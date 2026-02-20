#!/usr/bin/env python3
"""Train the analyst ensemble (AtomFinder + DefectClassifier + PhaseIdentifier).

Generates training data from HDF5 world files, trains all three models,
and saves checkpoints.

Usage:
    python scripts/train_atom_finder.py --world data/simulated/test_world.h5 \
                                        --output checkpoints/ \
                                        --epochs 30

For multi-world training:
    python scripts/train_atom_finder.py \
        --world data/simulated/test_world.h5 \
        --world data/simulated/srtio3_medium.h5 \
        --output checkpoints/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset, random_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset: tiles + ground truth from HDF5 worlds
# ---------------------------------------------------------------------------

class TileDataset(Dataset):
    """Dataset of tiles with atom position ground truth from HDF5 worlds.

    For each tile, generates:
        - tile image (128x128, normalized)
        - atom probability map (128x128, Gaussian blobs at atom positions)
        - per-atom defect labels
        - phase label
    """

    def __init__(
        self,
        world_paths: list[Path],
        augment: bool = True,
        atom_sigma: float = 2.0,
    ):
        self.augment = augment
        self.atom_sigma = atom_sigma
        self.tiles: list[np.ndarray] = []
        self.atom_maps: list[np.ndarray] = []
        self.defect_patches: list[list[tuple[np.ndarray, int]]] = []
        self.phase_labels: list[int] = []

        for path in world_paths:
            self._load_world(path)

        logger.info(f"Loaded {len(self.tiles)} tiles from {len(world_paths)} worlds")

    def _load_world(self, path: Path) -> None:
        """Extract tiles and ground truth from one HDF5 world."""
        with h5py.File(path, "r") as f:
            meta = f["metadata"]
            tile_size = int(meta.attrs["tile_size_px"])
            overlap = int(meta.attrs["overlap_px"])
            pixel_size = float(meta.attrs["pixel_size_nm"])
            grid_shape = tuple(meta.attrs["grid_shape"])
            stride = tile_size - overlap

            # Load ground truth
            gt = f["ground_truth"]
            atom_positions = np.array(gt["atom_positions"])  # (N, 2) in nm (x, y)
            atom_types = np.array(gt["atom_types"])           # 0=normal,1=vac,2=sub

            # Phase map if available
            phase_map = None
            if "phase_map" in gt:
                phase_map = np.array(gt["phase_map"])

            # Process each tile
            for r in range(grid_shape[0]):
                for c in range(grid_shape[1]):
                    key = f"tiles/{r}_{c}"
                    if key not in f:
                        continue
                    tile = np.array(f[key], dtype=np.float32)

                    # Tile bounds in nm
                    y0_px = r * stride
                    x0_px = c * stride
                    x0_nm = x0_px * pixel_size
                    y0_nm = y0_px * pixel_size
                    x1_nm = (x0_px + tile_size) * pixel_size
                    y1_nm = (y0_px + tile_size) * pixel_size

                    # Find atoms within this tile
                    in_tile = (
                        (atom_positions[:, 0] >= x0_nm) &
                        (atom_positions[:, 0] < x1_nm) &
                        (atom_positions[:, 1] >= y0_nm) &
                        (atom_positions[:, 1] < y1_nm)
                    )
                    tile_atom_pos = atom_positions[in_tile]
                    tile_atom_types = atom_types[in_tile]

                    # Create atom probability map
                    atom_map = np.zeros((tile_size, tile_size), dtype=np.float32)
                    for pos in tile_atom_pos:
                        # Convert nm → tile pixel
                        px_c = (pos[0] - x0_nm) / pixel_size
                        px_r = (pos[1] - y0_nm) / pixel_size
                        pr, pc = int(round(px_r)), int(round(px_c))
                        if 0 <= pr < tile_size and 0 <= pc < tile_size:
                            atom_map[pr, pc] = 1.0

                    # Apply Gaussian blur to create soft targets
                    if atom_map.max() > 0:
                        atom_map = gaussian_filter(atom_map, sigma=self.atom_sigma)
                        atom_map /= atom_map.max()

                    # Defect patches (32x32 around each atom)
                    patches = []
                    for pos, atype in zip(tile_atom_pos, tile_atom_types):
                        px_c = int(round((pos[0] - x0_nm) / pixel_size))
                        px_r = int(round((pos[1] - y0_nm) / pixel_size))
                        half = 16
                        if half <= px_r < tile_size - half and half <= px_c < tile_size - half:
                            patch = tile[px_r - half:px_r + half, px_c - half:px_c + half]
                            if patch.shape == (32, 32):
                                patches.append((patch, int(atype)))

                    # Phase label
                    phase_label = 0
                    if phase_map is not None:
                        center_r = y0_px + tile_size // 2
                        center_c = x0_px + tile_size // 2
                        if center_r < phase_map.shape[0] and center_c < phase_map.shape[1]:
                            phase_label = int(phase_map[center_r, center_c])

                    self.tiles.append(tile)
                    self.atom_maps.append(atom_map)
                    self.defect_patches.append(patches)
                    self.phase_labels.append(phase_label)

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tile = self.tiles[idx].copy()
        atom_map = self.atom_maps[idx].copy()
        phase_label = self.phase_labels[idx]

        # Normalize tile
        t_min, t_max = tile.min(), tile.max()
        if t_max > t_min:
            tile = (tile - t_min) / (t_max - t_min)

        if self.augment:
            # Random 90° rotation
            k = np.random.randint(0, 4)
            tile = np.rot90(tile, k).copy()
            atom_map = np.rot90(atom_map, k).copy()
            # Random flip
            if np.random.random() > 0.5:
                tile = np.fliplr(tile).copy()
                atom_map = np.fliplr(atom_map).copy()
            # Random noise
            noise_std = np.random.uniform(0, 0.05)
            tile = tile + np.random.normal(0, noise_std, tile.shape).astype(np.float32)
            tile = np.clip(tile, 0, 1)
            # Random contrast
            gamma = np.random.uniform(0.8, 1.2)
            tile = np.power(tile + 1e-6, gamma).astype(np.float32)

        return {
            "tile": torch.from_numpy(tile).unsqueeze(0),          # (1, H, W)
            "atom_map": torch.from_numpy(atom_map).unsqueeze(0),  # (1, H, W)
            "phase_label": torch.tensor(phase_label, dtype=torch.long),
        }


class DefectPatchDataset(Dataset):
    """Flat dataset of 32x32 defect patches extracted from TileDataset."""

    def __init__(self, tile_dataset: TileDataset, augment: bool = True):
        self.patches: list[np.ndarray] = []
        self.labels: list[int] = []
        self.augment = augment

        for patches in tile_dataset.defect_patches:
            for patch, label in patches:
                self.patches.append(patch)
                self.labels.append(label)

        logger.info(f"DefectPatchDataset: {len(self.patches)} patches")
        if self.patches:
            from collections import Counter
            counts = Counter(self.labels)
            for cls, cnt in sorted(counts.items()):
                logger.info(f"  class {cls}: {cnt} patches")

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        patch = self.patches[idx].copy()
        label = self.labels[idx]

        # Normalize
        p_min, p_max = patch.min(), patch.max()
        if p_max > p_min:
            patch = (patch - p_min) / (p_max - p_min)

        if self.augment:
            k = np.random.randint(0, 4)
            patch = np.rot90(patch, k).copy()
            if np.random.random() > 0.5:
                patch = np.fliplr(patch).copy()

        return (
            torch.from_numpy(patch).unsqueeze(0).float(),
            torch.tensor(label, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_atom_finder(
    dataset: TileDataset,
    n_members: int = 3,
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str = "cpu",
) -> "AtomFinderEnsemble":
    """Train the AtomFinder U-Net ensemble."""
    from stem_gym.agents.models import AtomFinderEnsemble

    ensemble = AtomFinderEnsemble(n_members=n_members).to(device)

    # Split data
    n_val = max(1, len(dataset) // 5)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    for member_idx in range(n_members):
        logger.info(f"Training AtomFinder member {member_idx+1}/{n_members}...")
        model = ensemble.members[member_idx]
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        best_val_loss = float("inf")
        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                tile = batch["tile"].to(device)
                target = batch["atom_map"].to(device)

                pred = model(tile)
                loss = F.binary_cross_entropy(pred, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * tile.size(0)
            train_loss /= n_train

            # Validate
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    tile = batch["tile"].to(device)
                    target = batch["atom_map"].to(device)
                    pred = model(tile)
                    val_loss += F.binary_cross_entropy(pred, target).item() * tile.size(0)
            val_loss /= n_val

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"  Member {member_idx+1} epoch {epoch+1}/{epochs}: "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss

    return ensemble


def train_defect_classifier(
    tile_dataset: TileDataset,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
) -> "DefectClassifierCNN":
    """Train the defect classifier CNN."""
    from stem_gym.agents.models import DefectClassifierCNN

    patch_ds = DefectPatchDataset(tile_dataset)
    if len(patch_ds) < 5:
        logger.warning("Not enough defect patches for training, returning untrained model")
        return DefectClassifierCNN()

    n_val = max(1, len(patch_ds) // 5)
    n_train = len(patch_ds) - n_val
    train_ds, val_ds = random_split(patch_ds, [n_train, n_val])

    model = DefectClassifierCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Class weights for imbalanced data
    from collections import Counter
    label_counts = Counter(patch_ds.labels)
    n_classes = 4
    weights = torch.ones(n_classes)
    total = len(patch_ds)
    for cls, cnt in label_counts.items():
        if cls < n_classes:
            weights[cls] = total / (n_classes * cnt) if cnt > 0 else 1.0
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total_samples = 0
        for patches, labels in train_loader:
            patches = patches.to(device)
            labels = labels.to(device)

            logits = model(patches)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * patches.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total_samples += patches.size(0)

        train_loss /= max(total_samples, 1)
        train_acc = correct / max(total_samples, 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            # Validate
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for patches, labels in val_loader:
                    patches = patches.to(device)
                    labels = labels.to(device)
                    logits = model(patches)
                    val_correct += (logits.argmax(1) == labels).sum().item()
                    val_total += patches.size(0)
            val_acc = val_correct / max(val_total, 1)
            logger.info(
                f"  DefectClassifier epoch {epoch+1}/{epochs}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
                f"val_acc={val_acc:.3f}"
            )

    return model


def train_phase_identifier(
    tile_dataset: TileDataset,
    n_phases: int = 2,
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str = "cpu",
) -> "PhaseIdentifierResNet":
    """Train the phase identifier ResNet."""
    from stem_gym.agents.models import PhaseIdentifierResNet

    # Check if we have multiple phases
    unique_phases = set(tile_dataset.phase_labels)
    if len(unique_phases) < 2:
        logger.warning("Only one phase present, returning untrained model")
        return PhaseIdentifierResNet(n_phases=n_phases)

    n_val = max(1, len(tile_dataset) // 5)
    n_train = len(tile_dataset) - n_val
    train_ds, val_ds = random_split(tile_dataset, [n_train, n_val])

    model = PhaseIdentifierResNet(n_phases=n_phases).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            tile = batch["tile"].to(device)
            # Crop center 64x64 for phase ID
            h, w = tile.shape[-2:]
            ch, cw = h // 2, w // 2
            tile_crop = tile[:, :, ch-32:ch+32, cw-32:cw+32]
            label = batch["phase_label"].to(device)

            logits = model(tile_crop)
            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train STEMGym analyst ensemble")
    parser.add_argument(
        "--world", type=Path, action="append", required=True,
        help="Path(s) to HDF5 world files for training data",
    )
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ensemble-size", type=int, default=3)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Load data
    dataset = TileDataset(args.world, augment=True)

    # Train AtomFinder ensemble
    logger.info("=" * 50)
    logger.info("Training AtomFinder ensemble...")
    atom_finder = train_atom_finder(
        dataset, n_members=args.ensemble_size,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, device=args.device,
    )
    torch.save(atom_finder.state_dict(), args.output / "atom_finder.pt")
    logger.info(f"Saved atom_finder.pt")

    # Train DefectClassifier
    logger.info("=" * 50)
    logger.info("Training DefectClassifier...")
    defect_clf = train_defect_classifier(
        dataset, epochs=args.epochs, batch_size=32,
        lr=args.lr, device=args.device,
    )
    torch.save(defect_clf.state_dict(), args.output / "defect_classifier.pt")
    logger.info(f"Saved defect_classifier.pt")

    # Train PhaseIdentifier
    logger.info("=" * 50)
    logger.info("Training PhaseIdentifier...")
    phase_id = train_phase_identifier(
        dataset, epochs=min(args.epochs, 20),
        batch_size=args.batch_size, lr=args.lr, device=args.device,
    )
    torch.save(phase_id.state_dict(), args.output / "phase_identifier.pt")
    logger.info(f"Saved phase_identifier.pt")

    logger.info("=" * 50)
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
