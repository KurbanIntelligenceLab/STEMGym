#!/usr/bin/env python3
"""Train the analyst ensemble (AtomFinder + DefectClassifier + PhaseIdentifier).

Generates training data from HDF5 world files, trains all three models,
and saves checkpoints.

Usage:
    uv run --locked python stem_gym/scripts/train_analyst.py --world data/simulated/test_world.h5 \
                                             --output checkpoints/ \
                                             --epochs 30

For multi-world training:
    uv run --locked python stem_gym/scripts/train_analyst.py \
        --world data/simulated/srtio3_clustered_easy.h5 \
        --world data/simulated/batio3_interface_easy.h5 \
        --world data/simulated/sige_gradient_clustered_easy.h5 \
        --output checkpoints/ --device cuda

Phase training auto-detects worlds with phase maps. Override with:
    --phase-world data/simulated/batio3_interface_easy.h5 \
    --phase-world data/simulated/gan_easy.h5
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stem_gym.agents.models import AtomFinderEnsemble, DefectClassifierCNN, PhaseIdentifierResNet

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset, Subset, random_split

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

    Loads tiles lazily from disk for large worlds, but caches atom maps.
    """

    def __init__(
        self,
        world_paths: list[Path],
        augment: bool = True,
        atom_sigma: float = 2.0,
        max_tiles_per_world: int = 0,
    ):
        self.augment = augment
        self.atom_sigma = atom_sigma

        # Per-tile metadata (lightweight)
        self.tile_refs: list[dict] = []  # {path, row, col, tile_size, stride, pixel_size}
        self.atom_maps: list[np.ndarray] = []
        self.defect_patches: list[list[tuple[np.ndarray, int]]] = []
        self.phase_labels: list[int] = []
        self.world_indices: list[int] = []  # which world each tile came from

        for i, path in enumerate(world_paths):
            self._load_world(path, world_idx=i, max_tiles=max_tiles_per_world)

        logger.info(f"Loaded {len(self.tile_refs)} tiles from {len(world_paths)} worlds")

    def _load_world(self, path: Path, world_idx: int, max_tiles: int = 0) -> None:
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

            # Build tile list
            tile_coords = [(r, c) for r in range(grid_shape[0]) for c in range(grid_shape[1])
                           if f"tiles/{r}_{c}" in f]

            # Subsample if requested
            if max_tiles > 0 and len(tile_coords) > max_tiles:
                rng = np.random.default_rng(42 + world_idx)
                indices = rng.choice(len(tile_coords), max_tiles, replace=False)
                tile_coords = [tile_coords[i] for i in sorted(indices)]

            for r, c in tile_coords:
                tile = np.array(f[f"tiles/{r}_{c}"], dtype=np.float32)

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
                    px_c = (pos[0] - x0_nm) / pixel_size
                    px_r = (pos[1] - y0_nm) / pixel_size
                    pr, pc = int(round(px_r)), int(round(px_c))
                    if 0 <= pr < tile_size and 0 <= pc < tile_size:
                        atom_map[pr, pc] = 1.0

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

                self.tile_refs.append({
                    "path": str(path), "row": r, "col": c,
                    "tile_size": tile_size, "stride": stride,
                    "pixel_size": pixel_size,
                })
                self.atom_maps.append(atom_map)
                self.defect_patches.append(patches)
                self.phase_labels.append(phase_label)
                self.world_indices.append(world_idx)

                # Store tile data in ref for retrieval
                self.tile_refs[-1]["tile_data"] = tile

    def __len__(self) -> int:
        return len(self.tile_refs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tile = self.tile_refs[idx]["tile_data"].copy()
        atom_map = self.atom_maps[idx].copy()
        phase_label = self.phase_labels[idx]

        # Normalize tile
        t_min, t_max = tile.min(), tile.max()
        if t_max > t_min:
            tile = (tile - t_min) / (t_max - t_min)

        if self.augment:
            k = np.random.randint(0, 4)
            tile = np.rot90(tile, k).copy()
            atom_map = np.rot90(atom_map, k).copy()
            if np.random.random() > 0.5:
                tile = np.fliplr(tile).copy()
                atom_map = np.fliplr(atom_map).copy()
            noise_std = np.random.uniform(0, 0.05)
            tile = tile + np.random.normal(0, noise_std, tile.shape).astype(np.float32)
            tile = np.clip(tile, 0, 1)
            gamma = np.random.uniform(0.8, 1.2)
            tile = np.power(tile + 1e-6, gamma).astype(np.float32)

        return {
            "tile": torch.from_numpy(tile).unsqueeze(0),
            "atom_map": torch.from_numpy(atom_map).unsqueeze(0),
            "phase_label": torch.tensor(phase_label, dtype=torch.long),
        }


class DefectPatchDataset(Dataset):
    """Flat dataset of 32x32 defect patches extracted from TileDataset.

    Supports max_patches with class-balanced subsampling.
    """

    def __init__(
        self,
        tile_dataset: TileDataset,
        augment: bool = True,
        max_patches: int = 0,
    ):
        self.augment = augment

        # Collect all patches
        all_patches: list[np.ndarray] = []
        all_labels: list[int] = []
        for patches in tile_dataset.defect_patches:
            for patch, label in patches:
                all_patches.append(patch)
                all_labels.append(label)

        # Subsample with class balance if needed
        if max_patches > 0 and len(all_patches) > max_patches:
            logger.info(f"Subsampling {len(all_patches)} patches to {max_patches} (class-balanced)")
            all_patches, all_labels = self._balanced_subsample(
                all_patches, all_labels, max_patches
            )

        self.patches = all_patches
        self.labels = all_labels

        logger.info(f"DefectPatchDataset: {len(self.patches)} patches")
        if self.patches:
            counts = Counter(self.labels)
            for cls, cnt in sorted(counts.items()):
                logger.info(f"  class {cls}: {cnt} patches")

    @staticmethod
    def _balanced_subsample(
        patches: list[np.ndarray],
        labels: list[int],
        max_patches: int,
    ) -> tuple[list[np.ndarray], list[int]]:
        """Subsample with class-balanced strategy: keep all minority, sample majority."""
        rng = np.random.default_rng(42)
        counts = Counter(labels)
        n_classes = len(counts)
        per_class = max_patches // n_classes

        # Group by class
        by_class: dict[int, list[int]] = {}
        for i, lbl in enumerate(labels):
            by_class.setdefault(lbl, []).append(i)

        selected: list[int] = []
        for cls, indices in by_class.items():
            if len(indices) <= per_class:
                selected.extend(indices)
            else:
                selected.extend(rng.choice(indices, per_class, replace=False).tolist())

        # Fill remaining budget from majority class
        remaining = max_patches - len(selected)
        if remaining > 0:
            unselected = list(set(range(len(labels))) - set(selected))
            if unselected:
                extra = rng.choice(unselected, min(remaining, len(unselected)), replace=False)
                selected.extend(extra.tolist())

        return [patches[i] for i in selected], [labels[i] for i in selected]

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        patch = self.patches[idx].copy()
        label = self.labels[idx]

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
    patience: int = 7,
) -> tuple["AtomFinderEnsemble", dict]:
    """Train the AtomFinder U-Net ensemble with best-model checkpointing."""
    from stem_gym.agents.models import AtomFinderEnsemble

    ensemble = AtomFinderEnsemble(n_members=n_members).to(device)

    # Split data
    n_val = max(1, len(dataset) // 5)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_indices = train_ds.indices if hasattr(train_ds, "indices") else list(range(n_train))
    val_indices = val_ds.indices if hasattr(val_ds, "indices") else list(range(n_train, len(dataset)))

    metadata = {
        "n_members": n_members, "epochs": epochs, "lr": lr,
        "n_train": n_train, "n_val": n_val,
        "train_indices": [int(i) for i in train_indices],
        "val_indices": [int(i) for i in val_indices],
        "loss_curves": {},
    }

    for member_idx in range(n_members):
        logger.info(f"Training AtomFinder member {member_idx+1}/{n_members}...")
        model = ensemble.members[member_idx]
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0
        train_losses, val_losses = [], []

        for epoch in range(epochs):
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

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    tile = batch["tile"].to(device)
                    target = batch["atom_map"].to(device)
                    pred = model(tile)
                    val_loss += F.binary_cross_entropy(pred, target).item() * tile.size(0)
            val_loss /= n_val

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"  Member {member_idx+1} epoch {epoch+1}/{epochs}: "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                logger.info(f"  Member {member_idx+1} early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)
            logger.info(f"  Member {member_idx+1} restored best model (val_loss={best_val_loss:.4f})")

        metadata["loss_curves"][f"member_{member_idx}"] = {
            "train": train_losses, "val": val_losses,
            "best_val_loss": best_val_loss,
        }

    return ensemble, metadata


def train_defect_classifier(
    tile_dataset: TileDataset,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    max_patches: int = 0,
    patience: int = 7,
) -> tuple["DefectClassifierCNN", dict]:
    """Train the defect classifier CNN with best-model checkpointing."""
    from stem_gym.agents.models import DefectClassifierCNN

    patch_ds = DefectPatchDataset(tile_dataset, max_patches=max_patches)
    if len(patch_ds) < 5:
        logger.warning("Not enough defect patches for training, returning untrained model")
        return DefectClassifierCNN(), {"error": "not_enough_patches"}

    n_val = max(1, len(patch_ds) // 5)
    n_train = len(patch_ds) - n_val
    train_ds, val_ds = random_split(
        patch_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    model = DefectClassifierCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Class weights for imbalanced data
    label_counts = Counter(patch_ds.labels)
    n_classes = 4
    weights = torch.ones(n_classes)
    total = len(patch_ds)
    for cls, cnt in label_counts.items():
        if cls < n_classes:
            weights[cls] = total / (n_classes * cnt) if cnt > 0 else 1.0
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    metadata = {
        "n_patches": len(patch_ds), "n_train": n_train, "n_val": n_val,
        "class_counts": dict(label_counts), "epochs": epochs, "lr": lr,
        "loss_curves": {"train": [], "val": [], "train_acc": [], "val_acc": []},
    }

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

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for patches, labels in val_loader:
                patches = patches.to(device)
                labels = labels.to(device)
                logits = model(patches)
                val_loss += criterion(logits, labels).item() * patches.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += patches.size(0)
        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        metadata["loss_curves"]["train"].append(train_loss)
        metadata["loss_curves"]["val"].append(val_loss)
        metadata["loss_curves"]["train_acc"].append(train_acc)
        metadata["loss_curves"]["val_acc"].append(val_acc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"  DefectClassifier epoch {epoch+1}/{epochs}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
                f"val_acc={val_acc:.3f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.info(f"  DefectClassifier early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"  DefectClassifier restored best model (val_loss={best_val_loss:.4f})")

    metadata["best_val_loss"] = best_val_loss
    return model, metadata


def train_phase_identifier(
    tile_dataset: TileDataset,
    phase_world_indices: list[int] | None = None,
    n_phases: int = 0,
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str = "cpu",
    patience: int = 7,
) -> tuple["PhaseIdentifierResNet", dict]:
    """Train the phase identifier ResNet.

    Args:
        phase_world_indices: If provided, only use tiles from these worlds.
            Worlds without phase maps are auto-excluded.
        n_phases: Number of phase classes. Auto-detected if 0.
    """
    from stem_gym.agents.models import PhaseIdentifierResNet

    # Filter to only tiles from phase worlds
    if phase_world_indices is not None:
        valid_idx = [i for i, w in enumerate(tile_dataset.world_indices)
                     if w in phase_world_indices]
    else:
        valid_idx = list(range(len(tile_dataset)))

    # Auto-detect n_phases from labels of valid tiles
    phase_labels_valid = [tile_dataset.phase_labels[i] for i in valid_idx]
    unique_phases = sorted(set(phase_labels_valid))

    if len(unique_phases) < 2:
        logger.warning("Only one phase present in selected worlds, returning untrained model")
        return PhaseIdentifierResNet(n_phases=max(n_phases, 2)), {"error": "single_phase"}

    if n_phases <= 0:
        n_phases = max(unique_phases) + 1
    logger.info(f"Phase training: {len(valid_idx)} tiles, {n_phases} phases, labels={unique_phases}")

    # Create subset
    subset = Subset(tile_dataset, valid_idx)
    n_val = max(1, len(subset) // 5)
    n_train = len(subset) - n_val
    train_ds, val_ds = random_split(
        subset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    model = PhaseIdentifierResNet(n_phases=n_phases).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    metadata = {
        "n_phases": n_phases, "n_tiles": len(valid_idx),
        "n_train": n_train, "n_val": n_val,
        "unique_phases": unique_phases, "epochs": epochs, "lr": lr,
        "loss_curves": {"train": [], "val": []},
    }

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_samples = 0
        for batch in train_loader:
            tile = batch["tile"].to(device)
            h, w = tile.shape[-2:]
            ch, cw = h // 2, w // 2
            tile_crop = tile[:, :, ch-32:ch+32, cw-32:cw+32]
            label = batch["phase_label"].to(device)
            logits = model(tile_crop)
            loss = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * tile.size(0)
            train_samples += tile.size(0)
        train_loss /= max(train_samples, 1)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                tile = batch["tile"].to(device)
                h, w = tile.shape[-2:]
                ch, cw = h // 2, w // 2
                tile_crop = tile[:, :, ch-32:ch+32, cw-32:cw+32]
                label = batch["phase_label"].to(device)
                logits = model(tile_crop)
                val_loss += criterion(logits, label).item() * tile.size(0)
                val_correct += (logits.argmax(1) == label).sum().item()
                val_total += tile.size(0)
        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        metadata["loss_curves"]["train"].append(train_loss)
        metadata["loss_curves"]["val"].append(val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"  PhaseIdentifier epoch {epoch+1}/{epochs}: "
                f"train_loss={train_loss:.4f}, val_acc={val_acc:.3f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.info(f"  PhaseIdentifier early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"  PhaseIdentifier restored best model (val_loss={best_val_loss:.4f})")

    metadata["best_val_loss"] = best_val_loss
    return model, metadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_phase_worlds(world_paths: list[Path]) -> list[int]:
    """Return indices of worlds that have a multi-valued phase_map in GT."""
    phase_indices = []
    for i, path in enumerate(world_paths):
        with h5py.File(path, "r") as f:
            if "ground_truth/phase_map" in f:
                pm = np.array(f["ground_truth/phase_map"])
                if len(np.unique(pm)) >= 2:
                    phase_indices.append(i)
                    logger.info(f"  Phase world detected: {path.name} ({len(np.unique(pm))} phases)")
                else:
                    logger.info(f"  Skipping {path.name} for phase training (single phase)")
            else:
                logger.info(f"  Skipping {path.name} for phase training (no phase_map)")
    return phase_indices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train STEMGym analyst ensemble")
    parser.add_argument(
        "--world", type=Path, action="append", required=True,
        help="Path(s) to HDF5 world files for training data",
    )
    parser.add_argument(
        "--phase-world", type=Path, action="append", default=None,
        help="Override: only these worlds for phase training (auto-detected if omitted)",
    )
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--max-patches", type=int, default=50000,
                        help="Max defect patches per dataset (0=unlimited)")
    parser.add_argument("--max-tiles-per-world", type=int, default=0,
                        help="Max tiles per world (0=all)")
    parser.add_argument("--patience", type=int, default=7,
                        help="Early stopping patience (epochs)")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # Load data
    logger.info(f"Loading data from {len(args.world)} worlds...")
    dataset = TileDataset(
        args.world, augment=True,
        max_tiles_per_world=args.max_tiles_per_world,
    )

    # Detect phase worlds
    if args.phase_world:
        # Map phase-world paths to indices in args.world
        phase_world_indices = []
        for pw in args.phase_world:
            for i, w in enumerate(args.world):
                if pw.resolve() == w.resolve():
                    phase_world_indices.append(i)
                    break
    else:
        logger.info("Auto-detecting phase worlds...")
        phase_world_indices = detect_phase_worlds(args.world)

    all_metadata = {
        "worlds": [str(w) for w in args.world],
        "phase_worlds": [str(args.world[i]) for i in phase_world_indices],
        "device": args.device,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Train AtomFinder ensemble
    logger.info("=" * 50)
    logger.info("Training AtomFinder ensemble...")
    atom_finder, af_meta = train_atom_finder(
        dataset, n_members=args.ensemble_size,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, device=args.device, patience=args.patience,
    )
    torch.save(atom_finder.state_dict(), args.output / "atom_finder.pt")
    all_metadata["atom_finder"] = af_meta
    logger.info("Saved atom_finder.pt")

    # Train DefectClassifier
    logger.info("=" * 50)
    logger.info("Training DefectClassifier...")
    defect_clf, dc_meta = train_defect_classifier(
        dataset, epochs=args.epochs, batch_size=32,
        lr=args.lr, device=args.device,
        max_patches=args.max_patches, patience=args.patience,
    )
    torch.save(defect_clf.state_dict(), args.output / "defect_classifier.pt")
    all_metadata["defect_classifier"] = dc_meta
    logger.info("Saved defect_classifier.pt")

    # Train PhaseIdentifier
    logger.info("=" * 50)
    logger.info("Training PhaseIdentifier...")
    phase_id, pi_meta = train_phase_identifier(
        dataset, phase_world_indices=phase_world_indices,
        epochs=min(args.epochs, 20),
        batch_size=args.batch_size, lr=args.lr, device=args.device,
        patience=args.patience,
    )
    torch.save(phase_id.state_dict(), args.output / "phase_identifier.pt")
    all_metadata["phase_identifier"] = pi_meta
    logger.info("Saved phase_identifier.pt")

    # Save training metadata
    elapsed = time.time() - t_start
    all_metadata["total_training_seconds"] = round(elapsed, 1)
    with open(args.output / "training_metadata.json", "w") as f:
        json.dump(all_metadata, f, indent=2, default=str)
    logger.info(f"Saved training_metadata.json")

    logger.info("=" * 50)
    logger.info(f"Training complete in {elapsed:.0f}s!")


if __name__ == "__main__":
    main()
