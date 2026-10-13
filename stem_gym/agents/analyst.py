"""Analyst ensemble: AtomFinder + DefectClassifier + PhaseIdentifier.

Wraps the neural network models into a unified analysis pipeline with
the same interface as SimpleAnalyst (analyze_tile → AnalysisResult).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from skimage.feature import peak_local_max

from stem_gym.agents.models import (
    AtomFinderEnsemble,
    DefectClassifierCNN,
    PhaseIdentifierResNet,
)
from stem_gym.agents.simple_analyst import AnalysisResult


DEFECT_CLASSES = ["pristine", "vacancy", "substitution", "interstitial"]


def _default_device() -> str:
    """Return 'cuda' if available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


class Analyst:
    """Neural network analyst ensemble.

    Pipeline per tile:
        1. AtomFinder U-Net ensemble → probability map + uncertainty
        2. Peak detection on probability map → atom positions
        3. Extract 32×32 patches at each atom → DefectClassifier → class + confidence
        4. Extract 64×64 patch at tile center → PhaseIdentifier → phase label

    Falls back to SimpleAnalyst if no model weights are loaded.
    """

    def __init__(
        self,
        atom_finder: AtomFinderEnsemble | None = None,
        defect_classifier: DefectClassifierCNN | None = None,
        phase_identifier: PhaseIdentifierResNet | None = None,
        device: str | None = None,
        peak_threshold: float = 0.15,
        peak_min_distance: int = 5,
        defect_patch_size: int = 32,
        phase_patch_size: int = 64,
    ):
        self.device = torch.device(device if device is not None else _default_device())
        self.peak_threshold = peak_threshold
        self.peak_min_distance = peak_min_distance
        self.defect_patch_size = defect_patch_size
        self.phase_patch_size = phase_patch_size

        # Models (can be None if not trained yet)
        self.atom_finder = atom_finder
        self.defect_classifier = defect_classifier
        self.phase_identifier = phase_identifier

        if self.atom_finder is not None:
            self.atom_finder = self.atom_finder.to(self.device).eval()
        if self.defect_classifier is not None:
            self.defect_classifier = self.defect_classifier.to(self.device).eval()
        if self.phase_identifier is not None:
            self.phase_identifier = self.phase_identifier.to(self.device).eval()

        self._loaded_defect_classifier = self.defect_classifier
        self._loaded_phase_identifier = self.phase_identifier

        # Track degenerate model detection (always predicts same class)
        self._defect_clf_check_count = 0
        self._defect_clf_pred_counts: dict[int, int] = {}
        self._phase_id_check_count = 0
        self._phase_id_pred_counts: dict[int, int] = {}

        # Accumulated results
        self._tile_results: list[AnalysisResult] = []
        self._all_atom_positions_nm: list[np.ndarray] = []
        self._all_defect_labels: list[str] = []

        # Phase heuristic: track mean intensity per tile for median-split
        self._phase_intensities: list[float] = []
        self._phase_tile_keys: list[tuple[int, int]] = []
        # Neural model phase predictions per tile (used when model is available)
        self._phase_model_predictions: dict[tuple[int, int], int] = {}

    def reset(self) -> None:
        self.defect_classifier = self._loaded_defect_classifier
        self.phase_identifier = self._loaded_phase_identifier
        self._tile_results.clear()
        self._all_atom_positions_nm.clear()
        self._all_defect_labels.clear()
        self._defect_clf_check_count = 0
        self._defect_clf_pred_counts.clear()
        self._phase_id_check_count = 0
        self._phase_id_pred_counts.clear()
        self._phase_intensities.clear()
        self._phase_tile_keys.clear()
        self._phase_model_predictions.clear()

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str | Path, device: str | None = None, *, compile_models: bool = False) -> "Analyst":
        """Load trained models from a checkpoint directory.

        Expected files:
            atom_finder.pt — AtomFinderEnsemble state dict
            defect_classifier.pt — DefectClassifierCNN state dict
            phase_identifier.pt — PhaseIdentifierResNet state dict
        """
        if device is None:
            device = _default_device()
        cp = Path(checkpoint_dir)

        atom_finder = None
        if (cp / "atom_finder.pt").exists():
            sd = torch.load(cp / "atom_finder.pt", map_location=device, weights_only=True)
            # Infer n_members from state dict keys
            member_ids = {int(k.split(".")[1]) for k in sd if k.startswith("members.")}
            n_members = max(member_ids) + 1 if member_ids else 3
            atom_finder = AtomFinderEnsemble(n_members=n_members)
            atom_finder.load_state_dict(sd)
            if compile_models:
                atom_finder = torch.compile(atom_finder)

        defect_classifier = None
        if (cp / "defect_classifier.pt").exists():
            sd = torch.load(cp / "defect_classifier.pt", map_location=device, weights_only=True)
            # Infer n_classes from classifier output dimension
            fc_key = "classifier.4.weight"
            n_classes = sd[fc_key].shape[0] if fc_key in sd else 4
            defect_classifier = DefectClassifierCNN(n_classes=n_classes)
            defect_classifier.load_state_dict(sd)
            if compile_models:
                defect_classifier = torch.compile(defect_classifier)

        phase_identifier = None
        if (cp / "phase_identifier.pt").exists():
            sd = torch.load(cp / "phase_identifier.pt", map_location=device, weights_only=True)
            # Infer n_phases from fc layer output dimension
            n_phases = sd["fc.weight"].shape[0] if "fc.weight" in sd else 2
            phase_identifier = PhaseIdentifierResNet(n_phases=n_phases)
            phase_identifier.load_state_dict(sd)
            if compile_models:
                phase_identifier = torch.compile(phase_identifier)

        return cls(
            atom_finder=atom_finder,
            defect_classifier=defect_classifier,
            phase_identifier=phase_identifier,
            device=device,
        )

    def _tile_to_tensor(self, tile: np.ndarray) -> torch.Tensor:
        """Convert (H, W) tile to (1, 1, H, W) tensor."""
        t = torch.from_numpy(tile.astype(np.float32))
        # Normalize to [0, 1]
        t_min, t_max = t.min(), t.max()
        if t_max > t_min:
            t = (t - t_min) / (t_max - t_min)
        return t.unsqueeze(0).unsqueeze(0).to(self.device)

    def _extract_patch(
        self, tile: np.ndarray, row: int, col: int, size: int
    ) -> np.ndarray:
        """Extract a square patch centered at (row, col), zero-padded at edges."""
        h, w = tile.shape
        half = size // 2
        patch = np.zeros((size, size), dtype=np.float32)

        r0 = max(0, row - half)
        r1 = min(h, row - half + size)
        c0 = max(0, col - half)
        c1 = min(w, col - half + size)

        pr0 = max(0, -(row - half))
        pc0 = max(0, -(col - half))
        pr1 = pr0 + (r1 - r0)
        pc1 = pc0 + (c1 - c0)

        if r1 > r0 and c1 > c0:
            patch[pr0:pr1, pc0:pc1] = tile[r0:r1, c0:c1]
        return patch

    def find_atoms(
        self, tile: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect atoms in a tile using the U-Net ensemble.

        Returns:
            positions: (N, 2) atom positions in tile pixels (row, col).
            confidences: (N,) detection confidence per atom.
            uncertainty_map: (H, W) predictive uncertainty.
        """
        if self.atom_finder is None:
            # Fallback: peak_local_max on raw tile
            from scipy.ndimage import gaussian_filter
            smoothed = gaussian_filter(tile.astype(np.float64), sigma=1.5)
            thresh = smoothed.max() * 0.2
            coords = peak_local_max(smoothed, min_distance=6, threshold_abs=thresh)
            if len(coords) == 0:
                return np.empty((0, 2)), np.empty(0), np.zeros_like(tile)
            confs = np.array([tile[r, c] for r, c in coords], dtype=np.float32)
            confs = confs / max(confs.max(), 1e-6)
            return coords.astype(np.float32), confs, np.zeros_like(tile)

        x = self._tile_to_tensor(tile)
        with torch.inference_mode():
            prob_map, unc_map = self.atom_finder(x)

        prob = prob_map[0, 0].cpu().numpy()
        unc = unc_map[0, 0].cpu().numpy()

        # Peak detection on probability map
        coords = peak_local_max(
            prob,
            min_distance=self.peak_min_distance,
            threshold_abs=self.peak_threshold,
        )

        if len(coords) == 0:
            return np.empty((0, 2)), np.empty(0), unc

        confidences = np.array([prob[r, c] for r, c in coords], dtype=np.float32)
        return coords.astype(np.float32), confidences, unc

    def classify_defects(
        self, tile: np.ndarray, positions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Classify atoms as pristine/vacancy/substitution/interstitial.

        Args:
            tile: (H, W) HAADF tile.
            positions: (N, 2) atom positions in tile pixels.

        Returns:
            labels: (N,) int class indices.
            confidences: (N,) classification confidence.
        """
        if len(positions) == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)

        if self.defect_classifier is None:
            # Fallback: intensity-based z-score classification.
            # When using the trained atom finder, positions may include
            # inter-column spots with low intensity. Filter to high-
            # intensity atoms (real column peaks) for robust z-scores.
            intensities = np.array(
                [tile[int(r), int(c)] for r, c in positions], dtype=np.float32
            )
            # Use only atoms above median intensity for statistics
            # (inter-column spots pull the mean down and inflate std)
            median_i = np.median(intensities)
            bright_mask = intensities >= median_i
            if bright_mask.sum() > 3:
                ref_intensities = intensities[bright_mask]
            else:
                ref_intensities = intensities
            mean_i = ref_intensities.mean()
            std_i = max(ref_intensities.std(), 1e-6)
            zscores = (intensities - mean_i) / std_i

            labels = np.zeros(len(positions), dtype=np.int32)
            labels[zscores < -1.5] = 1  # vacancy
            labels[zscores > 1.5] = 2   # substitution
            confs = np.clip(np.abs(zscores) / 3.0, 0.3, 1.0).astype(np.float32)
            return labels, confs

        # Extract patches
        patches = []
        for r, c in positions:
            patch = self._extract_patch(tile, int(r), int(c), self.defect_patch_size)
            patches.append(patch)

        batch = torch.from_numpy(np.stack(patches)).unsqueeze(1).to(self.device)
        # Normalize
        b_min = batch.amin(dim=(2, 3), keepdim=True)
        b_max = batch.amax(dim=(2, 3), keepdim=True)
        batch = (batch - b_min) / (b_max - b_min + 1e-6)

        with torch.inference_mode():
            logits = self.defect_classifier(batch)
            probs = F.softmax(logits, dim=-1)

        labels = probs.argmax(dim=-1).cpu().numpy().astype(np.int32)
        confidences = probs.max(dim=-1).values.cpu().numpy().astype(np.float32)

        # Detect degenerate model: if after 50+ atoms all predictions are
        # the same class, the model is broken — disable and use heuristic
        for lbl in labels:
            self._defect_clf_pred_counts[int(lbl)] = (
                self._defect_clf_pred_counts.get(int(lbl), 0) + 1
            )
        self._defect_clf_check_count += len(labels)
        if self._defect_clf_check_count >= 50 and len(self._defect_clf_pred_counts) == 1:
            import logging
            logging.getLogger(__name__).warning(
                "Defect classifier always predicts class %d — falling back to heuristic",
                next(iter(self._defect_clf_pred_counts)),
            )
            self.defect_classifier = None
            return self.classify_defects(tile, positions)

        return labels, confidences

    def _heuristic_phase(self, tile: np.ndarray) -> tuple[int, float]:
        """Intensity-based phase heuristic via running median split.

        Records each tile's mean intensity.  After >=4 tiles, splits at
        the running median: below → phase 0, above → phase 1.  Early
        tiles (< 4 observed) default to phase 0.

        The raw labels assigned here may be noise on single-phase worlds.
        ``get_corrected_tile_phases()`` applies a Cohen's d significance
        test to decide whether the split is real before exposing it as a
        phase map.
        """
        mean_intensity = float(tile.mean())
        self._phase_intensities.append(mean_intensity)

        if len(self._phase_intensities) < 4:
            return 0, 0.5

        median = float(np.median(self._phase_intensities))
        label = 0 if mean_intensity <= median else 1
        # Confidence: distance from median relative to spread
        spread = max(float(np.std(self._phase_intensities)), 1e-8)
        conf = min(abs(mean_intensity - median) / spread, 1.0)
        conf = 0.5 + 0.5 * conf  # scale to [0.5, 1.0]
        return label, float(conf)

    def identify_phase(self, tile: np.ndarray) -> tuple[int, float]:
        """Identify the phase of a tile.

        Returns:
            phase_label: int phase index.
            confidence: classification confidence.

        Without a trained model, falls back to adaptive intensity-based
        median split heuristic.
        """
        if self.phase_identifier is None:
            return self._heuristic_phase(tile)

        # Extract center patch
        h, w = tile.shape
        patch = self._extract_patch(
            tile, h // 2, w // 2, self.phase_patch_size
        )
        x = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(self.device)
        # Normalize
        x_min, x_max = x.min(), x.max()
        if x_max > x_min:
            x = (x - x_min) / (x_max - x_min)

        with torch.inference_mode():
            logits = self.phase_identifier(x)
            probs = F.softmax(logits, dim=-1)

        label = probs.argmax(dim=-1).item()
        conf = probs.max(dim=-1).values.item()

        # Detect degenerate model: if after 5+ tiles all predictions are
        # the same class, the model is broken — disable and use heuristic
        self._phase_id_pred_counts[label] = (
            self._phase_id_pred_counts.get(label, 0) + 1
        )
        self._phase_id_check_count += 1
        if self._phase_id_check_count >= 5 and len(self._phase_id_pred_counts) == 1:
            import logging
            logging.getLogger(__name__).warning(
                "Phase identifier always predicts class %d — falling back to heuristic",
                next(iter(self._phase_id_pred_counts)),
            )
            self.phase_identifier = None
            return self._heuristic_phase(tile)

        return int(label), float(conf)

    def analyze_tile(
        self,
        tile: np.ndarray,
        tile_position_nm: tuple[float, float] | None = None,
        pixel_size_nm: float = 0.005,
        tile_key: tuple[int, int] | None = None,
    ) -> AnalysisResult:
        """Full analysis pipeline for a single tile.

        Same interface as SimpleAnalyst.analyze_tile().

        Args:
            tile_key: Optional (row, col) grid index. If provided, enables
                phase correction via ``get_corrected_tile_phases()``.
        """
        if tile.max() <= 0:
            return AnalysisResult(
                atom_positions=np.empty((0, 2)),
                atom_intensities=np.empty(0),
            )

        # 1. Find atoms
        positions, atom_confs, unc_map = self.find_atoms(tile)

        # 2. Classify defects
        defect_labels, defect_confs = self.classify_defects(tile, positions)

        # 3. Identify phase
        phase_label, phase_conf = self.identify_phase(tile)

        # Record tile key for phase correction (must follow identify_phase
        # since _heuristic_phase appends to _phase_intensities)
        if tile_key is not None:
            self._phase_tile_keys.append(tile_key)
            # Store neural model prediction if model was used (not heuristic)
            if self.phase_identifier is not None:
                self._phase_model_predictions[tile_key] = phase_label

        # Count defects
        n_vac = int(np.sum(defect_labels == 1))
        n_sub = int(np.sum(defect_labels == 2))

        # Get intensities for AnalysisResult
        intensities = np.array(
            [tile[int(r), int(c)] for r, c in positions], dtype=np.float32
        ) if len(positions) > 0 else np.empty(0, dtype=np.float32)

        # Convert positions to world nm if tile position given
        if tile_position_nm is not None and len(positions) > 0:
            # positions are (row, col) in tile pixels → (x_nm, y_nm)
            tile_h, tile_w = tile.shape
            world_positions = np.column_stack([
                tile_position_nm[0] + (positions[:, 1] - tile_w / 2) * pixel_size_nm,
                tile_position_nm[1] + (positions[:, 0] - tile_h / 2) * pixel_size_nm,
            ])
            self._all_atom_positions_nm.append(world_positions)

        # Record defect labels
        for lbl in defect_labels:
            self._all_defect_labels.append(DEFECT_CLASSES[lbl])

        result = AnalysisResult(
            atom_positions=positions,
            atom_intensities=intensities,
            n_atoms=len(positions),
            n_vacancies=n_vac,
            n_substitutions=n_sub,
            mean_intensity=float(intensities.mean()) if len(intensities) > 0 else 0.0,
            intensity_std=float(intensities.std()) if len(intensities) > 1 else 0.0,
            phase_label=phase_label,
        )
        self._tile_results.append(result)
        return result

    def get_defect_census(self) -> dict[str, int]:
        total_vac = sum(r.n_vacancies for r in self._tile_results)
        total_sub = sum(r.n_substitutions for r in self._tile_results)
        census = {}
        if total_vac > 0:
            census["vacancy"] = total_vac
        if total_sub > 0:
            census["substitution"] = total_sub
        return census

    def get_total_atoms(self) -> int:
        return sum(r.n_atoms for r in self._tile_results)

    def get_confidence_summary(self) -> dict[str, float]:
        """Summary statistics of analyst confidence."""
        if not self._tile_results:
            return {"mean_atoms_per_tile": 0, "mean_phase_conf": 0}
        return {
            "mean_atoms_per_tile": np.mean([r.n_atoms for r in self._tile_results]),
            "mean_phase_conf": 0.0,
        }

    def get_corrected_tile_phases(self) -> dict[tuple[int, int], int] | None:
        """Return tile-level phase labels.

        When a trained neural phase identifier was used, returns its
        predictions directly (if non-degenerate: at least 2 distinct
        classes observed).

        When only the intensity heuristic was used (no neural model),
        applies a bimodality coefficient (BC) significance test on mean
        tile intensities to guard against false-positive phase maps on
        single-phase worlds.

        Returns:
            dict mapping (row, col) → phase label, or None if phase map
            cannot be determined (single-phase world or too few tiles).
        """
        # Prefer neural model predictions when available
        if self._phase_model_predictions:
            distinct = set(self._phase_model_predictions.values())
            if len(distinct) >= 2:
                return dict(self._phase_model_predictions)
            # Model predicted only one class — single-phase world
            return None

        # Fallback: intensity heuristic with bimodality test
        if len(self._phase_intensities) < 4 or len(self._phase_tile_keys) < 4:
            return None

        intensities = np.array(self._phase_intensities)

        # Bimodality coefficient: BC = (skew² + 1) / kurtosis
        # For normal: BC ≈ 0.33; threshold 5/9 ≈ 0.555 indicates bimodality
        m = intensities.mean()
        centered = intensities - m
        var = np.mean(centered**2)
        if var < 1e-10:
            return None
        std = np.sqrt(var)
        skew = np.mean(centered**3) / std**3
        kurt = np.mean(centered**4) / std**4  # regular (not excess) kurtosis
        if kurt < 1e-10:
            return None

        bc = (skew**2 + 1) / kurt
        if bc < 5 / 9:
            return None

        # Distribution is bimodal — re-label all tiles with global median
        median = float(np.median(intensities))
        result = {}
        for intensity, key in zip(self._phase_intensities, self._phase_tile_keys):
            result[key] = 0 if intensity <= median else 1
        return result

    @property
    def analysed_tiles(self) -> int:
        return len(self._tile_results)

    def get_findings(self, grid_shape: tuple[int, int]) -> dict[str, Any]:
        # Phase map from corrected tile phases (Cohen's d significance test)
        phase_map = None
        corrected = self.get_corrected_tile_phases()
        if corrected is not None:
            gh, gw = grid_shape
            phase_map = np.zeros((gh, gw), dtype=np.int32)
            for (r, c), phase in corrected.items():
                if 0 <= r < gh and 0 <= c < gw:
                    phase_map[r, c] = phase

        # Defect positions from analyst accumulated state
        defect_positions = []
        all_positions = self._all_atom_positions_nm
        all_labels = self._all_defect_labels

        if all_positions and all_labels:
            offset = 0
            for tile_pos in all_positions:
                n = len(tile_pos)
                for j in range(n):
                    label_idx = offset + j
                    if label_idx < len(all_labels):
                        lbl = all_labels[label_idx]
                        if lbl != "pristine":
                            defect_positions.append({
                                "position": (float(tile_pos[j, 0]), float(tile_pos[j, 1])),
                                "type": lbl,
                            })
                offset += n

        # Atom positions
        atom_positions = None
        if all_positions:
            atom_positions = np.concatenate(all_positions, axis=0)

        return {
            "defect_census": self.get_defect_census(),
            "defect_positions": defect_positions,
            "phase_map": phase_map,
            "atom_positions": atom_positions,
        }
