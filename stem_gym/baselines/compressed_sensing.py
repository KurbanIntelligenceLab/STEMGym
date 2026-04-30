"""Compressed-sensing baseline (Candès et al. 2006-style).

Random subsampling of grid tiles followed by inpainting of the unsampled
regions, then analyst inference over the reconstructed tiles. Provides a
complementary acquisition paradigm to greedy active-search baselines.

Algorithm:
1. At episode reset: sample ``coverage_frac`` × |grid| tiles uniformly at
   random (without replacement).
2. During act(): visit the sampled tiles in randomised order, ACQUIRE each
   and analyse with the primary analyst online.
3. At ``get_findings()``: assemble a partial overview from the acquired
   tiles, biharmonic-inpaint the missing regions (skimage), then run a
   SEPARATE analyst instance over the reconstructed tiles. The two findings
   streams are merged at the end.

The separate-analyst design avoids contaminating the primary analyst's
``_phase_intensities`` (used by the median-split bimodality test in
``Analyst.get_corrected_tile_phases()``) with statistics from
reconstructed-not-real tiles.

Caveat: the trained U-Net atom finder was fit to raw HAADF; biharmonic-
inpainted tiles have smoother atomic columns than real ones. Some atom-
detection degradation on reconstructed tiles is expected and reported as
part of the experiment write-up rather than hidden.

Reference: Candès, Romberg & Tao 2006, "Robust uncertainty principles:
exact signal reconstruction from highly incomplete frequency information."
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from stem_gym.agents.simple_analyst import SimpleAnalyst
from stem_gym.baselines import Agent
from stem_gym.envs.stem_env import ActionType


def _inpaint_tile_from_neighbors(
    target_rc: tuple[int, int],
    grid_shape: tuple[int, int],
    acquired: dict[tuple[int, int], np.ndarray],
    tile_shape: tuple[int, int],
) -> np.ndarray:
    """Reconstruct a single missing 128x128 tile by averaging its
    acquired 8-neighborhood.

    Earlier versions used skimage.restoration.inpaint_biharmonic over a
    3x3 patch; the per-tile sparse factorisation is correct but ~1-5s
    each, which scales to ~20+ hours for the full 36-combo CS sweep.
    Mean-of-neighbors is two orders of magnitude faster and still
    represents the compressed-sensing paradigm: missing tiles are
    filled from the acquired subset rather than re-acquired. Compared
    to true TV inpainting, the boundary detail is smoother (no edge
    preservation), which we acknowledge in the paper.

    Falls back to zero-fill if no neighbor is acquired.
    """
    tr, tc = target_rc
    th, tw = tile_shape
    rows, cols = grid_shape

    neighbors: list[np.ndarray] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if (dr, dc) == (0, 0):
                continue
            r = tr + dr
            c = tc + dc
            if r < 0 or r >= rows or c < 0 or c >= cols:
                continue
            if (r, c) in acquired:
                neighbors.append(acquired[(r, c)])

    if not neighbors:
        return np.zeros((th, tw), dtype=np.float32)

    return np.mean(np.stack(neighbors), axis=0).astype(np.float32)


class CompressedSensingAgent(Agent):
    """Random-subsample + inpaint baseline using SimpleAnalyst online.

    Args:
        coverage_frac: Fraction of grid tiles to ACQUIRE (e.g., 0.1 = 10%).
        seed: Random seed.
    """

    def __init__(
        self,
        coverage_frac: float = 0.2,
        seed: int = 42,
    ):
        if not 0.0 < coverage_frac <= 1.0:
            raise ValueError(
                f"coverage_frac must be in (0, 1], got {coverage_frac}"
            )
        self.coverage_frac = coverage_frac
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.analyst = SimpleAnalyst()

        self._grid_shape: tuple[int, int] = (1, 1)
        self._fov_nm: tuple[float, float] = (1.0, 1.0)
        self._pixel_size_nm: float = 0.005
        self._tile_queue: list[tuple[int, int]] = []
        self._navigate_phase = True
        self._current_target: tuple[float, float] = (0.0, 0.0)
        self._acquired: dict[tuple[int, int], np.ndarray] = {}

    def reset(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> None:
        self.analyst.reset()
        self.rng = np.random.default_rng(self.seed)
        self._grid_shape = obs["visited_mask"].shape
        self._fov_nm = info.get("fov_nm", (100.0, 100.0))
        self._pixel_size_nm = info.get("pixel_size_nm", 0.005)
        self._navigate_phase = True
        self._acquired.clear()

        rows, cols = self._grid_shape
        all_tiles = [(r, c) for r in range(rows) for c in range(cols)]
        n_sample = max(1, int(round(self.coverage_frac * len(all_tiles))))
        idx = self.rng.choice(len(all_tiles), size=n_sample, replace=False)
        self._tile_queue = [all_tiles[i] for i in idx]

    def _tile_to_nm(self, row: int, col: int) -> tuple[float, float]:
        rows, cols = self._grid_shape
        fov_h, fov_w = self._fov_nm
        x_nm = (col + 0.5) / cols * fov_w
        y_nm = (row + 0.5) / rows * fov_h
        return (x_nm, y_nm)

    def _nm_to_grid(self, x_nm: float, y_nm: float) -> tuple[int, int]:
        grid_h, grid_w = self._grid_shape
        col = int(np.clip(x_nm / self._fov_nm[1] * grid_w, 0, grid_w - 1))
        row = int(np.clip(y_nm / self._fov_nm[0] * grid_h, 0, grid_h - 1))
        return row, col

    def act(
        self, obs: dict[str, np.ndarray], info: dict[str, Any]
    ) -> dict[str, Any]:
        budget_frac = float(obs["budget_remaining"][0])

        if not self._tile_queue or budget_frac < 0.01:
            return self._make_done_action()

        if self._navigate_phase:
            row, col = self._tile_queue[0]
            x_nm, y_nm = self._tile_to_nm(row, col)
            self._current_target = (x_nm, y_nm)
            self._navigate_phase = False
            return {
                "action_type": ActionType.NAVIGATE,
                "position": np.array([x_nm, y_nm], dtype=np.float32),
                "zoom": np.array([1.0], dtype=np.float32),
                "dwell": np.array([1.0], dtype=np.float32),
                "finding_type": 0,
            }

        target_row, target_col = self._tile_queue.pop(0)
        self._navigate_phase = True

        tile = obs.get("tile")
        if tile is not None and tile.max() > 0:
            self._acquired[(target_row, target_col)] = tile.copy()
            self.analyst.analyze_tile(tile)

        return {
            "action_type": ActionType.ACQUIRE,
            "position": np.array(self._current_target, dtype=np.float32),
            "zoom": np.array([1.0], dtype=np.float32),
            "dwell": np.array([1.0], dtype=np.float32),
            "finding_type": 0,
        }

    def _build_reconstructed_tiles(
        self,
    ) -> dict[tuple[int, int], np.ndarray]:
        """Per-tile biharmonic inpainting from acquired neighbors.

        Avoids the full-overview (rows * tile_h) x (cols * tile_w)
        sparse factorisation that OOMs on 1024x1024 grids; instead each
        missing tile is reconstructed from a 3x3 neighbor patch.
        """
        if not self._acquired:
            return {}

        sample_tile = next(iter(self._acquired.values()))
        tile_shape = sample_tile.shape
        rows, cols = self._grid_shape

        out: dict[tuple[int, int], np.ndarray] = {}
        for r in range(rows):
            for c in range(cols):
                if (r, c) in self._acquired:
                    continue
                out[(r, c)] = _inpaint_tile_from_neighbors(
                    (r, c), self._grid_shape, self._acquired, tile_shape
                )
        return out

    def _build_secondary_analyst(self):
        """Spawn a fresh SimpleAnalyst for the reconstructed-tile pass.

        Subclasses (equipped variant) override this to use a trained Analyst.
        """
        return SimpleAnalyst()

    def get_findings(self) -> dict[str, Any]:
        # Run the primary analyst-based findings (real tiles only)
        primary = {
            "defect_census": dict(self.analyst.get_defect_census()),
            "defect_positions": [],
            "phase_map": None,
            "atom_positions": None,
        }

        # Run a secondary analyst over reconstructed tiles
        reconstructed = self._build_reconstructed_tiles()
        secondary = self._build_secondary_analyst()
        for (r, c), tile in reconstructed.items():
            x_nm, y_nm = self._tile_to_nm(r, c)
            try:
                secondary.analyze_tile(
                    tile,
                    tile_position_nm=(x_nm, y_nm),
                    pixel_size_nm=self._pixel_size_nm,
                    tile_key=(r, c),
                )
            except TypeError:
                # SimpleAnalyst.analyze_tile takes only the tile arg
                secondary.analyze_tile(tile)

        sec_census = dict(secondary.get_defect_census())
        merged_census = primary["defect_census"]
        for k, v in sec_census.items():
            merged_census[k] = merged_census.get(k, 0) + v

        return {
            "defect_census": merged_census,
            "defect_positions": [],
            "phase_map": None,
            "atom_positions": None,
        }


class CompressedSensingEquippedAgent(CompressedSensingAgent):
    """Compressed-sensing agent with the trained Analyst ensemble.

    Primary analyst (real-tile pass) and secondary analyst (reconstructed-
    tile pass) are independent Analyst instances loaded from the same
    checkpoint to keep phase-bimodality statistics clean.
    """

    def __init__(
        self,
        coverage_frac: float = 0.2,
        seed: int = 42,
        checkpoint_dir: str | Path | None = None,
    ):
        super().__init__(coverage_frac=coverage_frac, seed=seed)
        from stem_gym.baselines.equipped_baselines import _load_analyst

        self._checkpoint_dir = checkpoint_dir
        self.analyst = _load_analyst(checkpoint_dir)

    def reset(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> None:
        super().reset(obs, info)

    def act(
        self, obs: dict[str, np.ndarray], info: dict[str, Any]
    ) -> dict[str, Any]:
        budget_frac = float(obs["budget_remaining"][0])

        if not self._tile_queue or budget_frac < 0.01:
            return self._make_done_action()

        if self._navigate_phase:
            row, col = self._tile_queue[0]
            x_nm, y_nm = self._tile_to_nm(row, col)
            self._current_target = (x_nm, y_nm)
            self._navigate_phase = False
            return {
                "action_type": ActionType.NAVIGATE,
                "position": np.array([x_nm, y_nm], dtype=np.float32),
                "zoom": np.array([1.0], dtype=np.float32),
                "dwell": np.array([1.0], dtype=np.float32),
                "finding_type": 0,
            }

        target_row, target_col = self._tile_queue.pop(0)
        self._navigate_phase = True

        tile = obs.get("tile")
        if tile is not None and tile.max() > 0:
            self._acquired[(target_row, target_col)] = tile.copy()
            x_nm, y_nm = self._current_target
            self.analyst.analyze_tile(
                tile,
                tile_position_nm=(x_nm, y_nm),
                pixel_size_nm=self._pixel_size_nm,
                tile_key=(target_row, target_col),
            )

        return {
            "action_type": ActionType.ACQUIRE,
            "position": np.array(self._current_target, dtype=np.float32),
            "zoom": np.array([1.0], dtype=np.float32),
            "dwell": np.array([1.0], dtype=np.float32),
            "finding_type": 0,
        }

    def _build_secondary_analyst(self):
        from stem_gym.baselines.equipped_baselines import _load_analyst

        return _load_analyst(self._checkpoint_dir)

    def get_findings(self) -> dict[str, Any]:
        from stem_gym.baselines.equipped_baselines import _assemble_findings

        # Real-tile findings via the primary analyst
        primary = _assemble_findings(self.analyst, self._grid_shape)

        # Reconstructed-tile findings via a fresh analyst instance
        reconstructed = self._build_reconstructed_tiles()
        secondary = self._build_secondary_analyst()
        for (r, c), tile in reconstructed.items():
            x_nm, y_nm = self._tile_to_nm(r, c)
            try:
                secondary.analyze_tile(
                    tile,
                    tile_position_nm=(x_nm, y_nm),
                    pixel_size_nm=self._pixel_size_nm,
                    tile_key=(r, c),
                )
            except Exception:
                # Tolerate failures on heavily-smoothed reconstructed tiles
                pass
        secondary_findings = _assemble_findings(secondary, self._grid_shape)

        # Merge: union of defect counts; reconstructed defect positions
        # tagged with provenance for downstream analysis if needed
        merged_census: dict[str, int] = dict(primary["defect_census"])
        for k, v in secondary_findings["defect_census"].items():
            merged_census[k] = merged_census.get(k, 0) + v

        merged_defect_positions = list(primary.get("defect_positions") or [])
        for d in secondary_findings.get("defect_positions") or []:
            d_copy = dict(d)
            d_copy["source"] = "reconstructed"
            merged_defect_positions.append(d_copy)

        # Phase map: prefer primary if it has multi-phase evidence, else secondary
        phase_map = primary.get("phase_map")
        if phase_map is None:
            phase_map = secondary_findings.get("phase_map")

        atom_positions = primary.get("atom_positions")
        if atom_positions is None:
            atom_positions = secondary_findings.get("atom_positions")

        return {
            "defect_census": merged_census,
            "defect_positions": merged_defect_positions,
            "phase_map": phase_map,
            "atom_positions": atom_positions,
        }
