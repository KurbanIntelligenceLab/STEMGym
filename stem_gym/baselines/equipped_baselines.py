"""Equipped baselines: baseline navigation + trained Analyst.

Fair comparison agents that pair simple navigation strategies (GP-UCB,
raster scan) with the same trained Analyst ensemble used by STEMAgent.
This isolates the contribution of the planner/navigation coordination
from the perception advantage.

GPUCBEquippedAgent: GP-UCB navigation (identical to GPUCBAgent) + trained Analyst.
RasterEquippedAgent: Row-major raster scan + trained Analyst.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from stem_gym.baselines import Agent
from stem_gym.agents.analyst import Analyst
from stem_gym.envs.stem_env import ActionType


# Default checkpoint path — same as STEMAgent
_DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"


def _assemble_findings(
    analyst: Analyst,
    tile_phases: dict[tuple[int, int], int],
    grid_shape: tuple[int, int],
) -> dict[str, Any]:
    """Build findings dict from analyst state and tile phase labels.

    Shared by both equipped agents — same logic as STEMAgent.get_findings().
    """
    # Phase map
    phase_map = None
    if tile_phases:
        gh, gw = grid_shape
        phase_map = np.zeros((gh, gw), dtype=np.int32)
        for (r, c), phase in tile_phases.items():
            if 0 <= r < gh and 0 <= c < gw:
                phase_map[r, c] = phase

    # Defect positions from analyst accumulated state
    defect_positions = []
    all_positions = analyst._all_atom_positions_nm
    all_labels = analyst._all_defect_labels

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
        "defect_census": analyst.get_defect_census(),
        "defect_positions": defect_positions,
        "phase_map": phase_map,
        "atom_positions": atom_positions,
    }


def _load_analyst(checkpoint_dir: str | Path | None) -> Analyst:
    """Load trained Analyst from checkpoints, fallback to untrained."""
    if checkpoint_dir is not None:
        cp = Path(checkpoint_dir)
    else:
        cp = _DEFAULT_CHECKPOINT_DIR
    if cp.exists() and (cp / "atom_finder.pt").exists():
        try:
            return Analyst.from_checkpoint(cp)
        except Exception:
            pass
    return Analyst()


# ---------------------------------------------------------------------------
# GP-UCB + trained Analyst
# ---------------------------------------------------------------------------


class GPUCBEquippedAgent(Agent):
    """GP-UCB navigation with the trained Analyst ensemble.

    Navigation logic is identical to GPUCBAgent: GP-UCB acquisition
    function over tile interestingness (intensity std), fixed beta=2.0.
    The only difference is using the trained Analyst instead of SimpleAnalyst,
    with tile_position_nm passed for world-coordinate defect localization.
    """

    def __init__(
        self,
        beta: float = 2.0,
        seed: int = 42,
        initial_random: int = 3,
        checkpoint_dir: str | Path | None = None,
    ):
        self.beta = beta
        self.seed = seed
        self.initial_random = initial_random
        self.analyst = _load_analyst(checkpoint_dir)
        self.rng = np.random.default_rng(seed)

        # GP data
        self._X: list[list[float]] = []
        self._Y: list[float] = []
        self._grid_shape: tuple[int, int] = (1, 1)
        self._fov_nm: tuple[float, float] = (1.0, 1.0)
        self._pixel_size_nm: float = 0.005
        self._navigate_phase = True
        self._step = 0
        self._gp_model = None
        self._visited_set: set[tuple[int, int]] = set()
        self._tile_phases: dict[tuple[int, int], int] = {}

    def reset(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> None:
        self.analyst.reset()
        self._X.clear()
        self._Y.clear()
        self._grid_shape = obs["visited_mask"].shape
        self._fov_nm = info.get("fov_nm", (100.0, 100.0))
        self._pixel_size_nm = info.get("pixel_size_nm", 0.005)
        self._navigate_phase = True
        self._step = 0
        self._gp_model = None
        self._visited_set.clear()
        self._tile_phases.clear()
        self.rng = np.random.default_rng(self.seed)

    def _fit_gp(self) -> None:
        """Fit GP model to observed data."""
        if len(self._X) < 2:
            return

        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood

        train_X = torch.tensor(self._X, dtype=torch.float64)
        train_Y = torch.tensor(self._Y, dtype=torch.float64).unsqueeze(-1)

        y_mean = train_Y.mean()
        y_std = train_Y.std().clamp(min=1e-6)
        train_Y_norm = (train_Y - y_mean) / y_std

        try:
            self._gp_model = SingleTaskGP(train_X, train_Y_norm)
            mll = ExactMarginalLogLikelihood(
                self._gp_model.likelihood, self._gp_model
            )
            fit_gpytorch_mll(mll)
        except Exception:
            self._gp_model = SingleTaskGP(train_X, train_Y_norm)

    def _select_next_position(self, obs: dict[str, np.ndarray]) -> tuple[float, float]:
        """Select next position using GP-UCB."""
        grid_h, grid_w = self._grid_shape
        visited = obs["visited_mask"]

        if self._gp_model is None or self._step < self.initial_random:
            unvisited = np.argwhere(visited < 0.5)
            if len(unvisited) == 0:
                unvisited = np.argwhere(np.ones_like(visited))
            idx = self.rng.integers(0, len(unvisited))
            row, col = unvisited[idx]
            return self._grid_to_nm(int(row), int(col))

        candidates = []
        for r in range(grid_h):
            for c in range(grid_w):
                if (r, c) not in self._visited_set:
                    candidates.append((r, c))

        if not candidates:
            r = self.rng.integers(0, grid_h)
            c = self.rng.integers(0, grid_w)
            return self._grid_to_nm(r, c)

        cand_norm = torch.tensor(
            [[r / max(grid_h - 1, 1), c / max(grid_w - 1, 1)] for r, c in candidates],
            dtype=torch.float64,
        )

        self._gp_model.eval()
        with torch.no_grad():
            posterior = self._gp_model.posterior(cand_norm)
            mean = posterior.mean.squeeze(-1)
            std = posterior.variance.sqrt().squeeze(-1)

        ucb = mean + self.beta * std
        best_idx = ucb.argmax().item()
        row, col = candidates[best_idx]
        return self._grid_to_nm(row, col)

    def _grid_to_nm(self, row: int, col: int) -> tuple[float, float]:
        """Convert grid indices to nm coordinates."""
        grid_h, grid_w = self._grid_shape
        x_nm = (col + 0.5) / grid_w * self._fov_nm[1]
        y_nm = (row + 0.5) / grid_h * self._fov_nm[0]
        return (x_nm, y_nm)

    def _nm_to_grid(self, x_nm: float, y_nm: float) -> tuple[int, int]:
        """Convert nm coordinates to grid indices."""
        grid_h, grid_w = self._grid_shape
        col = int(np.clip(x_nm / self._fov_nm[1] * grid_w, 0, grid_w - 1))
        row = int(np.clip(y_nm / self._fov_nm[0] * grid_h, 0, grid_h - 1))
        return row, col

    @staticmethod
    def _tile_interestingness(tile: np.ndarray) -> float:
        """Intensity std as GP objective — identical to GPUCBAgent."""
        if tile.max() <= 0:
            return 0.0
        return float(np.std(tile))

    def act(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> dict[str, Any]:
        self._step += 1
        budget_frac = float(obs["budget_remaining"][0])

        if budget_frac < 0.01:
            return self._make_done_action()

        if self._navigate_phase:
            x_nm, y_nm = self._select_next_position(obs)
            self._navigate_phase = False
            return {
                "action_type": ActionType.NAVIGATE,
                "position": np.array([x_nm, y_nm], dtype=np.float32),
                "zoom": np.array([1.0], dtype=np.float32),
                "dwell": np.array([1.0], dtype=np.float32),
                "finding_type": 0,
            }
        else:
            self._navigate_phase = True

            tile = obs.get("tile")
            if tile is not None and tile.max() > 0:
                pos = obs["position"]
                x_nm, y_nm = float(pos[0]), float(pos[1])

                # Analyze with trained analyst (with position for world coords)
                result = self.analyst.analyze_tile(
                    tile,
                    tile_position_nm=(x_nm, y_nm),
                    pixel_size_nm=self._pixel_size_nm,
                )

                # Record phase label for phase map
                row, col = self._nm_to_grid(x_nm, y_nm)
                self._tile_phases[(row, col)] = result.phase_label

                # Feed interestingness to GP (same objective as GPUCBAgent)
                interestingness = self._tile_interestingness(tile)
                self._visited_set.add((row, col))

                grid_h, grid_w = self._grid_shape
                x_norm = row / max(grid_h - 1, 1)
                y_norm = col / max(grid_w - 1, 1)
                self._X.append([x_norm, y_norm])
                self._Y.append(interestingness)

                if len(self._X) >= self.initial_random and len(self._X) % 3 == 0:
                    self._fit_gp()

            return {
                "action_type": ActionType.ACQUIRE,
                "position": obs["position"].copy(),
                "zoom": np.array([1.0], dtype=np.float32),
                "dwell": np.array([1.0], dtype=np.float32),
                "finding_type": 0,
            }

    def _make_done_action(self) -> dict[str, Any]:
        return {
            "action_type": ActionType.DONE,
            "position": np.array([0.0, 0.0], dtype=np.float32),
            "zoom": np.array([1.0], dtype=np.float32),
            "dwell": np.array([1.0], dtype=np.float32),
            "finding_type": 0,
        }

    def get_findings(self) -> dict[str, Any]:
        return _assemble_findings(self.analyst, self._tile_phases, self._grid_shape)


# ---------------------------------------------------------------------------
# Raster scan + trained Analyst
# ---------------------------------------------------------------------------


class RasterEquippedAgent(Agent):
    """Raster scan navigation with the trained Analyst ensemble.

    Row-major grid traversal identical to RasterAgent, but using the
    trained Analyst instead of SimpleAnalyst for tile analysis.
    """

    def __init__(
        self,
        seed: int = 42,
        checkpoint_dir: str | Path | None = None,
    ):
        self.analyst = _load_analyst(checkpoint_dir)
        self._grid_shape: tuple[int, int] = (1, 1)
        self._fov_nm: tuple[float, float] = (1.0, 1.0)
        self._pixel_size_nm: float = 0.005
        self._tile_queue: list[tuple[int, int]] = []
        self._navigate_phase = True
        self._current_target: tuple[float, float] = (0.0, 0.0)
        self._tile_phases: dict[tuple[int, int], int] = {}

    def reset(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> None:
        self.analyst.reset()
        self._grid_shape = obs["visited_mask"].shape
        self._fov_nm = info.get("fov_nm", (100.0, 100.0))
        self._pixel_size_nm = info.get("pixel_size_nm", 0.005)
        self._navigate_phase = True
        self._tile_phases.clear()

        # Build row-major tile queue
        rows, cols = self._grid_shape
        self._tile_queue = [(r, c) for r in range(rows) for c in range(cols)]

    def _tile_to_nm(self, row: int, col: int) -> tuple[float, float]:
        """Convert tile grid position to nm coordinates."""
        rows, cols = self._grid_shape
        fov_h, fov_w = self._fov_nm
        x_nm = (col + 0.5) / cols * fov_w
        y_nm = (row + 0.5) / rows * fov_h
        return (x_nm, y_nm)

    def _nm_to_grid(self, x_nm: float, y_nm: float) -> tuple[int, int]:
        """Convert nm coordinates to grid indices."""
        grid_h, grid_w = self._grid_shape
        col = int(np.clip(x_nm / self._fov_nm[1] * grid_w, 0, grid_w - 1))
        row = int(np.clip(y_nm / self._fov_nm[0] * grid_h, 0, grid_h - 1))
        return row, col

    def act(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> dict[str, Any]:
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
        else:
            self._tile_queue.pop(0)
            self._navigate_phase = True

            tile = obs.get("tile")
            if tile is not None and tile.max() > 0:
                x_nm, y_nm = self._current_target

                # Analyze with trained analyst (with position for world coords)
                result = self.analyst.analyze_tile(
                    tile,
                    tile_position_nm=(x_nm, y_nm),
                    pixel_size_nm=self._pixel_size_nm,
                )

                # Record phase label for phase map
                row, col = self._nm_to_grid(x_nm, y_nm)
                self._tile_phases[(row, col)] = result.phase_label

            return {
                "action_type": ActionType.ACQUIRE,
                "position": np.array(self._current_target, dtype=np.float32),
                "zoom": np.array([1.0], dtype=np.float32),
                "dwell": np.array([1.0], dtype=np.float32),
                "finding_type": 0,
            }

    def _make_done_action(self) -> dict[str, Any]:
        return {
            "action_type": ActionType.DONE,
            "position": np.array([0.0, 0.0], dtype=np.float32),
            "zoom": np.array([1.0], dtype=np.float32),
            "dwell": np.array([1.0], dtype=np.float32),
            "finding_type": 0,
        }

    def get_findings(self) -> dict[str, Any]:
        return _assemble_findings(self.analyst, self._tile_phases, self._grid_shape)
