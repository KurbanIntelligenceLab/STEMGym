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

from stem_gym.baselines import Agent
from stem_gym.agents.analyst import Analyst
from stem_gym.baselines.gp_ucb import GPUCBAgent
from stem_gym.baselines.single_bo import SingleBOAgent
from stem_gym.envs.stem_env import ActionType


# Default checkpoint path — same as STEMAgent
_DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"


def _assemble_findings(
    analyst: Analyst,
    grid_shape: tuple[int, int],
) -> dict[str, Any]:
    """Build findings dict from analyst state and corrected phase labels.

    Shared by both equipped agents — same logic as STEMAgent.get_findings().
    """
    # Phase map from corrected tile phases (Cohen's d significance test)
    phase_map = None
    corrected = analyst.get_corrected_tile_phases()
    if corrected is not None:
        gh, gw = grid_shape
        phase_map = np.zeros((gh, gw), dtype=np.int32)
        for (r, c), phase in corrected.items():
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


class GPUCBEquippedAgent(GPUCBAgent):
    """GP-UCB navigation with the trained Analyst ensemble.

    Inherits all navigation/GP logic from GPUCBAgent. Overrides analyst
    initialization (trained vs simple) and tile analysis (passes position
    for world-coordinate defect localization).
    """

    def __init__(self, beta=2.0, seed=42, initial_random=3, checkpoint_dir=None, acq_func="ucb"):
        super().__init__(beta=beta, seed=seed, initial_random=initial_random, acq_func=acq_func)
        self.analyst = _load_analyst(checkpoint_dir)
        self._pixel_size_nm: float = 0.005

    def reset(self, obs, info):
        super().reset(obs, info)
        self._pixel_size_nm = info.get("pixel_size_nm", 0.005)
        self.rng = np.random.default_rng(self.seed)

    def act(self, obs, info):
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

                row, col = self._nm_to_grid(x_nm, y_nm)
                result = self.analyst.analyze_tile(
                    tile, tile_position_nm=(x_nm, y_nm),
                    pixel_size_nm=self._pixel_size_nm,
                    tile_key=(row, col),
                )

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

    def get_findings(self):
        return _assemble_findings(self.analyst, self._grid_shape)


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
        zoom: float = 1.0,
        dwell: float = 1.0,
    ):
        self.analyst = _load_analyst(checkpoint_dir)
        self._zoom = zoom
        self._dwell = dwell
        self._grid_shape: tuple[int, int] = (1, 1)
        self._fov_nm: tuple[float, float] = (1.0, 1.0)
        self._pixel_size_nm: float = 0.005
        self._tile_queue: list[tuple[int, int]] = []
        self._navigate_phase = True
        self._current_target: tuple[float, float] = (0.0, 0.0)

    def reset(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> None:
        self.analyst.reset()
        self._grid_shape = obs["visited_mask"].shape
        self._fov_nm = info.get("fov_nm", (100.0, 100.0))
        self._pixel_size_nm = info.get("pixel_size_nm", 0.005)
        self._navigate_phase = True

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
                "zoom": np.array([self._zoom], dtype=np.float32),
                "dwell": np.array([self._dwell], dtype=np.float32),
                "finding_type": 0,
            }
        else:
            self._tile_queue.pop(0)
            self._navigate_phase = True

            tile = obs.get("tile")
            if tile is not None and tile.max() > 0:
                x_nm, y_nm = self._current_target

                # Analyze with trained analyst (with position for world coords)
                row, col = self._nm_to_grid(x_nm, y_nm)
                self.analyst.analyze_tile(
                    tile,
                    tile_position_nm=(x_nm, y_nm),
                    pixel_size_nm=self._pixel_size_nm,
                    tile_key=(row, col),
                )

            return {
                "action_type": ActionType.ACQUIRE,
                "position": np.array(self._current_target, dtype=np.float32),
                "zoom": np.array([self._zoom], dtype=np.float32),
                "dwell": np.array([self._dwell], dtype=np.float32),
                "finding_type": 0,
            }

    def get_findings(self) -> dict[str, Any]:
        return _assemble_findings(self.analyst, self._grid_shape)


# ---------------------------------------------------------------------------
# Random navigation + trained Analyst
# ---------------------------------------------------------------------------


class RandomEquippedAgent(Agent):
    """Random navigation with the trained Analyst ensemble.

    Uniform random position selection identical to RandomAgent, but using the
    trained Analyst instead of SimpleAnalyst for tile analysis.
    """

    def __init__(
        self,
        seed: int = 42,
        checkpoint_dir: str | Path | None = None,
    ):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.analyst = _load_analyst(checkpoint_dir)
        self._grid_shape: tuple[int, int] = (1, 1)
        self._fov_nm: tuple[float, float] = (1.0, 1.0)
        self._pixel_size_nm: float = 0.005
        self._navigate_phase = True

    def reset(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> None:
        self.analyst.reset()
        self._grid_shape = obs["visited_mask"].shape
        self._fov_nm = info.get("fov_nm", (100.0, 100.0))
        self._pixel_size_nm = info.get("pixel_size_nm", 0.005)
        self._navigate_phase = True

    def _nm_to_grid(self, x_nm: float, y_nm: float) -> tuple[int, int]:
        """Convert nm coordinates to grid indices."""
        grid_h, grid_w = self._grid_shape
        col = int(np.clip(x_nm / self._fov_nm[1] * grid_w, 0, grid_w - 1))
        row = int(np.clip(y_nm / self._fov_nm[0] * grid_h, 0, grid_h - 1))
        return row, col

    def act(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> dict[str, Any]:
        budget_frac = float(obs["budget_remaining"][0])

        if budget_frac < 0.01:
            return self._make_done_action()

        if self._navigate_phase:
            fov_h, fov_w = self._fov_nm
            x = self.rng.uniform(0, fov_w)
            y = self.rng.uniform(0, fov_h)
            self._navigate_phase = False
            return {
                "action_type": ActionType.NAVIGATE,
                "position": np.array([x, y], dtype=np.float32),
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
                row, col = self._nm_to_grid(x_nm, y_nm)
                self.analyst.analyze_tile(
                    tile,
                    tile_position_nm=(x_nm, y_nm),
                    pixel_size_nm=self._pixel_size_nm,
                    tile_key=(row, col),
                )

            return {
                "action_type": ActionType.ACQUIRE,
                "position": obs["position"].copy(),
                "zoom": np.array([1.0], dtype=np.float32),
                "dwell": np.array([1.0], dtype=np.float32),
                "finding_type": 0,
            }

    def get_findings(self) -> dict[str, Any]:
        return _assemble_findings(self.analyst, self._grid_shape)


# ---------------------------------------------------------------------------
# Single-BO + trained Analyst
# ---------------------------------------------------------------------------


class SingleBOEquippedAgent(SingleBOAgent):
    """Joint BO over (x, y, zoom, dwell) with the trained Analyst ensemble.

    Inherits all BO logic from SingleBOAgent. Replaces the SimpleAnalyst
    with the trained Analyst and adds position-aware tile analysis.
    """

    def __init__(
        self,
        seed: int = 42,
        initial_random: int = 6,
        checkpoint_dir: str | Path | None = None,
    ):
        super().__init__(seed=seed, initial_random=initial_random)
        self.analyst = _load_analyst(checkpoint_dir)
        self._pixel_size_nm: float = 0.005

    def reset(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> None:
        super().reset(obs, info)
        self._pixel_size_nm = info.get("pixel_size_nm", 0.005)

    def _nm_to_grid(self, x_nm: float, y_nm: float) -> tuple[int, int]:
        """Convert nm coordinates to grid indices."""
        grid_h, grid_w = self._grid_shape
        col = int(np.clip(x_nm / self._fov_nm[1] * grid_w, 0, grid_w - 1))
        row = int(np.clip(y_nm / self._fov_nm[0] * grid_h, 0, grid_h - 1))
        return row, col

    def act(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> dict[str, Any]:
        self._step += 1
        budget_frac = float(obs["budget_remaining"][0])

        if budget_frac < 0.01:
            return self._make_done_action()

        if self._phase == "navigate":
            x_n, y_n, zoom_n, dwell_n = self._select_next(obs)
            x_nm, y_nm, zoom, dwell = self._denormalize(x_n, y_n, zoom_n, dwell_n)
            self._next_params = (x_n, y_n, zoom_n, dwell_n)
            self._phase = "acquire"

            return {
                "action_type": ActionType.NAVIGATE,
                "position": np.array([x_nm, y_nm], dtype=np.float32),
                "zoom": np.array([zoom], dtype=np.float32),
                "dwell": np.array([dwell], dtype=np.float32),
                "finding_type": 0,
            }
        else:
            self._phase = "navigate"

            tile = obs.get("tile")
            interestingness = 0.0
            if tile is not None and tile.max() > 0:
                pos = obs["position"]
                x_nm, y_nm = float(pos[0]), float(pos[1])
                row, col = self._nm_to_grid(x_nm, y_nm)
                self.analyst.analyze_tile(
                    tile,
                    tile_position_nm=(x_nm, y_nm),
                    pixel_size_nm=self._pixel_size_nm,
                    tile_key=(row, col),
                )
                interestingness = float(np.std(tile))

            if self._next_params is not None:
                self._X.append(list(self._next_params))
                self._Y.append(interestingness)

            if self._next_params:
                _, _, zoom_n, dwell_n = self._next_params
                zoom = 1.0 + zoom_n * 3.0
                dwell = 0.1 + dwell_n * 9.9
            else:
                zoom, dwell = 1.0, 1.0

            return {
                "action_type": ActionType.ACQUIRE,
                "position": obs["position"].copy(),
                "zoom": np.array([zoom], dtype=np.float32),
                "dwell": np.array([dwell], dtype=np.float32),
                "finding_type": 0,
            }

    def get_findings(self) -> dict[str, Any]:
        return _assemble_findings(self.analyst, self._grid_shape)
