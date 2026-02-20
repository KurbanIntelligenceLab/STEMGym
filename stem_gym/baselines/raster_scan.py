"""Raster scan baseline: systematic grid traversal.

Visits every tile in row-major order. Represents naive automation —
thorough but dose-inefficient. Should achieve high final coverage
but waste budget on uninteresting regions.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from stem_gym.baselines import Agent
from stem_gym.agents.simple_analyst import SimpleAnalyst
from stem_gym.envs.stem_env import ActionType


class RasterAgent(Agent):
    """Raster scan agent: visits tiles in row-major order.

    Alternates navigate → acquire for each tile position.
    Uses SimpleAnalyst to accumulate findings.
    """

    def __init__(self, seed: int = 42):
        self.analyst = SimpleAnalyst()
        self._grid_shape: tuple[int, int] = (1, 1)
        self._fov_nm: tuple[float, float] = (1.0, 1.0)
        self._tile_queue: list[tuple[int, int]] = []
        self._navigate_phase = True
        self._current_target: tuple[float, float] = (0.0, 0.0)

    def reset(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> None:
        self.analyst.reset()
        self._grid_shape = obs["visited_mask"].shape
        self._fov_nm = info.get("fov_nm", (100.0, 100.0))
        self._navigate_phase = True

        # Build row-major tile queue
        rows, cols = self._grid_shape
        self._tile_queue = [(r, c) for r in range(rows) for c in range(cols)]

    def _tile_to_nm(self, row: int, col: int) -> tuple[float, float]:
        """Convert tile grid position to nm coordinates."""
        rows, cols = self._grid_shape
        fov_h, fov_w = self._fov_nm
        # Map grid position to FOV center of each tile
        x_nm = (col + 0.5) / cols * fov_w
        y_nm = (row + 0.5) / rows * fov_h
        return (x_nm, y_nm)

    def act(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> dict[str, Any]:
        budget_frac = float(obs["budget_remaining"][0])

        # If out of tiles or budget, stop
        if not self._tile_queue or budget_frac < 0.01:
            return self._make_done_action()

        if self._navigate_phase:
            # Navigate to next tile
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
            # Acquire at current position
            self._tile_queue.pop(0)
            self._navigate_phase = True

            # Analyze the tile from observation
            tile = obs.get("tile")
            if tile is not None and tile.max() > 0:
                self.analyst.analyze_tile(tile)

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
        return {
            "defect_census": self.analyst.get_defect_census(),
            "defect_positions": [],
            "phase_map": None,
            "atom_positions": None,
        }
