from __future__ import annotations

from typing import Any

import numpy as np

from stem_gym.baselines import Agent
from stem_gym.agents.simple_analyst import SimpleAnalyst
from stem_gym.envs.stem_env import ActionType


class RasterAgent(Agent):


    def __init__(self, seed: int = 42):
        self.analyst = SimpleAnalyst()
        self._grid_shape: tuple[int, int] = (1, 1)
        self._fov_nm: tuple[float, float] = (1.0, 1.0)
        self._tile_queue: list[tuple[int, int]] = []
        self._navigate_phase = True
        self._current_target: tuple[float, float] = (0.0, 0.0)

    def reset(self, obs: dict[str, np.ndarray], info: dict[str, Any], *, seed: int | None = None) -> None:
        self.analyst.reset()
        self._observed_acquisitions = 0
        self._geometry = info.get("geometry")
        self._grid_shape = obs["visited_mask"].shape
        self._fov_nm = info.get("fov_nm", (100.0, 100.0))
        self._navigate_phase = True


        rows, cols = self._grid_shape
        self._tile_queue = [(r, c) for r in range(rows) for c in range(cols)]

    def _tile_to_nm(self, row: int, col: int) -> tuple[float, float]:
        """Convert tile grid position to nm coordinates."""
        if self._geometry is not None:
            return self._geometry.tile_to_nm(row, col)
        rows, cols = self._grid_shape
        fov_h, fov_w = self._fov_nm

        x_nm = (col + 0.5) / cols * fov_w
        y_nm = (row + 0.5) / rows * fov_h
        return (x_nm, y_nm)

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

            return {
                "action_type": ActionType.ACQUIRE,
                "position": np.array(self._current_target, dtype=np.float32),
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
