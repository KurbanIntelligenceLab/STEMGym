"""Random baseline: random positions + default acquisition params.

Lower bound on performance. Acquires at random positions with no
strategy. Uses SimpleAnalyst to produce findings for scoring.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from stem_gym.baselines import Agent
from stem_gym.agents.simple_analyst import SimpleAnalyst
from stem_gym.envs.stem_env import ActionType


class RandomAgent(Agent):
    """Random baseline agent.

    Each step: navigate to a random position, then acquire.
    After all acquisitions, declares findings based on SimpleAnalyst.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.analyst = SimpleAnalyst()
        self._fov: tuple[float, float] = (1.0, 1.0)
        self._step = 0
        self._navigate_phase = True  # alternate navigate → acquire

    def reset(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> None:
        self.analyst.reset()
        self._step = 0
        self._navigate_phase = True
        self._grid_shape = obs["visited_mask"].shape
        self._fov = info.get("fov_nm", (100.0, 100.0))

    def act(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> dict[str, Any]:
        self._step += 1
        budget_frac = float(obs["budget_remaining"][0])

        # If nearly out of budget, declare and stop
        if budget_frac < 0.05:
            return self._make_done_action()

        if self._navigate_phase:
            # Navigate to random position
            self._navigate_phase = False
            # Random position within the FOV
            fov_h, fov_w = self._fov
            x = self.rng.uniform(0, fov_w)
            y = self.rng.uniform(0, fov_h)
            return {
                "action_type": ActionType.NAVIGATE,
                "position": np.array([x, y], dtype=np.float32),
                "zoom": np.array([1.0], dtype=np.float32),
                "dwell": np.array([1.0], dtype=np.float32),
                "finding_type": 0,
            }
        else:
            # Acquire at current position
            self._navigate_phase = True
            # Analyze the previous tile if we have one
            tile = obs.get("tile")
            if tile is not None and tile.max() > 0:
                self.analyst.analyze_tile(tile)
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
        """Get accumulated findings for task scoring."""
        return {
            "defect_census": self.analyst.get_defect_census(),
            "defect_positions": [],
            "phase_map": None,
            "atom_positions": None,
        }
