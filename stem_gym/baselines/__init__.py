"""Baseline agents for STEMGym.

All baselines implement the Agent protocol:
    agent.reset(obs, info) -> None
    agent.act(obs, info) -> action dict
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Agent(ABC):
    """Base class for STEMGym agents (baselines and multi-agent)."""

    @abstractmethod
    def reset(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> None:
        """Called at episode start after env.reset()."""

    @abstractmethod
    def act(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> dict[str, Any]:
        """Select an action given current observation.

        Returns:
            Action dict compatible with STEMEnv.action_space.
        """

    def _make_done_action(self) -> dict[str, Any]:
        from stem_gym.envs.stem_env import ActionType
        return {
            "action_type": ActionType.DONE,
            "position": np.array([0.0, 0.0], dtype=np.float32),
            "zoom": np.array([1.0], dtype=np.float32),
            "dwell": np.array([1.0], dtype=np.float32),
            "finding_type": 0,
        }
