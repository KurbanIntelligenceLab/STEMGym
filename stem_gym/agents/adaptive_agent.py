"""Adaptive agent wrappers: base agents with online fine-tuning.

Uses composition to wrap existing agents (RasterEquippedAgent,
STEMAgent) with an AdaptiveAnalyst that fine-tunes the AtomFinder
ensemble during episodes using self-supervised consistency loss.

Designed for cross-family transfer scenarios where the trained models
may not generalize well to unseen material systems.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from stem_gym.baselines import Agent
from stem_gym.agents.adaptive_analyst import AdaptiveAnalyst
from stem_gym.agents.analyst import Analyst
from stem_gym.baselines.equipped_baselines import (
    RasterEquippedAgent,
    _assemble_findings,
    _load_analyst,
)
from stem_gym.agents.multi_agent import STEMAgent


class AdaptiveRasterAgent(Agent):
    """Raster scan + AdaptiveAnalyst for cross-family transfer.

    Creates a RasterEquippedAgent, then hot-swaps its analyst with
    an AdaptiveAnalyst wrapper that fine-tunes online.
    """

    def __init__(
        self,
        seed: int = 42,
        checkpoint_dir: str | Path | None = None,
        lr: float = 1e-5,
        update_every_n_tiles: int = 5,
        n_gradient_steps: int = 3,
    ):
        self._base_agent = RasterEquippedAgent(
            seed=seed, checkpoint_dir=checkpoint_dir,
        )
        # Wrap the loaded analyst with adaptive fine-tuning
        self._adaptive_analyst = AdaptiveAnalyst(
            self._base_agent.analyst,
            lr=lr,
            update_every_n_tiles=update_every_n_tiles,
            n_gradient_steps=n_gradient_steps,
        )
        # Hot-swap: the base agent now uses the adaptive analyst
        self._base_agent.analyst = self._adaptive_analyst

    def reset(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> None:
        self._base_agent.reset(obs, info)

    def act(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> dict[str, Any]:
        return self._base_agent.act(obs, info)

    def get_findings(self) -> dict[str, Any]:
        return self._base_agent.get_findings()


class AdaptiveSTEMAgent(Agent):
    """STEMAgent + AdaptiveAnalyst for cross-family transfer.

    Creates a STEMAgent, then hot-swaps its analyst with an
    AdaptiveAnalyst wrapper that fine-tunes online.
    """

    def __init__(
        self,
        seed: int = 42,
        checkpoint_dir: str | Path | None = None,
        lr: float = 1e-5,
        update_every_n_tiles: int = 5,
        n_gradient_steps: int = 3,
        acq_func: str = "ucb",
    ):
        self._base_agent = STEMAgent(
            seed=seed, checkpoint_dir=checkpoint_dir,
            acq_func=acq_func,
        )
        # Wrap with adaptive fine-tuning
        self._adaptive_analyst = AdaptiveAnalyst(
            self._base_agent.analyst,
            lr=lr,
            update_every_n_tiles=update_every_n_tiles,
            n_gradient_steps=n_gradient_steps,
        )
        # Hot-swap
        self._base_agent.analyst = self._adaptive_analyst

    def reset(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> None:
        self._base_agent.reset(obs, info)

    def act(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> dict[str, Any]:
        return self._base_agent.act(obs, info)

    def get_findings(self) -> dict[str, Any]:
        return self._base_agent.get_findings()
