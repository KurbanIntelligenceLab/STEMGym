"""VLM-equipped baselines: baseline navigation + VLMAnalyst.

Pairs simple navigation strategies (GP-UCB, raster scan) with a
vision-language model for tile analysis via OpenRouter.
Direct comparison with equipped_baselines.py (which uses trained CNNs).

Supported providers: "claude" (Haiku 4.5), "gpt5" (GPT-5-mini), "llama" (Llama 4 Scout).

GPUCBVLMAgent: GP-UCB navigation + VLMAnalyst.
RasterVLMAgent: Row-major raster scan + VLMAnalyst.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from stem_gym.baselines import Agent
from stem_gym.agents.vlm_analyst import VLMAnalyst
from stem_gym.baselines.equipped_baselines import _assemble_findings
from stem_gym.baselines.gp_ucb import GPUCBAgent
from stem_gym.envs.stem_env import ActionType


# ---------------------------------------------------------------------------
# GP-UCB + VLM Analyst
# ---------------------------------------------------------------------------


class GPUCBVLMAgent(GPUCBAgent):
    """GP-UCB navigation with VLM-based tile analysis.

    Inherits all navigation/GP logic from GPUCBAgent. Uses VLMAnalyst
    (Claude Sonnet or GPT-4o) instead of trained CNN ensemble.
    """

    def __init__(self, beta=2.0, seed=42, initial_random=3, provider="claude"):
        super().__init__(beta=beta, seed=seed, initial_random=initial_random)
        self.analyst = VLMAnalyst(provider=provider)
        self._pixel_size_nm: float = 0.005

    def reset(self, obs, info):
        super().reset(obs, info)
        self.analyst.reset()
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
                self.analyst.analyze_tile(
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
# Raster scan + VLM Analyst
# ---------------------------------------------------------------------------


class RasterVLMAgent(Agent):
    """Raster scan navigation with VLM-based tile analysis.

    Row-major grid traversal identical to RasterAgent, but using the
    VLMAnalyst instead of SimpleAnalyst for tile analysis.
    """

    def __init__(self, seed: int = 42, provider: str = "claude"):
        self.analyst = VLMAnalyst(provider=provider)
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

        rows, cols = self._grid_shape
        self._tile_queue = [(r, c) for r in range(rows) for c in range(cols)]

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
                "zoom": np.array([1.0], dtype=np.float32),
                "dwell": np.array([1.0], dtype=np.float32),
                "finding_type": 0,
            }

    def get_findings(self) -> dict[str, Any]:
        return _assemble_findings(self.analyst, self._grid_shape)
