"""Single-BO baseline: joint Bayesian Optimization over (x, y, zoom, dwell).

Ablation baseline for demonstrating the value of multi-agent decomposition.
This agent optimizes position AND acquisition parameters jointly, which
should perform worse than GP-UCB (which separates navigation from acquisition)
due to the higher-dimensional search space and conflated objectives.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from stem_gym.baselines import Agent
from stem_gym.agents.simple_analyst import SimpleAnalyst
from stem_gym.envs.stem_env import ActionType


class SingleBOAgent(Agent):
    """Single-agent BO over joint (x, y, zoom, dwell) space.

    Uses BoTorch Expected Improvement to select the next
    (position, zoom, dwell) tuple jointly. This conflates navigation
    and acquisition decisions, making the search space 4D instead of 2D.

    This should perform worse than GP-UCB to demonstrate that
    decomposition (separating navigation from acquisition into
    different agents) is beneficial.
    """

    def __init__(
        self,
        seed: int = 42,
        initial_random: int = 6,
    ):
        self.seed = seed
        self.initial_random = initial_random
        self.analyst = SimpleAnalyst()
        self.rng = np.random.default_rng(seed)

        # BO data: 4D input (x_norm, y_norm, zoom_norm, dwell_norm)
        self._X: list[list[float]] = []
        self._Y: list[float] = []
        self._grid_shape: tuple[int, int] = (1, 1)
        self._fov_nm: tuple[float, float] = (1.0, 1.0)
        self._step = 0
        self._pending_action: dict[str, Any] | None = None
        self._phase = "navigate"  # "navigate" or "acquire"
        self._next_params: tuple[float, float, float, float] | None = None

    def reset(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> None:
        self.analyst.reset()
        self._X.clear()
        self._Y.clear()
        self._grid_shape = obs["visited_mask"].shape
        self._fov_nm = info.get("fov_nm", (100.0, 100.0))
        self._step = 0
        self._phase = "navigate"
        self._next_params = None

    def _select_next(self, obs: dict[str, np.ndarray]) -> tuple[float, float, float, float]:
        """Select (x_nm, y_nm, zoom, dwell) via BO or random."""

        grid_h, grid_w = self._grid_shape

        if len(self._X) < self.initial_random:
            # Random exploration
            x = self.rng.uniform(0, 1)
            y = self.rng.uniform(0, 1)
            zoom = self.rng.uniform(0, 1)
            dwell = self.rng.uniform(0, 1)
            return x, y, zoom, dwell

        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from botorch.acquisition import ExpectedImprovement
        from botorch.optim import optimize_acqf
        from gpytorch.mlls import ExactMarginalLogLikelihood

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        train_X = torch.tensor(self._X, dtype=torch.float64, device=_device)
        train_Y = torch.tensor(self._Y, dtype=torch.float64, device=_device).unsqueeze(-1)

        # Normalize Y
        y_mean = train_Y.mean()
        y_std = train_Y.std().clamp(min=1e-6)
        train_Y_norm = (train_Y - y_mean) / y_std

        try:
            model = SingleTaskGP(train_X, train_Y_norm)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 50}})

            best_f = train_Y_norm.max()
            ei = ExpectedImprovement(model=model, best_f=best_f)

            candidate, _ = optimize_acqf(
                ei,
                bounds=torch.tensor([[0.0]*4, [1.0]*4], dtype=torch.float64, device=_device),
                q=1,
                num_restarts=5,
                raw_samples=64,
            )
            return tuple(candidate.squeeze(0).tolist())
        except Exception:
            # Fallback to random
            return (
                self.rng.uniform(0, 1),
                self.rng.uniform(0, 1),
                self.rng.uniform(0, 1),
                self.rng.uniform(0, 1),
            )

    def _denormalize(
        self, x_n: float, y_n: float, zoom_n: float, dwell_n: float
    ) -> tuple[float, float, float, float]:
        """Convert normalized [0,1] params to physical values."""
        grid_h, grid_w = self._grid_shape
        x_nm = x_n * self._fov_nm[1]
        y_nm = y_n * self._fov_nm[0]
        zoom = 1.0 + zoom_n * 3.0    # [1, 4]
        dwell = 0.1 + dwell_n * 9.9  # [0.1, 10]
        return x_nm, y_nm, zoom, dwell

    def act(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> dict[str, Any]:
        self._step += 1
        budget_frac = float(obs["budget_remaining"][0])

        if budget_frac < 0.01:
            return self._make_done_action()

        if self._phase == "navigate":
            # Select next params via BO
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
            # Acquire with the BO-selected parameters
            self._phase = "navigate"

            # Analyze previous tile
            tile = obs.get("tile")
            interestingness = 0.0
            if tile is not None and tile.max() > 0:
                self.analyst.analyze_tile(tile)
                interestingness = float(np.std(tile))

            # Record observation
            if self._next_params is not None:
                self._X.append(list(self._next_params))
                self._Y.append(interestingness)

            # Denormalize for the acquire action
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
