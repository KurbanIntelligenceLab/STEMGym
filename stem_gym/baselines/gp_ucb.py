"""GP-UCB baseline: Gaussian Process Upper Confidence Bound.

Single-agent GP-UCB over the spatial (x, y) domain. Acquires at every
navigated position. Models tile "interestingness" (intensity variance)
as the objective to maximize — explores high-variance regions first.

This supplies the GP navigation used by the manuscript reference evaluator.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from stem_gym.baselines import Agent
from stem_gym.agents.simple_analyst import SimpleAnalyst
from stem_gym.envs.stem_env import ActionType


class GPUCBAgent(Agent):
    """GP-UCB agent over spatial domain.

    Maintains a GP model of tile interestingness (intensity variance as
    proxy for information content). Uses UCB acquisition function to
    select next position. Always acquires at the selected position.

    Args:
        beta: UCB exploration parameter. Higher = more exploration.
        seed: Random seed.
        initial_random: Number of initial random acquisitions before
            fitting the GP (need ≥2 points).
    """

    def __init__(
        self,
        beta: float = 2.0,
        seed: int = 42,
        initial_random: int = 4,
        acq_func: str = "ucb",
    ):
        self.beta = beta
        self.seed = seed
        self.initial_random = initial_random
        self.acq_func = acq_func
        self.analyst = SimpleAnalyst()
        self.rng = np.random.default_rng(seed)

        # GP data
        self._X: list[list[float]] = []  # observed positions (normalized)
        self._Y: list[float] = []         # observed interestingness values
        self._grid_shape: tuple[int, int] = (1, 1)
        self._fov_nm: tuple[float, float] = (1.0, 1.0)
        self._navigate_phase = True
        self._step = 0
        self._gp_model = None
        self._visited_set: set[tuple[int, int]] = set()

    def reset(self, obs: dict[str, np.ndarray], info: dict[str, Any], *, seed: int | None = None) -> None:
        torch.manual_seed(self.seed if seed is None else seed)
        self.rng = np.random.default_rng(self.seed if seed is None else seed)
        self.analyst.reset()
        self._observed_acquisitions = 0
        self._geometry = info.get("geometry")
        self._X.clear()
        self._Y.clear()
        self._grid_shape = obs["visited_mask"].shape
        self._fov_nm = info.get("fov_nm", (100.0, 100.0))
        self._navigate_phase = True
        self._step = 0
        self._gp_model = None
        self._visited_set.clear()

    def _fit_gp(self) -> None:
        """Fit GP model to observed data."""
        if len(self._X) < 2:
            return

        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood

        import os

        force_cpu = os.environ.get("STEMGYM_FORCE_CPU_GP") == "1"
        _device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
        train_X = torch.tensor(self._X, dtype=torch.float64, device=_device)
        train_Y = torch.tensor(self._Y, dtype=torch.float64, device=_device).unsqueeze(-1)

        # Normalize Y
        y_mean = train_Y.mean()
        y_std = train_Y.std().clamp(min=1e-6)
        train_Y_norm = (train_Y - y_mean) / y_std

        try:
            self._gp_model = SingleTaskGP(train_X, train_Y_norm)
            mll = ExactMarginalLogLikelihood(
                self._gp_model.likelihood, self._gp_model
            )
            fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 50}})
        except Exception:
            # Fall back to unfitted model if optimization fails
            self._gp_model = SingleTaskGP(train_X, train_Y_norm)

    def _select_next_position(self, obs: dict[str, np.ndarray]) -> tuple[float, float]:
        """Select next position using GP-UCB."""
        grid_h, grid_w = self._grid_shape
        visited = obs["visited_mask"]

        # If no GP yet, pick random unvisited tile
        if self._gp_model is None or self._step < self.initial_random:
            unvisited = np.argwhere(visited < 0.5)
            if len(unvisited) == 0:
                unvisited = np.argwhere(np.ones_like(visited))
            idx = self.rng.integers(0, len(unvisited))
            row, col = unvisited[idx]
            return self._grid_to_nm(int(row), int(col))

        # Evaluate UCB over all unvisited tiles
        candidates = []
        for r in range(grid_h):
            for c in range(grid_w):
                if (r, c) not in self._visited_set:
                    candidates.append((r, c))

        if not candidates:
            # All visited — pick random
            r = self.rng.integers(0, grid_h)
            c = self.rng.integers(0, grid_w)
            return self._grid_to_nm(r, c)

        # Normalize candidate positions to [0, 1] on same device as GP
        _device = next(self._gp_model.parameters()).device
        cand_norm = torch.tensor(
            [[r / max(grid_h - 1, 1), c / max(grid_w - 1, 1)] for r, c in candidates],
            dtype=torch.float64,
            device=_device,
        )

        # Compute acquisition function values
        from stem_gym.agents.acquisition import evaluate_acquisition
        train_Y = torch.tensor(self._Y, dtype=torch.float64, device=_device).unsqueeze(-1) if self._Y else None
        acq_values = evaluate_acquisition(
            self._gp_model, cand_norm,
            acq_func=self.acq_func, beta=self.beta, train_Y=train_Y,
        )
        best_idx = acq_values.argmax().item()
        row, col = candidates[best_idx]
        return self._grid_to_nm(row, col)

    def _grid_to_nm(self, row: int, col: int) -> tuple[float, float]:
        """Convert grid indices to approximate nm coordinates."""
        if self._geometry is not None:
            return self._geometry.tile_to_nm(row, col)
        grid_h, grid_w = self._grid_shape
        x_nm = (col + 0.5) / grid_w * self._fov_nm[1]
        y_nm = (row + 0.5) / grid_h * self._fov_nm[0]
        return (x_nm, y_nm)

    def _nm_to_grid(self, x_nm: float, y_nm: float) -> tuple[int, int]:
        """Convert nm coordinates to grid indices."""
        if self._geometry is not None:
            return self._geometry.nm_to_tile(x_nm, y_nm)
        grid_h, grid_w = self._grid_shape
        col = int(np.clip(x_nm / self._fov_nm[1] * grid_w, 0, grid_w - 1))
        row = int(np.clip(y_nm / self._fov_nm[0] * grid_h, 0, grid_h - 1))
        return row, col

    def _tile_interestingness(self, tile: np.ndarray) -> float:
        """Compute interestingness score for a tile.

        Uses intensity variance as proxy — tiles with high variance
        likely contain defects or features of interest.
        """
        if tile.max() <= 0:
            return 0.0
        return float(np.std(tile))

    def act(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> dict[str, Any]:
        self._step += 1
        budget_frac = float(obs["budget_remaining"][0])

        if budget_frac < 0.01:
            return self._make_done_action()

        if self._navigate_phase:
            # Select position via GP-UCB
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
            # Acquire and analyze
            self._navigate_phase = True

            return {
                "action_type": ActionType.ACQUIRE,
                "position": obs["position"].copy(),
                "zoom": np.array([1.0], dtype=np.float32),
                "dwell": np.array([1.0], dtype=np.float32),
                "finding_type": 0,
            }

    def select_tile(self, obs: dict[str, np.ndarray]) -> tuple[int, int]:
        """Select a tile using the current surrogate and visited mask."""
        return self._nm_to_grid(*self._select_next_position(obs))

    def update(self, tile: tuple[int, int], signal: float, *, fit: bool = True) -> None:
        """Record a completed acquisition and refit every three acquisitions."""
        row, col = tile
        gh, gw = self._grid_shape
        if not (0 <= row < gh and 0 <= col < gw) or not np.isfinite(signal):
            raise ValueError("Invalid GP observation")
        self._X.append([row / max(gh - 1, 1), col / max(gw - 1, 1)])
        self._Y.append(float(signal))
        self._visited_set.add(tile)
        self._step = len(self._X)
        if fit and len(self._X) >= self.initial_random and len(self._X) % 3 == 0:
            self._fit_gp()

    def observe(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> None:
        count = info.get("acquisitions", 0)
        if count > self._observed_acquisitions:
            super().observe(obs, info)
            self.update(info["tile_position"], self._tile_interestingness(obs["tile"]))

    def get_findings(self) -> dict[str, Any]:
        return {
            "defect_census": self.analyst.get_defect_census(),
            "defect_positions": [],
            "phase_map": None,
            "atom_positions": None,
        }
