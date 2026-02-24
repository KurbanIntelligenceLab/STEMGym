"""GP-UCB baseline: Gaussian Process Upper Confidence Bound.

Single-agent GP-UCB over the spatial (x, y) domain. Acquires at every
navigated position. Models tile "interestingness" (intensity variance)
as the objective to maximize — explores high-variance regions first.

This is the primary single-agent baseline and the one STEMAgent must beat.
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
    ):
        self.beta = beta
        self.seed = seed
        self.initial_random = initial_random
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

    def reset(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> None:
        self.analyst.reset()
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

        train_X = torch.tensor(self._X, dtype=torch.float64)
        train_Y = torch.tensor(self._Y, dtype=torch.float64).unsqueeze(-1)

        # Normalize Y
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

        # Normalize candidate positions to [0, 1]
        cand_norm = torch.tensor(
            [[r / max(grid_h - 1, 1), c / max(grid_w - 1, 1)] for r, c in candidates],
            dtype=torch.float64,
        )

        # Compute posterior
        self._gp_model.eval()
        with torch.no_grad():
            posterior = self._gp_model.posterior(cand_norm)
            mean = posterior.mean.squeeze(-1)
            std = posterior.variance.sqrt().squeeze(-1)

        # UCB = mean + beta * std
        ucb = mean + self.beta * std
        best_idx = ucb.argmax().item()
        row, col = candidates[best_idx]
        return self._grid_to_nm(row, col)

    def _grid_to_nm(self, row: int, col: int) -> tuple[float, float]:
        """Convert grid indices to approximate nm coordinates."""
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

            # Analyze previous tile and record observation for GP
            tile = obs.get("tile")
            if tile is not None and tile.max() > 0:
                result = self.analyst.analyze_tile(tile)
                interestingness = self._tile_interestingness(tile)

                pos = obs["position"]
                row, col = self._nm_to_grid(float(pos[0]), float(pos[1]))
                self._visited_set.add((row, col))

                # Record for GP (normalized coordinates)
                grid_h, grid_w = self._grid_shape
                x_norm = row / max(grid_h - 1, 1)
                y_norm = col / max(grid_w - 1, 1)
                self._X.append([x_norm, y_norm])
                self._Y.append(interestingness)

                # Refit GP periodically
                if len(self._X) >= self.initial_random and len(self._X) % 3 == 0:
                    self._fit_gp()

            return {
                "action_type": ActionType.ACQUIRE,
                "position": obs["position"].copy(),
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
