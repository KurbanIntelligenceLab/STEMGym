"""GP-UCB Navigator: spatial exploration via Gaussian Process modeling.

Maintains a BoTorch SingleTaskGP over tile interestingness (intensity
variance). Uses UCB acquisition function with configurable beta to balance
exploration vs exploitation. Masks already-visited positions.

Interface:
    navigator.reset(state)
    navigator.select_next(state) -> (x_nm, y_nm)
    navigator.record_observation(state, tile, x_nm, y_nm)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from stem_gym.state import ExperimentState


class Navigator:
    """GP-UCB navigator for selecting where to acquire images.

    Builds a Gaussian Process surrogate of tile interestingness and uses
    Upper Confidence Bound to trade off exploration vs exploitation.
    Beta is set externally by the Planner (high in SURVEY, low in
    CHARACTERIZE mode).

    Args:
        beta: Initial UCB exploration parameter.
        initial_random: Number of random samples before fitting GP.
        refit_interval: Refit GP every N observations.
        seed: Random seed.
    """

    def __init__(
        self,
        beta: float = 2.0,
        initial_random: int = 4,
        refit_interval: int = 3,
        seed: int = 42,
    ):
        self.beta = beta
        self.initial_random = initial_random
        self.refit_interval = refit_interval
        self.rng = np.random.default_rng(seed)

        # GP training data
        self._X: list[list[float]] = []       # normalized (row, col) in [0, 1]
        self._Y: list[float] = []             # interestingness values
        self._gp_model = None
        self._visited_set: set[tuple[int, int]] = set()

        # World geometry (filled on reset)
        self._grid_shape: tuple[int, int] = (1, 1)
        self._fov_nm: tuple[float, float] = (1.0, 1.0)

    def reset(self, state: ExperimentState) -> None:
        """Reset navigator for a new episode."""
        self._X.clear()
        self._Y.clear()
        self._gp_model = None
        self._visited_set.clear()
        self._grid_shape = state.visited_mask.shape if state.visited_mask is not None else (1, 1)
        self._fov_nm = state.fov_nm

    def set_beta(self, beta: float) -> None:
        """Update the UCB exploration parameter (called by Planner)."""
        self.beta = beta

    def select_next(self, state: ExperimentState) -> tuple[float, float]:
        """Select the next position to navigate to.

        Uses GP-UCB if enough data has been collected, otherwise random.

        Returns:
            (x_nm, y_nm) target position in world coordinates.
        """
        grid_h, grid_w = self._grid_shape

        # Early phase: random unvisited tiles
        if self._gp_model is None or len(self._X) < self.initial_random:
            return self._random_unvisited(state)

        # Collect unvisited candidates
        candidates = []
        for r in range(grid_h):
            for c in range(grid_w):
                if (r, c) not in self._visited_set:
                    candidates.append((r, c))

        if not candidates:
            # All visited — pick random for revisit
            r = int(self.rng.integers(0, grid_h))
            c = int(self.rng.integers(0, grid_w))
            return self._grid_to_nm(r, c)

        # Normalize candidate positions to [0, 1] on same device as GP
        _device = next(self._gp_model.parameters()).device
        cand_norm = torch.tensor(
            [[r / max(grid_h - 1, 1), c / max(grid_w - 1, 1)] for r, c in candidates],
            dtype=torch.float64,
            device=_device,
        )

        # Compute GP posterior UCB
        self._gp_model.eval()
        with torch.no_grad():
            posterior = self._gp_model.posterior(cand_norm)
            mean = posterior.mean.squeeze(-1)
            std = posterior.variance.sqrt().squeeze(-1)

        ucb = mean + self.beta * std
        best_idx = ucb.argmax().item()
        row, col = candidates[best_idx]
        return self._grid_to_nm(row, col)

    def record_observation(
        self, state: ExperimentState, tile: np.ndarray, x_nm: float, y_nm: float,
    ) -> None:
        """Record an acquired tile for GP model update.

        Args:
            state: Current experiment state.
            tile: The acquired tile image.
            x_nm, y_nm: Position where tile was acquired.
        """
        row, col = self._nm_to_grid(x_nm, y_nm)
        self._visited_set.add((row, col))

        # Compute interestingness
        interestingness = self._tile_interestingness(tile)

        # Normalized coordinates for GP
        grid_h, grid_w = self._grid_shape
        self._X.append([row / max(grid_h - 1, 1), col / max(grid_w - 1, 1)])
        self._Y.append(interestingness)

        # Refit GP periodically
        if len(self._X) >= self.initial_random and len(self._X) % self.refit_interval == 0:
            self._fit_gp()

    def _random_unvisited(self, state: ExperimentState) -> tuple[float, float]:
        """Pick a random unvisited tile position."""
        visited = state.visited_mask
        if visited is not None:
            unvisited = np.argwhere(~visited)
        else:
            grid_h, grid_w = self._grid_shape
            unvisited = np.array([[r, c] for r in range(grid_h) for c in range(grid_w)])

        if len(unvisited) == 0:
            grid_h, grid_w = self._grid_shape
            r = int(self.rng.integers(0, grid_h))
            c = int(self.rng.integers(0, grid_w))
            return self._grid_to_nm(r, c)

        idx = self.rng.integers(0, len(unvisited))
        row, col = unvisited[idx]
        return self._grid_to_nm(int(row), int(col))

    def _fit_gp(self) -> None:
        """Fit GP to observed interestingness data."""
        if len(self._X) < 2:
            return

        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood

        _device = "cuda" if torch.cuda.is_available() else "cpu"
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
            self._gp_model = SingleTaskGP(train_X, train_Y_norm)

    def _tile_interestingness(self, tile: np.ndarray) -> float:
        """Compute interestingness score for a tile."""
        if tile.max() <= 0:
            return 0.0
        return float(np.std(tile))

    def _grid_to_nm(self, row: int, col: int) -> tuple[float, float]:
        """Convert grid (row, col) to (x_nm, y_nm)."""
        grid_h, grid_w = self._grid_shape
        x_nm = (col + 0.5) / grid_w * self._fov_nm[1]
        y_nm = (row + 0.5) / grid_h * self._fov_nm[0]
        return (x_nm, y_nm)

    def _nm_to_grid(self, x_nm: float, y_nm: float) -> tuple[int, int]:
        """Convert (x_nm, y_nm) to grid (row, col)."""
        grid_h, grid_w = self._grid_shape
        col = int(np.clip(x_nm / self._fov_nm[1] * grid_w, 0, grid_w - 1))
        row = int(np.clip(y_nm / self._fov_nm[0] * grid_h, 0, grid_h - 1))
        return row, col

    @property
    def n_observations(self) -> int:
        return len(self._X)
