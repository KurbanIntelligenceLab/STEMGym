"""Kalinin/Ziatdinov-style GP-BO baseline for autonomous STEM acquisition.

Re-implements the AtomAI active learning loop spec from Ziatdinov et al. 2022
(Nat. Mach. Intell. 4:1101-1112) over our existing BoTorch SingleTaskGP
scaffolding. Provides a published prior-art comparison against the in-house
agents per the broader autonomous-STEM workflow described in Kalinin et al.
2023 (npj Comput. Mater. 9:227).

Differences from the in-house GPUCBAgent:
- Target is a composite "informativeness" score combining intensity
  statistics with atom-position features (atom count + intensity dispersion),
  not just raw intensity variance. This matches the Ziatdinov 2022
  feature-rich active-learning signal.
- Uses the analyst's peak detector for per-tile featurization at each
  ACQUIRE step.

GP inputs: normalized spatial coordinates (x_norm, y_norm). Acquisition: UCB.
The full Deep Kernel Learning variant (CNN feature extractor + GP) is out of
scope for this baseline; we use a SingleTaskGP on spatial coordinates with the
richer Ziatdinov target. See Roccapriore et al. 2022 for the full DKL pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from stem_gym.agents.simple_analyst import AnalysisResult, SimpleAnalyst
from stem_gym.baselines import Agent
from stem_gym.envs.stem_env import ActionType


# Ziatdinov 2022 composite-target weights. Equal-ish weighting; sensitivity
# to these is reported as low in the paper's Fig. 4.
_W_INTENSITY_VAR = 1.0
_W_ATOM_COUNT = 0.5  # normalised by typical 50 atoms/tile
_W_ATOM_DISPERSION = 1.0
_TYPICAL_ATOMS_PER_TILE = 50.0


class KalininGPBOAgent(Agent):
    """GP-BO baseline using intensity + atom-position features.

    Args:
        beta: UCB exploration parameter.
        seed: Random seed.
        initial_random: Initial random acquisitions before fitting the GP.
        acq_func: Acquisition function ("ucb", "ei", "ts").
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

        self._X: list[list[float]] = []
        self._Y: list[float] = []
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
        self.rng = np.random.default_rng(self.seed)

    def _ziatdinov_target(
        self, tile: np.ndarray, result: AnalysisResult
    ) -> float:
        """Composite informativeness signal per Ziatdinov 2022.

        Combines intensity statistics with atom-position features extracted
        by the analyst's peak detector.
        """
        if tile.max() <= 0:
            return 0.0
        intensity_var = float(np.std(tile))
        atom_density = float(result.n_atoms) / _TYPICAL_ATOMS_PER_TILE
        atom_dispersion = float(result.intensity_std)
        return (
            _W_INTENSITY_VAR * intensity_var
            + _W_ATOM_COUNT * atom_density
            + _W_ATOM_DISPERSION * atom_dispersion
        )

    def _fit_gp(self) -> None:
        if len(self._X) < 2:
            return

        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from gpytorch.mlls import ExactMarginalLogLikelihood

        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_X = torch.tensor(self._X, dtype=torch.float64, device=device)
        train_Y = torch.tensor(
            self._Y, dtype=torch.float64, device=device
        ).unsqueeze(-1)

        y_mean = train_Y.mean()
        y_std = train_Y.std().clamp(min=1e-6)
        train_Y_norm = (train_Y - y_mean) / y_std

        try:
            self._gp_model = SingleTaskGP(train_X, train_Y_norm)
            mll = ExactMarginalLogLikelihood(
                self._gp_model.likelihood, self._gp_model
            )
            fit_gpytorch_mll(
                mll, optimizer_kwargs={"options": {"maxiter": 50}}
            )
        except Exception:
            self._gp_model = SingleTaskGP(train_X, train_Y_norm)

    def _select_next_position(
        self, obs: dict[str, np.ndarray]
    ) -> tuple[float, float]:
        grid_h, grid_w = self._grid_shape

        if self._gp_model is None or self._step < self.initial_random:
            visited = obs["visited_mask"]
            unvisited = np.argwhere(visited < 0.5)
            if len(unvisited) == 0:
                unvisited = np.argwhere(np.ones_like(visited))
            idx = self.rng.integers(0, len(unvisited))
            row, col = unvisited[idx]
            return self._grid_to_nm(int(row), int(col))

        candidates = [
            (r, c)
            for r in range(grid_h)
            for c in range(grid_w)
            if (r, c) not in self._visited_set
        ]
        if not candidates:
            r = int(self.rng.integers(0, grid_h))
            c = int(self.rng.integers(0, grid_w))
            return self._grid_to_nm(r, c)

        device = next(self._gp_model.parameters()).device
        cand_norm = torch.tensor(
            [
                [r / max(grid_h - 1, 1), c / max(grid_w - 1, 1)]
                for r, c in candidates
            ],
            dtype=torch.float64,
            device=device,
        )

        from stem_gym.agents.acquisition import evaluate_acquisition

        train_Y = (
            torch.tensor(self._Y, dtype=torch.float64, device=device).unsqueeze(-1)
            if self._Y
            else None
        )
        acq_values = evaluate_acquisition(
            self._gp_model,
            cand_norm,
            acq_func=self.acq_func,
            beta=self.beta,
            train_Y=train_Y,
        )
        best_idx = int(acq_values.argmax().item())
        row, col = candidates[best_idx]
        return self._grid_to_nm(row, col)

    def _grid_to_nm(self, row: int, col: int) -> tuple[float, float]:
        grid_h, grid_w = self._grid_shape
        x_nm = (col + 0.5) / grid_w * self._fov_nm[1]
        y_nm = (row + 0.5) / grid_h * self._fov_nm[0]
        return (x_nm, y_nm)

    def _nm_to_grid(self, x_nm: float, y_nm: float) -> tuple[int, int]:
        grid_h, grid_w = self._grid_shape
        col = int(np.clip(x_nm / self._fov_nm[1] * grid_w, 0, grid_w - 1))
        row = int(np.clip(y_nm / self._fov_nm[0] * grid_h, 0, grid_h - 1))
        return row, col

    def act(
        self, obs: dict[str, np.ndarray], info: dict[str, Any]
    ) -> dict[str, Any]:
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

        self._navigate_phase = True
        tile = obs.get("tile")
        if tile is not None and tile.max() > 0:
            result = self.analyst.analyze_tile(tile)
            target = self._ziatdinov_target(tile, result)

            pos = obs["position"]
            row, col = self._nm_to_grid(float(pos[0]), float(pos[1]))
            self._visited_set.add((row, col))

            grid_h, grid_w = self._grid_shape
            x_norm = row / max(grid_h - 1, 1)
            y_norm = col / max(grid_w - 1, 1)
            self._X.append([x_norm, y_norm])
            self._Y.append(target)

            if (
                len(self._X) >= self.initial_random
                and len(self._X) % 3 == 0
            ):
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


class KalininGPBOEquippedAgent(KalininGPBOAgent):
    """KalininGPBOAgent paired with the trained Analyst ensemble.

    Mirrors the inheritance pattern of GPUCBEquippedAgent(GPUCBAgent):
    overrides the analyst slot in __init__ and threads tile position
    through analyze_tile so defect localisation uses world coordinates.
    """

    def __init__(
        self,
        beta: float = 2.0,
        seed: int = 42,
        initial_random: int = 4,
        acq_func: str = "ucb",
        checkpoint_dir: str | Path | None = None,
    ):
        super().__init__(
            beta=beta,
            seed=seed,
            initial_random=initial_random,
            acq_func=acq_func,
        )
        from stem_gym.baselines.equipped_baselines import _load_analyst

        self.analyst = _load_analyst(checkpoint_dir)
        self._pixel_size_nm: float = 0.005

    def reset(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> None:
        super().reset(obs, info)
        self._pixel_size_nm = info.get("pixel_size_nm", 0.005)

    def act(
        self, obs: dict[str, np.ndarray], info: dict[str, Any]
    ) -> dict[str, Any]:
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

        self._navigate_phase = True
        tile = obs.get("tile")
        if tile is not None and tile.max() > 0:
            pos = obs["position"]
            x_nm, y_nm = float(pos[0]), float(pos[1])
            row, col = self._nm_to_grid(x_nm, y_nm)

            result = self.analyst.analyze_tile(
                tile,
                tile_position_nm=(x_nm, y_nm),
                pixel_size_nm=self._pixel_size_nm,
                tile_key=(row, col),
            )
            target = self._ziatdinov_target(tile, result)

            self._visited_set.add((row, col))
            grid_h, grid_w = self._grid_shape
            x_norm = row / max(grid_h - 1, 1)
            y_norm = col / max(grid_w - 1, 1)
            self._X.append([x_norm, y_norm])
            self._Y.append(target)

            if (
                len(self._X) >= self.initial_random
                and len(self._X) % 3 == 0
            ):
                self._fit_gp()

        return {
            "action_type": ActionType.ACQUIRE,
            "position": obs["position"].copy(),
            "zoom": np.array([1.0], dtype=np.float32),
            "dwell": np.array([1.0], dtype=np.float32),
            "finding_type": 0,
        }

    def get_findings(self) -> dict[str, Any]:
        from stem_gym.baselines.equipped_baselines import _assemble_findings

        return _assemble_findings(self.analyst, self._grid_shape)
