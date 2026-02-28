"""DQN RL baseline: Stable-Baselines3 DQN with discretized action space.

Demonstrates that standard RL struggles with sparse rewards and large
observation spaces in dose-limited STEM, motivating the benchmark as a
testbed for future RL research.

DiscreteSTEMWrapper: Wraps STEMEnv with a flat Discrete action space.
DQNAgent: Loads a trained SB3 DQN model for evaluation.
DQNEquippedAgent: DQN navigation + trained Analyst for causal decomposition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from stem_gym.baselines import Agent
from stem_gym.envs.stem_env import ActionType, STEMEnv


# ── Grid / action constants ────────────────────────────────────────
SPATIAL_GRID = 8           # 8×8 = 64 spatial bins
N_ACTION_TYPES = 3         # NAVIGATE, ACQUIRE, DONE
N_ACTIONS = SPATIAL_GRID * SPATIAL_GRID * N_ACTION_TYPES  # 192


# ── Gymnasium wrapper ──────────────────────────────────────────────

class DiscreteSTEMWrapper(gym.Wrapper):
    """Wraps STEMEnv with a flat Discrete action space for DQN.

    Action encoding:  action_id = (row * SPATIAL_GRID + col) * 3 + action_type_idx
      action_type_idx: 0=NAVIGATE, 1=ACQUIRE, 2=DONE

    Observation: flat vector (tile_mean_features + overview_features + scalars).
    """

    def __init__(self, env: STEMEnv, tile_pool: int = 8, overview_pool: int = 4):
        super().__init__(env)
        self.tile_pool = tile_pool
        self.overview_pool = overview_pool

        # Compute obs dimension: pooled tile + pooled overview + scalars
        tile_size = env.world.tile_size_px
        overview_size = env.overview_size
        tile_feat = (tile_size // tile_pool) ** 2
        overview_feat = (overview_size // overview_pool) ** 2
        scalar_feat = 4  # x, y, budget_remaining, step_count
        self._obs_dim = tile_feat + overview_feat + scalar_feat

        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._obs_dim,), dtype=np.float32,
        )

        self._fov_nm: tuple[float, float] = (1.0, 1.0)
        self._prev_tiles_acquired: int = 0

    def _pool(self, img: np.ndarray, pool_size: int) -> np.ndarray:
        """Average-pool a 2D array and flatten."""
        h, w = img.shape
        ph = h // pool_size
        pw = w // pool_size
        cropped = img[: ph * pool_size, : pw * pool_size]
        pooled = cropped.reshape(ph, pool_size, pw, pool_size).mean(axis=(1, 3))
        return pooled.flatten().astype(np.float32)

    def _flatten_obs(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        tile_feat = self._pool(obs["tile"], self.tile_pool)
        overview_feat = self._pool(obs["overview"], self.overview_pool)
        pos = obs["position"]
        fov_h, fov_w = self._fov_nm
        scalars = np.array([
            pos[0] / max(fov_w, 1.0),
            pos[1] / max(fov_h, 1.0),
            obs["budget_remaining"][0],
            obs["step_count"][0],
        ], dtype=np.float32)
        return np.concatenate([tile_feat, overview_feat, scalars])

    def _decode_action(self, action_id: int) -> dict[str, Any]:
        action_type_idx = action_id % N_ACTION_TYPES
        spatial_id = action_id // N_ACTION_TYPES
        row = spatial_id // SPATIAL_GRID
        col = spatial_id % SPATIAL_GRID

        fov_h, fov_w = self._fov_nm
        x_nm = (col + 0.5) / SPATIAL_GRID * fov_w
        y_nm = (row + 0.5) / SPATIAL_GRID * fov_h

        action_types = [ActionType.NAVIGATE, ActionType.ACQUIRE, ActionType.DONE]
        return {
            "action_type": action_types[action_type_idx],
            "position": np.array([x_nm, y_nm], dtype=np.float32),
            "zoom": np.array([1.0], dtype=np.float32),
            "dwell": np.array([1.0], dtype=np.float32),
            "finding_type": 0,
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._fov_nm = info.get("fov_nm", (100.0, 100.0))
        self._prev_tiles_acquired = 0
        return self._flatten_obs(obs), info

    def step(self, action: int):
        action_dict = self._decode_action(action)
        obs, reward, terminated, truncated, info = self.env.step(action_dict)

        # Reward shaping: small bonus for new tiles, small step penalty
        tiles_now = info.get("tiles_acquired", 0)
        new_tiles = tiles_now - self._prev_tiles_acquired
        self._prev_tiles_acquired = tiles_now

        shaped_reward = reward  # terminal task score
        if not (terminated or truncated):
            # 0.01 per new tile encourages exploration; -0.0001 step penalty
            # discourages idle actions without dominating the tile bonus
            shaped_reward = 0.01 * new_tiles - 0.0001

        return self._flatten_obs(obs), shaped_reward, terminated, truncated, info


# ── DQN Agent (evaluation wrapper) ────────────────────────────────

_DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent.parent / "checkpoints" / "dqn_agent.zip"


class DQNAgent(Agent):
    """DQN-based navigation agent using a trained Stable-Baselines3 model.

    Uses SimpleAnalyst for findings (same as RandomAgent).
    """

    def __init__(self, seed: int = 42, model_path: str | Path | None = None):
        self.seed = seed
        self._model_path = Path(model_path) if model_path else _DEFAULT_CHECKPOINT
        self._model = None
        self._fov_nm: tuple[float, float] = (1.0, 1.0)
        self._grid_shape: tuple[int, int] = (1, 1)
        self._tile_pool = 8
        self._overview_pool = 4

        from stem_gym.agents.simple_analyst import SimpleAnalyst
        self.analyst = SimpleAnalyst()

    def _ensure_model(self):
        if self._model is not None:
            return
        if not self._model_path.exists():
            # No trained model — fall back to random actions
            self._model = None
            return
        try:
            from stable_baselines3 import DQN
            self._model = DQN.load(str(self._model_path))
        except Exception:
            self._model = None

    def _pool(self, img: np.ndarray, pool_size: int) -> np.ndarray:
        h, w = img.shape
        ph = h // pool_size
        pw = w // pool_size
        cropped = img[: ph * pool_size, : pw * pool_size]
        pooled = cropped.reshape(ph, pool_size, pw, pool_size).mean(axis=(1, 3))
        return pooled.flatten().astype(np.float32)

    def _flatten_obs(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        tile_feat = self._pool(obs["tile"], self._tile_pool)
        overview_feat = self._pool(obs["overview"], self._overview_pool)
        pos = obs["position"]
        fov_h, fov_w = self._fov_nm
        scalars = np.array([
            pos[0] / max(fov_w, 1.0),
            pos[1] / max(fov_h, 1.0),
            obs["budget_remaining"][0],
            obs["step_count"][0],
        ], dtype=np.float32)
        return np.concatenate([tile_feat, overview_feat, scalars])

    def reset(self, obs, info):
        self.analyst.reset()
        self._fov_nm = info.get("fov_nm", (100.0, 100.0))
        self._grid_shape = obs["visited_mask"].shape
        self._ensure_model()
        self._navigate_phase = True
        self._rng = np.random.default_rng(self.seed)

    def act(self, obs, info):
        budget_frac = float(obs["budget_remaining"][0])
        if budget_frac < 0.01:
            return self._make_done_action()

        if self._navigate_phase:
            # Use DQN to pick a spatial target
            self._navigate_phase = False
            flat_obs = self._flatten_obs(obs)

            if self._model is not None:
                action_id, _ = self._model.predict(flat_obs, deterministic=True)
                action_id = int(action_id)
            else:
                # Random fallback
                action_id = int(self._rng.integers(0, N_ACTIONS))

            # Decode spatial target from action_id (ignore action type — we enforce navigate)
            spatial_id = action_id // N_ACTION_TYPES
            row = spatial_id // SPATIAL_GRID
            col = spatial_id % SPATIAL_GRID
            fov_h, fov_w = self._fov_nm
            x_nm = (col + 0.5) / SPATIAL_GRID * fov_w
            y_nm = (row + 0.5) / SPATIAL_GRID * fov_h

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
                self.analyst.analyze_tile(tile)
            return {
                "action_type": ActionType.ACQUIRE,
                "position": obs["position"].copy(),
                "zoom": np.array([1.0], dtype=np.float32),
                "dwell": np.array([1.0], dtype=np.float32),
                "finding_type": 0,
            }

    def get_findings(self):
        return {
            "defect_census": self.analyst.get_defect_census(),
            "defect_positions": [],
            "phase_map": None,
            "atom_positions": None,
        }


# ── DQN Equipped Agent ────────────────────────────────────────────

class DQNEquippedAgent(Agent):
    """DQN navigation + trained Analyst for causal decomposition.

    Same DQN model for navigation decisions, but uses the full trained
    Analyst ensemble instead of SimpleAnalyst. This isolates the navigation
    contribution of DQN from the perception advantage.
    """

    def __init__(self, seed: int = 42, model_path: str | Path | None = None,
                 checkpoint_dir: str | Path | None = None):
        self.seed = seed
        self._model_path = Path(model_path) if model_path else _DEFAULT_CHECKPOINT
        self._model = None
        self._fov_nm: tuple[float, float] = (1.0, 1.0)
        self._grid_shape: tuple[int, int] = (1, 1)
        self._pixel_size_nm: float = 0.005
        self._tile_pool = 8
        self._overview_pool = 4

        from stem_gym.baselines.equipped_baselines import _load_analyst
        self.analyst = _load_analyst(checkpoint_dir)

    def _ensure_model(self):
        if self._model is not None:
            return
        if not self._model_path.exists():
            self._model = None
            return
        try:
            from stable_baselines3 import DQN
            self._model = DQN.load(str(self._model_path))
        except Exception:
            self._model = None

    def _pool(self, img: np.ndarray, pool_size: int) -> np.ndarray:
        h, w = img.shape
        ph = h // pool_size
        pw = w // pool_size
        cropped = img[: ph * pool_size, : pw * pool_size]
        pooled = cropped.reshape(ph, pool_size, pw, pool_size).mean(axis=(1, 3))
        return pooled.flatten().astype(np.float32)

    def _flatten_obs(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        tile_feat = self._pool(obs["tile"], self._tile_pool)
        overview_feat = self._pool(obs["overview"], self._overview_pool)
        pos = obs["position"]
        fov_h, fov_w = self._fov_nm
        scalars = np.array([
            pos[0] / max(fov_w, 1.0),
            pos[1] / max(fov_h, 1.0),
            obs["budget_remaining"][0],
            obs["step_count"][0],
        ], dtype=np.float32)
        return np.concatenate([tile_feat, overview_feat, scalars])

    def _nm_to_grid(self, x_nm: float, y_nm: float) -> tuple[int, int]:
        grid_h, grid_w = self._grid_shape
        col = int(np.clip(x_nm / self._fov_nm[1] * grid_w, 0, grid_w - 1))
        row = int(np.clip(y_nm / self._fov_nm[0] * grid_h, 0, grid_h - 1))
        return row, col

    def reset(self, obs, info):
        self.analyst.reset()
        self._fov_nm = info.get("fov_nm", (100.0, 100.0))
        self._grid_shape = obs["visited_mask"].shape
        self._pixel_size_nm = info.get("pixel_size_nm", 0.005)
        self._ensure_model()
        self._navigate_phase = True
        self._rng = np.random.default_rng(self.seed)

    def act(self, obs, info):
        budget_frac = float(obs["budget_remaining"][0])
        if budget_frac < 0.01:
            return self._make_done_action()

        if self._navigate_phase:
            self._navigate_phase = False
            flat_obs = self._flatten_obs(obs)

            if self._model is not None:
                action_id, _ = self._model.predict(flat_obs, deterministic=True)
                action_id = int(action_id)
            else:
                action_id = int(self._rng.integers(0, N_ACTIONS))

            spatial_id = action_id // N_ACTION_TYPES
            row = spatial_id // SPATIAL_GRID
            col = spatial_id % SPATIAL_GRID
            fov_h, fov_w = self._fov_nm
            x_nm = (col + 0.5) / SPATIAL_GRID * fov_w
            y_nm = (row + 0.5) / SPATIAL_GRID * fov_h

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
                    tile,
                    tile_position_nm=(x_nm, y_nm),
                    pixel_size_nm=self._pixel_size_nm,
                    tile_key=(row, col),
                )
            return {
                "action_type": ActionType.ACQUIRE,
                "position": obs["position"].copy(),
                "zoom": np.array([1.0], dtype=np.float32),
                "dwell": np.array([1.0], dtype=np.float32),
                "finding_type": 0,
            }

    def get_findings(self):
        from stem_gym.baselines.equipped_baselines import _assemble_findings
        return _assemble_findings(self.analyst, self._grid_shape)
