"""STEMEnv: Gymnasium environment for autonomous STEM microscopy."""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from stem_gym.envs.world import World
from stem_gym.state import ExperimentState
from stem_gym.tasks import DummyTask, Task


class ActionType(IntEnum):
    NAVIGATE = 0   # Move probe — free (no dose cost)
    ACQUIRE = 1    # Acquire tile — costs dose
    DECLARE = 2    # Declare a finding
    DONE = 3       # End episode


# Default acquisition parameters
DEFAULT_TILE_SIZE_PX = 128
DEFAULT_DWELL_US = 1.0
DEFAULT_DOSE_PER_ACQUIRE = 100.0  # e⁻/Å² per acquisition


class STEMEnv(gym.Env):
    """Gymnasium environment for the STEMGym benchmark.

    Action space (Dict):
        action_type: Discrete(4) — NAVIGATE, ACQUIRE, DECLARE, DONE
        position: Box(2) — (x_nm, y_nm) target position (used by NAVIGATE)
        zoom: Box(1) — zoom level 1-4 (used by ACQUIRE)
        dwell: Box(1) — dwell time multiplier (used by ACQUIRE)
        finding_type: Discrete(8) — defect/phase class (used by DECLARE)

    Observation space (Dict):
        tile: Box(tile_size, tile_size) — last acquired tile (zeros if none)
        overview: Box(64, 64) — overview crop around current position
        position: Box(2) — current (x_nm, y_nm)
        budget_remaining: Box(1) — fraction of budget remaining [0, 1]
        step_count: Box(1) — fraction of max steps used [0, 1]
        visited_mask: Box(grid_h, grid_w) — binary mask of visited tiles
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        world_path: str | Path,
        task: Task | None = None,
        total_budget: float = 1e4,
        max_steps: int = 200,
        dose_per_acquire: float = DEFAULT_DOSE_PER_ACQUIRE,
        overview_size: int = 64,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.world = World(world_path)
        self.task = task or DummyTask()
        self.dose_per_acquire = dose_per_acquire
        self.overview_size = overview_size
        self.render_mode = render_mode

        # World geometry
        grid_h, grid_w = self.world.grid_shape
        tile_size = self.world.tile_size_px
        fov_h, fov_w = self.world.fov_nm

        # State template
        self._init_budget = total_budget
        self._init_max_steps = max_steps

        # --- Action space (MineRL-style Dict) ---
        self.action_space = spaces.Dict({
            "action_type": spaces.Discrete(4),
            "position": spaces.Box(
                low=np.array([0.0, 0.0], dtype=np.float32),
                high=np.array([fov_w, fov_h], dtype=np.float32),
            ),
            "zoom": spaces.Box(low=1.0, high=4.0, shape=(1,), dtype=np.float32),
            "dwell": spaces.Box(low=0.1, high=10.0, shape=(1,), dtype=np.float32),
            "finding_type": spaces.Discrete(8),
        })

        # --- Observation space ---
        self.observation_space = spaces.Dict({
            "tile": spaces.Box(
                low=0.0, high=np.inf, shape=(tile_size, tile_size), dtype=np.float32,
            ),
            "overview": spaces.Box(
                low=0.0, high=np.inf,
                shape=(overview_size, overview_size), dtype=np.float32,
            ),
            "position": spaces.Box(
                low=np.array([0.0, 0.0], dtype=np.float32),
                high=np.array([fov_w, fov_h], dtype=np.float32),
            ),
            "budget_remaining": spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32,
            ),
            "step_count": spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32,
            ),
            "visited_mask": spaces.Box(
                low=0.0, high=1.0, shape=(grid_h, grid_w), dtype=np.float32,
            ),
        })

        # State
        self.state: ExperimentState | None = None
        self._last_tile: np.ndarray = np.zeros((tile_size, tile_size), dtype=np.float32)
        self._terminated = False
        self._truncated = False

    def _build_state(self) -> ExperimentState:
        grid_h, grid_w = self.world.grid_shape
        return ExperimentState(
            world_name=self.world.path.stem,
            world_shape=self.world.world_shape_px,
            pixel_size_nm=self.world.pixel_size_nm,
            fov_nm=self.world.fov_nm,
            total_budget=self._init_budget,
            dose_used=0.0,
            step_count=0,
            max_steps=self._init_max_steps,
            position=(self.world.fov_nm[1] / 2, self.world.fov_nm[0] / 2),  # center
            visited_mask=np.zeros((grid_h, grid_w), dtype=bool),
            dose_map=np.zeros((grid_h, grid_w), dtype=np.float32),
            tiles_acquired=0,
        )

    def _get_obs(self) -> dict[str, np.ndarray]:
        s = self.state
        x, y = s.position
        overview = self.world.get_overview_at(x, y, self.overview_size)
        return {
            "tile": self._last_tile.copy(),
            "overview": overview,
            "position": np.array(s.position, dtype=np.float32),
            "budget_remaining": np.array(
                [s.budget_remaining / s.total_budget], dtype=np.float32
            ),
            "step_count": np.array(
                [s.step_count / max(s.max_steps, 1)], dtype=np.float32
            ),
            "visited_mask": s.visited_mask.astype(np.float32),
        }

    def _get_info(self) -> dict[str, Any]:
        s = self.state
        return {
            "dose_used": s.dose_used,
            "budget_remaining": s.budget_remaining,
            "tiles_acquired": s.tiles_acquired,
            "step_count": s.step_count,
            "position": s.position,
            "fov_nm": s.fov_nm,
            "grid_shape": self.world.grid_shape,
            "pixel_size_nm": s.pixel_size_nm,
            "task_score": None,  # filled on episode end
        }

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self.state = self._build_state()
        tile_size = self.world.tile_size_px
        self._last_tile = np.zeros((tile_size, tile_size), dtype=np.float32)
        self._terminated = False
        self._truncated = False
        return self._get_obs(), self._get_info()

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Must call reset() before step()")

        s = self.state
        s.step_count += 1
        reward = 0.0
        action_type = int(action["action_type"])

        if action_type == ActionType.NAVIGATE:
            # Move probe — free
            pos = action["position"]
            x = float(np.clip(pos[0], 0, s.fov_nm[1]))
            y = float(np.clip(pos[1], 0, s.fov_nm[0]))
            s.position = (x, y)

        elif action_type == ActionType.ACQUIRE:
            # Acquire tile at current position — costs dose
            zoom = float(action.get("zoom", [1.0])[0]) if "zoom" in action else 1.0
            dwell = float(action.get("dwell", [1.0])[0]) if "dwell" in action else 1.0
            dose_cost = self.dose_per_acquire * dwell * (zoom ** 2)

            if s.budget_remaining >= dose_cost:
                # Get tile at current position
                row, col = self.world.nm_to_tile(*s.position)
                clean_tile = self.world.get_tile(row, col)

                # Apply noise based on dose
                noisy_tile = self.world.apply_poisson_noise(clean_tile, dose_cost)
                self._last_tile = noisy_tile

                # Update bookkeeping
                s.dose_used += dose_cost
                s.dose_map[row, col] += dose_cost
                if not s.visited_mask[row, col]:
                    s.visited_mask[row, col] = True
                    s.tiles_acquired += 1
            else:
                # Not enough budget — no-op
                pass

        elif action_type == ActionType.DECLARE:
            # Declare a finding at current position
            finding_type_idx = int(action.get("finding_type", 0))
            finding_types = [
                "pristine", "vacancy", "substitution", "interstitial",
                "phase_boundary", "interface", "target", "other",
            ]
            ft = finding_types[min(finding_type_idx, len(finding_types) - 1)]
            from stem_gym.state import Finding
            s.findings.append(Finding(
                finding_type=ft,
                position=s.position,
            ))

        elif action_type == ActionType.DONE:
            # Agent declares episode finished
            self._terminated = True

        # Check termination conditions
        if s.budget_remaining <= 0:
            self._terminated = True
        if s.step_count >= s.max_steps:
            self._truncated = True

        # Score on episode end
        info = self._get_info()
        if self._terminated or self._truncated:
            gt = self.world.get_ground_truth()
            info["task_score"] = self.task.score(s.to_findings_dict(), gt)
            reward = info["task_score"]

        return self._get_obs(), reward, self._terminated, self._truncated, info

    def close(self) -> None:
        self.world.close()
        super().close()
