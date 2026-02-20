"""STEMAgent: multi-agent orchestrator (Navigator + Analyst + Planner).

Implements the Agent protocol and wires together the three sub-agents
in a sequential loop:

    1. planner.update_mode(state)
    2. pos = navigator.select_next(state)
    3. navigate to pos (free peek via overview)
    4. if planner.should_acquire(state, obs): acquire + analyst.analyze_tile()
    5. navigator.record_observation(state, tile, pos)
    6. check termination

The orchestrator alternates between NAVIGATE and ACQUIRE actions over
successive env.step() calls, maintaining an internal FSM for the
two-step acquire cycle.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pathlib import Path

from stem_gym.baselines import Agent
from stem_gym.agents.analyst import Analyst
from stem_gym.agents.navigator import Navigator
from stem_gym.agents.planner import Planner
from stem_gym.agents.simple_analyst import SimpleAnalyst
from stem_gym.envs.stem_env import ActionType
from stem_gym.state import ExperimentState, PlannerMode


class STEMAgent(Agent):
    """Multi-agent system: Planner + Navigator + Analyst.

    On each call to act(), the agent either navigates to a new position
    or acquires/skips at the current position. The internal state machine
    alternates between these phases.

    Args:
        analyst: Analyst ensemble (uses SimpleAnalyst if None).
        navigator: GP-UCB Navigator (default parameters if None).
        planner: FSM Planner (default parameters if None).
        seed: Random seed.
        checkpoint_dir: Path to trained model checkpoints. If provided and
            analyst is None, loads models from this directory.
    """

    # Default checkpoint path — set by user or auto-detected
    DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"

    def __init__(
        self,
        analyst: Analyst | None = None,
        navigator: Navigator | None = None,
        planner: Planner | None = None,
        seed: int = 42,
        checkpoint_dir: str | Path | None = None,
    ):
        if analyst is None:
            analyst = self._try_load_analyst(checkpoint_dir)
        self.analyst = analyst or Analyst()
        self.navigator = navigator or Navigator(seed=seed)
        self.planner = planner or Planner()
        self.seed = seed

        # Internal state
        self._state: ExperimentState | None = None
        self._phase: str = "navigate"  # "navigate" or "acquire"
        self._target_pos: tuple[float, float] = (0.0, 0.0)
        self._skip_acquire: bool = False

        # Track phase labels per tile for phase map reconstruction
        self._tile_phases: dict[tuple[int, int], int] = {}

    @staticmethod
    def _try_load_analyst(checkpoint_dir: str | Path | None) -> Analyst | None:
        """Try to load trained analyst from checkpoints."""
        if checkpoint_dir is not None:
            cp = Path(checkpoint_dir)
        else:
            cp = STEMAgent.DEFAULT_CHECKPOINT_DIR
        if cp.exists() and (cp / "atom_finder.pt").exists():
            try:
                return Analyst.from_checkpoint(cp)
            except Exception:
                pass
        return None

    def reset(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> None:
        """Initialize sub-agents and internal state for a new episode."""
        grid_shape = info.get("grid_shape", obs["visited_mask"].shape)
        fov_nm = info.get("fov_nm", (100.0, 100.0))
        pixel_size_nm = info.get("pixel_size_nm", 0.005)

        # Build internal ExperimentState for sub-agents
        grid_h, grid_w = grid_shape
        self._state = ExperimentState(
            fov_nm=fov_nm,
            pixel_size_nm=pixel_size_nm,
            total_budget=float(obs["budget_remaining"][0]) * 1.0,  # we'll track manually
            visited_mask=np.zeros((grid_h, grid_w), dtype=bool),
            dose_map=np.zeros((grid_h, grid_w), dtype=np.float32),
        )
        # Reconstruct total_budget from budget_remaining fraction
        # budget_remaining obs is fraction in [0, 1], info has absolute values
        if "budget_remaining" in info:
            self._state.total_budget = float(info["budget_remaining"])
        elif "dose_used" in info:
            budget_frac = float(obs["budget_remaining"][0])
            if budget_frac > 0:
                self._state.total_budget = float(info["dose_used"]) / max(1 - budget_frac, 1e-6)

        self.navigator.reset(self._state)
        self.planner.reset(self._state)
        self.analyst.reset()

        self._phase = "navigate"
        self._skip_acquire = False
        self._tile_phases.clear()

    def act(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> dict[str, Any]:
        """Select an action: navigate → acquire/skip → repeat."""
        s = self._state

        # Sync state from env observations
        self._sync_state(obs, info)

        budget_frac = float(obs["budget_remaining"][0])
        if budget_frac < 0.01 or s.mode == PlannerMode.TERMINATE:
            return self._make_action(ActionType.DONE)

        if self._phase == "navigate":
            # 1. Planner updates mode and beta
            self.planner.update_mode(s)
            self.navigator.set_beta(s.planner_beta)

            if s.mode == PlannerMode.TERMINATE:
                return self._make_action(ActionType.DONE)

            # 2. Navigator selects next position
            self._target_pos = self.navigator.select_next(s)

            # 3. Decide whether we'll acquire after navigating
            overview = obs.get("overview")
            overview_std = float(np.std(overview)) if overview is not None else 0.0
            self._skip_acquire = not self.planner.should_acquire(s, overview_std)

            self._phase = "acquire"
            return self._make_action(
                ActionType.NAVIGATE,
                position=np.array(self._target_pos, dtype=np.float32),
            )

        else:  # self._phase == "acquire"
            self._phase = "navigate"

            if self._skip_acquire:
                # Skip acquisition — go straight to next navigate
                # But planner may still want us to check other locations,
                # so we immediately select a new target and navigate
                self.planner.update_mode(s)
                self.navigator.set_beta(s.planner_beta)

                if s.mode == PlannerMode.TERMINATE:
                    return self._make_action(ActionType.DONE)

                self._target_pos = self.navigator.select_next(s)
                overview = obs.get("overview")
                overview_std = float(np.std(overview)) if overview is not None else 0.0
                self._skip_acquire = not self.planner.should_acquire(s, overview_std)

                self._phase = "acquire"
                return self._make_action(
                    ActionType.NAVIGATE,
                    position=np.array(self._target_pos, dtype=np.float32),
                )

            # 4. Acquire tile
            tile = obs.get("tile")
            if tile is not None and tile.max() > 0:
                # Run analyst on previously acquired tile (from last step's obs)
                pos = obs["position"]
                x_nm, y_nm = float(pos[0]), float(pos[1])
                result = self.analyst.analyze_tile(
                    tile,
                    tile_position_nm=(x_nm, y_nm),
                    pixel_size_nm=s.pixel_size_nm,
                )

                # Record phase label for this tile position
                if s.fov_nm[0] > 0 and s.fov_nm[1] > 0:
                    gh, gw = s.visited_mask.shape
                    tile_r = int(y_nm / s.fov_nm[0] * gh)
                    tile_c = int(x_nm / s.fov_nm[1] * gw)
                    tile_r = min(tile_r, gh - 1)
                    tile_c = min(tile_c, gw - 1)
                    self._tile_phases[(tile_r, tile_c)] = result.phase_label

                # Update navigator with new observation
                self.navigator.record_observation(s, tile, x_nm, y_nm)

                # Sync defect census to state
                s.defect_census = self.analyst.get_defect_census()

            return self._make_action(ActionType.ACQUIRE)

    def _sync_state(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> None:
        """Sync internal state from env observations."""
        s = self._state
        if "dose_used" in info:
            s.dose_used = float(info["dose_used"])
        if "step_count" in info:
            s.step_count = int(info["step_count"])
        if "tiles_acquired" in info:
            s.tiles_acquired = int(info["tiles_acquired"])

        # Update visited mask from obs
        visited = obs.get("visited_mask")
        if visited is not None:
            s.visited_mask = visited > 0.5

        # Update position
        pos = obs.get("position")
        if pos is not None:
            s.position = (float(pos[0]), float(pos[1]))

    def _make_action(
        self,
        action_type: ActionType,
        position: np.ndarray | None = None,
        zoom: float = 1.0,
        dwell: float = 1.0,
    ) -> dict[str, Any]:
        """Build an action dict."""
        if position is None:
            position = np.array([0.0, 0.0], dtype=np.float32)
        return {
            "action_type": action_type,
            "position": position,
            "zoom": np.array([zoom], dtype=np.float32),
            "dwell": np.array([dwell], dtype=np.float32),
            "finding_type": 0,
        }

    def get_findings(self) -> dict[str, Any]:
        """Return accumulated findings for task scoring."""
        # Build phase map from tile-level phase labels
        phase_map = None
        if self._state is not None and self._tile_phases:
            gh, gw = self._state.visited_mask.shape
            phase_map = np.zeros((gh, gw), dtype=np.int32)
            for (r, c), phase in self._tile_phases.items():
                if 0 <= r < gh and 0 <= c < gw:
                    phase_map[r, c] = phase

        # Collect defect positions from analyst — iterate positions and labels
        # in lockstep across all tiles
        defect_positions = []
        all_positions = getattr(self.analyst, '_all_atom_positions_nm', [])
        all_labels = getattr(self.analyst, '_all_defect_labels', [])

        if all_positions and all_labels:
            # Flatten world positions across all tiles
            offset = 0
            for tile_pos in all_positions:
                n = len(tile_pos)
                for j in range(n):
                    label_idx = offset + j
                    if label_idx < len(all_labels):
                        lbl = all_labels[label_idx]
                        if lbl != "pristine":
                            defect_positions.append({
                                "position": (float(tile_pos[j, 0]), float(tile_pos[j, 1])),
                                "type": lbl,
                            })
                offset += n

        # Atom positions
        atom_positions = None
        if all_positions:
            atom_positions = np.concatenate(all_positions, axis=0)

        return {
            "defect_census": self.analyst.get_defect_census(),
            "defect_positions": defect_positions,
            "phase_map": phase_map,
            "atom_positions": atom_positions,
        }
