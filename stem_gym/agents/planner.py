"""Rule-based FSM Planner for the multi-agent system.

Implements a finite state machine with modes:
    SURVEY → INVESTIGATE → CHARACTERIZE → CENSUS → TERMINATE

Each mode adjusts the Navigator's exploration parameter (beta) and decides
whether to acquire at the current position.  Transition conditions are based
on coverage, defect findings, and budget remaining.

Structured transition logging produces qualitative figures for the paper.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from stem_gym.state import ExperimentState, PlannerMode, TransitionLog

logger = logging.getLogger(__name__)


# Beta schedule per mode (higher = more exploration)
MODE_BETA = {
    PlannerMode.SURVEY: 3.0,
    PlannerMode.INVESTIGATE: 2.0,
    PlannerMode.CHARACTERIZE: 0.5,
    PlannerMode.CENSUS: 1.0,
    PlannerMode.TERMINATE: 0.0,
}

# Minimum coverage fraction to leave SURVEY
SURVEY_COVERAGE_THRESHOLD = 0.15

# Minimum defects found to trigger CHARACTERIZE
INVESTIGATE_DEFECT_THRESHOLD = 2

# Budget fraction to enter CENSUS (last portion of budget for confirming)
CENSUS_BUDGET_THRESHOLD = 0.20

# Budget fraction to trigger TERMINATE
TERMINATE_BUDGET_THRESHOLD = 0.05


class Planner:
    """Rule-based FSM planner for the STEMAgent.

    Manages the experiment lifecycle through a sequence of modes, each
    with different acquisition strategies:

    - **SURVEY**: Explore broadly to build a spatial model. High beta.
    - **INVESTIGATE**: Focus on interesting regions found during survey.
    - **CHARACTERIZE**: Deep analysis of specific defects/features. Low beta.
    - **CENSUS**: Final sweep to confirm findings before termination.
    - **TERMINATE**: Signal the orchestrator to end the episode.

    The planner reads from ExperimentState and writes mode + beta back.
    """

    def __init__(
        self,
        survey_coverage: float = SURVEY_COVERAGE_THRESHOLD,
        investigate_defects: int = INVESTIGATE_DEFECT_THRESHOLD,
        census_budget: float = CENSUS_BUDGET_THRESHOLD,
        terminate_budget: float = TERMINATE_BUDGET_THRESHOLD,
    ):
        self.survey_coverage = survey_coverage
        self.investigate_defects = investigate_defects
        self.census_budget = census_budget
        self.terminate_budget = terminate_budget

    def reset(self, state: ExperimentState) -> None:
        """Reset to SURVEY mode at episode start."""
        state.mode = PlannerMode.SURVEY
        state.planner_beta = MODE_BETA[PlannerMode.SURVEY]
        state.transition_log.clear()

    def update_mode(self, state: ExperimentState) -> PlannerMode:
        """Evaluate transition conditions and update the planner mode.

        Called once per orchestrator step. Returns the (possibly new) mode.
        """
        old_mode = state.mode
        new_mode = old_mode

        budget_frac = state.budget_remaining / max(state.total_budget, 1e-6)
        coverage = self._coverage_fraction(state)
        n_defects = self._total_defects(state)

        # --- Transition logic ---

        # Budget-based overrides (highest priority)
        if budget_frac <= self.terminate_budget:
            new_mode = PlannerMode.TERMINATE

        elif budget_frac <= self.census_budget and old_mode not in (
            PlannerMode.CENSUS, PlannerMode.TERMINATE
        ):
            new_mode = PlannerMode.CENSUS

        # Mode-specific transitions
        elif old_mode == PlannerMode.SURVEY:
            if coverage >= self.survey_coverage:
                # Have enough coverage — move to INVESTIGATE
                new_mode = PlannerMode.INVESTIGATE

        elif old_mode == PlannerMode.INVESTIGATE:
            if n_defects >= self.investigate_defects:
                # Found enough defects — characterize them
                new_mode = PlannerMode.CHARACTERIZE
            elif coverage >= 0.5:
                # Explored a lot without finding defects — go to CENSUS
                new_mode = PlannerMode.CENSUS

        elif old_mode == PlannerMode.CHARACTERIZE:
            # Stay in CHARACTERIZE until budget forces transition
            pass

        elif old_mode == PlannerMode.CENSUS:
            # Stay in CENSUS until budget forces TERMINATE
            pass

        # Apply transition
        if new_mode != old_mode:
            self._transition(state, old_mode, new_mode,
                             f"coverage={coverage:.2f}, defects={n_defects}, "
                             f"budget={budget_frac:.2f}")

        return state.mode

    def should_acquire(
        self, state: ExperimentState, overview_std: float = 0.0,
    ) -> bool:
        """Decide whether the orchestrator should acquire at the current position.

        In SURVEY mode, always acquire (building the spatial model).
        In other modes, may skip acquisition of low-interest regions.

        Args:
            state: Current experiment state.
            overview_std: Standard deviation of the overview image at current
                position (proxy for how interesting the location looks).

        Returns:
            True if the agent should acquire a tile, False to skip.
        """
        if state.mode == PlannerMode.TERMINATE:
            return False

        if state.mode == PlannerMode.SURVEY:
            # Always acquire in survey mode
            return True

        if state.mode == PlannerMode.CENSUS:
            # Acquire only unvisited tiles
            return not self._is_current_visited(state)

        if state.mode == PlannerMode.INVESTIGATE:
            # Acquire if position looks interesting or is unvisited
            if not self._is_current_visited(state):
                return True
            return overview_std > 0.05

        if state.mode == PlannerMode.CHARACTERIZE:
            # Always acquire — we're doing detailed analysis
            return True

        return True

    def _transition(
        self, state: ExperimentState,
        from_mode: PlannerMode, to_mode: PlannerMode,
        reason: str,
    ) -> None:
        """Record a mode transition."""
        log = TransitionLog(
            step=state.step_count,
            from_mode=from_mode,
            to_mode=to_mode,
            reason=reason,
            budget_remaining=state.budget_remaining,
        )
        state.transition_log.append(log)
        state.mode = to_mode
        state.planner_beta = MODE_BETA[to_mode]
        logger.debug(
            f"Planner: {from_mode.name} → {to_mode.name} "
            f"(step={state.step_count}, {reason})"
        )

    def _coverage_fraction(self, state: ExperimentState) -> float:
        """Fraction of tiles that have been visited."""
        if state.visited_mask is None:
            return 0.0
        total = state.visited_mask.size
        visited = int(np.sum(state.visited_mask))
        return visited / max(total, 1)

    def _total_defects(self, state: ExperimentState) -> int:
        """Total number of non-pristine defects found."""
        return sum(v for k, v in state.defect_census.items() if k != "pristine")

    def _is_current_visited(self, state: ExperimentState) -> bool:
        """Check if the current position has already been visited."""
        if state.visited_mask is None:
            return False
        # Convert position to grid index
        grid_h, grid_w = state.visited_mask.shape
        fov_h, fov_w = state.fov_nm
        x, y = state.position
        col = int(np.clip(x / fov_w * grid_w, 0, grid_w - 1))
        row = int(np.clip(y / fov_h * grid_h, 0, grid_h - 1))
        return bool(state.visited_mask[row, col])
