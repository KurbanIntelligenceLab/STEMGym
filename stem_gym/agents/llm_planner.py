"""LLM Planner: LangGraph ReAct agent for high-level experiment planning.

Uses an LLM (via OpenRouter) as a strategic planner that is invoked every
K steps or on significant events. Between invocations, the rule-based
Planner handles fast-loop decisions. The LLM can:

- Query experiment state (coverage, budget, defects, navigator model)
- Set the planner mode (SURVEY, INVESTIGATE, CHARACTERIZE, CENSUS)
- Adjust the exploration parameter (beta)
- Log hypotheses about the sample
- Trigger early termination

Architecture (agent harness pattern):
    Orchestrator loop (every step):
        rule_planner.update_mode(state)     # fast decisions
    Every K steps:
        llm_planner.invoke(state)           # strategic decisions
        -> LLM calls tools to read state and issue directives
        -> Directives modify state (mode, beta, hypotheses)
        -> Rule planner respects LLM-set mode until next invocation

Requires OPENROUTER_API_KEY environment variable. Falls back to rule-based
planner silently if unavailable or on API errors.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np

from stem_gym.agents.planner import Planner, MODE_BETA
from stem_gym.state import ExperimentState, PlannerMode, TransitionLog

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt for the LLM planner
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert STEM (Scanning Transmission Electron Microscopy) experiment planner.
You are controlling an autonomous microscope that scans a sample tile-by-tile.
Your job is to make strategic decisions about how to allocate the limited electron dose budget.

You will be invoked periodically during the experiment. Use the provided tools to:
1. Query the current experiment state (coverage, budget, defects found)
2. Decide the best strategy given what you've learned
3. Set the operating mode and exploration parameter

Operating modes:
- SURVEY: Broad exploration to build a spatial model. High beta (exploration).
- INVESTIGATE: Focus on regions of interest found during survey.
- CHARACTERIZE: Deep analysis of specific defects/features. Low beta (exploitation).
- CENSUS: Final sweep to confirm findings before termination.
- TERMINATE: End the experiment.

Key principles:
- Early in the experiment, SURVEY to build coverage
- When interesting features are found, switch to INVESTIGATE or CHARACTERIZE
- Reserve ~15-20% of budget for CENSUS to confirm findings
- Don't waste dose on already-visited uniform regions
- Log hypotheses when you notice patterns (e.g., "phase boundary near x=50nm")

Beta parameter controls exploration vs exploitation in the navigator's GP-UCB:
- High beta (2-5): explore unvisited regions
- Low beta (0.1-1): exploit known interesting regions
"""


class LLMPlanner:
    """LLM-augmented planner using LangGraph ReAct agent.

    Wraps the rule-based Planner and periodically invokes an LLM to make
    strategic decisions. The LLM reads experiment state via tools and
    issues directives (mode changes, beta adjustments, hypotheses).

    Args:
        model: OpenRouter model identifier (e.g. "anthropic/claude-sonnet-4-20250514",
            "openai/gpt-4o-mini").
        invoke_every: Invoke LLM every N orchestrator steps.
        api_key: OpenRouter API key. If None, reads OPENROUTER_API_KEY env var.
        fallback_planner: Rule-based planner for fast-loop decisions and fallback.
        max_llm_steps: Maximum tool-calling steps per LLM invocation.
        temperature: LLM temperature.
    """

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4-20250514",
        invoke_every: int = 10,
        api_key: str | None = None,
        fallback_planner: Planner | None = None,
        max_llm_steps: int = 5,
        temperature: float = 0.3,
    ):
        self.model_name = model
        self.invoke_every = invoke_every
        self.max_llm_steps = max_llm_steps
        self.temperature = temperature
        self.fallback = fallback_planner or Planner()

        # Resolve API key
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._available = bool(self._api_key)
        if not self._available:
            logger.warning(
                "OPENROUTER_API_KEY not set — LLMPlanner will use rule-based fallback"
            )

        # LangGraph agent (lazy init)
        self._agent = None
        self._step_counter = 0
        self._last_invoke_step = -1

        # Shared state reference (set during reset/update)
        self._state: ExperimentState | None = None

        # LLM invocation log
        self.invocation_log: list[dict[str, Any]] = []

    def _build_agent(self):
        """Lazily build the LangGraph ReAct agent with tools."""
        from langchain_openai import ChatOpenAI
        from langgraph.prebuilt import create_react_agent

        llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=self._api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        tools = [
            self._make_query_state_tool(),
            self._make_query_navigator_tool(),
            self._make_set_mode_tool(),
            self._make_set_beta_tool(),
            self._make_log_hypothesis_tool(),
            self._make_terminate_tool(),
        ]

        self._agent = create_react_agent(llm, tools)

    # -- Tool factories (bound to self._state) --

    def _make_query_state_tool(self):
        from langchain_core.tools import tool

        planner_ref = self

        @tool
        def query_state() -> str:
            """Query the current experiment state: coverage, budget, defects, mode, position."""
            s = planner_ref._state
            if s is None:
                return "No state available"

            coverage = 0.0
            if s.visited_mask is not None:
                coverage = float(np.sum(s.visited_mask)) / max(s.visited_mask.size, 1)

            budget_frac = s.budget_remaining / max(s.total_budget, 1e-6)
            n_defects = sum(v for k, v in s.defect_census.items() if k != "pristine")

            return json.dumps({
                "mode": s.mode.name,
                "beta": s.planner_beta,
                "coverage_fraction": round(coverage, 3),
                "tiles_acquired": s.tiles_acquired,
                "budget_remaining_fraction": round(budget_frac, 3),
                "dose_used": round(s.dose_used, 1),
                "total_budget": round(s.total_budget, 1),
                "step_count": s.step_count,
                "position_nm": [round(s.position[0], 2), round(s.position[1], 2)],
                "fov_nm": [round(s.fov_nm[0], 2), round(s.fov_nm[1], 2)],
                "defect_census": dict(s.defect_census),
                "total_defects": n_defects,
                "n_hypotheses": len(s.hypotheses),
                "n_transitions": len(s.transition_log),
            })

        return query_state

    def _make_query_navigator_tool(self):
        from langchain_core.tools import tool

        planner_ref = self

        @tool
        def query_navigator() -> str:
            """Query the navigator's GP model summary: number of observations, visited tile count, current beta."""
            s = planner_ref._state
            if s is None:
                return "No state available"

            visited_count = 0
            total_tiles = 1
            if s.visited_mask is not None:
                visited_count = int(np.sum(s.visited_mask))
                total_tiles = s.visited_mask.size

            # Find regions with high/low dose
            high_dose_tiles = 0
            low_dose_tiles = 0
            if s.dose_map is not None:
                high_dose_tiles = int(np.sum(s.dose_map > s.dose_map.mean() + s.dose_map.std()))
                low_dose_tiles = int(np.sum((s.dose_map > 0) & (s.dose_map < s.dose_map.mean())))

            return json.dumps({
                "visited_tiles": visited_count,
                "total_tiles": total_tiles,
                "unvisited_tiles": total_tiles - visited_count,
                "current_beta": s.planner_beta,
                "high_dose_tiles": high_dose_tiles,
                "low_dose_tiles": low_dose_tiles,
            })

        return query_navigator

    def _make_set_mode_tool(self):
        from langchain_core.tools import tool

        planner_ref = self

        @tool
        def set_mode(mode: str) -> str:
            """Set the planner operating mode. Valid modes: SURVEY, INVESTIGATE, CHARACTERIZE, CENSUS, TERMINATE.

            Args:
                mode: The mode name (case-insensitive).
            """
            s = planner_ref._state
            if s is None:
                return "Error: no state"

            mode_upper = mode.upper().strip()
            try:
                new_mode = PlannerMode[mode_upper]
            except KeyError:
                return f"Error: invalid mode '{mode}'. Valid: SURVEY, INVESTIGATE, CHARACTERIZE, CENSUS, TERMINATE"

            old_mode = s.mode
            if new_mode != old_mode:
                log = TransitionLog(
                    step=s.step_count,
                    from_mode=old_mode,
                    to_mode=new_mode,
                    reason=f"LLM planner directive",
                    budget_remaining=s.budget_remaining,
                )
                s.transition_log.append(log)
                s.mode = new_mode
                s.planner_beta = MODE_BETA.get(new_mode, s.planner_beta)
                logger.debug(f"LLMPlanner: {old_mode.name} -> {new_mode.name}")
                return f"Mode changed: {old_mode.name} -> {new_mode.name}, beta set to {s.planner_beta}"
            return f"Mode already {mode_upper}"

        return set_mode

    def _make_set_beta_tool(self):
        from langchain_core.tools import tool

        planner_ref = self

        @tool
        def set_beta(beta: float) -> str:
            """Set the GP-UCB exploration parameter (beta). Higher = more exploration.

            Args:
                beta: Exploration parameter, typically 0.1 to 5.0.
            """
            s = planner_ref._state
            if s is None:
                return "Error: no state"
            beta = max(0.0, min(10.0, beta))
            s.planner_beta = beta
            return f"Beta set to {beta:.2f}"

        return set_beta

    def _make_log_hypothesis_tool(self):
        from langchain_core.tools import tool

        planner_ref = self

        @tool
        def log_hypothesis(hypothesis: str) -> str:
            """Log a hypothesis about the sample (e.g., 'possible phase boundary near x=50nm').

            Args:
                hypothesis: A description of the hypothesis.
            """
            s = planner_ref._state
            if s is None:
                return "Error: no state"
            entry = {
                "step": s.step_count,
                "hypothesis": hypothesis,
                "budget_frac": round(s.budget_remaining / max(s.total_budget, 1e-6), 3),
            }
            s.hypotheses.append(entry)
            return f"Hypothesis logged (total: {len(s.hypotheses)})"

        return log_hypothesis

    def _make_terminate_tool(self):
        from langchain_core.tools import tool

        planner_ref = self

        @tool
        def terminate(reason: str) -> str:
            """Terminate the experiment early. Use when findings are sufficient or budget should be conserved.

            Args:
                reason: Why the experiment should end.
            """
            s = planner_ref._state
            if s is None:
                return "Error: no state"

            old_mode = s.mode
            if old_mode != PlannerMode.TERMINATE:
                log = TransitionLog(
                    step=s.step_count,
                    from_mode=old_mode,
                    to_mode=PlannerMode.TERMINATE,
                    reason=f"LLM terminate: {reason}",
                    budget_remaining=s.budget_remaining,
                )
                s.transition_log.append(log)
                s.mode = PlannerMode.TERMINATE
                s.planner_beta = 0.0
            return f"Experiment terminated: {reason}"

        return terminate

    # -- Main interface (same as Planner) --

    def reset(self, state: ExperimentState) -> None:
        """Reset for a new episode."""
        self._state = state
        self._step_counter = 0
        self._last_invoke_step = -1
        self.invocation_log.clear()
        self.fallback.reset(state)

    def update_mode(self, state: ExperimentState) -> PlannerMode:
        """Update mode — delegates to rule planner, periodically invokes LLM."""
        self._state = state
        self._step_counter += 1

        # Always run rule-based planner for budget safety checks
        self.fallback.update_mode(state)

        # Invoke LLM every K steps (if available)
        if (
            self._available
            and self._step_counter - self._last_invoke_step >= self.invoke_every
            and state.mode != PlannerMode.TERMINATE
        ):
            self._invoke_llm(state)
            self._last_invoke_step = self._step_counter

        return state.mode

    def should_acquire(
        self, state: ExperimentState, overview_std: float = 0.0,
    ) -> bool:
        """Delegate to fallback planner."""
        return self.fallback.should_acquire(state, overview_std)

    def _invoke_llm(self, state: ExperimentState) -> None:
        """Invoke the LLM agent to make strategic decisions."""
        if self._agent is None:
            try:
                self._build_agent()
            except Exception as e:
                logger.warning(f"Failed to build LLM agent: {e}")
                self._available = False
                return

        budget_frac = state.budget_remaining / max(state.total_budget, 1e-6)
        coverage = 0.0
        if state.visited_mask is not None:
            coverage = float(np.sum(state.visited_mask)) / max(state.visited_mask.size, 1)
        n_defects = sum(v for k, v in state.defect_census.items() if k != "pristine")

        user_msg = (
            f"Step {state.step_count}: The experiment is in {state.mode.name} mode. "
            f"Coverage: {coverage:.1%}, Budget remaining: {budget_frac:.1%}, "
            f"Defects found: {n_defects}, Tiles acquired: {state.tiles_acquired}. "
            f"Review the state and decide the next strategic action. "
            f"Use tools to query details and adjust the plan."
        )

        try:
            result = self._agent.invoke(
                {"messages": [("system", SYSTEM_PROMPT), ("user", user_msg)]},
                config={"recursion_limit": self.max_llm_steps + 2},
            )

            # Extract the final message content for logging
            final_msg = ""
            if result.get("messages"):
                last = result["messages"][-1]
                final_msg = getattr(last, "content", str(last))

            self.invocation_log.append({
                "step": state.step_count,
                "mode_after": state.mode.name,
                "beta_after": state.planner_beta,
                "summary": final_msg[:200] if final_msg else "",
            })

            logger.debug(
                f"LLMPlanner invoked at step {state.step_count}: "
                f"mode={state.mode.name}, beta={state.planner_beta:.2f}"
            )

        except Exception as e:
            logger.warning(f"LLM invocation failed: {e}")
            self.invocation_log.append({
                "step": state.step_count,
                "error": str(e),
            })
