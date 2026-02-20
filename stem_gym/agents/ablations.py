"""Ablation variants of STEMAgent for the ablation study.

Each variant removes or swaps one component to measure its contribution:

- STEMAgentNoPlanner: Navigator + Analyst, no FSM planner (always SURVEY mode)
- STEMAgentNoUncertainty: Full pipeline but analyst has no ensemble uncertainty
- STEMAgentLLM: Full pipeline with LLMPlanner instead of rule-based Planner
- STEMAgentRulePlanner: Alias for the standard STEMAgent (explicit baseline name)

All implement the Agent protocol for use with run_benchmark().
"""

from __future__ import annotations

from typing import Any

import numpy as np

from stem_gym.agents.analyst import Analyst
from stem_gym.agents.multi_agent import STEMAgent
from stem_gym.agents.navigator import Navigator
from stem_gym.agents.planner import Planner
from stem_gym.state import ExperimentState, PlannerMode


# ---------------------------------------------------------------------------
# No-Planner ablation: always stays in SURVEY mode
# ---------------------------------------------------------------------------

class _NullPlanner:
    """A planner that does nothing — always stays in SURVEY with fixed beta."""

    def __init__(self, beta: float = 2.0):
        self._beta = beta

    def reset(self, state: ExperimentState) -> None:
        state.mode = PlannerMode.SURVEY
        state.planner_beta = self._beta
        state.transition_log.clear()

    def update_mode(self, state: ExperimentState) -> PlannerMode:
        # Only terminate on budget exhaustion
        budget_frac = state.budget_remaining / max(state.total_budget, 1e-6)
        if budget_frac <= 0.02:
            state.mode = PlannerMode.TERMINATE
        return state.mode

    def should_acquire(
        self, state: ExperimentState, overview_std: float = 0.0,
    ) -> bool:
        return state.mode != PlannerMode.TERMINATE


class STEMAgentNoPlanner(STEMAgent):
    """STEMAgent without the FSM planner (ablation).

    Uses Navigator + Analyst but no strategic mode transitions.
    Always acquires (SURVEY mode), demonstrating the value of the planner.
    """

    def __init__(self, seed: int = 42, beta: float = 2.0):
        super().__init__(
            navigator=Navigator(seed=seed, beta=beta),
            planner=_NullPlanner(beta=beta),
            seed=seed,
        )


# ---------------------------------------------------------------------------
# No-Uncertainty ablation: analyst without ensemble
# ---------------------------------------------------------------------------

class STEMAgentNoUncertainty(STEMAgent):
    """STEMAgent where the analyst has no ensemble uncertainty (ablation).

    Uses the same trained models but forces the AtomFinder to use only
    one ensemble member, removing predictive uncertainty estimation.
    Falls back to SimpleAnalyst if no checkpoints found.
    """

    def __init__(self, seed: int = 42):
        from stem_gym.agents.simple_analyst import SimpleAnalyst as _SA

        # Try to load trained analyst, then disable ensemble uncertainty
        analyst = self._try_load_analyst(None)
        if analyst is not None and analyst.atom_finder is not None:
            # Force single-member prediction (no uncertainty)
            analyst.atom_finder._no_uncertainty = True
        elif analyst is None:
            # No trained models — use SimpleAnalyst (no uncertainty by design)
            analyst = _SA()

        super().__init__(
            analyst=analyst,
            navigator=Navigator(seed=seed),
            planner=Planner(),
            seed=seed,
        )


# ---------------------------------------------------------------------------
# LLM Planner variant
# ---------------------------------------------------------------------------

class STEMAgentLLM(STEMAgent):
    """STEMAgent with LLM Planner via OpenRouter (ablation / enhancement).

    Uses LangGraph ReAct agent for strategic planning decisions every K steps.
    Falls back to rule-based planner if OPENROUTER_API_KEY is not set.

    Args:
        model: OpenRouter model identifier.
        invoke_every: Invoke LLM every N steps.
        seed: Random seed.
    """

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4-20250514",
        invoke_every: int = 10,
        seed: int = 42,
    ):
        from stem_gym.agents.llm_planner import LLMPlanner

        llm_planner = LLMPlanner(
            model=model,
            invoke_every=invoke_every,
        )
        super().__init__(
            navigator=Navigator(seed=seed),
            planner=llm_planner,
            seed=seed,
        )

    @property
    def llm_planner(self):
        """Access the LLMPlanner for inspection."""
        return self.planner


# ---------------------------------------------------------------------------
# Explicit alias for the standard STEMAgent
# ---------------------------------------------------------------------------

class STEMAgentRulePlanner(STEMAgent):
    """Standard STEMAgent with rule-based planner (explicit ablation baseline name)."""

    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)
