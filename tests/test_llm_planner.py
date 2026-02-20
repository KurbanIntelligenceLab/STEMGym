"""Tests for Phase 7: LLM Planner, ablation variants."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from stem_gym.agents.llm_planner import LLMPlanner
from stem_gym.agents.planner import Planner
from stem_gym.agents.ablations import (
    STEMAgentNoPlanner,
    STEMAgentNoUncertainty,
    STEMAgentLLM,
    STEMAgentRulePlanner,
    _NullPlanner,
)
from stem_gym.agents.multi_agent import STEMAgent
from stem_gym.state import ExperimentState, PlannerMode
from stem_gym.envs.stem_env import STEMEnv, ActionType
from stem_gym.tasks.defect_census import DefectCensusTask
from stem_gym.runner import run_episode, run_benchmark


TEST_WORLD = Path(__file__).parent.parent / "data" / "simulated" / "test_world.h5"


@pytest.fixture(scope="module")
def world_path():
    if not TEST_WORLD.exists():
        pytest.skip("Test world not generated")
    return TEST_WORLD


def _make_state(grid=(8, 8), fov=(100.0, 100.0), budget=10000.0, dose_used=0.0):
    return ExperimentState(
        fov_nm=fov,
        total_budget=budget,
        dose_used=dose_used,
        visited_mask=np.zeros(grid, dtype=bool),
        dose_map=np.zeros(grid, dtype=np.float32),
    )


# ---- LLMPlanner unit tests ----

class TestLLMPlannerOffline:
    """Test LLMPlanner without an API key (fallback to rule-based)."""

    def test_fallback_without_api_key(self):
        """LLMPlanner without API key should use rule-based fallback."""
        with patch.dict(os.environ, {}, clear=False):
            # Remove key if set
            env = os.environ.copy()
            env.pop("OPENROUTER_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                planner = LLMPlanner(api_key="")
                assert not planner._available

                state = _make_state()
                planner.reset(state)
                assert state.mode == PlannerMode.SURVEY

                # Should delegate to fallback
                state.visited_mask[:2, :] = True  # 25% coverage
                mode = planner.update_mode(state)
                assert mode == PlannerMode.INVESTIGATE

    def test_reset_clears_state(self):
        planner = LLMPlanner(api_key="")
        state = _make_state()
        planner.reset(state)
        assert state.mode == PlannerMode.SURVEY
        assert planner._step_counter == 0
        assert len(planner.invocation_log) == 0

    def test_should_acquire_delegates(self):
        planner = LLMPlanner(api_key="")
        state = _make_state()
        planner.reset(state)
        assert planner.should_acquire(state) is True

        state.mode = PlannerMode.TERMINATE
        assert planner.should_acquire(state) is False

    def test_update_mode_increments_counter(self):
        planner = LLMPlanner(api_key="", invoke_every=5)
        state = _make_state()
        planner.reset(state)

        for _ in range(10):
            planner.update_mode(state)

        assert planner._step_counter == 10


class TestLLMPlannerTools:
    """Test LLMPlanner tool functions directly."""

    def test_query_state_tool(self):
        planner = LLMPlanner(api_key="")
        state = _make_state(budget=10000.0, dose_used=3000.0)
        state.defect_census = {"vacancy": 3, "substitution": 1}
        state.tiles_acquired = 5
        planner._state = state

        tool = planner._make_query_state_tool()
        result = tool.invoke({})
        import json
        data = json.loads(result)

        assert data["mode"] == "SURVEY"
        assert data["budget_remaining_fraction"] == pytest.approx(0.7, abs=0.01)
        assert data["total_defects"] == 4
        assert data["tiles_acquired"] == 5

    def test_set_mode_tool(self):
        planner = LLMPlanner(api_key="")
        state = _make_state()
        planner._state = state

        tool = planner._make_set_mode_tool()

        result = tool.invoke({"mode": "INVESTIGATE"})
        assert "INVESTIGATE" in result
        assert state.mode == PlannerMode.INVESTIGATE

    def test_set_mode_invalid(self):
        planner = LLMPlanner(api_key="")
        state = _make_state()
        planner._state = state

        tool = planner._make_set_mode_tool()
        result = tool.invoke({"mode": "INVALID"})
        assert "Error" in result
        assert state.mode == PlannerMode.SURVEY  # unchanged

    def test_set_beta_tool(self):
        planner = LLMPlanner(api_key="")
        state = _make_state()
        planner._state = state

        tool = planner._make_set_beta_tool()
        result = tool.invoke({"beta": 4.5})
        assert state.planner_beta == 4.5

    def test_set_beta_clamped(self):
        planner = LLMPlanner(api_key="")
        state = _make_state()
        planner._state = state

        tool = planner._make_set_beta_tool()
        tool.invoke({"beta": 100.0})
        assert state.planner_beta == 10.0  # clamped

    def test_log_hypothesis_tool(self):
        planner = LLMPlanner(api_key="")
        state = _make_state()
        planner._state = state

        tool = planner._make_log_hypothesis_tool()
        result = tool.invoke({"hypothesis": "Phase boundary near x=50nm"})
        assert len(state.hypotheses) == 1
        assert state.hypotheses[0]["hypothesis"] == "Phase boundary near x=50nm"

    def test_terminate_tool(self):
        planner = LLMPlanner(api_key="")
        state = _make_state()
        planner._state = state

        tool = planner._make_terminate_tool()
        result = tool.invoke({"reason": "Sufficient coverage achieved"})
        assert state.mode == PlannerMode.TERMINATE
        assert len(state.transition_log) == 1

    def test_query_navigator_tool(self):
        planner = LLMPlanner(api_key="")
        state = _make_state()
        state.visited_mask[:2, :] = True
        planner._state = state

        tool = planner._make_query_navigator_tool()
        result = tool.invoke({})
        import json
        data = json.loads(result)

        assert data["visited_tiles"] == 16  # 2 rows * 8 cols
        assert data["total_tiles"] == 64
        assert data["unvisited_tiles"] == 48


# ---- NullPlanner tests ----

class TestNullPlanner:
    def test_stays_in_survey(self):
        planner = _NullPlanner(beta=2.0)
        state = _make_state()
        planner.reset(state)
        assert state.mode == PlannerMode.SURVEY

        # Even with high coverage, stays in SURVEY
        state.visited_mask[:, :] = True
        mode = planner.update_mode(state)
        assert mode == PlannerMode.SURVEY

    def test_terminates_on_budget(self):
        planner = _NullPlanner()
        state = _make_state(budget=10000.0, dose_used=9900.0)
        planner.reset(state)
        mode = planner.update_mode(state)
        assert mode == PlannerMode.TERMINATE

    def test_always_acquires_except_terminate(self):
        planner = _NullPlanner()
        state = _make_state()
        planner.reset(state)
        assert planner.should_acquire(state) is True

        state.mode = PlannerMode.TERMINATE
        assert planner.should_acquire(state) is False


# ---- Ablation variant integration tests ----

class TestAblationVariants:
    def test_no_planner_completes_episode(self, world_path):
        env = STEMEnv(world_path, total_budget=5000.0, max_steps=100)
        agent = STEMAgentNoPlanner(seed=42)

        obs, info = env.reset(seed=42)
        agent.reset(obs, info)

        for _ in range(200):
            action = agent.act(obs, info)
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                break

        assert term or trunc
        assert info["tiles_acquired"] > 0
        env.close()

    def test_no_uncertainty_completes_episode(self, world_path):
        env = STEMEnv(world_path, total_budget=5000.0, max_steps=100)
        agent = STEMAgentNoUncertainty(seed=42)

        obs, info = env.reset(seed=42)
        agent.reset(obs, info)

        for _ in range(200):
            action = agent.act(obs, info)
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                break

        assert term or trunc
        assert info["tiles_acquired"] > 0
        env.close()

    def test_rule_planner_is_standard(self, world_path):
        """STEMAgentRulePlanner should behave identically to STEMAgent."""
        env = STEMEnv(world_path, total_budget=3000.0, max_steps=50)
        agent = STEMAgentRulePlanner(seed=42)

        obs, info = env.reset(seed=42)
        agent.reset(obs, info)

        for _ in range(100):
            action = agent.act(obs, info)
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                break

        assert term or trunc
        findings = agent.get_findings()
        assert isinstance(findings, dict)
        assert "defect_census" in findings
        env.close()

    def test_llm_agent_fallback_completes(self, world_path):
        """STEMAgentLLM without API key should complete via rule-based fallback."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}):
            env = STEMEnv(world_path, total_budget=5000.0, max_steps=100)
            agent = STEMAgentLLM(seed=42)

            obs, info = env.reset(seed=42)
            agent.reset(obs, info)

            for _ in range(200):
                action = agent.act(obs, info)
                obs, _, term, trunc, info = env.step(action)
                if term or trunc:
                    break

            assert term or trunc
            assert info["tiles_acquired"] > 0
            env.close()


# ---- Ablation benchmark comparison ----

class TestAblationBenchmark:
    """Compare ablation variants to verify they all produce valid results."""

    def test_all_ablations_produce_scores(self, world_path):
        budget = 5000.0
        max_steps = 100
        n_seeds = 2

        variants = {
            "full": lambda: STEMAgent(seed=42),
            "no_planner": lambda: STEMAgentNoPlanner(seed=42),
            "no_uncertainty": lambda: STEMAgentNoUncertainty(seed=42),
        }

        results = {}
        for name, make_agent in variants.items():
            env = STEMEnv(world_path, task=DefectCensusTask(),
                          total_budget=budget, max_steps=max_steps)
            agent = make_agent()
            bench = run_benchmark(
                env, agent, name, n_seeds=n_seeds,
                score_every_n_acquires=3,
            )
            results[name] = bench
            env.close()

        print(f"\n  Ablation DEC-AUC scores:")
        for name, bench in results.items():
            print(f"    {name:20s}: {bench.mean_dec_auc:.4f} ± {bench.std_dec_auc:.4f}"
                  f"  (tiles={bench.mean_tiles:.1f})")

        # All variants should produce non-negative DEC-AUC
        for name, bench in results.items():
            assert bench.mean_dec_auc >= 0.0, f"{name} should have valid DEC-AUC"
            assert bench.mean_tiles > 0, f"{name} should acquire tiles"


# ---- LLM Planner with mock LLM ----

class TestLLMPlannerMocked:
    """Test LLMPlanner with a mocked LLM agent."""

    def test_invoke_calls_agent(self):
        """When API key is set, LLM should be invoked at the right interval."""
        planner = LLMPlanner(api_key="fake-key", invoke_every=5)
        state = _make_state()
        planner.reset(state)

        # Mock the agent
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": []}
        planner._agent = mock_agent

        # Run 6 steps — should invoke at step 5
        for _ in range(6):
            planner.update_mode(state)

        assert mock_agent.invoke.call_count == 1

    def test_invoke_handles_error(self):
        """LLM invocation errors should be caught gracefully."""
        planner = LLMPlanner(api_key="fake-key", invoke_every=5)
        state = _make_state()
        planner.reset(state)

        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = Exception("API error")
        planner._agent = mock_agent

        # Should not raise — logs error and continues
        for _ in range(6):
            planner.update_mode(state)

        assert len(planner.invocation_log) == 1
        assert "error" in planner.invocation_log[0]

    def test_invoke_frequency(self):
        """LLM should be invoked exactly at the right intervals."""
        planner = LLMPlanner(api_key="fake-key", invoke_every=3)
        state = _make_state()
        planner.reset(state)

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": []}
        planner._agent = mock_agent

        for _ in range(12):
            planner.update_mode(state)

        # Steps: 3, 6, 9, 12 → 4 invocations
        assert mock_agent.invoke.call_count == 4

    def test_no_invoke_after_terminate(self):
        """LLM should not be invoked if already in TERMINATE mode."""
        planner = LLMPlanner(api_key="fake-key", invoke_every=3)
        state = _make_state()
        planner.reset(state)
        state.mode = PlannerMode.TERMINATE

        mock_agent = MagicMock()
        planner._agent = mock_agent

        for _ in range(10):
            planner.update_mode(state)

        assert mock_agent.invoke.call_count == 0
