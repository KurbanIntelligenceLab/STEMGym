"""Tests for Phase 5: Navigator, Planner, and STEMAgent orchestrator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stem_gym.agents.navigator import Navigator
from stem_gym.agents.planner import Planner
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


# ---- Navigator tests ----

class TestNavigator:
    def _make_state(self, grid=(8, 8), fov=(100.0, 100.0)):
        return ExperimentState(
            fov_nm=fov,
            total_budget=10000.0,
            visited_mask=np.zeros(grid, dtype=bool),
            dose_map=np.zeros(grid, dtype=np.float32),
        )

    def test_reset(self):
        nav = Navigator(seed=42)
        state = self._make_state()
        nav.reset(state)
        assert nav.n_observations == 0

    def test_select_next_returns_valid_position(self):
        nav = Navigator(seed=42)
        state = self._make_state()
        nav.reset(state)
        x, y = nav.select_next(state)
        assert 0 <= x <= 100.0
        assert 0 <= y <= 100.0

    def test_record_and_select(self):
        """After recording observations, navigator should still produce positions."""
        nav = Navigator(seed=42, initial_random=2)
        state = self._make_state()
        nav.reset(state)

        for i in range(5):
            x, y = nav.select_next(state)
            tile = np.random.rand(128, 128).astype(np.float32)
            nav.record_observation(state, tile, x, y)

        assert nav.n_observations == 5
        # Should still return valid position
        x, y = nav.select_next(state)
        assert 0 <= x <= 100.0
        assert 0 <= y <= 100.0

    def test_set_beta(self):
        nav = Navigator(beta=2.0, seed=42)
        nav.set_beta(5.0)
        assert nav.beta == 5.0


# ---- Planner tests ----

class TestPlanner:
    def _make_state(self, budget=10000.0, dose_used=0.0, grid=(8, 8)):
        s = ExperimentState(
            fov_nm=(100.0, 100.0),
            total_budget=budget,
            dose_used=dose_used,
            visited_mask=np.zeros(grid, dtype=bool),
            dose_map=np.zeros(grid, dtype=np.float32),
        )
        return s

    def test_initial_mode_is_survey(self):
        planner = Planner()
        state = self._make_state()
        planner.reset(state)
        assert state.mode == PlannerMode.SURVEY

    def test_survey_to_investigate(self):
        """After sufficient coverage, planner transitions to INVESTIGATE."""
        planner = Planner(survey_coverage=0.10)
        state = self._make_state(grid=(10, 10))
        planner.reset(state)

        # Visit 15% of tiles (15 out of 100)
        state.visited_mask[:2, :] = True  # 20 tiles = 20%
        mode = planner.update_mode(state)
        assert mode == PlannerMode.INVESTIGATE

    def test_investigate_to_characterize(self):
        """After finding defects, planner transitions to CHARACTERIZE."""
        planner = Planner(survey_coverage=0.05, investigate_defects=2)
        state = self._make_state(grid=(10, 10))
        planner.reset(state)

        # Force into INVESTIGATE
        state.visited_mask[:2, :] = True
        planner.update_mode(state)
        assert state.mode == PlannerMode.INVESTIGATE

        # Add defects
        state.defect_census = {"vacancy": 2, "substitution": 1}
        mode = planner.update_mode(state)
        assert mode == PlannerMode.CHARACTERIZE

    def test_low_budget_triggers_census(self):
        """Low budget should trigger CENSUS mode."""
        planner = Planner(census_budget=0.20)
        state = self._make_state(budget=10000.0, dose_used=8500.0)
        planner.reset(state)
        # budget_remaining = 1500, fraction = 0.15 < 0.20
        mode = planner.update_mode(state)
        assert mode == PlannerMode.CENSUS

    def test_very_low_budget_triggers_terminate(self):
        """Very low budget should trigger TERMINATE."""
        planner = Planner(terminate_budget=0.05)
        state = self._make_state(budget=10000.0, dose_used=9600.0)
        planner.reset(state)
        mode = planner.update_mode(state)
        assert mode == PlannerMode.TERMINATE

    def test_transition_log(self):
        """Transitions should be logged."""
        planner = Planner(survey_coverage=0.10)
        state = self._make_state(grid=(10, 10))
        planner.reset(state)

        state.visited_mask[:2, :] = True
        planner.update_mode(state)
        assert len(state.transition_log) == 1
        assert state.transition_log[0].from_mode == PlannerMode.SURVEY
        assert state.transition_log[0].to_mode == PlannerMode.INVESTIGATE

    def test_should_acquire_survey_always(self):
        """In SURVEY mode, always acquire."""
        planner = Planner()
        state = self._make_state()
        planner.reset(state)
        assert planner.should_acquire(state) is True

    def test_should_acquire_terminate_never(self):
        """In TERMINATE mode, never acquire."""
        planner = Planner()
        state = self._make_state()
        planner.reset(state)
        state.mode = PlannerMode.TERMINATE
        assert planner.should_acquire(state) is False


# ---- STEMAgent integration tests ----

class TestSTEMAgent:
    def test_agent_completes_episode(self, world_path):
        """STEMAgent should complete an episode without errors."""
        env = STEMEnv(world_path, total_budget=5000.0, max_steps=100)
        agent = STEMAgent(seed=42)

        obs, info = env.reset(seed=42)
        agent.reset(obs, info)

        terminated = False
        truncated = False
        for _ in range(200):
            action = agent.act(obs, info)
            assert "action_type" in action
            assert "position" in action
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        assert terminated or truncated, "STEMAgent didn't terminate"
        env.close()

    def test_agent_acquires_tiles(self, world_path):
        """STEMAgent should acquire tiles during an episode."""
        env = STEMEnv(world_path, total_budget=5000.0, max_steps=100)
        agent = STEMAgent(seed=42)

        obs, info = env.reset(seed=42)
        agent.reset(obs, info)

        for _ in range(200):
            action = agent.act(obs, info)
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                break

        assert info["tiles_acquired"] > 0, "STEMAgent should acquire tiles"
        env.close()

    def test_agent_has_get_findings(self, world_path):
        """STEMAgent should provide get_findings()."""
        env = STEMEnv(world_path, total_budget=3000.0, max_steps=50)
        agent = STEMAgent(seed=42)

        obs, info = env.reset(seed=42)
        agent.reset(obs, info)

        for _ in range(20):
            action = agent.act(obs, info)
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                break

        findings = agent.get_findings()
        assert isinstance(findings, dict)
        assert "defect_census" in findings
        env.close()

    def test_run_episode_stem_agent(self, world_path):
        """run_episode should work with STEMAgent."""
        env = STEMEnv(world_path, task=DefectCensusTask(),
                      total_budget=5000.0, max_steps=100)
        agent = STEMAgent(seed=42)
        result = run_episode(env, agent, seed=42, score_every_n_acquires=3)

        assert result.total_steps > 0
        assert result.total_dose_used >= 0
        assert result.tiles_acquired > 0
        env.close()

    def test_planner_transitions_happen(self, world_path):
        """STEMAgent should trigger planner transitions during an episode."""
        env = STEMEnv(world_path, total_budget=10000.0, max_steps=200)
        agent = STEMAgent(seed=42)

        obs, info = env.reset(seed=42)
        agent.reset(obs, info)

        for _ in range(400):
            action = agent.act(obs, info)
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                break

        # Agent should have gone through at least one transition
        # (SURVEY → something)
        assert len(agent._state.transition_log) >= 1, (
            "Expected planner transitions during episode"
        )
        env.close()
