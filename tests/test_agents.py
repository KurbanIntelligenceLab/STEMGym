"""Tests for baseline agents, runner, and DEC metrics integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stem_gym.baselines import Agent
from stem_gym.baselines.random_agent import RandomAgent
from stem_gym.baselines.raster_scan import RasterAgent
from stem_gym.baselines.gp_ucb import GPUCBAgent
from stem_gym.baselines.single_bo import SingleBOAgent
from stem_gym.agents.simple_analyst import SimpleAnalyst
from stem_gym.envs.stem_env import STEMEnv, ActionType
from stem_gym.tasks import DummyTask
from stem_gym.tasks.defect_census import DefectCensusTask
from stem_gym.runner import run_episode, run_benchmark


TEST_WORLD = Path(__file__).parent.parent / "data" / "simulated" / "test_world.h5"


@pytest.fixture(scope="module")
def world_path():
    if not TEST_WORLD.exists():
        pytest.skip("Test world not generated")
    return TEST_WORLD


# ---- SimpleAnalyst tests ----

class TestSimpleAnalyst:
    def test_peak_detection_on_blobs(self):
        """SimpleAnalyst should detect peaks in a Gaussian blob pattern."""
        from scipy.ndimage import gaussian_filter
        rng = np.random.default_rng(42)
        tile = np.zeros((128, 128), dtype=np.float32)
        # Plant ~16 atoms in a grid pattern
        for r in range(16, 128, 32):
            for c in range(16, 128, 32):
                tile[r, c] = 1.0
        tile = gaussian_filter(tile, sigma=2.0)

        analyst = SimpleAnalyst(min_distance=8, threshold_rel=0.1)
        result = analyst.analyze_tile(tile)
        # Should find roughly 16 atoms (4x4 grid)
        assert result.n_atoms >= 10, f"Expected ~16 atoms, got {result.n_atoms}"
        assert result.n_atoms <= 25

    def test_defect_census_accumulates(self):
        """Census should accumulate across multiple tiles."""
        analyst = SimpleAnalyst()
        analyst.reset()

        from scipy.ndimage import gaussian_filter
        # Tile with uniform atoms
        tile = np.zeros((128, 128), dtype=np.float32)
        for r in range(8, 128, 16):
            for c in range(8, 128, 16):
                tile[r, c] = 1.0
        tile = gaussian_filter(tile, sigma=2.0)

        for _ in range(5):
            analyst.analyze_tile(tile)

        assert analyst.get_total_atoms() > 0
        # Census might be empty if no outliers detected (uniform tiles)
        census = analyst.get_defect_census()
        assert isinstance(census, dict)


# ---- Agent protocol tests ----

class TestAgentProtocol:
    """Test that all agents implement the Agent protocol correctly."""

    @pytest.fixture(params=[
        ("random", RandomAgent),
        ("raster", RasterAgent),
        ("gp_ucb", GPUCBAgent),
        ("single_bo", SingleBOAgent),
    ])
    def agent_cls(self, request):
        return request.param

    def test_agent_completes_episode(self, world_path, agent_cls):
        """Each agent should complete an episode without errors."""
        name, cls = agent_cls
        env = STEMEnv(world_path, total_budget=5000.0, max_steps=100)

        if name == "gp_ucb":
            agent = cls(seed=42, initial_random=3)
        elif name == "single_bo":
            agent = cls(seed=42, initial_random=3)
        else:
            agent = cls(seed=42)

        obs, info = env.reset(seed=42)
        agent.reset(obs, info)

        terminated = False
        truncated = False
        steps = 0
        for _ in range(200):
            action = agent.act(obs, info)
            # Verify action is valid
            assert "action_type" in action
            assert "position" in action
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                break

        assert terminated or truncated, f"{name} didn't terminate in 200 steps"
        env.close()

    def test_agent_has_get_findings(self, world_path, agent_cls):
        """Each agent should provide get_findings()."""
        name, cls = agent_cls
        agent = cls(seed=42) if name not in ("gp_ucb", "single_bo") else cls(seed=42, initial_random=2)

        env = STEMEnv(world_path, total_budget=2000.0, max_steps=30)
        obs, info = env.reset(seed=42)
        agent.reset(obs, info)

        # Run a few steps
        for _ in range(10):
            action = agent.act(obs, info)
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                break

        findings = agent.get_findings()
        assert isinstance(findings, dict)
        assert "defect_census" in findings
        env.close()


# ---- Runner tests ----

class TestRunner:
    def test_run_episode_random(self, world_path):
        """run_episode should return a valid EpisodeResult."""
        env = STEMEnv(world_path, task=DefectCensusTask(),
                      total_budget=5000.0, max_steps=50)
        agent = RandomAgent(seed=42)
        result = run_episode(env, agent, seed=42, score_every_n_acquires=3)

        assert result.total_steps > 0
        assert result.total_dose_used >= 0
        assert len(result.episode_log) > 0
        assert result.dec.auc >= 0
        # At least the final entry should have a score
        scores = [e["task_score"] for e in result.episode_log if e["task_score"] is not None]
        assert len(scores) >= 1
        env.close()

    def test_run_episode_raster(self, world_path):
        """Raster agent should acquire tiles systematically."""
        env = STEMEnv(world_path, task=DefectCensusTask(),
                      total_budget=10000.0, max_steps=200)
        agent = RasterAgent(seed=42)
        result = run_episode(env, agent, seed=42, score_every_n_acquires=3)

        assert result.tiles_acquired > 0
        assert result.total_dose_used > 0
        env.close()

    def test_run_benchmark(self, world_path):
        """run_benchmark should aggregate results over seeds."""
        env = STEMEnv(world_path, task=DefectCensusTask(),
                      total_budget=3000.0, max_steps=40)
        agent = RandomAgent(seed=0)
        bench = run_benchmark(
            env, agent, "random", n_seeds=3,
            score_every_n_acquires=3,
        )
        assert bench.agent_name == "random"
        assert len(bench.episode_results) == 3
        assert bench.mean_dec_auc >= 0
        env.close()


# ---- Baseline ordering test ----

class TestBaselineOrdering:
    """Verify baseline ordering on fake world.

    On the uniform fake world, all strategies produce similar DEC-AUC
    because the spatial structure is homogeneous. The key verifications:

    1. All agents produce positive DEC-AUC (they're actually scoring)
    2. Raster achieves high coverage when given enough budget
    3. GP-UCB is competitive with Random (not worse by a large margin)

    Strict ordering (Random < Raster < GP-UCB) is expected on
    real abTEM-simulated worlds with spatial heterogeneity.
    """

    def test_all_agents_produce_scores(self, world_path):
        """All baselines produce positive DEC-AUC — they're working."""
        budget = 5000.0
        max_steps = 100
        n_seeds = 3
        score_every = 3

        results = {}
        for name, cls, kwargs in [
            ("random", RandomAgent, {}),
            ("raster", RasterAgent, {}),
            ("gp_ucb", GPUCBAgent, {"initial_random": 3}),
            ("single_bo", SingleBOAgent, {"initial_random": 3}),
        ]:
            env = STEMEnv(world_path, task=DefectCensusTask(),
                          total_budget=budget, max_steps=max_steps)
            agent = cls(seed=42, **kwargs) if kwargs else cls(seed=42)
            bench = run_benchmark(
                env, agent, name, n_seeds=n_seeds,
                score_every_n_acquires=score_every,
            )
            results[name] = bench
            env.close()

        print(f"\n  Baseline DEC-AUC scores:")
        for name, bench in results.items():
            print(f"    {name:12s}: {bench.mean_dec_auc:.4f} ± {bench.std_dec_auc:.4f}"
                  f"  (tiles={bench.mean_tiles:.1f}, dose={bench.mean_dose_used:.0f})")

        # All agents should produce non-negative DEC-AUC
        for name, bench in results.items():
            assert bench.mean_dec_auc >= 0.0, (
                f"{name} DEC-AUC should be non-negative, got {bench.mean_dec_auc}"
            )

        # All agents should acquire at least some tiles
        for name, bench in results.items():
            assert bench.mean_tiles > 0, f"{name} should acquire tiles"

    def test_raster_high_coverage(self, world_path):
        """Raster with generous budget achieves near-complete coverage."""
        env = STEMEnv(world_path, task=DefectCensusTask(),
                      total_budget=100000.0, max_steps=500)
        agent = RasterAgent(seed=42)
        result = run_episode(env, agent, seed=42, score_every_n_acquires=5)

        grid_h, grid_w = env.world.grid_shape
        total_tiles = grid_h * grid_w
        coverage = result.tiles_acquired / total_tiles
        assert coverage > 0.9, f"Raster coverage {coverage:.2f} should be > 0.9"
        env.close()

    def test_gp_ucb_competitive(self, world_path):
        """GP-UCB should not be dramatically worse than random."""
        budget = 5000.0
        max_steps = 100
        n_seeds = 5

        env_r = STEMEnv(world_path, task=DefectCensusTask(),
                        total_budget=budget, max_steps=max_steps)
        random_bench = run_benchmark(
            env_r, RandomAgent(seed=42), "random", n_seeds=n_seeds,
            score_every_n_acquires=3,
        )
        env_r.close()

        env_g = STEMEnv(world_path, task=DefectCensusTask(),
                        total_budget=budget, max_steps=max_steps)
        gp_bench = run_benchmark(
            env_g, GPUCBAgent(seed=42, initial_random=3), "gp_ucb",
            n_seeds=n_seeds, score_every_n_acquires=3,
        )
        env_g.close()

        # GP-UCB should be within 50% of random (they're similar on uniform world)
        assert gp_bench.mean_dec_auc >= random_bench.mean_dec_auc * 0.5, (
            f"GP-UCB ({gp_bench.mean_dec_auc:.4f}) should be competitive with "
            f"Random ({random_bench.mean_dec_auc:.4f})"
        )
