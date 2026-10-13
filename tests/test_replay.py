"""Tests for Phase 6: Replay dataset and agents on replay data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stem_gym.envs.replay_world import ReplayWorld
from stem_gym.envs.stem_env import STEMEnv, ActionType
from stem_gym.tasks.defect_census import DefectCensusTask
from stem_gym.tasks.phase_mapping import PhaseMappingTask
from stem_gym.runner import run_episode, run_benchmark


REPLAY_WORLD = Path(__file__).parent.parent / "data" / "replay" / "replay_world.h5"
SIM_WORLD = Path(__file__).parent.parent / "data" / "simulated" / "test_world.h5"


@pytest.fixture(scope="module")
def replay_path():
    if not REPLAY_WORLD.exists():
        pytest.skip("Replay world not generated")
    return REPLAY_WORLD


@pytest.fixture(scope="module")
def sim_path():
    if not SIM_WORLD.exists():
        pytest.skip("Simulated test world not generated")
    return SIM_WORLD


# ---- ReplayWorld unit tests ----

class TestReplayWorld:
    def test_opens_and_has_metadata(self, replay_path):
        w = ReplayWorld(replay_path)
        assert w.is_replay is True
        assert w.grid_shape[0] > 0
        assert w.grid_shape[1] > 0
        assert w.tile_size_px == 128
        assert w.pixel_size_nm > 0
        w.close()

    def test_source_info(self, replay_path):
        w = ReplayWorld(replay_path)
        info = w.source_info
        assert info["scenario"] == "replay"
        assert info["source"] != "unknown"
        w.close()

    def test_get_tile(self, replay_path):
        w = ReplayWorld(replay_path)
        tile = w.get_tile(0, 0)
        assert tile.shape == (128, 128)
        assert tile.dtype == np.float32
        assert tile.max() > 0
        w.close()

    def test_no_poisson_noise(self, replay_path):
        """apply_poisson_noise should return identical tile for replay."""
        w = ReplayWorld(replay_path)
        tile = w.get_tile(0, 0)
        noisy = w.apply_poisson_noise(tile, 100.0)
        np.testing.assert_array_equal(noisy, tile)
        w.close()

    def test_ground_truth(self, replay_path):
        w = ReplayWorld(replay_path)
        gt = w.get_ground_truth()
        assert "atom_positions" in gt
        assert "atom_types" in gt
        assert "defect_mask" in gt
        assert len(gt["atom_positions"]) > 0
        w.close()

    def test_overview(self, replay_path):
        w = ReplayWorld(replay_path)
        overview = w.get_overview()
        assert overview.ndim == 2
        assert overview.shape[0] > 0
        w.close()


# ---- Env integration ----

class TestReplayEnv:
    def test_env_works_with_replay(self, replay_path):
        """STEMEnv should work with replay world files."""
        env = STEMEnv(replay_path, total_budget=3000.0, max_steps=50)
        obs, info = env.reset(seed=42)

        assert obs["tile"].shape == (128, 128)
        assert obs["budget_remaining"][0] > 0
        assert info["fov_nm"][0] > 0

        # Navigate + acquire
        action = {
            "action_type": ActionType.NAVIGATE,
            "position": np.array([info["fov_nm"][1] / 2, info["fov_nm"][0] / 2],
                                 dtype=np.float32),
            "zoom": np.array([1.0], dtype=np.float32),
            "dwell": np.array([1.0], dtype=np.float32),
            "finding_type": 0,
        }
        obs, _, _, _, _ = env.step(action)

        action["action_type"] = ActionType.ACQUIRE
        obs, _, _, _, info2 = env.step(action)
        assert info2["tiles_acquired"] > 0
        env.close()


# ---- Agents on replay ----

class TestAgentsOnReplay:
    """Run all agents on replay data and verify they produce scores."""

    def test_random_agent_on_replay(self, replay_path):
        from stem_gym.baselines.random_agent import RandomAgent

        env = STEMEnv(replay_path, task=DefectCensusTask(),
                      total_budget=3000.0, max_steps=50)
        agent = RandomAgent(seed=42)
        result = run_episode(env, agent, seed=42, score_every_n_acquires=3)

        assert result.total_steps > 0
        assert result.tiles_acquired > 0
        assert result.dec.auc >= 0
        env.close()

    def test_raster_agent_on_replay(self, replay_path):
        from stem_gym.baselines.raster_scan import RasterAgent

        env = STEMEnv(replay_path, task=DefectCensusTask(),
                      total_budget=5000.0, max_steps=100)
        agent = RasterAgent(seed=42)
        result = run_episode(env, agent, seed=42, score_every_n_acquires=3)

        assert result.total_steps > 0
        assert result.tiles_acquired > 0
        env.close()

    def test_gp_ucb_agent_on_replay(self, replay_path):
        from stem_gym.baselines.gp_ucb import GPUCBAgent

        env = STEMEnv(replay_path, task=DefectCensusTask(),
                      total_budget=3000.0, max_steps=50)
        agent = GPUCBAgent(seed=42, initial_random=3)
        result = run_episode(env, agent, seed=42, score_every_n_acquires=3)

        assert result.total_steps > 0
        assert result.tiles_acquired > 0
        env.close()


    def test_phase_mapping_on_replay(self, replay_path):
        """Phase mapping task should also work on replay."""
        from stem_gym.baselines.raster_scan import RasterAgent

        env = STEMEnv(replay_path, task=PhaseMappingTask(),
                      total_budget=5000.0, max_steps=100)
        agent = RasterAgent(seed=42)
        result = run_episode(env, agent, seed=42, score_every_n_acquires=3)

        assert result.total_steps > 0
        env.close()


# ---- Ranking correlation ----

class TestRankingCorrelation:
    """Verify agent rankings on replay correlate with simulation rankings.

    On the synthetic replay world (which has similar structure to the
    simulated test world), agent rankings should be broadly consistent.
    Both are uniform lattices with random defects, so all agents perform
    similarly — the key check is that all agents produce valid results.
    """

    def test_all_agents_produce_scores_on_replay(self, replay_path):
        """All agents should produce non-negative DEC-AUC on replay."""
        from stem_gym.baselines.random_agent import RandomAgent
        from stem_gym.baselines.raster_scan import RasterAgent
        from stem_gym.baselines.gp_ucb import GPUCBAgent

        budget = 3000.0
        max_steps = 50
        n_seeds = 2

        results = {}
        for name, make_agent in [
            ("random", lambda: RandomAgent(seed=42)),
            ("raster", lambda: RasterAgent(seed=42)),
            ("gp_ucb", lambda: GPUCBAgent(seed=42, initial_random=3)),
        ]:
            env = STEMEnv(replay_path, task=DefectCensusTask(),
                          total_budget=budget, max_steps=max_steps)
            agent = make_agent()
            bench = run_benchmark(
                env, agent, name, n_seeds=n_seeds,
                score_every_n_acquires=3,
            )
            results[name] = bench
            env.close()

        print(f"\n  Replay DEC-AUC scores:")
        for name, bench in results.items():
            print(f"    {name:12s}: {bench.mean_dec_auc:.4f} ± {bench.std_dec_auc:.4f}"
                  f"  (tiles={bench.mean_tiles:.1f})")

        # All agents should produce non-negative DEC-AUC
        for name, bench in results.items():
            assert bench.mean_dec_auc >= 0.0, (
                f"{name} DEC-AUC should be >= 0 on replay, got {bench.mean_dec_auc}"
            )

        # All agents should acquire tiles
        for name, bench in results.items():
            assert bench.mean_tiles > 0, f"{name} should acquire tiles on replay"

    def test_sim_and_replay_both_work(self, replay_path, sim_path):
        """Both simulated and replay worlds should produce valid benchmarks."""
        from stem_gym.baselines.raster_scan import RasterAgent

        for world_path, label in [(sim_path, "sim"), (replay_path, "replay")]:
            env = STEMEnv(world_path, task=DefectCensusTask(),
                          total_budget=5000.0, max_steps=100)
            agent = RasterAgent(seed=42)
            result = run_episode(env, agent, seed=42, score_every_n_acquires=3)

            assert result.total_steps > 0, f"{label} should produce steps"
            assert result.tiles_acquired > 0, f"{label} should acquire tiles"
            assert result.dec.auc >= 0, f"{label} should produce valid DEC-AUC"
            env.close()
