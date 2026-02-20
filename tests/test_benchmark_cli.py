"""Tests for Phase 8: Benchmark Runner CLI."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from experiments.run_benchmark import (
    AGENT_REGISTRY,
    TASK_REGISTRY,
    WORLD_REGISTRY,
    ExperimentConfig,
    _import_class,
    _serialize_result,
    load_config,
    make_agent,
    make_task,
    print_summary_table,
    resolve_world,
    run_full_benchmark,
)


TEST_WORLD = Path(__file__).parent.parent / "data" / "simulated" / "test_world.h5"


@pytest.fixture(scope="module")
def world_path():
    if not TEST_WORLD.exists():
        pytest.skip("Test world not generated")
    return TEST_WORLD


# ---- Registry tests ----

class TestRegistries:
    def test_agent_registry_has_all_agents(self):
        expected = {
            "random", "raster", "gp_ucb", "single_bo",
            "stem_agent", "no_planner", "no_uncertainty",
            "llm_planner", "rule_planner",
            "gp_ucb_equipped", "raster_equipped",
        }
        assert expected == set(AGENT_REGISTRY.keys())

    def test_task_registry_has_all_tasks(self):
        expected = {"defect_census", "phase_mapping", "targeted"}
        assert expected == set(TASK_REGISTRY.keys())

    def test_world_registry_discovers_files(self):
        # Should find at least the test world
        assert len(WORLD_REGISTRY) > 0
        assert "test_world" in WORLD_REGISTRY

    def test_all_agent_classes_importable(self):
        for name, entry in AGENT_REGISTRY.items():
            cls = _import_class(entry["class"])
            assert cls is not None, f"Failed to import {name}"

    def test_all_task_classes_importable(self):
        for name, entry in TASK_REGISTRY.items():
            cls = _import_class(entry["class"])
            assert cls is not None, f"Failed to import {name}"


# ---- Instantiation tests ----

class TestInstantiation:
    def test_make_agent_random(self):
        agent = make_agent("random", seed=42)
        assert agent is not None

    def test_make_agent_stem_agent(self):
        agent = make_agent("stem_agent", seed=42)
        assert agent is not None

    def test_make_agent_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown agent"):
            make_agent("nonexistent_agent")

    def test_make_task_defect_census(self):
        task = make_task("defect_census")
        assert task.name == "defect_census"

    def test_make_task_phase_mapping(self):
        task = make_task("phase_mapping")
        assert task.name == "phase_mapping"

    def test_make_task_targeted(self):
        task = make_task("targeted")
        assert task.name == "targeted"

    def test_make_task_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            make_task("nonexistent_task")

    def test_resolve_world_by_name(self):
        if "test_world" not in WORLD_REGISTRY:
            pytest.skip("test_world not in registry")
        path = resolve_world("test_world")
        assert Path(path).exists()

    def test_resolve_world_by_path(self, world_path):
        path = resolve_world(str(world_path))
        assert path == str(world_path)

    def test_resolve_world_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown world"):
            resolve_world("nonexistent_world_xyz")


# ---- Config loading tests ----

class TestConfigLoading:
    def test_load_smoke_config(self):
        config_path = Path(__file__).parent.parent / "experiments" / "configs" / "quick_smoke.yaml"
        if not config_path.exists():
            pytest.skip("Smoke config not found")
        config = load_config(str(config_path))
        assert config.name == "smoke_test"
        assert "random" in config.agents
        assert "defect_census" in config.tasks
        assert config.seeds == 2

    def test_load_ablation_config(self):
        config_path = Path(__file__).parent.parent / "experiments" / "configs" / "ablation.yaml"
        if not config_path.exists():
            pytest.skip("Ablation config not found")
        config = load_config(str(config_path))
        assert config.name == "ablation_study"
        assert "stem_agent" in config.agents
        assert "no_planner" in config.agents

    def test_load_full_benchmark_config(self):
        config_path = Path(__file__).parent.parent / "experiments" / "configs" / "full_benchmark.yaml"
        if not config_path.exists():
            pytest.skip("Full benchmark config not found")
        config = load_config(str(config_path))
        assert config.seeds == 5
        assert config.budget == 5000.0
        assert len(config.agents) >= 4
        assert len(config.tasks) >= 3


# ---- Serialization tests ----

class TestSerialization:
    def test_serialize_result(self, world_path):
        from stem_gym.envs.stem_env import STEMEnv
        from stem_gym.runner import run_benchmark

        env = STEMEnv(world_path, total_budget=3000.0, max_steps=50)
        agent = make_agent("random", seed=42)
        bench = run_benchmark(env, agent, "random", n_seeds=2,
                              score_every_n_acquires=3)
        env.close()

        result = _serialize_result(bench, "defect_census", "test_world")

        assert result["agent"] == "random"
        assert result["task"] == "defect_census"
        assert result["world"] == "test_world"
        assert result["n_episodes"] == 2
        assert "mean_dec_auc" in result
        assert "std_dec_auc" in result
        assert len(result["episodes"]) == 2

        # Should be JSON-serializable
        json_str = json.dumps(result)
        restored = json.loads(json_str)
        assert restored["agent"] == "random"

    def test_print_summary_table(self, capsys):
        results = [
            {"agent": "random", "task": "defect_census", "world": "test_world",
             "mean_dec_auc": 0.1234, "std_dec_auc": 0.01,
             "mean_final_score": 0.5, "mean_tiles": 10.0, "mean_dose_used": 2000.0},
            {"agent": "stem_agent", "task": "defect_census", "world": "test_world",
             "mean_dec_auc": 0.5678, "std_dec_auc": 0.02,
             "mean_final_score": 0.8, "mean_tiles": 15.0, "mean_dose_used": 3000.0},
        ]
        print_summary_table(results)
        captured = capsys.readouterr()
        assert "random" in captured.out
        assert "stem_agent" in captured.out


# ---- Integration: run benchmark ----

class TestBenchmarkIntegration:
    def test_smoke_benchmark(self, world_path):
        """Run a minimal benchmark to verify end-to-end flow."""
        config = ExperimentConfig(
            name="test_smoke",
            agents=["random", "raster"],
            tasks=["defect_census"],
            worlds=[str(world_path)],
            seeds=2,
            budget=3000.0,
            max_steps=50,
            score_every_n_acquires=3,
        )

        results = run_full_benchmark(config, verbose=False)

        assert len(results) == 2  # 2 agents × 1 task × 1 world
        for r in results:
            assert r["mean_dec_auc"] >= 0
            assert r["mean_tiles"] > 0
            assert r["n_episodes"] == 2

    def test_benchmark_output_json(self, world_path, tmp_path):
        """Verify JSON output is written correctly."""
        output_file = tmp_path / "test_results.json"

        config = ExperimentConfig(
            name="test_output",
            agents=["random"],
            tasks=["defect_census"],
            worlds=[str(world_path)],
            seeds=1,
            budget=2000.0,
            max_steps=30,
            score_every_n_acquires=3,
            output=str(output_file),
        )

        results = run_full_benchmark(config, verbose=False)

        # Manually save like cmd_run does
        with open(output_file, "w") as f:
            json.dump({
                "config": {
                    "name": config.name,
                    "agents": config.agents,
                    "tasks": config.tasks,
                    "worlds": config.worlds,
                    "seeds": config.seeds,
                    "budget": config.budget,
                },
                "results": results,
            }, f, indent=2)

        # Verify saved file
        with open(output_file) as f:
            data = json.load(f)

        assert data["config"]["name"] == "test_output"
        assert len(data["results"]) == 1
        assert data["results"][0]["agent"] == "random"

    def test_multi_task_benchmark(self, world_path):
        """Verify benchmark works with multiple tasks."""
        config = ExperimentConfig(
            name="multi_task",
            agents=["random"],
            tasks=["defect_census", "phase_mapping"],
            worlds=[str(world_path)],
            seeds=1,
            budget=2000.0,
            max_steps=30,
            score_every_n_acquires=3,
        )

        results = run_full_benchmark(config, verbose=False)
        assert len(results) == 2  # 1 agent × 2 tasks
        tasks_run = {r["task"] for r in results}
        assert tasks_run == {"defect_census", "phase_mapping"}

    def test_stem_agent_benchmark(self, world_path):
        """Verify STEMAgent runs through benchmark pipeline."""
        config = ExperimentConfig(
            agents=["stem_agent"],
            tasks=["defect_census"],
            worlds=[str(world_path)],
            seeds=1,
            budget=3000.0,
            max_steps=50,
            score_every_n_acquires=3,
        )

        results = run_full_benchmark(config, verbose=False)
        assert len(results) == 1
        assert results[0]["agent"] == "stem_agent"
        assert results[0]["mean_dec_auc"] >= 0
