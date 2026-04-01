#!/usr/bin/env python3
"""STEMGym Benchmark Runner CLI.

Usage:
    python -m experiments.run_benchmark --config experiments/configs/quick_smoke.yaml
    python -m experiments.run_benchmark --agent stem_agent --task defect_census --world data/simulated/test_world.h5
    stemgym run --config experiments/configs/main_defect_census.yaml --output results/defect.json

Examples:
    # Quick smoke test
    stemgym run --agent random raster --task defect_census --world data/simulated/test_world.h5 --seeds 2 --budget 3000

    # Full benchmark from config
    stemgym run --config experiments/configs/main_defect_census.yaml --output results/defect.json

    # Dry run (print plan without executing)
    stemgym run --config experiments/configs/main_defect_census.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Ensure stem_gym is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stem_gym.baselines import Agent
from stem_gym.envs.stem_env import STEMEnv
from stem_gym.runner import BenchmarkResult, run_benchmark
from stem_gym.tasks import Task


# ---------------------------------------------------------------------------
# Registry: all available agents, tasks, worlds
# ---------------------------------------------------------------------------

AGENT_REGISTRY: dict[str, dict[str, Any]] = {
    "random": {
        "class": "stem_gym.baselines.random_agent.RandomAgent",
        "description": "Random navigation baseline",
    },
    "raster": {
        "class": "stem_gym.baselines.raster_scan.RasterAgent",
        "description": "Systematic grid traversal",
    },
    "gp_ucb": {
        "class": "stem_gym.baselines.gp_ucb.GPUCBAgent",
        "description": "Single-agent GP-UCB",
        "kwargs": {"initial_random": 3},
    },
    "single_bo": {
        "class": "stem_gym.baselines.single_bo.SingleBOAgent",
        "description": "Joint BO over (x,y,zoom,dwell)",
    },
    "stem_agent": {
        "class": "stem_gym.agents.multi_agent.STEMAgent",
        "description": "Full multi-agent system (Navigator + Analyst + Planner)",
    },
    "no_planner": {
        "class": "stem_gym.agents.ablations.STEMAgentNoPlanner",
        "description": "Ablation: no FSM planner (always SURVEY)",
    },
    "no_uncertainty": {
        "class": "stem_gym.agents.ablations.STEMAgentNoUncertainty",
        "description": "Ablation: analyst without ensemble uncertainty",
    },
    "llm_planner": {
        "class": "stem_gym.agents.ablations.STEMAgentLLM",
        "description": "LLM planner via OpenRouter (falls back to rule-based)",
    },
    "rule_planner": {
        "class": "stem_gym.agents.ablations.STEMAgentRulePlanner",
        "description": "Explicit rule-based planner baseline",
    },
    "gp_ucb_equipped": {
        "class": "stem_gym.baselines.equipped_baselines.GPUCBEquippedAgent",
        "description": "GP-UCB + trained Analyst (fair comparison)",
        "kwargs": {"initial_random": 3},
    },
    "raster_equipped": {
        "class": "stem_gym.baselines.equipped_baselines.RasterEquippedAgent",
        "description": "Raster + trained Analyst (fair comparison)",
    },
    "random_equipped": {
        "class": "stem_gym.baselines.equipped_baselines.RandomEquippedAgent",
        "description": "Random navigation + trained Analyst (fair comparison)",
    },
    "single_bo_equipped": {
        "class": "stem_gym.baselines.equipped_baselines.SingleBOEquippedAgent",
        "description": "Joint BO over (x,y,zoom,dwell) + trained Analyst",
    },
    "raster_vlm_claude": {
        "class": "stem_gym.baselines.vlm_baselines.RasterVLMAgent",
        "description": "Raster + Claude Haiku 4.5 VLM analyst",
        "kwargs": {"provider": "claude"},
    },
    "raster_vlm_gpt5": {
        "class": "stem_gym.baselines.vlm_baselines.RasterVLMAgent",
        "description": "Raster + GPT-5-mini VLM analyst",
        "kwargs": {"provider": "gpt5"},
    },
    "raster_vlm_llama": {
        "class": "stem_gym.baselines.vlm_baselines.RasterVLMAgent",
        "description": "Raster + Llama 4 Scout VLM analyst",
        "kwargs": {"provider": "llama"},
    },
    "gp_ucb_vlm_claude": {
        "class": "stem_gym.baselines.vlm_baselines.GPUCBVLMAgent",
        "description": "GP-UCB + Claude Haiku 4.5 VLM analyst",
        "kwargs": {"provider": "claude"},
    },
    "gp_ucb_vlm_gpt5": {
        "class": "stem_gym.baselines.vlm_baselines.GPUCBVLMAgent",
        "description": "GP-UCB + GPT-5-mini VLM analyst",
        "kwargs": {"provider": "gpt5"},
    },
    "gp_ucb_vlm_llama": {
        "class": "stem_gym.baselines.vlm_baselines.GPUCBVLMAgent",
        "description": "GP-UCB + Llama 4 Scout VLM analyst",
        "kwargs": {"provider": "llama"},
    },
    "raster_vlm_gemini": {
        "class": "stem_gym.baselines.vlm_baselines.RasterVLMAgent",
        "description": "Raster + Gemini 2.0 Flash VLM analyst",
        "kwargs": {"provider": "gemini_flash"},
    },
    "gp_ucb_vlm_gemini": {
        "class": "stem_gym.baselines.vlm_baselines.GPUCBVLMAgent",
        "description": "GP-UCB + Gemini 2.0 Flash VLM analyst",
        "kwargs": {"provider": "gemini_flash"},
    },
    # -- Zoom/dwell ablation variants --
    "raster_equipped_z2": {
        "class": "stem_gym.baselines.equipped_baselines.RasterEquippedAgent",
        "description": "Raster + Analyst, zoom=2",
        "kwargs": {"zoom": 2.0},
    },
    "raster_equipped_z4": {
        "class": "stem_gym.baselines.equipped_baselines.RasterEquippedAgent",
        "description": "Raster + Analyst, zoom=4",
        "kwargs": {"zoom": 4.0},
    },
    "raster_equipped_d05": {
        "class": "stem_gym.baselines.equipped_baselines.RasterEquippedAgent",
        "description": "Raster + Analyst, dwell=0.5",
        "kwargs": {"dwell": 0.5},
    },
    "raster_equipped_d2": {
        "class": "stem_gym.baselines.equipped_baselines.RasterEquippedAgent",
        "description": "Raster + Analyst, dwell=2.0",
        "kwargs": {"dwell": 2.0},
    },
    "raster_equipped_z2_d05": {
        "class": "stem_gym.baselines.equipped_baselines.RasterEquippedAgent",
        "description": "Raster + Analyst, zoom=2 dwell=0.5",
        "kwargs": {"zoom": 2.0, "dwell": 0.5},
    },
    "raster_equipped_z2_d2": {
        "class": "stem_gym.baselines.equipped_baselines.RasterEquippedAgent",
        "description": "Raster + Analyst, zoom=2 dwell=2.0",
        "kwargs": {"zoom": 2.0, "dwell": 2.0},
    },
    "raster_equipped_z4_d05": {
        "class": "stem_gym.baselines.equipped_baselines.RasterEquippedAgent",
        "description": "Raster + Analyst, zoom=4 dwell=0.5",
        "kwargs": {"zoom": 4.0, "dwell": 0.5},
    },
    "raster_equipped_z4_d2": {
        "class": "stem_gym.baselines.equipped_baselines.RasterEquippedAgent",
        "description": "Raster + Analyst, zoom=4 dwell=2.0",
        "kwargs": {"zoom": 4.0, "dwell": 2.0},
    },
    # -- Acquisition function variants --
    "gp_ucb_ei": {
        "class": "stem_gym.baselines.gp_ucb.GPUCBAgent",
        "description": "GP-UCB with Expected Improvement",
        "kwargs": {"initial_random": 3, "acq_func": "ei"},
    },
    "gp_ucb_ts": {
        "class": "stem_gym.baselines.gp_ucb.GPUCBAgent",
        "description": "GP-UCB with Thompson Sampling",
        "kwargs": {"initial_random": 3, "acq_func": "ts"},
    },
    "gp_ucb_ei_equipped": {
        "class": "stem_gym.baselines.equipped_baselines.GPUCBEquippedAgent",
        "description": "GP-EI + trained Analyst",
        "kwargs": {"initial_random": 3, "acq_func": "ei"},
    },
    "gp_ucb_ts_equipped": {
        "class": "stem_gym.baselines.equipped_baselines.GPUCBEquippedAgent",
        "description": "GP-TS + trained Analyst",
        "kwargs": {"initial_random": 3, "acq_func": "ts"},
    },
    "stem_agent_ei": {
        "class": "stem_gym.agents.multi_agent.STEMAgent",
        "description": "Full STEMAgent with EI acquisition",
        "kwargs": {"acq_func": "ei"},
    },
    "stem_agent_ts": {
        "class": "stem_gym.agents.multi_agent.STEMAgent",
        "description": "Full STEMAgent with Thompson Sampling",
        "kwargs": {"acq_func": "ts"},
    },
    # -- Adaptive (online fine-tuning) variants --
    "adaptive_raster": {
        "class": "stem_gym.agents.adaptive_agent.AdaptiveRasterAgent",
        "description": "Raster + AdaptiveAnalyst (online fine-tuning)",
    },
    "adaptive_stem": {
        "class": "stem_gym.agents.adaptive_agent.AdaptiveSTEMAgent",
        "description": "STEMAgent + AdaptiveAnalyst (online fine-tuning)",
    },
    # -- DECLARE variants --
    "stem_agent_declare": {
        "class": "stem_gym.agents.multi_agent.STEMAgent",
        "description": "STEMAgent with DECLARE annotations enabled",
        "kwargs": {"use_declare": True},
    },
    "stem_agent_no_declare": {
        "class": "stem_gym.agents.multi_agent.STEMAgent",
        "description": "STEMAgent without DECLARE (explicit baseline)",
        "kwargs": {"use_declare": False},
    },
    # -- DQN RL baselines --
    "dqn": {
        "class": "stem_gym.baselines.rl_agent.DQNAgent",
        "description": "DQN RL navigation baseline (SB3)",
    },
    "dqn_equipped": {
        "class": "stem_gym.baselines.rl_agent.DQNEquippedAgent",
        "description": "DQN navigation + trained Analyst",
    },
    # -- PPO RL baselines --
    "ppo": {
        "class": "stem_gym.baselines.rl_agent.PPOAgent",
        "description": "PPO RL navigation baseline (SB3, discrete)",
    },
    "ppo_equipped": {
        "class": "stem_gym.baselines.rl_agent.PPOEquippedAgent",
        "description": "PPO navigation + trained Analyst",
    },
    # -- SAC RL baselines --
    "sac": {
        "class": "stem_gym.baselines.rl_agent.SACAgent",
        "description": "SAC RL navigation baseline (SB3, continuous)",
    },
    "sac_equipped": {
        "class": "stem_gym.baselines.rl_agent.SACEquippedAgent",
        "description": "SAC navigation + trained Analyst",
    },
}

TASK_REGISTRY: dict[str, dict[str, Any]] = {
    "defect_census": {
        "class": "stem_gym.tasks.defect_census.DefectCensusTask",
        "description": "F1 score for defect detection",
    },
    "phase_mapping": {
        "class": "stem_gym.tasks.phase_mapping.PhaseMappingTask",
        "description": "IoU score for phase identification",
    },
    "targeted": {
        "class": "stem_gym.tasks.targeted.TargetedTask",
        "description": "F1 for locating target features",
        "kwargs": {"target_type": "substitution", "tolerance_nm": 5.0},
    },
    "particle_census": {
        "class": "stem_gym.tasks.particle_census.ParticleCensusTask",
        "description": "Detection + sizing + morphology for nanoparticles",
    },
    "composite": {
        "class": "stem_gym.tasks.composite_task.CompositeTask",
        "description": "Multi-objective: defect census + phase mapping (weighted)",
    },
}

# Auto-discover world files
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

WORLD_REGISTRY: dict[str, str] = {}


def _discover_worlds():
    """Scan data/ for HDF5 world files."""
    for subdir in ["simulated", "replay"]:
        d = DATA_DIR / subdir
        if d.exists():
            for f in sorted(d.glob("*.h5")):
                key = f.stem
                WORLD_REGISTRY[key] = str(f)


_discover_worlds()


# ---------------------------------------------------------------------------
# Instantiation helpers
# ---------------------------------------------------------------------------

def _import_class(dotted_path: str):
    """Import a class from a dotted module path."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def make_agent(name: str, seed: int = 42, **overrides) -> Agent:
    """Instantiate an agent by registry name."""
    if name not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent '{name}'. Available: {list(AGENT_REGISTRY.keys())}"
        )
    entry = AGENT_REGISTRY[name]
    cls = _import_class(entry["class"])
    kwargs = {**entry.get("kwargs", {}), "seed": seed, **overrides}
    return cls(**kwargs)


def make_task(name: str, **overrides) -> Task:
    """Instantiate a task by registry name."""
    if name not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task '{name}'. Available: {list(TASK_REGISTRY.keys())}"
        )
    entry = TASK_REGISTRY[name]
    cls = _import_class(entry["class"])
    kwargs = {**entry.get("kwargs", {}), **overrides}
    return cls(**kwargs)


def resolve_world(name_or_path: str) -> str:
    """Resolve a world name (from registry) or file path."""
    if name_or_path in WORLD_REGISTRY:
        return WORLD_REGISTRY[name_or_path]
    p = Path(name_or_path)
    if p.exists():
        return str(p)
    raise ValueError(
        f"Unknown world '{name_or_path}'. Available: {list(WORLD_REGISTRY.keys())} "
        f"or provide a path to an HDF5 file."
    )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Configuration for a benchmark experiment."""
    name: str = "benchmark"
    agents: list[str] | None = None
    tasks: list[str] | None = None
    worlds: list[str] | None = None
    seeds: int = 5
    budget: float = 5000.0
    max_steps: int = 200
    score_every_n_acquires: int = 5
    output: str | None = None
    workers: int = 1
    save_trajectories: bool = False


def load_config(path: str) -> ExperimentConfig:
    """Load experiment config from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)

    return ExperimentConfig(
        name=data.get("name", "benchmark"),
        agents=data.get("agents"),
        tasks=data.get("tasks"),
        worlds=data.get("worlds"),
        seeds=data.get("seeds", 5),
        budget=data.get("budget", 5000.0),
        max_steps=data.get("max_steps", 200),
        score_every_n_acquires=data.get("score_every_n_acquires", 5),
        output=data.get("output"),
        workers=data.get("workers", 1),
        save_trajectories=data.get("save_trajectories", False),
    )


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------

def _serialize_result(
    bench: BenchmarkResult, task_name: str, world_name: str,
    save_trajectories: bool = False,
) -> dict:
    """Convert BenchmarkResult to JSON-serializable dict."""
    episodes = []
    for i, r in enumerate(bench.episode_results):
        ep = {
            "seed": i,
            "dec_auc": round(r.dec.auc, 6),
            "final_score": round(r.final_score, 6),
            "tiles_acquired": r.tiles_acquired,
            "total_dose_used": round(r.total_dose_used, 1),
            "total_steps": r.total_steps,
            "elapsed_seconds": round(r.elapsed_seconds, 3),
            "dec_dose_fractions": r.dec.dose_fractions.tolist(),
            "dec_scores": r.dec.scores.tolist(),
        }
        if save_trajectories:
            ep["trajectory"] = [
                {
                    "step": e["step"],
                    "action_type": e.get("action_type", -1),
                    "dose_used": e["dose_used"],
                    "tiles_acquired": e["tiles_acquired"],
                    "position": list(e["position"]) if e.get("position") else None,
                }
                for e in r.episode_log
            ]
        episodes.append(ep)

    return {
        "agent": bench.agent_name,
        "task": task_name,
        "world": world_name,
        "n_episodes": len(bench.episode_results),
        "mean_dec_auc": round(bench.mean_dec_auc, 6),
        "std_dec_auc": round(bench.std_dec_auc, 6),
        "mean_final_score": round(bench.mean_final_score, 6),
        "mean_dose_used": round(bench.mean_dose_used, 1),
        "mean_tiles": round(bench.mean_tiles, 1),
        "episodes": episodes,
    }


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def _load_existing_results(output_path: str | None) -> list[dict]:
    """Load previously saved results from an output file for resume support."""
    if not output_path:
        return []
    p = Path(output_path)
    if not p.exists():
        return []
    try:
        with open(p) as f:
            data = json.load(f)
        results = data.get("results", [])
        if results:
            print(f"  Resuming: loaded {len(results)} existing results from {output_path}")
        return results
    except (json.JSONDecodeError, KeyError):
        return []


def _save_results_incremental(
    output_path: str | None,
    config: ExperimentConfig,
    results: list[dict],
) -> None:
    """Save results incrementally to the output file."""
    if not output_path:
        return
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "config": {
                "name": config.name,
                "agents": config.agents,
                "tasks": config.tasks,
                "worlds": config.worlds,
                "seeds": config.seeds,
                "budget": config.budget,
                "max_steps": config.max_steps,
                "workers": config.workers,
            },
            "results": results,
        }, f, indent=2)


def run_full_benchmark(
    config: ExperimentConfig,
    verbose: bool = True,
    shard: tuple[int, int] | None = None,
) -> list[dict]:
    """Run the full benchmark matrix specified by config.

    Args:
        config: Experiment configuration.
        verbose: Print progress.
        shard: Optional (shard_index, total_shards) for distributed runs.
            shard_index is 1-based. Only runs the slice of combos for this shard.

    Returns list of serialized result dicts.
    """
    agents = config.agents or list(AGENT_REGISTRY.keys())
    tasks = config.tasks or list(TASK_REGISTRY.keys())
    worlds = config.worlds or list(WORLD_REGISTRY.keys())

    # Build full job list: (world_name, task_name, agent_name)
    combos = [
        (world_name, task_name, agent_name)
        for world_name in worlds
        for task_name in tasks
        for agent_name in agents
    ]

    # Apply sharding
    if shard is not None:
        shard_idx, total_shards = shard
        combos = [c for i, c in enumerate(combos) if i % total_shards == shard_idx - 1]

    total_runs = len(combos)
    total_episodes = total_runs * config.seeds
    workers = config.workers

    # Resume support: load existing results and skip completed combos
    existing_results = _load_existing_results(config.output)
    completed_keys = {
        (r["agent"], r["task"], r["world"]) for r in existing_results
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"  STEMGym Benchmark: {config.name}")
        print(f"{'='*70}")
        print(f"  Agents: {agents}")
        print(f"  Tasks:  {tasks}")
        print(f"  Worlds: {worlds}")
        print(f"  Seeds:  {config.seeds}")
        print(f"  Budget: {config.budget}")
        print(f"  Workers: {workers}")
        if shard:
            print(f"  Shard:  {shard[0]}/{shard[1]}")
        print(f"  Total combinations: {total_runs} ({total_episodes} episodes)")
        if completed_keys:
            print(f"  Already completed: {len(completed_keys)} (resuming)")
        print(f"{'='*70}\n")

    # --- Parallel execution ---
    if workers > 1:
        return _run_full_benchmark_parallel(
            config, combos, verbose=verbose,
            existing_results=existing_results,
            completed_keys=completed_keys,
        )

    # --- Sequential execution (original path) ---
    all_results = list(existing_results)
    for run_idx, (world_name, task_name, agent_name) in enumerate(combos, 1):
        # Skip already completed combos
        if (agent_name, task_name, world_name) in completed_keys:
            if verbose:
                print(f"  [{run_idx}/{total_runs}] {agent_name} / {task_name} / {world_name} — skipped (done)")
            continue

        label = f"[{run_idx}/{total_runs}] {agent_name} / {task_name} / {world_name}"

        if verbose:
            print(f"\n  {label}")
            print(f"  {'-'*len(label)}")

        t0 = time.time()
        world_path = resolve_world(world_name)
        task = make_task(task_name)
        env = STEMEnv(
            world_path,
            task=task,
            total_budget=config.budget,
            max_steps=config.max_steps,
        )

        agent = make_agent(agent_name)
        bench = run_benchmark(
            env, agent, agent_name,
            n_seeds=config.seeds,
            score_every_n_acquires=config.score_every_n_acquires,
            verbose=verbose,
        )
        env.close()
        elapsed = time.time() - t0

        result = _serialize_result(bench, task_name, world_name, save_trajectories=config.save_trajectories)
        result["elapsed_seconds"] = round(elapsed, 2)
        all_results.append(result)

        # Save incrementally after each combo
        _save_results_incremental(config.output, config, all_results)

        if verbose:
            print(
                f"  => DEC-AUC: {bench.mean_dec_auc:.4f} +/- {bench.std_dec_auc:.4f}"
                f"  | tiles: {bench.mean_tiles:.1f}"
                f"  | time: {elapsed:.1f}s"
            )

    return all_results


def _run_full_benchmark_parallel(
    config: ExperimentConfig,
    combos: list[tuple[str, str, str]],
    verbose: bool = True,
    existing_results: list[dict] | None = None,
    completed_keys: set[tuple[str, str, str]] | None = None,
) -> list[dict]:
    """Run benchmark with parallel episode execution via ProcessPoolExecutor."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from stem_gym.runner import _run_episode_isolated

    if existing_results is None:
        existing_results = []
    if completed_keys is None:
        completed_keys = set()

    # Use 'spawn' for CUDA safety
    ctx = mp.get_context("spawn")

    # Filter out already completed combos
    remaining_combos = [
        c for c in combos
        if (c[2], c[1], c[0]) not in completed_keys  # (agent, task, world)
    ]

    if verbose and len(remaining_combos) < len(combos):
        skipped = len(combos) - len(remaining_combos)
        print(f"  Skipping {skipped} already-completed combinations\n")

    # Flatten all jobs: (agent, task, world, seed)
    jobs = []
    for world_name, task_name, agent_name in remaining_combos:
        world_path = resolve_world(world_name)
        for seed in range(config.seeds):
            jobs.append({
                "agent_name": agent_name,
                "task_name": task_name,
                "world_path": world_path,
                "world_name": world_name,
                "seed": seed,
                "budget": config.budget,
                "max_steps": config.max_steps,
                "score_every_n_acquires": config.score_every_n_acquires,
                "save_trajectories": config.save_trajectories,
            })

    total_jobs = len(jobs)
    if verbose:
        print(f"  Dispatching {total_jobs} episodes across {config.workers} workers...\n")

    t0 = time.time()

    # Collect results keyed by (agent, task, world)
    results_by_combo: dict[tuple[str, str, str], list[dict]] = {}
    completed_count = 0
    # Track which combos are fully done (all seeds collected) for incremental saves
    combo_seeds_expected = {
        (c[2], c[1], c[0]): config.seeds for c in remaining_combos
    }
    saved_combos: set[tuple[str, str, str]] = set()
    all_results = list(existing_results)

    with ProcessPoolExecutor(
        max_workers=config.workers,
        mp_context=ctx,
    ) as executor:
        # Submit all jobs
        future_to_job = {}
        for job in jobs:
            future = executor.submit(_run_episode_isolated, job)
            future_to_job[future] = job

        # Collect as completed
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            completed_count += 1
            key = (job["agent_name"], job["task_name"], job["world_name"])

            try:
                episode_result = future.result()
                results_by_combo.setdefault(key, []).append(episode_result)
                if verbose:
                    print(
                        f"  [{completed_count}/{total_jobs}] "
                        f"{key[0]}/{key[1]}/{key[2]} seed={job['seed']} "
                        f"DEC-AUC={episode_result['dec_auc']:.4f}"
                    )

                # Incremental save: when all seeds for a combo are done
                if (
                    key not in saved_combos
                    and len(results_by_combo.get(key, [])) >= combo_seeds_expected.get(key, config.seeds)
                ):
                    saved_combos.add(key)
                    episodes = sorted(results_by_combo[key], key=lambda e: e["seed"])
                    aucs = [e["dec_auc"] for e in episodes]
                    combo_result = {
                        "agent": key[0],
                        "task": key[1],
                        "world": key[2],
                        "n_episodes": len(episodes),
                        "mean_dec_auc": round(float(np.mean(aucs)), 6),
                        "std_dec_auc": round(float(np.std(aucs)), 6),
                        "mean_final_score": round(float(np.mean([e["final_score"] for e in episodes])), 6),
                        "mean_dose_used": round(float(np.mean([e["total_dose_used"] for e in episodes])), 1),
                        "mean_tiles": round(float(np.mean([e["tiles_acquired"] for e in episodes])), 1),
                        "episodes": [
                            {
                                "seed": e["seed"],
                                "dec_auc": round(e["dec_auc"], 6),
                                "final_score": round(e["final_score"], 6),
                                "tiles_acquired": e["tiles_acquired"],
                                "total_dose_used": round(e["total_dose_used"], 1),
                                "total_steps": e["total_steps"],
                                "dec_dose_fractions": e["dec_dose_fractions"],
                                "dec_scores": e["dec_scores"],
                                **({"trajectory": e["trajectory"]} if "trajectory" in e else {}),
                            }
                            for e in episodes
                        ],
                        "elapsed_seconds": round(time.time() - t0, 2),
                    }
                    all_results.append(combo_result)
                    _save_results_incremental(config.output, config, all_results)
                    if verbose:
                        print(f"    -> saved {key[0]}/{key[1]}/{key[2]} ({len(all_results)} total combos)")

            except Exception as e:
                if verbose:
                    print(
                        f"  [{completed_count}/{total_jobs}] "
                        f"{key[0]}/{key[1]}/{key[2]} seed={job['seed']} "
                        f"FAILED: {e}"
                    )

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  Total parallel time: {elapsed:.1f}s")

    # Add any combos that weren't incrementally saved (shouldn't happen, but safety)
    for world_name, task_name, agent_name in remaining_combos:
        key = (agent_name, task_name, world_name)
        if key in saved_combos:
            continue
        episodes = results_by_combo.get(key, [])
        if not episodes:
            continue

        episodes.sort(key=lambda e: e["seed"])
        aucs = [e["dec_auc"] for e in episodes]
        result = {
            "agent": agent_name,
            "task": task_name,
            "world": world_name,
            "n_episodes": len(episodes),
            "mean_dec_auc": round(float(np.mean(aucs)), 6),
            "std_dec_auc": round(float(np.std(aucs)), 6),
            "mean_final_score": round(float(np.mean([e["final_score"] for e in episodes])), 6),
            "mean_dose_used": round(float(np.mean([e["total_dose_used"] for e in episodes])), 1),
            "mean_tiles": round(float(np.mean([e["tiles_acquired"] for e in episodes])), 1),
            "episodes": [
                {
                    "seed": e["seed"],
                    "dec_auc": round(e["dec_auc"], 6),
                    "final_score": round(e["final_score"], 6),
                    "tiles_acquired": e["tiles_acquired"],
                    "total_dose_used": round(e["total_dose_used"], 1),
                    "total_steps": e["total_steps"],
                    "dec_dose_fractions": e["dec_dose_fractions"],
                    "dec_scores": e["dec_scores"],
                    **({"trajectory": e["trajectory"]} if "trajectory" in e else {}),
                }
                for e in episodes
            ],
            "elapsed_seconds": round(elapsed, 2),
        }
        all_results.append(result)

    return all_results


def print_summary_table(results: list[dict]) -> None:
    """Print a formatted summary table of results."""
    if not results:
        print("  No results to display.")
        return

    # Group by task/world
    groups: dict[str, list[dict]] = {}
    for r in results:
        key = f"{r['task']} / {r['world']}"
        groups.setdefault(key, []).append(r)

    for group_key, group_results in groups.items():
        print(f"\n  Task/World: {group_key}")
        print(f"  {'Agent':<20s} {'DEC-AUC':>12s} {'Final':>10s} {'Tiles':>8s} {'Dose':>10s}")
        print(f"  {'-'*60}")

        # Sort by DEC-AUC descending
        sorted_results = sorted(group_results, key=lambda x: x["mean_dec_auc"], reverse=True)
        for r in sorted_results:
            print(
                f"  {r['agent']:<20s} "
                f"{r['mean_dec_auc']:>8.4f} +/- {r['std_dec_auc']:<5.4f}"
                f"{r['mean_final_score']:>10.4f}"
                f"{r['mean_tiles']:>8.1f}"
                f"{r['mean_dose_used']:>10.1f}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stemgym",
        description="STEMGym Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # -- run command --
    run_parser = sub.add_parser("run", help="Run benchmark experiments")
    run_parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="Path to YAML config file",
    )
    run_parser.add_argument(
        "--agent", "-a", nargs="+", default=None,
        help=f"Agent(s) to evaluate. Available: {list(AGENT_REGISTRY.keys())}",
    )
    run_parser.add_argument(
        "--task", "-t", nargs="+", default=None,
        help=f"Task(s). Available: {list(TASK_REGISTRY.keys())}",
    )
    run_parser.add_argument(
        "--world", "-w", nargs="+", default=None,
        help=f"World(s). Names: {list(WORLD_REGISTRY.keys())} or paths to .h5 files",
    )
    run_parser.add_argument("--seeds", "-s", type=int, default=5, help="Number of seeds")
    run_parser.add_argument("--budget", "-b", type=float, default=5000.0, help="Dose budget")
    run_parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    run_parser.add_argument("--score-every", type=int, default=5, help="Score every N acquires")
    run_parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON file path")
    run_parser.add_argument("--quiet", "-q", action="store_true", help="Suppress verbose output")
    run_parser.add_argument("--dry-run", action="store_true", help="Print plan without executing")
    run_parser.add_argument(
        "--workers", "-j", type=int, default=1,
        help="Number of parallel workers (1=sequential, >1=parallel episodes)",
    )
    run_parser.add_argument(
        "--shard", type=str, default=None,
        help="Shard spec 'K/N' — run shard K of N total (1-based). For distributed runs.",
    )
    run_parser.add_argument(
        "--save-trajectories", action="store_true", default=False,
        help="Save per-step trajectory (position, action_type, dose) in results JSON",
    )

    # -- merge command --
    merge_parser = sub.add_parser("merge", help="Merge shard result JSON files")
    merge_parser.add_argument("inputs", nargs="+", help="Input JSON files to merge")
    merge_parser.add_argument("--output", "-o", required=True, help="Output merged JSON file")

    # -- list command --
    list_parser = sub.add_parser("list", help="List available agents, tasks, worlds")
    list_parser.add_argument(
        "what", nargs="?", default="all",
        choices=["all", "agents", "tasks", "worlds"],
    )

    return parser


def cmd_list(what: str = "all"):
    """List available components."""
    if what in ("all", "agents"):
        print("\n  Available Agents:")
        for name, info in AGENT_REGISTRY.items():
            print(f"    {name:<20s} {info['description']}")

    if what in ("all", "tasks"):
        print("\n  Available Tasks:")
        for name, info in TASK_REGISTRY.items():
            print(f"    {name:<20s} {info['description']}")

    if what in ("all", "worlds"):
        print("\n  Available Worlds:")
        if WORLD_REGISTRY:
            for name, path in WORLD_REGISTRY.items():
                print(f"    {name:<25s} {path}")
        else:
            print("    (No world files found in data/)")
    print()


def cmd_run(args):
    """Execute a benchmark run."""
    # Build config from args or file
    if args.config:
        config = load_config(args.config)
        # CLI overrides
        if args.agent:
            config.agents = args.agent
        if args.task:
            config.tasks = args.task
        if args.world:
            config.worlds = args.world
        if args.seeds != 5:
            config.seeds = args.seeds
        if args.budget != 5000.0:
            config.budget = args.budget
        if args.output:
            config.output = args.output
        if args.workers != 1:
            config.workers = args.workers
        if args.save_trajectories:
            config.save_trajectories = True
    else:
        config = ExperimentConfig(
            agents=args.agent,
            tasks=args.task or ["defect_census"],
            worlds=args.world,
            seeds=args.seeds,
            budget=args.budget,
            max_steps=args.max_steps,
            score_every_n_acquires=args.score_every,
            output=args.output,
            workers=args.workers,
            save_trajectories=args.save_trajectories,
        )

    # Validate
    if not config.worlds:
        # Default to all discovered worlds
        if not WORLD_REGISTRY:
            print("Error: No world files found. Generate them first with scripts/generate_worlds/")
            sys.exit(1)
        config.worlds = list(WORLD_REGISTRY.keys())

    if not config.agents:
        config.agents = list(AGENT_REGISTRY.keys())

    # Parse shard
    shard = None
    if args.shard:
        parts = args.shard.split("/")
        if len(parts) != 2:
            print("Error: --shard must be 'K/N' (e.g. '1/4')")
            sys.exit(1)
        shard = (int(parts[0]), int(parts[1]))
        if shard[0] < 1 or shard[0] > shard[1]:
            print(f"Error: shard index must be 1..{shard[1]}, got {shard[0]}")
            sys.exit(1)

    # Dry run
    if args.dry_run:
        total = len(config.agents) * len(config.tasks) * len(config.worlds) * config.seeds
        print(f"\n  Dry-run plan:")
        print(f"    Agents: {config.agents}")
        print(f"    Tasks:  {config.tasks}")
        print(f"    Worlds: {config.worlds}")
        print(f"    Seeds:  {config.seeds}")
        print(f"    Budget: {config.budget}")
        print(f"    Workers: {config.workers}")
        if shard:
            print(f"    Shard:  {shard[0]}/{shard[1]}")
        print(f"    Total episodes: {total}")
        return

    # Run
    verbose = not args.quiet
    results = run_full_benchmark(config, verbose=verbose, shard=shard)

    # Print summary
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")
    print_summary_table(results)

    # Final save (also done incrementally during run for crash safety)
    output_path = config.output or args.output
    if output_path:
        _save_results_incremental(output_path, config, results)
        print(f"\n  Results saved to: {output_path}")


def cmd_merge(args):
    """Merge multiple shard result files into one."""
    all_results = []
    merged_config = None

    for input_path in args.inputs:
        with open(input_path) as f:
            data = json.load(f)
        if merged_config is None:
            merged_config = data.get("config", {})
        all_results.extend(data.get("results", []))

    # Deduplicate: if same (agent, task, world) appears in multiple shards, keep all
    # (they should be disjoint if sharded properly)
    if not all_results:
        print("  No results found in input files.")
        return

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "config": merged_config or {},
            "results": all_results,
        }, f, indent=2)

    print(f"  Merged {len(all_results)} result entries from {len(args.inputs)} files")
    print(f"  Output: {args.output}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)
    elif args.command == "list":
        cmd_list(args.what)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "merge":
        cmd_merge(args)


if __name__ == "__main__":
    main()
