#!/usr/bin/env python3
"""STEMGym Benchmark Runner CLI.

Usage:
    python -m experiments.run_benchmark --config experiments/configs/full_benchmark.yaml
    python -m experiments.run_benchmark --agent stem_agent --task defect_census --world data/simulated/test_world.h5
    stemgym run --config experiments/configs/ablation.yaml --output results/ablation.json

Examples:
    # Quick smoke test
    stemgym run --agent random raster --task defect_census --world data/simulated/test_world.h5 --seeds 2 --budget 3000

    # Full benchmark from config
    stemgym run --config experiments/configs/full_benchmark.yaml --output results/full.json

    # Dry run (print plan without executing)
    stemgym run --config experiments/configs/full_benchmark.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Ensure stem_gym is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stem_gym.baselines import Agent
from stem_gym.envs.stem_env import STEMEnv
from stem_gym.metrics import DECResult
from stem_gym.runner import BenchmarkResult, EpisodeResult, run_benchmark, run_episode
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
    )


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------

def _serialize_result(bench: BenchmarkResult, task_name: str, world_name: str) -> dict:
    """Convert BenchmarkResult to JSON-serializable dict."""
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
        "episodes": [
            {
                "seed": i,
                "dec_auc": round(r.dec.auc, 6),
                "final_score": round(r.final_score, 6),
                "tiles_acquired": r.tiles_acquired,
                "total_dose_used": round(r.total_dose_used, 1),
                "total_steps": r.total_steps,
            }
            for i, r in enumerate(bench.episode_results)
        ],
    }


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_full_benchmark(config: ExperimentConfig, verbose: bool = True) -> list[dict]:
    """Run the full benchmark matrix specified by config.

    Returns list of serialized result dicts.
    """
    agents = config.agents or list(AGENT_REGISTRY.keys())
    tasks = config.tasks or list(TASK_REGISTRY.keys())
    worlds = config.worlds or list(WORLD_REGISTRY.keys())

    total_runs = len(agents) * len(tasks) * len(worlds)
    if verbose:
        print(f"\n{'='*70}")
        print(f"  STEMGym Benchmark: {config.name}")
        print(f"{'='*70}")
        print(f"  Agents: {agents}")
        print(f"  Tasks:  {tasks}")
        print(f"  Worlds: {worlds}")
        print(f"  Seeds:  {config.seeds}")
        print(f"  Budget: {config.budget}")
        print(f"  Total combinations: {total_runs}")
        print(f"{'='*70}\n")

    all_results = []
    run_idx = 0

    for world_name in worlds:
        world_path = resolve_world(world_name)

        for task_name in tasks:
            task = make_task(task_name)

            for agent_name in agents:
                run_idx += 1
                label = f"[{run_idx}/{total_runs}] {agent_name} / {task_name} / {world_name}"

                if verbose:
                    print(f"\n  {label}")
                    print(f"  {'-'*len(label)}")

                t0 = time.time()
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

                result = _serialize_result(bench, task_name, world_name)
                result["elapsed_seconds"] = round(elapsed, 2)
                all_results.append(result)

                if verbose:
                    print(
                        f"  => DEC-AUC: {bench.mean_dec_auc:.4f} +/- {bench.std_dec_auc:.4f}"
                        f"  | tiles: {bench.mean_tiles:.1f}"
                        f"  | time: {elapsed:.1f}s"
                    )

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
        )

    # Validate
    if not config.worlds:
        # Default to all discovered worlds
        if not WORLD_REGISTRY:
            print("Error: No world files found. Generate them first with data/scripts/")
            sys.exit(1)
        config.worlds = list(WORLD_REGISTRY.keys())

    if not config.agents:
        config.agents = list(AGENT_REGISTRY.keys())

    # Dry run
    if args.dry_run:
        total = len(config.agents) * len(config.tasks) * len(config.worlds) * config.seeds
        print(f"\n  Dry-run plan:")
        print(f"    Agents: {config.agents}")
        print(f"    Tasks:  {config.tasks}")
        print(f"    Worlds: {config.worlds}")
        print(f"    Seeds:  {config.seeds}")
        print(f"    Budget: {config.budget}")
        print(f"    Total episodes: {total}")
        return

    # Run
    verbose = not args.quiet
    results = run_full_benchmark(config, verbose=verbose)

    # Print summary
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")
    print_summary_table(results)

    # Save output
    output_path = config.output or args.output
    if output_path:
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
                },
                "results": results,
            }, f, indent=2)
        print(f"\n  Results saved to: {output_path}")


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


if __name__ == "__main__":
    main()
