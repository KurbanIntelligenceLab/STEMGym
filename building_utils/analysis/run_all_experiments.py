#!/usr/bin/env python3
"""Master script to run all STEMGym benchmark experiments.

Usage:
    python scripts/run_all_experiments.py              # run all experiments
    python scripts/run_all_experiments.py --dry-run     # print plan only
    python scripts/run_all_experiments.py --only fairness  # run specific experiment
    python scripts/run_all_experiments.py --list        # list experiments

Each experiment writes results to results/benchmark/<experiment_name>.json.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "benchmark"
CONFIG_DIR = PROJECT_ROOT / "experiments" / "configs" / "benchmark"

EXPERIMENTS = [
    {
        "name": "fairness_5seeds",
        "config": "fairness_5seeds.yaml",
        "description": "Fairness comparison at budget=5000, 5 seeds",
    },
    {
        "name": "tight_budget_2000",
        "config": "tight_budget_2000.yaml",
        "description": "Tight budget (2000) comparison, 5 seeds",
    },
    {
        "name": "tight_budget_1000",
        "config": "tight_budget_1000.yaml",
        "description": "Extreme budget (1000), 5 seeds",
    },
    {
        "name": "clustered_fairness_5000",
        "config": "clustered_fairness_5000.yaml",
        "description": "Clustered worlds fairness comparison, budget=5000, 5 seeds",
    },
    {
        "name": "clustered_tight_2000",
        "config": "clustered_tight_2000.yaml",
        "description": "Clustered worlds tight budget (2000), 5 seeds",
    },
    {
        "name": "clustered_extreme_1000",
        "config": "clustered_extreme_1000.yaml",
        "description": "Clustered worlds extreme budget (1000), 5 seeds",
    },
]


def run_experiment(name: str, config: str, dry_run: bool = False) -> bool:
    """Run a single experiment. Returns True on success."""
    config_path = CONFIG_DIR / config
    output_path = RESULTS_DIR / f"{name}.json"

    if not config_path.exists():
        print(f"  [SKIP] Config not found: {config_path}")
        return False

    if output_path.exists():
        print(f"  [SKIP] Results already exist: {output_path}")
        print(f"         Delete the file to re-run.")
        return True

    cmd = [
        sys.executable, "-m", "experiments.run_benchmark",
        "run",
        "--config", str(config_path),
        "--output", str(output_path),
    ]

    if dry_run:
        cmd.append("--dry-run")

    print(f"  [RUN] {' '.join(cmd)}")
    t0 = time.time()

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"  [OK] Completed in {elapsed:.0f}s")
        return True
    else:
        print(f"  [FAIL] Exit code {result.returncode} after {elapsed:.0f}s")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run all STEMGym benchmark experiments")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only")
    parser.add_argument("--only", type=str, help="Run only this experiment (by name)")
    parser.add_argument("--list", action="store_true", help="List experiments and exit")
    args = parser.parse_args()

    if args.list:
        print("\nSTEMGym Benchmark Experiments:")
        for i, exp in enumerate(EXPERIMENTS, 1):
            config_path = CONFIG_DIR / exp["config"]
            status = "ready" if config_path.exists() else "config missing"
            result_path = RESULTS_DIR / f"{exp['name']}.json"
            if result_path.exists():
                status = "done"
            print(f"  [{i}] {exp['name']:<30s} [{status}] {exp['description']}")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    experiments = EXPERIMENTS
    if args.only:
        experiments = [e for e in EXPERIMENTS if e["name"] == args.only]
        if not experiments:
            names = [e["name"] for e in EXPERIMENTS]
            print(f"Unknown experiment '{args.only}'. Available: {names}")
            sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  STEMGym Benchmark Suite")
    print(f"  Results directory: {RESULTS_DIR}")
    print(f"  Experiments: {len(experiments)}")
    print(f"{'='*70}\n")

    successes = 0
    failures = 0

    for i, exp in enumerate(experiments, 1):
        print(f"\n--- [{i}] {exp['name']}: {exp['description']} ---")
        if run_experiment(exp["name"], exp["config"], dry_run=args.dry_run):
            successes += 1
        else:
            failures += 1

    print(f"\n{'='*70}")
    print(f"  Done: {successes} succeeded, {failures} failed/skipped")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
