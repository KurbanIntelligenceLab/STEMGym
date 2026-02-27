#!/usr/bin/env python3
"""Cross-material transfer experiment.

Tests whether models trained on one material generalize to others.
Trains separate checkpoints per material, then evaluates each on all
materials (5×5 train/eval matrix).

Usage:
    python scripts/run_transfer_experiment.py \
        --checkpoint-base checkpoints/transfer \
        --seeds 5 --device cuda \
        --output results/transfer_experiment.json

Expects checkpoint layout:
    checkpoints/transfer/
        srtio3/  (atom_finder.pt, defect_classifier.pt, phase_identifier.pt)
        batio3/
        sige/
        gan/
        pt/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Ensure stem_gym is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stem_gym.agents.multi_agent import STEMAgent
from stem_gym.baselines.equipped_baselines import (
    GPUCBEquippedAgent,
    RasterEquippedAgent,
)
from stem_gym.envs.stem_env import STEMEnv
from stem_gym.runner import BenchmarkResult, run_benchmark
from stem_gym.tasks.defect_census import DefectCensusTask

# Material short names → easy world file stems
MATERIALS = {
    "srtio3": "srtio3_clustered_easy",
    "batio3": "batio3_interface_easy",
    "sige": "sige_gradient_clustered_easy",
    "gan": "gan_easy",
    "pt": "pt_nanoparticles_easy",
}

# Agents to test (each supports checkpoint_dir kwarg)
AGENT_CONFIGS = {
    "stem_agent": STEMAgent,
    "gp_ucb_equipped": GPUCBEquippedAgent,
    "raster_equipped": RasterEquippedAgent,
}


def find_world(world_stem: str, data_dir: Path) -> str:
    """Find a world HDF5 file by stem name."""
    for subdir in ["simulated", "replay"]:
        d = data_dir / subdir
        if d.exists():
            for f in sorted(d.glob("*.h5")):
                if f.stem == world_stem:
                    return str(f)
    raise FileNotFoundError(f"World not found: {world_stem} in {data_dir}")


def serialize_result(
    bench: BenchmarkResult,
    agent_name: str,
    train_mat: str,
    eval_mat: str,
    world_name: str,
) -> dict:
    """Convert BenchmarkResult to JSON-serializable dict."""
    return {
        "agent": agent_name,
        "train_material": train_mat,
        "eval_material": eval_mat,
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
                "dec_dose_fractions": r.dec.dose_fractions.tolist(),
                "dec_scores": r.dec.scores.tolist(),
            }
            for i, r in enumerate(bench.episode_results)
        ],
    }


def run_transfer(
    checkpoint_base: Path,
    data_dir: Path,
    output_path: Path,
    n_seeds: int = 5,
    budget: float = 5000.0,
    max_steps: int = 200,
    score_every: int = 5,
    verbose: bool = True,
) -> list[dict]:
    """Run the full 5×5 transfer matrix for all agents with resume support."""
    materials = list(MATERIALS.keys())

    # --- Resume: load existing results and build completed set ---
    all_results = []
    completed = set()
    if output_path.exists():
        try:
            with open(output_path) as f:
                existing = json.load(f)
            all_results = existing.get("results", [])
            for r in all_results:
                key = (r["agent"], r["train_material"], r["eval_material"])
                completed.add(key)
            print(f"  Resuming: {len(completed)} combos already done, skipping them")
        except (json.JSONDecodeError, KeyError):
            print("  WARNING: Could not parse existing results, starting fresh")
            all_results = []

    total_combos = len(AGENT_CONFIGS) * len(materials) * len(materials)
    combo_idx = 0

    for agent_name, agent_cls in AGENT_CONFIGS.items():
        for train_mat in materials:
            cp_dir = checkpoint_base / train_mat
            if not cp_dir.exists():
                print(f"  WARNING: Checkpoint dir missing: {cp_dir}, skipping")
                combo_idx += len(materials)
                continue

            for eval_mat in materials:
                combo_idx += 1
                combo_key = (agent_name, train_mat, eval_mat)

                # Skip already-completed combos
                if combo_key in completed:
                    if verbose:
                        print(f"  [{combo_idx}/{total_combos}] {agent_name} train={train_mat} eval={eval_mat} — SKIPPED (done)")
                    continue

                world_stem = MATERIALS[eval_mat]

                try:
                    world_path = find_world(world_stem, data_dir)
                except FileNotFoundError as e:
                    print(f"  WARNING: {e}, skipping")
                    continue

                label = (
                    f"[{combo_idx}/{total_combos}] "
                    f"{agent_name} train={train_mat} eval={eval_mat}"
                )
                if verbose:
                    print(f"\n  {label}")

                t0 = time.time()

                task = DefectCensusTask()
                env = STEMEnv(
                    world_path,
                    task=task,
                    total_budget=budget,
                    max_steps=max_steps,
                )

                # Instantiate agent with specific checkpoint dir
                if agent_cls == GPUCBEquippedAgent:
                    agent = agent_cls(
                        checkpoint_dir=str(cp_dir),
                        initial_random=3,
                    )
                else:
                    agent = agent_cls(checkpoint_dir=str(cp_dir))

                bench = run_benchmark(
                    env,
                    agent,
                    agent_name,
                    n_seeds=n_seeds,
                    score_every_n_acquires=score_every,
                    verbose=verbose,
                )
                env.close()
                elapsed = time.time() - t0

                result = serialize_result(
                    bench, agent_name, train_mat, eval_mat, world_stem
                )
                result["elapsed_seconds"] = round(elapsed, 2)
                all_results.append(result)

                # --- Incremental save after each combo ---
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w") as f:
                    json.dump(
                        {
                            "config": {
                                "name": "transfer_experiment",
                                "checkpoint_base": str(checkpoint_base),
                                "materials": MATERIALS,
                                "agents": list(AGENT_CONFIGS.keys()),
                                "seeds": n_seeds,
                                "budget": budget,
                                "max_steps": max_steps,
                            },
                            "results": all_results,
                        },
                        f,
                        indent=2,
                    )

                if verbose:
                    print(
                        f"  => DEC-AUC: {bench.mean_dec_auc:.4f} "
                        f"+/- {bench.std_dec_auc:.4f} | "
                        f"time: {elapsed:.1f}s"
                    )

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Cross-material transfer experiment"
    )
    parser.add_argument(
        "--checkpoint-base",
        type=str,
        default="checkpoints/transfer",
        help="Base directory with per-material checkpoint subdirs",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory containing simulated/ worlds",
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--budget", type=float, default=5000.0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--score-every", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results/transfer_experiment.json",
    )
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    # Set CUDA device if specified
    if args.device != "cpu":
        import torch
        if torch.cuda.is_available():
            torch.set_default_device(args.device)

    checkpoint_base = Path(args.checkpoint_base)
    data_dir = Path(args.data_dir)

    if not checkpoint_base.exists():
        print(f"Error: Checkpoint base directory not found: {checkpoint_base}")
        print("Run training first:")
        print("  for mat in srtio3 batio3 sige gan pt; do")
        print('    python scripts/train_atom_finder.py --world data/simulated/${mat}*_easy*.h5 \\')
        print(f"      --output checkpoints/transfer/$mat --epochs 30 --device cuda")
        print("  done")
        sys.exit(1)

    verbose = not args.quiet

    print(f"\n{'='*70}")
    print(f"  Cross-Material Transfer Experiment")
    print(f"{'='*70}")
    print(f"  Checkpoint base: {checkpoint_base}")
    print(f"  Materials: {list(MATERIALS.keys())}")
    print(f"  Agents: {list(AGENT_CONFIGS.keys())}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Budget: {args.budget}")
    total = len(AGENT_CONFIGS) * len(MATERIALS) ** 2 * args.seeds
    print(f"  Total episodes: {total}")
    print(f"{'='*70}\n")

    out = Path(args.output)
    results = run_transfer(
        checkpoint_base=checkpoint_base,
        data_dir=data_dir,
        output_path=out,
        n_seeds=args.seeds,
        budget=args.budget,
        max_steps=args.max_steps,
        score_every=args.score_every,
        verbose=verbose,
    )

    # Final save (in case no new combos were run)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(
            {
                "config": {
                    "name": "transfer_experiment",
                    "checkpoint_base": str(checkpoint_base),
                    "materials": MATERIALS,
                    "agents": list(AGENT_CONFIGS.keys()),
                    "seeds": args.seeds,
                    "budget": args.budget,
                    "max_steps": args.max_steps,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\n  Results saved to: {args.output}")

    # Print summary matrix
    print(f"\n{'='*70}")
    print(f"  TRANSFER MATRIX (DEC-AUC)")
    print(f"{'='*70}")

    materials = list(MATERIALS.keys())
    for agent_name in AGENT_CONFIGS:
        print(f"\n  {agent_name}:")
        header = "  train\\eval  " + "  ".join(f"{m:>8s}" for m in materials)
        print(header)
        print(f"  {'-' * len(header)}")
        for train_mat in materials:
            row = f"  {train_mat:<10s}  "
            for eval_mat in materials:
                match = [
                    r for r in results
                    if r["agent"] == agent_name
                    and r["train_material"] == train_mat
                    and r["eval_material"] == eval_mat
                ]
                if match:
                    row += f"  {match[0]['mean_dec_auc']:>7.4f}"
                else:
                    row += f"  {'---':>7s}"
            print(row)


if __name__ == "__main__":
    main()
