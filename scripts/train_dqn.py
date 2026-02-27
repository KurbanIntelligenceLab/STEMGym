#!/usr/bin/env python3
"""Train a DQN agent on STEMGym environments.

Uses Stable-Baselines3 DQN with the DiscreteSTEMWrapper.
Trains on a few easy worlds and saves the model checkpoint.

Usage:
    python scripts/train_dqn.py
    python scripts/train_dqn.py --worlds data/simulated/srtio3_clustered_easy.h5 --steps 100000
    python scripts/train_dqn.py --output checkpoints/dqn_agent.zip --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def find_easy_worlds() -> list[str]:
    """Auto-discover easy world files for training."""
    data_dir = Path(__file__).resolve().parent.parent / "data" / "simulated"
    if not data_dir.exists():
        return []
    worlds = []
    for pattern in ["*_easy.h5", "*_easy_verified.h5"]:
        for p in sorted(data_dir.glob(pattern)):
            # Avoid duplicates from verified symlinks
            resolved = str(p.resolve())
            if resolved not in [str(Path(w).resolve()) for w in worlds]:
                worlds.append(str(p))
    return worlds[:5]  # Cap at 5 worlds


def make_env(world_path: str, budget: float = 5000.0, max_steps: int = 200):
    """Create a wrapped STEMEnv for DQN training."""
    from stem_gym.baselines.rl_agent import DiscreteSTEMWrapper
    from stem_gym.envs.stem_env import STEMEnv
    from stem_gym.tasks.defect_census import DefectCensusTask

    task = DefectCensusTask()
    env = STEMEnv(world_path, task=task, total_budget=budget, max_steps=max_steps)
    return DiscreteSTEMWrapper(env)


def main():
    parser = argparse.ArgumentParser(description="Train DQN agent for STEMGym")
    parser.add_argument(
        "--worlds", nargs="+", default=None,
        help="World HDF5 files to train on (default: auto-discover easy worlds)",
    )
    parser.add_argument("--steps", type=int, default=50000, help="Total training steps")
    parser.add_argument("--output", type=str, default="checkpoints/dqn_agent.zip",
                        help="Output model path")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--budget", type=float, default=5000.0, help="Dose budget")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--buffer-size", type=int, default=50000, help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--exploration-fraction", type=float, default=0.3,
                        help="Fraction of training for epsilon decay")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    args = parser.parse_args()

    # Find worlds
    worlds = args.worlds or find_easy_worlds()
    if not worlds:
        print("ERROR: No world files found. Run `python scripts/download_data.py` first.")
        sys.exit(1)

    print(f"Training DQN on {len(worlds)} worlds:")
    for w in worlds:
        print(f"  {w}")
    print(f"Total steps: {args.steps}")
    print(f"Device: {args.device}")
    print()

    # Use first world for training (SB3 DQN expects single env)
    # Cycle through worlds by training in stages
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import EvalCallback

    steps_per_world = max(args.steps // len(worlds), 5000)
    model = None

    for i, world_path in enumerate(worlds):
        print(f"\n{'='*60}")
        print(f"  Training on world {i+1}/{len(worlds)}: {Path(world_path).stem}")
        print(f"  Steps: {steps_per_world}")
        print(f"{'='*60}")

        env = make_env(world_path, budget=args.budget, max_steps=args.max_steps)

        if model is None:
            model = DQN(
                "MlpPolicy",
                env,
                learning_rate=args.lr,
                buffer_size=args.buffer_size,
                batch_size=args.batch_size,
                exploration_fraction=args.exploration_fraction,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                train_freq=4,
                target_update_interval=1000,
                gamma=0.99,
                verbose=args.verbose,
                device=args.device,
            )
        else:
            model.set_env(env)

        model.learn(total_timesteps=steps_per_world, reset_num_timesteps=False)
        env.close()

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path).replace(".zip", ""))  # SB3 adds .zip automatically
    print(f"\nModel saved to: {args.output}")
    print(f"Size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
