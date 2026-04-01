#!/usr/bin/env python3
"""Train RL agents (DQN, PPO, SAC) on STEMGym environments.

Uses Stable-Baselines3 with appropriate wrappers per algorithm:
- DQN / PPO: DiscreteSTEMWrapper (Discrete(192) action space)
- SAC: ContinuousSTEMWrapper (Box(2) continuous action space)

Usage:
    python scripts/train_rl.py --algo dqn
    python scripts/train_rl.py --algo ppo --steps 50000
    python scripts/train_rl.py --algo sac --steps 50000 --device cuda
    python scripts/train_rl.py --algo dqn --worlds data/simulated/srtio3_clustered_easy.h5
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
            resolved = str(p.resolve())
            if resolved not in [str(Path(w).resolve()) for w in worlds]:
                worlds.append(str(p))
    return worlds[:5]  # Cap at 5 worlds


def make_env(world_path: str, algo: str, budget: float = 5000.0, max_steps: int = 200):
    """Create a wrapped STEMEnv for RL training."""
    from stem_gym.envs.stem_env import STEMEnv
    from stem_gym.tasks.defect_census import DefectCensusTask

    task = DefectCensusTask()
    env = STEMEnv(world_path, task=task, total_budget=budget, max_steps=max_steps)

    if algo == "sac":
        from stem_gym.baselines.rl_agent import ContinuousSTEMWrapper
        return ContinuousSTEMWrapper(env)
    else:
        from stem_gym.baselines.rl_agent import DiscreteSTEMWrapper
        return DiscreteSTEMWrapper(env)


def make_model(algo: str, env, args):
    """Create a fresh SB3 model for the given algorithm."""
    if algo == "dqn":
        from stable_baselines3 import DQN
        return DQN(
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
    elif algo == "ppo":
        from stable_baselines3 import PPO
        return PPO(
            "MlpPolicy",
            env,
            learning_rate=args.lr,
            n_steps=2048,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            clip_range=0.2,
            verbose=args.verbose,
            device=args.device,
        )
    elif algo == "sac":
        from stable_baselines3 import SAC
        return SAC(
            "MlpPolicy",
            env,
            learning_rate=args.lr,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            gamma=0.99,
            tau=0.005,
            ent_coef="auto",
            train_freq=1,
            verbose=args.verbose,
            device=args.device,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def main():
    parser = argparse.ArgumentParser(description="Train RL agents for STEMGym")
    parser.add_argument(
        "--algo", type=str, default="dqn", choices=["dqn", "ppo", "sac"],
        help="RL algorithm to train (default: dqn)",
    )
    parser.add_argument(
        "--worlds", nargs="+", default=None,
        help="World HDF5 files to train on (default: auto-discover easy worlds)",
    )
    parser.add_argument("--steps", type=int, default=50000, help="Total training steps")
    parser.add_argument("--output", type=str, default=None,
                        help="Output model path (default: checkpoints/{algo}_agent.zip)")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--budget", type=float, default=5000.0, help="Dose budget")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: 1e-4 for DQN, 3e-4 for PPO/SAC)")
    parser.add_argument("--buffer-size", type=int, default=50000, help="Replay buffer size (DQN/SAC)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--exploration-fraction", type=float, default=0.3,
                        help="Fraction of training for epsilon decay (DQN only)")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    args = parser.parse_args()

    # Default learning rate per algorithm
    if args.lr is None:
        args.lr = 1e-4 if args.algo == "dqn" else 3e-4

    # Default output path
    if args.output is None:
        args.output = f"checkpoints/{args.algo}_agent.zip"

    # Find worlds
    worlds = args.worlds or find_easy_worlds()
    if not worlds:
        print("ERROR: No world files found. Run `python scripts/download_data.py` first.")
        sys.exit(1)

    print(f"Training {args.algo.upper()} on {len(worlds)} worlds:")
    for w in worlds:
        print(f"  {w}")
    print(f"Total steps: {args.steps}")
    print(f"Device: {args.device}")
    print()

    steps_per_world = max(args.steps // len(worlds), 5000)
    model = None

    for i, world_path in enumerate(worlds):
        print(f"\n{'='*60}")
        print(f"  Training on world {i+1}/{len(worlds)}: {Path(world_path).stem}")
        print(f"  Steps: {steps_per_world}")
        print(f"{'='*60}")

        env = make_env(world_path, args.algo, budget=args.budget, max_steps=args.max_steps)

        if model is None:
            model = make_model(args.algo, env, args)
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
