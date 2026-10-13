"""Episode runner: execute agents on environments, collect DEC-ready logs.

Provides run_episode() for single episodes and run_benchmark() for
multi-seed evaluation. Handles periodic scoring checkpoints for DEC
curve computation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from stem_gym.baselines import Agent
from stem_gym.envs.stem_env import STEMEnv
from stem_gym.envs.world import World
from stem_gym.metrics import DECResult, compute_dec


from stem_gym.scoring import filter_gt_to_explored


@dataclass
class EpisodeResult:
    """Result of running one episode."""
    episode_log: list[dict[str, Any]]   # per-step info dicts with task_score
    total_dose_used: float
    total_steps: int
    tiles_acquired: int
    final_score: float
    dec: DECResult
    agent_findings: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    seed: int = 0


def run_episode(
    env: STEMEnv,
    agent: Agent,
    seed: int = 42,
    score_every_n_acquires: int = 5,
) -> EpisodeResult:
    """Run a single episode, scoring periodically for DEC computation.

    The agent's findings are scored at periodic intervals (every N
    acquisitions) to build the DEC curve, not just at episode end.

    Args:
        env: STEMGym environment (already constructed with task).
        agent: Agent implementing reset()/act() protocol.
        seed: Random seed for env reset.
        score_every_n_acquires: Score findings every N tile acquisitions.

    Returns:
        EpisodeResult with episode log, DEC curve, etc.
    """
    if score_every_n_acquires < 1:
        raise ValueError("Scoring interval must be positive")
    obs, info = env.reset(seed=seed)
    agent.reset(obs, info, seed=seed)

    episode_log: list[dict[str, Any]] = []
    acquires_since_score = 0
    last_tiles_acquired = 0

    # +10 buffer: DECLARE annotation actions don't consume dose or count
    # toward max_steps, so the loop may need extra iterations to converge
    for step in range(env.max_steps + 10):
        action = agent.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)

        agent.observe(obs, info)

        # Track acquisitions
        current_tiles = env.state.acquisitions
        if current_tiles > last_tiles_acquired:
            acquires_since_score += (current_tiles - last_tiles_acquired)
            last_tiles_acquired = current_tiles

        # Periodic scoring checkpoint
        log_entry = {
            "step": step,
            "action_type": int(action.get("action_type", -1)),
            "dose_used": env.state.dose_used,
            "tiles_acquired": env.state.tiles_acquired,
            "acquisitions": current_tiles,
            "position": env.state.position,
            "n_declarations": len(env.state.findings),
            "task_score": None,
        }

        if acquires_since_score >= score_every_n_acquires:
            # Score current findings
            findings = agent.get_findings()
            # Also update env state findings for scoring
            env.state.defect_census = findings.get("defect_census", {})
            env.state.declared_phase_map = findings.get("phase_map")
            gt = env.world.get_ground_truth()
            # Filter GT to explored region for atom-level tasks
            if env.task.name != "phase_mapping":
                gt = filter_gt_to_explored(gt, env.state.visited_mask, env.world)
            score = env.task.score(findings, gt)
            log_entry["task_score"] = score
            acquires_since_score = 0

        # Final scoring at episode end
        if terminated or truncated:
            findings = agent.get_findings()
            gt = env.world.get_ground_truth()
            # Filter GT to explored region for atom-level tasks
            if env.task.name != "phase_mapping":
                gt = filter_gt_to_explored(gt, env.state.visited_mask, env.world)
            score = env.task.score(findings, gt)
            log_entry["task_score"] = score
            episode_log.append(log_entry)
            break

        episode_log.append(log_entry)
    else:
        raise RuntimeError("Agent exceeded the interaction limit without terminating")

    # Compute DEC
    dec = compute_dec(episode_log, env.total_budget)

    return EpisodeResult(
        episode_log=episode_log,
        total_dose_used=env.state.dose_used,
        total_steps=env.state.step_count,
        tiles_acquired=env.state.tiles_acquired,
        final_score=dec.scores[-1] if len(dec.scores) > 0 else 0.0,
        dec=dec,
        agent_findings=agent.get_findings(),
        seed=seed,
    )


@dataclass
class BenchmarkResult:
    """Result of running multiple episodes (seeds) of one agent."""
    agent_name: str
    episode_results: list[EpisodeResult]
    mean_dec_auc: float
    std_dec_auc: float
    mean_final_score: float
    mean_dose_used: float
    mean_tiles: float


def run_benchmark(
    env: STEMEnv,
    agent: Agent,
    agent_name: str,
    seeds: list[int] | None = None,
    n_seeds: int = 5,
    score_every_n_acquires: int = 5,
    verbose: bool = False,
) -> BenchmarkResult:
    """Run an agent over multiple seeds, compute summary statistics.

    Args:
        env: STEMGym environment.
        agent: Agent to evaluate.
        agent_name: Name for reporting.
        seeds: Explicit seed list. If None, uses range(n_seeds).
        n_seeds: Number of seeds if seeds not provided.
        score_every_n_acquires: Scoring frequency.
        verbose: Print per-seed results.

    Returns:
        BenchmarkResult with aggregated statistics.
    """
    if seeds is None:
        seeds = list(range(n_seeds))

    if not seeds:
        raise ValueError("At least one seed is required")
    results = []
    for seed in seeds:
        t0 = time.perf_counter()
        result = run_episode(env, agent, seed=seed,
                             score_every_n_acquires=score_every_n_acquires)
        result.elapsed_seconds = time.perf_counter() - t0
        result.seed = seed
        results.append(result)
        if verbose:
            print(
                f"  {agent_name} seed={seed}: "
                f"DEC-AUC={result.dec.auc:.4f}, "
                f"final={result.final_score:.4f}, "
                f"tiles={result.tiles_acquired}, "
                f"dose={result.total_dose_used:.0f}"
            )

    aucs = [r.dec.auc for r in results]
    return BenchmarkResult(
        agent_name=agent_name,
        episode_results=results,
        mean_dec_auc=float(np.mean(aucs)),
        std_dec_auc=float(np.std(aucs)),
        mean_final_score=float(np.mean([r.final_score for r in results])),
        mean_dose_used=float(np.mean([r.total_dose_used for r in results])),
        mean_tiles=float(np.mean([r.tiles_acquired for r in results])),
    )
