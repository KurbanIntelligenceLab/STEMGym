#!/usr/bin/env python3
"""Post-hoc Pareto hypervolume analysis for composite task results.

Loads composite task results JSON, extracts per-checkpoint component
scores, and plots:
  1. Pareto front at final dose checkpoint (score_a vs score_b)
  2. Hypervolume vs dose fraction (HV-DEC curves per agent)

Usage:
    python scripts/plot_pareto_hypervolume.py results/composite_task.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stem_gym.metrics import compute_pareto_hypervolume


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data.get("results", [])


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/plot_pareto_hypervolume.py <results.json>")
        sys.exit(1)

    results = load_results(sys.argv[1])
    if not results:
        print("No results found.")
        sys.exit(1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Collect final (score_a, score_b) per agent for Pareto front
    agent_final_scores: dict[str, list[tuple[float, float]]] = {}

    for combo in results:
        agent = combo["agent"]
        world = combo["world"]
        budget = combo.get("budget", 5000.0)

        for ep in combo.get("episodes", []):
            # Reconstruct episode_log from dec_dose_fractions if trajectory exists
            traj = ep.get("trajectory", [])
            if not traj:
                continue

            # Check if trajectory entries have sub-scores
            has_sub = any("task_score_a" in t for t in traj)
            if not has_sub:
                continue

            hv_result = compute_pareto_hypervolume(traj, budget)

            # Final scores
            final_a = float(hv_result.scores_a[-1])
            final_b = float(hv_result.scores_b[-1])
            agent_final_scores.setdefault(agent, []).append((final_a, final_b))

    # Plot 1: Pareto front scatter
    for agent, scores in agent_final_scores.items():
        a_vals = [s[0] for s in scores]
        b_vals = [s[1] for s in scores]
        ax1.scatter(a_vals, b_vals, label=agent, alpha=0.7, s=50)

    ax1.set_xlabel("Defect Census F1")
    ax1.set_ylabel("Phase Mapping IoU")
    ax1.set_title("Pareto Front (Final Dose)")
    ax1.legend(fontsize=8)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Plot 2: HV-DEC curves — aggregate across episodes per agent
    # For simplicity, use the scalar composite DEC curves from the results
    for combo in results:
        agent = combo["agent"]
        world = combo["world"]
        label = f"{agent} / {world}"

        # Average DEC curves across episodes
        all_scores = []
        for ep in combo.get("episodes", []):
            if "dec_scores" in ep:
                all_scores.append(ep["dec_scores"])

        if all_scores:
            mean_scores = np.mean(all_scores, axis=0)
            dose_fracs = np.linspace(0, 1, len(mean_scores))
            ax2.plot(dose_fracs, mean_scores, label=label, alpha=0.8)

    ax2.set_xlabel("Dose Fraction")
    ax2.set_ylabel("Composite Score")
    ax2.set_title("Composite DEC Curves")
    ax2.legend(fontsize=7, ncol=2)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(sys.argv[1]).stem + "_pareto.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
