#!/usr/bin/env python3
"""Compare human baseline sessions against automated agents.

Loads human session JSONs from results/human_baseline/ and agent result
JSONs, then plots DEC curves on the same axes for comparison.

Usage:
    python scripts/compare_human_vs_agents.py results/main_defect_census.json
    python scripts/compare_human_vs_agents.py results/*.json --human-dir results/human_baseline/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_human_sessions(human_dir: str) -> list[dict]:
    """Load all human session JSON files from a directory."""
    sessions = []
    d = Path(human_dir)
    if not d.exists():
        return sessions
    for f in sorted(d.glob("human_*.json")):
        with open(f) as fp:
            data = json.load(fp)
        data["_source"] = str(f)
        sessions.append(data)
    return sessions


def load_agent_results(paths: list[str]) -> list[dict]:
    """Load agent results from benchmark JSON files."""
    all_results = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        all_results.extend(data.get("results", []))
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Compare human vs agent DEC curves")
    parser.add_argument("agent_results", nargs="*", help="Agent result JSON files")
    parser.add_argument(
        "--human-dir", type=str, default="results/human_baseline/",
        help="Directory containing human session JSONs",
    )
    parser.add_argument("--output", "-o", type=str, default=None, help="Output PNG path")
    parser.add_argument("--task", "-t", type=str, default=None, help="Filter by task name")
    parser.add_argument("--world", "-w", type=str, default=None, help="Filter by world name")
    args = parser.parse_args()

    human_sessions = load_human_sessions(args.human_dir)
    agent_results = load_agent_results(args.agent_results) if args.agent_results else []

    if not human_sessions and not agent_results:
        print("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot agent DEC curves
    for combo in agent_results:
        agent = combo.get("agent", "?")
        task = combo.get("task", "?")
        world = combo.get("world", "?")

        if args.task and task != args.task:
            continue
        if args.world and world != args.world:
            continue

        # Average DEC across episodes
        all_scores = []
        for ep in combo.get("episodes", []):
            if "dec_scores" in ep:
                all_scores.append(ep["dec_scores"])

        if all_scores:
            mean_scores = np.mean(all_scores, axis=0)
            dose_fracs = np.linspace(0, 1, len(mean_scores))
            label = f"{agent} (AUC={combo.get('mean_dec_auc', 0):.3f})"
            ax.plot(dose_fracs, mean_scores, label=label, alpha=0.8)

    # Plot human DEC curves
    for session in human_sessions:
        task = session.get("task", "?")
        world = session.get("world", "?")

        if args.task and task != args.task:
            continue
        if args.world and world != args.world:
            continue

        dec_scores = session.get("dec_scores", [])
        dec_fracs = session.get("dec_dose_fractions", [])
        dec_auc = session.get("dec_auc", 0)

        if dec_scores and dec_fracs:
            label = f"Human ({world}, AUC={dec_auc:.3f})"
            ax.plot(dec_fracs, dec_scores, label=label, linewidth=2.5,
                    linestyle="--", color="red", alpha=0.9)

    ax.set_xlabel("Dose Fraction")
    ax.set_ylabel("Task Score")
    ax.set_title("Human vs Agent DEC Curves")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out_path = args.output or "human_vs_agents.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
