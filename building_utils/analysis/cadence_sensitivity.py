#!/usr/bin/env python3
"""Scoring cadence sensitivity analysis for DEC-AUC.

Tests whether DEC-AUC rankings are stable across different scoring cadences
by resampling stored 50-checkpoint DEC curves at sparser intervals.

Reviewer concern (Q2): "How sensitive is DEC-AUC to the scoring cadence
(every 5 acquisitions)?"

Approach:
  For each cadence N (in acquisitions), compute the dose fractions where
  scoring would occur, resample from the stored smooth DEC curve via np.interp,
  recompute AUC via trapezoidal integration, and compare rankings via Spearman ρ.

Usage:
    python scripts/cadence_sensitivity.py
    python scripts/cadence_sensitivity.py --results-dir results/ --output paper/figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT = REPO / "paper" / "figures"

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

AGENT_LABELS = {
    "random": "Random",
    "raster": "Raster",
    "gp_ucb": "GP-UCB",
    "gp_ucb_equipped": "GP-UCB+CNN",
    "raster_equipped": "Raster+CNN",
    "stem_agent": "STEMAgent",
    "no_planner": "No Planner",
    "no_uncertainty": "No Unc.",
    "rule_planner": "Rule Plan.",
    "llm_planner": "LLM Plan.",
}

# Default environment parameters
DOSE_PER_ACQUIRE = 100  # e-/A^2 per acquisition
DEFAULT_BUDGET = 5000   # e-/A^2 total budget
MAX_ACQUIRES = DEFAULT_BUDGET // DOSE_PER_ACQUIRE  # 50 acquisitions


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(path: Path) -> list[dict]:
    """Load results from a JSON file."""
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return []
    with open(path) as f:
        data = json.load(f)
    results = data.get("results", data if isinstance(data, list) else [])
    for r in results:
        if "world" in r:
            r["world"] = r["world"].replace("_verified", "")
    return results


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_dec_at_cadence(
    dose_fractions: np.ndarray,
    scores: np.ndarray,
    cadence_n: int,
    budget: float = DEFAULT_BUDGET,
    dose_per_acquire: float = DOSE_PER_ACQUIRE,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a DEC curve at a given scoring cadence.

    Args:
        dose_fractions: Original 50-point dose fraction array [0, ..., 1].
        scores: Original 50-point score array.
        cadence_n: Score every N acquisitions.
        budget: Total dose budget.
        dose_per_acquire: Dose per acquisition.

    Returns:
        (new_dose_fractions, new_scores) at the resampled cadence.
    """
    max_acquires = int(budget / dose_per_acquire)
    # Scoring happens at acquisitions: cadence_n, 2*cadence_n, ..., up to max_acquires
    scoring_acquires = np.arange(cadence_n, max_acquires + 1, cadence_n)
    scoring_fractions = scoring_acquires * dose_per_acquire / budget
    # Clip to [0, 1]
    scoring_fractions = scoring_fractions[scoring_fractions <= 1.0]

    # Prepend 0.0 (score at zero dose is 0)
    new_fracs = np.concatenate([[0.0], scoring_fractions])
    # Interpolate from original smooth curve
    new_scores = np.interp(new_fracs, dose_fractions, scores)
    # Score at zero dose should be 0
    new_scores[0] = 0.0

    return new_fracs, new_scores


def compute_auc(dose_fractions: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUC via trapezoidal integration."""
    if len(dose_fractions) < 2:
        return 0.0
    return float(np.trapezoid(scores, dose_fractions))


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_cadence_sensitivity(
    results: list[dict],
    cadences: list[int] | None = None,
    task: str = "defect_census",
) -> dict:
    """Analyze DEC-AUC sensitivity to scoring cadence.

    Returns dict with:
      - per_cadence: {cadence -> {agent -> mean_auc}}
      - rank_correlations: {cadence -> spearman_rho vs baseline}
      - agents: list of agents found
    """
    if cadences is None:
        cadences = [1, 2, 5, 10, 20]

    # Collect all episodes per agent
    agent_episodes: dict[str, list[dict]] = {}
    for r in results:
        if r.get("task") != task:
            continue
        agent = r["agent"]
        for ep in r.get("episodes", []):
            fracs = ep.get("dec_dose_fractions")
            scores = ep.get("dec_scores")
            if fracs and scores:
                agent_episodes.setdefault(agent, []).append({
                    "fracs": np.array(fracs),
                    "scores": np.array(scores),
                })

    if not agent_episodes:
        print("  No episodes found!")
        return {"per_cadence": {}, "rank_correlations": {}, "agents": []}

    agents = sorted(agent_episodes.keys())

    # Compute AUC at each cadence
    per_cadence: dict[int, dict[str, float]] = {}
    for cadence in cadences:
        per_cadence[cadence] = {}
        for agent in agents:
            aucs = []
            for ep in agent_episodes[agent]:
                new_fracs, new_scores = resample_dec_at_cadence(
                    ep["fracs"], ep["scores"], cadence,
                )
                aucs.append(compute_auc(new_fracs, new_scores))
            per_cadence[cadence][agent] = float(np.mean(aucs))

    # Compute Spearman rank correlations vs baseline cadence (5)
    baseline_cadence = 5
    baseline_ranks = [per_cadence[baseline_cadence].get(a, 0) for a in agents]

    rank_correlations: dict[int, float] = {}
    for cadence in cadences:
        cadence_values = [per_cadence[cadence].get(a, 0) for a in agents]
        if len(agents) >= 3:
            rho, _ = spearmanr(baseline_ranks, cadence_values)
            rank_correlations[cadence] = float(rho)
        else:
            rank_correlations[cadence] = 1.0

    return {
        "per_cadence": per_cadence,
        "rank_correlations": rank_correlations,
        "agents": agents,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_cadence_figure(
    analysis: dict,
    output_dir: Path,
    extra_rank_corrs: dict[str, dict[int, float]] | None = None,
) -> None:
    """Generate cadence sensitivity figure with two panels.

    Args:
        extra_rank_corrs: Optional dict of {task_label -> {cadence -> rho}} for
            additional Spearman lines in panel (b).
    """
    per_cadence = analysis["per_cadence"]
    rank_corrs = analysis["rank_correlations"]
    agents = analysis["agents"]
    cadences = sorted(per_cadence.keys())

    if not cadences:
        print("  No data to plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # Panel A: DEC-AUC vs cadence for each agent
    agent_colors = {
        "random": "#999999", "raster": "#E65100", "gp_ucb": "#1565C0",
        "gp_ucb_equipped": "#2196F3", "raster_equipped": "#FF9800",
        "stem_agent": "#4CAF50", "no_planner": "#81C784",
        "no_uncertainty": "#66BB6A", "rule_planner": "#A5D6A7",
        "llm_planner": "#C8E6C9",
    }

    for agent in agents:
        aucs = [per_cadence[c].get(agent, 0) for c in cadences]
        label = AGENT_LABELS.get(agent, agent)
        color = agent_colors.get(agent, "#333333")
        ax1.plot(cadences, aucs, "o-", color=color, label=label,
                 markersize=4, linewidth=1.2)

    ax1.set_xlabel("Scoring cadence (every $N$ acquisitions)")
    ax1.set_ylabel("DEC-AUC")
    ax1.set_xticks(cadences)
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=5.5, loc="best", ncol=2, frameon=True,
               fancybox=False, edgecolor="#CCCCCC")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_title("(a) DEC-AUC vs scoring cadence", fontsize=8, fontweight="bold")

    # Panel B: Spearman rank correlation — per task
    task_colors = {
        "Defect Census": "#4CAF50",
        "Phase Mapping": "#2196F3",
        "Targeted": "#FF9800",
    }
    task_markers = {
        "Defect Census": "s",
        "Phase Mapping": "D",
        "Targeted": "^",
    }

    rhos = [rank_corrs[c] for c in cadences]
    ax2.plot(cadences, rhos, marker=task_markers["Defect Census"], linestyle="-",
             color=task_colors["Defect Census"], markersize=5, linewidth=1.5,
             label="Defect Census")

    if extra_rank_corrs:
        for task_label, corrs in extra_rank_corrs.items():
            rhos_extra = [corrs.get(c, np.nan) for c in cadences]
            ax2.plot(cadences, rhos_extra,
                     marker=task_markers.get(task_label, "o"), linestyle="-",
                     color=task_colors.get(task_label, "#333333"),
                     markersize=5, linewidth=1.5, label=task_label)

    ax2.axhline(0.95, color="#999999", linestyle="--", linewidth=0.8, label=r"$\rho = 0.95$")
    ax2.set_xlabel("Scoring cadence (every $N$ acquisitions)")
    ax2.set_ylabel(r"Spearman $\rho$ (vs cadence = 5)")
    ax2.set_xticks(cadences)
    ax2.set_ylim(0.85, 1.03)
    ax2.legend(fontsize=6, frameon=True, fancybox=False, edgecolor="#CCCCCC")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_title("(b) Rank stability across tasks", fontsize=8, fontweight="bold")

    fig.tight_layout()
    outpath = output_dir / "fig_cadence_sensitivity.pdf"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DEC-AUC cadence sensitivity analysis")
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print("DEC-AUC Cadence Sensitivity Analysis")
    print(f"Results dir: {args.results_dir}")
    print()

    # Load primary dataset
    results_dc = load_results(args.results_dir / "main_defect_census.json")
    if not results_dc:
        print("ERROR: No results found. Run experiments first.")
        return

    print(f"Loaded {len(results_dc)} defect census entries")

    # Run analysis for primary task
    cadences = [1, 2, 5, 10, 20]
    analysis = analyze_cadence_sensitivity(results_dc, cadences=cadences)

    # Print results table
    agents = analysis["agents"]
    per_cadence = analysis["per_cadence"]
    rank_corrs = analysis["rank_correlations"]

    print(f"\n{'Agent':<20}", end="")
    for c in cadences:
        print(f"  N={c:>2}", end="")
    print()
    print("-" * (20 + 7 * len(cadences)))

    for agent in agents:
        label = AGENT_LABELS.get(agent, agent)
        print(f"{label:<20}", end="")
        for c in cadences:
            auc = per_cadence[c].get(agent, 0)
            print(f" {auc:>.3f}", end="")
        print()

    print()
    print("Spearman rank correlations (vs cadence=5):")
    for c in cadences:
        rho = rank_corrs[c]
        print(f"  Cadence {c:>2}: ρ = {rho:.4f}")

    # Run cadence analysis for additional tasks
    extra_rank_corrs = {}

    results_pm = load_results(args.results_dir / "main_phase_mapping.json")
    if results_pm:
        print(f"\nLoaded {len(results_pm)} phase mapping entries")
        pm_analysis = analyze_cadence_sensitivity(results_pm, cadences=cadences, task="phase_mapping")
        if pm_analysis["rank_correlations"]:
            extra_rank_corrs["Phase Mapping"] = pm_analysis["rank_correlations"]
            print("Phase Mapping Spearman ρ:", {c: f"{r:.4f}" for c, r in pm_analysis["rank_correlations"].items()})

    results_tgt = load_results(args.results_dir / "main_targeted.json")
    if results_tgt:
        print(f"\nLoaded {len(results_tgt)} targeted entries")
        tgt_analysis = analyze_cadence_sensitivity(results_tgt, cadences=cadences, task="targeted")
        if tgt_analysis["rank_correlations"]:
            extra_rank_corrs["Targeted"] = tgt_analysis["rank_correlations"]
            print("Targeted Spearman ρ:", {c: f"{r:.4f}" for c, r in tgt_analysis["rank_correlations"].items()})

    # Plot
    plot_cadence_figure(analysis, args.output, extra_rank_corrs=extra_rank_corrs)


if __name__ == "__main__":
    main()
