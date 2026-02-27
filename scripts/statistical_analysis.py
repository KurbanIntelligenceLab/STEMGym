#!/usr/bin/env python3
"""Statistical analysis for STEMGym benchmark results.

Produces:
  - Bootstrap 95% confidence intervals on mean DEC-AUC
  - Paired Wilcoxon signed-rank tests for key agent comparisons
  - LaTeX table output for the paper

Usage:
    python scripts/statistical_analysis.py
    python scripts/statistical_analysis.py --results-dir results/ --output paper/figures/table_statistics.tex
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT = REPO / "paper" / "figures"


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


def collect_episode_aucs(
    results: list[dict], agent: str, task: str | None = None,
) -> dict[tuple[str, int], float]:
    """Collect per-episode DEC-AUC keyed by (world, seed).

    Returns dict mapping (world, seed) -> dec_auc.
    """
    aucs: dict[tuple[str, int], float] = {}
    for r in results:
        if r["agent"] != agent:
            continue
        if task is not None and r.get("task") != task:
            continue
        world = r["world"]
        for ep in r.get("episodes", []):
            seed = ep.get("seed", 0)
            auc = ep.get("dec_auc")
            if auc is not None:
                aucs[(world, seed)] = auc
    return aucs


def collect_paired_aucs(
    results: list[dict],
    agent_a: str,
    agent_b: str,
    task: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect paired DEC-AUC values for two agents on matched (world, seed) pairs.

    Returns (aucs_a, aucs_b) arrays of equal length.
    """
    aucs_a = collect_episode_aucs(results, agent_a, task)
    aucs_b = collect_episode_aucs(results, agent_b, task)
    common_keys = sorted(set(aucs_a.keys()) & set(aucs_b.keys()))
    if not common_keys:
        return np.array([]), np.array([])
    vals_a = np.array([aucs_a[k] for k in common_keys])
    vals_b = np.array([aucs_b[k] for k in common_keys])
    return vals_a, vals_b


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval on the mean.

    Returns (mean, ci_lower, ci_upper).
    """
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(values)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()
    alpha = 1 - ci
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(values.mean()), float(lower), float(upper)


def paired_wilcoxon(
    values_a: np.ndarray, values_b: np.ndarray,
) -> tuple[float, float, int, str]:
    """Paired Wilcoxon signed-rank test.

    Returns (statistic, p_value, n_pairs, direction).
    direction is 'A > B', 'B > A', or 'A ≈ B'.
    """
    if len(values_a) < 6:
        return float("nan"), float("nan"), len(values_a), "too few pairs"
    diff = values_a - values_b
    # Remove zero differences (ties)
    nonzero = diff[diff != 0]
    if len(nonzero) < 6:
        return float("nan"), float("nan"), len(values_a), "too few non-tied pairs"
    try:
        stat, p = stats.wilcoxon(nonzero, alternative="two-sided")
    except ValueError:
        return float("nan"), float("nan"), len(values_a), "test failed"
    mean_diff = diff.mean()
    if p < 0.05:
        direction = "A > B" if mean_diff > 0 else "B > A"
    else:
        direction = "A ≈ B"
    return float(stat), float(p), len(values_a), direction


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def run_comparisons(results_dir: Path) -> list[dict]:
    """Run all key comparisons and return structured results."""
    # Load data sources
    main_dc = load_results(results_dir / "main_defect_census.json")
    ablation = load_results(results_dir / "trained_ablation_study.json")

    comparisons = [
        # From main_defect_census: perception gap
        {
            "name": "Raster vs Raster+CNN",
            "hypothesis": "Perception gap (central claim)",
            "agent_a": "raster_equipped",
            "agent_b": "raster",
            "task": "defect_census",
            "source": main_dc,
            "source_name": "main_defect_census",
        },
        # Navigation effect
        {
            "name": "Raster+CNN vs GP-UCB+CNN",
            "hypothesis": "Navigation adds little",
            "agent_a": "raster_equipped",
            "agent_b": "gp_ucb_equipped",
            "task": "defect_census",
            "source": main_dc,
            "source_name": "main_defect_census",
        },
        # Full system benefit
        {
            "name": "Raster+CNN vs STEMAgent",
            "hypothesis": "Full system vs equipped baseline",
            "agent_a": "stem_agent",
            "agent_b": "raster_equipped",
            "task": "defect_census",
            "source": main_dc,
            "source_name": "main_defect_census",
        },
        # Ablation: uncertainty ensemble
        {
            "name": "STEMAgent vs No Uncertainty",
            "hypothesis": "Uncertainty ensemble contribution",
            "agent_a": "stem_agent",
            "agent_b": "no_uncertainty",
            "task": "defect_census",
            "source": ablation,
            "source_name": "trained_ablation_study",
        },
        # Ablation: planner
        {
            "name": "STEMAgent vs No Planner",
            "hypothesis": "Planner contribution",
            "agent_a": "stem_agent",
            "agent_b": "no_planner",
            "task": "defect_census",
            "source": ablation,
            "source_name": "trained_ablation_study",
        },
    ]

    results = []
    for comp in comparisons:
        vals_a, vals_b = collect_paired_aucs(
            comp["source"], comp["agent_a"], comp["agent_b"], comp["task"],
        )
        mean_a, lo_a, hi_a = bootstrap_ci(vals_a)
        mean_b, lo_b, hi_b = bootstrap_ci(vals_b)
        stat, p, n_pairs, direction = paired_wilcoxon(vals_a, vals_b)

        result = {
            "name": comp["name"],
            "hypothesis": comp["hypothesis"],
            "source": comp["source_name"],
            "task": comp["task"],
            "agent_a": comp["agent_a"],
            "agent_b": comp["agent_b"],
            "n_pairs": n_pairs,
            "mean_a": mean_a,
            "ci_a": (lo_a, hi_a),
            "mean_b": mean_b,
            "ci_b": (lo_b, hi_b),
            "mean_diff": mean_a - mean_b,
            "wilcoxon_stat": stat,
            "p_value": p,
            "direction": direction,
        }
        results.append(result)

    return results


def run_per_agent_bootstrap(results_dir: Path) -> list[dict]:
    """Compute bootstrap CIs for each agent on defect census."""
    main_dc = load_results(results_dir / "main_defect_census.json")
    agents = [
        "random", "raster", "gp_ucb",
        "raster_equipped", "gp_ucb_equipped", "stem_agent",
    ]

    rows = []
    for agent in agents:
        aucs = collect_episode_aucs(main_dc, agent, "defect_census")
        if not aucs:
            continue
        values = np.array(list(aucs.values()))
        mean, lo, hi = bootstrap_ci(values)
        rows.append({
            "agent": agent,
            "n_episodes": len(values),
            "mean": mean,
            "ci_lower": lo,
            "ci_upper": hi,
        })
    return rows


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_p_value(p: float) -> str:
    """Format p-value for display."""
    if np.isnan(p):
        return "---"
    if p < 0.001:
        return f"$< 0.001$"
    if p < 0.01:
        return f"${p:.3f}$"
    return f"${p:.3f}$"


def format_ci(mean: float, lo: float, hi: float) -> str:
    """Format mean [CI] for display."""
    return f"${mean:.3f}$ [{lo:.3f}, {hi:.3f}]"


def print_comparison_table(results: list[dict]) -> None:
    """Print comparison results in a readable table."""
    print("\n" + "=" * 90)
    print("PAIRED WILCOXON SIGNED-RANK TESTS")
    print("=" * 90)
    print(f"{'Comparison':<30} {'N':>4} {'Mean A':>8} {'Mean B':>8} {'Diff':>8} {'p-value':>10} {'Direction'}")
    print("-" * 90)
    for r in results:
        p_str = f"{r['p_value']:.4f}" if not np.isnan(r["p_value"]) else "---"
        print(
            f"{r['name']:<30} {r['n_pairs']:>4} "
            f"{r['mean_a']:>8.3f} {r['mean_b']:>8.3f} "
            f"{r['mean_diff']:>+8.3f} {p_str:>10} {r['direction']}"
        )
    print()


def print_bootstrap_table(rows: list[dict]) -> None:
    """Print per-agent bootstrap CI table."""
    print("\n" + "=" * 70)
    print("BOOTSTRAP 95% CONFIDENCE INTERVALS (Defect Census)")
    print("=" * 70)
    print(f"{'Agent':<20} {'N':>4} {'Mean':>8} {'95% CI':>20}")
    print("-" * 70)
    for r in rows:
        ci_str = f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
        print(f"{r['agent']:<20} {r['n_episodes']:>4} {r['mean']:>8.3f} {ci_str:>20}")
    print()


def generate_latex_table(
    comparison_results: list[dict],
    bootstrap_rows: list[dict],
    output_path: Path | None = None,
) -> str:
    """Generate LaTeX table for the paper."""
    lines = []

    # Part 1: Per-agent bootstrap CIs
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Statistical analysis of benchmark results. Top: bootstrap 95\% confidence intervals on mean DEC-AUC (defect census). Bottom: paired Wilcoxon signed-rank tests on matched (world, seed) episodes.}")
    lines.append(r"\label{tab:statistics}")
    lines.append(r"\begin{tabular}{lrcc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Agent} & \textbf{$N$} & \textbf{Mean DEC-AUC} & \textbf{95\% CI} \\")
    lines.append(r"\midrule")

    for r in bootstrap_rows:
        agent_label = {
            "random": "Random", "raster": "Raster", "gp_ucb": "GP-UCB",
            "raster_equipped": "Raster+CNN", "gp_ucb_equipped": "GP-UCB+CNN",
            "stem_agent": "STEMAgent",
        }.get(r["agent"], r["agent"])
        lines.append(
            f"{agent_label} & {r['n_episodes']} & "
            f"${r['mean']:.3f}$ & $[{r['ci_lower']:.3f},\\, {r['ci_upper']:.3f}]$ \\\\"
        )

    lines.append(r"\botrule")
    lines.append(r"\end{tabular}")
    lines.append("")
    lines.append(r"\vspace{1em}")
    lines.append("")

    # Part 2: Paired comparisons
    lines.append(r"\begin{tabular}{lrccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Comparison} & \textbf{$N$} & \textbf{$\Delta$ DEC-AUC} & \textbf{$p$-value} & \textbf{Sig.} \\")
    lines.append(r"\midrule")

    for r in comparison_results:
        p = r["p_value"]
        if np.isnan(p):
            p_str = "---"
            sig = ""
        elif p < 0.001:
            p_str = "$< 0.001$"
            sig = "***"
        elif p < 0.01:
            p_str = f"${p:.3f}$"
            sig = "**"
        elif p < 0.05:
            p_str = f"${p:.3f}$"
            sig = "*"
        else:
            p_str = f"${p:.3f}$"
            sig = ""

        lines.append(
            f"{r['name']} & {r['n_pairs']} & "
            f"${r['mean_diff']:+.3f}$ & {p_str} & {sig} \\\\"
        )

    lines.append(r"\botrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    table = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(table)
        print(f"  Saved LaTeX table to {output_path}")

    return table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Statistical analysis of STEMGym results")
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path for LaTeX table (default: paper/figures/table_statistics.tex)")
    args = parser.parse_args()

    output = args.output or OUT / "table_statistics.tex"

    print("STEMGym Statistical Analysis")
    print(f"Results dir: {args.results_dir}")
    print()

    # Run analyses
    comparison_results = run_comparisons(args.results_dir)
    bootstrap_rows = run_per_agent_bootstrap(args.results_dir)

    # Print results
    print_bootstrap_table(bootstrap_rows)
    print_comparison_table(comparison_results)

    # Generate LaTeX
    print("\n--- LaTeX Table ---")
    table = generate_latex_table(comparison_results, bootstrap_rows, output)
    print(table)


if __name__ == "__main__":
    main()
