#!/usr/bin/env python3
"""Compute alternative scoring metrics from saved STEMGym episode data and
quantify ranking stability against DEC-AUC.

Reads canonical `results/*.json` files locally, computes five alternative
metrics from each episode's `dec_dose_fractions` / `dec_scores` arrays,
aggregates per (agent, task) across worlds and seeds, and reports
Spearman/Kendall rank correlations between DEC-AUC and each alternative.

Usage:
    python -m building_utils.analysis.metric_alternatives
    python -m building_utils.analysis.metric_alternatives \
        --results-dir results/ \
        --json-output results/metric_alternatives.json \
        --latex-output building_utils/analysis/metric_alternatives_table.tex
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import stats


def final_accuracy(scores: np.ndarray, dose_fractions: np.ndarray) -> float:
    """Score at the final acquisition (s(f=1))."""
    if len(scores) == 0:
        return 0.0
    return float(scores[-1])


def time_to_threshold(
    scores: np.ndarray,
    dose_fractions: np.ndarray,
    threshold: float = 0.8,
) -> float:
    """Dose fraction at which `scores` first reaches `threshold`. Returns 1.0
    (worst case: full budget) if the threshold is never reached.

    Order-discontinuous: an agent that reaches 0.799 then 0.801 jumps from
    1.0 to ~the corresponding dose fraction. This discontinuity is part of
    why DEC-AUC is preferred for stable rankings, and quantifying it is the
    point of this analysis.
    """
    if len(scores) == 0:
        return 1.0
    above = np.where(scores >= threshold)[0]
    if len(above) == 0:
        return 1.0
    return float(dose_fractions[above[0]])


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration with numpy version compatibility (np.trapezoid in 2.x, np.trapz in 1.x)."""
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    if fn is None:
        raise RuntimeError("numpy has neither trapezoid nor trapz; numpy >= 1.24 is required")
    return float(fn(y, x))


def early_weighted_auc(scores: np.ndarray, dose_fractions: np.ndarray) -> float:
    """Trapezoidal integral of s(f) * w_early(f) where w_early(f) = 2(1-f).

    The weight ramps linearly from 2 at f=0 to 0 at f=1 and integrates to
    1 over [0, 1], so a constant-1 score yields exactly 1.0. Privileges
    front-loaded acquisition strategies that reach high scores early.
    """
    if len(scores) == 0:
        return 0.0
    weights = 2.0 * (1.0 - dose_fractions)
    return _trapz(scores * weights, dose_fractions)


def late_weighted_auc(scores: np.ndarray, dose_fractions: np.ndarray) -> float:
    """Trapezoidal integral of s(f) * w_late(f) where w_late(f) = 2f.

    Mirror image of `early_weighted_auc`. Privileges asymptotic accuracy.
    """
    if len(scores) == 0:
        return 0.0
    weights = 2.0 * dose_fractions
    return _trapz(scores * weights, dose_fractions)


def half_budget_auc(scores: np.ndarray, dose_fractions: np.ndarray) -> float:
    """DEC-AUC restricted to the first half of dose, rescaled to [0, 1].

    Computed as (1/0.5) * integral from f=0 to f=0.5 of s(f). Tests
    whether the DEC-AUC ranking depends on the budget cap by truncating
    the curve at half budget. Returns NaN when the curve is supplied but
    has fewer than two points in the first half (under-defined truncation),
    so downstream rank correlation can drop the agent rather than treating
    it as a true zero score.
    """
    if len(scores) == 0:
        return 0.0
    mask = dose_fractions <= 0.5
    if mask.sum() < 2:
        return float("nan")
    f_trunc = dose_fractions[mask]
    s_trunc = scores[mask]
    return _trapz(s_trunc, f_trunc) / 0.5


METRIC_NAMES = (
    "dec_auc",
    "final_accuracy",
    "time_to_threshold",
    "early_weighted_auc",
    "late_weighted_auc",
    "half_budget_auc",
)


def dec_auc_from_curve(scores: np.ndarray, dose_fractions: np.ndarray) -> float:
    """Standard trapezoidal DEC-AUC (matches stem_gym.metrics.compute_dec)."""
    if len(scores) == 0:
        return 0.0
    return _trapz(scores, dose_fractions)


def compute_all_metrics(
    scores: np.ndarray,
    dose_fractions: np.ndarray,
) -> dict[str, float]:
    """Compute all six metrics on a single dose-score curve.

    Returns a dict keyed by METRIC_NAMES. Each value is a float in [0, 1]
    (or [0, 1] for time_to_threshold where 1.0 means threshold never
    reached).
    """
    return {
        "dec_auc": dec_auc_from_curve(scores, dose_fractions),
        "final_accuracy": final_accuracy(scores, dose_fractions),
        "time_to_threshold": time_to_threshold(scores, dose_fractions),
        "early_weighted_auc": early_weighted_auc(scores, dose_fractions),
        "late_weighted_auc": late_weighted_auc(scores, dose_fractions),
        "half_budget_auc": half_budget_auc(scores, dose_fractions),
    }


def aggregate_results(
    results: list[dict],
) -> dict[tuple[str, str], dict[str, float]]:
    """Aggregate per-episode metrics across worlds and seeds, grouped by
    (agent, task).

    Args:
        results: list of combo dicts as found in `results/*.json` under the
            top-level `"results"` key. Each combo has `agent`, `task`,
            `world`, and `episodes` fields. Each episode has at minimum
            `seed`, `dec_auc`, `final_score`, and (preferably)
            `dec_dose_fractions` + `dec_scores` arrays for trajectory
            metrics.

    Returns:
        dict mapping (agent, task) -> {metric_name: mean value across all
        episodes for that pair}. Trajectory-dependent metrics fall back to
        NaN for episodes missing `dec_dose_fractions`/`dec_scores`.
    """
    per_pair = defaultdict(lambda: defaultdict(list))
    for combo in results:
        agent = combo.get("agent")
        task = combo.get("task")
        episodes = combo.get("episodes") or []
        if not episodes:
            continue
        for ep in episodes:
            f = ep.get("dec_dose_fractions")
            s = ep.get("dec_scores")
            if f is not None and s is not None and len(s) > 0:
                arr_f = np.asarray(f, dtype=float)
                arr_s = np.asarray(s, dtype=float)
                metrics = compute_all_metrics(arr_s, arr_f)
                # Prefer authoritative top-level fields when present so we
                # round-trip exactly with the runner's own computation.
                if "dec_auc" in ep:
                    metrics["dec_auc"] = float(ep["dec_auc"])
                if "final_score" in ep:
                    metrics["final_accuracy"] = float(ep["final_score"])
            else:
                # Fall back to top-level fields, NaN for trajectory metrics
                metrics = {name: float("nan") for name in METRIC_NAMES}
                metrics["dec_auc"] = float(ep.get("dec_auc", 0.0))
                metrics["final_accuracy"] = float(ep.get("final_score", 0.0))
            for name, value in metrics.items():
                per_pair[(agent, task)][name].append(value)

    # Reduce: nanmean per metric per (agent, task)
    out: dict[tuple[str, str], dict[str, float]] = {}
    for key, metric_lists in per_pair.items():
        out[key] = {
            name: float(np.nanmean(values)) if len(values) > 0 and not all(np.isnan(values)) else float("nan")
            for name, values in metric_lists.items()
        }
    return out


def rank_correlation(
    agg: dict[tuple[str, str], dict[str, float]],
    task: str,
) -> dict[str, dict[str, float]] | None:
    """Per-task: compute Spearman ρ and Kendall τ between DEC-AUC ranking
    and each alternative-metric ranking. Returns None if fewer than two
    agents are present for the given task.

    For `time_to_threshold`, lower values mean faster convergence (better),
    so the metric is negated before ranking to align with DEC-AUC's
    "higher is better" convention.

    Returns:
        dict mapping alt-metric-name -> {"spearman": ρ, "kendall": τ}.
        Excludes `dec_auc` itself (trivially 1.0).
    """
    pairs = [(agent, m["dec_auc"], m) for (agent, t), m in agg.items() if t == task]
    if len(pairs) < 2:
        return None

    dec_values = np.array([p[1] for p in pairs])

    out: dict[str, dict[str, float]] = {}
    for name in METRIC_NAMES:
        if name == "dec_auc":
            continue
        alt_values = np.array([p[2][name] for p in pairs])
        # time_to_threshold is "lower is better", invert for fair comparison
        if name == "time_to_threshold":
            alt_values = -alt_values
        # Drop pairs where either side is NaN
        mask = ~(np.isnan(dec_values) | np.isnan(alt_values))
        if mask.sum() < 2:
            out[name] = {"spearman": float("nan"), "kendall": float("nan")}
            continue
        sp, _ = stats.spearmanr(dec_values[mask], alt_values[mask])
        kt, _ = stats.kendalltau(dec_values[mask], alt_values[mask])
        out[name] = {"spearman": float(sp), "kendall": float(kt)}
    return out


def find_disagreements(
    agg: dict[tuple[str, str], dict[str, float]],
    task: str,
) -> list[dict]:
    """Identify agent pairs (a, b) where DEC-AUC ranks a > b but at least
    one alternative metric ranks b > a (or vice versa).

    Returns:
        list of dicts with keys {alt_metric, agent_a, agent_b,
        dec_auc_a, dec_auc_b, alt_a, alt_b}. Each record corresponds to
        a single (alt_metric, ordered pair) inversion.
    """
    agents = sorted({a for (a, t) in agg if t == task})
    out: list[dict] = []

    for name in METRIC_NAMES:
        if name == "dec_auc":
            continue
        for i, agent_a in enumerate(agents):
            for agent_b in agents[i + 1:]:
                m_a = agg.get((agent_a, task), {})
                m_b = agg.get((agent_b, task), {})
                if name not in m_a or name not in m_b:
                    continue
                dec_a, dec_b = m_a.get("dec_auc", 0.0), m_b.get("dec_auc", 0.0)
                alt_a, alt_b = m_a[name], m_b[name]
                if np.isnan(alt_a) or np.isnan(alt_b):
                    continue
                # Invert time_to_threshold so "higher is better" applies
                if name == "time_to_threshold":
                    alt_a, alt_b = -alt_a, -alt_b
                # Disagreement: DEC-AUC and alt have opposite orderings
                if (dec_a > dec_b and alt_a < alt_b) or (dec_a < dec_b and alt_a > alt_b):
                    out.append({
                        "alt_metric": name,
                        "agent_a": agent_a,
                        "agent_b": agent_b,
                        "dec_auc_a": dec_a,
                        "dec_auc_b": dec_b,
                        "alt_a": m_a[name],
                        "alt_b": m_b[name],
                    })
    return out
