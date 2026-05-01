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

import numpy as np


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
    """Trapezoidal integration with numpy version compatibility."""
    fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
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
    the curve at half budget.
    """
    if len(scores) == 0:
        return 0.0
    mask = dose_fractions <= 0.5
    if mask.sum() < 2:
        return 0.0
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
