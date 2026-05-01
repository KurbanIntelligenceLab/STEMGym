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
