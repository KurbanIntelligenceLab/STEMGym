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
