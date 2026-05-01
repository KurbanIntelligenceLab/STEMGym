"""Tests for DEC-AUC alternative-metric ablation analysis."""

from __future__ import annotations

import numpy as np
import pytest

from building_utils.analysis.metric_alternatives import (
    final_accuracy,
    time_to_threshold,
    early_weighted_auc,
    late_weighted_auc,
    half_budget_auc,
    compute_all_metrics,
    METRIC_NAMES,
)


class TestFinalAccuracy:
    def test_returns_last_score(self):
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        dose_fractions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        assert final_accuracy(scores, dose_fractions) == pytest.approx(0.9)

    def test_single_point(self):
        scores = np.array([0.42])
        dose_fractions = np.array([1.0])
        assert final_accuracy(scores, dose_fractions) == pytest.approx(0.42)

    def test_empty_returns_zero(self):
        assert final_accuracy(np.array([]), np.array([])) == 0.0


class TestTimeToThreshold:
    def test_reaches_threshold_early(self):
        scores = np.array([0.0, 0.85, 0.85, 0.85, 0.85])
        dose_fractions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        # First crossing at 0.25
        assert time_to_threshold(scores, dose_fractions, threshold=0.8) == pytest.approx(0.25)

    def test_never_reaches(self):
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        dose_fractions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        # Returns 1.0 (full budget) when threshold is never hit
        assert time_to_threshold(scores, dose_fractions, threshold=0.8) == 1.0

    def test_reaches_at_end(self):
        scores = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
        dose_fractions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        assert time_to_threshold(scores, dose_fractions, threshold=0.8) == pytest.approx(1.0)

    def test_empty_returns_one(self):
        assert time_to_threshold(np.array([]), np.array([]), threshold=0.8) == 1.0


class TestWeightedAUCs:
    """Reference: a constant-1 score gives all variants exactly 1.0
    because each weighting integrates to 1 over [0, 1].
    """

    def test_constant_one_yields_one_for_all(self):
        f = np.linspace(0, 1, 51)
        s = np.ones_like(f)
        assert early_weighted_auc(s, f) == pytest.approx(1.0, abs=1e-3)
        assert late_weighted_auc(s, f) == pytest.approx(1.0, abs=1e-3)
        assert half_budget_auc(s, f) == pytest.approx(1.0, abs=1e-3)

    def test_constant_zero_yields_zero(self):
        f = np.linspace(0, 1, 51)
        s = np.zeros_like(f)
        assert early_weighted_auc(s, f) == 0.0
        assert late_weighted_auc(s, f) == 0.0
        assert half_budget_auc(s, f) == 0.0

    def test_step_function_at_half_distinguishes_variants(self):
        # Score is 0 for f < 0.5, 1 for f >= 0.5.
        f = np.linspace(0, 1, 51)
        s = (f >= 0.5).astype(float)
        # Late-weighted picks up most of the weight (rising weight on the
        # high-score half). Early-weighted picks up little (rising weight
        # on the zero half). Half-budget AUC is exactly 0 (no score in
        # the first half).
        assert late_weighted_auc(s, f) > early_weighted_auc(s, f)
        assert half_budget_auc(s, f) == pytest.approx(0.0, abs=0.05)

    def test_step_function_at_quarter_distinguishes_variants(self):
        # Score is 0 for f < 0.25, 1 for f >= 0.25.
        f = np.linspace(0, 1, 51)
        s = (f >= 0.25).astype(float)
        # Now early-weighted picks up substantial weight too because the
        # score rises early.
        assert early_weighted_auc(s, f) > 0.5
        assert late_weighted_auc(s, f) > 0.5

    def test_empty_returns_zero(self):
        empty = np.array([])
        assert early_weighted_auc(empty, empty) == 0.0
        assert late_weighted_auc(empty, empty) == 0.0
        assert half_budget_auc(empty, empty) == 0.0


class TestComputeAllMetrics:
    def test_returns_dict_with_six_keys(self):
        f = np.linspace(0, 1, 51)
        s = np.ones_like(f)
        result = compute_all_metrics(s, f)
        assert set(result.keys()) == set(METRIC_NAMES)
        assert len(METRIC_NAMES) == 6

    def test_dec_auc_matches_trapezoidal(self):
        f = np.linspace(0, 1, 51)
        s = f.copy()  # Linear ramp; integral = 0.5
        result = compute_all_metrics(s, f)
        assert result["dec_auc"] == pytest.approx(0.5, abs=1e-3)
        assert result["final_accuracy"] == pytest.approx(1.0)
        assert result["half_budget_auc"] == pytest.approx(0.25, abs=1e-3)

    def test_handles_empty_input(self):
        empty = np.array([])
        result = compute_all_metrics(empty, empty)
        assert result["dec_auc"] == 0.0
        assert result["final_accuracy"] == 0.0
        assert result["time_to_threshold"] == 1.0
