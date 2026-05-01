"""Tests for DEC-AUC alternative-metric ablation analysis."""

from __future__ import annotations

import numpy as np
import pytest

from building_utils.analysis.metric_alternatives import (
    final_accuracy,
    time_to_threshold,
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
