"""Tests for DEC-AUC alternative-metric ablation analysis."""

from __future__ import annotations

import numpy as np
import pytest

from building_utils.analysis.metric_alternatives import (
    final_accuracy,
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
