"""Tests for DEC metrics."""

from __future__ import annotations

import numpy as np
import pytest

from stem_gym.metrics import compute_dec, dec_auc


class TestDECCurve:
    def test_empty_log(self):
        result = compute_dec([], total_budget=1e4)
        assert result.auc == 0.0
        assert result.time_to_threshold is None

    def test_perfect_from_start(self):
        log = [{"dose_used": 0.0, "task_score": 1.0}]
        result = compute_dec(log, total_budget=1e4)
        assert result.auc == pytest.approx(1.0, abs=0.05)
        assert result.time_to_threshold is not None
        assert result.time_to_threshold == pytest.approx(0.0, abs=0.05)

    def test_monotonic_improvement(self):
        log = [
            {"dose_used": 0.0, "task_score": 0.0},
            {"dose_used": 5000, "task_score": 0.5},
            {"dose_used": 10000, "task_score": 1.0},
        ]
        result = compute_dec(log, total_budget=10000)
        assert 0.0 < result.auc < 1.0
        assert result.time_to_threshold is not None

    def test_auc_convenience(self):
        log = [
            {"dose_used": 0.0, "task_score": 0.0},
            {"dose_used": 10000, "task_score": 1.0},
        ]
        auc = dec_auc(log, total_budget=10000)
        assert 0.0 < auc < 1.0

    def test_no_scored_entries(self):
        log = [
            {"dose_used": 100.0},
            {"dose_used": 200.0},
        ]
        result = compute_dec(log, total_budget=1e4)
        assert result.auc == 0.0
