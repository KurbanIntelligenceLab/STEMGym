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
    aggregate_results,
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

    def test_half_budget_returns_nan_when_no_first_half_points(self):
        # Single point at f=1.0: mask.sum() < 2, so half-budget AUC is undefined.
        scores = np.array([0.5])
        dose_fractions = np.array([1.0])
        assert np.isnan(half_budget_auc(scores, dose_fractions))

    def test_half_budget_returns_nan_when_one_first_half_point(self):
        # Two points but only f=0.0 is in the first half; not enough to integrate.
        scores = np.array([0.5, 0.5])
        dose_fractions = np.array([0.0, 1.0])
        # mask = [True, False], mask.sum() = 1, returns NaN.
        assert np.isnan(half_budget_auc(scores, dose_fractions))


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


class TestAggregateResults:
    def _fake_combo(self, agent: str, task: str, world: str, episode_aucs: list[float]):
        """Build a minimal results-JSON-shaped combo entry."""
        episodes = []
        for i, auc in enumerate(episode_aucs):
            f = np.linspace(0, 1, 11).tolist()
            # Construct a curve whose AUC is approximately `auc`: linear
            # ramp from 0 to 2*auc (truncated at 1.0).
            scores = [min(1.0, 2.0 * auc * x) for x in f]
            episodes.append({
                "seed": i,
                "dec_auc": auc,
                "final_score": scores[-1],
                "dec_dose_fractions": f,
                "dec_scores": scores,
            })
        return {
            "agent": agent,
            "task": task,
            "world": world,
            "episodes": episodes,
        }

    def test_aggregates_two_agents_one_task(self):
        results = [
            self._fake_combo("agent_a", "defect_census", "world_1", [0.3, 0.4]),
            self._fake_combo("agent_a", "defect_census", "world_2", [0.5, 0.6]),
            self._fake_combo("agent_b", "defect_census", "world_1", [0.1, 0.2]),
            self._fake_combo("agent_b", "defect_census", "world_2", [0.2, 0.3]),
        ]
        agg = aggregate_results(results)
        assert ("agent_a", "defect_census") in agg
        assert ("agent_b", "defect_census") in agg
        # agent_a mean DEC-AUC = (0.3+0.4+0.5+0.6) / 4 = 0.45
        assert agg[("agent_a", "defect_census")]["dec_auc"] == pytest.approx(0.45, abs=1e-3)
        # agent_b mean DEC-AUC = (0.1+0.2+0.2+0.3) / 4 = 0.20
        assert agg[("agent_b", "defect_census")]["dec_auc"] == pytest.approx(0.20, abs=1e-3)
        # Returns dicts of all six metric means
        for key in METRIC_NAMES:
            assert key in agg[("agent_a", "defect_census")]

    def test_skips_combos_with_no_episodes(self):
        results = [
            {"agent": "agent_a", "task": "defect_census", "world": "w1", "episodes": []},
        ]
        agg = aggregate_results(results)
        assert agg == {}

    def test_handles_missing_curve_arrays(self):
        # Some legacy result files may not include dec_dose_fractions/dec_scores.
        # In that case we still record dec_auc and final_score from the top-level
        # episode fields.
        results = [{
            "agent": "agent_a", "task": "defect_census", "world": "w1",
            "episodes": [{"seed": 0, "dec_auc": 0.5, "final_score": 0.7}],
        }]
        agg = aggregate_results(results)
        # dec_auc and final_accuracy still computable; trajectory metrics fall back to NaN
        assert agg[("agent_a", "defect_census")]["dec_auc"] == pytest.approx(0.5)
        assert agg[("agent_a", "defect_census")]["final_accuracy"] == pytest.approx(0.7)
        # Trajectory-dependent metrics are NaN when arrays are missing
        assert np.isnan(agg[("agent_a", "defect_census")]["early_weighted_auc"])
