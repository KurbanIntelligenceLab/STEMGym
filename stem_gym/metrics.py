"""DEC (Dose-Efficiency Curve) metrics for STEMGym."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DECResult:
    """Result of a DEC computation."""
    dose_fractions: np.ndarray   # (K,) dose fraction checkpoints [0, 1]
    scores: np.ndarray           # (K,) task scores at each checkpoint
    auc: float                   # Area under the DEC curve (normalized to [0, 1])
    time_to_threshold: float | None  # dose fraction to reach threshold (None if never)


def compute_dec(
    episode_log: list[dict],
    total_budget: float,
    n_checkpoints: int = 50,
    threshold: float = 0.8,
) -> DECResult:
    """Compute the Dose-Efficiency Curve from an episode log.

    Args:
        episode_log: List of dicts with keys:
            - "dose_used": cumulative dose at this step
            - "task_score": task score at this step (can be None for intermediate steps)
        total_budget: Total dose budget for normalization.
        n_checkpoints: Number of evenly-spaced dose fraction checkpoints.
        threshold: Score threshold for time-to-threshold metric.

    Returns:
        DECResult with dose fractions, scores, AUC, and time-to-threshold.
    """
    if not episode_log:
        dose_fracs = np.linspace(0, 1, n_checkpoints)
        return DECResult(
            dose_fractions=dose_fracs,
            scores=np.zeros(n_checkpoints),
            auc=0.0,
            time_to_threshold=None,
        )

    # Extract (dose_fraction, score) pairs — only entries with scores
    scored_entries = []
    for entry in episode_log:
        score = entry.get("task_score")
        if score is not None:
            dose_frac = entry["dose_used"] / total_budget if total_budget > 0 else 1.0
            scored_entries.append((dose_frac, score))

    if not scored_entries:
        dose_fracs = np.linspace(0, 1, n_checkpoints)
        return DECResult(
            dose_fractions=dose_fracs,
            scores=np.zeros(n_checkpoints),
            auc=0.0,
            time_to_threshold=None,
        )

    # Sort by dose fraction
    scored_entries.sort(key=lambda x: x[0])
    dose_vals = np.array([e[0] for e in scored_entries])
    score_vals = np.array([e[1] for e in scored_entries])

    # Interpolate to uniform checkpoints
    dose_fracs = np.linspace(0, 1, n_checkpoints)
    scores = np.interp(dose_fracs, dose_vals, score_vals, left=0.0, right=score_vals[-1])

    # AUC via trapezoidal rule
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = float(_trapz(scores, dose_fracs))

    # Time to threshold
    above = np.where(scores >= threshold)[0]
    time_to_thresh = float(dose_fracs[above[0]]) if len(above) > 0 else None

    return DECResult(
        dose_fractions=dose_fracs,
        scores=scores,
        auc=auc,
        time_to_threshold=time_to_thresh,
    )


def dec_auc(episode_log: list[dict], total_budget: float) -> float:
    """Convenience: compute just the DEC-AUC scalar."""
    return compute_dec(episode_log, total_budget).auc


@dataclass
class ParetoHVResult:
    """Result of Pareto hypervolume computation over two objectives."""
    dose_fractions: np.ndarray  # (K,) dose fraction checkpoints
    hv_values: np.ndarray       # (K,) hypervolume at each checkpoint
    hv_auc: float               # AUC of HV-DEC curve
    scores_a: np.ndarray        # (K,) interpolated scores for objective A
    scores_b: np.ndarray        # (K,) interpolated scores for objective B


def compute_pareto_hypervolume(
    episode_log: list[dict],
    total_budget: float,
    n_checkpoints: int = 50,
    ref_point: tuple[float, float] = (0.0, 0.0),
) -> ParetoHVResult:
    """Compute 2D Pareto hypervolume DEC from composite task episode log.

    For a single trajectory, the hypervolume at each dose checkpoint is
    simply ``(score_a - ref_a) * (score_b - ref_b)`` since there is one
    point on the Pareto front. For multi-trajectory aggregation, compute
    the dominated hypervolume of the point set.

    Args:
        episode_log: Episode log entries with ``task_score_a`` and
            ``task_score_b`` fields (produced by CompositeTask).
        total_budget: Total dose budget for normalization.
        n_checkpoints: Number of uniform dose fraction checkpoints.
        ref_point: Reference point for hypervolume (lower-left).

    Returns:
        ParetoHVResult with HV values at each checkpoint and HV-DEC-AUC.
    """
    scored = []
    for entry in episode_log:
        sa = entry.get("task_score_a")
        sb = entry.get("task_score_b")
        if sa is not None and sb is not None:
            dose_frac = entry["dose_used"] / total_budget if total_budget > 0 else 1.0
            scored.append((dose_frac, sa, sb))

    dose_fracs = np.linspace(0, 1, n_checkpoints)

    if not scored:
        return ParetoHVResult(
            dose_fractions=dose_fracs,
            hv_values=np.zeros(n_checkpoints),
            hv_auc=0.0,
            scores_a=np.zeros(n_checkpoints),
            scores_b=np.zeros(n_checkpoints),
        )

    scored.sort(key=lambda x: x[0])
    d = np.array([s[0] for s in scored])
    a = np.array([s[1] for s in scored])
    b = np.array([s[2] for s in scored])

    scores_a = np.interp(dose_fracs, d, a, left=0.0, right=a[-1])
    scores_b = np.interp(dose_fracs, d, b, left=0.0, right=b[-1])

    # Hypervolume: area of rectangle from ref_point to (score_a, score_b)
    hv = np.maximum(scores_a - ref_point[0], 0.0) * np.maximum(scores_b - ref_point[1], 0.0)

    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    hv_auc = float(_trapz(hv, dose_fracs))

    return ParetoHVResult(
        dose_fractions=dose_fracs,
        hv_values=hv,
        hv_auc=hv_auc,
        scores_a=scores_a,
        scores_b=scores_b,
    )
