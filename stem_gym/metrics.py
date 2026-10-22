from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DECResult:

    dose_fractions: np.ndarray
    scores: np.ndarray
    auc: float
    time_to_threshold: float | None


def compute_dec(
    episode_log: list[dict],
    total_budget: float,
    n_checkpoints: int = 50,
    threshold: float = 0.8,
) -> DECResult:
    """Compute the Dose-Efficiency Curve from an episode log."""
    if not episode_log:
        dose_fracs = np.linspace(0, 1, n_checkpoints)
        return DECResult(
            dose_fractions=dose_fracs,
            scores=np.zeros(n_checkpoints),
            auc=0.0,
            time_to_threshold=None,
        )


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


    scored_entries.sort(key=lambda x: x[0])
    dose_vals = np.array([e[0] for e in scored_entries])
    score_vals = np.array([e[1] for e in scored_entries])


    dose_fracs = np.linspace(0, 1, n_checkpoints)
    scores = np.interp(dose_fracs, dose_vals, score_vals, left=0.0, right=score_vals[-1])


    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = float(_trapz(scores, dose_fracs))


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
