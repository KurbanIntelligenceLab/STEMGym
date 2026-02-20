"""Defect Census task: F1 score for defect detection and classification.

Blends count-based F1 (how many of each type) with spatial F1 (did the
agent locate real defects) when spatial positions are available.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from stem_gym.tasks import Task


class DefectCensusTask(Task):
    """Score agent's defect census against ground truth.

    Primary metric: blended F1 combining count accuracy and spatial
    localization when defect positions are available.

    - Count F1: macro-averaged F1 on {type: count} vectors.
    - Spatial F1: for agents that report defect_positions, compute
      precision/recall based on proximity to ground truth defects.
    - Final score = max(count_f1, spatial_f1) to reward either approach.
    """

    def __init__(self, tolerance_nm: float = 5.0):
        self.tolerance_nm = tolerance_nm

    @property
    def name(self) -> str:
        return "defect_census"

    def _count_f1(
        self, pred_census: dict[str, int], gt_census: dict[str, int]
    ) -> float:
        """Macro-averaged F1 on defect type counts."""
        if len(gt_census) == 0 and len(pred_census) == 0:
            return 1.0

        all_types = set(gt_census.keys()) | set(pred_census.keys())
        f1_scores = []

        for defect_type in all_types:
            gt_count = gt_census.get(defect_type, 0)
            pred_count = pred_census.get(defect_type, 0)

            if gt_count == 0 and pred_count == 0:
                f1_scores.append(1.0)
                continue

            tp = min(gt_count, pred_count)
            fp = max(0, pred_count - gt_count)
            fn = max(0, gt_count - pred_count)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            if precision + recall > 0:
                f1_scores.append(2 * precision * recall / (precision + recall))
            else:
                f1_scores.append(0.0)

        return float(np.mean(f1_scores)) if f1_scores else 0.0

    def _spatial_f1(
        self,
        pred_positions: list[dict],
        gt_positions: np.ndarray,
        gt_defect_mask: np.ndarray,
    ) -> float:
        """F1 based on spatial proximity to ground truth defects."""
        gt_defect_pos = gt_positions[gt_defect_mask]
        if len(gt_defect_pos) == 0 or len(pred_positions) == 0:
            return 0.0

        # Match predicted → ground truth within tolerance
        hits = 0
        matched_gt = set()
        for pred in pred_positions:
            pos = np.array(pred["position"])
            distances = np.linalg.norm(gt_defect_pos - pos, axis=1)
            best_idx = int(np.argmin(distances))
            if distances[best_idx] < self.tolerance_nm and best_idx not in matched_gt:
                hits += 1
                matched_gt.add(best_idx)

        precision = hits / len(pred_positions) if len(pred_positions) > 0 else 0.0
        recall = hits / len(gt_defect_pos) if len(gt_defect_pos) > 0 else 0.0

        if precision + recall > 0:
            return 2 * precision * recall / (precision + recall)
        return 0.0

    def score(self, findings: dict[str, Any], ground_truth: dict[str, Any]) -> float:
        pred_census = findings.get("defect_census", {})
        gt_types = ground_truth.get("defect_types", np.array([]))

        if len(gt_types) == 0:
            return 1.0 if len(pred_census) == 0 else 0.0

        # Ground truth census
        gt_census: dict[str, int] = {}
        for t in gt_types:
            t_str = str(t)
            if t_str != "pristine":
                gt_census[t_str] = gt_census.get(t_str, 0) + 1

        # Count-based F1
        count_f1 = self._count_f1(pred_census, gt_census)

        # Spatial F1 (if agent provides positions)
        spatial_f1 = 0.0
        pred_positions = findings.get("defect_positions", [])
        gt_positions = ground_truth.get("atom_positions", np.array([]))
        gt_atom_types = ground_truth.get("atom_types", np.array([]))

        if len(pred_positions) > 0 and len(gt_positions) > 0 and len(gt_atom_types) > 0:
            gt_defect_mask = gt_atom_types > 0  # any non-pristine
            spatial_f1 = self._spatial_f1(pred_positions, gt_positions, gt_defect_mask)

        # Blend: reward agents that provide spatial accuracy
        # Use max so count-only agents aren't penalized, but spatial agents get a boost
        return max(count_f1, spatial_f1)
