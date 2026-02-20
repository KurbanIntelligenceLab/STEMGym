"""Targeted Characterization task: find a specific feature."""

from __future__ import annotations

from typing import Any

import numpy as np

from stem_gym.tasks import Task


class TargetedTask(Task):
    """Score whether the agent found the target feature.

    Primary metric: binary correctness + localization bonus.
    """

    def __init__(self, target_type: str = "target", tolerance_nm: float = 5.0):
        self.target_type = target_type
        self.tolerance_nm = tolerance_nm

    @property
    def name(self) -> str:
        return "targeted"

    def score(self, findings: dict[str, Any], ground_truth: dict[str, Any]) -> float:
        # Get target locations from ground truth
        gt_positions = ground_truth.get("atom_positions", np.array([]))
        gt_types = ground_truth.get("defect_types", np.array([]))

        if len(gt_positions) == 0 or len(gt_types) == 0:
            return 0.0

        # Find target atoms in ground truth
        target_mask = np.array([str(t) == self.target_type for t in gt_types])
        if not np.any(target_mask):
            # No exact match — try any non-pristine defect as target
            gt_atom_types = ground_truth.get("atom_types", np.array([]))
            # Try substitutions first, then vacancies
            target_mask = gt_atom_types == 2  # substitution
            if not np.any(target_mask):
                target_mask = gt_atom_types == 1  # vacancy
            if not np.any(target_mask):
                # Truly no defects in ground truth
                return 1.0 if len(findings.get("defect_positions", [])) == 0 else 0.0

        target_positions = gt_positions[target_mask]

        # Get agent's declared positions
        pred_positions = findings.get("defect_positions", [])
        if len(pred_positions) == 0:
            return 0.0

        # Check if any predicted position is close to a target
        hits = 0
        for pred in pred_positions:
            pos = np.array(pred["position"])
            distances = np.linalg.norm(target_positions - pos, axis=1)
            if np.min(distances) < self.tolerance_nm:
                hits += 1

        # Score: fraction of targets found (capped at 1.0)
        recall = min(1.0, hits / len(target_positions))
        # Penalize false positives lightly
        precision = hits / len(pred_positions) if len(pred_positions) > 0 else 0.0

        if precision + recall > 0:
            return 2 * precision * recall / (precision + recall)
        return 0.0
