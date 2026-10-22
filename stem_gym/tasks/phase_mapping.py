from __future__ import annotations

from typing import Any

import numpy as np

from stem_gym.tasks import Task


class PhaseMappingTask(Task):


    @property
    def name(self) -> str:
        return "phase_mapping"

    def score(self, findings: dict[str, Any], ground_truth: dict[str, Any]) -> float:
        pred_map = findings.get("phase_map")
        gt_map = ground_truth.get("phase_map")

        if gt_map is None:
            return 1.0 if pred_map is None else 0.0
        if pred_map is None:
            return 0.0

        gt_map = np.asarray(gt_map)
        pred_map = np.asarray(pred_map)


        if pred_map.shape != gt_map.shape:
            from scipy.ndimage import zoom
            scale_h = gt_map.shape[0] / pred_map.shape[0]
            scale_w = gt_map.shape[1] / pred_map.shape[1]
            pred_map = zoom(pred_map, (scale_h, scale_w), order=0).astype(int)


        all_phases = set(np.unique(gt_map)) | set(np.unique(pred_map))
        ious = []

        for phase in all_phases:
            gt_mask = gt_map == phase
            pred_mask = pred_map == phase
            intersection = np.sum(gt_mask & pred_mask)
            union = np.sum(gt_mask | pred_mask)
            if union > 0:
                ious.append(intersection / union)
            else:
                ious.append(1.0)

        return float(np.mean(ious)) if ious else 0.0
