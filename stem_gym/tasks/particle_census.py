"""Particle Census task: detection F1 + sizing + morphology classification.

Designed for the Pt nanoparticle scenario where the agent must detect
discrete particles, estimate their diameters, and classify morphology
(single_crystal / icosahedral / decahedral).

Scoring formula (from Pt generator docstring):
    Detection F1:  Greedy nearest-neighbour matching, r_match = 1 nm
    Sizing score:  exp(-W1 / <d_true>)  via scipy.stats.wasserstein_distance
    Morphology F1: Macro-averaged F1 over {single_crystal, icosahedral, decahedral}
    Overall:       0.5 * F1_det + 0.3 * S_size + 0.2 * F1_morph
"""

from __future__ import annotations

from typing import Any

import numpy as np

from stem_gym.tasks import Task

MORPH_NAMES = {0: "single_crystal", 1: "icosahedral", 2: "decahedral"}
MORPH_CLASSES = ["single_crystal", "icosahedral", "decahedral"]


class ParticleCensusTask(Task):
    """Score agent's particle census against ground truth.

    Combines three sub-scores:
    - Detection F1 (50%): greedy nearest-neighbour matching of centroids
    - Sizing score (30%): Wasserstein distance between diameter distributions
    - Morphology F1 (20%): macro-averaged F1 over 3 morphology classes

    Gracefully degrades when agents don't report diameters or morphology.
    """

    def __init__(self, match_radius_nm: float = 1.0):
        self.match_radius_nm = match_radius_nm

    @property
    def name(self) -> str:
        return "particle_census"

    def _detection_f1(
        self,
        pred_positions: list[dict],
        gt_positions: np.ndarray,
    ) -> tuple[float, list[int]]:
        """Greedy nearest-neighbour detection F1.

        Returns (f1, matched_gt_indices) where matched_gt_indices[i] is the
        GT index matched to pred_positions[i], or -1 if unmatched.
        """
        if len(gt_positions) == 0 and len(pred_positions) == 0:
            return 1.0, []
        if len(gt_positions) == 0 or len(pred_positions) == 0:
            return 0.0, [-1] * len(pred_positions)

        # Build distance matrix and greedily match
        pred_pts = np.array([p["position"] for p in pred_positions], dtype=np.float64)
        gt_pts = gt_positions.astype(np.float64)

        # Pairwise distances: (n_pred, n_gt)
        diffs = pred_pts[:, None, :] - gt_pts[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)

        matched_gt = set()
        matched_indices: list[int] = [-1] * len(pred_positions)
        hits = 0

        # Greedy: sort all (pred, gt) pairs by distance, match closest first
        flat_order = np.argsort(dists, axis=None)
        for flat_idx in flat_order:
            pi = int(flat_idx // dists.shape[1])
            gi = int(flat_idx % dists.shape[1])
            if dists[pi, gi] > self.match_radius_nm:
                break  # all remaining are too far
            if matched_indices[pi] != -1 or gi in matched_gt:
                continue
            matched_indices[pi] = gi
            matched_gt.add(gi)
            hits += 1

        precision = hits / len(pred_positions)
        recall = hits / len(gt_positions)
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        return f1, matched_indices

    def _sizing_score(
        self,
        pred_diameters: list[float],
        gt_diameters: np.ndarray,
    ) -> float:
        """Sizing score: exp(-W1 / mean(gt_diameters))."""
        if len(pred_diameters) == 0 or len(gt_diameters) == 0:
            return 0.0

        from scipy.stats import wasserstein_distance

        w1 = wasserstein_distance(gt_diameters, pred_diameters)
        mean_d = float(np.mean(gt_diameters))
        if mean_d <= 0:
            return 0.0
        return float(np.exp(-w1 / mean_d))

    def _morphology_f1(
        self,
        pred_labels: list[str],
        gt_labels: list[str],
    ) -> float:
        """Macro-averaged F1 over morphology classes."""
        if len(pred_labels) == 0 or len(gt_labels) == 0:
            return 0.0

        f1_scores = []
        for cls in MORPH_CLASSES:
            tp = sum(1 for p, g in zip(pred_labels, gt_labels) if p == cls and g == cls)
            fp = sum(1 for p, g in zip(pred_labels, gt_labels) if p == cls and g != cls)
            fn = sum(1 for p, g in zip(pred_labels, gt_labels) if p != cls and g == cls)

            if tp + fp == 0 and tp + fn == 0:
                # Class absent from both pred and gt in matched set
                continue
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision + recall > 0:
                f1_scores.append(2 * precision * recall / (precision + recall))
            else:
                f1_scores.append(0.0)

        return float(np.mean(f1_scores)) if f1_scores else 0.0

    def score(self, findings: dict[str, Any], ground_truth: dict[str, Any]) -> float:
        # --- Ground truth ---
        gt_positions = ground_truth.get("atom_positions", np.array([]))
        gt_diameters = ground_truth.get("particle_diameters_nm", np.array([]))
        gt_morph_labels = ground_truth.get("particle_morphology_labels", np.array([]))

        # --- Predictions ---
        pred_positions = findings.get("defect_positions", [])

        if len(gt_positions) == 0 and len(pred_positions) == 0:
            return 1.0
        if len(gt_positions) == 0 or len(pred_positions) == 0:
            return 0.0

        # 1. Detection F1
        f1_det, matched_indices = self._detection_f1(pred_positions, gt_positions)

        # 2. Sizing score
        pred_diameters = findings.get("particle_diameters", [])
        s_size = self._sizing_score(pred_diameters, gt_diameters)

        # 3. Morphology F1 (only over matched pairs)
        f1_morph = 0.0
        if len(gt_morph_labels) > 0:
            pred_morph: list[str] = []
            gt_morph: list[str] = []
            for pi, gi in enumerate(matched_indices):
                if gi < 0:
                    continue
                pred_type = pred_positions[pi].get("type", "unknown")
                gt_label = int(gt_morph_labels[gi])
                gt_name = MORPH_NAMES.get(gt_label, "unknown")
                pred_morph.append(pred_type)
                gt_morph.append(gt_name)

            f1_morph = self._morphology_f1(pred_morph, gt_morph)

        return 0.5 * f1_det + 0.3 * s_size + 0.2 * f1_morph
