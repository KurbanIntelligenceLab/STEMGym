"""Composite Task: simultaneous defect census + phase mapping scoring.

Enables multi-objective DEC-AUC evaluation on worlds that have both
defect ground truth and a phase map (e.g. BaTiO3). Returns a weighted
scalar for DEC-AUC compatibility while storing per-objective component
scores for Pareto hypervolume analysis.
"""

from __future__ import annotations

from typing import Any

from stem_gym.tasks import Task
from stem_gym.tasks.defect_census import DefectCensusTask
from stem_gym.tasks.phase_mapping import PhaseMappingTask


class CompositeTask(Task):
    """Score both defect census AND phase mapping simultaneously.

    The scalar score is a weighted combination of the two sub-task scores,
    suitable for standard DEC-AUC computation. Component scores are stored
    in ``_last_scores`` for post-hoc Pareto analysis.

    Args:
        weight_a: Weight for defect census score.
        weight_b: Weight for phase mapping score.
    """

    def __init__(self, weight_a: float = 0.5, weight_b: float = 0.5):
        self.task_a = DefectCensusTask()
        self.task_b = PhaseMappingTask()
        self.weight_a = weight_a
        self.weight_b = weight_b
        self._last_scores: tuple[float, float] = (0.0, 0.0)

    @property
    def name(self) -> str:
        return "composite"

    def score(self, findings: dict[str, Any], ground_truth: dict[str, Any]) -> float:
        sa = self.task_a.score(findings, ground_truth)
        sb = self.task_b.score(findings, ground_truth)
        self._last_scores = (sa, sb)
        return self.weight_a * sa + self.weight_b * sb
