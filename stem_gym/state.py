from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Finding:

    finding_type: str
    position: tuple[float, float]
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentState:


    world_name: str = ""
    world_shape: tuple[int, int] = (0, 0)
    pixel_size_nm: float = 0.0
    fov_nm: tuple[float, float] = (0.0, 0.0)


    total_budget: float = 1e6
    dose_used: float = 0.0
    step_count: int = 0
    max_steps: int = 200


    position: tuple[float, float] = (0.0, 0.0)


    visited_mask: np.ndarray | None = None

    dose_map: np.ndarray | None = None

    tiles_acquired: int = 0
    acquisitions: int = 0


    atom_positions: np.ndarray | None = None
    atom_confidences: np.ndarray | None = None
    defect_census: dict[str, int] = field(default_factory=dict)
    defect_positions: list[Finding] = field(default_factory=list)
    phase_map: np.ndarray | None = None
    confidence_map: np.ndarray | None = None


    findings: list[Finding] = field(default_factory=list)
    declared_phase_map: np.ndarray | None = None

    @property
    def budget_remaining(self) -> float:
        return max(0.0, self.total_budget - self.dose_used)

    @property
    def budget_fraction_used(self) -> float:
        if self.total_budget <= 0:
            return 1.0
        return self.dose_used / self.total_budget

    def to_findings_dict(self) -> dict[str, Any]:
        """Package findings for task scoring."""
        return {
            "defect_census": dict(self.defect_census),
            "defect_positions": [
                {"type": f.finding_type, "position": f.position,
                 "confidence": f.confidence, "metadata": f.metadata}
                for f in self.findings
            ],
            "phase_map": self.declared_phase_map,
            "atom_positions": self.atom_positions,
        }
