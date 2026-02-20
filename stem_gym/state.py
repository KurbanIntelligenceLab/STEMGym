"""ExperimentState: blackboard shared by all agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np


class PlannerMode(Enum):
    """FSM planner states."""
    SURVEY = auto()
    INVESTIGATE = auto()
    CHARACTERIZE = auto()
    CENSUS = auto()
    TERMINATE = auto()


@dataclass
class Finding:
    """A single declared finding from the agent."""
    finding_type: str          # e.g. "vacancy", "substitution", "phase_boundary"
    position: tuple[float, float]  # (x_nm, y_nm) in world coordinates
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransitionLog:
    """Record of a planner FSM transition."""
    step: int
    from_mode: PlannerMode
    to_mode: PlannerMode
    reason: str
    budget_remaining: float


@dataclass
class ExperimentState:
    """Blackboard dataclass shared by Navigator, Analyst, and Planner.

    This is the central state object that all agents read from and write to.
    It is NOT the Gymnasium observation — the env constructs obs from this +
    the world data.
    """

    # --- World metadata ---
    world_name: str = ""
    world_shape: tuple[int, int] = (0, 0)  # (H, W) in pixels
    pixel_size_nm: float = 0.0             # nm per pixel
    fov_nm: tuple[float, float] = (0.0, 0.0)  # (H_nm, W_nm)

    # --- Budget tracking ---
    total_budget: float = 1e6     # total electron dose budget (e⁻/Å²)
    dose_used: float = 0.0        # cumulative dose spent
    step_count: int = 0           # environment steps taken
    max_steps: int = 200          # episode step limit

    # --- Agent position ---
    position: tuple[float, float] = (0.0, 0.0)  # (x_nm, y_nm) current probe center

    # --- Coverage tracking ---
    # Boolean mask: which tiles have been acquired (on the tile grid)
    visited_mask: np.ndarray | None = None
    # Float mask: accumulated dose per tile position
    dose_map: np.ndarray | None = None
    # Number of unique tiles acquired
    tiles_acquired: int = 0

    # --- Analyst outputs ---
    atom_positions: np.ndarray | None = None    # (N, 2) detected atom positions
    atom_confidences: np.ndarray | None = None  # (N,) confidence per atom
    defect_census: dict[str, int] = field(default_factory=dict)  # type -> count
    defect_positions: list[Finding] = field(default_factory=list)
    phase_map: np.ndarray | None = None         # (H, W) phase labels
    confidence_map: np.ndarray | None = None    # (H, W) analyst confidence

    # --- Planner state ---
    mode: PlannerMode = PlannerMode.SURVEY
    planner_beta: float = 2.0               # GP-UCB exploration parameter
    transition_log: list[TransitionLog] = field(default_factory=list)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)

    # --- Agent findings (for task scoring) ---
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
