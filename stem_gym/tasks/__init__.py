"""Task definitions for STEMGym."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


class Task(ABC):
    """Base class for STEMGym evaluation tasks.

    A task defines how agent findings are scored against ground truth.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this task."""

    @abstractmethod
    def score(self, findings: dict[str, Any], ground_truth: dict[str, Any]) -> float:
        """Score agent findings against ground truth.

        Args:
            findings: Agent's declared findings (from ExperimentState).
            ground_truth: Ground truth from the world file.

        Returns:
            Score in [0, 1]. Higher is better.
        """


@dataclass
class DummyTask(Task):
    """Placeholder task that always returns 0. Used for env testing."""

    @property
    def name(self) -> str:
        return "dummy"

    def score(self, findings: dict[str, Any], ground_truth: dict[str, Any]) -> float:
        return 0.0
