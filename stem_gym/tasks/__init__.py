from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


class Task(ABC):


    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this task."""

    @abstractmethod
    def score(self, findings: dict[str, Any], ground_truth: dict[str, Any]) -> float:
        """Score agent findings against ground truth."""


@dataclass
class DummyTask(Task):


    @property
    def name(self) -> str:
        return "dummy"

    def score(self, findings: dict[str, Any], ground_truth: dict[str, Any]) -> float:
        return 0.0
