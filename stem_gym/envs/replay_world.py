from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from stem_gym.envs.world import World


class ReplayWorld(World):


    @property
    def is_replay(self) -> bool:
        return True

    @property
    def source_info(self) -> dict[str, Any]:
        """Provenance metadata for the replay dataset."""
        return {
            "scenario": self._metadata.get("scenario", "replay"),
            "source": self._metadata.get("source", "unknown"),
            "processing": self._metadata.get("processing", "unknown"),
        }

    def apply_poisson_noise(self, tile: np.ndarray, dose: float) -> np.ndarray:
        """No-op for replay worlds — real data already has experimental noise."""
        return tile.copy()
