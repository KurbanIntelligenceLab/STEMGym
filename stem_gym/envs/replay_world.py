"""ReplayWorld: World backed by real experimental STEM data.

Same interface as World, but with modifications for real data:
- No Poisson noise (real data already has experimental noise)
- Ground truth derived from automated analysis (may be approximate)
- Metadata tracks provenance (source dataset, processing params)

Replay worlds are created from py4DSTEM 4D-STEM datasets or
pre-existing HAADF images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from stem_gym.envs.world import World


class ReplayWorld(World):
    """World backed by real experimental STEM data.

    Identical to World except:
    - apply_poisson_noise returns the tile unmodified (real noise is baked in)
    - is_replay property returns True
    - source_info provides provenance metadata
    """

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
        """No-op for replay worlds — real data already has experimental noise.

        Returns a copy of the tile (same convention as base class).
        """
        return tile.copy()
