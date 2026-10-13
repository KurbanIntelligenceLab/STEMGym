"""Coordinate transforms shared by environments and acquisition policies."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class WorldGeometry:
    grid_shape: tuple[int, int]
    pixel_size_nm: float
    tile_size_px: int
    stride_px: int

    def tile_to_nm(self, row: int, col: int) -> tuple[float, float]:
        if not (0 <= row < self.grid_shape[0] and 0 <= col < self.grid_shape[1]):
            raise ValueError("Tile coordinates outside the grid")
        return (
            (col * self.stride_px + self.tile_size_px / 2) * self.pixel_size_nm,
            (row * self.stride_px + self.tile_size_px / 2) * self.pixel_size_nm,
        )

    def nm_to_tile(self, x_nm: float, y_nm: float) -> tuple[int, int]:
        if not np.isfinite([x_nm, y_nm]).all():
            raise ValueError("Position must be finite")
        scale = self.pixel_size_nm * self.stride_px
        return (
            int(np.clip(y_nm / scale, 0, self.grid_shape[0] - 1)),
            int(np.clip(x_nm / scale, 0, self.grid_shape[1] - 1)),
        )
