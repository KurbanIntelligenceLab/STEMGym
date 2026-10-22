from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


class World:


    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.rng = np.random.default_rng()
        self._file: h5py.File | None = None
        self._metadata: dict[str, Any] = {}
        self._ground_truth_cache: dict[str, Any] | None = None
        self._overview_cache: np.ndarray | None = None
        self._valid_region_cache: np.ndarray | None = None
        self._open()

    def _open(self) -> None:
        self._file = h5py.File(self.path, "r")
        meta = self._file["metadata"]
        self._metadata = {
            "pixel_size_nm": float(meta.attrs["pixel_size_nm"]),
            "tile_size_px": int(meta.attrs["tile_size_px"]),
            "overlap_px": int(meta.attrs["overlap_px"]),
            "grid_shape": tuple(meta.attrs["grid_shape"]),
            "fov_nm": tuple(meta.attrs["fov_nm"]),
            "world_shape_px": tuple(meta.attrs["world_shape_px"]),
            "scenario": str(meta.attrs.get("scenario", "unknown")),
            "difficulty": str(meta.attrs.get("difficulty", "medium")),
        }

        for key in meta.attrs:
            if key not in self._metadata:
                self._metadata[key] = meta.attrs[key]

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    @property
    def pixel_size_nm(self) -> float:
        return self._metadata["pixel_size_nm"]

    @property
    def tile_size_px(self) -> int:
        return self._metadata["tile_size_px"]

    @property
    def overlap_px(self) -> int:
        return self._metadata["overlap_px"]

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self._metadata["grid_shape"]

    @property
    def fov_nm(self) -> tuple[float, float]:
        return self._metadata["fov_nm"]

    @property
    def world_shape_px(self) -> tuple[int, int]:
        return self._metadata["world_shape_px"]

    @property
    def stride_px(self) -> int:
        """Tile step size = tile_size - overlap."""
        return self.tile_size_px - self.overlap_px

    @property
    def geometry(self):
        from stem_gym.geometry import WorldGeometry
        return WorldGeometry(self.grid_shape, self.pixel_size_nm, self.tile_size_px, self.stride_px)

    def nm_to_tile(self, x_nm: float, y_nm: float) -> tuple[int, int]:
        return self.geometry.nm_to_tile(x_nm, y_nm)

    def tile_to_nm(self, row: int, col: int) -> tuple[float, float]:
        return self.geometry.tile_to_nm(row, col)

    def get_tile(self, row: int, col: int) -> np.ndarray:
        """Load a single tile from HDF5. Returns (tile_size, tile_size) float32."""
        key = f"tiles/{row}_{col}"
        if key not in self._file:
            raise KeyError(f"Tile {key} not found in {self.path}")
        return np.array(self._file[key], dtype=np.float32)

    def get_overview(self) -> np.ndarray:
        """Load the low-resolution overview image (cached after first call)."""
        if self._overview_cache is None:
            self._overview_cache = np.array(self._file["overview"], dtype=np.float32)
        return self._overview_cache

    def get_overview_at(self, x_nm: float, y_nm: float, size_px: int = 64) -> np.ndarray:
        """Extract a crop from the overview centered at (x_nm, y_nm)."""
        overview = self.get_overview()
        oh, ow = overview.shape

        scale = oh / self.world_shape_px[0]
        cy = int(y_nm / self.pixel_size_nm * scale)
        cx = int(x_nm / self.pixel_size_nm * scale)
        half = size_px // 2


        out = np.zeros((size_px, size_px), dtype=np.float32)
        sy_start = max(0, cy - half)
        sy_end = min(oh, cy - half + size_px)
        sx_start = max(0, cx - half)
        sx_end = min(ow, cx - half + size_px)

        dy_start = max(0, -(cy - half))
        dx_start = max(0, -(cx - half))
        dy_end = dy_start + (sy_end - sy_start)
        dx_end = dx_start + (sx_end - sx_start)

        if sy_end > sy_start and sx_end > sx_start:
            out[dy_start:dy_end, dx_start:dx_end] = overview[sy_start:sy_end, sx_start:sx_end]
        return out

    def get_ground_truth(self) -> dict[str, Any]:
        """Load all ground truth data (cached after first call)."""
        if self._ground_truth_cache is not None:
            return self._ground_truth_cache

        gt = {}
        gt_group = self._file["ground_truth"]
        for key in gt_group:
            data = gt_group[key]
            if isinstance(data, h5py.Dataset):
                arr = np.array(data)

                if arr.dtype.kind == "S":
                    arr = arr.astype(str)
                elif arr.dtype.kind == "O":

                    try:
                        arr = np.char.decode(arr.astype(bytes), "utf-8")
                    except (UnicodeDecodeError, TypeError):
                        arr = np.array([x.decode() if isinstance(x, bytes) else x for x in arr])
                gt[key] = arr

        for key, val in gt_group.attrs.items():
            gt[key] = val

        self._ground_truth_cache = gt
        return gt

    def get_valid_region(self) -> np.ndarray:
        """Load the valid region mask (cached after first call). Returns (H, W) bool array."""
        if self._valid_region_cache is None:
            if "valid_region" in self._file:
                self._valid_region_cache = np.array(self._file["valid_region"], dtype=bool)
            else:
                h, w = self.world_shape_px
                self._valid_region_cache = np.ones((h, w), dtype=bool)
        return self._valid_region_cache

    def get_tile_valid_mask(self, row: int, col: int) -> np.ndarray:
        """Get the valid_region crop for a specific tile."""
        full_mask = self.get_valid_region()
        h, w = full_mask.shape
        ts = self.tile_size_px
        ovlp = self.overlap_px
        half_ovlp = ovlp // 2


        y0 = row * self.stride_px
        x0 = col * self.stride_px

        tile_mask = np.zeros((ts, ts), dtype=bool)


        inner_y0 = half_ovlp if row > 0 else 0
        inner_y1 = ts - half_ovlp if row < self.grid_shape[0] - 1 else ts
        inner_x0 = half_ovlp if col > 0 else 0
        inner_x1 = ts - half_ovlp if col < self.grid_shape[1] - 1 else ts


        gy0 = y0 + inner_y0
        gy1 = min(y0 + inner_y1, h)
        gx0 = x0 + inner_x0
        gx1 = min(x0 + inner_x1, w)

        if gy1 > gy0 and gx1 > gx0:
            tile_mask[inner_y0:inner_y0 + (gy1 - gy0), inner_x0:inner_x0 + (gx1 - gx0)] = (
                full_mask[gy0:gy1, gx0:gx1]
            )

        return tile_mask

    def apply_poisson_noise(self, tile: np.ndarray, dose: float) -> np.ndarray:
        """Apply Poisson noise to a tile at a given dose level."""
        if not np.isfinite(dose) or dose <= 0:
            raise ValueError("Dose must be finite and positive")

        expected = np.clip(tile * dose, 0, None)
        noisy = self.rng.poisson(expected).astype(np.float32)

        if dose > 0:
            noisy /= dose
        return noisy
