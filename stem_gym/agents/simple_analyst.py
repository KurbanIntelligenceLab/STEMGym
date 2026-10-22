from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max


@dataclass
class AnalysisResult:

    atom_positions: np.ndarray
    atom_intensities: np.ndarray
    n_atoms: int = 0
    n_vacancies: int = 0
    n_substitutions: int = 0
    mean_intensity: float = 0.0
    intensity_std: float = 0.0
    phase_label: int = -1


class SimpleAnalyst:


    def __init__(
        self,
        min_distance: int = 6,
        threshold_rel: float = 0.2,
        smooth_sigma: float = 1.5,
        vacancy_zscore: float = -1.5,
        substitution_zscore: float = 1.5,
    ):
        self.min_distance = min_distance
        self.threshold_rel = threshold_rel
        self.smooth_sigma = smooth_sigma
        self.vacancy_zscore = vacancy_zscore
        self.substitution_zscore = substitution_zscore


        self._all_intensities: list[float] = []
        self._tile_results: list[AnalysisResult] = []
        self._global_mean: float = 0.0
        self._global_std: float = 1.0

    def reset(self) -> None:
        """Reset accumulated state for a new episode."""
        self._all_intensities.clear()
        self._tile_results.clear()
        self._global_mean = 0.0
        self._global_std = 1.0

    def analyze_tile(
        self,
        tile: np.ndarray,
        tile_position_nm: tuple[float, float] | None = None,
        pixel_size_nm: float = 0.005,
    ) -> AnalysisResult:
        """Analyze a single tile for atoms and defects."""
        if tile.max() <= 0:
            result = AnalysisResult(
                atom_positions=np.empty((0, 2)),
                atom_intensities=np.empty(0),
            )
            self._tile_results.append(result)
            return result


        smoothed = gaussian_filter(tile.astype(np.float64), sigma=self.smooth_sigma)


        threshold_abs = smoothed.max() * self.threshold_rel
        coords = peak_local_max(
            smoothed,
            min_distance=self.min_distance,
            threshold_abs=threshold_abs,
        )

        if len(coords) == 0:
            result = AnalysisResult(
                atom_positions=np.empty((0, 2)),
                atom_intensities=np.empty(0),
                mean_intensity=float(tile.mean()),
            )
            self._tile_results.append(result)
            return result


        intensities = np.array([tile[r, c] for r, c in coords], dtype=np.float32)


        self._all_intensities.extend(intensities.tolist())
        if len(self._all_intensities) > 1:
            self._global_mean = float(np.mean(self._all_intensities))
            self._global_std = max(float(np.std(self._all_intensities)), 1e-6)


        n_vac = 0
        n_sub = 0
        if self._global_std > 0 and len(self._all_intensities) > 10:
            zscores = (intensities - self._global_mean) / self._global_std
            n_vac = int(np.sum(zscores < self.vacancy_zscore))
            n_sub = int(np.sum(zscores > self.substitution_zscore))

        result = AnalysisResult(
            atom_positions=coords.astype(np.float32),
            atom_intensities=intensities,
            n_atoms=len(coords),
            n_vacancies=n_vac,
            n_substitutions=n_sub,
            mean_intensity=float(intensities.mean()),
            intensity_std=float(intensities.std()) if len(intensities) > 1 else 0.0,
        )
        self._tile_results.append(result)
        return result

    def get_defect_census(self) -> dict[str, int]:
        """Aggregate defect counts across all analyzed tiles."""
        total_vac = sum(r.n_vacancies for r in self._tile_results)
        total_sub = sum(r.n_substitutions for r in self._tile_results)
        census = {}
        if total_vac > 0:
            census["vacancy"] = total_vac
        if total_sub > 0:
            census["substitution"] = total_sub
        return census

    def get_total_atoms(self) -> int:
        """Total atoms detected across all tiles."""
        return sum(r.n_atoms for r in self._tile_results)

    def estimate_phase(self, tile: np.ndarray) -> int:
        """Rough phase estimate based on lattice spacing."""
        if tile.max() <= 0:
            return 0

        from scipy.signal import fftconvolve
        tile_norm = tile - tile.mean()
        acf = fftconvolve(tile_norm, tile_norm[::-1, ::-1], mode="same")
        acf /= max(acf.max(), 1e-10)


        center = np.array(acf.shape) // 2

        mask = np.ones_like(acf, dtype=bool)
        r = self.min_distance
        mask[center[0]-r:center[0]+r+1, center[1]-r:center[1]+r+1] = False
        acf_masked = acf * mask

        if acf_masked.max() < 0.1:
            return 0

        peak = np.unravel_index(acf_masked.argmax(), acf.shape)
        spacing = np.sqrt((peak[0] - center[0])**2 + (peak[1] - center[1])**2)


        return 1 if spacing > self.min_distance * 1.5 else 0

    @property
    def analysed_tiles(self) -> int:
        return len(self._tile_results)
