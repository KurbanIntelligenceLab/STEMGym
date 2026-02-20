"""Shared abTEM simulation pipeline utilities for scenario generators.

All three scenario generators (SrTiO3, BaTiO3, Si/Ge) share this pipeline:
  MP structure → ASE → supercell → plant defects → record ground truth
  → frozen phonons → PRISM → Poisson noise → tile → HDF5
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from ase import Atoms

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """Full simulation configuration for a scenario."""
    # Identity
    scenario: str = "unknown"
    difficulty: str = "medium"

    # Electron optics
    energy_eV: float = 200e3       # 200 keV
    semiangle_cutoff: float = 21.0  # mrad, convergence semi-angle
    haadf_inner: float = 68.0      # mrad
    haadf_outer: float = 200.0     # mrad

    # PRISM
    interpolation: int = 2
    device: str = "cpu"

    # Potential
    sampling: float = 0.05         # Angstrom per pixel
    slice_thickness: float = 1.0   # Angstrom

    # Frozen phonons
    num_phonon_configs: int = 8
    phonon_sigmas: dict[str, float] = field(default_factory=dict)

    # Tiling (HDF5 output)
    tile_size_px: int = 128
    overlap_px: int = 4

    # Simulation tiling (for large FOVs that don't fit in GPU memory)
    sim_tile_nm: float = 10.0  # tile size for PRISM sub-simulations (nm)
    sim_tile_overlap_nm: float = 1.0  # overlap between sim tiles (nm)

    # Output
    seed: int = 42


# ---------------------------------------------------------------------------
# Structure fetching
# ---------------------------------------------------------------------------

def fetch_mp_structure(mp_id: str, api_key: str | None = None) -> Atoms:
    """Fetch a structure from Materials Project and convert to ASE Atoms.

    Args:
        mp_id: Materials Project ID (e.g. "mp-5229").
        api_key: MP API key. If None, uses MP_API_KEY env var.

    Returns:
        ASE Atoms object.
    """
    from mp_api.client import MPRester
    from pymatgen.io.ase import AseAtomsAdaptor

    with MPRester(api_key) as mpr:
        structure = mpr.get_structure_by_material_id(mp_id)

    return AseAtomsAdaptor.get_atoms(structure)


def build_supercell(
    atoms: Atoms,
    target_size_angstrom: tuple[float, float, float],
) -> Atoms:
    """Build an orthorhombic supercell of approximately the target size.

    Args:
        atoms: Unit cell.
        target_size_angstrom: (x, y, z) target dimensions in Angstrom.

    Returns:
        Supercell as ASE Atoms.
    """
    import abtem

    # First orthogonalize
    atoms = abtem.orthogonalize_cell(atoms)
    cell = atoms.cell.array

    # Compute repetitions needed
    reps = []
    for i in range(3):
        length = np.linalg.norm(cell[i])
        n = max(1, int(np.round(target_size_angstrom[i] / length)))
        reps.append(n)

    logger.info(f"Supercell repetitions: {reps}")
    supercell = atoms * tuple(reps)
    return supercell


def standardize_for_stem(atoms: Atoms, zone_axis: tuple[int, int, int] = (0, 0, 1)) -> Atoms:
    """Standardize cell for STEM: beam along z, orthorhombic.

    For [001] zone axis the default orientation is already correct.
    For [110] zone axis we need to rotate.
    """
    import abtem

    if zone_axis == (0, 0, 1):
        # Default: beam along z
        return abtem.standardize_cell(atoms)
    elif zone_axis == (1, 1, 0):
        # Rotate so [110] is along z
        from ase.build import cut
        # cut to get [110] along z: a1=[1,-1,0], a2=[0,0,1], a3=[1,1,0]
        rotated = cut(atoms, a=(1, -1, 0), b=(0, 0, 1), c=(1, 1, 0))
        return abtem.standardize_cell(rotated)
    else:
        raise ValueError(f"Unsupported zone axis: {zone_axis}")


# ---------------------------------------------------------------------------
# Defect planting
# ---------------------------------------------------------------------------

def plant_vacancies(
    atoms: Atoms,
    species: str,
    rate: float,
    rng: np.random.Generator,
) -> tuple[Atoms, np.ndarray]:
    """Remove atoms to create vacancies.

    Args:
        atoms: Input structure.
        species: Chemical symbol to create vacancies for (e.g. "O").
        rate: Fraction of atoms of that species to remove.
        rng: Random number generator.

    Returns:
        (modified_atoms, vacancy_indices) — indices are into the ORIGINAL atoms.
    """
    symbols = np.array(atoms.get_chemical_symbols())
    species_mask = symbols == species
    species_indices = np.where(species_mask)[0]

    n_vacancies = max(1, int(len(species_indices) * rate))
    vacancy_indices = rng.choice(species_indices, size=n_vacancies, replace=False)

    # Remove atoms (reverse sort to avoid index shifting)
    keep_mask = np.ones(len(atoms), dtype=bool)
    keep_mask[vacancy_indices] = False

    modified = atoms[keep_mask]
    return modified, vacancy_indices


def plant_vacancies_weighted(
    atoms: Atoms,
    species: str,
    rate: float,
    weights: np.ndarray,
    rng: np.random.Generator,
    cluster_fraction: float = 0.7,
) -> tuple[Atoms, np.ndarray]:
    """Remove atoms to create vacancies, biased by per-atom weights.

    Splits total vacancies into clustered (sampled proportional to weights)
    and background (uniform from remaining atoms).

    Args:
        atoms: Input structure.
        species: Chemical symbol to create vacancies for (e.g. "O").
        rate: Fraction of atoms of that species to remove.
        weights: Per-species-atom array (higher = more likely to be a vacancy).
                 Must have length equal to the number of atoms of `species`.
        rng: Random number generator.
        cluster_fraction: Fraction of vacancies drawn from weighted distribution.

    Returns:
        (modified_atoms, vacancy_indices) — indices are into the ORIGINAL atoms.
    """
    symbols = np.array(atoms.get_chemical_symbols())
    species_mask = symbols == species
    species_indices = np.where(species_mask)[0]

    if len(weights) != len(species_indices):
        raise ValueError(
            f"weights length ({len(weights)}) != number of {species} atoms "
            f"({len(species_indices)})"
        )

    n_vacancies = max(1, int(len(species_indices) * rate))
    n_clustered = int(n_vacancies * cluster_fraction)
    n_background = n_vacancies - n_clustered

    # --- Clustered vacancies: sample proportional to weights ---
    w = np.asarray(weights, dtype=np.float64)
    w_sum = w.sum()
    if w_sum > 0:
        prob = w / w_sum
    else:
        prob = np.ones(len(w)) / len(w)

    clustered_local = rng.choice(len(species_indices), size=n_clustered, replace=False, p=prob)
    chosen_set = set(clustered_local.tolist())

    # --- Background vacancies: uniform from remainder ---
    remaining_local = np.array([i for i in range(len(species_indices)) if i not in chosen_set])
    if n_background > 0 and len(remaining_local) > 0:
        n_background = min(n_background, len(remaining_local))
        bg_local = rng.choice(remaining_local, size=n_background, replace=False)
        all_local = np.concatenate([clustered_local, bg_local])
    else:
        all_local = clustered_local

    vacancy_indices = species_indices[all_local]

    # Remove atoms
    keep_mask = np.ones(len(atoms), dtype=bool)
    keep_mask[vacancy_indices] = False
    modified = atoms[keep_mask]

    return modified, vacancy_indices


def plant_substitutions(
    atoms: Atoms,
    original_species: str,
    substitute_species: str,
    rate: float,
    rng: np.random.Generator,
) -> tuple[Atoms, np.ndarray]:
    """Replace atoms to create substitutional defects.

    Returns:
        (modified_atoms, substitution_indices) — indices into ORIGINAL atoms.
    """
    symbols = np.array(atoms.get_chemical_symbols())
    species_mask = symbols == original_species
    species_indices = np.where(species_mask)[0]

    n_subs = max(1, int(len(species_indices) * rate))
    sub_indices = rng.choice(species_indices, size=n_subs, replace=False)

    modified = atoms.copy()
    new_symbols = list(modified.get_chemical_symbols())
    for idx in sub_indices:
        new_symbols[idx] = substitute_species
    modified.set_chemical_symbols(new_symbols)

    return modified, sub_indices


# ---------------------------------------------------------------------------
# Ground truth recording
# ---------------------------------------------------------------------------

@dataclass
class GroundTruth:
    """Ground truth for a simulated world."""
    # All positions in nm (x, y) — projected along beam direction
    atom_positions_nm: np.ndarray     # (N, 2) all atoms in pristine structure
    atom_types: np.ndarray            # (N,) int — 0=normal, 1=vacancy, 2=substitution
    atom_species: list[str]           # (N,) element symbols
    defect_mask: np.ndarray           # (N,) bool
    defect_types: list[str]           # (N,) "pristine"/"vacancy"/"substitution"
    phase_map: np.ndarray | None = None  # (H, W) int phase labels
    extra: dict[str, Any] = field(default_factory=dict)


def record_ground_truth(
    pristine_atoms: Atoms,
    vacancy_indices: np.ndarray | None = None,
    substitution_indices: np.ndarray | None = None,
    pixel_size_nm: float = 0.005,
) -> GroundTruth:
    """Record ground truth from pristine structure and defect indices.

    Projects atom positions along z (beam direction) to get 2D positions.
    """
    positions_angstrom = pristine_atoms.get_positions()
    # Project to xy plane, convert to nm
    positions_nm = positions_angstrom[:, :2] / 10.0  # Angstrom -> nm

    n_atoms = len(pristine_atoms)
    atom_types = np.zeros(n_atoms, dtype=np.int32)
    defect_types = ["pristine"] * n_atoms
    defect_mask = np.zeros(n_atoms, dtype=bool)

    if vacancy_indices is not None:
        for idx in vacancy_indices:
            if idx < n_atoms:
                atom_types[idx] = 1
                defect_types[idx] = "vacancy"
                defect_mask[idx] = True

    if substitution_indices is not None:
        for idx in substitution_indices:
            if idx < n_atoms:
                atom_types[idx] = 2
                defect_types[idx] = "substitution"
                defect_mask[idx] = True

    species = pristine_atoms.get_chemical_symbols()

    return GroundTruth(
        atom_positions_nm=positions_nm.astype(np.float32),
        atom_types=atom_types,
        atom_species=species,
        defect_mask=defect_mask,
        defect_types=defect_types,
    )


# ---------------------------------------------------------------------------
# abTEM simulation
# ---------------------------------------------------------------------------

def _run_single_tile_simulation(
    atoms: Atoms,
    config: SimConfig,
) -> np.ndarray:
    """Run PRISM STEM-HAADF on a single (small) structure tile.

    The structure must be small enough that the S-matrix fits in memory.
    """
    import abtem

    frozen = abtem.FrozenPhonons(
        atoms,
        num_configs=config.num_phonon_configs,
        sigmas=config.phonon_sigmas,
        seed=config.seed,
    )

    potential = abtem.Potential(
        frozen,
        sampling=config.sampling,
        slice_thickness=config.slice_thickness,
        device=config.device,
    )

    interpolation = config.interpolation
    if config.device == "gpu":
        interpolation = max(interpolation, 4)

    s_matrix = abtem.SMatrix(
        semiangle_cutoff=config.semiangle_cutoff,
        energy=config.energy_eV,
        potential=potential,
        interpolation=interpolation,
        device=config.device,
        store_on_host=(config.device == "gpu"),
    )

    flex_detector = abtem.FlexibleAnnularDetector(
        step_size=1.0,
        inner=0.0,
        outer=None,
    )

    scan = abtem.GridScan()
    measurement = s_matrix.scan(scan, detectors=flex_detector)

    if hasattr(measurement, "compute"):
        measurement = measurement.compute()

    angle_axis = [ax for ax in measurement.axes_metadata if hasattr(ax, "offset")]
    if angle_axis:
        ax = angle_axis[-1]
        max_avail = ax.offset + ax.sampling * measurement.array.shape[-1]
    else:
        max_avail = config.haadf_outer

    haadf_inner = min(config.haadf_inner, max_avail * 0.9)
    haadf_outer = min(config.haadf_outer, max_avail)
    haadf = measurement.integrate_radial(inner=haadf_inner, outer=haadf_outer)

    image = np.array(haadf.array, dtype=np.float32)
    if image.ndim > 2:
        image = np.mean(image, axis=tuple(range(image.ndim - 2)))

    return image.astype(np.float32)


def _crop_atoms_xy(atoms: Atoms, x0: float, y0: float,
                   lx: float, ly: float) -> Atoms:
    """Crop atoms to a rectangular region in XY, creating a new cell."""
    positions = atoms.get_positions()
    mask = (
        (positions[:, 0] >= x0) & (positions[:, 0] < x0 + lx)
        & (positions[:, 1] >= y0) & (positions[:, 1] < y0 + ly)
    )
    cropped = atoms[mask]
    new_positions = cropped.get_positions()
    new_positions[:, 0] -= x0
    new_positions[:, 1] -= y0
    cell = atoms.cell.array.copy()
    cell[0, 0] = lx
    cell[1, 1] = ly
    cropped.set_cell(cell)
    cropped.set_positions(new_positions)
    cropped.set_pbc(atoms.pbc)
    return cropped


def run_stem_simulation(
    atoms: Atoms,
    config: SimConfig,
) -> np.ndarray:
    """Run PRISM STEM-HAADF simulation, tiling if the structure is too large.

    For structures larger than ``config.sim_tile_nm``, the XY plane is
    divided into overlapping tiles, each simulated independently, then
    blended together with cosine-weighted overlap.

    Args:
        atoms: Structure (with or without frozen phonons already applied).
        config: Simulation configuration.

    Returns:
        HAADF image as (H, W) float32 numpy array.
    """
    cell = atoms.cell.array
    extent_x = cell[0, 0]  # Angstrom
    extent_y = cell[1, 1]

    tile_ang = config.sim_tile_nm * 10.0
    overlap_ang = config.sim_tile_overlap_nm * 10.0

    needs_tiling = (extent_x > tile_ang * 1.5) or (extent_y > tile_ang * 1.5)

    if not needs_tiling:
        logger.info("Structure fits in single tile, running directly...")
        return _run_single_tile_simulation(atoms, config)

    # --- Tiled simulation ---
    stride_ang = tile_ang - overlap_ang

    n_tiles_x = max(1, int(np.ceil((extent_x - overlap_ang) / stride_ang)))
    n_tiles_y = max(1, int(np.ceil((extent_y - overlap_ang) / stride_ang)))

    # Recompute tile_ang to cover the full extent evenly
    tile_x = (extent_x + overlap_ang * (n_tiles_x - 1)) / n_tiles_x if n_tiles_x > 1 else extent_x
    tile_y = (extent_y + overlap_ang * (n_tiles_y - 1)) / n_tiles_y if n_tiles_y > 1 else extent_y
    stride_x = tile_x - overlap_ang if n_tiles_x > 1 else extent_x
    stride_y = tile_y - overlap_ang if n_tiles_y > 1 else extent_y

    total_tiles = n_tiles_x * n_tiles_y
    logger.info(
        f"Tiling simulation: {n_tiles_x}x{n_tiles_y} = {total_tiles} tiles, "
        f"tile={tile_x:.0f}x{tile_y:.0f} Å, stride={stride_x:.0f}x{stride_y:.0f} Å, "
        f"overlap={overlap_ang:.0f} Å"
    )

    # Output image size
    full_px_x = int(round(extent_x / config.sampling))
    full_px_y = int(round(extent_y / config.sampling))
    image = np.zeros((full_px_y, full_px_x), dtype=np.float64)
    weights = np.zeros((full_px_y, full_px_x), dtype=np.float64)

    overlap_px = int(round(overlap_ang / config.sampling))

    for ti in range(n_tiles_x):
        for tj in range(n_tiles_y):
            tile_idx = ti * n_tiles_y + tj + 1
            x0 = ti * stride_x
            y0 = tj * stride_y
            # Clamp to extent
            x0 = min(x0, extent_x - tile_x)
            y0 = min(y0, extent_y - tile_y)

            logger.info(f"  Tile {tile_idx}/{total_tiles}: "
                        f"origin=({x0:.0f}, {y0:.0f}) Å")

            tile_atoms = _crop_atoms_xy(atoms, x0, y0, tile_x, tile_y)
            tile_image = _run_single_tile_simulation(tile_atoms, config)

            # Map tile pixels into the full image
            px_x0 = int(round(x0 / config.sampling))
            px_y0 = int(round(y0 / config.sampling))
            th, tw = tile_image.shape

            # Build blend weights for this tile
            w = np.ones_like(tile_image, dtype=np.float64)
            # Clamp overlap to actual tile dims
            olap_w = min(overlap_px, tw // 3) if overlap_px > 0 else 0
            olap_h = min(overlap_px, th // 3) if overlap_px > 0 else 0
            if olap_w > 0:
                ramp_w = np.linspace(0, 1, olap_w)
                if ti > 0:
                    w[:, :olap_w] *= ramp_w[np.newaxis, :]
                if ti < n_tiles_x - 1:
                    w[:, -olap_w:] *= ramp_w[np.newaxis, ::-1]
            if olap_h > 0:
                ramp_h = np.linspace(0, 1, olap_h)
                if tj > 0:
                    w[:olap_h, :] *= ramp_h[:, np.newaxis]
                if tj < n_tiles_y - 1:
                    w[-olap_h:, :] *= ramp_h[::-1, np.newaxis]

            # Accumulate
            py_end = min(px_y0 + th, full_px_y)
            px_end = min(px_x0 + tw, full_px_x)
            sh = py_end - px_y0
            sw = px_end - px_x0
            image[px_y0:py_end, px_x0:px_end] += tile_image[:sh, :sw] * w[:sh, :sw]
            weights[px_y0:py_end, px_x0:px_end] += w[:sh, :sw]

            logger.info(f"  Tile {tile_idx}/{total_tiles} complete.")

    # Normalize by weights
    mask = weights > 0
    image[mask] /= weights[mask]

    return image.astype(np.float32)


# ---------------------------------------------------------------------------
# HDF5 writing
# ---------------------------------------------------------------------------

def write_world_hdf5(
    output_path: Path,
    image: np.ndarray,
    ground_truth: GroundTruth,
    config: SimConfig,
    pixel_size_nm: float | None = None,
) -> None:
    """Write a simulated world to HDF5 in STEMGym format.

    Args:
        output_path: Path to write HDF5 file.
        image: (H, W) HAADF image.
        ground_truth: Ground truth data.
        config: Simulation config.
        pixel_size_nm: Override pixel size. If None, computed from config.sampling.
    """
    from scipy.ndimage import zoom as scipy_zoom

    if pixel_size_nm is None:
        pixel_size_nm = config.sampling / 10.0  # Angstrom -> nm

    h, w = image.shape
    tile_size = config.tile_size_px
    overlap = config.overlap_px
    stride = tile_size - overlap

    grid_h = max(1, (h - tile_size) // stride + 1)
    grid_w = max(1, (w - tile_size) // stride + 1)

    fov_h = h * pixel_size_nm
    fov_w = w * pixel_size_nm

    # Normalize image to [0, 1]
    img_min, img_max = image.min(), image.max()
    if img_max > img_min:
        image_norm = (image - img_min) / (img_max - img_min)
    else:
        image_norm = image.copy()

    # Overview: downsample by 4x
    scale = min(1.0, 256.0 / max(h, w))
    overview = scipy_zoom(image_norm, scale, order=1).astype(np.float32)

    # Valid region: entire image
    valid_region = np.ones((h, w), dtype=bool)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing HDF5 to {output_path}...")
    with h5py.File(output_path, "w") as f:
        # Metadata
        meta = f.create_group("metadata")
        meta.attrs["pixel_size_nm"] = pixel_size_nm
        meta.attrs["tile_size_px"] = tile_size
        meta.attrs["overlap_px"] = overlap
        meta.attrs["grid_shape"] = [grid_h, grid_w]
        meta.attrs["fov_nm"] = [fov_h, fov_w]
        meta.attrs["world_shape_px"] = [h, w]
        meta.attrs["scenario"] = config.scenario
        meta.attrs["difficulty"] = config.difficulty

        # Overview
        f.create_dataset("overview", data=overview, dtype=np.float32)

        # Valid region
        f.create_dataset("valid_region", data=valid_region, dtype=bool)

        # Tiles
        tiles_group = f.create_group("tiles")
        for r in range(grid_h):
            for c in range(grid_w):
                y0 = r * stride
                x0 = c * stride
                y1 = min(y0 + tile_size, h)
                x1 = min(x0 + tile_size, w)
                tile = image_norm[y0:y1, x0:x1]
                # Pad if needed (edge tiles)
                if tile.shape != (tile_size, tile_size):
                    padded = np.zeros((tile_size, tile_size), dtype=np.float32)
                    padded[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded
                tiles_group.create_dataset(
                    f"{r}_{c}", data=tile, dtype=np.float32, chunks=True,
                )

        # Ground truth
        gt = f.create_group("ground_truth")
        gt.create_dataset("atom_positions", data=ground_truth.atom_positions_nm)
        gt.create_dataset("atom_types", data=ground_truth.atom_types)
        gt.create_dataset("defect_mask", data=ground_truth.defect_mask)
        defect_bytes = [s.encode("utf-8") for s in ground_truth.defect_types]
        gt.create_dataset("defect_types", data=defect_bytes)
        species_bytes = [s.encode("utf-8") for s in ground_truth.atom_species]
        gt.create_dataset("atom_species", data=species_bytes)

        if ground_truth.phase_map is not None:
            gt.create_dataset("phase_map", data=ground_truth.phase_map)

        gt.attrs["n_atoms"] = len(ground_truth.atom_positions_nm)
        gt.attrs["n_vacancies"] = int(np.sum(ground_truth.atom_types == 1))
        gt.attrs["n_substitutions"] = int(np.sum(ground_truth.atom_types == 2))

        for key, val in ground_truth.extra.items():
            if isinstance(val, np.ndarray):
                gt.create_dataset(key, data=val)
            else:
                gt.attrs[key] = val

    n_atoms = len(ground_truth.atom_positions_nm)
    n_defects = int(ground_truth.defect_mask.sum())
    logger.info(
        f"Wrote {output_path}: {h}x{w} image, {grid_h}x{grid_w} tiles, "
        f"{n_atoms} atoms, {n_defects} defects"
    )


# ---------------------------------------------------------------------------
# Difficulty presets
# ---------------------------------------------------------------------------

DIFFICULTY_PARAMS = {
    "easy": {"vacancy_rate": 0.05, "num_phonon_configs": 4, "noise_dose": 1e4},
    "medium": {"vacancy_rate": 0.03, "num_phonon_configs": 8, "noise_dose": 5e3},
    "hard": {"vacancy_rate": 0.01, "num_phonon_configs": 16, "noise_dose": 1e3},
}
