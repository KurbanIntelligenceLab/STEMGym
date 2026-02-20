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

    # Tiling
    tile_size_px: int = 128
    overlap_px: int = 4

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

def run_stem_simulation(
    atoms: Atoms,
    config: SimConfig,
) -> np.ndarray:
    """Run PRISM STEM-HAADF simulation.

    Args:
        atoms: Structure (with or without frozen phonons already applied).
        config: Simulation configuration.

    Returns:
        HAADF image as (H, W) float32 numpy array.
    """
    import abtem

    logger.info("Building frozen phonons...")
    frozen = abtem.FrozenPhonons(
        atoms,
        num_configs=config.num_phonon_configs,
        sigmas=config.phonon_sigmas,
        seed=config.seed,
    )

    logger.info("Building potential...")
    potential = abtem.Potential(
        frozen,
        sampling=config.sampling,
        slice_thickness=config.slice_thickness,
        device=config.device,
    )

    logger.info("Building S-matrix (PRISM)...")
    s_matrix = abtem.SMatrix(
        semiangle_cutoff=config.semiangle_cutoff,
        energy=config.energy_eV,
        potential=potential,
        interpolation=config.interpolation,
        device=config.device,
    )

    logger.info("Setting up detector...")
    # Use FlexibleAnnularDetector to get radially-resolved diffraction,
    # then integrate over HAADF range. This avoids max-angle issues
    # since we can clamp post-hoc.
    flex_detector = abtem.FlexibleAnnularDetector(
        step_size=1.0,
        inner=0.0,
        outer=None,  # auto: use max available angle
    )

    logger.info("Running PRISM scan...")
    scan = abtem.GridScan()
    measurement = s_matrix.scan(scan, detectors=flex_detector)

    # Compute if lazy (dask-backed)
    if hasattr(measurement, "compute"):
        logger.info("Computing measurement (may be lazy/dask)...")
        measurement = measurement.compute()

    # Clamp HAADF angles to available range, then integrate
    # The angular axis is the last axis of FlexibleAnnularDetector output
    angle_axis = [ax for ax in measurement.axes_metadata if hasattr(ax, "offset")]
    if angle_axis:
        ax = angle_axis[-1]
        max_avail = ax.offset + ax.sampling * measurement.array.shape[-1]
    else:
        max_avail = config.haadf_outer

    haadf_inner = min(config.haadf_inner, max_avail * 0.9)
    haadf_outer = min(config.haadf_outer, max_avail)
    if haadf_outer < config.haadf_outer:
        logger.warning(
            f"Clamping HAADF outer from {config.haadf_outer:.0f} to "
            f"{haadf_outer:.0f} mrad (max available: {max_avail:.0f} mrad)"
        )
    logger.info(f"Integrating HAADF: {haadf_inner:.0f}-{haadf_outer:.0f} mrad...")
    haadf = measurement.integrate_radial(inner=haadf_inner, outer=haadf_outer)

    image = np.array(haadf.array, dtype=np.float32)
    if image.ndim > 2:
        # Average over frozen phonon configs if not already done
        image = np.mean(image, axis=tuple(range(image.ndim - 2)))

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
