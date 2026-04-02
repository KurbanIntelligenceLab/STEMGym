#!/usr/bin/env python3
"""Generate BaTiO3 interface scenario: phase boundaries + clustered O vacancies.

Material: BaTiO3 (mp-5986), [001] zone axis, ~100nm FOV
Defects: Cubic/tetragonal phase boundaries AND O vacancies concentrated
         near the boundaries (as seen in real ferroelectric ceramics)
Task: Defect Census + Phase Mapping (dual-objective)

Extends the uniform BaTiO3 generator by adding vacancy accumulation near
phase boundaries.  After building the composite structure and phase map,
we compute a distance-to-nearest-boundary field with
scipy.ndimage.distance_transform_edt.  O atoms close to boundaries get
higher vacancy probability via plant_vacancies_weighted.

Usage:
    python scripts/generate_worlds/generate_batio3_interface.py [--difficulty easy|medium|hard]
                                                      [--output data/simulated/batio3_interface.h5]
                                                      [--device cpu|gpu]
                                                      [--seed 42]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from ase import Atoms
from scipy.ndimage import distance_transform_edt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from sim_utils import (
    SimConfig,
    build_supercell,
    plant_vacancies_weighted,
    record_ground_truth,
    run_stem_simulation,
    run_stem_simulation_shard,
    merge_simulation_shards,
    write_world_hdf5,
    add_shard_args,
    DIFFICULTY_PARAMS,
)

# BaTiO3 lattice parameters
# Cubic phase: a = 4.009 Å (above Curie temperature)
BTO_CUBIC_A = 4.009
# Tetragonal phase: a = 3.994 Å, c = 4.038 Å (room temperature)
BTO_TETRA_A = 3.994
BTO_TETRA_C = 4.038

# Debye-Waller factors for BaTiO3
BTO_PHONON_SIGMAS = {
    "Ba": 0.080,
    "Ti": 0.080,
    "O": 0.090,
}


def build_batio3_cell(phase: str = "cubic") -> Atoms:
    """Build BaTiO3 perovskite unit cell.

    Args:
        phase: "cubic" or "tetragonal".

    Cubic: a = b = c = 4.009 Å
    Tetragonal: a = b = 3.994 Å, c = 4.038 Å, Ti displaced along c
    """
    if phase == "cubic":
        a = BTO_CUBIC_A
        atoms = Atoms(
            symbols=["Ba", "Ti", "O", "O", "O"],
            positions=[
                [0.0, 0.0, 0.0],
                [a/2, a/2, a/2],
                [a/2, a/2, 0.0],
                [a/2, 0.0, a/2],
                [0.0, a/2, a/2],
            ],
            cell=[a, a, a],
            pbc=True,
        )
    elif phase == "tetragonal":
        a, c = BTO_TETRA_A, BTO_TETRA_C
        # Ti is displaced ~0.05 Å along c in tetragonal phase
        ti_disp = 0.05
        atoms = Atoms(
            symbols=["Ba", "Ti", "O", "O", "O"],
            positions=[
                [0.0, 0.0, 0.0],
                [a/2, a/2, c/2 + ti_disp],
                [a/2, a/2, 0.0],
                [a/2, 0.0, c/2],
                [0.0, a/2, c/2],
            ],
            cell=[a, a, c],
            pbc=True,
        )
    else:
        raise ValueError(f"Unknown phase: {phase}")
    return atoms


def make_phase_boundary_mask(
    shape_px: tuple[int, int],
    n_boundaries: int = 1,
    waviness: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Create a phase map with boundaries between cubic (0) and tetragonal (1).

    Args:
        shape_px: (H, W) image shape in pixels.
        n_boundaries: Number of phase boundaries.
        waviness: Amplitude of boundary waviness as fraction of width.
        rng: Random number generator.

    Returns:
        (H, W) int32 array: 0 = cubic, 1 = tetragonal.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    h, w = shape_px
    phase_map = np.zeros(shape_px, dtype=np.int32)

    # Evenly space boundaries
    boundary_positions = np.linspace(0, w, n_boundaries + 2)[1:-1]

    for i, bx in enumerate(boundary_positions):
        # Create wavy boundary
        if waviness > 0:
            # Sinusoidal + random perturbation
            freq = rng.uniform(1, 4)
            phase_offset = rng.uniform(0, 2 * np.pi)
            wave = waviness * w * np.sin(
                2 * np.pi * freq * np.arange(h) / h + phase_offset
            )
            # Add some random roughness
            roughness = waviness * w * 0.3 * rng.normal(size=h)
            from scipy.ndimage import gaussian_filter1d
            roughness = gaussian_filter1d(roughness, sigma=h * 0.05)
            boundary = bx + wave + roughness
        else:
            boundary = np.full(h, bx)

        for row in range(h):
            col_boundary = int(np.clip(boundary[row], 0, w - 1))
            # Alternate phases at each boundary
            if i % 2 == 0:
                phase_map[row, col_boundary:] = 1
            else:
                phase_map[row, col_boundary:] = 0

    return phase_map


def build_composite_structure(
    fov_angstrom: float,
    z_thickness: float,
    phase_map: np.ndarray,
    sampling_angstrom: float = 0.5,
) -> tuple[Atoms, np.ndarray]:
    """Build a composite cubic+tetragonal BaTiO3 structure.

    Places cubic or tetragonal unit cells based on the phase map.
    Returns the full atomic structure and per-atom phase labels.

    For efficiency, we build separate supercells for each phase region
    and merge them.
    """
    import abtem

    # Build supercells for both phases
    cubic_cell = build_batio3_cell("cubic")
    tetra_cell = build_batio3_cell("tetragonal")

    # Build full cubic supercell as base
    cubic_super = build_supercell(
        cubic_cell,
        target_size_angstrom=(fov_angstrom, fov_angstrom, z_thickness),
    )
    tetra_super = build_supercell(
        tetra_cell,
        target_size_angstrom=(fov_angstrom, fov_angstrom, z_thickness),
    )

    # For a first-order approximation: use cubic everywhere but modify
    # atom positions in tetragonal regions to match tetragonal cell.
    # This gives physically correct column positions.
    composite = cubic_super.copy()
    cell_x = composite.cell[0, 0]
    cell_y = composite.cell[1, 1]
    map_h, map_w = phase_map.shape

    positions = composite.get_positions()
    atom_phases = np.zeros(len(composite), dtype=np.int32)

    for i, pos in enumerate(positions):
        # Map atom position to phase map pixel
        px = int(pos[0] / cell_x * map_w) % map_w
        py = int(pos[1] / cell_y * map_h) % map_h

        if phase_map[py, px] == 1:
            atom_phases[i] = 1
            # Apply tetragonal distortion: slightly compress a,b and stretch c
            scale_ab = BTO_TETRA_A / BTO_CUBIC_A
            scale_c = BTO_TETRA_C / BTO_CUBIC_A
            positions[i, 0] *= scale_ab
            positions[i, 1] *= scale_ab
            positions[i, 2] *= scale_c

    composite.set_positions(positions)

    return composite, atom_phases

# Difficulty params for BaTiO3 interface scenario
INTERFACE_DIFFICULTY = {
    "easy": {
        "vacancy_rate": 0.04,
        "cluster_fraction": 0.60,
        "boundary_sigma_px": 80,
        "n_boundaries": 1,
        "waviness": 0.0,
    },
    "medium": {
        "vacancy_rate": 0.03,
        "cluster_fraction": 0.70,
        "boundary_sigma_px": 50,
        "n_boundaries": 2,
        "waviness": 0.05,
    },
    "hard": {
        "vacancy_rate": 0.015,
        "cluster_fraction": 0.80,
        "boundary_sigma_px": 25,
        "n_boundaries": 3,
        "waviness": 0.10,
    },
}


def compute_boundary_distance_weights(
    atoms,
    phase_map: np.ndarray,
    boundary_sigma_px: float,
    sampling_angstrom: float = 0.05,
) -> np.ndarray:
    """Compute per-O-atom weights from distance to nearest phase boundary.

    Uses scipy.ndimage.distance_transform_edt on the phase boundary map
    to get a pixel-level distance field.  Each O atom's xy position is
    mapped to the distance field and converted to a Gaussian weight.

    Args:
        atoms: Structure containing O atoms.
        phase_map: (H, W) int32 phase map (0=cubic, 1=tetragonal).
        boundary_sigma_px: Gaussian sigma in pixels.
        sampling_angstrom: Pixel size in Angstrom.

    Returns:
        weights array of length == number of O atoms.
    """
    # Find boundary pixels: where neighbouring pixels differ
    boundary_mask = np.zeros_like(phase_map, dtype=bool)
    boundary_mask[:-1, :] |= phase_map[:-1, :] != phase_map[1:, :]
    boundary_mask[:, :-1] |= phase_map[:, :-1] != phase_map[:, 1:]

    # Distance from every pixel to nearest boundary pixel
    # distance_transform_edt computes distance from 0-valued pixels
    # So invert: boundary=0, rest=1
    dist_input = (~boundary_mask).astype(np.float64)
    dist_field = distance_transform_edt(dist_input)

    # Map each O atom to the distance field
    positions = atoms.get_positions()
    symbols = np.array(atoms.get_chemical_symbols())
    o_mask = symbols == "O"
    o_positions = positions[o_mask]

    cell_x = atoms.cell[0, 0]
    cell_y = atoms.cell[1, 1]
    map_h, map_w = phase_map.shape

    # Atom position -> pixel coordinate
    px_x = (o_positions[:, 0] / cell_x * map_w).astype(int) % map_w
    px_y = (o_positions[:, 1] / cell_y * map_h).astype(int) % map_h

    distances_px = dist_field[px_y, px_x]

    # Gaussian weights: closer to boundary = higher weight
    weights = np.exp(-distances_px**2 / (2.0 * boundary_sigma_px**2))
    return weights


def generate_batio3_interface(
    difficulty: str = "medium",
    output_path: Path | None = None,
    device: str = "cpu",
    seed: int = 42,
    fov_nm: float = 100.0,
    shard_id: int | None = None,
    total_shards: int | None = None,
    merge_shards: int | None = None,
) -> Path:
    """Generate BaTiO3 world with phase boundaries and clustered O vacancies.

    Args:
        difficulty: "easy", "medium", or "hard".
        output_path: HDF5 output path.
        device: "cpu" or "gpu" for abTEM.
        seed: Random seed.
        fov_nm: Field of view in nm.

    Returns:
        Path to generated HDF5 file.
    """
    if output_path is None:
        output_path = Path("data/simulated/batio3_interface.h5")

    rng = np.random.default_rng(seed)
    diff = DIFFICULTY_PARAMS[difficulty]
    idiff = INTERFACE_DIFFICULTY[difficulty]

    fov_angstrom = fov_nm * 10.0
    z_thickness = 100.0
    sampling = 0.05  # Angstrom/pixel

    # --- 1. Create phase boundary map ---
    img_size = int(fov_angstrom / sampling)
    logger.info(f"Expected image size: ~{img_size}x{img_size} pixels")

    phase_map = make_phase_boundary_mask(
        (img_size, img_size),
        n_boundaries=idiff["n_boundaries"],
        waviness=idiff["waviness"],
        rng=rng,
    )
    logger.info(
        f"Phase map: {np.sum(phase_map == 0)} cubic, "
        f"{np.sum(phase_map == 1)} tetragonal pixels"
    )

    # --- 2. Build composite structure ---
    logger.info("Building composite cubic/tetragonal structure...")
    composite, atom_phases = build_composite_structure(
        fov_angstrom, z_thickness, phase_map, sampling,
    )
    logger.info(f"Composite: {len(composite)} atoms")

    pristine = composite.copy()

    # --- 3. Plant clustered O vacancies near boundaries ---
    vacancy_rate = idiff["vacancy_rate"]
    cluster_fraction = idiff["cluster_fraction"]
    boundary_sigma_px = idiff["boundary_sigma_px"]

    weights = compute_boundary_distance_weights(
        composite, phase_map, boundary_sigma_px, sampling,
    )
    logger.info(
        f"Planting boundary-clustered O vacancies: rate={vacancy_rate*100:.1f}%, "
        f"cluster_frac={cluster_fraction:.0%}, sigma={boundary_sigma_px}px"
    )

    defected, vacancy_indices = plant_vacancies_weighted(
        composite, species="O", rate=vacancy_rate,
        weights=weights, rng=rng, cluster_fraction=cluster_fraction,
    )
    logger.info(f"Created {len(vacancy_indices)} O vacancies")

    # --- 4. Record ground truth ---
    gt = record_ground_truth(
        pristine,
        vacancy_indices=vacancy_indices,
        pixel_size_nm=sampling / 10.0,
    )
    gt.phase_map = phase_map
    gt.extra["material"] = "BaTiO3"
    gt.extra["zone_axis"] = "001"
    gt.extra["n_boundaries"] = idiff["n_boundaries"]
    gt.extra["vacancy_rate"] = vacancy_rate
    gt.extra["cluster_fraction"] = cluster_fraction
    gt.extra["boundary_sigma_px"] = boundary_sigma_px
    gt.extra["clustering"] = "boundary_proximity"

    # --- 5. Configure simulation ---
    config = SimConfig(
        scenario="batio3_interface",
        difficulty=difficulty,
        energy_eV=200e3,
        semiangle_cutoff=21.0,
        haadf_inner=68.0,
        haadf_outer=200.0,
        interpolation=2,
        device=device,
        sampling=sampling,
        slice_thickness=1.0,
        num_phonon_configs=diff["num_phonon_configs"],
        phonon_sigmas=BTO_PHONON_SIGMAS,
        tile_size_px=128,
        overlap_px=4,
        seed=seed,
    )

    # --- 6. Run STEM simulation ---
    if shard_id is not None and total_shards is not None:
        stem = str(output_path.with_suffix(''))
        npz_path = Path(f"{stem}_shard{shard_id}.npz")
        logger.info(f"Running shard {shard_id}/{total_shards}...")
        run_stem_simulation_shard(defected, config, shard_id, total_shards, npz_path)
        return output_path

    if merge_shards is not None:
        stem = str(output_path.with_suffix(''))
        shard_paths = [Path(f"{stem}_shard{i}.npz") for i in range(merge_shards)]
        logger.info(f"Merging {merge_shards} shards...")
        image = merge_simulation_shards(shard_paths)
    else:
        logger.info("Running STEM-HAADF simulation...")
        image = run_stem_simulation(defected, config)
    logger.info(f"Image shape: {image.shape}")

    # --- 7. Resize phase map to match actual image ---
    from scipy.ndimage import zoom as scipy_zoom
    if phase_map.shape != image.shape:
        scale_h = image.shape[0] / phase_map.shape[0]
        scale_w = image.shape[1] / phase_map.shape[1]
        gt.phase_map = scipy_zoom(
            phase_map, (scale_h, scale_w), order=0,
        ).astype(np.int32)

    # --- 8. Write HDF5 ---
    write_world_hdf5(output_path, image, gt, config)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate BaTiO3 interface scenario for STEMGym",
    )
    parser.add_argument(
        "--difficulty", choices=["easy", "medium", "hard"], default="medium",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fov-nm", type=float, default=100.0)
    add_shard_args(parser)
    parser.add_argument(
        "--all-difficulties", action="store_true",
        help="Generate all 3 difficulty levels",
    )
    args = parser.parse_args()

    if args.all_difficulties:
        for diff in ["easy", "medium", "hard"]:
            out = args.output or Path(f"data/simulated/batio3_interface_{diff}.h5")
            if args.output and diff != "medium":
                out = args.output.parent / f"batio3_interface_{diff}.h5"
            generate_batio3_interface(
                difficulty=diff, output_path=out,
                device=args.device, seed=args.seed, fov_nm=args.fov_nm,
                shard_id=args.shard, total_shards=args.total_shards,
                merge_shards=args.merge_shards,
            )
    else:
        generate_batio3_interface(
            difficulty=args.difficulty,
            output_path=args.output,
            device=args.device,
            seed=args.seed,
            fov_nm=args.fov_nm,
            shard_id=args.shard,
            total_shards=args.total_shards,
            merge_shards=args.merge_shards,
        )


if __name__ == "__main__":
    main()
