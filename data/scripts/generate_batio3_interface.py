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
    python data/scripts/generate_batio3_interface.py [--difficulty easy|medium|hard]
                                                      [--output data/simulated/batio3_interface.h5]
                                                      [--device cpu|gpu]
                                                      [--seed 42]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from sim_utils import (
    SimConfig,
    plant_vacancies_weighted,
    record_ground_truth,
    run_stem_simulation,
    write_world_hdf5,
    DIFFICULTY_PARAMS,
)
from generate_batio3 import (
    make_phase_boundary_mask,
    build_composite_structure,
    BTO_PHONON_SIGMAS,
)

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
    logger.info("Running STEM-HAADF simulation...")
    image = run_stem_simulation(defected, config)
    logger.info(f"Simulation complete: {image.shape}")

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
            )
    else:
        generate_batio3_interface(
            difficulty=args.difficulty,
            output_path=args.output,
            device=args.device,
            seed=args.seed,
            fov_nm=args.fov_nm,
        )


if __name__ == "__main__":
    main()
