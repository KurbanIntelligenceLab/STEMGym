#!/usr/bin/env python3
"""Generate Si/Ge gradient-clustered scenario: Ge concentrated in one quadrant.

Material: Si/Ge heterostructure, [110] zone axis, ~50nm FOV
Defects: Ge substitutions concentrated in quadrant (0.75, 0.75) instead of
         a smooth 1D sigmoid gradient — ~70% of Ge in ~25% of area
Task: Targeted Characterization (tests adaptive exploration vs raster)

Replaces the uniform sigmoid gradient with a 2D Gaussian-based Ge
concentration profile.  Each Si atom has a local ge_frac determined by
its distance from the quadrant center, then Si→Ge substitution is
stochastic based on that probability.

Usage:
    python data/scripts/generate_sige_gradient_clustered.py [--difficulty easy|medium|hard]
                                                             [--output data/simulated/sige_gradient_clustered.h5]
                                                             [--device cpu|gpu]
                                                             [--seed 42]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from sim_utils import (
    SimConfig,
    record_ground_truth,
    run_stem_simulation,
    write_world_hdf5,
    DIFFICULTY_PARAMS,
)
from generate_sige import (
    build_si_supercell_110,
    SI_A,
    GE_A,
    SIGE_PHONON_SIGMAS,
)

# Quadrant center in fractional coordinates
QUADRANT_CENTER = np.array([0.75, 0.75])

# Difficulty params for gradient-clustered SiGe
CLUSTERED_DIFFICULTY = {
    "easy": {
        "ge_max": 0.8,
        "cluster_fraction": 0.60,
        "quadrant_sigma_frac": 0.20,  # wide blob
    },
    "medium": {
        "ge_max": 0.5,
        "cluster_fraction": 0.70,
        "quadrant_sigma_frac": 0.15,
    },
    "hard": {
        "ge_max": 0.3,
        "cluster_fraction": 0.80,
        "quadrant_sigma_frac": 0.10,  # tight blob
    },
}


def apply_ge_gradient_clustered(
    atoms,
    ge_max: float,
    cluster_fraction: float,
    quadrant_sigma_frac: float,
    rng: np.random.Generator,
) -> tuple:
    """Replace Si atoms with Ge using a 2D Gaussian concentration profile.

    The Ge fraction at each Si atom is:
        ge_frac = ge_max * (cluster_fraction * exp(-d^2/2σ^2) + (1 - cluster_fraction))

    where d is the 2D distance from the quadrant center in fractional coords.

    Args:
        atoms: Si supercell.
        ge_max: Maximum Ge fraction (at the quadrant center).
        cluster_fraction: Fraction of Ge concentrated in the blob vs uniform.
        quadrant_sigma_frac: Gaussian sigma as fraction of cell size.
        rng: Random number generator.

    Returns:
        (modified_atoms, substitution_indices, ge_fraction_2d)
    """
    modified = atoms.copy()
    positions = modified.get_positions()
    symbols = list(modified.get_chemical_symbols())
    cell_x = modified.cell[0, 0]
    cell_y = modified.cell[1, 1]

    # Quadrant center in Angstrom
    center_x = QUADRANT_CENTER[0] * cell_x
    center_y = QUADRANT_CENTER[1] * cell_y

    # Sigma in Angstrom (average of x and y cell)
    sigma = quadrant_sigma_frac * (cell_x + cell_y) / 2.0

    substitution_indices = []

    for i, (pos, sym) in enumerate(zip(positions, symbols)):
        if sym != "Si":
            continue

        # 2D distance from quadrant center
        dx = pos[0] - center_x
        dy = pos[1] - center_y
        d2 = dx**2 + dy**2

        # Local Ge fraction: clustered Gaussian + uniform background
        gaussian = np.exp(-d2 / (2.0 * sigma**2))
        ge_frac = ge_max * (cluster_fraction * gaussian + (1.0 - cluster_fraction))

        # Stochastic substitution
        if rng.random() < ge_frac:
            symbols[i] = "Ge"
            substitution_indices.append(i)
            # Adjust position for lattice mismatch
            expansion = (GE_A - SI_A) / SI_A * ge_frac
            positions[i] *= (1.0 + expansion * 0.5)

    modified.set_chemical_symbols(symbols)
    modified.set_positions(positions)

    # Compute 2D Ge fraction map for ground truth
    n_bins = 50
    ge_map = np.zeros((n_bins, n_bins), dtype=np.float32)
    total_map = np.zeros((n_bins, n_bins), dtype=np.float32)
    for i, (pos, sym) in enumerate(zip(positions, symbols)):
        bx = min(int(pos[0] / cell_x * n_bins), n_bins - 1)
        by = min(int(pos[1] / cell_y * n_bins), n_bins - 1)
        total_map[by, bx] += 1
        if sym == "Ge":
            ge_map[by, bx] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        ge_fraction_2d = np.where(total_map > 0, ge_map / total_map, 0.0)

    return modified, np.array(substitution_indices), ge_fraction_2d.astype(np.float32)


def make_clustered_phase_map(
    shape_px: tuple[int, int],
    quadrant_sigma_frac: float,
) -> np.ndarray:
    """Create a phase map for the clustered Si/Ge structure.

    0 = pure Si, 1 = low-Ge transition, 2 = high-Ge core
    """
    h, w = shape_px
    yy, xx = np.mgrid[0:h, 0:w]

    # Fractional coordinates
    fx = xx / w
    fy = yy / h

    # Distance from quadrant center
    dx = fx - QUADRANT_CENTER[0]
    dy = fy - QUADRANT_CENTER[1]
    d = np.sqrt(dx**2 + dy**2)

    phase_map = np.zeros(shape_px, dtype=np.int32)
    # Transition zone: within 2 sigma
    phase_map[d < 2.0 * quadrant_sigma_frac] = 1
    # Core: within 1 sigma
    phase_map[d < quadrant_sigma_frac] = 2

    return phase_map


def generate_sige_gradient_clustered(
    difficulty: str = "medium",
    output_path: Path | None = None,
    device: str = "cpu",
    seed: int = 42,
    fov_nm: float = 50.0,
) -> Path:
    """Generate Si/Ge world with Ge concentrated in one quadrant.

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
        output_path = Path("data/simulated/sige_gradient_clustered.h5")

    rng = np.random.default_rng(seed)
    diff = DIFFICULTY_PARAMS[difficulty]
    cdiff = CLUSTERED_DIFFICULTY[difficulty]

    fov_angstrom = fov_nm * 10.0
    z_thickness = 80.0  # Slightly thinner for [110]

    # --- 1. Build Si supercell in [110] orientation ---
    logger.info("Building Si [110] supercell...")
    supercell = build_si_supercell_110(fov_angstrom, z_thickness)
    logger.info(f"Supercell: {len(supercell)} atoms, cell={supercell.cell.lengths()}")

    pristine = supercell.copy()

    # --- 2. Apply clustered Ge gradient ---
    logger.info(
        f"Applying clustered Ge gradient: ge_max={cdiff['ge_max']}, "
        f"cluster_frac={cdiff['cluster_fraction']:.0%}, "
        f"sigma={cdiff['quadrant_sigma_frac']}"
    )
    modified, sub_indices, ge_fraction_2d = apply_ge_gradient_clustered(
        supercell,
        ge_max=cdiff["ge_max"],
        cluster_fraction=cdiff["cluster_fraction"],
        quadrant_sigma_frac=cdiff["quadrant_sigma_frac"],
        rng=rng,
    )
    n_ge = sum(1 for s in modified.get_chemical_symbols() if s == "Ge")
    logger.info(f"Substituted {n_ge} Si → Ge ({len(sub_indices)} substitutions)")

    # --- 3. Record ground truth ---
    gt = record_ground_truth(
        pristine,
        substitution_indices=sub_indices,
        pixel_size_nm=0.005,
    )
    gt.extra["material"] = "Si_Ge"
    gt.extra["zone_axis"] = "110"
    gt.extra["ge_fraction_2d"] = ge_fraction_2d
    gt.extra["ge_max"] = cdiff["ge_max"]
    gt.extra["cluster_fraction"] = cdiff["cluster_fraction"]
    gt.extra["quadrant_sigma_frac"] = cdiff["quadrant_sigma_frac"]
    gt.extra["clustering"] = "quadrant_gaussian"

    # Phase map
    sampling = 0.05
    img_size = int(fov_angstrom / sampling)
    gt.phase_map = make_clustered_phase_map(
        (img_size, img_size),
        cdiff["quadrant_sigma_frac"],
    )

    # --- 4. Simulate ---
    config = SimConfig(
        scenario="sige_gradient_clustered",
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
        phonon_sigmas=SIGE_PHONON_SIGMAS,
        tile_size_px=128,
        overlap_px=4,
        seed=seed,
    )

    logger.info("Running STEM-HAADF simulation...")
    image = run_stem_simulation(modified, config)
    logger.info(f"Simulation complete: {image.shape}")

    # Resize phase map to match image
    from scipy.ndimage import zoom as scipy_zoom
    if gt.phase_map.shape != image.shape:
        scale_h = image.shape[0] / gt.phase_map.shape[0]
        scale_w = image.shape[1] / gt.phase_map.shape[1]
        gt.phase_map = scipy_zoom(
            gt.phase_map, (scale_h, scale_w), order=0,
        ).astype(np.int32)

    # --- 5. Write HDF5 ---
    write_world_hdf5(output_path, image, gt, config)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate Si/Ge gradient-clustered scenario for STEMGym",
    )
    parser.add_argument(
        "--difficulty", choices=["easy", "medium", "hard"], default="medium",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fov-nm", type=float, default=50.0)
    parser.add_argument(
        "--all-difficulties", action="store_true",
        help="Generate all 3 difficulty levels",
    )
    args = parser.parse_args()

    if args.all_difficulties:
        for diff in ["easy", "medium", "hard"]:
            out = args.output or Path(f"data/simulated/sige_gradient_clustered_{diff}.h5")
            if args.output and diff != "medium":
                out = args.output.parent / f"sige_gradient_clustered_{diff}.h5"
            generate_sige_gradient_clustered(
                difficulty=diff, output_path=out,
                device=args.device, seed=args.seed, fov_nm=args.fov_nm,
            )
    else:
        generate_sige_gradient_clustered(
            difficulty=args.difficulty,
            output_path=args.output,
            device=args.device,
            seed=args.seed,
            fov_nm=args.fov_nm,
        )


if __name__ == "__main__":
    main()
