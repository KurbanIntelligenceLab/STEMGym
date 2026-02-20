#!/usr/bin/env python3
"""Generate Si/Ge scenario (S6): Targeted Characterization task.

Material: Si/Ge heterostructure (mp-149 Si + mp-32 Ge), [110] zone axis, ~50nm FOV
Defects: Interface + Ge composition gradient
Task: Targeted Characterization (find interface, measure Ge profile)

The world has a Si substrate on one side and gradually increasing Ge
concentration across an interface region. The agent must locate the
interface and characterize the composition profile.

Pipeline: Build Si supercell → add Ge gradient → rotate to [110] →
         record ground truth → frozen phonons → PRISM → HDF5

Usage:
    python data/scripts/generate_sige.py [--difficulty easy|medium|hard]
                                         [--output data/simulated/sige_medium.h5]
                                         [--device cpu|gpu]
                                         [--seed 42]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk, cut

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from sim_utils import (
    SimConfig,
    GroundTruth,
    build_supercell,
    record_ground_truth,
    run_stem_simulation,
    write_world_hdf5,
    DIFFICULTY_PARAMS,
)

# Lattice parameters
SI_A = 5.431  # Angstrom, diamond cubic
GE_A = 5.658  # Angstrom, diamond cubic

# Debye-Waller factors
SIGE_PHONON_SIGMAS = {
    "Si": 0.078,
    "Ge": 0.065,
}

# Difficulty controls interface sharpness and Ge gradient
SIGE_DIFFICULTY = {
    "easy": {
        "gradient_width_frac": 0.3,    # Wide gradient (easy to detect)
        "ge_max_fraction": 0.8,         # High Ge at edge
        "interface_position": 0.5,      # Middle of FOV
    },
    "medium": {
        "gradient_width_frac": 0.2,
        "ge_max_fraction": 0.5,
        "interface_position": 0.5,
    },
    "hard": {
        "gradient_width_frac": 0.1,    # Sharp, narrow gradient
        "ge_max_fraction": 0.3,         # Low Ge concentration
        "interface_position": 0.35,     # Off-center
    },
}


def build_si_supercell_110(fov_angstrom: float, z_thickness: float) -> Atoms:
    """Build a Si supercell oriented along [110] zone axis.

    For [110] viewing:
    - x along [-1,1,0]
    - y along [0,0,1]
    - z along [1,1,0] (beam direction)
    """
    import abtem

    si = bulk("Si", "diamond", a=SI_A)

    # Cut to get [110] orientation: z along [110]
    # a=[-1,1,0], b=[0,0,1], c=[1,1,0]
    si_110 = cut(si, a=(-1, 1, 0), b=(0, 0, 1), c=(1, 1, 0))

    # Build supercell
    supercell = build_supercell(
        si_110,
        target_size_angstrom=(fov_angstrom, fov_angstrom, z_thickness),
    )
    return supercell


def apply_ge_gradient(
    atoms: Atoms,
    interface_position: float,
    gradient_width_frac: float,
    ge_max_fraction: float,
    rng: np.random.Generator,
) -> tuple[Atoms, np.ndarray, np.ndarray]:
    """Replace Si atoms with Ge according to a composition gradient.

    The Ge fraction increases from 0 to ge_max_fraction as x increases
    past the interface position, with a smooth sigmoid transition.

    Args:
        atoms: Si supercell.
        interface_position: Position of interface as fraction of cell x [0, 1].
        gradient_width_frac: Width of gradient as fraction of cell x.
        ge_max_fraction: Maximum Ge fraction at the far edge.
        rng: Random number generator.

    Returns:
        (modified_atoms, substitution_indices, ge_fraction_profile)
    """
    modified = atoms.copy()
    positions = modified.get_positions()
    symbols = list(modified.get_chemical_symbols())
    cell_x = modified.cell[0, 0]

    # Interface position in Angstrom
    x_interface = interface_position * cell_x
    gradient_width = gradient_width_frac * cell_x

    # Compute Ge fraction at each atom position based on x
    substitution_indices = []
    n_si = sum(1 for s in symbols if s == "Si")

    for i, (pos, sym) in enumerate(zip(positions, symbols)):
        if sym != "Si":
            continue

        # Sigmoid-like Ge fraction
        x = pos[0]
        # Distance past interface, normalized by gradient width
        if gradient_width > 0:
            t = (x - x_interface) / gradient_width
        else:
            t = np.inf if x > x_interface else -np.inf

        # Sigmoid: 0 for t << 0, ge_max for t >> 0
        ge_frac = ge_max_fraction / (1.0 + np.exp(-4.0 * t))

        # Stochastic substitution
        if rng.random() < ge_frac:
            symbols[i] = "Ge"
            substitution_indices.append(i)
            # Adjust position slightly for Ge lattice mismatch
            # Ge is ~4% larger than Si
            expansion = (GE_A - SI_A) / SI_A * ge_frac
            positions[i] *= (1.0 + expansion * 0.5)

    modified.set_chemical_symbols(symbols)
    modified.set_positions(positions)

    # Compute Ge fraction profile (1D along x)
    n_bins = 100
    bin_edges = np.linspace(0, cell_x, n_bins + 1)
    ge_profile = np.zeros(n_bins)
    total_profile = np.zeros(n_bins)
    for i, (pos, sym) in enumerate(zip(positions, symbols)):
        bin_idx = min(int(pos[0] / cell_x * n_bins), n_bins - 1)
        total_profile[bin_idx] += 1
        if sym == "Ge":
            ge_profile[bin_idx] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        ge_fraction = np.where(total_profile > 0, ge_profile / total_profile, 0.0)

    return modified, np.array(substitution_indices), ge_fraction.astype(np.float32)


def make_interface_phase_map(
    shape_px: tuple[int, int],
    interface_position: float,
    gradient_width_frac: float,
) -> np.ndarray:
    """Create a phase map for the Si/Ge interface.

    0 = pure Si, 1 = interface/gradient region, 2 = Ge-rich
    """
    h, w = shape_px
    phase_map = np.zeros(shape_px, dtype=np.int32)

    x_interface = int(interface_position * w)
    gradient_half = int(gradient_width_frac * w / 2)

    # Interface region
    x_start = max(0, x_interface - gradient_half)
    x_end = min(w, x_interface + gradient_half)

    phase_map[:, x_start:x_end] = 1
    phase_map[:, x_end:] = 2

    return phase_map


def generate_sige(
    difficulty: str = "medium",
    output_path: Path | None = None,
    device: str = "cpu",
    seed: int = 42,
    fov_nm: float = 50.0,
) -> Path:
    """Generate Si/Ge heterostructure world."""
    if output_path is None:
        output_path = Path(f"data/simulated/sige_{difficulty}.h5")

    rng = np.random.default_rng(seed)
    diff = DIFFICULTY_PARAMS[difficulty]
    sige_diff = SIGE_DIFFICULTY[difficulty]

    fov_angstrom = fov_nm * 10.0
    z_thickness = 80.0  # Slightly thinner for [110]

    # --- 1. Build Si supercell in [110] orientation ---
    logger.info("Building Si [110] supercell...")
    supercell = build_si_supercell_110(fov_angstrom, z_thickness)
    logger.info(f"Supercell: {len(supercell)} atoms, cell={supercell.cell.lengths()}")

    # Record pristine
    pristine = supercell.copy()

    # --- 2. Apply Ge gradient ---
    logger.info("Applying Ge composition gradient...")
    modified, sub_indices, ge_profile = apply_ge_gradient(
        supercell,
        interface_position=sige_diff["interface_position"],
        gradient_width_frac=sige_diff["gradient_width_frac"],
        ge_max_fraction=sige_diff["ge_max_fraction"],
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
    gt.extra["ge_fraction_profile"] = ge_profile
    gt.extra["interface_position"] = sige_diff["interface_position"]
    gt.extra["gradient_width"] = sige_diff["gradient_width_frac"]

    # Phase map
    sampling = 0.05
    img_size = int(fov_angstrom / sampling)
    gt.phase_map = make_interface_phase_map(
        (img_size, img_size),
        sige_diff["interface_position"],
        sige_diff["gradient_width_frac"],
    )

    # --- 4. Simulate ---
    config = SimConfig(
        scenario="sige",
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
        gt.phase_map = scipy_zoom(gt.phase_map, (scale_h, scale_w), order=0).astype(np.int32)

    # --- 5. Write HDF5 ---
    write_world_hdf5(output_path, image, gt, config)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate Si/Ge scenario for STEMGym")
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
            out = args.output or Path(f"data/simulated/sige_{diff}.h5")
            if args.output and diff != "medium":
                out = args.output.parent / f"sige_{diff}.h5"
            generate_sige(
                difficulty=diff, output_path=out,
                device=args.device, seed=args.seed, fov_nm=args.fov_nm,
            )
    else:
        generate_sige(
            difficulty=args.difficulty,
            output_path=args.output,
            device=args.device,
            seed=args.seed,
            fov_nm=args.fov_nm,
        )


if __name__ == "__main__":
    main()
