#!/usr/bin/env python3
"""Generate BaTiO3 scenario (S4): Phase Mapping task.

Material: BaTiO3 (mp-5986), [001] zone axis, ~100nm FOV
Defects: Cubic/tetragonal phase boundaries
Task: Phase Mapping (IoU score)

The world is split into cubic and tetragonal BaTiO3 regions with a
phase boundary. The agent must identify which regions are cubic vs
tetragonal by acquiring and analyzing tiles.

Pipeline: Build cubic + tetragonal supercells → stitch along boundary →
         record phase map ground truth → frozen phonons → PRISM → HDF5

Usage:
    python data/scripts/generate_batio3.py [--difficulty easy|medium|hard]
                                           [--output data/simulated/batio3_medium.h5]
                                           [--device cpu|gpu]
                                           [--seed 42]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from ase import Atoms

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

# Difficulty controls boundary complexity
# easy: straight vertical boundary, clear contrast
# medium: slightly wavy boundary
# hard: irregular boundary, subtle contrast
PHASE_DIFFICULTY = {
    "easy": {"boundary_waviness": 0.0, "n_boundaries": 1},
    "medium": {"boundary_waviness": 0.05, "n_boundaries": 2},
    "hard": {"boundary_waviness": 0.10, "n_boundaries": 3},
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


def generate_batio3(
    difficulty: str = "medium",
    output_path: Path | None = None,
    device: str = "cpu",
    seed: int = 42,
    fov_nm: float = 100.0,
) -> Path:
    """Generate BaTiO3 phase mapping world."""
    if output_path is None:
        output_path = Path(f"data/simulated/batio3_{difficulty}.h5")

    rng = np.random.default_rng(seed)
    diff = DIFFICULTY_PARAMS[difficulty]
    phase_diff = PHASE_DIFFICULTY[difficulty]

    fov_angstrom = fov_nm * 10.0
    z_thickness = 100.0

    # --- 1. Create phase boundary map ---
    # Estimate image size from FOV and sampling
    sampling = 0.05  # Angstrom/pixel
    img_size = int(fov_angstrom / sampling)
    logger.info(f"Expected image size: ~{img_size}x{img_size} pixels")

    phase_map = make_phase_boundary_mask(
        (img_size, img_size),
        n_boundaries=phase_diff["n_boundaries"],
        waviness=phase_diff["boundary_waviness"],
        rng=rng,
    )
    logger.info(
        f"Phase map: {np.sum(phase_map == 0)} cubic pixels, "
        f"{np.sum(phase_map == 1)} tetragonal pixels"
    )

    # --- 2. Build composite structure ---
    logger.info("Building composite cubic/tetragonal structure...")
    composite, atom_phases = build_composite_structure(
        fov_angstrom, z_thickness, phase_map, sampling,
    )
    logger.info(f"Composite: {len(composite)} atoms")

    # Record pristine for ground truth
    pristine = composite.copy()

    # --- 3. Record ground truth ---
    gt = record_ground_truth(pristine, pixel_size_nm=sampling / 10.0)
    gt.phase_map = phase_map
    gt.extra["material"] = "BaTiO3"
    gt.extra["zone_axis"] = "001"
    gt.extra["n_boundaries"] = phase_diff["n_boundaries"]

    # --- 4. Simulate ---
    config = SimConfig(
        scenario="batio3",
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

    logger.info("Running STEM-HAADF simulation...")
    image = run_stem_simulation(composite, config)
    logger.info(f"Simulation complete: {image.shape}")

    # --- 5. Resize phase map to match actual image ---
    from scipy.ndimage import zoom as scipy_zoom
    if phase_map.shape != image.shape:
        scale_h = image.shape[0] / phase_map.shape[0]
        scale_w = image.shape[1] / phase_map.shape[1]
        gt.phase_map = scipy_zoom(phase_map, (scale_h, scale_w), order=0).astype(np.int32)

    # --- 6. Write HDF5 ---
    write_world_hdf5(output_path, image, gt, config)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate BaTiO3 scenario for STEMGym")
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
            out = args.output or Path(f"data/simulated/batio3_{diff}.h5")
            if args.output and diff != "medium":
                out = args.output.parent / f"batio3_{diff}.h5"
            generate_batio3(
                difficulty=diff, output_path=out,
                device=args.device, seed=args.seed, fov_nm=args.fov_nm,
            )
    else:
        generate_batio3(
            difficulty=args.difficulty,
            output_path=args.output,
            device=args.device,
            seed=args.seed,
            fov_nm=args.fov_nm,
        )


if __name__ == "__main__":
    main()
