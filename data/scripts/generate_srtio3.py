#!/usr/bin/env python3
"""Generate SrTiO3 scenario (S1): Defect Census task.

Material: SrTiO3 (mp-5229), [001] zone axis, ~100nm FOV
Defects: Oxygen vacancies at 1-5% depending on difficulty
Task: Defect Census (F1 score for vacancy detection)

Pipeline: MP structure → ASE → supercell → plant O vacancies →
         record ground truth → frozen phonons → PRISM → HDF5

Usage:
    python data/scripts/generate_srtio3.py [--difficulty easy|medium|hard]
                                           [--output data/simulated/srtio3_medium.h5]
                                           [--device cpu|gpu]
                                           [--mp-api-key KEY]
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

# Add parent to path for sim_utils
import sys
sys.path.insert(0, str(Path(__file__).parent))
from sim_utils import (
    SimConfig,
    build_supercell,
    fetch_mp_structure,
    plant_vacancies,
    record_ground_truth,
    run_stem_simulation,
    standardize_for_stem,
    write_world_hdf5,
    DIFFICULTY_PARAMS,
)

# SrTiO3 lattice parameter
STO_A = 3.905  # Angstrom, cubic perovskite

# Debye-Waller factors (RMS displacement in Angstrom) for SrTiO3
# From: Abramov et al., Acta Cryst. B 51, 942-951, 1995 (values at 296 K)
STO_PHONON_SIGMAS = {
    "Sr": 0.084,
    "Ti": 0.060,
    "O": 0.085,
}


def build_srtio3_unit_cell() -> Atoms:
    """Build SrTiO3 cubic perovskite unit cell.

    Layout (cubic, a=3.905 Å):
        Sr at corners: (0, 0, 0)
        Ti at body center: (0.5, 0.5, 0.5)
        O at face centers: (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)
    """
    a = STO_A
    atoms = Atoms(
        symbols=["Sr", "Ti", "O", "O", "O"],
        positions=[
            [0.0, 0.0, 0.0],           # Sr
            [a/2, a/2, a/2],            # Ti
            [a/2, a/2, 0.0],            # O (xy face)
            [a/2, 0.0, a/2],            # O (xz face)
            [0.0, a/2, a/2],            # O (yz face)
        ],
        cell=[a, a, a],
        pbc=True,
    )
    return atoms


def build_srtio3_from_mp(api_key: str | None = None) -> Atoms:
    """Fetch SrTiO3 from Materials Project (mp-5229)."""
    try:
        atoms = fetch_mp_structure("mp-5229", api_key=api_key)
        logger.info(f"Fetched SrTiO3 from MP: {len(atoms)} atoms")
        return atoms
    except Exception as e:
        logger.warning(f"Could not fetch from MP ({e}), using built-in unit cell")
        return build_srtio3_unit_cell()


def generate_srtio3(
    difficulty: str = "medium",
    output_path: Path | None = None,
    device: str = "cpu",
    mp_api_key: str | None = None,
    seed: int = 42,
    fov_nm: float = 100.0,
) -> Path:
    """Generate SrTiO3 world at specified difficulty.

    Args:
        difficulty: "easy" (5% vacancies, low noise), "medium" (3%), "hard" (1%, high noise).
        output_path: HDF5 output path.
        device: "cpu" or "gpu" for abTEM.
        mp_api_key: Materials Project API key.
        seed: Random seed.
        fov_nm: Field of view in nm.

    Returns:
        Path to generated HDF5 file.
    """
    if output_path is None:
        output_path = Path(f"data/simulated/srtio3_{difficulty}.h5")

    rng = np.random.default_rng(seed)
    diff = DIFFICULTY_PARAMS[difficulty]

    # --- 1. Get unit cell ---
    logger.info("Building SrTiO3 structure...")
    unit_cell = build_srtio3_from_mp(mp_api_key)

    # --- 2. Build supercell for ~fov_nm field of view ---
    # Convert FOV from nm to Angstrom
    fov_angstrom = fov_nm * 10.0
    # z thickness: ~10 nm is typical for STEM
    z_thickness = 100.0  # 10 nm in Angstrom

    supercell = build_supercell(
        unit_cell,
        target_size_angstrom=(fov_angstrom, fov_angstrom, z_thickness),
    )
    logger.info(f"Supercell: {len(supercell)} atoms, cell={supercell.cell.lengths()}")

    # Record pristine structure for ground truth
    pristine = supercell.copy()

    # --- 3. Plant oxygen vacancies ---
    vacancy_rate = diff["vacancy_rate"]
    logger.info(f"Planting O vacancies at {vacancy_rate*100:.1f}%...")
    defected, vacancy_indices = plant_vacancies(
        supercell, species="O", rate=vacancy_rate, rng=rng,
    )
    logger.info(f"Created {len(vacancy_indices)} O vacancies out of {len(supercell)} atoms")

    # --- 4. Record ground truth ---
    ground_truth = record_ground_truth(
        pristine,
        vacancy_indices=vacancy_indices,
        pixel_size_nm=0.005,
    )
    ground_truth.extra["vacancy_rate"] = vacancy_rate
    ground_truth.extra["material"] = "SrTiO3"
    ground_truth.extra["zone_axis"] = "001"

    # --- 5. Configure simulation ---
    config = SimConfig(
        scenario="srtio3",
        difficulty=difficulty,
        energy_eV=200e3,
        semiangle_cutoff=21.0,
        haadf_inner=68.0,
        haadf_outer=200.0,
        interpolation=2,
        device=device,
        sampling=0.05,  # ~0.005 nm/pixel
        slice_thickness=1.0,
        num_phonon_configs=diff["num_phonon_configs"],
        phonon_sigmas=STO_PHONON_SIGMAS,
        tile_size_px=128,
        overlap_px=4,
        seed=seed,
    )

    # --- 6. Run STEM simulation ---
    logger.info("Running STEM-HAADF simulation (this may take a while)...")
    image = run_stem_simulation(defected, config)
    logger.info(f"Simulation complete: image shape={image.shape}")

    # --- 7. Write HDF5 ---
    write_world_hdf5(output_path, image, ground_truth, config)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate SrTiO3 scenario for STEMGym")
    parser.add_argument(
        "--difficulty", choices=["easy", "medium", "hard"], default="medium",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--mp-api-key", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fov-nm", type=float, default=100.0)
    parser.add_argument(
        "--all-difficulties", action="store_true",
        help="Generate all 3 difficulty levels",
    )
    args = parser.parse_args()

    if args.all_difficulties:
        for diff in ["easy", "medium", "hard"]:
            out = args.output or Path(f"data/simulated/srtio3_{diff}.h5")
            if args.output and diff != "medium":
                out = args.output.parent / f"srtio3_{diff}.h5"
            generate_srtio3(
                difficulty=diff, output_path=out,
                device=args.device, mp_api_key=args.mp_api_key,
                seed=args.seed, fov_nm=args.fov_nm,
            )
    else:
        generate_srtio3(
            difficulty=args.difficulty,
            output_path=args.output,
            device=args.device,
            mp_api_key=args.mp_api_key,
            seed=args.seed,
            fov_nm=args.fov_nm,
        )


if __name__ == "__main__":
    main()
