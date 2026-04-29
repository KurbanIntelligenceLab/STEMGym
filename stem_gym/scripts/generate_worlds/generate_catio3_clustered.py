#!/usr/bin/env python3
"""Generate CaTiO3 clustered-defect scenario: O vacancies along a grain boundary.

Material: CaTiO3 (mp-5827, cubic Pm-3m), [001] zone axis, ~100 nm FOV.
Defects: O vacancies concentrated along a diagonal grain boundary.
Task: Defect Census + cross-material transfer (third perovskite for the
      transfer matrix, complementing SrTiO3 and BaTiO3).

The cubic high-symmetry phase is used for consistency with the existing
SrTiO3/BaTiO3 generators (both use cubic phases). At room temperature the
thermodynamically stable phase is orthorhombic Pnma (mp-4019), which would
require a substantially larger supercell — out of scope for the rebuttal
experiment.

Phonon sigmas are interpolated from SrTiO3/BaTiO3 (Sr2+ ionic radius 1.18 A
and Ba2+ 1.35 A bracket Ca2+ 1.00 A; Ca leans toward Sr). For first-principles
phonon calculations on CaTiO3 see Cockayne & Burton, Phys. Rev. B 62 (2000)
3735 ("Phonons and static dielectric constant in CaTiO3 from first
principles").

Usage:
    python stem_gym/scripts/generate_worlds/generate_catio3_clustered.py \
        [--difficulty easy|medium|hard] \
        [--output data/simulated/catio3_clustered.h5] \
        [--device cpu|gpu] \
        [--mp-api-key KEY] \
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
    build_supercell,
    fetch_mp_structure,
    plant_vacancies_weighted,
    record_ground_truth,
    run_stem_simulation,
    run_stem_simulation_shard,
    merge_simulation_shards,
    write_world_hdf5,
    add_shard_args,
    DIFFICULTY_PARAMS,
)

# CaTiO3 cubic lattice parameter (Pm-3m, mp-5827). High-temperature phase;
# room-temperature CaTiO3 is orthorhombic Pnma (a~5.38, b~7.64, c~5.44 A).
CTO_A = 3.84  # Angstrom

# Debye-Waller factors (RMS displacement in Angstrom) for cubic CaTiO3.
# Interpolated from STO (Abramov 1995) and BTO (Kwei 1993): Ca2+ ionic
# radius (1.00 A) sits between Sr2+ (1.18 A) and Ba2+ (1.35 A) and is
# closer to Sr, so values track SrTiO3 with a slight downward shift on Ca.
# Ti and O sigmas track the cubic-perovskite norm.
CTO_PHONON_SIGMAS = {
    "Ca": 0.077,
    "Ti": 0.063,
    "O": 0.087,
}


def build_catio3_unit_cell() -> Atoms:
    """Build cubic CaTiO3 (Pm-3m) unit cell.

    Layout (a = 3.84 A):
        Ca at corner: (0, 0, 0)
        Ti at body center: (0.5, 0.5, 0.5)
        O at face centers: (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)
    """
    a = CTO_A
    return Atoms(
        symbols=["Ca", "Ti", "O", "O", "O"],
        positions=[
            [0.0, 0.0, 0.0],
            [a / 2, a / 2, a / 2],
            [a / 2, a / 2, 0.0],
            [a / 2, 0.0, a / 2],
            [0.0, a / 2, a / 2],
        ],
        cell=[a, a, a],
        pbc=True,
    )


def build_catio3_from_mp(api_key: str | None = None) -> Atoms:
    """Fetch cubic CaTiO3 from Materials Project (mp-5827).

    Falls back to the built-in unit cell if MP is unreachable or returns
    a non-cubic polymorph (e.g., the orthorhombic mp-4019 entry).
    """
    try:
        atoms = fetch_mp_structure("mp-5827", api_key=api_key)
        logger.info(f"Fetched CaTiO3 from MP (mp-5827): {len(atoms)} atoms")
        return atoms
    except Exception as e:
        logger.warning(
            f"Could not fetch from MP ({e}), using built-in cubic unit cell"
        )
        return build_catio3_unit_cell()


# Same clustered-vacancy schedule as SrTiO3/BaTiO3 for transfer-matrix
# comparability.
CLUSTERED_DIFFICULTY = {
    "easy":   {"vacancy_rate": 0.05, "cluster_fraction": 0.60, "line_width_frac": 0.08},
    "medium": {"vacancy_rate": 0.03, "cluster_fraction": 0.70, "line_width_frac": 0.05},
    "hard":   {"vacancy_rate": 0.01, "cluster_fraction": 0.80, "line_width_frac": 0.03},
}

GB_START = np.array([0.15, 0.35])
GB_END = np.array([0.85, 0.65])


def compute_grain_boundary_weights(
    atoms,
    line_width_frac: float,
) -> np.ndarray:
    """Per-O-atom Gaussian weights from a diagonal grain boundary.

    Mirrors ``compute_grain_boundary_weights`` in the SrTiO3 generator;
    duplicated to keep this script self-contained in line with the existing
    generator-per-material convention.
    """
    positions = atoms.get_positions()
    symbols = np.array(atoms.get_chemical_symbols())
    o_mask = symbols == "O"
    o_positions = positions[o_mask]

    cell_x = atoms.cell[0, 0]
    cell_y = atoms.cell[1, 1]

    p0 = GB_START * np.array([cell_x, cell_y])
    p1 = GB_END * np.array([cell_x, cell_y])

    line_vec = p1 - p0
    line_len = np.linalg.norm(line_vec)
    line_dir = line_vec / line_len

    dp = o_positions[:, :2] - p0
    cross = np.abs(dp[:, 0] * line_dir[1] - dp[:, 1] * line_dir[0])

    diag_len = np.sqrt(cell_x**2 + cell_y**2)
    sigma = line_width_frac * diag_len

    return np.exp(-cross**2 / (2.0 * sigma**2))


def generate_catio3_clustered(
    difficulty: str = "easy",
    output_path: Path | None = None,
    device: str = "cpu",
    mp_api_key: str | None = None,
    seed: int = 42,
    fov_nm: float = 100.0,
    shard_id: int | None = None,
    total_shards: int | None = None,
    merge_shards: int | None = None,
) -> Path:
    """Generate a CaTiO3 world with clustered O vacancies along a grain boundary."""
    if output_path is None:
        output_path = Path("data/simulated/catio3_clustered.h5")

    rng = np.random.default_rng(seed)
    diff = DIFFICULTY_PARAMS[difficulty]
    cdiff = CLUSTERED_DIFFICULTY[difficulty]

    logger.info("Building CaTiO3 structure...")
    unit_cell = build_catio3_from_mp(mp_api_key)

    fov_angstrom = fov_nm * 10.0
    z_thickness = 100.0

    supercell = build_supercell(
        unit_cell,
        target_size_angstrom=(fov_angstrom, fov_angstrom, z_thickness),
    )
    logger.info(f"Supercell: {len(supercell)} atoms, cell={supercell.cell.lengths()}")

    pristine = supercell.copy()

    vacancy_rate = cdiff["vacancy_rate"]
    cluster_fraction = cdiff["cluster_fraction"]
    line_width_frac = cdiff["line_width_frac"]

    weights = compute_grain_boundary_weights(supercell, line_width_frac)
    logger.info(
        f"Planting clustered O vacancies: rate={vacancy_rate*100:.1f}%, "
        f"cluster_frac={cluster_fraction:.0%}, line_width={line_width_frac}"
    )

    defected, vacancy_indices = plant_vacancies_weighted(
        supercell, species="O", rate=vacancy_rate,
        weights=weights, rng=rng, cluster_fraction=cluster_fraction,
    )
    logger.info(
        f"Created {len(vacancy_indices)} O vacancies out of {len(supercell)} atoms"
    )

    ground_truth = record_ground_truth(
        pristine,
        vacancy_indices=vacancy_indices,
        pixel_size_nm=0.005,
    )
    ground_truth.extra["vacancy_rate"] = vacancy_rate
    ground_truth.extra["cluster_fraction"] = cluster_fraction
    ground_truth.extra["line_width_frac"] = line_width_frac
    ground_truth.extra["material"] = "CaTiO3"
    ground_truth.extra["zone_axis"] = "001"
    ground_truth.extra["clustering"] = "grain_boundary"
    ground_truth.extra["mp_id"] = "mp-5827"
    ground_truth.extra["space_group"] = "Pm-3m"

    config = SimConfig(
        scenario="catio3_clustered",
        difficulty=difficulty,
        energy_eV=200e3,
        semiangle_cutoff=21.0,
        haadf_inner=68.0,
        haadf_outer=200.0,
        interpolation=2,
        device=device,
        sampling=0.05,
        slice_thickness=1.0,
        num_phonon_configs=diff["num_phonon_configs"],
        phonon_sigmas=CTO_PHONON_SIGMAS,
        tile_size_px=128,
        overlap_px=4,
        seed=seed,
    )

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
        logger.info("Running STEM-HAADF simulation (this may take a while)...")
        image = run_stem_simulation(defected, config)
    logger.info(f"Image shape: {image.shape}")

    write_world_hdf5(output_path, image, ground_truth, config)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate CaTiO3 clustered-defect scenario for STEMGym",
    )
    parser.add_argument(
        "--difficulty", choices=["easy", "medium", "hard"], default="easy",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--mp-api-key", type=str, default=None)
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
            out = args.output or Path(f"data/simulated/catio3_clustered_{diff}.h5")
            if args.output and diff != "easy":
                out = args.output.parent / f"catio3_clustered_{diff}.h5"
            generate_catio3_clustered(
                difficulty=diff, output_path=out,
                device=args.device, mp_api_key=args.mp_api_key,
                seed=args.seed, fov_nm=args.fov_nm,
                shard_id=args.shard, total_shards=args.total_shards,
                merge_shards=args.merge_shards,
            )
    else:
        generate_catio3_clustered(
            difficulty=args.difficulty,
            output_path=args.output,
            device=args.device,
            mp_api_key=args.mp_api_key,
            seed=args.seed,
            fov_nm=args.fov_nm,
            shard_id=args.shard,
            total_shards=args.total_shards,
            merge_shards=args.merge_shards,
        )


if __name__ == "__main__":
    main()
