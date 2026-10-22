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


STO_A = 3.905


STO_PHONON_SIGMAS = {
    "Sr": 0.084,
    "Ti": 0.060,
    "O": 0.085,
}


def build_srtio3_unit_cell() -> Atoms:
    """Build SrTiO3 cubic perovskite unit cell."""
    a = STO_A
    atoms = Atoms(
        symbols=["Sr", "Ti", "O", "O", "O"],
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
    """Compute per-O-atom weights based on distance to diagonal grain boundary."""
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

    weights = np.exp(-cross**2 / (2.0 * sigma**2))
    return weights


def generate_srtio3_clustered(
    difficulty: str = "medium",
    output_path: Path | None = None,
    device: str = "cpu",
    mp_api_key: str | None = None,
    seed: int = 42,
    fov_nm: float = 100.0,
    shard_id: int | None = None,
    total_shards: int | None = None,
    merge_shards: int | None = None,
) -> Path:
    """Generate SrTiO3 world with clustered O vacancies along a grain boundary."""
    if output_path is None:
        output_path = Path("data/simulated/srtio3_clustered.h5")

    rng = np.random.default_rng(seed)
    diff = DIFFICULTY_PARAMS[difficulty]
    cdiff = CLUSTERED_DIFFICULTY[difficulty]


    logger.info("Building SrTiO3 structure...")
    unit_cell = build_srtio3_from_mp(mp_api_key)


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
    logger.info(f"Created {len(vacancy_indices)} O vacancies out of {len(supercell)} atoms")


    ground_truth = record_ground_truth(
        pristine,
        vacancy_indices=vacancy_indices,
        pixel_size_nm=0.005,
    )
    ground_truth.extra["vacancy_rate"] = vacancy_rate
    ground_truth.extra["cluster_fraction"] = cluster_fraction
    ground_truth.extra["line_width_frac"] = line_width_frac
    ground_truth.extra["material"] = "SrTiO3"
    ground_truth.extra["zone_axis"] = "001"
    ground_truth.extra["clustering"] = "grain_boundary"


    config = SimConfig(
        scenario="srtio3_clustered",
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
        phonon_sigmas=STO_PHONON_SIGMAS,
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
        description="Generate SrTiO3 clustered-defect scenario for STEMGym",
    )
    parser.add_argument(
        "--difficulty", choices=["easy", "medium", "hard"], default="medium",
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
            out = args.output or Path(f"data/simulated/srtio3_clustered_{diff}.h5")
            if args.output and diff != "medium":
                out = args.output.parent / f"srtio3_clustered_{diff}.h5"
            generate_srtio3_clustered(
                difficulty=diff, output_path=out,
                device=args.device, mp_api_key=args.mp_api_key,
                seed=args.seed, fov_nm=args.fov_nm,
                shard_id=args.shard, total_shards=args.total_shards,
                merge_shards=args.merge_shards,
            )
    else:
        generate_srtio3_clustered(
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
