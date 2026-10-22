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
    build_supercell,
    record_ground_truth,
    run_stem_simulation,
    run_stem_simulation_shard,
    merge_simulation_shards,
    write_world_hdf5,
    add_shard_args,
    DIFFICULTY_PARAMS,
)


SI_A = 5.431
GE_A = 5.658


SIGE_PHONON_SIGMAS = {
    "Si": 0.078,
    "Ge": 0.085,
}


def build_si_supercell_110(fov_angstrom: float, z_thickness: float) -> Atoms:
    """Build a Si supercell oriented along [110] zone axis."""
    import abtem

    si = bulk("Si", "diamond", a=SI_A)


    si_110 = cut(si, a=(-1, 1, 0), b=(0, 0, 1), c=(1, 1, 0))


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
    """Replace Si atoms with Ge according to a composition gradient."""
    modified = atoms.copy()
    positions = modified.get_positions()
    symbols = list(modified.get_chemical_symbols())
    cell_x = modified.cell[0, 0]


    x_interface = interface_position * cell_x
    gradient_width = gradient_width_frac * cell_x


    substitution_indices = []
    n_si = sum(1 for s in symbols if s == "Si")

    for i, (pos, sym) in enumerate(zip(positions, symbols)):
        if sym != "Si":
            continue


        x = pos[0]

        if gradient_width > 0:
            t = (x - x_interface) / gradient_width
        else:
            t = np.inf if x > x_interface else -np.inf


        ge_frac = ge_max_fraction / (1.0 + np.exp(-4.0 * t))


        if rng.random() < ge_frac:
            symbols[i] = "Ge"
            substitution_indices.append(i)


            expansion = (GE_A - SI_A) / SI_A * ge_frac
            positions[i] *= (1.0 + expansion * 0.5)

    modified.set_chemical_symbols(symbols)
    modified.set_positions(positions)


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
    """Create a phase map for the Si/Ge interface."""
    h, w = shape_px
    phase_map = np.zeros(shape_px, dtype=np.int32)

    x_interface = int(interface_position * w)
    gradient_half = int(gradient_width_frac * w / 2)


    x_start = max(0, x_interface - gradient_half)
    x_end = min(w, x_interface + gradient_half)

    phase_map[:, x_start:x_end] = 1
    phase_map[:, x_end:] = 2

    return phase_map


QUADRANT_CENTER = np.array([0.75, 0.75])


CLUSTERED_DIFFICULTY = {
    "easy": {
        "ge_max": 0.8,
        "cluster_fraction": 0.60,
        "quadrant_sigma_frac": 0.20,
    },
    "medium": {
        "ge_max": 0.5,
        "cluster_fraction": 0.70,
        "quadrant_sigma_frac": 0.15,
    },
    "hard": {
        "ge_max": 0.3,
        "cluster_fraction": 0.80,
        "quadrant_sigma_frac": 0.10,
    },
}


def apply_ge_gradient_clustered(
    atoms,
    ge_max: float,
    cluster_fraction: float,
    quadrant_sigma_frac: float,
    rng: np.random.Generator,
) -> tuple:
    """Replace Si atoms with Ge using a 2D Gaussian concentration profile."""
    modified = atoms.copy()
    positions = modified.get_positions()
    symbols = list(modified.get_chemical_symbols())
    cell_x = modified.cell[0, 0]
    cell_y = modified.cell[1, 1]


    center_x = QUADRANT_CENTER[0] * cell_x
    center_y = QUADRANT_CENTER[1] * cell_y


    sigma = quadrant_sigma_frac * (cell_x + cell_y) / 2.0

    substitution_indices = []

    for i, (pos, sym) in enumerate(zip(positions, symbols)):
        if sym != "Si":
            continue


        dx = pos[0] - center_x
        dy = pos[1] - center_y
        d2 = dx**2 + dy**2


        gaussian = np.exp(-d2 / (2.0 * sigma**2))
        ge_frac = ge_max * (cluster_fraction * gaussian + (1.0 - cluster_fraction))


        if rng.random() < ge_frac:
            symbols[i] = "Ge"
            substitution_indices.append(i)

            expansion = (GE_A - SI_A) / SI_A * ge_frac
            positions[i] *= (1.0 + expansion * 0.5)

    modified.set_chemical_symbols(symbols)
    modified.set_positions(positions)


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
    """Create a phase map for the clustered Si/Ge structure."""
    h, w = shape_px
    yy, xx = np.mgrid[0:h, 0:w]


    fx = xx / w
    fy = yy / h


    dx = fx - QUADRANT_CENTER[0]
    dy = fy - QUADRANT_CENTER[1]
    d = np.sqrt(dx**2 + dy**2)

    phase_map = np.zeros(shape_px, dtype=np.int32)

    phase_map[d < 2.0 * quadrant_sigma_frac] = 1

    phase_map[d < quadrant_sigma_frac] = 2

    return phase_map


def generate_sige_gradient_clustered(
    difficulty: str = "medium",
    output_path: Path | None = None,
    device: str = "cpu",
    seed: int = 42,
    fov_nm: float = 50.0,
    shard_id: int | None = None,
    total_shards: int | None = None,
    merge_shards: int | None = None,
) -> Path:
    """Generate Si/Ge world with Ge concentrated in one quadrant."""
    if output_path is None:
        output_path = Path("data/simulated/sige_gradient_clustered.h5")

    rng = np.random.default_rng(seed)
    diff = DIFFICULTY_PARAMS[difficulty]
    cdiff = CLUSTERED_DIFFICULTY[difficulty]

    fov_angstrom = fov_nm * 10.0
    z_thickness = 80.0


    logger.info("Building Si [110] supercell...")
    supercell = build_si_supercell_110(fov_angstrom, z_thickness)
    logger.info(f"Supercell: {len(supercell)} atoms, cell={supercell.cell.lengths()}")

    pristine = supercell.copy()


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


    sampling = 0.05
    img_size = int(fov_angstrom / sampling)
    gt.phase_map = make_clustered_phase_map(
        (img_size, img_size),
        cdiff["quadrant_sigma_frac"],
    )


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

    if shard_id is not None and total_shards is not None:
        stem = str(output_path.with_suffix(''))
        npz_path = Path(f"{stem}_shard{shard_id}.npz")
        logger.info(f"Running shard {shard_id}/{total_shards}...")
        run_stem_simulation_shard(modified, config, shard_id, total_shards, npz_path)
        return output_path

    if merge_shards is not None:
        stem = str(output_path.with_suffix(''))
        shard_paths = [Path(f"{stem}_shard{i}.npz") for i in range(merge_shards)]
        logger.info(f"Merging {merge_shards} shards...")
        image = merge_simulation_shards(shard_paths)
    else:
        logger.info("Running STEM-HAADF simulation...")
        image = run_stem_simulation(modified, config)
    logger.info(f"Image shape: {image.shape}")


    from scipy.ndimage import zoom as scipy_zoom
    if gt.phase_map.shape != image.shape:
        scale_h = image.shape[0] / gt.phase_map.shape[0]
        scale_w = image.shape[1] / gt.phase_map.shape[1]
        gt.phase_map = scipy_zoom(
            gt.phase_map, (scale_h, scale_w), order=0,
        ).astype(np.int32)


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
    add_shard_args(parser)
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
                shard_id=args.shard, total_shards=args.total_shards,
                merge_shards=args.merge_shards,
            )
    else:
        generate_sige_gradient_clustered(
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
