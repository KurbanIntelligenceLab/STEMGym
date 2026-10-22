from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from ase import Atoms
from scipy.spatial.transform import Rotation

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from sim_utils import (
    SimConfig,
    GroundTruth,
    run_stem_simulation,
    run_stem_simulation_shard,
    merge_simulation_shards,
    write_world_hdf5,
    add_shard_args,
    DIFFICULTY_PARAMS,
)


PT_A = 3.9242


PT_SURFACES = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
PT_ENERGIES = [1.81, 1.72, 1.49]


PT_PHONON_SIGMAS = {
    "Pt": 0.070,
    "C":  0.080,
}


MORPH_SINGLE_CRYSTAL = 0
MORPH_ICOSAHEDRAL    = 1
MORPH_DECAHEDRAL     = 2
MORPH_NAMES = {0: "single_crystal", 1: "icosahedral", 2: "decahedral"}


PT_DIFFICULTY = {
    "easy": {
        "n_particles":       8,
        "diameter_mean_nm":  4.0,
        "diameter_sigma_nm": 0.8,
        "diameter_min_nm":   2.5,
        "morph_weights":     [0.80, 0.10, 0.10],
        "min_sep_nm":        10.0,
        "support_thick_nm":  3.0,
    },
    "medium": {
        "n_particles":       20,
        "diameter_mean_nm":  3.0,
        "diameter_sigma_nm": 1.0,
        "diameter_min_nm":   1.5,
        "morph_weights":     [0.50, 0.25, 0.25],
        "min_sep_nm":        5.0,
        "support_thick_nm":  5.0,
    },
    "hard": {
        "n_particles":       40,
        "diameter_mean_nm":  1.8,
        "diameter_sigma_nm": 0.8,
        "diameter_min_nm":   0.8,
        "morph_weights":     [0.30, 0.30, 0.40],
        "min_sep_nm":        2.0,
        "support_thick_nm":  8.0,
    },
}


def _natoms_from_diameter(d_nm: float) -> int:
    """Approximate atom count for an FCC sphere of given diameter."""
    d_ang = d_nm * 10.0
    v_sphere = (4.0 / 3.0) * np.pi * (d_ang / 2.0) ** 3
    v_atom = PT_A ** 3 / 4.0
    return max(13, int(round(v_sphere / v_atom)))


def _measure_diameter(particle: Atoms) -> float:
    """Bounding-sphere diameter of a particle, in nm."""
    pos = particle.get_positions()
    radii = np.linalg.norm(pos - pos.mean(axis=0), axis=1)
    return (2.0 * radii.max() + 2.8) / 10.0


def make_wulff_particle(n_target: int) -> Atoms:
    """Single-crystal Pt via Wulff construction (truncated octahedron)."""
    from ase.cluster import wulff_construction
    p = wulff_construction(
        "Pt", PT_SURFACES, PT_ENERGIES,
        size=n_target, structure="fcc",
        rounding="closest", latticeconstant=PT_A,
    )
    p.positions -= p.positions.mean(axis=0)
    return p


def make_icosahedral_particle(n_target: int) -> Atoms:
    """Multiply-twinned Mackay icosahedron."""
    from ase.cluster import Icosahedron
    best, best_diff = 2, float("inf")
    for k in range(2, 12):
        n = (10 * k**3 + 15 * k**2 + 11 * k + 3) // 3
        if abs(n - n_target) < best_diff:
            best, best_diff = k, abs(n - n_target)
        if n > 2 * n_target:
            break
    p = Icosahedron("Pt", noshells=best, latticeconstant=PT_A)
    p.positions -= p.positions.mean(axis=0)
    return p


def make_decahedral_particle(n_target: int) -> Atoms:
    """Multiply-twinned Marks decahedron."""
    from ase.cluster import Decahedron
    best_p, best_diff = 2, float("inf")
    for pp in range(2, 10):
        qq = max(1, pp // 2)
        try:
            test = Decahedron("Pt", p=pp, q=qq, r=0, latticeconstant=PT_A)
            if abs(len(test) - n_target) < best_diff:
                best_p, best_diff = pp, abs(len(test) - n_target)
            if len(test) > 2 * n_target:
                break
        except Exception:
            continue
    qq = max(1, best_p // 2)
    p = Decahedron("Pt", p=best_p, q=qq, r=0, latticeconstant=PT_A)
    p.positions -= p.positions.mean(axis=0)
    return p


_BUILDERS = [make_wulff_particle, make_icosahedral_particle, make_decahedral_particle]
_LABELS   = [MORPH_SINGLE_CRYSTAL, MORPH_ICOSAHEDRAL, MORPH_DECAHEDRAL]


def make_amorphous_carbon(
    lx: float, ly: float, thickness: float,
    density_gcc: float = 2.0,
    rng: np.random.Generator | None = None,
) -> Atoms:
    """Generate an amorphous carbon slab by random placement."""
    if rng is None:
        rng = np.random.default_rng(42)


    n_density = density_gcc * 6.022e23 / 12.011 * 1e-24
    volume = lx * ly * thickness
    n_atoms = int(n_density * volume)

    logger.info(
        f"Generating amorphous carbon: {lx:.0f}×{ly:.0f}×{thickness:.0f} Å, "
        f"ρ={density_gcc} g/cm³, {n_atoms} atoms"
    )


    grid_a = (1.0 / n_density) ** (1.0 / 3.0)

    nx = max(1, int(np.ceil(lx / grid_a)))
    ny = max(1, int(np.ceil(ly / grid_a)))
    nz = max(1, int(np.ceil(thickness / grid_a)))


    gx = np.linspace(grid_a / 2, lx - grid_a / 2, nx)
    gy = np.linspace(grid_a / 2, ly - grid_a / 2, ny)
    gz = np.linspace(grid_a / 2, thickness - grid_a / 2, nz)
    xx, yy, zz = np.meshgrid(gx, gy, gz, indexing="ij")
    positions = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


    jitter_sigma = grid_a * 0.40
    positions += rng.normal(scale=jitter_sigma, size=positions.shape)


    positions[:, 0] = np.clip(positions[:, 0], 0.2, lx - 0.2)
    positions[:, 1] = np.clip(positions[:, 1], 0.2, ly - 0.2)
    positions[:, 2] = np.clip(positions[:, 2], 0.2, thickness - 0.2)


    if len(positions) > n_atoms:
        idx = rng.choice(len(positions), size=n_atoms, replace=False)
        positions = positions[idx]

    n_placed = len(positions)
    logger.info(f"Placed {n_placed} C atoms")


    vacuum_above = 100.0
    atoms = Atoms(
        "C" * n_placed,
        positions=positions,
        cell=[lx, ly, thickness + vacuum_above],
        pbc=[True, True, False],
    )
    return atoms


def place_particles_on_support(
    support: Atoms,
    pdiff: dict,
    fov_angstrom: float,
    rng: np.random.Generator,
) -> tuple[Atoms, list[dict]]:
    """Place Pt nanoparticles on the carbon support."""
    n_particles   = pdiff["n_particles"]
    d_mean        = pdiff["diameter_mean_nm"]
    d_sigma       = pdiff["diameter_sigma_nm"]
    d_min         = pdiff["diameter_min_nm"]
    morph_w       = pdiff["morph_weights"]
    min_sep_ang   = pdiff["min_sep_nm"] * 10.0
    support_thick = pdiff["support_thick_nm"] * 10.0

    margin = 30.0

    combined = support.copy()
    meta: list[dict] = []
    placed_xy: list[list[float]] = []

    for p_idx in range(n_particles):

        for _ in range(200):
            d_nm = rng.lognormal(mean=np.log(d_mean), sigma=d_sigma / d_mean)
            if d_nm >= d_min:
                break
        d_nm = max(d_min, d_nm)


        morph_idx = rng.choice(3, p=morph_w)
        builder   = _BUILDERS[morph_idx]
        label     = _LABELS[morph_idx]


        n_target = _natoms_from_diameter(d_nm)
        try:
            particle = builder(n_target)
        except Exception as exc:
            logger.warning(
                f"Could not build {MORPH_NAMES[label]} "
                f"(~{n_target} atoms): {exc}. Wulff fallback."
            )
            particle = make_wulff_particle(n_target)
            label = MORPH_SINGLE_CRYSTAL

        actual_d = _measure_diameter(particle)


        rot = Rotation.random(random_state=rng.integers(2**31))
        particle.positions = rot.apply(particle.positions)
        particle.positions -= particle.positions.mean(axis=0)


        placed = False
        for _ in range(500):
            cx = rng.uniform(margin, fov_angstrom - margin)
            cy = rng.uniform(margin, fov_angstrom - margin)
            if placed_xy:
                dists = np.hypot(
                    np.array(placed_xy)[:, 0] - cx,
                    np.array(placed_xy)[:, 1] - cy,
                )
                if dists.min() < min_sep_ang:
                    continue
            placed = True
            break

        if not placed:
            logger.warning(
                f"Could not place particle {p_idx+1}/{n_particles} "
                f"(d={d_nm:.1f} nm) — skipping."
            )
            continue


        half_h = (particle.positions[:, 2].max() -
                  particle.positions[:, 2].min()) / 2.0
        cz = support_thick + half_h + 1.0

        particle.positions[:, 0] += cx
        particle.positions[:, 1] += cy
        particle.positions[:, 2] += cz
        particle.set_cell(combined.cell)
        particle.set_pbc(combined.pbc)

        combined = combined + particle
        placed_xy.append([cx, cy])

        meta.append({
            "centroid_nm": (cx / 10.0, cy / 10.0),
            "diameter_nm": actual_d,
            "morphology_label": label,
            "morphology_name": MORPH_NAMES[label],
            "n_atoms": len(particle),
        })

        logger.info(
            f"  Particle {p_idx+1}/{n_particles}: "
            f"{MORPH_NAMES[label]}, d={actual_d:.1f} nm, "
            f"{len(particle)} atoms at ({cx/10:.1f}, {cy/10:.1f}) nm"
        )

    return combined, meta


def build_particle_ground_truth(meta: list[dict]) -> GroundTruth:
    """Package particle metadata into GroundTruth."""
    n = len(meta)

    centroids = np.array(
        [m["centroid_nm"] for m in meta], dtype=np.float32,
    ).reshape(-1, 2)
    if n == 0:
        centroids = np.zeros((0, 2), dtype=np.float32)

    gt = GroundTruth(
        atom_positions_nm=centroids,
        atom_types=np.full(n, 2, dtype=np.int32),
        atom_species=["Pt"] * n,
        defect_mask=np.ones(n, dtype=bool),
        defect_types=[m["morphology_name"] for m in meta],
    )
    gt.extra["material"]                   = "Pt_on_aC"
    gt.extra["task_type"]                  = "particle_census"
    gt.extra["n_particles"]                = n
    gt.extra["particle_diameters_nm"]      = np.array(
        [m["diameter_nm"] for m in meta], dtype=np.float32,
    )
    gt.extra["particle_morphology_labels"] = np.array(
        [m["morphology_label"] for m in meta], dtype=np.int32,
    )
    gt.extra["particle_n_atoms"]           = np.array(
        [m["n_atoms"] for m in meta], dtype=np.int32,
    )
    gt.extra["morphology_names"]           = str(MORPH_NAMES)
    return gt


def generate_pt_nanoparticles(
    difficulty: str = "medium",
    output_path: Path | None = None,
    device: str = "cpu",
    seed: int = 42,
    fov_nm: float = 100.0,
    shard_id: int | None = None,
    total_shards: int | None = None,
    merge_shards: int | None = None,
) -> Path:
    """Generate Pt nanoparticle world on amorphous carbon support."""
    if output_path is None:
        output_path = Path(f"data/simulated/pt_nanoparticles_{difficulty}.h5")

    rng  = np.random.default_rng(seed)
    diff  = DIFFICULTY_PARAMS[difficulty]
    pdiff = PT_DIFFICULTY[difficulty]

    fov_angstrom = fov_nm * 10.0
    support_thick_ang = pdiff["support_thick_nm"] * 10.0


    logger.info("Generating amorphous carbon support...")
    support = make_amorphous_carbon(
        lx=fov_angstrom, ly=fov_angstrom,
        thickness=support_thick_ang,
        density_gcc=2.0, rng=rng,
    )
    logger.info(f"Support: {len(support)} C atoms")


    logger.info(f"Placing {pdiff['n_particles']} Pt nanoparticles...")
    combined, particle_meta = place_particles_on_support(
        support, pdiff, fov_angstrom, rng,
    )
    n_pt = sum(1 for s in combined.get_chemical_symbols() if s == "Pt")
    n_c  = sum(1 for s in combined.get_chemical_symbols() if s == "C")
    logger.info(
        f"Combined structure: {len(combined)} atoms "
        f"({n_pt} Pt + {n_c} C), {len(particle_meta)} particles"
    )


    gt = build_particle_ground_truth(particle_meta)
    gt.extra["support_thick_nm"] = pdiff["support_thick_nm"]
    gt.extra["difficulty"] = difficulty
    gt.extra["fov_nm"] = fov_nm


    config = SimConfig(
        scenario="pt_nanoparticles",
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
        phonon_sigmas=PT_PHONON_SIGMAS,
        tile_size_px=128,
        overlap_px=4,
        seed=seed,
    )


    if shard_id is not None and total_shards is not None:
        stem = str(output_path.with_suffix(''))
        npz_path = Path(f"{stem}_shard{shard_id}.npz")
        logger.info(f"Running shard {shard_id}/{total_shards}...")
        run_stem_simulation_shard(combined, config, shard_id, total_shards, npz_path)
        return output_path

    if merge_shards is not None:
        stem = str(output_path.with_suffix(''))
        shard_paths = [Path(f"{stem}_shard{i}.npz") for i in range(merge_shards)]
        logger.info(f"Merging {merge_shards} shards...")
        image = merge_simulation_shards(shard_paths)
    else:
        logger.info("Running STEM-HAADF simulation...")
        image = run_stem_simulation(combined, config)
    logger.info(f"Image shape: {image.shape}")


    write_world_hdf5(output_path, image, gt, config)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate Pt nanoparticle scenario for STEMGym",
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
            out = args.output or Path(f"data/simulated/pt_nanoparticles_{diff}.h5")
            if args.output and diff != "medium":
                out = args.output.parent / f"pt_nanoparticles_{diff}.h5"
            generate_pt_nanoparticles(
                difficulty=diff, output_path=out,
                device=args.device, seed=args.seed, fov_nm=args.fov_nm,
                shard_id=args.shard, total_shards=args.total_shards,
                merge_shards=args.merge_shards,
            )
    else:
        generate_pt_nanoparticles(
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
