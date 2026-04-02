#!/usr/bin/env python3
"""Generate GaN scenario: InGaN substitution defects in wurtzite GaN.

Material: GaN (mp-804), wurtzite P6_3mc (#186), [11-20] zone axis, ~50nm FOV
Defects: In substitution on Ga sites (InGaN alloy / quantum-well band)
Task: Targeted Characterisation + Defect Census (dual-objective)

The [11-20] cross-section resolves pure-element Ga and N columns as
dumbbell pairs separated by u*c ≈ 1.96 Å along c.  In (Z=49) on
Ga (Z=31) sites gives ~2.18× per-atom HAADF intensity increase.
Substitutions are concentrated in a horizontal quantum-well band at
~60-70 % of the FOV height (along c), with a tunable fraction in the
band vs. uniform background — testing whether adaptive agents can
locate and characterise the In-rich region more efficiently than
systematic raster scanning.

References
----------
Crystal structure:
    Schulz, H. & Thiemann, K.H.
    "Crystal structure refinement of AlN and GaN"
    Solid State Commun. 23, 815–819, 1977.
    Lattice: a = 3.189 Å, c = 5.185 Å, u = 0.377

    Materials Project mp-804 (P6₃mc, #186, 4 atoms / cell)
    https://next-gen.materialsproject.org/materials/mp-804

Debye–Waller / frozen phonon:
    Schowalter, M. et al.
    "Temperature-dependent Debye–Waller factors for semiconductors
     with the wurtzite-type structure"
    Acta Cryst. A65, 227–244, 2009.
    σ_Ga = 0.059 Å, σ_N = 0.063 Å at 300 K (isotropic approx.)

In thermal displacement:
    Schowalter et al. 2009 (InN data, interpolated for dilute InGaN)
    σ_In ≈ 0.065 Å at 300 K

InGaN HAADF-STEM quantification:
    Rosenauer, A. et al.
    "Composition mapping in InGaN by scanning TEM"
    Ultramicroscopy 111, 1316–1327, 2011.

    Mehrtens, T. et al.
    "Measuring composition in InGaN from HAADF-STEM images and
     studying the temperature dependence of Z-contrast"
    J. Phys.: Conf. Ser. 471, 012009, 2013.

Static displacement correction:
    Schowalter, M. et al.
    "Including static atomic displacements in quantitative STEM
     simulations for AlGaN"
    Microsc. Microanal. 20, 1463–1469, 2014.

Electron optics (shared benchmark parameters):
    200 keV, 21 mrad convergence, HAADF 68–200 mrad
    Hartel criterion: 68 / 21 = 3.24 > 3.0 (incoherent Z-contrast)
    0.05 Å potential sampling, 1.0 Å slice thickness, PRISM

Usage:
    python stem_gym/scripts/generate_worlds/generate_gan.py [--difficulty easy|medium|hard]
                                        [--output data/simulated/gan.h5]
                                        [--device cpu|gpu]
                                        [--seed 42]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from sim_utils import (
    SimConfig,
    record_ground_truth,
    run_stem_simulation,
    run_stem_simulation_shard,
    merge_simulation_shards,
    write_world_hdf5,
    add_shard_args,
    DIFFICULTY_PARAMS,
)

# ---------------------------------------------------------------------------
# GaN crystallographic constants
# ---------------------------------------------------------------------------

# Wurtzite lattice parameters (Schulz & Thiemann 1977, mp-804)
GAN_A = 3.189   # Å, in-plane
GAN_C = 5.185   # Å, along c-axis
GAN_U = 0.377   # internal parameter for N sublattice

# InN lattice parameters (for mismatch / partial relaxation)
INN_A = 3.545   # Å
INN_C = 5.703   # Å

# Debye–Waller: isotropic RMS displacement at 300 K
# Schowalter et al., Acta Cryst. A65, 227-244, 2009
GAN_PHONON_SIGMAS = {
    "Ga": 0.059,
    "N":  0.063,
    "In": 0.065,   # interpolated from InN data, Schowalter et al. 2009
}

# ---------------------------------------------------------------------------
# Difficulty parameters
# ---------------------------------------------------------------------------

GAN_DIFFICULTY = {
    "easy": {
        "in_fraction":      0.25,   # 25 % In in QW → large ΔI (~30 %)
        "cluster_fraction": 0.70,   # 70 % of In concentrated in QW band
        "qw_width_frac":    0.15,   # QW band σ as fraction of FOV
        "qw_center_frac":   0.65,   # QW centre along c-axis
    },
    "medium": {
        "in_fraction":      0.12,   # 12 % In → moderate ΔI (~12 %)
        "cluster_fraction": 0.75,
        "qw_width_frac":    0.10,
        "qw_center_frac":   0.65,
    },
    "hard": {
        "in_fraction":      0.05,   # 5 % In → subtle ΔI (~5 %, near noise floor)
        "cluster_fraction": 0.80,
        "qw_width_frac":    0.06,
        "qw_center_frac":   0.60,
    },
}


# ---------------------------------------------------------------------------
# Structure building
# ---------------------------------------------------------------------------

def build_gan_unit_cell() -> Atoms:
    """Build orthorhombic wurtzite GaN unit cell via ASE.

    Uses ``ase.build.bulk`` with *orthorhombic=True* to produce a
    rectangular cell suitable for STEM scan grids::

        x  →  [2-1-10]  a         = 3.189 Å
        y  →  [01-10]   a√3       = 5.524 Å
        z  →  [0001]    c         = 5.185 Å

    4 atoms per cell (2 Ga + 2 N).
    """
    return bulk("GaN", crystalstructure="wurtzite",
                a=GAN_A, c=GAN_C, orthorhombic=True)


def build_gan_supercell_1120(fov_angstrom: float, z_thickness: float) -> Atoms:
    """Build a GaN supercell oriented for [11-20] zone axis STEM.

    For [11-20] beam direction (beam along the *a*-axis)::

        image x  →  [01-10]  horizontal, inter-dumbbell  (a√3 = 5.524 Å)
        image y  →  [0001]   vertical, c-axis dumbbells   (c   = 5.185 Å)
        beam  z  →  [2-1-10] specimen thickness            (a   = 3.189 Å)

    We permute the ASE orthorhombic cell axes (x,y,z) → (y,z,x).

    Key column separations in [11-20] projection
    =============================================
    ================  ==========
    Ga–N dumbbell     u·c = 1.955 Å  (along c)
    Inter-pair horiz  a√3/2 = 2.762 Å
    Vertical repeat   c = 5.185 Å
    Horizontal repeat a√3 = 5.524 Å
    ================  ==========
    """
    gan = build_gan_unit_cell()

    # Permute: (x,y,z) → (y,z,x)  so that old-x = [2-1-10] becomes beam
    # The original orthorhombic cell is diagonal, so we build a new diagonal
    # cell with the axes reordered and positions columns swapped consistently.
    cell = gan.cell.array
    old_pos = gan.get_positions()
    # New Cartesian x ← old y ([01-10]), new y ← old z ([0001]), new z ← old x ([2-1-10])
    new_cell = np.diag([cell[1, 1], cell[2, 2], cell[0, 0]])
    new_positions = old_pos[:, [1, 2, 0]]

    reoriented = Atoms(
        symbols=gan.get_chemical_symbols(),
        positions=new_positions,
        cell=new_cell,
        pbc=True,
    )

    # Tile to target dimensions
    target = (fov_angstrom, fov_angstrom, z_thickness)
    reps = []
    for i in range(3):
        length = np.linalg.norm(reoriented.cell[i])
        n = max(1, int(np.round(target[i] / length)))
        reps.append(n)

    logger.info(f"GaN [11-20] supercell repetitions: {reps}")
    return reoriented * tuple(reps)


# ---------------------------------------------------------------------------
# In substitution (InGaN defects)
# ---------------------------------------------------------------------------

def apply_in_substitution(
    atoms: Atoms,
    in_fraction: float,
    cluster_fraction: float,
    qw_center_frac: float,
    qw_width_frac: float,
    rng: np.random.Generator,
) -> tuple[Atoms, np.ndarray, np.ndarray]:
    """Replace Ga atoms with In using a QW-band + background distribution.

    In concentration is highest in a horizontal band (quantum well)
    centred at *qw_center_frac* of the cell y-extent, with Gaussian
    fall-off.  (1 − cluster_fraction) of substitutions are distributed
    uniformly as background.  Mimics InGaN/GaN MQW structures in LEDs.

    Returns
    -------
    modified : Atoms
        Structure with In substitutions applied.
    sub_indices : ndarray
        Indices (into original atoms) of substituted Ga sites.
    in_fraction_1d : ndarray
        1-D In fraction profile along y (c-axis), 100 bins.
    """
    modified = atoms.copy()
    positions = modified.get_positions()
    symbols = list(modified.get_chemical_symbols())
    # After [11-20] permutation, cell[1] may not be diagonal;
    # use positions extent along dim 1 (c-axis) instead.
    cell_y = float(np.linalg.norm(modified.cell[1]))  # y = [0001]

    # Identify Ga atoms
    ga_indices = np.array([i for i, s in enumerate(symbols) if s == "Ga"])
    n_ga = len(ga_indices)
    n_total_subs = max(1, int(n_ga * in_fraction))

    # QW band centre and sigma in Angstrom
    qw_center = qw_center_frac * cell_y
    qw_sigma = qw_width_frac * cell_y

    # Per-Ga-atom Gaussian weights based on distance to QW centre
    ga_y = positions[ga_indices, 1]
    gaussian_weights = np.exp(-(ga_y - qw_center) ** 2 / (2.0 * qw_sigma ** 2))

    # Split clustered (QW) / background
    n_clustered = int(n_total_subs * cluster_fraction)
    n_background = n_total_subs - n_clustered

    w = gaussian_weights.astype(np.float64)
    w_sum = w.sum()
    prob = w / w_sum if w_sum > 0 else np.ones(len(w)) / len(w)
    clustered_local = rng.choice(
        len(ga_indices), size=min(n_clustered, len(ga_indices)),
        replace=False, p=prob,
    )
    chosen_set = set(clustered_local.tolist())

    remaining_local = np.array([
        i for i in range(len(ga_indices)) if i not in chosen_set
    ])
    if n_background > 0 and len(remaining_local) > 0:
        n_background = min(n_background, len(remaining_local))
        bg_local = rng.choice(remaining_local, size=n_background, replace=False)
        all_local = np.concatenate([clustered_local, bg_local])
    else:
        all_local = clustered_local

    sub_indices = ga_indices[all_local]

    # Apply Ga → In with partial lattice relaxation
    expansion = (INN_A - GAN_A) / GAN_A * 0.3  # constrained epitaxy
    for idx in sub_indices:
        symbols[idx] = "In"
        positions[idx, 0] *= (1.0 + expansion)
        positions[idx, 1] *= (1.0 + expansion)

    modified.set_chemical_symbols(symbols)
    modified.set_positions(positions)

    # 1-D In fraction profile along y (c-axis) for ground truth
    n_bins = 100
    in_profile = np.zeros(n_bins, dtype=np.float32)
    total_profile = np.zeros(n_bins, dtype=np.float32)
    for i in ga_indices:
        b = min(int(positions[i, 1] / cell_y * n_bins), n_bins - 1)
        total_profile[b] += 1
        if symbols[i] == "In":
            in_profile[b] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        in_fraction_1d = np.where(total_profile > 0, in_profile / total_profile, 0.0)

    return modified, sub_indices, in_fraction_1d.astype(np.float32)


def make_gan_phase_map(
    shape_px: tuple[int, int],
    qw_center_frac: float,
    qw_width_frac: float,
) -> np.ndarray:
    """Create a phase map for the GaN/InGaN structure.

    Labels:  0 = GaN matrix,  1 = InGaN transition (|Δy| < 2σ),
             2 = InGaN core (|Δy| < 1σ).
    """
    h, w = shape_px
    yy = np.arange(h) / h
    dy = np.abs(yy - qw_center_frac)

    phase_map = np.zeros(shape_px, dtype=np.int32)
    phase_map[dy < 2.0 * qw_width_frac, :] = 1
    phase_map[dy < qw_width_frac, :] = 2
    return phase_map


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_gan(
    difficulty: str = "medium",
    output_path: Path | None = None,
    device: str = "cpu",
    seed: int = 42,
    fov_nm: float = 50.0,
    shard_id: int | None = None,
    total_shards: int | None = None,
    merge_shards: int | None = None,
) -> Path:
    """Generate GaN world with InGaN substitution defects.

    Args:
        difficulty: "easy", "medium", or "hard".
        output_path: HDF5 output path.
        device: "cpu" or "gpu" for abTEM.
        seed: Random seed.
        fov_nm: Field of view in nm (default 50).

    Returns:
        Path to generated HDF5 file.
    """
    if output_path is None:
        output_path = Path(f"data/simulated/gan_{difficulty}.h5")

    rng = np.random.default_rng(seed)
    diff = DIFFICULTY_PARAMS[difficulty]
    gdiff = GAN_DIFFICULTY[difficulty]

    fov_angstrom = fov_nm * 10.0
    z_thickness = 200.0  # 20 nm along [11-20] beam (~63 unit cells)

    # --- 1. Build GaN supercell in [11-20] orientation ---
    logger.info("Building GaN [11-20] supercell...")
    supercell = build_gan_supercell_1120(fov_angstrom, z_thickness)
    logger.info(
        f"Supercell: {len(supercell)} atoms, "
        f"cell={[f'{x:.1f}' for x in supercell.cell.lengths()]}"
    )

    pristine = supercell.copy()

    # --- 2. Apply In substitution (InGaN QW + background) ---
    in_frac = gdiff["in_fraction"]
    cluster_frac = gdiff["cluster_fraction"]
    qw_center = gdiff["qw_center_frac"]
    qw_width = gdiff["qw_width_frac"]

    logger.info(
        f"Applying In substitution: in_frac={in_frac*100:.0f}%, "
        f"cluster={cluster_frac:.0%}, "
        f"QW centre={qw_center:.2f}, width={qw_width:.2f}"
    )
    modified, sub_indices, in_profile = apply_in_substitution(
        supercell,
        in_fraction=in_frac,
        cluster_fraction=cluster_frac,
        qw_center_frac=qw_center,
        qw_width_frac=qw_width,
        rng=rng,
    )
    n_in = sum(1 for s in modified.get_chemical_symbols() if s == "In")
    n_ga = sum(1 for s in modified.get_chemical_symbols() if s == "Ga")
    logger.info(
        f"Substituted {n_in} Ga → In "
        f"({n_in}/{n_in + n_ga} = {n_in / (n_in + n_ga) * 100:.1f}%)"
    )

    # --- 3. Record ground truth ---
    gt = record_ground_truth(
        pristine,
        substitution_indices=sub_indices,
        pixel_size_nm=0.005,
    )
    gt.extra["material"] = "GaN"
    gt.extra["zone_axis"] = "11-20"
    gt.extra["in_fraction_target"] = in_frac
    gt.extra["in_fraction_actual"] = n_in / (n_in + n_ga)
    gt.extra["in_fraction_profile"] = in_profile
    gt.extra["cluster_fraction"] = cluster_frac
    gt.extra["qw_center_frac"] = qw_center
    gt.extra["qw_width_frac"] = qw_width
    gt.extra["clustering"] = "quantum_well_band"

    # Phase map
    sampling = 0.05  # Å / pixel
    img_size = int(fov_angstrom / sampling)
    gt.phase_map = make_gan_phase_map(
        (img_size, img_size), qw_center, qw_width,
    )

    # --- 4. Configure simulation ---
    config = SimConfig(
        scenario="gan",
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
        phonon_sigmas=GAN_PHONON_SIGMAS,
        tile_size_px=128,
        overlap_px=4,
        seed=seed,
    )

    # --- 5. Run STEM simulation ---
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

    # --- 6. Resize phase map to match actual image ---
    from scipy.ndimage import zoom as scipy_zoom
    if gt.phase_map.shape != image.shape:
        scale_h = image.shape[0] / gt.phase_map.shape[0]
        scale_w = image.shape[1] / gt.phase_map.shape[1]
        gt.phase_map = scipy_zoom(
            gt.phase_map, (scale_h, scale_w), order=0,
        ).astype(np.int32)

    # --- 7. Write HDF5 ---
    write_world_hdf5(output_path, image, gt, config)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate GaN InGaN scenario for STEMGym",
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
            out = args.output or Path(f"data/simulated/gan_{diff}.h5")
            if args.output and diff != "medium":
                out = args.output.parent / f"gan_{diff}.h5"
            generate_gan(
                difficulty=diff, output_path=out,
                device=args.device, seed=args.seed, fov_nm=args.fov_nm,
                shard_id=args.shard, total_shards=args.total_shards,
                merge_shards=args.merge_shards,
            )
    else:
        generate_gan(
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
