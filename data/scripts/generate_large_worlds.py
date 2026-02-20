#!/usr/bin/env python3
"""Generate larger scenario worlds for benchmarking (no abTEM required).

Creates 1024×1024 (8×8 grid) worlds with scenario-specific structure:
  - SrTiO3: Perovskite lattice with oxygen vacancies (cubic)
  - BaTiO3: Two-phase perovskite with phase boundary (cubic/tetragonal)
  - SiGe:   Diamond lattice with Ge gradient interface

Each world has realistic lattice spacing, defect contrast, and ground truth
labels suitable for all three tasks (defect census, phase mapping, targeted).

Usage:
    python data/scripts/generate_large_worlds.py
    python data/scripts/generate_large_worlds.py --scenario srtio3 --difficulty medium
    python data/scripts/generate_large_worlds.py --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter, zoom as scipy_zoom


def make_perovskite_lattice(
    shape: tuple[int, int],
    a_px: int = 16,
    sigma: float = 2.0,
    rng: np.random.Generator | None = None,
    z_contrast: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a perovskite-like lattice (ABO3).

    Sublattices:
      A-site: corners → (0, 0)
      B-site: body center → (a/2, a/2)
      O-site: face centers → (a/2, 0), (0, a/2)

    Returns:
        image, positions (N, 2) in pixels (row, col), atom_sublattice (N,) str
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if z_contrast is None:
        z_contrast = {"A": 1.0, "B": 0.7, "O": 0.3}

    h, w = shape
    image = np.zeros(shape, dtype=np.float32)
    positions = []
    sublattices = []

    half = a_px // 2

    for r in range(half, h - half // 2, a_px):
        for c in range(half, w - half // 2, a_px):
            jitter = lambda: rng.normal(0, 0.3)

            # A-site (corner)
            rr, cc = int(np.clip(r + jitter(), 0, h-1)), int(np.clip(c + jitter(), 0, w-1))
            image[rr, cc] += z_contrast["A"]
            positions.append([rr, cc])
            sublattices.append("A")

            # B-site (body center)
            rr = int(np.clip(r + half + jitter(), 0, h-1))
            cc = int(np.clip(c + half + jitter(), 0, w-1))
            image[rr, cc] += z_contrast["B"]
            positions.append([rr, cc])
            sublattices.append("B")

            # O-sites (face centers, projected in [001])
            for dr, dc in [(half, 0), (0, half)]:
                rr = int(np.clip(r + dr + jitter(), 0, h-1))
                cc = int(np.clip(c + dc + jitter(), 0, w-1))
                image[rr, cc] += z_contrast["O"]
                positions.append([rr, cc])
                sublattices.append("O")

    image = gaussian_filter(image, sigma=sigma)
    if image.max() > 0:
        image /= image.max()

    return image, np.array(positions), np.array(sublattices)


def make_diamond_lattice(
    shape: tuple[int, int],
    a_px: int = 20,
    sigma: float = 2.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a diamond-like lattice (Si/Ge projected along [110]).

    Returns:
        image, positions (N, 2) in pixels
    """
    if rng is None:
        rng = np.random.default_rng(42)

    h, w = shape
    image = np.zeros(shape, dtype=np.float32)
    positions = []

    # Diamond [110] projection: pairs of atoms ("dumbbells")
    for r in range(a_px // 2, h - a_px // 4, a_px):
        for c in range(a_px // 2, w - a_px // 4, a_px):
            jitter = lambda: rng.normal(0, 0.3)
            # Dumbbell pair
            sep = a_px // 5  # separation within pair
            for offset in [-sep, sep]:
                rr = int(np.clip(r + offset + jitter(), 0, h-1))
                cc = int(np.clip(c + jitter(), 0, w-1))
                image[rr, cc] += 1.0
                positions.append([rr, cc])

    image = gaussian_filter(image, sigma=sigma)
    if image.max() > 0:
        image /= image.max()

    return image, np.array(positions)


# ---------------------------------------------------------------------------
# Scenario generators
# ---------------------------------------------------------------------------

def generate_srtio3_large(
    output_path: Path,
    difficulty: str = "medium",
    world_size: int = 1024,
    tile_size: int = 128,
    overlap: int = 4,
    pixel_size_nm: float = 0.1,
    seed: int = 42,
):
    """SrTiO3 with oxygen vacancies — Defect Census scenario."""
    rng = np.random.default_rng(seed)
    vacancy_rates = {"easy": 0.05, "medium": 0.03, "hard": 0.01}
    noise_levels = {"easy": 0.02, "medium": 0.04, "hard": 0.08}
    vac_rate = vacancy_rates[difficulty]
    noise_level = noise_levels[difficulty]

    print(f"  Generating SrTiO3 {world_size}x{world_size} ({difficulty})...")

    # Sr is heavy (Z=38), Ti moderate (Z=22), O light (Z=8)
    image, positions, sublattices = make_perovskite_lattice(
        (world_size, world_size), a_px=16, sigma=2.0, rng=rng,
        z_contrast={"A": 1.0, "B": 0.65, "O": 0.25},
    )

    # Plant O vacancies
    o_mask = sublattices == "O"
    o_indices = np.where(o_mask)[0]
    n_vac = int(len(o_indices) * vac_rate)
    vac_indices = rng.choice(o_indices, size=n_vac, replace=False)

    atom_types = np.zeros(len(positions), dtype=np.int32)
    defect_type_names = np.array(["pristine"] * len(positions))
    atom_types[vac_indices] = 1
    defect_type_names[vac_indices] = "vacancy"

    # Dim vacancy sites
    defect_image = image.copy()
    for idx in vac_indices:
        r, c = positions[idx]
        r0, r1 = max(0, r-3), min(world_size, r+4)
        c0, c1 = max(0, c-3), min(world_size, c+4)
        defect_image[r0:r1, c0:c1] *= 0.15

    # Add noise
    defect_image += rng.normal(0, noise_level, defect_image.shape).astype(np.float32)
    defect_image = np.clip(defect_image, 0, None)
    if defect_image.max() > 0:
        defect_image /= defect_image.max()

    # Phase map: uniform (single phase)
    phase_map = np.zeros((world_size, world_size), dtype=np.int32)

    # Write
    _write_world(
        output_path, defect_image, positions, atom_types,
        defect_type_names, phase_map, tile_size, overlap, pixel_size_nm,
        scenario="srtio3", difficulty=difficulty,
    )
    n_vac_total = int(np.sum(atom_types == 1))
    print(f"  SrTiO3: {len(positions)} atoms, {n_vac_total} vacancies ({vac_rate*100:.0f}%)")


def generate_batio3_large(
    output_path: Path,
    difficulty: str = "medium",
    world_size: int = 1024,
    tile_size: int = 128,
    overlap: int = 4,
    pixel_size_nm: float = 0.1,
    seed: int = 42,
):
    """BaTiO3 with phase boundary — Phase Mapping scenario."""
    rng = np.random.default_rng(seed)
    boundary_widths = {"easy": 100, "medium": 50, "hard": 20}
    noise_levels = {"easy": 0.02, "medium": 0.04, "hard": 0.08}
    bw = boundary_widths[difficulty]
    noise_level = noise_levels[difficulty]

    print(f"  Generating BaTiO3 {world_size}x{world_size} ({difficulty})...")

    # Ba is heavy (Z=56), Ti moderate (Z=22), O light (Z=8)
    image, positions, sublattices = make_perovskite_lattice(
        (world_size, world_size), a_px=16, sigma=2.0, rng=rng,
        z_contrast={"A": 1.0, "B": 0.55, "O": 0.20},
    )

    # Phase boundary: cubic (left) vs tetragonal (right)
    # In tetragonal phase, Ti shifts slightly along c-axis → changes HAADF contrast
    boundary_x = world_size // 2 + rng.integers(-world_size // 8, world_size // 8)

    phase_map = np.zeros((world_size, world_size), dtype=np.int32)
    # Smooth boundary with gradient
    for c in range(world_size):
        frac = 1.0 / (1.0 + np.exp(-(c - boundary_x) / (bw / 4)))
        phase_map[:, c] = (frac > 0.5).astype(np.int32)

    defect_image = image.copy()

    # Modify tetragonal phase: slightly different B-site contrast
    b_mask = sublattices == "B"
    for idx in np.where(b_mask)[0]:
        r, c = positions[idx]
        if c > boundary_x:
            # Tetragonal: Ti shifts, appears slightly brighter
            r0, r1 = max(0, r-2), min(world_size, r+3)
            c0, c1 = max(0, c-2), min(world_size, c+3)
            defect_image[r0:r1, c0:c1] *= 1.15  # subtle increase

    # Also add some vacancies for defect census compatibility
    o_indices = np.where(sublattices == "O")[0]
    n_vac = int(len(o_indices) * 0.02)
    vac_indices = rng.choice(o_indices, size=n_vac, replace=False)

    atom_types = np.zeros(len(positions), dtype=np.int32)
    defect_type_names = np.array(["pristine"] * len(positions))
    atom_types[vac_indices] = 1
    defect_type_names[vac_indices] = "vacancy"

    for idx in vac_indices:
        r, c = positions[idx]
        r0, r1 = max(0, r-3), min(world_size, r+4)
        c0, c1 = max(0, c-3), min(world_size, c+4)
        defect_image[r0:r1, c0:c1] *= 0.15

    # Add noise
    defect_image += rng.normal(0, noise_level, defect_image.shape).astype(np.float32)
    defect_image = np.clip(defect_image, 0, None)
    if defect_image.max() > 0:
        defect_image /= defect_image.max()

    _write_world(
        output_path, defect_image, positions, atom_types,
        defect_type_names, phase_map, tile_size, overlap, pixel_size_nm,
        scenario="batio3", difficulty=difficulty,
    )
    print(f"  BaTiO3: {len(positions)} atoms, boundary at x={boundary_x}px, {n_vac} vacancies")


def generate_sige_large(
    output_path: Path,
    difficulty: str = "medium",
    world_size: int = 1024,
    tile_size: int = 128,
    overlap: int = 4,
    pixel_size_nm: float = 0.1,
    seed: int = 42,
):
    """Si/Ge with interface gradient — Targeted Characterization scenario."""
    rng = np.random.default_rng(seed)
    gradient_widths = {"easy": 200, "medium": 100, "hard": 40}
    noise_levels = {"easy": 0.02, "medium": 0.04, "hard": 0.08}
    gw = gradient_widths[difficulty]
    noise_level = noise_levels[difficulty]

    print(f"  Generating Si/Ge {world_size}x{world_size} ({difficulty})...")

    image, positions = make_diamond_lattice(
        (world_size, world_size), a_px=20, sigma=2.0, rng=rng,
    )

    # Ge gradient: left is pure Si, right is high-Ge
    interface_x = world_size // 2 + rng.integers(-world_size // 8, world_size // 8)
    defect_image = image.copy()

    atom_types = np.zeros(len(positions), dtype=np.int32)
    defect_type_names = np.array(["pristine"] * len(positions))

    # Phase map: Si (0) vs SiGe (1)
    phase_map = np.zeros((world_size, world_size), dtype=np.int32)
    phase_map[:, interface_x:] = 1

    for idx in range(len(positions)):
        r, c = positions[idx]
        # Ge fraction increases across interface
        ge_frac = 1.0 / (1.0 + np.exp(-(c - interface_x) / (gw / 4)))

        if ge_frac > 0.3:  # In the Ge-rich region
            # Ge is heavier (Z=32) than Si (Z=14) → brighter in HAADF
            brightness_boost = 1.0 + ge_frac * 0.5
            r0, r1 = max(0, r-2), min(world_size, r+3)
            c0, c1 = max(0, c-2), min(world_size, c+3)
            defect_image[r0:r1, c0:c1] *= brightness_boost

            if ge_frac > 0.5:
                atom_types[idx] = 2  # substitution (Ge replacing Si)
                defect_type_names[idx] = "substitution"

    # Add noise
    defect_image += rng.normal(0, noise_level, defect_image.shape).astype(np.float32)
    defect_image = np.clip(defect_image, 0, None)
    if defect_image.max() > 0:
        defect_image /= defect_image.max()

    _write_world(
        output_path, defect_image, positions, atom_types,
        defect_type_names, phase_map, tile_size, overlap, pixel_size_nm,
        scenario="sige", difficulty=difficulty,
    )
    n_sub = int(np.sum(atom_types == 2))
    print(f"  SiGe: {len(positions)} atoms, {n_sub} Ge substitutions, interface at x={interface_x}px")


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------

def _write_world(
    output_path: Path,
    image: np.ndarray,
    positions: np.ndarray,
    atom_types: np.ndarray,
    defect_type_names: np.ndarray,
    phase_map: np.ndarray,
    tile_size: int,
    overlap: int,
    pixel_size_nm: float,
    scenario: str,
    difficulty: str,
):
    """Write a world to HDF5 in standard STEMGym format."""
    world_size = image.shape[0]
    stride = tile_size - overlap
    grid_h = (world_size - tile_size) // stride + 1
    grid_w = (world_size - tile_size) // stride + 1
    fov_h = world_size * pixel_size_nm
    fov_w = world_size * pixel_size_nm

    # Atom positions in nm (x, y)
    positions_nm_xy = positions[:, ::-1].astype(np.float32) * pixel_size_nm

    # Overview (4x downsampled)
    overview = scipy_zoom(image, 0.25, order=1)

    # Valid region
    valid_region = np.ones((world_size, world_size), dtype=bool)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["pixel_size_nm"] = pixel_size_nm
        meta.attrs["tile_size_px"] = tile_size
        meta.attrs["overlap_px"] = overlap
        meta.attrs["grid_shape"] = [grid_h, grid_w]
        meta.attrs["fov_nm"] = [fov_h, fov_w]
        meta.attrs["world_shape_px"] = [world_size, world_size]
        meta.attrs["scenario"] = scenario
        meta.attrs["difficulty"] = difficulty

        f.create_dataset("overview", data=overview, dtype=np.float32)
        f.create_dataset("valid_region", data=valid_region, dtype=bool)

        tiles_group = f.create_group("tiles")
        for r in range(grid_h):
            for c in range(grid_w):
                y0 = r * stride
                x0 = c * stride
                tile = image[y0:y0 + tile_size, x0:x0 + tile_size]
                tiles_group.create_dataset(
                    f"{r}_{c}", data=tile, dtype=np.float32, chunks=True,
                )

        gt = f.create_group("ground_truth")
        gt.create_dataset("atom_positions", data=positions_nm_xy)
        gt.create_dataset("atom_types", data=atom_types)
        gt.create_dataset("defect_mask", data=(atom_types > 0))
        defect_bytes = [s.encode("utf-8") for s in defect_type_names]
        gt.create_dataset("defect_types", data=defect_bytes)
        gt.create_dataset("phase_map", data=phase_map)
        gt.attrs["n_atoms"] = len(positions)
        gt.attrs["n_vacancies"] = int(np.sum(atom_types == 1))
        gt.attrs["n_substitutions"] = int(np.sum(atom_types == 2))
        gt.attrs["scenario"] = scenario
        gt.attrs["difficulty"] = difficulty

    print(f"  Written to {output_path} (grid: {grid_h}x{grid_w})")


def main():
    parser = argparse.ArgumentParser(description="Generate larger scenario worlds")
    parser.add_argument("--scenario", choices=["srtio3", "batio3", "sige", "all"], default="all")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    parser.add_argument("--size", type=int, default=1024, help="World size in pixels")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true", help="Generate all scenarios × all difficulties")
    parser.add_argument("--output-dir", type=Path, default=Path("data/simulated"))
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = {
        "srtio3": generate_srtio3_large,
        "batio3": generate_batio3_large,
        "sige": generate_sige_large,
    }

    if args.all or args.scenario == "all":
        difficulties = ["easy", "medium", "hard"] if args.all else [args.difficulty]
        for diff in difficulties:
            for name, gen_fn in scenarios.items():
                out = output_dir / f"{name}_{diff}.h5"
                gen_fn(out, difficulty=diff, world_size=args.size, seed=args.seed)
    else:
        gen_fn = scenarios[args.scenario]
        out = output_dir / f"{args.scenario}_{args.difficulty}.h5"
        gen_fn(out, difficulty=args.difficulty, world_size=args.size, seed=args.seed)


if __name__ == "__main__":
    main()
