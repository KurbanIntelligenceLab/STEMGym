from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter


def make_lattice(
    shape: tuple[int, int],
    spacing_px: int = 16,
    sigma: float = 2.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a 2D lattice of Gaussian blobs."""
    if rng is None:
        rng = np.random.default_rng(42)

    h, w = shape
    image = np.zeros(shape, dtype=np.float32)
    positions = []

    for r in range(spacing_px // 2, h - spacing_px // 4, spacing_px):
        for c in range(spacing_px // 2, w - spacing_px // 4, spacing_px):

            dr = rng.normal(0, 0.5)
            dc = rng.normal(0, 0.5)
            rr = int(np.clip(r + dr, 0, h - 1))
            cc = int(np.clip(c + dc, 0, w - 1))
            positions.append([rr, cc])
            image[rr, cc] += 1.0


    image = gaussian_filter(image, sigma=sigma)

    if image.max() > 0:
        image /= image.max()

    return image, np.array(positions)


def plant_defects(
    image: np.ndarray,
    positions: np.ndarray,
    vacancy_rate: float = 0.03,
    substitution_rate: float = 0.02,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Plant vacancies and substitutions into the lattice."""
    if rng is None:
        rng = np.random.default_rng(42)

    n_atoms = len(positions)
    atom_types = np.zeros(n_atoms, dtype=np.int32)
    defect_type_names = np.array(["pristine"] * n_atoms)

    defect_image = image.copy()


    n_vac = int(n_atoms * vacancy_rate)
    vac_indices = rng.choice(n_atoms, size=n_vac, replace=False)
    atom_types[vac_indices] = 1
    defect_type_names[vac_indices] = "vacancy"


    for idx in vac_indices:
        r, c = positions[idx]

        r0, r1 = max(0, r - 3), min(image.shape[0], r + 4)
        c0, c1 = max(0, c - 3), min(image.shape[1], c + 4)
        defect_image[r0:r1, c0:c1] *= 0.1


    remaining = np.where(atom_types == 0)[0]
    n_sub = int(n_atoms * substitution_rate)
    sub_indices = rng.choice(remaining, size=min(n_sub, len(remaining)), replace=False)
    atom_types[sub_indices] = 2
    defect_type_names[sub_indices] = "substitution"


    for idx in sub_indices:
        r, c = positions[idx]
        r0, r1 = max(0, r - 3), min(image.shape[0], r + 4)
        c0, c1 = max(0, c - 3), min(image.shape[1], c + 4)
        defect_image[r0:r1, c0:c1] *= 1.5

    defect_mask = atom_types > 0


    if defect_image.max() > 0:
        defect_image /= defect_image.max()

    return defect_image, atom_types, defect_mask, defect_type_names


def make_phase_map(
    shape: tuple[int, int],
    n_phases: int = 2,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Create a simple phase map with n_phases regions."""
    if rng is None:
        rng = np.random.default_rng(42)
    h, w = shape
    phase_map = np.zeros(shape, dtype=np.int32)

    boundary = w // 2 + rng.integers(-w // 8, w // 8)
    phase_map[:, boundary:] = 1
    if n_phases > 2:
        boundary2 = h // 2 + rng.integers(-h // 8, h // 8)
        phase_map[boundary2:, :boundary] = 2
    return phase_map


def generate_world(
    output_path: Path,
    world_size: int = 1024,
    tile_size: int = 128,
    overlap: int = 4,
    spacing_px: int = 16,
    pixel_size_nm: float = 0.1,
    vacancy_rate: float = 0.03,
    substitution_rate: float = 0.02,
    seed: int = 42,
) -> None:
    """Generate a complete synthetic HDF5 world."""
    rng = np.random.default_rng(seed)

    print(f"Generating {world_size}x{world_size} world...")
    image, positions = make_lattice(
        (world_size, world_size), spacing_px=spacing_px, rng=rng,
    )
    defect_image, atom_types, defect_mask, defect_type_names = plant_defects(
        image, positions,
        vacancy_rate=vacancy_rate,
        substitution_rate=substitution_rate,
        rng=rng,
    )
    phase_map = make_phase_map((world_size, world_size), rng=rng)


    stride = tile_size - overlap
    grid_h = (world_size - tile_size) // stride + 1
    grid_w = (world_size - tile_size) // stride + 1
    fov_h = world_size * pixel_size_nm
    fov_w = world_size * pixel_size_nm


    positions_nm = positions.astype(np.float32) * pixel_size_nm

    positions_nm_xy = positions_nm[:, ::-1].copy()


    from scipy.ndimage import zoom as scipy_zoom
    overview = scipy_zoom(defect_image, 0.25, order=1)


    valid_region = np.ones((world_size, world_size), dtype=bool)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing HDF5 to {output_path}...")
    with h5py.File(output_path, "w") as f:

        meta = f.create_group("metadata")
        meta.attrs["pixel_size_nm"] = pixel_size_nm
        meta.attrs["tile_size_px"] = tile_size
        meta.attrs["overlap_px"] = overlap
        meta.attrs["grid_shape"] = [grid_h, grid_w]
        meta.attrs["fov_nm"] = [fov_h, fov_w]
        meta.attrs["world_shape_px"] = [world_size, world_size]
        meta.attrs["scenario"] = "test"
        meta.attrs["difficulty"] = "easy"


        f.create_dataset("overview", data=overview, dtype=np.float32)


        f.create_dataset("valid_region", data=valid_region, dtype=bool)


        tiles_group = f.create_group("tiles")
        for r in range(grid_h):
            for c in range(grid_w):
                y0 = r * stride
                x0 = c * stride
                tile = defect_image[y0:y0 + tile_size, x0:x0 + tile_size]
                tiles_group.create_dataset(
                    f"{r}_{c}", data=tile, dtype=np.float32,
                    chunks=True,
                )


        gt = f.create_group("ground_truth")
        gt.create_dataset("atom_positions", data=positions_nm_xy)
        gt.create_dataset("atom_types", data=atom_types)
        gt.create_dataset("defect_mask", data=defect_mask)

        defect_bytes = [s.encode("utf-8") for s in defect_type_names]
        gt.create_dataset("defect_types", data=defect_bytes)
        gt.create_dataset("phase_map", data=phase_map)
        gt.attrs["n_atoms"] = len(positions)
        gt.attrs["n_vacancies"] = int(np.sum(atom_types == 1))
        gt.attrs["n_substitutions"] = int(np.sum(atom_types == 2))
        gt.attrs["vacancy_rate"] = vacancy_rate
        gt.attrs["substitution_rate"] = substitution_rate

    n_atoms = len(positions)
    n_vac = int(np.sum(atom_types == 1))
    n_sub = int(np.sum(atom_types == 2))
    print(f"Done. {n_atoms} atoms, {n_vac} vacancies, {n_sub} substitutions")
    print(f"Grid: {grid_h}x{grid_w} tiles, tile_size={tile_size}, overlap={overlap}")


def main():
    parser = argparse.ArgumentParser(description="Generate test world for STEMGym")
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/simulated/test_world.h5"),
    )
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_world(
        output_path=args.output,
        world_size=args.size,
        tile_size=args.tile_size,
        overlap=args.overlap,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
