#!/usr/bin/env python3
"""Prepare a replay world from py4DSTEM experimental 4D-STEM data.

Downloads a py4DSTEM sample dataset (or loads a user-provided HAADF image),
computes virtual HAADF, derives ground truth via automated analysis
(peak-finding, intensity-based defect labeling, autocorrelation phase
estimation), and writes an HDF5 file in the standard World format.

Usage:
    # From py4DSTEM sample (downloads ~200MB 4D-STEM data)
    python data/scripts/prepare_replay.py --source py4dstem \
        --dataset small_datacube --output data/replay/replay_world.h5

    # From an existing HAADF image (npy, tiff, etc.)
    python data/scripts/prepare_replay.py --source image \
        --image path/to/haadf.npy --output data/replay/replay_world.h5

    # Generate synthetic replay for testing (no py4DSTEM needed)
    python data/scripts/prepare_replay.py --source synthetic \
        --output data/replay/replay_world.h5
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter, zoom as scipy_zoom
from skimage.feature import peak_local_max

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------

def load_haadf_from_py4dstem(
    dataset_name: str,
    download_dir: Path,
    inner_frac: float = 0.15,
    outer_frac: float = 0.6,
) -> tuple[np.ndarray, dict]:
    """Download a py4DSTEM sample and compute virtual HAADF.

    Args:
        dataset_name: py4DSTEM sample key (e.g. 'small_datacube', 'Si_SiGe_exp').
        download_dir: Directory to save downloaded files.
        inner_frac: Inner radius as fraction of max Q.
        outer_frac: Outer radius as fraction of max Q.

    Returns:
        haadf: (H, W) virtual HAADF image.
        metadata: Dict with calibration info.
    """
    import py4DSTEM
    from py4DSTEM.io.google_drive_downloader import gdrive_download

    download_dir.mkdir(parents=True, exist_ok=True)

    # Download if not already present
    logger.info(f"Downloading py4DSTEM sample '{dataset_name}'...")
    gdrive_download(dataset_name, destination=str(download_dir), overwrite=False)

    # Find the downloaded file
    from py4DSTEM.io.google_drive_downloader import file_ids
    filename, _ = file_ids[dataset_name]
    filepath = download_dir / filename

    logger.info(f"Loading DataCube from {filepath}...")
    dc = py4DSTEM.import_file(str(filepath))
    Rx, Ry, Qx, Qy = dc.shape
    logger.info(f"DataCube shape: {dc.shape}")

    # Compute virtual HAADF (annular dark field)
    center = (Qx // 2, Qy // 2)
    inner_r = int(max(Qx, Qy) * inner_frac)
    outer_r = int(max(Qx, Qy) * outer_frac)
    logger.info(f"Computing virtual HAADF: center={center}, inner={inner_r}, outer={outer_r}")

    vi = dc.get_virtual_image(
        mode="annular",
        geometry=(center, (inner_r, outer_r)),
        name="HAADF",
    )
    haadf = vi.data.astype(np.float32)

    # Get calibration
    cal = dc.calibration
    r_pixel_size = cal.get_R_pixel_size()
    r_pixel_units = cal.get_R_pixel_units() if hasattr(cal, "get_R_pixel_units") else "pixels"

    metadata = {
        "source": f"py4DSTEM/{dataset_name}",
        "r_pixel_size": float(r_pixel_size),
        "r_pixel_units": str(r_pixel_units),
        "diffraction_shape": (Qx, Qy),
        "inner_radius": inner_r,
        "outer_radius": outer_r,
    }
    return haadf, metadata


def load_haadf_from_image(image_path: Path) -> tuple[np.ndarray, dict]:
    """Load an HAADF image from a file (npy, tiff, png, etc.).

    Returns:
        haadf: (H, W) image.
        metadata: Source info.
    """
    suffix = image_path.suffix.lower()
    if suffix == ".npy":
        haadf = np.load(image_path).astype(np.float32)
    elif suffix in (".tif", ".tiff"):
        from skimage.io import imread
        haadf = imread(str(image_path)).astype(np.float32)
    elif suffix in (".png", ".jpg", ".jpeg"):
        from skimage.io import imread
        haadf = imread(str(image_path), as_gray=True).astype(np.float32)
    else:
        raise ValueError(f"Unsupported image format: {suffix}")

    if haadf.ndim > 2:
        haadf = haadf.mean(axis=-1)

    metadata = {"source": f"image/{image_path.name}"}
    return haadf, metadata


def generate_synthetic_replay(
    size: int = 512,
    spacing_px: int = 12,
    sigma: float = 1.8,
    seed: int = 123,
) -> tuple[np.ndarray, dict]:
    """Generate a synthetic HAADF image for testing the replay pipeline.

    Creates a realistic-looking HAADF with:
    - Slightly irregular atom spacing
    - Background gradient
    - Some intensity variation (simulating different Z)
    - Poisson noise (simulating real detector noise)

    Returns:
        haadf: (size, size) image.
        metadata: Source info.
    """
    rng = np.random.default_rng(seed)

    # Background gradient
    y, x = np.mgrid[0:size, 0:size]
    background = 0.05 + 0.02 * (x / size) + 0.01 * np.sin(4 * np.pi * y / size)

    # Atom lattice with slight irregularity
    image = np.zeros((size, size), dtype=np.float64)
    for r in range(spacing_px // 2, size - spacing_px // 4, spacing_px):
        for c in range(spacing_px // 2, size - spacing_px // 4, spacing_px):
            dr = rng.normal(0, 0.8)
            dc = rng.normal(0, 0.8)
            rr = int(np.clip(r + dr, 0, size - 1))
            cc = int(np.clip(c + dc, 0, size - 1))
            # Vary intensity slightly
            intensity = rng.normal(1.0, 0.15)
            image[rr, cc] += max(intensity, 0.3)

    image = gaussian_filter(image, sigma=sigma)
    image = image + background.astype(np.float64)

    # Add Poisson noise (simulating real detector)
    scale = 1000.0  # counts
    image_counts = np.clip(image * scale, 0, None)
    image_noisy = rng.poisson(image_counts).astype(np.float64) / scale

    # Normalize
    haadf = image_noisy.astype(np.float32)
    if haadf.max() > 0:
        haadf = haadf / haadf.max()

    metadata = {
        "source": "synthetic_replay",
        "spacing_px": spacing_px,
        "sigma": sigma,
    }
    return haadf, metadata


# ---------------------------------------------------------------------------
# Ground truth derivation
# ---------------------------------------------------------------------------

def derive_ground_truth(
    haadf: np.ndarray,
    pixel_size_nm: float,
    min_distance: int = 6,
    threshold_rel: float = 0.15,
    defect_z_threshold: float = 1.5,
) -> dict:
    """Derive ground truth from an HAADF image via automated analysis.

    Uses peak-finding for atom positions and intensity-based z-score
    analysis for defect labeling. This is approximate — the ground truth
    is derived, not known, which is an inherent property of replay data.

    Args:
        haadf: (H, W) HAADF image.
        pixel_size_nm: Pixel size in nm.
        min_distance: Minimum distance between peaks.
        threshold_rel: Relative threshold for peak detection.
        defect_z_threshold: Z-score threshold for defect classification.

    Returns:
        Dictionary with ground truth arrays.
    """
    # Smooth for peak detection
    smoothed = gaussian_filter(haadf.astype(np.float64), sigma=1.5)

    # Find atom positions
    thresh_abs = smoothed.max() * threshold_rel
    coords = peak_local_max(
        smoothed,
        min_distance=min_distance,
        threshold_abs=thresh_abs,
    )
    n_atoms = len(coords)
    logger.info(f"Found {n_atoms} atoms via peak-finding")

    if n_atoms == 0:
        return {
            "atom_positions": np.empty((0, 2), dtype=np.float32),
            "atom_types": np.empty(0, dtype=np.int32),
            "defect_mask": np.empty(0, dtype=bool),
            "defect_types": [],
            "n_atoms": 0,
        }

    # Positions in nm (x, y) — note coords are (row, col)
    positions_nm = coords[:, ::-1].astype(np.float32) * pixel_size_nm

    # Extract intensities at atom positions
    intensities = np.array([haadf[r, c] for r, c in coords], dtype=np.float32)

    # Z-score classification
    mean_i = intensities.mean()
    std_i = max(intensities.std(), 1e-6)
    zscores = (intensities - mean_i) / std_i

    atom_types = np.zeros(n_atoms, dtype=np.int32)
    defect_type_names = ["pristine"] * n_atoms

    for i, z in enumerate(zscores):
        if z < -defect_z_threshold:
            atom_types[i] = 1  # vacancy-like (dim)
            defect_type_names[i] = "vacancy"
        elif z > defect_z_threshold:
            atom_types[i] = 2  # substitution-like (bright)
            defect_type_names[i] = "substitution"

    defect_mask = atom_types > 0

    n_vac = int(np.sum(atom_types == 1))
    n_sub = int(np.sum(atom_types == 2))
    logger.info(f"Derived defects: {n_vac} vacancy-like, {n_sub} substitution-like")

    # Phase map via autocorrelation
    phase_map = derive_phase_map(haadf)

    return {
        "atom_positions": positions_nm,
        "atom_types": atom_types,
        "defect_mask": defect_mask,
        "defect_types": defect_type_names,
        "phase_map": phase_map,
        "n_atoms": n_atoms,
        "n_vacancies": n_vac,
        "n_substitutions": n_sub,
    }


def derive_phase_map(
    haadf: np.ndarray,
    block_size: int = 64,
) -> np.ndarray:
    """Derive a coarse phase map from HAADF intensity statistics.

    Divides the image into blocks and clusters them by mean intensity
    into 2 phases. Simple but effective for images with phase boundaries.

    Args:
        haadf: (H, W) HAADF image.
        block_size: Size of blocks for phase analysis.

    Returns:
        phase_map: (H, W) integer phase labels (0 or 1).
    """
    h, w = haadf.shape
    phase_map = np.zeros((h, w), dtype=np.int32)

    # Compute mean intensity per block
    block_means = []
    blocks = []
    for r in range(0, h, block_size):
        for c in range(0, w, block_size):
            block = haadf[r:min(r + block_size, h), c:min(c + block_size, w)]
            block_means.append(block.mean())
            blocks.append((r, c))

    if len(block_means) < 2:
        return phase_map

    # Simple threshold at median
    threshold = np.median(block_means)
    for (r, c), mean_val in zip(blocks, block_means):
        label = 1 if mean_val > threshold else 0
        r_end = min(r + block_size, h)
        c_end = min(c + block_size, w)
        phase_map[r:r_end, c:c_end] = label

    return phase_map


# ---------------------------------------------------------------------------
# Write HDF5 world
# ---------------------------------------------------------------------------

def write_replay_world(
    output_path: Path,
    haadf: np.ndarray,
    ground_truth: dict,
    pixel_size_nm: float,
    tile_size: int = 128,
    overlap: int = 4,
    source_metadata: dict | None = None,
) -> None:
    """Write a replay world HDF5 in the standard World format.

    If the HAADF image is smaller than one tile, it is upscaled.
    """
    h, w = haadf.shape

    # Ensure image is large enough for tiling
    min_size = tile_size * 2
    if h < min_size or w < min_size:
        scale = max(min_size / h, min_size / w)
        logger.info(f"Upscaling HAADF from {h}x{w} by {scale:.1f}x")
        haadf = scipy_zoom(haadf, scale, order=3)
        pixel_size_nm /= scale
        # Scale ground truth positions
        if len(ground_truth["atom_positions"]) > 0:
            # Positions are in nm — pixel_size changed so no rescaling needed
            pass
        # Recompute from upscaled image
        h, w = haadf.shape
        ground_truth = derive_ground_truth(haadf, pixel_size_nm)

    # Tile grid
    stride = tile_size - overlap
    grid_h = max(1, (h - tile_size) // stride + 1)
    grid_w = max(1, (w - tile_size) // stride + 1)
    fov_h = h * pixel_size_nm
    fov_w = w * pixel_size_nm

    # Overview (4x downsampled)
    overview = scipy_zoom(haadf, 0.25, order=1)

    # Valid region
    valid_region = np.ones((h, w), dtype=bool)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing replay world to {output_path}...")
    logger.info(f"  Image: {h}x{w}, pixel_size={pixel_size_nm:.4f} nm")
    logger.info(f"  Grid: {grid_h}x{grid_w}, tile_size={tile_size}, overlap={overlap}")

    with h5py.File(output_path, "w") as f:
        # Metadata
        meta = f.create_group("metadata")
        meta.attrs["pixel_size_nm"] = pixel_size_nm
        meta.attrs["tile_size_px"] = tile_size
        meta.attrs["overlap_px"] = overlap
        meta.attrs["grid_shape"] = [grid_h, grid_w]
        meta.attrs["fov_nm"] = [fov_h, fov_w]
        meta.attrs["world_shape_px"] = [h, w]
        meta.attrs["scenario"] = "replay"
        meta.attrs["difficulty"] = "real"
        if source_metadata:
            meta.attrs["source"] = str(source_metadata.get("source", "unknown"))
            meta.attrs["processing"] = "virtual_HAADF+peak_finding"

        # Overview
        f.create_dataset("overview", data=overview, dtype=np.float32)

        # Valid region
        f.create_dataset("valid_region", data=valid_region, dtype=bool)

        # Tiles
        tiles_group = f.create_group("tiles")
        for r in range(grid_h):
            for c in range(grid_w):
                y0 = r * stride
                x0 = c * stride
                tile = haadf[y0:y0 + tile_size, x0:x0 + tile_size]
                if tile.shape != (tile_size, tile_size):
                    padded = np.zeros((tile_size, tile_size), dtype=np.float32)
                    padded[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded
                tiles_group.create_dataset(
                    f"{r}_{c}", data=tile, dtype=np.float32, chunks=True,
                )

        # Ground truth
        gt = f.create_group("ground_truth")
        gt.create_dataset("atom_positions", data=ground_truth["atom_positions"])
        gt.create_dataset("atom_types", data=ground_truth["atom_types"])
        gt.create_dataset("defect_mask", data=ground_truth["defect_mask"])

        defect_bytes = [s.encode("utf-8") for s in ground_truth["defect_types"]]
        gt.create_dataset("defect_types", data=defect_bytes)

        if "phase_map" in ground_truth and ground_truth["phase_map"] is not None:
            gt.create_dataset("phase_map", data=ground_truth["phase_map"])

        gt.attrs["n_atoms"] = ground_truth.get("n_atoms", 0)
        gt.attrs["n_vacancies"] = ground_truth.get("n_vacancies", 0)
        gt.attrs["n_substitutions"] = ground_truth.get("n_substitutions", 0)
        gt.attrs["ground_truth_source"] = "automated_analysis"

    logger.info(f"Done. {ground_truth.get('n_atoms', 0)} atoms detected.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare a replay world from experimental STEM data"
    )
    parser.add_argument(
        "--source", choices=["py4dstem", "image", "synthetic"],
        default="synthetic",
        help="Data source type",
    )
    parser.add_argument(
        "--dataset", type=str, default="small_datacube",
        help="py4DSTEM sample name (for --source py4dstem)",
    )
    parser.add_argument(
        "--image", type=Path, default=None,
        help="Path to HAADF image (for --source image)",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/replay/replay_world.h5"),
    )
    parser.add_argument(
        "--pixel-size", type=float, default=0.05,
        help="Pixel size in nm (default 0.05 for ~5pm resolution)",
    )
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    # Load HAADF image
    if args.source == "py4dstem":
        download_dir = args.output.parent / "downloads"
        haadf, metadata = load_haadf_from_py4dstem(args.dataset, download_dir)
        # Use calibrated pixel size if available
        if metadata.get("r_pixel_units") == "A":
            args.pixel_size = metadata["r_pixel_size"] / 10.0  # Å → nm
        elif metadata.get("r_pixel_units") == "nm":
            args.pixel_size = metadata["r_pixel_size"]

    elif args.source == "image":
        if args.image is None:
            parser.error("--image required for --source image")
        haadf, metadata = load_haadf_from_image(args.image)

    elif args.source == "synthetic":
        haadf, metadata = generate_synthetic_replay(seed=args.seed)

    else:
        parser.error(f"Unknown source: {args.source}")

    # Normalize to [0, 1]
    haadf_min, haadf_max = haadf.min(), haadf.max()
    if haadf_max > haadf_min:
        haadf = (haadf - haadf_min) / (haadf_max - haadf_min)

    logger.info(f"HAADF image shape: {haadf.shape}")

    # Derive ground truth
    ground_truth = derive_ground_truth(haadf, args.pixel_size)

    # Write HDF5
    write_replay_world(
        args.output, haadf, ground_truth,
        pixel_size_nm=args.pixel_size,
        tile_size=args.tile_size,
        overlap=args.overlap,
        source_metadata=metadata,
    )


if __name__ == "__main__":
    main()
