from pathlib import Path
import hashlib, json, urllib.request
import h5py
import numpy as np
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUTPUT = ROOT / "results/verification"
OUTPUT.mkdir(parents=True, exist_ok=True)
REV = "aa12b7380fed7a4362a18aa67df1ea135f7e15ff"
URL = f"https://huggingface.co/api/datasets/kurbanintelligencelab/stemgym/revision/{REV}?blobs=true"


def main():
    with urllib.request.urlopen(URL, timeout=60) as response:
        remote = json.load(response)
    records = []
    for name in ["srtio3_clustered_hard", "pt_nanoparticles_hard"]:
        path = ROOT / "data/simulated" / f"{name}.h5"
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(8 * 1024 * 1024):
                digest.update(block)
        digest = digest.hexdigest()
        expected = next(
            (
                x["lfs"]["sha256"]
                for x in remote["siblings"]
                if x["rfilename"] == f"worlds/{name}.h5"
            )
        )
        if not digest == expected:
            raise ValueError("Validation failed: digest == expected")
        with h5py.File(path) as f:
            m = f["metadata"].attrs
            gt = f["ground_truth"]
            p = gt["atom_positions"][...]
            shape = np.array(m["world_shape_px"])
            size = float(m["pixel_size_nm"])
            fov = np.array(m["fov_nm"])
            if not np.allclose(shape * size, fov):
                raise ValueError("Validation failed: np.allclose(shape * size, fov)")
            outside = (p < 0).any(1) | (p[:, 0] > fov[1]) | (p[:, 1] > fov[0])
            pix = p.astype(float) / size
            outside_pixels = (
                (pix < 0).any(1) | (pix[:, 0] > shape[1]) | (pix[:, 1] > shape[0])
            )
            if not (np.array_equal(outside, outside_pixels) and outside.any()):
                raise ValueError(
                    "Validation failed: np.array_equal(outside, outside_pixels) and outside.any()"
                )
            records.append(
                dict(
                    file=path.name,
                    sha256=digest,
                    matches_fresh_public_manifest=True,
                    image_shape_yx=shape.tolist(),
                    pixel_size_nm=size,
                    declared_fov_yx_nm=fov.tolist(),
                    ground_truth_fov_nm=float(gt.attrs["fov_nm"])
                    if "fov_nm" in gt.attrs
                    else None,
                    position_max_xy_nm=p.max(0).tolist(),
                    position_count=len(p),
                    positions_outside_declared_field=int(outside.sum()),
                    defects_outside_declared_field=int(
                        (outside & gt["defect_mask"][...]).sum()
                    ),
                    independent_pixel_bounds_check_agrees=True,
                    axis_swap_cannot_resolve_square_field=bool(fov[0] == fov[1]),
                )
            )
    lattice = []
    for level in ["easy", "medium", "hard"]:
        with h5py.File(ROOT / "data/simulated" / f"srtio3_clustered_{level}.h5") as f:
            gt = f["ground_truth"]
            pos = gt["atom_positions"][:500000]
            species = gt["atom_species"][:500000]
            sr = pos[species == b"Sr"]
            delta = np.diff(np.unique(sr[:, 0]))
            spacing = float(np.median(delta[delta > 0.0001]))
            per = []
            for name in ["0_0", "10_10", "20_20"]:
                a = f["tiles/" + name][...].astype(float)
                cor = [
                    float(np.corrcoef(a[:, :-lag].ravel(), a[:, lag:].ravel())[0, 1])
                    for lag in range(1, 45)
                ]
                peaks = find_peaks(cor)[0] + 1
                per.append(
                    dict(
                        tile=name,
                        zero_fraction=float(np.mean(a == 0)),
                        correlation_peaks_px=[
                            int(k) for k in peaks if cor[k - 1] > 0.8
                        ],
                        correlation_by_lag_1_to_44=cor,
                    )
                )
            lattice.append(
                dict(
                    world=f"srtio3_clustered_{level}",
                    stored_Sr_period_nm=spacing,
                    metadata_pixel_nm=float(f["metadata"].attrs["pixel_size_nm"]),
                    expected_Sr_period_pixels_from_metadata=spacing
                    / float(f["metadata"].attrs["pixel_size_nm"]),
                    tiles=per,
                )
            )
    coverage = []
    for r in json.loads(
        (ROOT / "results/manuscript/release_validation.json").read_text()
    )["records"]:
        if not r["file"].endswith(".h5"):
            continue
        with h5py.File(ROOT / "data/simulated" / Path(r["file"]).name) as f:
            coverage.append(
                dict(
                    world=Path(r["file"]).stem,
                    overview_zero_fraction=float(np.mean(f["overview"][...] == 0)),
                    valid_region_all_true=bool(f["valid_region"][...].all()),
                )
            )
    result = dict(
        public_manifest_url=URL,
        revision=REV,
        conclusion="Both flagged published files have inconsistent image-coordinate metadata and ground-truth positions. Bounds consistency of the other 13 does not establish physical calibration. Lattice and image-coverage checks flag a broader registration issue.",
        disposition="Preserve original records; list the two bounds failures as additional records; require scan-sampling, coverage and image-label registration checks before physically calibrated evaluation.",
        records=records,
        lattice_registration_checks=lattice,
        image_coverage_checks=coverage,
        source_review="The supplied generator uses GridScan() without explicit sampling but assembles and writes geometry using potential sampling times PRISM interpolation. Official abTEM documentation says unspecified scan sampling follows probe Nyquist sampling. This is a candidate cause, not a verified correction.",
        source_url="https://abtem.github.io/doc/reference/api/_autosummary/abtem.scan.GridScan.html",
    )
    (OUTPUT / "calibration_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        "Confirmed both field-bounds failures; recorded independent lattice and coverage checks."
    )


if __name__ == "__main__":
    main()
