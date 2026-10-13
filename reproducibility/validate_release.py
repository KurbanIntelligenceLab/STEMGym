"""Verify pinned release files and HDF5 records; download absent worlds only."""

from pathlib import Path
import concurrent.futures
import hashlib
import json
import urllib.request
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUTPUT = ROOT / "development/verification"
OUTPUT.mkdir(parents=True, exist_ok=True)
manifest = json.loads((ROOT / "results/manuscript/release_manifest.json").read_text())
revision = manifest["sha"]


def verify(record):
    remote = record["rfilename"]
    path = (
        ROOT
        / ("data/simulated" if remote.startswith("worlds/") else "checkpoints")
        / Path(remote).name
    )
    from stem_gym.data import download_verified

    download_verified(manifest, remote, path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    result = {
        "file": remote,
        "sha256": h.hexdigest(),
        "bytes": path.stat().st_size,
        "matches_release": h.hexdigest() == record["lfs"]["sha256"],
    }
    if remote.endswith(".h5"):
        with h5py.File(path, "r") as f:
            meta = f["metadata"].attrs
            result["metadata"] = {k: np.asarray(v).tolist() for (k, v) in meta.items()}
            tiles = f["tiles"]
            expected = int(np.prod(meta["grid_shape"]))
            (finite, shaped, lo, hi) = (True, True, float("inf"), float("-inf"))
            for name in tiles:
                tile = tiles[name][...]
                finite &= bool(np.isfinite(tile).all())
                shaped &= tile.shape == (int(meta["tile_size_px"]),) * 2
                (lo, hi) = (min(lo, float(tile.min())), max(hi, float(tile.max())))
            gt = f["ground_truth"]
            n = len(gt["atom_positions"])
            aligned = all(
                (len(gt[k]) == n for k in ["atom_types", "defect_mask", "defect_types"])
            )
            pos = gt["atom_positions"][...]
            (fov_y, fov_x) = meta["fov_nm"]
            outside = (
                (pos[:, 0] < 0)
                | (pos[:, 1] < 0)
                | (pos[:, 0] > fov_x)
                | (pos[:, 1] > fov_y)
            )
            result.update(
                positions_outside_field=int(outside.sum()),
                geometry_consistent=not bool(outside.any()),
                position_min=pos.min(axis=0).tolist(),
                position_max=pos.max(axis=0).tolist(),
            )
            result.update(
                tile_count=len(tiles),
                expected_tile_count=expected,
                finite_tiles=finite,
                correct_tile_shapes=shaped,
                tile_range=[lo, hi],
                atom_count=n,
                aligned_annotation_lengths=aligned,
                finite_positions=bool(np.isfinite(pos).all()),
                ground_truth_fields=list(gt.keys()),
                defect_count=int(gt["defect_mask"][...].sum()),
                overview_shape=list(f["overview"].shape),
                mask_shape=list(f["valid_region"].shape),
            )
    print(remote, result["matches_release"], flush=True)
    return result


if __name__ == "__main__":
    records = [
        r
        for r in manifest["siblings"]
        if r["rfilename"].startswith("worlds/")
        and Path(r["rfilename"]).name not in ("test_world.h5", "replay_world.h5")
        or r["rfilename"]
        in (
            "checkpoints/atom_finder.pt",
            "checkpoints/defect_classifier.pt",
            "checkpoints/phase_identifier.pt",
        )
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(verify, records))
    result = {"repository": manifest["id"], "revision": revision, "records": results}
    (OUTPUT / "release_validation.json").write_text(json.dumps(result, indent=2))
    if not all((r["matches_release"] for r in results)):
        raise ValueError("Local files differ from pinned release")
    for r in results:
        if "tile_count" in r:
            if not (
                r["tile_count"] == r["expected_tile_count"]
                and r["finite_tiles"]
                and r["correct_tile_shapes"]
                and r["aligned_annotation_lengths"]
                and r["finite_positions"]
            ):
                raise ValueError(
                    "Validation failed: r['tile_count'] == r['expected_tile_count'] and r['finite_tiles'] and r['correct_tile_shapes'] and r['aligned_annotation_lengths'] and r['finite_positions']"
                )
