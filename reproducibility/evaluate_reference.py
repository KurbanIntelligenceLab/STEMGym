"""Reproduce the compact dataset-paper evaluation from the repository root.

Every acquisition is analysed at its true tile centre before scoring or
selecting the next tile. Results are saved in one resumable JSON file.
"""

from __future__ import annotations
import os

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stem_gym.evaluation import WORLDS, POLICIES, evaluate_episode


def episode(job):
    return evaluate_episode(job, ROOT)


def source_hash():
    """Fingerprint the evaluator, imported implementation, and lockfile."""
    h = hashlib.sha256()
    paths = [Path(__file__), ROOT / "uv.lock"] + sorted(
        (ROOT / "stem_gym").rglob("*.py")
    )
    for path in paths:
        h.update(str(path.relative_to(ROOT)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def key(j):
    return (j["world"], j["policy"], j["seed"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument(
        "--output", type=Path, default=ROOT / "results/runs/reference.json"
    )
    a = p.parse_args()
    if a.seeds < 1 or a.workers < 1:
        p.error("seeds and workers must be positive")
    from stem_gym.artifacts import writable_output

    a.output = writable_output(a.output, ROOT)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    jobs = [
        {"world": w, "policy": pol, "seed": seed}
        for w in WORLDS
        for pol in POLICIES
        for seed in range(a.seeds)
    ]
    jobs += [
        {"world": "batio3_interface_" + d, "policy": "raster_trained", "seed": seed}
        for d in ("medium", "hard")
        for seed in range(a.seeds)
    ]
    from stem_gym.data import download_verified, release_manifest

    manifest = release_manifest()
    for world in sorted({job["world"] for job in jobs}):
        download_verified(
            manifest,
            "worlds/" + world + ".h5",
            ROOT / "data/simulated" / (world + ".h5"),
        )
    for name in ["atom_finder.pt", "defect_classifier.pt", "phase_identifier.pt"]:
        download_verified(manifest, "checkpoints/" + name, ROOT / "checkpoints" / name)
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    content = {
        "protocol": {
            "script_sha256": digest,
            "dataset_revision": "aa12b7380fed7a4362a18aa67df1ea135f7e15ff",
            "source_sha256": source_hash(),
            "protocol_version": "2",
            "seeds": a.seeds,
            "budget": 5000,
            "dose_per_tile": 100,
            "task": "defect_census",
            "score_every": 5,
            "integration_checkpoints": 50,
            "cpu_threads_per_worker": 1,
            "description": "Synchronous acquisition, seeded detector noise, true tile centres; released analyst including built-in fallbacks.",
        },
        "episodes": [],
    }
    if a.output.exists():
        content = json.loads(a.output.read_text())
        if (
            content["protocol"].get("source_sha256") != source_hash()
            or content["protocol"]["seeds"] != a.seeds
        ):
            raise ValueError(
                "Source, lockfile or seed count changed: choose a fresh output file"
            )
    done = {key(j) for j in content["episodes"]}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        pending = {pool.submit(episode, j): j for j in jobs if key(j) not in done}
        for future in as_completed(pending):
            r = future.result()
            content["episodes"].append(r)
            content["episodes"].sort(key=key)
            tmp = a.output.with_suffix(".tmp")
            tmp.write_text(json.dumps(content, indent=2))
            tmp.replace(a.output)
            print(
                f"{len(content['episodes'])}/{len(jobs)} {r['world']} {r['policy']} s{r['seed']}: {r['dec_auc']:.4f} ({r['elapsed_seconds']:.1f}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
