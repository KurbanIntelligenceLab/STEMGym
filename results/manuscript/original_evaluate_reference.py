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
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
WORLDS = ["srtio3_clustered_easy", "batio3_interface_easy", "sige_gradient_clustered_easy", "gan_easy"]
POLICIES = ["raster_heuristic", "raster_trained", "random_trained", "gp_intensity_trained", "gp_defect_trained"]


def episode(job):
    import numpy as np
    import torch
    from stem_gym.envs.stem_env import STEMEnv
    from stem_gym.agents.analyst import Analyst
    from stem_gym.agents.simple_analyst import SimpleAnalyst
    from stem_gym.baselines.equipped_baselines import _assemble_findings
    from stem_gym.baselines.gp_ucb import GPUCBAgent
    from stem_gym.tasks.defect_census import DefectCensusTask
    from stem_gym.runner import filter_gt_to_explored
    from stem_gym.metrics import compute_dec
    torch.set_num_threads(1)
    torch.manual_seed(job["seed"])
    task = DefectCensusTask()
    env = STEMEnv(ROOT / "data/simulated" / (job["world"] + ".h5"), task=task, total_budget=5000, max_steps=200)
    obs, info = env.reset(seed=job["seed"])
    trained = job["policy"] != "raster_heuristic"
    analyst = Analyst.from_checkpoint(ROOT / "checkpoints", device="cpu") if trained else SimpleAnalyst()
    if trained:
        assert all(getattr(analyst, k) is not None for k in ("atom_finder", "defect_classifier", "phase_identifier")), "Missing trained checkpoint"
    nav = GPUCBAgent(seed=job["seed"], initial_random=4, beta=2.0)
    nav.reset(obs, info)
    rng = np.random.default_rng(job["seed"])
    shape = env.world.grid_shape
    log, tiles = [], []
    started = time.perf_counter()
    for step in range(50):
        if job["policy"].startswith("raster"):
            row, col = divmod(step, shape[1])
        elif job["policy"].startswith("random"):
            candidates = np.argwhere(~env.state.visited_mask)
            row, col = map(int, candidates[rng.integers(len(candidates))])
        else:
            nav._step = step * 2
            nx, ny = nav._select_next_position(obs)
            row, col = nav._nm_to_grid(nx, ny)
        assert not env.state.visited_mask[row, col]
        x, y = env.world.tile_to_nm(row, col)
        env.step({"action_type": 0, "position": np.array([x, y])})
        obs, _, terminated, truncated, info = env.step({"action_type": 1})
        tile = obs["tile"]
        if trained:
            result = analyst.analyze_tile(tile, tile_position_nm=(x, y), pixel_size_nm=env.world.pixel_size_nm, tile_key=(row, col))
        else:
            result = analyst.analyze_tile(tile)
        tiles.append([int(row), int(col)])
        if job["policy"].startswith("gp"):
            signal = float(result.n_vacancies + result.n_substitutions) if "defect" in job["policy"] else float(np.std(tile))
            nav._X.append([row / (shape[0] - 1), col / (shape[1] - 1)])
            nav._Y.append(signal)
            nav._visited_set.add((row, col))
            if len(nav._X) >= 4 and len(nav._X) % 3 == 0 and step < 49:
                nav._fit_gp()
        if (step + 1) % 5 == 0:
            findings = _assemble_findings(analyst, shape) if trained else {"defect_census": analyst.get_defect_census(), "defect_positions": []}
            gt = filter_gt_to_explored(env.world.get_ground_truth(), env.state.visited_mask, env.world)
            score = task.score(findings, gt)
            log.append({"dose_used": env.state.dose_used, "task_score": score})
        if terminated or truncated:
            break
    assert len(tiles) == 50 and env.state.tiles_acquired == 50
    assert len(analyst._tile_results) == 50
    dec = compute_dec(log, 5000)
    flags = {k: getattr(analyst, k, None) is not None for k in ("atom_finder", "defect_classifier", "phase_identifier")}
    env.close()
    return {**job, "dec_auc": dec.auc, "final_score": float(dec.scores[-1]), "tiles": tiles, "analysed_tiles": 50, "dose_used": 5000, "score_log": log, "dec_dose_fractions": dec.dose_fractions.tolist(), "dec_scores": dec.scores.tolist(), "models_active_at_end": flags, "elapsed_seconds": time.perf_counter() - started}


def key(j):
    return (j["world"], j["policy"], j["seed"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--output", type=Path, default=Path(__file__).with_name("reference_results.json"))
    a = p.parse_args()
    jobs = [{"world": w, "policy": pol, "seed": seed} for w in WORLDS for pol in POLICIES for seed in range(a.seeds)]
    jobs += [{"world": "batio3_interface_" + d, "policy": "raster_trained", "seed": seed} for d in ("medium", "hard") for seed in range(a.seeds)]
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    content = {"protocol": {"script_sha256": digest, "dataset_revision": "aa12b7380fed7a4362a18aa67df1ea135f7e15ff", "seeds": a.seeds, "budget": 5000, "dose_per_tile": 100, "task": "defect_census", "score_every": 5, "integration_checkpoints": 50, "cpu_threads_per_worker": 1, "description": "Synchronous acquisition, seeded detector noise, true tile centres; released analyst including built-in fallbacks."}, "episodes": []}
    if a.output.exists():
        content = json.loads(a.output.read_text())
        assert content["protocol"]["script_sha256"] == digest, "Changed script: choose a fresh output file"
        assert content["protocol"]["seeds"] == a.seeds
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
            print(f"{len(content['episodes'])}/{len(jobs)} {r['world']} {r['policy']} s{r['seed']}: {r['dec_auc']:.4f} ({r['elapsed_seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
