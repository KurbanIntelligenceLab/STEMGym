"""Independently verify a selected run and its generated tables and summary."""

from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from evaluate_reference import WORLDS, POLICIES, source_hash

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EVIDENCE = ROOT / "results/manuscript"


def verify(results_path, artifacts_dir=None):
    results = json.loads(results_path.read_text())
    from stem_gym.artifacts import verify_preservation

    verify_preservation(EVIDENCE)
    original = results_path.resolve() == (EVIDENCE / "reference_results.json").resolve()
    evaluator = (
        EVIDENCE / "original_evaluate_reference.py"
        if original
        else HERE / "evaluate_reference.py"
    )
    if (
        not results["protocol"]["script_sha256"]
        == hashlib.sha256(evaluator.read_bytes()).hexdigest()
    ):
        raise ValueError(
            "Validation failed: results['protocol']['script_sha256'] == hashlib.sha256(evaluator.read_bytes()).hexdigest()"
        )
    if not original:
        if not results["protocol"]["source_sha256"] == source_hash():
            raise ValueError(
                "Validation failed: results['protocol']['source_sha256'] == source_hash()"
            )
    for field, value in {
        "dataset_revision": "aa12b7380fed7a4362a18aa67df1ea135f7e15ff",
        "seeds": 10,
        "budget": 5000,
        "dose_per_tile": 100,
        "task": "defect_census",
        "score_every": 5,
        "integration_checkpoints": 50,
    }.items():
        if not results["protocol"][field] == value:
            raise ValueError("Validation failed: results['protocol'][field] == value")
    expected = {(w, p, s) for w in WORLDS for p in POLICIES for s in range(10)}
    expected |= {
        ("batio3_interface_" + d, "raster_trained", s)
        for d in ["medium", "hard"]
        for s in range(10)
    }
    actual = {(e["world"], e["policy"], e["seed"]) for e in results["episodes"]}
    if not (actual == expected and len(results["episodes"]) == 220):
        raise ValueError(
            "Validation failed: actual == expected and len(results['episodes']) == 220"
        )
    for e in results["episodes"]:
        if (
            not len(e["tiles"])
            == len({tuple(t) for t in e["tiles"]})
            == e["analysed_tiles"]
            == 50
        ):
            raise ValueError(
                "Validation failed: len(e['tiles']) == len({tuple(t) for t in e['tiles']}) == e['analysed_tiles'] == 50"
            )
        if not e["dose_used"] == 5000:
            raise ValueError("Validation failed: e['dose_used'] == 5000")
        log = e["score_log"]
        if not [z["dose_used"] for z in log] == list(range(500, 5001, 500)):
            raise ValueError(
                "Validation failed: [z['dose_used'] for z in log] == list(range(500, 5001, 500))"
            )
        x = np.linspace(0, 1, 50)
        scores = np.interp(
            x,
            np.array([z["dose_used"] for z in log]) / 5000,
            [z["task_score"] for z in log],
            left=0,
            right=log[-1]["task_score"],
        )
        if not np.all((scores >= 0) & (scores <= 1)):
            raise ValueError("Validation failed: np.all((scores >= 0) & (scores <= 1))")
        # Interpolation may differ by a few ULPs across NumPy builds/CPUs.
        # Use the same absolute precision as the independent integral checks.
        if not (
            np.allclose(x, e["dec_dose_fractions"], rtol=0, atol=1e-12)
            and np.allclose(scores, e["dec_scores"], rtol=0, atol=1e-12)
        ):
            raise ValueError(
                "Recomputed dose fractions or scores differ from stored curves (atol=1e-12)"
            )
        area = np.sum(np.diff(x) * (scores[1:] + scores[:-1]) / 2)
        if not abs(float(area) - e["dec_auc"]) < 1e-12:
            raise ValueError(
                "Validation failed: abs(float(area) - e['dec_auc']) < 1e-12"
            )
        if not scores[-1] == e["final_score"]:
            raise ValueError("Validation failed: scores[-1] == e['final_score']")
    if not original and artifacts_dir is None:
        raise ValueError("New studies require --artifacts-dir with a generated summary")
    summary_path = (
        EVIDENCE / "reference_summary.json"
        if original and (artifacts_dir is None or not (artifacts_dir / "reproducibility/reference_summary.json").exists())
        else artifacts_dir / "reproducibility/reference_summary.json"
    )
    summary = json.loads(summary_path.read_text())
    if not set(summary) == set(WORLDS):
        raise ValueError("Validation failed: set(summary) == set(WORLDS)")
    table = (artifacts_dir / "sec_experiments.tex").read_text() if artifacts_dir else None
    for w in WORLDS:
        if not set(summary[w]) == set(POLICIES):
            raise ValueError("Validation failed: set(summary[w]) == set(POLICIES)")
        for p in POLICIES:
            stats = summary[w][p]
            v = np.array(
                [
                    e["dec_auc"]
                    for e in results["episodes"]
                    if e["world"] == w and e["policy"] == p
                ]
            )
            if not len(v) == 10:
                raise ValueError("Validation failed: len(v) == 10")
            if not abs(v.mean() - stats["mean"]) < 1e-12:
                raise ValueError(
                    "Validation failed: abs(v.mean() - stats['mean']) < 1e-12"
                )
            if not abs(v.std(ddof=1) - stats["sample_sd"]) < 1e-12:
                raise ValueError(
                    "Validation failed: abs(v.std(ddof=1) - stats['sample_sd']) < 1e-12"
                )
    labels = [
        "Raster + heuristic",
        "Raster + trained",
        "Random + trained",
        "GP intensity + trained",
        "GP defects + trained",
    ]
    for p, label in zip(POLICIES, labels) if table is not None else ():
        cells = [
            f"${summary[w][p]['mean']:.3f} \\pm {summary[w][p]['sample_sd']:.3f}$"
            for w in WORLDS
        ]
        if not label + " & " + " & ".join(cells) + " \\\\" in table:
            raise ValueError(
                "Validation failed: label + ' & ' + ' & '.join(cells) + ' \\\\\\\\' in table"
            )
    return {
        "episodes": 220,
        "all_expected_configurations_and_seeds": True,
        "all_unique_50_acquisitions": True,
        "all_curves_and_integrals_recomputed": True,
        "summary_values_recomputed": True,
        "table_values_recomputed": table is not None,
        "results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=EVIDENCE / "reference_results.json"
    )
    parser.add_argument(
        "--artifacts-dir", type=Path, help="Optional directory containing the summary and results table"
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional JSON destination for this verification only",
    )
    args = parser.parse_args()
    checks = verify(args.results, args.artifacts_dir)
    if args.report:
        from stem_gym.artifacts import writable_output

        args.report = writable_output(args.report, ROOT)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(checks, indent=2) + "\n")
    print(
        "220 episodes: complete configurations, unique acquisitions, curves, integrals and summary values verified."
    )


if __name__ == "__main__":
    main()
