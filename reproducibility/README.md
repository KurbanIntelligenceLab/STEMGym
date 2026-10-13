# Reproduction tools

Run these scripts from the repository root using `uv run --locked python reproducibility/<script>.py`. See the root README for complete commands.

- `prepare_test_data.py`: download and hash-check both test fixtures.
- `validate_release.py`: verify all 15 worlds and three reference checkpoints; write a new report under `results/verification/`.
- `audit_calibration.py`: repeat the bounds, lattice and coverage audit; write a new report under `results/verification/`.
- `verify_reference.py`: check frozen evidence and recompute the original 220 episodes' metrics and summary values; `--artifacts-dir` also checks table values. `--results` and `--artifacts-dir` select a complete new study.
- `evaluate_reference.py`: run the current implementation with ten seeds by default. `--seeds` can select a smaller study; the complete-table builder and verifier require ten. `--output` defaults to one resumable JSON file under `results/runs/`.
- `build_figures.py`: reproduce the two article figures, main results table, inventory table and summary; write into `results/generated/` by default.
- `package_source.py`: create a source archive under `dist/` and verify every archived byte against its manifest.

The reference study and its source snapshot are in `results/manuscript/`. Each new evaluation records the evaluator, package source, and lockfile fingerprints for reproducibility.
