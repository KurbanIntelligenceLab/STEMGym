# Reference results and provenance

`preservation.json` records SHA-256 identities for the reference results, evaluator, and source archive. `original_source.zip` contains the source, tests, reproduction scripts, and dependency metadata used for the reference study, with a per-file integrity manifest.

| File | Purpose |
|---|---|
| `reference_results.json` | Original 220 episodes, visited cells and score trajectories |
| `reference_summary.json` | Means and sample standard deviations for the main table |
| `release_manifest.json` | Pinned dataset revision and remote file hashes |
| `release_validation.json` | Reference integrity, schema and bounds checks |
| `calibration_audit.json` | Reference sampling, registration and coverage audit |
| `figure_sources.json` | Original figure tile selection |
| `repeatability_checks.json` | Reference repeated-episode checks |
| `verification.json` | Reference environment and verification record |
| `original_evaluate_reference.py` | Exact evaluator whose hash appears in the original results |
| `original_source.zip` | Complete preserved implementation and supporting scripts |
| `preservation.json` | Integrity manifest for the files above |

Run `uv run --locked python reproducibility/verify_reference.py` from the repository root to check file integrity and recompute curves, integrals, and summary statistics. After generating figures and tables, add `--artifacts-dir results/generated` to check the generated table.

To run the reference implementation, extract `original_source.zip` into a separate directory, recreate the runtime recorded in `verification.json`, and provide the pinned data and checkpoints. Use the repository's `uv.lock` for the current implementation. Each implementation has its own recorded provenance.
