# Preserved manuscript evidence

The eight original JSON records are preserved byte-for-byte from the working repository before the refactor. `preservation.json` records their SHA-256 identities, together with the frozen evaluator and source archive. The commit identifier in that file is context; it does not identify the uncommitted source. `original_source.zip` contains the exact pre-refactor source, tests, reproduction scripts and dependency metadata, with its own per-file manifest.

| File | Purpose |
|---|---|
| `reference_results.json` | Original 220 episodes, visited cells and score trajectories |
| `reference_summary.json` | Means and sample standard deviations for the main table |
| `release_manifest.json` | Pinned dataset revision and remote file hashes |
| `release_validation.json` | Historical integrity, schema and bounds checks |
| `calibration_audit.json` | Historical sampling, registration and coverage audit |
| `figure_sources.json` | Original figure tile selection |
| `repeatability_checks.json` | Historical repeated-episode checks |
| `verification.json` | Historical environment and verification record |
| `original_evaluate_reference.py` | Exact evaluator whose hash appears in the original results |
| `original_source.zip` | Complete preserved implementation and supporting scripts |
| `preservation.json` | Integrity manifest for the files above |

Run `uv run --locked python reproducibility/verify_reference.py` from the repository root to verify preservation, recompute curves and integrals, and compare the preserved summary. Add `--artifacts-dir development/final_manuscript_mlst` to also check the local article's main table. Historical reports describe the original study, not the current refactored environment. No historical hashes are rewritten to make current code appear to have generated those results.

For historical execution, extract `original_source.zip` into a separate working directory and recreate its runtime from `verification.json`; supply the pinned data and checkpoints there. The frozen source retains its original paths. The active `uv.lock` describes the current implementation and does not claim to recreate the historical environment.

Current evaluations and verification reports are written under ignored `development/`. They never replace the original evidence. Figure-generation commands use the preserved results by default and write separate derived files.
