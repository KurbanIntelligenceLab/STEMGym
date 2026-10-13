# STEMGym

Source and reproducibility evidence for **STEMGym: Benchmarking Sequential Decision-Making under Dose Budgets in Autonomous Electron Microscopy**.

The article and supplement are kept locally in ignored `development/final_manuscript_mlst/` and are excluded from this repository and its distributions. The original 220-episode study, release identities, audit records, and frozen source are retained in [`results/manuscript/`](results/manuscript/). New evaluations use the corrected synchronous implementation and have separate provenance.

## Installation with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run from the repository root:

```sh
uv sync --locked
uv run --locked python reproducibility/prepare_test_data.py
uv run --locked pytest -m "not slow and not training"
```

`pyproject.toml` declares dependencies; `uv.lock` records their resolved versions. `.python-version` selects Python 3.14 for local development. Both CI jobs use Python 3.14, the newest version supported by the current dependency wheels. Package metadata permits Python 3.10 and later; older versions are not part of CI. On macOS with Python 3.10, SciPy 1.14.1 is pinned to avoid the [upstream wheel-loading issue](https://github.com/scipy/scipy/issues/25635); this combination passes the test suite. The environment is isolated and does not require an existing Conda installation. Do not copy an inherited system-site-packages environment into `.venv`.

Simulation dependencies are optional:

```sh
uv sync --locked --extra sim
uv run --locked --extra sim pytest -m slow
```

For CUDA simulation on Linux x86-64, add `--extra cuda` alongside `--extra sim`. CuPy is separate from the CPU simulation dependencies. CI runs one-epoch training smoke tests on eight tiles and at most 32 defect patches, checking finite losses, parameter updates, and checkpoint roundtrips. Run them with `uv run --locked pytest -m training_smoke`. Longer convergence tests remain available with `uv run --locked pytest -m "training and not training_smoke"`.

## Data and manuscript evidence

Worlds and checkpoints are hosted on [Hugging Face](https://huggingface.co/datasets/kurbanintelligencelab/stemgym/tree/aa12b7380fed7a4362a18aa67df1ea135f7e15ff). Downloads are pinned to that revision and checked against SHA-256 hashes before publication to the local cache. Existing mismatched files cause an error and are not overwritten.

```sh
uv run --locked stemgym-download
uv run --locked python reproducibility/validate_release.py
uv run --locked python reproducibility/verify_reference.py
```

The release contains 15 material worlds. The 13-world core contains 13,600 tiles and passes coordinate-bounds checks. Hard SrTiO3 and hard Pt/C remain additional records with inconsistent field bounds. Coordinate-bounds consistency does not establish physical calibration; the preserved image audit documents sampling and registration findings. Two separate test/replay fixtures are excluded from benchmark results.

`stemgym-download --worlds`, `--checkpoints`, and `--test-only` select subsets. HDF5 data and model weights remain ignored under `data/` and `checkpoints/`.

## Evaluation and figures

The manuscript compares five configurations on four easy crystalline worlds over ten seeds, with 20 additional trained-raster episodes on medium/hard BaTiO3. Every reference episode acquires 50 distinct tiles, consumes 5,000 dose units, and scores after each five acquisitions.

The original study is immutable evidence. The current evaluator fixes generic acquisition ordering, reset behavior, and coordinate handling; fresh results must be treated as a new implementation version. Recompute figures and tables from the preserved study without changing the manuscript:

```sh
uv run --locked python reproducibility/build_figures.py
uv run --locked python reproducibility/verify_reference.py --artifacts-dir development/generated
```

Run the current complete reference protocol into a new results file:

```sh
uv run --locked python reproducibility/evaluate_reference.py --output development/runs/reference.json
uv run --locked python reproducibility/build_figures.py --results development/runs/reference.json --output-dir development/generated-current
uv run --locked python reproducibility/verify_reference.py --results development/runs/reference.json --artifacts-dir development/generated-current
```

The evaluator resumes one JSON file per run. It refuses to resume when its source, imported implementation, lockfile, or seed count changes. Default outputs are under ignored `development/`; output guards protect the canonical manuscript and original evidence. Use `CUDA_VISIBLE_DEVICES=""` for CPU-only GP fitting. Analyst inference uses CPU.

## Implementation

| Location | Responsibility |
|---|---|
| `stem_gym/envs/` | World access, observations, validated actions and dose accounting |
| `stem_gym/geometry.py` | Shared tile/position conversion |
| `stem_gym/agents/` | Trained and heuristic image analysis, public findings |
| `stem_gym/baselines/` | Seeded action selection and acquisition feedback |
| `stem_gym/runner.py` | Shared acquire → observe/analyze → score loop |
| `stem_gym/evaluation.py` | Reference configuration and policy/analyst composition |
| `stem_gym/scoring.py`, `tasks/`, `metrics.py` | Ground-truth selection and scoring |
| `stem_gym/data.py`, `scripts/` | Verified downloads, training and generation |
| `reproducibility/` | Verification, figure generation, auditing and source packaging |
| `results/manuscript/` | Tracked historical manuscript evidence |
| `development/` | Ignored exploratory code, old manuscripts, fresh runs and build outputs |

Agents implement `reset(obs, info, seed=...)`, `act(obs, info)`, `observe(obs, info)` and `get_findings()`. When driving an agent manually, call `observe` immediately after every environment step, including the terminating acquisition. `run_episode` handles this automatically. Successful acquisitions and unique visited tiles have separate counters. Analyst reset restores the initially loaded heads and clears episode-specific fallback state.

## Verification and packaging

```sh
uv lock --check
uv pip check
uv run --locked ruff check stem_gym tests reproducibility
uv run --locked pytest -m "not slow"
uv run --locked python reproducibility/audit_calibration.py
uv run --locked python reproducibility/package_source.py
uv build --out-dir development/dist
```

The review archive is `development/releases/stemgym_mlst_source.zip`; every included file is rechecked against its embedded hash manifest. The wheel is the runtime library and download CLI, with the pinned dataset manifest included. The source distribution and review archive additionally include reproduction scripts, tests, lockfile and reference evidence. Development files, data caches, credentials and trained weights are excluded.

Compile the multi-file journal project with TeX Live and `latexmk`, retaining the supplied IOP class. Generate build products outside the canonical manuscript directory:

```sh
mkdir -p development/latex
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd -outdir=../latex development/final_manuscript_mlst/main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd -outdir=../latex development/final_manuscript_mlst/supplement.tex
```

## License

Code: MIT. Data: CC-BY-4.0. Scientific authorship is recorded in the manuscript.
