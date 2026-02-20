# STEMGym

A Gymnasium-based benchmark for evaluating autonomous dose-efficient scanning transmission electron microscopy (STEM) agents. Agents navigate a simulated specimen under a limited electron dose budget to characterize defects, phases, and atomic structure. Performance is measured via Dose-Efficiency Curves (DEC-AUC).

## Installation

```bash
# Core package
pip install -e .

# With development tools (pytest)
pip install -e ".[dev]"

# With abTEM simulation support (heavy)
pip install -e ".[sim]"
```

Requires Python >= 3.10.

## Data Setup

Download the simulated worlds and model checkpoints from Hugging Face:

```bash
python scripts/download_data.py
```

Or generate the test world locally (no GPU needed):

```bash
python data/scripts/generate_test_world.py
```

## Quick Start

```bash
# Run a quick smoke test
stemgym run --config experiments/configs/quick_smoke.yaml

# Run specific agents on a task
stemgym run --agent raster stem_agent --task defect_census --world test_world --seeds 2

# List available agents, tasks, and worlds
stemgym list agents
stemgym list tasks
stemgym list worlds

# Dry run (print plan without executing)
stemgym run --config experiments/configs/full_benchmark.yaml --dry-run
```

## Agents

| Agent | Description |
|-------|-------------|
| `random` | Random navigation baseline |
| `raster` | Systematic grid traversal |
| `gp_ucb` | GP-UCB exploration |
| `single_bo` | Joint Bayesian optimization |
| `stem_agent` | Full multi-agent system (Navigator + Analyst + Planner) |
| `raster_equipped` | Raster + trained Analyst (fair comparison) |
| `gp_ucb_equipped` | GP-UCB + trained Analyst (fair comparison) |

## Tasks

| Task | Metric | Description |
|------|--------|-------------|
| `defect_census` | F1 | Detect and classify crystal defects |
| `phase_mapping` | IoU | Identify material phases |
| `targeted` | Spatial F1 | Locate specific target features |

## Project Structure

```
stem_gym/          Core package
  envs/            Gymnasium environment + HDF5 world
  agents/          Multi-agent system (Navigator, Analyst, Planner)
  baselines/       Baseline agents (Random, Raster, GP-UCB, SingleBO)
  tasks/           Task definitions and scoring
  metrics.py       DEC curve computation
  runner.py        Episode and benchmark execution
experiments/       Configs and CLI runner
data/              Simulated worlds (HDF5) and generation scripts
checkpoints/       Trained neural model weights
tests/             Test suite
```

## License

Code: MIT. Data: CC-BY-4.0.

## Citation

```bibtex
@inproceedings{stemgym2026,
  title={STEMGym: A Benchmark for Dose-Efficient Autonomous STEM Microscopy},
  author={Polat, John},
  year={2026}
}
```
