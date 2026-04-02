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
python stem_gym/scripts/download_data.py              # everything (worlds + checkpoints + transfer)
python stem_gym/scripts/download_data.py --worlds     # only HDF5 worlds
python stem_gym/scripts/download_data.py --checkpoints # main checkpoints + DQN
python stem_gym/scripts/download_data.py --transfer   # transfer-learning checkpoints (~388 MB)
```

Or generate the test world locally (no GPU needed):

```bash
python stem_gym/scripts/generate_worlds/generate_test_world.py
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
stemgym run --config experiments/configs/main_defect_census.yaml --dry-run
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
| `random_equipped` | Random + trained Analyst |
| `single_bo_equipped` | Joint BO + trained Analyst |
| `dqn` | DQN RL navigation baseline (SB3) |
| `dqn_equipped` | DQN navigation + trained Analyst |
| `llm_planner` | LLM-based planner via OpenRouter |

Run `stemgym list agents` for the full list (41 agents including VLM, ablation, and acquisition function variants).

## Tasks

| Task | Metric | Description |
|------|--------|-------------|
| `defect_census` | F1 | Detect and classify crystal defects |
| `phase_mapping` | IoU | Identify material phases |
| `targeted` | Spatial F1 | Locate specific target features |
| `particle_census` | Detection + sizing | Nanoparticle detection, sizing, and morphology |
| `composite` | Weighted | Multi-objective: defect census + phase mapping |

## Project Structure

```
stem_gym/            Core package
  envs/              Gymnasium environment + HDF5 world
  agents/            Multi-agent system (Navigator, Analyst, Planner)
  baselines/         Baseline agents (Random, Raster, GP-UCB, SingleBO, DQN, VLM)
  tasks/             Task definitions and scoring
  scripts/           Data download, model training, human baseline UI
    generate_worlds/ World-generation scripts (abTEM simulation)
  metrics.py         DEC curve computation
  runner.py          Episode and benchmark execution
experiments/         Configs and CLI runner
building_utils/      Internal analysis and cloud deployment tooling
data/                Simulated worlds (HDF5)
checkpoints/         Trained neural model weights
tests/               Test suite
```

## License

Code: MIT. Data: CC-BY-4.0.

## Authors

Anonymous (double-blind review)

## Citation

Citation will be provided after review.
