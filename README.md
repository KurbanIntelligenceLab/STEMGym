# STEMGym

A Gymnasium Environment for Benchmarking Dose-Efficient Autonomous Scanning Transmission Electron Microscopy.

Agents navigate a simulated specimen under a limited electron dose budget to characterize defects, phases, and atomic structure. Performance is measured via Dose-Efficiency Curves (DEC-AUC).

**Dataset & Checkpoints:** [https://huggingface.co/datasets/johnpolat/STEMGym](https://huggingface.co/datasets/johnpolat/STEMGym)

## Installation

```bash
# Core package
pip install -e .

# With development tools (pytest)
pip install -e ".[dev]"

# With RL baselines (DQN, PPO, SAC via Stable-Baselines3)
pip install -e ".[rl]"

# With human baseline UI (Gradio)
pip install -e ".[human]"

# With abTEM simulation support (heavy, for world generation only)
pip install -e ".[sim]"
```

Requires Python >= 3.10.

## Data Setup

Download the simulated worlds and model checkpoints from [Hugging Face](https://huggingface.co/datasets/johnpolat/STEMGym):

```bash
python stem_gym/scripts/download_data.py              # everything (worlds + checkpoints + transfer)
python stem_gym/scripts/download_data.py --worlds     # only HDF5 worlds (~7.1 GB)
python stem_gym/scripts/download_data.py --checkpoints # main checkpoints + RL agents (~107 MB)
python stem_gym/scripts/download_data.py --transfer   # transfer-learning checkpoints (~388 MB)
python stem_gym/scripts/download_data.py --test-only  # only test_world.h5 (~10 MB, minimal)
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

## Materials

| Material | Zone Axis | FOV | Defect Types | Primary Task |
|----------|-----------|-----|-------------|-------------|
| SrTiO3 | [001] | ~100 nm | O vacancies clustered along grain boundary | Defect Census |
| BaTiO3 | [001] | ~100 nm | Cubic/tetragonal phase boundaries + O vacancies | Phase Mapping |
| SiGe | [110] | ~50 nm | Ge substitutions concentrated in one quadrant | Targeted |
| GaN | [11-20] | ~80 nm | InGaN quantum-well substitutions | Defect Census |
| Pt nanoparticles | — | ~60 nm | Pt nanoparticles on amorphous carbon | Particle Census |

Each material has 3 difficulty levels (easy, medium, hard) controlling defect density, phonon configurations, and dose budget. 15 worlds + test world + replay world = 17 total.

## Agents

| Agent | Type | Description |
|-------|------|-------------|
| `random` | Baseline | Random navigation |
| `raster` | Baseline | Systematic grid traversal |
| `gp_ucb` | Baseline | GP-UCB exploration |
| `single_bo` | Baseline | Joint Bayesian optimization |
| `stem_agent` | Multi-agent | Navigator (GP-UCB) + Analyst (neural ensemble) + Planner (FSM) |
| `raster_equipped` | Equipped | Raster navigation + trained Analyst |
| `gp_ucb_equipped` | Equipped | GP-UCB navigation + trained Analyst |
| `random_equipped` | Equipped | Random navigation + trained Analyst |
| `single_bo_equipped` | Equipped | Joint BO + trained Analyst |
| `dqn` / `ppo` / `sac` | RL | Stable-Baselines3 RL navigation baselines |
| `dqn_equipped` / `ppo_equipped` / `sac_equipped` | RL+Analyst | RL navigation + trained Analyst |
| `raster_vlm_*` / `gp_ucb_vlm_*` | VLM | Vision-language model analysis (Claude, Gemini, GPT-5, Llama) |
| `llm_planner` | LLM | LLM-based strategic planner via LangGraph |
| `stem_agent_ei` / `stem_agent_ts` | Acquisition | Alternative acquisition functions (EI, Thompson sampling) |
| `no_planner` / `no_uncertainty` / `rule_planner` | Ablation | Ablation variants of stem_agent |

45 agents total. Run `stemgym list agents` for the full list.

## Tasks

| Task | Metric | Description |
|------|--------|-------------|
| `defect_census` | F1 | Detect and classify crystal defects |
| `phase_mapping` | IoU | Identify material phases |
| `targeted` | Spatial F1 | Locate specific target features |
| `particle_census` | Detection + sizing | Nanoparticle detection, sizing, and morphology |
| `composite` | Weighted | Multi-objective: defect census + phase mapping |

## Training

```bash
# Train the analyst ensemble (AtomFinder + DefectClassifier + PhaseIdentifier)
python stem_gym/scripts/train_analyst.py --world data/simulated/test_world.h5 --output checkpoints/

# Train RL agents (DQN, PPO, SAC)
python stem_gym/scripts/train_rl.py --algo dqn --steps 50000
python stem_gym/scripts/train_rl.py --algo ppo --steps 50000
python stem_gym/scripts/train_rl.py --algo sac --steps 50000 --device cuda

# Human baseline UI
python stem_gym/scripts/human_baseline.py --world data/simulated/test_world.h5
```

## Project Structure

```
stem_gym/              Core package
  envs/                Gymnasium environment + HDF5 world
  agents/              Multi-agent system (Navigator, Analyst, Planner)
  baselines/           Baseline agents (Random, Raster, GP-UCB, SingleBO, RL, VLM)
  tasks/               Task definitions and scoring
  scripts/             Data download, model training, human baseline UI
    generate_worlds/   World-generation scripts (abTEM simulation)
  metrics.py           DEC curve computation
  runner.py            Episode and benchmark execution
experiments/           Configs and CLI runner
data/                  Simulated worlds (HDF5)
checkpoints/           Trained neural model weights + RL agents
tests/                 Test suite
```

## License

Code: MIT. Data: CC-BY-4.0.

## Authors

ANON

## Citation

```bibtex
@inproceedings{2026stemgym,
  title={STEMGym: A Gymnasium Environment for Benchmarking Dose-Efficient Autonomous Scanning Transmission Electron Microscopy},
  author={anon},
  year={2026}
}
```
