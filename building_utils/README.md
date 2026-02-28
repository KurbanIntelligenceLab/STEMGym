# Building Utilities

Internal tooling — not needed by reviewers, but essential for development.

## Analysis Scripts

| Script | Purpose | Example |
|--------|---------|---------|
| `statistical_analysis.py` | Bootstrap CIs, Wilcoxon tests, LaTeX tables | `python building_utils/analysis/statistical_analysis.py results/` |
| `cadence_sensitivity.py` | Scoring cadence robustness (reviewer Q2) | `python building_utils/analysis/cadence_sensitivity.py` |
| `run_transfer_experiment.py` | 5×5 cross-material transfer matrix | `python building_utils/analysis/run_transfer_experiment.py` |
| `plot_trajectories.py` | Agent navigation path visualization | `python building_utils/analysis/plot_trajectories.py results/*.json` |
| `plot_pareto_hypervolume.py` | Pareto front + hypervolume for composite task | `python building_utils/analysis/plot_pareto_hypervolume.py` |
| `compare_human_vs_agents.py` | Overlay human vs agent DEC curves | `python building_utils/analysis/compare_human_vs_agents.py` |
| `run_all_experiments.py` | Batch orchestrator (deprecated — use `stemgym run`) | — |

## Cloud Deployment (Vast.ai)

| Script | Purpose |
|--------|---------|
| `run_expansion_experiments.sh` | Tier 2/3 + DQN experiments on A100 |
| `vastai_run_expansion.sh` | Vast.ai provisioning wrapper for expansion |
| `run_remaining_experiments.sh` | VLM/LLM experiments on H200 |
| `vastai_run_remaining.sh` | Vast.ai provisioning wrapper for H200 |
| `run_experiments_on_instance.sh` | Generic instance runner (legacy) |
| `vastai_run_experiments.sh` | Generic Vast.ai wrapper (legacy) |
| `vastai_generate_worlds.sh` | World generation via abTEM on GPU |

## Legacy Configs

Superseded experiment configs preserved for reference. Use the configs in
`experiments/configs/` instead.
