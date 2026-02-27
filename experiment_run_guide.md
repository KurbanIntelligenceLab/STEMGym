# Experiment Run Guide

Checklist of all experiments needed for the revised paper. Organized by dependency and priority.

---

## A. Analysis Scripts (run locally, no GPU, uses existing data)

### A1. Statistical Analysis
```bash
python scripts/statistical_analysis.py
# Outputs: table to stdout + paper/figures/table_statistics.tex
# Uses: results/main_defect_census.json (14 matched pairs per comparison)
#       results/trained_ablation_study.json (15 matched pairs for ablations)
# Produces: Bootstrap 95% CIs + Wilcoxon p-values for key agent comparisons
```

### A2. Scoring Cadence Sensitivity
```bash
python scripts/cadence_sensitivity.py
# Outputs: paper/figures/fig_cadence_sensitivity.pdf + rank correlation table
# Uses: results/main_defect_census.json
# Tests: DEC-AUC ranking stability across scoring cadences [1, 2, 5, 10, 20]
```

### A3. Regenerate Figures (with bootstrap CIs)
```bash
python paper/generate_result_figures.py
# Regenerates all 7 figures + 2 tables with updated CI error bars
# Requires: all result JSONs in results/
```

### A4. Compile Paper
```bash
cd paper/npj && tectonic sn-article.tex
# Verify no LaTeX errors after text additions
```

---

## B. Experiments In Progress (H200, instance 32091145)

### B1. VLM Comparison (RUNNING)
```
# ssh -p 14009 root@208.64.254.72
# PID 23131, log: /workspace/vlm_v3.log
# Config: vlm_comparison.yaml — 900 episodes (280 done, 620 remaining)
# When done: scp results/vlm_comparison.json locally
```

### B2. LLM Planner (RUNNING)
```
# PID 23132, log: /workspace/llm_planner.log
# Config: llm_planner.yaml — 200 episodes
```

---

## C. Tier 1 Experiments (need to run)

### C1. Random Equipped Agent (CPU, ~2 hours)
```bash
stemgym run --config experiments/configs/random_equipped_run.yaml
# Output: results/random_equipped_run.json
```

### C2. Single BO Agent (GPU needed — BoTorch)
```bash
stemgym run --config experiments/configs/single_bo_run.yaml
# Output: results/single_bo_run.json
```

### C3. Defect Census 5 Seeds (CPU, ~4 hours)
```bash
stemgym run --config experiments/configs/defect_census_5seeds.yaml
# 10 agents × defect_census × 7 worlds × 5 seeds (fresh output file)
# Output: results/defect_census_5seeds.json
```

---

## D. Tier 2 Experiments (reviewer-motivated)

### D1. Zoom/Dwell Ablation (CPU) — addresses Q7
```bash
stemgym run --config experiments/configs/zoom_dwell_ablation.yaml
# Output: results/zoom_dwell_ablation.json
```

### D2. Acquisition Function Comparison (GPU) — addresses reviewer
```bash
stemgym run --config experiments/configs/acq_func_comparison.yaml
# Output: results/acq_func_comparison.json
```

### D3. DECLARE Ablation (CPU)
```bash
stemgym run --config experiments/configs/declare_ablation.yaml
# Output: results/declare_ablation.json
```

### D4. Trajectory Capture (CPU)
```bash
stemgym run --config experiments/configs/trajectory_capture.yaml
# Output: results/trajectory_capture.json
```

---

## E. Tier 3 Experiments (new capabilities)

### E1. Composite Task (GPU recommended)
```bash
stemgym run --config experiments/configs/composite_task.yaml
# Output: results/composite_task.json
```

### E2. Adaptive Transfer (GPU needed)
```bash
stemgym run --config experiments/configs/adaptive_transfer.yaml
# Output: results/adaptive_transfer.json
```

### E3. Human Baseline (manual, interactive)
```bash
python scripts/human_baseline_ui.py --world srtio3_clustered_easy --task defect_census
# Opens Gradio web UI on localhost:7860
```

---

## F. Post-Experiment Tasks

1. Download VLM results from H200:
   ```bash
   scp -P 14009 root@208.64.254.72:/workspace/STEMGym/results/vlm_comparison.json results/
   ```
2. Regenerate all figures: `python paper/generate_result_figures.py`
3. Rerun statistical analysis: `python scripts/statistical_analysis.py`
4. Compile paper: `cd paper/npj && tectonic sn-article.tex`

---

## Reviewer Comment → Experiment/Change Mapping

| Reviewer Concern | Response | Status |
|------------------|----------|--------|
| Statistical testing | `scripts/statistical_analysis.py` — bootstrap CIs + Wilcoxon | Script ready |
| Scoring cadence (Q2) | `scripts/cadence_sensitivity.py` — Spearman rank stability | Script ready |
| Analyst training details (Q1) | Paper: new paragraph + Appendix B table | Done |
| Particle census remediation (Q4) | Paper: expanded discussion + SimpleAnalyst reference | Done |
| Variable dwell (Q7) | `zoom_dwell_ablation.yaml` | Config ready, needs run |
| VLM comparison | `vlm_comparison.yaml` | Running on H200 |
| RL baselines (Q5) | Paper: acknowledged in Extensibility paragraph | Done |
| Navigation cost (Q3) | Paper: new Navigation Cost Model paragraph | Done |
| Real HAADF data (Q6) | Paper: new Real HAADF-STEM Data paragraph | Done |
| Compressed sensing | Paper: new Compressed Sensing paragraph | Done |
| Bootstrap CIs in figures | `generate_result_figures.py` updated | Done |

## Priority Order

1. **Section A** — run analysis scripts (immediate, from existing data)
2. **Section C** — fill gaps in main results table
3. **Section D1** — directly answers reviewer Q7
4. **Section B** — wait for H200 VLM to finish
5. **Section D2-D4** — strengthen specific claims
6. **Section E** — nice-to-have new capabilities
