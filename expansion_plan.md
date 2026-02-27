# STEMGym Expansion Plan

**Goal:** Strengthen the benchmark for npj Computational Materials submission — more agents, deeper analysis, richer diagnostics — using existing worlds and infrastructure only.

**Approach:** Three tiers ordered by impact-to-effort ratio. Each tier is self-contained: complete it, update figures/tables, then decide whether to continue.

---

## Current State (2026-02-27)

| Metric | Value |
|--------|-------|
| Total episodes completed | 2,422 across 28 result files |
| Agents tested | 10 of 19 registered |
| Agents registered but untested | 9 (8 VLM + `single_bo`) |
| Missing equipped baseline | `RandomEquippedAgent` (not implemented) |
| Incomplete experiments | `defect_census` needs 210 more episodes (70 combos × 3 seeds) |
| VLM/LLM experiments | Running on H200 instance 32091145 |

---

## Tier 1 — Low-Hanging Fruit (1–2 days)

### 1.1 Implement & Run RandomEquippedAgent

**Why:** The causal decomposition (equipped vs raw) is the paper's secondary contribution. Random+Analyst is the missing cell — without it, the "analysis dominates navigation" claim has a gap.

**What:**
- Add `RandomEquippedAgent` to `stem_gym/baselines/equipped_baselines.py`, following the existing `RasterEquippedAgent` pattern (inherits from `RandomAgent`, adds analyst pipeline)
- Register as `random_equipped` in `experiments/run_benchmark.py` AGENT_REGISTRY
- Run across all 4 tasks × 5 worlds × 5 seeds (matching existing equipped agent coverage)

**Expected outcome:** Should score close to `raster_equipped` (~0.288 on defect census), reinforcing that navigation strategy is noise once analysis is fixed.

**Files to modify:**
- `stem_gym/baselines/equipped_baselines.py` — add class
- `experiments/run_benchmark.py` — register agent
- `experiments/configs/` — add config YAML or extend existing

### 1.2 Run SingleBOAgent

**Why:** Already implemented (`stem_gym/baselines/single_bo.py:21`), registered (`run_benchmark.py:63`), but zero results. It's the only agent that varies zoom ∈ [1,4] and dwell ∈ [0.1,10], making it uniquely interesting for the dose-efficiency story.

**What:**
- Create config YAML covering defect_census and phase_mapping × all 15 worlds × 5 seeds
- Run on GPU instance (BoTorch EI needs CUDA)
- Also create `SingleBOEquippedAgent` variant for causal decomposition completeness

**Expected outcome:** Likely poor raw performance (no trained analyst), but equipped variant tests whether adaptive zoom/dwell improves over fixed 1.0/1.0.

### 1.3 Complete Defect Census Seeds

**Why:** `main_defect_census.json` has 70 combos × 2 seeds. Other main experiments have 5 seeds. Inconsistency weakens statistical claims.

**What:**
- Re-run `main_defect_census.yaml` — incremental resume (commit `a52bf58`) will skip the 70 completed combos and run only the 210 remaining episodes (70 combos × 3 seeds)
- Runs on CPU, no GPU needed

---

## Tier 2 — Deeper Analysis (2–4 days)

### 2.1 Zoom & Dwell Ablation

**Why:** Every agent hardcodes `zoom=1.0, dwell=1.0`. The action space supports continuous zoom ∈ [1,4] and dwell ∈ [0.1,10], but this is entirely unexplored. A controlled ablation would reveal whether the benchmark's dose-efficiency story changes with non-trivial acquisition parameters.

**Hardcoded locations (all confirmed):**
- `random_agent.py:57,72`
- `equipped_baselines.py:130-131,162-163,236-237,260-261`
- `multi_agent.py:228-244` (default params, never overridden)

**What:**
- Create parametric variants of `RasterEquippedAgent` with fixed zoom ∈ {1.0, 2.0, 4.0} and dwell ∈ {0.5, 1.0, 2.0}
- Run 3×3 = 9 parameter combos × subset of worlds (e.g., 5 easy worlds × 5 seeds = 225 episodes)
- Compare DEC-AUC curves across parameter space

**Expected outcome:** Higher zoom = fewer tiles to cover same area but more dose per tile. Should see a Pareto front between coverage and per-tile signal quality.

### 2.2 Trajectory Serialization

**Why:** Position is logged per step (`runner.py:122`) but never serialized to JSON (`run_benchmark.py:266-291` omits it). Without trajectories, we can't analyze spatial exploration patterns, compute coverage metrics, or visualize agent behavior.

**What:**
- Extend `_serialize_result()` in `run_benchmark.py` to include per-step position data (optionally, behind a `--save-trajectories` flag to avoid bloating default output)
- Add trajectory visualization script (heatmaps of visited positions overlaid on specimen overview)
- Re-run a subset of experiments with trajectory saving enabled

**Files to modify:**
- `experiments/run_benchmark.py` — extend serialization
- New: `scripts/plot_trajectories.py`

### 2.3 DECLARE Action Integration

**Why:** The DECLARE action exists in the environment (`stem_env.py:222-234`) but no agent uses it. It was designed for agents to explicitly mark findings during exploration, enabling online scoring and a richer agent-environment protocol.

**What:**
- Add DECLARE calls to `STEMAgent` after analyst makes high-confidence detections
- Score declared findings against ground truth for a "precision of declarations" metric
- This creates a new evaluation dimension: not just final-score DEC-AUC, but also "when did the agent know?"

**Complexity:** Medium — requires modifying `multi_agent.py` and `analyst.py` to emit DECLARE actions, plus a new scoring pathway in tasks.

### 2.4 Acquisition Function Comparison

**Why:** GP-UCB is the only acquisition function (`navigator.py:115`: `ucb = mean + beta * std`). Comparing against Expected Improvement (EI) and Thompson Sampling (TS) would validate whether UCB is actually optimal for dose-efficient exploration.

**What:**
- Parameterize `Navigator` to accept acquisition function type
- Implement EI and TS variants (BoTorch has both built-in)
- Run STEMAgent with {UCB, EI, TS} × 5 easy worlds × defect_census × 5 seeds = 75 episodes

**Files to modify:**
- `stem_gym/agents/navigator.py` — add EI/TS options
- `stem_gym/agents/multi_agent.py` — pass acquisition function config

---

## Tier 3 — New Capabilities (1–2 weeks)

### 3.1 Multi-Objective DEC-AUC

**Why:** Currently each task produces a single score. Real microscopy often requires simultaneous defect detection AND phase mapping. A multi-objective evaluation would test whether agents can balance competing objectives under a shared dose budget.

**What:**
- Define composite task that scores both defect census AND phase mapping on BaTiO3 worlds (which have both ground truths)
- Compute Pareto-optimal DEC curves across the two objectives
- New metric: hypervolume indicator of the Pareto front at each dose checkpoint

### 3.2 Curriculum / Transfer Agents

**Why:** Transfer experiment shows perovskite→perovskite is strong (0.72–0.83 DEC-AUC) but cross-family is poor. An agent that adapts its analysis pipeline online could close this gap.

**What:**
- Implement `AdaptiveAnalyst` that fine-tunes atom finder on acquired tiles (self-supervised, using consistency between ensemble members)
- Test on cross-family transfer scenarios (e.g., SrTiO3-trained models on SiGe worlds)

### 3.3 Human Baseline via Replay

**Why:** `replay_world.h5` exists on HuggingFace. A human baseline (even from 1–2 expert microscopists) would ground all agent scores against the actual target user.

**What:**
- Build a simple web UI (Gradio/Streamlit) that presents the replay world as an interactive microscopy session
- Record human trajectories and decisions
- Score using identical DEC-AUC pipeline

---

## Execution Checklist

```
Tier 1 (do first — immediate paper impact)
  [x] 1.1 RandomEquippedAgent: implement, register, run (commit 99e6006)
  [x] 1.2 SingleBOAgent: create config, run (+ equipped variant) (commit 99e6006)
      - RandomEquippedAgent added to equipped_baselines.py
      - SingleBOEquippedAgent added to equipped_baselines.py
      - Both registered in AGENT_REGISTRY
      - Configs: random_equipped_run.yaml (200 eps), single_bo_run.yaml (300 eps)
      - Smoke tested: random_equipped DEC-AUC=0.005, single_bo=0.048, single_bo_equipped=0.292
      - Equipped analyst effect confirmed: 6× improvement (0.292 vs 0.048)
  [x] 1.3 defect_census_5seeds.yaml created (350 eps, fresh output to avoid resume skip)
      - Resume would skip 2-seed combos — needs fresh output file
  [ ] Run configs on GPU instance (single_bo_run needs CUDA for BoTorch)
  [ ] Run random_equipped_run on CPU
  [ ] Run defect_census_5seeds (CPU, but llm_planner needs API keys)
  [ ] Update paper Table 1 and Figure 2 with new agent data

Tier 2 (do next — deeper analysis for reviewers)
  [x] 2.1 Zoom/dwell ablation: implement parametric variants (commit 7d9756b)
      - 8 parametric variants registered (z2, z4, d05, d2, z2_d05, z2_d2, z4_d05, z4_d2)
      - Config: zoom_dwell_ablation.yaml
  [x] 2.2 Trajectory serialization: --save-trajectories flag (commit 7d9756b)
      - Position serialized in both sequential and parallel paths
      - Config: trajectory_serialization.yaml
  [x] 2.3 DECLARE integration: STEMAgent use_declare option (commit 7d9756b)
      - stem_agent_declare / stem_agent_no_declare registered
      - Config: declare_ablation.yaml
  [x] 2.4 Acquisition function comparison: EI/TS in acquisition.py (commit 7d9756b)
      - gp_ucb_ei, gp_ucb_ts, gp_ucb_ei_equipped, gp_ucb_ts_equipped, stem_agent_ei, stem_agent_ts
      - Config: acq_func_comparison.yaml
      - BUG FIX: EI normalization — callers passed raw Y, now normalized defensively
  [ ] Run Tier 2 configs on GPU instance
  [ ] Generate new figures for each completed experiment

Tier 3 (implemented, needs running)
  [x] 3.1 Multi-objective DEC-AUC
      - CompositeTask (defect_census + phase_mapping) in stem_gym/tasks/composite_task.py
      - Runner logs task_score_a, task_score_b for composite sub-scores
      - compute_pareto_hypervolume() in metrics.py for HV-DEC-AUC
      - Config: composite_task.yaml (5 agents × 3 BaTiO3 worlds × 5 seeds = 75 eps)
      - Plot script: scripts/plot_pareto_hypervolume.py
  [x] 3.2 Adaptive analyst (self-supervised transfer)
      - AdaptiveAnalyst in stem_gym/agents/adaptive_analyst.py
      - AdaptiveRasterAgent, AdaptiveSTEMAgent in stem_gym/agents/adaptive_agent.py
      - Online fine-tuning: member 0 pseudo-labels → BCE loss for members 1+2
      - BN layers frozen, conv weights only, lr=1e-5, 3 gradient steps per update
      - Config: adaptive_transfer.yaml (4 agents × 3 worlds × 5 seeds = 60 eps)
  [x] 3.3 Human baseline via Gradio
      - scripts/human_baseline_ui.py — HumanSession + Gradio Blocks UI
      - Click-to-navigate, acquire, declare findings, DEC-AUC scoring
      - Results saved to results/human_baseline/ as JSON
      - scripts/compare_human_vs_agents.py — overlay DEC curves
      - pip install -e ".[human]" for gradio dependency
  [ ] Run Tier 3 experiments on GPU instance
```

---

## Episode Count Estimates

| Item | Episodes | GPU Required |
|------|----------|-------------|
| 1.1 RandomEquipped (4 tasks × 5 worlds × 5 seeds) | 100 | No (CPU) |
| 1.2 SingleBO raw (2 tasks × 15 worlds × 5 seeds) | 150 | Yes (BoTorch) |
| 1.2 SingleBO equipped (same) | 150 | Yes |
| 1.3 Defect census remaining | 210 | No (CPU) |
| 2.1 Zoom/dwell ablation (9 combos × 5 worlds × 5 seeds) | 225 | No (CPU) |
| 2.4 Acquisition function (3 AF × 5 worlds × 5 seeds) | 75 | Yes (BoTorch) |
| **Total Tier 1+2** | **~910** | |

All Tier 1+2 episodes can run on existing infrastructure (H200 instance for GPU tasks, local/CPU for the rest).
