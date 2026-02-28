#!/bin/bash
# =============================================================================
# run_expansion_experiments.sh
#
# Runs expansion experiments (Tier 2/3 + DQN RL baseline) on a Vast.ai instance:
#   1.  Trajectory Capture         (24 episodes)
#   2.  DECLARE Ablation           (60 episodes)
#   3.  Zoom/Dwell Ablation        (225 episodes)
#   4.  Composite Task             (75 episodes)
#   5.  Adaptive Transfer          (45 episodes)
#   6.  Acq. Function Comparison   (150 episodes)
#   7.  Random Equipped Run        (200 episodes)
#   8.  Single BO Run              (300 episodes)
#   9.  Defect Census 5-seed       (350 episodes)
#  10.  DQN Training               (~50k steps)
#  11.  RL Baseline Evaluation     (50 episodes)
#
# Total: ~1,479 episodes + 1 DQN training
#
# Assumes: cwd is the repo root, Python/CUDA available.
# =============================================================================
set -euo pipefail

export TERM=dumb
export NO_COLOR=1
export PYTHONUNBUFFERED=1
export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda-12.4}"

echo "================================================================"
echo "  STEMGym — Expansion Experiments"
echo "  $(date)"
echo "================================================================"
echo ""

# --- Environment setup ---
echo "=== Environment setup ==="
pip uninstall -y colorama 2>/dev/null || true
pip install -e ".[dev,rl]" 2>&1 | tail -10
pip install openai langchain-openai langgraph 2>&1 | tail -5
echo ""

# --- Download worlds and checkpoints from HuggingFace ---
echo "=== Downloading worlds from HuggingFace ==="
python scripts/download_data.py 2>&1 | tee /workspace/download.log

# --- Create _verified symlinks (some configs use _verified suffix) ---
echo ""
echo "=== Creating _verified symlinks ==="
cd data/simulated
for f in *.h5; do
    base="${f%.h5}"
    if [[ "$base" != *_verified ]]; then
        ln -sf "$f" "${base}_verified.h5" 2>/dev/null || true
    fi
done
cd /workspace/STEMGym
echo "  Symlinks created."

# --- Verify key world files ---
echo ""
echo "=== Verifying world files ==="
python3 -c "
import os, h5py

checks = [
    'data/simulated/srtio3_clustered_easy.h5',
    'data/simulated/batio3_interface_easy.h5',
    'data/simulated/batio3_interface_medium.h5',
    'data/simulated/batio3_interface_hard.h5',
    'data/simulated/sige_gradient_clustered_easy.h5',
    'data/simulated/gan_easy.h5',
    'data/simulated/pt_nanoparticles_easy.h5',
]
ok = 0
missing = 0
for path in checks:
    name = os.path.basename(path)
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024*1024)
        try:
            with h5py.File(path, 'r') as f:
                pass
            print(f'  OK  {name:45s} {size_mb:8.1f} MB')
            ok += 1
        except Exception as e:
            print(f'  ERR {name:45s} {e}')
            missing += 1
    else:
        print(f'  MISSING {name:45s}')
        missing += 1
print(f'\n  {ok} OK, {missing} missing/error')
if missing > 0:
    print('  WARNING: Some worlds missing. Affected experiments may fail.')
"

# --- Verify GPU ---
echo ""
echo "=== GPU check ==="
python3 -c "
import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1e9
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {vram:.1f} GB')
    print(f'  CUDA: {torch.version.cuda}')
else:
    print('  WARNING: No GPU detected. Experiments will run on CPU (slow).')
" || true

# --- Verify DQN agents registered ---
echo ""
echo "=== Verifying agent registry ==="
python3 -c "
from experiments.run_benchmark import AGENT_REGISTRY
for name in ['dqn', 'dqn_equipped']:
    if name in AGENT_REGISTRY:
        print(f'  OK  {name}')
    else:
        print(f'  MISSING  {name}')
print(f'  Total registered: {len(AGENT_REGISTRY)} agents')
"

mkdir -p results

# ===================================================================
# Experiment 1: Trajectory Capture (24 episodes)
# ===================================================================
echo ""
echo "================================================================"
echo "  [1/11] TRAJECTORY CAPTURE — 24 episodes"
echo "  4 agents x defect_census x 2 worlds x 3 seeds"
echo "================================================================"
RESULT_FILE="results/trajectory_capture.json"
if [[ -f "$RESULT_FILE" ]]; then
    COMBOS=$(python3 -c "import json; d=json.load(open('$RESULT_FILE')); print(len(d.get('results',[])))")
    echo "  EXISTS: $RESULT_FILE ($COMBOS combos) — will resume"
fi
stemgym run --config experiments/configs/trajectory_capture.yaml 2>&1 \
    | tee "/workspace/trajectory_capture.log"
echo "  DONE: trajectory_capture"

# ===================================================================
# Experiment 2: DECLARE Ablation (60 episodes)
# ===================================================================
echo ""
echo "================================================================"
echo "  [2/11] DECLARE ABLATION — 60 episodes"
echo "  4 agents x defect_census x 3 worlds x 5 seeds"
echo "================================================================"
RESULT_FILE="results/declare_ablation.json"
if [[ -f "$RESULT_FILE" ]]; then
    COMBOS=$(python3 -c "import json; d=json.load(open('$RESULT_FILE')); print(len(d.get('results',[])))")
    echo "  EXISTS: $RESULT_FILE ($COMBOS combos) — will resume"
fi
stemgym run --config experiments/configs/declare_ablation.yaml -j 2 2>&1 \
    | tee "/workspace/declare_ablation.log"
echo "  DONE: declare_ablation"

# ===================================================================
# Experiment 3: Zoom/Dwell Ablation (225 episodes)
# ===================================================================
echo ""
echo "================================================================"
echo "  [3/11] ZOOM/DWELL ABLATION — 225 episodes"
echo "  9 agents x defect_census x 5 worlds x 5 seeds"
echo "================================================================"
RESULT_FILE="results/zoom_dwell_ablation.json"
if [[ -f "$RESULT_FILE" ]]; then
    COMBOS=$(python3 -c "import json; d=json.load(open('$RESULT_FILE')); print(len(d.get('results',[])))")
    echo "  EXISTS: $RESULT_FILE ($COMBOS combos) — will resume"
fi
stemgym run --config experiments/configs/zoom_dwell_ablation.yaml -j 4 2>&1 \
    | tee "/workspace/zoom_dwell_ablation.log"
echo "  DONE: zoom_dwell_ablation"

# ===================================================================
# Experiment 4: Composite Task (75 episodes)
# ===================================================================
echo ""
echo "================================================================"
echo "  [4/11] COMPOSITE TASK — 75 episodes"
echo "  5 agents x composite x 3 BaTiO3 worlds x 5 seeds"
echo "================================================================"
RESULT_FILE="results/composite_task.json"
if [[ -f "$RESULT_FILE" ]]; then
    COMBOS=$(python3 -c "import json; d=json.load(open('$RESULT_FILE')); print(len(d.get('results',[])))")
    echo "  EXISTS: $RESULT_FILE ($COMBOS combos) — will resume"
fi
stemgym run --config experiments/configs/composite_task.yaml -j 4 2>&1 \
    | tee "/workspace/composite_task.log"
echo "  DONE: composite_task"

# ===================================================================
# Experiment 5: Adaptive Transfer (45 episodes)
# ===================================================================
echo ""
echo "================================================================"
echo "  [5/11] ADAPTIVE TRANSFER — 45 episodes"
echo "  3 agents x defect_census x 3 worlds x 5 seeds"
echo "================================================================"
RESULT_FILE="results/adaptive_transfer.json"
if [[ -f "$RESULT_FILE" ]]; then
    COMBOS=$(python3 -c "import json; d=json.load(open('$RESULT_FILE')); print(len(d.get('results',[])))")
    echo "  EXISTS: $RESULT_FILE ($COMBOS combos) — will resume"
fi
stemgym run --config experiments/configs/adaptive_transfer.yaml -j 4 2>&1 \
    | tee "/workspace/adaptive_transfer.log"
echo "  DONE: adaptive_transfer"

# ===================================================================
# Experiment 6: Acquisition Function Comparison (150 episodes)
# ===================================================================
echo ""
echo "================================================================"
echo "  [6/11] ACQ. FUNCTION COMPARISON — 150 episodes"
echo "  6 agents x defect_census x 5 worlds x 5 seeds"
echo "================================================================"
RESULT_FILE="results/acq_func_comparison.json"
if [[ -f "$RESULT_FILE" ]]; then
    COMBOS=$(python3 -c "import json; d=json.load(open('$RESULT_FILE')); print(len(d.get('results',[])))")
    echo "  EXISTS: $RESULT_FILE ($COMBOS combos) — will resume"
fi
stemgym run --config experiments/configs/acq_func_comparison.yaml -j 4 2>&1 \
    | tee "/workspace/acq_func_comparison.log"
echo "  DONE: acq_func_comparison"

# ===================================================================
# Experiment 7: Random Equipped Run (200 episodes)
# ===================================================================
echo ""
echo "================================================================"
echo "  [7/11] RANDOM EQUIPPED RUN — 200 episodes"
echo "  1 agent x 4 tasks x 10 worlds x 5 seeds"
echo "================================================================"
RESULT_FILE="results/random_equipped_run.json"
if [[ -f "$RESULT_FILE" ]]; then
    COMBOS=$(python3 -c "import json; d=json.load(open('$RESULT_FILE')); print(len(d.get('results',[])))")
    echo "  EXISTS: $RESULT_FILE ($COMBOS combos) — will resume"
fi
stemgym run --config experiments/configs/random_equipped_run.yaml -j 4 2>&1 \
    | tee "/workspace/random_equipped_run.log"
echo "  DONE: random_equipped_run"

# ===================================================================
# Experiment 8: Single BO Run (300 episodes)
# ===================================================================
echo ""
echo "================================================================"
echo "  [8/11] SINGLE BO RUN — 300 episodes"
echo "  1 agent x 4 tasks x 15 worlds x 5 seeds"
echo "================================================================"
RESULT_FILE="results/single_bo_run.json"
if [[ -f "$RESULT_FILE" ]]; then
    COMBOS=$(python3 -c "import json; d=json.load(open('$RESULT_FILE')); print(len(d.get('results',[])))")
    echo "  EXISTS: $RESULT_FILE ($COMBOS combos) — will resume"
fi
stemgym run --config experiments/configs/single_bo_run.yaml -j 4 2>&1 \
    | tee "/workspace/single_bo_run.log"
echo "  DONE: single_bo_run"

# ===================================================================
# Experiment 9: Defect Census 5-seed (350 episodes)
# ===================================================================
echo ""
echo "================================================================"
echo "  [9/11] DEFECT CENSUS 5-SEED — 350 episodes"
echo "  10 agents x defect_census x 7 worlds x 5 seeds"
echo "================================================================"
RESULT_FILE="results/defect_census_5seeds.json"
if [[ -f "$RESULT_FILE" ]]; then
    COMBOS=$(python3 -c "import json; d=json.load(open('$RESULT_FILE')); print(len(d.get('results',[])))")
    echo "  EXISTS: $RESULT_FILE ($COMBOS combos) — will resume"
fi
stemgym run --config experiments/configs/defect_census_5seeds.yaml -j 4 2>&1 \
    | tee "/workspace/defect_census_5seeds.log"
echo "  DONE: defect_census_5seeds"

# ===================================================================
# Experiment 10: DQN Training (~50k steps)
# ===================================================================
echo ""
echo "================================================================"
echo "  [10/11] DQN TRAINING — ~50k steps"
echo "  Training DQN on easy worlds"
echo "================================================================"
DQN_CHECKPOINT="checkpoints/dqn_agent.zip"
if [[ -f "$DQN_CHECKPOINT" ]]; then
    echo "  SKIP: $DQN_CHECKPOINT already exists"
else
    python scripts/train_dqn.py \
        --steps 50000 \
        --output "$DQN_CHECKPOINT" \
        --device cuda 2>&1 \
        | tee "/workspace/dqn_training.log"
    echo "  DONE: DQN training -> $DQN_CHECKPOINT"
fi

# ===================================================================
# Experiment 11: RL Baseline Evaluation (50 episodes)
# ===================================================================
echo ""
echo "================================================================"
echo "  [11/11] RL BASELINE — 50 episodes"
echo "  2 agents x defect_census x 5 easy worlds x 5 seeds"
echo "================================================================"
RESULT_FILE="results/rl_baseline.json"
if [[ -f "$RESULT_FILE" ]]; then
    COMBOS=$(python3 -c "import json; d=json.load(open('$RESULT_FILE')); print(len(d.get('results',[])))")
    echo "  EXISTS: $RESULT_FILE ($COMBOS combos) — will resume"
fi
stemgym run --config experiments/configs/rl_baseline.yaml -j 4 2>&1 \
    | tee "/workspace/rl_baseline.log"
echo "  DONE: rl_baseline"

# ===================================================================
# Verification
# ===================================================================
echo ""
echo "================================================================"
echo "  VERIFICATION"
echo "================================================================"
python3 -c "
import json, os

expected = {
    'trajectory_capture.json': 8,
    'declare_ablation.json': 12,
    'zoom_dwell_ablation.json': 45,
    'composite_task.json': 15,
    'adaptive_transfer.json': 9,
    'acq_func_comparison.json': 30,
    'random_equipped_run.json': 40,
    'single_bo_run.json': 60,
    'defect_census_5seeds.json': 70,
    'rl_baseline.json': 10,
}

all_ok = True
total_combos = 0
for fname, expected_n in expected.items():
    path = f'results/{fname}'
    if not os.path.exists(path):
        print(f'  MISSING  {fname}')
        all_ok = False
        continue
    with open(path) as f:
        data = json.load(f)
    n = len(data.get('results', []))
    total_combos += n
    status = 'OK' if n >= expected_n else f'LOW ({n}/{expected_n})'
    if n < expected_n:
        all_ok = False
    print(f'  {status:8s}  {fname:35s}  {n:4d} combos (expected >= {expected_n})')

# Check DQN checkpoint
dqn_path = 'checkpoints/dqn_agent.zip'
if os.path.exists(dqn_path):
    size_kb = os.path.getsize(dqn_path) / 1024
    print(f'  OK        dqn_agent.zip                        {size_kb:.0f} KB')
else:
    print(f'  MISSING  dqn_agent.zip')
    all_ok = False

print(f'\n  Total combos: {total_combos}')
if all_ok:
    print('  ALL EXPERIMENTS PASSED VERIFICATION')
else:
    print('  WARNING: Some experiments have fewer results than expected')
"

echo ""
echo "================================================================"
echo "  ALL EXPANSION EXPERIMENTS COMPLETE"
echo "  $(date)"
echo "================================================================"
touch /workspace/DONE
