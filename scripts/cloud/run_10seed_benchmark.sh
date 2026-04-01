#!/usr/bin/env bash
# =============================================================================
# STEMGym 10-Seed Benchmark + RL Training
# Run on a fresh Vast.ai A100 SXM4 instance
#
# Pre-requisites:
#   1. Deploy key at /root/.ssh/id_ed25519 (scp from local before running)
#   2. Instance created with: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
#   3. apt-get install -y git openssh-client (done in onstart)
#
# Usage:
#   bash run_10seed_benchmark.sh 2>&1 | tee /workspace/benchmark_run.log
# =============================================================================

set -euo pipefail

export TERM=dumb
export NO_COLOR=1
export PYTHONUNBUFFERED=1

WORKSPACE=/workspace
REPO_DIR=$WORKSPACE/STEMGym

echo "============================================================"
echo "  STEMGym 10-Seed Benchmark"
echo "  $(date)"
echo "============================================================"

# ------------------------------------------------------------------
# Step 1: Clone repo + install dependencies
# ------------------------------------------------------------------
echo ""
echo "[Step 1/6] Environment setup..."

ssh-keyscan github.com >> /root/.ssh/known_hosts 2>/dev/null

if [ -d "$REPO_DIR" ]; then
    echo "  Repo already exists, pulling latest..."
    cd "$REPO_DIR"
    git pull
else
    cd "$WORKSPACE"
    git clone git@github.com:KurbanIntelligenceLab/STEMGym.git
    cd "$REPO_DIR"
fi

pip install -e ".[rl]" 2>&1 | tail -5
pip uninstall -y colorama 2>/dev/null || true  # transitive dep, RecursionError on Vast.ai

echo "  Done. Python: $(python --version), torch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# ------------------------------------------------------------------
# Step 2: Download worlds + neural model checkpoints
# ------------------------------------------------------------------
echo ""
echo "[Step 2/6] Downloading data from HuggingFace..."

python scripts/download_data.py --worlds --checkpoints

# Delete old DQN checkpoint from HF — we train all RL agents fresh
rm -f checkpoints/dqn_agent.zip

# Verify worlds
WORLD_COUNT=$(ls data/simulated/*.h5 2>/dev/null | wc -l)
echo "  World files found: $WORLD_COUNT (expect 17+)"
if [ "$WORLD_COUNT" -lt 15 ]; then
    echo "  ERROR: Too few world files. Check download_data.py output."
    exit 1
fi

# Verify neural model checkpoints (needed by equipped agents)
for f in atom_finder.pt defect_classifier.pt phase_identifier.pt; do
    if [ ! -f "checkpoints/$f" ]; then
        echo "  ERROR: Missing checkpoint: checkpoints/$f"
        exit 1
    fi
done
echo "  Neural model checkpoints: OK"

# ------------------------------------------------------------------
# Step 3: Train RL agents (DQN, PPO, SAC) — all fresh at 50K steps
# ------------------------------------------------------------------
echo ""
echo "[Step 3/6] Training RL agents (50K steps each)..."

for algo in dqn ppo sac; do
    echo "  Training $algo..."
    python scripts/train_rl.py --algo $algo --steps 50000 --device cuda \
        2>&1 | tee "$WORKSPACE/train_${algo}.log"

    if [ ! -f "checkpoints/${algo}_agent.zip" ]; then
        echo "  ERROR: Training failed — checkpoints/${algo}_agent.zip not found"
        exit 1
    fi
    echo "  $algo checkpoint: $(ls -la checkpoints/${algo}_agent.zip | awk '{print $5}') bytes"
done

echo "  All RL agents trained successfully."

# ------------------------------------------------------------------
# Step 4: Clear old results (fresh 10-seed run)
# ------------------------------------------------------------------
echo ""
echo "[Step 4/6] Clearing old results..."

for f in main_defect_census main_phase_mapping main_targeted particle_census trained_ablation_study; do
    rm -f "results/${f}.json"
done
echo "  Done."

# ------------------------------------------------------------------
# Step 5: Run 5 main configs (-j 4 parallel workers each)
# ------------------------------------------------------------------
echo ""
echo "[Step 5/6] Running benchmark configs..."

CONFIGS=(
    "main_defect_census"
    "main_phase_mapping"
    "main_targeted"
    "particle_census"
    "trained_ablation_study"
)

for config in "${CONFIGS[@]}"; do
    echo ""
    echo "  === Running $config ==="
    echo "  Started: $(date)"

    stemgym run --config "experiments/configs/${config}.yaml" -j 4 \
        2>&1 | tee "$WORKSPACE/run_${config}.log"

    echo "  Finished: $(date)"

    if [ ! -f "results/${config}.json" ]; then
        echo "  WARNING: results/${config}.json not found after run"
    fi
done

# ------------------------------------------------------------------
# Step 6: Verification
# ------------------------------------------------------------------
echo ""
echo "[Step 6/6] Verifying results..."

python -c "
import json, sys

expected = {
    'results/main_defect_census.json': 112,
    'results/main_phase_mapping.json': 48,
    'results/main_targeted.json': 32,
    'results/particle_census.json': 45,
    'results/trained_ablation_study.json': 45,
}

ok = True
for f, n in expected.items():
    try:
        d = json.load(open(f))
        actual = len(d['results'])
        status = 'OK' if actual >= n else 'MISSING'
        print(f'  {f}: {actual}/{n} combos [{status}]')
        if actual < n:
            ok = False
    except Exception as e:
        print(f'  {f}: ERROR - {e}')
        ok = False

if ok:
    print()
    print('  ALL CHECKS PASSED')
else:
    print()
    print('  SOME CHECKS FAILED — review logs')
    sys.exit(1)
"

touch "$WORKSPACE/DONE"

echo ""
echo "============================================================"
echo "  BENCHMARK COMPLETE"
echo "  $(date)"
echo "  Results in: $REPO_DIR/results/"
echo "  RL checkpoints in: $REPO_DIR/checkpoints/"
echo "  Signal file: $WORKSPACE/DONE"
echo "============================================================"
echo ""
echo "  To download results (from local machine):"
echo "    scp -P <PORT> root@<HOST>:$REPO_DIR/results/main_*.json results/"
echo "    scp -P <PORT> root@<HOST>:$REPO_DIR/results/particle_census.json results/"
echo "    scp -P <PORT> root@<HOST>:$REPO_DIR/results/trained_ablation_study.json results/"
echo "    scp -P <PORT> root@<HOST>:$REPO_DIR/checkpoints/*_agent.zip checkpoints/"
echo "    scp -P <PORT> root@<HOST>:$WORKSPACE/*.log results/logs/10seed_run/"
