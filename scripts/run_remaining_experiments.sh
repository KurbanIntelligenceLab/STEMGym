#!/bin/bash
# =============================================================================
# run_remaining_experiments.sh
#
# Runs remaining experiments on an H200 Vast.ai instance:
#   1. Particle Census     (135 episodes)
#   2. Pareto Sweep 1k/3k/10k (180 episodes)
#   3. Replay Comparison   (24 episodes)
#   4. Transfer Experiment (training + 240 episodes)
#
# Total: ~579 episodes + 4 training runs
#
# Assumes: cwd is the repo root, Python/CUDA available.
# =============================================================================
set -euo pipefail

export TERM=dumb
export NO_COLOR=1
export PYTHONUNBUFFERED=1

# H200 CUDA path for CuPy (not needed for experiments, but just in case)
export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda-12.4}"

echo "================================================================"
echo "  STEMGym — Remaining Experiments"
echo "  $(date)"
echo "================================================================"
echo ""

# --- Environment setup ---
echo "=== Environment setup ==="
pip uninstall -y colorama 2>/dev/null || true
pip install -e ".[dev]" 2>&1 | tail -5
echo ""

# --- Download worlds and checkpoints from HuggingFace ---
echo "=== Downloading worlds from HuggingFace ==="
python scripts/download_data.py 2>&1 | tee /workspace/download.log

# Download replay_world.h5 separately (not in standard download script)
REPLAY_DIR="data/replay"
REPLAY_FILE="$REPLAY_DIR/replay_world.h5"
HF_BASE="https://huggingface.co/datasets/johnpolat/STEMGym/resolve/main"
if [[ ! -f "$REPLAY_FILE" ]]; then
    echo "=== Downloading replay_world.h5 ==="
    mkdir -p "$REPLAY_DIR"
    wget -q --show-progress -O "$REPLAY_FILE" \
        "${HF_BASE}/worlds/replay_world.h5" 2>&1 || {
        echo "WARNING: replay_world.h5 not available on HuggingFace."
        echo "  Replay comparison will be skipped."
    }
fi

# --- Verify key world files ---
echo ""
echo "=== Verifying world files ==="
python3 -c "
import os, h5py, sys

checks = {
    'pt_nanoparticles_easy': 'data/simulated/pt_nanoparticles_easy.h5',
    'pt_nanoparticles_medium': 'data/simulated/pt_nanoparticles_medium.h5',
    'pt_nanoparticles_hard': 'data/simulated/pt_nanoparticles_hard.h5',
    'batio3_interface_easy': 'data/simulated/batio3_interface_easy.h5',
    'batio3_interface_medium': 'data/simulated/batio3_interface_medium.h5',
    'batio3_interface_hard': 'data/simulated/batio3_interface_hard.h5',
    'test_world': 'data/simulated/test_world.h5',
    'replay_world': 'data/replay/replay_world.h5',
    'srtio3_easy': 'data/simulated/srtio3_clustered_easy.h5',
    'batio3_easy': 'data/simulated/batio3_interface_easy.h5',
    'sige_easy': 'data/simulated/sige_gradient_clustered_easy.h5',
    'gan_easy': 'data/simulated/gan_easy.h5',
}
ok = 0
missing = 0
for name, path in checks.items():
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024*1024)
        try:
            with h5py.File(path, 'r') as f:
                n_tiles = sum(1 for k in f.keys() if k == 'tiles')
            print(f'  OK  {name:35s} {size_mb:8.1f} MB')
            ok += 1
        except Exception as e:
            print(f'  ERR {name:35s} {e}')
            missing += 1
    else:
        print(f'  MISSING {name:35s} {path}')
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

mkdir -p results

# ===================================================================
# Experiment 1: Particle Census (135 episodes)
# ===================================================================
echo ""
echo "================================================================"
echo "  [1/6] PARTICLE CENSUS — 135 episodes"
echo "  9 agents x particle_census x 3 Pt worlds x 5 seeds"
echo "================================================================"
RESULT_FILE="results/particle_census.json"
if [[ -f "$RESULT_FILE" ]]; then
    echo "  SKIP: $RESULT_FILE already exists"
else
    stemgym run --config experiments/configs/particle_census.yaml -j 4 2>&1 \
        | tee "/workspace/particle_census.log"
    echo "  DONE: particle_census -> $RESULT_FILE"
fi

# ===================================================================
# Experiment 2a: Pareto Sweep 1k (60 episodes)
# ===================================================================
echo ""
echo "================================================================"
echo "  [2/6] PARETO SWEEP 1k — 60 episodes"
echo "  4 agents x defect_census x 3 BaTiO3 x 5 seeds, budget=1000"
echo "================================================================"
RESULT_FILE="results/pareto_sweep_1k.json"
if [[ -f "$RESULT_FILE" ]]; then
    echo "  SKIP: $RESULT_FILE already exists"
else
    stemgym run --config experiments/configs/pareto_sweep_1k.yaml -j 4 2>&1 \
        | tee "/workspace/pareto_sweep_1k.log"
    echo "  DONE: pareto_sweep_1k -> $RESULT_FILE"
fi

# ===================================================================
# Experiment 2b: Pareto Sweep 3k (60 episodes)
# ===================================================================
echo ""
echo "================================================================"
echo "  [3/6] PARETO SWEEP 3k — 60 episodes"
echo "  4 agents x defect_census x 3 BaTiO3 x 5 seeds, budget=3000"
echo "================================================================"
RESULT_FILE="results/pareto_sweep_3k.json"
if [[ -f "$RESULT_FILE" ]]; then
    echo "  SKIP: $RESULT_FILE already exists"
else
    stemgym run --config experiments/configs/pareto_sweep_3k.yaml -j 4 2>&1 \
        | tee "/workspace/pareto_sweep_3k.log"
    echo "  DONE: pareto_sweep_3k -> $RESULT_FILE"
fi

# ===================================================================
# Experiment 2c: Pareto Sweep 10k (60 episodes)
# ===================================================================
echo ""
echo "================================================================"
echo "  [4/6] PARETO SWEEP 10k — 60 episodes"
echo "  4 agents x defect_census x 3 BaTiO3 x 5 seeds, budget=10000"
echo "================================================================"
RESULT_FILE="results/pareto_sweep_10k.json"
if [[ -f "$RESULT_FILE" ]]; then
    echo "  SKIP: $RESULT_FILE already exists"
else
    stemgym run --config experiments/configs/pareto_sweep_10k.yaml -j 4 2>&1 \
        | tee "/workspace/pareto_sweep_10k.log"
    echo "  DONE: pareto_sweep_10k -> $RESULT_FILE"
fi

# ===================================================================
# Experiment 3: Replay Comparison (24 episodes)
# ===================================================================
echo ""
echo "================================================================"
echo "  [5/6] REPLAY COMPARISON — 24 episodes"
echo "  4 agents x defect_census x 2 worlds x 3 seeds"
echo "================================================================"
RESULT_FILE="results/replay_comparison.json"
if [[ -f "$RESULT_FILE" ]]; then
    echo "  SKIP: $RESULT_FILE already exists"
elif [[ ! -f "data/replay/replay_world.h5" ]]; then
    echo "  SKIP: replay_world.h5 not available"
else
    stemgym run --config experiments/configs/replay_comparison.yaml 2>&1 \
        | tee "/workspace/replay_comparison.log"
    echo "  DONE: replay_comparison -> $RESULT_FILE"
fi

# ===================================================================
# Experiment 4: Transfer Experiment (training + 240 episodes)
# ===================================================================
echo ""
echo "================================================================"
echo "  [6/6] TRANSFER EXPERIMENT"
echo "  Step A: Train per-material checkpoints (4 materials x 3 models)"
echo "  Step B: Run 3 agents x 4 train x 4 eval x 5 seeds = 240 episodes"
echo "================================================================"
RESULT_FILE="results/transfer_experiment.json"
if [[ -f "$RESULT_FILE" ]]; then
    echo "  SKIP: $RESULT_FILE already exists"
else
    # Step 4a: Train per-material checkpoints
    TRANSFER_BASE="checkpoints/transfer"
    MATERIALS=("srtio3" "batio3" "sige" "gan")
    WORLD_FILES=(
        "data/simulated/srtio3_clustered_easy.h5"
        "data/simulated/batio3_interface_easy.h5"
        "data/simulated/sige_gradient_clustered_easy.h5"
        "data/simulated/gan_easy.h5"
    )

    for i in "${!MATERIALS[@]}"; do
        mat="${MATERIALS[$i]}"
        world="${WORLD_FILES[$i]}"
        cp_dir="$TRANSFER_BASE/$mat"

        if [[ -f "$cp_dir/atom_finder.pt" ]]; then
            echo "  SKIP training $mat: checkpoints already exist"
            continue
        fi

        if [[ ! -f "$world" ]]; then
            echo "  SKIP training $mat: world file missing ($world)"
            continue
        fi

        echo ""
        echo "  --- Training $mat ($world) ---"
        mkdir -p "$cp_dir"
        python scripts/train_atom_finder.py \
            --world "$world" \
            --output "$cp_dir" \
            --epochs 30 \
            --device cuda 2>&1 \
            | tee "/workspace/train_${mat}.log"
        echo "  DONE training $mat -> $cp_dir/"
    done

    # Step 4b: Run transfer matrix
    echo ""
    echo "  --- Running transfer experiment (240 episodes) ---"
    python scripts/run_transfer_experiment.py \
        --device cuda \
        --seeds 5 \
        --output "$RESULT_FILE" 2>&1 \
        | tee "/workspace/transfer_experiment.log"
    echo "  DONE: transfer_experiment -> $RESULT_FILE"
fi

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
    'particle_census.json': 135,
    'pareto_sweep_1k.json': 60,
    'pareto_sweep_3k.json': 60,
    'pareto_sweep_10k.json': 60,
    'replay_comparison.json': 24,
    'transfer_experiment.json': 48,
}

all_ok = True
for fname, expected_n in expected.items():
    path = f'results/{fname}'
    if not os.path.exists(path):
        print(f'  MISSING  {fname}')
        all_ok = False
        continue
    with open(path) as f:
        data = json.load(f)
    n = len(data.get('results', []))
    status = 'OK' if n >= expected_n else f'LOW ({n}/{expected_n})'
    if n < expected_n:
        all_ok = False
    print(f'  {status:8s}  {fname:35s}  {n:4d} results (expected {expected_n})')

if all_ok:
    print('\n  ALL EXPERIMENTS PASSED VERIFICATION')
else:
    print('\n  WARNING: Some experiments have fewer results than expected')
"

echo ""
echo "================================================================"
echo "  ALL REMAINING EXPERIMENTS COMPLETE"
echo "  $(date)"
echo "================================================================"
touch /workspace/DONE
