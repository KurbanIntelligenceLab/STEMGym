#!/usr/bin/env bash
# =============================================================================
# vastai_run_expansion.sh
#
# Provisions an A100 instance, runs expansion experiments (Tier 2/3 + DQN RL),
# downloads results, and prompts before destroying.
#
# Usage:
#   bash scripts/vastai_run_expansion.sh
#   bash scripts/vastai_run_expansion.sh --gpu A100_SXM4 --dry-run
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOY_KEY="$SCRIPT_DIR/vastai_deploy_key"
LABEL="stemgym_expansion"
INSTANCE_ID=""
GPU_NAME="A100_SXM4"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu) GPU_NAME="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --label) LABEL="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

cleanup_reminder() {
    local exit_code=$?
    if [[ $exit_code -ne 0 && -n "$INSTANCE_ID" ]]; then
        echo ""
        echo "=========================================="
        echo "  SCRIPT FAILED — instance $INSTANCE_ID may still be running!"
        echo "  To clean up:  vastai destroy instance $INSTANCE_ID"
        echo "=========================================="
    fi
}
trap cleanup_reminder EXIT

# ── Preflight ─────────────────────────────────────────────────────
echo "==> Preflight checks..."

if ! command -v vastai &>/dev/null; then
    echo "ERROR: vastai CLI not found. Install with: pip install vastai"
    exit 1
fi

DEPLOY_KEY_CONTENT=""
if [[ -f "$DEPLOY_KEY" ]]; then
    DEPLOY_KEY_CONTENT=$(cat "$DEPLOY_KEY")
    echo "    Deploy key: $DEPLOY_KEY"
else
    echo "    WARNING: No deploy key. Will clone via HTTPS."
fi

echo "    GPU target: $GPU_NAME"
echo "    Label: $LABEL"

echo ""
echo "==> Experiment plan:"
echo "     1. Trajectory Capture        4 agents x 2 worlds x 3 seeds     =  24 episodes"
echo "     2. DECLARE Ablation          4 agents x 3 worlds x 5 seeds     =  60 episodes"
echo "     3. Zoom/Dwell Ablation       9 agents x 5 worlds x 5 seeds     = 225 episodes"
echo "     4. Composite Task            5 agents x 3 worlds x 5 seeds     =  75 episodes"
echo "     5. Adaptive Transfer         3 agents x 3 worlds x 5 seeds     =  45 episodes"
echo "     6. Acq. Func. Comparison     6 agents x 5 worlds x 5 seeds     = 150 episodes"
echo "     7. Random Equipped           1 agent  x 10 worlds x 4 tasks    = 200 episodes"
echo "     8. Single BO Run             1 agent  x 15 worlds x 4 tasks    = 300 episodes"
echo "     9. Defect Census 5-seed      10 agents x 7 worlds x 5 seeds    = 350 episodes"
echo "    10. DQN Training              ~50k steps"
echo "    11. RL Baseline               2 agents x 5 worlds x 5 seeds     =  50 episodes"
echo "    Total: ~1,479 episodes + DQN training"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN] Would provision $GPU_NAME and run ~1,479 episodes."
    exit 0
fi

# ── Search for GPU offer ──────────────────────────────────────────
echo "==> Searching for $GPU_NAME offers..."

OFFER_ID=$(vastai search offers \
    "num_gpus=1 gpu_ram>=40 reliability>0.95 rentable=true gpu_name=$GPU_NAME cuda_max_good>=12.1 disk_space>=200 inet_down>=200" \
    -o 'dph_total' --limit 1 --raw \
    | python3 -c "import sys,json; data=json.load(sys.stdin); print(data[0]['id']) if data else sys.exit(1)" 2>/dev/null) || true

if [[ -z "$OFFER_ID" ]]; then
    echo "    No $GPU_NAME offers found. Trying A100 (any variant)..."
    OFFER_ID=$(vastai search offers \
        'num_gpus=1 gpu_ram>=40 reliability>0.95 rentable=true gpu_name=A100 cuda_max_good>=12.1 disk_space>=200 inet_down>=200' \
        -o 'dph_total' --limit 1 --raw \
        | python3 -c "import sys,json; data=json.load(sys.stdin); print(data[0]['id']) if data else sys.exit(1)" 2>/dev/null) || true
fi

if [[ -z "$OFFER_ID" ]]; then
    echo "    No A100 found. Trying any GPU >=40GB VRAM..."
    OFFER_ID=$(vastai search offers \
        'num_gpus=1 gpu_ram>=40 reliability>0.95 rentable=true cuda_max_good>=12.1 disk_space>=200 inet_down>=200' \
        -o 'dph_total' --limit 1 --raw \
        | python3 -c "import sys,json; data=json.load(sys.stdin); print(data[0]['id']) if data else sys.exit(1)" 2>/dev/null) || true
fi

if [[ -z "$OFFER_ID" ]]; then
    echo "ERROR: No suitable GPU offers found."
    exit 1
fi
echo "    Found offer: $OFFER_ID"

# ── Build onstart script ─────────────────────────────────────────
read -r -d '' ONSTART_SCRIPT << 'ONSTART_HEREDOC' || true
#!/bin/bash
set -ex
export TERM=dumb
export NO_COLOR=1
export PYTHONUNBUFFERED=1
apt-get update && apt-get install -y git openssh-client wget 2>/dev/null || true
pip uninstall -y colorama 2>/dev/null || true
__DEPLOY_KEY_BLOCK__
cd /workspace
if [[ -d STEMGym ]]; then cd STEMGym && git pull; else __CLONE_CMD__; cd STEMGym; fi
export CUDA_PATH=/usr/local/cuda-12.4
bash scripts/run_expansion_experiments.sh 2>&1 | tee /workspace/expansion_run.log
ONSTART_HEREDOC

if [[ -n "$DEPLOY_KEY_CONTENT" ]]; then
    DEPLOY_KEY_BLOCK=$(cat <<DKEOF
mkdir -p /root/.ssh
cat > /root/.ssh/id_ed25519 << 'KEYEOF'
$DEPLOY_KEY_CONTENT
KEYEOF
chmod 600 /root/.ssh/id_ed25519
ssh-keyscan github.com >> /root/.ssh/known_hosts 2>/dev/null
DKEOF
    )
    CLONE_CMD='git clone git@github.com:KurbanIntelligenceLab/STEMGym.git STEMGym'
else
    DEPLOY_KEY_BLOCK='echo "No deploy key — using HTTPS clone"'
    CLONE_CMD='git clone https://github.com/KurbanIntelligenceLab/STEMGym.git STEMGym'
fi

ONSTART_SCRIPT="${ONSTART_SCRIPT//__DEPLOY_KEY_BLOCK__/$DEPLOY_KEY_BLOCK}"
ONSTART_SCRIPT="${ONSTART_SCRIPT//__CLONE_CMD__/$CLONE_CMD}"

# ── Create instance ──────────────────────────────────────────────
echo ""
echo "==> Creating instance from offer $OFFER_ID..."

CREATE_OUTPUT=$(vastai create instance "$OFFER_ID" \
    --image "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel" \
    --ssh --direct \
    --disk 200 \
    --label "$LABEL" \
    --onstart-cmd "$ONSTART_SCRIPT" \
    --raw 2>&1)

echo "    Create output: $CREATE_OUTPUT"

INSTANCE_ID=$(echo "$CREATE_OUTPUT" | python3 -c "
import sys, json, re
text = sys.stdin.read()
try:
    data = json.loads(text)
    print(data.get('new_contract', data.get('id', '')))
except:
    m = re.search(r'(\d{7,})', text)
    if m: print(m.group(1))
    else: sys.exit(1)
" 2>/dev/null) || true

if [[ -z "$INSTANCE_ID" ]]; then
    INSTANCE_ID=$(echo "$CREATE_OUTPUT" | grep -oE '[0-9]{7,}' | head -1)
fi

if [[ -z "$INSTANCE_ID" ]]; then
    echo "ERROR: Could not extract instance ID"
    exit 1
fi

echo "    Instance ID: $INSTANCE_ID"
echo "    *** SAFETY: Only instance $INSTANCE_ID will be destroyed ***"

# Record instance
echo "" >> "$REPO_DIR/vast_ai_instances.txt"
echo "$(date '+%Y-%m-%d %H:%M') | Instance $INSTANCE_ID | $LABEL | $GPU_NAME" \
    >> "$REPO_DIR/vast_ai_instances.txt"

# ── Wait for startup ─────────────────────────────────────────────
echo ""
echo "==> Waiting for instance $INSTANCE_ID to start..."

MAX_WAIT_STARTUP=600
ELAPSED=0
while true; do
    STATUS=$(vastai show instances --raw 2>/dev/null \
        | python3 -c "
import sys, json
for i in json.load(sys.stdin):
    if i['id'] == $INSTANCE_ID:
        print(i.get('actual_status', 'unknown'))
        break
else:
    print('not_found')
" 2>/dev/null) || STATUS="unknown"

    echo "    [$ELAPSED s] Status: $STATUS"

    if [[ "$STATUS" == "running" ]]; then
        echo "    Instance is running!"
        break
    elif [[ "$STATUS" == "exited" || "$STATUS" == "error" ]]; then
        echo "ERROR: Instance entered '$STATUS'"
        exit 1
    fi

    if [[ $ELAPSED -ge $MAX_WAIT_STARTUP ]]; then
        echo "ERROR: Instance did not start within ${MAX_WAIT_STARTUP}s"
        exit 1
    fi

    sleep 30
    ELAPSED=$((ELAPSED + 30))
done

echo "    Waiting 90s for onstart to initialize..."
sleep 90

# ── Get SSH info ──────────────────────────────────────────────────
echo ""
echo "==> Getting SSH connection info..."

SSH_URL=$(vastai ssh-url "$INSTANCE_ID" 2>/dev/null || true)
SSH_HOST=$(echo "$SSH_URL" | sed -n 's|ssh://root@\([^:]*\):.*|\1|p')
SSH_PORT=$(echo "$SSH_URL" | sed -n 's|ssh://root@[^:]*:\([0-9]*\)|\1|p')

if [[ -z "$SSH_HOST" || -z "$SSH_PORT" ]]; then
    read -r SSH_HOST SSH_PORT < <(vastai show instances --raw 2>/dev/null \
        | python3 -c "
import sys, json
for i in json.load(sys.stdin):
    if i['id'] == $INSTANCE_ID:
        print(i.get('public_ipaddr', i.get('ssh_host', '')), i.get('direct_port_start', i.get('ssh_port', '')))
        break
" 2>/dev/null)
fi

echo "    SSH: root@$SSH_HOST:$SSH_PORT"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10"

# Update instance tracking with SSH info
echo "  SSH: ssh -p $SSH_PORT root@$SSH_HOST" >> "$REPO_DIR/vast_ai_instances.txt"

# ── Poll for completion ───────────────────────────────────────────
echo ""
echo "==> Polling for experiment completion..."
echo "    Monitor with: ssh $SSH_OPTS -p $SSH_PORT root@$SSH_HOST 'tail -50 /workspace/expansion_run.log'"
echo ""

MAX_WAIT=86400
ELAPSED=0
POLL_INTERVAL=300
while true; do
    DONE_CHECK=$(ssh $SSH_OPTS -p "$SSH_PORT" root@"$SSH_HOST" \
        "test -f /workspace/DONE && echo 'DONE' || echo 'NOT_DONE'" 2>/dev/null || echo "SSH_FAIL")

    PROGRESS=$(ssh $SSH_OPTS -p "$SSH_PORT" root@"$SSH_HOST" \
        "ls /workspace/STEMGym/results/*.json 2>/dev/null | wc -l; \
         tail -1 /workspace/expansion_run.log 2>/dev/null" 2>/dev/null || echo "?")

    HOURS=$((ELAPSED / 3600))
    MINS=$(((ELAPSED % 3600) / 60))
    echo "    [${HOURS}h${MINS}m] Status: $DONE_CHECK | $PROGRESS"

    if [[ "$DONE_CHECK" == "DONE" ]]; then
        echo ""
        echo "    *** ALL EXPERIMENTS COMPLETE! ***"
        break
    fi

    if [[ $ELAPSED -ge $MAX_WAIT ]]; then
        echo "ERROR: Experiments did not complete within 24 hours"
        echo "    Instance $INSTANCE_ID still running. SSH in to check."
        exit 1
    fi

    sleep "$POLL_INTERVAL"
    ELAPSED=$((ELAPSED + POLL_INTERVAL))
done

# ── Download results ──────────────────────────────────────────────
echo ""
echo "==> Downloading results..."
mkdir -p "$REPO_DIR/results"

RESULT_FILES=(
    "trajectory_capture.json"
    "declare_ablation.json"
    "zoom_dwell_ablation.json"
    "composite_task.json"
    "adaptive_transfer.json"
    "acq_func_comparison.json"
    "random_equipped_run.json"
    "single_bo_run.json"
    "defect_census_5seeds.json"
    "rl_baseline.json"
)

for rf in "${RESULT_FILES[@]}"; do
    echo "    Downloading $rf..."
    scp $SSH_OPTS -P "$SSH_PORT" \
        "root@$SSH_HOST:/workspace/STEMGym/results/$rf" \
        "$REPO_DIR/results/$rf" 2>/dev/null || echo "    WARNING: $rf not found"
done

# Download DQN checkpoint
echo "    Downloading dqn_agent.zip..."
mkdir -p "$REPO_DIR/checkpoints"
scp $SSH_OPTS -P "$SSH_PORT" \
    "root@$SSH_HOST:/workspace/STEMGym/checkpoints/dqn_agent.zip" \
    "$REPO_DIR/checkpoints/dqn_agent.zip" 2>/dev/null || echo "    WARNING: dqn_agent.zip not found"

# Download logs
mkdir -p "$REPO_DIR/results/logs"
scp $SSH_OPTS -P "$SSH_PORT" \
    "root@$SSH_HOST:/workspace/*.log" \
    "$REPO_DIR/results/logs/" 2>/dev/null || true

echo ""
echo "==> Downloaded results:"
ls -lh "$REPO_DIR/results/"*.json 2>/dev/null

# ── Verify ────────────────────────────────────────────────────────
echo ""
echo "==> Verifying results..."
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
for fname, exp_n in expected.items():
    path = os.path.join('$REPO_DIR/results', fname)
    if not os.path.exists(path):
        print(f'  MISSING  {fname}')
        all_ok = False
        continue
    with open(path) as f:
        data = json.load(f)
    n = len(data.get('results', []))
    ok = 'OK' if n >= exp_n else f'LOW ({n}/{exp_n})'
    if n < exp_n:
        all_ok = False
    print(f'  {ok:8s}  {fname:35s}  {n:4d} combos')

# DQN checkpoint
dqn = os.path.join('$REPO_DIR/checkpoints', 'dqn_agent.zip')
if os.path.exists(dqn):
    print(f'  OK        dqn_agent.zip')
else:
    print(f'  MISSING  dqn_agent.zip')
    all_ok = False

if all_ok:
    print('\n  ALL RESULTS VERIFIED')
else:
    print('\n  WARNING: Some results missing or incomplete')
"

# ── Destroy prompt ────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Results downloaded. Ready to destroy instance $INSTANCE_ID."
echo "=========================================="
echo ""
read -p "Destroy instance $INSTANCE_ID? (y/N) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    vastai destroy instance "$INSTANCE_ID"
    echo "    Instance $INSTANCE_ID destroyed."
else
    echo "    Instance $INSTANCE_ID left running."
    echo "    To destroy later: vastai destroy instance $INSTANCE_ID"
fi

echo ""
echo "=========================================="
echo "  EXPANSION EXPERIMENTS COMPLETE"
echo "=========================================="
echo ""
echo "Results in: $REPO_DIR/results/"
ls -lh "$REPO_DIR/results/"*.json 2>/dev/null
echo ""
echo "Next steps:"
echo "  python scripts/statistical_analysis.py"
echo "  python scripts/cadence_sensitivity.py"
echo "  python paper/generate_result_figures.py"
