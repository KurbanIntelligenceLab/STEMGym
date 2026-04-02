#!/usr/bin/env bash
# =============================================================================
# vastai_run_experiments.sh
#
# End-to-end: provision a Vast.ai GPU instance, download worlds + checkpoints
# from HuggingFace, run all benchmark experiments, download results, destroy.
#
# Prerequisites:
#   - vastai CLI authenticated (`vastai set api-key <key>`)
#   - Deploy key at scripts/vastai_deploy_key (or use --no-deploy-key)
#   - OPENROUTER_API_KEY in .env (optional, for LLM planner; falls back to
#     rule-based if missing)
#
# Usage:
#   bash scripts/vastai_run_experiments.sh
#   bash scripts/vastai_run_experiments.sh --gpu L40S    # specific GPU
#   bash scripts/vastai_run_experiments.sh --dry-run     # print plan only
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOY_KEY="$SCRIPT_DIR/vastai_deploy_key"
LABEL="stemgym_experiments"
INSTANCE_ID=""
GPU_NAME="RTX_4090"   # Good balance of speed, availability, and cost
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu) GPU_NAME="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --label) LABEL="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Safety: trap to remind about cleanup on failure ──────────────────────────
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

# ── Preflight checks ────────────────────────────────────────────────────────
echo "==> Preflight checks..."

if ! command -v vastai &>/dev/null; then
    echo "ERROR: vastai CLI not found. Install with: pip install vastai"
    exit 1
fi

DEPLOY_KEY_CONTENT=""
if [[ -f "$DEPLOY_KEY" ]]; then
    DEPLOY_KEY_CONTENT=$(cat "$DEPLOY_KEY")
    echo "    Deploy key:  $DEPLOY_KEY"
else
    echo "    WARNING: No deploy key found. Will clone via HTTPS (repo must be public)."
fi

# Load optional env vars
OPENROUTER_KEY=""
if [[ -f "$REPO_DIR/.env" ]]; then
    OPENROUTER_KEY=$(grep '^OPENROUTER_API_KEY=' "$REPO_DIR/.env" 2>/dev/null | cut -d= -f2- || true)
fi
if [[ -n "$OPENROUTER_KEY" ]]; then
    echo "    OPENROUTER_API_KEY: ${OPENROUTER_KEY:0:8}... (LLM planner enabled)"
else
    echo "    OPENROUTER_API_KEY: not set (LLM planner will use rule-based fallback)"
fi

echo "    GPU target: $GPU_NAME"
echo "    Label: $LABEL"

# ── Plan ─────────────────────────────────────────────────────────────────────
echo ""
echo "==> Experiment plan:"
echo "    1. difficulty_scaling   (4 agents  x 3 worlds x 2 tasks x 5 seeds = 120 episodes)"
echo "    2. ablation_study       (5 agents  x 3 worlds x 3 tasks x 5 seeds = 225 episodes)"
echo "    3. main_targeted        (10 agents x 2 worlds x 1 task  x 5 seeds = 100 episodes)"
echo "    4. main_defect_census   (10 agents x 7 worlds x 1 task  x 5 seeds = 350 episodes)"
echo "    5. main_phase_mapping   (10 agents x 3 worlds x 1 task  x 5 seeds = 150 episodes)"
echo "    Total: 945 episodes"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN] Would provision $GPU_NAME instance and run 945 episodes."
    exit 0
fi

# ── Step 1: Search for GPU offer ─────────────────────────────────────────────
echo "==> Searching for $GPU_NAME offers..."

OFFER_ID=$(vastai search offers \
    "num_gpus=1 gpu_ram>=16 reliability>0.95 rentable=true gpu_name=$GPU_NAME cuda_max_good>=12.1 disk_space>=100 inet_down>=200" \
    -o 'dph_total' --limit 1 --raw \
    | python3 -c "import sys,json; data=json.load(sys.stdin); print(data[0]['id']) if data else sys.exit(1)" 2>/dev/null) || true

if [[ -z "$OFFER_ID" ]]; then
    echo "    No $GPU_NAME offers found. Trying any GPU with >=16GB VRAM..."
    OFFER_ID=$(vastai search offers \
        'num_gpus=1 gpu_ram>=16 reliability>0.95 rentable=true cuda_max_good>=12.1 disk_space>=100 inet_down>=200' \
        -o 'dph_total' --limit 1 --raw \
        | python3 -c "import sys,json; data=json.load(sys.stdin); print(data[0]['id']) if data else sys.exit(1)" 2>/dev/null) || true
fi

if [[ -z "$OFFER_ID" ]]; then
    echo "ERROR: No suitable GPU offers found."
    exit 1
fi

echo "    Found offer: $OFFER_ID"

# ── Step 2: Build onstart script ─────────────────────────────────────────────
# The onstart script runs inside the container when the instance starts.
# It: clones the repo, downloads data from HuggingFace, runs all experiments,
# and writes a DONE sentinel when finished.

read -r -d '' ONSTART_SCRIPT << 'ONSTART_HEREDOC' || true
#!/bin/bash
set -ex
export TERM=dumb
export NO_COLOR=1
export PYTHONUNBUFFERED=1
apt-get update && apt-get install -y git openssh-client 2>/dev/null || true
__DEPLOY_KEY_BLOCK__
cd /workspace
if [[ -d STEMGym ]]; then cd STEMGym && git pull; else __CLONE_CMD__; cd STEMGym; fi
__ENV_BLOCK__
bash scripts/run_experiments_on_instance.sh 2>&1 | tee /workspace/experiment_run.log
ONSTART_HEREDOC

# ── Inject deploy key and env vars into onstart script ───────────────────────
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
    CLONE_CMD='git clone git@github.com:anonymous/STEMGym.git STEMGym'
else
    DEPLOY_KEY_BLOCK='echo "No deploy key — using HTTPS clone"'
    CLONE_CMD='git clone https://github.com/anonymous/STEMGym.git STEMGym'
fi

ENV_BLOCK=""
if [[ -n "$OPENROUTER_KEY" ]]; then
    ENV_BLOCK="export OPENROUTER_API_KEY=\"$OPENROUTER_KEY\""
fi

ONSTART_SCRIPT="${ONSTART_SCRIPT//__DEPLOY_KEY_BLOCK__/$DEPLOY_KEY_BLOCK}"
ONSTART_SCRIPT="${ONSTART_SCRIPT//__CLONE_CMD__/$CLONE_CMD}"
ONSTART_SCRIPT="${ONSTART_SCRIPT//__ENV_BLOCK__/$ENV_BLOCK}"

# ── Step 3: Create instance ──────────────────────────────────────────────────
echo ""
echo "==> Creating instance from offer $OFFER_ID..."

CREATE_OUTPUT=$(vastai create instance "$OFFER_ID" \
    --image "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel" \
    --ssh --direct \
    --disk 100 \
    --label "$LABEL" \
    --onstart-cmd "$ONSTART_SCRIPT" \
    --raw 2>&1)

echo "    Create output: $CREATE_OUTPUT"

# Extract instance ID
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
echo ""
echo "    *** SAFETY: Only instance $INSTANCE_ID will be destroyed ***"

# ── Step 4: Wait for instance to start ────────────────────────────────────────
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

echo "    Waiting 90s for onstart script to initialize..."
sleep 90

# ── Step 5: Get SSH connection info ───────────────────────────────────────────
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

# ── Step 6: Poll for completion ───────────────────────────────────────────────
echo ""
echo "==> Polling for experiment completion..."
echo "    (This will take several hours. Check progress with:"
echo "     ssh $SSH_OPTS -p $SSH_PORT root@$SSH_HOST 'tail -20 /workspace/*.log')"
echo ""

MAX_WAIT=86400  # 24 hours
ELAPSED=0
POLL_INTERVAL=300  # 5 minutes
while true; do
    DONE_CHECK=$(ssh $SSH_OPTS -p "$SSH_PORT" root@"$SSH_HOST" \
        "test -f /workspace/DONE && echo 'DONE' || echo 'NOT_DONE'" 2>/dev/null || echo "SSH_FAIL")

    # Get last experiment progress
    PROGRESS=$(ssh $SSH_OPTS -p "$SSH_PORT" root@"$SSH_HOST" \
        "ls -lh /workspace/STEMGym/results/*.json 2>/dev/null | wc -l; \
         grep -c 'DEC-AUC' /workspace/*.log 2>/dev/null | tail -1" 2>/dev/null || echo "?")

    HOURS=$((ELAPSED / 3600))
    MINS=$(((ELAPSED % 3600) / 60))
    echo "    [${HOURS}h${MINS}m] Status: $DONE_CHECK  |  Progress: $PROGRESS"

    if [[ "$DONE_CHECK" == "DONE" ]]; then
        echo ""
        echo "    *** ALL EXPERIMENTS COMPLETE! ***"
        break
    fi

    if [[ $ELAPSED -ge $MAX_WAIT ]]; then
        echo "ERROR: Experiments did not complete within 24 hours"
        echo "    Instance $INSTANCE_ID is still running. Check logs manually."
        echo "    ssh $SSH_OPTS -p $SSH_PORT root@$SSH_HOST"
        exit 1
    fi

    sleep "$POLL_INTERVAL"
    ELAPSED=$((ELAPSED + POLL_INTERVAL))
done

# ── Step 7: Download results ────────────────────────────────────────────────
echo ""
echo "==> Downloading results..."
mkdir -p "$REPO_DIR/results"

# Download result JSON files
RESULT_FILES=(
    "difficulty_scaling.json"
    "ablation_study.json"
    "main_targeted.json"
    "main_defect_census.json"
    "main_phase_mapping.json"
)

for rf in "${RESULT_FILES[@]}"; do
    echo "    Downloading $rf..."
    scp $SSH_OPTS -P "$SSH_PORT" \
        "root@$SSH_HOST:/workspace/STEMGym/results/$rf" \
        "$REPO_DIR/results/$rf" 2>/dev/null || echo "    WARNING: $rf not found"
done

# Also download experiment logs
mkdir -p "$REPO_DIR/results/logs"
scp $SSH_OPTS -P "$SSH_PORT" \
    "root@$SSH_HOST:/workspace/*.log" \
    "$REPO_DIR/results/logs/" 2>/dev/null || true

echo ""
echo "==> Downloaded results:"
ls -lh "$REPO_DIR/results/"*.json 2>/dev/null

# ── Step 8: Verify results ───────────────────────────────────────────────────
echo ""
echo "==> Verifying results..."
python3 -c "
import json, os

results_dir = '$REPO_DIR/results'
for rf in os.listdir(results_dir):
    if not rf.endswith('.json'):
        continue
    path = os.path.join(results_dir, rf)
    try:
        with open(path) as f:
            data = json.load(f)
        n = len(data.get('results', []))
        print(f'  {rf:30s}  {n:4d} result entries  OK')
    except Exception as e:
        print(f'  {rf:30s}  ERROR: {e}')
"

# ── Step 9: Confirm and destroy ──────────────────────────────────────────────
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

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  EXPERIMENT RUN COMPLETE"
echo "=========================================="
echo ""
echo "Results in: $REPO_DIR/results/"
ls -lh "$REPO_DIR/results/"*.json 2>/dev/null
echo ""
echo "Next steps:"
echo "  1. Run the analysis notebook:  jupyter notebook notebooks/analyze_results.ipynb"
echo "  2. Compile the paper:          cd paper && pdflatex main.tex"
