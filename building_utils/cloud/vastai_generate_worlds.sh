#!/usr/bin/env bash
# =============================================================================
# vastai_generate_worlds.sh
#
# Provisions a Vast.ai A100 SXM4 (80 GB) instance, runs the 3 clustered-defect
# world generators, downloads the HDF5 files, and destroys ONLY the new
# instance.  The 8 existing training instances are never touched.
#
# Prerequisites:
#   - vastai CLI authenticated (`~/.config/vastai/vast_api_key`)
#   - Deploy key generated at scripts/vastai_deploy_key
#   - .env file with MP_API_KEY
#
# Usage:
#   bash scripts/vastai_generate_worlds.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOY_KEY="$SCRIPT_DIR/vastai_deploy_key"
LABEL="stemgym_worldgen"
INSTANCE_ID=""  # will be set after creation

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
if [[ ! -f "$DEPLOY_KEY" ]]; then
    echo "ERROR: Deploy key not found at $DEPLOY_KEY"
    echo "Run: ssh-keygen -t ed25519 -f $DEPLOY_KEY -N '' -C 'stemgym-vastai-deploy'"
    exit 1
fi

if [[ ! -f "$REPO_DIR/.env" ]]; then
    echo "ERROR: .env file not found at $REPO_DIR/.env"
    exit 1
fi

# Load MP_API_KEY from .env
MP_API_KEY=$(grep '^MP_API_KEY=' "$REPO_DIR/.env" | cut -d= -f2-)
if [[ -z "$MP_API_KEY" ]]; then
    echo "ERROR: MP_API_KEY not found in .env"
    exit 1
fi

DEPLOY_KEY_CONTENT=$(cat "$DEPLOY_KEY")

echo "==> Preflight OK"
echo "    Deploy key:  $DEPLOY_KEY"
echo "    MP_API_KEY:  ${MP_API_KEY:0:8}..."

# ── Step 1: Search for an A100 SXM4 80GB offer ──────────────────────────────
echo ""
echo "==> Searching for A100 SXM4 80GB offers..."

OFFER_ID=$(vastai search offers \
    'num_gpus=1 gpu_ram>=70 reliability>0.95 rentable=true gpu_name=A100_SXM4' \
    -o 'dph_total' --limit 1 --raw \
    | python3 -c "import sys,json; data=json.load(sys.stdin); print(data[0]['id']) if data else sys.exit(1)")

if [[ -z "$OFFER_ID" ]]; then
    echo "ERROR: No A100 SXM4 offers found. Trying A100 PCIE as fallback..."
    OFFER_ID=$(vastai search offers \
        'num_gpus=1 gpu_ram>=70 reliability>0.95 rentable=true gpu_name=A100_PCIE' \
        -o 'dph_total' --limit 1 --raw \
        | python3 -c "import sys,json; data=json.load(sys.stdin); print(data[0]['id']) if data else sys.exit(1)")
fi

if [[ -z "$OFFER_ID" ]]; then
    echo "ERROR: No suitable GPU offers found."
    exit 1
fi

echo "    Found offer: $OFFER_ID"

# ── Step 2: Create instance ─────────────────────────────────────────────────
echo ""
echo "==> Creating instance from offer $OFFER_ID..."

# Build the onstart script that will run inside the container
ONSTART_SCRIPT=$(cat <<'ONSTART_EOF'
#!/bin/bash
set -ex

# --- Install prerequisites ---
apt-get update && apt-get install -y git openssh-client

# --- Set up deploy key for GitHub ---
mkdir -p /root/.ssh
cat > /root/.ssh/id_ed25519 << 'KEYEOF'
__DEPLOY_KEY_PLACEHOLDER__
KEYEOF
chmod 600 /root/.ssh/id_ed25519
ssh-keyscan github.com >> /root/.ssh/known_hosts 2>/dev/null

# --- Clone repository ---
git clone git@github.com:KurbanIntelligenceLab/STEMGym.git /workspace/STEMGym

# --- Install dependencies ---
pip install cupy-cuda12x
pip install -e "/workspace/STEMGym[sim]"

# --- Set MP_API_KEY ---
export MP_API_KEY="__MP_API_KEY_PLACEHOLDER__"

# --- Run generators ---
cd /workspace/STEMGym

echo "=== [1/3] Generating SrTiO3 clustered worlds ==="
python data/scripts/generate_srtio3_clustered.py --device gpu --all-difficulties 2>&1 | tee /workspace/gen_srtio3.log

echo "=== [2/3] Generating BaTiO3 interface worlds ==="
python data/scripts/generate_batio3_interface.py --device gpu --all-difficulties 2>&1 | tee /workspace/gen_batio3.log

echo "=== [3/3] Generating SiGe gradient clustered worlds ==="
python data/scripts/generate_sige_gradient_clustered.py --device gpu --all-difficulties 2>&1 | tee /workspace/gen_sige.log

echo "=== ALL GENERATORS COMPLETE ==="
touch /workspace/DONE
ONSTART_EOF
)

# Substitute the actual deploy key and MP_API_KEY into the onstart script
ONSTART_SCRIPT="${ONSTART_SCRIPT//__DEPLOY_KEY_PLACEHOLDER__/$DEPLOY_KEY_CONTENT}"
ONSTART_SCRIPT="${ONSTART_SCRIPT//__MP_API_KEY_PLACEHOLDER__/$MP_API_KEY}"

CREATE_OUTPUT=$(vastai create instance "$OFFER_ID" \
    --image "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel" \
    --ssh --direct \
    --disk 80 \
    --label "$LABEL" \
    --onstart-cmd "$ONSTART_SCRIPT" \
    --raw 2>&1)

echo "    Create output: $CREATE_OUTPUT"

# Extract instance ID from create output
INSTANCE_ID=$(echo "$CREATE_OUTPUT" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('new_contract', data.get('id', '')))
except:
    # Try parsing from text output like 'Started. 12345678'
    import re
    text = sys.stdin.read() if not data else json.dumps(data)
    m = re.search(r'(\d{7,})', text)
    if m: print(m.group(1))
    else: sys.exit(1)
")

if [[ -z "$INSTANCE_ID" ]]; then
    # Fallback: try to extract from raw text
    INSTANCE_ID=$(echo "$CREATE_OUTPUT" | grep -oE '[0-9]{7,}' | head -1)
fi

if [[ -z "$INSTANCE_ID" ]]; then
    echo "ERROR: Could not extract instance ID from create output"
    echo "Output was: $CREATE_OUTPUT"
    exit 1
fi

echo "    Instance ID: $INSTANCE_ID"
echo "    Label: $LABEL"
echo ""
echo "    *** SAFETY: Only this instance ID ($INSTANCE_ID) will be destroyed ***"

# ── Step 3: Wait for instance to be running ──────────────────────────────────
echo ""
echo "==> Waiting for instance $INSTANCE_ID to start running..."

MAX_WAIT_STARTUP=600  # 10 minutes
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
")
    echo "    [$ELAPSED s] Instance status: $STATUS"

    if [[ "$STATUS" == "running" ]]; then
        echo "    Instance is running!"
        break
    elif [[ "$STATUS" == "exited" || "$STATUS" == "error" ]]; then
        echo "ERROR: Instance entered status '$STATUS'"
        exit 1
    fi

    if [[ $ELAPSED -ge $MAX_WAIT_STARTUP ]]; then
        echo "ERROR: Instance did not start within ${MAX_WAIT_STARTUP}s"
        exit 1
    fi

    sleep 30
    ELAPSED=$((ELAPSED + 30))
done

# Give the onstart script time to begin
echo "    Waiting 60s for onstart script to initialize..."
sleep 60

# ── Step 4: Poll for completion ──────────────────────────────────────────────
echo ""
echo "==> Polling for /workspace/DONE sentinel file..."

# Get SSH connection info
SSH_URL=$(vastai ssh-url "$INSTANCE_ID" 2>/dev/null || true)
echo "    SSH URL: $SSH_URL"

# Parse SSH host and port from the ssh-url output (format: ssh://root@host:port)
SSH_HOST=$(echo "$SSH_URL" | sed -n 's|ssh://root@\([^:]*\):.*|\1|p')
SSH_PORT=$(echo "$SSH_URL" | sed -n 's|ssh://root@[^:]*:\([0-9]*\)|\1|p')

if [[ -z "$SSH_HOST" || -z "$SSH_PORT" ]]; then
    # Fallback: get from show instances
    read SSH_HOST SSH_PORT < <(vastai show instances --raw 2>/dev/null \
        | python3 -c "
import sys, json
for i in json.load(sys.stdin):
    if i['id'] == $INSTANCE_ID:
        print(i.get('public_ipaddr', i.get('ssh_host', '')), i.get('direct_port_start', i.get('ssh_port', '')))
        break
")
fi

echo "    SSH target: root@$SSH_HOST:$SSH_PORT"

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10"

MAX_WAIT_GEN=14400  # 4 hours
ELAPSED=0
POLL_INTERVAL=120   # check every 2 minutes
while true; do
    # Try to check for DONE sentinel
    DONE_CHECK=$(ssh $SSH_OPTS -p "$SSH_PORT" root@"$SSH_HOST" \
        "test -f /workspace/DONE && echo 'DONE' || echo 'NOT_DONE'" 2>/dev/null || echo "SSH_FAIL")

    echo "    [$ELAPSED s] Status: $DONE_CHECK"

    if [[ "$DONE_CHECK" == "DONE" ]]; then
        echo ""
        echo "    *** GENERATION COMPLETE! ***"
        break
    fi

    if [[ $ELAPSED -ge $MAX_WAIT_GEN ]]; then
        echo "ERROR: Generation did not complete within ${MAX_WAIT_GEN}s (4 hours)"
        echo "Check logs: ssh $SSH_OPTS -p $SSH_PORT root@$SSH_HOST 'cat /workspace/gen_*.log'"
        exit 1
    fi

    sleep "$POLL_INTERVAL"
    ELAPSED=$((ELAPSED + POLL_INTERVAL))
done

# ── Step 5: Download HDF5 files ─────────────────────────────────────────────
echo ""
echo "==> Downloading generated HDF5 files..."

mkdir -p "$REPO_DIR/data/simulated"

# Download all HDF5 files from the simulated directory
vastai copy "${INSTANCE_ID}:/workspace/STEMGym/data/simulated/" "local:$REPO_DIR/data/simulated/"

echo ""
echo "==> Downloaded files:"
ls -lh "$REPO_DIR/data/simulated/"*clustered* "$REPO_DIR/data/simulated/"*interface* 2>/dev/null || echo "    (checking for files...)"

# Verify we got the expected files
EXPECTED_FILES=(
    "srtio3_clustered_easy.h5"
    "srtio3_clustered_medium.h5"
    "srtio3_clustered_hard.h5"
    "batio3_interface_easy.h5"
    "batio3_interface_medium.h5"
    "batio3_interface_hard.h5"
    "sige_gradient_clustered_easy.h5"
    "sige_gradient_clustered_medium.h5"
    "sige_gradient_clustered_hard.h5"
)

MISSING=0
for f in "${EXPECTED_FILES[@]}"; do
    if [[ ! -f "$REPO_DIR/data/simulated/$f" ]]; then
        echo "    MISSING: $f"
        MISSING=$((MISSING + 1))
    fi
done

if [[ $MISSING -gt 0 ]]; then
    echo "WARNING: $MISSING expected files are missing!"
    echo "Check logs on instance before destroying."
    read -p "Destroy instance anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Instance $INSTANCE_ID left running. Clean up manually."
        exit 1
    fi
fi

# ── Step 6: Destroy ONLY the new instance ────────────────────────────────────
echo ""
echo "==> Destroying instance $INSTANCE_ID (label: $LABEL)..."
echo "    (Only this instance — the 8 training instances are untouched)"

vastai destroy instance "$INSTANCE_ID"

echo "    Instance $INSTANCE_ID destroyed."

# ── Step 7: Verify existing instances are untouched ──────────────────────────
echo ""
echo "==> Verifying existing instances..."

PROTECTED_LABELS=("schnet" "dimenetpp" "gotennet" "equiformer_v2" "nequip" "painn" "torchmdnet" "clifford")

vastai show instances --raw 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
protected = set($( printf "'%s'," "${PROTECTED_LABELS[@]}" | sed 's/,$//' | sed 's/^/[/;s/$/]/' ))
found = {i['label']: i['actual_status'] for i in data if i.get('label') in protected}
print(f'  Found {len(found)}/{len(protected)} protected instances:')
for label, status in sorted(found.items()):
    print(f'    {label}: {status}')
if len(found) < len(protected):
    missing = protected - set(found.keys())
    print(f'  WARNING: Missing instances: {missing}')
"

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  WORLD GENERATION COMPLETE"
echo "=========================================="
echo ""
echo "Generated files in data/simulated/:"
ls -lh "$REPO_DIR/data/simulated/"*clustered* "$REPO_DIR/data/simulated/"*interface* 2>/dev/null
echo ""
echo "Next steps:"
echo "  1. Verify files with: python -c \"import h5py; f=h5py.File('data/simulated/srtio3_clustered_medium.h5'); print(list(f.keys()))\""
echo "  2. Upload to HuggingFace:"
for f in "${EXPECTED_FILES[@]}"; do
    NAME="${f%.h5}"
    echo "     huggingface-cli upload johnpolat/STEMGym data/simulated/$f worlds/$f --repo-type=dataset"
done
