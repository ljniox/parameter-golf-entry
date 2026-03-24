#!/bin/bash
# =============================================================================
# PARAMETER GOLF - RUNPOD QUICKSTART
# =============================================================================
# Deploy a pod using the official template first:
#   https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th
# Then SSH in and run this script.
# =============================================================================

set -e

echo "============================================"
echo "  PARAMETER GOLF - SETUP"
echo "============================================"

# ---- 1. Clone repo ----
cd /workspace
if [ ! -d "parameter-golf" ]; then
    echo "[1/4] Cloning repository..."
    git clone https://github.com/openai/parameter-golf.git
else
    echo "[1/4] Repo already exists, pulling latest..."
    cd parameter-golf && git pull && cd ..
fi
cd parameter-golf

# ---- 2. Download dataset ----
# Use --train-shards 1 for dev (fast, ~100M tokens)
# Use --train-shards 10 for full (8B tokens, needed for final eval)
SHARDS=${TRAIN_SHARDS:-1}
echo "[2/4] Downloading dataset (${SHARDS} shards)..."
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards ${SHARDS}

# ---- 3. Copy our custom training script ----
echo "[3/4] Setting up our training script..."
# If our_train_gpt.py exists in /workspace, copy it in
if [ -f "/workspace/our_train_gpt.py" ]; then
    cp /workspace/our_train_gpt.py ./our_train_gpt.py
    echo "       Copied our_train_gpt.py from /workspace"
fi

# ---- 4. Quick sanity check ----
echo "[4/4] Sanity check..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPUs: {torch.cuda.device_count()}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

echo ""
echo "============================================"
echo "  SETUP COMPLETE"
echo "============================================"
echo ""
echo "  Quick commands:"
echo ""
echo "  # DEV: Fast smoke test (1 GPU, 2 min):"
echo "  bash run_dev.sh"
echo ""
echo "  # EXPERIMENT: Single GPU, 10 min:"
echo "  bash run_experiment.sh"
echo ""
echo "  # RECURRENCE: 2x depth, single GPU, 10 min:"
echo "  bash run_recurrence.sh"
echo ""
echo "  # FINAL: 8xH100, 10 min (for leaderboard submission):"
echo "  bash run_final.sh"
echo ""
echo "============================================"
