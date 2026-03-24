#!/bin/bash
# =============================================================================
# FINAL SUBMISSION RUN: 8xH100 SXM, 10 min
# This is the official evaluation config for leaderboard submission.
# Run 3 seeds for reproducibility: SEED=1337, SEED=42, SEED=2024
# Cost: ~$3.50 per run ($10.50 for 3 seeds)
# =============================================================================
cd /workspace/parameter-golf

# Make sure we have full dataset
if [ ! -f "./data/datasets/fineweb10B_sp1024/fineweb_train_000079.bin" ]; then
    echo "Downloading full dataset (80 shards)..."
    python3 data/cached_challenge_fineweb.py --variant sp1024
fi

SEED=${SEED:-1337}
echo "Running FINAL submission with SEED=${SEED} on 8xH100..."

# Set RECURRENCE_FACTOR based on best experiment results
# Change this after your experiments determine the best config!
RECURRENCE=${RECURRENCE_FACTOR:-1}

RUN_ID=final_seed${SEED}_r${RECURRENCE}_$(date +%s) \
ITERATIONS=9000 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_BATCH_TOKENS=786432 \
VAL_LOSS_EVERY=4000 \
NUM_LAYERS=11 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=3 \
TRAIN_SEQ_LEN=2048 \
EVAL_SEQ_LEN=2048 \
EVAL_STRIDE=64 \
BIGRAM_VOCAB_SIZE=2048 \
XSA_LAST_N=4 \
ROPE_DIMS=16 \
LN_SCALE=1 \
VE_ENABLED=1 \
VE_DIM=128 \
VE_LAYERS=9,10 \
SWA_ENABLED=1 \
SWA_EVERY=50 \
RECURRENCE_FACTOR=${RECURRENCE} \
PASS_SCALE_INIT=1.0 \
WARMDOWN_ITERS=3500 \
MATRIX_LR=0.025 \
SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
MUON_WD=0.04 \
ADAM_WD=0.04 \
GRAD_CLIP_NORM=0.3 \
LATE_QAT_THRESHOLD=0.15 \
SEED=${SEED} \
torchrun --standalone --nproc_per_node=8 our_train_gpt.py

echo ""
echo "====================================="
echo "  FINAL RUN COMPLETE (seed=${SEED})"
echo "====================================="
echo ""
echo "Check logs/ for val_bpb score."
echo "Check final_model.int6.ptz for artifact."
echo ""
echo "To run all 3 seeds:"
echo "  SEED=1337 bash run_final.sh"
echo "  SEED=42   bash run_final.sh"
echo "  SEED=2024 bash run_final.sh"
