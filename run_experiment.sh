#!/bin/bash
# =============================================================================
# EXPERIMENT RUN: Full 10-min training on 1 GPU
# Use on RTX 4090 or 1xH100 for meaningful BPB numbers.
# Cost: ~$0.06 (4090) or ~$0.25 (H100) per run
# Note: 1 GPU will train fewer steps than 8xH100. Use for relative comparisons.
# =============================================================================
cd /workspace/parameter-golf

NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "Running on ${NUM_GPUS} GPU(s)"

RUN_ID=exp_1gpu_$(date +%s) \
ITERATIONS=20000 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_BATCH_TOKENS=524288 \
VAL_LOSS_EVERY=2000 \
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
EMA_ENABLED=1 \
SWA_ENABLED=1 \
SWA_EVERY=50 \
RECURRENCE_FACTOR=1 \
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
SEED=${SEED:-1337} \
torchrun --standalone --nproc_per_node=1 our_train_gpt.py

echo ""
echo "Experiment complete. Check logs/ for results."
