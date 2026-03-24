#!/bin/bash
# =============================================================================
# DEV RUN: Fast smoke test on 1 GPU (~2 min)
# Use on cheap GPUs (RTX A5000, RTX 3090, RTX 4090) for rapid iteration.
# Cost: ~$0.01-0.02 per run
# =============================================================================
cd /workspace/parameter-golf

RUN_ID=dev_smoke_$(date +%s) \
ITERATIONS=500 \
MAX_WALLCLOCK_SECONDS=120 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=131072 \
NUM_LAYERS=11 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=3 \
TRAIN_SEQ_LEN=1024 \
EVAL_SEQ_LEN=1024 \
BIGRAM_VOCAB_SIZE=2048 \
XSA_LAST_N=4 \
ROPE_DIMS=16 \
LN_SCALE=1 \
VE_ENABLED=1 \
RECURRENCE_FACTOR=1 \
WARMDOWN_ITERS=100 \
MUON_WD=0.04 \
ADAM_WD=0.04 \
SEED=${SEED:-1337} \
torchrun --standalone --nproc_per_node=1 our_train_gpt.py

echo ""
echo "Dev run complete. Check logs/ for results."
