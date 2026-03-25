#!/bin/bash
# =============================================================================
# AUTOSEARCH: Run overnight autonomous experiment loop
# Cost: ~$0.017 per 5-min experiment on A40 spot ($0.20/hr)
# 8 hours = ~80 experiments = ~$1.60
# =============================================================================
cd /workspace/parameter-golf-entry

# Make sure data is available
if [ ! -f "/workspace/parameter-golf/data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin" ]; then
    echo "Data not found. Run runpod_quickstart.sh first."
    exit 1
fi

HOURS=${HOURS:-8}
BUDGET=${BUDGET_MINUTES:-5}

echo "============================================"
echo "  AUTOSEARCH"
echo "  ${HOURS}h runtime, ${BUDGET}min/experiment"
echo "  Estimated experiments: ~$(echo "$HOURS * 60 / ($BUDGET + 2)" | bc)"
echo "  Estimated cost: ~\$$(echo "scale=2; $HOURS * 0.20" | bc)"
echo "============================================"

python autosearch.py \
    --hours ${HOURS} \
    --budget-minutes ${BUDGET} \
    --results results.tsv \
    --randomize \
    2>&1 | tee autosearch.log

echo ""
echo "Results saved to results.tsv"
echo "Log saved to autosearch.log"
