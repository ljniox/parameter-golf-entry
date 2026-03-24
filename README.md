# Parameter Golf Entry

Our submission for the [OpenAI Parameter Golf Challenge](https://github.com/openai/parameter-golf).

## Key Innovation: Depth Recurrence

22 effective transformer layers from 11 unique blocks via weight tiling — same artifact size, more representational depth.

Built on top of the current #1 submission (1.1228 BPB) with:
- 2x depth recurrence with learnable per-pass scales
- All SOTA techniques: XSA, Partial RoPE, BigramHash, SmearGate, EMA, GPTQ-lite
- FlashAttention 3 with automatic fallback for non-Hopper GPUs

## Quick Start on RunPod

```bash
cd /workspace
git clone https://github.com/ljniox/parameter-golf-entry.git
cd parameter-golf-entry
bash runpod_quickstart.sh
bash run_dev.sh
```

## Scripts

| Script | Purpose | Cost |
|--------|---------|------|
| `run_dev.sh` | 2-min smoke test, 1 GPU | ~$0.01 |
| `run_experiment.sh` | Full 10-min standard training, 1 GPU | ~$0.06 |
| `run_recurrence.sh` | 10-min 2x depth recurrence, 1 GPU | ~$0.06 |
| `run_final.sh` | Official 8xH100 submission run | ~$3.50/seed |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RECURRENCE_FACTOR` | 1 | Set to 2 for 2x depth recurrence |
| `PASS_SCALE_INIT` | 1.0 | Initial value for per-pass learnable scales |
| `NUM_LAYERS` | 11 | Number of unique transformer blocks |
| `SEED` | 1337 | Random seed |
