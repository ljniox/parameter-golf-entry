"""
Size Budget Analysis for Parameter Golf
Estimates compressed artifact size for different model configurations.
Helps decide how to fill the 16MB budget optimally.
"""

import io
import zlib
import torch
import torch.nn as nn
import torch.nn.functional as F


def estimate_artifact_size(
    num_layers: int,
    model_dim: int,
    num_heads: int,
    num_kv_heads: int,
    mlp_mult: float,
    vocab_size: int = 1024,
    quant_bits: int = 8,  # 8 or 6
    bigram_vocab: int = 0,
    bigram_dim: int = 128,
    ve_enabled: bool = False,
    ve_dim: int = 128,
    ve_layers: int = 2,
    shared_layers: int = 0,  # 0 = no sharing, >0 = N unique blocks tiled
    tie_embeddings: bool = True,
):
    """Estimate compressed artifact size without building the full model."""
    head_dim = model_dim // num_heads
    kv_dim = num_kv_heads * head_dim
    mlp_hidden = int(mlp_mult * model_dim)

    # Count parameters per block
    # Attention: Q, K, V projections + output projection + q_gain + norms + scales
    attn_params = (
        model_dim * model_dim +      # Q proj
        model_dim * kv_dim +          # K proj
        model_dim * kv_dim +          # V proj
        model_dim * model_dim +       # O proj
        num_heads                     # q_gain
    )
    # MLP: fc + proj
    mlp_params = (
        model_dim * mlp_hidden +      # fc
        mlp_hidden * model_dim        # proj
    )
    # Per-block small params: attn_scale, mlp_scale, resid_mix
    block_small = model_dim + model_dim + 2 * model_dim  # attn_scale + mlp_scale + resid_mix(2,dim)

    block_total = attn_params + mlp_params + block_small

    # With weight sharing, only unique blocks count
    if shared_layers > 0:
        unique_blocks = shared_layers
    else:
        unique_blocks = num_layers

    total_block_params = block_total * unique_blocks

    # Embedding
    embed_params = vocab_size * model_dim

    # Skip weights
    num_skip = min(num_layers // 2, num_layers - num_layers // 2)
    skip_params = num_skip * model_dim

    # BigramHash
    bigram_params = 0
    if bigram_vocab > 0:
        bigram_params = bigram_vocab * bigram_dim  # embed
        if bigram_dim != model_dim:
            bigram_params += bigram_dim * model_dim  # proj
        bigram_params += 1  # scale

    # Value Embedding
    ve_params = 0
    if ve_enabled:
        ve_params = vocab_size * ve_dim  # embed
        if ve_dim != kv_dim:
            ve_params += ve_dim * kv_dim  # proj
        ve_params += 1 + ve_layers  # scale + per-layer scales

    # Head (tied = 0 extra)
    head_params = 0 if tie_embeddings else vocab_size * model_dim

    # SmearGate
    smear_params = model_dim  # gate

    total_params = total_block_params + embed_params + skip_params + bigram_params + ve_params + head_params + smear_params

    # Estimate quantized size
    # Large 2D tensors → int{quant_bits} + fp16 scale per row
    # Small tensors → fp16 passthrough
    # Compression ratio from real submissions: ~0.92-0.95 for int6, ~0.85-0.90 for int8

    large_2d_params = total_block_params + (embed_params if embed_params > 65536 else 0)
    small_params = total_params - large_2d_params

    if quant_bits == 6:
        # Int6: 6 bits per param + fp16 scale per row
        quant_bytes = (large_2d_params * 6) / 8
        # Embed stays int8
        if embed_params > 65536:
            quant_bytes = ((large_2d_params - embed_params) * 6) / 8 + embed_params  # embed at int8
        # Row scales: ~2 bytes per row (fp16)
        n_rows = unique_blocks * (4 + 2)  # Q,K,V,O + fc,proj per block
        if embed_params > 65536:
            n_rows += vocab_size
        scale_bytes = n_rows * 2
    else:
        quant_bytes = large_2d_params  # int8 = 1 byte per param
        n_rows = unique_blocks * 6
        if embed_params > 65536:
            n_rows += vocab_size
        scale_bytes = n_rows * 2

    small_bytes = small_params * 2  # fp16

    raw_bytes = quant_bytes + scale_bytes + small_bytes

    # zstd-22 compression ratio (empirical from submissions)
    if quant_bits == 6:
        compress_ratio = 0.93  # int6 doesn't compress as well
    else:
        compress_ratio = 0.30  # int8 + zlib compresses very well (seen 5MB from 17MB)

    compressed_bytes = raw_bytes * compress_ratio

    # Add estimated code size
    code_size = 70_000  # ~70KB for a competitive train_gpt.py

    total_artifact = compressed_bytes + code_size

    return {
        "total_params": total_params,
        "block_params": block_total,
        "unique_blocks": unique_blocks,
        "effective_layers": num_layers,
        "embed_params": embed_params,
        "bigram_params": bigram_params,
        "ve_params": ve_params,
        "raw_bytes": raw_bytes,
        "compressed_est": compressed_bytes,
        "code_size": code_size,
        "total_artifact": total_artifact,
        "headroom_mb": (16_000_000 - total_artifact) / 1e6,
        "fits": total_artifact <= 16_000_000,
    }


def print_config(name, **kwargs):
    r = estimate_artifact_size(**kwargs)
    status = "PASS" if r["fits"] else "FAIL"
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(f"  Layers: {r['effective_layers']} ({r['unique_blocks']} unique)")
    print(f"  Params: {r['total_params']:>12,}  ({r['total_params']/1e6:.1f}M)")
    print(f"    Block:    {r['block_params']:>10,}  x {r['unique_blocks']}")
    print(f"    Embed:    {r['embed_params']:>10,}")
    print(f"    Bigram:   {r['bigram_params']:>10,}")
    print(f"    VE:       {r['ve_params']:>10,}")
    print(f"  Raw bytes:        {r['raw_bytes']:>12,.0f}  ({r['raw_bytes']/1e6:.2f} MB)")
    print(f"  Compressed (est): {r['compressed_est']:>12,.0f}  ({r['compressed_est']/1e6:.2f} MB)")
    print(f"  Code:             {r['code_size']:>12,}  ({r['code_size']/1e6:.2f} MB)")
    print(f"  Total artifact:   {r['total_artifact']:>12,.0f}  ({r['total_artifact']/1e6:.2f} MB)")
    print(f"  Headroom:         {r['headroom_mb']:>+.2f} MB")
    print(f"  [{status}]")


def main():
    print("=" * 70)
    print("PARAMETER GOLF - SIZE BUDGET ANALYSIS")
    print("=" * 70)

    # ---- Baseline (from repo) ----
    print_config("Baseline (9L, 512d, int8, 2x MLP)",
        num_layers=9, model_dim=512, num_heads=8, num_kv_heads=4,
        mlp_mult=2, quant_bits=8)

    # ---- Current SOTA config ----
    print_config("SOTA #1 (11L, 512d, int6, 3x MLP, bigram, VE)",
        num_layers=11, model_dim=512, num_heads=8, num_kv_heads=4,
        mlp_mult=3, quant_bits=6, bigram_vocab=2048, bigram_dim=128,
        ve_enabled=True, ve_dim=128, ve_layers=2)

    # ---- Depth Recurrence: 16 effective layers from 4 unique blocks ----
    print_config("EXPERIMENT: 16L from 4 unique (depth recurrence), int6, 3x MLP",
        num_layers=16, model_dim=512, num_heads=8, num_kv_heads=4,
        mlp_mult=3, quant_bits=6, shared_layers=4,
        bigram_vocab=2048, bigram_dim=128, ve_enabled=True, ve_dim=128, ve_layers=2)

    # ---- Depth Recurrence: 14L from 7 unique ----
    print_config("EXPERIMENT: 14L from 7 unique, int6, 3x MLP",
        num_layers=14, model_dim=512, num_heads=8, num_kv_heads=4,
        mlp_mult=3, quant_bits=6, shared_layers=7,
        bigram_vocab=2048, bigram_dim=128, ve_enabled=True, ve_dim=128, ve_layers=2)

    # ---- Max capacity: 11L all unique, wider model (640d) ----
    print_config("EXPERIMENT: 11L, 640d (wider), int6, 3x MLP",
        num_layers=11, model_dim=640, num_heads=10, num_kv_heads=5,
        mlp_mult=3, quant_bits=6, bigram_vocab=2048, bigram_dim=128,
        ve_enabled=True, ve_dim=128, ve_layers=2)

    # ---- 13L all unique, same width ----
    print_config("EXPERIMENT: 13L (deeper), 512d, int6, 3x MLP",
        num_layers=13, model_dim=512, num_heads=8, num_kv_heads=4,
        mlp_mult=3, quant_bits=6, bigram_vocab=2048, bigram_dim=128,
        ve_enabled=True, ve_dim=128, ve_layers=2)

    # ---- MoE: 4 experts, 2 active ----
    print_config("EXPERIMENT: 11L, MoE 4 experts (2x MLP each), int6",
        num_layers=11, model_dim=512, num_heads=8, num_kv_heads=4,
        mlp_mult=8, quant_bits=6,  # 4 experts x 2x = 8x total MLP params
        bigram_vocab=2048, bigram_dim=128, ve_enabled=True, ve_dim=128, ve_layers=2)

    # ---- Depth Recurrence + wider ----
    print_config("EXPERIMENT: 22L from 11 unique (2x recurrence), 512d, int6",
        num_layers=22, model_dim=512, num_heads=8, num_kv_heads=4,
        mlp_mult=3, quant_bits=6, shared_layers=11,
        bigram_vocab=2048, bigram_dim=128, ve_enabled=True, ve_dim=128, ve_layers=2)


if __name__ == "__main__":
    main()
