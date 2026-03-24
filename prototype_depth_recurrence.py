"""
Prototype: Depth Recurrence for Parameter Golf
Tests weight sharing across transformer layers to increase effective depth
without increasing parameter count or artifact size.

Key idea: Instead of 11 unique layer blocks, use N unique blocks tiled K times
to create N*K effective layers. Each pass through the same weights sees a
different residual stream state, learning increasingly refined representations.

This is similar to:
- Universal Transformers (Dehghani et al., 2019)
- ALBERT (Lan et al., 2020) — cross-layer parameter sharing
- Depth recurrence in recent efficient architectures
"""

import io
import math
import os
import time
import zlib

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---- Compat shim ----
def rms_norm(x, normalized_shape, eps=None):
    dims = tuple(range(-len(normalized_shape), 0))
    rms = x.to(torch.float32).pow(2).mean(dim=dims, keepdim=True).add(eps or 1e-6).rsqrt()
    return (x * rms).to(x.dtype)


class RMSNorm(nn.Module):
    def forward(self, x):
        return rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        self._zero_init = False
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype), self.bias)


class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hdim = int(dim * mlp_mult)
        self.fc = CastedLinear(dim, hdim)
        self.proj = CastedLinear(hdim, dim)
        self.proj._zero_init = True
    def forward(self, x):
        return self.proj(F.relu(self.fc(x)).square())


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, layer_idx=0, ln_scale=False):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.mlp = MLP(dim, mlp_mult)
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_proj = CastedLinear(dim, num_heads * head_dim)
        self.k_proj = CastedLinear(dim, num_kv_heads * head_dim)
        self.v_proj = CastedLinear(dim, num_kv_heads * head_dim)
        self.o_proj = CastedLinear(num_heads * head_dim, dim)
        self.o_proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads, 1, head_dim), 1.5))
        self.attn_scale = nn.Parameter(torch.ones(dim))
        self.mlp_scale = nn.Parameter(torch.ones(dim))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))))
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0

    def forward(self, x, x0):
        B, T, C = x.shape
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        # Attention
        h = self.attn_norm(x_in) * self.ln_scale_factor
        q = self.q_proj(h).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.num_kv_heads < self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        q = q * self.q_gain.unsqueeze(0).to(dtype=q.dtype)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(B, T, -1)

        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * self.o_proj(attn)
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor)
        return x_out


class GPT_Standard(nn.Module):
    """Standard GPT with unique layers (baseline for comparison)."""
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads, mlp_mult):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, layer_idx=i, ln_scale=True)
            for i in range(num_layers)
        ])
        self.num_enc = num_layers // 2
        self.num_dec = num_layers - self.num_enc
        self.num_skip = min(self.num_enc, self.num_dec)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip, model_dim))
        self.final_norm = RMSNorm()
        self.logit_softcap = 30.0
        nn.init.normal_(self.tok_emb.weight, std=0.005)

    def forward(self, input_ids, target_ids):
        x = self.tok_emb(input_ids)
        x = rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []
        for i in range(self.num_enc):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_dec):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_enc + i](x, x0)
        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        logits = F.linear(x, self.tok_emb.weight)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


class GPT_DepthRecurrence(nn.Module):
    """GPT with weight sharing / depth recurrence.

    `num_unique` blocks are tiled to create `effective_layers` total passes.
    Each recurrence pass uses the same weights but sees different residual state.

    Strategies:
    - "tile": Repeat blocks [0,1,...,N-1, 0,1,...,N-1, ...]
    - "cycle": Like tile but with learnable per-pass scale factors
    """
    def __init__(self, vocab_size, num_unique, effective_layers, model_dim,
                 num_heads, num_kv_heads, mlp_mult, strategy="cycle"):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.effective_layers = effective_layers
        self.num_unique = num_unique
        self.strategy = strategy

        # Only allocate unique blocks
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, layer_idx=i, ln_scale=True)
            for i in range(num_unique)
        ])

        # Per-pass scale factors (learnable, allows each recurrence to adapt)
        if strategy == "cycle":
            self.pass_scales = nn.Parameter(torch.ones(effective_layers, dtype=torch.float32))
        else:
            self.pass_scales = None

        # U-Net skip connections (based on effective depth)
        self.num_enc = effective_layers // 2
        self.num_dec = effective_layers - self.num_enc
        self.num_skip = min(self.num_enc, self.num_dec)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip, model_dim))

        self.final_norm = RMSNorm()
        self.logit_softcap = 30.0
        nn.init.normal_(self.tok_emb.weight, std=0.005)

    def _block_idx(self, layer_i: int) -> int:
        """Map effective layer index to unique block index."""
        return layer_i % self.num_unique

    def forward(self, input_ids, target_ids):
        x = self.tok_emb(input_ids)
        x = rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []

        for i in range(self.num_enc):
            block = self.blocks[self._block_idx(i)]
            out = block(x, x0)
            if self.pass_scales is not None:
                scale = self.pass_scales[i].to(dtype=out.dtype)
                out = x + scale * (out - x)  # Learnable interpolation
            x = out
            skips.append(x)

        for i in range(self.num_dec):
            eff_i = self.num_enc + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            block = self.blocks[self._block_idx(eff_i)]
            out = block(x, x0)
            if self.pass_scales is not None:
                scale = self.pass_scales[eff_i].to(dtype=out.dtype)
                out = x + scale * (out - x)
            x = out

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        logits = F.linear(x, self.tok_emb.weight)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def estimate_compressed_size(model, quant_bits=6):
    """Quick estimate of compressed artifact size."""
    total = 0
    for name, p in model.named_parameters():
        if p.ndim == 2 and p.numel() > 65536:
            total += (p.numel() * quant_bits) / 8 + p.shape[0] * 2  # quant + scales
        else:
            total += p.numel() * 2  # fp16
    # zstd compression ratio
    ratio = 0.93 if quant_bits == 6 else 0.30
    return total * ratio


def main():
    print("=" * 70)
    print("DEPTH RECURRENCE PROTOTYPE")
    print("=" * 70)

    configs = [
        ("Standard 11L (SOTA baseline)", "standard",
         dict(vocab_size=1024, num_layers=11, model_dim=512, num_heads=8, num_kv_heads=4, mlp_mult=3)),

        ("Recurrence: 22L from 11 unique (2x tile)", "recurrence",
         dict(vocab_size=1024, num_unique=11, effective_layers=22, model_dim=512,
              num_heads=8, num_kv_heads=4, mlp_mult=3, strategy="cycle")),

        ("Recurrence: 16L from 8 unique (2x tile)", "recurrence",
         dict(vocab_size=1024, num_unique=8, effective_layers=16, model_dim=512,
              num_heads=8, num_kv_heads=4, mlp_mult=3, strategy="cycle")),

        ("Recurrence: 33L from 11 unique (3x tile)", "recurrence",
         dict(vocab_size=1024, num_unique=11, effective_layers=33, model_dim=512,
              num_heads=8, num_kv_heads=4, mlp_mult=3, strategy="cycle")),

        ("Recurrence: 15L from 5 unique (3x tile)", "recurrence",
         dict(vocab_size=1024, num_unique=5, effective_layers=15, model_dim=512,
              num_heads=8, num_kv_heads=4, mlp_mult=3, strategy="cycle")),

        ("Standard 13L (more params, deeper)", "standard",
         dict(vocab_size=1024, num_layers=13, model_dim=512, num_heads=8, num_kv_heads=4, mlp_mult=3)),
    ]

    dummy_x = torch.randint(0, 1024, (2, 64))
    dummy_y = torch.randint(0, 1024, (2, 64))

    for name, kind, kwargs in configs:
        print(f"\n{'='*70}")
        print(f"  {name}")
        print(f"{'='*70}")

        if kind == "standard":
            model = GPT_Standard(**kwargs)
        else:
            model = GPT_DepthRecurrence(**kwargs)

        params = count_params(model)
        est_size = estimate_compressed_size(model, quant_bits=6)
        total_est = est_size + 70_000  # code size

        t0 = time.time()
        with torch.no_grad():
            loss = model(dummy_x, dummy_y)
        fwd_time = time.time() - t0

        t0 = time.time()
        model.train()
        loss2 = model(dummy_x, dummy_y)
        loss2.backward()
        bwd_time = time.time() - t0

        grad_count = sum(1 for p in model.parameters() if p.grad is not None)

        if kind == "recurrence":
            eff = kwargs["effective_layers"]
            uniq = kwargs["num_unique"]
            print(f"  Effective layers: {eff} (from {uniq} unique, {eff/uniq:.0f}x recurrence)")
        else:
            print(f"  Layers: {kwargs['num_layers']}")

        print(f"  Parameters: {params:,} ({params/1e6:.1f}M)")
        print(f"  Est artifact: {total_est/1e6:.2f} MB (headroom: {(16e6-total_est)/1e6:+.2f} MB)")
        fits = "PASS" if total_est <= 16e6 else "FAIL"
        print(f"  Size check: [{fits}]")
        print(f"  Forward:  {fwd_time:.2f}s  loss={loss.item():.4f}")
        print(f"  Backward: {bwd_time:.2f}s  grads={grad_count}")

        # Key insight: same params, different forward cost
        if kind == "recurrence":
            std_11_params = count_params(GPT_Standard(1024, 11, 512, 8, 4, 3))
            savings = (1 - params / std_11_params) * 100
            if savings > 0:
                print(f"  Param savings vs 11L standard: {savings:.1f}% fewer params")
            else:
                print(f"  Same param count as 11L standard (weight sharing = free depth)")

    print(f"\n{'='*70}")
    print("KEY INSIGHT")
    print("="*70)
    print("""
  Depth recurrence gives us MORE effective layers at the SAME artifact size.
  The 22L-from-11 config has identical param count to standard 11L but
  processes the residual stream through 22 transformations.

  Trade-off: ~2x forward/backward compute time (more passes through blocks).
  Within the 10-min H100 budget, this means ~half the training steps,
  BUT each step is more expressive. The question is whether the extra
  depth per step outweighs fewer total steps.

  The "cycle" strategy with learnable per-pass scales lets the model
  control how much each recurrence pass contributes, avoiding the
  diminishing returns problem of naive weight tying.
    """)


if __name__ == "__main__":
    main()
