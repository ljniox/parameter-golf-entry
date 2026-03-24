"""
CPU Smoke Test for Parameter Golf
- Builds the baseline GPT model on CPU
- Runs a forward pass with dummy data
- Tests int8 quantization + zlib compression
- Verifies artifact fits under 16MB
- No CUDA or data files required
"""

import io
import math
import os
import sys
import time
import zlib

# Force CPU-only before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---- Import model components from train_gpt.py ----
# We inline the minimal architecture here to avoid CUDA checks in main()

def rms_norm(x, normalized_shape, eps=None):
    """Manual RMS norm for PyTorch < 2.4 compatibility."""
    dims = tuple(range(-len(normalized_shape), 0))
    rms = x.to(torch.float32).pow(2).mean(dim=dims, keepdim=True).add(eps or 1e-6).rsqrt()
    return (x * rms).to(x.dtype)

class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        return rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        self._zero_init = False

    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype), self.bias)


class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hdim = dim * mlp_mult
        self.fc1 = CastedLinear(dim, hdim)
        self.fc2 = CastedLinear(hdim, dim)
        self.fc2._zero_init = True

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)).square())


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base=10000.0, qk_gain_init=1.5):
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
        self.q_gain = nn.Parameter(torch.full((num_heads, 1, head_dim), qk_gain_init))

    def forward(self, x, x0):
        B, T, C = x.shape
        # Attention
        h = self.attn_norm(x)
        q = self.q_proj(h).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # GQA: expand kv heads
        if self.num_kv_heads < self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        q = q * self.q_gain.unsqueeze(0)

        # Scaled dot-product attention (causal)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(B, T, -1)
        x = x + self.o_proj(attn)

        # MLP
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size=1024, num_layers=9, model_dim=512, num_heads=8,
                 num_kv_heads=4, mlp_mult=2, tie_embeddings=True,
                 tied_embed_init_std=0.005, logit_softcap=30.0, rope_base=10000.0,
                 qk_gain_init=1.5):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim))
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.005)
        for m in self.modules():
            if isinstance(m, nn.Linear) and getattr(m, "_zero_init", False):
                nn.init.zeros_(m.weight)

    def forward(self, input_ids, target_ids):
        x = self.tok_emb(input_ids)
        x = rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []

        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


# ---- Quantization (from train_gpt.py) ----

def quantize_int8(state_dict):
    quantized, scales, dtypes, passthrough = {}, {}, {}, {}
    for name, t in state_dict.items():
        t = t.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= 65536:
            passthrough[name] = t.half() if t.is_floating_point() and t.numel() > 1 else t
            continue
        t32 = t.float()
        if t32.ndim == 2:
            clip_abs = torch.quantile(t32.abs(), 0.9999984, dim=1)
            scale = (clip_abs / 127.0).clamp_min(1/127)
            q = torch.clamp(torch.round(t32 / scale[:, None]), -127, 127).to(torch.int8)
            scales[name] = scale.half()
        else:
            clip_abs = float(torch.quantile(t32.abs().flatten(), 0.9999984).item())
            scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
            q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8)
            scales[name] = scale
        quantized[name] = q.contiguous()
        dtypes[name] = str(t.dtype).removeprefix("torch.")
    return {"quantized": quantized, "scales": scales, "dtypes": dtypes, "passthrough": passthrough}


def compress_artifact(obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    raw = buf.getvalue()
    compressed = zlib.compress(raw, 9)
    return raw, compressed


# ---- Main smoke test ----

def main():
    print("=" * 60)
    print("PARAMETER GOLF - CPU SMOKE TEST")
    print("=" * 60)

    # 1. Build model
    print("\n[1/5] Building baseline GPT model (9L, 512d, vocab=1024)...")
    t0 = time.time()
    model = GPT(
        vocab_size=1024, num_layers=9, model_dim=512,
        num_heads=8, num_kv_heads=4, mlp_mult=2,
        tie_embeddings=True
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"       Parameters: {num_params:,}")
    print(f"       Built in {time.time()-t0:.2f}s")

    # 2. Forward pass
    print("\n[2/5] Running forward pass (batch=2, seq=64)...")
    t0 = time.time()
    dummy_x = torch.randint(0, 1024, (2, 64))
    dummy_y = torch.randint(0, 1024, (2, 64))
    with torch.no_grad():
        loss = model(dummy_x, dummy_y)
    print(f"       Loss: {loss.item():.4f} (expected ~6.93 = ln(1024) for random init)")
    print(f"       Forward pass in {time.time()-t0:.2f}s")

    # 3. Backward pass
    print("\n[3/5] Running backward pass...")
    t0 = time.time()
    model.train()
    loss = model(dummy_x, dummy_y)
    loss.backward()
    grad_norms = {n: p.grad.norm().item() for n, p in model.named_parameters() if p.grad is not None}
    print(f"       Gradients computed for {len(grad_norms)} parameters")
    print(f"       Backward pass in {time.time()-t0:.2f}s")

    # 4. Quantization
    print("\n[4/5] Quantizing to int8...")
    t0 = time.time()
    sd = model.state_dict()
    qobj = quantize_int8(sd)
    n_quantized = len(qobj["quantized"])
    n_passthrough = len(qobj["passthrough"])
    print(f"       Quantized: {n_quantized} tensors, Passthrough: {n_passthrough} tensors")
    print(f"       Quantized in {time.time()-t0:.2f}s")

    # 5. Compression & size check
    print("\n[5/5] Compressing artifact...")
    t0 = time.time()
    raw_bytes, compressed_bytes = compress_artifact(qobj)

    # Add code size estimate
    code_path = os.path.join(os.path.dirname(__file__), "train_gpt.py")
    code_size = os.path.getsize(code_path) if os.path.exists(code_path) else 50_000
    total_size = len(compressed_bytes) + code_size

    print(f"       Raw size:        {len(raw_bytes):>12,} bytes ({len(raw_bytes)/1e6:.2f} MB)")
    print(f"       Compressed:      {len(compressed_bytes):>12,} bytes ({len(compressed_bytes)/1e6:.2f} MB)")
    print(f"       Code size:       {code_size:>12,} bytes ({code_size/1e6:.2f} MB)")
    print(f"       Total artifact:  {total_size:>12,} bytes ({total_size/1e6:.2f} MB)")
    print(f"       Compressed in {time.time()-t0:.2f}s")

    limit = 16_000_000
    if total_size <= limit:
        print(f"\n  PASS - Artifact fits under 16MB ({total_size/1e6:.2f} MB / 16.00 MB)")
    else:
        over = total_size - limit
        print(f"\n  FAIL - Artifact exceeds 16MB by {over:,} bytes ({total_size/1e6:.2f} MB)")

    # Summary
    print("\n" + "=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)
    print(f"  Model:       9L x 512d, {num_params:,} params")
    print(f"  Forward:     OK (loss={loss.item():.4f})")
    print(f"  Backward:    OK ({len(grad_norms)} grads)")
    print(f"  Quantize:    OK ({n_quantized} tensors)")
    print(f"  Artifact:    {total_size/1e6:.2f} MB / 16.00 MB limit")
    print(f"  Headroom:    {(limit - total_size)/1e6:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
