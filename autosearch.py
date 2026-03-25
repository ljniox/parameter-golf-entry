"""
AUTOSEARCH: Autonomous experiment loop for Parameter Golf
Inspired by Karpathy's autoresearch — runs 5-min experiments in a loop,
keeps improvements, discards failures. No LLM needed.

Usage:
    python autosearch.py [--hours N] [--budget-minutes M]

Each experiment:
1. Picks a configuration mutation from the search space
2. Runs training for BUDGET minutes
3. Evaluates val_bpb
4. Keeps if improved, discards if not
5. Logs to results.tsv
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

# ============================================================================
# SEARCH SPACE
# ============================================================================
# Each experiment mutates ONE thing from the current best config.
# This keeps the search focused and allows clean attribution.

BASELINE_CONFIG = {
    "NUM_LAYERS": 11,
    "MODEL_DIM": 512,
    "NUM_HEADS": 8,
    "NUM_KV_HEADS": 4,
    "MLP_MULT": 3,
    "TRAIN_SEQ_LEN": 2048,
    "EVAL_SEQ_LEN": 2048,
    "EVAL_STRIDE": 64,
    "BIGRAM_VOCAB_SIZE": 2048,
    "XSA_LAST_N": 4,
    "ROPE_DIMS": 16,
    "LN_SCALE": 1,
    "VE_ENABLED": 1,
    "VE_DIM": 128,
    "VE_LAYERS": "9,10",
    "SWA_ENABLED": 1,
    "SWA_EVERY": 50,
    "RECURRENCE_FACTOR": 1,
    "PASS_SCALE_INIT": 1.0,
    "WARMDOWN_ITERS": 3500,
    "MATRIX_LR": 0.025,
    "SCALAR_LR": 0.025,
    "TIED_EMBED_LR": 0.035,
    "MUON_MOMENTUM": 0.99,
    "MUON_MOMENTUM_WARMUP_START": 0.92,
    "MUON_MOMENTUM_WARMUP_STEPS": 1500,
    "MUON_WD": 0.04,
    "ADAM_WD": 0.04,
    "GRAD_CLIP_NORM": 0.3,
    "LATE_QAT_THRESHOLD": 0.15,
    "QK_GAIN_INIT": 1.5,
    "LOGIT_SOFTCAP": 30.0,
    "TRAIN_BATCH_TOKENS": 131072,  # Smaller batch for A40 speed
    "TRAIN_SEQ_LEN": 1024,  # Shorter seq for A40 speed
    "EVAL_SEQ_LEN": 1024,
    "SEED": 1337,
}

# Mutations: each is (name, param, values_to_try)
MUTATIONS = [
    # --- Depth recurrence ---
    ("recurrence_2x", {"RECURRENCE_FACTOR": 2, "PASS_SCALE_INIT": 1.0}),
    ("recurrence_2x_damped", {"RECURRENCE_FACTOR": 2, "PASS_SCALE_INIT": 0.7}),
    ("recurrence_2x_low_scale", {"RECURRENCE_FACTOR": 2, "PASS_SCALE_INIT": 0.5}),

    # --- Learning rates ---
    ("matrix_lr_0.03", {"MATRIX_LR": 0.03}),
    ("matrix_lr_0.02", {"MATRIX_LR": 0.02}),
    ("matrix_lr_0.035", {"MATRIX_LR": 0.035}),
    ("tied_embed_lr_0.04", {"TIED_EMBED_LR": 0.04}),
    ("tied_embed_lr_0.03", {"TIED_EMBED_LR": 0.03}),
    ("scalar_lr_0.03", {"SCALAR_LR": 0.03}),
    ("scalar_lr_0.02", {"SCALAR_LR": 0.02}),

    # --- Architecture ---
    ("xsa_last_5", {"XSA_LAST_N": 5}),
    ("xsa_last_6", {"XSA_LAST_N": 6}),
    ("xsa_last_3", {"XSA_LAST_N": 3}),
    ("rope_dims_32", {"ROPE_DIMS": 32}),
    ("rope_dims_8", {"ROPE_DIMS": 8}),
    ("rope_dims_48", {"ROPE_DIMS": 48}),
    ("bigram_4096", {"BIGRAM_VOCAB_SIZE": 4096}),
    ("bigram_1024", {"BIGRAM_VOCAB_SIZE": 1024}),
    ("ve_dim_64", {"VE_DIM": 64}),
    ("ve_dim_256", {"VE_DIM": 256}),
    ("ve_3layers", {"VE_LAYERS": "8,9,10"}),
    ("ve_4layers", {"VE_LAYERS": "7,8,9,10"}),

    # --- Training dynamics ---
    ("warmdown_4000", {"WARMDOWN_ITERS": 4000}),
    ("warmdown_3000", {"WARMDOWN_ITERS": 3000}),
    ("warmdown_5000", {"WARMDOWN_ITERS": 5000}),
    ("muon_wd_0.05", {"MUON_WD": 0.05}),
    ("muon_wd_0.03", {"MUON_WD": 0.03}),
    ("muon_wd_0.06", {"MUON_WD": 0.06}),
    ("grad_clip_0.5", {"GRAD_CLIP_NORM": 0.5}),
    ("grad_clip_0.2", {"GRAD_CLIP_NORM": 0.2}),
    ("grad_clip_1.0", {"GRAD_CLIP_NORM": 1.0}),
    ("swa_every_30", {"SWA_EVERY": 30}),
    ("swa_every_100", {"SWA_EVERY": 100}),
    ("late_qat_0.1", {"LATE_QAT_THRESHOLD": 0.1}),
    ("late_qat_0.2", {"LATE_QAT_THRESHOLD": 0.2}),
    ("late_qat_0.25", {"LATE_QAT_THRESHOLD": 0.25}),

    # --- Muon optimizer ---
    ("muon_momentum_0.95", {"MUON_MOMENTUM": 0.95}),
    ("muon_momentum_0.98", {"MUON_MOMENTUM": 0.98}),
    ("muon_warmup_1000", {"MUON_MOMENTUM_WARMUP_STEPS": 1000}),
    ("muon_warmup_2000", {"MUON_MOMENTUM_WARMUP_STEPS": 2000}),

    # --- Batch size ---
    ("batch_786k", {"TRAIN_BATCH_TOKENS": 786432}),
    ("batch_262k", {"TRAIN_BATCH_TOKENS": 262144}),

    # --- Logit softcap ---
    ("softcap_20", {"LOGIT_SOFTCAP": 20.0}),
    ("softcap_50", {"LOGIT_SOFTCAP": 50.0}),

    # --- QK gain ---
    ("qk_gain_1.0", {"QK_GAIN_INIT": 1.0}),
    ("qk_gain_2.0", {"QK_GAIN_INIT": 2.0}),

    # --- Combo experiments (multiple changes) ---
    ("combo_recurrence_higher_lr", {"RECURRENCE_FACTOR": 2, "MATRIX_LR": 0.03, "PASS_SCALE_INIT": 0.8}),
    ("combo_deeper_xsa_ve", {"XSA_LAST_N": 6, "VE_LAYERS": "7,8,9,10", "VE_DIM": 64}),
    ("combo_aggressive_wd", {"MUON_WD": 0.06, "ADAM_WD": 0.06, "WARMDOWN_ITERS": 4000}),
]


def load_results(tsv_path):
    """Load previous results from TSV."""
    results = []
    if os.path.exists(tsv_path):
        with open(tsv_path) as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 5:
                    results.append({
                        "name": parts[0],
                        "val_bpb": float(parts[1]) if parts[1] != "crash" else 999.0,
                        "steps": int(parts[2]),
                        "status": parts[3],
                        "description": parts[4],
                    })
    return results


def save_result(tsv_path, name, val_bpb, steps, status, description):
    """Append a result to TSV."""
    header_needed = not os.path.exists(tsv_path)
    with open(tsv_path, "a") as f:
        if header_needed:
            f.write("name\tval_bpb\tsteps\tstatus\tdescription\n")
        bpb_str = f"{val_bpb:.6f}" if val_bpb < 100 else "crash"
        f.write(f"{name}\t{bpb_str}\t{steps}\t{status}\t{description}\n")


def run_experiment(config, budget_seconds, run_id):
    """Run a single training experiment and return (val_bpb, steps) or (None, 0) on crash."""
    env = os.environ.copy()
    env["RUN_ID"] = run_id
    env["ITERATIONS"] = "20000"
    env["MAX_WALLCLOCK_SECONDS"] = str(budget_seconds)
    env["VAL_LOSS_EVERY"] = "0"  # only validate at the end
    env["VAL_BATCH_SIZE"] = "131072"  # smaller val batch for speed
    env["TRAIN_LOG_EVERY"] = "100"
    env["EVAL_STRIDE"] = "0"  # disable slow sliding window eval

    # Data paths
    env["DATA_PATH"] = os.environ.get("DATA_PATH", "/workspace/parameter-golf/data/datasets/fineweb10B_sp1024")
    env["TOKENIZER_PATH"] = os.environ.get("TOKENIZER_PATH", "/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model")

    # Apply config
    for k, v in config.items():
        env[k] = str(v)

    cmd = ["torchrun", "--standalone", "--nproc_per_node=1", "train_gpt.py"]

    print(f"\n{'='*60}")
    print(f"  RUNNING: {run_id}")
    print(f"  Config changes: {json.dumps({k: v for k, v in config.items() if k not in BASELINE_CONFIG or BASELINE_CONFIG.get(k) != v}, indent=2)}")
    print(f"  Budget: {budget_seconds}s")
    print(f"{'='*60}\n")

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            timeout=budget_seconds + 600  # extra 10 min for warmup + eval on slow GPUs
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {time.time()-t0:.0f}s")
        return None, 0
    except Exception as e:
        print(f"  CRASH: {e}")
        return None, 0

    elapsed = time.time() - t0
    print(f"  Finished in {elapsed:.0f}s")

    # Parse val_bpb from output
    val_bpb = None
    steps = 0
    for line in output.split("\n"):
        # Look for the final validation line
        if "val_bpb:" in line and "step:" in line:
            try:
                bpb_part = line.split("val_bpb:")[1].split()[0]
                val_bpb = float(bpb_part)
                step_part = line.split("step:")[1].split("/")[0]
                steps = int(step_part)
            except (ValueError, IndexError):
                pass
        # Also check for final int6 roundtrip
        if "final_int6_roundtrip val_loss:" in line and "val_bpb:" in line:
            try:
                bpb_part = line.split("val_bpb:")[1].split()[0]
                val_bpb = float(bpb_part)
            except (ValueError, IndexError):
                pass
        if "stopping_early" in line or "step:" in line:
            try:
                step_part = line.split("step:")[1].split("/")[0]
                steps = max(steps, int(step_part))
            except (ValueError, IndexError):
                pass

    if val_bpb is not None:
        print(f"  val_bpb: {val_bpb:.4f} | steps: {steps}")
    else:
        print(f"  CRASH — could not parse val_bpb")
        # Print last 20 lines for debugging
        lines = output.strip().split("\n")
        for l in lines[-20:]:
            print(f"    {l}")

    return val_bpb, steps


def main():
    parser = argparse.ArgumentParser(description="Autosearch: autonomous experiment loop")
    parser.add_argument("--hours", type=float, default=8, help="Total hours to run")
    parser.add_argument("--budget-minutes", type=float, default=3, help="Minutes per experiment")
    parser.add_argument("--results", type=str, default="results.tsv", help="Results file")
    parser.add_argument("--randomize", action="store_true", help="Randomize experiment order")
    args = parser.parse_args()

    budget_seconds = int(args.budget_minutes * 60)
    total_seconds = int(args.hours * 3600)
    tsv_path = args.results

    print("=" * 60)
    print("  AUTOSEARCH - Parameter Golf Experiment Loop")
    print("=" * 60)
    print(f"  Budget per experiment: {args.budget_minutes} min")
    print(f"  Total runtime: {args.hours} hours")
    print(f"  Max experiments: ~{int(total_seconds / (budget_seconds + 120))}")
    print(f"  Results: {tsv_path}")
    print("=" * 60)

    # Load previous results to skip already-run experiments
    prev_results = load_results(tsv_path)
    done_names = {r["name"] for r in prev_results}
    best_bpb = min((r["val_bpb"] for r in prev_results if r["status"] == "keep"), default=999.0)

    # First run: establish baseline if not done
    if "baseline" not in done_names:
        print("\n>>> Running baseline first...")
        bpb, steps = run_experiment(BASELINE_CONFIG, budget_seconds, "baseline")
        if bpb is not None:
            best_bpb = bpb
            save_result(tsv_path, "baseline", bpb, steps, "keep", "Standard SOTA config")
            done_names.add("baseline")
            print(f"\n  BASELINE: val_bpb={bpb:.4f} ({steps} steps)")
        else:
            save_result(tsv_path, "baseline", 999.0, 0, "crash", "Baseline crashed")
            print("\n  BASELINE CRASHED — check setup")
            return

    # Build experiment queue
    queue = [(name, mut) for name, mut in MUTATIONS if name not in done_names]
    if args.randomize:
        random.shuffle(queue)

    print(f"\n  Baseline BPB: {best_bpb:.4f}")
    print(f"  Experiments remaining: {len(queue)}")

    start_time = time.time()
    exp_count = 0

    for name, mutation in queue:
        elapsed = time.time() - start_time
        if elapsed >= total_seconds:
            print(f"\n  Time limit reached ({args.hours}h). Stopping.")
            break

        remaining = total_seconds - elapsed
        if remaining < budget_seconds + 60:
            print(f"\n  Not enough time for another experiment. Stopping.")
            break

        # Build config
        config = dict(BASELINE_CONFIG)
        config.update(mutation)
        config["SEED"] = 1337  # Keep seed fixed for fair comparison

        run_id = f"auto_{name}_{int(time.time())}"
        bpb, steps = run_experiment(config, budget_seconds, run_id)
        exp_count += 1

        if bpb is None:
            save_result(tsv_path, name, 999.0, 0, "crash", json.dumps(mutation))
            print(f"  [{exp_count}] {name}: CRASH")
            continue

        delta = bpb - best_bpb
        if bpb < best_bpb:
            status = "keep"
            best_bpb = bpb
            marker = "*** NEW BEST ***"
        else:
            status = "discard"
            marker = ""

        save_result(tsv_path, name, bpb, steps, status, json.dumps(mutation))

        print(f"  [{exp_count}] {name}: val_bpb={bpb:.4f} (delta={delta:+.4f}) [{status}] {marker}")

        # Summary so far
        all_results = load_results(tsv_path)
        kept = [r for r in all_results if r["status"] == "keep"]
        print(f"  Progress: {exp_count} experiments, {len(kept)} kept, best={best_bpb:.4f}")
        print(f"  Time: {elapsed/3600:.1f}h / {args.hours}h")

    # Final summary
    print("\n" + "=" * 60)
    print("  AUTOSEARCH COMPLETE")
    print("=" * 60)
    all_results = load_results(tsv_path)
    kept = sorted([r for r in all_results if r["status"] == "keep"], key=lambda r: r["val_bpb"])
    print(f"  Total experiments: {len(all_results)}")
    print(f"  Kept: {len(kept)}")
    print(f"  Best: {kept[0]['name']} = {kept[0]['val_bpb']:.4f}" if kept else "  No successful runs")
    print("\n  Top 5:")
    for i, r in enumerate(kept[:5]):
        print(f"    {i+1}. {r['name']}: {r['val_bpb']:.4f} ({r['steps']} steps)")
    print("=" * 60)


if __name__ == "__main__":
    main()
