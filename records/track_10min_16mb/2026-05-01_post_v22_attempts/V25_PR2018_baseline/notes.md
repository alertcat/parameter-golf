# V25: PR #2018 (simon-marcus) Baseline Replication — Killed at Pre-quant Stage

**Status**: Killed at pre-quant 1.06316 (env mismatch +0.014 BPB vs simon-marcus's 1.04930)

**Postscript**: cocohearts's official audit (PR #2146, 2026-05-02) banned PR #2018 entirely for "train/validation document overlap" — even successful replication would have inherited the ban.

## Target

PR #2018 (simon-marcus) reported:
- 3-seed mean **1.04617 BPB** (later corrected to 1.04722 after partial C1 fix)
- Stack: V21 base + Gated XSA + LQER top-1 + In-timer N-gram tilt + Cheaper Phased TTT
- Created 2026-04-30 21:24 UTC (2h35m before deadline)

## Environment

Same 8× H100 SXM5 80GB on Hyperbolic.

## Run Result

```
seed 42:
  stopping_early: wallclock_cap train_time:596024ms step:4913/20000
  diagnostic pre-quantization post-ema val_bpb: 1.06316272  ← +0.014 BPB worse than PR #2018 (1.04930)
  diagnostic quantized val_bpb: 1.07094742                  ← +0.013 BPB worse than PR #2018 (1.05774)
  
  [killed before TTT phase to free GPU for V24 attempt]
```

## Root Cause Analysis

**Environment mismatch produced systematically worse trained model.**

Our pre-quant val_bpb was +0.014 BPB worse than PR #2018's reported number. Since we used the EXACT same `train_gpt.py` and env vars, the difference must come from:

1. **PyTorch/CUDA stack**: Our PyTorch 2.9.1+cu128 with CUDA 13 forward-compat driver 580 vs simon-marcus's exact build
2. **Triton compile**: Custom kernels (fused softcapped CE) compile to slightly different SASS
3. **bfloat16 numerical determinism**: NCCL all_reduce with different topology produces slightly different gradients
4. **TF32/cuBLAS**: Default ON in PyTorch can produce ~1e-4 numerical differences accumulating over 4900 steps

Confirmed by V26 (same stack with 2 extra env vars): pre-quant 1.06305 — almost identical regression.

## Insight: Why V25 Baseline Wouldn't Have Helped Anyway

On 2026-05-01 16:44 UTC (~22h after V25 was killed), @sharpobject commented on PR #2018:

> "the first 40k documents of your training data are the same as the last 40k documents of your validation data"

This is a **C1 causality violation** — training on validation tokens. simon-marcus did not respond to this finding and pivoted to PR #2140 (different stack, 1.056).

cocohearts's official audit (PR #2146, 2026-05-02 18:08 UTC) confirmed:

> "PR #2018: invalid due train/validation document overlap in the submitted CaseOps data construction"

**Implication**: If V25 replication had succeeded at simon-marcus's claimed 1.046, we would have inherited the same train/val overlap (we used the same dataset construction script) and our submission would have been banned alongside PR #2018.

Our env mismatch was actually protective.

## Lessons

### 1. Cross-environment compliance reproduction needs identical infra

The 0.014 BPB pre-quant gap is reproducible (V25 baseline + V26 both showed it) and large enough to invalidate any attempt to claim simon-marcus's reported number on our infra.

### 2. Score-first compliance audit must precede replication attempts

If I had checked the dataset construction script first (or grep'd for train/val overlap), I would have caught the issue before launching any replication. cocohearts's auditors did this in 1.5 hours; I should have done it in 5 minutes.

### 3. "Pivot" PRs near deadline are high-risk

simon-marcus's PR #2140 pivot (created 24 minutes before audited cutoff) was ALSO banned by cocohearts: "active within/word n-gram gates, same C1 target-token issue; #2140's final scored state was also post-cutoff". Lesson: when the SOTA holder pivots in panic, follow-on attempts inherit the underlying compliance debt.

## Files

- `run.sh` — exact replication launcher (PR #2018 env vars, seeds 42/1337/2026 matching simon-marcus)
- `reference_train_gpt.py` — PR #2018's train_gpt.py (4385 lines)
