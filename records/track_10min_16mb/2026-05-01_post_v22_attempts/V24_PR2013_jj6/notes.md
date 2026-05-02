# V24: PR #2013 (Wilbatronic) jj6 Eval Trick Stack — Catastrophic Failure

**Status**: Failed — val_bpb 1.4284 + artifact 17.2 MB (over 16MB cap)

## Target

PR #2013 (Wilbatronic) reported:
- 1.0543 BPB (single seed=99 in `launch_8xh100_competition.sh`)
- Stack: 11L attention + LeakyReLU² + DEPTH_RECUR=4 + Residual Signs + Stochastic Eval (N=16) + TTT-LoRA rank 8 + Outlier Filter + INT4+lzma
- vocab_size=1024 (different from our sp8192 CaseOps base)
- Created 2026-04-30 20:51 UTC

## Environment

Same 8× H100 SXM5 80GB on Hyperbolic. **Required separate sp1024 dataset** (downloaded ~14 GB FineWeb sp1024 to pod before launch).

## Run Result

```
seed 42:
  step:964/5000   ← only 964 steps in 600s wallclock!
  val_loss:2.2403 val_bpb:1.3461  (mid-training)
  stopping_early: wallclock_cap train_time:600132ms step:964/5000
  Total submission size mixed+lzma: 17,199,979 bytes  ← OVER 16MB CAP
  final_mixed_lzma_roundtrip val_bpb:1.42838065  ← model didn't converge
```

## Root Cause

**DEPTH_RECUR=4** means each forward pass loops through the 11-layer block stack 4× (effective 44 layers of compute). On our env:
- step_avg ~623 ms
- Wallclock budget: 600s
- Max steps achievable: ~964 (with warmup overhead)
- PR #2013 reportedly needed 4000+ steps for convergence

PR #2013 must have either:
1. Faster compute environment than ours (different H100 SXM revision, or better cuDNN/Triton tuning)
2. Cherry-picked seed 99 from a longer training run (not 600s constrained)
3. Different definition of "wallclock cap" timing

## Lessons

### 1. Always pre-flight check `step_avg × max_steps × cap`

Before running:
- Estimate step_avg from architecture (DEPTH_RECUR=4 ≈ 4× normal)
- Compute max steps in 600s budget
- Compare to expected stop step from author's logs
- If our budget < author's reported steps, model won't converge

### 2. Mixed precision INT4 + lzma is fragile

Our artifact came in at 17.2 MB despite Wilbatronic's reportedly fitting in 16MB. The lzma compression ratio is content-dependent — different training trajectories produce different compressibility.

### 3. Cross-stack replication is high-risk in time-constrained scenarios

Switching from sp8192 CaseOps (V22 base) to sp1024 BPE required:
- Re-downloading 14 GB tokenized dataset
- Loading completely different vocabulary
- Different `train_gpt.py` (1500 lines, totally separate codebase)
- 5+ min setup overhead before first training step

**Lesson**: In a 60-minute window before deadline, never attempt a cross-vocab replication. The setup risk is too high.

## Files

- `run.sh` — exact launcher (matches PR #2013's `launch_8xh100_competition.sh` env vars, except seeds 42/0/1234 instead of 99 for 3-seed validity)
- `reference_train_gpt.py` — PR #2013's train_gpt.py (2459 lines)
