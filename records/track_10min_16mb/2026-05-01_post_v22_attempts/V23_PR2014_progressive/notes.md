# V23: PR #2014 (simonbissonnette) Progressive Context Growth + Short-Doc TTT

**Status**: Killed mid-run (seed 42 only, ~70% wallclock done) when better target appeared.

## Target

PR #2014 (simonbissonnette) reported:
- 3-seed mean **1.05759 BPB** (std 0.00034)
- Stack: PR #1855/#1953 base + Progressive context growth (1024 → 2048 → 3072) + Short-doc score-first TTT
- Created 2026-04-30 21:00 UTC

## Environment

- 8× H100 SXM5 80GB (Hyperbolic eu-north-4)
- PyTorch 2.9.1+cu128 (with CUDA 13 forward-compat driver 580)
- sp8192 CaseOps lossless caps tokenizer (from PR #1729)

## Run

```bash
# Started: 2026-04-30T21:48:19Z UTC
# Killed: 2026-04-30T22:30 UTC (when PR #2018 simon-marcus appeared at 1.04617)
# Progress: seed 42 at step ~3500/5000 (70% wallclock), pre-quant val_bpb tracking ~1.05x
```

## Why killed

simon-marcus's PR #2018 (1.04617 BPB, 0.011 BPB lower than PR #2014's 1.05759) appeared 24 minutes after V23 launch. Decision: kill V23, switch to V25 (PR #2018 replication) for higher target.

## Reflection

In hindsight, V23 was the safest bet — it was a real replication of a clean, no-controversy stack at 1.05759 BPB. Even if V25 had succeeded, V25/V26 inherited the eventual PR #2018 train/val overlap ban risk (sharpobject flagged 5-1 16:44).

If I had let V23 finish all 3 seeds (~57 min), I might have submitted a clean 1.05759 update to PR #1945 — that would have been better than V22's 1.05877 by 0.001 BPB.

**Lesson**: Don't switch targets mid-run unless the new target has materially higher confidence. PR #2018 was 9 minutes old when I switched — not enough time to verify compliance.

## Files

- `run.sh` — exact launcher used on pod (env vars match PR #2014 README)
- `reference_train_gpt.py` — PR #2014's train_gpt.py (4535 lines)
