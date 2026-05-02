# V26: V25 + GLOBAL_TTT_EPOCHS=2 + PHASED_TTT_PREFIX_DOCS=1500 — NON-COMPLIANT eval timeout

**Status**: Failed — val_bpb 1.05741 (no improvement over V22), eval 642s exceeding 600s cap

## Hypothesis

Could the n-gram tilt + Gated XSA stack (PR #2018, claimed 1.04617) be improved by:
1. **GLOBAL_TTT_EPOCHS=2** — run global TTT phase 2 epochs over the 1500-doc prefix instead of 1 epoch
2. **PHASED_TTT_PREFIX_DOCS=1500** — middle ground between PR #2018's 1000 and the train_gpt.py default 2000

Both were eval-side only changes, train numerics identical to PR #2018.

## Environment

Same 8× H100 SXM5 80GB on Hyperbolic. Same `train_gpt.py` as V25.

## Run Result

```
seed 42:
  stopping_early: wallclock_cap train_time:596057ms step:4918/20000
  diagnostic pre-quantization post-ema val_bpb: 1.06304590    ← same env mismatch as V25 (+0.014 BPB)
  diagnostic quantized val_bpb: 1.07085572                    ← same as V25
  quantized_ttt_phased val_bpb: 1.05741642
  total_eval_time: 642.2s   ← OVER 600s CAP — NON-COMPLIANT ❌
  
  [seed 1337 killed because seed 42 was non-compliant]
```

## Root Cause

**GLOBAL_TTT_EPOCHS=2 doubled the global TTT phase compute, pushing total eval over 600s.**

PR #2018's eval timing breakdown:
- Diagnostic pre-quant: ~9s
- Diagnostic quantized: ~14s
- Phased TTT (1 epoch over 1000 prefix docs): ~542s
- Total: ~565s, well within 600s cap

Our V26 eval:
- Diagnostic pre-quant: ~11s
- Diagnostic quantized: ~65s (longer than PR #2018's 14s — possibly compile overhead)
- Phased TTT (2 epochs over 1500 prefix docs): ~566s
- Total: 642s, **42s over cap**

The 2-epoch + larger prefix combination consumed all the timing margin and then some.

## Why Final BPB Was 1.05741 (Same as V22)

Even if eval had been compliant, the final post-TTT BPB was 1.05741 — almost identical to V22's seed 42 (1.05733) and V21 v2's seed 42 (1.05867).

This is because:
1. **Pre-quant model was +0.014 BPB worse than simon-marcus** due to env mismatch (same as V25 baseline)
2. The N-gram tilt's expected -0.012 BPB recovery during TTT couldn't fully compensate for the +0.014 deficit
3. GLOBAL_TTT_EPOCHS=2 also consumed time that should have been TTT iteration steps, partially defeating the purpose

Net: V26 ≈ V22 in BPB but with much higher compute and non-compliant eval.

## Lessons

### 1. Always reserve eval-time budget margin

PR #2018's 565s/600s eval has only 35s margin. Any compute amplification (2× epoch, larger prefix, more TTT chunks) easily exceeds the cap. Lesson: assume 10-15% safety margin when modifying eval-time budget.

### 2. Compounding env mismatch with novel optimization confuses the signal

V26 changed 2 axes (epochs + prefix docs) on top of V25's already-broken pre-quant baseline. Even if the optimization had helped, we couldn't have measured its effect because V25's baseline was unreliable. Lesson: fix environment first, optimize second.

### 3. Eval timeout produces a "ghost result" that's worse than failure

When eval exceeds 600s cap, the scoring is technically invalid (the run is non-compliant), so the BPB number is meaningless for ranking. But the eval still produces a number, which I initially read as "no improvement". A non-compliant run is worse than no run because it consumes GPU time without producing usable data.

### 4. Post-deadline pivot also failed

After V26 failed, I attempted V24 (PR #2013) as a desperation move at ~SGT 06:40 (50 min before what I incorrectly thought was the deadline). V24 also failed catastrophically (1.4284 BPB + 17.2MB artifact). 

Both V25/V26 and V24 were burned in the false-urgency window. The correct decision (in retrospect) was to keep V22 on PR #1945 and accept the position.

## Files

- `run.sh` — exact V26 launcher (V25 env vars + GLOBAL_TTT_EPOCHS=2 + PHASED_TTT_PREFIX_DOCS=1500)
- `reference_train_gpt.py` — PR #2018's train_gpt.py (same as V25, no train_gpt.py changes)
