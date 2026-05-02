# Post-V22 Final-Day Attempts (Archive — All Failed/Aborted)

This directory documents 4 alternative submissions attempted after V22 (PR #1945, 1.05877) was committed on 2026-04-30 21:02 UTC, before the deadline (2026-05-01 00:00 UTC PT-cutoff).

**None of these attempts beat V22.** They are archived for transparency, learning, and future-reference.

## Context

After V22 was committed (1.05877 BPB), competitors PR #2018 (simon-marcus 1.04617), PR #2009 (SlavH 1.0500), PR #2013 (Wilbatronic 1.0543) appeared in the final hours, threatening V22's position. I attempted 4 different replication / improvement runs on Hyperbolic eu-north-4 8×H100 SXM5 80GB pod, all failed.

## Summary of Attempts

| Version | Stack | Target BPB | Actual Result | Reason for Failure |
|---|---|---:|---:|---|
| **V23** | PR #2014 (simonbissonnette) Progressive Context Growth + Short-Doc TTT | 1.05759 | killed mid-run | V25 (PR #2018) appeared as better target, V23 sacrificed before seed 42 finished |
| **V24** | PR #2013 (Wilbatronic) jj6 Eval Trick Stack (vocab=1024) | 1.0543 | **1.4284 + 17.2MB artifact ❌** | DEPTH_RECUR=4 too slow on our env: only 964/5000 steps in 600s, model didn't converge; artifact also exceeded 16MB cap |
| **V25** | PR #2018 (simon-marcus) Gated XSA + LQER top-1 + In-timer N-gram TTT | 1.04617 | killed at pre-quant 1.06316 ❌ | Environment mismatch: our pre-quant was +0.014 BPB worse than simon-marcus's, suggesting our PyTorch/CUDA/Triton stack produces slightly different trained model |
| **V26** | V25 + GLOBAL_TTT_EPOCHS=2 + PHASED_TTT_PREFIX_DOCS=1500 (novel attempt to beat 1.04617) | < 1.04617 | **1.05741, eval 642s NON-COMPLIANT ❌** | GLOBAL_TTT_EPOCHS=2 doubled global TTT phase, pushing eval above 600s cap |

## Critical Insights from These Failures

### 1. Environment matters — V25 baseline pre-quant gap

Running PR #2018's exact code with our identical env vars produced **+0.014 BPB worse pre-quantization val_bpb** compared to simon-marcus's logs. Possible causes:
- PyTorch 2.9.1+cu128 vs simon-marcus's exact version
- CUDA 13 forward-compat driver 580 (our Hyperbolic VM)
- Triton compile output differences
- bfloat16 numerical determinism

**Lesson**: For competitive replication, the runtime environment is non-trivial. A "compliance reproduction" requires identical hardware + drivers + framework versions.

### 2. The post-deadline simon-marcus PR #2018 was retroactively banned

After deadline, on 2026-05-01 16:44 UTC, @sharpobject flagged that PR #2018's training set's first 40k documents overlapped with the validation set's last 40k documents — **C1 causality violation** (training on validation tokens). simon-marcus did not respond and pivoted to PR #2140 (1.0560).

cocohearts's official audit (PR #2146 on 2026-05-02) explicitly excluded PR #2018: "invalid due train/validation document overlap in the submitted CaseOps data construction".

**This means**: even if our V25 baseline replication had succeeded at simon-marcus's reported 1.046, it would have inherited the same training-validation overlap and been banned. Our env mismatch protected us from accidentally publishing a non-compliant claim.

### 3. PR #2013 (Wilbatronic) was always going to fail in 600s wallclock

DEPTH_RECUR=4 means each forward pass goes through the 11-layer block stack 4× (effective 44 layers). On our 8×H100 SXM5, we got ~623ms/step → only 964 steps in 600s. PR #2013 likely needs 4000+ steps for convergence, suggesting either:
- Wilbatronic had faster compute than us
- Or PR #2013's reported 1.0543 is single-seed cherry-picked from a longer training run

**Lesson**: Before committing to a 600s replication, check the reported `step_avg_ms` × stop_step to verify feasibility on target hardware.

### 4. V26's novel attempt taught a lesson about eval-time budget

V26 added `GLOBAL_TTT_EPOCHS=2` (run global TTT 2 epochs over prefix docs) hoping for marginal improvement. Result:
- val_bpb = 1.05741 (slightly worse than V22)
- **eval_time = 642s, exceeding 600s cap → NON-COMPLIANT**

simon-marcus's PR #2018 had eval ~542s with GLOBAL_TTT_EPOCHS=1. Doubling phase 1 added ~100s, just enough to bust the cap.

**Lesson**: When working with budget-constrained eval, ANY increase in compute must be measured against the budget margin. The headroom in PR #2018's 542s was not enough to absorb 2× global TTT.

## Final Outcome

**V22 (PR #1945, 1.05877 BPB, V21 v2 commit `7006753`) remains the official submission**. No replacement was made. cocohearts's audited leaderboard (PR #2146, 2026-05-02) places alertcat at the V21 v2 frontier row at 1.0594, in the chronological progression chain.

## Files in Each Subfolder

Each `V23/V24/V25/V26` subfolder contains:
- `run.sh` — exact launcher script used on Hyperbolic 8×H100 pod
- `reference_train_gpt.py` — the source PR's `train_gpt.py` we tried to replicate
- (no `train_seed*.log` — pod was offline before logs could be exported, except V26 seed 42 partial)

## Reproduction (if you want to verify)

```bash
# Clone this fork and check out v19-frontier
git clone https://github.com/alertcat/parameter-golf.git
cd parameter-golf
git checkout v19-frontier

# For each attempt, the reference_train_gpt.py is the source PR's train_gpt.py
# (downloaded from the original author's fork at the SHA they submitted)
# The run.sh contains exact env vars used on Hyperbolic 8×H100 SXM5 80GB

# To reproduce V25 baseline (PR #2018), you'd need:
# - Same Hyperbolic eu-north-4 region (or near-identical PyTorch+CUDA)
# - sp8192 CaseOps lossless caps tokenizer (from PR #1729)
# - fineweb10B sp8192 CaseOps reserved dataset
```

## Lineage Disclosure

These are not record submissions. They are documented attempts to replicate / improve on:
- **PR #2014** simonbissonnette (V23)
- **PR #2013** Wilbatronic (V24)  
- **PR #2018** simon-marcus (V25 baseline + V26 optimized)

Original credit for the stacks belongs to those authors (V25 was banned post-deadline by cocohearts for train/val overlap, separate from this archive).

---

*Author: alertcat (PR #1945 author)*
*Archive date: 2026-05-03*
