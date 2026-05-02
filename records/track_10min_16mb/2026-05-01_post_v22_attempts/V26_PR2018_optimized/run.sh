#!/bin/bash
# V26 = V25 (PR #2018) + 2 untested levers (novel attempt to beat 1.04617)
# 1. GLOBAL_TTT_EPOCHS=2 (V25 default=1)         <- Phase 1 runs 2 epochs
# 2. PHASED_TTT_PREFIX_DOCS=1500 (V25=1000)      <- Mid-value between 1000 and 2000
# Target: < 1.04617 (3-seed mean strict beat simon-marcus)
# Seeds: 42, 1337, 2026 (THEIR seeds for direct comparability)
set -e

# Inherit login PATH
export PATH="/home/ubuntu/.local/bin:/usr/local/cuda-13.0/bin:$PATH"
export PYTHONUNBUFFERED=1
echo "PATH=$PATH"
which torchrun || { echo "FATAL: torchrun missing"; exit 1; }

V26_DIR=/workspace/pr2018_data
cd "$V26_DIR"

echo "===================================================="
echo "  V26 = V25 + GLOBAL_TTT_EPOCHS=2 + PREFIX_DOCS=1500"
echo "  Target: < 1.04617 (beat simon-marcus PR #2018)"
echo "  3-seed: 42, 1337, 2026  Start: $(date)"
echo "===================================================="

# Same as V25 except GLOBAL_TTT_EPOCHS=2 and PHASED_TTT_PREFIX_DOCS=1500
ENV_VARS_V26="DATA_PATH=/workspace/caseops_data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
  TOKENIZER_PATH=/workspace/caseops_data/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
  CASEOPS_ENABLED=1 VOCAB_SIZE=8192 \
  ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
  TTT_ENABLED=1 PHASED_TTT_ENABLED=1 PHASED_TTT_NUM_PHASES=1 PHASED_TTT_PREFIX_DOCS=1500 \
  GLOBAL_TTT_EPOCHS=2 \
  TTT_LORA_RANK=80 TTT_MASK=no_qv TTT_Q_LORA=0 TTT_V_LORA=0 \
  TTT_LOCAL_LR_MULT=0.75 \
  EVAL_SEQ_LEN=2560 TTT_EVAL_SEQ_LEN=2560 \
  QK_GAIN_INIT=5.25 \
  MATRIX_LR=0.026 MIN_LR=0.1 EMBED_BITS=7 GRAD_CLIP_NORM=0.3 \
  MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 \
  MLP_CLIP_SIGMAS=11.5 EMBED_CLIP_SIGMAS=14.0 \
  FUSED_CE_ENABLED=1 \
  SMEAR_GATE_ENABLED=1 GATE_WINDOW=12 \
  SPARSE_ATTN_GATE_ENABLED=1 \
  LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=1 \
  LQER_GROUP_SIZE=64 LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64 \
  AWQ_LITE_ENABLED=1 \
  ASYM_LOGIT_RESCALE=1 \
  NGRAM_TILT_ENABLED=1 \
  NGRAM_HINT_PRECOMPUTE_OUTSIDE=0 \
  GATED_XSA=1 \
  SKYLIGHT_MUON=0 \
  GPTQ_RESERVE_SECONDS=4.0 GPTQ_CALIBRATION_BATCHES=16 \
  COMPRESSOR=pergroup \
  VAL_LOSS_EVERY=0 \
  NCCL_NET=Socket"

for SEED in 42 1337 2026; do
  echo ""
  echo "========================================"
  echo "  V26 SEED $SEED  Start: $(date)"
  echo "========================================"

  env SEED=$SEED $ENV_VARS_V26 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py \
    > /workspace/scout_v26_seed${SEED}.log 2>&1 || {
      echo "!!! V26 SEED $SEED FAILED at $(date) - log tail:"
      tail -40 /workspace/scout_v26_seed${SEED}.log
      continue
    }

  cp final_model.int6.ptz /workspace/v26_seed${SEED}_model.int6.ptz 2>/dev/null || true

  echo "--- V26 Seed $SEED done at $(date) ---"
  grep -E "stopping_early|train_time|quantized_ttt_phased|Total submission|total_eval_time" /workspace/scout_v26_seed${SEED}.log | tail -8
done

echo ""
echo "===================================================="
echo "  V26 3-SEED FINAL RESULTS  $(date)"
echo "===================================================="
python3 << 'PYEOF'
import re

def get_data(seed):
    try:
        with open(f'/workspace/scout_v26_seed{seed}.log') as f:
            c = f.read()
    except FileNotFoundError:
        return None
    bpb_m = re.search(r'quantized_ttt_phased\s+val_loss:[\d.]+\s+val_bpb:([\d.]+)', c)
    sz_m  = re.search(r'Total submission size quantized\+pergroup:\s+(\d+)', c)
    wt_m  = re.search(r'stopping_early:\s+wallclock_cap\s+train_time:\s+(\d+)ms', c)
    et_m  = re.search(r'total_eval_time:([\d.]+)s', c)
    return {
        'val_bpb': float(bpb_m.group(1)) if bpb_m else None,
        'artifact': int(sz_m.group(1)) if sz_m else None,
        'train_ms': int(wt_m.group(1)) if wt_m else None,
        'eval_s': float(et_m.group(1)) if et_m else None,
    }

results = {s: get_data(s) for s in [42, 1337, 2026]}
print(f"{'seed':>6} {'val_bpb':>11} {'artifact':>12} {'train':>10} {'eval':>10}")
for s in [42, 1337, 2026]:
    r = results[s]
    if r and r['val_bpb']:
        print(f"{s:>6} {r['val_bpb']:>11.6f} {r['artifact']:>12,} {r['train_ms']/1000 if r['train_ms'] else 0:>9.2f}s {r['eval_s'] if r['eval_s'] else 0:>9.2f}s")
    else:
        print(f"{s:>6} MISSING")

vals = [r['val_bpb'] for r in results.values() if r and r['val_bpb']]
if len(vals) == 3:
    mean = sum(vals)/3
    std = (sum((v-mean)**2 for v in vals)/3)**0.5
    print(f"\n  V26 3-SEED MEAN: {mean:.6f}")
    print(f"  V26 3-SEED STD:  {std:.6f}")
    print()
    print(f"  vs PR #2018    (1.04617): delta {1.04617 - mean:+.6f}  ** TARGET **")
    print(f"  vs PR #2009    (1.05003): delta {1.05003 - mean:+.6f}")
    print(f"  vs V22         (1.05877): delta {1.05877 - mean:+.6f}")
    if mean < 1.04617:
        delta = 1.04617 - mean
        print(f"\n  *** V26 BEATS PR #2018 1.04617! delta={delta:+.6f} BPB ***")
        print(f"  *** RECORD CANDIDATE -- PUSH TO NEW PR ***")
    elif mean < 1.046:
        print(f"\n  *** V26 close to or matching PR #2018 ***")
    else:
        print(f"\n  V26 doesn't beat PR #2018 - investigate")
PYEOF
