#!/bin/bash
# V25 = PR #2018 (simon-marcus Gated XSA + LQER top-1 + In-timer N-gram TTT) replication
# Target: beat V22 (1.05877) and replicate PR #2018 (1.04617 mean) baseline
# Stack: V21 + LeakyReLU 0.3 + Gated XSA + LQER top-1 + In-timer ngram tilt + Cheaper phased TTT
# Seeds: 42, 1337, 2026 (THEIR seeds, for direct reproducibility verification)
set -e

# Inherit login PATH so torchrun is found under nohup
export PATH="/home/ubuntu/.local/bin:/usr/local/cuda-13.0/bin:$PATH"
export PYTHONUNBUFFERED=1
echo "PATH=$PATH"
which torchrun || { echo "FATAL: torchrun missing"; exit 1; }

V25_DIR=/workspace/pr2018_data
cd "$V25_DIR"
ls -la train_gpt.py online_ngram_tilt.py online_ngram_state.c | head

echo "===================================================="
echo "  V25 = PR #2018 simon-marcus 1.04617 BPB"
echo "  3-seed: 42, 1337, 2026  Start: $(date)"
echo "===================================================="

# Env vars from PR #2018 README reproduction command (verified against submission.json)
# CaseOps tokenizer + sp8192 data already on pod from V22/V23 setup
ENV_VARS_V25="DATA_PATH=/workspace/caseops_data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
  TOKENIZER_PATH=/workspace/caseops_data/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
  CASEOPS_ENABLED=1 VOCAB_SIZE=8192 \
  ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
  TTT_ENABLED=1 PHASED_TTT_ENABLED=1 PHASED_TTT_NUM_PHASES=1 PHASED_TTT_PREFIX_DOCS=1000 \
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
  echo "  V25 SEED $SEED  Start: $(date)"
  echo "========================================"

  env SEED=$SEED $ENV_VARS_V25 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py \
    > /workspace/scout_v25_seed${SEED}.log 2>&1 || {
      echo "!!! V25 SEED $SEED FAILED at $(date) - log tail:"
      tail -40 /workspace/scout_v25_seed${SEED}.log
      continue
    }

  cp final_model.int6.ptz /workspace/v25_seed${SEED}_model.int6.ptz 2>/dev/null || true

  echo "--- V25 Seed $SEED done at $(date) ---"
  grep -E "stopping_early|train_time|quantized_ttt_phased|Total submission|total_eval_time" /workspace/scout_v25_seed${SEED}.log | tail -8
done

echo ""
echo "===================================================="
echo "  V25 3-SEED FINAL RESULTS  $(date)"
echo "===================================================="
python3 << 'PYEOF'
import re

def get_data(seed):
    try:
        with open(f'/workspace/scout_v25_seed{seed}.log') as f:
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
    print(f"\n  V25 3-SEED MEAN: {mean:.6f}")
    print(f"  V25 3-SEED STD:  {std:.6f}")
    print()
    print(f"  vs PR #2018    (1.04617):  delta {1.04617 - mean:+.6f}  (target replication)")
    print(f"  vs PR #2009    (1.05003):  delta {1.05003 - mean:+.6f}")
    print(f"  vs V22         (1.05877):  delta {1.05877 - mean:+.6f}")
    if mean < 1.046:
        print(f"\n  *** V25 BEATS PR #2018 base 1.04617! ***")
    elif mean < 1.048:
        print(f"\n  *** V25 close to PR #2018 (within 0.002), strong #1-2 ***")
    elif mean < 1.052:
        print(f"\n  *** V25 in 1.046-1.052 range, between PR #2018 and PR #2009 ***")
    else:
        print(f"\n  V25 worse than expected, investigate")
PYEOF
