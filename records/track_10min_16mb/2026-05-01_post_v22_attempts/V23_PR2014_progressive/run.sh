#!/bin/bash
# V23 = PR #2014 (Progressive3k + ShortDocTTT) replication
# Goal: beat V22 (1.05877) and PR #2014 (1.05759) baseline
# Stack: PR #1855 base + AWQ-lite + AsymLogit + WD=2.0 + Progressive 3k context + Short-doc TTT
set -e

# Inherit login PATH so torchrun is found under nohup
export PATH="/home/ubuntu/.local/bin:/usr/local/cuda-13.0/bin:$PATH"
echo "PATH=$PATH"
which torchrun || { echo "FATAL: torchrun missing"; exit 1; }

# Use PR #2014 train_gpt.py (4535 lines) staged at /workspace/PR2014_train_gpt.py
# Seeds: 42, 0, 1234 (V22-consistent for apples-to-apples comparison)

# Stage in dedicated v23 dir
V23_DIR=/workspace/parameter-golf-v23
mkdir -p "$V23_DIR"
cp /workspace/PR2014_train_gpt.py "$V23_DIR/train_gpt.py"
cp /workspace/PR2014_lossless_caps.py "$V23_DIR/lossless_caps.py" 2>/dev/null || true

cd "$V23_DIR"

echo "===================================================="
echo "  V23 = PR #2014 Progressive3k + ShortDocTTT"
echo "  3-seed: 42, 0, 1234   Start: $(date)"
echo "  Train file: $(wc -l train_gpt.py) lines"
echo "===================================================="

# All env vars from PR #2014 README reproduction command (verified vs train_seed42.log)
ENV_VARS_V23="DATA_DIR=/workspace/caseops_data/datasets/ \
  DATA_PATH=/workspace/caseops_data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
  TOKENIZER_PATH=/workspace/caseops_data/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
  CASEOPS_ENABLED=1 VOCAB_SIZE=8192 \
  ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
  EVAL_INCLUDE_TAIL=1 \
  TRAIN_SEQ_LEN=3072 ROPE_TRAIN_SEQ_LEN=3072 \
  TRAIN_SEQ_SCHEDULE=1024@0.100,2048@0.700,3072@1.000 \
  TRAIN_SEQ_SCHEDULE_MODE=wallclock \
  SEQ_CHANGE_WARMUP_STEPS=32 \
  EVAL_SEQ_LEN=3072 EVAL_STRIDE=1536 \
  TTT_ENABLED=1 \
  TTT_EVAL_SEQ_LEN=3072 TTT_BATCH_SIZE=24 TTT_CHUNK_SIZE=48 \
  TTT_SHORT_SCORE_FIRST_ENABLED=1 \
  TTT_SHORT_DOC_LEN=2000 \
  TTT_SHORT_CHUNK_SIZE=24 \
  TTT_SHORT_SCORE_FIRST_STEPS=256:8,2000:24 \
  TTT_LORA_RANK=80 TTT_LORA_LR=0.0001 \
  TTT_LOCAL_LR_MULT=0.75 \
  TTT_MASK=no_qv TTT_Q_LORA=0 TTT_V_LORA=0 \
  TTT_WEIGHT_DECAY=0.5 TTT_BETA2=0.99 \
  PHASED_TTT_PREFIX_DOCS=2500 PHASED_TTT_NUM_PHASES=1 \
  WARMDOWN_FRAC=0.85 BETA2=0.99 \
  QK_GAIN_INIT=5.25 \
  SPARSE_ATTN_GATE_ENABLED=1 SPARSE_ATTN_GATE_SCALE=0.5 \
  GATED_ATTN_QUANT_GATE=1 \
  SMEAR_GATE_ENABLED=1 GATE_WINDOW=12 \
  FUSED_CE_ENABLED=1 \
  MATRIX_LR=0.026 MIN_LR=0.1 GRAD_CLIP_NORM=0.3 \
  EMBED_BITS=7 \
  EMBED_CLIP_SIGMAS=14.0 MATRIX_CLIP_SIGMAS=12.85 \
  ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=11.5 \
  LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 \
  LQER_FACTOR_BITS=4 LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64 \
  AWQ_LITE_ENABLED=1 AWQ_LITE_BITS=8 \
  AWQ_LITE_GROUP_TOP_K=1 AWQ_LITE_GROUP_SIZE=64 \
  ASYM_LOGIT_RESCALE=1 \
  GPTQ_RESERVE_SECONDS=4.0 GPTQ_CALIBRATION_BATCHES=16 \
  COMPRESSOR=pergroup \
  VAL_LOSS_EVERY=0 \
  NCCL_NET=Socket"

for SEED in 42 0 1234; do
  echo ""
  echo "========================================"
  echo "  V23 SEED $SEED  Start: $(date)"
  echo "========================================"

  env SEED=$SEED $ENV_VARS_V23 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py \
    > /workspace/scout_v23_seed${SEED}.log 2>&1 || {
      echo "!!! V23 SEED $SEED FAILED at $(date) - see log tail:"
      tail -30 /workspace/scout_v23_seed${SEED}.log
      continue
    }

  cp "$V23_DIR/final_model.int6.ptz" /workspace/v23_seed${SEED}_model.int6.ptz 2>/dev/null || true

  echo "--- V23 Seed $SEED done at $(date) ---"
  grep -E "stopping_early|train_time|quantized_ttt_phased|Total submission|total_eval_time|val_tokens|target_tokens" /workspace/scout_v23_seed${SEED}.log | tail -10
done

echo ""
echo "===================================================="
echo "  V23 3-SEED FINAL RESULTS  $(date)"
echo "===================================================="
python3 << 'PYEOF'
import re

def get_data(seed):
    try:
        with open(f'/workspace/scout_v23_seed{seed}.log') as f:
            c = f.read()
    except FileNotFoundError:
        return None
    bpb_m = re.search(r'quantized_ttt_phased\s+val_loss:[\d.]+\s+val_bpb:([\d.]+)', c)
    sz_m  = re.search(r'Total submission size quantized\+pergroup:\s+(\d+)', c)
    wt_m  = re.search(r'stopping_early:\s+wallclock_cap\s+train_time:\s+(\d+)ms', c)
    et_m  = re.search(r'total_eval_time:([\d.]+)s', c)
    tok_m = re.search(r'val_tokens:(\d+)\s+target_tokens:(\d+)', c)
    return {
        'val_bpb': float(bpb_m.group(1)) if bpb_m else None,
        'artifact': int(sz_m.group(1)) if sz_m else None,
        'train_ms': int(wt_m.group(1)) if wt_m else None,
        'eval_s': float(et_m.group(1)) if et_m else None,
        'val_tok': int(tok_m.group(1)) if tok_m else None,
        'tgt_tok': int(tok_m.group(2)) if tok_m else None,
    }

results = {s: get_data(s) for s in [42, 0, 1234]}
print(f"{'seed':>6} {'val_bpb':>11} {'artifact':>12} {'train':>10} {'eval':>10} {'val/target':>20}")
for s in [42, 0, 1234]:
    r = results[s]
    if r and r['val_bpb']:
        cov = f"{r['val_tok']}/{r['tgt_tok']}" if r['val_tok'] else "?/?"
        print(f"{s:>6} {r['val_bpb']:>11.6f} {r['artifact']:>12,} {r['train_ms']/1000:>9.2f}s {r['eval_s']:>9.2f}s {cov:>20}")
    else:
        print(f"{s:>6} MISSING")

vals = [r['val_bpb'] for r in results.values() if r and r['val_bpb']]
if len(vals) == 3:
    mean = sum(vals)/3
    std = (sum((v-mean)**2 for v in vals)/3)**0.5
    print(f"\n  V23 3-SEED MEAN: {mean:.6f}")
    print(f"  V23 3-SEED STD:  {std:.6f}")
    print()
    print(f"  vs V22         (1.05877):  delta {1.05877 - mean:+.6f}")
    print(f"  vs PR #2014    (1.05759):  delta {1.05759 - mean:+.6f}")
    print(f"  vs PR #1953    (1.05855):  delta {1.05855 - mean:+.6f}")
    print(f"  vs PR #1967    (1.05851):  delta {1.05851 - mean:+.6f}")
    print(f"  vs PR #2009    (1.0500):   delta {1.0500 - mean:+.6f}")
    if mean < 1.0500:
        print(f"\n  *** V23 BEATS PR #2009 (DepthShare4096)! NEW SOTA ***")
    elif mean < 1.05759:
        print(f"\n  *** V23 BEATS PR #2014 (Progressive3k)! ***")
    elif mean < 1.05851:
        print(f"\n  *** V23 BEATS PR #1953/1967 — top 3 ***")
    elif mean < 1.05877:
        print(f"\n  *** V23 BEATS V22 — improvement ***")
    else:
        print(f"\n  V23 doesn't improve V22 - regression")
PYEOF
