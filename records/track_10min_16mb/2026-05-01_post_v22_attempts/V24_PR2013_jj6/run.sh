#!/bin/bash
# V24 = PR #2013 (Wilbatronic jj6 Eval Trick Stack) replication
# Target: beat V22 (1.05877) and PR #2013 (1.0543) baseline
# Stack: 10L attention SwiGLU + BigramHash + SmearGate + DepthRecur4 + ResidualSigns + StochEval + TTT-LoRA
set -e

# Inherit login PATH so torchrun is found under nohup
export PATH="/home/ubuntu/.local/bin:/usr/local/cuda-13.0/bin:$PATH"
export PYTHONUNBUFFERED=1
echo "PATH=$PATH"
which torchrun || { echo "FATAL: torchrun missing"; exit 1; }

# Use PR #2013 train_gpt.py + attention_playground.py + sp1024 data (already staged)
V24_DIR=/workspace/pr2013_data
cd "$V24_DIR"
ls -la train_gpt.py attention_playground.py data/datasets/fineweb10B_sp1024/ | head -10

echo "===================================================="
echo "  V24 = PR #2013 jj6 Eval Trick Stack"
echo "  3-seed: 42, 0, 1234  Start: $(date)"
echo "  Train file: $(wc -l train_gpt.py) lines"
echo "===================================================="

# Env vars from launch_8xh100_competition.sh (verified vs PR #2013 README target 1.0543)
# We use seeds 42, 0, 1234 instead of single seed 99 (community 3-seed standard)
ENV_VARS_V24="DATA_PATH=/workspace/pr2013_data/data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=/workspace/pr2013_data/data/tokenizers/fineweb_1024_bpe.model \
  ARCH=attention \
  RESET_ON_BOS=1 \
  VOCAB_SIZE=1024 \
  NUM_LAYERS=11 \
  MLP_ACTIVATION=leakyrelu2 \
  TRAIN_SEQ_LEN=2048 \
  WARMUP_STEPS=20 \
  SOFTCAP=10.0 \
  ROPE_DIMS=32 \
  QK_GAIN=1.0 \
  BWCE=1 \
  CGGR_RATIO=1.0 \
  TRIGRAM_HASH_SIZE=5120 \
  XSA_LAST_N=11 \
  SCALAR_LR=0.004 \
  TIED_EMBED_LR=0.005 \
  COMPILE_MODEL=1 \
  VAL_MAX_TOKENS=4194304 \
  TRAIN_LOG_EVERY=100 \
  VAL_LOSS_EVERY=200 \
  TRAIN_BATCH_TOKENS=524288 \
  GRAD_ACCUM_STEPS=8 \
  MAX_WALLCLOCK_SECONDS=600 \
  ITERATIONS=5000 \
  WARMDOWN_ITERS=20 \
  SUBMIT_FMT=mixed_lzma \
  INT4_QUANT=1 \
  INT4_GROUP_SIZE=32 \
  INT4_STOCH_QUANT=1 \
  QAT_START_FRACTION=0.0 \
  DEPTH_RECUR=4 \
  EMA_DECAY=0.995 \
  RESIDUAL_SIGNS=1 \
  RESIDUAL_SIGNS_FILTER=ffn \
  RESIDUAL_SIGNS_FFN_LAYER=down \
  RESIDUAL_SIGNS_BLOCKS=0,2,3,5,6,7,8,9,10 \
  SELECTIVE_INT6_BLOCKS=0 \
  STOCH_EVAL_N=16 \
  STOCH_EVAL_EPS=0.5 \
  TTT_RANK=8 \
  TTT_STEPS=5 \
  TTT_LR=0.01 \
  TTT_CHUNK_SEQS=4 \
  TTT_MOMENTUM=0.9 \
  OUTLIER_FILTER=all \
  OUTLIER_HESSIAN=1 \
  OUTLIER_HESSIAN_BATCHES=8 \
  OUTLIER_TOPK_FRAC=0.0029 \
  OUTLIER_FRAC_TOP=0.00485 \
  OUTLIER_FRAC_BOTTOM=0.00095 \
  LOGIT_TEMP=1.02"

for SEED in 42 0 1234; do
  RUN_ID="v24_seed${SEED}"
  echo ""
  echo "========================================"
  echo "  V24 SEED $SEED  Start: $(date)"
  echo "========================================"

  env SEED=$SEED RUN_ID=$RUN_ID $ENV_VARS_V24 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py \
    > /workspace/scout_v24_seed${SEED}.log 2>&1 || {
      echo "!!! V24 SEED $SEED FAILED at $(date) - log tail:"
      tail -40 /workspace/scout_v24_seed${SEED}.log
      continue
    }

  echo "--- V24 Seed $SEED done at $(date) ---"
  grep -E "final_(mixed_lzma|mixed_zstd|int8_zlib|int4_zlib)_roundtrip_exact|Total submission size|train_time:|val_bpb:" /workspace/scout_v24_seed${SEED}.log | tail -10
done

echo ""
echo "===================================================="
echo "  V24 3-SEED FINAL RESULTS  $(date)"
echo "===================================================="
python3 << 'PYEOF'
import re

def get_data(seed):
    try:
        with open(f'/workspace/scout_v24_seed{seed}.log') as f:
            c = f.read()
    except FileNotFoundError:
        return None
    # Roundtrip val_bpb (the official submission number)
    rt = re.findall(r'final_\w+_roundtrip_exact[^\n]*val_bpb:([\d.]+)', c)
    rt_bpb = float(rt[-1]) if rt else None
    # Pre-quant val_bpb (last step)
    pre_lines = re.findall(r'val_bpb:([\d.]+)', c)
    pre_bpb = float(pre_lines[-2]) if len(pre_lines) >= 2 else None
    # Submission size
    sz = re.findall(r'Total submission size [^:]+:\s*(\d+)', c)
    artifact = int(sz[-1]) if sz else None
    # Train time
    tt = re.findall(r'train_time:\s*(\d+)', c)
    train_ms = int(tt[-1]) if tt else None
    return {
        'val_bpb': rt_bpb,
        'pre_quant_bpb': pre_bpb,
        'artifact': artifact,
        'train_ms': train_ms,
    }

results = {s: get_data(s) for s in [42, 0, 1234]}
print(f"{'seed':>6} {'val_bpb':>11} {'pre_quant':>11} {'artifact':>12} {'train':>10}")
for s in [42, 0, 1234]:
    r = results[s]
    if r and r['val_bpb']:
        print(f"{s:>6} {r['val_bpb']:>11.6f} {r.get('pre_quant_bpb', 0):>11.6f} {r['artifact']:>12,} {r['train_ms']/1000 if r['train_ms'] else 0:>9.2f}s")
    else:
        print(f"{s:>6} MISSING")

vals = [r['val_bpb'] for r in results.values() if r and r['val_bpb']]
if len(vals) == 3:
    mean = sum(vals)/3
    std = (sum((v-mean)**2 for v in vals)/3)**0.5
    print(f"\n  V24 3-SEED MEAN: {mean:.6f}")
    print(f"  V24 3-SEED STD:  {std:.6f}")
    print()
    print(f"  vs V22         (1.05877):  delta {1.05877 - mean:+.6f}")
    print(f"  vs PR #2014    (1.05759):  delta {1.05759 - mean:+.6f}")
    print(f"  vs PR #2013    (1.0543):   delta {1.0543 - mean:+.6f}")
    print(f"  vs PR #2009    (1.0500):   delta {1.0500 - mean:+.6f}")
    if mean < 1.0500:
        print(f"\n  *** V24 BEATS PR #2009 (DepthShare4096)! NEW SOTA ***")
    elif mean < 1.0543:
        print(f"\n  *** V24 BEATS PR #2013 (jj6)! TOP 2 ***")
    elif mean < 1.05759:
        print(f"\n  *** V24 BEATS PR #2014 (Progressive3k)! TOP 3 ***")
    elif mean < 1.05877:
        print(f"\n  *** V24 BEATS V22 ***")
    else:
        print(f"\n  V24 doesn't improve V22 - regression")
PYEOF
