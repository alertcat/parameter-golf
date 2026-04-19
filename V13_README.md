# 11L + 3-Layer Recur + Parallel Resid + QK-Gain 5.5 + DS Bias + Legal TTT

**val_bpb: TBD** | 8xH100 SXM, 600s

Built on PR #1493 (bigbag, 1.0810 BPB). Two added innovations:

## What is new

1. **QK-Gain 5.5** (monotone increase from 5.25). Continues the sweep: dexhunter 5.0, bigbag 5.25, this 5.5.
2. **Per-position Document-Start Bias**. `nn.Parameter(torch.zeros(64, vocab_size))` applied to logits at positions 0-63. Captures high document-start entropy cheaply (524K params ~0.5MB at int8).

## Carried from PR #1493

Entire bigbag stack: 11L d=512 GQA 8/4, MLP 4x LeakyReLU(0.5)^2, 3-layer recurrence, parallel residuals from layer 7, partial RoPE 16/64, U-Net skip gates, MuonEq-R, score-first TTT (Issue #1017 Track B), full-Hessian GPTQ + SDClip, byte-shuffle + Brotli-11, SP8192 tokenizer.

## Compliance (Issue #1017 Track B)

DS bias is a pre-training learnable prior, identical to positional embeddings. Trained during 10-minute training phase. No post-hoc modification.

- Causality PASS
- Normalized output PASS
- Score-before-update PASS
- Single pass PASS

## Command

```bash
pip install brotli sentencepiece zstandard
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

SEED=42 QK_GAIN_INIT=5.5 DS_BIAS_ENABLED=1 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```
