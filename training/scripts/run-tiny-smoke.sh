#!/usr/bin/env bash
set -euo pipefail
cd /home/nink/pfor-work/training
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="/home/nink/pfor-work/training"
export CUDA_VISIBLE_DEVICES=0
mkdir -p /home/nink/pfor-ckpts/tiny-smoke
.venv/bin/python -m llmm_llm.train \
  --model tiny \
  --seq-len 64 \
  --overfit-single-batch \
  --epochs 1 \
  --checkpoint-dir /home/nink/pfor-ckpts/tiny-smoke \
  --eval-interval 1 \
  --eval-batches 1
echo SMOKE_OK
