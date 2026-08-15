#!/usr/bin/env bash
set -euo pipefail
cd /home/nink/pfor-work/training
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="/home/nink/pfor-work/training"
export CUDA_VISIBLE_DEVICES=0
mkdir -p /home/nink/pfor-ckpts/main-smoke
.venv/bin/python -m llmm_llm.train \
  --model main \
  --seq-len 1024 \
  --overfit-single-batch \
  --max-steps 3 \
  --checkpoint-interval 1 \
  --checkpoint-dir /home/nink/pfor-ckpts/main-smoke \
  --eval-interval 1 \
  --eval-batches 1
echo MAIN_SMOKE_OK
