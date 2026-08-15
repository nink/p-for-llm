#!/usr/bin/env bash
set -euo pipefail
cd /home/nink/pfor-work/training
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="/home/nink/pfor-work/training"
export CUDA_VISIBLE_DEVICES=0
mkdir -p /home/nink/pfor-ckpts/main-fineweb
exec .venv/bin/python -m llmm_llm.train \
  --model main \
  --seq-len 1024 \
  --data-dir /home/nink/pfor-work/training/data/raw/pretraining/smollm-corpus/fineweb-edu-dedup \
  --checkpoint-dir /home/nink/pfor-ckpts/main-fineweb \
  --checkpoint-interval 200 \
  --eval-interval 50 \
  --eval-batches 2 \
  --epochs 1
