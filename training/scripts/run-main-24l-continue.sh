#!/usr/bin/env bash
set -euo pipefail
cd /home/nink/pfor-work/training
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="/home/nink/pfor-work/training"
export CUDA_VISIBLE_DEVICES=0
mkdir -p /home/nink/pfor-ckpts/main-24l-continue
# 24L checkpoints are ~2x 180M; interval 2000 keeps disk under ~200 GB for 3 epochs.
exec .venv/bin/python -m llmm_llm.train \
  --model main \
  --n-layers 24 \
  --seq-len 1024 \
  --router-top-k 1 \
  --learning-rate 1e-4 \
  --resume-weights /home/nink/pfor-ckpts/main-24l-upcycle/step-00000000.pt \
  --data-dir /home/nink/pfor-work/training/data/raw/pretraining/smollm-corpus/fineweb-edu-dedup \
  --checkpoint-dir /home/nink/pfor-ckpts/main-24l-continue \
  --checkpoint-interval 2000 \
  --eval-interval 50 \
  --eval-batches 2 \
  --epochs 3
