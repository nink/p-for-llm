#!/usr/bin/env bash
set -euo pipefail
cd /home/nink/pfor-work/training
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="/home/nink/pfor-work/training"
export CUDA_VISIBLE_DEVICES=0
mkdir -p /home/nink/pfor-ckpts/main-24l-upcycle /home/nink/pfor-ckpts/main-24l-continue
.venv/bin/python scripts/upcycle_depth.py \
  --source /home/nink/pfor-ckpts/original-reconstructed/step-00000000.pt \
  --out /home/nink/pfor-ckpts/main-24l-upcycle/step-00000000.pt \
  --n-layers 24 \
  --device cpu
.venv/bin/python -m llmm_llm.train \
  --model main \
  --n-layers 24 \
  --seq-len 1024 \
  --router-top-k 1 \
  --learning-rate 1e-4 \
  --resume-weights /home/nink/pfor-ckpts/main-24l-upcycle/step-00000000.pt \
  --data-dir /home/nink/pfor-work/training/data/raw/pretraining/smollm-corpus/fineweb-edu-dedup \
  --checkpoint-dir /home/nink/pfor-ckpts/main-24l-continue \
  --max-steps 3 \
  --eval-interval 3 \
  --eval-batches 1 \
  --epochs 1
echo SMOKE_24L_OK
