#!/usr/bin/env bash
set -euo pipefail
cd /home/nink/pfor-work/training
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="/home/nink/pfor-work/training"
export CUDA_VISIBLE_DEVICES=0
mkdir -p /home/nink/pfor-ckpts/sft-original
exec .venv/bin/python -m llmm_llm.train \
  --model main \
  --stage sft \
  --router-top-k 1 \
  --learning-rate 1e-4 \
  --resume-weights /home/nink/pfor-ckpts/original-reconstructed/step-00000000.pt \
  --sft-pool /home/nink/pfor-ckpts/sft-v2-pool \
  --replay-data-dir /home/nink/pfor-work/training/data/raw/pretraining/dclm-sft-v2-replay \
  --checkpoint-dir /home/nink/pfor-ckpts/sft-original-smoke \
  --max-steps 3 \
  --eval-interval 3 \
  --eval-batches 1 \
  --epochs 1
