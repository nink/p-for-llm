#!/usr/bin/env bash
set -euo pipefail
cd /home/nink/pfor-work/training
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="/home/nink/pfor-work/training"
export CUDA_VISIBLE_DEVICES=0
LOG_DIR=/home/nink/pfor-ckpts/sft-original
mkdir -p "$LOG_DIR"

echo "=== fetch ==="
bash scripts/fetch-sft-v2.sh

echo "=== prepare pools ==="
.venv/bin/python scripts/prepare-sft-v2.py

echo "=== smoke ==="
bash scripts/smoke-sft-original.sh

echo "=== full sft ==="
bash scripts/run-sft-original.sh

echo "=== greedy gate ==="
.venv/bin/python scripts/greedy_compare.py \
  --candidate-dir /home/nink/pfor-ckpts/sft-original \
  --out /home/nink/pfor-ckpts/sft-original/greedy-vs-original.json

echo SFT_PIPELINE_OK
