#!/usr/bin/env bash
set -euo pipefail
TRAIN_PID="${1:-}"
CKPT_DIR=/home/nink/pfor-ckpts/main-fineweb
OUT_DIR=/home/nink/pfor-ckpts/main-fineweb
TRAINING=/home/nink/pfor-work/training
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$TRAINING"
export CUDA_VISIBLE_DEVICES=0
cd "$TRAINING"

if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
  echo "waiting for train pid $TRAIN_PID ..."
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 30
  done
fi
echo "train exited $(date -Is)" | tee "$OUT_DIR/train-exited.txt"

# Prefer the last full checkpoint (also saved at total_steps).
.venv/bin/python -m export_aircraft \
  --checkpoint-dir "$CKPT_DIR" \
  --output "$OUT_DIR/pfor-180m-fineweb.llmcraft" \
  --load-device cuda
echo EXPORT_OK | tee -a "$OUT_DIR/train-exited.txt"

.venv/bin/python scripts/generate_prompts.py \
  --checkpoint-dir "$CKPT_DIR" \
  --tokenizer assets/qwen3.5-english-tokenizer/tokenizer.json \
  | tee "$OUT_DIR/gpu-samples.txt"
echo GENERATE_OK | tee -a "$OUT_DIR/train-exited.txt"
echo ALL_DONE | tee -a "$OUT_DIR/train-exited.txt"
