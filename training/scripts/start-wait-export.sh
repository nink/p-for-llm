#!/usr/bin/env bash
set -euo pipefail
PID=$(pgrep -f 'llmm_llm.train --model main' | head -n1 || true)
echo "TRAIN_PID=${PID}"
if [ -z "${PID}" ]; then
  echo "no trainer; running export now"
  exec bash /home/nink/pfor-work/training/scripts/wait-export.sh
fi
setsid nohup bash /home/nink/pfor-work/training/scripts/wait-export.sh "$PID" \
  >> /home/nink/pfor-ckpts/main-fineweb/wait-export.log 2>&1 < /dev/null &
echo "WAIT_EXPORT_STARTED pid=$! watching $PID"
