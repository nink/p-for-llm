#!/usr/bin/env bash
set -euo pipefail
DEST=/home/nink/pfor-work/training/data/raw/pretraining/smollm-corpus/fineweb-edu-dedup
mkdir -p "$DEST"
FILE="$DEST/train-00000-of-00234.parquet"
URL="https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/resolve/3ba9d605774198c5868892d7a8deda78031a781f/fineweb-edu-dedup/train-00000-of-00234.parquet"
if [ -f "$FILE" ]; then
  echo "already have $FILE"
  ls -l "$FILE"
  exit 0
fi
curl -L --fail --retry 8 --continue-at - -o "$FILE.partial" "$URL"
mv "$FILE.partial" "$FILE"
ls -l "$FILE"
sha256sum "$FILE"
echo FINEWEB_OK
