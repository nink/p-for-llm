#!/usr/bin/env bash
set -euo pipefail
DEST=/home/nink/pfor-work/training/data/raw/tokenizer-validation/tinystories
mkdir -p "$DEST"
URL="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/TinyStories-valid.txt"
if [ ! -f "$DEST/TinyStories-valid.txt" ]; then
  curl -L --fail --retry 5 -o "$DEST/TinyStories-valid.txt" "$URL"
fi
ls -l "$DEST/TinyStories-valid.txt"
sha256sum "$DEST/TinyStories-valid.txt"
echo DOWNLOAD_OK
