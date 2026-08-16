#!/usr/bin/env bash
set -euo pipefail
# Download locked sft-v2 conversation shards + one DCLM replay shard.
ROOT=/home/nink/pfor-work/training/data/raw
SFT="$ROOT/sft-v2"
REPLAY="$ROOT/pretraining/dclm-sft-v2-replay"
mkdir -p "$SFT" "$REPLAY"

fetch_one() {
  local dest="$1" url="$2" sha="$3"
  mkdir -p "$(dirname "$dest")"
  if [ -f "$dest" ]; then
    local got
    got="$(sha256sum "$dest" | awk '{print $1}')"
    if [ "$got" = "$sha" ]; then
      echo "ok $dest"
      return 0
    fi
    echo "hash mismatch $dest (got $got want $sha); re-download"
    rm -f "$dest"
  fi
  echo "fetch $dest"
  curl -L --fail --retry 8 --continue-at - -o "$dest.partial" "$url"
  mv "$dest.partial" "$dest"
  local got
  got="$(sha256sum "$dest" | awk '{print $1}')"
  if [ "$got" != "$sha" ]; then
    echo "hash mismatch after download $dest (got $got want $sha)" >&2
    exit 1
  fi
  echo "ok $dest"
}

HF=https://huggingface.co/datasets
SMOL_REV=f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc
TULU_REV=fe0c7d350c9b4542b8d829a6f1daa1c259f0ba0e
SQUAD_REV=3ffb306f725f7d2ce8394bc1873b24868140c412
DCLM_REV=a3b142c183aebe5af344955ae20836eb34dcf69b

fetch_one \
  "$SFT/smol-smoltalk/data/train-00000-of-00004.parquet" \
  "$HF/HuggingFaceTB/smol-smoltalk/resolve/$SMOL_REV/data/train-00000-of-00004.parquet" \
  498cd4580014f42c40cfde066573564ed28e3e1548a5566715b615f5ea932856

fetch_one \
  "$SFT/smol-smoltalk/data/train-00001-of-00004.parquet" \
  "$HF/HuggingFaceTB/smol-smoltalk/resolve/$SMOL_REV/data/train-00001-of-00004.parquet" \
  5412c03df579fc7dd7d06c7f5249627d99d0621882fb8f2cd5c13582eb2a80f6

fetch_one \
  "$SFT/smol-smoltalk/data/train-00002-of-00004.parquet" \
  "$HF/HuggingFaceTB/smol-smoltalk/resolve/$SMOL_REV/data/train-00002-of-00004.parquet" \
  aefcac6571abf275b4e1629baa01f326c0fe279d35cd0349b7439319923815a7

fetch_one \
  "$SFT/smol-smoltalk/data/train-00003-of-00004.parquet" \
  "$HF/HuggingFaceTB/smol-smoltalk/resolve/$SMOL_REV/data/train-00003-of-00004.parquet" \
  c15edfca888c9924a626dad7b7f4e9f08d8f406f997ac13be46bc9be0a6192db

fetch_one \
  "$SFT/tulu-3-sft-personas-instruction-following/data/train-00000-of-00001.parquet" \
  "$HF/allenai/tulu-3-sft-personas-instruction-following/resolve/$TULU_REV/data/train-00000-of-00001.parquet" \
  19a16c5f1649d367f69899b3cfadbbeb5ffef91f24e20c6617588bdd87cd3e60

fetch_one \
  "$SFT/.downloads/squad-v2/train-00000-of-00001.parquet" \
  "$HF/rajpurkar/squad_v2/resolve/$SQUAD_REV/squad_v2/train-00000-of-00001.parquet" \
  f6da32ffb482ff463ad056477740d1bb284b96a45db3a08bee6a225ca6abf291

fetch_one \
  "$REPLAY/shard_00000081_processed.jsonl.zst" \
  "$HF/mlfoundations/dclm-baseline-1.0/resolve/$DCLM_REV/global-shard_03_of_10/local-shard_0_of_10/shard_00000081_processed.jsonl.zst" \
  f20398f065bb47075ba4165513fc2ccf5fe642bcdf0aefe64f0bf423b1edb889

echo SFT_FETCH_OK
du -sh "$SFT" "$REPLAY"
