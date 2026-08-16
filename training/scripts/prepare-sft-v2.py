#!/usr/bin/env python3
"""Convert SQuAD v2 and pack the sft-v2 conversation pool."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

from data.pretraining import discover_data_profile, load_or_prepare_packed_pool
from data.sft import convert_squad_v2, load_or_prepare_sft_pool

ROOT = Path("/home/nink/pfor-work/training")
RAW = ROOT / "data/raw/sft-v2"
REPLAY = ROOT / "data/raw/pretraining/dclm-sft-v2-replay"
TOKENIZER = ROOT / "assets/qwen3.5-english-tokenizer/tokenizer.json"
SQUAD_SRC = RAW / ".downloads/squad-v2/train-00000-of-00001.parquet"
SQUAD_JSONL = RAW / "squad-v2-messages.jsonl"
POOL = Path("/home/nink/pfor-ckpts/sft-v2-pool")


def main() -> None:
    print(f"convert_squad={SQUAD_SRC} -> {SQUAD_JSONL}", flush=True)
    metadata = convert_squad_v2(SQUAD_SRC, SQUAD_JSONL)
    print(json.dumps(metadata, indent=2), flush=True)

    inputs = [
        RAW / "smol-smoltalk/data/train-00000-of-00004.parquet",
        RAW / "smol-smoltalk/data/train-00001-of-00004.parquet",
        RAW / "smol-smoltalk/data/train-00002-of-00004.parquet",
        RAW / "smol-smoltalk/data/train-00003-of-00004.parquet",
        RAW / "tulu-3-sft-personas-instruction-following/data/train-00000-of-00001.parquet",
        SQUAD_JSONL,
    ]
    missing = [path for path in inputs if not path.is_file()]
    if missing:
        raise SystemExit(f"missing SFT inputs: {missing}")

    print(f"packing_sft_pool={POOL}", flush=True)
    pool = load_or_prepare_sft_pool(inputs, TOKENIZER, POOL)
    print(
        "sft_pool_ready "
        f"fingerprint={pool.fingerprint} "
        + " ".join(
            f"{split}.s{bucket}={pool.sequence_count(split, bucket)}"
            for split in ("train", "validation")
            for bucket in (256, 512, 1024)
        ),
        flush=True,
    )

    print(f"packing_dclm_replay={REPLAY}", flush=True)
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    replay = load_or_prepare_packed_pool(
        discover_data_profile(REPLAY),
        tokenizer,
        TOKENIZER,
        1024,
        0.01,
    )
    print(
        "replay_pool_ready "
        f"path={replay.path} fingerprint={replay.fingerprint} "
        f"train_sequences={replay.sequence_count('train')} "
        f"val_sequences={replay.sequence_count('validation')}",
        flush=True,
    )
    print("SFT_PREPARE_OK", flush=True)


if __name__ == "__main__":
    sys.exit(main())
