"""Build the compact Qwen-derived tokenizer used by PFor."""

from __future__ import annotations

import argparse
from pathlib import Path

from .qwen_bpe import PruneConfig, prune_qwen_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tokenizer", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-size", type=int, default=32_768)
    parser.add_argument("--max-seq-len", type=int, default=1_024)
    args = parser.parse_args()

    result = prune_qwen_tokenizer(
        args.source_tokenizer,
        args.source_config,
        args.output,
        PruneConfig(
            target_vocab_size=args.target_size,
            max_seq_len=args.max_seq_len,
        ),
    )
    print(result)


if __name__ == "__main__":
    main()
