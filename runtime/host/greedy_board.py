#!/usr/bin/env python3
"""Greedy (temp=0) prompts on a named board for GPU reconstruction compare."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .boards import board_connection, get_board
    from .p4 import P4Device, ProtocolError, ensure_ready, format_chat_prompt
except ImportError:
    from boards import board_connection, get_board
    from p4 import P4Device, ProtocolError, ensure_ready, format_chat_prompt

PROMPTS = [
    "What is 2+2?",
    "What is the capital of France?",
    "Explain photosynthesis in one sentence.",
    "Who wrote Romeo and Juliet?",
    "The boiling point of water in Celsius is",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", default="sun")
    parser.add_argument("--out", type=Path, default=Path("reports/sun-greedy-original.json"))
    parser.add_argument("--tokens", type=int, default=64)
    args = parser.parse_args()
    board = get_board(args.board)
    port, host = board_connection(board)
    rows = []
    with P4Device.connect(port, timeout=120.0, reset=False, host=host) as device:
        ensure_ready(device, None)
        for prompt in PROMPTS:
            device.clear()
            chat = format_chat_prompt([{"role": "user", "content": prompt}])
            print(f"you> {prompt}", flush=True)
            print("assistant> ", end="", flush=True)
            pieces: list[str] = []

            def on_chunk(piece: str, bucket: list[str] = pieces) -> None:
                bucket.append(piece)
                print(piece, end="", flush=True)

            result = device.text(
                chat,
                requested_tokens=args.tokens,
                temperature=0.0,
                top_k=20,
                random_state=1,
                on_chunk=on_chunk,
            )
            print(flush=True)
            rows.append(
                {
                    "prompt": prompt,
                    "answer": "".join(pieces),
                    "generated": result.generated_tokens,
                    "elapsed_us": result.elapsed_us,
                }
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ProtocolError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"error: {error}")
        raise SystemExit(1)
