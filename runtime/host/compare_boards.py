#!/usr/bin/env python3
"""Run the same prompts on two named boards and print a comparison."""

from __future__ import annotations

import argparse
import json
import sys
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


def ask(board_name: str, artifact: Path | None, prompts: list[str], timeout: float) -> list[dict]:
    board = get_board(board_name)
    port, host = board_connection(board)
    print(f"=== {board.name} ({board.transport} {port or host}) ===", flush=True)
    rows: list[dict] = []
    with P4Device.connect(port, timeout=timeout, reset=False, host=host) as device:
        ensure_ready(device, artifact)
        for prompt in prompts:
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
                requested_tokens=64,
                temperature=0.7,
                top_k=20,
                random_state=1,
                on_chunk=on_chunk,
            )
            print(flush=True)
            rows.append(
                {
                    "board": board.name,
                    "prompt": prompt,
                    "answer": "".join(pieces),
                    "prompt_tokens": result.prompt_tokens,
                    "generated": result.generated_tokens,
                    "elapsed_us": result.elapsed_us,
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-board", default="sun")
    parser.add_argument("--candidate-board", default="mercury")
    parser.add_argument("--baseline-artifact", type=Path)
    parser.add_argument("--candidate-artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/fineweb-vs-original.json"))
    parser.add_argument("--timeout", type=float, default=1200.0)
    args = parser.parse_args()

    baseline = ask(args.baseline_board, args.baseline_artifact, PROMPTS, args.timeout)
    candidate = ask(args.candidate_board, args.candidate_artifact, PROMPTS, args.timeout)
    report = {"baseline": baseline, "candidate": candidate}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}", flush=True)
    print("\n--- side by side ---", flush=True)
    for left, right in zip(baseline, candidate):
        print(f"\nQ: {left['prompt']}")
        print(f"  {args.baseline_board}: {left['answer']!r}")
        print(f"  {args.candidate_board}: {right['answer']!r}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ProtocolError, RuntimeError, TimeoutError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
