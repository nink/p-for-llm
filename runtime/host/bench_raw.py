#!/usr/bin/env python3
"""Bench TTFT / tok/s for short vs raw-long (on-device compress) prompts."""

from __future__ import annotations

import time
from pathlib import Path

try:
    from .p4 import P4Device, ensure_ready, format_chat_prompt
except ImportError:
    from p4 import P4Device, ensure_ready, format_chat_prompt

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "testdata" / "sample_long_context.md"
REQ_TOKENS = 64


def run_case(device: P4Device, label: str, prompt: str) -> dict:
    device.clear()
    raw_b = len(prompt.encode("utf-8"))
    first_t: float | None = None
    t0 = time.perf_counter()

    def on_chunk(piece: str) -> None:
        nonlocal first_t
        if first_t is None and piece:
            first_t = time.perf_counter()
        print(piece, end="", flush=True)

    print(f"\n=== {label} ===", flush=True)
    print(f"wire_bytes={raw_b}", flush=True)
    print("assistant> ", end="", flush=True)
    result = device.text(
        prompt,
        requested_tokens=REQ_TOKENS,
        temperature=0.3,
        top_k=20,
        random_state=1,
        on_chunk=on_chunk,
    )
    t1 = time.perf_counter()
    print(flush=True)

    wall_s = t1 - t0
    ttft_s = (first_t - t0) if first_t is not None else None
    board_s = result.elapsed_us / 1e6
    gen = result.generated_tokens
    tps_board = gen / board_s if board_s > 0 else 0.0
    tps_decode = None
    if ttft_s is not None and wall_s > ttft_s and gen > 1:
        tps_decode = (gen - 1) / (wall_s - ttft_s)

    ttft_ms = ttft_s * 1000 if ttft_s is not None else float("nan")
    print(
        f"metrics: prompt_tok={result.prompt_tokens} gen={gen} "
        f"TTFT={ttft_ms:.0f}ms wall={wall_s:.2f}s board={board_s:.2f}s "
        f"tok/s_board={tps_board:.2f}"
        + (f" tok/s_decode~{tps_decode:.2f}" if tps_decode else ""),
        flush=True,
    )
    return {
        "label": label,
        "wire_bytes": raw_b,
        "prompt_tok": result.prompt_tokens,
        "gen": gen,
        "ttft_ms": ttft_ms,
        "wall_s": wall_s,
        "board_s": board_s,
        "tps_board": tps_board,
        "tps_decode": tps_decode or 0.0,
        "text": result.text.strip().replace("\n", " / "),
    }


def main() -> int:
    source = SAMPLE.read_text(encoding="utf-8")
    question = "Why do plant cells need chloroplasts?"
    raw_user = (
        f"CONTEXT:\n{source}\n\nQUESTION: {question}\nAnswer using only CONTEXT."
    )
    cases = [
        ("A short (no context)", format_chat_prompt([{"role": "user", "content": question}])),
        ("B raw long (on-device)", format_chat_prompt([{"role": "user", "content": raw_user}])),
        ("C short math", format_chat_prompt([{"role": "user", "content": "What is 2+2?"}])),
    ]

    rows: list[dict] = []
    with P4Device.connect("COM5", timeout=1200, reset=True) as device:
        ensure_ready(device, "pfor-180m.llmcraft")
        for label, prompt in cases:
            rows.append(run_case(device, label, prompt))

    print("\n======== SUMMARY ========", flush=True)
    hdr = f"{'case':<28} {'wireB':>6} {'ptok':>5} {'gen':>4} {'TTFT_ms':>8} {'board_s':>8} {'tok/s':>7} {'decode~':>8}"
    print(hdr, flush=True)
    for row in rows:
        print(
            f"{row['label']:<28} {row['wire_bytes']:>6} {row['prompt_tok']:>5} {row['gen']:>4} "
            f"{row['ttft_ms']:>8.0f} {row['board_s']:>8.2f} {row['tps_board']:>7.2f} "
            f"{row['tps_decode']:>8.2f}",
            flush=True,
        )
        print(f"  -> {row['text'][:120]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
