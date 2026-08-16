#!/usr/bin/env python3
"""Request an LLMPRQ05 stage profile from a named board (needs LLMM_DEBUG firmware)."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

try:
    from .boards import board_connection, get_board
    from .p4 import P4Device, ProtocolError, ensure_ready, format_chat_prompt
except ImportError:
    from boards import board_connection, get_board
    from p4 import P4Device, ProtocolError, ensure_ready, format_chat_prompt

STAGES = [
    "embedding",
    "ple_embedding",
    "qkv",
    "attention",
    "attention_output",
    "router",
    "expert",
    "ple_adapter",
    "output_head",
    "sampling",
    "pad",
]
PHASES = ["prefill", "decode"]
PROFILE_PHASES = 2
PROFILE_STAGES = 11


def parse_profile(blob: bytes, cpu_hz: int) -> dict:
    if len(blob) < PROFILE_PHASES * PROFILE_STAGES * 8:
        raise ValueError(f"profile blob is {len(blob)} bytes")
    # Layout matches llmm.zig Profile / main.c llmm_profile_t: stage_cycles first.
    offset = 0

    def take(count: int) -> tuple[int, ...]:
        nonlocal offset
        values = struct.unpack_from(f"<{count}Q", blob, offset)
        offset += count * 8
        return values

    stage_cycles = take(PROFILE_PHASES * PROFILE_STAGES)
    _max = take(PROFILE_PHASES * PROFILE_STAGES)
    stage_calls = take(PROFILE_PHASES * PROFILE_STAGES)
    rows = []
    for phase_index, phase in enumerate(PHASES):
        for stage_index, stage in enumerate(STAGES):
            idx = phase_index * PROFILE_STAGES + stage_index
            cycles = stage_cycles[idx]
            calls = stage_calls[idx]
            ms = (cycles / cpu_hz) * 1000.0 if cpu_hz else 0.0
            per_call_us = (cycles / calls / cpu_hz) * 1e6 if calls and cpu_hz else 0.0
            rows.append(
                {
                    "phase": phase,
                    "stage": stage,
                    "cycles": cycles,
                    "calls": calls,
                    "ms": round(ms, 3),
                    "us_per_call": round(per_call_us, 1),
                }
            )
    decode_expert = next(row for row in rows if row["phase"] == "decode" and row["stage"] == "expert")
    decode_head = next(row for row in rows if row["phase"] == "decode" and row["stage"] == "output_head")
    return {
        "cpu_hz": cpu_hz,
        "bytes": len(blob),
        "stages": rows,
        "decode_expert_ms": decode_expert["ms"],
        "decode_output_head_ms": decode_head["ms"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", default="sun")
    parser.add_argument("--out", type=Path, default=Path("reports/sun-stage-profile.json"))
    parser.add_argument("--tokens", type=int, default=32)
    args = parser.parse_args()
    board = get_board(args.board)
    port, host = board_connection(board)
    prompt = format_chat_prompt([{"role": "user", "content": "What is the capital of France?"}])
    with P4Device.connect(port, timeout=120.0, reset=False, host=host) as device:
        ensure_ready(device, None)
        result = device.text(
            prompt,
            requested_tokens=args.tokens,
            temperature=0.0,
            top_k=20,
            random_state=1,
            profile=True,
        )
    if result.profile is None:
        raise ProtocolError("no profile payload")
    parsed = parse_profile(result.profile, result.cpu_hz)
    parsed["elapsed_us"] = result.elapsed_us
    parsed["generated"] = result.generated_tokens
    parsed["text"] = result.text
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    print(
        f"decode expert={parsed['decode_expert_ms']}ms "
        f"output_head={parsed['decode_output_head_ms']}ms "
        f"board={result.elapsed_us/1000:.1f}ms gen={result.generated_tokens}",
        flush=True,
    )
    for row in parsed["stages"]:
        if row["calls"]:
            print(
                f"  {row['phase']:<8} {row['stage']:<18} {row['ms']:8.3f} ms  "
                f"calls={row['calls']}  {row['us_per_call']:.1f} us/call",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ProtocolError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"error: {error}")
        raise SystemExit(1)
