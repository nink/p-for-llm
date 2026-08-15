#!/usr/bin/env python3
"""One-shot smoke test: load model if needed and ask a short question."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .p4 import P4Device, ProtocolError, ensure_ready, format_chat_prompt
except ImportError:
    from p4 import P4Device, ProtocolError, ensure_ready, format_chat_prompt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="reboot board via RTS (forces PSRAM reload unless SD provides weights)",
    )
    args = parser.parse_args()

    print(f"connecting to {args.port} ...", flush=True)
    with P4Device.connect(args.port, timeout=args.timeout, reset=args.reset) as device:
        print("handshake / load model if needed ...", flush=True)
        layout = ensure_ready(device, args.artifact)
        print(f"ready: {layout.path if layout else 'board payload'}", flush=True)
        device.clear()
        prompt = format_chat_prompt([{"role": "user", "content": args.prompt}])
        print(f"you> {args.prompt}", flush=True)
        print("assistant> ", end="", flush=True)
        result = device.text(
            prompt,
            requested_tokens=args.max_new_tokens,
            temperature=0.7,
            top_k=20,
            random_state=1,
            on_chunk=lambda piece: print(piece, end="", flush=True),
        )
        print(flush=True)
        toks_per_s = (
            result.generated_tokens / (result.elapsed_us / 1e6)
            if result.elapsed_us > 0
            else 0.0
        )
        print(
            f"ok: prompt_tokens={result.prompt_tokens} "
            f"generated={result.generated_tokens} "
            f"{toks_per_s:.1f} tok/s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ProtocolError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
