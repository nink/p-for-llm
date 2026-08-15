#!/usr/bin/env python3
"""Interactive terminal chat for the ESP32-P4 runtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .compress import DEFAULT_PACKET_BYTES, DEFAULT_TARGET_RATIO, compress_context
    from .p4 import CONTEXT_LENGTH, P4Device, ProtocolError, TEXT_MAX_BYTES, ensure_ready, format_chat_prompt
except ImportError:
    from compress import DEFAULT_PACKET_BYTES, DEFAULT_TARGET_RATIO, compress_context
    from p4 import CONTEXT_LENGTH, P4Device, ProtocolError, TEXT_MAX_BYTES, ensure_ready, format_chat_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="USB Serial/JTAG port (Espressif), not the CH343 console UART")
    parser.add_argument("--artifact", type=Path, help="release model artifact used when the board is not loaded")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1200.0, help="seconds; PSRAM USB load needs ~10+ minutes")
    parser.add_argument(
        "--compress",
        action="store_true",
        help="enable Phase 1 host context compression for --context-file turns",
    )
    parser.add_argument(
        "--context-file",
        type=Path,
        help="long source document; with --compress, trimmed to ~8:1 before each question",
    )
    parser.add_argument("--max-packet-bytes", type=int, default=DEFAULT_PACKET_BYTES)
    parser.add_argument("--target-ratio", type=float, default=DEFAULT_TARGET_RATIO)
    return parser.parse_args()


def print_help(compress_enabled: bool) -> None:
    print("/help    show this list")
    print("/clear   clear the board session and chat history")
    print("/reload  reload the model payload from --artifact")
    print("/exit    leave the chat")
    if compress_enabled:
        print("/stats   show last compression stats")


def run(args: argparse.Namespace) -> None:
    source_text = ""
    if args.context_file is not None:
        source_text = args.context_file.read_text(encoding="utf-8")
    if args.compress and not source_text:
        raise ValueError("--compress requires --context-file")

    with P4Device.connect(args.port, timeout=args.timeout) as device:
        print("handshake / load model (PSRAM transfer can take several minutes) ...", flush=True)
        layout = ensure_ready(device, args.artifact)
        device.clear()
        if layout is None:
            print("ready: using the model already loaded on the board")
        else:
            print(f"ready: loaded {layout.path}")
        if args.compress:
            print(
                f"compress: on  target≈{args.target_ratio}:1  "
                f"max_packet_bytes={args.max_packet_bytes}  "
                f"source_chars={len(source_text)}",
                flush=True,
            )
        print("type /help for commands")

        continuing = False
        session_tokens = 0
        last_stats = None
        while True:
            try:
                line = input("you> ")
            except EOFError:
                print()
                return
            except KeyboardInterrupt:
                print()
                return
            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                command = line.lower()
                if command == "/help":
                    print_help(args.compress)
                elif command == "/clear":
                    device.clear()
                    continuing = False
                    session_tokens = 0
                    print("session cleared")
                elif command == "/reload":
                    if layout is None:
                        print("reload requires --artifact")
                        continue
                    device.reload(layout)
                    continuing = False
                    session_tokens = 0
                    print("model reloaded and session cleared")
                elif command == "/stats":
                    if last_stats is None:
                        print("no compression stats yet")
                    else:
                        print(
                            f"ratio~{last_stats.ratio_tokens:.2f}:1 "
                            f"src~{last_stats.source_tokens_est} "
                            f"pkt~{last_stats.packet_tokens_est} "
                            f"bytes={last_stats.packet_bytes} "
                            f"kept={last_stats.kept_sentences} "
                            f"dropped={last_stats.dropped_sentences}"
                        )
                elif command in {"/exit", "/quit"}:
                    return
                else:
                    print(f"unknown command: {line}")
                continue

            user_content = line
            if args.compress:
                result = compress_context(
                    source_text,
                    line,
                    max_packet_bytes=args.max_packet_bytes,
                    target_ratio=args.target_ratio,
                )
                last_stats = result
                user_content = result.packet
                print(
                    f"[compress ratio~{result.ratio_tokens:.2f}:1 "
                    f"{result.source_tokens_est}->{result.packet_tokens_est} tok_est "
                    f"{result.packet_bytes}B]",
                    flush=True,
                )
                # Compressed turns are self-contained; reset KV session.
                if continuing:
                    device.clear()
                    continuing = False
                    session_tokens = 0

            if continuing:
                prompt = f"<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
            else:
                prompt = format_chat_prompt([{"role": "user", "content": user_content}])

            prompt_bytes = len(prompt.encode("utf-8"))
            if prompt_bytes > TEXT_MAX_BYTES:
                print(
                    f"error: prompt is {prompt_bytes} bytes; wire limit is {TEXT_MAX_BYTES}. "
                    "Lower --max-packet-bytes.",
                    flush=True,
                )
                continue

            if session_tokens + prompt_bytes + args.max_new_tokens > CONTEXT_LENGTH:
                device.clear()
                continuing = False
                session_tokens = 0
                prompt = format_chat_prompt([{"role": "user", "content": user_content}])

            print("assistant> ", end="", flush=True)
            try:
                result = device.text(
                    prompt,
                    requested_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    random_state=args.seed,
                    on_chunk=lambda piece: print(piece, end="", flush=True),
                )
            except (ProtocolError, RuntimeError, TimeoutError, ValueError) as error:
                print(f"\nerror: {error}")
                continue
            print()
            if result.session_evicted:
                session_tokens = 0
            session_tokens += result.prompt_tokens + result.generated_tokens
            continuing = True


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except (OSError, ProtocolError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
