#!/usr/bin/env python3
"""Interactive terminal chat for the ESP32-P4 runtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .p4 import (
        CONTEXT_LENGTH,
        RAW_TEXT_MAX_BYTES,
        COMPRESS_FITTED_MAX_BYTES,
        TEXT_MAX_BYTES,
        P4Device,
        ProtocolError,
        ensure_ready,
        format_chat_prompt,
    )
except ImportError:
    from p4 import (
        CONTEXT_LENGTH,
        RAW_TEXT_MAX_BYTES,
        COMPRESS_FITTED_MAX_BYTES,
        TEXT_MAX_BYTES,
        P4Device,
        ProtocolError,
        ensure_ready,
        format_chat_prompt,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="CH343 UART port (same COM as flash.py, e.g. COM5)")
    parser.add_argument("--artifact", type=Path, help="release model artifact used when the board is not loaded")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1200.0, help="seconds; UART PSRAM load needs ~10+ minutes if SD missing")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="pulse RTS to reboot the board (clears PSRAM; avoid unless needed)",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="send raw long CONTEXT+question; board compresses on-device before inference",
    )
    parser.add_argument(
        "--context-file",
        type=Path,
        help="long source document; with --compress, sent raw (board trims to ~8:1)",
    )
    return parser.parse_args()


def print_help(compress_enabled: bool) -> None:
    print("/help    show this list")
    print("/clear   clear the board session and chat history")
    print("/reload  reload the model payload from --artifact")
    print("/exit    leave the chat")
    if compress_enabled:
        print("/stats   show last raw-prompt size sent to the board")


def run(args: argparse.Namespace) -> None:
    source_text = ""
    if args.context_file is not None:
        source_text = args.context_file.read_text(encoding="utf-8")
    if args.compress and not source_text:
        raise ValueError("--compress requires --context-file")

    with P4Device.connect(args.port, timeout=args.timeout, reset=args.reset) as device:
        layout = ensure_ready(device, args.artifact)
        device.clear()
        if layout is None:
            print("ready: using board payload")
        else:
            print(f"ready: artifact {layout.path.name}")
        if args.compress:
            print(
                f"compress: on-device  wire_max={RAW_TEXT_MAX_BYTES}B  "
                f"fitted_max={COMPRESS_FITTED_MAX_BYTES}B  bypass<{TEXT_MAX_BYTES}B  "
                f"source_chars={len(source_text)}",
                flush=True,
            )
        print("type /help for commands")

        continuing = False
        session_tokens = 0
        last_raw_bytes = 0
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
                    if last_raw_bytes <= 0:
                        print("no compression stats yet")
                    else:
                        print(
                            f"last raw prompt {last_raw_bytes}B "
                            f"(board fits to <= {COMPRESS_FITTED_MAX_BYTES}B on-device)"
                        )
                elif command in {"/exit", "/quit"}:
                    return
                else:
                    print(f"unknown command: {line}")
                continue

            user_content = line
            if args.compress:
                user_content = (
                    f"CONTEXT:\n{source_text}\n\n"
                    f"QUESTION: {line}\n"
                    f"Answer using only CONTEXT."
                )
                # Self-contained long turns; reset KV session.
                if continuing:
                    device.clear()
                    continuing = False
                    session_tokens = 0

            if continuing:
                prompt = f"<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
            else:
                prompt = format_chat_prompt([{"role": "user", "content": user_content}])

            prompt_bytes = len(prompt.encode("utf-8"))
            if prompt_bytes > RAW_TEXT_MAX_BYTES:
                print(
                    f"error: prompt is {prompt_bytes} bytes; wire limit is {RAW_TEXT_MAX_BYTES}.",
                    flush=True,
                )
                continue

            if args.compress:
                last_raw_bytes = prompt_bytes
                print(
                    f"[raw {prompt_bytes}B → board compress ≤{COMPRESS_FITTED_MAX_BYTES}B]",
                    flush=True,
                )
            elif prompt_bytes > TEXT_MAX_BYTES:
                print(
                    f"error: prompt is {prompt_bytes} bytes; without --compress "
                    f"limit is {TEXT_MAX_BYTES} (or enable --compress for on-device trim).",
                    flush=True,
                )
                continue

            if session_tokens + min(prompt_bytes, TEXT_MAX_BYTES) + args.max_new_tokens > CONTEXT_LENGTH:
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
