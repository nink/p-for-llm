#!/usr/bin/env python3
"""Interactive terminal chat for the ESP32-P4 runtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .p4 import CONTEXT_LENGTH, P4Device, ProtocolError, ensure_ready, format_chat_prompt
except ImportError:
    from p4 import CONTEXT_LENGTH, P4Device, ProtocolError, ensure_ready, format_chat_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="USB serial port exposed by the board")
    parser.add_argument("--artifact", type=Path, help="release model artifact used when the board is not loaded")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def print_help() -> None:
    print("/help    show this list")
    print("/clear   clear the board session and chat history")
    print("/reload  reload the model payload from --artifact")
    print("/exit    leave the chat")


def run(args: argparse.Namespace) -> None:
    with P4Device.connect(args.port, timeout=args.timeout) as device:
        layout = ensure_ready(device, args.artifact)
        device.clear()
        if layout is None:
            print("ready: using the model already loaded on the board")
        else:
            print(f"ready: loaded {layout.path}")
        print("type /help for commands")

        continuing = False
        session_tokens = 0
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
                    print_help()
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
                elif command in {"/exit", "/quit"}:
                    return
                else:
                    print(f"unknown command: {line}")
                continue

            if continuing:
                prompt = f"<|im_end|>\n<|im_start|>user\n{line}<|im_end|>\n<|im_start|>assistant\n"
            else:
                prompt = format_chat_prompt([{"role": "user", "content": line}])

            if session_tokens + len(prompt.encode("utf-8")) + args.max_new_tokens > CONTEXT_LENGTH:
                device.clear()
                continuing = False
                session_tokens = 0
                prompt = format_chat_prompt([{"role": "user", "content": line}])

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
