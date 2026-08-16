#!/usr/bin/env python3
"""Query (and optionally upload) canada.kpack on a P4. Does not run the neural net."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HOST = Path(__file__).resolve().parents[1] / "runtime" / "host"
PACK = Path(__file__).resolve().parent
if str(HOST) not in sys.path:
    sys.path.insert(0, str(HOST))

from boards import board_connection, get_board  # noqa: E402
from p4 import P4Device  # noqa: E402

DEFAULT_KPACK = PACK / "out" / "canada.kpack"

DEMO = [
    "Which province or territory is Oka in?",
    "Which province or territory is Cherry Point in?",
    "What is the capital of British Columbia?",
    "St. John's is the capital of which province or territory?",
    "What is the boiling point of water?",
]


def ask(device: P4Device, question: str) -> None:
    started = time.perf_counter()
    try:
        answer = device.pack_query(question)
    except Exception as exc:
        print(f"error: {exc}", flush=True)
        return
    ms = (time.perf_counter() - started) * 1000.0
    print(f"pack> {answer}  [{ms:.0f} ms]", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", default="mercury", help="sun or mercury (Mercury only for this firmware)")
    parser.add_argument("--host", help="Ethernet IP (preferred). Default: board eth_ip if set.")
    parser.add_argument("--port", help="UART COM port")
    parser.add_argument("--put", action="store_true", help="upload canada.kpack to the card first")
    parser.add_argument("--demo", action="store_true", help="run the short demo set and exit")
    parser.add_argument("--kpack", type=Path, default=DEFAULT_KPACK)
    parser.add_argument("question", nargs="*", help="one question; omit for an interactive prompt")
    args = parser.parse_args()

    host = args.host
    port = args.port
    if host is None and port is None:
        board = get_board(args.board)
        host = board.eth_ip
        if host is None:
            port, host = board_connection(board)

    print(f"connect host={host} port={port}", flush=True)
    with P4Device.connect(port=port, host=host, timeout=600.0 if args.put else 120.0) as device:
        info = device.handshake()
        print(
            f"board loaded={info.loaded} payload=0x{info.payload_id:08x} status={info.status}",
            flush=True,
        )
        if args.put:
            print(f"uploading {args.kpack} ({args.kpack.stat().st_size} bytes) ...", flush=True)
            device.pack_put(args.kpack)
            print("upload done", flush=True)

        if args.question:
            print(f"\nq> {' '.join(args.question)}", flush=True)
            ask(device, " ".join(args.question))
            return 0

        if args.demo:
            for q in DEMO:
                print(f"\nq> {q}", flush=True)
                ask(device, q)
            return 0

        print("pack retrieve (not the neural net). /exit to leave.", flush=True)
        while True:
            try:
                line = input("pack> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not line or line in {"/exit", "/quit"}:
                return 0
            ask(device, line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
