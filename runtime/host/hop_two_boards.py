#!/usr/bin/env python3
"""Host-orchestrated 1-hop split: Sun layers 0-5, Mercury 6-11, original 180M.

Both boards need hop firmware (LLMHOP05) and the original payload.
Does not flash. Sun Type-C (COM5) is required to app-flash hop firmware.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:
    from .boards import get_board
    from .p4 import P4Device, ProtocolError, ensure_ready
except ImportError:
    from boards import get_board
    from p4 import P4Device, ProtocolError, ensure_ready


def connect_board(name: str, timeout: float) -> P4Device:
    board = get_board(name)
    if not board.eth_ip:
        raise RuntimeError(f"{board.name} has no eth_ip")
    print(f"board {board.name} ethernet {board.eth_ip}:8742", flush=True)
    device = P4Device.connect(None, timeout=timeout, reset=False, host=board.eth_ip)
    ensure_ready(device, None)
    return device


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", type=int, default=1234)
    parser.add_argument("--split", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--out", type=Path, default=Path("reports/two-board-hop.json"))
    args = parser.parse_args()

    sun = None
    mercury = None
    try:
        mercury = connect_board("mercury", args.timeout)
        sun = connect_board("sun", args.timeout)
        mercury.clear()
        sun.clear()
        started = time.perf_counter()
        first = sun.hop(
            args.token,
            position=0,
            layer_begin=0,
            layer_end=args.split,
            score_output=False,
        )
        second = mercury.hop(
            args.token,
            position=0,
            layer_begin=args.split,
            layer_end=12,
            score_output=True,
            hidden=first.hidden,
        )
        host_ms = (time.perf_counter() - started) * 1000.0
        mercury.clear()
        loopback = mercury.split_loopback(args.token, split_layer=args.split)
        report = {
            "token": args.token,
            "split": args.split,
            "sun_layers0_us": first.elapsed_us,
            "mercury_layers_rest_us": second.elapsed_us,
            "host_ms": round(host_ms, 2),
            "next_token": second.next_token,
            "mercury_loopback_token": loopback.full_token,
            "match": second.next_token == loopback.full_token,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        print(f"wrote {args.out}", flush=True)
        return 0 if report["match"] else 2
    except (OSError, ProtocolError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"error: {error}", flush=True)
        print(
            "Sun still needs hop firmware: plug Type-C (CH343 serial 5B90158129 / COM5), "
            "then idf.py -p COM5 app-flash. Do not full-flash; keep original weights.",
            flush=True,
        )
        return 1
    finally:
        if sun is not None:
            sun.close()
        if mercury is not None:
            mercury.close()


if __name__ == "__main__":
    raise SystemExit(main())
