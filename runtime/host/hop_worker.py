#!/usr/bin/env python3
"""Mercury-only 1-hop layer-split smoke: loopback match + timed worker hop.

Does not talk to Sun. Prefers Mercury Ethernet when eth_ip is set.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

try:
    from .boards import get_board, resolve_uart_port
    from .p4 import P4Device, ProtocolError, ensure_ready
except ImportError:
    from boards import get_board, resolve_uart_port
    from p4 import P4Device, ProtocolError, ensure_ready


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", default="mercury")
    parser.add_argument("--uart", action="store_true", help="force CH343 UART instead of Ethernet")
    parser.add_argument("--token", type=int, default=1234)
    parser.add_argument("--split", type=int, default=6)
    parser.add_argument("--hops", type=int, default=20)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/mercury-layer-hop.json"),
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def connect(args: argparse.Namespace) -> tuple[P4Device, str]:
    board = get_board(args.board)
    host = None
    port = None
    if args.uart or not board.eth_ip:
        port = resolve_uart_port(board)
        if not port:
            raise RuntimeError(f"{board.name} has no UART port")
        via = f"uart {port}"
    else:
        host = board.eth_ip
        via = f"ethernet {host}:8742"
    print(f"board {board.name} ({board.role}) via {via}", flush=True)
    device = P4Device.connect(port, timeout=args.timeout, reset=False, host=host)
    return device, via


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def main() -> int:
    args = parse_args()
    device, via = connect(args)
    report: dict = {"board": args.board, "via": via, "token": args.token, "split": args.split}
    try:
        ensure_ready(device, None)
        device.clear()
        print(f"loopback token={args.token} split={args.split} ...", flush=True)
        loopback = device.split_loopback(args.token, split_layer=args.split)
        report["loopback"] = {
            "tokens_match": loopback.tokens_match,
            "split_token": loopback.split_token,
            "full_token": loopback.full_token,
            "max_abs_diff": loopback.max_abs_diff,
            "elapsed_us": loopback.elapsed_us,
        }
        print(
            f"loopback match={loopback.tokens_match} "
            f"split={loopback.split_token} full={loopback.full_token} "
            f"max_abs_diff={loopback.max_abs_diff:.3e} "
            f"board_us={loopback.elapsed_us}",
            flush=True,
        )

        device.clear()
        print("protocol hop 0-6 then 6-12 ...", flush=True)
        host_started = time.perf_counter()
        first = device.hop(
            args.token,
            position=0,
            layer_begin=0,
            layer_end=args.split,
            score_output=False,
        )
        second = device.hop(
            args.token,
            position=0,
            layer_begin=args.split,
            layer_end=12,
            score_output=True,
            hidden=first.hidden,
        )
        host_ms = (time.perf_counter() - host_started) * 1000.0
        hop_match = second.next_token == loopback.full_token
        report["protocol_split"] = {
            "first_board_us": first.elapsed_us,
            "second_board_us": second.elapsed_us,
            "host_ms": round(host_ms, 2),
            "next_token": second.next_token,
            "matches_loopback": hop_match,
        }
        print(
            f"hop tokens={second.next_token} match_loopback={hop_match} "
            f"layers0-{args.split}={first.elapsed_us}us "
            f"layers{args.split}-12={second.elapsed_us}us "
            f"host_rtt_pair={host_ms:.1f}ms",
            flush=True,
        )

        times_ms: list[float] = []
        board_us: list[int] = []
        for _ in range(args.hops):
            started = time.perf_counter()
            hop = device.hop(
                args.token,
                position=0,
                layer_begin=args.split,
                layer_end=12,
                score_output=True,
                hidden=first.hidden,
            )
            times_ms.append((time.perf_counter() - started) * 1000.0)
            board_us.append(hop.elapsed_us)
        report["worker_hop"] = {
            "count": args.hops,
            "host_ms_avg": round(statistics.fmean(times_ms), 2),
            "host_ms_p50": round(percentile(times_ms, 50), 2),
            "host_ms_p99": round(percentile(times_ms, 99), 2),
            "board_ms_avg": round(statistics.fmean(board_us) / 1000.0, 2),
        }
        print(
            f"worker hop x{args.hops}: host p50={report['worker_hop']['host_ms_p50']}ms "
            f"p99={report['worker_hop']['host_ms_p99']}ms "
            f"board_avg={report['worker_hop']['board_ms_avg']}ms",
            flush=True,
        )

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}", flush=True)
        if not loopback.tokens_match or not hop_match:
            return 2
        return 0
    except (OSError, ProtocolError, RuntimeError, ValueError, TimeoutError) as error:
        print(f"error: {error}", flush=True)
        return 1
    finally:
        device.close()


if __name__ == "__main__":
    raise SystemExit(main())
