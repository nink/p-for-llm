#!/usr/bin/env python3
"""After power-on: UART bring-up, then short vs ~8k bench on Ethernet and UART."""

from __future__ import annotations

import socket
import time

from bench_short_vs_8k import ensure_long_context, format_chat_prompt, run_case
from p4 import P4Device, discover_eth_host, ensure_ready

ARTIFACT = None  # SD already loaded; skip local .llmcraft CRC check


def wait_uart_ready(timeout_s: float = 90.0) -> tuple[object, str | None]:
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    reset_tried = False
    while time.monotonic() < deadline:
        remaining = max(5.0, deadline - time.monotonic())
        device = None
        try:
            device = P4Device.connect("COM5", timeout=min(40.0, remaining), reset=False)
            print("handshake ...", flush=True)
            info = device.handshake()
            ip_txt = (
                socket.inet_ntoa(info.eth_ip.to_bytes(4, "big")) if info.eth_ip else "none"
            )
            print(
                f"board: loaded={info.loaded} payload_id=0x{info.payload_id:08x} "
                f"psram={info.psram_bytes} eth_ip={ip_txt}",
                flush=True,
            )
            if info.loaded and info.eth_ip:
                host = socket.inet_ntoa(info.eth_ip.to_bytes(4, "big"))
                return device, host
            if info.loaded and not info.eth_ip:
                device.close()
                print("model loaded, waiting for Ethernet DHCP...", flush=True)
                time.sleep(2.0)
                continue
            device.close()
            print("model not in PSRAM yet — waiting for SD...", flush=True)
            time.sleep(3.0)
        except Exception as exc:
            last_err = exc
            print(f"UART wait: {exc}", flush=True)
            if device is not None:
                try:
                    device.close()
                except Exception:
                    pass
                device = None
            if not reset_tried:
                reset_tried = True
                print("retry with --reset ...", flush=True)
                try:
                    device = P4Device.connect("COM5", timeout=40.0, reset=True)
                    info = device.handshake()
                    ip_txt = (
                        socket.inet_ntoa(info.eth_ip.to_bytes(4, "big"))
                        if info.eth_ip
                        else "none"
                    )
                    print(
                        f"board: loaded={info.loaded} payload_id=0x{info.payload_id:08x} "
                        f"psram={info.psram_bytes} eth_ip={ip_txt}",
                        flush=True,
                    )
                    if info.loaded:
                        host = (
                            socket.inet_ntoa(info.eth_ip.to_bytes(4, "big"))
                            if info.eth_ip
                            else None
                        )
                        return device, host
                    device.close()
                except Exception as reset_exc:
                    last_err = reset_exc
                    print(f"reset handshake failed: {reset_exc}", flush=True)
            time.sleep(2.0)
    raise RuntimeError(f"board not ready after {timeout_s:.0f}s: {last_err}")


def suite(label: str, **conn) -> list[dict]:
    long_text, long_tok = ensure_long_context()
    question = "Why do plant cells need chloroplasts?"
    short_prompt = format_chat_prompt([{"role": "user", "content": question}])
    long_prompt = format_chat_prompt(
        [
            {
                "role": "user",
                "content": (
                    f"CONTEXT:\n{long_text}\n\n"
                    f"QUESTION: {question}\n"
                    "Answer using only CONTEXT."
                ),
            }
        ]
    )
    rows: list[dict] = []
    with P4Device.connect(timeout=120, **conn) as device:
        ensure_ready(device, ARTIFACT)
        rows.append(run_case(device, f"{label} SHORT", short_prompt, None))
        rows.append(run_case(device, f"{label} LONG", long_prompt, long_tok))
    return rows, len(long_prompt.encode()), long_tok


def main() -> int:
    print("=== Ethernet scan (TCP 8742) ===", flush=True)
    host = None
    deadline = time.monotonic() + 45.0
    while time.monotonic() < deadline:
        host = discover_eth_host()
        if host:
            print(f"found {host}:8742", flush=True)
            break
        print("no listener yet, retrying scan ...", flush=True)
        time.sleep(4.0)
    if not host:
        print("no TCP listener — trying UART COM5 (close Arduino Serial Monitor if this fails)", flush=True)
        print("=== UART bring-up ===", flush=True)
        device, host = wait_uart_ready()
        device.close()
        print(f"using ethernet host={host}", flush=True)

    all_rows: list[dict] = []
    wire_b = 0
    long_tok = 0
    if host:
        rows, wire_b, long_tok = suite("ETH", host=host)
        all_rows.extend(rows)
    else:
        rows, wire_b, long_tok = suite("UART", port="COM5", reset=False)
        all_rows.extend(rows)

    print("\n======== SUMMARY ========", flush=True)
    print(f"long wire={wire_b}B src_tok~{long_tok}", flush=True)
    print(
        f"{'case':<28} {'wireB':>6} {'ptok':>5} {'TTFT_ms':>8} "
        f"{'board_s':>8} {'tok/s':>7} {'decode~':>8}",
        flush=True,
    )
    for row in all_rows:
        print(
            f"{row['label']:<28} {row['wire_bytes']:>6} {row['prompt_tok']:>5} "
            f"{row['ttft_ms']:>8.0f} {row['board_s']:>8.2f} "
            f"{row['tps_board']:>7.2f} {row['tps_decode']:>8.2f}",
            flush=True,
        )
        print(f"  -> {row['text'][:140]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
