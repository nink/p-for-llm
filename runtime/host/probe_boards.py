#!/usr/bin/env python3
"""Probe COM ports and Ethernet for PFor LLMHOST5 handshakes."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runtime" / "host"))
from p4 import P4Device, discover_eth_host, ensure_ready  # noqa: E402

PORTS = ["COM3", "COM4", "COM5", "COM6", "COM7", "COM8"]


def probe_uart(port: str) -> str:
    try:
        with P4Device.connect(port, timeout=4.0, reset=False) as device:
            info = device.handshake()
            ip = ""
            if info.eth_ip:
                ip = socket.inet_ntoa(info.eth_ip.to_bytes(4, "big"))
            return (
                f"OK loaded={info.loaded} payload=0x{info.payload_id:08x} "
                f"psram={info.psram_bytes} eth={ip or 'none'} status={info.status}"
            )
    except Exception as exc:
        return f"fail: {type(exc).__name__}: {exc}"


def main() -> int:
    print("=== UART probe (no RTS reset) ===", flush=True)
    for port in PORTS:
        print(f"{port}: {probe_uart(port)}", flush=True)
    print("=== Ethernet TCP 8742 ===", flush=True)
    host = discover_eth_host()
    print(f"scan: {host or 'none'}", flush=True)
    if host:
        with P4Device.connect(host=host, timeout=8.0) as device:
            ensure_ready(device, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
