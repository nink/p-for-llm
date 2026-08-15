#!/usr/bin/env python3
"""Named P4 boards (planets). Sun = first ETH, Mercury = second."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

BOARDS_PATH = Path(__file__).resolve().parent / "boards.json"


@dataclass(frozen=True)
class Board:
    id: str
    name: str
    role: str
    transport: str
    uart_hint: str | None
    usb_serial: str | None
    eth_ip: str | None
    mac: str | None


def load_boards(path: Path = BOARDS_PATH) -> dict[str, Board]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    boards: dict[str, Board] = {}
    for key, row in raw.items():
        transport = str(row.get("transport") or "").strip().casefold()
        if not transport:
            transport = "ethernet" if row.get("eth_ip") else "uart"
        boards[key] = Board(
            id=key,
            name=str(row["name"]),
            role=str(row.get("role") or ""),
            transport=transport,
            uart_hint=row.get("uart_hint"),
            usb_serial=row.get("usb_serial"),
            eth_ip=row.get("eth_ip"),
            mac=row.get("mac"),
        )
    return boards


def resolve_uart_port(board: Board) -> str | None:
    """Match CH343 by USB serial when known so COM numbers can move."""
    serial_id = (board.usb_serial or "").strip()
    if serial_id:
        try:
            from serial.tools import list_ports
        except ImportError:
            list_ports = None
        if list_ports is not None:
            wanted = serial_id.casefold()
            for port in list_ports.comports():
                found = (port.serial_number or "").strip()
                if found and found.casefold() == wanted:
                    return port.device
    return board.uart_hint


def board_connection(board: Board) -> tuple[str | None, str | None]:
    """Return (uart_port, eth_host) for the board's current transport."""
    if board.transport == "ethernet":
        if not board.eth_ip:
            raise RuntimeError(f"{board.name} is set to ethernet but has no eth_ip")
        return None, board.eth_ip
    port = resolve_uart_port(board)
    if not port:
        raise RuntimeError(f"{board.name} is set to uart but no COM port is present")
    return port, None


def get_board(name: str) -> Board:
    key = name.strip().casefold()
    boards = load_boards()
    if key in boards:
        return boards[key]
    for board in boards.values():
        if board.name.casefold() == key:
            return board
    known = ", ".join(sorted(boards))
    raise KeyError(f"unknown board {name!r}; known: {known}")
