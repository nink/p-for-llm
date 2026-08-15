#!/usr/bin/env python3
"""Prepare the PSRAM model blob for an SD card (Waveshare ESP32-P4-ETH).

Writes pfor-psram.bin with header:
  magic P4SD | version=1 | payload_bytes | crc32 | raw PSRAM payload

Copy the file to the root of a FAT32 microSD card, then boot firmware that
loads it into PSRAM (seconds instead of a ~10 minute UART transfer).
"""

from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

try:
    from .p4 import validate_artifact
except ImportError:
    from p4 import validate_artifact

MAGIC = b"P4SD"
VERSION = 1
HEADER = struct.Struct("<4sIII")


def write_sd_payload(artifact: Path, output: Path) -> tuple[int, int]:
    layout = validate_artifact(artifact)
    payload = bytearray()
    with artifact.open("rb") as source:
        source.seek(layout.psram_offset)
        remaining = layout.psram_bytes
        while remaining:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("artifact ended while reading PSRAM payload")
            payload.extend(chunk)
            remaining -= len(chunk)
    if len(payload) != layout.psram_bytes:
        raise ValueError("PSRAM payload size mismatch")
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as target:
        target.write(HEADER.pack(MAGIC, VERSION, layout.psram_bytes, crc))
        target.write(payload)
    return layout.psram_bytes, crc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True, help="pfor-180m.llmcraft")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("pfor-psram.bin"),
        help="output path (copy to SD card root as pfor-psram.bin)",
    )
    args = parser.parse_args()
    size, crc = write_sd_payload(args.artifact, args.out)
    print(f"wrote {args.out} bytes={size} crc32=0x{crc:08x}")
    print("Copy pfor-psram.bin to the root of a FAT32 microSD, insert in the board, power cycle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
