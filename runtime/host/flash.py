#!/usr/bin/env python3
"""Flash a PFor firmware package and model artifact to ESP32-P4."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

try:
    from .p4 import AIRCRAFT_HEADER, AIRCRAFT_HEADER_BYTES, validate_artifact
except ImportError:
    from p4 import AIRCRAFT_HEADER, AIRCRAFT_HEADER_BYTES, validate_artifact


PACKAGE_FORMAT = "pfor-esp32p4-firmware"
PACKAGE_VERSION = 1
FLASH_BYTES = 16 * 1024 * 1024
BAUD_RATE = 460_800
MANIFEST_OFFSET = 0x255000
MANIFEST_LIMIT = 0x1B000
PLE_OFFSET = 0x270000
PLE_BYTES = 0xD38000
EXPECTED_IMAGES = {
    0x2000: "bootloader.bin",
    0x8000: "partition-table.bin",
    0x10000: "pfor.bin",
    0x110000: "tokenizer.bin",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_firmware_package(path: Path, model: Path) -> dict[int, bytes]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("firmware package contains duplicate files")
        damaged = archive.testzip()
        if damaged is not None:
            raise ValueError(f"firmware package contains a damaged file: {damaged}")
        try:
            release = json.loads(archive.read("release.json"))
        except KeyError as error:
            raise ValueError("firmware package has no release.json") from error

        if (
            release.get("format") != PACKAGE_FORMAT
            or release.get("format_version") != PACKAGE_VERSION
            or release.get("chip") != "esp32p4"
        ):
            raise ValueError("firmware package format is unsupported")
        if release.get("flash") != {"mode": "dio", "size": "16MB", "frequency": "80m"}:
            raise ValueError("firmware package Flash settings are unsupported")

        model_record = release.get("model")
        if not isinstance(model_record, dict) or model_record.get("file") != "pfor-180m.llmcraft":
            raise ValueError("firmware package model record is invalid")
        expected_model_hash = model_record.get("sha256")
        if not isinstance(expected_model_hash, str) or sha256_file(model) != expected_model_hash:
            raise ValueError("model artifact does not match the firmware release")

        records = release.get("images")
        if not isinstance(records, list):
            raise ValueError("firmware package image list is invalid")
        images: dict[int, bytes] = {}
        expected_names = {"release.json", *EXPECTED_IMAGES.values()}
        if set(names) != expected_names:
            raise ValueError("firmware package contains an unexpected file set")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("firmware package image record is invalid")
            try:
                offset = int(record["offset"], 0)
                name = record["file"]
                expected_bytes = record["bytes"]
                expected_hash = record["sha256"]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("firmware package image record is invalid") from error
            if EXPECTED_IMAGES.get(offset) != name or offset in images:
                raise ValueError("firmware package image layout is invalid")
            data = archive.read(name)
            if expected_bytes != len(data) or expected_hash != sha256_bytes(data):
                raise ValueError(f"firmware package image verification failed: {name}")
            if offset + len(data) > FLASH_BYTES:
                raise ValueError(f"firmware package image exceeds Flash: {name}")
            images[offset] = data
        if set(images) != set(EXPECTED_IMAGES):
            raise ValueError("firmware package image layout is incomplete")
        return images


def copy_region(source_path: Path, offset: int, size: int, output_path: Path) -> None:
    remaining = size
    with source_path.open("rb") as source, output_path.open("wb") as output:
        source.seek(offset)
        while remaining:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("model artifact ended before its declared payload")
            output.write(chunk)
            remaining -= len(chunk)


def model_flash_regions(model: Path) -> tuple[int, int]:
    validate_artifact(model)
    with model.open("rb") as source:
        header_bytes = source.read(AIRCRAFT_HEADER_BYTES)
    header = AIRCRAFT_HEADER.unpack_from(header_bytes)
    manifest_bytes = int(header[14])
    ple_bytes = int(header[15])
    if manifest_bytes > MANIFEST_LIMIT or ple_bytes != PLE_BYTES:
        raise ValueError("model artifact Flash regions do not match the PFor partition layout")
    return manifest_bytes, ple_bytes


def flash(firmware: Path, model: Path, port: str, baud: int) -> None:
    images = load_firmware_package(firmware, model)
    manifest_bytes, ple_bytes = model_flash_regions(model)
    if importlib.util.find_spec("esptool") is None:
        raise RuntimeError("esptool is not installed; run: python3 -m pip install esptool")

    with tempfile.TemporaryDirectory(prefix="pfor-flash-") as directory:
        temporary = Path(directory)
        paths: dict[int, Path] = {}
        for offset, data in images.items():
            path = temporary / EXPECTED_IMAGES[offset]
            path.write_bytes(data)
            paths[offset] = path

        manifest_path = temporary / "manifest.bin"
        ple_path = temporary / "model-ple.bin"
        copy_region(model, 0, manifest_bytes, manifest_path)
        copy_region(model, manifest_bytes, ple_bytes, ple_path)
        paths[MANIFEST_OFFSET] = manifest_path
        paths[PLE_OFFSET] = ple_path

        command = [
            sys.executable,
            "-m",
            "esptool",
            "--chip",
            "esp32p4",
            "--port",
            port,
            "--baud",
            str(baud),
            "--before",
            "default-reset",
            "--after",
            "hard-reset",
            "write-flash",
            "--flash-mode",
            "dio",
            "--flash-freq",
            "80m",
            "--flash-size",
            "16MB",
        ]
        for offset, path in sorted(paths.items()):
            command.extend((f"0x{offset:x}", str(path)))
        subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firmware", type=Path, required=True, help="pfor-esp32p4.zip")
    parser.add_argument("--model", type=Path, required=True, help="pfor-180m.llmcraft")
    parser.add_argument("--port", required=True, help="CH343 UART port (same COM as chat.py, e.g. COM5)")
    parser.add_argument("--baud", type=int, default=BAUD_RATE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        flash(args.firmware, args.model, args.port, args.baud)
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
