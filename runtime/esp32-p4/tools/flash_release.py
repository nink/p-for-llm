#!/usr/bin/env python3
"""Build and provision the complete ESP32-P4 LLMM release image."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import struct
import subprocess
from pathlib import Path

from tokenizer_asset import build_asset


PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PROJECT_DIR.parent
DEFAULT_TOKENIZER = PROJECT_DIR / "assets/qwen3.5-english-tokenizer"
BUILD_DIR = PROJECT_DIR / "build-release"
AIRCRAFT_HEADER = struct.Struct("<8sHH10I5Q")
AIRCRAFT_RECORD = struct.Struct("<HBBHQQQH")
AIRCRAFT_HEADER_BYTES = 96
AIRCRAFT_RECORD_BYTES = 32
FLASH_BYTES = 16 * 1024 * 1024
BAUD_RATE = 460_800
NO_SCALE = 0xFFFF

P4_CONFIG = (32_768, 192, 12, 6, 2, 512, 29, 176, 1_024)
P4_RECORD_COUNT = 2_276
REGION_FLASH = 1
REGION_PSRAM = 2
STORAGE_TERNARY_BASE3 = 1
STORAGE_FP16 = 2
STORAGE_Q8 = 3

PARTITIONS = {
    "factory": (0x10000, 0x100000),
    "tokenizer": (0x110000, 0x145000),
    "manifest": (0x255000, 0x1B000),
    "model_ple": (0x270000, 0xD38000),
}
FIRMWARE_OFFSETS = {0x2000, 0x8000, 0x10000}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_number(value: str) -> int:
    return int(value.strip(), 0)


def validate_partitions() -> None:
    actual: dict[str, tuple[int, int]] = {}
    with (PROJECT_DIR / "partitions.csv").open(newline="", encoding="ascii") as source:
        for row in csv.reader(line for line in source if not line.lstrip().startswith("#")):
            if not row or not row[0].strip():
                continue
            actual[row[0].strip()] = (parse_number(row[3]), parse_number(row[4]))
    for name, expected in PARTITIONS.items():
        if actual.get(name) != expected:
            raise RuntimeError(
                f"partition {name!r} is {actual.get(name)!r}, expected {expected!r}"
            )


def lexical_layer(layer: int) -> int:
    if layer < 2:
        return layer
    return layer + 2 if layer < 10 else layer - 8


def validate_aircraft_records(artifact: bytes, header: tuple[object, ...]) -> None:
    if tuple(header[3:12]) != P4_CONFIG or header[12] != P4_RECORD_COUNT:
        raise ValueError("Aircraft model configuration does not match the P4 runtime")

    flash_bytes = int(header[15])
    psram_bytes = int(header[17])
    records = [
        AIRCRAFT_RECORD.unpack_from(artifact, AIRCRAFT_HEADER_BYTES + index * AIRCRAFT_RECORD_BYTES)
        for index in range(P4_RECORD_COUNT)
    ]
    for index, record in enumerate(records):
        tensor_id, region, storage, scale_index, elements, offset, payload_bytes, reserved = record
        if tensor_id != index or reserved != 0:
            raise ValueError(f"Aircraft tensor record {index} has an invalid identity or reserved field")
        limit = flash_bytes if region == REGION_FLASH else psram_bytes if region == REGION_PSRAM else -1
        if limit < 0 or offset > limit or payload_bytes > limit - offset:
            raise ValueError(f"Aircraft tensor record {index} exceeds its storage region")
        if storage == STORAGE_TERNARY_BASE3:
            expected_bytes = (
                P4_CONFIG[0] * ((P4_CONFIG[2] * P4_CONFIG[7] + 4) // 5)
                if index == 0
                else (elements + 4) // 5
            )
            if scale_index == NO_SCALE or payload_bytes != expected_bytes:
                raise ValueError(f"Aircraft ternary tensor record {index} is malformed")
        elif storage == STORAGE_FP16:
            if scale_index != NO_SCALE or payload_bytes != elements * 2:
                raise ValueError(f"Aircraft FP16 tensor record {index} is malformed")
        elif storage == STORAGE_Q8:
            if scale_index != NO_SCALE or payload_bytes != elements:
                raise ValueError(f"Aircraft Q8 tensor record {index} is malformed")
        else:
            raise ValueError(f"Aircraft tensor record {index} has unsupported storage {storage}")

    def expect(index: int, label: str, region: int, storage: int, elements: int) -> None:
        record = records[index]
        actual = (record[1], record[2], record[4])
        expected = (region, storage, elements)
        if actual != expected:
            raise ValueError(
                f"Aircraft {label} record {index} is {actual}, expected {expected}"
            )

    vocab, width, layers, _, kv_heads, ffn, experts, ple_dim, _ = P4_CONFIG
    expect(0, "PLE table", REGION_FLASH, STORAGE_TERNARY_BASE3, vocab * layers * ple_dim)
    expect(1, "tied embedding", REGION_PSRAM, STORAGE_Q8, vocab * width)
    expect(2, "tied embedding scales", REGION_PSRAM, STORAGE_FP16, vocab)

    attention_shapes = ((kv_heads * 32, width), (width, width), (width, width), (kv_heads * 32, width))
    expert_shapes = ((width, ffn), (ffn, width), (ffn, width))
    for layer in range(layers):
        block_base = 3 + lexical_layer(layer) * 94
        norm_base = 1_132 + lexical_layer(layer) * 95
        for which, (rows, columns) in enumerate(attention_shapes):
            expect(block_base + which, f"layer {layer} attention weight {which}", REGION_PSRAM, STORAGE_TERNARY_BASE3, rows * columns)
            expect(norm_base + which, f"layer {layer} attention norm {which}", REGION_PSRAM, STORAGE_FP16, columns)
        for expert in range(experts):
            for which, (rows, columns) in enumerate(expert_shapes):
                expect(block_base + 4 + expert + which * experts, f"layer {layer} expert {expert} weight {which}", REGION_PSRAM, STORAGE_TERNARY_BASE3, rows * columns)
                expect(norm_base + 4 + expert + which * experts, f"layer {layer} expert {expert} norm {which}", REGION_PSRAM, STORAGE_FP16, columns)
        expect(block_base + 91, f"layer {layer} router weight", REGION_PSRAM, STORAGE_TERNARY_BASE3, experts * width)
        expect(norm_base + 91, f"layer {layer} router norm", REGION_PSRAM, STORAGE_FP16, width)
        expect(block_base + 92, f"layer {layer} PLE gate weight", REGION_PSRAM, STORAGE_TERNARY_BASE3, ple_dim * width)
        expect(block_base + 93, f"layer {layer} PLE projection weight", REGION_PSRAM, STORAGE_TERNARY_BASE3, width * ple_dim)
        expect(norm_base + 92, f"layer {layer} PLE gate norm", REGION_PSRAM, STORAGE_FP16, width)
        expect(norm_base + 93, f"layer {layer} PLE output norm", REGION_PSRAM, STORAGE_FP16, width)
        expect(norm_base + 94, f"layer {layer} PLE projection norm", REGION_PSRAM, STORAGE_FP16, ple_dim)

    expect(1_131, "top-level PLE projection", REGION_PSRAM, STORAGE_TERNARY_BASE3, layers * ple_dim * width)
    expect(2_272, "output norm", REGION_PSRAM, STORAGE_FP16, width)
    expect(2_273, "top-level PLE norm", REGION_PSRAM, STORAGE_FP16, width)
    expect(2_274, "top-level PLE projection norm", REGION_PSRAM, STORAGE_FP16, ple_dim)
    ternary_count = sum(record[2] == STORAGE_TERNARY_BASE3 for record in records)
    expect(2_275, "ternary scale bank", REGION_PSRAM, STORAGE_FP16, ternary_count)
    if sorted(record[3] for record in records if record[2] == STORAGE_TERNARY_BASE3) != list(range(ternary_count)):
        raise ValueError("Aircraft ternary scale indexes are not contiguous")


def split_aircraft(path: Path) -> tuple[bytes, bytes, int]:
    artifact = path.read_bytes()
    if len(artifact) < AIRCRAFT_HEADER_BYTES:
        raise ValueError("Aircraft file is shorter than its header")
    header = AIRCRAFT_HEADER.unpack_from(artifact)
    if header[0] != b"LLMCRAFT" or header[1:3] != (2, AIRCRAFT_HEADER_BYTES):
        raise ValueError("Aircraft magic or format version is unsupported")

    record_count = header[12]
    index_offset = header[13]
    flash_offset = header[14]
    flash_bytes = header[15]
    psram_offset = header[16]
    psram_bytes = header[17]
    manifest_bytes = AIRCRAFT_HEADER_BYTES + record_count * AIRCRAFT_RECORD_BYTES
    if (
        index_offset != AIRCRAFT_HEADER_BYTES
        or manifest_bytes > PARTITIONS["manifest"][1]
        or flash_offset != manifest_bytes
        or flash_bytes != PARTITIONS["model_ple"][1]
        or psram_offset != flash_offset + flash_bytes
        or psram_offset + psram_bytes != len(artifact)
    ):
        raise ValueError("Aircraft payload layout does not match the P4 deployment contract")
    validate_aircraft_records(artifact, header)
    return artifact[:manifest_bytes], artifact[flash_offset:psram_offset], psram_bytes


def find_idf_path(value: Path | None) -> Path:
    candidates = [value] if value is not None else []
    configured = os.environ.get("IDF_PATH")
    if configured:
        candidates.append(Path(configured))
    for candidate in candidates:
        if candidate is not None and (candidate / "export.sh").is_file():
            return candidate.resolve()
    raise RuntimeError("ESP-IDF was not found; pass --idf-path")


def run_in_idf(idf_path: Path, arguments: list[str]) -> None:
    command = (
        'set -euo pipefail\n'
        'source "$1/export.sh" >/dev/null\n'
        'shift\n'
        'exec "$@"\n'
    )
    subprocess.run(
        ["bash", "-c", command, "llmm-release", str(idf_path), *arguments],
        cwd=PROJECT_DIR,
        check=True,
    )


def fresh_release_build(idf_path: Path) -> None:
    if BUILD_DIR.is_symlink():
        raise RuntimeError(f"refusing to replace symlinked build directory: {BUILD_DIR}")
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    run_in_idf(
        idf_path,
        ["idf.py", "-B", str(BUILD_DIR), "build"],
    )


def firmware_images() -> dict[int, Path]:
    description = json.loads((BUILD_DIR / "flasher_args.json").read_text(encoding="utf-8"))
    settings = description.get("flash_settings")
    if settings != {"flash_mode": "dio", "flash_size": "16MB", "flash_freq": "80m"}:
        raise RuntimeError(f"unexpected firmware Flash settings: {settings!r}")
    images = {
        int(offset, 0): BUILD_DIR / relative
        for offset, relative in description["flash_files"].items()
    }
    if set(images) != FIRMWARE_OFFSETS or not all(path.is_file() for path in images.values()):
        raise RuntimeError(f"unexpected firmware image layout: {images!r}")
    return images


def write_assets(tokenizer: bytes, manifest: bytes, ple: bytes) -> dict[int, Path]:
    limits = {
        "tokenizer.bin": PARTITIONS["tokenizer"][1],
        "manifest.bin": PARTITIONS["manifest"][1],
        "model_ple.bin": PARTITIONS["model_ple"][1],
    }
    payloads = {
        "tokenizer.bin": tokenizer,
        "manifest.bin": manifest,
        "model_ple.bin": ple,
    }
    assets_dir = BUILD_DIR / "assets"
    assets_dir.mkdir()
    for name, payload in payloads.items():
        if len(payload) > limits[name]:
            raise RuntimeError(f"{name} is larger than its Flash partition")
        path = assets_dir / name
        path.write_bytes(payload)
        print(f"asset {name}: {len(payload)} bytes sha256={sha256(payload)}", flush=True)
    return {
        PARTITIONS["tokenizer"][0]: assets_dir / "tokenizer.bin",
        PARTITIONS["manifest"][0]: assets_dir / "manifest.bin",
        PARTITIONS["model_ple"][0]: assets_dir / "model_ple.bin",
    }


def provision(idf_path: Path, port: str, images: dict[int, Path]) -> None:
    command = [
        "python",
        "-m",
        "esptool",
        "--chip",
        "esp32p4",
        "--port",
        port,
        "--baud",
        str(BAUD_RATE),
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
    for offset, path in sorted(images.items()):
        if offset + path.stat().st_size > FLASH_BYTES:
            raise RuntimeError(f"image at 0x{offset:x} exceeds physical Flash")
        command.extend((f"0x{offset:x}", str(path)))
    print("provisioning firmware and persistent model assets", flush=True)
    run_in_idf(idf_path, command)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="LLMCRAFT v2 model artifact")
    parser.add_argument("--port", required=True, help="USB serial port exposed by the board")
    parser.add_argument("--idf-path", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER)
    args = parser.parse_args()

    validate_partitions()
    manifest, ple, psram_bytes = split_aircraft(args.artifact)
    tokenizer = build_asset(args.tokenizer_dir)
    print(f"artifact: {args.artifact}", flush=True)
    print(f"USB-only PSRAM payload: {psram_bytes} bytes", flush=True)

    idf_path = find_idf_path(args.idf_path)
    fresh_release_build(idf_path)
    images = firmware_images()
    images.update(write_assets(tokenizer, manifest, ple))
    provision(idf_path, args.port, images)


if __name__ == "__main__":
    main()
