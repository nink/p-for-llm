#!/usr/bin/env python3
"""Build the firmware and create the fixed PFor GitHub Release assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

from flash_release import (
    PARTITIONS,
    find_idf_path,
    firmware_images,
    fresh_release_build,
    split_aircraft,
)
from tokenizer_asset import build_asset


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER = PROJECT_DIR / "assets/qwen3.5-english-tokenizer"
FIRMWARE_NAME = "pfor-esp32p4.zip"
MODEL_NAME = "pfor-180m.llmcraft"
CHECKSUM_NAME = "SHA256SUMS"
PACKAGE_NAMES = {
    0x2000: "bootloader.bin",
    0x8000: "partition-table.bin",
    0x10000: "pfor.bin",
    PARTITIONS["tokenizer"][0]: "tokenizer.bin",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_archive_file(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    entry.compress_type = zipfile.ZIP_DEFLATED
    entry.external_attr = 0o100644 << 16
    archive.writestr(entry, data)


def package_release(
    model: Path,
    output_dir: Path,
    release_version: str,
    idf_path: Path | None,
    skip_build: bool,
) -> None:
    split_aircraft(model)
    model_hash = sha256_file(model)
    if not skip_build:
        fresh_release_build(find_idf_path(idf_path))

    built_images = firmware_images()
    payloads = {
        offset: path.read_bytes()
        for offset, path in built_images.items()
    }
    payloads[PARTITIONS["tokenizer"][0]] = build_asset(DEFAULT_TOKENIZER)
    if set(payloads) != set(PACKAGE_NAMES):
        raise RuntimeError("release firmware image layout is incomplete")

    output_dir.mkdir(parents=True, exist_ok=True)
    firmware_target = output_dir / FIRMWARE_NAME
    model_target = output_dir / MODEL_NAME
    checksum_target = output_dir / CHECKSUM_NAME
    for target in (firmware_target, model_target, checksum_target):
        if target.exists():
            raise FileExistsError(f"release asset already exists: {target}")

    images = [
        {
            "offset": f"0x{offset:x}",
            "file": PACKAGE_NAMES[offset],
            "bytes": len(payloads[offset]),
            "sha256": sha256_bytes(payloads[offset]),
        }
        for offset in sorted(payloads)
    ]
    release = {
        "format": "pfor-esp32p4-firmware",
        "format_version": 1,
        "release": release_version,
        "chip": "esp32p4",
        "flash": {"mode": "dio", "size": "16MB", "frequency": "80m"},
        "model": {"file": MODEL_NAME, "sha256": model_hash},
        "images": images,
    }
    release_bytes = (json.dumps(release, indent=2, sort_keys=True) + "\n").encode("ascii")

    with zipfile.ZipFile(firmware_target, "w") as archive:
        write_archive_file(archive, "release.json", release_bytes)
        for offset in sorted(payloads):
            write_archive_file(archive, PACKAGE_NAMES[offset], payloads[offset])
    shutil.copyfile(model, model_target)
    firmware_hash = sha256_file(firmware_target)
    checksum_target.write_text(
        f"{firmware_hash}  {FIRMWARE_NAME}\n{model_hash}  {MODEL_NAME}\n",
        encoding="ascii",
    )
    print(f"firmware={firmware_target}")
    print(f"model={model_target}")
    print(f"checksums={checksum_target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="LLMCRAFT v2 model artifact")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--release-version", required=True)
    parser.add_argument("--idf-path", type=Path)
    parser.add_argument("--skip-build", action="store_true", help="package the existing build-release directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        package_release(
            args.model,
            args.output_dir,
            args.release_version,
            args.idf_path,
            args.skip_build,
        )
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
