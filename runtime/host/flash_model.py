#!/usr/bin/env python3
"""Flash only llmcraft Flash regions (manifest + PLE). Firmware/tokenizer stay as-is."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from .flash import MANIFEST_OFFSET, PLE_OFFSET, copy_region, model_flash_regions
except ImportError:
    from flash import MANIFEST_OFFSET, PLE_OFFSET, copy_region, model_flash_regions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=460800)
    args = parser.parse_args()
    manifest_bytes, ple_bytes = model_flash_regions(args.model)
    with tempfile.TemporaryDirectory(prefix="pfor-model-") as directory:
        temporary = Path(directory)
        manifest_path = temporary / "manifest.bin"
        ple_path = temporary / "model-ple.bin"
        copy_region(args.model, 0, manifest_bytes, manifest_path)
        copy_region(args.model, manifest_bytes, ple_bytes, ple_path)
        command = [
            sys.executable,
            "-m",
            "esptool",
            "--chip",
            "esp32p4",
            "--port",
            args.port,
            "--baud",
            str(args.baud),
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
            f"0x{MANIFEST_OFFSET:x}",
            str(manifest_path),
            f"0x{PLE_OFFSET:x}",
            str(ple_path),
        ]
        print(" ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
