#!/usr/bin/env bash
# Build ESP32-P4 firmware with UART host transport (CH343 boards).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ZIG_VER="${ZIG_VER:-0.14.1}"
IDF_IMAGE="${IDF_IMAGE:-espressif/idf:v6.0.1}"

docker run --rm \
  -v "${ROOT}:/project" \
  -w /project \
  -e ZIG_VER="${ZIG_VER}" \
  "${IDF_IMAGE}" \
  bash -lc '
    set -euo pipefail
    if ! command -v zig >/dev/null 2>&1; then
      apt-get update -qq
      apt-get install -y -qq curl xz-utils >/dev/null
      curl -fsSL "https://ziglang.org/download/${ZIG_VER}/zig-linux-x86_64-${ZIG_VER}.tar.xz" -o /tmp/zig.tar.xz
      tar -xJf /tmp/zig.tar.xz -C /opt
      export PATH="/opt/zig-linux-x86_64-${ZIG_VER}:$PATH"
    fi
    zig version
    . /opt/esp/export-esp.sh 2>/dev/null || . "${IDF_PATH}/export.sh"
    rm -rf build sdkconfig
    idf.py set-target esp32p4
    idf.py build
    echo "BUILT=$(pwd)/build/p_for_llm_esp32p4.bin"
  '
