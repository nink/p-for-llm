#!/usr/bin/env python3
"""Standard-library host client for the ESP32-P4 runtime."""

from __future__ import annotations

import math
import os
import select
import struct
import time
import zlib
import codecs
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

try:
    import termios
except ImportError:  # pragma: no cover - this client currently targets POSIX USB serial ports.
    termios = None


BAUD_RATE = 460_800
READ_CHUNK_BYTES = 16 * 1024
TEXT_MAX_BYTES = 1_024
TOP_K_MAX = 64
CONTEXT_LENGTH = 1_024

P4_CONFIG = (32_768, 192, 12, 6, 2, 512, 29, 176, 1_024)
P4_RECORD_COUNT = 2_276
AIRCRAFT_HEADER = struct.Struct("<8sHH10I5Q")
AIRCRAFT_RECORD = struct.Struct("<HBBHQQQH")
AIRCRAFT_HEADER_BYTES = 96
AIRCRAFT_RECORD_BYTES = 32
FLASH_BYTES = 0xD38000
MANIFEST_PARTITION_BYTES = 0x1B000
NO_SCALE = 0xFFFF


class ProtocolError(RuntimeError):
    """The board returned an invalid frame or a runtime error status."""


@dataclass(frozen=True)
class DeviceInfo:
    status: int
    psram_bytes: int
    loaded: bool
    payload_id: int
    session_tokens: int


@dataclass(frozen=True)
class LoadResult:
    status: int
    psram_bytes: int
    payload_id: int


@dataclass(frozen=True)
class TextResult:
    text: str
    generated_tokens: int
    checksum: int
    elapsed_us: int
    prompt_tokens: int
    session_evicted: bool


@dataclass(frozen=True)
class ArtifactLayout:
    path: Path
    psram_offset: int
    psram_bytes: int
    file_bytes: int


class SerialTransport:
    """USB serial transport for POSIX (termios) or Windows (pyserial)."""

    def __init__(self, port: str, timeout: float = 30.0) -> None:
        self.port = port
        self.timeout = timeout
        self.fd: int | None = None
        self._serial = None

    def open(self) -> "SerialTransport":
        if os.name == "nt":
            try:
                import serial
            except ImportError as error:
                raise RuntimeError(
                    "Windows host client requires pyserial (python -m pip install pyserial)"
                ) from error
            try:
                self._serial = serial.Serial(
                    port=self.port,
                    baudrate=BAUD_RATE,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=0.2,
                    write_timeout=5.0,
                    dsrdtr=False,
                    rtscts=False,
                )
                # Opening COM ports on Windows often asserts DTR/RTS and holds ESP in reset.
                self._serial.dtr = False
                self._serial.rts = False
                time.sleep(0.05)
                # Pulse RTS to reboot into application firmware, then release.
                self._serial.rts = True
                time.sleep(0.05)
                self._serial.rts = False
                # ESP32-P4 app needs ~2–3s after reset before LLMHOST is ready.
                time.sleep(3.0)
                self._serial.reset_input_buffer()
                self._serial.reset_output_buffer()
                time.sleep(0.2)
            except Exception:
                self.close()
                raise
            return self

        if termios is None:
            raise RuntimeError("the host client requires a POSIX USB serial port")
        flags = os.O_RDWR | os.O_NOCTTY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        try:
            self.fd = os.open(self.port, flags)
            attributes = termios.tcgetattr(self.fd)
            attributes[0] = 0
            attributes[1] = 0
            attributes[2] = termios.CS8 | termios.CLOCAL | termios.CREAD
            attributes[3] = 0
            speed = getattr(termios, "B460800", termios.B115200)
            attributes[4] = speed
            attributes[5] = speed
            attributes[6][termios.VMIN] = 0
            attributes[6][termios.VTIME] = 0
            termios.tcsetattr(self.fd, termios.TCSANOW, attributes)
            termios.tcflush(self.fd, termios.TCIOFLUSH)
        except OSError:
            self.close()
            raise
        return self

    def _require_open(self) -> int:
        if self._serial is not None:
            return -1
        if self.fd is None:
            raise RuntimeError("USB serial port is not open")
        return self.fd

    def _wait(self, writable: bool, deadline: float) -> None:
        if self._serial is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out communicating with {self.port}")
            return
        fd = self._require_open()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"timed out communicating with {self.port}")
        readable, writable_fds, _ = select.select(
            [] if writable else [fd],
            [fd] if writable else [],
            [],
            remaining,
        )
        if not (writable_fds if writable else readable):
            raise TimeoutError(f"timed out communicating with {self.port}")

    def read_exact(self, size: int, timeout: float | None = None) -> bytes:
        if size < 0:
            raise ValueError("read size cannot be negative")
        if size == 0:
            return b""
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        result = bytearray()
        if self._serial is not None:
            while len(result) < size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"timed out communicating with {self.port}")
                chunk = self._serial.read(size - len(result))
                if chunk:
                    result.extend(chunk)
                    continue
                time.sleep(min(0.01, remaining))
            return bytes(result)

        fd = self._require_open()
        while len(result) < size:
            self._wait(False, deadline)
            chunk = os.read(fd, size - len(result))
            if not chunk:
                raise ProtocolError(f"USB serial port {self.port!r} closed")
            result.extend(chunk)
        return bytes(result)

    def write_all(self, data: bytes | bytearray | memoryview, timeout: float | None = None) -> None:
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        view = memoryview(data)
        if self._serial is not None:
            while view:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"timed out communicating with {self.port}")
                written = self._serial.write(view)
                if written is None or written <= 0:
                    time.sleep(min(0.01, remaining))
                    continue
                view = view[written:]
            return

        fd = self._require_open()
        while view:
            self._wait(True, deadline)
            written = os.write(fd, view)
            if written <= 0:
                raise ProtocolError(f"USB serial port {self.port!r} closed")
            view = view[written:]

    def close(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            finally:
                self._serial = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self) -> "SerialTransport":
        return self.open()

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()


def _check_status(operation: str, status: int) -> None:
    if status != 0:
        raise ProtocolError(f"{operation} failed with status {status}")


def validate_artifact(path: str | Path) -> ArtifactLayout:
    """Validate the deployment header and return the PSRAM payload location."""

    artifact_path = Path(path)
    file_bytes = artifact_path.stat().st_size
    if file_bytes < AIRCRAFT_HEADER_BYTES:
        raise ValueError("model artifact is shorter than its header")
    with artifact_path.open("rb") as source:
        header_bytes = source.read(AIRCRAFT_HEADER_BYTES)
        header = AIRCRAFT_HEADER.unpack_from(header_bytes)
        if header[0] != b"LLMCRAFT" or header[1:3] != (2, AIRCRAFT_HEADER_BYTES):
            raise ValueError("model artifact magic or format version is unsupported")
        if tuple(header[3:12]) != P4_CONFIG or header[12] != P4_RECORD_COUNT:
            raise ValueError("model artifact configuration does not match the P4 runtime")

        index_offset = header[13]
        flash_offset = header[14]
        flash_bytes = header[15]
        psram_offset = header[16]
        psram_bytes = header[17]
        manifest_bytes = AIRCRAFT_HEADER_BYTES + P4_RECORD_COUNT * AIRCRAFT_RECORD_BYTES
        if (
            index_offset != AIRCRAFT_HEADER_BYTES
            or manifest_bytes > MANIFEST_PARTITION_BYTES
            or flash_offset != manifest_bytes
            or flash_bytes != FLASH_BYTES
            or psram_offset != flash_offset + flash_bytes
            or psram_offset + psram_bytes != file_bytes
        ):
            raise ValueError("model artifact payload layout does not match the P4 runtime")

        source.seek(index_offset)
        for index in range(P4_RECORD_COUNT):
            record = AIRCRAFT_RECORD.unpack(source.read(AIRCRAFT_RECORD_BYTES))
            tensor_id, region, storage, scale_index, elements, offset, payload_bytes, reserved = record
            if tensor_id != index or reserved != 0:
                raise ValueError(f"model artifact tensor record {index} is malformed")
            limit = flash_bytes if region == 1 else psram_bytes if region == 2 else -1
            if limit < 0 or offset > limit or payload_bytes > limit - offset:
                raise ValueError(f"model artifact tensor record {index} exceeds its storage region")
            if storage == 1:
                expected_bytes = (
                    P4_CONFIG[0] * ((P4_CONFIG[2] * P4_CONFIG[7] + 4) // 5)
                    if index == 0
                    else (elements + 4) // 5
                )
                if scale_index == NO_SCALE or payload_bytes != expected_bytes:
                    raise ValueError(f"model artifact ternary tensor record {index} is malformed")
            elif storage == 2:
                if scale_index != NO_SCALE or payload_bytes != elements * 2:
                    raise ValueError(f"model artifact FP16 tensor record {index} is malformed")
            elif storage == 3:
                if scale_index != NO_SCALE or payload_bytes != elements:
                    raise ValueError(f"model artifact Q8 tensor record {index} is malformed")
            else:
                raise ValueError(f"model artifact tensor record {index} has unsupported storage")

    return ArtifactLayout(artifact_path, psram_offset, psram_bytes, file_bytes)


def _file_crc32(layout: ArtifactLayout) -> int:
    checksum = 0
    remaining = layout.psram_bytes
    with layout.path.open("rb") as source:
        source.seek(layout.psram_offset)
        while remaining:
            chunk = source.read(min(READ_CHUNK_BYTES, remaining))
            if not chunk:
                raise ValueError("model artifact ended while reading the PSRAM payload")
            checksum = zlib.crc32(chunk, checksum)
            remaining -= len(chunk)
    return checksum & 0xFFFFFFFF


def _iter_file_region(layout: ArtifactLayout) -> Iterable[bytes]:
    remaining = layout.psram_bytes
    with layout.path.open("rb") as source:
        source.seek(layout.psram_offset)
        while remaining:
            chunk = source.read(min(READ_CHUNK_BYTES, remaining))
            if not chunk:
                raise ValueError("model artifact ended while reading the PSRAM payload")
            yield chunk
            remaining -= len(chunk)


class P4Device:
    """High-level protocol client for a flashed ESP32-P4 board."""

    def __init__(self, transport: SerialTransport) -> None:
        self.transport = transport
        self._last_info: DeviceInfo | None = None
        self._closed = False

    @classmethod
    def connect(cls, port: str, timeout: float = 30.0) -> "P4Device":
        return cls(SerialTransport(port, timeout).open())

    def _read_magic(self, expected: set[bytes]) -> bytes:
        if not expected or any(len(magic) != 8 for magic in expected):
            raise ValueError("protocol frame magic must be exactly 8 bytes")
        deadline = time.monotonic() + self.transport.timeout
        window = bytearray()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                names = ", ".join(sorted(magic.decode("ascii") for magic in expected))
                raise TimeoutError(f"timed out waiting for protocol frame {names}")
            window.extend(self.transport.read_exact(1, remaining))
            if len(window) > 8:
                del window[0]
            candidate = bytes(window)
            if candidate in expected:
                return candidate

    def _read_frame(self, magic: bytes, size: int) -> bytes:
        self._read_magic({magic})
        return magic + self.transport.read_exact(size - len(magic))

    def handshake(self) -> DeviceInfo:
        self.transport.write_all(b"LLMHOST5")
        frame = self._read_frame(b"LLMRDY05", 28)
        _, status, psram_bytes, loaded, payload_id, session_tokens = struct.unpack("<8siIIII", frame)
        info = DeviceInfo(status, psram_bytes, bool(loaded), payload_id, session_tokens)
        self._last_info = info
        return info

    def clear(self) -> None:
        self.transport.write_all(b"LLMCLR05")
        frame = self._read_frame(b"LLMCLRD5", 12)
        _, status = struct.unpack("<8si", frame)
        _check_status("clear", status)
        if self._last_info is not None:
            self._last_info = DeviceInfo(
                self._last_info.status,
                self._last_info.psram_bytes,
                self._last_info.loaded,
                self._last_info.payload_id,
                0,
            )

    def reload(self, layout: ArtifactLayout) -> LoadResult:
        return self.load_artifact(layout)

    def load_artifact(self, layout: ArtifactLayout) -> LoadResult:
        info = self._last_info or self.handshake()
        _check_status("handshake", info.status)
        if info.psram_bytes != layout.psram_bytes:
            raise ValueError(
                f"artifact PSRAM payload is {layout.psram_bytes} bytes, board expects {info.psram_bytes}"
            )
        checksum = _file_crc32(layout)
        self.transport.write_all(b"LLMPSR05")
        self.transport.write_all(struct.pack("<II", layout.psram_bytes, checksum))
        sent = 0
        last_report = 0
        for chunk in _iter_file_region(layout):
            self.transport.write_all(chunk)
            sent += len(chunk)
            if sent - last_report >= 1024 * 1024 or sent == layout.psram_bytes:
                pct = 100.0 * sent / layout.psram_bytes if layout.psram_bytes else 100.0
                print(
                    f"PSRAM load: {sent}/{layout.psram_bytes} bytes ({pct:.1f}%)",
                    flush=True,
                )
                last_report = sent
        frame = self._read_frame(b"LLMLOAD5", 20)
        _, status, psram_bytes, payload_id = struct.unpack("<8siII", frame)
        _check_status("PSRAM load", status)
        if psram_bytes != layout.psram_bytes or payload_id != checksum:
            raise ProtocolError("board acknowledged an unexpected PSRAM payload")
        result = LoadResult(status, psram_bytes, payload_id)
        self._last_info = DeviceInfo(status, psram_bytes, True, payload_id, 0)
        return result

    def text(
        self,
        prompt: str | bytes,
        *,
        requested_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 20,
        random_state: int = 1,
        on_chunk: Callable[[str], None] | None = None,
    ) -> TextResult:
        prompt_bytes = prompt.encode("utf-8") if isinstance(prompt, str) else bytes(prompt)
        if not 1 <= len(prompt_bytes) <= TEXT_MAX_BYTES:
            raise ValueError(f"prompt must be between 1 and {TEXT_MAX_BYTES} bytes")
        if not 1 <= requested_tokens <= CONTEXT_LENGTH:
            raise ValueError("requested token count is outside the runtime context")
        if not 1 <= top_k <= TOP_K_MAX:
            raise ValueError(f"top_k must be between 1 and {TOP_K_MAX}")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be a finite non-negative number")
        if not 0 <= random_state <= 0xFFFFFFFF:
            raise ValueError("random_state must fit in an unsigned 32-bit integer")

        request = struct.pack(
            "<IIfII",
            len(prompt_bytes),
            requested_tokens,
            temperature,
            top_k,
            random_state,
        )
        self.transport.write_all(b"LLMTXT05" + request + prompt_bytes)

        chunks = bytearray()
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        prompt_tokens = 0
        session_evicted = False
        while True:
            magic = self._read_magic({b"LLMTOK05", b"LLMCHN05", b"LLMDONE5"})
            if magic == b"LLMTOK05":
                count, flags = struct.unpack("<II", self.transport.read_exact(8))
                if count > CONTEXT_LENGTH:
                    raise ProtocolError("board returned an invalid prompt token count")
                self.transport.read_exact(count * 4)
                prompt_tokens = count
                session_evicted = bool(flags & 1)
            elif magic == b"LLMCHN05":
                (count,) = struct.unpack("<I", self.transport.read_exact(4))
                if count > TEXT_MAX_BYTES:
                    raise ProtocolError("board returned an oversized text chunk")
                chunk = self.transport.read_exact(count)
                chunks.extend(chunk)
                if on_chunk is not None:
                    piece = decoder.decode(chunk, final=False)
                    if piece:
                        on_chunk(piece)
            elif magic == b"LLMDONE5":
                status, generated, checksum, elapsed_us = struct.unpack(
                    "<iIII", self.transport.read_exact(16)
                )
                _check_status("text request", status)
                tail = decoder.decode(b"", final=True)
                if tail and on_chunk is not None:
                    on_chunk(tail)
                result = TextResult(
                    chunks.decode("utf-8", errors="replace"),
                    generated,
                    checksum,
                    elapsed_us,
                    prompt_tokens,
                    session_evicted,
                )
                if self._last_info is not None:
                    self._last_info = DeviceInfo(
                        self._last_info.status,
                        self._last_info.psram_bytes,
                        self._last_info.loaded,
                        self._last_info.payload_id,
                        self._last_info.session_tokens,
                    )
                return result
            else:
                received = magic.decode("ascii", errors="replace")
                raise ProtocolError(f"unexpected text frame {received!r}")

    def bye(self) -> None:
        self.transport.write_all(b"LLMBYE05")
        frame = self._read_frame(b"LLMBYED5", 12)
        _, status = struct.unpack("<8si", frame)
        _check_status("close", status)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            port_open = self.transport.fd is not None or self.transport._serial is not None
            if port_open and self._last_info is not None:
                self.bye()
        except (OSError, ProtocolError, TimeoutError):
            pass
        finally:
            self.transport.close()

    def __enter__(self) -> "P4Device":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()


def ensure_ready(device: P4Device, artifact: str | Path | None, *, reload: bool = False) -> ArtifactLayout | None:
    """Handshake and load the artifact when the board does not have a payload."""

    layout = validate_artifact(artifact) if artifact is not None else None
    info = device.handshake()
    _check_status("handshake", info.status)
    expected_payload_id = _file_crc32(layout) if layout is not None else None
    if reload or not info.loaded or (
        expected_payload_id is not None and info.payload_id != expected_payload_id
    ):
        if layout is None:
            raise RuntimeError("the board has no loaded model; pass --artifact")
        device.load_artifact(layout)
    return layout


def format_chat_prompt(messages: list[dict[str, str]], *, add_generation_prompt: bool = True) -> str:
    """Render the compact ChatML form used by the bundled tokenizer."""

    if not messages:
        raise ValueError("chat history cannot be empty")
    parts: list[str] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "").strip()
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported chat role: {role!r}")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)
