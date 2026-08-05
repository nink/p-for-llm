#!/usr/bin/env python3
"""Build the fixed P4 binary asset for the locked English BPE tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


MAGIC = b"LLMTOK01"
VERSION = 1
HEADER_BYTES = 64
HEADER = struct.Struct("<8sHH12I")
VOCAB_SIZE = 32_768
BASE_VOCAB_SIZE = 32_753
BYTE_IDS_BYTES = 256 * 2
TOKEN_OFFSET_BYTES = (VOCAB_SIZE + 1) * 4
MERGE_RECORD = struct.Struct("<HHHH")
EOS_TOKEN = 32_753


def byte_decoder() -> dict[int, int]:
    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(ord("\u00a1"), ord("\u00ac") + 1))
    visible += list(range(ord("\u00ae"), ord("\u00ff") + 1))
    codepoints = visible[:]
    extra = 0
    for byte in range(256):
        if byte not in visible:
            visible.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    return dict(zip(codepoints, visible, strict=True))


BYTE_DECODER = byte_decoder()


def decode_byte_level(token: str) -> bytes:
    try:
        return bytes(BYTE_DECODER[ord(character)] for character in token)
    except KeyError as error:
        raise ValueError(f"unsupported byte-level token: {token!r}") from error


def align(value: int, alignment: int = 4) -> int:
    return (value + alignment - 1) & -alignment


def load_tokens(directory: Path) -> tuple[list[str], dict[str, int], list[str]]:
    vocab = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
    if not isinstance(vocab, dict):
        raise ValueError("vocab.json must contain an object")
    tokens = [""] * BASE_VOCAB_SIZE
    for token, token_id in vocab.items():
        if not isinstance(token, str) or not isinstance(token_id, int):
            raise ValueError("vocab.json contains an invalid entry")
        if not 0 <= token_id < BASE_VOCAB_SIZE or tokens[token_id]:
            raise ValueError("vocab IDs are not the expected dense base range")
        tokens[token_id] = token
    if any(not token for token in tokens):
        raise ValueError("vocab.json has a missing base token")

    config = json.loads((directory / "tokenizer_config.json").read_text(encoding="utf-8"))
    decoder = config.get("added_tokens_decoder")
    if not isinstance(decoder, dict):
        raise ValueError("tokenizer_config.json has no added token definitions")
    added: list[str] = []
    for token_id in range(BASE_VOCAB_SIZE, VOCAB_SIZE):
        entry = decoder.get(str(token_id))
        if not isinstance(entry, dict) or not isinstance(entry.get("content"), str):
            raise ValueError(f"missing added token {token_id}")
        added.append(entry["content"])
    return tokens, vocab, added


def build_asset(directory: Path) -> bytes:
    base_tokens, vocab, added_tokens = load_tokens(directory)
    token_bytes = [decode_byte_level(token) for token in base_tokens]
    token_bytes.extend(token.encode("ascii") for token in added_tokens)
    if len(token_bytes) != VOCAB_SIZE:
        raise ValueError("unexpected tokenizer vocabulary size")

    byte_ids = [0] * 256
    for token_id, payload in enumerate(token_bytes[:256]):
        if len(payload) != 1 or byte_ids[payload[0]] != 0:
            raise ValueError("first 256 BPE tokens are not a byte permutation")
        byte_ids[payload[0]] = token_id + 1
    if any(token_id == 0 for token_id in byte_ids):
        raise ValueError("BPE byte primitive is missing")
    byte_ids = [token_id - 1 for token_id in byte_ids]

    token_offsets = [0]
    joined_tokens = bytearray()
    for payload in token_bytes:
        joined_tokens.extend(payload)
        token_offsets.append(len(joined_tokens))
    max_token_bytes = max(len(payload) for payload in token_bytes)

    merge_records: list[tuple[int, int, int, int]] = []
    for rank, line in enumerate((directory / "merges.txt").read_text(encoding="utf-8").splitlines()):
        left, separator, right = line.partition(" ")
        if not separator or not left or not right:
            raise ValueError(f"invalid merge at rank {rank}: {line!r}")
        try:
            left_id = vocab[left]
            right_id = vocab[right]
            result_id = vocab[left + right]
        except KeyError as error:
            raise ValueError(f"merge references an absent base token: {line!r}") from error
        merge_records.append((left_id, right_id, result_id, rank))
    merge_records.sort(key=lambda record: (record[0], record[1]))

    byte_ids_offset = HEADER_BYTES
    token_offsets_offset = align(byte_ids_offset + BYTE_IDS_BYTES)
    token_bytes_offset = align(token_offsets_offset + TOKEN_OFFSET_BYTES)
    merge_offset = align(token_bytes_offset + len(joined_tokens))
    total_bytes = merge_offset + len(merge_records) * MERGE_RECORD.size
    header = HEADER.pack(
        MAGIC,
        VERSION,
        HEADER_BYTES,
        total_bytes,
        VOCAB_SIZE,
        len(merge_records),
        byte_ids_offset,
        token_offsets_offset,
        token_bytes_offset,
        len(joined_tokens),
        merge_offset,
        max_token_bytes,
        EOS_TOKEN,
        0,
        0,
    ).ljust(HEADER_BYTES, b"\0")
    asset = bytearray(header)
    asset.extend(struct.pack("<256H", *byte_ids))
    asset.extend(b"\0" * (token_offsets_offset - len(asset)))
    asset.extend(struct.pack(f"<{len(token_offsets)}I", *token_offsets))
    asset.extend(b"\0" * (token_bytes_offset - len(asset)))
    asset.extend(joined_tokens)
    asset.extend(b"\0" * (merge_offset - len(asset)))
    for record in merge_records:
        asset.extend(MERGE_RECORD.pack(*record))
    if len(asset) != total_bytes:
        raise AssertionError("tokenizer asset length calculation is wrong")
    return bytes(asset)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=root / "assets/qwen3.5-english-tokenizer",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    asset = build_asset(args.tokenizer_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(asset)
    print(f"tokenizer={args.tokenizer_dir}")
    print(f"output={args.output}")
    print(f"bytes={len(asset)}")
    print(f"sha256={hashlib.sha256(asset).hexdigest()}")


if __name__ == "__main__":
    main()
