#!/usr/bin/env python3
"""Build a seekable canada.kpack for ESP32-P4 SD (same fold as lookup.py / llmm_pack.c)."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lookup import PREFERRED, folded_map, load_index, rank_rows  # noqa: E402

OUT = ROOT / "out" / "canada.kpack"
MAGIC = b"KPK1"
HEADER = struct.Struct("<4sHHIIII")  # magic, version, flags, n_keys, dir_off, rec_off, rec_bytes
DIR_ENT = struct.Struct("<IIHH")  # fold_off, rec_off, n_rec, fold_len
HEADER_BYTES = 32
MAX_NAME = 80
MAX_PT = 40
MAX_TERM = 40
MAX_REC = 16


def _clip(text: str, limit: int) -> bytes:
    raw = (text or "").encode("utf-8", errors="replace")
    return raw[:limit]


def _best_by_pt(rows: list[dict]) -> list[dict]:
    chosen: dict[str, dict] = {}
    for row in rank_rows(rows):
        pt = (row.get("pt") or "").strip()
        if not pt or pt in chosen:
            continue
        chosen[pt] = row
        if len(chosen) >= MAX_REC:
            break
    return list(chosen.values())


def main() -> int:
    print("loading gazetteer ...", flush=True)
    index = load_index()
    fmap = folded_map(index)
    keys = sorted(k for k in fmap if k)
    print(f"folded names={len(keys)}", flush=True)

    fold_blob = bytearray()
    rec_blob = bytearray()
    directory: list[tuple[int, int, int, int]] = []

    for key in keys:
        rows = _best_by_pt(fmap[key])
        if not rows:
            continue
        fold_b = key.encode("ascii")
        if len(fold_b) > 64:
            fold_b = fold_b[:64]
        fold_off = len(fold_blob)
        fold_blob.extend(fold_b)
        rec_off = len(rec_blob)
        rec_blob.append(len(rows))
        for row in rows:
            name = _clip(row.get("name") or "", MAX_NAME)
            pt = _clip(row.get("pt") or "", MAX_PT)
            term = _clip((row.get("term") or row.get("kind") or "")[:MAX_TERM], MAX_TERM)
            code = ((row.get("code") or "") + "    ")[:4].encode("ascii", errors="replace")
            rec_blob.append(len(name))
            rec_blob.extend(name)
            rec_blob.append(len(pt))
            rec_blob.extend(pt)
            rec_blob.append(len(term))
            rec_blob.extend(term)
            rec_blob.extend(code)
        directory.append((fold_off, rec_off, len(rows), len(fold_b)))

    n_keys = len(directory)
    dir_off = HEADER_BYTES
    rec_off = dir_off + n_keys * DIR_ENT.size
    fold_off = rec_off + len(rec_blob)
    # Store fold blob after records; dir fold_off is absolute file offset.
    # Redefine: fold_off in dir is absolute.
    fold_file_off = rec_off + len(rec_blob)
    rec_file_off = rec_off

    header = bytearray(HEADER_BYTES)
    HEADER.pack_into(header, 0, MAGIC, 1, 0, n_keys, dir_off, rec_file_off, len(rec_blob))

    dir_bytes = bytearray()
    for f_off, r_off, n_rec, f_len in directory:
        dir_bytes.extend(
            DIR_ENT.pack(fold_file_off + f_off, rec_file_off + r_off, n_rec, f_len)
        )

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("wb") as handle:
        handle.write(header)
        handle.write(dir_bytes)
        handle.write(rec_blob)
        handle.write(fold_blob)

    print(
        f"wrote {OUT} keys={n_keys} bytes={OUT.stat().st_size} "
        f"dir={len(dir_bytes)} rec={len(rec_blob)} fold={len(fold_blob)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
