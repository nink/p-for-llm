#!/usr/bin/env python3
"""Read canada.kpack (same layout as llmm_pack.c)."""

from __future__ import annotations

import struct
from pathlib import Path

HEADER = struct.Struct("<4sHHIIII")
DIR_ENT = struct.Struct("<IIHH")
HEADER_BYTES = 32


class KPack:
    def __init__(self, path: Path) -> None:
        self.data = path.read_bytes()
        magic, version, _flags, n_keys, dir_off, rec_off, rec_bytes = HEADER.unpack_from(self.data, 0)
        if magic != b"KPK1" or version != 1:
            raise ValueError(f"bad kpack header {magic!r} v{version}")
        self.n_keys = n_keys
        self.dir_off = dir_off
        self.rec_off = rec_off
        self.rec_bytes = rec_bytes

    def _dir(self, index: int) -> tuple[int, int, int, int]:
        return DIR_ENT.unpack_from(self.data, self.dir_off + index * DIR_ENT.size)

    def _fold(self, index: int) -> str:
        fold_off, _rec_off, _n_rec, fold_len = self._dir(index)
        return self.data[fold_off : fold_off + fold_len].decode("ascii")

    def find(self, folded: str) -> list[dict]:
        if not folded:
            return []
        lo, hi = 0, self.n_keys
        while lo < hi:
            mid = (lo + hi) // 2
            got = self._fold(mid)
            if got < folded:
                lo = mid + 1
            else:
                hi = mid
        if lo >= self.n_keys or self._fold(lo) != folded:
            return []
        _fold_off, rec_off, n_rec, _fold_len = self._dir(lo)
        pos = rec_off
        stored = self.data[pos]
        pos += 1
        rows = []
        count = min(stored, n_rec)
        for _ in range(count):
            nlen = self.data[pos]
            pos += 1
            name = self.data[pos : pos + nlen].decode("utf-8", errors="replace")
            pos += nlen
            plen = self.data[pos]
            pos += 1
            pt = self.data[pos : pos + plen].decode("utf-8", errors="replace")
            pos += plen
            tlen = self.data[pos]
            pos += 1
            term = self.data[pos : pos + tlen].decode("utf-8", errors="replace")
            pos += tlen
            code = self.data[pos : pos + 4].decode("ascii", errors="replace").strip()
            pos += 4
            rows.append({"name": name, "pt": pt, "term": term, "code": code})
        return rows
