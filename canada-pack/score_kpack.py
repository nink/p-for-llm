#!/usr/bin/env python3
"""Score eval-200 against canada.kpack (the file the P4 will seek)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from kpack import KPack  # noqa: E402
from lookup import (  # noqa: E402
    PT_IS_IN,
    STOP,
    _capital_answer,
    fold,
    format_provinces,
)

EVAL_PATH = ROOT / "out" / "eval-200.jsonl"
KPACK_PATH = ROOT / "out" / "canada.kpack"


def pick_name(question: str, pack: KPack) -> str | None:
    extracted = PT_IS_IN.search(question.strip())
    if extracted:
        key = fold(extracted.group("name"))
        return key if pack.find(key) else None
    tokens = fold(question).split()
    hits: list[str] = []
    max_len = min(12, len(tokens))
    for length in range(1, max_len + 1):
        for start in range(0, len(tokens) - length + 1):
            span = " ".join(tokens[start : start + length])
            if length == 1 and span in STOP:
                continue
            if pack.find(span):
                hits.append(span)
    if not hits:
        return None
    return max(hits, key=lambda n: (len(n.split()), len(n), n))


def answer(question: str, pack: KPack) -> str:
    cap = _capital_answer(question.strip())
    if cap is not None:
        return cap
    name = pick_name(question, pack)
    if not name:
        return (
            "Unknown — no official Canadian name in that question. "
            "I only answer from the gazetteer (places, lakes, rivers, parks, peaks)."
        )
    rows = pack.find(name)
    return format_provinces(rows)


def main() -> int:
    if not KPACK_PATH.is_file():
        print("run build_kpack.py first", file=sys.stderr)
        return 2
    pack = KPack(KPACK_PATH)
    rows = [
        json.loads(line)
        for line in EVAL_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    hits = unk = wrong = 0
    for row in rows:
        text = answer(row["question"], pack)
        gold = row["answer"]
        if text.startswith("Unknown"):
            unk += 1
            print(f"UNK {row['id']} {row['question']!r} gold={gold!r}")
            continue
        if gold.casefold() in text.casefold():
            hits += 1
        else:
            wrong += 1
            print(f"WRONG {row['id']} gold={gold!r} got={text[:180]!r}")
    n = len(rows)
    print(f"\nKPACK {hits}/{n} ({100 * hits / n:.1f}%) unknown={unk} wrong={wrong}")
    return 0 if hits / n >= 0.95 else 1


if __name__ == "__main__":
    raise SystemExit(main())
