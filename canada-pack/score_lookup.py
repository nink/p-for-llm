#!/usr/bin/env python3
"""Lookup-only scoreboard for eval-200. Prints every miss."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lookup import answer, context_for, load_index, pick_name  # noqa: E402

EVAL_PATH = ROOT / "out" / "eval-200.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    index = load_index()
    rows = load_jsonl(EVAL_PATH)
    hits = unk = wrong = 0
    for row in rows:
        ctx = context_for(row["question"], index)
        gold = row["answer"]
        if not ctx:
            unk += 1
            kind = "UNK"
        elif gold.casefold() in ctx.casefold():
            hits += 1
            continue
        else:
            wrong += 1
            kind = "WRONG"
        name = pick_name(row["question"], index)
        print(f"{kind} id={row['id']} gold={gold!r} name={name!r}")
        print(f"  q={row['question']}")
        print(f"  a={answer(row['question'], index)[:240]}")
    n = len(rows)
    print(f"\nLOOKUP {hits}/{n} ({100 * hits / n:.1f}%)  unknown={unk} wrong={wrong}")
    return 0 if hits / n >= 0.95 else 1


if __name__ == "__main__":
    raise SystemExit(main())
