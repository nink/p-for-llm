#!/usr/bin/env python3
"""Compact name index over the full NRCan CGNDB (official Canadian names)."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CGN = ROOT / "cgn" / "cgn_canada_csv_eng.csv"
OUT = ROOT / "out" / "cgn-index.json"

# Drop the long tail of unnamed/tiny water duplicates? Keep all official names.
# Duplicate names (Long Lake) stay as a list so lookup can disambiguate.


def main() -> None:
    index: dict[str, list[dict]] = defaultdict(list)
    n = 0
    with CGN.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("Geographical Name") or "").strip()
            if not name:
                continue
            rec = {
                "name": name,
                "kind": (row.get("Generic Category") or "").strip(),
                "term": (row.get("Generic Term") or "").strip(),
                "code": (row.get("Concise Code") or "").strip(),
                "pt": (row.get("Province - Territory") or "").strip(),
                "lat": row.get("Latitude") or "",
                "lon": row.get("Longitude") or "",
            }
            index[name.casefold()].append(rec)
            n += 1
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(index, ensure_ascii=True, separators=(",", ":")), encoding="utf-8")
    print(f"features={n} unique_names={len(index)} wrote {OUT} bytes={OUT.stat().st_size}")


if __name__ == "__main__":
    main()
