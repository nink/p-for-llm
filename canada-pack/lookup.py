#!/usr/bin/env python3
"""High-accuracy Canada geography answers: retrieve from CGNDB, never invent."""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INDEX_PATH = ROOT / "out" / "cgn-index.json"
NAME_INDEX_PATH = ROOT / "out" / "name-index.json"

CAPITALS = {
    "canada": "Ottawa, Ontario (federal capital)",
    "alberta": "Edmonton",
    "british columbia": "Victoria",
    "manitoba": "Winnipeg",
    "new brunswick": "Fredericton",
    "newfoundland and labrador": "St. John's",
    "newfoundland": "St. John's",
    "nova scotia": "Halifax",
    "northwest territories": "Yellowknife",
    "nunavut": "Iqaluit",
    "ontario": "Toronto",
    "prince edward island": "Charlottetown",
    "quebec": "Quebec City",
    "saskatchewan": "Regina",
    "yukon": "Whitehorse",
    "bc": "Victoria",
    "pei": "Charlottetown",
    "nwt": "Yellowknife",
}

CITY_IS_CAPITAL_OF = {
    "ottawa": "Canada (federal capital, in Ontario)",
    "edmonton": "Alberta",
    "victoria": "British Columbia",
    "winnipeg": "Manitoba",
    "fredericton": "New Brunswick",
    "st johns": "Newfoundland and Labrador",
    "halifax": "Nova Scotia",
    "yellowknife": "Northwest Territories",
    "iqaluit": "Nunavut",
    "toronto": "Ontario",
    "charlottetown": "Prince Edward Island",
    "quebec city": "Quebec",
    "regina": "Saskatchewan",
    "whitehorse": "Yukon",
}

PT_HINT = re.compile(
    r"\b(alberta|british columbia|manitoba|new brunswick|newfoundland|"
    r"nova scotia|ontario|quebec|saskatchewan|yukon|nunavut|"
    r"northwest territories|pei|nwt|b\.?c\.?)\b",
    re.I,
)
PT_IS_IN = re.compile(
    r"which province or territory is (?P<name>.+?) in\??\s*$",
    re.I,
)
REV_CAPITAL = re.compile(
    r"^(?P<city>.+?)\s+is the capital of which\b",
    re.I,
)
FWD_CAPITAL = re.compile(r"\bcapital of (?P<place>.+?)\??$", re.I)

STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "or",
        "in",
        "is",
        "which",
        "what",
        "where",
        "province",
        "territory",
        "capital",
    }
)
PREFERRED = ("CITY", "TOWN", "VILG", "HAM", "UNP", "MUN1", "MUN2", "PARK")

_FOLDED: dict[int, dict[str, list[dict]]] = {}


# UTF-8 C3 xx (U+00C0..00FF) → ASCII letter; '.' means not a letter. Same table as llmm_pack.c.
_C3_ASCII = b"aaaaaa.ceeeeiiii.nooooo..uuuuy..aaaaaa.ceeeeiiii.nooooo..uuuuy.y"


def fold(text: str) -> str:
    """ASCII fold matching P4 C. Développement-Delage == developpement delage."""
    raw = text.encode("utf-8", errors="replace")
    out: list[str] = []
    i = 0
    while i < len(raw):
        b0 = raw[i]
        if b0 < 0x80:
            if 48 <= b0 <= 57 or 97 <= b0 <= 122:
                out.append(chr(b0))
            elif 65 <= b0 <= 90:
                out.append(chr(b0 + 32))
            elif b0 in (0x27, 0x60):
                pass
            else:
                out.append(" ")
            i += 1
            continue
        if b0 == 0xC3 and i + 1 < len(raw):
            mapped = _C3_ASCII[raw[i + 1] - 0x80] if 0x80 <= raw[i + 1] <= 0xBF else 0
            out.append(chr(mapped) if mapped and mapped != ord(".") else " ")
            i += 2
            continue
        if b0 == 0xE2 and i + 2 < len(raw) and raw[i + 1] == 0x80 and raw[i + 2] in (0x98, 0x99):
            i += 3
            continue
        if (b0 & 0xE0) == 0xC0:
            i += 2
        elif (b0 & 0xF0) == 0xE0:
            i += 3
        elif (b0 & 0xF8) == 0xF0:
            i += 4
        else:
            i += 1
        out.append(" ")
    return " ".join("".join(out).split())


def _geonames_row(place: dict) -> dict:
    return {
        "name": place.get("name") or "",
        "kind": "Populated Place",
        "term": "populated place",
        "code": "UNP",
        "pt": place.get("pt") or "",
        "lat": str(place.get("lat") or ""),
        "lon": str(place.get("lon") or ""),
    }


def _build_folded(raw: dict) -> dict[str, list[dict]]:
    folded: dict[str, list[dict]] = defaultdict(list)
    for key, rows in raw.items():
        if not isinstance(rows, list):
            continue
        folded[fold(key)].extend(rows)
    if NAME_INDEX_PATH.is_file():
        extra = json.loads(NAME_INDEX_PATH.read_text(encoding="utf-8"))
        for key, places in extra.items():
            fk = fold(key)
            if fk in folded or not isinstance(places, list):
                continue
            folded[fk] = [_geonames_row(p) for p in places if isinstance(p, dict)]
    return dict(folded)


def load_index() -> dict:
    raw = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    _FOLDED[id(raw)] = _build_folded(raw)
    return raw


def folded_map(index: dict) -> dict[str, list[dict]]:
    cached = _FOLDED.get(id(index))
    if cached is None:
        cached = _build_folded(index)
        _FOLDED[id(index)] = cached
    return cached


def rank_rows(rows: list[dict]) -> list[dict]:
    def key(row: dict) -> tuple:
        code = row.get("code") or ""
        try:
            pref = PREFERRED.index(code)
        except ValueError:
            pref = len(PREFERRED)
        return (pref, row.get("pt") or "")

    return sorted(rows, key=key)


def format_row(row: dict) -> str:
    bits = [row["name"]]
    if row.get("term"):
        bits.append(row["term"].lower())
    elif row.get("kind"):
        bits.append(row["kind"].lower())
    if row.get("pt"):
        bits.append(f"in {row['pt']}")
    return ", ".join(bits[:1]) + " — " + ", ".join(bits[1:]) if len(bits) > 1 else bits[0]


def format_provinces(rows: list[dict]) -> str:
    """Name + every province that has this official name. Homonyms stay listed."""
    rows = rank_rows(rows)
    display = rows[0]["name"]
    pts: list[str] = []
    seen: set[str] = set()
    for row in rows:
        pt = row.get("pt") or ""
        if pt and pt not in seen:
            seen.add(pt)
            pts.append(pt)
    if not pts:
        return f"Unknown — no province packed for {display!r}."
    if len(pts) == 1:
        return format_row(next(row for row in rows if row.get("pt") == pts[0])) + "."
    if len(pts) == 2:
        return f"{display} is an official name in {pts[0]} and {pts[1]}."
    return f"{display} is an official name in {', '.join(pts[:-1])}, and {pts[-1]}."


def pick_name(question: str, index: dict) -> str | None:
    """Return a folded gazetteer key, or None."""
    fmap = folded_map(index)
    extracted = PT_IS_IN.search(question.strip())
    if extracted:
        key = fold(extracted.group("name"))
        if key in fmap:
            return key
        return None

    tokens = fold(question).split()
    hits: list[str] = []
    max_len = min(12, len(tokens))
    for length in range(1, max_len + 1):
        for start in range(0, len(tokens) - length + 1):
            span = " ".join(tokens[start : start + length])
            if length == 1 and span in STOP:
                continue
            if span in fmap:
                hits.append(span)
    if not hits:
        return None
    return max(hits, key=lambda n: (len(n.split()), len(n), n))


def _capital_answer(question: str) -> str | None:
    low = question.strip()
    rev = REV_CAPITAL.search(low)
    if rev:
        city = fold(rev.group("city"))
        pt = CITY_IS_CAPITAL_OF.get(city)
        if pt:
            return f"{rev.group('city').strip()} is the capital of {pt}."
        return f"Unknown — no capital packed for {rev.group('city').strip()!r}."

    fwd = FWD_CAPITAL.search(low)
    if not fwd:
        return None
    key = fold(fwd.group("place"))
    if key in {"which province or territory", "which province", "which territory"}:
        return None
    if key in CAPITALS:
        return CAPITALS[key]
    aliases = {"b c": "british columbia", "nwt": "northwest territories"}
    key = aliases.get(key, key)
    if key in CAPITALS:
        return CAPITALS[key]
    return f"Unknown — no capital packed for {fwd.group('place').strip()!r}."


def answer(question: str, index: dict) -> str:
    q = question.strip()
    cap = _capital_answer(q)
    if cap is not None:
        return cap

    name = pick_name(q, index)
    if not name:
        return (
            "Unknown — no official Canadian name in that question. "
            "I only answer from the gazetteer (places, lakes, rivers, parks, peaks)."
        )

    rows = list(folded_map(index)[name])
    hint = PT_HINT.search(q)
    asking_pt = bool(PT_IS_IN.search(q))
    if hint and len({r.get("pt") for r in rows}) > 1:
        h = hint.group(1).casefold()
        aliases = {
            "bc": "british columbia",
            "b.c": "british columbia",
            "pei": "prince edward island",
            "nwt": "northwest territories",
        }
        h = aliases.get(h.replace(".", ""), h)
        filtered = [r for r in rows if h in (r.get("pt") or "").casefold()]
        if filtered:
            rows = filtered

    if asking_pt or len({r.get("pt") for r in rows if r.get("pt")}) > 1:
        return format_provinces(rows)

    rows = rank_rows(rows)
    top = rows[0]
    if len(rows) == 1:
        return format_row(top) + "."
    if top.get("code") in PREFERRED[:5] and "lake" not in fold(q) and "river" not in fold(q):
        return format_row(top) + "."

    pts = sorted({r["pt"] for r in rows if r.get("pt")})
    if len(pts) == 1:
        kinds = sorted({r.get("term") or r.get("kind") or "feature" for r in rows})
        return (
            f"{rows[0]['name']} matches {len(rows)} official features in {pts[0]} "
            f"({', '.join(kinds[:6])})."
        )
    return format_provinces(rows)


def context_for(question: str, index: dict) -> str:
    """Gazetteer row(s) to condition a speaker model. Empty if nothing retrieved."""
    result = answer(question, index)
    if result.startswith("Unknown"):
        return ""
    return result


def main() -> int:
    if not INDEX_PATH.is_file():
        print("run build_cgn_index.py first", file=sys.stderr)
        return 1
    index = load_index()
    if len(sys.argv) > 1:
        print(answer(" ".join(sys.argv[1:]), index))
        return 0
    print("Canada gazetteer. Answers only from NRCan names. Ctrl-C to quit.")
    while True:
        try:
            line = input("q> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        print(answer(line, index))


if __name__ == "__main__":
    raise SystemExit(main())
