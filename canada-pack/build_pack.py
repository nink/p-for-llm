#!/usr/bin/env python3
"""Build a compact Canada gazetteer + template Q&A from GeoNames + CGNDB."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GEONAMES = ROOT / "geonames" / "CA.txt"
CGN = ROOT / "cgn" / "cgn_canada_csv_eng.csv"
OUT = ROOT / "out"

ADMIN1 = {
    "01": ("Alberta", "AB"),
    "02": ("British Columbia", "BC"),
    "03": ("Manitoba", "MB"),
    "04": ("New Brunswick", "NB"),
    "05": ("Newfoundland and Labrador", "NL"),
    "07": ("Nova Scotia", "NS"),
    "08": ("Ontario", "ON"),
    "09": ("Prince Edward Island", "PE"),
    "10": ("Quebec", "QC"),
    "11": ("Saskatchewan", "SK"),
    "12": ("Yukon", "YT"),
    "13": ("Northwest Territories", "NT"),
    "14": ("Nunavut", "NU"),
}

PT_CAPITALS = {
    "AB": ("Edmonton", "Alberta"),
    "BC": ("Victoria", "British Columbia"),
    "MB": ("Winnipeg", "Manitoba"),
    "NB": ("Fredericton", "New Brunswick"),
    "NL": ("St. John's", "Newfoundland and Labrador"),
    "NS": ("Halifax", "Nova Scotia"),
    "NT": ("Yellowknife", "Northwest Territories"),
    "NU": ("Iqaluit", "Nunavut"),
    "ON": ("Toronto", "Ontario"),
    "PE": ("Charlottetown", "Prince Edward Island"),
    "QC": ("Quebec City", "Quebec"),
    "SK": ("Regina", "Saskatchewan"),
    "YT": ("Whitehorse", "Yukon"),
}

COUNTRY_CAPITAL = ("Ottawa", "Ontario", "ON")

# GeoNames feature codes we keep as "places people ask about"
KEEP_FC = {
    "PPLC",
    "PPLA",
    "PPLA2",
    "PPLA3",
    "PPLA4",
    "PPL",
    "PPLL",
    "PPLS",
}


def load_geonames() -> list[dict]:
    places: list[dict] = []
    with GEONAMES.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 15:
                continue
            feature_class, feature_code = parts[6], parts[7]
            if feature_class != "P" or feature_code not in KEEP_FC:
                continue
            admin1 = parts[10]
            if admin1 not in ADMIN1:
                continue
            population = int(parts[14] or 0)
            if feature_code == "PPL" and population < 1000:
                continue
            pt_name, pt_code = ADMIN1[admin1]
            places.append(
                {
                    "name": parts[1],
                    "ascii": parts[2] or parts[1],
                    "lat": float(parts[4]),
                    "lon": float(parts[5]),
                    "feature": feature_code,
                    "pt": pt_name,
                    "pt_code": pt_code,
                    "population": population,
                    "source": "geonames",
                }
            )
    return places


def load_cgn_populated(limit_unp: int = 8000) -> list[dict]:
    """Official names for populated places / cities from NRCan."""
    rows: list[dict] = []
    unp = 0
    with CGN.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            concise = (row.get("Concise Code") or "").strip()
            category = (row.get("Generic Category") or "").strip()
            if concise not in {"CITY", "TOWN", "VIL", "UNP", "HAM", "MUN1"} and category != "Populated Place":
                continue
            if concise == "UNP":
                unp += 1
                if unp > limit_unp:
                    continue
            pt = (row.get("Province - Territory") or "").strip()
            pt_code = ""
            for adm, (name, short) in ADMIN1.items():
                if name == pt:
                    pt_code = short
                    break
            if not pt_code:
                continue
            try:
                lat = float(row["Latitude"])
                lon = float(row["Longitude"])
            except (KeyError, ValueError):
                continue
            rows.append(
                {
                    "name": row["Geographical Name"].strip(),
                    "ascii": row["Geographical Name"].strip(),
                    "lat": lat,
                    "lon": lon,
                    "feature": concise or "POP",
                    "pt": pt,
                    "pt_code": pt_code,
                    "population": 0,
                    "source": "cgndb",
                }
            )
    return rows


def make_qa(places: list[dict], rng: random.Random) -> list[dict]:
    qa: list[dict] = []
    qa.append(
        {
            "id": "capital-ca",
            "question": "What is the capital of Canada?",
            "answer": f"{COUNTRY_CAPITAL[0]}, {COUNTRY_CAPITAL[1]}",
            "type": "capital",
        }
    )
    for code, (city, pt) in PT_CAPITALS.items():
        qa.append(
            {
                "id": f"capital-{code}",
                "question": f"What is the capital of {pt}?",
                "answer": city,
                "type": "capital",
            }
        )
        qa.append(
            {
                "id": f"capital-{code}-city",
                "question": f"{city} is the capital of which province or territory?",
                "answer": pt,
                "type": "capital",
            }
        )

    # Prefer larger / admin seats for place→province questions
    ranked = sorted(places, key=lambda p: (p["feature"] != "PPLA", -p["population"], p["name"]))
    seen: set[str] = set()
    for place in ranked:
        key = place["name"].casefold()
        if key in seen:
            continue
        seen.add(key)
        qa.append(
            {
                "id": f"pt-{place['pt_code']}-{len(seen)}",
                "question": f"Which province or territory is {place['name']} in?",
                "answer": place["pt"],
                "type": "place-pt",
            }
        )
        if len(seen) >= 8000:
            break

    rng.shuffle(qa)
    return qa


def main() -> None:
    OUT.mkdir(exist_ok=True)
    geo = load_geonames()
    cgn = load_cgn_populated()
    # GeoNames first (has population); CGNDB fills official names
    by_name: dict[str, dict] = {}
    for row in cgn + geo:
        by_name.setdefault(f"{row['name'].casefold()}|{row['pt_code']}", row)
        if row["source"] == "geonames":
            by_name[f"{row['name'].casefold()}|{row['pt_code']}"] = row
    places = list(by_name.values())
    places.sort(key=lambda p: (-p["population"], p["name"]))

    pack_path = OUT / "places.jsonl"
    with pack_path.open("w", encoding="utf-8") as handle:
        for row in places:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    qa = make_qa(places, random.Random(42))
    holdout = qa[:200]
    train = qa[200:]
    (OUT / "eval-200.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in holdout),
        encoding="utf-8",
    )
    (OUT / "qa-train.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in train),
        encoding="utf-8",
    )
    index = defaultdict(list)
    for row in places:
        index[row["name"].casefold()].append(row)
    (OUT / "name-index.json").write_text(json.dumps(index, ensure_ascii=True), encoding="utf-8")
    print(f"places={len(places)} qa={len(qa)} eval={len(holdout)} train={len(train)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
