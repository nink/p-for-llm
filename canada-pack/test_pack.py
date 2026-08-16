#!/usr/bin/env python3
"""Test Canada pack: gazetteer lookup accuracy, then retrieve→PFor on a slice."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOST = ROOT.parent / "runtime" / "host"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HOST) not in sys.path:
    sys.path.insert(0, str(HOST))

from lookup import context_for, load_index  # noqa: E402

EVAL_PATH = ROOT / "out" / "eval-200.jsonl"
EXTRA = [
    {"id": "capital-ca", "question": "What is the capital of Canada?", "answer": "Ottawa", "type": "capital"},
    {"id": "capital-ON", "question": "What is the capital of Ontario?", "answer": "Toronto", "type": "capital"},
    {"id": "capital-BC", "question": "What is the capital of British Columbia?", "answer": "Victoria", "type": "capital"},
]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def contains_answer(text: str, answer: str) -> bool:
    return answer.casefold() in (text or "").casefold()


def score_lookup(rows: list[dict], index: dict) -> dict:
    hits = 0
    unknown = 0
    misses: list[tuple[str, str, str]] = []
    for row in rows:
        ctx = context_for(row["question"], index)
        if not ctx or ctx.startswith("Unknown"):
            unknown += 1
            if len(misses) < 8:
                misses.append((row["question"], row["answer"], ctx or "(empty)"))
            continue
        if contains_answer(ctx, row["answer"]):
            hits += 1
        else:
            if len(misses) < 8:
                misses.append((row["question"], row["answer"], ctx[:160]))
    return {
        "n": len(rows),
        "hits": hits,
        "unknown": unknown,
        "wrong": len(rows) - hits - unknown,
        "misses": misses,
    }


def p4_slice(eval_rows: list[dict]) -> list[dict]:
    picked: list[dict] = list(EXTRA)
    by_id = {row["id"]: row for row in eval_rows}
    for key in ("capital-NL-city", "pt-NB-9", "pt-ON-6417", "pt-NS-2151", "pt-QC-1356"):
        if key in by_id:
            picked.append(by_id[key])
    # a couple of random-looking eval rows that are likely ambiguous
    for row in eval_rows:
        if row["id"] not in {p["id"] for p in picked} and "Lake" in row["question"]:
            picked.append(row)
            break
    return picked[:8]


def run_p4(rows: list[dict], index: dict) -> list[dict]:
    from p4 import P4Device, discover_eth_host, ensure_ready, format_chat_prompt

    print("scanning Ethernet for P4 ...", flush=True)
    host = discover_eth_host()
    if not host:
        raise RuntimeError("no P4 on TCP 8742 — close Arduino Serial Monitor if using COM5, or plug Ethernet")
    print(f"found {host}:8742", flush=True)

    results: list[dict] = []
    with P4Device.connect(host=host, timeout=120) as device:
        ensure_ready(device, None)
        for row in rows:
            ctx = context_for(row["question"], index)
            ctx = (
                ctx.replace("\u2014", "-")
                .replace("\u2013", "-")
                .encode("ascii", "replace")
                .decode("ascii")
            )
            user = (
                f"CONTEXT:\n{ctx}\n\nQUESTION: {row['question']}\n"
                "Answer using only CONTEXT. Reply with a short fact."
            )
            prompt = format_chat_prompt([{"role": "user", "content": user}])
            device.clear()
            first_t: float | None = None
            t0 = time.perf_counter()

            def on_chunk(piece: str, _ft=[None]) -> None:
                nonlocal first_t
                if first_t is None and piece:
                    first_t = time.perf_counter()
                print(piece, end="", flush=True)

            print(f"\n=== {row['id']} ===", flush=True)
            print(f"q: {row['question']}", flush=True)
            print(f"gold: {row['answer']}", flush=True)
            print(f"lookup: {ctx[:180]}", flush=True)
            print("pfor> ", end="", flush=True)
            try:
                result = device.text(
                    prompt,
                    requested_tokens=48,
                    temperature=0.2,
                    top_k=20,
                    random_state=1,
                    on_chunk=on_chunk,
                )
            except Exception as exc:
                print(f"\nerror: {exc}", flush=True)
                results.append(
                    {
                        "id": row["id"],
                        "lookup_ok": contains_answer(ctx, row["answer"]),
                        "pfor_ok": False,
                        "ptok": 0,
                        "ttft": float("nan"),
                        "text": str(exc)[:160],
                    }
                )
                continue
            t1 = time.perf_counter()
            print(flush=True)
            ttft = (first_t - t0) if first_t else float("nan")
            text = result.text.strip().replace("\n", " / ")
            lookup_ok = contains_answer(ctx, row["answer"])
            pfor_ok = contains_answer(text, row["answer"])
            print(
                f"ptok={result.prompt_tokens} TTFT={ttft:.2f}s "
                f"lookup_hit={lookup_ok} pfor_hit={pfor_ok}",
                flush=True,
            )
            results.append(
                {
                    "id": row["id"],
                    "lookup_ok": lookup_ok,
                    "pfor_ok": pfor_ok,
                    "ptok": result.prompt_tokens,
                    "ttft": ttft,
                    "text": text[:160],
                }
            )
    return results


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Score Canada pack lookup; optionally PFor.")
    parser.add_argument("--p4", action="store_true", help="also run retrieve→PFor on a tiny slice")
    args = parser.parse_args()

    print("loading CGN index ...", flush=True)
    index = load_index()
    eval_rows = load_jsonl(EVAL_PATH)
    stats = score_lookup(eval_rows, index)
    print(
        f"\nLOOKUP eval-200: {stats['hits']}/{stats['n']} contain gold "
        f"({100 * stats['hits'] / stats['n']:.1f}%)  "
        f"unknown={stats['unknown']} wrong={stats['wrong']}",
        flush=True,
    )
    for q, gold, ctx in stats["misses"]:
        print(f"  miss q={q!r} gold={gold!r} ctx={ctx.encode('ascii', 'replace').decode()!r}", flush=True)

    if not args.p4:
        ok = stats["hits"] / stats["n"] >= 0.95
        return 0 if ok else 1

    slice_rows = p4_slice(eval_rows)
    print(f"\nP4 retrieve->generate on {len(slice_rows)} questions", flush=True)
    rows = run_p4(slice_rows, index)
    print("\n======== P4 SLICE ========", flush=True)
    print(f"{'id':<22} {'lu':>3} {'p4':>3} {'ptok':>5} {'TTFT':>6}  reply", flush=True)
    for row in rows:
        print(
            f"{row['id']:<22} {str(row['lookup_ok']):>3} {str(row['pfor_ok']):>3} "
            f"{row['ptok']:>5} {row['ttft']:>5.2f}s  {row['text'][:70]}",
            flush=True,
        )
    p4_hits = sum(1 for row in rows if row["pfor_ok"])
    lu_hits = sum(1 for row in rows if row["lookup_ok"])
    print(f"lookup {lu_hits}/{len(rows)}  pfor {p4_hits}/{len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
