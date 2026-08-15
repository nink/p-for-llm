#!/usr/bin/env python3
"""Short (<1024B bypass) vs ~8k-token raw long (on-device compress) compare."""

from __future__ import annotations

import time
from pathlib import Path

try:
    from .compress import estimate_tokens
    from .p4 import P4Device, RAW_TEXT_MAX_BYTES, TEXT_MAX_BYTES, ensure_ready, format_chat_prompt
except ImportError:
    from compress import estimate_tokens
    from p4 import P4Device, RAW_TEXT_MAX_BYTES, TEXT_MAX_BYTES, ensure_ready, format_chat_prompt

SAMPLE_DIR = Path(__file__).resolve().parent / "testdata"
LONG_PATH = SAMPLE_DIR / "long_8k_context.md"
REQ = 64


def ensure_long_context() -> tuple[str, int]:
    base = (SAMPLE_DIR / "sample_long_context.md").read_text(encoding="utf-8")
    paras = [
        "Chloroplasts capture photons and convert carbon dioxide into sugars during photosynthesis.",
        "Mitochondria oxidize sugars and release usable energy as ATP for cellular work.",
        "The nucleus stores DNA and coordinates gene expression across plant tissues.",
        "Cellulose walls provide structure while the plasma membrane controls transport.",
        "Students often confuse respiration with photosynthesis; healthy plants perform both.",
        "Root cells usually lack chloroplasts and rely on sugar transported from leaves.",
        "Guard cells open stomata when light and water status allow gas exchange.",
        "Chlorophyll absorbs mainly blue and red wavelengths and reflects green light.",
    ]
    chunks = [base]
    n = 1
    while True:
        block = f"\n\n## Section {n}\n" + " ".join(paras * 4)
        trial = "\n".join(chunks + [block])
        prompt = format_chat_prompt(
            [
                {
                    "role": "user",
                    "content": (
                        f"CONTEXT:\n{trial}\n\n"
                        "QUESTION: Why do plant cells need chloroplasts?\n"
                        "Answer using only CONTEXT."
                    ),
                }
            ]
        )
        if len(prompt.encode("utf-8")) > RAW_TEXT_MAX_BYTES - 128:
            break
        chunks.append(block)
        if estimate_tokens(trial) >= 8000:
            break
        n += 1

    text = "\n".join(chunks)
    while estimate_tokens(text) > 8200:
        text = text[: int(len(text) * 0.99)]
    LONG_PATH.write_text(text, encoding="utf-8")
    return text, estimate_tokens(text)


def run_case(device: P4Device, label: str, prompt: str, src_tok_est: int | None) -> dict:
    device.clear()
    raw_b = len(prompt.encode("utf-8"))
    bypass = raw_b <= TEXT_MAX_BYTES
    first_t: float | None = None
    t0 = time.perf_counter()

    def on_chunk(piece: str) -> None:
        nonlocal first_t
        if first_t is None and piece:
            first_t = time.perf_counter()
        print(piece, end="", flush=True)

    print(f"\n=== {label} ===", flush=True)
    print(
        f"wire_bytes={raw_b} bypass_compress={bypass} "
        f"src_tok_est={src_tok_est if src_tok_est is not None else 'n/a'}",
        flush=True,
    )
    print("assistant> ", end="", flush=True)
    result = device.text(
        prompt,
        requested_tokens=REQ,
        temperature=0.3,
        top_k=20,
        random_state=1,
        on_chunk=on_chunk,
    )
    t1 = time.perf_counter()
    print(flush=True)
    wall_s = t1 - t0
    ttft_s = (first_t - t0) if first_t is not None else None
    board_s = result.elapsed_us / 1e6
    gen = result.generated_tokens
    tps_board = gen / board_s if board_s > 0 else 0.0
    tps_decode = 0.0
    if ttft_s is not None and wall_s > ttft_s and gen > 1:
        tps_decode = (gen - 1) / (wall_s - ttft_s)
    ttft_ms = ttft_s * 1000 if ttft_s is not None else float("nan")
    print(
        f"metrics: prompt_tok={result.prompt_tokens} gen={gen} "
        f"TTFT={ttft_ms:.0f}ms board={board_s:.2f}s "
        f"tok/s_board={tps_board:.2f} tok/s_decode~{tps_decode:.2f}",
        flush=True,
    )
    return {
        "label": label,
        "wire_bytes": raw_b,
        "bypass": bypass,
        "src_tok_est": src_tok_est,
        "prompt_tok": result.prompt_tokens,
        "gen": gen,
        "ttft_ms": ttft_ms,
        "board_s": board_s,
        "tps_board": tps_board,
        "tps_decode": tps_decode,
        "text": result.text.strip().replace("\n", " / "),
    }


def main() -> int:
    long_text, long_tok = ensure_long_context()
    question = "Why do plant cells need chloroplasts?"
    short_prompt = format_chat_prompt([{"role": "user", "content": question}])
    long_prompt = format_chat_prompt(
        [
            {
                "role": "user",
                "content": (
                    f"CONTEXT:\n{long_text}\n\n"
                    f"QUESTION: {question}\n"
                    "Answer using only CONTEXT."
                ),
            }
        ]
    )
    print(
        f"prepared long: chars={len(long_text)} tok_est={long_tok} "
        f"wire={len(long_prompt.encode())}/{RAW_TEXT_MAX_BYTES}",
        flush=True,
    )

    rows: list[dict] = []
    with P4Device.connect("COM5", timeout=1200, reset=True) as device:
        ensure_ready(device, "pfor-180m.llmcraft")
        rows.append(run_case(device, "SHORT (<1024B, bypass)", short_prompt, None))
        rows.append(run_case(device, "LONG (~8k tok, compress)", long_prompt, long_tok))

    print("\n======== SUMMARY ========", flush=True)
    print(
        f"{'case':<28} {'wireB':>6} {'bypass':>6} {'srcTok':>6} {'ptok':>5} "
        f"{'TTFT_ms':>8} {'board_s':>8} {'tok/s':>7} {'decode~':>8}",
        flush=True,
    )
    for row in rows:
        src = row["src_tok_est"] if row["src_tok_est"] is not None else 0
        print(
            f"{row['label']:<28} {row['wire_bytes']:>6} {str(row['bypass']):>6} {src:>6} "
            f"{row['prompt_tok']:>5} {row['ttft_ms']:>8.0f} {row['board_s']:>8.2f} "
            f"{row['tps_board']:>7.2f} {row['tps_decode']:>8.2f}",
            flush=True,
        )
        print(f"  -> {row['text'][:140]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
