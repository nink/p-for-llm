#!/usr/bin/env python3
"""Phase 1 host-side context compression (~8:1 extractive budget trim).

Compresses long source text into a short packet that fits the PFor wire/prompt
budget. Intentionally simple: keep the question, prefer informative sentences,
drop filler until under budget. No ML compressor in Phase 1.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass


# Wire protocol caps the whole ChatML prompt at TEXT_MAX_BYTES (1024).
# Leave headroom for ChatML wrappers + reply generation on-device.
DEFAULT_PACKET_BYTES = 700
DEFAULT_TARGET_RATIO = 8.0

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\s]", re.UNICODE)
_NUMBER_RE = re.compile(r"\d")
_NAMEISH_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6}\s+\S|.{1,80}:)\s*$")


@dataclass(frozen=True)
class CompressResult:
    packet: str
    source_chars: int
    source_tokens_est: int
    packet_chars: int
    packet_tokens_est: int
    packet_bytes: int
    ratio_tokens: float
    kept_sentences: int
    dropped_sentences: int

    @property
    def ratio_chars(self) -> float:
        if self.packet_chars <= 0:
            return 0.0
        return self.source_chars / self.packet_chars


def estimate_tokens(text: str) -> int:
    """Cheap host-side token estimate (not the on-device tokenizer)."""

    if not text:
        return 0
    words = _WORD_RE.findall(text)
    # English-ish: ~0.75 words per token → tokens ≈ words / 0.75
    return max(1, int(round(len(words) / 0.75))) if words else 0


def _split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]
    return [part for part in parts if len(part) >= 8]


def _score_sentence(sentence: str, question_terms: set[str]) -> float:
    score = 1.0
    lower = sentence.lower()
    if _HEADING_RE.match(sentence) or sentence.isupper():
        score += 3.0
    if _NUMBER_RE.search(sentence):
        score += 2.0
    if _NAMEISH_RE.search(sentence):
        score += 1.0
    if any(term in lower for term in question_terms):
        score += 4.0
    # Prefer mid-length informative lines; penalize tiny fragments and huge walls.
    length = len(sentence)
    if 40 <= length <= 280:
        score += 1.0
    elif length < 20:
        score -= 1.0
    elif length > 500:
        score -= 2.0
    # Light redundancy penalty for boilerplate.
    if "as mentioned" in lower or "in conclusion" in lower or "click here" in lower:
        score -= 2.0
    return score


def _question_terms(question: str) -> set[str]:
    stop = {
        "a", "an", "the", "is", "are", "was", "were", "what", "which", "who", "whom",
        "whose", "where", "when", "why", "how", "do", "does", "did", "can", "could",
        "would", "should", "to", "of", "in", "on", "for", "and", "or", "with", "from",
        "about", "this", "that", "these", "those", "it", "its", "be", "as", "at", "by",
    }
    return {
        token.lower()
        for token in _WORD_RE.findall(question)
        if len(token) > 2 and token.lower() not in stop
    }


def compress_context(
    source: str,
    question: str,
    *,
    max_packet_bytes: int = DEFAULT_PACKET_BYTES,
    target_ratio: float = DEFAULT_TARGET_RATIO,
) -> CompressResult:
    """Compress ``source`` for ``question`` into a budgeted packet.

    Tries to approach ``target_ratio`` while never exceeding ``max_packet_bytes``.
    """

    if max_packet_bytes < 64:
        raise ValueError("max_packet_bytes must be at least 64")
    if target_ratio < 1.0:
        raise ValueError("target_ratio must be >= 1")

    source = source.strip()
    question = question.strip()
    if not question:
        raise ValueError("question is required")
    if not source:
        raise ValueError("source context is required")

    source_tokens = estimate_tokens(source)
    # Ideal packet size from ratio, but never above byte budget.
    ratio_budget_tokens = max(32, int(source_tokens / target_ratio))
    terms = _question_terms(question)
    sentences = _split_sentences(source)
    ranked = sorted(
        enumerate(sentences),
        key=lambda item: (_score_sentence(item[1], terms), -item[0]),
        reverse=True,
    )

    selected: dict[int, str] = {}
    for index, sentence in ranked:
        candidate_map = dict(selected)
        candidate_map[index] = sentence
        ordered = [candidate_map[i] for i in sorted(candidate_map)]
        body = "\n".join(ordered)
        packet = (
            f"CONTEXT:\n{body}\n\n"
            f"QUESTION: {question}\n"
            f"Answer using only CONTEXT."
        )
        packet_bytes = len(packet.encode("utf-8"))
        packet_tokens = estimate_tokens(packet)
        if packet_bytes > max_packet_bytes:
            continue
        # Prefer staying near target ratio once we have enough meat.
        if selected and packet_tokens > ratio_budget_tokens * 1.15:
            continue
        selected[index] = sentence

    if not selected:
        # Fallback: hard trim source head to fit budget.
        prefix = "CONTEXT:\n"
        suffix = f"\n\nQUESTION: {question}\nAnswer using only CONTEXT."
        room = max_packet_bytes - len(prefix.encode("utf-8")) - len(suffix.encode("utf-8"))
        clipped = source.encode("utf-8")[: max(0, room)].decode("utf-8", errors="ignore")
        packet = prefix + clipped + suffix
        kept = 0
        dropped = len(sentences)
    else:
        body = "\n".join(selected[i] for i in sorted(selected))
        packet = (
            f"CONTEXT:\n{body}\n\n"
            f"QUESTION: {question}\n"
            f"Answer using only CONTEXT."
        )
        kept = len(selected)
        dropped = max(0, len(sentences) - kept)

    packet_tokens = estimate_tokens(packet)
    ratio = (source_tokens / packet_tokens) if packet_tokens else 0.0
    return CompressResult(
        packet=packet,
        source_chars=len(source),
        source_tokens_est=source_tokens,
        packet_chars=len(packet),
        packet_tokens_est=packet_tokens,
        packet_bytes=len(packet.encode("utf-8")),
        ratio_tokens=ratio,
        kept_sentences=kept,
        dropped_sentences=dropped,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=argparse.FileType("r", encoding="utf-8"), required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-packet-bytes", type=int, default=DEFAULT_PACKET_BYTES)
    parser.add_argument("--target-ratio", type=float, default=DEFAULT_TARGET_RATIO)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.read()
    result = compress_context(
        source,
        args.question,
        max_packet_bytes=args.max_packet_bytes,
        target_ratio=args.target_ratio,
    )
    print(result.packet)
    print(
        f"\n# ratio~{result.ratio_tokens:.2f}:1 "
        f"src_tokens~{result.source_tokens_est} "
        f"pkt_tokens~{result.packet_tokens_est} "
        f"pkt_bytes={result.packet_bytes} "
        f"kept={result.kept_sentences} dropped={result.dropped_sentences}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
