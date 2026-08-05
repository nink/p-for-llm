"""Convert locked SQuAD v2 rows into deterministic short-answer SFT messages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from data.manifest import hash_file


FORMAT_VERSION = 1
REJECTION_FRACTION = 0.05
REJECTION_NAMESPACE = b"llmm-squad-v2-unanswerable-v1"
REJECTION_TEXT = "The answer is not provided in the context."
PARQUET_CHUNK_ROWS = 4096


def keep_unanswerable(sample_id: str) -> bool:
    digest = hashlib.sha256(
        REJECTION_NAMESPACE + b"\0" + sample_id.encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") < int(REJECTION_FRACTION * (1 << 64))


def convert_row(row: dict[str, Any]) -> tuple[dict[str, Any], bool] | None:
    sample_id = row.get("id")
    context = row.get("context")
    question = row.get("question")
    answers = row.get("answers")
    if not all(isinstance(value, str) and value.strip() for value in (sample_id, context, question)):
        raise ValueError("SQuAD v2 row has invalid id, context, or question")
    if not isinstance(answers, dict) or not isinstance(answers.get("text"), list):
        raise ValueError("SQuAD v2 row has invalid answers")

    answer_texts = [text.strip() for text in answers["text"] if isinstance(text, str) and text.strip()]
    unanswerable = not answer_texts
    if unanswerable and not keep_unanswerable(sample_id):
        return None
    answer = REJECTION_TEXT if unanswerable else answer_texts[0]
    prompt = f"Context:\n{context.strip()}\n\nQuestion: {question.strip()}"
    return (
        {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
        },
        unanswerable,
    )


def convert_squad_v2(source: Path, output: Path) -> dict[str, Any]:
    source_hash = hash_file(source)
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    contract = {
        "format_version": FORMAT_VERSION,
        "source_sha256": source_hash,
        "rejection_fraction": REJECTION_FRACTION,
        "rejection_namespace": REJECTION_NAMESPACE.decode("ascii"),
    }
    if output.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(metadata.get(key) == value for key, value in contract.items()):
            print(f"verified {output}")
            return metadata

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    counts = {"source": 0, "answerable": 0, "unanswerable": 0, "written": 0}
    parquet = pq.ParquetFile(source)
    columns = ["id", "context", "question", "answers"]
    with temporary_output.open("w", encoding="utf-8") as handle:
        for batch in parquet.iter_batches(batch_size=PARQUET_CHUNK_ROWS, columns=columns):
            for row in batch.to_pylist():
                counts["source"] += 1
                converted = convert_row(row)
                if converted is None:
                    continue
                record, unanswerable = converted
                counts["unanswerable" if unanswerable else "answerable"] += 1
                counts["written"] += 1
                handle.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")))
                handle.write("\n")
    metadata = {**contract, **counts, "output_bytes": temporary_output.stat().st_size}
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    temporary_output.replace(output)
    temporary_metadata.replace(metadata_path)
    return metadata
