"""Load locked local and formal pretraining sources without downloading them."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from data.manifest import hash_file, load_manifest

from .selection import DocumentSampling, select_document


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data/manifests/pretraining.json"
DEFAULT_TOKENIZER_PATH = (
    PROJECT_ROOT / "assets/qwen3.5-english-tokenizer/tokenizer.json"
)
PARQUET_CHUNK_ROWS = 4096


@dataclass(frozen=True, slots=True)
class DataSource:
    name: str
    path: Path
    file_format: str
    size_bytes: int
    sha256: str
    text_column: str | None = None
    document_separator: str | None = None
    document_sampling: DocumentSampling | None = None


@dataclass(frozen=True, slots=True)
class DataProfile:
    name: str
    purpose: str
    sources: tuple[DataSource, ...]


def discover_data_profile(root: Path) -> DataProfile:
    """Recursively discover complete supported training shards under ``root``."""
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"training data directory does not exist: {root}")

    sources: list[DataSource] = []
    for path in sorted(root.rglob("*")):
        relative_path = path.relative_to(root)
        if (
            not path.is_file()
            or path.name.endswith(".incomplete")
            or any(part.startswith(".") for part in relative_path.parts)
        ):
            continue
        if path.suffix == ".parquet":
            parquet = pq.ParquetFile(path)
            if "text" not in parquet.schema_arrow.names:
                # A mixed SFT root also contains conversation-only Parquet.
                continue
            file_format = "parquet"
            text_column = "text"
        elif path.suffix == ".zst":
            file_format = "jsonl_zstd"
            text_column = "text"
        else:
            continue
        sources.append(
            DataSource(
                name=relative_path.as_posix(),
                path=path,
                file_format=file_format,
                size_bytes=path.stat().st_size,
                sha256=hash_file(path),
                text_column=text_column,
            )
        )
    if not sources:
        raise ValueError(f"no supported complete training shards under: {root}")
    return DataProfile(
        name=f"directory-{root.name}",
        purpose="Discovered training shards",
        sources=tuple(sources),
    )


def _validate_source_file(source: DataSource, profile_name: str) -> None:
    if not source.path.is_file():
        raise FileNotFoundError(
            f"pretraining source is missing: {source.path}. "
            f"The '{profile_name}' profile never downloads data automatically."
        )
    if source.path.stat().st_size != source.size_bytes:
        raise ValueError(
            f"pretraining source size does not match manifest: {source.path}"
        )
    if hash_file(source.path) != source.sha256:
        raise ValueError(
            f"pretraining source hash does not match manifest: {source.path}"
        )


def load_data_profile(
    name: str,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    project_root: Path = PROJECT_ROOT,
) -> DataProfile:
    manifest = load_manifest(manifest_path)
    try:
        record = manifest["profiles"][name]
    except KeyError as error:
        raise ValueError(f"unknown pretraining data profile: {name}") from error

    sources: list[DataSource] = []
    for source_record in record["sources"]:
        source = DataSource(
            name=source_record["name"],
            path=project_root / source_record["local_path"],
            file_format=source_record["format"],
            size_bytes=int(source_record["size_bytes"]),
            sha256=source_record["sha256"],
            text_column=source_record.get("text_column"),
            document_separator=source_record.get("document_separator"),
            document_sampling=(
                DocumentSampling(
                    namespace=source_record["document_sampling"]["namespace"],
                    numerator=int(source_record["document_sampling"]["numerator"]),
                    denominator=int(source_record["document_sampling"]["denominator"]),
                )
                if "document_sampling" in source_record
                else None
            ),
        )
        if source.file_format not in {"text", "parquet", "jsonl_zstd"}:
            raise ValueError(
                f"unsupported pretraining source format: {source.file_format}"
            )
        _validate_source_file(source, name)
        sources.append(source)

    if not sources:
        raise ValueError(f"pretraining data profile has no sources: {name}")
    return DataProfile(name=name, purpose=record["purpose"], sources=tuple(sources))


def _iter_delimited_text(source: DataSource) -> Iterator[str]:
    if source.document_separator is None:
        raise ValueError(f"text source has no document separator: {source.name}")

    lines: list[str] = []
    with source.path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.rstrip("\r\n")
            if text == source.document_separator:
                document = "\n".join(lines)
                if document and select_document(document, source.document_sampling):
                    yield document
                lines.clear()
            else:
                lines.append(text)
    document = "\n".join(lines)
    if document and select_document(document, source.document_sampling):
        yield document


def _iter_parquet_text(
    source: DataSource,
    progress_callback: Callable[[int], None] | None = None,
) -> Iterator[str]:
    if source.text_column is None:
        raise ValueError(f"Parquet source has no text column: {source.name}")

    parquet = pq.ParquetFile(source.path)
    if source.text_column not in parquet.schema_arrow.names:
        raise ValueError(
            f"Parquet source is missing column {source.text_column!r}: {source.path}"
        )
    processed_rows = 0
    total_rows = parquet.metadata.num_rows
    for batch in parquet.iter_batches(
        batch_size=PARQUET_CHUNK_ROWS,
        columns=[source.text_column],
    ):
        for value in batch.column(0).to_pylist():
            if value is None or value == "":
                continue
            if not isinstance(value, str):
                raise ValueError(f"pretraining text is not a string: {source.name}")
            if select_document(value, source.document_sampling):
                yield value
        processed_rows += batch.num_rows
        if progress_callback is not None:
            progress_callback(source.size_bytes * processed_rows // total_rows)


def _iter_jsonl_zstd_text(
    source: DataSource,
    progress_callback: Callable[[int], None] | None = None,
) -> Iterator[str]:
    if source.text_column is None:
        raise ValueError(f"JSONL-Zstd source has no text column: {source.name}")
    with pa.OSFile(str(source.path), "rb") as raw:
        with pa.CompressedInputStream(raw, "zstd") as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    value = record.get(source.text_column)
                    if value is None or value == "":
                        continue
                    if not isinstance(value, str):
                        raise ValueError(
                            f"JSONL-Zstd text is not a string at "
                            f"{source.path}:{line_number}"
                        )
                    if select_document(value, source.document_sampling):
                        yield value
                    if (
                        progress_callback is not None
                        and line_number % PARQUET_CHUNK_ROWS == 0
                    ):
                        progress_callback(raw.tell())
    if progress_callback is not None:
        progress_callback(source.size_bytes)


def iter_source_documents(
    source: DataSource,
    progress_callback: Callable[[int], None] | None = None,
) -> Iterator[str]:
    if source.file_format == "text":
        yield from _iter_delimited_text(source)
        return
    if source.file_format == "parquet":
        yield from _iter_parquet_text(source, progress_callback)
        return
    if source.file_format == "jsonl_zstd":
        yield from _iter_jsonl_zstd_text(source, progress_callback)
        return
    raise ValueError(f"unsupported pretraining source format: {source.file_format}")
