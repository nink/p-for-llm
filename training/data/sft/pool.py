"""Build fixed-shape, masked-target SFT pools without cross-example packing."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import shutil
import struct
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tqdm import tqdm

from data.manifest import hash_file


POOL_FORMAT_VERSION = 2
TOKEN_DTYPE = np.dtype("<u2")
LENGTH_DTYPE = np.dtype("<u2")
BUCKET_LENGTHS = (256, 512, 1024)
RECORD_CHUNK_ROWS = 4096
DEFAULT_SFT_WORKERS = 8
SPLIT_NAMESPACE = "llmm-sft-v1"
EOS_TOKEN = "<|endoftext|>"
IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"
PoolSplit = Literal["train", "validation"]

_ROLE_MAP = {
    "assistant": "assistant",
    "bot": "assistant",
    "gpt": "assistant",
    "human": "user",
    "system": "system",
    "user": "user",
}

_WORKER_TOKENIZER: Tokenizer | None = None
_WORKER_PROGRESS_QUEUE: object | None = None


def _init_sft_worker(tokenizer_path: str, progress_queue: object) -> None:
    global _WORKER_TOKENIZER, _WORKER_PROGRESS_QUEUE
    _WORKER_TOKENIZER = Tokenizer.from_file(tokenizer_path)
    _WORKER_PROGRESS_QUEUE = progress_queue


@dataclass(frozen=True, slots=True)
class SFTExample:
    token_ids: tuple[int, ...]
    supervision: tuple[bool, ...]

    @property
    def length(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True, slots=True)
class SFTDataPool:
    path: Path
    fingerprint: str
    sequence_counts: dict[PoolSplit, dict[int, int]]

    def sequence_count(self, split: PoolSplit, bucket_length: int) -> int:
        return self.sequence_counts[split][bucket_length]

    def iter_batches(
        self,
        split: PoolSplit,
        bucket_length: int,
        batch_size: int,
        *,
        epoch: int = 0,
        seed: int = 42,
        shuffle: bool = False,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if bucket_length not in BUCKET_LENGTHS:
            raise ValueError(f"unsupported SFT bucket: {bucket_length}")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        count = self.sequence_count(split, bucket_length)
        usable_count = count // batch_size * batch_size
        if usable_count == 0:
            return

        tokens = np.memmap(
            self.path / f"{split}.{bucket_length}.tokens.bin",
            dtype=TOKEN_DTYPE,
            mode="c",
            shape=(count, bucket_length),
        )
        masks = np.memmap(
            self.path / f"{split}.{bucket_length}.mask.bin",
            dtype=np.uint8,
            mode="c",
            shape=(count, (bucket_length + 7) // 8),
        )
        lengths = np.memmap(
            self.path / f"{split}.{bucket_length}.lengths.bin",
            dtype=LENGTH_DTYPE,
            mode="c",
            shape=(count,),
        )
        order = np.arange(usable_count)
        if shuffle:
            np.random.default_rng(np.random.SeedSequence([seed, epoch, bucket_length])).shuffle(order)
        for start in range(0, usable_count, batch_size):
            indices = order[start : start + batch_size]
            batch_masks = np.unpackbits(masks[indices], axis=1, bitorder="little")[
                :, :bucket_length
            ]
            yield (
                np.asarray(tokens[indices]),
                np.asarray(batch_masks, dtype=np.bool_),
                np.asarray(lengths[indices]),
            )


def _encode(tokenizer: Tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False).ids


def _special_token_id(tokenizer: Tokenizer, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"tokenizer is missing required token: {token}")
    return token_id


def _normalise_messages(record: dict[str, Any]) -> list[tuple[str, str]] | None:
    raw_messages = record.get("messages", record.get("conversations"))
    if not isinstance(raw_messages, list):
        return None
    messages: list[tuple[str, str]] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            return None
        raw_role = item.get("role", item.get("from"))
        content = item.get("content", item.get("value"))
        if not isinstance(raw_role, str) or not isinstance(content, str):
            return None
        role = _ROLE_MAP.get(raw_role.strip().lower())
        if role is None or not content.strip():
            return None
        messages.append((role, content))
    if not messages or not any(role == "assistant" for role, _ in messages):
        return None
    return messages


def _contains_thinking(messages: Sequence[tuple[str, str]]) -> bool:
    return any(
        role == "assistant" and ("<think>" in text.lower() or "</think>" in text.lower())
        for role, text in messages
    )


def _canonical_messages(messages: Sequence[tuple[str, str]]) -> bytes:
    return json.dumps(
        [{"role": role, "content": text} for role, text in messages],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")


def build_sft_example(tokenizer: Tokenizer, messages: Sequence[tuple[str, str]]) -> SFTExample:
    """Encode one complete ChatML conversation and mark assistant-only targets."""
    im_start = _special_token_id(tokenizer, IM_START_TOKEN)
    im_end = _special_token_id(tokenizer, IM_END_TOKEN)
    eos = _special_token_id(tokenizer, EOS_TOKEN)
    token_ids: list[int] = []
    supervision: list[bool] = []
    newline = _encode(tokenizer, "\n")
    for role, content in messages:
        role_ids = _encode(tokenizer, role)
        content_ids = _encode(tokenizer, content)
        token_ids.extend((im_start, *role_ids, *newline, *content_ids, im_end, *newline))
        supervision.extend(
            (
                False,
                *([False] * len(role_ids)),
                *([False] * len(newline)),
                *([role == "assistant"] * len(content_ids)),
                role == "assistant",
                *([False] * len(newline)),
            )
        )
    token_ids.append(eos)
    supervision.append(True)
    return SFTExample(tuple(token_ids), tuple(supervision))


def build_event_sft_example(
    tokenizer: Tokenizer, events: Sequence[dict[str, Any]]
) -> SFTExample:
    """Encode a compact agent transcript using its persisted loss mask."""
    if not events:
        raise ValueError("agent events must not be empty")
    token_ids: list[int] = []
    supervision: list[bool] = []
    newline = _encode(tokenizer, "\n")
    for index, event in enumerate(events):
        text = event.get("text")
        loss = event.get("loss")
        if not isinstance(text, str) or not isinstance(loss, bool) or not text.isascii():
            raise ValueError("agent event requires ASCII text and a boolean loss flag")
        text_ids = _encode(tokenizer, text)
        token_ids.extend(text_ids)
        supervision.extend([loss] * len(text_ids))
        if index + 1 != len(events):
            token_ids.extend(newline)
            supervision.extend([loss] * len(newline))
    eos = _special_token_id(tokenizer, EOS_TOKEN)
    token_ids.append(eos)
    supervision.append(True)
    return SFTExample(tuple(token_ids), tuple(supervision))


def _iter_input_files(inputs: Iterable[Path]) -> Iterator[Path]:
    discovered: set[Path] = set()
    for value in inputs:
        if value.is_file():
            discovered.add(value.resolve())
        elif value.is_dir():
            for pattern in ("*.parquet", "*.jsonl"):
                discovered.update(
                    path.resolve()
                    for path in value.rglob(pattern)
                    if not any(
                        part.startswith(".")
                        for part in path.relative_to(value).parts
                    )
                )
        else:
            raise FileNotFoundError(f"SFT input does not exist: {value}")
    if not discovered:
        raise ValueError("no .parquet or .jsonl SFT inputs were found")
    yield from sorted(discovered, key=lambda path: str(path))


def _iter_records(
    path: Path,
    progress_callback: callable | None = None,
) -> Iterator[dict[str, Any]]:
    size_bytes = path.stat().st_size
    if path.suffix == ".parquet":
        parquet = pq.ParquetFile(path)
        columns = set(parquet.schema_arrow.names)
        if "events" in columns:
            selected = ["events"] + (["split"] if "split" in columns else [])
            processed_rows = 0
            total_rows = parquet.metadata.num_rows
            for batch in parquet.iter_batches(batch_size=RECORD_CHUNK_ROWS, columns=selected):
                values = batch.to_pylist()
                yield from values
                processed_rows += batch.num_rows
                if progress_callback is not None:
                    progress_callback(size_bytes * processed_rows // total_rows)
            return
        column = "messages" if "messages" in columns else "conversations"
        if column not in columns:
            # A complete SFT root may also contain causal-only shards. They
            # are intentionally ignored here; the SFT pool accepts only
            # conversation/event records.
            return
        processed_rows = 0
        total_rows = parquet.metadata.num_rows
        for batch in parquet.iter_batches(batch_size=RECORD_CHUNK_ROWS, columns=[column]):
            for value in batch.column(0).to_pylist():
                yield {column: value}
            processed_rows += batch.num_rows
            if progress_callback is not None:
                progress_callback(size_bytes * processed_rows // total_rows)
        return
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
                    if not isinstance(value, dict):
                        raise ValueError(f"JSONL record is not an object at {path}:{line_number}")
                    yield value
                if progress_callback is not None and line_number % RECORD_CHUNK_ROWS == 0:
                    progress_callback(handle.buffer.tell())
        if progress_callback is not None:
            progress_callback(size_bytes)
        return
    raise ValueError(f"unsupported SFT input type: {path}")


def _bucket_for_length(length: int) -> int | None:
    return next((bucket for bucket in BUCKET_LENGTHS if length <= bucket), None)


def _is_validation(canonical: bytes, validation_fraction: float) -> bool:
    digest = hashlib.sha256(SPLIT_NAMESPACE.encode("ascii") + b"\0" + canonical).digest()
    return int.from_bytes(digest[:8], "big") < int(validation_fraction * (1 << 64))


def _normalise_record(
    tokenizer: Tokenizer,
    record: dict[str, Any],
    validation_fraction: float,
    drop_thinking: bool,
) -> tuple[SFTExample, PoolSplit] | None:
    if "events" in record:
        try:
            example = build_event_sft_example(tokenizer, record["events"])
        except ValueError:
            return None
        split_group = record.get("split_group")
        if split_group is None:
            canonical = json.dumps(
                record["events"], ensure_ascii=True, separators=(",", ":")
            ).encode("ascii")
        elif isinstance(split_group, str) and split_group and split_group.isascii():
            canonical = b"agent-group\0" + split_group.encode("ascii")
        else:
            return None
        requested_split = record.get("split")
        split: PoolSplit = (
            requested_split
            if requested_split in {"train", "validation"}
            else (
                "validation"
                if _is_validation(canonical, validation_fraction)
                else "train"
            )
        )
        return example, split

    messages = _normalise_messages(record)
    if messages is None or (drop_thinking and _contains_thinking(messages)):
        return None
    example = build_sft_example(tokenizer, messages)
    split = (
        "validation"
        if _is_validation(_canonical_messages(messages), validation_fraction)
        else "train"
    )
    return example, split


class _SFTWriter:
    def __init__(self, path: Path, bucket_length: int, pad_token_id: int) -> None:
        self._bucket_length = bucket_length
        self._pad_token_id = pad_token_id
        self._tokens = (path.parent / f"{path.name}.tokens.bin").open("wb")
        self._mask = (path.parent / f"{path.name}.mask.bin").open("wb")
        self._lengths = (path.parent / f"{path.name}.lengths.bin").open("wb")
        self.count = 0

    def add(self, example: SFTExample) -> None:
        padding = self._bucket_length - example.length
        if padding < 0:
            raise ValueError("SFT example exceeds its assigned bucket")
        np.asarray(
            (*example.token_ids, *([self._pad_token_id] * padding)), dtype=TOKEN_DTYPE
        ).tofile(self._tokens)
        mask = np.asarray((*example.supervision, *([False] * padding)), dtype=np.uint8)
        np.packbits(mask, bitorder="little").tofile(self._mask)
        self._lengths.write(struct.pack("<H", example.length))
        self.count += 1

    def close(self) -> None:
        self._tokens.close()
        self._mask.close()
        self._lengths.close()


def _process_sft_source(
    task: tuple[int, str, str, float, bool],
) -> tuple[int, dict[str, int], dict[str, dict[int, int]]]:
    source_index, source_name, worker_root, validation_fraction, drop_thinking = task
    tokenizer = _WORKER_TOKENIZER
    if tokenizer is None:
        raise RuntimeError("SFT worker tokenizer was not initialized")
    source_path = Path(source_name)
    output_path = Path(worker_root) / f"{source_index:06d}"
    output_path.mkdir(parents=True, exist_ok=True)
    eos_token_id = _special_token_id(tokenizer, EOS_TOKEN)
    writers = {
        (split, bucket): _SFTWriter(output_path / f"{split}.{bucket}", bucket, eos_token_id)
        for split in ("train", "validation")
        for bucket in BUCKET_LENGTHS
    }
    statistics = {
        "read_records": 0,
        "accepted_records": 0,
        "dropped_invalid": 0,
        "dropped_thinking": 0,
        "dropped_overlength": 0,
    }

    def report_progress(processed_bytes: int) -> None:
        if _WORKER_PROGRESS_QUEUE is not None:
            _WORKER_PROGRESS_QUEUE.put((source_index, processed_bytes))

    try:
        for record in _iter_records(source_path, report_progress):
            statistics["read_records"] += 1
            if "events" not in record:
                messages = _normalise_messages(record)
                if messages is None:
                    statistics["dropped_invalid"] += 1
                    continue
                if drop_thinking and _contains_thinking(messages):
                    statistics["dropped_thinking"] += 1
                    continue
            normalized = _normalise_record(
                tokenizer,
                record,
                validation_fraction,
                drop_thinking,
            )
            if normalized is None:
                statistics["dropped_invalid"] += 1
                continue
            example, split = normalized
            bucket = _bucket_for_length(example.length)
            if bucket is None:
                statistics["dropped_overlength"] += 1
                continue
            writers[(split, bucket)].add(example)
            statistics["accepted_records"] += 1
    finally:
        for writer in writers.values():
            writer.close()
    counts = {
        split: {bucket: writers[(split, bucket)].count for bucket in BUCKET_LENGTHS}
        for split in ("train", "validation")
    }
    return source_index, statistics, counts


def _append_binary_file(source_path: Path, destination) -> None:
    with source_path.open("rb") as source:
        offset = 0
        remaining = source_path.stat().st_size
        while remaining:
            sent = os.sendfile(destination.fileno(), source.fileno(), offset, remaining)
            if sent == 0:
                raise OSError(f"failed to append SFT worker output: {source_path}")
            offset += sent
            remaining -= sent


def _pool_contract(
    input_files: Sequence[Path],
    tokenizer_path: Path,
    validation_fraction: float,
    drop_thinking: bool,
) -> dict[str, object]:
    return {
        "format_version": POOL_FORMAT_VERSION,
        "buckets": list(BUCKET_LENGTHS),
        "chat_format": "chatml-and-masked-agent-events-v2",
        "drop_thinking": drop_thinking,
        "padding": "right-masked-eos-id-v1",
        "split_namespace": SPLIT_NAMESPACE,
        "tokenizer_sha256": hash_file(tokenizer_path),
        "validation_fraction": validation_fraction,
        "sources": [
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": hash_file(path),
            }
            for path in input_files
        ],
    }


def _fingerprint(contract: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_pool(path: Path, contract: dict[str, object], fingerprint: str) -> SFTDataPool | None:
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("contract") != contract:
        raise ValueError(f"SFT pool contract does not match: {path}")
    counts = metadata.get("sequence_counts")
    if not isinstance(counts, dict):
        raise ValueError(f"SFT pool metadata is missing sequence counts: {path}")
    result: dict[PoolSplit, dict[int, int]] = {"train": {}, "validation": {}}
    for split in ("train", "validation"):
        split_counts = counts.get(split)
        if not isinstance(split_counts, dict):
            raise ValueError(f"SFT pool metadata is missing {split} counts: {path}")
        for bucket in BUCKET_LENGTHS:
            count = int(split_counts.get(str(bucket), 0))
            expected_files = {
                f"{split}.{bucket}.tokens.bin": count * bucket * TOKEN_DTYPE.itemsize,
                f"{split}.{bucket}.mask.bin": count * ((bucket + 7) // 8),
                f"{split}.{bucket}.lengths.bin": count * LENGTH_DTYPE.itemsize,
            }
            for name, expected_size in expected_files.items():
                data_path = path / name
                if not data_path.is_file() or data_path.stat().st_size != expected_size:
                    raise ValueError(f"SFT pool file is incomplete: {data_path}")
            result[split][bucket] = count
    return SFTDataPool(path=path, fingerprint=fingerprint, sequence_counts=result)


def load_sft_pool(path: Path) -> SFTDataPool:
    """Load and validate an already materialized SFT pool."""
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"SFT pool metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    contract = metadata.get("contract")
    if not isinstance(contract, dict):
        raise ValueError(f"SFT pool contract is missing: {path}")
    fingerprint = _fingerprint(contract)
    if metadata.get("fingerprint") != fingerprint:
        raise ValueError(f"SFT pool fingerprint does not match contract: {path}")
    pool = _load_pool(path, contract, fingerprint)
    if pool is None:
        raise ValueError(f"SFT pool metadata could not be loaded: {path}")
    return pool


def load_or_prepare_sft_pool(
    inputs: Iterable[Path],
    tokenizer_path: Path,
    output_path: Path,
    *,
    validation_fraction: float = 0.01,
    drop_thinking: bool = True,
    workers: int = DEFAULT_SFT_WORKERS,
) -> SFTDataPool:
    """Materialize a reusable SFT pool, or validate and reuse an identical one."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    if workers <= 0:
        raise ValueError("workers must be positive")
    input_files = tuple(_iter_input_files(inputs))
    contract = _pool_contract(input_files, tokenizer_path, validation_fraction, drop_thinking)
    fingerprint = _fingerprint(contract)
    existing = _load_pool(output_path, contract, fingerprint)
    if existing is not None:
        return existing
    if output_path.exists():
        raise FileExistsError(f"SFT output path exists without a matching pool: {output_path}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size(with_added_tokens=True) > 1 << 16:
        raise ValueError("SFT pool uint16 format supports at most 65,536 tokens")
    eos_token_id = _special_token_id(tokenizer, EOS_TOKEN)
    _special_token_id(tokenizer, IM_START_TOKEN)
    _special_token_id(tokenizer, IM_END_TOKEN)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=output_path.parent))
    worker_root = temporary_path / ".workers"
    worker_root.mkdir()
    statistics = {
        "read_records": 0,
        "accepted_records": 0,
        "dropped_invalid": 0,
        "dropped_thinking": 0,
        "dropped_overlength": 0,
    }
    progress = tqdm(desc="packing sft", unit="conversation", dynamic_ncols=True)
    completion = tqdm(
        total=sum(path.stat().st_size for path in input_files),
        desc="packing sft ETA",
        unit="B",
        unit_scale=True,
        dynamic_ncols=True,
    )
    previous_rayon_threads = os.environ.get("RAYON_NUM_THREADS")
    try:
        # Each worker writes compact bucket bytes directly. The parent only
        # concatenates those bytes in source order, so worker completion order
        # cannot affect the materialized pool.
        os.environ["RAYON_NUM_THREADS"] = "1"
        context = mp.get_context("fork")
        progress_queue = context.Queue()
        tasks = [
            (index, str(path), str(worker_root), validation_fraction, drop_thinking)
            for index, path in enumerate(input_files)
        ]
        source_sizes = {index: path.stat().st_size for index, path in enumerate(input_files)}
        reported_bytes = {index: 0 for index in range(len(input_files))}
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)),
            mp_context=context,
            initializer=_init_sft_worker,
            initargs=(str(tokenizer_path), progress_queue),
        ) as executor:
            futures = {executor.submit(_process_sft_source, task): task[0] for task in tasks}
            pending = set(futures)
            completed: dict[int, tuple[dict[str, int], dict[str, dict[int, int]]]] = {}
            while pending:
                try:
                    source_index, processed_bytes = progress_queue.get(timeout=0.25)
                    processed_bytes = min(processed_bytes, source_sizes[source_index])
                    completion.update(max(0, processed_bytes - reported_bytes[source_index]))
                    reported_bytes[source_index] = max(reported_bytes[source_index], processed_bytes)
                except Empty:
                    pass
                done, pending = wait(pending, timeout=0, return_when=FIRST_COMPLETED)
                for future in done:
                    source_index, source_statistics, counts = future.result()
                    completed[source_index] = (source_statistics, counts)
                    completion.update(source_sizes[source_index] - reported_bytes[source_index])
                    reported_bytes[source_index] = source_sizes[source_index]

        output_handles = {
            (split, bucket, suffix): (
                temporary_path / f"{split}.{bucket}.{suffix}.bin"
            ).open("wb")
            for split in ("train", "validation")
            for bucket in BUCKET_LENGTHS
            for suffix in ("tokens", "mask", "lengths")
        }
        sequence_counts = {
            split: {str(bucket): 0 for bucket in BUCKET_LENGTHS}
            for split in ("train", "validation")
        }
        try:
            for source_index in range(len(input_files)):
                source_statistics, counts = completed[source_index]
                for name, value in source_statistics.items():
                    statistics[name] += value
                progress.update(source_statistics["accepted_records"])
                source_path = worker_root / f"{source_index:06d}"
                for split in ("train", "validation"):
                    for bucket in BUCKET_LENGTHS:
                        for suffix in ("tokens", "mask", "lengths"):
                            _append_binary_file(
                                source_path / f"{split}.{bucket}.{suffix}.bin",
                                output_handles[(split, bucket, suffix)],
                            )
                        sequence_counts[split][str(bucket)] += counts[split][bucket]
        finally:
            for handle in output_handles.values():
                handle.close()
    finally:
        progress.close()
        completion.close()
        if previous_rayon_threads is None:
            os.environ.pop("RAYON_NUM_THREADS", None)
        else:
            os.environ["RAYON_NUM_THREADS"] = previous_rayon_threads
        shutil.rmtree(worker_root, ignore_errors=True)

    metadata = {
        "contract": contract,
        "fingerprint": fingerprint,
        "sequence_counts": sequence_counts,
        "statistics": statistics,
    }
    (temporary_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_path, output_path)
    pool = _load_pool(output_path, contract, fingerprint)
    if pool is None:
        raise RuntimeError("newly created SFT pool failed validation")
    return pool
