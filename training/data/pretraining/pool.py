"""Build and iterate a reusable packed token pool."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import random
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Iterator, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from tokenizers import Tokenizer

from data.manifest import hash_file

from .packing import EOS_TOKEN
from .selection import DocumentSampling, select_document
from .sources import DataProfile, DataSource, iter_source_documents


POOL_FORMAT_VERSION = 1
TOKEN_DTYPE = np.dtype("<u2")
TOKENIZATION_CHUNK_DOCUMENTS = 4096
LARGE_ZSTD_CHUNK_DOCUMENTS = 512
# Write complete sequences in bounded chunks instead of performing one file
# write per document. This is deliberately measured in tokens so it stays
# useful when the sequence length changes.
TOKEN_WRITE_BUFFER_TOKENS = 65_536
PARALLEL_ZSTD_MIN_BYTES = 512 * 1024 * 1024
PoolSplit = Literal["train", "validation"]

_WORKER_TOKENIZER: Tokenizer | None = None
_WORKER_PROGRESS_QUEUE: object | None = None


def _init_packing_worker(
    tokenizer_path: str,
    progress_queue: object | None = None,
) -> None:
    global _WORKER_PROGRESS_QUEUE, _WORKER_TOKENIZER
    _WORKER_TOKENIZER = Tokenizer.from_file(tokenizer_path)
    _WORKER_PROGRESS_QUEUE = progress_queue


def _available_packing_workers() -> int:
    available = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    if cpu_max.is_file():
        quota, period = cpu_max.read_text(encoding="utf-8").split()
        if quota != "max":
            return min(available, max(1, (int(quota) + int(period) - 1) // int(period)))

    quota_path = Path("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us")
    period_path = Path("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us")
    if quota_path.is_file() and period_path.is_file():
        quota = int(quota_path.read_text(encoding="utf-8"))
        period = int(period_path.read_text(encoding="utf-8"))
        if quota > 0:
            return min(available, max(1, (quota + period - 1) // period))
    return available


def _pack_row_group_task(
    task: tuple[
        str,
        str,
        list[int],
        int,
        int,
        int,
        int,
        str,
        DocumentSampling | None,
        str,
        int,
    ],
) -> tuple[int, int, dict[str, str], dict[str, list[int]]]:
    (
        source_path,
        text_column,
        row_groups,
        threshold,
        eos_token_id,
        sequence_length,
        task_index,
        output_dir,
        document_sampling,
        task_signature,
        tokenization_chunk_documents,
    ) = task
    tokenizer = _WORKER_TOKENIZER
    if tokenizer is None:
        raise RuntimeError("packing worker was not initialized")

    output_root = Path(output_dir)
    paths = {
        split: output_root / f"{task_index:04d}.{split}.bin"
        for split in ("train", "validation")
    }
    pending = {"train": [], "validation": []}
    document_count = 0

    with paths["train"].open("wb") as train_handle, paths["validation"].open(
        "wb"
    ) as validation_handle:
        handles = {"train": train_handle, "validation": validation_handle}

        def flush(split: str) -> None:
            tokens = pending[split]
            complete = len(tokens) // sequence_length * sequence_length
            if complete:
                np.asarray(tokens[:complete], dtype=TOKEN_DTYPE).tofile(handles[split])
                del tokens[:complete]

        parquet = pq.ParquetFile(source_path)
        for batch in parquet.iter_batches(
            batch_size=tokenization_chunk_documents,
            row_groups=row_groups,
            columns=[text_column],
        ):
            documents = [
                document
                for document in batch.column(0).to_pylist()
                if isinstance(document, str)
                and document
                and select_document(document, document_sampling)
            ]
            if not documents:
                continue
            encodings = tokenizer.encode_batch(documents, add_special_tokens=False)
            for document, encoding in zip(documents, encodings, strict=True):
                digest = hashlib.sha256(document.encode("utf-8")).digest()
                split = (
                    "validation"
                    if int.from_bytes(digest[:8], "big") < threshold
                    else "train"
                )
                pending[split].extend(encoding.ids)
                pending[split].append(eos_token_id)
            flush("train")
            flush("validation")
            document_count += len(documents)

    result = (
        task_index,
        document_count,
        {split: str(path) for split, path in paths.items()},
        pending,
    )
    _write_task_result(output_root, task_index, task_signature, result)
    return result


def _pack_source_task(
    task: tuple[
        DataSource,
        int,
        int,
        int,
        int,
        str,
        str,
    ],
) -> tuple[int, int, dict[str, str], dict[str, list[int]]]:
    (
        source,
        threshold,
        eos_token_id,
        sequence_length,
        task_index,
        output_dir,
        task_signature,
    ) = task
    tokenizer = _WORKER_TOKENIZER
    if tokenizer is None:
        raise RuntimeError("packing worker was not initialized")

    output_root = Path(output_dir)
    paths = {
        split: output_root / f"{task_index:04d}.{split}.bin"
        for split in ("train", "validation")
    }
    pending = {"train": [], "validation": []}
    document_count = 0

    with paths["train"].open("wb") as train_handle, paths["validation"].open(
        "wb"
    ) as validation_handle:
        handles = {"train": train_handle, "validation": validation_handle}

        def flush(split: str) -> None:
            tokens = pending[split]
            complete = len(tokens) // sequence_length * sequence_length
            if complete:
                np.asarray(tokens[:complete], dtype=TOKEN_DTYPE).tofile(handles[split])
                del tokens[:complete]

        def report_progress(processed_bytes: int) -> None:
            if _WORKER_PROGRESS_QUEUE is not None:
                _WORKER_PROGRESS_QUEUE.put((task_index, processed_bytes))

        documents: list[str] = []
        for document in iter_source_documents(source, report_progress):
            documents.append(document)
            if len(documents) != TOKENIZATION_CHUNK_DOCUMENTS:
                continue
            encodings = tokenizer.encode_batch(documents, add_special_tokens=False)
            for value, encoding in zip(documents, encodings, strict=True):
                digest = hashlib.sha256(value.encode("utf-8")).digest()
                split = (
                    "validation"
                    if int.from_bytes(digest[:8], "big") < threshold
                    else "train"
                )
                pending[split].extend(encoding.ids)
                pending[split].append(eos_token_id)
            document_count += len(documents)
            documents.clear()
            flush("train")
            flush("validation")
        if documents:
            encodings = tokenizer.encode_batch(documents, add_special_tokens=False)
            for value, encoding in zip(documents, encodings, strict=True):
                digest = hashlib.sha256(value.encode("utf-8")).digest()
                split = (
                    "validation"
                    if int.from_bytes(digest[:8], "big") < threshold
                    else "train"
                )
                pending[split].extend(encoding.ids)
                pending[split].append(eos_token_id)
            document_count += len(documents)
            flush("train")
            flush("validation")

    result = (
        task_index,
        document_count,
        {split: str(path) for split, path in paths.items()},
        pending,
    )
    _write_task_result(output_root, task_index, task_signature, result)
    return result


def _task_result_path(output_root: Path, task_index: int) -> Path:
    return output_root / f"{task_index:04d}.json"


def _write_task_result(
    output_root: Path,
    task_index: int,
    signature: str,
    result: tuple[int, int, dict[str, str], dict[str, list[int]]],
) -> None:
    _, documents, paths, remainders = result
    result_path = _task_result_path(output_root, task_index)
    temporary_path = result_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(
            {
                "signature": signature,
                "documents": documents,
                "paths": paths,
                "remainders": remainders,
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    temporary_path.replace(result_path)


def _load_task_result(
    output_root: Path,
    task_index: int,
    signature: str,
) -> tuple[int, int, dict[str, str], dict[str, list[int]]] | None:
    result_path = _task_result_path(output_root, task_index)
    if not result_path.is_file():
        return None
    record = json.loads(result_path.read_text(encoding="utf-8"))
    paths = {str(split): str(path) for split, path in record["paths"].items()}
    if record.get("signature") != signature or not all(
        Path(path).is_file() for path in paths.values()
    ):
        result_path.unlink(missing_ok=True)
        return None
    return (
        task_index,
        int(record["documents"]),
        paths,
        {
            str(split): [int(token) for token in tokens]
            for split, tokens in record["remainders"].items()
        },
    )


def _prepare_zstd_parquet_cache(
    source: DataSource,
    output_root: Path,
    task_index: int,
) -> Path:
    cache_path = output_root / f"{task_index:04d}.expanded.parquet"
    marker_path = output_root / f"{task_index:04d}.expanded.json"
    if marker_path.is_file() and cache_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("source_sha256") == source.sha256:
            return cache_path

    temporary_path = cache_path.with_suffix(".parquet.tmp")
    temporary_path.unlink(missing_ok=True)
    schema = pa.schema([("text", pa.string())])
    writer = pq.ParquetWriter(temporary_path, schema, compression="NONE")
    documents: list[str] = []
    processed_bytes = 0
    progress = tqdm(
        total=source.size_bytes,
        desc=f"splitting {source.name}",
        unit="B",
        unit_scale=True,
        dynamic_ncols=True,
    )

    def report_progress(value: int) -> None:
        nonlocal processed_bytes
        value = min(value, source.size_bytes)
        progress.update(max(0, value - processed_bytes))
        processed_bytes = max(processed_bytes, value)

    try:
        for document in iter_source_documents(source, report_progress):
            documents.append(document)
            if len(documents) == TOKENIZATION_CHUNK_DOCUMENTS:
                writer.write_table(pa.table({"text": documents}, schema=schema))
                documents.clear()
        if documents:
            writer.write_table(pa.table({"text": documents}, schema=schema))
    finally:
        writer.close()
        progress.close()
    temporary_path.replace(cache_path)
    temporary_marker = marker_path.with_suffix(".json.tmp")
    temporary_marker.write_text(
        json.dumps({"source_sha256": source.sha256}, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary_marker.replace(marker_path)
    return cache_path


def _merge_task_results(
    output_root: Path,
    task_index: int,
    signature: str,
    sequence_length: int,
    results: list[tuple[int, int, dict[str, str], dict[str, list[int]]]],
) -> tuple[int, int, dict[str, str], dict[str, list[int]]]:
    paths = {
        split: output_root / f"{task_index:04d}.{split}.bin"
        for split in ("train", "validation")
    }
    pending = {"train": [], "validation": []}
    document_count = 0
    with paths["train"].open("wb") as train_handle, paths["validation"].open(
        "wb"
    ) as validation_handle:
        handles = {"train": train_handle, "validation": validation_handle}

        def flush(split: str) -> None:
            tokens = pending[split]
            complete = len(tokens) // sequence_length * sequence_length
            if complete:
                np.asarray(tokens[:complete], dtype=TOKEN_DTYPE).tofile(handles[split])
                del tokens[:complete]

        for _, documents, part_paths, remainders in results:
            document_count += documents
            for split in ("train", "validation"):
                with Path(part_paths[split]).open("rb") as handle:
                    while True:
                        chunk = np.fromfile(
                            handle,
                            dtype=TOKEN_DTYPE,
                            count=TOKEN_WRITE_BUFFER_TOKENS,
                        )
                        if chunk.size == 0:
                            break
                        pending[split].extend(chunk.tolist())
                        flush(split)
                pending[split].extend(remainders[split])
                flush(split)

    result = (
        task_index,
        document_count,
        {split: str(path) for split, path in paths.items()},
        pending,
    )
    _write_task_result(output_root, task_index, signature, result)
    return result


def _pack_large_zstd_task(
    task: tuple[DataSource, int, int, int, int, str, str],
    tokenizer_path: Path,
    worker_count: int,
) -> None:
    source, threshold, eos_token_id, sequence_length, task_index, output_dir, signature = task
    output_root = Path(output_dir)
    if _load_task_result(output_root, task_index, signature) is not None:
        return

    cache_path = _prepare_zstd_parquet_cache(source, output_root, task_index)
    part_root = output_root / f"{task_index:04d}.parts"
    part_root.mkdir(exist_ok=True)
    row_group_count = pq.ParquetFile(cache_path).num_row_groups
    chunk_size = 1
    part_tasks = []
    for part_index, start in enumerate(range(0, row_group_count, chunk_size)):
        stop = min(start + chunk_size, row_group_count)
        part_signature = f"{signature}:part:{start}:{stop}"
        part_tasks.append(
            (
                str(cache_path),
                "text",
                list(range(start, stop)),
                threshold,
                eos_token_id,
                sequence_length,
                part_index,
                str(part_root),
                None,
                part_signature,
                LARGE_ZSTD_CHUNK_DOCUMENTS,
            )
        )

    results: dict[int, tuple[int, int, dict[str, str], dict[str, list[int]]]] = {}
    pending_tasks = []
    for part_task in part_tasks:
        cached = _load_task_result(part_root, part_task[6], part_task[9])
        if cached is None:
            pending_tasks.append(part_task)
        else:
            results[part_task[6]] = cached

    executor: ProcessPoolExecutor | None = None
    progress = tqdm(
        total=len(part_tasks),
        initial=len(results),
        desc=f"packing {source.name}",
        unit="part",
        dynamic_ncols=True,
    )
    try:
        if pending_tasks:
            executor = ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=mp.get_context("spawn"),
                initializer=_init_packing_worker,
                initargs=(str(tokenizer_path),),
            )
            for result in executor.map(_pack_row_group_task, pending_tasks):
                results[result[0]] = result
                progress.update(1)
    except KeyboardInterrupt:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
            executor = None
        raise
    finally:
        progress.close()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    ordered = [results[index] for index in range(len(part_tasks))]
    _merge_task_results(
        output_root,
        task_index,
        signature,
        sequence_length,
        ordered,
    )
    for part_index, _, part_paths, _ in ordered:
        for path in part_paths.values():
            Path(path).unlink(missing_ok=True)
        _task_result_path(part_root, part_index).unlink(missing_ok=True)
    part_root.rmdir()
    cache_path.unlink(missing_ok=True)
    (output_root / f"{task_index:04d}.expanded.json").unlink(missing_ok=True)

@dataclass(frozen=True, slots=True)
class PackedDataPool:
    path: Path
    fingerprint: str
    sequence_length: int
    train_sequences: int
    validation_sequences: int

    def sequence_count(self, split: PoolSplit) -> int:
        return (
            self.train_sequences
            if split == "train"
            else self.validation_sequences
        )

    def batch_count(self, split: PoolSplit, batch_size: int) -> int:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return self.sequence_count(split) // batch_size

    def iter_batches(
        self,
        split: PoolSplit,
        batch_size: int,
        *,
        epoch: int = 0,
        seed: int = 42,
        start_batch: int = 0,
        stop_batch: int | None = None,
        shuffle: bool = False,
    ) -> Iterator[np.ndarray]:
        batch_count = self.batch_count(split, batch_size)
        stop = batch_count if stop_batch is None else stop_batch
        if not 0 <= start_batch <= stop <= batch_count:
            raise ValueError("batch range exceeds the packed pool")
        if start_batch == stop:
            return

        order = list(range(batch_count))
        if shuffle:
            random.Random(seed + epoch).shuffle(order)

        tokens = np.memmap(
            self.path / f"{split}.bin",
            dtype=TOKEN_DTYPE,
            mode="c",
            shape=(self.sequence_count(split), self.sequence_length),
        )
        for order_index in range(start_batch, stop):
            batch_index = order[order_index]
            begin = batch_index * batch_size
            yield np.asarray(tokens[begin : begin + batch_size])


class _PackedSequenceWriter:
    def __init__(self, path: Path, sequence_length: int) -> None:
        self._handle = path.open("wb")
        self._sequence_length = sequence_length
        self._pending: list[int] = []
        self.sequence_count = 0

    def add_document(self, token_ids: list[int], eos_token_id: int) -> None:
        self._pending.extend(token_ids)
        self._pending.append(eos_token_id)
        self._flush_complete_sequences()

    def add_tokens(self, token_ids: list[int]) -> None:
        self._pending.extend(token_ids)
        self._flush_complete_sequences()

    def _flush_complete_sequences(self, *, force: bool = False) -> None:
        complete_tokens = (
            len(self._pending) // self._sequence_length * self._sequence_length
        )
        if complete_tokens == 0:
            return

        chunk_tokens = max(
            self._sequence_length,
            TOKEN_WRITE_BUFFER_TOKENS
            // self._sequence_length
            * self._sequence_length,
        )
        written_tokens = 0
        while complete_tokens >= chunk_tokens or (force and complete_tokens > 0):
            write_tokens = min(complete_tokens, chunk_tokens)
            np.asarray(
                self._pending[written_tokens : written_tokens + write_tokens],
                dtype=TOKEN_DTYPE,
            ).tofile(self._handle)
            self.sequence_count += write_tokens // self._sequence_length
            written_tokens += write_tokens
            complete_tokens -= write_tokens
        if written_tokens:
            del self._pending[:written_tokens]

    @property
    def dropped_tokens(self) -> int:
        return len(self._pending)

    def close(self) -> None:
        self._flush_complete_sequences(force=True)
        self._handle.close()


def _default_pool_root(profile: DataProfile) -> Path:
    source_parents = [str(source.path.resolve().parent) for source in profile.sources]
    return Path(os.path.commonpath(source_parents)) / ".llmm-pools"


def _pool_contract(
    profile: DataProfile,
    tokenizer_path: Path,
    sequence_length: int,
    validation_fraction: float,
) -> dict[str, object]:
    return {
        "format_version": POOL_FORMAT_VERSION,
        "profile": profile.name,
        "sequence_length": sequence_length,
        "validation_fraction": validation_fraction,
        "tokenizer_sha256": hash_file(tokenizer_path),
        "sources": [
            {
                "name": source.name,
                "sha256": source.sha256,
                "format": source.file_format,
                "text_column": source.text_column,
                "document_separator": source.document_separator,
                **(
                    {
                        "document_sampling": source.document_sampling.as_dict(),
                    }
                    if source.document_sampling is not None
                    else {}
                ),
            }
            for source in profile.sources
        ],
    }


def _fingerprint(contract: dict[str, object]) -> str:
    encoded = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_pool(
    pool_path: Path,
    contract: dict[str, object],
    fingerprint: str,
) -> PackedDataPool | None:
    metadata_path = pool_path / "metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("contract") != contract:
        raise ValueError(f"packed pool contract does not match: {pool_path}")

    sequence_length = int(contract["sequence_length"])
    counts = metadata["sequence_counts"]
    for split in ("train", "validation"):
        expected_size = int(counts[split]) * sequence_length * TOKEN_DTYPE.itemsize
        path = pool_path / f"{split}.bin"
        if not path.is_file() or path.stat().st_size != expected_size:
            raise ValueError(f"packed pool file is incomplete: {path}")
    return PackedDataPool(
        path=pool_path,
        fingerprint=fingerprint,
        sequence_length=sequence_length,
        train_sequences=int(counts["train"]),
        validation_sequences=int(counts["validation"]),
    )


def _iter_document_batches(profile: DataProfile) -> Iterator[list[str]]:
    for source in profile.sources:
        yield from _iter_source_document_batches(source)


def _iter_source_document_batches(source) -> Iterator[list[str]]:
    pending: list[str] = []
    for document in iter_source_documents(source):
        pending.append(document)
        if len(pending) == TOKENIZATION_CHUNK_DOCUMENTS:
            yield pending
            pending = []
    if pending:
        yield pending


def _prepare_pool_files(
    profile: DataProfile,
    tokenizer: Tokenizer,
    tokenizer_path: Path,
    pool_path: Path,
    sequence_length: int,
    validation_fraction: float,
    contract: dict[str, object],
) -> None:
    eos_token_id = tokenizer.token_to_id(EOS_TOKEN)
    if eos_token_id is None:
        raise ValueError(f"tokenizer is missing EOS token: {EOS_TOKEN}")
    if tokenizer.get_vocab_size(with_added_tokens=True) > 1 << 16:
        raise ValueError("packed pool uint16 format supports at most 65,536 tokens")

    pool_path.mkdir(parents=True, exist_ok=True)
    temporary_paths = {
        split: pool_path / f".{split}.bin.tmp"
        for split in ("train", "validation")
    }
    writers = {
        split: _PackedSequenceWriter(path, sequence_length)
        for split, path in temporary_paths.items()
    }
    threshold = int(validation_fraction * (1 << 64))
    document_count = 0
    progress = tqdm(
        desc=f"packing {profile.name}",
        unit="doc",
        dynamic_ncols=True,
    )

    worker_count = _available_packing_workers()
    worker_dir = pool_path / ".workers"
    worker_dir.mkdir(exist_ok=True)
    for stale_path in worker_dir.iterdir():
        if stale_path.suffix == ".tmp":
            stale_path.unlink()

    tasks: list[
        tuple[
            str,
            str,
            list[int],
            int,
            int,
            int,
            int,
            str,
            DocumentSampling | None,
            str,
            int,
        ]
    ] = []
    source_tasks: list[
        tuple[DataSource, int, int, int, int, str, str]
    ] = []
    large_source_tasks: list[
        tuple[DataSource, int, int, int, int, str, str]
    ] = []
    task_bytes: dict[int, int] = {}
    task_signatures: dict[int, str] = {}
    task_index = 0
    if worker_count > 1 and all(
        source.file_format in {"parquet", "jsonl_zstd"}
        for source in profile.sources
    ):
        for source in profile.sources:
            if source.file_format == "parquet":
                if source.text_column is None:
                    raise ValueError(f"parquet source has no text column: {source.name}")
                row_group_count = pq.ParquetFile(source.path).num_row_groups
                chunk_size = max(
                    1,
                    (row_group_count + worker_count * 4 - 1) // (worker_count * 4),
                )
                for start in range(0, row_group_count, chunk_size):
                    stop = min(start + chunk_size, row_group_count)
                    signature = f"parquet:{source.sha256}:{start}:{stop}"
                    tasks.append(
                        (
                            str(source.path),
                            source.text_column,
                            list(range(start, stop)),
                            threshold,
                            eos_token_id,
                            sequence_length,
                            task_index,
                            str(worker_dir),
                            source.document_sampling,
                            signature,
                            TOKENIZATION_CHUNK_DOCUMENTS,
                        )
                    )
                    task_bytes[task_index] = (
                        source.size_bytes * (stop - start) // row_group_count
                    )
                    task_signatures[task_index] = signature
                    task_index += 1
                continue
            signature = f"source:{source.sha256}"
            source_task = (
                source,
                threshold,
                eos_token_id,
                sequence_length,
                task_index,
                str(worker_dir),
                signature,
            )
            if source.size_bytes >= PARALLEL_ZSTD_MIN_BYTES:
                large_source_tasks.append(source_task)
            else:
                source_tasks.append(source_task)
            task_bytes[task_index] = source.size_bytes
            task_signatures[task_index] = signature
            task_index += 1

    executor: ProcessPoolExecutor | None = None
    completion = tqdm(
        total=sum(source.size_bytes for source in profile.sources),
        desc=f"packing ETA {profile.name}",
        unit="B",
        unit_scale=True,
        dynamic_ncols=True,
    )
    previous_rayon_threads = os.environ.get("RAYON_NUM_THREADS")
    try:
        if tasks or source_tasks or large_source_tasks:
            os.environ["RAYON_NUM_THREADS"] = "1"
        for large_task in large_source_tasks:
            _pack_large_zstd_task(
                large_task,
                tokenizer_path,
                worker_count,
            )

        if tasks or source_tasks or large_source_tasks:
            context = mp.get_context("spawn")
            progress_queue = context.Queue()
            executor = ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=context,
                initializer=_init_packing_worker,
                initargs=(str(tokenizer_path), progress_queue),
            )
            completed: dict[
                int, tuple[int, dict[str, str], dict[str, list[int]]]
            ] = {}
            reported_bytes = {index: 0 for index in task_bytes}
            for index, signature in task_signatures.items():
                cached = _load_task_result(worker_dir, index, signature)
                if cached is None:
                    continue
                _, documents, paths, remainders = cached
                completed[index] = (documents, paths, remainders)
                document_count += documents
                progress.update(documents)
                completion.update(task_bytes[index])
                reported_bytes[index] = task_bytes[index]

            futures = {}
            for task in tasks:
                if task[6] not in completed:
                    futures[executor.submit(_pack_row_group_task, task)] = task[6]
            for task in source_tasks:
                if task[4] not in completed:
                    futures[executor.submit(_pack_source_task, task)] = task[4]
            pending_futures = set(futures)
            while pending_futures:
                try:
                    task_index, processed_bytes = progress_queue.get(timeout=0.25)
                    processed_bytes = min(processed_bytes, task_bytes[task_index])
                    completion.update(
                        max(0, processed_bytes - reported_bytes[task_index])
                    )
                    reported_bytes[task_index] = max(
                        reported_bytes[task_index], processed_bytes
                    )
                except Empty:
                    pass
                done, pending_futures = wait(
                    pending_futures,
                    timeout=0,
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    task_index, documents, paths, remainders = future.result()
                    completed[task_index] = (documents, paths, remainders)
                    document_count += documents
                    progress.update(documents)
                    completion.update(task_bytes[task_index] - reported_bytes[task_index])
                    reported_bytes[task_index] = task_bytes[task_index]
            for task_index in range(len(task_bytes)):
                documents, paths, remainders = completed[task_index]
                for split in ("train", "validation"):
                    with Path(paths[split]).open("rb") as handle:
                        while True:
                            chunk = np.fromfile(
                                handle,
                                dtype=TOKEN_DTYPE,
                                count=TOKEN_WRITE_BUFFER_TOKENS,
                            )
                            if chunk.size == 0:
                                break
                            writers[split].add_tokens(chunk.tolist())
                    writers[split].add_tokens(remainders[split])
                for path in paths.values():
                    Path(path).unlink()
                _task_result_path(worker_dir, task_index).unlink(missing_ok=True)
            completion.update(completion.total - completion.n)
        else:
            for source in profile.sources:
                source_progress = 0
                source_rows = (
                    pq.ParquetFile(source.path).metadata.num_rows
                    if source.file_format == "parquet"
                    else None
                )
                source_documents = 0
                for documents in _iter_source_document_batches(source):
                    encodings = tokenizer.encode_batch(
                        documents,
                        add_special_tokens=False,
                    )
                    encoded = []
                    for document, encoding in zip(documents, encodings, strict=True):
                        digest = hashlib.sha256(document.encode("utf-8")).digest()
                        encoded.append(
                            (
                                int.from_bytes(digest[:8], "big") < threshold,
                                encoding.ids,
                            )
                        )
                    for is_validation, token_ids in encoded:
                        split = "validation" if is_validation else "train"
                        writers[split].add_document(token_ids, eos_token_id)
                    document_count += len(encoded)
                    progress.update(len(encoded))
                    if source_rows is not None:
                        source_documents += len(documents)
                        estimated = min(
                            source.size_bytes,
                            source.size_bytes * source_documents // source_rows,
                        )
                        completion.update(estimated - source_progress)
                        source_progress = estimated
                completion.update(source.size_bytes - source_progress)
    except KeyboardInterrupt:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
            executor = None
        raise
    finally:
        progress.close()
        completion.close()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        if previous_rayon_threads is None:
            os.environ.pop("RAYON_NUM_THREADS", None)
        else:
            os.environ["RAYON_NUM_THREADS"] = previous_rayon_threads
        for writer in writers.values():
            writer.close()

    for split, temporary_path in temporary_paths.items():
        temporary_path.replace(pool_path / f"{split}.bin")
    metadata = {
        "contract": contract,
        "documents": document_count,
        "sequence_counts": {
            split: writer.sequence_count for split, writer in writers.items()
        },
        "dropped_tokens": {
            split: writer.dropped_tokens for split, writer in writers.items()
        },
    }
    temporary_metadata = pool_path / ".metadata.json.tmp"
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_metadata.replace(pool_path / "metadata.json")


def load_or_prepare_packed_pool(
    profile: DataProfile,
    tokenizer: Tokenizer,
    tokenizer_path: Path,
    sequence_length: int,
    validation_fraction: float,
    pool_root: Path | None = None,
) -> PackedDataPool:
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    contract = _pool_contract(
        profile,
        tokenizer_path,
        sequence_length,
        validation_fraction,
    )
    fingerprint = _fingerprint(contract)
    root = _default_pool_root(profile) if pool_root is None else pool_root
    pool_path = root / f"{profile.name}-{fingerprint[:16]}"
    existing = _load_pool(pool_path, contract, fingerprint)
    if existing is not None:
        return existing

    print(f"preparing_data_pool={pool_path}", flush=True)
    _prepare_pool_files(
        profile,
        tokenizer,
        tokenizer_path,
        pool_path,
        sequence_length,
        validation_fraction,
        contract,
    )
    prepared = _load_pool(pool_path, contract, fingerprint)
    assert prepared is not None
    return prepared
