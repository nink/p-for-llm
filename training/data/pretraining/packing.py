"""Tokenize documents and pack them into fixed-length causal LM sequences."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from typing import Literal

from tokenizers import Tokenizer

from .sources import DataProfile, iter_source_documents


EOS_TOKEN = "<|endoftext|>"
DataSplit = Literal["all", "train", "validation"]


def iter_split_documents(
    documents: Iterable[str],
    split: DataSplit,
    validation_fraction: float,
) -> Iterator[str]:
    if split == "all":
        yield from documents
        return
    if split not in {"train", "validation"}:
        raise ValueError(f"unknown data split: {split}")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    threshold = int(validation_fraction * (1 << 64))
    for document in documents:
        digest = hashlib.sha256(document.encode("utf-8")).digest()
        is_validation = int.from_bytes(digest[:8], "big") < threshold
        if (split == "validation") == is_validation:
            yield document


def iter_token_sequences(
    documents: Iterable[str],
    tokenizer: Tokenizer,
    sequence_length: int,
) -> Iterator[list[int]]:
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    eos_token_id = tokenizer.token_to_id(EOS_TOKEN)
    if eos_token_id is None:
        raise ValueError(f"tokenizer is missing EOS token: {EOS_TOKEN}")

    packed: list[int] = []
    for document in documents:
        document_ids = tokenizer.encode(document, add_special_tokens=False).ids
        document_ids.append(eos_token_id)
        cursor = 0
        while cursor < len(document_ids):
            take = min(sequence_length - len(packed), len(document_ids) - cursor)
            packed.extend(document_ids[cursor : cursor + take])
            cursor += take
            if len(packed) == sequence_length:
                yield packed
                packed = []


def iter_profile_documents(profile: DataProfile) -> Iterator[str]:
    for source in profile.sources:
        yield from iter_source_documents(source)


def iter_profile_sequences(
    profile: DataProfile,
    tokenizer: Tokenizer,
    sequence_length: int,
    split: DataSplit,
    validation_fraction: float,
) -> Iterator[list[int]]:
    yield from iter_token_sequences(
        iter_split_documents(
            iter_profile_documents(profile), split, validation_fraction
        ),
        tokenizer,
        sequence_length,
    )


def iter_profile_batches(
    profile: DataProfile,
    tokenizer: Tokenizer,
    batch_size: int,
    sequence_length: int,
    split: DataSplit = "all",
    validation_fraction: float = 0.01,
) -> Iterator[list[list[int]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    batch: list[list[int]] = []
    for sequence in iter_profile_sequences(
        profile,
        tokenizer,
        sequence_length,
        split,
        validation_fraction,
    ):
        batch.append(sequence)
        if len(batch) == batch_size:
            yield batch
            batch = []
