"""Deterministic source and document selection for capability pretraining."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DocumentSampling:
    """Keep a stable fraction of documents without depending on read order."""

    namespace: str
    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if not self.namespace:
            raise ValueError("document sampling namespace must not be empty")
        if self.denominator <= 0 or not 0 < self.numerator <= self.denominator:
            raise ValueError("document sampling requires 0 < numerator <= denominator")

    def as_dict(self) -> dict[str, object]:
        return {
            "namespace": self.namespace,
            "numerator": self.numerator,
            "denominator": self.denominator,
        }


def _digest(namespace: str, value: str) -> bytes:
    return hashlib.sha256(
        namespace.encode("utf-8") + b"\0" + value.encode("utf-8")
    ).digest()


def select_document(document: str, sampling: DocumentSampling | None) -> bool:
    if sampling is None:
        return True
    value = int.from_bytes(_digest(sampling.namespace, document)[:8], "big")
    return value % sampling.denominator < sampling.numerator


def deterministic_shard_order(shard_count: int, namespace: str) -> list[int]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if not namespace:
        raise ValueError("shard namespace must not be empty")
    return sorted(range(shard_count), key=lambda index: _digest(namespace, str(index)))
