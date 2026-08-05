"""Validate the immutable multi-wave capability-pretraining plan."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CapabilitySourceBudget:
    name: str
    target_tokens: int
    selection: dict[str, object]


@dataclass(frozen=True, slots=True)
class CapabilityWave:
    name: str
    target_tokens: int
    sequence_length: int
    batch_size: int
    sources: tuple[CapabilitySourceBudget, ...]


@dataclass(frozen=True, slots=True)
class CapabilityPlan:
    name: str
    total_tokens: int
    waves: tuple[CapabilityWave, ...]
    fingerprint: str


def _fingerprint(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_capability_plan(path: Path) -> CapabilityPlan:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported capability plan schema")
    total_tokens = int(payload["total_tokens"])
    waves: list[CapabilityWave] = []
    seen_names: set[str] = set()
    for record in payload["waves"]:
        name = str(record["name"])
        if name in seen_names:
            raise ValueError(f"duplicate capability wave: {name}")
        seen_names.add(name)
        source_records = record["sources"]
        sources = tuple(
            CapabilitySourceBudget(
                name=str(source["name"]),
                target_tokens=int(source["target_tokens"]),
                selection=dict(source["selection"]),
            )
            for source in source_records
        )
        wave = CapabilityWave(
            name=name,
            target_tokens=int(record["target_tokens"]),
            sequence_length=int(record["sequence_length"]),
            batch_size=int(record["batch_size"]),
            sources=sources,
        )
        if wave.target_tokens <= 0 or any(
            source.target_tokens <= 0 for source in wave.sources
        ):
            raise ValueError(f"wave {name} has a non-positive token budget")
        if sum(source.target_tokens for source in wave.sources) != wave.target_tokens:
            raise ValueError(f"wave {name} source budgets do not match its token budget")
        waves.append(wave)
    if sum(wave.target_tokens for wave in waves) != total_tokens:
        raise ValueError("wave token budgets do not match total_tokens")
    return CapabilityPlan(
        name=str(payload["name"]),
        total_tokens=total_tokens,
        waves=tuple(waves),
        fingerprint=_fingerprint(payload),
    )
