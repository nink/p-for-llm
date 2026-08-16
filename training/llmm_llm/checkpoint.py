"""Atomic training checkpoint save and restore."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torchao.prototype.quantized_training import BitNetTrainingLinearWeight


CHECKPOINT_FORMAT_VERSION = 2
CHECKPOINT_PATTERN = re.compile(r"^step-(\d{8,})\.pt$")


@dataclass(frozen=True, slots=True)
class TrainingState:
    step: int = 0
    micro_batches_consumed: int = 0
    tokens_seen: int = 0
    schedule_step: int = 0


def training_checkpoint_path(directory: Path, step: int) -> Path:
    return directory / f"step-{step:08d}.pt"


def latest_training_checkpoint(directory: Path) -> Path | None:
    if not directory.is_dir():
        return None
    checkpoints = [
        (int(match.group(1)), path)
        for path in directory.iterdir()
        if (match := CHECKPOINT_PATTERN.fullmatch(path.name)) is not None
    ]
    return max(checkpoints)[1] if checkpoints else None


def save_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizers: dict[str, Optimizer],
    state: TrainingState,
    run_contract: dict[str, Any],
    device: torch.device,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "run_contract": run_contract,
        "training_state": asdict(state),
        "model": model.state_dict(),
        "optimizers": {
            name: optimizer.state_dict()
            for name, optimizer in optimizers.items()
        },
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        ),
    }
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def load_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizers: dict[str, Optimizer],
    expected_run_contract: dict[str, Any],
    device: torch.device,
    allowed_contract_differences: frozenset[str] = frozenset(),
) -> TrainingState:
    with torch.serialization.safe_globals([BitNetTrainingLinearWeight]):
        payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format: {path}")

    saved_contract = payload["run_contract"]
    differences = {
        key
        for key in set(saved_contract) | set(expected_run_contract)
        if saved_contract.get(key) != expected_run_contract.get(key)
    }
    if differences - allowed_contract_differences:
        difference = ", ".join(sorted(differences))
        raise ValueError(f"checkpoint run contract differs in: {difference}")

    model.load_state_dict(payload["model"])
    saved_optimizers = payload["optimizers"]
    if saved_optimizers.keys() != optimizers.keys():
        raise ValueError("checkpoint optimizer set differs from the current run")
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(saved_optimizers[name])

    torch.set_rng_state(payload["cpu_rng_state"].cpu())
    cuda_rng_state = payload["cuda_rng_state"]
    if device.type == "cuda" and cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state.cpu(), device)
    return TrainingState(**payload["training_state"])


def load_model_weights_only(
    path: Path,
    model: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """Load model tensors from a checkpoint and ignore optimizer/run-contract."""
    with torch.serialization.safe_globals([BitNetTrainingLinearWeight]):
        payload = torch.load(path, map_location=device, weights_only=True)
    if "model" not in payload:
        raise ValueError(f"checkpoint has no model tensors: {path}")
    model.load_state_dict(payload["model"])
    return payload.get("run_contract") or {}
