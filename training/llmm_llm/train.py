import argparse
import hashlib
import json
import math
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from itertools import islice, repeat
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np
import torch
from torch import nn
from tokenizers import Tokenizer
from tqdm import tqdm

from data.pretraining import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_TOKENIZER_PATH,
    discover_data_profile,
    DataProfile,
    PackedDataPool,
    load_capability_plan,
    load_data_profile,
    load_or_prepare_packed_pool,
)
from data.sft import SFTDataPool, load_sft_pool

from .checkpoint import (
    TrainingState,
    latest_training_checkpoint,
    load_model_weights_only,
    load_training_checkpoint,
    save_training_checkpoint,
    training_checkpoint_path,
)
from .config import ModelConfig
from .fused_loss import DEFAULT_LOSS_CHUNK_SIZE
from .model import LLMM
from .optimization import (
    CachedFusedAdamW,
    FusedSparseAdam,
    clip_grad_norm_sparse_,
    fused_adamw_clip_scale_,
    learning_rate_for_step,
    set_learning_rate,
)
from .quantization import QUANTIZATION_FORMAT


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    token_ids: np.ndarray
    labels: np.ndarray
    source: str
    valid_mask: np.ndarray
    non_ignore_count: int


@dataclass(frozen=True, slots=True)
class DeviceTrainingBatch:
    token_ids: torch.Tensor
    labels: torch.Tensor
    source: str
    valid_mask: torch.Tensor
    non_ignore_count: int


PackedBatchIterator = Iterator[TrainingBatch]
LOSS_TELEMETRY_BUFFER_SIZE = 16


def _clear_inactive_expert_grads_from_mask_(
    model: LLMM,
    active_experts: torch.Tensor,
) -> None:
    if bool(active_experts.all()):
        return
    for layer_index, block in enumerate(model.blocks):
        inactive = ~active_experts[layer_index].to(dtype=torch.bool)
        for parameter in block.moe.experts.parameters():
            if parameter.grad is not None:
                parameter.grad[inactive] = 0


def clear_inactive_expert_grads_(
    model: LLMM,
    expert_counts: torch.Tensor,
) -> None:
    """Keep AdamW state and weight decay frozen for experts unused this step."""
    active_experts = expert_counts.ne(0).to(device="cpu")
    _clear_inactive_expert_grads_from_mask_(model, active_experts)


def _expert_parameter_coordinates(
    model: LLMM,
) -> dict[nn.Parameter, int]:
    return {
        parameter: layer_index
        for layer_index, block in enumerate(model.blocks)
        for parameter in block.moe.experts.parameters()
    }


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    total_loss: float
    causal_loss: float
    balance_loss: float
    z_loss: float
    expert_counts: torch.Tensor
    router_entropy: torch.Tensor


class DeviceBatchIterator(Iterator[DeviceTrainingBatch]):
    """Stage NumPy batches through pinned memory and prefetch CUDA copies."""

    def __init__(
        self,
        batches: PackedBatchIterator,
        device: torch.device,
    ) -> None:
        self._batches = iter(batches)
        self._device = device
        self._copy_stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )
        self._next_batch: DeviceTrainingBatch | None = None
        self._next_slot: int | None = None
        self._last_returned_slot: int | None = None
        self._host_token_buffers: list[torch.Tensor] = []
        self._host_label_buffers: list[torch.Tensor] = []
        self._host_valid_mask_buffers: list[torch.Tensor] = []
        self._device_token_buffers: list[torch.Tensor] = []
        self._device_label_buffers: list[torch.Tensor] = []
        self._device_valid_mask_buffers: list[torch.Tensor] = []
        self._slot_metadata: list[
            tuple[tuple[int, ...], str, int] | None
        ] = [None, None]
        self._ready_events: list[torch.cuda.Event] = []
        self._consumed_events: list[torch.cuda.Event] = []
        self._slot_loaded = [False, False]
        self._slot_returned = [False, False]
        self._initialize()

    def __iter__(self) -> "DeviceBatchIterator":
        return self

    def __next__(self) -> DeviceTrainingBatch:
        if self._copy_stream is None:
            batch = self._next_batch
            if batch is None:
                raise StopIteration
            self._preload_cpu()
            return batch

        slot = self._next_slot
        if slot is None:
            raise StopIteration
        current_stream = torch.cuda.current_stream(self._device)
        if self._last_returned_slot is not None:
            self._consumed_events[self._last_returned_slot].record(current_stream)
        current_stream.wait_event(self._ready_events[slot])
        metadata = self._slot_metadata[slot]
        if metadata is None:
            raise RuntimeError("prefetched batch metadata is missing")
        shape, source, non_ignore_count = metadata
        batch = DeviceTrainingBatch(
            self._device_token_buffers[slot].view(shape),
            self._device_label_buffers[slot].view(shape),
            source,
            self._device_valid_mask_buffers[slot].view(shape),
            non_ignore_count,
        )
        self._slot_returned[slot] = True
        self._last_returned_slot = slot

        try:
            packed_batch = next(self._batches)
        except StopIteration:
            self._next_slot = None
        else:
            next_slot = 1 - slot
            self._preload_cuda(next_slot, packed_batch)
            self._next_slot = next_slot
        return batch

    def _initialize(self) -> None:
        try:
            packed_batch = next(self._batches)
        except StopIteration:
            return

        if self._copy_stream is None:
            self._next_batch = DeviceTrainingBatch(
                torch.from_numpy(packed_batch.token_ids).to(dtype=torch.long),
                torch.from_numpy(packed_batch.labels).to(dtype=torch.long),
                packed_batch.source,
                torch.from_numpy(packed_batch.valid_mask).to(dtype=torch.bool),
                packed_batch.non_ignore_count,
            )
            return

        token_source = torch.from_numpy(packed_batch.token_ids.reshape(-1))
        label_source = torch.from_numpy(packed_batch.labels.reshape(-1))
        valid_mask_source = torch.from_numpy(packed_batch.valid_mask.reshape(-1))
        self._host_token_buffers = [
            torch.empty_like(token_source, dtype=torch.long, pin_memory=True),
            torch.empty_like(token_source, dtype=torch.long, pin_memory=True),
        ]
        self._host_label_buffers = [
            torch.empty_like(label_source, dtype=torch.long, pin_memory=True),
            torch.empty_like(label_source, dtype=torch.long, pin_memory=True),
        ]
        self._host_valid_mask_buffers = [
            torch.empty_like(valid_mask_source, dtype=torch.bool, pin_memory=True),
            torch.empty_like(valid_mask_source, dtype=torch.bool, pin_memory=True),
        ]
        self._device_token_buffers = [
            torch.empty(token_source.numel(), dtype=torch.long, device=self._device),
            torch.empty(token_source.numel(), dtype=torch.long, device=self._device),
        ]
        self._device_label_buffers = [
            torch.empty(label_source.numel(), dtype=torch.long, device=self._device),
            torch.empty(label_source.numel(), dtype=torch.long, device=self._device),
        ]
        self._device_valid_mask_buffers = [
            torch.empty(
                valid_mask_source.numel(), dtype=torch.bool, device=self._device
            ),
            torch.empty(
                valid_mask_source.numel(), dtype=torch.bool, device=self._device
            ),
        ]
        self._ready_events = [torch.cuda.Event(), torch.cuda.Event()]
        self._consumed_events = [torch.cuda.Event(), torch.cuda.Event()]
        self._preload_cuda(0, packed_batch)
        self._next_slot = 0

    def _preload_cpu(self) -> None:
        try:
            packed_batch = next(self._batches)
        except StopIteration:
            self._next_batch = None
            return
        self._next_batch = DeviceTrainingBatch(
            torch.from_numpy(packed_batch.token_ids).to(dtype=torch.long),
            torch.from_numpy(packed_batch.labels).to(dtype=torch.long),
            packed_batch.source,
            torch.from_numpy(packed_batch.valid_mask).to(dtype=torch.bool),
            packed_batch.non_ignore_count,
        )

    def _preload_cuda(self, slot: int, packed_batch: TrainingBatch) -> None:
        token_source = torch.from_numpy(packed_batch.token_ids.reshape(-1))
        label_source = torch.from_numpy(packed_batch.labels.reshape(-1))
        valid_mask_source = torch.from_numpy(packed_batch.valid_mask.reshape(-1))
        token_host_buffer = self._host_token_buffers[slot]
        label_host_buffer = self._host_label_buffers[slot]
        valid_mask_host_buffer = self._host_valid_mask_buffers[slot]
        if (
            token_source.numel() != token_host_buffer.numel()
            or label_source.numel() != label_host_buffer.numel()
            or valid_mask_source.numel() != valid_mask_host_buffer.numel()
            or packed_batch.token_ids.shape != packed_batch.labels.shape
            or packed_batch.token_ids.shape != packed_batch.valid_mask.shape
        ):
            raise ValueError("all prefetched batches must have the same token count")
        if self._slot_loaded[slot]:
            self._ready_events[slot].synchronize()
        token_host_buffer.copy_(token_source)
        label_host_buffer.copy_(label_source)
        valid_mask_host_buffer.copy_(valid_mask_source)

        assert self._copy_stream is not None
        with torch.cuda.stream(self._copy_stream):
            if self._slot_returned[slot]:
                self._copy_stream.wait_event(self._consumed_events[slot])
            self._device_token_buffers[slot].copy_(token_host_buffer, non_blocking=True)
            self._device_label_buffers[slot].copy_(label_host_buffer, non_blocking=True)
            self._device_valid_mask_buffers[slot].copy_(
                valid_mask_host_buffer, non_blocking=True
            )
            self._ready_events[slot].record(self._copy_stream)
        self._slot_metadata[slot] = (
            packed_batch.token_ids.shape,
            packed_batch.source,
            packed_batch.non_ignore_count,
        )
        self._slot_loaded[slot] = True


class LossTelemetryBuffer:
    def __init__(self, device: torch.device) -> None:
        self._buffer = (
            torch.empty(
                LOSS_TELEMETRY_BUFFER_SIZE,
                dtype=torch.float32,
                device=device,
            )
            if device.type == "cuda"
            else None
        )
        self._count = 0

    def record(self, loss: torch.Tensor) -> list[float]:
        if self._buffer is None:
            return [loss.detach().float().item()]
        self._buffer[self._count].copy_(loss.detach().float())
        self._count += 1
        return self.flush() if self._count == self._buffer.numel() else []

    def flush(self) -> list[float]:
        if self._buffer is None or self._count == 0:
            return []
        values = self._buffer[: self._count].tolist()
        self._count = 0
        return values


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _next_batch(batches: PackedBatchIterator, pool_name: str) -> TrainingBatch:
    try:
        return next(batches)
    except StopIteration as error:
        raise RuntimeError(f"data pool {pool_name!r} was exhausted") from error


def _replay_batches(
    pool: PackedDataPool,
    split: Literal["train", "validation"],
    batch_size: int,
    *,
    epoch: int = 0,
    seed: int = 42,
    shuffle: bool = False,
) -> Iterator[TrainingBatch]:
    for token_ids in pool.iter_batches(
        split,
        batch_size,
        epoch=epoch,
        seed=seed,
        shuffle=shuffle,
    ):
        yield TrainingBatch(
            token_ids,
            token_ids,
            "replay",
            np.ones(token_ids.shape, dtype=np.bool_),
            token_ids.size - token_ids.shape[0],
        )


def _sft_batch_size(bucket_length: int, sft_batch_size: int) -> int:
    if bucket_length not in (256, 512, 1024):
        raise ValueError(f"unsupported SFT bucket: {bucket_length}")
    if sft_batch_size != 32:
        raise ValueError("SFT batch size must be 32")
    return sft_batch_size


SFT_RUNTIME_LENGTH = 1024


def _sft_batches(
    pool: SFTDataPool,
    split: Literal["train", "validation"],
    bucket_length: int,
    sft_batch_size: int,
    *,
    epoch: int = 0,
    seed: int = 42,
    shuffle: bool = False,
) -> Iterator[TrainingBatch]:
    batch_size = _sft_batch_size(bucket_length, sft_batch_size)
    for token_ids, mask, lengths in pool.iter_batches(
        split,
        bucket_length,
        batch_size,
        epoch=epoch,
        seed=seed,
        shuffle=shuffle,
    ):
        labels = np.where(mask, token_ids.astype(np.int64), -100)
        valid_mask = (
            np.arange(bucket_length, dtype=np.uint16)[None, :]
            < lengths[:, None]
        )
        # CUDA prefetch and torch.compile(dynamic=False) need one sequence
        # length. Right-pad shorter SFT buckets; causal attention keeps the
        # real tokens unchanged, and pad labels stay -100.
        if bucket_length < SFT_RUNTIME_LENGTH:
            pad = SFT_RUNTIME_LENGTH - bucket_length
            token_ids = np.pad(token_ids, ((0, 0), (0, pad)), constant_values=0)
            labels = np.pad(labels, ((0, 0), (0, pad)), constant_values=-100)
            valid_mask = np.pad(
                valid_mask, ((0, 0), (0, pad)), constant_values=False
            )
        yield TrainingBatch(
            token_ids,
            labels,
            f"sft-{bucket_length}",
            valid_mask,
            int(np.count_nonzero(labels[:, 1:] != -100)),
        )


def _allocate_sft_batches(
    available: dict[int, int],
    requested: int,
) -> dict[int, int]:
    total = sum(available.values())
    if requested < 0 or requested > total:
        raise ValueError("requested SFT batch count exceeds the pool")
    allocation = {
        bucket: available[bucket] * requested // total if total else 0
        for bucket in available
    }
    remaining = requested - sum(allocation.values())
    for bucket in sorted(
        available,
        key=lambda value: (
            available[value] * requested % total if total else 0,
            value,
        ),
        reverse=True,
    ):
        if remaining == 0:
            break
        if allocation[bucket] < available[bucket]:
            allocation[bucket] += 1
            remaining -= 1
    return allocation


def _sft_epoch_schedule(
    sft_pool: SFTDataPool,
    *,
    sft_batch_size: int,
    gradient_accumulation_steps: int,
    epoch: int,
    seed: int,
) -> tuple[list[str], dict[str, int]]:
    return _mixed_sft_epoch_schedule(
        sft_pool,
        replay_pool=None,
        replay_batch_size=None,
        sft_batch_size=sft_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        epoch=epoch,
        seed=seed,
    )


def _mixed_sft_epoch_schedule(
    sft_pool: SFTDataPool,
    replay_pool: PackedDataPool | None,
    replay_batch_size: int | None,
    *,
    sft_batch_size: int,
    gradient_accumulation_steps: int,
    epoch: int,
    seed: int,
) -> tuple[list[str], dict[str, int]]:
    available = {
        f"sft-{bucket}": sft_pool.sequence_count("train", bucket)
        // _sft_batch_size(bucket, sft_batch_size)
        for bucket in (256, 512, 1024)
    }
    if replay_pool is not None:
        if replay_batch_size is None:
            raise ValueError("replay batch size is required with a replay pool")
        available["replay"] = replay_pool.batch_count("train", replay_batch_size)
    total_available = sum(available.values())
    if total_available == 0:
        raise ValueError("SFT and replay pools have no complete training microbatch")
    total_micro_batches = total_available - (
        total_available % gradient_accumulation_steps
    )
    if total_micro_batches == 0:
        raise ValueError("SFT and replay pools have no complete optimizer step")
    schedule = [source for source, count in available.items() for _ in range(count)]
    random.Random(seed + epoch).shuffle(schedule)
    schedule = schedule[:total_micro_batches]
    allocation = {source: schedule.count(source) for source in available}
    return schedule, allocation


def _sft_epoch_batches(
    sft_pool: SFTDataPool,
    *,
    replay_pool: PackedDataPool | None = None,
    replay_batch_size: int | None = None,
    sft_batch_size: int,
    gradient_accumulation_steps: int,
    epoch: int,
    seed: int,
    start_batch: int = 0,
) -> Iterator[TrainingBatch]:
    schedule, allocation = _mixed_sft_epoch_schedule(
        sft_pool,
        replay_pool,
        replay_batch_size,
        sft_batch_size=sft_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        epoch=epoch,
        seed=seed,
    )
    sources: dict[str, Iterator[TrainingBatch]] = {}
    for bucket in (256, 512, 1024):
        source = f"sft-{bucket}"
        count = allocation[source]
        if count:
            sources[source] = iter(
                _sft_batches(
                    sft_pool,
                    "train",
                    bucket,
                    sft_batch_size,
                    epoch=epoch,
                    seed=seed,
                    shuffle=True,
                )
            )
    if replay_pool is not None and allocation.get("replay", 0):
        assert replay_batch_size is not None
        sources["replay"] = iter(
            _replay_batches(
                replay_pool,
                "train",
                replay_batch_size,
                epoch=epoch,
                seed=seed,
                shuffle=True,
            )
        )
    for index, source in enumerate(schedule):
        batch = _next_batch(sources[source], source)
        if index >= start_batch:
            yield batch


def _sft_validation_batches(
    pool: SFTDataPool,
    sft_batch_size: int,
) -> Iterator[TrainingBatch]:
    for bucket in (256, 512, 1024):
        yield from _sft_batches(pool, "validation", bucket, sft_batch_size)


def _validate_arguments(args: argparse.Namespace) -> None:
    positive_values = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "eval_interval": args.eval_interval,
        "eval_batches": args.eval_batches,
        "checkpoint_interval": args.checkpoint_interval,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if args.batch_size != 32:
        raise ValueError("batch_size must be 32")
    if args.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if args.n_layers is not None and args.n_layers <= 0:
        raise ValueError("n_layers must be positive")
    if not 0.0 <= args.warmup_ratio <= 1.0:
        raise ValueError("warmup_ratio must be in [0, 1]")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if args.capability_transition and (
        args.capability_plan is None or args.resume_from is None
    ):
        raise ValueError(
            "--capability-transition requires --capability-plan and --resume-from"
        )
    if args.stage == "sft":
        if args.sft_pool is None:
            raise ValueError("--stage sft requires --sft-pool")
        if args.capability_plan is not None or args.capability_transition:
            raise ValueError("SFT stage cannot use capability-plan options")
    elif args.sft_pool is not None or args.sft_transition:
        raise ValueError("--sft-pool and --sft-transition require --stage sft")
    if args.sft_transition and args.resume_from is None:
        raise ValueError("--sft-transition requires --resume-from")
    if args.router_top_k < 1:
        raise ValueError("router_top_k must be at least 1")
    if args.resume_weights is not None and not args.resume_weights.is_file():
        raise ValueError(f"--resume-weights does not exist: {args.resume_weights}")


def _load_replay_profile(args: argparse.Namespace) -> DataProfile:
    if args.data_dir is not None and args.replay_data_dir:
        raise ValueError("use either --data-dir or --replay-data-dir, not both")
    replay_directories = args.replay_data_dir or (
        [args.data_dir] if args.data_dir is not None else []
    )
    if not replay_directories:
        return load_data_profile(args.data_profile, args.data_manifest)
    profiles = [discover_data_profile(path) for path in replay_directories]
    if len(profiles) == 1:
        return profiles[0]
    sources = tuple(
        replace(source, name=f"{profile.name}/{source.name}")
        for profile in profiles
        for source in profile.sources
    )
    name_digest = hashlib.sha256(
        "\n".join(source.sha256 for source in sources).encode("ascii")
    ).hexdigest()[:16]
    return DataProfile(
        name=f"replay-{name_digest}",
        purpose=";".join(profile.purpose for profile in profiles),
        sources=sources,
    )


def _run_contract(
    args: argparse.Namespace,
    config: ModelConfig,
    profile: DataProfile | None,
    pool: PackedDataPool | None,
    capability_plan_fingerprint: str | None,
    capability_wave: str | None,
    schedule_total_steps: int,
    sft_pool: SFTDataPool | None = None,
) -> dict[str, Any]:
    contract = {
        "architecture": "llmm-ple-moe-en32k-v1",
        "model_name": args.model,
        "quantization": QUANTIZATION_FORMAT,
        "model_config": asdict(config),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "sequence_length": args.seq_len,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "peak_learning_rate": args.learning_rate,
        "optimizers": {
            "dense": "fused AdamW on CUDA",
            "ple_table": "SparseAdam without weight decay",
            "ple_absmean": "exact incremental tensor-wide scale",
        },
        "training_loss": (
            f"exact causal CE plus Top-{config.router_top_k} router auxiliary losses"
        ),
        "router": {
            "routing": f"dropless Top-{config.router_top_k}",
            "top_k": config.router_top_k,
            "experts_per_layer": config.n_experts,
            "balance_loss_coefficient": (
                config.router_balance_loss_coefficient
            ),
            "z_loss_coefficient": config.router_z_loss_coefficient,
        },
        "warmup_ratio": args.warmup_ratio,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "overfit_single_batch": args.overfit_single_batch,
    }
    if profile is not None and pool is not None:
        contract.update(
            {
                "data_profile": profile.name,
                "data_pool": pool.fingerprint,
            }
        )
    if sft_pool is not None:
        contract.update(
            {
                "stage": "sft",
                "sft_pool": sft_pool.fingerprint,
                "sft_loss": "masked-target exact causal CE",
            }
        )
    if capability_plan_fingerprint is not None:
        contract.update(
            {
                "capability_plan": capability_plan_fingerprint,
                "capability_wave": capability_wave,
                "schedule_total_steps": schedule_total_steps,
            }
        )
    return contract


@torch.no_grad()
def _evaluate_batches(
    model: LLMM,
    batches: PackedBatchIterator,
    device: torch.device,
    batch_count: int,
) -> EvaluationMetrics:
    was_training = model.training
    model.eval()
    loss_sums = torch.zeros(4, dtype=torch.float32, device=device)
    expert_counts = torch.zeros(
        model.config.n_layers,
        model.config.n_experts,
        dtype=torch.int64,
        device=device,
    )
    router_entropy = torch.zeros(
        model.config.n_layers,
        dtype=torch.float32,
        device=device,
    )
    evaluated_batches = 0
    device_batches = DeviceBatchIterator(
        islice(batches, batch_count),
        device,
    )
    for batch in device_batches:
        router_valid_mask = (
            None if batch.source == "replay" else batch.valid_mask
        )
        with _autocast_context(device):
            total, causal, balance, z_loss, counts, entropy = (
                model.training_loss_components(
                    batch.token_ids,
                    batch.labels,
                    valid_mask=router_valid_mask,
                    non_ignore_count=batch.non_ignore_count,
                )
            )
        loss_sums.add_(
            torch.stack((total, causal, balance, z_loss)).detach().float()
        )
        expert_counts.add_(counts.detach())
        router_entropy.add_(entropy.detach().float())
        evaluated_batches += 1
    if was_training:
        model.train()
    mean_losses = (loss_sums / evaluated_batches).cpu().tolist()
    return EvaluationMetrics(
        total_loss=mean_losses[0],
        causal_loss=mean_losses[1],
        balance_loss=mean_losses[2],
        z_loss=mean_losses[3],
        expert_counts=expert_counts.cpu(),
        router_entropy=(router_entropy / evaluated_batches).cpu(),
    )


@torch.no_grad()
def evaluate(
    model: LLMM,
    pool: PackedDataPool,
    device: torch.device,
    batch_size: int,
    batch_count: int,
) -> EvaluationMetrics:
    return _evaluate_batches(
        model,
        _replay_batches(pool, "validation", batch_size),
        device,
        batch_count,
    )


def _router_telemetry(
    expert_counts: torch.Tensor,
    router_entropy: torch.Tensor,
) -> str:
    counts = expert_counts.detach().float().cpu()
    mean_load = counts.mean(dim=-1)
    max_to_mean = counts.max(dim=-1).values / mean_load.clamp_min(1.0)
    payload = {
        "expert_counts": counts.to(dtype=torch.int64).tolist(),
        "max_to_mean": max_to_mean.tolist(),
        "router_entropy": router_entropy.detach().float().cpu().tolist(),
        "dropped_tokens": 0,
    }
    return json.dumps(payload, separators=(",", ":"))


def train(args: argparse.Namespace) -> list[float]:
    _validate_arguments(args)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    config = (
        ModelConfig.tiny(vocab_size=vocab_size)
        if args.model == "tiny"
        else ModelConfig()
    )
    if args.n_layers is not None:
        if args.model != "main":
            raise ValueError("--n-layers is only valid with --model main")
        config = replace(config, n_layers=args.n_layers)
    if args.router_top_k != config.router_top_k:
        config = replace(config, router_top_k=args.router_top_k)
    if vocab_size != config.vocab_size:
        raise ValueError(
            f"tokenizer vocabulary has {vocab_size} entries; "
            f"model expects {config.vocab_size}"
        )
    if args.seq_len > config.max_seq_len:
        raise ValueError("seq_len exceeds the selected model's max_seq_len")

    capability_plan = (
        load_capability_plan(args.capability_plan)
        if args.capability_plan is not None
        else None
    )
    capability_wave = None
    if capability_plan is not None:
        if args.capability_wave is None:
            raise ValueError("--capability-wave is required with --capability-plan")
        capability_wave = next(
            (
                wave
                for wave in capability_plan.waves
                if wave.name == args.capability_wave
            ),
            None,
        )
        if capability_wave is None:
            raise ValueError(f"unknown capability wave: {args.capability_wave}")
        if (
            args.seq_len != capability_wave.sequence_length
            or args.batch_size != capability_wave.batch_size
        ):
            raise ValueError(
                "capability wave requires "
                f"--seq-len {capability_wave.sequence_length} "
                f"--batch-size {capability_wave.batch_size}"
            )

    sft_pool = load_sft_pool(args.sft_pool) if args.stage == "sft" else None
    load_replay_pool = (
        sft_pool is None
        or args.data_dir is not None
        or bool(args.replay_data_dir)
    )
    if load_replay_pool:
        profile = _load_replay_profile(args)
        pool = load_or_prepare_packed_pool(
            profile,
            tokenizer,
            args.tokenizer,
            1024 if sft_pool is not None else args.seq_len,
            args.validation_fraction,
        )
    else:
        profile = None
        pool = None
    sft_batch_size = args.batch_size
    replay_batch_size = args.batch_size if pool is not None else None
    if sft_pool is None:
        assert pool is not None
        available_micro_batches = pool.batch_count("train", args.batch_size)
        micro_batches_per_epoch = (
            available_micro_batches
            // args.gradient_accumulation_steps
            * args.gradient_accumulation_steps
        )
    else:
        schedule, _ = _mixed_sft_epoch_schedule(
            sft_pool,
            pool,
            replay_batch_size,
            sft_batch_size=sft_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            epoch=0,
            seed=args.seed,
        )
        micro_batches_per_epoch = len(schedule)
    if micro_batches_per_epoch == 0:
        raise ValueError("data pool has no complete optimizer step")
    if (
        pool is not None
        and replay_batch_size is not None
        and pool.batch_count("validation", replay_batch_size) < args.eval_batches
    ):
        raise ValueError("data pool has fewer validation batches than eval_batches")
    if sft_pool is not None:
        sft_validation_count = sum(
            sft_pool.sequence_count("validation", bucket)
            // _sft_batch_size(bucket, sft_batch_size)
            for bucket in (256, 512, 1024)
        )
        if sft_validation_count < args.eval_batches:
            raise ValueError("SFT pool has fewer validation batches than eval_batches")
    steps_per_epoch = (
        micro_batches_per_epoch // args.gradient_accumulation_steps
    )
    total_steps = steps_per_epoch * args.epochs
    total_micro_batches = micro_batches_per_epoch * args.epochs
    schedule_total_steps = (
        sum(
            math.ceil(
                wave.target_tokens / (wave.sequence_length * wave.batch_size)
            )
            for wave in capability_plan.waves
        )
        if capability_plan is not None
        else total_steps
    )

    model = LLMM(config).to(device)
    ple_parameter = model.ple_table.weight
    dense_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter is not ple_parameter
    ]
    dense_optimizer = torch.optim.AdamW(
        dense_parameters,
        lr=args.learning_rate,
        fused=device.type == "cuda",
    )
    ple_optimizer = torch.optim.SparseAdam(
        [ple_parameter],
        lr=args.learning_rate,
    )
    optimizers = {
        "dense": dense_optimizer,
        "ple": ple_optimizer,
    }
    run_contract = _run_contract(
        args,
        config,
        profile,
        pool,
        capability_plan.fingerprint if capability_plan is not None else None,
        capability_wave.name if capability_wave is not None else None,
        schedule_total_steps,
        sft_pool,
    )
    state = TrainingState()
    resume_from = args.resume_from
    if resume_from is None and args.checkpoint_dir is not None:
        resume_from = latest_training_checkpoint(args.checkpoint_dir)
    if resume_from is not None:
        state = load_training_checkpoint(
            resume_from,
            model,
            optimizers,
            expected_run_contract=run_contract,
            device=device,
            allowed_contract_differences=(
                frozenset(
                    {
                        "data_profile",
                        "data_pool",
                        "epochs",
                        "batch_size",
                        "sequence_length",
                        "capability_wave",
                        "capability_plan",
                        "schedule_total_steps",
                        "stage",
                        "sft_pool",
                        "sft_loss",
                        "peak_learning_rate",
                        "warmup_ratio",
                    }
                )
                if args.capability_transition or args.sft_transition
                else frozenset()
            ),
        )
        if args.capability_transition:
            state = replace(state, step=0, micro_batches_consumed=0)
        if args.sft_transition:
            state = replace(state, step=0, micro_batches_consumed=0, schedule_step=0)
        if device.type == "cuda":
            for parameter_group in dense_optimizer.param_groups:
                parameter_group["fused"] = True
        if state.step >= total_steps:
            raise ValueError("checkpoint has already completed the requested epochs")
        if (
            state.micro_batches_consumed
            != state.step * args.gradient_accumulation_steps
        ):
            raise ValueError("checkpoint optimizer and data positions disagree")
        if state.micro_batches_consumed >= total_micro_batches:
            raise ValueError("checkpoint data position exceeds the requested epochs")
        print(
            f"resumed_from={resume_from} step={state.step} "
            f"tokens={state.tokens_seen:,} schedule_step={state.schedule_step:,}"
        )
    elif args.resume_weights is not None:
        load_model_weights_only(args.resume_weights, model, device)
        print(
            f"loaded_weights={args.resume_weights} fresh_optimizer=True "
            f"router_top_k={config.router_top_k}"
        )
    dense_stepper = (
        CachedFusedAdamW(
            dense_optimizer,
            _expert_parameter_coordinates(model),
        )
        if device.type == "cuda"
        else None
    )
    ple_stepper = (
        FusedSparseAdam(ple_optimizer, ple_parameter)
        if device.type == "cuda"
        else None
    )
    repeated_batch = None
    if args.overfit_single_batch:
        if sft_pool is None:
            assert pool is not None
            repeated_batch = _next_batch(
                _replay_batches(pool, "train", args.batch_size),
                pool.path.name,
            )
        else:
            assert sft_pool is not None
            repeated_batch = _next_batch(
                _sft_epoch_batches(
                    sft_pool,
                    replay_pool=pool,
                    replay_batch_size=replay_batch_size,
                    sft_batch_size=sft_batch_size,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    epoch=0,
                    seed=args.seed,
                ),
                sft_pool.path.name,
            )
    precision = "bf16" if device.type == "cuda" else "fp32"
    print(
        f"device={device.type} precision={precision} model={args.model} "
        f"n_layers={config.n_layers} "
        f"quantization={QUANTIZATION_FORMAT} parameters={model.parameter_count():,}"
    )
    if profile is not None and pool is not None:
        print(
            f"data_pool={profile.name} train_sequences={pool.train_sequences:,} "
            f"validation_sequences={pool.validation_sequences:,} epochs={args.epochs}"
        )
    else:
        print(f"epochs={args.epochs}")
    if sft_pool is not None:
        print(
            "sft_pool="
            f"{sft_pool.path} "
            + " ".join(
                f"s{bucket}={sft_pool.sequence_count('train', bucket):,}"
                for bucket in (256, 512, 1024)
            )
        )
    print(
        f"batch_size={args.batch_size} sequence_length={args.seq_len} "
        f"steps_per_epoch={steps_per_epoch:,} total_steps={total_steps:,} "
        f"schedule_total_steps={schedule_total_steps:,}"
    )
    print(
        "dense_optimizer=cached-fused-AdamW "
        "ple_optimizer=fused-SparseAdam ple_weight_decay=0"
        if dense_stepper is not None
        else "dense_optimizer=AdamW ple_optimizer=SparseAdam ple_weight_decay=0"
    )

    losses: list[float] = []
    loss_telemetry = LossTelemetryBuffer(device)
    displayed_loss: float | None = None
    model.train()
    cuda_compile_mode = "reduce-overhead"
    cuda_compile_fullgraph = True
    if device.type == "cuda":
        hidden_states = torch.compile(
            model._hidden_states_from_ple,
            mode=cuda_compile_mode,
            dynamic=False,
            fullgraph=cuda_compile_fullgraph,
        )

        def training_loss_components(
            token_ids: torch.Tensor,
            labels: torch.Tensor,
            valid_mask: torch.Tensor | None,
            non_ignore_count: int,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]:
            ple_table = model.ple_table(token_ids)
            hidden, balance, z_loss, counts, entropy = hidden_states(
                token_ids,
                ple_table,
                valid_mask,
            )
            total, causal, balance, z_loss = (
                model._training_losses_from_hidden_states(
                    hidden,
                    labels,
                    balance,
                    z_loss,
                    non_ignore_count=non_ignore_count,
                )
            )
            return total, causal, balance, z_loss, counts, entropy

    else:
        def training_loss_components(
            token_ids: torch.Tensor,
            labels: torch.Tensor,
            valid_mask: torch.Tensor | None,
            non_ignore_count: int,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]:
            return model.training_loss_components(
                token_ids,
                labels,
                valid_mask=valid_mask,
                non_ignore_count=non_ignore_count,
            )
    if device.type == "cuda":
        print(
            f"cuda_compile={cuda_compile_mode} "
            "compiled_graph=transformer-core cuda_moe=grouped-triton fullgraph=true "
            f"loss_chunk_size={DEFAULT_LOSS_CHUNK_SIZE} "
            "first_step_includes_compilation=true"
        )
    starting_epoch = state.micro_batches_consumed // micro_batches_per_epoch
    for epoch_index in range(starting_epoch, args.epochs):
        start_batch = (
            state.micro_batches_consumed % micro_batches_per_epoch
            if epoch_index == starting_epoch
            else 0
        )
        batches: PackedBatchIterator
        if sft_pool is None:
            batches = islice(
                _replay_batches(
                    pool,
                    "train",
                    args.batch_size,
                    epoch=epoch_index,
                    seed=args.seed,
                    shuffle=not args.overfit_single_batch,
                ),
                start_batch,
                micro_batches_per_epoch,
            )
        else:
            batches = _sft_epoch_batches(
                sft_pool,
                replay_pool=pool,
                replay_batch_size=replay_batch_size,
                sft_batch_size=sft_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                epoch=epoch_index,
                seed=args.seed,
                start_batch=start_batch,
            )
        optimizer_steps = (
            micro_batches_per_epoch - start_batch
        ) // args.gradient_accumulation_steps
        if args.max_steps is not None:
            remaining = args.max_steps - state.step
            if remaining <= 0:
                break
            optimizer_steps = min(optimizer_steps, remaining)
        batch_source: PackedBatchIterator = (
            batches
            if repeated_batch is None
            else repeat(
                repeated_batch,
                optimizer_steps * args.gradient_accumulation_steps,
            )
        )
        device_batches = DeviceBatchIterator(batch_source, device)

        progress = tqdm(
            total=optimizer_steps,
            desc=f"epoch {epoch_index + 1}/{args.epochs}",
            unit="step",
            dynamic_ncols=True,
        )
        try:
            for optimizer_step_index in range(optimizer_steps):
                step = state.step + 1
                schedule_step = state.schedule_step + 1
                learning_rate = learning_rate_for_step(
                    schedule_step,
                    schedule_total_steps,
                    args.learning_rate,
                    args.warmup_ratio,
                )
                set_learning_rate(dense_optimizer, learning_rate)
                set_learning_rate(ple_optimizer, learning_rate)
                if dense_stepper is None:
                    dense_optimizer.zero_grad(set_to_none=True)
                else:
                    dense_stepper.zero_grad()
                if ple_stepper is None:
                    ple_optimizer.zero_grad(set_to_none=True)
                else:
                    ple_stepper.zero_grad()
                accumulated_metrics: torch.Tensor | None = None
                accumulated_expert_counts: torch.Tensor | None = None
                accumulated_router_entropy: torch.Tensor | None = None
                micro_batches_consumed = state.micro_batches_consumed
                tokens_seen = state.tokens_seen

                for micro_batch_index in range(args.gradient_accumulation_steps):
                    batch = next(device_batches)
                    router_valid_mask = (
                        None if batch.source == "replay" else batch.valid_mask
                    )
                    with _autocast_context(device):
                        (
                            loss,
                            causal_loss,
                            balance_loss,
                            z_loss,
                            expert_counts,
                            router_entropy,
                        ) = training_loss_components(
                            batch.token_ids,
                            batch.labels,
                            router_valid_mask,
                            batch.non_ignore_count,
                        )
                    detached_metrics = torch.stack(
                        (loss, causal_loss, balance_loss, z_loss)
                    ).detach().float()
                    accumulated_metrics = (
                        detached_metrics.clone()
                        if accumulated_metrics is None
                        else accumulated_metrics.add_(detached_metrics)
                    )
                    detached_counts = expert_counts.detach()
                    accumulated_expert_counts = (
                        detached_counts.clone()
                        if accumulated_expert_counts is None
                        else accumulated_expert_counts.add_(detached_counts)
                    )
                    detached_entropy = router_entropy.detach().float()
                    accumulated_router_entropy = (
                        detached_entropy.clone()
                        if accumulated_router_entropy is None
                        else accumulated_router_entropy.add_(detached_entropy)
                    )
                    (loss / args.gradient_accumulation_steps).backward()
                    micro_batches_consumed += 1
                    tokens_seen += batch.token_ids.numel()

                if device.type != "cuda":
                    clear_inactive_expert_grads_(model, accumulated_expert_counts)
                    active_experts = None
                else:
                    active_experts = accumulated_expert_counts.ne(0)
                if dense_stepper is None:
                    clip_grad_norm_sparse_(model.parameters(), max_norm=1.0)
                    dense_grad_scale = None
                else:
                    _, dense_grad_scale = fused_adamw_clip_scale_(
                        dense_stepper.gradients(),
                        ple_parameter,
                        max_norm=1.0,
                    )
                absmean_update = model.ple_table.prepare_absmean_update()
                if dense_stepper is None:
                    dense_optimizer.step()
                else:
                    assert dense_grad_scale is not None
                    dense_stepper.step(
                        grad_scale=dense_grad_scale,
                        active_experts=active_experts,
                    )
                if ple_stepper is None:
                    ple_optimizer.step()
                else:
                    ple_stepper.step()
                model.ple_table.finish_absmean_update(absmean_update)
                assert accumulated_metrics is not None
                assert accumulated_expert_counts is not None
                assert accumulated_router_entropy is not None
                current_metrics = (
                    accumulated_metrics / args.gradient_accumulation_steps
                )
                current_loss = current_metrics[0]
                current_router_entropy = (
                    accumulated_router_entropy
                    / args.gradient_accumulation_steps
                )
                state = TrainingState(
                    step=step,
                    micro_batches_consumed=micro_batches_consumed,
                    tokens_seen=tokens_seen,
                    schedule_step=schedule_step,
                )
                should_evaluate = (
                    step % args.eval_interval == 0 or step == total_steps
                )
                should_checkpoint = args.checkpoint_dir is not None and (
                    step % args.checkpoint_interval == 0 or step == total_steps
                )
                end_of_epoch = optimizer_step_index + 1 == optimizer_steps
                recorded_losses = loss_telemetry.record(current_loss)
                if should_evaluate or should_checkpoint or end_of_epoch:
                    recorded_losses.extend(loss_telemetry.flush())
                if recorded_losses:
                    losses.extend(recorded_losses)
                    displayed_loss = recorded_losses[-1]

                postfix = {
                    "lr": f"{learning_rate:.2e}",
                    "tokens": f"{tokens_seen:,}",
                    "schedule": f"{schedule_step:,}/{schedule_total_steps:,}",
                }
                if displayed_loss is not None:
                    postfix["loss"] = f"{displayed_loss:.4f}"
                progress.set_postfix(postfix, refresh=False)
                progress.update(1)

                if should_evaluate:
                    training_values = current_metrics.cpu().tolist()
                    progress.write(
                        f"step={step:06d} "
                        f"training_total_loss={training_values[0]:.6f} "
                        f"training_causal_loss={training_values[1]:.6f} "
                        f"router_balance_loss={training_values[2]:.6f} "
                        f"router_z_loss={training_values[3]:.6f}"
                    )
                    progress.write(
                        "training_router="
                        + _router_telemetry(
                            accumulated_expert_counts,
                            current_router_entropy,
                        )
                    )
                    if sft_pool is None:
                        assert pool is not None
                        validation = evaluate(
                            model,
                            pool,
                            device,
                            batch_size=args.batch_size,
                            batch_count=args.eval_batches,
                        )
                    else:
                        validation = _evaluate_batches(
                            model,
                            _sft_validation_batches(sft_pool, sft_batch_size),
                            device,
                            args.eval_batches,
                        )
                    progress.write(
                        f"step={step:06d} "
                        f"validation_total_loss={validation.total_loss:.6f} "
                        f"validation_causal_loss={validation.causal_loss:.6f} "
                        f"router_balance_loss={validation.balance_loss:.6f} "
                        f"router_z_loss={validation.z_loss:.6f}"
                    )
                    progress.write(
                        "validation_router="
                        + _router_telemetry(
                            validation.expert_counts,
                            validation.router_entropy,
                        )
                    )
                    if sft_pool is not None:
                        sft_validation = validation
                        progress.write(
                            f"step={step:06d} "
                            f"sft_validation_total_loss={sft_validation.total_loss:.6f} "
                            f"sft_validation_causal_loss={sft_validation.causal_loss:.6f}"
                        )

                if should_checkpoint:
                    assert args.checkpoint_dir is not None
                    checkpoint_path = training_checkpoint_path(
                        args.checkpoint_dir,
                        state.step,
                    )
                    save_training_checkpoint(
                        checkpoint_path,
                        model,
                        optimizers,
                        state,
                        run_contract,
                        device,
                    )
                    progress.write(
                        f"checkpoint={checkpoint_path} step={step}"
                    )
        finally:
            progress.close()

        print(f"epoch={epoch_index + 1}/{args.epochs} completed")
    return losses


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the LLMM reference model")
    parser.add_argument("--model", choices=("tiny", "main"), default="main")
    parser.add_argument(
        "--n-layers",
        type=int,
        default=None,
        help="Override ModelConfig.n_layers (main only). Use 24 for the two-board depth upcycle.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--stage", choices=("base", "sft"), default="base")
    parser.add_argument("--data-profile", default="local")
    parser.add_argument("--data-manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--replay-data-dir",
        type=Path,
        action="append",
        help="Replay corpus directory; may be repeated to combine old and new shards.",
    )
    parser.add_argument("--sft-pool", type=Path)
    parser.add_argument(
        "--sft-transition",
        action="store_true",
        help="Resume a base checkpoint with a fresh SFT scheduler while retaining optimizer state.",
    )
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="stop after this many optimizer steps (smoke / short runs)",
    )
    parser.add_argument("--batch-size", type=int, choices=(32,), default=32)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.02)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--resume-weights",
        type=Path,
        help="Load model tensors only (fresh optimizer). Used to continue the reconstructed original.",
    )
    parser.add_argument(
        "--router-top-k",
        type=int,
        default=1,
        help="Experts active per token (1 = shipped PFor, 2 = single-board top-2).",
    )
    parser.add_argument("--capability-plan", type=Path)
    parser.add_argument("--capability-wave")
    parser.add_argument("--capability-transition", action="store_true")
    parser.add_argument("--overfit-single-batch", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
