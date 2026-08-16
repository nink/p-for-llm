"""Rebuild a training checkpoint from a deployed LLMCRAFT v2 artifact."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor
from torchao.prototype.quantized_training import BitNetTrainingLinearWeight

from export_aircraft import (
    HEADER,
    HEADER_SIZE,
    MAGIC,
    NO_SCALE,
    P4_CONFIG,
    RECORD,
    RECORD_SIZE,
    REGION_FLASH,
    REGION_PSRAM,
    STORAGE_FP16,
    STORAGE_Q8,
    STORAGE_TERNARY_BASE3,
    VERSION,
    _assert_config,
    _build_records,
    _is_packed_expert_tensor,
    _is_training_only,
    _validate_contract,
    _write_artifact,
)
from llmm_llm.config import ModelConfig
from llmm_llm.model import LLMM
from llmm_llm.quantization import QUANTIZATION_FORMAT


def _unpack_five_trits(payload: bytes, numel: int) -> Tensor:
    packed = torch.frombuffer(bytearray(payload), dtype=torch.uint8).to(torch.int32)
    codes = torch.empty(packed.numel() * 5, dtype=torch.int8)
    remainder = packed.clone()
    for lane in range(5):
        codes[lane::5] = (remainder % 3).to(dtype=torch.int8)
        remainder //= 3
    return codes[:numel] - 1


def _unpack_ple_rows(payload: bytes, rows: int, cols: int) -> Tensor:
    row_bytes = (cols + 4) // 5
    expected = rows * row_bytes
    if len(payload) != expected:
        raise ValueError(f"PLE payload is {len(payload)} bytes, expected {expected}")
    rows_out = []
    for index in range(rows):
        start = index * row_bytes
        rows_out.append(_unpack_five_trits(payload[start : start + row_bytes], cols))
    return torch.stack(rows_out, dim=0)


def _master_from_codes(codes: Tensor, scale: float) -> Tensor:
    codes_f = codes.to(dtype=torch.float32)
    mean_abs = float(codes_f.abs().mean().item())
    if mean_abs == 0.0:
        return torch.zeros_like(codes_f)
    return codes_f * (scale / mean_abs)


def _copy_parameter(module: torch.nn.Module, name: str, value: Tensor) -> None:
    parts = name.split(".")
    target = module
    for part in parts[:-1]:
        target = target[int(part)] if part.isdigit() else getattr(target, part)
    current = getattr(target, parts[-1])
    master = getattr(current, "original_weight_tensor", current)
    if tuple(master.shape) != tuple(value.shape):
        raise ValueError(f"{name} shape {tuple(value.shape)} != {tuple(master.shape)}")
    master.data.copy_(value.to(dtype=master.dtype, device=master.device))


def parse_artifact(path: Path) -> tuple[dict[str, int], list[tuple], bytes, bytes]:
    data = path.read_bytes()
    magic, version, header_size, *rest = HEADER.unpack_from(data)
    if magic != MAGIC or version != VERSION or header_size != HEADER_SIZE:
        raise ValueError(f"unsupported artifact header: {path}")
    config_values = rest[:9]
    record_count = rest[9]
    index_offset, flash_offset, flash_len, psram_offset, psram_len = rest[10:15]
    config = dict(zip(P4_CONFIG, config_values))
    _assert_config(config)
    records = []
    for index in range(record_count):
        offset = index_offset + index * RECORD_SIZE
        tensor_id, region, storage, scale_index, elements, rel, nbytes, _pad = RECORD.unpack_from(
            data, offset
        )
        records.append((tensor_id, region, storage, scale_index, elements, rel, nbytes))
    flash = data[flash_offset : flash_offset + flash_len]
    psram = data[psram_offset : psram_offset + psram_len]
    if len(flash) != flash_len or len(psram) != psram_len:
        raise ValueError("artifact region truncated")
    return config, records, flash, psram


def _payload(record: tuple, flash: bytes, psram: bytes) -> bytes:
    _tensor_id, region, _storage, _scale, _elements, rel, nbytes = record
    blob = flash if region == REGION_FLASH else psram
    return blob[rel : rel + nbytes]


def reconstruct_state(artifact: Path) -> tuple[ModelConfig, dict[str, Tensor]]:
    config_dict, records, flash, psram = parse_artifact(artifact)
    model_config = ModelConfig(**{key: config_dict[key] for key in ModelConfig.__dataclass_fields__ if key in config_dict})
    if records[0][2] != STORAGE_TERNARY_BASE3 or records[1][2] != STORAGE_Q8:
        raise ValueError("artifact record order does not match export_aircraft")
    if records[2][2] != STORAGE_FP16 or records[-1][2] != STORAGE_FP16:
        raise ValueError("artifact is missing Q8 row scales or ternary scales")

    ternary_scales = torch.frombuffer(
        bytearray(_payload(records[-1], flash, psram)), dtype=torch.float16
    ).to(torch.float32)
    reconstructed: dict[str, Tensor] = {}

    ple_codes = _unpack_ple_rows(
        _payload(records[0], flash, psram),
        model_config.vocab_size,
        model_config.n_layers * model_config.ple_dim,
    )
    reconstructed["ple_table.weight"] = _master_from_codes(
        ple_codes, float(ternary_scales[records[0][3]].item())
    )

    q8 = torch.frombuffer(bytearray(_payload(records[1], flash, psram)), dtype=torch.int8).view(
        model_config.vocab_size, model_config.d_model
    )
    row_scales = torch.frombuffer(
        bytearray(_payload(records[2], flash, psram)), dtype=torch.float16
    ).to(torch.float32)
    reconstructed["token_embedding.weight"] = q8.to(torch.float32) * row_scales[:, None]
    reconstructed["lm_head.weight"] = reconstructed["token_embedding.weight"]

    cursor = 3
    norm_queue: list[tuple[str, int, Tensor]] = []

    template = LLMM(model_config)
    model_state = template.state_dict()
    del template

    def next_ternary(shape: tuple[int, ...], name: str) -> Tensor:
        nonlocal cursor
        record = records[cursor]
        cursor += 1
        if record[2] != STORAGE_TERNARY_BASE3:
            raise ValueError(f"expected ternary record for {name}, got storage {record[2]}")
        codes = _unpack_five_trits(_payload(record, flash, psram), record[4]).view(shape)
        return _master_from_codes(codes, float(ternary_scales[record[3]].item()))

    def queue_norm(name: str, elements: int) -> None:
        norm_queue.append((name, elements, None))

    for name in sorted(model_state):
        if name in {"ple_table.weight", "token_embedding.weight", "lm_head.weight"}:
            continue
        if _is_training_only(name):
            continue
        value = model_state[name]
        if _is_packed_expert_tensor(name) and value.ndim == 3:
            experts = [next_ternary(tuple(value.shape[1:]), f"{name}[{index}]") for index in range(value.shape[0])]
            reconstructed[name] = torch.stack(experts, dim=0)
            continue
        if _is_packed_expert_tensor(name) and value.ndim == 2:
            for expert_index, expert_norm in enumerate(value):
                queue_norm(f"{name}[{expert_index}]", expert_norm.numel())
            continue
        if value.ndim == 2:
            reconstructed[name] = next_ternary(tuple(value.shape), name)
        elif value.ndim == 1:
            queue_norm(name, value.numel())
        else:
            raise ValueError(f"unsupported deployment tensor: {name}")

    # Expert 2D norms were queued as placeholders; consume remaining FP16 records in order.
    # records[cursor:-1] after ternary consumption should be the norm tensors.
    leftover = records[cursor:-1]
    placeholder_i = 0
    filled: dict[str, list[Tensor]] = {}
    scalar_norms: dict[str, Tensor] = {}
    for record in leftover:
        if record[2] != STORAGE_FP16:
            raise ValueError(f"expected FP16 norm record, got storage {record[2]} at {record[0]}")
        payload = torch.frombuffer(
            bytearray(_payload(record, flash, psram)), dtype=torch.float16
        ).to(torch.float32)
        slot_name, elements, _ = norm_queue[placeholder_i]
        placeholder_i += 1
        if payload.numel() != elements:
            raise ValueError(f"{slot_name} expected {elements} FP16 values, got {payload.numel()}")
        if slot_name.endswith("]") and "[" in slot_name:
            base = slot_name[: slot_name.rindex("[")]
            filled.setdefault(base, []).append(payload)
        else:
            scalar_norms[slot_name] = payload
    if placeholder_i != len(norm_queue):
        raise ValueError(f"norm record count {placeholder_i} != queued {len(norm_queue)}")

    for name, slices in filled.items():
        reconstructed[name] = torch.stack(slices, dim=0)
    reconstructed.update(scalar_norms)
    return model_config, reconstructed


def load_reconstructed_model(artifact: Path, device: torch.device) -> LLMM:
    config, tensors = reconstruct_state(artifact)
    model = LLMM(config).to(device)
    for name, value in tensors.items():
        _copy_parameter(model, name, value)
    model.ple_table.reset_absmean_cache()
    model.eval()
    return model


def save_checkpoint(model: LLMM, artifact: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "run_contract": {
            "architecture": "llmm-ple-moe-en32k-v1",
            "model_name": "main",
            "quantization": QUANTIZATION_FORMAT,
            "model_config": asdict(model.config),
            "source_artifact": artifact.name,
            "reconstructed": True,
        },
        "training_state": {
            "step": 0,
            "micro_batches_consumed": 0,
            "tokens_seen": 12_000_000_000,
            "schedule_step": 0,
        },
        "model": model.state_dict(),
        "optimizers": {"dense": {}, "ple": {}},
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": None,
    }
    torch.save(payload, output)


def verify_reexport(model: LLMM, original: Path, reexport: Path) -> dict[str, str]:
    records = _build_records(model.state_dict())
    _validate_contract(records)
    _write_artifact(reexport, dict(P4_CONFIG), records)
    original_hash = hashlib.sha256(original.read_bytes()).hexdigest()
    new_hash = hashlib.sha256(reexport.read_bytes()).hexdigest()
    return {
        "original_sha256": original_hash,
        "reexport_sha256": new_hash,
        "match": str(original_hash == new_hash),
        "original_bytes": str(original.stat().st_size),
        "reexport_bytes": str(reexport.stat().st_size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reexport", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but not available")
    model = load_reconstructed_model(args.artifact, device)
    save_checkpoint(model, args.artifact, args.output)
    print(f"checkpoint={args.output}")
    print(f"bytes={args.output.stat().st_size}")
    if args.reexport is not None:
        report = verify_reexport(model, args.artifact, args.reexport)
        for key, value in report.items():
            print(f"{key}={value}")


if __name__ == "__main__":
    main()
