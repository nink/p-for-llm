"""Export the fixed ESP32-P4 W1.58A8 deployment artifact."""

from __future__ import annotations

import argparse
import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torchao.prototype.quantized_training import BitNetTrainingLinearWeight
from torchao.prototype.quantized_training.bitnet import quantize_bitnet_weight

from llmm_llm.checkpoint import latest_training_checkpoint


MAGIC = b"LLMCRAFT"
VERSION = 2
HEADER = struct.Struct("<8sHH10I5Q")
HEADER_SIZE = 96
RECORD = struct.Struct("<HBBHQQQH")
RECORD_SIZE = RECORD.size

REGION_FLASH = 1
REGION_PSRAM = 2
STORAGE_TERNARY_BASE3 = 1
STORAGE_FP16 = 2
STORAGE_Q8 = 3
NO_SCALE = 0xFFFF

P4_CONFIG = {
    "vocab_size": 32_768,
    "d_model": 192,
    "n_layers": 12,
    "n_heads": 6,
    "n_kv_heads": 2,
    "ffn_hidden": 512,
    "n_experts": 29,
    "ple_dim": 176,
    "max_seq_len": 1_024,
}
P4_PLE_BYTES = 13_860_864
P4_NON_PLE_TERNARY_BYTES = 21_018_713
P4_TIED_Q8_BYTES = 6_291_456
P4_TIED_ROW_SCALE_BYTES = 65_536
P4_NORM_BYTES = 661_216
P4_TERNARY_SCALE_BYTES = 2_260
P4_FLASH_MODEL_BYTES = P4_PLE_BYTES
P4_PSRAM_STATIC_BYTES = (
    P4_NON_PLE_TERNARY_BYTES
    + P4_TIED_Q8_BYTES
    + P4_TIED_ROW_SCALE_BYTES
    + P4_NORM_BYTES
    + P4_TERNARY_SCALE_BYTES
)
P4_FLASH_CAPACITY = 16 * 1024 * 1024
P4_PSRAM_CAPACITY = 32 * 1024 * 1024
P4_FLASH_NON_MODEL_BUDGET = int(2.47 * 1024 * 1024)
P4_PSRAM_KV_BYTES = 1_671_168
P4_PSRAM_RUNTIME_RESERVE = int(2.3284 * 1024 * 1024)
P4_MANIFEST_BUDGET = int(0.10 * 1024 * 1024)


@dataclass(frozen=True, slots=True)
class TensorRecord:
    tensor_id: int
    region: int
    storage: int
    elements: int
    payload: bytes
    scale_index: int = NO_SCALE


def _pack_five_trits(codes: Tensor) -> bytes:
    values = codes.reshape(-1).to(dtype=torch.int16) + 1
    padding = (-values.numel()) % 5
    if padding:
        values = torch.nn.functional.pad(values, (0, padding))
    values = values.view(-1, 5)
    packed = (
        values[:, 0]
        + values[:, 1] * 3
        + values[:, 2] * 9
        + values[:, 3] * 27
        + values[:, 4] * 81
    )
    return packed.to(dtype=torch.uint8).numpy().tobytes()


def _pack_ple_rows(codes: Tensor) -> bytes:
    if codes.ndim != 2:
        raise ValueError("PLE table must be rank two")
    # A row must be independently packed: its token ID maps directly to Flash.
    return b"".join(_pack_five_trits(row) for row in codes)


def _float32(value: Tensor) -> Tensor:
    return value.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _ternary(value: Tensor) -> tuple[bytes, float]:
    tensor = _float32(value)
    scale = float(tensor.abs().mean().item())
    if scale == 0.0:
        scale = 1.0
    codes = quantize_bitnet_weight(tensor, torch.tensor(scale))
    return _pack_five_trits(codes), scale


def _q8_rows(value: Tensor) -> tuple[bytes, bytes]:
    tensor = _float32(value)
    if tensor.ndim != 2:
        raise ValueError("tied embedding must be rank two")
    maximum = tensor.abs().amax(dim=1)
    scales = maximum / 127.0
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    quantized = torch.round(tensor / scales[:, None]).clamp(-127, 127)
    return (
        quantized.to(dtype=torch.int8).numpy().tobytes(),
        scales.to(dtype=torch.float16).numpy().tobytes(),
    )


def _is_training_only(name: str) -> bool:
    return name in {"ple_table.weight_abs_sum", "ple_table.row_abs_sums"}


def _is_packed_expert_tensor(name: str) -> bool:
    return ".moe.experts." in name


def _append_ternary_record(
    records: list[TensorRecord],
    scales: list[float],
    value: Tensor,
) -> None:
    payload, scale = _ternary(value)
    records.append(
        TensorRecord(
            tensor_id=len(records),
            region=REGION_PSRAM,
            storage=STORAGE_TERNARY_BASE3,
            elements=value.numel(),
            payload=payload,
            scale_index=len(scales),
        )
    )
    scales.append(scale)


def _assert_config(config: dict[str, int | float]) -> None:
    mismatches = [
        f"{key}={config.get(key)!r} (expected {expected!r})"
        for key, expected in P4_CONFIG.items()
        if config.get(key) != expected
    ]
    if mismatches:
        raise ValueError("not the contracted ESP32-P4 model: " + ", ".join(mismatches))


def _build_records(model_state: dict[str, Tensor]) -> list[TensorRecord]:
    if "token_embedding.weight" not in model_state or "ple_table.weight" not in model_state:
        raise KeyError("checkpoint is missing tied embeddings or the PLE table")
    if not torch.equal(model_state["token_embedding.weight"], model_state["lm_head.weight"]):
        raise ValueError("lm_head is not tied to token_embedding")

    records: list[TensorRecord] = []
    ternary_scales: list[float] = []

    ple = _float32(model_state["ple_table.weight"])
    ple_scale = float(ple.abs().mean().item())
    if ple_scale == 0.0:
        ple_scale = 1.0
    ple_codes = quantize_bitnet_weight(ple, torch.tensor(ple_scale))
    ternary_scales.append(ple_scale)
    records.append(
        TensorRecord(
            tensor_id=len(records),
            region=REGION_FLASH,
            storage=STORAGE_TERNARY_BASE3,
            elements=ple.numel(),
            payload=_pack_ple_rows(ple_codes),
            scale_index=0,
        )
    )

    tied_q8, tied_scales = _q8_rows(model_state["token_embedding.weight"])
    records.extend(
        (
            TensorRecord(
                tensor_id=len(records),
                region=REGION_PSRAM,
                storage=STORAGE_Q8,
                elements=model_state["token_embedding.weight"].numel(),
                payload=tied_q8,
            ),
            TensorRecord(
                tensor_id=len(records) + 1,
                region=REGION_PSRAM,
                storage=STORAGE_FP16,
                elements=model_state["token_embedding.weight"].shape[0],
                payload=tied_scales,
            ),
        )
    )

    norm_records: list[TensorRecord] = []
    for name in sorted(model_state):
        if name in {"ple_table.weight", "token_embedding.weight", "lm_head.weight"}:
            continue
        if _is_training_only(name):
            continue
        value = model_state[name]
        if not isinstance(value, Tensor):
            raise TypeError(f"unsupported checkpoint value: {name}")
        if _is_packed_expert_tensor(name) and value.ndim == 3:
            # Each projection is one contiguous expert bank. Projection banks
            # follow lexical state_dict order; experts inside a bank stay 0..E-1.
            for expert_weight in value:
                _append_ternary_record(records, ternary_scales, expert_weight)
            continue
        if _is_packed_expert_tensor(name) and value.ndim == 2:
            for expert_norm in value:
                norm_records.append(
                    TensorRecord(
                        tensor_id=0,
                        region=REGION_PSRAM,
                        storage=STORAGE_FP16,
                        elements=expert_norm.numel(),
                        payload=_float32(expert_norm)
                        .to(dtype=torch.float16)
                        .numpy()
                        .tobytes(),
                    )
                )
            continue
        if value.ndim == 2:
            _append_ternary_record(records, ternary_scales, value)
        elif value.ndim == 1:
            norm_records.append(
                TensorRecord(
                    tensor_id=0,
                    region=REGION_PSRAM,
                    storage=STORAGE_FP16,
                    elements=value.numel(),
                    payload=_float32(value).to(dtype=torch.float16).numpy().tobytes(),
                )
            )
        else:
            raise ValueError(f"unsupported deployment tensor: {name}")

    for record in norm_records:
        records.append(
            TensorRecord(
                tensor_id=len(records),
                region=record.region,
                storage=record.storage,
                elements=record.elements,
                payload=record.payload,
            )
        )
    records.append(
        TensorRecord(
            tensor_id=len(records),
            region=REGION_PSRAM,
            storage=STORAGE_FP16,
            elements=len(ternary_scales),
            payload=torch.tensor(ternary_scales, dtype=torch.float16).numpy().tobytes(),
        )
    )
    return records


def _region_payload(records: list[TensorRecord], region: int) -> bytes:
    return b"".join(record.payload for record in records if record.region == region)


def _validate_contract(records: list[TensorRecord]) -> None:
    flash = _region_payload(records, REGION_FLASH)
    psram = _region_payload(records, REGION_PSRAM)
    ternary = [record for record in records if record.storage == STORAGE_TERNARY_BASE3]
    norm_bytes = sum(
        len(record.payload)
        for record in records
        if record.storage == STORAGE_FP16 and record.elements != len(ternary)
    ) - P4_TIED_ROW_SCALE_BYTES
    non_ple_ternary_bytes = sum(len(record.payload) for record in ternary[1:])

    actual = {
        "flash PLE": len(flash),
        "non-PLE Base-3": non_ple_ternary_bytes,
        "Q8 tied table": len(records[1].payload),
        "Q8 row scales": len(records[2].payload),
        "FP16 norms": norm_bytes,
        "FP16 ternary scales": len(records[-1].payload),
        "PSRAM static": len(psram),
        "ternary scale count": len(ternary),
    }
    expected = {
        "flash PLE": P4_PLE_BYTES,
        "non-PLE Base-3": P4_NON_PLE_TERNARY_BYTES,
        "Q8 tied table": P4_TIED_Q8_BYTES,
        "Q8 row scales": P4_TIED_ROW_SCALE_BYTES,
        "FP16 norms": P4_NORM_BYTES,
        "FP16 ternary scales": P4_TERNARY_SCALE_BYTES,
        "PSRAM static": P4_PSRAM_STATIC_BYTES,
        "ternary scale count": 1_130,
    }
    mismatches = [
        f"{name}: {actual[name]} != {value}" for name, value in expected.items() if actual[name] != value
    ]
    manifest_bytes = HEADER_SIZE + RECORD_SIZE * len(records)
    if manifest_bytes > P4_MANIFEST_BUDGET:
        mismatches.append(f"manifest: {manifest_bytes} > {P4_MANIFEST_BUDGET}")
    # The 0.10 MiB manifest allocation is already included in the Flash table's
    # 2.47 MiB non-model budget; the check above proves this actual index fits it.
    if P4_FLASH_NON_MODEL_BUDGET + len(flash) > P4_FLASH_CAPACITY:
        mismatches.append("Flash capacity exceeded")
    if P4_PSRAM_KV_BYTES + P4_PSRAM_RUNTIME_RESERVE + len(psram) > P4_PSRAM_CAPACITY:
        mismatches.append("PSRAM capacity exceeded")
    if mismatches:
        raise ValueError("P4 deployment contract mismatch: " + "; ".join(mismatches))


def _write_artifact(
    output: Path,
    config: dict[str, int | float],
    records: list[TensorRecord],
) -> None:
    flash = _region_payload(records, REGION_FLASH)
    psram = _region_payload(records, REGION_PSRAM)
    index_offset = HEADER_SIZE
    flash_offset = HEADER_SIZE + RECORD_SIZE * len(records)
    psram_offset = flash_offset + len(flash)

    header = HEADER.pack(
        MAGIC,
        VERSION,
        HEADER_SIZE,
        *(int(config[key]) for key in P4_CONFIG),
        len(records),
        index_offset,
        flash_offset,
        len(flash),
        psram_offset,
        len(psram),
    )
    temporary = output.with_suffix(output.suffix + ".tmp")
    output.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as stream:
        stream.write(header)
        stream.write(b"\0" * (HEADER_SIZE - len(header)))
        offsets = {REGION_FLASH: 0, REGION_PSRAM: 0}
        for record in records:
            stream.write(
                RECORD.pack(
                    record.tensor_id,
                    record.region,
                    record.storage,
                    record.scale_index,
                    record.elements,
                    offsets[record.region],
                    len(record.payload),
                    0,
                )
            )
            offsets[record.region] += len(record.payload)
        stream.write(flash)
        stream.write(psram)
    temporary.replace(output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="ascii"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--load-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device used while loading the complete training checkpoint.",
    )
    args = parser.parse_args()

    checkpoint = latest_training_checkpoint(args.checkpoint_dir)
    if checkpoint is None:
        raise FileNotFoundError(f"no checkpoint in {args.checkpoint_dir}")
    load_device = torch.device(args.load_device)
    if load_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--load-device cuda requires an available CUDA device")
    with torch.serialization.safe_globals([BitNetTrainingLinearWeight]):
        payload = torch.load(
            checkpoint,
            map_location=load_device,
            weights_only=True,
            mmap=load_device.type == "cpu",
        )
    config = payload["run_contract"]["model_config"]
    _assert_config(config)
    model_state = payload["model"]
    del payload
    records = _build_records(model_state)
    _validate_contract(records)
    _write_artifact(args.output, config, records)
    print(f"checkpoint={checkpoint}")
    print(f"artifact={args.output}")
    print(f"records={len(records)}")
    print(f"flash_model_bytes={P4_FLASH_MODEL_BYTES}")
    print(f"psram_static_bytes={P4_PSRAM_STATIC_BYTES}")
    print(f"bytes={args.output.stat().st_size}")
    print(f"sha256={args.output.with_suffix(args.output.suffix + '.sha256')}")


if __name__ == "__main__":
    main()
