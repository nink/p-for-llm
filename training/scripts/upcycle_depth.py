#!/usr/bin/env python3
"""Stack a 12-layer PFor checkpoint into 24 layers by copying each block twice.

Embeddings, norms, and the first 12 blocks are copied. Layers 12-23 are copies of
0-11. PLE projection/table are concatenated along the layer axis. No random extra
layers. Output is a weights-only checkpoint for --resume-weights.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import torch
from torchao.prototype.quantized_training import BitNetTrainingLinearWeight

from llmm_llm.checkpoint import CHECKPOINT_FORMAT_VERSION, TrainingState
from llmm_llm.config import ModelConfig
from llmm_llm.model import LLMM


def load_source(path: Path, device: torch.device) -> tuple[LLMM, dict]:
    with torch.serialization.safe_globals([BitNetTrainingLinearWeight]):
        payload = torch.load(path, map_location=device, weights_only=True)
    raw = payload["run_contract"]["model_config"]
    allowed = {field.name for field in dataclasses.fields(ModelConfig)}
    config = ModelConfig(**{key: value for key, value in raw.items() if key in allowed})
    if config.n_layers != 12:
        raise ValueError(f"expected a 12-layer source, got n_layers={config.n_layers}")
    model = LLMM(config).to(device)
    model.load_state_dict(payload["model"])
    return model, payload


def stack_layers(source: LLMM, dest_layers: int) -> LLMM:
    src_layers = source.config.n_layers
    if dest_layers % src_layers != 0:
        raise ValueError("dest layer count must be a multiple of the source")
    repeats = dest_layers // src_layers
    dest_config = dataclasses.replace(source.config, n_layers=dest_layers)
    dest = LLMM(dest_config).to(device=next(source.parameters()).device)
    src_sd = source.state_dict()
    dest_sd = dest.state_dict()
    copies: dict[str, torch.Tensor] = {}

    for key, tensor in dest_sd.items():
        if key.startswith("blocks."):
            rest = key[len("blocks.") :]
            layer_s, _, suffix = rest.partition(".")
            src_layer = int(layer_s) % src_layers
            copies[key] = src_sd[f"blocks.{src_layer}.{suffix}"]
            continue
        if key in src_sd and src_sd[key].shape == tensor.shape:
            copies[key] = src_sd[key]
            continue
        if key == "ple_table.weight" or key == "ple_model_projection.weight":
            src = src_sd[key]
            copies[key] = torch.cat([src.detach()] * repeats, dim=1 if key == "ple_table.weight" else 0)
            continue
        copies[key] = tensor

    missing = dest.load_state_dict(copies, strict=True)
    if missing.missing_keys or missing.unexpected_keys:
        raise RuntimeError(f"upcycle state_dict mismatch: {missing}")
    dest.ple_table.reset_absmean_cache()
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/home/nink/pfor-ckpts/original-reconstructed/step-00000000.pt"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/nink/pfor-ckpts/main-24l-upcycle/step-00000000.pt"),
    )
    parser.add_argument("--n-layers", type=int, default=24)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    source, payload = load_source(args.source, device)
    dest = stack_layers(source, args.n_layers)
    contract = dict(payload.get("run_contract") or {})
    contract["model_config"] = dataclasses.asdict(dest.config)
    contract["upcycle"] = {
        "source": str(args.source),
        "from_layers": source.config.n_layers,
        "to_layers": dest.config.n_layers,
        "method": "repeat-stack",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "run_contract": contract,
            "training_state": dataclasses.asdict(TrainingState()),
            "model": dest.state_dict(),
            "optimizers": {},
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state": None,
        },
        args.out,
    )
    print(
        f"upcycle {source.config.n_layers}->{dest.config.n_layers} "
        f"parameters={dest.parameter_count():,} wrote {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
