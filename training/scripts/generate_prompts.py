#!/usr/bin/env python3
"""Greedy-sample a few prompts from a PFor training checkpoint (GPU sanity)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torchao.prototype.quantized_training import BitNetTrainingLinearWeight

from llmm_llm.checkpoint import latest_training_checkpoint
from llmm_llm.config import ModelConfig
from llmm_llm.model import LLMM

PROMPTS = [
    "What is 2+2?",
    "What is the capital of France?",
    "Explain photosynthesis in one sentence.",
    "Who wrote Romeo and Juliet?",
    "The boiling point of water in Celsius is",
]


def format_chat(prompt: str) -> str:
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("assets/qwen3.5-english-tokenizer/tokenizer.json"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint = latest_training_checkpoint(args.checkpoint_dir)
    if checkpoint is None:
        raise SystemExit(f"no checkpoint in {args.checkpoint_dir}")
    device = torch.device(args.device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    eos = tokenizer.token_to_id("<|im_end|>")
    with torch.serialization.safe_globals([BitNetTrainingLinearWeight]):
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
    config = ModelConfig(**payload["run_contract"]["model_config"])
    model = LLMM(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    print(f"checkpoint={checkpoint} step={payload['training_state']['step']}", flush=True)

    for prompt in PROMPTS:
        encoded = tokenizer.encode(format_chat(prompt)).ids
        tokens = torch.tensor([encoded], device=device, dtype=torch.long)
        cache = None
        generated: list[int] = []
        with torch.inference_mode():
            logits, cache = model.forward_cached(tokens, cache=None)
            for _ in range(args.max_new_tokens):
                next_id = int(logits[0, -1].argmax().item())
                generated.append(next_id)
                if eos is not None and next_id == eos:
                    break
                nxt = torch.tensor([[next_id]], device=device, dtype=torch.long)
                logits, cache = model.forward_cached(nxt, cache=cache)
        text = tokenizer.decode(generated)
        print(f"\nQ: {prompt}\nA: {text.strip()}", flush=True)


if __name__ == "__main__":
    main()
