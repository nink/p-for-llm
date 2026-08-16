#!/usr/bin/env python3
"""Greedy 5-prompt gate: reconstructed original vs an SFT checkpoint."""

from __future__ import annotations

import argparse
import dataclasses
import json
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


def load_model(checkpoint: Path, device: torch.device) -> tuple[LLMM, int]:
    with torch.serialization.safe_globals([BitNetTrainingLinearWeight]):
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
    raw = payload["run_contract"]["model_config"]
    allowed = {field.name for field in dataclasses.fields(ModelConfig)}
    config = ModelConfig(**{key: value for key, value in raw.items() if key in allowed})
    model = LLMM(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    step = int(payload["training_state"]["step"])
    return model, step


def greedy(
    model: LLMM,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    eos = tokenizer.token_to_id("<|im_end|>")
    encoded = tokenizer.encode(format_chat(prompt)).ids
    tokens = torch.tensor([encoded], device=device, dtype=torch.long)
    generated: list[int] = []
    with torch.inference_mode():
        logits, cache = model.forward_cached(tokens, cache=None)
        for _ in range(max_new_tokens):
            next_id = int(logits[0, -1].argmax().item())
            generated.append(next_id)
            if eos is not None and next_id == eos:
                break
            nxt = torch.tensor([[next_id]], device=device, dtype=torch.long)
            logits, cache = model.forward_cached(nxt, cache=cache)
    return tokenizer.decode(generated).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original",
        type=Path,
        default=Path("/home/nink/pfor-ckpts/original-reconstructed/step-00000000.pt"),
    )
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("assets/qwen3.5-english-tokenizer/tokenizer.json"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.candidate is not None:
        candidate = args.candidate
    elif args.candidate_dir is not None:
        candidate = latest_training_checkpoint(args.candidate_dir)
    else:
        raise SystemExit("pass --candidate or --candidate-dir")
    if candidate is None or not candidate.is_file():
        raise SystemExit(f"no candidate checkpoint at {candidate}")

    device = torch.device(args.device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    original, original_step = load_model(args.original, device)
    original_answers = {
        prompt: greedy(original, tokenizer, prompt, device, args.max_new_tokens)
        for prompt in PROMPTS
    }
    del original
    torch.cuda.empty_cache()

    sft, sft_step = load_model(candidate, device)
    sft_answers = {
        prompt: greedy(sft, tokenizer, prompt, device, args.max_new_tokens)
        for prompt in PROMPTS
    }

    rows = [
        {
            "q": prompt,
            "original": original_answers[prompt],
            "sft": sft_answers[prompt],
        }
        for prompt in PROMPTS
    ]
    report = {
        "original_checkpoint": str(args.original),
        "original_step": original_step,
        "sft_checkpoint": str(candidate),
        "sft_step": sft_step,
        "prompts": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
