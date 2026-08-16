#!/usr/bin/env python3
"""Fine-tune Flan-T5-small to answer from retrieved Canada gazetteer context."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lookup import context_for, load_index  # noqa: E402

OUT = ROOT / "out"
MODEL_ID = "google/flan-t5-small"
SAVE = OUT / "flan-t5-canada"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prompt(question: str, context: str) -> str:
    if not context:
        return (
            "Answer the Canadian geography question. If unknown, say Unknown.\n"
            f"question: {question}"
        )
    return (
        "Answer using only the context. Copy the province/territory or place from context.\n"
        f"question: {question}\ncontext: {context}"
    )


def build_pairs(rows: list[dict], index: dict) -> list[dict]:
    pairs = []
    for row in rows:
        ctx = context_for(row["question"], index)
        pairs.append(
            {
                "input": prompt(row["question"], ctx),
                "target": row["answer"],
            }
        )
    return pairs


def main() -> None:
    import torch
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    index = load_index()
    train_rows = load_jsonl(OUT / "qa-train.jsonl")
    eval_rows = load_jsonl(OUT / "eval-200.jsonl")
    train_pairs = build_pairs(train_rows, index)
    eval_pairs = build_pairs(eval_rows, index)
    print(f"train={len(train_pairs)} eval={len(eval_pairs)} model={MODEL_ID}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

    def tokenize(pair: dict) -> dict:
        src = tokenizer(
            pair["input"],
            max_length=160,
            truncation=True,
            padding=False,
        )
        tgt = tokenizer(
            pair["target"],
            max_length=32,
            truncation=True,
            padding=False,
        )
        src["labels"] = tgt["input_ids"]
        return src

    train_ds = [tokenize(p) for p in train_pairs]
    eval_ds = [tokenize(p) for p in eval_pairs]

    args = Seq2SeqTrainingArguments(
        output_dir=str(OUT / "flan-runs"),
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        learning_rate=3e-4,
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=32,
        report_to=[],
        fp16=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )
    trainer.train()
    SAVE.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(SAVE))
    tokenizer.save_pretrained(str(SAVE))

    model.eval()
    correct = 0
    shown = 0
    with torch.no_grad():
        for row, pair in zip(eval_rows, eval_pairs):
            enc = tokenizer(pair["input"], return_tensors="pt", truncation=True, max_length=160)
            out = model.generate(**enc, max_new_tokens=32)
            pred = tokenizer.decode(out[0], skip_special_tokens=True).strip()
            ok = pred.casefold() == row["answer"].casefold()
            correct += ok
            if shown < 12:
                print(f"{'ok' if ok else 'miss'}  {row['question']} -> {pred!r} (gold {row['answer']!r})", flush=True)
                shown += 1
    print(f"eval_exact={correct / len(eval_rows):.3f} {correct}/{len(eval_rows)}")
    print(f"saved {SAVE}")


if __name__ == "__main__":
    main()
