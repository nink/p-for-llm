#!/usr/bin/env python3
"""Train a tiny hashed-linear model on canada-pack Q&A (numpy only)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
DIM = 4096
EPOCHS = 8
LR = 0.4
NGRAM = (3, 4, 5)

TOKEN = re.compile(r"[a-z0-9']+")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def hashes(text: str) -> list[int]:
    s = f"  {text.casefold()}  "
    feats = [0]
    for n in NGRAM:
        for i in range(len(s) - n + 1):
            feats.append(hash(s[i : i + n]) % DIM)
    for tok in TOKEN.findall(s):
        feats.append(hash(f"w:{tok}") % DIM)
    return feats


def featurize(text: str) -> np.ndarray:
    x = np.zeros(DIM, dtype=np.float32)
    idx = hashes(text)
    for i in idx:
        x[i] += 1.0
    n = np.linalg.norm(x)
    if n > 0:
        x /= n
    return x


def encode_split(rows: list[dict], answers: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if answers is None:
        answers = sorted({row["answer"] for row in rows})
    inv = {name: i for i, name in enumerate(answers)}
    x = np.stack([featurize(row["question"]) for row in rows])
    y = np.array([inv.get(row["answer"], -1) for row in rows], dtype=np.int32)
    return x, y, answers


def train(x: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    w = rng.normal(0, 0.01, size=(n_classes, DIM)).astype(np.float32)
    order = np.arange(len(y))
    for epoch in range(EPOCHS):
        rng.shuffle(order)
        correct = 0
        for i in order:
            logits = w @ x[i]
            logits -= logits.max()
            exp = np.exp(logits)
            p = exp / exp.sum()
            pred = int(p.argmax())
            correct += pred == y[i]
            p[y[i]] -= 1.0
            w -= LR * np.outer(p, x[i])
        acc = correct / len(y)
        print(f"epoch {epoch + 1}/{EPOCHS} train_acc={acc:.3f}", flush=True)
    return w


def predict(w: np.ndarray, x: np.ndarray, answers: list[str]) -> list[str]:
    logits = x @ w.T
    return [answers[i] if i >= 0 else "" for i in logits.argmax(axis=1)]


def accuracy(pred: list[str], rows: list[dict]) -> float:
    return sum(p == r["answer"] for p, r in zip(pred, rows)) / max(len(rows), 1)


def main() -> None:
    train_rows = load_jsonl(OUT / "qa-train.jsonl")
    eval_rows = load_jsonl(OUT / "eval-200.jsonl")
    x_train, y_train, answers = encode_split(train_rows)
    print(f"train={len(train_rows)} eval={len(eval_rows)} classes={len(answers)} dim={DIM}")
    w = train(x_train, y_train, len(answers))
    train_pred = predict(w, x_train, answers)
    x_eval, y_eval, _ = encode_split(eval_rows, answers)
    eval_pred = predict(w, x_eval, answers)
    known = y_eval >= 0
    print(f"train_exact={accuracy(train_pred, train_rows):.3f}")
    print(f"eval_exact={accuracy(eval_pred, eval_rows):.3f} (all rows)")
    if known.any():
        subset = [eval_rows[i] for i, ok in enumerate(known) if ok]
        subset_pred = [eval_pred[i] for i, ok in enumerate(known) if ok]
        print(f"eval_exact_known_answers={accuracy(subset_pred, subset):.3f} n={len(subset)}")
    np.savez(
        OUT / "model.npz",
        w=w,
        answers=np.array(answers),
        dim=np.array([DIM]),
    )
    print(f"wrote {OUT / 'model.npz'}")
    for row, pred in list(zip(eval_rows, eval_pred))[:8]:
        mark = "ok" if pred == row["answer"] else "miss"
        print(f"  [{mark}] {row['question']} -> {pred!r} (gold {row['answer']!r})")


if __name__ == "__main__":
    main()
