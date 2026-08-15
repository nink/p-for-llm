# Training (nink fork)

This tree trains the PFor 180.9M PLE-MoE. It is meant to run on the **NVIDIA CMP 170HX 64 GB** at `192.168.72.70` (`nink-ROMED8-2T`), not the upstream RTX 5060 Ti write-up.

| | |
| --- | --- |
| GPU | CMP 170HX 64 GB · `CUDA_VISIBLE_DEVICES=0` |
| Python | 3.12 + `uv` venv (see `scripts/setup-uv.sh`) |
| Tokenizer | `assets/qwen3.5-english-tokenizer/` — 32,768 vocab (must match on-device) |
| Smoke | `scripts/run-main-smoke.sh` (`--model main --max-steps 3`) |
| FineWeb epoch | `scripts/run-main-fineweb.sh` |

`--model tiny` is not used here (Triton gather issues). Dual-P4 config draft: `configs/p4-dual.json` (58 experts, top-2) — firmware RPC is not in that file.

The two RTX 3090s in that chassis are **not** visible under the CMP unlocker driver; do not point training at them on this boot.

Export to `.llmcraft` after a run: `python -m export_aircraft --checkpoint-dir … --output … --load-device cuda`. Stock `flash.py` refuses a custom artifact SHA; use `runtime/host/flash_model.py` on **Mercury** only while Sun stays on the original image.
