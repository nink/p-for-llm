#!/usr/bin/env bash
set -euo pipefail
cd /home/nink/pfor-work/training
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="/home/nink/pfor-work/training"
SRC=/tmp/qwen-tokenizer-src
mkdir -p "$SRC"
.venv/bin/python - <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path
dest = Path("/tmp/qwen-tokenizer-src")
dest.mkdir(parents=True, exist_ok=True)
for name in ("tokenizer.json", "tokenizer_config.json"):
    path = hf_hub_download(
        repo_id="Qwen/Qwen3.5-0.8B-Base",
        filename=name,
        revision="dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
        local_dir=str(dest),
    )
    print("got", path)
PY
.venv/bin/python -m tokenizer.pruning \
  --source-tokenizer "$SRC/tokenizer.json" \
  --source-config "$SRC/tokenizer_config.json" \
  --output assets/qwen3.5-english-tokenizer
.venv/bin/python - <<'PY'
import hashlib
from pathlib import Path
from tokenizers import Tokenizer
p = Path("assets/qwen3.5-english-tokenizer/tokenizer.json")
digest = hashlib.sha256(p.read_bytes()).hexdigest()
expected = "35bd0d23242520d31f0ba3c5587599164a380ad4f5f61fb16ca92aaef82eb491"
tok = Tokenizer.from_file(str(p))
print("sha256", digest)
print("match", digest == expected)
print("vocab", tok.get_vocab_size(with_added_tokens=True))
if digest != expected:
    raise SystemExit("pruned tokenizer hash mismatch")
if tok.get_vocab_size(with_added_tokens=True) != 32768:
    raise SystemExit("pruned tokenizer is not 32768")
PY
rm -rf data/raw/tokenizer-validation/tinystories/.llmm-pools
echo TOKENIZER_OK
