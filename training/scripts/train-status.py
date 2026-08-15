#!/usr/bin/env python3
from pathlib import Path
import glob
import os
import subprocess

log = Path("/home/nink/pfor-ckpts/main-fineweb/train.log")
raw = log.read_bytes().replace(b"\r", b"\n").decode("utf-8", "replace")
lines = [ln for ln in raw.split("\n") if ln.strip()]
print("--- last ---")
print("\n".join(lines[-8:]))
print("--- val ---")
hits = [ln for ln in lines if "validation_causal_loss=" in ln]
print("\n".join(hits[-12:]) if hits else "(none)")
print("--- files ---")
print("exited", Path("/home/nink/pfor-ckpts/main-fineweb/train-exited.txt").exists())
print("llmcraft", Path("/home/nink/pfor-ckpts/main-fineweb/pfor-180m-fineweb.llmcraft").exists())
files = sorted(glob.glob("/home/nink/pfor-ckpts/main-fineweb/*.pt"), key=os.path.getmtime)
for f in files[-4:]:
    print(os.path.basename(f), os.path.getsize(f))
print("--- procs ---")
subprocess.run(["bash", "-lc", "pgrep -af 'llmm_llm.train --model main' || true; pgrep -af wait-export || true"])
