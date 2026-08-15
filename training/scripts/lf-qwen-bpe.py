from pathlib import Path
p = Path("/home/nink/pfor-work/training/tokenizer/pruning/qwen_bpe.py")
p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n"))
print("lf-ok", p.stat().st_size)
