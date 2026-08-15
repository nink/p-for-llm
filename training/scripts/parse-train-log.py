from pathlib import Path
p = Path("/home/nink/pfor-ckpts/main-fineweb/train.log")
text = p.read_text(errors="replace").replace("\r", "\n")
keys = (
    "device=",
    "data_pool=",
    "batch_size=",
    "steps_per_epoch",
    "preparing_data",
    "packing ",
    "step=",
    "checkpoint=",
)
for line in text.splitlines():
    if "Failed to load" in line:
        continue
    if any(k in line for k in keys):
        print(line[:500])
print("---TAIL---")
for line in text.splitlines()[-40:]:
    if line.strip() and "Failed to load" not in line:
        print(line[:500])
