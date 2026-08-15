# PFor (nink fork)

[中文文档](README_ZH.md)

Fork of [cyfrit/p-for-llm](https://github.com/cyfrit/p-for-llm) at [nink/p-for-llm](https://github.com/nink/p-for-llm).

Offline **180.9M** PLE-MoE on ESP32-P4. Native context is still **1,024 tokens**. This fork’s first goal is **on-device context compression** so a long paste (~8k tokens) can still run on that window.

Details: **[docs/CONTEXT-COMPRESSION.md](docs/CONTEXT-COMPRESSION.md)** · **[docs/SD-PAYLOAD.md](docs/SD-PAYLOAD.md)**

## Hardware (this fork)

Brought up on **[Waveshare ESP32-P4-ETH](https://www.waveshare.com/wiki/ESP32-P4-ETH)** — not the upstream WT9932P4-Tiny.

| | |
| --- | --- |
| Board | Waveshare ESP32-P4-ETH |
| SoC | ESP32-P4 **v1.3** (360 MHz in this firmware) |
| Memory | **32 MB** PSRAM · **16 MB** Flash |
| USB | Single Type-C → **CH343 UART** (host + flash). No Espressif USB-Serial-JTAG on that connector. |
| Host UART | UART0 **460800** · GPIO37 TX / GPIO38 RX |
| Storage | microSD (TF) for `pfor-psram.bin` · SDMMC 4-bit (CLK 43, CMD 44, D0–D3 39–42, GPIO45 power low, LDO ch. 4) |

On Windows this CH343 shows up as **COM5** (`USB-Enhanced-SERIAL CH343`). Flash, chat, and model load all use that port.

Upstream’s stock host protocol expects USB-Serial-JTAG (Espressif VID `303A`). That path is unused on this single-USB board.

## Changes vs upstream (high level)

1. **UART host protocol** (`LLMM_HOST_UART=1`) so chat works on the ETH board’s CH343 instead of USB-Serial-JTAG.
2. **Windows host client** (`pyserial` in `runtime/host/p4.py`).
3. **SD boot load** — copy `pfor-psram.bin` to a FAT32 card and skip the ~10 minute UART weight transfer. See [docs/SD-PAYLOAD.md](docs/SD-PAYLOAD.md).
4. **`--reset` is opt-in** — default connect does not pulse RTS (that reboot would drop PSRAM).
5. **On-device extractive compression** (`runtime/esp32-p4/main/llmm_compress.c`):
   - Host sends the **raw** long prompt (up to **48 KiB**).
   - Board trims to a **≤400 B** fitted packet (~80–100 tokens) for faster prefill.
   - Raw prompts **≤1024 B** skip compression (passthrough).
6. **Measured on this board** (decode ~**8 tok/s**): short TTFT ~**1.6 s**; ~8k-token source compressed on-device TTFT ~**7.6 s** (was ~20 s before the 400 B cap).

Model quality is still upstream’s undertrained 180.9M — compression changes **what fits**, not how well it writes.

## Quick start (Waveshare ETH / COM5)

Clone this fork (not upstream) if you want the UART + SD + compress path:

```bash
git clone https://github.com/nink/p-for-llm.git
cd p-for-llm
```

Download from the [upstream latest Release](https://github.com/cyfrit/p-for-llm/releases/latest) into the repo root:

- [`pfor-esp32p4.zip`](https://github.com/cyfrit/p-for-llm/releases/latest/download/pfor-esp32p4.zip)
- [`pfor-180m.llmcraft`](https://github.com/cyfrit/p-for-llm/releases/latest/download/pfor-180m.llmcraft)
- [`SHA256SUMS`](https://github.com/cyfrit/p-for-llm/releases/latest/download/SHA256SUMS)

```powershell
python -m pip install esptool pyserial
python runtime/host/flash.py --firmware pfor-esp32p4.zip --model pfor-180m.llmcraft --port COM5
```

That flashes **upstream** images. For UART host + SD + on-device compress, **rebuild and flash this fork’s firmware** (ESP-IDF v6 + Zig), then:

```powershell
# Optional: fast PSRAM load from microSD
python runtime/host/prepare_sd_payload.py --artifact pfor-180m.llmcraft --out pfor-psram.bin
# copy pfor-psram.bin to the FAT32 card root, insert, then:

python runtime/host/chat.py --port COM5 --artifact pfor-180m.llmcraft

# Long context: PC sends raw text; P4 compresses
python runtime/host/chat.py --port COM5 --artifact pfor-180m.llmcraft `
  --compress --context-file runtime/host/testdata/sample_long_context.md
```

Do **not** pass `--reset` on every chat unless you intend to reboot (clears PSRAM; SD will reload if present).

Commands: `/help`, `/clear`, `/reload`, `/exit`.

## PLE-MoE-W1.58A8

## Architecture

![PFor architecture](docs/architecture.png)

![WT9932P4-Tiny development board (upstream)](docs/wt9932p4-tiny.jpg)

PFor is an LLM running on ESP32-P4 (technically an SLM). It has Instruct (ChatML) and Agent capabilities, both early and unstable. Decode on this P4 is about **8–9 tokens/s**.

The USB cable does not mean the PC is doing inference. Flash cannot hold the PSRAM weights; without an SD payload they are transferred over UART at startup. With `pfor-psram.bin` on microSD, the board can load weights without that long transfer.

## Model

| Parameter | Value |
| --- | --- |
| Parameters | 180,920,432 (~180.9M) |
| Layers | 12 |
| Hidden size | 192 |
| Vocabulary | 32,768 |
| Attention | 6 Q heads, 2 KV heads, head dim 32 |
| MoE | 29 experts per layer, Top-1 routing |
| Expert FFN | 512 |
| PLE dimension | 176 |
| Context | 1,024 native (this fork: ~8k **effective** via compress) |
| Quantization | W1.58A8 |

With mixed ternary, Q8, and FP16 storage, the 180.9M-parameter model occupies about 44 MiB across Flash and PSRAM.

PFor uses a MoE + PLE architecture. MoE weights sit in PSRAM, PLE in Flash (XIP), KV and workspace in on-chip RAM. The runtime uses Espressif XespV.

This ETH board’s P4 is **v1.3** (360 MHz). Newer v3 silicon can run at 400 MHz.

The vocabulary is pruned from Qwen3.5. Thanks to Qwen.

## Examples

### Terminal chat

```text
user
What is the capital of France?

assistant
The capital of France is Paris.
```

Output is often weak or off-topic — that is the undertrained 180.9M, not the UART/compress path.

### Agent

```text
Task: In config/cache.ini, change cache_mode from lazy to eager.

Agent: Search config/cache.ini cache_mode
Tool: OK
4: cache_mode = lazy

Agent: Replace config/cache.ini 4 cache_mode = eager
Tool: OK replaced config/cache.ini:4

Agent: Finish Updated cache_mode to eager.
```

Agent mode is not general-purpose; it was trained with a fixed tool prompt.

## Training

PFor was trained on approximately 12B of raw data using only RTX 5060 Ti. Due to insufficient training data, PFor is highly unstable and its capabilities are extremely limited. However, it still shows the characteristics of an LLM and can complete simple Agent tasks in specific formats.

Based on its current performance, more training should improve it considerably. Small-model architectures such as LFM2.5 have already shown good capabilities. With complete training, a 180.9M-parameter model should have much better results. With training data tailored to a specific vertical domain, the current architecture can already support on-device instruction analysis, information extraction, intent classification, and structured command routing.

| Dataset |
| --- |
| FineWeb-Edu Dedup |
| DCLM |
| Cosmopedia v2 |
| peS2o |
| Wikipedia English |
| FineMath 4+ |
| Python-Edu-Cleaned |
| Magpie-Pro-300K-Filtered |
| UltraChat 200K |
| Smol-SmolTalk |
| Tulu 3 instruction-following personas |
| Manual Agent task cards |
| Verified programmatic 50,000-card corpus |

## Future

Because PFor uses MoE, experts could theoretically be distributed across multiple MCUs. With Top-1 routing, 8-bit activations, hidden size 192, and 12 layers, each generated token requires about 4,608 bytes of expert activation transfer. At 9 tokens/s, the theoretical bandwidth is about **40.5 KiB/s** in both directions combined, excluding protocol overhead.

This fork’s near-term work is **better compression quality at the same fitted budget**, not a larger native KV.

## License

See [LICENSE](LICENSE).
