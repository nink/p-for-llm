# PFor (nink fork)

[中文文档](README_ZH.md)

Fork of [cyfrit/p-for-llm](https://github.com/cyfrit/p-for-llm) at [nink/p-for-llm](https://github.com/nink/p-for-llm).

Offline **180.9M** PLE-MoE on **Waveshare ESP32-P4-ETH**. Native context is **1,024 tokens**; this fork adds on-device compression so a long paste (~8k tokens) can still run on that window.

This repo’s hardware, host tools, and training path are **not** the upstream WT9932P4-Tiny / USB-Serial-JTAG / RTX 5060 Ti setup.

Details: **[docs/CONTEXT-COMPRESSION.md](docs/CONTEXT-COMPRESSION.md)** · **[docs/SD-PAYLOAD.md](docs/SD-PAYLOAD.md)**

## Hardware (this fork)

![Waveshare ESP32-P4-ETH](docs/esp32-p4-eth.jpg)

Brought up on **[Waveshare ESP32-P4-ETH](https://www.waveshare.com/wiki/ESP32-P4-ETH)**.

| | |
| --- | --- |
| Board | Waveshare ESP32-P4-ETH |
| SoC | ESP32-P4 **v1.3** (360 MHz in this firmware) |
| Memory | **32 MB** PSRAM · **16 MB** Flash |
| Ethernet | 10/100 RJ45 (IP101 PHY, RMII) · host protocol **TCP 8742** |
| USB Type-C | **CH343 UART** — flash + host when Ethernet is down. Not Espressif USB-Serial-JTAG. |
| Host UART | UART0 **460800** · GPIO37 TX / GPIO38 RX |
| Storage | microSD `pfor-psram.bin` · SDMMC 4-bit |

Boards are named as planets in `runtime/host/boards.json`:

| Name | Role | Transport (now) |
| --- | --- | --- |
| **Sun** | Coordinator | Ethernet `192.168.72.42:8742` |
| **Mercury** | Expert worker | UART COM6 until a switch is in place |

Next desk cluster: **6× P4-ETH** on an **8-port gigabit switch** (P4 PHY still 100M). Same SKU, 32 MB PSRAM, A1/A2 microSD per board.

## Host I/O

```powershell
# Ethernet (preferred)
python runtime/host/chat.py --board sun

# UART fallback (CH343)
python runtime/host/chat.py --board mercury
# or: python runtime/host/chat.py --port COM6 --artifact pfor-180m.llmcraft
```

Do **not** pass `--reset` unless you intend to reboot (clears PSRAM; SD reloads if present).

Custom `.llmcraft` (FineWeb run) cannot use stock `flash.py` SHA checks — use `runtime/host/flash_model.py` (manifest + PLE only) and keep Sun on the original image until Mercury is compared.

## Quick start

```bash
git clone https://github.com/nink/p-for-llm.git
cd p-for-llm
python -m pip install esptool pyserial
```

Stock model weights still come from the [upstream Release](https://github.com/cyfrit/p-for-llm/releases/latest) (`pfor-180m.llmcraft`). **Firmware must be this fork** (ESP-IDF v6 + Zig): UART, Ethernet, SD, compress.

```powershell
# After building/flashing this fork's firmware:
python runtime/host/prepare_sd_payload.py --artifact pfor-180m.llmcraft --out pfor-psram.bin
# copy pfor-psram.bin to FAT32 card root, insert, power on

python runtime/host/chat.py --board sun
python runtime/host/chat.py --board sun --compress --context-file runtime/host/testdata/sample_long_context.md
```

Commands: `/help`, `/clear`, `/reload`, `/exit`.

## Model (single chip)

![PFor PLE-MoE on ESP32-P4](docs/architecture.png)

The diagram is the **single-chip** PLE-MoE (unchanged math). This fork loads PSRAM from **microSD** (or Ethernet/UART host), not USB-Serial-JTAG.

| Parameter | Value |
| --- | --- |
| Parameters | 180,920,432 (~180.9M) |
| Layers | 12 |
| Hidden size | 192 |
| Vocabulary | 32,768 (Qwen3.5-pruned; `training/assets/qwen3.5-english-tokenizer`) |
| Attention | 6 Q heads, 2 KV heads, head dim 32 |
| MoE | 29 experts/layer, **Top-1** on one P4 |
| Expert FFN | 512 |
| PLE dimension | 176 |
| Context | 1,024 native (~8k **effective** via compress) |
| Quantization | W1.58A8 |

~44 MiB across Flash + PSRAM. MoE in PSRAM, PLE in Flash (XIP), KV in on-chip RAM. Runtime uses Espressif XespV. Decode on this board is about **8 tok/s**.

## Training (this fork)

Upstream PFor was trained ~12B tokens on an **RTX 5060 Ti**. That run is **not** reproduced here (no original `.pt`).

This fork trains on a Linux box **`nink-ROMED8-2T` (`192.168.72.70`)**:

| | |
| --- | --- |
| GPU | **NVIDIA CMP 170HX 64 GB** (`CUDA_VISIBLE_DEVICES=0`) |
| Note | Two RTX 3090s are in PCI on that machine but **not** claimed by the CMP unlocker driver |
| Data | FineWeb-Edu Dedup parquet (one shard) + TinyStories for smokes |
| Code | `training/` with uv + torch CUDA |

```bash
# on the 170HX box
bash training/scripts/setup-uv.sh
bash training/scripts/run-main-smoke.sh      # 3-step sanity
bash training/scripts/run-main-fineweb.sh    # 1 epoch ~28k steps
```

A FineWeb-from-scratch 180M is a **learning run** (~1B tokens), not a replacement for the 12B upstream checkpoint until eval on Sun vs Mercury says otherwise.

## Multi-P4 (next)

Top-1 on one chip already holds 29 experts. Extra P4s buy **capacity**, not 32× speed.

| Stage | Plan |
| --- | --- |
| 2 boards | **58 experts, top-2** — Sun keeps attention/KV/router; Mercury holds 29 experts; Ethernet RPC of 192-byte activations |
| 6 boards | Same RPC; extra boards as more expert shards and/or knowledge-pack retrieve |
| Interconnect | **Switched Ethernet** (8-port gigabit). Not a hub. USB OTG is theoretically faster wire, worse for a cluster |

See `training/configs/p4-dual.json`.

## License

See [LICENSE](LICENSE). Upstream architecture and 180.9M recipe: [cyfrit/p-for-llm](https://github.com/cyfrit/p-for-llm).
