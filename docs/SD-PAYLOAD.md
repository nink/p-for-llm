# SD card model payload (Waveshare ESP32-P4-ETH)

Cold boot can load PSRAM weights from microSD in seconds instead of a ~10 minute UART transfer.

## Card

- FAT32 microSD (8 GB is fine)
- File at **root**: `pfor-psram.bin`

## Prepare the file (on PC)

```powershell
cd "c:\Users\peter\OneDrive\Documents\Ninks DealCheck\p-for-llm"
python runtime/host/prepare_sd_payload.py --artifact pfor-180m.llmcraft --out pfor-psram.bin
```

Copy `pfor-psram.bin` to the SD card root, eject, insert in the board TF slot.

## Boot behavior

1. Firmware mounts SD (SDMMC 4-bit, LDO + GPIO45 power for ETH board).
2. If `pfor-psram.bin` is present and CRC matches, PSRAM is filled and `loaded=1`.
3. If missing/invalid, board waits for UART `LLMPSR05` load (old slow path).

## Host chat

Do **not** pass `--reset` unless you want a reboot:

```powershell
python runtime/host/chat.py --port COM5 --artifact pfor-180m.llmcraft
```

With SD payload present, handshake should report the model already loaded after power-on.
