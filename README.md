# PFor

[中文文档](README_ZH.md)

## PLE-MoE-W1.58A8

## Architecture

![PFor architecture](docs/architecture.png)

![WT9932P4-Tiny development board](docs/wt9932p4-tiny.jpg)

PFor is an LLM running on ESP32-P4, although technically it should be called an SLM. It has Instruct(ChatML) and Agent capabilities, despite both being extremely early and highly unstable. Its inference speed on ESP32-P4 is about 9 tokens/s.

I use the WT9932P4-Tiny development board (I bought mine on Taobao for CNY 39.9, about USD 6; it is around USD 10 on AliExpress), which has 32 MB PSRAM and 16 MB Flash.

The USB connection admittedly makes "offline" look somewhat questionable. The board's Flash cannot hold the weights loaded into PSRAM, so they are transferred over USB at startup; the host performs no inference. Give it an SD card for the weights, and it can run entirely without a host.

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
| Context | 1,024 |
| Quantization | W1.58A8 |

With mixed ternary, Q8, and FP16 storage, the 180.9M-parameter model occupies about 44 MiB across Flash and PSRAM.

PFor uses a MoE + PLE architecture to make full use of the hardware. MoE weights are stored in PSRAM, PLE weights are stored in Flash and accessed with XIP, and the cache and working memory are stored in DIRAM. The runtime mainly uses Espressif XespV acceleration. During development, XespV tests showed speedups of up to 30x.

The WT9932P4-Tiny uses ESP32-P4 v1.3. Early P4 revisions can only run at 360 MHz. If you have a newer v3 chip, it can run at 400 MHz and may provide higher generation speed.

Compared with ESP32-S3 (thanks to @slvDev for the work), P4 provides much stronger performance, including new hardware accelerators and support for up to 32 MB PSRAM.

The vocabulary is pruned from Qwen3.5. Thanks to Qwen.

## Examples

### Terminal chat

```text
user
What is the capital of France?

assistant
The capital of France is Paris.
```

````text
user
Write a Python program.

assistant
Here's an example of a Python program that accomplishes this task:

```python
def print_and_print():
    for i, item in enumerate(items):
        print(i)
    print()
```

This program will create a new list called `items` of items called `item_list` with items from `items` that are not already in the `items`. The `print()` function is used to display the item's `item` in the `items` list.
````

```text
user
What is the United States?

assistant
The United States is the third largest country in the world, with around 1.8% of the country's population.
```

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

Although its output is often little more than nonsense and frequently goes off the rails, it still demonstrates some ability to follow instructions and stay roughly on topic. Its Agent capability is not general-purpose: it was trained with a fixed tool prompt. Even so, it shows some ability to understand tasks and drive tools, albeit very unstably.

## Installation

Clone the repository:

```bash
git clone https://github.com/cyfrit/p-for-llm.git
cd p-for-llm
```

Download these files from the [latest Release](https://github.com/cyfrit/p-for-llm/releases/latest) and place them in the repository root:

- [`pfor-esp32p4.zip`](https://github.com/cyfrit/p-for-llm/releases/latest/download/pfor-esp32p4.zip)
- [`pfor-180m.llmcraft`](https://github.com/cyfrit/p-for-llm/releases/latest/download/pfor-180m.llmcraft)
- [`SHA256SUMS`](https://github.com/cyfrit/p-for-llm/releases/latest/download/SHA256SUMS)

Verify the downloads and install `esptool`:

```bash
sha256sum --check SHA256SUMS
python3 -m pip install esptool
```

Flash the firmware and model:

```bash
python3 runtime/host/flash.py \
  --firmware pfor-esp32p4.zip \
  --model pfor-180m.llmcraft \
  --port <PORT>
```

Start terminal chat:

```bash
python3 runtime/host/chat.py --port <PORT> --artifact pfor-180m.llmcraft
```

Commands: `/help`, `/clear`, `/reload`, `/exit`.

Run the Agent demo:

```bash
python3 runtime/host/agent_demo.py --port <PORT> --artifact pfor-180m.llmcraft
```

The released firmware targets pre-v3 P4 hardware at 360 MHz. Newer P4 revisions may require rebuilding from source with a matching configuration.

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

## License

See [LICENSE](LICENSE).
