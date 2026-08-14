# Goal 1: Context compression (effective long context)

This document is the first product/engineering goal for the **nink/p-for-llm** fork.

Upstream PFor exposes a native context window of **1,024 tokens** (KV/RAM limited on ESP32-P4). We are **not** trying to raise native KV to 8K in this goal. We are raising **effective** context via compression.

## Target

| Metric | Goal |
| --- | --- |
| Compression ratio | **~8:1** (source tokens → tokens sent to the model) |
| Native window | Still **1,024** |
| Effective input | ~**8,000** tokens of source material per turn (before reply budget) |
| First implementation | **Host-side** (PC), wrapping the existing USB chat path |
| Later (optional) | On-device distill pass; finetune model on compressed “pack state” dialect |

Example: an ~8k-token document or chat history is reduced to ~1k tokens of dense state, then passed to `P4Device.text()` as today.

## Architecture

Effective long context = **compress outside the model**, then run normal PFor inference inside the native 1,024-token window.

```mermaid
flowchart TB
  subgraph sources ["Long source (~8k tokens)"]
    DOC["Document / paste"]
    HIST["Chat history"]
    Q["User question"]
  end

  subgraph host ["Host PC — nink fork v1"]
    IN["Assemble source + question"]
    CMP["compress.py<br/>~8:1 compression"]
    PACK["Pack state (~1k tokens)<br/>facts · numbers · negations · MEM"]
    CHAT["chat.py<br/>format_chat_prompt()"]
  end

  subgraph device ["ESP32-P4 — unchanged native window"]
    USB["USB session"]
    PFOR["PFor inference<br/>max_seq_len = 1024"]
    OUT["Short answer"]
  end

  DOC --> IN
  HIST --> IN
  Q --> IN
  IN --> CMP
  CMP --> PACK
  PACK --> CHAT
  CHAT -->|"prompt ≤ ~1024"| USB
  USB --> PFOR
  PFOR --> OUT
  OUT -->|"distill turn → MEM"| HIST
```

### Trust boundary

| Stage | Where | Role |
| --- | --- | --- |
| Ingest long text | Host | Accept ~8k-token source |
| Compress 8:1 | Host `compress.py` | Lossy but structured packet |
| Generate | P4 PFor | Native 1,024 KV only |
| Roll memory | Host | `MEM` keeps multi-turn effective context |

```mermaid
sequenceDiagram
  actor User
  participant Chat as chat.py
  participant Comp as compress.py
  participant P4 as ESP32-P4 PFor

  User->>Chat: long context + question
  Chat->>Comp: source tokens (~8k)
  Comp-->>Chat: pack state (~1k)
  Chat->>P4: ChatML prompt (fits 1024)
  P4-->>Chat: answer tokens
  Chat-->>User: answer
  Note over Chat,Comp: Optional: answer+history → updated MEM for next turn
```

### Compression sketch (pack state)

```text
SRC  ████████████████████████████████  ~8000 tokens
         │  8:1 compress
         ▼
PKT  ████                              ~1000 tokens  →  PFor (1024 window)
         + question + reply budget
```

## What success looks like

1. User (or tool) provides long text + a question.
2. Fork compresses to a short packet (schema and/or extractive summary).
3. Board generates an answer using only that packet + question.
4. Measured ratio ≈ 8:1 on a fixed eval set, with acceptable factual retention (numbers, names, negations preserved).

Multi-turn: compress running history into a rolling `MEM` blob so sessions stay long without growing the native window.

## Non-goals (for this milestone)

- Expanding on-chip KV cache to true 8K context
- Domain apps (vehicle diagnostics, appliances, education packs)
- Replacing upstream training/runtime architecture

## Planned code touchpoints

- `runtime/host/compress.py` — compression API (8:1 target)
- `runtime/host/chat.py` — optional `--compress` path before `device.text(...)`
- Eval fixtures under `docs/` or `runtime/host/testdata/` — ratio + quality checks

## Status

**Planned.** Hardware bring-up (stock PFor on ESP32-P4) comes first; then host-side compression on this fork.
