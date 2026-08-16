# Canada gazetteer pack (parked)

NRCan CGNDB names → seekable `canada.kpack` (KPK1) → ESP32-P4 retrieve. Templated “which province is X in?” scored **200/200** on `out/eval-200.jsonl`. Free-form questions (nearest city, population, typos) are **not** RAG; substring match is not an AI demo. Paused for rethink.

## In this folder

| Path | What |
| --- | --- |
| `lookup.py` | PC retrieve (fold + capitals + n-gram keys) |
| `build_kpack.py` / `kpack.py` | Binary pack matching `runtime/esp32-p4/main/llmm_pack.c` |
| `pack_query.py` | Ethernet/UART query (and optional `--put`) |
| `score_lookup.py` / `score_kpack.py` | Eval against `out/eval-200.jsonl` |
| `out/eval-200.jsonl` / `out/qa-train.jsonl` | Question sets |
| `build_cgn_index.py` | Rebuild `out/cgn-index.json` from NRCan CSV (not in git) |

Not committed: `canada.kpack` (~17 MB), CGNDB CSV, GeoNames `CA.txt`, Flan-T5 run dirs, the 55 MB JSON index.

## Query a board that already has the file

```powershell
python canada-pack/pack_query.py --host 192.168.72.77
```

Do not pass `--put` unless you intend to rewrite `/sdcard/canada.kpack`. Sun (`192.168.72.42`) was not flashed with pack firmware.
