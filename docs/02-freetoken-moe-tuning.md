# 02 — MoE Hybrid Prefill Tuning: −35 % Wall Time with One Flag

**Engine: [FreeToken](https://github.com/FlashML-org/FreeToken) (`ft serve`), model: Qwen3.6-35B-A3B NPFP4 (MoE, 35 B total / 3 B active), 12 GB GPU + 32 GB RAM laptop.**

## The setup

The engine runs MoE models in a **hybrid** mode: attention and dense parts live on the GPU, while **expert weights live in system RAM and are either fetched over PCIe or computed on the CPU**.

```bash
ft serve --model-path <model> --host 0.0.0.0 --port 1919 \
  --memory-ratio 0.90 --num-pages 163840 \
  --moe-backend hybrid --moe-cache-size 1400 \
  --attention-backend fi --cache-type radix
```

The relevant flag defaults to **auto**:

```
--moe-hybrid-max-fetch MOE_HYBRID_MAX_FETCH
    For --moe-backend hybrid: max experts fetched over PCIe per (layer, decode step);
    the rest of that step's misses are computed on the CPU, overlapped.
    -1 (default) = auto: fetch the benched pcie/cpu bandwidth [ratio]
```

**Auto benchmarks the PCIe/CPU ratio once at startup.** That heuristic is computed for a stock system — and it went stale the moment we unlocked the GPU from 80 W to 175 W (see [01-power-unlock.md](01-power-unlock.md)): the GPU side got ~2× faster, so more work should be routed to it.

## The result

**`--moe-hybrid-max-fetch 128`** — one flag, measured on 42 K-token cold prompts:

| Configuration | Prefill wall time | Throughput | vs baseline |
|---|---|---|---|
| Default (`-1`, auto) | 20.8 s | ~2000 tok/s | — |
| `--moe-hybrid-max-fetch 64` | 16.7 s | ~2500 tok/s | −20 % |
| **`--moe-hybrid-max-fetch 128`** | **13.6 s** | **~3090 tok/s** | **−35 %** |
| `--moe-hybrid-max-fetch 256` | 17.6 s | ~2380 tok/s | −15 % |

Sample spread for the 128 setting (four cold prompts): **12.7 / 13.6 / 13.7 / 13.7 / 14.2 s** — stable.

### Why 256 is *slower* than 128

Because fetching is not free. The laptop's GPU link is **PCIe Gen4 x8** (the vendor routes only 8 lanes to the GPU — `lspci -vv` shows `LnkCap: 16GT/s x16` but `LnkSta: 16GT/s x8`). At ~16 GB/s, asking for more experts per step just queues transfers. **The optimum sits where PCIe throughput and CPU throughput balance — here, ~64–128.**

Check your own link before tuning:

```bash
sudo lspci -vv -s $(lspci | grep -i nvidia | awk '{print $1}' | head -1) | grep -E "LnkCap:|LnkSta:"
# measure under load — idle links are power-saved down to 2.5GT/s
```

## What did *not* help (and why the naive reading was wrong)

Two other flags looked promising early and were **removed after proper A/B**:

| Flag | Alone (no max-fetch) | Combined with max-fetch 128 | Verdict |
|---|---|---|---|
| `--moe-prefill-hit-d2d` | included in a −7 % combo | **hurt** (19.35 s vs 13.6 s) | **leave off** |
| `--moe-cpu-threads 32` | included in the same combo | **hurt** | **leave at 0 (auto = 16 physical cores)** |

**The lesson — parameter interactions are real.** A flag that helps in isolation can hurt once a bigger knob is turned, because it changes the workload balance. Concretely:

- With more experts going to the GPU, the CPU has *less* expert work — so `--moe-cpu-threads 32` (hyper-threading) becomes pure contention, slowing the CPU-side remainder.
- We re-ran the full matrix **with the new baseline (max-fetch=128 present)** before concluding. **Always re-baseline after any change that shifts the workload.**

## Full matrix (all runs: fresh cold prompt, engine restarted each time)

| # | `max-fetch` | d2d | cpu-threads | Wall time | Notes |
|---|---|---|---|---|---|
| 0 | −1 (auto) | off | 0 | 20.8 s | baseline |
| 1 | −1 | **on** | **32** | 19.35 s | looked like −7 %… |
| 2 | 64 | on | 32 | 16.7 s | …but composed |
| 3 | 128 | on | 32 | 16.7 s | |
| 4 | 256 | on | 32 | 17.6 s | PCIe saturation |
| 5 | 128 | off | 0 | **13.6 s** | **best — drop the other two flags** |
| 6 | 64 | off | 0 | 16.7 s → (later re-verified) | ≈ 128 |

Rows 1→5 are the story: the two "small" flags helped only while the *big* flag was absent. Once `max-fetch=128` was set, dropping them recovered another **3.1 s (−19 %)**.

## Final configuration

```ini
# in the systemd unit / serve command line
--moe-backend hybrid
--moe-cache-size 1400          # GPU expert slots — raise only if you have VRAM headroom
--moe-hybrid-max-fetch 128     # ← the one that matters
# do NOT set --moe-cpu-threads (leave auto), do NOT enable --moe-prefill-hit-d2d
```

`--moe-cache-size` is the *other* obvious lever — cache more experts on the GPU so nothing needs fetching — but on a 12 GB card with a 160 K-token KV cache we had only ~800 MB free (11470/12282 MiB used), and earlier attempts at larger cache values OOM'd at startup. **If you have VRAM headroom, raising the cache is the cleanest win; if not, `max-fetch` is the lever that works within the same footprint.**

## Measuring your own runs (things that will fool you)

1. **Radix cache makes repeat prompts instant.** The second identical request returns in ~1.5 s with `#cached-token: 42880` — that is *not* prefill time. **Every measurement needs a fresh prompt** (or a restarted server).
2. **First run after a restart is systematically slower.** Our first sample read 23.3 s where the steady state was 19.3–19.6 s. **Discard the first sample or warm up.**
3. **The API's "TTFT" on a streaming request is useless** here: the stream's first byte arrives in ~0.004 s (headers), long before prefill completes. Use the wall time of a `max_tokens: 5` request, or read the engine's own log line `Prefill batch ... input throughput (token/s)`.
4. **"Model loaded" ≠ ready.** The HTTP API answers while weights are still loading (`{"error": "model is still loading"}` for ~60–90 s on a 35 B model). Wait for `nvidia-smi` memory to reach its plateau **and** the API to stop erroring.

See [docs/04-methodology.md](04-methodology.md) for the full checklist.

## What this means in practice

For an agent workload that repeatedly sends 40–80 K-token contexts, prefill dominates the wall clock. Going from 20.8 s to 13.6 s per long turn is the difference between "usable" and "waiting"; and it cost **one command-line flag** with no memory-space tradeoff at all — because the work was already there, merely routed to the slower processor.
