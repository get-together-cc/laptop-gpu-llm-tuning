# 05 — llama.cpp + MTP Speculative Decoding: 2× Decode for a 35B MoE on a 12 GB Laptop

**Engine: [PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) (a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp)) `prism-b10743` (`llama-server`, build 10743 / commit `adfffbe`). Model: [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) UD-Q4_K_XL MTP build ([unsloth](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)) + BF16 mmproj (vision).**

The third production deployment on the same laptop, and the decode-speed one. [FreeToken](02-freetoken-moe-tuning.md) is the prefill/long-context engine; this llama.cpp path trades some of that away for **speculative decoding with the model's own MTP (multi-token prediction) head**: short generations reach **72–76 tok/s — about 2× the hybrid engine's ~34 tok/s** — while a **160K-token q4-quantized KV cache still fits in 12 GB** by keeping part of the MoE experts on the CPU. Unlike the dense 27B ([06](06-bonsai2-packing-and-kv.md)), this engine *does* have a knob left: speculative drafting.

## Deployed configuration

```bash
./llama-server \
  -m Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --mmproj mmproj-BF16.gguf --no-mmproj-offload --image-min-tokens 1024 \
  --alias qwen35b-mtp --host 0.0.0.0 --port 8096 \
  -c 163840 -fa on -np 1 -ngl 99 --n-cpu-moe 28 \
  -ctk q4_0 -ctv q4_0 \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --cache-reuse 256 --temp 0.7 --top-p 0.95 --top-k 20 --min-p 0
```

| Flag group | Why |
|---|---|
| `--n-cpu-moe 28` | Part of the MoE expert weights stay on the CPU side (see the split sweep below) — this is what leaves VRAM headroom for everything else. **28 is the hard edge on 12 GB: 24 does not boot** (see *Limits*) |
| `-ctk q4_0 -ctv q4_0` | q4 KV quantization: the 160K-token context coexists with the model in ~10.9/12 GB |
| `--spec-type draft-mtp --spec-draft-n-max 3` | Use the model's **own MTP head as the speculative draft** — no separate draft model; up to 3 drafted tokens per step (2 → 3 measured **+5–10 % decode at every context size**, +24 MiB VRAM) |
| `--cache-reuse 256` | Radix-cache reuse window (chat continuity without full re-prefill) |
| `--no-mmproj-offload --image-min-tokens 1024` | Vision projector (BF16, 902 MB) runs on the **CPU** — the GPU has no spare VRAM; ≥1024 tokens per image |

## Decode (256-token outputs, `temp=0`, `cache_prompt=false` — same protocol for all rows)

| Load | tok/s | drafted / accepted | acceptance |
|---|---:|---:|---:|
| short chat (arithmetic) | **72.37** | 134 / 112 | 83.6 % |
| long copy (product text) | 66.55 | 212 / 148 | 69.8 % |
| list (10 items) | 64.94 | 118 / 74 | 62.7 % |
| single-shot check (long prompt) | 67.18 | 212 / 148 | 69.8 % |

`draft_n` / `draft_n_accepted` are read from the server's own `timings` block — decode speed follows the acceptance rate, which depends on the task (83.6 % → 62.7 % across these loads).

## Large-context decode: does MTP survive a full KV cache? (16K → 158K)

Every row: synthetic random-token filler (fixed seed, so every config sees the *same* prompt at the *same* length), length calibrated through the server's `/tokenize`, `cache_prompt=false` (**full re-prefill, no cache reuse**), `temp=0`, 256-token output. `--spec-draft-n-max 3` unless noted.

| Prompt | prefill (tok/s) | **decode (tok/s)** | MTP acceptance | wall |
|---|---:|---:|---:|---:|
| 16,025 | 693.3 | **76.08** | 95.5 % | 26.7 s |
| 48,002 | 656.2 | **70.29** | 93.5 % | 81.1 s |
| 95,988 | 609.9 | **60.15** | 90.3 % | 162.3 s |
| 144,084 | 577.3 | **55.52** | 88.1 % | 260.7 s |
| **158,091** | **577.2** | **49.52** * | 97.7 % * | 311.8 s |

\* 158K row was measured with `--spec-draft-n-max 2` (the n=3 setting landed afterwards); applying the measured +5–10 % delta puts it around **52–55 tok/s**.

**The answer is yes.** Decode falls **−30 % from 16K to 144K** (76.1 → 55.5) and then flattens — the curve tops out near **50 tok/s** rather than collapsing, because only ~11 of the model's 41 layers carry KV state, and q4_0 storage keeps the attention reads cheap. Prefill decays far less (−17 %).

**The 160K context is real, not a parameter-sheet claim**: a **158,091-token prompt ran to completion with `truncated = 0`** (context occupied 158,346 tokens including the generation) — i.e. 160K is usable end-to-end on a 12 GB laptop.

> **Read the acceptance column with care.** These synthetic-filler runs accepted 88–97 % of drafts, well above the 62–84 % seen on real prose (previous table). Filler output is more predictable, so the *absolute* decode numbers here are optimistic relative to production traffic. The **shape** of the curve, and any A/B inside the same table, are unaffected — same protocol throughout.

## A/B: `--spec-draft-n-max 2 → 3`

| Prompt | n = 2 | n = 3 | Δ | acceptance (2 → 3) |
|---|---:|---:|---:|---:|
| 16K | 72.33 | **76.08** | **+5.2 %** | 95.4 % → 95.5 % |
| 48K | 64.61 | **70.29** | **+8.8 %** | 96.6 % → 93.5 % |
| 96K | 56.27 | **60.15** | **+6.9 %** | 96.6 % → 90.3 % |
| 144K | 50.39 | **55.52** | **+10.2 %** | 94.3 % → 88.1 % |

More drafted tokens *lower* the acceptance rate (as expected) but the net effect is a win at every size — and it costs **+24 MiB** of VRAM (10,868 → 10,892 MiB), i.e. nothing. Prefill is unchanged within noise. Rule of thumb: **when the MTP acceptance rate is high, draft more; verify with a curve, not a single point.**

## Limits: where the 12 GB actually runs out

| Knob | Setting | Result |
|---|---|---|
| MoE split | `--n-cpu-moe 28` | ✅ boots, ~10.9 GB, decode per the curve above |
| MoE split | `--n-cpu-moe 24` (4 more layers on GPU) | ❌ **`failed to create context` → `exiting due to model loading error`** — the KV cache and the compute buffers are already at the edge; the ~300 MiB this knob asks for does not exist |
| Context | `-c 163840` (160K) | ✅ 158K-token prompt end-to-end, `truncated = 0` |
| KV precision | `q4_0` | ✅ deployed; `q8_0` would need ~1.6 GB more — not available at 160K |

**Take-away: at 160K context with a 35B MoE on 12 GB, `--n-cpu-moe 28` *is* the budget.** Beyond it there is ~24 MiB of slack, not gigabytes. If you need more *context*, you must move experts back to the CPU (`m32`/`m36`) — i.e. buy tokens with speed. See [07](07-vram-budget-and-long-context.md).

## Multimodal (vision on CPU)

- "What color is the square?" (test image: a red square) → correct **"Red" in 30.1 s**, including the model's thinking chain. The BF16 projector runs on the CPU because the GPU is full.
- If images matter more than raw decode speed, the FreeToken engine keeps the vision tower on the GPU and answers faster — this is the trade-off, not a defect.

## Streaming (SSE) — and one gateway pitfall

| Path | First chunk | Total |
|---|---:|---:|
| via an OpenAI-compatible gateway (fixed) | **0.27 s** | 4.6 s |
| the same gateway, before a buffering fix | 36.7 s | 39.8 s |
| direct to the engine (for reference) | 0.66 s | 7.2 s |

The >100× first-chunk difference was **not** the model: the gateway read the SSE stream in 8 KiB chunks and could not forward a partial line, so it sat on the first token until the buffer filled. If your first-token latency looks impossible, check the proxy's stream buffering **before** touching any inference flag.

## Where it sits vs the hybrid engine (same laptop)

| | llama.cpp + MTP (this doc) | FreeToken hybrid ([02](02-freetoken-moe-tuning.md)) |
|---|---|---|
| Decode, short context | **76.4 tok/s** | ~34 tok/s |
| Decode, ~100K context | **60.2 tok/s** | — |
| **158K-token prefill** | 577 tok/s (274 s) | **~1700–1900 tok/s** |
| Vision | CPU projector, ~30 s | GPU tower, faster |
| Long context | 160K q4 KV in 12 GB | experts paged over PCIe |

Roughly: **decode ≈ 2× faster here (and it stays ~2× at 100K+ context); prefill / long-context reads / vision stay FreeToken's wins.** Both engines fit on the same 12 GB laptop, so the workload picks the engine.

## Tuning history (same machine)

| Config | Decode |
|---|---:|
| MTP off (historical) | 44.6–45.8 tok/s |
| MTP on, earlier config | 55.15 (long) / 55.71 (short) |
| Consolidated `--n-cpu-moe 28`, `--spec-draft-n-max 2` | 66.55 / **72.37** (**+20 %**) |
| `--spec-draft-n-max 3` (today) | **70.29 (48K) / 76.08 (16K)**, +5–10 % everywhere |

## Caveats

- The 155K-token prefill on the earlier config was never re-measured; **the 158K row above is the re-measurement** (577 tok/s, 274 s, zero truncation).
- The MTP-off comparison is historical — not an A/B on the day's final configuration.
- Synthetic-filler prompts inflate MTP acceptance versus real text (see the warning above); use the *ratios*, not the absolute tok/s, for planning.
- All numbers follow the [methodology](04-methodology.md); absolute values are this machine's.

Machine-readable rows: [measurements.csv](../data/measurements.csv), filter `cpp_*`.
