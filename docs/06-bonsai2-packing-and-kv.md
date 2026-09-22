# 06 — Bonsai 2 27B on a 12 GB Laptop: Which Ternary Packing to Run, and How Far the KV Cache Stretches

**Model: [Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) (dense, {−1, 0, +1} weights + FP16 group scaling; a ternary quantization of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)). Engine: [PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) (a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp)) `prism-b10743` (build 10743 / commit `adfffbe`). The same ternary weights ship in two GGUF packings: `PTQ1_0` (dense trits, 1.75 bits/weight, 5.95 GB) and `PQ2_0` (2-bit slots, 2.13 bits/weight, 7.21 GB).**

This is the capacity story of the same laptop. The two packings are *containers for identical weights*, yet the choice changes decode speed by ~9 % on this Ada-generation GPU — and, more importantly, how much KV cache fits in 12 GB. Measured cold: the KV cache **boots all the way to the model's 256K maximum**, prompts up to **200K tokens were processed end-to-end** on the 224K configuration, and the deployment has since moved to **256K (the model maximum)**, verified by filling **85 % of that context** — see the section below.

## The two packings are the same weights — the trade-off is hardware-generation dependent

Both files decode to the same ternary values (packing is a container choice, not a different model). Per the upstream notes: `PTQ1_0` packs trits densely and is the faster decode on Ada-class and small accelerators — unpacking costs arithmetic that only pays off where batch-1 decode is not bandwidth-starved; `PQ2_0` is faster at prompt processing everywhere and the faster decode on H100-class parts.

Measured on this machine (RTX 4080 Laptop, Ada; `prism-b10743`; `-np 1 --cache-ram 24576`):

| Metric | PQ2_0 | PTQ1_0 |
|---|---:|---:|
| Decode (900-token outputs ×3, median) | 47.0 tok/s | **51.3 tok/s (+9.1 %)** |
| Prefill, 42K-token cold random prompt | **992 tok/s** | 975 tok/s (−1.7 %) |
| 136K-token prompt | ✗ HTTP 500 above ~130K | **✓ 536 tok/s (255.6 s)** |
| Resident weights | 7.21 GB | **5.95 GB** |

**Same-weights sanity check**: a fixed prompt, greedy decode (`temp=0`), fixed seed, run on both packings, produces outputs that agree **verbatim for the first ~40–60 tokens** and then diverge slightly in wording ("...sunlight contains all visible colors, and when it enters..." vs "...sunlight, which contains all visible colors, scatters as it enters..."). That is the expected signature of identical weights running through two different kernels — floating-point accumulation differences flipping near-tie tokens — not a quality gap. Both outputs are coherent.

## The KV-cache ceiling on 12 GB (cold-boot ladder)

All rows: q4_0 KV, flash attention, single slot, vision projector on CPU.

| `-c` (context) | PTQ1_0 VRAM | PQ2_0 VRAM |
|---|---:|---:|
| 192K (196608) | ✓ 10.3 GB | ✓ 11.45 GB |
| 224K (229376) | ✓ 11.0 GB — fallback (former default) | — |
| **256K (262144, model max)** | ✓ **11.8 GB — deployed** | — |

The KV cache that fits is only half the story — **the prompt-side compute buffers break first** (that is how the previous 160K config failed above ~130K tokens on `PQ2_0`):

| Prompt | Config | Result |
|---|---|---|
| 136K tokens | PTQ1_0 @160K | ✓ 536 tok/s |
| 180K tokens | PTQ1_0 @192K | ✓ 441 tok/s (409.6 s) |
| 200K tokens | PTQ1_0 @224K | ✓ 408 tok/s (492.2 s) |
| **222,729 tokens (85 % of 256K)** | **PTQ1_0 @256K** | **✓ 376 tok/s · prefill 592 s · `truncated = 0`** |
| >130K tokens | PQ2_0 @160K | ✗ HTTP 500 — chunk buffers do not fit once the KV cache eats the headroom |

## Filling 85 % of a 256K context, end to end

The 256K deployment was not accepted on "it boots" — it was verified by filling most of the context in one request (cold cache, synthetic filler, `/tokenize`-calibrated):

| Metric | Value |
|---|---:|
| Prompt tokens | **222,729** (= 85.0 % of 262,144) |
| **Prefill** | **376.1 tok/s** (592 s) |
| **Decode** | **16.14 tok/s** (256-token answer) |
| Total wall time | **608 s** |
| Engine-side result | `stop processing: n_tokens = 222,984, **truncated = 0**` ✅ |
| VRAM | 11,782 MiB — **83 MiB free**, identical before/after the run (KV is pre-allocated per `-c`, so a long prompt does not grow it) |
| Errors during the run | none (only the harmless boot-time `common_fit_params` note about `-ngl 99` being user-pinned) |
| Thermals | 85–87 °C → fans to maximum (~5800/5500 rpm), 175 W, GPU 99 % |

**Decode depth decay is the real cost, and it is steep for a dense model:**

| Depth | Decode |
|---|---:|
| short context | 51.3 tok/s |
| ~100K | ~25 tok/s |
| 222K (85 % of 256K) | **16.1 tok/s** |

That is the fundamental difference against the MoE + MTP engine on the same machine, whose decode decays far more gently (76 → 49.5 tok/s across 10× the prompt) — see [07](07-vram-budget-and-long-context.md) for the side-by-side and *why* (a dense 27B must read all of its weights and a proportionally larger KV per token; a 3B-active MoE does not).

**Deployment choice (updated): 256K.** The model's maximum context, verified with 85 % of it filled end-to-end. The trade is headroom: **83 MiB free** versus ~1.2 GB at 224K. Since the three engines on this laptop are strictly mutually exclusive (VRAM arbitration), the knife-edge is manageable — but treat the machine as *full*: any other GPU consumer, or a settings change that adds an allocation, will fail to start rather than degrade. Roll back to `-c 229376` (224K, ~1.2 GB headroom) if very large single reads stop being routine.

## Deployed configuration

```bash
./llama-server \
  -m Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --mmproj Ternary-Bonsai-2-27B-mmproj-BF16.gguf --no-mmproj-offload \
  --host 0.0.0.0 --port 8080 --alias bonsai-2-27b \
  -ngl 99 -fa on -ub 512 -c 262144 \
  -np 1 --cache-ram 24576 --cache-reuse 256 \
  --reasoning-budget 10000 \
  --reasoning-budget-message "Okay, I have thought long enough. Time to give the final answer now." \
  --temp 1.0 --top-p 0.95 --top-k 20 --jinja \
  --cache-type-k q4_0 --cache-type-v q4_0
```

| Flag group | Why |
|---|---|
| `-np 1 --cache-ram 24576` | Upstream known-issues: the default multi-slot cache split makes single conversations re-process their prompts; one slot plus a large RAM prompt cache fixes exactly that |
| `--cache-reuse 256` | KV-shift reuse window — the same continuity setting as doc [05](05-llamacpp-mtp-35b.md) |
| `--reasoning-budget 10000` + `--reasoning-budget-message "..."` | This model reasons at length; the budget caps the *thinking block* and injects a soft-landing sentence before the forced `</think>` |
| `-c 262144` | 256K context (model max), boot-verified at 11.8 GB with **83 MiB** to spare — and verified end-to-end at 85 % fill |

## Reasoning-budget behaviour (the fine print)

- A **per-request top-level** `thinking_budget_tokens` works; the same value inside `chat_template_kwargs` is silently ignored (upstream-documented). The budget caps the thinking block **only** — the model can still write long answers (an open-ended prompt hit the `max_tokens` ceiling with or without the budget; the thinking section shrank 23132 → 3841 chars at a 1500 budget).
- With `--reasoning-budget-message`, the injected sentence appears **verbatim at the end of the thinking block** when the budget expires — an observable, clean soft close instead of a silent truncation.

## Caveats

- The large-prompt runs used synthetic cold-cache prompts (tokenized filler): they measure how much *prompt* the machine can chew, not answer quality — real chat templates and vision preprocessing consume part of the margin.
- Boot ≠ usable for the largest prompts — the prompt column above exists because that is the real limit.
- At 256K the machine is effectively full (~83 MiB). Expect hard failures from any additional GPU consumer, not graceful degradation.
- Absolute numbers are this machine's; the ratios and the two-mechanism model (KV size vs compute buffers) transfer.
- Methodology: [04](04-methodology.md).

Machine-readable rows: [../data/measurements.csv](../data/measurements.csv), filter `bonsai_*`.
