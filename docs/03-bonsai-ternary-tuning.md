# 03 — Ternary Dense Model (Bonsai 27B) on a 12 GB Laptop: What Tuning Can and Cannot Do

**Model: Ternary Bonsai 2 27B — a dense 27 B model with {−1, 0, +1} weights + FP16 group scaling (1.76 bits/weight, 7.2 GB GGUF), running on the [PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) (`prism-b10709-9a9394a`).**

This file documents the *negative* results. They are the useful half of the tuning story that [02](02-freetoken-moe-tuning.md) tells positively: **dense models have no reallocation to tune.**

## Baseline and the one real win

The **power unlock alone** (80 W → 175 W, see [01](01-power-unlock.md)) moved prefill from **757 → 1085–1143 tok/s (≈ +50 %)** and decode from 41.4 → 45.9 tok/s. That is where the structural gains end.

| Config | Prefill (36 K-class cold prompt) | Notes |
|---|---|---|
| PQ2_0 weight format, 175 W | **1125 tok/s** (25.7 s @ 28952 tokens) | baseline for this file |
| Same, `-ub 512` | **1143 tok/s** (23.6 s @ 26951 tokens) | **±2 % — noise** |
| Same, `-ub 2048` | **crash** | see below |

## Every micro-batch value beyond that is noise or fatal

### `-ub 512` vs `-ub 1024`: statistically identical

1143 vs 1125 tok/s (the prompts differ slightly in length, which alone accounts for the gap). Both are on the plateau.

### `-ub 2048`: a hard abort

```
... common_fit_params: failed to fit params to free device memory: n_gpu_layers ...
... llama threadpool init, n_threads = 16
/home/runner/work/llama.cpp/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:107: CUDA error
Aborted
```

**Why**: a larger µ-batch needs proportionally larger *temporary* compute buffers. With a 7.2 GB model + 160 K-token KV cache already filling 11.4 of 12 GB, there is no room for them. **A crash here is harmless** — kill the process and restart; nothing is corrupted.

### Small batch may look "faster" for a subtle reason

If you compare runs at different prompt lengths, remember the throughput metric already divides by tokens — a shorter prompt finishing sooner is not a speedup. **Match prompt token counts (check `prompt_n` in the timings) or the comparison is meaningless.**

## Why dense has nothing to tune (the structural reason)

Pictured as a pipeline:

```
MoE hybrid (35B-A3B)                    Dense (27B ternary)
─────────────────────                   ───────────────────
attention:      GPU  ← tunable          attention:   GPU
dense parts:    GPU  ← tunable          MLP/FFN:     GPU   ← no alternative exists
expert MLPs:    CPU or PCIe → GPU       (all of it)
                ↑
        --moe-hybrid-max-fetch  ← 35 % win
        --moe-cpu-threads
        --moe-cache-size
```

A dense model computes *everything* on the GPU all the time. There is no CPU/GPU division to re-balance, no fetch-vs-compute decision, no cache-miss policy. What remains — µ-batch sizing, kernel scheduling — is already handled well by llama.cpp's defaults, and the power unlock means the GPU is no longer starved.

**Practical corollary**: when someone reports a large tuning win on a laptop, ask *what got reallocated*. If the answer is "nothing", the win probably came from a configuration fix (like the power limit), not from parameter tuning — and it will not repeat.

## Idle vs load behaviour (a free sanity check)

```
idle:  210 MHz,  ~7 W      ← GPU powers down on its own
load:  2400 MHz, 145–148 W ← boosts to the max on demand
```

Because the hardware already scales to demand, the tempting "lock the clocks high" trick is a net loss on a laptop:

```bash
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc 2400,3105     # works — clock floor 2400
# but: idle draw jumped 7 W → 35 W for zero throughput gain
sudo nvidia-smi -rgc               # we reverted it
```

## VRAM ledger (why memory-side tricks are out)

| Item | Size |
|---|---|
| Model weights (PQ2_0) | 7.2 GB |
| KV cache (160 K tokens, Q4_0) | ~2.5 GB |
| Vision projector (Q8_0 mmproj) | 0.6 GB |
| Runtime buffers | ~1 GB |
| **Used / total** | **11.4 / 12.3 GB** |

With ~800 MB headroom there is no room to trade memory for speed (no larger batch buffers, no bigger caches, no higher-precision KV). This is why the µ-batch ceiling is physical, not a misconfiguration.

## Suggested configuration (dense, 12 GB card, long context)

```bash
./llama-server \
  -m Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --mmproj Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf --no-mmproj-offload \
  -ngl 99 -fa on -ub 512 -c 163840 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --jinja --host 0.0.0.0 --port 8080
```

Notes:
- `-ub 512` (equal to 1024, smaller buffers — pick either)
- `--no-mmproj-offload` keeps the vision tower in RAM, freeing ~0.6 GB of VRAM
- `-fa on` (flash attention) and CUDA graph reuse are on by default in this fork; don't fight them
- Keep the OS headless: the desktop compositor steals ~2 GB of VRAM on a 12 GB card

## The one-line summary

**For dense models on a small laptop GPU: fix the power limit, keep the defaults, and spend your tuning budget elsewhere.**
