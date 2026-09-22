# 08 — ik_llama.cpp: +31 % Prefill, and the "Loadable ≠ Usable" Context Trap (35B MoE, 12 GB)

**Engine: [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) (MIT) — build `5f89bfc` (`llama-server`). Model: [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) UD-Q4_K_XL MTP build ([unsloth](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)) + BF16 mmproj (vision).**

The fourth deployment on the same laptop, and the one that retired the previous 35B engine. [05](05-llamacpp-mtp-35b.md) won decode with the model's own MTP head on the PrismML llama.cpp fork; this doc is what happens when you run **the same weights, same KV settings, same MoE split** on [**ik_llama.cpp**](https://github.com/ikawrakow/ik_llama.cpp) — a llama.cpp fork focused on MoE performance. The result is a **+27–31 % prefill / −22–27 % wall-clock** engine swap with no hardware change — plus a trap that cost most of a day: **a context size that boots is not a context size you can use**.

## Deployed configuration

```bash
./llama-server \
  -m Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --mmproj mmproj-BF16.gguf --no-mmproj-offload --image-min-tokens 1024 \
  --host 0.0.0.0 --port 8098 \
  -c 163840 -fa on -np 1 -ngl 99 --n-cpu-moe 28 \
  -ctk q4_0 -ctv q4_0 \
  --spec-type mtp:n_max=3,p_min=0.0 -mtprot q8_0 \
  --prefetch-experts -ger \
  --jinja \
  --reasoning-budget 3000
```

| Flag | Why |
|---|---|
| `--n-cpu-moe 28` | Same budget point as [05](05-llamacpp-mtp-35b.md) — 28 layers of experts on the CPU is what fits next to a 160K q4 KV cache. **The scan below confirms it is also the only *usable* point at 160K** |
| `--spec-type mtp:n_max=3,p_min=0.0 -mtprot q8_0` | ik's speculative-decoding syntax differs from upstream (`mtp:` spec string instead of `--spec-draft-n-max`); `-mtprot q8_0` quantizes the MTP head's tensors. See the depth sweep below |
| `--prefetch-experts -ger` | ik-specific MoE flags: expert **prefetching** and **grouped expert routing** — this is where much of the prefill gain comes from |
| `--jinja` | **Required for any request carrying a `tools` array** — without it every tool-calling request fails with a 500. See the acceptance-test section |
| `--reasoning-budget 3000` | ik honors the **startup** budget but **ignores per-request `thinking_budget_tokens`** — a gateway that injects budgets per request must move them here |
| `--no-mmproj-offload --image-min-tokens 1024` | Vision projector on the **CPU** (no spare VRAM), same as [05](05-llamacpp-mtp-35b.md) |

Build note: ik needs `LD_LIBRARY_PATH` pointed at the CUDA 13 runtime (`libcublas.so.13` is not in `ldconfig` on this box); the build here bakes in an rpath, but a plain build will not.

## Three engines, one model, one GPU

Same protocol for every cell (fixed seed, `/tokenize`-calibrated prompt length, `cache_prompt=false` = full re-prefill, `temp=0`, 256-token output, KV `q4_0`, `--n-cpu-moe 28`, `-c 163840`). Cell = **prefill tok/s / decode tok/s / wall clock**.

| Prompt | PrismML fork ([05](05-llamacpp-mtp-35b.md), spec n=2) | **ik_llama.cpp (spec n=3)** | upstream llama.cpp (spec n=3) |
|---|---:|---:|---:|
| 16K | 693.3 / 72.33 / 26.7 s | **899.5 / 90.59 / 20.7 s** | 678.4 / 85.85 / 26.6 s |
| 48K | 656.2 / 64.61 / 81.1 s | **861.1 / 78.62 / 59.4 s** | 648.5 / 74.15 / 77.7 s |
| 96K | 609.9 / 56.27 / 162.3 s | **788.3 / 62.14 / 126.8 s** | 597.8 / 65.08 / 164.9 s |
| 144K | 577.3 / 50.39 / 260.7 s | **731.6 / 53.57 / 203.0 s** | 562.9 / 59.85 / 260.9 s |

- **Prefill: ik wins by +27–31 % at every depth** vs the PrismML fork, and **+30–33 %** vs an upstream build on the same box. Prefill does not use speculative decoding, so this is the engine itself (fused MoE ops, grouped expert routing, expert prefetch).
- **Wall clock follows prefill: −22–27 %** on every request.
- **Decode: read with care — equalize the spec depth first.** With the draft depth equalized (all engines at n=3) at 48K: **ik 78.62 / PrismML re-run 75.04 / upstream 74.15 tok/s** (ik +5 % / +6 %). At 96K/144K the *upstream* build's raw decode is ~5–10 % faster than ik's in this run — the two forks trade places with depth, so don't extrapolate a single point. (An earlier "ik is slower at decode" conclusion of ours was **this exact mistake**: ik was running its most conservative draft setting while the other engine ran an aggressive one. See below.)

## The MTP depth sweep — and the mistake that produced it

48K prompt, same engine, only the spec settings change:

| Setting | prefill | decode | acceptance |
|---|---:|---:|---:|
| `n_max=1` (most conservative) | 901.0 | 57.09 | 99.2 % |
| `n_max=2` | 871.1 | 65.27 | 94.9 % |
| `n_max=3` | 874.1 | 77.32 | 94.5 % |
| `n_max=3` + `--spec-autotune` | 874.5 | 77.99 | 94.5 % |
| **`n_max=3` + `-mtprot q8_0`** ⭐ | 865.5 | **78.38** | 94.5 % |
| `n_max=3`, all heads | 870.6 | 77.68 | 94.5 % |

**`n_max` 1→3 is +37 % decode** — the depth knob dominates everything else (`mtprot q8_0` adds +1.4 %, autotune ≈ n=3, all-heads ≈ n=3). Rule: **before comparing two engines on decode, make sure both are at the same speculative depth** — otherwise you measure your own flag choices, not the engines.

## Context ladder: 256K boots, 160K works

The model's native maximum is 256K and the KV cache is q4_0 — so how much context fits? The honest answer required testing the **prompt**, not the boot:

| Context | Boots? | Long prompt (on a loaded context)? | Verdict |
|---|---|---|---|
| 256K (`-c 262144`) | ✅ (11,276 MiB) | ❌ 235K prompt → flash-attention OOM; `-ub 256` variant fails to load at all | **boots, unusable** |
| 240K (`-c 245760`) | ✅ (11,292 MiB, 990 free) | ❌ 156K prompt → OOM | **boots, unusable** |
| **160K (`-c 163840`)** | ✅ (10,754 MiB, **1,528 MiB spare**) | ✅ **158,091-token prompt (96.5 % of the window) completed: 738.4 tok/s prefill, `truncated = 0`, 0 OOM** | **deployed** |

**Why the larger windows fail is the interesting part.** KV cache is pre-allocated from the context size; flash-attention **work-space** is allocated at request time, proportional to the actual token count. A 256K cache eats the VRAM that a 235K prompt would need to *process* — so the bigger the declared context, the *smaller* the biggest prompt that can actually be run. **Loadable ≠ usable: plan from the worst-case prompt, not from the boot log.** The 160K window passes 96.5 %-full prompts and still leaves 1.5 GB free.

## The MoE split scan: at 160K, `m28` is the only usable point

More experts on the GPU should decode faster — in an earlier sweep (`m24` vs `m28` on a smaller context) it did, by ~7 %. At 160K the budget says no:

| `--n-cpu-moe` | VRAM at load | 158K prompt | decode |
|---|---:|---|---|
| **28 ⭐ deployed** | 10,754 MiB | ✅ | 53.2 tok/s |
| 26 | 11,678 MiB (+924) | ❌ prefill OOM | — |
| 24 | — | ❌ KV cache cannot even be allocated | — |

The extra ~0.9 GB the smaller split wants does not exist next to the 160K KV cache. **"Fewer CPU experts" is not a spare knob — it trades against context, and at this size the trade is closed.**

## Parameter-surface differences (checklist when switching forks)

| Capability | PrismML fork | **ik_llama.cpp** | upstream |
|---|---|---|---|
| Speculative switch | `--spec-type draft-mtp --spec-draft-n-max N` | `--spec-type mtp:n_max=N,p_min=…` (+ `-mtprot` to quantize the head) | same as PrismML |
| `--cache-reuse` | ✅ | ❌ **not available** | ✅ |
| MoE extras | `--n-cpu-moe` | `--n-cpu-moe` + `--prefetch-experts` + `-ger` + `-ser` + `--defer-experts` | `--n-cpu-moe` |
| Ternary GGUF (docs [03](03-bonsai-ternary-tuning.md)/[06](06-bonsai2-packing-and-kv.md)) | ✅ GPU kernels | loads, but **CPU-only kernels** (~80× slower; the log says `Prism ternary Hadamard rotations are CPU-only`) | not supported |
| Request-level thinking budget | `--reasoning-budget` / message | `--reasoning-budget` (startup only; **per-request budget is ignored**) | `--reasoning-budget` |
| `/slots` fields | standard (`is_processing`) | private (`state`, `id_task`; **`state` can read 0 while busy** — poll `id_task` instead) | standard |
| Runtime deps | prebuilt package (CUDA 12 runtime) | CUDA 13 build needs `LD_LIBRARY_PATH` (or an rpath) | needs `LD_LIBRARY_PATH` on this box |

## The acceptance test a bare curl misses: `--jinja`

ik does not enable its Jinja chat-template path by default. Bare `/v1/chat/completions` calls — plain prompts, streaming or not — **all passed**. Then every request carrying a **`tools` array** (any agent-style client) failed with:

```
500 - {'error': {'code': 500, 'message': 'tools param requires --jinja flag', ...}}
```

**Lesson for any engine swap: put a tool-calling request in the acceptance checklist.** A completion smoke test cannot see the template layer; the first real client will. With `--jinja` set, both streaming and non-streaming tool-carrying requests pass (and the response carries the separated `reasoning_content`).

## Gotchas (engine bring-up)

- **Startup allocates ~11.5 GB of pinned host memory** (`Allocating 11.54 GiB of pinned host memory`) — boots take tens of seconds to a minute; give the health wait enough time (`GGML_CUDA_NO_PINNED=1` disables it).
- **Wait for VRAM to drain before starting the next engine instance.** Starting too early gives `CUDA error: out of memory` at `cudaMemGetInfo` and reads like a model incompatibility. Poll until used VRAM < ~300 MiB.
- **A `pkill -f` pattern that matches your own command line kills your own SSH session.** Run the kill from a script file, or resolve the PID first and exclude yourself.
- Two instances on the same port → `couldn't bind` (TIME_WAIT; the server does not reuse the port). Use different ports for A/B runs.
- Don't classify failures from log keywords (things like `rope_finetuned = unknown` are harmless) — check whether the process actually died.
- CUDA 13 packages are split: headers in `libcublas-dev-13-0`, runtime in `libcublas-13-0` — "it compiles but won't run" is usually the missing runtime half.

## Take-aways

1. **The fork is worth real money on MoE prefill**: same model, same flags where possible — **+27–31 % prefill, −22–27 % wall** just by switching to ik on this GPU.
2. **Loadable ≠ usable.** A context size that boots can be *hindered* by its own KV cache: the workspace for processing long prompts competes with the pre-allocated cache. Test the prompt, not the boot.
3. **Equalize speculative depth before comparing engines** — our own first conclusion ("ik decodes slower") was an artifact of running it at `n_max=1`.
4. **A completion smoke test cannot catch a missing `--jinja`** — tool-calling requests are the agent-era acceptance test.
