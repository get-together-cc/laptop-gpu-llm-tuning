# 07 — One 12 GB Budget, Spent Three Ways

**A cross-engine note on the same machine: a laptop RTX 4080 (12 GB) running three deployments (a 35B MoE hybrid, a 27B dense ternary model, and the same 35B MoE under llama.cpp + MTP). Every wall described here was hit in production, not simulated.**

The interesting question on a small GPU is never "is it fast?" — it is **what am I buying with each megabyte?** This note collects the three currencies we spend VRAM on, what each one costs on this hardware, and the measurements that show where the ceiling really is.

## The three currencies

| Currency | Buys | Costs (measured, this machine) | Hard limit we hit |
|---|---|---|---|
| **MoE expert residency** | decode/prefill speed for a MoE model | ~74 MiB per layer moved from CPU to GPU | 35B MoE @160K: `--n-cpu-moe 28` boots, **24 does not** (`failed to create context`) |
| **KV cache size (context)** | how much prompt fits at once | q4_0 on this 35B (KV in ~11 of 41 layers): **≈13 MiB per 1K tokens** — 160K ≈ 2.1 GB, +32K ≈ +0.4 GB | 160K deployed; 192K needs ~0.4 GB *plus* headroom for compute buffers |
| **KV precision** | numerical stability on very long contexts, fewer re-reads | same 160K context at q8_0 ≈ +1.6 GB vs q4_0 | does not fit at 160K on 12 GB — you would have to shrink the context to pay for it |
| **Headroom (the non-negotiable)** | prompt-processing buffers; anything allocated at request time | the residual | ~1.2 GB at 224K context, **~0.08 GB at 256K** |

Two structural facts make this concrete:

- **The dense 27B and the MoE 35B spend their budget differently.** The dense model computes everything on the GPU — there is nothing to move, so it has no residency knob ([03](03-bonsai-ternary-tuning.md)); the MoE model has a CPU/GPU split that responds to tuning ([02](02-freetoken-moe-tuning.md), [05](05-llamacpp-mtp-35b.md)).
- **Boot ≠ usable.** Every one of these engines boots larger than it can serve. The measured prompt-size column in the tables below exists because that is where the failures show up (the previous 160K configuration served ≤~130K tokens and started returning HTTP 500 above it).

## The long-context decode curves (why this is a design choice, not a bug)

Same machine, same protocol (cold cache, `cache_prompt=false`, greedy), two different engines:

| Prompt depth | 27B dense ternary (Bonsai, [06](06-bonsai2-packing-and-kv.md)) | 35B MoE + MTP (llama.cpp, [05](05-llamacpp-mtp-35b.md)) |
|---|---:|---:|
| short / 16K | 51.3 (short) | **76.1** |
| ~100K | ~25 | **60.2** |
| ~144K | — | **55.5** |
| **85 % of 256K (222,729)** | **16.1** | — (160K config) |
| 158K | — | **49.5** |
| Prefill at depth | 376–441 tok/s (180–222K) | 577 tok/s (158K) |

Read it this way: **the MoE+MTP engine keeps ~2× the decode rate at any depth, and its decay is shallower** (76 → 49.5, −35 % across 10× the context) **than the dense ternary model's** (51 → 16, −69 %). That matches the mechanism: the 27B dense model must read its ternary weights *and* a proportionally larger KV per token, while the MoE engine activates ~3B parameters — its decode cost is dominated by draft verification, not weight traffic.

So "which engine?" is answered by the workload mix, not by a single number:

| Workload | Use | Why |
|---|---|---|
| Big review/read-then-answer jobs (50–160K prompt, long answer) | **35B MoE + MTP** | decode 55–70 tok/s at depth; 158K verified |
| Very long single reads (>160K) or vision | **FreeToken hybrid** (or the 27B at 256K) | prefill 3× faster; KV up to the model's 256K max |
| Short interactive chat | **35B MoE + MTP** | 76 tok/s |
| Batch/throughput with a full cache | FreeToken (prefill-bound workloads) | ~1700–1900 tok/s prefill |

## Where the wall actually is, in numbers

| Attempt | Result |
|---|---|
| 35B MoE, `--n-cpu-moe 28`, 160K ctx, q4 KV | ✅ 10.9 GB, 158K prompt end-to-end (`truncated = 0`) |
| 35B MoE, `--n-cpu-moe 24` | ❌ **OOM at context creation** — 4 layers (~300 MiB) do not fit |
| 27B ternary, `-c 262144` (model max) | ✅ 11.8 GB — **but only 83 MiB free**; 85 % of the context (222,729 tokens) ran end-to-end with zero truncation |
| 27B ternary, `-c 229376` (224K) | ✅ 11.0 GB, ~1.2 GB free — the safer default until large single reads are routine |

**Practical rule:** decide the *context ceiling you will actually serve* first, then give the model whatever residency that leaves, and never spend the last ~0.5 GB — prompt-processing buffers are allocated at request time and their size scales with the µ-batch, not with your optimism.

## The integrated GPU is not a free win

The same laptop has an AMD Radeon iGPU (Ryzen 7945HX3D's Raphael). Measured state:

| Fact | Value |
|---|---|
| iGPU VRAM (BIOS UMA carve-out) | **512 MiB** (idle use: ~31 MiB) |
| Addressable system RAM via GTT | **15.6 GB** |
| Role in practice | **display adapter only** — the internal panel is driven by it, `gpu_busy = 0 %`, no compute process attached |
| Usable by our engines? | **No.** Both engines are **CUDA-only builds** (no Vulkan/ROCm backend ships with them) |

Could it help? In principle a Vulkan build could run the MoE experts the CPU currently runs — but the iGPU has **no dedicated VRAM** (it reads DDR5 at the same ~83 GB/s the CPU uses), so it competes for exactly the bandwidth that is already the bottleneck, while adding a second engine build to maintain. **We did not pursue it, and would not recommend it as a first move.** The useful takeaway is the opposite one: because the *panel* hangs off the iGPU, **the discrete GPU is 100 % free for inference** — which is why the deployments above can spend 10.9 of 12 GB on model + KV without any display penalty.

## Caveats

- Absolute numbers are this machine's (RTX 4080 Laptop, Ada). The *mechanisms* — the three currencies, boot-vs-usable, MTP acceptance inflation on synthetic prompts — transfer to other small-VRAM setups.
- KV-per-1K-token is estimated from the boot ladder (192K/224K/256K measurements) plus the per-layer KV geometry, not from a dedicated micro-benchmark.
- Methodology: [04](04-methodology.md). Machine-readable rows: [../data/measurements.csv](../data/measurements.csv).
