# Laptop GPU LLM Inference Tuning

**Real-world tuning notes for running large language models on a laptop RTX 4080 (12 GB) under Linux** — measured data, exact commands, and the pitfalls we hit.

This is not a benchmark paper. It is a field report: four production deployments tuned on the same laptop, every number measured with the same methodology, every failed attempt documented.

## What you get

| Achievement | Before | After | Gain |
|---|---|---|---|
| **GPU power limit (Linux)** | 80 W (locked) | **175 W** (Dynamic Boost fully unlocked) | **+119 %** power budget |
| **MoE hybrid prefill** (35B-A3B) | ~2000 tok/s | **~3090 tok/s** | **+54 %** |
| **MoE hybrid decode** | 42–58 tok/s | **60 tok/s** | best-ever |
| **MTP speculative decode** (35B-A3B) | 44.6–45.8 tok/s (MTP off) | **76.4 tok/s** (`--spec-draft-n-max 3`) | **+67 %** |
| **MoE prefill, second fork** (35B-A3B) | 693 tok/s (PrismML fork) | **900 tok/s** (ik_llama.cpp) | **+27–31 %** across 16K→144K |
| **Long-context decode hold-up** (35B MoE) | ~34 tok/s (hybrid engine, any depth) | **60.2 @96K · 49.5 @158K** | **≈2× at 100K+ depth** |
| **Ternary dense prefill** (27B) | 757 tok/s | **1143 tok/s** | +51 % (power unlock) |
| **Ternary packing** (27B decode, Ada) | 47.0 tok/s (PQ2_0) | **51.3 tok/s** (PTQ1_0) | **+9 %** |
| **Ternary KV ceiling** (27B on 12 GB) | ~128K usable prompt (PQ2_0 @160K) | **222.7K prompt tested (85 % of max) · 256K deployed** | **1.7× context** |

All measured on the same machine, same methodology (see [methodology](docs/04-methodology.md)).

## Hardware

| Component | Detail |
|---|---|
| Laptop | MSI Raider A18 HX A7VHG (MS-182K) |
| GPU | NVIDIA GeForce RTX 4080 Laptop GPU, 12 GB GDDR6 (`10de:27a0`, subsystem `1462:1440`) |
| CPU | AMD Ryzen 9 7945HX3D, 16 cores / 32 threads (integrated Radeon — display only, see finding 7) |
| RAM | 32 GB DDR5-5200 (dual channel, ~83 GB/s) |
| Boot disk | USB SSD (114 GB) running the OS; internal NVMe (954 GB) hosts Windows |
| PSU | 330 W (20 V / 16.5 A), 95 Wh battery |
| BIOS | AMI E182KAMS.10C (2025-06) |
| **GPU PCIe link** | **Gen4 x8** (capable of x16; the rest of the lanes are shared with M.2/USB4 by the vendor) |

## Software

| Layer | Version |
|---|---|
| OS | Ubuntu 24.04.4 LTS (Noble) |
| Kernel | **7.0.0-31-generic** (HWE stack) |
| NVIDIA driver | **580.178.04** (DKMS) |
| EC control | [`msi-ec`](https://github.com/BeardOverflow/msi-ec) DKMS module (EC firmware `182KIMS1.113`; kernel 7.0 also ships a mainline `msi-ec`) |
| Dynamic Boost | `nvidia-powerd` 2.0 |
| Model A | [**FreeToken**](https://github.com/FlashML-org/FreeToken) engine ([Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) NVFP4, MoE hybrid — experts on CPU) |
| Model B | [**PrismML** llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) `prism-b10743` ([Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf), dense ternary {−1,0,+1}; **PTQ1_0**, q4 KV, **256K context — model max, 85 %-fill verified**) |
| Model C | [**PrismML** llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) `prism-b10743` ([Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) UD-Q4_K_XL + MTP speculative decoding `--spec-draft-n-max 3`, MoE CPU offload, 160K q4 KV) |
| Model D | [**ik_llama.cpp**](https://github.com/ikawrakow/ik_llama.cpp) `5f89bfc` ([Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) UD-Q4_K_XL + MTP speculative decoding `--spec-type mtp:n_max=3` + `-mtprot q8_0`, `--prefetch-experts -ger`, 160K q4 KV — **now the production 35B engine, +27–31 % prefill over Model C**) |

## The big findings

### 1. Linux locks your laptop GPU to base TGP — two hidden causes

Out of the box the GPU power limit reported **80 W** while the same hardware ran at **175 W under Windows**.

**Cause A**: `nvidia-powerd` was never running. On Ubuntu the systemd unit and the D-Bus policy file are shipped *inside the driver's doc directory* and are not installed automatically.

**Cause B**: The MSI EC "shift mode" (performance profile) has no driver on Linux — FN hotkeys reach the kernel as `Unknown key` and the EC never raises the GPU power budget.

Fixing both takes about ten minutes and immediately unlocks **150 W, then 175 W** under load. See [docs/01-power-unlock.md](docs/01-power-unlock.md).

### 2. MoE hybrid models have a huge, non-obvious tuning knob

For the FreeToken hybrid engine (`--moe-backend hybrid`), experts that miss the GPU cache are either **fetched over PCIe** or **computed on the CPU**. The default `--moe-hybrid-max-fetch -1` (auto) benchmarks the PCIe/CPU ratio once and stays conservative.

**After unlocking the GPU to 175 W, the GPU side got ~2× faster — so the auto heuristic is stale.** Forcing `--moe-hybrid-max-fetch 128` cut prefill time **from 20.8 s to 13.6 s (−35 %)**, i.e. **~2000 → ~3090 tok/s**.

Going higher (256) makes it *slower* — **PCIe Gen4 x8 (~16 GB/s) is the next wall**. See [docs/02-freetoken-moe-tuning.md](docs/02-freetoken-moe-tuning.md).

### 3. Dense models have nothing left to tune (and that's the lesson)

The same power unlock gave the dense ternary 27B a **+51 %** prefill gain — but every further knob (µ-batch 512/1024/2048) is noise-level or fatal:

- `-ub 512` vs `-ub 1024`: 1143 vs 1125 tok/s (±2 %, noise)
- `-ub 2048`: **CUDA abort** (`ggml-cuda.cu:107`) — the larger µ-batch needs temporary buffers that do not fit in the already-full 11.4/12 GB VRAM

**Tunable space comes from having something to re-allocate.** Dense models compute everything on the GPU — there is nothing to move. MoE hybrids have the CPU/GPU split — and that split responds to tuning. See [docs/03-bonsai-ternary-tuning.md](docs/03-bonsai-ternary-tuning.md).

### 4. The same class of model can go ~2× faster in decode on a second engine — and it holds up at depth

The llama.cpp (PrismML fork) deployment uses the model's **MTP head as its own speculative draft**. Short generations hit **76.4 tok/s vs ~34 tok/s** on the hybrid engine, and — the part that is easy to assume away — **the advantage survives long contexts**: measured with synthetic filler prompts and a cold cache (`cache_prompt=false`), decode goes **76.1 (16K) → 70.3 (48K) → 60.2 (96K) → 55.5 (144K) tok/s** (−30 % over 10× the prompt), and a **158,091-token prompt completed with zero truncation** (160K q4 KV inside 12 GB, experts partially on CPU). Bumping the draft depth (`--spec-draft-n-max 2 → 3`) added **+5–10 % at every size for +24 MiB** of VRAM. See [docs/05-llamacpp-mtp-35b.md](docs/05-llamacpp-mtp-35b.md).

### 5. The same ternary weights, two packings — on Ada, the smaller one is also the faster one

Bonsai 2 27B ships as two GGUF packings of identical ternary weights: `PTQ1_0` (dense trits, 5.95 GB) and `PQ2_0` (2-bit slots, 7.21 GB). On this Ada laptop the smaller packing is also the **faster decode (51.3 vs 47.0 tok/s, +9 %)** at a −1.7 % prefill cost — and the 1.3 GB it saves raises the KV ceiling: the cache boots up to the model's **256K maximum** (11.8 GB), **200K-token prompts run end-to-end**, and the deployment now uses the **full 256K**, verified by filling **222,729 tokens (85 % of the maximum) in one request — prefill 376 tok/s, zero truncation, with only 83 MiB of VRAM to spare**. The cost is decode depth-decay: 51.3 tok/s short-context → **16.1 tok/s at 222K**. A fixed-seed greedy cross-check confirms the packings are the same weights (identical output prefix, then kernel-level wording divergence). See [docs/06-bonsai2-packing-and-kv.md](docs/06-bonsai2-packing-and-kv.md).

### 6. One 12 GB budget, three ways to spend it — and a hard edge you can measure

On a small GPU the question is never "which flag is fastest" but **what each megabyte buys**. Three currencies, all measured here:

- **MoE expert residency**: ~74 MiB per layer moved from CPU to GPU. At 160K context, `--n-cpu-moe 28` boots; **`--n-cpu-moe 24` fails with `failed to create context`** — the ~300 MiB it wants do not exist. That is the hard edge, not a tuning preference.
- **KV size (context)**: ≈13 MiB per 1K tokens (q4_0, 35B; only ~11 of 41 layers carry KV). +32K of context costs ~0.4 GB.
- **Headroom**: never spend the last ~0.5 GB — prompt-processing buffers are allocated at request time. At 256K context the 27B leaves **83 MiB**; at 224K it leaves ~1.2 GB.

Cross-engine curves, the boot-vs-usable distinction, and the workload-to-engine table: [docs/07-vram-budget-and-long-context.md](docs/07-vram-budget-and-long-context.md).

### 7. The integrated GPU is a display adapter, not a spare compute device

The laptop's AMD Radeon iGPU (Ryzen 7945HX3D) has **512 MiB of BIOS-set UMA VRAM** and can address **15.6 GB of system RAM via GTT** — and it is idle: `gpu_busy = 0 %`, no compute process attached, because **our engines are CUDA-only builds** (no Vulkan/ROCm backend). Even with a Vulkan build, the iGPU shares the same DDR5 bandwidth the CPU already uses for MoE experts, so it competes rather than adds. The useful consequence runs the other way: **the internal panel is driven by the iGPU, so the discrete GPU is 100 % free for inference** — which is why these deployments can spend 10.9 of 12 GB on model + KV with no display penalty. See [docs/07](07-vram-budget-and-long-context.md).

### 8. The engine fork is worth +30 % prefill — and a context size that boots is not one you can use

Same model, same KV quantization, same MoE split, same GPU: switching the 35B from the PrismML fork to [**ik_llama.cpp**](https://github.com/ikawrakow/ik_llama.cpp) raised prefill **+27–31 % at every depth (16K→144K)** and cut wall clock **−22–27 %** — weeks of flag-tuning had bought less than changing the fork. The same exercise produced the most expensive lesson of the day: **ik boots happily at `-c 245760` (240K) — and then OOMs on a 156K prompt**, because the KV cache pre-allocated for a 240K window competes with the flash-attention workspace the prompt itself needs. The deployed point is **160K**, verified by running a **158,091-token prompt (96.5 % full) to completion with zero truncation** — and the MoE split scan says `--n-cpu-moe 28` is the *only* usable split there. One more trap for agent-era clients: **a bare completion smoke test passes while every request carrying a `tools` array 500s** unless the server runs with `--jinja`. See [docs/08](docs/08-ik-llamacpp-35b.md).

## Repository layout

```
docs/01-power-unlock.md          # 80 W → 175 W: complete steps + verification
docs/02-freetoken-moe-tuning.md  # MoE hybrid prefill -35 %: full experiment matrix
docs/03-bonsai-ternary-tuning.md # Ternary dense model: boundaries and dead ends
docs/04-methodology.md           # How we measure (and why naive A/B lies)
docs/05-llamacpp-mtp-35b.md      # llama.cpp + MTP: 2× decode, and the 16K→158K decode curve
docs/06-bonsai2-packing-and-kv.md# Ternary packing choice + KV ceiling: 256K filled to 85 %
docs/07-vram-budget-and-long-context.md # One 12 GB budget, three currencies + the iGPU question
docs/08-ik-llamacpp-35b.md        # ik_llama.cpp: +30 % prefill + the loadable≠usable context trap
data/measurements.csv            # Every number in this repo, machine-readable
```

## Key parameter cheat-sheet

| Setting | Value | Where |
|---|---|---|
| `nvidia-powerd` | enabled | systemd |
| D-Bus policy | `nvidia-powerd.conf` in `/etc/dbus-1/system.d/` | required with powerd |
| EC shift mode | `turbo` | `/sys/devices/platform/msi-ec/shift_mode` |
| EC fan | `auto` + cooler_boost `off` | `/sys/devices/platform/msi-ec/` |
| **MoE hybrid fetch** | **`--moe-hybrid-max-fetch 128`** | FreeToken serve args |
| MoE GPU cache | `--moe-cache-size 1400` (VRAM-bound) | FreeToken serve args |
| µ-batch (dense) | `-ub 512` or `1024` (equal) | llama.cpp |
| **MTP draft** | `--spec-type draft-mtp --spec-draft-n-max 3` | llama.cpp (35B MTP) |
| **MoE split (35B @160K)** | `--n-cpu-moe 28` (**24 = OOM**) | llama.cpp (35B) |
| **Ternary packing (Ada)** | **`PTQ1_0`** (5.95 GB, +9 % decode) | llama.cpp (Bonsai) |
| **ik spec decode** | `--spec-type mtp:n_max=3,p_min=0.0 -mtprot q8_0` | ik_llama.cpp (35B) |
| **ik MoE flags** | `--prefetch-experts -ger` | ik_llama.cpp (35B) |
| **Tool-calling clients** | `--jinja` (**required** — without it: 500) | ik_llama.cpp (35B) |
| **ik context (35B)** | `-c 163840` (160K; 240K boots but OOMs on long prompts) | ik_llama.cpp (35B) |
| **Bonsai context** | `-c 262144` (256K, 83 MiB spare) — fallback `-c 229376` (224K, ~1.2 GB spare) | llama.cpp (Bonsai) |
| Reasoning soft close | `--reasoning-budget-message "..."` | llama.cpp (Bonsai) |
| CPU governor | `performance` | cpufreq |

## Reproduce

Everything here is copy-paste runnable. Start with [docs/01-power-unlock.md](docs/01-power-unlock.md) — it applies to **any MSI laptop with an NVIDIA GPU on Linux**, not just this model.

## Credits & sources

This repo is field notes on top of other people's work — sources:

| Component | Source | License |
|---|---|---|
| llama.cpp (upstream inference engine) | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT |
| PrismML llama.cpp fork (`prism-b10743`, docs 03/05/06/07) | [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) | MIT |
| Ternary Bonsai 2 27B GGUF (ternary quant of Qwen3.8-27B) | [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) | Apache-2.0 |
| Qwen3.8-27B (base model) | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | Apache-2.0 |
| Qwen3.6-35B-A3B (MoE base model; UD-Q4_K_XL MTP build by Unsloth) | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) · [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) | Apache-2.0 |
| ik_llama.cpp (MoE-optimized llama.cpp fork, doc 08) | [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) | MIT |
| FreeToken (MoE hybrid engine, doc 02) | [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken) | Apache-2.0 |
| msi-ec (MSI EC kernel module, doc 01) | [BeardOverflow/msi-ec](https://github.com/BeardOverflow/msi-ec) | GPL-2.0 |

Model and project names belong to their respective owners.

## License

MIT © 2026 Joe. Use it, fork it, improve it.

*All measurements are from a single physical machine; treat absolute numbers as that machine's, and the ratios/method as transferable.*
