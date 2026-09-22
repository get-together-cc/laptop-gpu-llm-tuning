# Laptop GPU LLM Inference Tuning

**Real-world tuning notes for running large language models on a laptop RTX 4080 (12 GB) under Linux** — measured data, exact commands, and the pitfalls we hit.

This is not a benchmark paper. It is a field report: two production models tuned on the same laptop, every number measured with the same methodology, every failed attempt documented.

## What you get

| Achievement | Before | After | Gain |
|---|---|---|---|
| **GPU power limit (Linux)** | 80 W (locked) | **175 W** (Dynamic Boost fully unlocked) | **+119 %** power budget |
| **MoE hybrid prefill** (35B-A3B) | ~2000 tok/s | **~3090 tok/s** | **+54 %** |
| **MoE hybrid decode** | 42–58 tok/s | **60 tok/s** | best-ever |
| **Ternary dense prefill** (27B) | 757 tok/s | **1143 tok/s** | +51 % (power unlock) |

All measured on the same machine, same day, same methodology (see [methodology](docs/04-methodology.md)).

## Hardware

| Component | Detail |
|---|---|
| Laptop | MSI Raider A18 HX A7VHG (MS-182K) |
| GPU | NVIDIA GeForce RTX 4080 Laptop GPU, 12 GB GDDR6 (`10de:27a0`, subsystem `1462:1440`) |
| CPU | AMD Ryzen 9 7945HX3D, 16 cores / 32 threads |
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
| Model A | **FreeToken** engine (Qwen3.6-35B-A3B NVFP4, MoE hybrid — experts on CPU) |
| Model B | **PrismML** llama.cpp fork `prism-b10709-9a9394a` (Ternary Bonsai 2 27B, PQ2_0, dense ternary {−1,0,+1}) |

## The three big findings

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

## Repository layout

```
docs/01-power-unlock.md          # 80 W → 175 W: complete steps + verification
docs/02-freetoken-moe-tuning.md  # MoE hybrid prefill -35 %: full experiment matrix
docs/03-bonsai-ternary-tuning.md # Ternary dense model: boundaries and dead ends
docs/04-methodology.md           # How we measure (and why naive A/B lies)
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
| CPU governor | `performance` | cpufreq |

## Reproduce

Everything here is copy-paste runnable. Start with [docs/01-power-unlock.md](docs/01-power-unlock.md) — it applies to **any MSI laptop with an NVIDIA GPU on Linux**, not just this model.

## License

MIT © 2026 Joe. Use it, fork it, improve it.

*All measurements are from a single physical machine; treat absolute numbers as that machine's, and the ratios/method as transferable.*
