# 09 — KV Slot Save/Restore: −79 % Prefill Across Engine Restarts (35B MoE, 12 GB)

**Engine: [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) (MIT) — build `5f89bfc` (`llama-server`), and the [PrismML llama.cpp fork](https://github.com/PrismML/llama.cpp) (`b10754`) as the second subject. Models: [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) UD-Q4_K_XL MTP build ([unsloth](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)) + BF16 mmproj (vision) for ik; [Bonsai-2-27B PTQ1_0](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) (ternary, Apache-2.0, quantized from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)) for the fork.**

## The question

Agent and review pipelines re-run long prompts constantly, and this box runs a **single-engine-at-a-time discipline** (one 12 GB GPU, four engines in rotation). Every engine restart — a planned switch, an OOM recovery — **wipes the in-RAM KV cache**, so the next request pays a full re-prefill: ~30 s for a ~30 K-token prompt on this laptop (measured below), minutes at 100 K+.

The llama.cpp server family can persist a slot's KV state to a file (`--slot-save-path`, `POST /slots/{id}?action=save|restore`). The question: **is the "save before the restart, restore after" routine actually worth it, and does the restored state participate in prefix reuse on the next, possibly different, prompt?**

## Short answer

- **ik_llama.cpp: yes — −79 % total prefill** on a 6-leg workload where the engine is restarted before every leg (180.5 s → 37.9 s).
- **PrismML fork: no reuse.** The restore loads the state (0.12 s) but the subsequent request still re-prefills the full prompt. The save/restore plumbing works; the reuse step does not engage on this build.
- **Media choice:** RAM (`/dev/shm`) makes the *save* ~13× faster (0.41 s vs 5.2 s), restores are comparable; but each slot is ~1.3 GB on disk, and piling 10 of them into tmpfs (12 GB) **OOM-killed the resident engine** — see the incident section.

## Mechanism & file layout

Server flag: `--slot-save-path <dir>`. API: `POST http://host:port/slots/0?action=save|restore` with `{"filename": "name.slot"}`. `-np 1` = a single slot, id 0.

One "slot" writes **three files**:

| File | Size (ik, ~30 K-token state) | Role |
|---|---:|---|
| `X.slot` | 230 MB | the KV/state blob |
| `X.slot.checkpoints` | **1.07 GB** | checkpoint metadata (the bulk!) |
| `X.slot.tokens.json` | 0.1 MB | token list — what the prefix matcher compares against |

**≈ 1.3 GB per slot.** Budget accordingly (this was the source of a real incident below).

## Experiment: 36 leg-runs, engine restarted before *every* leg

**Design.** A synthetic review round-trip mimicking a real multi-pass review pipeline:

- Prompt structure = a ~24 K-token shared material block, **plus per-round appended blocks (R1 → R3, append-only)** — the stable-blocks-first ordering exists precisely for prefix caching — **plus** a short task tail that differs per leg (`test` view vs `review` view).
- Legs, in order: `test-R1, review-R1, test-R2, review-R2, test-R3, review-R3` — the two "legs" alternate, as real pipelines do, so a naive single-slot cache gets thrashed.
- **Groups:** A = no cache (each leg pays full prefill) · B = slot files on an NVMe partition (ext4) · C = slot files on tmpfs (`/dev/shm`). Each group stops the engine, restarts it fresh (simulating a switch/rebuild), then runs the leg. B/C also `save` after each leg and `restore` the previous leg's slot before the next.
- Protocol per leg: full process restart → health check → [restore] → request → [save]. Temperature 0, default prompt caching on. The two engines were run back to back with identical prompts and per-leg full restarts.

**The traps are the point of the design:** the slot is *one*, the legs *alternate*, and every leg *starts from a fresh process* — exactly where a naive "just keep it warm" assumption dies.

### Results — ik_llama.cpp

Per-leg **prefill** (seconds, ~30 K tokens per leg):

| Leg | A (no cache) | B (SSD) | C (RAM) |
|---|---:|---:|---:|
| test-R1 | 29.5 (cold) | 29.1 (cold) | 28.8 (cold) |
| review-R1 | 29.4 | **1.3** | **1.3** |
| test-R2 | 30.0 | **2.1** | **2.1** |
| review-R2 | 29.7 | **2.1** | **2.1** |
| test-R3 | 30.6 | **2.8** | **2.8** |
| review-R3 | 31.3 | **0.5** | **0.6** |
| **Total** | **180.5 s** | **37.9 s** | **37.6 s** |

**−79 % total prefill**, and the win *grows* with the round count: leg 1 must be cold (nothing to restore), every subsequent leg lands 0.5–2.8 s. The residual 0.5–2.8 s per leg is exactly the divergent tail (new round block + the differing task tail) plus the restore load — the shared material never recomputes again.

Decisive probes (isolating the mechanisms, same engine, ~30 K-token state):

| Probe | `prompt_n` processed | Prefill | Note |
|---|---:|---:|---|
| restore → **same** prompt | **5** | 104 ms | of a 30,322-token prompt, only the last 5 tokens computed — full reuse |
| restore → **different tail** prompt | **1,605** | 2.1 s | only the divergent tail recomputed (the matching is longest-common-prefix) |
| no restore, warm process, same prompt | 1,603 | 2.0 s | radix cache inside a live process — the baseline the slot lifecycle *extends across restarts* |

A separate 96 K-token single-prompt check (`110,016` tokens): cold prefill 137.7 s; save 11.3 s (2.7 GB written); process restart; restore 2.8 s; **re-request = `prompt_n` 5, prefill 0.184 s** — a ~750× improvement on that request.

### Results — PrismML fork (Bonsai)

Save/restore both work and are fast (save 2.7 s / restore **0.12 s**), but **every leg still re-prefills 24.8–26.0 s regardless of restore** — total 152.5 s for 6 legs in *all three* groups. The restored state is loaded; the prefix reuse step simply does not engage on this build. For this fork, slot files are still useful as *state persistence* (resume an interrupted conversation after a restart), but **not** as a prefill accelerator. Root cause is on the fork's side; worth re-testing on future builds.

## Save/restore timings (the media account)

| Combination | Save | Restore | Bytes written |
|---|---:|---:|---:|
| ik · SSD (`/mnt/win`, NVMe) | 5.2–5.5 s | 1.0–1.5 s | 1.36 GB |
| ik · RAM (`/dev/shm`) | **0.41 s** (13×) | 1.0 s | 1.36 GB |
| Bonsai · SSD | 2.7–3.1 s | 0.12 s | 705 MB |
| Bonsai · RAM | **0.25 s** (11×) | 0.11 s | 705 MB |

- RAM is a clear win for the *save*; restores are close (the OS page cache serves the SSD reads after a while).
- With ~1.3 GB per slot, **cap RAM slots at 1–3 and default to disk**; clean tmpfs after experiments.

## The incident: 10 slots in tmpfs OOM-killed the resident engine

Ten-plus benchmark slots (~1.3 GB each) left in `/dev/shm` consumed **12 GB of the 30 GB box** — invisible to the benchmark, until the resident engine's next model load hit the OOM killer: `systemd: freetoken.service: A process of this unit has been killed by the OOM killer` … `Failed with result 'oom-kill'`. The engine's `Restart=always` brought it back once tmpfs was cleared. **Lessons (the hard way):**

1. **Slot entities are ~1.3 GB each — the `.checkpoints` file is the bulk, not the `.slot` blob.** Plan tmpfs budget from the checkpoints number.
2. **Clean `/dev/shm` at test teardown.** The slots look tiny on `ls` (230 MB), and the hidden companion file is 4× bigger.
3. **The slot files in tmpfs count against every future allocation** — the OOM hit an *unrelated* service minutes later, not the benchmark.

And a testing-rig lesson, from the same night: **copy launch parameters from the unit that actually runs in production, not from a script lying around.** A stale start script (f16 KV on a 163 K context) needs 10 GB of KV and dies in `cudaMalloc` on a 12 GB card; the live unit (q4_0 KV, 262 K context) fits in ~2.5 GB of KV. Same engine, same model, two worlds.

## Memory budget (side audit, same night)

Host RSS of each engine while resident (this box, 30 GB RAM):

| Engine | Host RSS | Notes |
|---|---:|---|
| ik_llama.cpp | **18.8 GB** | `--n-cpu-moe 28` puts 28 layers of experts on the CPU |
| PrismML fork (Bonsai) | **2.3 GB** | everything fits on the GPU; the lightest resident by far |
| GLM-4.7-Flash (ik build) | 10.1 GB | `--n-cpu-moe 28`, q4 KV |
| FreeToken | ~4–6 GB | during load; peaks higher |

Peak discipline matters: with ~20 GB already committed to a resident engine, **+2 GB of tmpfs slots is already inside the OOM's blast radius** on this box.

## Why this matters for agent pipelines

- The win is **per-leg, per-restart** — i.e. proportional to how much leg switching / engine cycling a pipeline does. For a review pipeline that restarts between rounds (or alternates two engines), −79 % of prefill time is real wall-clock.
- The reuse keys on the **longest common prefix**, so the stable-blocks-first prompt discipline (shared material at the top, round blocks appended later) compounds: the earlier document order work and this save/restore work multiply rather than just add.
- The residual cost per leg is the divergent tail + the restore load: ~0.5–2.8 s here for ~30 K-token prompts, scaling with the *delta*, not the total.

## Reproduce

1. Start the server with `--slot-save-path <dir>` and `-np 1`.
2. Per leg: `POST /slots/0?action=save -d '{"filename":"<leg>.slot"}'`; before the next leg `action=restore`. Restart the process between legs to model a switch.
3. Read `timings.prompt_n` / `timings.prompt_ms` from the chat completion response: `prompt_n` near zero = everything was reused.

## Credits & sources

- Engines: [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) (MIT) · [PrismML llama.cpp fork](https://github.com/PrismML/llama.cpp) · upstream [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- Models: [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) · [unsloth UD-Q4_K_XL MTP GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) · [Bonsai-2-27B PTQ1_0 gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) (quantized from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B))
- Hardware: see [07](07-vram-budget-and-long-context.md) — same laptop (RTX 4080 Laptop 12 GB, Ryzen 9 7945HX3D, 30 GB RAM), same power wall

---
MIT © Joe. Part of a series (see [README](../README.md)): same laptop, four engines, one 12 GB budget.
