# 04 — Methodology: How We Measure (and Why Naive A/B Lies)

Every number in this repository comes from the procedure below. It exists because each shortcut we tried produced a wrong answer at least once.

## The measurement protocol

```bash
# One cold measurement, per configuration:
#  1. restart the engine (clears the radix cache)
#  2. wait for REAL readiness (see below)
#  3. send a FRESH prompt of the target size, max_tokens=5
#  4. take the wall time; discard the first sample after any restart
#  5. run 2–4 samples with DIFFERENT prompt texts, keep the spread
```

### 1. Fresh prompts only — caches lie

Both engines here use a radix prefix cache. Re-sending the same prompt returns in **1.5 s instead of 20 s**, with the engine log plainly saying `#cached-token: 42880`. That is a cache hit, not a prefill. Every measurement uses a newly generated prompt (same length, different wording).

### 2. Real readiness, not "the port answers"

```bash
# WRONG: the HTTP server answers while weights are still loading
curl -s http://127.0.0.1:1919/v1/chat/completions ... → {"error":"model is still loading"}

# RIGHT: poll until memory plateaus AND the API stops erroring
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # wait for >10 GB again
```

A 35 B model takes **60–90 s** to load (GPU part first, then MoE experts into RAM). We have wasted three separate test cycles on "8-second readiness".

### 3. The first sample after a restart is slow — discard it

Observed on the same configuration: `23.3 / 19.4 / 19.3 s`. The first run pays cold-start costs (CUDA kernel selection, allocator warm-up). **Report the steady state, or measure twice and throw the first away.**

### 4. Never trust a single sample

Our final configuration measured `12.7 / 13.7 / 13.7 / 14.2 s` across four fresh prompts — an 11 % spread on identical settings. Any claimed improvement smaller than that spread needs more samples, not more confidence.

## The two traps that cost us the most time

### Trap 1 — parameter interactions invalidate old conclusions

We measured `--moe-prefill-hit-d2d + --moe-cpu-threads 32` as **−7 %** before `--moe-hybrid-max-fetch` existed. After adding `max-fetch=128`, the same two flags became **+19 % slower**, and removing them recovered the time.

**Rule: after any change that shifts where work runs, re-run the whole matrix from a new baseline.** Old A/B results are not transferable across workload rebalancing.

### Trap 2 — "the flag returned success" ≠ "the flag took effect"

```bash
$ sudo nvidia-smi -lmc 9500
Memory clocks set to "(memClkMin 9500, memClkMax 9500)"     # smiles at you
$ nvidia-smi --query-gpu=clocks.mem --format=csv,noheader
9001 MHz                                                     # and does nothing
```

Always **verify the effect**, not the exit status: read the sysfs attribute, the engine's own startup log (`ServerArgs(...)`), or the live clock — whichever reflects truth.

## Engine-specific notes

| Engine | Where the truth lives |
|---|---|
| FreeToken (`ft serve`) | `journalctl -u <svc>` → `ServerArgs(...)` line = what was *actually* parsed; `Prefill batch ... input throughput (token/s)` per batch |
| llama.cpp (`llama-server`) | response `timings.prompt_per_second` (per-request); startup log for `n_threads`, graph reuse |
| FreeToken streaming requests | first byte arrives in ~4 ms (headers) — TTFT is meaningless; use wall time with `max_tokens=5` |
| Both | `nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu -l 2` under load to confirm the hardware actually engaged |

## Operational hygiene

- **Long apt/upgrade runs over SSH**: wrap in `sudo nohup bash -c '...' > /tmp/run.log 2>&1 &` — an SSH disconnect kills plain remote commands mid-upgrade (we hit this; the package set was left half-configured and needed `dpkg --configure -a`).
- **Restart your engine after changing its systemd unit** — and check the parsed-config log line, not your editor.
- **Keep a config backup before every experiment**: `cp service.service service.service.bak-$(date +%F)`.
- **One change at a time**, and a **fresh baseline after any structural change** (power limit, engine version, kernel).

## The checklist we run before believing any number

1. Prompt is new (cache cold) ✓
2. Engine reported ready (memory plateau + no loading errors) ✓
3. First-after-restart sample discarded ✓
4. ≥2 samples, spread noted ✓
5. Hardware state confirmed under load (power/clocks) ✓
6. Effect verified independently (not just "command returned 0") ✓
7. Baseline re-measured if any structural variable changed ✓
