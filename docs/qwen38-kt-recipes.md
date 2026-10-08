# Qwen3.8-Flash-Next-NVFP4 on ktransformers (par1-llm1) — recipes

Two verified configurations. Every number below was measured on this host and is traceable to a
log file; see `running-qwen38-flash-next-on-ktransformers.md` for the full account and the
provenance table.

Hardware: par1-llm1 (Ryzen 9 7945HX, 16 cores for CPU inference, 92 GB RAM, RTX PRO 2000
Blackwell sm_120, 15.93 GiB usable VRAM). Model: `/home/kletorch/models/qwen38-stage` (124 GB,
123.55 GiB of tensors: 65.63 experts + 47.75 PLE + 10.16 other).

**The checkpoint is never modified.** All 11 safetensors still carry their download mtimes.

---

## Recipe A — `g=0`, maximum context (RECOMMENDED for anything long-context)

All 24,576 experts on CPU via KT's AMX path; GPU holds only weights, KV and graphs. This is the
configuration that has **never segfaulted** across every run in this work.

```bash
cd ~/Projects/Infra/upstream-pr/sglang/scripts
PLE_BACKEND=file ISOLATE=1 MEMFRAC=0.95 MEMCAP=200G MEMSWAP=32G GPUEXPERTS=0 \
  ./run-qwen38-kt.sh --host 0.0.0.0 --port 8210 \
  --cuda-graph-backend-prefill=disabled
```

| | measured |
|---|---|
| KV pool | **86,528 tokens** (33% of the model's 262,144) |
| prefill | 494.40 tok/s |
| decode | 8.74 tok/s |
| mean TTFT | 1,114 ms |
| mean ITL | 109.52 ms |
| E2E (c=1, 16k in / 1k out) | 26,854 ms |
| stability | 0 segfaults, every run |

The three PLE flags travel together (`--ple-offload-backend file`, the host-side gather, and
`SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK=1`); the run script sets them from `PLE_BACKEND=file`.
`file` needs the 48 GB derived mmap at
`~/.cache/sglang/ple/.../ple_table_320001536x160_float8_e4m3fn_...bin` — reuse it across restarts,
never delete it.

---

## Recipe B — `g=8`, maximum throughput (38% faster, NOT stable)

Eight experts per layer resident on the GPU (384 of 512 x 48, via `flashinfer_cutlass`).

```bash
cd ~/Projects/Infra/upstream-pr/sglang/scripts
PLE_BACKEND=file ISOLATE=1 MEMFRAC=0.88 MEMCAP=200G MEMSWAP=32G \
  GPUEXPERTS=8 PLACEMENT=frequency \
  ./run-qwen38-kt.sh --host 0.0.0.0 --port 8210 \
  --moe-runner-backend flashinfer_cutlass \
  --cuda-graph-backend-prefill=disabled
```

| | measured |
|---|---|
| KV pool | 44,736 tokens (17%) |
| prefill | **681.16 tok/s** (1.38x) |
| decode | **12.04 tok/s** (1.38x) |
| mean TTFT | **916 ms** (1.22x faster) |
| mean ITL | **79.09 ms** (1.38x faster) |
| E2E | **19,504 ms** (1.38x faster) |
| stability | **segfaults nondeterministically** |

### Do not deploy Recipe B as-is

Six full server runs of the `g>0` path:

| run | g | MEMFRAC | tokens | segfault |
|---|---:|---:|---:|---:|
| plbA | 0 | 0.85 | 53,440 | 0 |
| plbB | 0 | 0.95 | 86,528 | 0 |
| g8cut | 8 | 0.95 | 67,392 | 0 |
| g8cut2 | 8 | 0.88 | 44,736 | 0 |
| sw_g4 (x2) | 4 | 0.88 | 52,032 | **2/2** |
| sw_g8b | 8 | 0.88 | 44,736 | **1** |

`sw_g8b` used the same config that had been clean twice. The crash is
`Subprocess scheduler_0 (pid=...) crashed with exit code -11`, on the first real request just
after `#Input tokens: 13350`. **Of six benchmark attempts, exactly one completed.** The 38% is
real; the reliability is not.

### Why these are the only two points

`--moe-runner-backend flashinfer_cutlass` is the only FP4 MoE backend that runs on sm_120:
`auto` selects TRT-LLM (whose cubins stop at sm107) and `marlin` dies in `sglang_kernel`'s
`moe_sum_reduce` (sm80/sm90 cubins only).

`g=16` cannot load at all, and the gate is the **mamba state cache**, not the KV pool:

    max_running_requests is capped to 0 by the mamba state cache
      (max_mamba_cache_size=2, 5 state slots per request)

| g | MEMFRAC | KV tokens | max_mamba_cache_size |
|---:|---:|---:|---:|
| 0 | 0.95 | 86,528 | — |
| 4 | 0.88 | 52,032 | 9 |
| 8 | 0.88 | 44,736 | 6 |
| 16 | 0.88 | — | 2 (fatal) |

### Two caveats that bound both recipes

1. **The frequency split is not frequency-based.** The logs report
   `Using frequency-based strategy WITHOUT activation frequency data (uniform distribution
   fallback)`: `kt_ep_wrapper.py:4629` gates on `init_loc.endswith(".pt")` and the EPLB field
   `init_expert_location` defaults to the string `"trivial"`, so no file is read and every expert
   ranks identically. **681 tok/s is a floor, not a tuned result.**
2. **The headroom costs more than the experts.** At equal MEMFRAC, 8 GPU experts cost 19,136
   tokens; dropping 0.95 -> 0.88 to survive the first real request (it OOM'd at
   `avail mem 0.13 GB`) cost 22,656.

---

## Bench command

Always pass the served name AND the real tokenizer path — `--model default` alone makes
`bench_serving` try to fetch "default" from HuggingFace and fail.

```bash
cd ~/Projects/kt071-venv && source bin/activate
python3 -m sglang.bench_serving \
  --backend sglang-oai --base-url http://127.0.0.1:8210 \
  --model /home/kletorch/models/qwen38-stage --served-model-name default \
  --tokenizer /home/kletorch/models/qwen38-stage \
  --dataset-name random --random-input-len 16384 --random-output-len 1024 \
  --num-prompts 1 --max-concurrency 1
```

## Reading a run's outcome

    grep -E 'max_total_num_tokens=|max_mamba_cache_size=|Uvicorn running' <log>
    grep -c 'exit code -11' <log>        # 0 = survived; 1 = the instability above

A run that loaded but crashed during benchmarking is the common outcome for `g>0`. That is the
finding, not a setup error.

---

## What else can be tuned — ranked, with the evidence

The honest starting point: this rebuild exposes **22 CLI args total**. Most of the levers the
vendor page and upstream recipes use do not exist here, so the list below separates what we can
actually change from what would need a base bump.

### Tier 1 — available now, untested, most likely to help

**1. `--kt-cpuinfer` 16 -> 24 or 32.** The host has **32 logical CPUs** (16 cores x 2 threads,
1 socket) and we give CPU inference **16 threads**. For Recipe A, where all 24,576 experts run on
CPU, CPU inference is the throughput bottleneck — so this is the single most promising untested
knob, and `--kt-cpuinfer` is already a script parameter (`CPUINFER`). Try 24 and 32 and compare
prefill/decode with the bench command above. Watch for diminishing returns once the threads
contend for the same cores.

**2. `--kt-threadpool-count`.** Currently `1`, which the help says is "one-to-one with the number
of NUMA nodes". This host reports `NUMA node(s): 1`, so `1` is already correct — **no change
available here**, recorded so nobody re-derives it.

**3. `--kt-max-deferred-experts-per-token`.** Present and untouched: "Maximum number of experts
deferred to CPU per token; all MoE layers except the final one use this value." Not yet swept.
Plausible small win for decode; needs a sweep to say.

**4. `--kt-gpu-experts-ratio`** as an alternative to `--kt-num-gpu-experts` (0.0-1.0 of *all*
experts). At ratio 0.0156 (= 8/512) it should reproduce `g=8`. Useful only for expressing a target
fraction rather than a per-layer count; not a perf lever by itself.

### Tier 2 — available, but blocked by something

**5. `--kt-expert-placement-strategy frequency` with a real distribution.** The mechanism exists
and the consumer is wired (`kt_ep_wrapper.py:4629`, reading `{"logical_count": ...}` from
`init_expert_location`), but this rebuild cannot produce the `.pt`: `--record-kt-gpu-expert-distribution`
and `--expert-distribution-recorder-mode` are both absent. **This is the highest-value missing
piece** — it converts the arbitrary uniform split into a real frequency split, and the vendor
measured that lever as worth ~2x on a 256-expert model. Anticipate it being worth less here (our
top-10-of-512 routing is flatter than top-8-of-256), but it should still beat 681 tok/s.

**6. `--kv-cache-dtype fp8_e4m3`** — would ~double the KV pool (the vendor measured 2.00x), i.e.
86,528 -> ~173,000 tokens for Recipe A. **Absent from this build**; the string appears only in a
docstring. Needs a base bump.

**7. `--max-mamba-cache-size` / `--mamba-full-memory-ratio`** — the pair that would make `g=16`
loadable. **Both absent** (0 references). Without them the mamba gate caps `g` at 8 on this card.

### Tier 3 — structural, larger effort

**8. Speculative decoding (MTP/NEXTN).** The vendor's best single-stream number (52.5 tok/s) comes
from `--speculative-algorithm NEXTN` with a draft head. **`--speculative-algorithm` is absent**,
and the checkpoint ships an MTP tensor that this build does not consume. Potentially the biggest
single-stream win available anywhere on this list, and the largest piece of work.

**9. `--cuda-graph-bs` / `--cuda-graph-max-bs`.** Absent. We capture 12 shapes via the config
object instead, which already worked (both graphs captured, 103 s, 1.20 GB).

**10. The `g>0` segfault.** Not a tuning item — a correctness blocker. Until it is understood,
every throughput number from Recipe B is a number you cannot rely on. Hypothesis (unproven): the
fused cutlass grouped-GEMM path over per-layer GPU-expert buffers.

### What I would do first

Run **#1** (`--kt-cpuinfer` 24/32 against Recipe A). It needs no new code, no base bump, and
Recipe A is the configuration that has never crashed — so it is the one place a clean win is
still available. Then, if the goal is throughput rather than context, revisit **#5**, which is a
contained piece of work on this branch.

### The trap to avoid

Do **not** tune `g` upward to gain throughput. `g` costs context (28% of the pool at `g=8`), and
its headroom costs more than the experts themselves. And `g=16` cannot load at all. If you want
speed on this card, the tunable space is CPU threads and placement quality — not more experts on
the GPU.
