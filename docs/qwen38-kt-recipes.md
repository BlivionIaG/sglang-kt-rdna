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
