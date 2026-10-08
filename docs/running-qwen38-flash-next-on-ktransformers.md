# Running Qwen3.8-Flash-Next-NVFP4 on ktransformers

What it took to serve a 123 GB NVFP4 checkpoint on a **single 16 GB GPU + 92 GB host**, without
touching a byte of the model. Every number below comes from a command run against the live
server; the raw logs are named so each can be re-derived.

---

## The machine

    GPU    RTX PRO 2000 Blackwell, 15.93 GiB total, 15.17 GiB usable (sm_120)
    RAM    92 GiB  (AMD Ryzen 9 7945HX, 32 threads, 1 NUMA node)
    disk   one LUKS+btrfs NVMe, 464 GB
    host   par1-llm1, 192.168.1.121

## The checkpoint

`nvidia/Qwen3.8-Flash-Next-NVFP4`, 123.55 GiB total, **11 safetensors**:

| component | GiB | what it is |
|---|---:|---|
| routed experts | **65.63** | 297,984 tensors, NVFP4 — 512 experts x 48 layers |
| PLE n-gram table | **47.75** | `model-fp8-mtp-ple.safetensors`, 51.7 GB, FP8 e4m3fn, `[2500012, 160]` |
| GDN / linear attention | 3.89 | 36 of the 48 layers are linear attention |
| embed / lm_head | 2.37 | |
| everything else | 3.91 | norms, conv, router, shared expert, 12 full-attn layers |

Model shape: hidden 2560, 48 layers, 512 experts/layer top-10, moe_intermediate 640,
24 q-heads / 2 kv-heads, head_dim 256, vocab 248,320, `context_len` 262,144.

---

## The placement: three tiers, and why each is forced

    15.93 GiB GPU  <-  non-expert weights + KV cache
    92 GiB RAM     <-  48 x 512 routed experts, via ktransformers' AMX path
    464 GB disk    <-  the PLE n-gram table, as a sparse mmap

**The arithmetic that forces it** (measured, not estimated):

| | GiB | can go on GPU? |
|---|---:|---|
| routed experts | 65.63 | no — 4x the card |
| PLE table | 47.75 | no — 3x the card |
| everything else | 10.17 | yes |

The experts go to host RAM; that is ktransformers' whole purpose and it works:
`[NVFP4SafeTensorLoader] Loaded 512 experts` for each of 48 layers, `AVX2_MXFP4_MOE_TP` created
per layer, 16 AMX threads.

**One measured surprise.** The loader does not hold the checkpoint's packed bytes. Budgeting
from disk says 63.28 GiB for the experts; the kernel reports **83.16 GiB** of resident memory at
40 of 48 layers — a **1.31x** overhead, because the KT path materialises per-expert host buffers
in an AMX-workable layout. That discrepancy is the single number that decides whether a machine
can run this model, and getting it wrong (from disk instead of from the kernel) is what made the
first several attempts look impossible.

---

## The PLE table: the part that does not work the obvious way

Upstream's own module docstring settles it:

> Matched for unified-memory parts (GB10 / DGX Spark and similar), where pinned host memory comes
> out of the *same* pool as the model weights and `pinned` therefore frees nothing:
> **Qwen3.8-Flash-Next is 126.0 GiB of weights on a 121.63 GiB box and does not boot with
> `pinned`.**

So `--ple-offload-backend pinned` — the default — cannot boot this model at this size, and it is
what wedged the host three times during bring-up.

The alternative is `--ple-offload-backend file`: a sparse mmap, paged in on demand, with a
built-in `WILLNEED` prefetcher and an RSS trimmer. On this GPU it is refused, because its gather
is a Triton kernel that dereferences the host pointer **from the device**:

    values = tl.load(weight_ptr + local_idx * embedding_dim + offsets, ...)

and that needs `cudaDevAttrPageableMemoryAccessUsesHostPageTables` — a unified-memory feature
this discrete card reports as False.

**The fix is to move the read across the bus.** `SGLANG_QWEN4_PLE_HOST_SIDE_GATHER=1` gathers the
same rows on the CPU from the mmap and copies them over, mirroring the kernel including
out-of-shard masking. The traffic is small by construction — 16 rows per token (8 two-gram +
8 three-gram heads), ~2.5 KB in fp8 — so this is a correctness path, not a fast one.

    # all three flags belong together
    --ple-offload-embedding --ple-offload-backend file
    SGLANG_QWEN4_PLE_HOST_SIDE_GATHER=1
    SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK=1

The device check exists to stop the Triton kernel reading garbage; the host-side gather is what
makes that kernel irrelevant. Bypassing the check **without** the gather reintroduces exactly the
failure it guards against.

---

## CUDA graphs: a 106x TTFT win, and what it cost to find

The bring-up config had both graphs disabled. Enabling them is where the performance is:

| metric | graphs off | graphs on | change |
|---|---:|---:|---|
| TTFT | 117,927 ms | **1,114 ms** | **106x** |
| ITL (mean) | 122.24 ms | **109.52 ms** | 1.12x |
| prefill | 91.01 tok/s | **494.40 tok/s** | **5.4x** |
| decode | 1.61 tok/s | 8.74 tok/s | 5.4x |
| E2E | 146.7 s | **26.9 s** | 5.5x |

c=1, 16k input / 1k output, `sglang.bench_serving --dataset-name random`. Both runs produced
identical token counts (13,350 in / 236 out), so the comparison is like-for-like.

**The trap: the default graph set is 42 shapes and OOMs.**

    MEMFRAC=0.85 + 42 shapes -> CUDA out of memory at 64% of capture, after ~46 minutes
    MEMFRAC=0.85 + 12 shapes -> 100% in 103 s using 1.20 GB  (decode: 1.59 s, 0.04 GB)

Capture cost scales with **shape count**, not with the memory fraction. And
`chunked_prefill_size=2048` means a single prefill graph never exceeds 2048 tokens, so a
power-of-two list to 2048 covers the real workload:

    --cuda-graph-bs-prefill 1 2 4 8 16 32 64 128 256 512 1024 2048

**`--cuda-graph-prefill-max-context` does not apply here.** It is rejected outright:

    ValueError: --cuda-graph-prefill-max-context is only supported by attention backends that
    implement fixed-context prefill graph metadata; got Hybrid...

This model's `triton` backend is hybrid — 36 linear-attention + 12 full-attention layers.

**Stability caveat, stated plainly.** With prefill capture *enabled* alongside decode, the
scheduler segfaulted (`exit code -11`) shortly after serving. With
`--cuda-graph-backend-prefill=disabled` it is stable and decode graphs engage
(`cuda graph: True` on every decode batch). The shipped configuration is the stable one. The
suspected cause — graph replay touching the KT path's host-pinned expert buffers — is a
hypothesis, not a finding.

---

## The memory split, and the context ceiling

sglang reports its own numbers; these are quoted, not derived:

    Load weight end. avail mem=4.77 GB          (10.40 GB of weights)
    KV Cache is allocated. dtype: torch.bfloat16, #tokens: 53440, K 0.61 GB, V 0.61 GB
    max_total_num_tokens=53440, chunked_prefill_size=2048, max_prefill_tokens=16384,
      max_running_requests=1, context_len=262144, available_gpu_mem=3.63 GB

**KV cost: 22,829 bytes/token**, measured two ways and stable across both:

    53,440 tokens -> K 0.61 GB + V 0.61 GB   (MEMFRAC=0.85)
    86,528 tokens -> K 0.99 GB + V 0.99 GB   (MEMFRAC=0.95)

## Max usable context: 86,528 tokens (measured, not modelled)

The model advertises `context_len=262144`. On this card the real ceiling is **33% of that**,
because VRAM — not the architecture — is the binding constraint. Measured by raising `MEMFRAC`
and reading what sglang actually allocates:

| MEMFRAC | KV pool allocated | of the model's 262,144 |
|---:|---:|---:|
| 0.75 | 20,416 tokens | 7.8% |
| 0.85 | 53,440 tokens | 20% |
| **0.95** | **86,528 tokens** | **33%** |

At `MEMFRAC=0.95` the server reaches `health: 200`, answers coherently, and captures decode
graphs (`elapsed=4.15 s, mem usage=0.26 GB`) with **zero segfaults**. `max_running_requests=2`.

The full 262,144-token context would need **5.57 GiB** of KV. After 10.40 GiB of weights plus
activations and the graph pool, a 15.93 GiB card cannot supply that. **The split that maximises
context is `MEMFRAC=0.95` with decode-only graphs.**

**A caution about modelling this.** A straightforward budget model — `pool = MEMFRAC x GPU -
weights - overhead` — predicted ~146,000 tokens at 0.95, against the 86,528 sglang measured, an
over-prediction of 69%. The per-token cost was right; the budget was wrong, because sglang
reserves more for activations and fragmentation than the model allowed. Measure the pool; do
not compute it.

---

## Reproduce

    # max context (86,528 tokens), verified serving:
    PLE_BACKEND=file ISOLATE=1 MEMFRAC=0.95 MEMCAP=200G \
      ./scripts/run-qwen38-kt.sh --host 0.0.0.0 --port 8210 \
      --cuda-graph-backend-prefill=disabled

    # the split the benchmark numbers above were taken at (53,440 tokens):
    #   ... MEMFRAC=0.85 ...

Prerequisites that are easy to get wrong:

- **torch 2.14.1+cu130**, and **`sglang-kernel` 0.4.9** — NOT `sgl-kernel` 0.3.21, whose
  `common_ops.abi3.so` links against torch 2.9 and fails with an undefined symbol.
- **flashinfer 0.7.0.post1**: `flashinfer-cubin` and `flashinfer-jit-cache` are **not on PyPI
  above 0.6.13**; they come from `https://flashinfer.ai/whl` (the jit-cache path carries a
  `cu${CUINDEX}` suffix, cubin's does not).
- **`ISOLATE=1`** runs the load under `systemd-run --scope -p MemoryMax=...`. Without it, a
  failing load wedges the host — the kernel ends up unable to fork, so sshd accepts TCP and
  never sends a banner.

---

## What went wrong, and what each taught

Recorded because these are the traps a reader would hit, not war stories:

1. **Two ServerArgs subtleties.** This base uses `__slots__`, so a CLI default must come from
   `_declared_default("field")` — `default=ServerArgs.field` registers the slot *descriptor*.
   And `dataclasses.field(...)` must not be used for new fields, because `record_fields` reads
   the class attribute and the `Field` object then leaks into `resolved_dict()` as
   `TypeError: cannot pickle 'mappingproxy' object`.
2. **`until ssh ...; do sleep; done` is the wrong watcher shape**: it terminates exactly when the
   host is unavailable, which is the only time it matters. Use `while true; do out=$(ssh ...);
   case "$out" in ...) break;; esac; done`.
3. **A shell script that drops its arguments.** `exec ... "${ARGS[@]}"` with no `"$@"` silently
   ignored every extra flag — including `--cuda-graph-backend-prefill=disabled` and `--host`.
   The flag appeared to be unsupported; it was simply never passed.
4. **`pkill -f sglang.launch_server` kills your own SSH session**, because the pattern matches the
   ssh command line. Several launches failed with a staged script, no log, and no process.
5. **Read the resolved config, not an init message.** `Init Unified Radix Cache` appears in the
   log whether or not radix cache is enabled. Only `/get_server_info` -> `internal_states[0]`
   tells the truth.
6. **`rc=0` through a pipe is not the exit code.** `rm -rf DIR 2>&1 | head; echo $?` reports
   *head's* status — a total failure looked like success, twice, in different forms.
7. **The kernel OOM table's first column is `total_vm`, not `rss`.** Reading `total_vm` (330 GB)
   as memory usage produced a confident, entirely wrong diagnosis.
8. **Verify a run is NEW before concluding a fix failed.** Twice I read a log from an earlier
   attempt as the current one. Compare mtime against `date`, and check for a live PID.

---

## Honest limits

- The **host-side PLE gather is a correctness path**, not an optimised one: a CPU gather plus one
  H2D copy per call. A unified-memory part (GB10 / DGX Spark) would run the original UPM kernel
  with no copy at all.
- **Decode graphs only.** Prefill capture is disabled for stability; the segfault under both
  phases is unexplained.
- **Token throughput is modest** (8.74 tok/s decode at c=1) because the experts run on 16 AMX
  threads over 65.6 GiB of host RAM. That is the cost of the offload, and it is the trade that
  makes the model fit at all.
- **`max_running_requests=1`.** This is a single-stream configuration; concurrency is untested.

---

## Provenance

Every figure was checked back against a log line from the session that produced it:

| figure | source |
|---|---|
| 65.63 / 47.75 / 10.16 / 123.53 GiB | summed from `model.safetensors.index.json` with `safe_open` |
| 1.31x in-kernel overhead | kernel OOM record (`anon-rss:72665692kB` at 40/48 layers) vs the same 40 layers on disk |
| TTFT / ITL / prefill / decode, both rows | `sglang.bench_serving` output for each run |
| 42 shapes OOM | `torch.AcceleratorError: CUDA error: out of memory` at `avail_mem=0.01 GB` |
| 12 shapes in 103 s / 1.20 GB | `Capture target prefill CUDA graph end. elapsed=102.93 s` |
| `cuda graph: True` | decode batch lines in the server log (`x10` during the benchmark) |
| 53,440 / 86,528 tokens | `KV Cache is allocated ... #tokens:` and `max_total_num_tokens=` |
| segfault under both phases | `scheduler_0 (pid=33237) crashed with exit code -11` |

Log files, on the host: `/tmp/bench-c1.log` and `/tmp/bench2.log` (the two benchmark rows),
`/tmp/plb5.log` (graphs off), `/tmp/plb9.log` (both graph phases, segfault), `/tmp/plbA.log`
(decode-only, stable), `/tmp/plbB.log` (the 86,528-token ceiling).
