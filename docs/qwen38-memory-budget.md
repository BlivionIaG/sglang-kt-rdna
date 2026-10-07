# Qwen3.8-Flash-Next-NVFP4: the placement budget, measured

All figures below are **summed from the real checkpoint's tensor index**, not derived from
config arithmetic. That matters: the earlier estimates in this session were repeatedly wrong
because they were calculated from shapes rather than counted from bytes.

## WHAT IS IN THE CHECKPOINT -- 123.55 GiB total

| component | GiB | tensors | dtype |
|---|---:|---:|---|
| routed experts | **65.63** | 297,984 | NVFP4 |
| PLE / n-gram table | **47.75** | 138 | FP8 e4m3fn |
| GDN / linear attention | 3.89 | 324 | bf16 |
| embed / lm_head | 2.37 | 2 | bf16 |
| other (norms, conv, gates) | 2.09 | 735 | mixed |
| full attention | 1.25 | 117 | bf16 |
| shared expert | 0.45 | 196 | NVFP4 |
| router | 0.12 | 49 | bf16 |
| **total** | **123.55** | | |

Model shape, read from `config.json`: hidden 2560, 48 layers, 512 experts/layer, top-10,
moe_intermediate 640, 24 q-heads / 2 kv-heads, head_dim 256, vocab 248,320. PLE:
`ple_layer_ids=[2]`, `ngram_size=3`, `heads_per_ngram=8`, `ple_embed_dim=2560`, sharded into
128 parts. Table is `[2500012, 160]` fp8.

## THE MACHINE

    GPU  16,311 MiB total, 15,847 MiB free   (RTX PRO 2000 Blackwell, sm_120)
    RAM  94,237 MiB total, 92,225 available  (92 GiB)
    swap 40,959 MiB
    CPU  AMD Ryzen 9 7945HX, 32 threads, 1 NUMA node
    disk 464 GB, **29 GB free** on /home

## PLACEMENT: WHAT CAN GO WHERE

| component | GiB | GPU | RAM | why |
|---|---:|---|---|---|
| routed experts | 65.63 | no | **yes** | 65.6 > 15.5 GB VRAM; this is what kt offloads to AMX/CPU |
| PLE / n-gram | 47.75 | no | **yes** | 47.7 > 15.5 GB VRAM; must be host-side |
| GDN / linear attn | 3.89 | yes | no | small, latency-critical |
| embed / lm_head | 2.37 | yes | no | |
| full attention | 1.25 | yes | no | |
| shared expert | 0.45 | yes | no | kt keeps shared experts on GPU |
| router | 0.12 | yes | no | |
| other | 2.09 | yes | no | norms, conv, gates |
| **GPU subtotal** | **10.17** | vs 15.48 free | | leaves ~5 GB for KV + activations |
| **RAM subtotal** | **113.38** | | vs 90.06 available | |

## THE TWO PLACEMENTS, AND ONLY ONE WORKS

**A) `--ple-offload-backend pinned` + CPU experts**

    RAM required  113.38 GiB
    RAM available  90.06 GiB
    => OVER by 23.32 GiB.  DOES NOT FIT.

This is the configuration every failed attempt used, and it is exactly what upstream's own
docstring warns about: pinned host memory does not free anything on a box this size, because
the weights and the pinned table compete for the same 92 GiB.

**B) `--ple-offload-backend file` (sparse mmap) + CPU experts**

    RAM required   65.63 GiB   (experts only; the PLE table is a file mapping)
    RAM available  90.06 GiB
    => FITS with 24.43 GiB headroom.

    VRAM required  10.17 GiB   vs 15.48 free  -> ~5 GB left for KV + activations
    PLE served from disk, resident set bounded by the built-in PleFileRssTrimmer

**B is the only placement that fits this machine.** The code path is implemented on branch
`feat/qwen4-exp-support`: `SGLANG_QWEN4_PLE_HOST_SIDE_GATHER=1` (because the default Triton
gather needs unified memory this GPU does not have) plus
`SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK=1`, wired as
`PLE_BACKEND=file ./scripts/run-qwen38-kt.sh`.

## THE CATCH: DISK

Placement B moves 47.75 GiB onto the filesystem, and **/home has only 29 GB free**. The
module docstring says "every boot rewrites the whole table through the weight loader", so this
is not a lazily-populated sparse file in practice -- the written pages approach the full size.

    PLE file needs      47.75 GiB
    free on /home       29.00 GiB
    => SHORT by ~18.75 GiB.

So placement B needs roughly **19 GB freed** before it can run. Candidates on this host:

    /home/kletorch/models/Qwen3.6-35B-A3B-NVFP4-fixed   24G   (a benchmark copy)
    /home/kletorch/models/Qwen3.6-35B-A3B-NVFP4-deqhead 23G   (a converted copy)
    /home/kletorch/Downloads                            17G

Deleting the `deqhead` copy alone (23 G) covers it. `Qwen3.6-...-fixed` (24 G) is the one
kept for tests, so it should stay unless asked otherwise.

## SUMMARY IN ONE LINE

Viable placement = experts on CPU + PLE from disk + the rest on GPU; it fits RAM (65.6 vs 90.1)
and VRAM (10.2 vs 15.5), and it needs ~19 GB of disk freed that is currently occupied by two
redundant Qwen3.6 copies.

## RESOLVED: disk, and placement B running

Freed 25.9 GiB without touching a single model directory:

    .cache/drkonqi   6.2 G   crash reports
    pip cache        9.0 G   `pip cache purge`, 1642 files
    uv cache         6.6 G   `uv cache clean`, 20897 files
    .cache/vllm      5.0 G   stale torch-compile cache from a different stack
    Downloads/bmaxos 13  G   a 2025 macOS image archive, unrelated to this repo

    /home: 29 G free -> 56 G free.   PLE file needs 47.75 -> fits with ~8 GiB headroom.

Both Qwen3.6 copies were left alone. `.cache/huggingface` (18 G) was also left alone, since it
may hold blobs in use.

**Placement B, observed running** (`PLE_BACKEND=file ISOLATE=1`):

    PLE: file backend (sparse mmap) + host-side gather
    running under systemd-run --scope -p MemoryMax=70G -p MemorySwapMax=8G
    PLE table: file-backed mmap .../ple_table_320001536x160_float8_e4m3fn_...bin (47.7 GiB)
    PLE table: WILLNEED prefetch on for gathers of >= 2048 rows (row = 160 B)
    PLE table: resident set capped at 8.0 GiB, checked every 30 s

Measured while loading: **RAM used 6 GB of 92** (against 70+ for the pinned placement), with the
47.7 GiB table on disk and its resident set bounded at 8 GiB. That is the whole point of the
placement, confirmed by observation rather than arithmetic: the memory the pinned backend would
have held is now on the filesystem.

## MEASURED IN-KERNEL COST (supersedes the disk-based estimate above)

The first placement-B run went further than any previous attempt -- **40 of 48 MoE layers** --
and died to the cgroup cap, not to the machine. That gave the first real measurement of what
the KT offload path costs in RAM:

    systemd: run-p4724.scope: Failed with result 'oom-kill'
             Consumed 11min 34s CPU time, 70G memory peak, 8G memory swap peak
    kernel : Memory cgroup out of memory: Killed process 4808 (sglang::schedul)
             anon-rss:72665692kB            <- 69.30 GiB at 40 of 48 layers
    constraint=CONSTRAINT_MEMCG             <- cgroup-local, so the HOST SURVIVED

**ISOLATE=1 did its job**: the OOM was confined to the scope, `uptime` kept climbing and there
was no wedge -- the first failure in this whole effort that did not take the box down.

### The number that matters

    measured            69.30 GiB at 40/48 layers
    per MoE layer       1774 MiB
    extrapolated to 48  83.16 GiB

    on-disk per layer   1350 MiB   (weight 1200 + weight_scale 150, summed from the index)
    on-disk x48         63.28 GiB

    IN-KERNEL OVERHEAD FACTOR: 1.314x

So the KT loader does not hold the checkpoint's packed bytes; it materialises per-expert host
buffers in an AMX-workable layout that costs ~31% more. That is a property of the offload path,
and it is the number that decides placement -- the earlier budget used the on-disk 63.28 GiB and
therefore UNDERESTIMATED the requirement by ~20 GiB.

### WHERE THAT LEAVES THE PLACEMENT

    RAM available                90.06 GiB
    experts, all 48 layers       83.16 GiB  (measured extrapolation)
    headroom for everything else  6.9 GiB

6.9 GiB must cover the process, KV pools, CUDA context and buffers. It is tight but it is not
obviously impossible -- the run was killed at the 70G cap with 8G of swap, so the cap itself
(not the machine) is what stopped it. The next attempt should raise MEMCAP (the scope had
memory.swap.max=8G and the system has 40G of swap) and see whether the remaining 8 layers
complete.

### NEXT ATTEMPT

    MEMCAP=88G PLE_BACKEND=file ISOLATE=1 ./scripts/run-qwen38-kt.sh

with the swap allowance raised too. If it still dies, the lever is `--kt-gpu-experts-ratio`
(put some experts on the 16 GB card) or fewer `--kt-cpuinfer` threads.

### ONE THING TO CLEAN UP FIRST

The PLE file is 48 G of disk and `/home` is now at **7.6 G free (99%)**. The file is reused
across restarts (deterministic name), so it should NOT be deleted -- but there is no longer room
for another 48 G if anything repopulates it. Watch this.
