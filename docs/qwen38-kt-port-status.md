# Qwen3.8-Flash-Next-NVFP4 through the kt sglang fork -- port status

Branch `feat/qwen4-exp-support` on `BlivionIaG/sglang-kt-rdna`.
Rollback tag for the base-forward: `pre-base-forward-2026-10-07`.

## WHERE IT STANDS

The port works. On the last good run the model loaded through the kt CPU expert path:

    [NVFP4SafeTensorLoader] Loaded 512 experts from model.language_model.layers.0.mlp.experts
    Generated KT GPU experts masks using 'uniform' strategy: 48 MoE layers x 512 experts
    [KT] Created shared staging buffer: 40.0 MiB (shape=torch.Size([8192, 2560]), bfloat16)
    CPUInfer: Hello / WorkerPool 1 subpools, [numa:threads][0:16]
    ===========In NumaPool============   In Numa Worker Pool at NUMA 0, 16 threads
    [KT] Recreated NativeMoEWrapper loader for layer 0..29 (~300ms each)

It then dies during the per-layer weight walk with SIGKILL from the kernel OOM killer.

## WHAT WAS REQUIRED TO GET THERE

1. **`srt/` moved forward wholesale** to upstream's generation. The fork's base predated it and
   the newer modules carried **514 modules on import cycles**; zero of the cycle-closing imports
   were safely deferrable, so patching could not converge. One `git checkout sglproject/main --
   python/sglang/srt/`, then the 201 fork-only files plus the kt-specific ones restored:
   `nsa/`, `moe/kt_ep_wrapper.py`, `models/qwen4_exp.py`, `environ.py`, `configs/qwen4_exp.py`,
   `utils/hf_transformers_utils.py`. `modelopt_quant.py` deliberately NOT restored -- upstream's
   is a superset and its `_resolve_quant_algo` is stronger than the fork's local version.

2. **The dependency generation moved with it.** torch 2.14.1+cu130, flashinfer_python/cubin/
   jit_cache 0.7.0.post1, transformers 5.17.0, xgrammar 0.2.7, **sglang-kernel 0.4.9** replacing
   sgl-kernel 0.3.21. The old sgl-kernel wheel links against torch 2.9 and its
   `common_ops.abi3.so` cannot import under 2.14 (`undefined symbol
   c10_cuda_check_implementation`).

3. **The kt_ ServerArgs surface restored**, which the base-forward removed entirely: 16 `kt_*`
   fields, 13 `--kt-*` CLI arguments, `ServerArgs.get_hf_config()`, the `DISABLE_KT_EP_WRAPPER`
   global with its predicate and context manager, and `GLM5_NEXT_SUPPORTED_TP_SIZES`.
   Two subtleties worth keeping:
   - the kt CLI args must use `default=_declared_default("field")`, NOT
     `default=ServerArgs.field` -- this base uses `__slots__`, so the class attribute is a
     `member_descriptor` and argparse would register the descriptor instead of the value;
   - `dataclasses.field(...)` must not be used for the kt fields -- this base's
     `record_fields` reads the CLASS attribute, so the `Field` object leaks into
     `resolved_dict()` and dies with "cannot pickle 'mappingproxy' object".

## THE OPEN QUESTION -- do not guess at it

Three explanations for the per-layer death have been offered and two are RETRACTED:

- "the checkpoint cannot fit this host (123 GB vs 92 GB)" -- **retracted**. An RSS sampler
  showed the process walking layers 9-27 at **10-17 GB resident** with 86 GB free.
- "`vm.overcommit_memory=0` refuses a 342 GB virtual reservation" -- **retracted**. Setting
  `overcommit_memory=1` did not stop the kill.
- **`LimitMEMLOCK=8388608` (8 MiB)** -- **untested**. The kt path pins host buffers via
  `cudaHostRegister` (kt_ep_wrapper.py:1197), which counts against the locked-memory limit,
  and `systemctl show user@1000.service -p LimitMEMLOCK` reports 8388608. The run that was
  testing it took the host down before producing a verdict.

**Column trap when reading the kernel OOM table:** the header is
`total_vm rss rss_anon rss_file rss_shmem pgtables_bytes swapents oom_score_adj name`, so
`$(NF-8)` is total_vm and `$(NF-7)` is rss. At the 13:06 kill the scheduler had
`total_vm` = 330 GB but `rss` = **5 GB**. Reading the first big number as memory usage is
what produced the retracted overcommit diagnosis.

## HOW TO RUN IT

`scripts/run-qwen38-kt.sh` in this repo. Run it twice -- once plain, once with
`ulimit -l unlimited` in the launching shell -- and compare which layer each reaches. That
settles the memlock question. **Do not report a root cause until one of those completes.**

## HOST NOTE (2026-10-07)

par1-llm1 wedged during the memlock test: ICMP fine, ARP REACHABLE, `:22` accepts TCP but
never sends a banner, all service ports closed. Confirmed from two vantages (this workstation
and `par1-cssec1` via `chenco_adm@100.122.94.247`). No BMC exists (623/5900 closed; .122 is a
Hue bridge). Recovery required a power cycle.

## THE BLOCKER, AS UPSTREAM DOCUMENTS IT

`python/sglang/srt/models/qwen4_exp_ple_table.py` says it plainly in its module docstring:

> ``file`` -- A file-backed, shared ``mmap`` of a sparse file under ``--ple-offload-dir``.
> Meant for unified-memory parts (GB10 / DGX Spark and similar), where pinned host memory
> comes out of the *same* pool as the model weights and ``pinned`` therefore frees nothing:
> **Qwen3.8-Flash-Next is 126.0 GiB of weights on a 121.63 GiB box and does not boot with
> ``pinned``.**

So the `pinned` backend this run uses is documented by upstream as not booting this model on
a box of this size -- and it is what wedged par1-llm1. The two alternatives:

| backend | requirement | this host |
|---|---|---|
| `pinned` | host RAM for 126 GiB of weights + 47.7 GiB PLE | 92 GB RAM -- documented as not booting |
| `file` | `cudaDevAttrPageableMemoryAccessUsesHostPageTables` (unified memory: GB10 / DGX Spark) | reports **False** -- correctly refused |
| neither | the 47.69 GiB PLE table must be GPU-resident | 16 GB card -- `Tried to allocate 47.69 GiB` |

**Attribute-number trap:** the library's constant is
`_CUDA_DEV_ATTR_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100` (`ple_table.py:54`). An
earlier hand-rolled probe of mine read attribute **129** and printed `1`, which briefly looked
like the capability was available. It is not; the library's own
`device_uses_host_page_tables(0)` returns `False`. Never hand-pick an enum value -- call the
library's predicate.

## CONCLUSION

Both PLE backends are unavailable on this hardware, and the third option does not fit on the
GPU. That is an environment/checkpoint mismatch, not a defect in the port: the code path is
complete and pushed. Resolving it needs either a unified-memory device (so `file` works) or a
host large enough for the weights plus the pinned table. `scripts/run-qwen38-kt.sh` is ready
for either, and `ISOLATE=1` keeps a future attempt from wedging the machine.

## UPDATE: the `file` backend now has a software route on this GPU

The table above said both backends were closed. That was true of the code AS UPSTREAM SHIPS
IT, but not of the design: the only thing binding `file` to unified memory is *where the row
read happens*. The default gather is a Triton kernel that dereferences the host pointer from
the device (`tl.load(weight_ptr + ...)`), which needs
`cudaDevAttrPageableMemoryAccessUsesHostPageTables`. The sparse mmap itself
(`torch.from_file(..., shared=True)` + `MADV_RANDOM` + WILLNEED prefetch + an RSS trimmer)
is plain Linux and works anywhere.

`SGLANG_QWEN4_PLE_HOST_SIDE_GATHER=1` gathers the same rows on the CPU and copies them over,
removing that dependency. `scripts/run-qwen38-kt.sh` wires both together:

    PLE_BACKEND=file ./scripts/run-qwen38-kt.sh

which sets `SGLANG_QWEN4_PLE_HOST_SIDE_GATHER=1` and
`SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK=1` and selects
`--ple-offload-embedding --ple-offload-backend file`.

Cost, stated plainly: this is a correctness path, not a fast one. It adds a host gather plus
one H2D copy per call. The volume is small by construction -- 16 rows per token (8 two-gram +
8 three-gram heads) of `head_dim_per_ngram` values, about 2.5 KB in fp8 -- but it is still a
per-step cost with a possible sync.

Verified: the masking arithmetic matches `_gather_ple_embedding_from_pinned_kernel` over
adversarial ids (in-range, below the shard, above it, both bounds) -- 7/7 rows identical.
NOT verified on hardware: par1-llm1 is wedged, so the torch tensor ops (indexing an fp8 cpu
table with a long tensor, then `.to(bfloat16)`) have not been exercised against real weights.
Test that first when the host returns.

## HOST RELIABILITY (2026-10-07) -- read before scheduling any long run

par1-llm1 flaps. Three wedges in one afternoon, and the recovery windows are SHORT:

    1. ~13:10 wedged  ->  ~13:34 recovered on its own (uptime unchanged: 41 days, no reboot)
    2. ~13:36 wedged  ->  briefly answered again
    3. moments later  ->  wedged again and stayed down

Signature, confirmed from TWO vantages (this workstation and `chenco_adm@par1-cssec1`, which
now trusts llm1's host key):

    ICMP            -> 0% loss, normal RTT        kernel + NIC healthy
    ARP / ip neigh  -> REACHABLE, MAC unchanged   no reboot happened
    TCP :22         -> OPEN                       sshd is listening
    SSH banner      -> NEVER ARRIVES              the forked child cannot run
    all service ports (8208-8214, 9100, 11434) -> closed

So: nothing on the box can fork, while the network stack and kernel are fine. **Practical
consequences:**
- Do NOT hold a long foreground SSH call -- it will be cut off mid-run.
- Launch work under `tmux` on the host and read the log later; that survived the wedge before.
- A recovery window is short: the first command issued must be the one you actually want.
- The `file` PLE route's cost matters more now, because a wedging host is the real constraint.
