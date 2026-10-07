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
