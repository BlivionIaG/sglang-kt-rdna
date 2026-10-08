# What we changed, per project — and what is worth contributing upstream

Scope: branch `feat/qwen4-exp-support` on the fork `BlivionIaG/sglang-kt-rdna`, HEAD `80ca360f35`.
Base for comparison: `18c0de72e593` (upstream `main` at the time of the port).

**Read the diff with the base-forward excluded.** Of the 91 commits on this branch, `025b624476`
("move the shared `srt` base forward wholesale to upstream's generation") replaces **1,228 files**
because the fork's base predated upstream's. That is bulk synchronisation, not our work, and a
raw `git diff` shows 2,629 changed files. Everything after `025b624476` is ours:

    16 files changed, 1,861 insertions(+), 38 deletions(-)

---

## Our actual changes, by project

### A. `ktransformers` — nothing

We did not touch `kt-kernel` or any ktransformers source. Everything KT-related lives in the
*consuming* fork and is either a call into the existing `kt_kernel` binding or a restoration of
what the base-forward dropped. Worth stating plainly because it bounds the contribution: we are
not proposing kernel changes.

### B. `sglang` (glue for the KT hybrid path) — the bulk of our work

| file | what | ours? |
|---|---|---|
| `layers/moe/kt_ep_wrapper.py` | `chunked_prefill_size or 8192` — the newer base computes it in `arg_groups/memory_hook.py`, so it is `None` and `kt_kernel`'s typed C++ `max_len: int` rejects it | **yes** |
| `layers/moe/utils.py` | restored `DISABLE_KT_EP_WRAPPER` + `is_kt_ep_wrapper_disabled()` + `speculative_kt_ep_disabled_context()` | **yes** (restoration) |
| `eplb/expert_distribution.py` | restored `ExpertDistributionRecorder.on_gpu_expert_mask` — the KT wrapper calls it and the newer base's ABC does not declare it, so the no-op subclass raised `AttributeError` | **yes** (restoration) |
| `configs/glm5_next.py` | restored `GLM5_NEXT_SUPPORTED_TP_SIZES` (5 lines), consumed by the KT TP-size check | **yes** (restoration) |
| `server_args.py` | +130 lines: the 16 `kt_*` fields and 13 `--kt-*` CLI args the base-forward dropped; `ServerArgs.get_hf_config()` | **yes** (restoration) |
| `environ.py` | +191 lines: 16 env names the newer base references that the fork's merged `environ.py` lacked (`EXA_API_KEY` alone killed the FastAPI lifespan) | **yes** |
| `models/qwen4_exp.py` | 72 insertions: import-path fixes for the forwarded base | **yes** |
| `pyproject.toml` | dependency pins: torch 2.14.1+cu130, sglang-kernel 0.4.9 (replacing sgl-kernel 0.3.21, whose wheel links torch 2.9), flashinfer 0.7.0.post1 | **yes** |

### C. Two genuine bug fixes on the GPU-expert path — the strongest contribution

`layers/quantization/modelopt_quant.py`, 34 insertions in commit `c5d1367be7`:

1. **Width mismatch.** `w13_input_scale` is created `(num_experts, num_shards)` and marked
   `_sglang_require_global_experts`, so it stays at *global* width, while `w13_weight_scale_2` is
   sized from `num_local_experts` — which the KT wrapper overwrites to the GPU subset. The `else`
   branch at line 2730 multiplied 512 x 384 and raised
   `The size of tensor a (512) must match the size of tensor b (384)`. Fixed by reducing to the
   local window via the file's own `_input_scale_to_local_experts` helper (also applied to
   `w2_input_scale`).
2. **`auto` resolved two different ways.** `__init__` read the global predicate (so
   `enable_flashinfer_trtllm_moe` was False and `process_weights_after_loading` skipped creating
   `g1_scale_c`), while `create_moe_runner` resolved the same `auto` to `FLASHINFER_TRTLLM` (so
   `apply()` demanded the attribute). Fixed by resolving `auto` identically in both places.

Both are **reachable only at `--kt-num-gpu-experts > 0`**, because `kt_ep_wrapper.py:5989`
short-circuits at `g=0`. That is why nobody had hit them.

### D. Our own docs and tooling — not upstream material

`docs/qwen38-*` (5 files), `docs/running-qwen38-flash-next-on-ktransformers.md`, `RESUME-qwen38.md`,
`scripts/run-qwen38-kt.sh`. Host-specific by design: they hardcode `par1-llm1` paths and the model
location.

---

## Upstreamability, sorted by value

### Contribute to `sglang` (fork: `sgl-project/sglang` or `kvcache-ai/sglang`)

**1. The two `modelopt_quant.py` fixes — highest value, best shape.** They are:
   - upstream code (not KT-specific),
   - small and self-contained (34 lines, two hunks),
   - each carries a clear failure signature and a stated mechanism,
   - and they fix a naming/consistency bug (`auto` resolved differently in two places) that a
     maintainer can verify by reading, not by owning our hardware.
   The honest caveat: the reporter path is `--kt-num-gpu-experts`, so demonstrating them needs the
   KT fork. They are still correct `modelopt_quant` fixes on their own terms.

**2. `on_gpu_expert_mask` on the `ExpertDistributionRecorder` ABC — good value, small.** The base
   already simulates a callback the KT wrapper invokes; the ABC simply lacks it. Add the method to
   the abstract interface with a default no-op. Low risk, obviously correct.

**3. `chunked_prefill_size or 8192` — small but real.** A `None` reaching a typed C++ int is a
   latent bug for any consumer reading `server_args.chunked_prefill_size` after the memory hook was
   introduced. Better as a default in the memory hook than a local `or`, but the diagnosis is
   useful upstream.

**4. The 16 missing env names in `environ.py` — report, probably do not propose.** These look like
   fork-merge fallout rather than an upstream defect; the right fix is likely a rebase, not a patch.

### Report, not propose

**5. `auto` selects TRT-LLM on sm_120, which has no cubins for it.** `create_moe_runner` gates on
   `capability >= (10, 0)`, and sm_120 passes while the TRT-LLM FP4 MoE cubins stop at sm107. Any
   sm_120 user with NVFP4 MoE hits `The trtllm-gen batched GEMM cubin manifest contains no kernels
   runnable on sm120`. This deserves an upstream **issue** with the manifest evidence — the fix is
   a capability check or a clearer error, not our patch. It affects non-KT users too, which makes
   it the most broadly interesting thing we found.

**6. The mamba gate on hybrid models.** `kv_cache_configurator.py:2344` derives the concurrency
   ceiling from `max_mamba_cache_size // ratio` and raises when it reaches 0. On a 36-linear-layer
   hybrid the message suggests flags that may not exist in a given build. A better error
   (naming the derived value and the minimum viable `g`) would have saved this session hours.

### Do not upstream

- The `kt_*` surface restorations (B): these exist in the kt fork already. Upstreaming them would
  duplicate KT's own work.
- The dependency pins and everything in D: environment-specific.

---

## What we found that would change how others tune this

These are findings, not patches, but they are the part with the most reuse value:

- **`flashinfer_cutlass` is the only FP4 MoE backend that runs on sm_120**, and the
  `NotImplementedError` in `apply()` already names it. `auto` picks TRT-LLM (no sm120 cubins);
  `marlin` loads but dies in `sglang_kernel`'s `moe_sum_reduce` (sm80/sm90 cubins only).
- **The headroom costs more than the experts.** At equal MEMFRAC, 8 GPU experts per layer cost
  19,136 KV tokens, but dropping 0.95 -> 0.88 to survive the first real request cost 22,656.
- **The mamba state cache, not the KV pool, caps `g`** on this hybrid model.
- **`frequency` placement silently degrades to uniform** without an `init_expert_location` `.pt`,
  so the flag looks honoured while doing nothing.
- **`--kt-num-gpu-experts > 0` is nondeterministically unstable on this stack** — 1 of 6 benchmark
  attempts completed. Whatever the eventual fix, this belongs in any "is it ready" discussion.

---

## How to verify any of this

    git -C <fork> diff --stat 025b624476..HEAD             # our work: 16 files, +1861/-38
    git -C <fork> diff 025b624476..HEAD --stat             # the 16-file inventory above
    git -C <fork> show c5d1367be7                          # the two bug fixes in full
