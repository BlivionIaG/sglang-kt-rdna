# SERVED: Qwen3.8-Flash-Next-NVFP4 on the kt sglang fork

    The server is fired up and ready to roll!
    GET /health HTTP/1.1" 200 OK

    Q: What is 17+25? Reply with just the number.
    A: 42
    Q: The capital of France is
    A: ... Paris

## The checkpoint was NOT modified

Served from `/home/kletorch/models/qwen38-stage` as shipped. The safetensors mtimes are
`2026-10-07 02:01`-`03:17`, i.e. the download, which predates all of this work; `find -newermt
'2026-10-07 00:00'` returns only HuggingFace's own download-metadata files. No tensor was
rewritten, requantised, or dequantised on disk. (`git status` on the repo is clean and the
branch carries the numbers.)

## The placement, all three tiers confirmed by observation

    GPU   12.6 GiB used of 15.5   weights for the non-expert layers + KV cache
    RAM   87 GiB used of 92       48 MoE layers x 512 experts, kt's AMX path
    disk  47.7 GiB                the PLE table as a sparse mmap, RSS trimmed
                                  "trimmed resident set 11.7 -> 0.1 GiB (budget 8.0 GiB)"

**MoE-to-disk was not needed.** The experts fit in RAM at 83.16 GiB measured in-kernel; the
thing that had to leave memory was the PLE n-gram embedding, and that is exactly what
`--ple-offload-backend file` exists for.

## What made it work, in dependency order

1. **`srt/` moved forward wholesale.** 514 modules were on import cycles and none of the
   cycle-closing imports were deferrable, so patching could not converge. Then the 201
   fork-only files and the kt-specific ones were restored.
2. **The dependency generation** moved with it: torch 2.14.1+cu130, flashinfer 0.7.0.post1
   (cubin and jit-cache live on https://flashinfer.ai/whl, not PyPI above 0.6.13),
   transformers 5.17.0, `sglang-kernel` 0.4.9 replacing `sgl-kernel` 0.3.21.
3. **The kt_ ServerArgs surface restored**: 16 fields, 13 `--kt-*` CLI args,
   `get_hf_config()`. With two subtleties -- the CLI defaults must come from
   `_declared_default(...)` because this base uses `__slots__`, and the kt fields must not use
   `dataclasses.field(...)` because `record_fields` reads the class attribute.
4. **The PLE route.** Upstream's own docstring says `pinned` "does not boot" this model at this
   size (126 GiB of weights on a 92 GB box) -- and it wedged the host three times.
   `--ple-offload-backend file` needs a unified-memory device, so
   `SGLANG_QWEN4_PLE_HOST_SIDE_GATHER=1` gathers the rows on the CPU instead, removing that
   dependency. The two flags must go together: the device check exists to stop the Triton
   kernel reading garbage.
5. **`ExpertDistributionRecorder.on_gpu_expert_mask`** restored -- `kt_ep_wrapper.py:5734` calls
   a callback that exists only in the fork.
6. **16 env names** the newer base references but the fork's merged `environ.py` lacked.
   Found by diffing every `envs.<NAME>` reference against the declarations rather than fixing
   the one that crashed (`EXA_API_KEY` killed the FastAPI lifespan).
7. **`mem-fraction-static` past the computed KV floor** (0.6911 measured by the configurator
   after the weights land). `MEMFRAC` is now a script parameter.

## And the host

It flapped three times before `ISOLATE=1` was added; after that, every failure was confined to
the scope and the host stayed up. `MEMCAP`/`MEMSWAP` are parameters because the 70 G cap
(successfully) killed a run at layer 40 -- the cap was the limit, not the machine.

## Reproduce

    PLE_BACKEND=file ISOLATE=1 MEMFRAC=0.75 MEMCAP=200G \
      ./scripts/run-qwen38-kt.sh --host 0.0.0.0 --port 8210
