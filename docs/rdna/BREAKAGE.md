# RDNA breakage inventory

Repos:

- This serving host: https://github.com/BlivionIaG/sglang-kt-rdna
- Sibling kernels (fa_rdna2, q_gemm_rdna2, gemv_f16_rdna2, moe_q_gemm_rdna2): https://github.com/BlivionIaG/ktransformers-rdna

This file lists every CUDA-only op or assumption this port hit, the verbatim in-source string, and a status. Status words are `fixed`, `fallback`, `gated`, `open`, or `needs the vllm-rdna Triton/ROCm toolchain pin`.

## What was checked on the agent VM

This VM has no AMD GPU, no `hipcc`, no `/opt/rocm`, and no Docker. Compiler stderr was not captured here. Nothing in this change was compile-checked or run on hardware. The GitHub Actions job `.github/workflows/rdna-sgl-kernel.yml` is the compile check: it cross-compiles `sgl-kernel` for gfx1030 and gfx1100 inside `rocm/pytorch:rocm7.14.1_ubuntu24.04_py3.12_pytorch_release_2.12.0`. A human runs `scripts/rdna/smoke_qwen3_moe.sh` on hardware.

`python3 -m unittest` of `sgl-kernel/tests/test_rdna_targets.py` does not need a GPU and was run on this VM. The GPU reference tests in `sgl-kernel/tests/test_rdna_ops_reference.py` skip when `torch.cuda.is_available()` is false.

NVIDIA `sgl-kernel/CMakeLists.txt` is unchanged.

## Status of the hybrid Triton kernels

Qwen3.5 (Gated DeltaNet hybrid) and Qwen4exp (QSA split-K full attention) are **not** gfx1030 blockers. Those Triton kernels already run on gfx1030 in opengfx1030/vllm-rdna with its pinned Triton build. Upstream Triton dropping RDNA2 (2026-09-30) only changes the **default full-attention** backend on gfx1030 to `torch_native`. It does not retire GDN or QSA.

Status for both kernels: **needs the vllm-rdna Triton/ROCm toolchain pin**.

For these models the port's job is SGLang's hybrid-state cache (Mamba / GDN recurrent state): keep that state on the GPU next to attention, and send only routed experts to kt-kernel (`--kt-num-gpu-experts 0` in phase 0). The recurrent state is not offloaded.

Qwen4exp's model class (`Qwen4ExpForConditionalGeneration` / `qwen_sparse_attention`) is absent from this fork. That is an open serving gap, not a missing-kernel gap. The QSA kernel status stays the pin above.

## Hybrid-state cache: what is CUDA-specific

`HybridReqToTokenPool` and the decode-side `HybridMambaDecodeReqToTokenPool` are constructed with `device=self.device` (`model_runner_kv_cache_mixin.py`). Attention KV and `MambaPool` therefore share the runner device. On ROCm, PyTorch's device string for the HIP device is `"cuda"`, and HIP tensors report `tensor.is_cuda == True`.

| Path | CUDA-specific piece | Status |
| --- | --- | --- |
| `MambaPool` conv and temporal buffers | `torch.zeros(..., device=device)`. Not hardcoded to a CUDA context. | fixed |
| Speculative intermediate SSM and conv-window caches | Were `device="cuda"`. They now use the pool `device`, so they stay on the same GPU as attention (and on `cuda:1` when that is the pool device). | fixed |
| `alloc` / `free` / `copy_from` / `fork_from` | Torch indexing on those tensors. | fixed |
| `mamba_radix_cache.py` | Uses `self.device`. No `device="cuda"` literal. | fixed |
| `mamba_ssm_dtype` | Taken from the HF config (`mamba_utils.py`). Qwen3.5 publishes `mamba_ssm_dtype: float32`. | fixed |
| `fused_mamba_state_scatter_with_mask` | Raises `fused_mamba_state_scatter_with_mask only supports CUDA tensors.` when `tensor.is_cuda` is false. HIP tensors pass that check, so the string is not hit on ROCm. gfx1030 then uses `torch_mamba_state_scatter_with_mask` and writes `dst` in place. gfx1100 and NVIDIA keep the Triton kernel. | fallback (gfx1030 torch copy) |
| GDN conv state update (`gdn_backend.py`) | Default import is Triton `causal_conv1d`. `is_cuda()` replaces `causal_conv1d_fn` with the CUDA `sgl_kernel` conv. HIP keeps Triton and writes `MambaPool.conv`. | needs the vllm-rdna Triton/ROCm toolchain pin |
| GDN temporal state (`gdn_triton.py`) | `TritonGDNKernel` calls FLA `fused_sigmoid_gating_delta_rule_update`, `chunk_gated_delta_rule`, and `fused_recurrent_gated_delta_rule_update` with `ssm_states` as `initial_state_source`. Imported whenever the process is not CPU. Default `--linear-attn-backend triton`. | needs the vllm-rdna Triton/ROCm toolchain pin |
| QSA split-K full attention | Not in this tree. Already runs on gfx1030 under the vllm-rdna pin. | needs the vllm-rdna Triton/ROCm toolchain pin |
| CuTe DSL GDN (`gdn_cutedsl.py`) | `GDNKernelDispatcher` raises `CuTe DSL backend requires CUDA` when the decode backend is cutedsl and `is_cuda()` is false. Not the default. | gated |
| Mamba2 mixer conv (`mamba.py`) | CUDA imports `sgl_kernel` causal conv. HIP imports Triton causal conv and aliases both the default and `_triton` names. Qwen3.5 does not run this mixer; it uses `mamba_v2_sharded_weight_loader` plus `RadixLinearAttention` / `Qwen3_5GatedDeltaNet`. | needs the vllm-rdna Triton/ROCm toolchain pin |
| Mamba2 scan ops (`mamba/ops/ssd_*.py`, `mamba_ssm.py`) | `with torch.cuda.device(...)`. HIP tensors still enter that API. Not the Qwen3.5 path. | needs the vllm-rdna Triton/ROCm toolchain pin |
| FLA platform flags (`fla/utils.py`) | Triton backend `hip` is remapped to torch device `"cuda"`. `is_nvidia_hopper` and TF32 flags stay false on AMD, so warp counts take the non-Hopper list. `layernorm_gated._get_sm_count` calls `torch.cuda.get_device_properties`, which exists for HIP tensors. | fixed (vendor check); kernel itself needs the pin |
| CUDA graph metadata (`hybrid_linear_attn_backend.py`, `hybrid_attn_backend.py`) | `init_cuda_graph_state` / capture / replay allocate index tensors on `self.device`. Phase 0 passes `--disable-cuda-graph`. HIP graph replay plus kt-kernel host callbacks is unverified. | gated |
| Mooncake custom mem pool | `maybe_init_custom_mem_pool` is on only when `SGLANG_MOONCAKE_CUSTOM_MEM_POOL` is set. It wraps allocation in `torch.cuda.use_mem_pool`. `get_contiguous_buf_infos` exposes conv/temporal pointers for RDMA (`disaggregation/mooncake/conn.py` `_send_mamba_state` and `_send_mamba_state_slice`, and the NIXL sender). Phase 0 does not enable disaggregation. | open |
| NSA fused KV+index store | `_can_fuse_kv_index_store` requires `_is_cuda`. MiniMax path, not GDN state. | open |
| Routed experts | kt-kernel. Phase 0: `--kt-num-gpu-experts 0`. GPU INT4 repack is CUDA-only (`gptq_marlin_repack is CUDA-only...`). | gated |

## DeepSeek-V4-Flash: genuine new-kernel gap (inventory only)

Not implemented in this change. The new kernels are sparse MLA, the C4/NSA indexer, and the GPU MXFP4 expert tile.

| Piece | Where | Status |
| --- | --- | --- |
| Sparse MLA + C4 indexer + NSA | `models/deepseek_v4.py`: `forward_c4_indexer`, `rms_normalize_triton`, `apply_rotary_emb_triton`, NSA `rotate_activation`. Attention backend name `compressed` maps to `DeepseekV4BackendRadix`. No sparse-MLA or indexer kernel is added here. | open |
| GPU MXFP4 expert tile | `--kt-method MXFP4` CPU experts live in kt-kernel (sibling repo). The GPU tile (`mxfp4_deepseek` / triton_kernels / marlin in `kt_ep_wrapper.py`) is CUDA. | open |
| `dsv4_norm_rope` | `fused_qknorm_rope_kernel.cu` uses CUDA headers (`cuda_fp8.h`, `cuda_runtime`) and a hardcoded warp of 32. It is not in the ROCm source list. The Python model currently calls Triton RMS and RoPE instead. | open |
| `deepseek_v4` fused hash top-k | `SGLANG_OPT_USE_FUSED_HASH_TOPK` JIT (`sglang.jit_kernel.deepseek_v4.hash_topk`) is skipped on HIP. Log: `deepseek_v4 hash_topk JIT is CUDA-only. Using the torch HashTopK path on ROCm.` `biased_topk_impl` is already torch. | fallback |
| FP8 layerwise expert transport | kt-kernel, not this repo. | open |

## Phase-0 ops (Qwen3-30B-A3B-GPTQ-Int4)

Smoke command is `scripts/rdna/smoke_qwen3_moe.sh`. gfx1100 uses `--attention-backend triton`. gfx1030 uses `--attention-backend torch_native`. Both set `SGLANG_USE_AITER=0`, `--disable-cuda-graph`, `--kt-method GPTQ_INT4`, `--kt-num-gpu-experts 0`.

| Op | What happened | Status |
| --- | --- | --- |
| Arch gate | Was gfx942/gfx950 only. `rdna_targets.py` accepts a single gfx1030 or gfx1100 module and refuses mixing them with each other or with CDNA. CMake fatal text: `Unsupported AMDGPU_TARGET '${AMDGPU_TARGETS}'. Expected 'gfx942', 'gfx950', 'gfx1030', or 'gfx1100'.` Python gate text: `Warning: Unsupported GPU architecture detected '<arch>'. Expected one of gfx942, gfx950, gfx1030, gfx1100.` | fixed |
| Wave32 | Host HIP compile used `WARP_SIZE 64` unless `__HIP_DEVICE_COMPILE__` and not `__GFX9__`. RDNA passes `-DSGL_RDNA_WAVE32` for host and device so both see `WARP_SIZE 32`. NVIDIA (`#ifndef USE_ROCM`) stays 32. | fixed |
| FP8 | RDNA has no FP8 units. Build passes `-DSGL_RDNA_NO_FP8` and does not define `HIP_FP8_TYPE_FNUZ` or `HIP_FP8_TYPE_E4M3`. `utils.h` then does not define `FP8_TYPE`. The leftover CDNA string, if neither macro is set, is `#error "fp8 is not supported in this processor (arch < gfx942)."`. gfx942 stays FNUZ, gfx950 stays E4M3. Software E4M3 is not enabled. | gated |
| Custom / quick / deterministic all-reduce | `quick_all_reduce_base.h` only defines MUBUF acquire/release for gfx942 and gfx908/gfx90a, and `kWavefront = 64`. Those sources are omitted on RDNA (`-DSGL_RDNA_NO_CUSTOM_AR`). No hipcc diagnostic was captured; the exclusion is source-level. TP uses RCCL through PyTorch's NCCL backend. | gated |
| TopK dynamic shared memory | gfx942, gfx1030, and gfx1100 use 48 KiB. gfx950 and a combined gfx942;gfx950 build keep 128 KiB. The old CMake branch read unset `AMDGPU_TARGET_ONE`, so every CMake build previously got 128 KiB. | fixed |
| silu_and_mul / gelu_and_mul / gelu_tanh | HIP import of `sgl_kernel` is try/except. `SiluAndMul`, `GeluAndMul`, and `QuickGELU` `forward_hip` call `forward_native` if the import failed. | fallback (native if the .so is absent); kernel is in the ROCm source list |
| topk_softmax / topk_sigmoid | Import is soft. `fused_topk` calls `fused_topk_torch_native` when the op is missing. `moe_fused_gate` stays CUDA (`if _is_cuda`); grouped top-k on HIP uses `biased_grouped_topk_impl`. | fallback |
| moe_align_block_size | Import is soft. `moe_align_block_size_torch` is the fallback and the reference. Expert slot is `topk_id + 1` so `-1` is slot 0; `expert_ids[block] = slot - 1`. Within-expert order is increasing token index in torch; the HIP kernel does not promise that order. Compare per-expert sets, padded count, and `expert_ids`. | fallback |
| RoPE | `pos_enc.cu` / `rotary_embedding` is in the ROCm source list. | fixed (compile list); runtime not run |
| rms_norm | `fused_add_rms_norm_kernel.cu` includes FlashInfer `norm.cuh` and is not in the ROCm source list. `common_extension_rocm.cc` does not register `rmsnorm`. RDNA layernorm skips AITER and vLLM custom ops and uses `forward_native`. | fallback |
| KV cache transfer | `kvcacheio/transfer.cu` is in the ROCm source list. | fixed (compile list); runtime not run |
| Grammar bitmask | `apply_token_bitmask_inplace_cuda.cu` is in the ROCm source list. | fixed (compile list); runtime not run |
| AITER | `_disable_aiter_on_rdna()` sets `SGLANG_USE_AITER=0` at import when a visible device is RDNA. `_handle_amd_specifics` repeats that and replaces an explicit `aiter` backend. | gated |
| gfx1030 default full attention | `rdna_default_attention_backend()` returns `torch_native`. This does not change `--linear-attn-backend`. Asking for Triton full attention logs that stock Triton needs the vllm-rdna pin. | fallback |
| gfx1100 default full attention | `triton`. | fixed (selection); runtime not run |
| fa_rdna2 / moe_q_gemm_rdna2 / q_gemm / gemv | Hooks only (`--attention-backend fa_rdna2`, `load_rdna_w4a16_moe_method`). Missing `kt_rdna` raises `NotImplementedError` pointing at the sibling repo. WMMA must live in a gfx1100-only object there, never in the gfx1030 sgl-kernel module. LDS per workgroup is 64 KiB. | open (sibling) |
| FlashInfer attention, FlashMLA, CUTLASS MLA | Not ported. | open |
| GGUF GPU dequant | Left on the CUDA path. | open |

## Qwen3.5 shape notes (not a phase-0 smoke)

Fetched HF configs, not executed:

- `Qwen/Qwen3.5-9B`: `Qwen3_5ForConditionalGeneration`, layer pattern 3× `linear_attention` + `full_attention`, 32 layers, `head_dim` 256, 16 heads / 4 KV heads, linear K/V head 128, `attn_output_gate`, partial RoPE 0.25, mrope, `mamba_ssm_dtype` float32. Dense.
- `Qwen/Qwen3.5-35B-A3B`: `Qwen3_5MoeForConditionalGeneration`, 40 layers, hidden 2048, `head_dim` 256, 16 heads / 2 KV heads, 256 experts, top 8, `moe_intermediate_size` 512. GPTQ-Int4 sibling: `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`.

Full-attention `head_dim` 256 is inside the sibling fa_rdna2 D=128/256 claim. Phase-0 smoke stays on Qwen3-30B-A3B (standard GQA), not Qwen3.5. `attn_output_gate` and partial mRoPE are extra assumptions the phase-0 GQA path does not cover.
