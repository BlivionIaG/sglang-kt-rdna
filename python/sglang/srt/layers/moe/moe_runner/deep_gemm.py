from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import einops
import torch

from sglang.srt.environ import envs, is_large_dummy_model
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_post_permute,
    register_pre_permute,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.utils import (
    ceil_div,
    dispose_tensor,
    get_bool_env_var,
    is_cuda,
    is_hip,
    is_npu,
)
from sglang.srt.utils.offloader import get_offloader
from typing import TYPE_CHECKING, Any, List, Optional, Tuple
from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import MoeRunnerBackend, get_moe_a2a_backend
import triton
import triton.language as tl

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLCombineInput,
        DeepEPLLDispatchOutput,
        DeepEPNormalCombineInput,
        DeepEPNormalDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )

_is_hip = is_hip()
_is_npu = is_npu()
_is_cuda = is_cuda()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

# Imported only for the SGLANG_OPT_FIX_MEGA_MOE_MEMORY=False fallback path.
if not (_is_npu or _is_hip):
    from sgl_kernel import silu_and_mul as _legacy_silu_and_mul
else:
    _legacy_silu_and_mul = None


_MASKED_GEMM_FAST_ACT = get_bool_env_var("SGLANG_MASKED_GEMM_FAST_ACT")
_DEEPGEMM_ON_H20 = get_bool_env_var("SGLANG_DEEPGEMM_ON_H20")


# TODO(kaixih@nvidia): ideally we should merge this logic into
# `fill_gateup_input_triton_kernel` to directly generate e8m0 scale.
@torch.compile(disable=_is_hip or _is_npu)
def _cast_to_e8m0_with_rounding_up(x: torch.Tensor) -> torch.Tensor:
    temp = x.to(torch.float32).view(torch.int32)
    exp = torch.bitwise_right_shift(temp, 23)
    mant = torch.bitwise_and(temp, 0x7FFFFF)
    is_ru = torch.logical_and(
        torch.logical_and((mant > 0), (exp != 0xFE)),
        ~torch.logical_and((exp == 0), (mant <= 0x400000)),
    )
    exp = torch.where(is_ru, exp + 1, exp)
    new_x = exp.to(torch.uint8).view(torch.int)
    return new_x.transpose(1, 2).contiguous().transpose(1, 2)


def copy_list_to_gpu_no_ce(arr: List[int]):
    from sgl_kernel.elementwise import copy_to_gpu_no_ce

    tensor_cpu = torch.tensor(arr, dtype=torch.int32, device="cpu")
    tensor_gpu = torch.empty_like(tensor_cpu, device="cuda")
    copy_to_gpu_no_ce(tensor_cpu, tensor_gpu)
    return tensor_gpu


@dataclass
class DeepGemmRunnerInput(RunnerInput):
    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor
    use_masked_gemm: bool
    masked_m: Optional[torch.Tensor] = None
    expected_m: Optional[int] = None
    m_indices: Optional[torch.Tensor] = None

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.DEEP_GEMM


@dataclass
class DeepGemmRunnerOutput(RunnerOutput):
    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.DEEP_GEMM


@dataclass
class DeepGemmMoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    use_fp8: bool
    w13_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None
    block_shape: Optional[List[int]] = None


class DeepGemmRunnerCore(MoeRunnerCore):
    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)
        assert self.config.activation == "silu"
        assert self.config.is_gated
        self.swiglu_limit = self.config.swiglu_limit
        self.use_swizzle = False
        if envs.SGLANG_OPT_FIX_MEGA_MOE_MEMORY.get():
            assert envs.SGLANG_OPT_SWIGLU_CLAMP_FUSION.get()
            assert envs.SGLANG_OPT_USE_JIT_EP_ACTIVATION.get()
            assert envs.SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE.get()
            self.use_swizzle = True

    def run(
        self,
        runner_input: DeepGemmRunnerInput,
        quant_info: DeepGemmMoeQuantInfo,
        running_state: dict,
    ) -> DeepGemmRunnerOutput:
        if not runner_input.use_masked_gemm:
            hidden_states = self._run_contiguous_gemm(
                runner_input, quant_info, running_state
            )
        else:
            hidden_states = self._run_masked_gemm(
                runner_input, quant_info, running_state
            )
        return DeepGemmRunnerOutput(hidden_states=hidden_states)

    def _run_contiguous_gemm(
        self,
        runner_input: DeepGemmRunnerInput,
        quant_info: DeepGemmMoeQuantInfo,
        running_state: dict,
    ) -> torch.Tensor:
        from sglang.jit_kernel.deepseek_v4 import silu_and_mul_contig_post_quant
        from sglang.srt.layers.moe.ep_moe.kernels import tma_align_input_scale
        from sglang.srt.layers.quantization.fp8_kernel import (
            create_per_token_group_quant_fp8_output_scale,
        )

        hidden_states = runner_input.hidden_states
        hidden_states_scale = runner_input.hidden_states_scale
        all_tokens = running_state["all_tokens"]
        hidden_states_device = running_state["hidden_states_device"]
        hidden_states_dtype = running_state["hidden_states_dtype"]
        hidden_states_shape = running_state["hidden_states_shape"]
        m_indices = runner_input.m_indices

        N = quant_info.w13_weight.size(1)
        K = hidden_states_shape[1]
        scale_block_size = 128

        w13_weight_fp8 = (
            quant_info.w13_weight,
            quant_info.w13_scale,
        )
        w2_weight_fp8 = (quant_info.w2_weight, quant_info.w2_scale)

        gateup_output = torch.empty(
            (all_tokens, N),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        if not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
            hidden_states_scale = tma_align_input_scale(hidden_states_scale)
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
            (hidden_states, hidden_states_scale),
            w13_weight_fp8,
            gateup_output,
            m_indices,
        )

        dispose_tensor(hidden_states)
        dispose_tensor(hidden_states_scale)

        if envs.SGLANG_OPT_FIX_MEGA_MOE_MEMORY.get():
            is_2604b = envs.SGLANG_DSV4_2604_SUBMODE.get() == "2604B"
            swiglu_limit_arg: Optional[float] = None
            if is_2604b:
                swiglu_limit_arg = self.swiglu_limit

            down_input_fp8 = torch.empty(
                (all_tokens, N // 2),
                device=gateup_output.device,
                dtype=torch.float8_e4m3fn,
            )
            down_input_scale = create_per_token_group_quant_fp8_output_scale(
                x_shape=(all_tokens, N // 2),
                device=gateup_output.device,
                group_size=scale_block_size,
                column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            )
            silu_and_mul_contig_post_quant(
                input=gateup_output,
                output=down_input_fp8,
                output_scale=down_input_scale,
                quant_group_size=scale_block_size,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                transposed=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                swiglu_limit=swiglu_limit_arg,
                swizzle=self.use_swizzle,
            )
            del gateup_output
        else:
            # Hacky byte-equal fallback that reproduces the optimize-branch
            # code path exactly: bf16 silu_and_mul then a separate per-token
            # group fp8 quant. Kept behind the mega-moe-memory flag.
            from sglang.srt.layers.quantization.fp8_kernel import (
                sglang_per_token_group_quant_fp8,
            )

            if envs.SGLANG_DSV4_2604_SUBMODE.get() == "2604B":
                from sglang.srt.debug_utils.deepseek_v4_debug_utils import (
                    deepseek_v4_moe_code_path_checker,
                )

                gateup_output = _apply_swiglu_limit(
                    gateup_output, swiglu_limit=self.swiglu_limit
                )
                deepseek_v4_moe_code_path_checker.observed += 1

            down_input = torch.empty(
                (all_tokens, N // 2),
                device=gateup_output.device,
                dtype=torch.bfloat16,
            )
            _legacy_silu_and_mul(gateup_output.view(-1, N), down_input)
            del gateup_output

            down_input_fp8, down_input_scale = sglang_per_token_group_quant_fp8(
                down_input,
                scale_block_size,
                column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            )
            del down_input

        down_output = torch.empty(
            (all_tokens, K),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        if not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
            down_input_scale = tma_align_input_scale(down_input_scale)

        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
            (down_input_fp8, down_input_scale),
            w2_weight_fp8,
            down_output,
            m_indices,
        )

        return down_output

    def _run_masked_gemm(
        self,
        runner_input: DeepGemmRunnerInput,
        quant_info: DeepGemmMoeQuantInfo,
        running_state: dict,
    ) -> torch.Tensor:
        from sglang.srt.layers import deep_gemm_wrapper

        hidden_states = runner_input.hidden_states
        hidden_states_scale = runner_input.hidden_states_scale
        masked_m = runner_input.masked_m
        expected_m = runner_input.expected_m

        w13_weight = quant_info.w13_weight
        w2_weight = quant_info.w2_weight
        w13_scale = quant_info.w13_scale
        w2_scale = quant_info.w2_scale

        hidden_states_device = running_state["hidden_states_device"]

        # GroupGemm-0
        if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
            if hidden_states_scale.dtype != torch.int:
                b, s_mn, s_k = hidden_states_scale.shape
                assert (
                    s_mn % 4 == 0 and s_k % 4 == 0
                ), f"scales must be aligned to 4, but got ({b}, {s_mn}, {s_k})"
                hidden_states_scale = _cast_to_e8m0_with_rounding_up(
                    hidden_states_scale
                )
        else:
            hidden_states_scale = deep_gemm_wrapper.get_mn_major_tma_aligned_tensor(
                hidden_states_scale
            )

        num_groups, m, k = hidden_states.shape
        n = w13_weight.size(1)
        gateup_output = torch.empty(
            (num_groups, m, n), device=hidden_states_device, dtype=torch.bfloat16
        )
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
            (hidden_states, hidden_states_scale),
            (w13_weight, w13_scale),
            gateup_output,
            masked_m,
            expected_m,
        )
        dispose_tensor(hidden_states)
        dispose_tensor(hidden_states_scale)

        is_2604b = envs.SGLANG_DSV4_2604_SUBMODE.get() == "2604B"
        assert is_2604b == (
            self.swiglu_limit is not None
        ), f"swiglu_limit must be non-None iff submode=2604B (got submode={envs.SGLANG_DSV4_2604_SUBMODE.get()!r}, swiglu_limit={self.swiglu_limit!r})"
        swiglu_limit_arg: Optional[float] = None
        if is_2604b:
            assert (
                not _MASKED_GEMM_FAST_ACT
            ), "DSV4 2604 submode 2604B does not support SGLANG_MASKED_GEMM_FAST_ACT"
            assert (
                envs.SGLANG_OPT_USE_JIT_EP_ACTIVATION.get()
            ), "DSV4 2604 submode 2604B requires SGLANG_OPT_USE_JIT_EP_ACTIVATION=True"

            if envs.SGLANG_OPT_SWIGLU_CLAMP_FUSION.get():
                swiglu_limit_arg = self.swiglu_limit
            else:
                from sglang.srt.debug_utils.deepseek_v4_debug_utils import (
                    deepseek_v4_moe_code_path_checker,
                )

                gateup_output = einops.rearrange(
                    gateup_output, "grp tok hidden -> (grp tok) hidden"
                )
                gateup_output = _apply_swiglu_limit(
                    gateup_output, swiglu_limit=self.swiglu_limit
                )
                gateup_output = einops.rearrange(
                    gateup_output, "(grp tok) hidden -> grp tok hidden", grp=num_groups
                )
                deepseek_v4_moe_code_path_checker.observed += 1

        # Act
        down_input, down_input_scale = _varlen_deep_gemm_silu_mul_quant(
            gateup_output,
            masked_m,
            group_size=128,
            topk=self.config.top_k,
            swiglu_limit=swiglu_limit_arg,
            swizzle=self.use_swizzle,
        )
        del gateup_output

        # GroupGemm-1
        n = w2_weight.shape[1]

        if not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
            down_input_scale = deep_gemm_wrapper.get_mn_major_tma_aligned_tensor(
                down_input_scale
            )

        down_output = torch.empty(
            (num_groups, m, n), device=hidden_states_device, dtype=torch.bfloat16
        )

        down_gemm_overlap_args = running_state.get("down_gemm_overlap_args", None)
        if down_gemm_overlap_args is None:
            gemm_overlap_args_dict = {}
        else:
            down_gemm_overlap_args.start_event.record()
            max_block_n = (
                160 if (_DEEPGEMM_ON_H20 and runner_input.expected_m <= 64) else 256
            )
            gemm_overlap_args_dict = {
                "overlap_args": down_gemm_overlap_args,
                "max_block_n": max_block_n,
            }

        deep_gemm_return_value = deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
            (down_input, down_input_scale),
            (w2_weight, w2_scale),
            down_output,
            masked_m,
            expected_m,
            **gemm_overlap_args_dict,
        )
        meta_overlap_args = running_state.get("meta_overlap_args", None)
        if meta_overlap_args is not None:
            block_m, threshold = deep_gemm_return_value
            meta_overlap_args["block_m"] = block_m
            meta_overlap_args["threshold"] = threshold

        return down_output

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.DEEP_GEMM


@register_pre_permute("standard", "deep_gemm")
def pre_permute_standard_to_deep_gemm(
    dispatch_output: StandardDispatchOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepGemmRunnerInput:
    from sglang.srt.layers.moe.ep_moe.kernels import moe_ep_deepgemm_preprocess

    hidden_states, topk_output = (
        dispatch_output.hidden_states,
        dispatch_output.topk_output,
    )
    topk_weights, topk_ids, _ = topk_output

    hidden_states_shape = hidden_states.shape
    hidden_states_dtype = hidden_states.dtype
    hidden_states_device = hidden_states.device
    hidden_states_ref = hidden_states

    topk_weights, topk_ids = topk_weights, topk_ids

    # PreReorder
    masked_m, expected_m, src2dst, hidden_states, hidden_states_scale = (
        moe_ep_deepgemm_preprocess(
            topk_ids,
            runner_config.num_local_experts,
            hidden_states,
            runner_config.top_k,
            quant_info.block_shape,
        )
    )

    dispose_tensor(hidden_states_ref)

    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights
    running_state["hidden_states_shape"] = hidden_states_shape
    running_state["hidden_states_dtype"] = hidden_states_dtype
    running_state["hidden_states_device"] = hidden_states_device
    running_state["src2dst"] = src2dst

    return DeepGemmRunnerInput(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        use_masked_gemm=True,
        masked_m=masked_m,
        expected_m=expected_m,
    )


@register_post_permute("deep_gemm", "standard")
def post_permute_deep_gemm_to_standard(
    runner_output: DeepGemmRunnerOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.ep_moe.kernels import post_reorder_triton_kernel
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    hidden_states_shape = running_state["hidden_states_shape"]
    hidden_states_dtype = running_state["hidden_states_dtype"]
    hidden_states_device = running_state["hidden_states_device"]
    src2dst = running_state["src2dst"]
    topk_ids = running_state["topk_ids"]
    topk_weights = running_state["topk_weights"]

    output = torch.empty(
        hidden_states_shape, dtype=hidden_states_dtype, device=hidden_states_device
    )
    post_reorder_triton_kernel[(hidden_states_shape[0],)](
        runner_output.hidden_states,
        output,
        src2dst,
        topk_ids,
        topk_weights,
        runner_config.top_k,
        hidden_states_shape[1],
        BLOCK_SIZE=512,
    )

    dispose_tensor(runner_output.hidden_states)

    if runner_config.routed_scaling_factor is not None:
        output *= runner_config.routed_scaling_factor

    return StandardCombineInput(
        hidden_states=output,
    )


@register_pre_permute("deepep_ll", "deep_gemm")
def pre_permute_deepep_ll_to_deep_gemm(
    dispatch_output: DeepEPLLDispatchOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepGemmRunnerInput:
    hidden_states, hidden_states_scale, topk_ids, topk_weights, masked_m, expected_m = (
        dispatch_output
    )

    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights
    running_state["hidden_states_shape"] = hidden_states.shape
    running_state["hidden_states_dtype"] = hidden_states.dtype
    running_state["hidden_states_device"] = hidden_states.device

    return DeepGemmRunnerInput(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        use_masked_gemm=True,
        masked_m=masked_m,
        expected_m=expected_m,
    )


@register_post_permute("deep_gemm", "deepep_ll")
def post_permute_deep_gemm_to_deepep_ll(
    runner_output: DeepGemmRunnerOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepEPLLCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLCombineInput

    return DeepEPLLCombineInput(
        hidden_states=runner_output.hidden_states,
        topk_ids=running_state["topk_ids"],
        topk_weights=running_state["topk_weights"],
    )


@register_pre_permute("deepep_normal", "deep_gemm")
def pre_permute_deepep_normal_to_deep_gemm(
    dispatch_output: DeepEPNormalDispatchOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepGemmRunnerInput:
    from sglang.srt.layers.moe.ep_moe.kernels import ep_scatter

    (
        hidden_states,
        hidden_states_scale,
        topk_ids,
        topk_weights,
        num_recv_tokens_per_expert,
    ) = dispatch_output
    assert runner_config.activation == "silu"

    all_tokens = sum(num_recv_tokens_per_expert)
    running_state["all_tokens"] = all_tokens

    K = hidden_states.shape[1]

    hidden_states_shape = hidden_states.shape
    hidden_states_device = hidden_states.device
    hidden_states_dtype = hidden_states.dtype

    running_state["hidden_states_shape"] = hidden_states_shape
    running_state["hidden_states_device"] = hidden_states_device
    running_state["hidden_states_dtype"] = hidden_states_dtype
    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights

    input_tensor = torch.empty(
        (all_tokens, K),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
        # TODO check whether need `zeros`
        input_tensor_scale = torch.zeros(
            (ceil_div(K // 128, 4), all_tokens),
            device=hidden_states.device,
            dtype=torch.int,
        ).transpose(0, 1)
    else:
        input_tensor_scale = torch.empty(
            (all_tokens, K // 128),
            device=hidden_states.device,
            dtype=torch.float32,
        )
    m_indices = torch.empty(all_tokens, device=hidden_states.device, dtype=torch.int32)
    output_index = torch.empty_like(topk_ids)

    if get_offloader().forbid_copy_engine_usage:
        num_recv_tokens_per_expert_gpu = copy_list_to_gpu_no_ce(
            num_recv_tokens_per_expert
        )
    else:
        num_recv_tokens_per_expert_gpu = torch.tensor(
            num_recv_tokens_per_expert,
            dtype=torch.int32,
            pin_memory=True,
            device="cpu",
        ).cuda(non_blocking=True)
    expert_start_loc = torch.empty_like(num_recv_tokens_per_expert_gpu)

    ep_scatter(
        hidden_states,
        hidden_states_scale,
        topk_ids,
        num_recv_tokens_per_expert_gpu,
        expert_start_loc,
        input_tensor,
        input_tensor_scale,
        m_indices,
        output_index,
        scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
    )
    dispose_tensor(hidden_states)
    dispose_tensor(hidden_states_scale)

    running_state["output_index"] = output_index

    return DeepGemmRunnerInput(
        hidden_states=input_tensor,
        hidden_states_scale=input_tensor_scale,
        use_masked_gemm=False,
        m_indices=m_indices,
    )


@register_post_permute("deep_gemm", "deepep_normal")
def post_permute_deep_gemm_to_deepep_normal(
    runner_output: DeepGemmRunnerOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepEPNormalCombineInput:
    from sglang.srt.layers.moe.ep_moe.kernels import ep_gather
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPNormalCombineInput

    hidden_states = runner_output.hidden_states
    topk_ids = running_state["topk_ids"]
    topk_weights = running_state["topk_weights"]
    output_index = running_state["output_index"]

    gather_out = torch.empty(
        running_state["hidden_states_shape"],
        device=running_state["hidden_states_device"],
        dtype=torch.bfloat16,
    )
    ep_gather(hidden_states, topk_ids, topk_weights, output_index, gather_out)

    return DeepEPNormalCombineInput(
        hidden_states=gather_out,
        topk_ids=running_state["topk_ids"],
        topk_weights=running_state["topk_weights"],
    )


def _varlen_deep_gemm_silu_mul_quant(
    gateup_output: torch.Tensor,
    masked_m: Optional[torch.Tensor],
    group_size: int,
    topk: int,
    swiglu_limit: Optional[float] = None,
    swizzle: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from sglang.srt.layers.moe.ep_moe.kernels import silu_and_mul_masked_post_quant_fwd
    from sglang.srt.layers.quantization.fp8_kernel import (
        sglang_per_token_group_quant_8bit,
    )

    if _MASKED_GEMM_FAST_ACT:
        assert not swizzle, (
            "SGLANG_OPT_FIX_MEGA_MOE_MEMORY is incompatible with "
            "SGLANG_MASKED_GEMM_FAST_ACT (swizzled layout only supported by JIT act)"
        )
        assert (
            swiglu_limit is None
        ), "swiglu_limit (DSV4 2604 submode 2604B) is not supported together with SGLANG_MASKED_GEMM_FAST_ACT"
        return sglang_per_token_group_quant_8bit(
            x=gateup_output,
            dst_dtype=torch.float8_e4m3fn,
            group_size=group_size,
            masked_m=masked_m,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            fuse_silu_and_mul=True,
            enable_v2=True,
        )

    assert masked_m is not None
    hidden_states_device = gateup_output.device
    E, N, D_2 = gateup_output.shape
    D = D_2 // 2
    del D_2
    G = D // group_size
    down_input = torch.empty(
        (E, N, D),
        device=hidden_states_device,
        dtype=torch.float8_e4m3fn,
    )

    if envs.SGLANG_OPT_USE_JIT_EP_ACTIVATION.get():
        from sglang.jit_kernel.deepseek_v4 import silu_and_mul_masked_post_quant

        assert N % 4 == 0 and G % 4 == 0
        packed_ue8m0 = deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
        down_input_scale = torch.empty(
            (E, G // 4, N) if packed_ue8m0 else (E, N, G),
            device=hidden_states_device,
            dtype=torch.int32 if packed_ue8m0 else torch.float32,
        )
        silu_and_mul_masked_post_quant(
            gateup_output,
            down_input,
            down_input_scale,
            group_size,
            masked_m,
            scale_ue8m0=packed_ue8m0,
            topk=topk,
            transposed=packed_ue8m0,
            swiglu_limit=swiglu_limit,
            swizzle=swizzle,
        )
        if packed_ue8m0:
            down_input_scale = down_input_scale.transpose(-1, -2)
    else:
        assert (
            swiglu_limit is None
        ), "swiglu_limit (DSV4 2604 submode 2604B) requires SGLANG_OPT_USE_JIT_EP_ACTIVATION=True"
        assert (
            not swizzle
        ), "SGLANG_OPT_FIX_MEGA_MOE_MEMORY requires SGLANG_OPT_USE_JIT_EP_ACTIVATION=True"
        down_input_scale = torch.empty(
            (E, N, G),
            device=hidden_states_device,
            dtype=torch.float32,
        )
        silu_and_mul_masked_post_quant_fwd(
            gateup_output,
            down_input,
            down_input_scale,
            group_size,
            masked_m,
            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        )
    return down_input, down_input_scale


def _apply_swiglu_limit(
    gateup_output: torch.Tensor, swiglu_limit: float
) -> torch.Tensor:
    assert swiglu_limit == 10

    num_tokens, hidden_size_x2 = gateup_output.shape
    if envs.SGLANG_DEBUG_SANITY_CHECK_CONFIG.get() and not is_large_dummy_model():
        assert hidden_size_x2 == 2048 * 2
    assert gateup_output.dtype == torch.bfloat16

    gate, up = torch.chunk(gateup_output, chunks=2, dim=-1)
    assert gate.shape == (num_tokens, hidden_size_x2 // 2)
    assert up.shape == (num_tokens, hidden_size_x2 // 2)

    up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
    gate = torch.clamp(gate, max=swiglu_limit)

    out = torch.cat([gate, up], dim=-1)
    assert out.shape == (num_tokens, hidden_size_x2)
    return out


# --- imported with the qwen4 subsystem (sgl-project/sglang) ---


def set_masked_standard_layout_memory_budget(
    available_memory_bytes: int,
) -> int:
    """Cache the masked-layout share of free non-static device memory."""
    global _masked_standard_layout_memory_budget_bytes
    fraction = envs.SGLANG_DEEPGEMM_MASKED_MEMORY_BUDGET_FRACTION.get()
    if not 0.0 < fraction <= 1.0:
        raise ValueError(
            "SGLANG_DEEPGEMM_MASKED_MEMORY_BUDGET_FRACTION must be in (0, 1]"
        )
    _masked_standard_layout_memory_budget_bytes = int(available_memory_bytes * fraction)
    return _masked_standard_layout_memory_budget_bytes


def _estimate_masked_standard_layout_peak_bytes(
    runner_config: MoeRunnerConfig,
    quant_info: DeepGemmMoeQuantInfo,
    hidden_states: torch.Tensor,
) -> int:
    padded_m = (hidden_states.shape[0] // 256 + 1) * 256
    activation_dtype = (
        torch.bfloat16
        if quant_info.w13_weight.dtype == torch.bfloat16
        else torch.float8_e4m3fn
    )
    hidden_size = hidden_states.shape[1]
    gateup_size = quant_info.w13_weight.shape[1]
    gateup_row_bytes = gateup_size * torch.bfloat16.itemsize
    down_output_row_bytes = quant_info.w2_weight.shape[1] * torch.bfloat16.itemsize
    input_row_bytes = hidden_size * activation_dtype.itemsize
    down_input_row_bytes = gateup_size // 2 * activation_dtype.itemsize

    if activation_dtype == torch.bfloat16:
        input_scale_row_bytes = 0
        down_scale_row_bytes = 0
    else:
        block_k = quant_info.block_shape[1] if quant_info.block_shape else 128
        packed_scales = quant_info.use_mxfp8 or deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
        scale_item_bytes = (
            torch.uint8.itemsize if packed_scales else torch.float32.itemsize
        )
        input_scale_row_bytes = ceil_div(hidden_size, block_k) * scale_item_bytes
        down_scale_row_bytes = ceil_div(gateup_size // 2, block_k) * scale_item_bytes

    peak_row_bytes = max(
        input_row_bytes + input_scale_row_bytes + gateup_row_bytes,
        gateup_row_bytes + down_input_row_bytes + down_scale_row_bytes,
        down_input_row_bytes + down_scale_row_bytes + down_output_row_bytes,
    )
    return runner_config.num_local_experts * padded_m * peak_row_bytes


def _masked_activation_unsupported_reason(
    runner_config: MoeRunnerConfig, quant_info: DeepGemmMoeQuantInfo
) -> Optional[str]:
    if runner_config.swiglu_limit is None and not get_moe_a2a_backend().is_megamoe():
        return None
    d = runner_config.intermediate_size_per_partition
    e = runner_config.num_local_experts
    if d is None or e is None:
        return None
    group_size = quant_info.block_shape[1] if quant_info.block_shape else 128
    if d // 8 < e:
        return f"D // 8 ({d // 8}) < num_local_experts ({e})"
    if group_size != 128:
        return (
            f"masked activation group_size {group_size}, DSV4 JIT kernel requires 128"
        )
    if d % (group_size * 4) != 0:
        return f"D ({d}) not divisible by 4 * group_size"
    return None


def _should_use_masked_standard_layout(
    runner_config: MoeRunnerConfig,
    quant_info: DeepGemmMoeQuantInfo,
    hidden_states: torch.Tensor,
) -> bool:
    # Preserve the Oakhaven WideEP escape hatch while adopting upstream's
    # memory-budget-based auto policy. CUDA graph capture remains masked.
    if (
        envs.SGLANG_OPT_DG_COMPACT_EAGER.get()
        and not get_flags().capture.disable_dispose_tensor
    ):
        return False

    reason = _masked_activation_unsupported_reason(runner_config, quant_info)
    if reason is not None:
        global _masked_activation_fallback_logged
        if not _masked_activation_fallback_logged:
            _masked_activation_fallback_logged = True
            logger.info(
                "DeepGEMM masked standard layout disabled: %s. "
                "Clamped/swizzled activations on this config must use the "
                "compact layout.",
                reason,
            )
        return False
    mode = envs.SGLANG_DEEPGEMM_STANDARD_LAYOUT.get().lower()
    if mode not in ("auto", "masked", "compact"):
        raise ValueError(
            "SGLANG_DEEPGEMM_STANDARD_LAYOUT must be one of: auto, masked, compact"
        )
    if mode != "auto":
        return mode == "masked"

    global _masked_standard_layout_memory_budget_bytes
    if _masked_standard_layout_memory_budget_bytes is None:
        # Serving sets an all-rank budget before capture. Direct eager callers
        # fall back to this rank's free memory without querying inside capture.
        # Import lazily to avoid a module-initialization cycle through
        # runner_utils -> DeepEP -> MoE -> this module.
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            get_is_capture_mode,
        )

        if get_is_capture_mode():
            return False
        free_memory, _ = torch.cuda.mem_get_info(hidden_states.device)
        set_masked_standard_layout_memory_budget(free_memory)

    return (
        _estimate_masked_standard_layout_peak_bytes(
            runner_config, quant_info, hidden_states
        )
        <= _masked_standard_layout_memory_budget_bytes
    )


def _get_compact_all_tokens(
    num_assignments: int, num_experts: int, block_e: int = 128
) -> int:
    """Return the maximum padded rows over all routings of the assignments."""
    max_nonempty_experts = min(num_assignments, num_experts)
    return block_e * (
        max_nonempty_experts + (num_assignments - max_nonempty_experts) // block_e
    )


@register_pre_permute("flashinfer", "deep_gemm")
def pre_permute_flashinfer_to_deep_gemm(
    dispatch_output: FlashinferDispatchOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepGemmRunnerInput:
    """Feed one-sided A2A output into DeepGEMM with fused expert remapping."""

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput

    if dispatch_output.hidden_states.dtype != torch.bfloat16:
        raise TypeError(
            "FlashInfer A2A + DeepGEMM requires a BF16 dispatch payload, got "
            f"{dispatch_output.hidden_states.dtype}."
        )
    if dispatch_output.hidden_states_scale is not None:
        raise ValueError(
            "FlashInfer A2A + DeepGEMM expects unquantized BF16 dispatch; "
            "hidden_states_scale must be None."
        )
    if dispatch_output.topk_output.topk_ids.dtype != torch.int32:
        raise TypeError(
            "FlashInfer A2A expert IDs must be int32 before DeepGEMM, got "
            f"{dispatch_output.topk_output.topk_ids.dtype}."
        )

    standard_output = StandardDispatchOutput(
        hidden_states=dispatch_output.hidden_states,
        hidden_states_scale=None,
        topk_output=dispatch_output.topk_output,
    )
    expert_start = get_parallel().moe_ep_rank * runner_config.num_local_experts
    return pre_permute_standard_to_deep_gemm(
        standard_output,
        quant_info,
        runner_config,
        running_state,
        expert_start=expert_start,
    )


@register_post_permute("deep_gemm", "flashinfer")
def post_permute_deep_gemm_to_flashinfer(
    runner_output: DeepGemmRunnerOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
):
    """Reuse DeepGEMM's weighted post-permute and hand BF16 to A2A combine."""

    from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
        FlashinferCombineInput,
    )

    standard_input = post_permute_deep_gemm_to_standard(
        runner_output, quant_info, runner_config, running_state
    )
    if standard_input.hidden_states.dtype != torch.bfloat16:
        raise TypeError(
            "FlashInfer A2A + DeepGEMM combine payload must be BF16, got "
            f"{standard_input.hidden_states.dtype}."
        )
    return FlashinferCombineInput(hidden_states=standard_input.hidden_states)


def _varlen_deep_gemm_situ_mul_quant(
    gateup_output: torch.Tensor,
    masked_m: torch.Tensor,
    group_size: int,
    topk: int,
    beta: float,
    linear_beta: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused SiTU activation + per-group fp8 quant via CUDA JIT kernel."""
    from sglang.kernels.ops.moe import situ_and_mul_masked_post_quant

    E, N, D_2 = gateup_output.shape
    D = D_2 // 2
    G = D // group_size
    packed_ue8m0 = deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0

    down_input = torch.empty(
        (E, N, D), device=gateup_output.device, dtype=torch.float8_e4m3fn
    )
    if packed_ue8m0:
        down_input_scale = torch.empty(
            (E, G // 4, N), device=gateup_output.device, dtype=torch.int32
        )
    else:
        down_input_scale = torch.empty(
            (E, N, G), device=gateup_output.device, dtype=torch.float32
        )

    situ_and_mul_masked_post_quant(
        gateup_output,
        down_input,
        down_input_scale,
        group_size,
        masked_m,
        beta=beta,
        linear_beta=linear_beta,
        scale_ue8m0=packed_ue8m0,
        topk=topk,
        transposed=packed_ue8m0,
    )

    if packed_ue8m0:
        down_input_scale = down_input_scale.transpose(-1, -2)

    return down_input, down_input_scale


@triton.jit
def _situ_mul_quant_contig_kernel(
    g_ptr,  # [rows, 2N] bf16, non-interleaved [gate; up] halves
    q_ptr,  # [rows, N] fp8 out
    s_ptr,  # [rows, KG] fp32 scales out
    N,
    KG,
    situ_beta,
    situ_linear_beta,
    GROUP: tl.constexpr,
    KG_POW2: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    rows2d = tl.arange(0, KG_POW2)[:, None]
    cols = tl.arange(0, GROUP)[None, :]
    offs = rows2d * GROUP + cols
    mask = rows2d < KG
    gate = tl.load(g_ptr + row * 2 * N + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(g_ptr + row * 2 * N + N + offs, mask=mask, other=0.0).to(tl.float32)
    # tanh(x) == 2*sigmoid(2x) - 1 (avoids a libdevice dependency)
    gate_t = 2.0 * tl.sigmoid(2.0 * gate / situ_beta) - 1.0
    gate = situ_beta * gate_t * tl.sigmoid(gate)
    up_t = 2.0 * tl.sigmoid(2.0 * up / situ_linear_beta) - 1.0
    y = gate * situ_linear_beta * up_t
    amax = tl.clamp(tl.max(tl.abs(y), axis=1), min=1e-10, max=float("inf"))
    q = (y * (448.0 / amax)[:, None]).to(tl.float8e4nv)
    tl.store(q_ptr + row * N + offs, q, mask=mask)
    srow = tl.arange(0, KG_POW2)
    tl.store(s_ptr + row * KG + srow, amax / 448.0, mask=srow < KG)


@register_pre_permute("deepep_v2", "deep_gemm")
def pre_permute_deepep_v2_to_deep_gemm(
    dispatch_output: DeepEPv2DispatchOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepGemmRunnerInput:
    from sglang.kernels.ops.moe.ep_moe_kernels import (
        ep_scatter_from_psum,
        fill_m_indices_from_psum,
    )

    hidden_states = dispatch_output.hidden_states
    hidden_states_scale = dispatch_output.hidden_states_scale
    topk_ids = dispatch_output.topk_ids
    topk_weights = dispatch_output.topk_weights
    psum_num_recv_tokens_per_expert = dispatch_output.psum_num_recv_tokens_per_expert
    is_expanded = dispatch_output.is_expanded
    deepep_v2_use_masked = dispatch_output.use_masked_gemm
    deepep_v2_expected_m = dispatch_output.expected_m
    deepep_v2_masked_max_m = dispatch_output.masked_max_m
    deepep_v2_total_expanded = dispatch_output.total_expanded
    deepep_v2_expert_alignment = dispatch_output.expert_alignment
    is_fp8 = hidden_states_scale is not None
    if not is_fp8 and hidden_states.dtype != torch.bfloat16:
        raise RuntimeError(
            "DeepEP v2 -> DeepGEMM requires either FP8 dispatch output with "
            "activation scales or BF16 dispatch output, but the dispatch "
            f"output carried {hidden_states.dtype} without scales."
        )
    assert runner_config.activation == "silu"

    if is_expanded:
        if psum_num_recv_tokens_per_expert is None:
            raise RuntimeError(
                "DeepEP v2 requires the native expert prefix sums from the "
                "ElasticBuffer dispatch handle."
            )
        all_tokens = hidden_states.shape[0]
        running_state["all_tokens"] = all_tokens
        running_state["hidden_states_shape"] = hidden_states.shape
        running_state["hidden_states_device"] = hidden_states.device
        running_state["hidden_states_dtype"] = hidden_states.dtype
        running_state["topk_ids"] = None
        running_state["topk_weights"] = topk_weights
        running_state["deepep_v2_expanded"] = True

        if deepep_v2_use_masked:
            # masked_m bounds each expert independently of buffer capacity.
            from sglang.kernels.ops.moe.ep_moe_kernels import expand_to_masked_slab

            num_local_experts = psum_num_recv_tokens_per_expert.shape[0]
            input_tensor, input_tensor_scale, masked_m = expand_to_masked_slab(
                hidden_states,
                hidden_states_scale,
                psum_num_recv_tokens_per_expert,
                num_local_experts,
                deepep_v2_masked_max_m,
                deepep_v2_expert_alignment,
            )
            running_state["deepep_v2_masked"] = True
            running_state["deepep_v2_psum"] = psum_num_recv_tokens_per_expert
            running_state["deepep_v2_total_expanded"] = deepep_v2_total_expanded
            running_state["deepep_v2_expert_alignment"] = deepep_v2_expert_alignment
            return DeepGemmRunnerInput(
                hidden_states=input_tensor,
                hidden_states_scale=input_tensor_scale,
                use_masked_gemm=True,
                masked_m=masked_m,
                expected_m=deepep_v2_expected_m,
                activation_scale_block_size=(
                    dispatch_output.activation_scale_block_size
                ),
            )

        num_local_experts = psum_num_recv_tokens_per_expert.shape[0]
        m_indices = fill_m_indices_from_psum(
            psum_num_recv_tokens_per_expert,
            num_local_experts,
            all_tokens,
            deepep_v2_expert_alignment,
        )
        return DeepGemmRunnerInput(
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            use_masked_gemm=False,
            m_indices=m_indices,
            activation_scale_block_size=dispatch_output.activation_scale_block_size,
        )

    all_tokens = int(psum_num_recv_tokens_per_expert[-1].item())
    K = hidden_states.shape[1]
    scale_block_size = dispatch_output.activation_scale_block_size
    running_state["all_tokens"] = all_tokens
    running_state["hidden_states_shape"] = hidden_states.shape
    running_state["hidden_states_device"] = hidden_states.device
    running_state["hidden_states_dtype"] = hidden_states.dtype
    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights

    input_tensor = torch.empty(
        (all_tokens, K), device=hidden_states.device, dtype=hidden_states.dtype
    )
    if not is_fp8:
        input_tensor_scale = None
    elif deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
        # Packed UE8M0 scales require zero padding lanes.
        input_tensor_scale = torch.zeros(
            (ceil_div(K // scale_block_size, 4), all_tokens),
            device=hidden_states.device,
            dtype=torch.int,
        ).transpose(0, 1)
    else:
        input_tensor_scale = torch.empty(
            (all_tokens, K // scale_block_size),
            device=hidden_states.device,
            dtype=torch.float32,
        )
    m_indices = torch.empty(all_tokens, device=hidden_states.device, dtype=torch.int32)
    output_index = torch.empty_like(topk_ids)
    # Contiguous psum already includes the 128-row expert alignment.
    expert_start_loc = torch.empty_like(psum_num_recv_tokens_per_expert)
    ep_scatter_from_psum(
        hidden_states,
        hidden_states_scale,
        topk_ids,
        psum_num_recv_tokens_per_expert,
        expert_start_loc,
        input_tensor,
        input_tensor_scale,
        m_indices,
        output_index,
        scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        quant_block_size=scale_block_size,
    )
    dispose_tensor(hidden_states)
    if hidden_states_scale is not None:
        dispose_tensor(hidden_states_scale)
    running_state["output_index"] = output_index

    return DeepGemmRunnerInput(
        hidden_states=input_tensor,
        hidden_states_scale=input_tensor_scale,
        use_masked_gemm=False,
        m_indices=m_indices,
        activation_scale_block_size=dispatch_output.activation_scale_block_size,
    )


@register_post_permute("deep_gemm", "deepep_v2")
def post_permute_deep_gemm_to_deepep_v2(
    runner_output: DeepGemmRunnerOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepEPv2CombineInput:
    from sglang.kernels.ops.moe.ep_moe_kernels import ep_gather
    from sglang.srt.layers.moe.token_dispatcher.base import RoutewiseLayout
    from sglang.srt.layers.moe.token_dispatcher.deepep_v2 import DeepEPv2CombineInput

    return_unweighted_routes = runner_config.no_combine
    if running_state.get("deepep_v2_expanded", False):
        hidden_states = runner_output.hidden_states
        topk_weights = running_state["topk_weights"]
        if running_state.get("deepep_v2_masked", False):
            # A routewise finalizer must run before router weighting. Preserve
            # one raw row per route and carry its 1-D weight to that finalizer.
            from sglang.kernels.ops.moe.ep_moe_kernels import masked_slab_to_expand

            output_capacity = running_state["deepep_v2_total_expanded"]
            if topk_weights.ndim != 1 or topk_weights.shape[0] < output_capacity:
                raise ValueError(
                    "DeepEP v2 expanded output exceeds router-weight capacity"
                )
            hidden_states = masked_slab_to_expand(
                hidden_states,
                running_state["deepep_v2_psum"],
                output_capacity,
                running_state["deepep_v2_expert_alignment"],
                topk_weights=None if return_unweighted_routes else topk_weights,
            )
            if not return_unweighted_routes:
                return DeepEPv2CombineInput(hidden_states, None)
            # Match the communication-capacity weights to the output slab.
            return DeepEPv2CombineInput(
                hidden_states=hidden_states,
                topk_weights=topk_weights[: hidden_states.shape[0]],
                routewise_layout=RoutewiseLayout.EXPANDED,
            )
        if return_unweighted_routes:
            return DeepEPv2CombineInput(
                hidden_states, topk_weights, RoutewiseLayout.EXPANDED
            )
        if topk_weights is not None and not running_state.get(
            "deepep_v2_weight_prefused", False
        ):
            # Expanded combine does not consume top-k weights;
            # skip when fold-into-scale already applied them before down_proj.
            # In-place with fp32 weights rounds once; casting the weights to
            # bf16 first costs measurable accuracy.
            hidden_states.mul_(topk_weights.unsqueeze(-1))
        return DeepEPv2CombineInput(hidden_states, None)

    hidden_states = runner_output.hidden_states
    topk_ids = running_state["topk_ids"]
    topk_weights = running_state["topk_weights"]
    output_index = running_state["output_index"]
    if return_unweighted_routes:
        # Restore the route dimension required by a routewise finalizer.
        # output_index maps each received token/expert slot back to the compact
        # expert-sorted DeepGEMM output; -1 denotes a non-local route.
        valid = output_index >= 0
        if hidden_states.shape[0] == 0:
            route_out = hidden_states.new_zeros(
                (*output_index.shape, hidden_states.shape[-1])
            )
        else:
            safe_output_index = output_index.clamp_min(0).to(torch.int64)
            route_out = hidden_states[safe_output_index]
            route_out.masked_fill_(~valid.unsqueeze(-1), 0)
        return DeepEPv2CombineInput(
            hidden_states=route_out,
            topk_weights=topk_weights,
            routewise_layout=RoutewiseLayout.TOKEN_TOPK,
        )
    gather_out = torch.empty(
        running_state["hidden_states_shape"],
        device=running_state["hidden_states_device"],
        dtype=torch.bfloat16,
    )
    ep_gather(hidden_states, topk_ids, topk_weights, output_index, gather_out)
    return DeepEPv2CombineInput(
        hidden_states=gather_out,
        topk_weights=topk_weights,
    )
