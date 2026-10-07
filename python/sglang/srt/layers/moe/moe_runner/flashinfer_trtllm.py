from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    register_fused_func,
)
from sglang.srt.layers.quantization.fp8_kernel import (
    per_token_group_quant_fp8,
    scaled_fp8_quant,
)
from sglang.srt.utils.common import (
    is_cuda_alike,
    is_flashinfer_available,
    is_sm120_supported,
    next_power_of_2,
)
from typing import Optional
from contextlib import contextmanager

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        StandardCombineInput,
        StandardDispatchOutput,
    )

if is_flashinfer_available() and is_sm120_supported():
    from flashinfer import fp4_quantize
elif is_cuda_alike():
    from sgl_kernel import scaled_fp4_quant as fp4_quantize
else:
    fp4_quantize = None


def align_fp8_moe_weights_for_flashinfer_trtllm(
    layer: Module, swap_w13_halves: bool = False
) -> None:
    """Prepare FP8 MoE weights/scales for FlashInfer TRT-LLM kernels.

    Args:
        layer: The MoE layer to process.
        swap_w13_halves: If True, swap W13 halves from [Up, Gate] to [Gate, Up].
            This is needed for ModelOpt FP8 checkpoints which store weights in
            [Up, Gate] order, while regular FP8 checkpoints store them in [Gate, Up].
    """
    from flashinfer import reorder_rows_for_gated_act_gemm, shuffle_matrix_a

    w13_weight = cast(torch.Tensor, layer.w13_weight)
    w2_weight = cast(torch.Tensor, layer.w2_weight)
    num_experts, two_n, hidden = w13_weight.shape

    # Optionally swap W13 halves: [Up, Gate] -> [Gate, Up]
    if swap_w13_halves:
        inter = two_n // 2
        w13_weight = (
            w13_weight.reshape(num_experts, 2, inter, hidden)
            .flip(dims=[1])
            .reshape(num_experts, two_n, hidden)
        )

    w13_interleaved_list = [
        reorder_rows_for_gated_act_gemm(w13_weight[i]) for i in range(num_experts)
    ]
    w13_interleaved: torch.Tensor = torch.stack(w13_interleaved_list).reshape(
        num_experts, two_n, hidden
    )

    # Shuffle weights for transposed MMA output (both W13, W2)
    epilogue_tile_m = 128
    w13_shuffled = [
        shuffle_matrix_a(w13_interleaved[i].view(torch.uint8), epilogue_tile_m)
        for i in range(num_experts)
    ]
    w2_shuffled = [
        shuffle_matrix_a(w2_weight[i].view(torch.uint8), epilogue_tile_m)
        for i in range(num_experts)
    ]

    layer.w13_weight = Parameter(
        torch.stack(w13_shuffled).view(torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.w2_weight = Parameter(
        torch.stack(w2_shuffled).view(torch.float8_e4m3fn),
        requires_grad=False,
    )

    # Precompute and register per-expert output scaling factors for FI MoE.
    # Note: w13_input_scale and w2_input_scale are scalar Parameters post-reduction.
    assert hasattr(layer, "w13_input_scale") and layer.w13_input_scale is not None
    assert hasattr(layer, "w2_input_scale") and layer.w2_input_scale is not None
    assert hasattr(layer, "w13_weight_scale") and layer.w13_weight_scale is not None
    assert hasattr(layer, "w2_weight_scale") and layer.w2_weight_scale is not None

    input_scale = cast(torch.Tensor, layer.w13_input_scale).to(torch.float32)
    activation_scale = cast(torch.Tensor, layer.w2_input_scale).to(torch.float32)
    w13_weight_scale = cast(torch.Tensor, layer.w13_weight_scale).to(torch.float32)
    w2_weight_scale = cast(torch.Tensor, layer.w2_weight_scale).to(torch.float32)

    output1_scales_scalar = w13_weight_scale * input_scale * (1.0 / activation_scale)
    output1_scales_gate_scalar = w13_weight_scale * input_scale
    output2_scales_scalar = activation_scale * w2_weight_scale

    layer.output1_scales_scalar = Parameter(output1_scales_scalar, requires_grad=False)
    layer.output1_scales_gate_scalar = Parameter(
        output1_scales_gate_scalar, requires_grad=False
    )
    layer.output2_scales_scalar = Parameter(output2_scales_scalar, requires_grad=False)


def align_fp4_moe_weights_for_flashinfer_trtllm(layer: Module) -> None:
    """Prepare FP4 MoE weights/scales for FlashInfer TRT-LLM kernels.

    This function handles the weight transformation needed for FP4 TRTLLM MoE:
    - Reorders weights for gated activation GEMM
    - Shuffles weights and scales for transposed MMA output
    - Computes the output scale factors
    """
    from sglang.srt.layers.quantization.utils import (
        prepare_static_weights_for_trtllm_fp4_moe,
    )

    w13_weight = cast(torch.Tensor, layer.w13_weight)
    w2_weight = cast(torch.Tensor, layer.w2_weight)
    w13_weight_scale = cast(torch.Tensor, layer.w13_weight_scale)
    w2_weight_scale = cast(torch.Tensor, layer.w2_weight_scale)

    (
        gemm1_weights_fp4_shuffled,
        gemm1_scales_fp4_shuffled,
        gemm2_weights_fp4_shuffled,
        gemm2_scales_fp4_shuffled,
    ) = prepare_static_weights_for_trtllm_fp4_moe(
        w13_weight,
        w2_weight,
        w13_weight_scale,
        w2_weight_scale,
        w2_weight.size(-2),  # hidden_size
        w13_weight.size(-2) // 2,  # intermediate_size
        w13_weight.size(0),  # num_experts
    )

    # Set flashinfer parameters
    layer.gemm1_weights_fp4_shuffled = Parameter(
        gemm1_weights_fp4_shuffled, requires_grad=False
    )
    layer.gemm2_weights_fp4_shuffled = Parameter(
        gemm2_weights_fp4_shuffled, requires_grad=False
    )
    layer.gemm1_scales_fp4_shuffled = Parameter(
        gemm1_scales_fp4_shuffled, requires_grad=False
    )
    layer.gemm2_scales_fp4_shuffled = Parameter(
        gemm2_scales_fp4_shuffled, requires_grad=False
    )

    # Compute additional scaling factor needed for TRT-LLM
    w2_input_scale_quant = cast(torch.Tensor, layer.w2_input_scale_quant)
    g1_alphas = cast(torch.Tensor, layer.g1_alphas)
    layer.g1_scale_c = Parameter(
        (w2_input_scale_quant * g1_alphas).to(torch.float32),
        requires_grad=False,
    )

    # Clean up weights that won't be used by TRT-LLM
    del (
        layer.w2_weight,
        layer.w2_weight_scale,
        layer.w13_weight,
        layer.w13_weight_scale,
    )


@dataclass
class FlashInferTrtllmFp8MoeQuantInfo(MoeQuantInfo):
    """Quantization payload consumed by FlashInfer TRT-LLM FP8 MoE kernels."""

    # Weights
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor

    # Expert-parallel metadata
    global_num_experts: int
    local_expert_offset: int
    local_num_experts: int
    intermediate_size: int

    routing_method_type: int

    # Block-quant path
    block_quant: bool
    weight_block_k: int | None = None
    w13_weight_scale_inv: torch.Tensor | None = None
    w2_weight_scale_inv: torch.Tensor | None = None

    # Per-tensor path
    w13_input_scale: torch.Tensor | None = None
    output1_scales_scalar: torch.Tensor | None = None
    output1_scales_gate_scalar: torch.Tensor | None = None
    output2_scales_scalar: torch.Tensor | None = None
    use_routing_scales_on_input: bool = False


def fused_experts_none_to_flashinfer_trtllm_fp8(
    dispatch_output: StandardDispatchOutput,
    quant_info: FlashInferTrtllmFp8MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from flashinfer.fused_moe import (
        trtllm_fp8_block_scale_moe,
        trtllm_fp8_per_tensor_scale_moe,
    )

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.layers.moe.utils import RoutingMethodType

    assert runner_config.activation == "silu", "Only silu is supported."
    assert not runner_config.no_combine, "no_combine is not supported for flashinfer."

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    assert TopKOutputChecker.format_is_bypassed(topk_output)

    router_logits = topk_output.router_logits
    topk_config = topk_output.topk_config
    correction_bias = (
        None
        if topk_config.correction_bias is None
        else topk_config.correction_bias.to(hidden_states.dtype)
    )

    routing_method_type = quant_info.routing_method_type

    if quant_info.block_quant:
        assert quant_info.weight_block_k is not None
        assert quant_info.w13_weight_scale_inv is not None
        assert quant_info.w2_weight_scale_inv is not None

        a_q, a_sf = per_token_group_quant_fp8(hidden_states, quant_info.weight_block_k)
        a_sf_t = a_sf.t().contiguous()

        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            # FIXME: there is a bug in the trtllm_fp8_block_scale_moe.
            # It ignored the `output` argument. https://github.com/flashinfer-ai/flashinfer/blob/da01b1bd8f9f22aec8c0eea189ad54860b034947/flashinfer/fused_moe/core.py#L1323-L1325
            # so we put the whole function under the ``use_symmetric_memory`` context manager.
            # If the bug is fixed, we can only put the output tensor allocation under the context manager.
            output = trtllm_fp8_block_scale_moe(
                routing_logits=(
                    router_logits.to(torch.float32)
                    if routing_method_type == RoutingMethodType.DeepSeekV3
                    else router_logits
                ),
                routing_bias=correction_bias,
                hidden_states=a_q,
                hidden_states_scale=a_sf_t,
                gemm1_weights=quant_info.w13_weight,
                gemm1_weights_scale=quant_info.w13_weight_scale_inv,
                gemm2_weights=quant_info.w2_weight,
                gemm2_weights_scale=quant_info.w2_weight_scale_inv,
                num_experts=quant_info.global_num_experts,
                top_k=topk_config.top_k,
                n_group=(
                    topk_config.num_expert_group if topk_config.num_expert_group else 0
                ),
                topk_group=topk_config.topk_group if topk_config.topk_group else 0,
                intermediate_size=quant_info.intermediate_size,
                local_expert_offset=quant_info.local_expert_offset,
                local_num_experts=quant_info.local_num_experts,
                routed_scaling_factor=(
                    runner_config.routed_scaling_factor
                    if runner_config.routed_scaling_factor is not None
                    else 1.0
                ),
                routing_method_type=routing_method_type,
                use_shuffled_weight=False,
                tune_max_num_tokens=next_power_of_2(a_q.shape[0]),
            )
    else:
        assert quant_info.w13_input_scale is not None
        assert quant_info.output1_scales_scalar is not None
        assert quant_info.output1_scales_gate_scalar is not None
        assert quant_info.output2_scales_scalar is not None

        a_q, _ = scaled_fp8_quant(hidden_states, quant_info.w13_input_scale)
        routing_bias_cast = (
            None if correction_bias is None else correction_bias.to(torch.bfloat16)
        )

        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            output = trtllm_fp8_per_tensor_scale_moe(
                routing_logits=router_logits.to(torch.bfloat16),
                routing_bias=routing_bias_cast,
                hidden_states=a_q,
                gemm1_weights=quant_info.w13_weight,
                output1_scales_scalar=quant_info.output1_scales_scalar,
                output1_scales_gate_scalar=quant_info.output1_scales_gate_scalar,
                gemm2_weights=quant_info.w2_weight,
                output2_scales_scalar=quant_info.output2_scales_scalar,
                num_experts=quant_info.global_num_experts,
                top_k=topk_config.top_k,
                n_group=(
                    topk_config.num_expert_group if topk_config.num_expert_group else 0
                ),
                topk_group=topk_config.topk_group if topk_config.topk_group else 0,
                intermediate_size=quant_info.intermediate_size,
                local_expert_offset=quant_info.local_expert_offset,
                local_num_experts=quant_info.local_num_experts,
                routed_scaling_factor=(
                    runner_config.routed_scaling_factor
                    if runner_config.routed_scaling_factor is not None
                    else 1.0
                ),
                use_routing_scales_on_input=quant_info.use_routing_scales_on_input,
                routing_method_type=routing_method_type,
                tune_max_num_tokens=next_power_of_2(a_q.shape[0]),
            )

    return StandardCombineInput(hidden_states=output)


@dataclass
class FlashInferTrtllmFp4MoeQuantInfo(MoeQuantInfo):
    """Quantization payload consumed by FlashInfer TRT-LLM FP4 MoE kernels."""

    # Shuffled FP4 weights (processed by align_fp4_moe_weights_for_flashinfer_trtllm)
    gemm1_weights_fp4_shuffled: torch.Tensor
    gemm2_weights_fp4_shuffled: torch.Tensor
    gemm1_scales_fp4_shuffled: torch.Tensor
    gemm2_scales_fp4_shuffled: torch.Tensor

    # Scaling factors
    g1_scale_c: torch.Tensor
    g1_alphas: torch.Tensor
    g2_alphas: torch.Tensor
    w13_input_scale_quant: torch.Tensor

    # Expert-parallel metadata
    global_num_experts: int
    local_expert_offset: int
    local_num_experts: int
    intermediate_size_per_partition: int

    routing_method_type: int


def quantize_hidden_states_fp4(
    hidden_states: torch.Tensor,
    input_scale_quant: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize hidden states to FP4 for TRTLLM MoE.

    Global scale factor is set by ModelOptNvFp4FusedMoEMethod during weight loading.
    Only block scales are computed at runtime for efficiency.

    Returns (packed_fp4_uint8, scale_float8_e4m3fn_runtime)
    """

    # flashinfer.fp4_quantize returns (packed_uint8, scale_fp8)
    # Only the block scales are computed at runtime
    hs_fp4_bytes, hs_sf_bytes = fp4_quantize(
        hidden_states,
        input_scale_quant,
        16,  # sf_vec_size
        False,  # use_ue8m0
        False,  # is_sf_swizzled_layout
    )

    seq_len, hidden_size = hidden_states.shape
    hs_fp4 = hs_fp4_bytes.reshape(seq_len, hidden_size // 2)
    # TRT-LLM expects hidden state scales shaped as [seq_len, hidden_size // 16]
    hs_sf = hs_sf_bytes.view(torch.float8_e4m3fn).reshape(seq_len, hidden_size // 16)

    return hs_fp4, hs_sf


def fused_experts_none_to_flashinfer_trtllm_fp4(
    dispatch_output: StandardDispatchOutput,
    quant_info: FlashInferTrtllmFp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    """FlashInfer TRTLLM FP4 MoE forward pass.

    This function handles the FP4 TRTLLM MoE path that was previously in
    FlashInferFP4MoE.forward_impl and ModelOptNvFp4FusedMoEMethod.apply.
    """
    from flashinfer.fused_moe import trtllm_fp4_block_scale_moe

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.layers.moe.utils import RoutingMethodType

    assert runner_config.activation == "silu", "Only silu is supported for FP4 MoE."
    assert runner_config.is_gated, "Only gated MoEs are supported for FP4 MoE."

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    assert TopKOutputChecker.format_is_bypassed(topk_output)

    router_logits = topk_output.router_logits
    topk_config = topk_output.topk_config
    routing_method_type = quant_info.routing_method_type

    # Quantize hidden states to FP4
    hs_fp4, hs_scale_linear = quantize_hidden_states_fp4(
        hidden_states, quant_info.w13_input_scale_quant
    )

    # DeepSeekV3 style routing requires float32 router logits
    if routing_method_type == RoutingMethodType.DeepSeekV3:
        router_logits = router_logits.to(torch.float32)

    correction_bias = (
        None
        if topk_config.correction_bias is None
        else topk_config.correction_bias.to(hidden_states.dtype)
    )

    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
        num_tokens = hs_fp4.shape[0]
        hidden_size = (
            hs_fp4.shape[-1] * 2 if hs_fp4.dtype == torch.uint8 else hs_fp4.shape[-1]
        )
        symm_output = torch.empty(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=hs_fp4.device
        )

    result = trtllm_fp4_block_scale_moe(
        routing_logits=router_logits,
        routing_bias=correction_bias,
        hidden_states=hs_fp4,
        hidden_states_scale=hs_scale_linear.view(torch.float8_e4m3fn).flatten(),
        gemm1_weights=quant_info.gemm1_weights_fp4_shuffled,
        gemm1_weights_scale=quant_info.gemm1_scales_fp4_shuffled.view(
            torch.float8_e4m3fn
        ),
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=quant_info.gemm2_weights_fp4_shuffled,
        gemm2_weights_scale=quant_info.gemm2_scales_fp4_shuffled.view(
            torch.float8_e4m3fn
        ),
        gemm2_bias=None,
        output1_scale_scalar=quant_info.g1_scale_c,
        output1_scale_gate_scalar=quant_info.g1_alphas,
        output2_scale_scalar=quant_info.g2_alphas,
        num_experts=quant_info.global_num_experts,
        top_k=topk_config.top_k,
        n_group=topk_config.num_expert_group,
        topk_group=topk_config.topk_group,
        intermediate_size=quant_info.intermediate_size_per_partition,
        local_expert_offset=quant_info.local_expert_offset,
        local_num_experts=quant_info.local_num_experts,
        routed_scaling_factor=runner_config.routed_scaling_factor,
        tile_tokens_dim=None,
        routing_method_type=(
            routing_method_type
            if routing_method_type is not None
            else RoutingMethodType.Default
        ),
        do_finalize=True,
        tune_max_num_tokens=next_power_of_2(hs_fp4.shape[0]),
        output=symm_output,
    )[0]

    return StandardCombineInput(hidden_states=result)


@dataclass
class FlashInferTrtllmBf16MoeQuantInfo(MoeQuantInfo):
    """Quantization payload consumed by FlashInfer TRT-LLM BF16 MoE kernels."""

    gemm1_weights: torch.Tensor
    gemm2_weights: torch.Tensor

    # Expert-parallel metadata
    global_num_experts: int
    local_expert_offset: int


def fused_experts_none_to_flashinfer_trtllm_bf16(
    dispatch_output: StandardDispatchOutput,
    quant_info: FlashInferTrtllmBf16MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    # lazy import
    try:
        from flashinfer.fused_moe import trtllm_bf16_moe
    except ImportError as e:
        raise ImportError(
            "Can't import trtllm_bf16_moe from flashinfer. "
            "Please check flashinfer version to use bf16 with flashinfer_trtllm backend."
        ) from e

    assert (
        runner_config.activation == "silu"
    ), "Only silu is supported for flashinfer trtllm moe"
    if hasattr(dispatch_output.topk_output, "topk_config"):
        assert (
            dispatch_output.topk_output.topk_config.renormalize
        ), "Renormalize is required for flashinfer trtllm moe"
    assert (
        runner_config.num_fused_shared_experts == 0
    ), "Fused shared experts are not supported for flashinfer trtllm moe"
    assert (
        runner_config.is_gated
    ), "Only gated MoEs are supported for flashinfer trtllm moe"
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    assert TopKOutputChecker.format_is_bypassed(
        dispatch_output.topk_output
    ) or TopKOutputChecker.format_is_standard(dispatch_output.topk_output)

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output

    # Handle both bypassed and standard topk output formats
    if TopKOutputChecker.format_is_bypassed(dispatch_output.topk_output):
        assert TopKOutputChecker.format_is_bypassed(
            dispatch_output.topk_output
        ), "BF16 MoE requires bypassed topk output"
        router_logits = topk_output.router_logits
        topk_config = topk_output.topk_config
        correction_bias = topk_config.correction_bias
        routed_scaling_factor = runner_config.routed_scaling_factor
    else:
        # Standard format - extract from topk_output directly
        router_logits = topk_output.router_logits
        topk_config = topk_output.topk_config
        correction_bias = topk_config.correction_bias
        routed_scaling_factor = runner_config.routed_scaling_factor

    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):

        # Call the fused kernel
        final_hidden_states = trtllm_bf16_moe(
            routing_logits=router_logits,
            routing_bias=correction_bias,
            hidden_states=hidden_states,
            gemm1_weights=quant_info.gemm1_weights,
            gemm2_weights=quant_info.gemm2_weights,
            num_experts=quant_info.global_num_experts,
            top_k=topk_config.top_k,
            n_group=topk_config.num_expert_group,
            topk_group=topk_config.topk_group,
            intermediate_size=runner_config.intermediate_size_per_partition,
            local_expert_offset=quant_info.local_expert_offset,
            local_num_experts=runner_config.num_local_experts,
            routing_method_type=runner_config.routing_method_type,
            routed_scaling_factor=routed_scaling_factor,
            tune_max_num_tokens=next_power_of_2(hidden_states.shape[0]),
        )

    return StandardCombineInput(hidden_states=final_hidden_states)


@register_fused_func("none", "flashinfer_trtllm")
def fused_experts_none_to_flashinfer_trtllm(
    dispatch_output: StandardDispatchOutput,
    quant_info: MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    """Dispatch to FP8 or FP4 FlashInfer TRT-LLM MoE based on quant_info type."""
    if isinstance(quant_info, FlashInferTrtllmFp4MoeQuantInfo):
        return fused_experts_none_to_flashinfer_trtllm_fp4(
            dispatch_output, quant_info, runner_config
        )
    if isinstance(quant_info, FlashInferTrtllmFp8MoeQuantInfo):
        return fused_experts_none_to_flashinfer_trtllm_fp8(
            dispatch_output, quant_info, runner_config
        )
    if isinstance(quant_info, FlashInferTrtllmBf16MoeQuantInfo):
        return fused_experts_none_to_flashinfer_trtllm_bf16(
            dispatch_output, quant_info, runner_config
        )
    raise TypeError(
        f"Unexpected quant_info type for flashinfer_trtllm: {type(quant_info)}"
    )


# --- imported with the qwen4 subsystem (sgl-project/sglang) ---


def trtllm_moe_enable_pdl(num_tokens: int) -> bool:
    from sglang.kernels.jit.utils import is_arch_support_pdl

    return is_arch_support_pdl() and num_tokens <= _TRTLLM_MOE_PDL_MAX_TOKENS


@dataclass
class FlashInferTrtllmDeferredFinalizeOutput:
    gemm2_out: torch.Tensor
    expert_weights: torch.Tensor
    expanded_idx_to_permuted_idx: torch.Tensor
    top_k: int


@contextmanager
def flashinfer_trtllm_deferred_finalize_context(
    enabled: bool = True,
) -> Generator[None, None, None]:
    token = _deferred_finalize_enabled.set(enabled)
    try:
        yield
    finally:
        _deferred_finalize_enabled.reset(token)


def is_deferred_finalize_enabled() -> bool:
    return _deferred_finalize_enabled.get()


def finalize_flashinfer_trtllm_deferred_output(
    deferred_output: FlashInferTrtllmDeferredFinalizeOutput,
    shared_output: Optional[torch.Tensor],
) -> torch.Tensor:
    from sglang.kernels.ops.moe.moe_finalize_fuse_shared import moe_finalize_fuse_shared

    return moe_finalize_fuse_shared(
        deferred_output.gemm2_out,
        deferred_output.expanded_idx_to_permuted_idx,
        deferred_output.expert_weights,
        shared_output,
        deferred_output.top_k,
        enable_pdl=trtllm_moe_enable_pdl(deferred_output.expert_weights.shape[0]),
    )


def _make_deferred_finalize_output(
    result,
    *,
    top_k: int,
) -> FlashInferTrtllmDeferredFinalizeOutput:
    """Validate and adapt FlashInfer's ``do_finalize=False`` output ABI."""
    gemm2_out, expert_weights, expanded_idx_to_permuted_idx = result[:3]
    # FlashInfer >= 0.6.18 types this buffer by content (flashinfer #3595):
    # bf16 for packed routing, the caller's dtype for unpacked routing.
    if expert_weights.dtype not in (torch.bfloat16, torch.float32):
        raise RuntimeError(
            "FlashInfer deferred finalize must return BF16 or FP32 expert weights, got "
            f"{expert_weights.dtype}"
        )
    if gemm2_out.dtype != torch.bfloat16:
        raise RuntimeError(
            "FlashInfer deferred finalize must return BF16 GEMM2 output, got "
            f"{gemm2_out.dtype}"
        )
    if expanded_idx_to_permuted_idx.dtype != torch.int32:
        raise RuntimeError(
            "FlashInfer deferred finalize must return Int32 permuted indices, got "
            f"{expanded_idx_to_permuted_idx.dtype}"
        )
    return FlashInferTrtllmDeferredFinalizeOutput(
        gemm2_out=gemm2_out,
        expert_weights=expert_weights,
        expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
        top_k=top_k,
    )


def round_up_to_multiple(x: int, m: int) -> int:
    """Round up *x* to the nearest multiple of *m*."""
    return (x + m - 1) // m * m


def clear_mxfp8_shuffle_index_cache() -> None:
    """Drop the cached MXFP8 MoE row-index permutations.
    The cached index tensors are GPU-resident; sglang reuses the weights-region
    memory across weight-update cycles
    """
    _flashinfer_trtllm_shuffle_row_indices_cache_mxfp8.clear()


def _is_gated(layer: Module) -> bool:
    """Return whether the MoE layer uses a gated activation (default True)."""
    is_gated = (
        getattr(layer, "moe_runner_config", None) and layer.moe_runner_config.is_gated
    )
    return True if is_gated is None else is_gated


def _get_routing_for_flashinfer_routed(topk_output) -> FlashInferRouting:
    """Return the `topk_ids` kernel argument for the trtllm routed MoEs."""
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    if TopKOutputChecker.format_is_packed(topk_output):
        return topk_output.packed_topk_ids

    assert TopKOutputChecker.format_is_standard(topk_output)
    return (
        topk_output.topk_ids.contiguous(),
        topk_output.topk_weights.contiguous(),
    )


def _routing_top_k(routing: FlashInferRouting) -> int:
    """Both routing forms carry top_k as the last dim of their ids tensor."""
    ids = routing[0] if isinstance(routing, tuple) else routing
    return ids.shape[1]


def _align_fp8_moe_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
    is_gated: bool,
    min_alignment: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pad intermediate size so FlashInfer TRTLLM FP8 kernels' alignment holds.

    Returns (w13, w2, padded_intermediate).
    """
    num_experts, hidden_size, intermediate = w2.shape

    padded_intermediate = round_up_to_multiple(intermediate, min_alignment)
    if padded_intermediate == intermediate:
        return w13, w2, intermediate

    logger.info(
        "FP8 MoE: padding intermediate size from %d to %d (alignment=%d)",
        intermediate,
        padded_intermediate,
        min_alignment,
    )

    up_mult = 2 if is_gated else 1
    padded_gate_up = up_mult * padded_intermediate

    padded_w13 = w13.new_zeros((num_experts, padded_gate_up, w13.shape[2]))
    padded_w13[:, : w13.shape[1], :] = w13

    padded_w2 = w2.new_zeros((num_experts, hidden_size, padded_intermediate))
    padded_w2[:, :, :intermediate] = w2

    return padded_w13, padded_w2, padded_intermediate


def _align_mxfp8_moe_weights(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    is_gated: bool,
    min_alignment: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pad intermediate size so FlashInfer TRTLLM MXFP8 kernels' alignment holds.

    Returns (w13, w13_scale, w2, w2_scale, padded_intermediate).
    """
    num_experts, hidden_size, intermediate = w2.shape

    padded_intermediate = round_up_to_multiple(intermediate, min_alignment)
    if padded_intermediate == intermediate:
        return w13, w13_scale, w2, w2_scale, intermediate

    logger.info(
        "MXFP8 MoE: padding intermediate size from %d to %d (alignment=%d)",
        intermediate,
        padded_intermediate,
        min_alignment,
    )

    up_mult = 2 if is_gated else 1
    padded_gate_up = up_mult * padded_intermediate

    padded_w13 = w13.new_zeros((num_experts, padded_gate_up, w13.shape[2]))
    padded_w13[:, : w13.shape[1], :] = w13

    padded_w2 = w2.new_zeros((num_experts, hidden_size, padded_intermediate))
    padded_w2[:, :, :intermediate] = w2

    padded_w13_scale = w13_scale.new_zeros(
        (num_experts, padded_gate_up, w13_scale.shape[2])
    )
    padded_w13_scale[:, : w13_scale.shape[1], :] = w13_scale

    # Scale's last dim tracks intermediate / block_size (MXFP8 block_size = 32)
    scale_block_k = intermediate // w2_scale.shape[2] if w2_scale.shape[2] > 0 else 32
    padded_w2_scale = w2_scale.new_zeros(
        (num_experts, hidden_size, padded_intermediate // scale_block_k)
    )
    padded_w2_scale[:, :, : w2_scale.shape[2]] = w2_scale

    return padded_w13, padded_w13_scale, padded_w2, padded_w2_scale, padded_intermediate


def align_mxfp8_moe_weights_for_flashinfer_trtllm(layer: Module) -> None:
    """Prepare MXFP8 MoE weights/scales for FlashInfer TRT-LLM kernels."""
    from flashinfer import block_scale_interleave
    from flashinfer.fused_moe.core import (
        get_reorder_rows_for_gated_act_gemm_row_indices,
    )
    from flashinfer.utils import (
        get_shuffle_matrix_a_row_indices,
        get_shuffle_matrix_sf_a_row_indices,
    )

    is_gated = _is_gated(layer)

    w13_weight = cast(torch.Tensor, layer.w13_weight).contiguous()
    w2_weight = cast(torch.Tensor, layer.w2_weight).contiguous()
    w13_scale = cast(torch.Tensor, layer.w13_weight_scale_inv).contiguous()
    w2_scale = cast(torch.Tensor, layer.w2_weight_scale_inv).contiguous()

    assert w13_scale.dtype == torch.uint8
    assert w2_scale.dtype == torch.uint8

    if not is_gated:
        intermediate = w2_weight.shape[2]
        w13_weight = w13_weight[:, :intermediate, :].contiguous()
        w13_scale = w13_scale[:, :intermediate, :].contiguous()

    # Pad for kernel alignment (non-gated needs 128, gated needs 16)
    min_alignment = 16 if is_gated else 128
    w13_weight, w13_scale, w2_weight, w2_scale, _ = _align_mxfp8_moe_weights(
        w13_weight, w13_scale, w2_weight, w2_scale, is_gated, min_alignment
    )

    num_experts, gate_up_dim, _ = w13_weight.shape
    _, hidden_size, _ = w2_weight.shape
    epilogue_tile_m = 128

    # Reuse precomputed row-index transforms whenever shape/device are unchanged.
    w13_weight_u8 = w13_weight.view(torch.uint8)
    w2_weight_u8 = w2_weight.view(torch.uint8)
    cache_key = (
        gate_up_dim,
        hidden_size,
        w2_weight.shape[-1],
        w13_scale.shape[-1],
        w2_scale.shape[-1],
        epilogue_tile_m,
        (w13_weight.device.type, w13_weight.device.index),
        (w2_weight.device.type, w2_weight.device.index),
        (w13_scale.device.type, w13_scale.device.index),
        (w2_scale.device.type, w2_scale.device.index),
    )
    cache = _flashinfer_trtllm_shuffle_row_indices_cache_mxfp8.get(cache_key)
    if cache is None:
        if is_gated:
            reorder_row_indices = get_reorder_rows_for_gated_act_gemm_row_indices(
                w13_weight_u8[0]
            ).to(w13_weight.device)
        else:
            reorder_row_indices = torch.arange(
                gate_up_dim, device=w13_weight.device, dtype=torch.long
            )
        w13_shuffle_row_indices = get_shuffle_matrix_a_row_indices(
            w13_weight_u8[0], epilogue_tile_m
        ).to(w13_weight.device)
        w2_shuffle_row_indices = get_shuffle_matrix_a_row_indices(
            w2_weight_u8[0], epilogue_tile_m
        ).to(w2_weight.device)
        w13_scale_shuffle_row_indices = get_shuffle_matrix_sf_a_row_indices(
            w13_scale[0].reshape(gate_up_dim, -1), epilogue_tile_m
        ).to(w13_scale.device)
        w2_scale_shuffle_row_indices = get_shuffle_matrix_sf_a_row_indices(
            w2_scale[0].reshape(hidden_size, -1), epilogue_tile_m
        ).to(w2_scale.device)
        cache = {
            "reorder_row_indices": reorder_row_indices,
            "w13_shuffle_row_indices": w13_shuffle_row_indices,
            "w2_shuffle_row_indices": w2_shuffle_row_indices,
            "w13_scale_shuffle_row_indices": w13_scale_shuffle_row_indices,
            "w2_scale_shuffle_row_indices": w2_scale_shuffle_row_indices,
        }
        _flashinfer_trtllm_shuffle_row_indices_cache_mxfp8[cache_key] = cache

    reorder_row_indices = cache["reorder_row_indices"]
    w13_shuffle_row_indices = cache["w13_shuffle_row_indices"]
    w2_shuffle_row_indices = cache["w2_shuffle_row_indices"]
    w13_scale_shuffle_row_indices = cache["w13_scale_shuffle_row_indices"]
    w2_scale_shuffle_row_indices = cache["w2_scale_shuffle_row_indices"]

    w13_shuffled_u8 = torch.empty_like(w13_weight_u8)
    w2_shuffled_u8 = torch.empty_like(w2_weight_u8)
    w13_scale_shuffled = torch.empty_like(w13_scale)
    w2_scale_shuffled = torch.empty_like(w2_scale)

    for i in range(num_experts):
        w13_interleaved_u8 = w13_weight_u8[i].index_select(0, reorder_row_indices)
        w13_scale_interleaved = w13_scale[i].index_select(0, reorder_row_indices)

        w13_shuffled_u8[i].copy_(
            w13_interleaved_u8.index_select(0, w13_shuffle_row_indices)
        )
        w2_shuffled_u8[i].copy_(w2_weight_u8[i].index_select(0, w2_shuffle_row_indices))

        w13_scale_linear = w13_scale_interleaved.reshape(gate_up_dim, -1)
        w13_scale_shuffled[i].copy_(
            block_scale_interleave(
                w13_scale_linear.index_select(0, w13_scale_shuffle_row_indices)
            ).reshape_as(w13_scale_shuffled[i])
        )

        w2_scale_linear = w2_scale[i].reshape(hidden_size, -1)
        w2_scale_shuffled[i].copy_(
            block_scale_interleave(
                w2_scale_linear.index_select(0, w2_scale_shuffle_row_indices)
            ).reshape_as(w2_scale_shuffled[i])
        )

    # Keep parameter identities stable for CUDA graph capture reuse.
    copy_or_rebind_param(layer, "w13_weight", w13_shuffled_u8.view(torch.float8_e4m3fn))
    copy_or_rebind_param(layer, "w2_weight", w2_shuffled_u8.view(torch.float8_e4m3fn))
    copy_or_rebind_param(
        layer,
        "w13_weight_scale_inv",
        w13_scale_shuffled.contiguous(),
    )
    copy_or_rebind_param(
        layer,
        "w2_weight_scale_inv",
        w2_scale_shuffled.contiguous(),
    )
    layer.w13_weight_scale_inv.format_ue8m0 = True
    layer.w2_weight_scale_inv.format_ue8m0 = True


def _align_fp4_moe_weights(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    is_gated: bool,
    min_alignment: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pad intermediate size so FlashInfer TRTLLM FP4 kernels' alignment holds.

    Returns (w13, w13_scale, w2, w2_scale, padded_intermediate).
    """
    num_experts, hidden_size, intermediate_packed = w2.shape
    intermediate = intermediate_packed * 2  # FP4 packs 2 values per byte

    padded_intermediate = round_up_to_multiple(intermediate, min_alignment)
    if padded_intermediate == intermediate:
        return w13, w13_scale, w2, w2_scale, intermediate

    logger.info(
        "FP4 MoE: padding intermediate size from %d to %d (alignment=%d)",
        intermediate,
        padded_intermediate,
        min_alignment,
    )

    up_mult = 2 if is_gated else 1
    padded_gate_up = up_mult * padded_intermediate

    padded_w13 = w13.new_zeros((num_experts, padded_gate_up, w13.shape[2]))
    padded_w13[:, : w13.shape[1], :] = w13

    padded_w2 = w2.new_zeros((num_experts, hidden_size, padded_intermediate // 2))
    padded_w2[:, :, : w2.shape[2]] = w2

    padded_w13_scale = w13_scale.new_zeros(
        (num_experts, padded_gate_up, w13_scale.shape[2])
    )
    padded_w13_scale[:, : w13_scale.shape[1], :] = w13_scale

    padded_w2_scale = w2_scale.new_zeros(
        (num_experts, hidden_size, padded_intermediate // 16)
    )
    padded_w2_scale[:, :, : w2_scale.shape[2]] = w2_scale

    return padded_w13, padded_w13_scale, padded_w2, padded_w2_scale, padded_intermediate


def _compute_g1_scale_c(
    w2_input_scale_quant: torch.Tensor,
    g1_alphas: torch.Tensor,
    g1_alphas_up: torch.Tensor,
    is_gated: bool,
    activation: Optional[str] = None,
) -> torch.Tensor:
    """TRT-LLM GEMM1-output scale for the up (w3) half.

    TRT-LLM dequantizes the two halves of the fused GEMM1 separately: g1_alphas
    covers the gate half, this scalar the up half (hence g1_alphas_up). The
    1/a2_scale factor (w2_input_scale_quant) requantizes GEMM2's input. A shared
    scale passes g1_alphas as g1_alphas_up and recovers the single-scale value;
    non-gated (Relu2) has no gate half, so it is just 1/a2_scale per expert.
    """
    if activation == "situ":
        # SiTU consumes both GEMM1 scales before tanh; scale_c carries only
        # the GEMM2 input requantization factor.
        num_experts = g1_alphas.shape[0]
        return w2_input_scale_quant.to(torch.float32).expand(num_experts).contiguous()
    if is_gated:
        return (w2_input_scale_quant * g1_alphas_up).to(torch.float32)
    num_experts = g1_alphas.shape[0]
    return w2_input_scale_quant.to(torch.float32).expand(num_experts).contiguous()


def get_activation_type(activation: str, is_gated: bool = True) -> int:
    """Map SGLang activation string to FlashInfer ActivationType int value."""
    from flashinfer.fused_moe.core import ActivationType

    if is_gated:
        _ACTIVATION_STR_TO_TYPE = {
            "silu": ActivationType.Swiglu,
            "gelu": ActivationType.Geglu,
            "situ": ActivationType.Situ,
        }
    else:
        _ACTIVATION_STR_TO_TYPE = {
            "silu": ActivationType.Silu,
            "gelu": ActivationType.Gelu,
            "relu2": ActivationType.Relu2,
        }
    act = _ACTIVATION_STR_TO_TYPE.get(activation)
    if act is None:
        raise ValueError(
            f"Unsupported activation '{activation}' for TRTLLM MoE "
            f"(is_gated={is_gated}). "
            f"Expected one of {list(_ACTIVATION_STR_TO_TYPE.keys())}."
        )
    return act.value


@dataclass
class FlashInferTrtllmGenMxfp4MoeQuantInfo(MoeQuantInfo):
    """Payload for the SM100 (Blackwell) trtllm-gen MXFP4 MoE path."""

    # Packed MXFP4 weights: uint8, e2m1 x 2.
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    w13_weight_scale: torch.Tensor
    w2_weight_scale: torch.Tensor

    # fp32 per expert. GPT-OSS sets these.
    w13_weight_bias: torch.Tensor
    w2_weight_bias: torch.Tensor
    gemm1_alpha: torch.Tensor
    gemm1_beta: torch.Tensor
    gemm1_clamp_limit: torch.Tensor

    global_num_experts: int
    local_expert_offset: int
    local_num_experts: int
    intermediate_size_per_partition: int

    hidden_size: int
    flashinfer_mxfp4_moe_precision: str
    routing_bias: Optional[torch.Tensor] = None


def _fused_experts_flashinfer_mxfp4_sm100_trtllm_gen(
    dispatch_output: StandardDispatchOutput,
    quant_info: FlashInferTrtllmGenMxfp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    """SM100 (Blackwell) trtllm-gen MXFP4 fused experts."""
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    x = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output

    origin_hidden_states_dim = x.shape[-1]
    prepared_packed_topk = None
    if quant_info.flashinfer_mxfp4_moe_precision == "bf16":
        assert x.dtype == torch.bfloat16
        x_quant = x
        x_scale = None
        if quant_info.hidden_size != origin_hidden_states_dim:
            x_quant = torch.nn.functional.pad(
                x_quant,
                (0, quant_info.hidden_size - origin_hidden_states_dim),
                mode="constant",
                value=0.0,
            )
    elif quant_info.flashinfer_mxfp4_moe_precision == "default":
        # Deferred import: mxfp4 dispatches into this module, and the helper
        # stays there so its registered unit test keeps its import path.
        from sglang.srt.layers.quantization.mxfp4 import (
            _prepare_flashinfer_mxfp8_activations,
        )

        x, prepared_packed_topk, x_quant, x_scale = (
            _prepare_flashinfer_mxfp8_activations(x, quant_info.hidden_size)
        )
    else:
        raise NotImplementedError(
            f"Unsupported flashinfer_mxfp4_moe_precision: "
            f"{quant_info.flashinfer_mxfp4_moe_precision}"
        )

    assert x_quant.shape[-1] == quant_info.hidden_size
    is_standard = TopKOutputChecker.format_is_standard(topk_output)
    assert is_standard or TopKOutputChecker.format_is_bypassed(topk_output), (
        f"unsupported topk format: {topk_output.format}"
    )
    if is_standard:
        assert runner_config.activation == "situ", (
            "standard topk output only wired for the situ path"
        )
        top_k = topk_output.topk_ids.shape[1]
        router_logits = None
    else:
        top_k = topk_output.topk_config.top_k
        router_logits = topk_output.router_logits

    num_tokens = x_quant.shape[0]
    from sglang.srt.layers import zero_copy_context

    symm_output = zero_copy_context.get_moe_output_spec(
        torch.Size((num_tokens, origin_hidden_states_dim)),
        torch.bfloat16,
        x_quant.device,
    )
    if symm_output is None:
        with use_symmetric_memory(
            get_parallel().tp_group, disabled=not is_allocation_symmetric()
        ):
            symm_output = torch.empty(
                num_tokens,
                origin_hidden_states_dim,
                dtype=torch.bfloat16,
                device=x_quant.device,
            )

    if runner_config.activation == "situ":
        from flashinfer import trtllm_fp4_block_scale_moe
        from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
        from flashinfer.tllm_enums import ActivationType, RoutingMethodType

        if is_standard:
            routing = (
                prepared_packed_topk
                if prepared_packed_topk is not None
                else _get_routing_for_flashinfer_routed(topk_output)
            )
            routed_top_k = _routing_top_k(routing)

            defer_finalize = _deferred_finalize_enabled.get()
            result = trtllm_fp4_block_scale_routed_moe(
                topk_ids=routing,
                routing_bias=None,
                hidden_states=x_quant,
                hidden_states_scale=x_scale,
                gemm1_weights=quant_info.w13_weight,
                gemm1_weights_scale=quant_info.w13_weight_scale,
                gemm1_bias=None,
                gemm1_alpha=quant_info.gemm1_alpha,
                gemm1_beta=quant_info.gemm1_clamp_limit,
                gemm1_clamp_limit=None,
                gemm2_weights=quant_info.w2_weight,
                gemm2_weights_scale=quant_info.w2_weight_scale,
                gemm2_bias=None,
                output1_scale_scalar=None,
                output1_scale_gate_scalar=None,
                output2_scale_scalar=None,
                num_experts=quant_info.global_num_experts,
                top_k=routed_top_k,
                n_group=None,
                topk_group=None,
                intermediate_size=quant_info.intermediate_size_per_partition,
                local_expert_offset=quant_info.local_expert_offset,
                local_num_experts=quant_info.local_num_experts,
                routed_scaling_factor=None,
                routing_method_type=RoutingMethodType.TopK.value,
                activation_type=ActivationType.Situ.value,
                tune_max_num_tokens=next_power_of_2(x_quant.shape[0]),
                output=symm_output,
                do_finalize=not defer_finalize,
                enable_pdl=trtllm_moe_enable_pdl(x_quant.shape[0]),
            )
            if defer_finalize:
                gemm2_out, topk_weights, expanded_idx = result
                result = FlashInferTrtllmDeferredFinalizeOutput(
                    gemm2_out=gemm2_out,
                    expert_weights=topk_weights,
                    expanded_idx_to_permuted_idx=expanded_idx,
                    top_k=routed_top_k,
                )
                return StandardCombineInput(hidden_states=result)
            # The finalized kernel writes to its explicit output argument. Do
            # not propagate the FFI return tensor: some SiTU runner versions
            # return a distinct wrapper/allocation even though symm_output
            # contains the published result. Returning the destination makes
            # the pointer contract explicit for K3's zero-copy latent buffer.
            return StandardCombineInput(hidden_states=symm_output)

        trtllm_fp4_block_scale_moe(
            routing_logits=router_logits.to(torch.bfloat16).contiguous(),
            routing_bias=quant_info.routing_bias,
            hidden_states=x_quant,
            hidden_states_scale=x_scale,
            gemm1_weights=quant_info.w13_weight,
            gemm1_weights_scale=quant_info.w13_weight_scale,
            gemm1_bias=None,
            gemm1_alpha=quant_info.gemm1_alpha,
            gemm1_beta=quant_info.gemm1_clamp_limit,
            gemm1_clamp_limit=None,
            gemm2_weights=quant_info.w2_weight,
            gemm2_weights_scale=quant_info.w2_weight_scale,
            gemm2_bias=None,
            output1_scale_scalar=None,
            output1_scale_gate_scalar=None,
            output2_scale_scalar=None,
            num_experts=quant_info.global_num_experts,
            top_k=top_k,
            n_group=topk_output.topk_config.num_expert_group,
            topk_group=topk_output.topk_config.topk_group,
            intermediate_size=quant_info.intermediate_size_per_partition,
            routed_scaling_factor=(
                topk_output.topk_config.routed_scaling_factor or 1.0
            ),
            routing_method_type=RoutingMethodType.DeepSeekV3.value,
            activation_type=ActivationType.Situ.value,
            norm_topk_prob=topk_output.topk_config.renormalize,
            local_expert_offset=quant_info.local_expert_offset,
            local_num_experts=quant_info.local_num_experts,
            tune_max_num_tokens=next_power_of_2(x_quant.shape[0]),
            output=symm_output,
            enable_pdl=trtllm_moe_enable_pdl(x_quant.shape[0]),
        )
        return StandardCombineInput(hidden_states=symm_output)

    from flashinfer import trtllm_fp4_block_scale_moe

    trtllm_gen_output = trtllm_fp4_block_scale_moe(
        router_logits.to(torch.bfloat16),
        None,  # routing_bias
        x_quant,
        x_scale,
        quant_info.w13_weight,  # uint8 (e2m1 x 2)
        quant_info.w13_weight_scale,  # uint8 (e4m3 x 2)
        quant_info.w13_weight_bias,  # fp32 per expert per channel
        quant_info.gemm1_alpha,  # fp32 per expert
        quant_info.gemm1_beta,  # fp32 per expert
        quant_info.gemm1_clamp_limit,  # fp32 per expert
        quant_info.w2_weight,  # uint8 (e2m1 x 2)
        quant_info.w2_weight_scale,  # ue8m0
        quant_info.w2_weight_bias,  # fp32 per expert per channel
        None,  # output1_scale_scalar
        None,  # output1_scale_gate_scalar
        None,  # output2_scale_scalar
        quant_info.global_num_experts,
        top_k,
        None,  # n_group      # TODO: support n_group
        None,  # topk_group   # TODO: support topk_group
        quant_info.intermediate_size_per_partition,  # padded to multiple of 128
        quant_info.local_expert_offset,
        quant_info.local_num_experts,
        None,  # routed_scaling_factor
        1,  # routing_method_type, renormalize
        True,  # do finalize
        tune_max_num_tokens=next_power_of_2(x_quant.shape[0]),
        output=symm_output,
        enable_pdl=trtllm_moe_enable_pdl(x_quant.shape[0]),
    )[0]
    return StandardCombineInput(hidden_states=trtllm_gen_output)


@register_fused_func("none", "flashinfer_trtllm_routed")
def fused_experts_none_to_flashinfer_trtllm_routed(
    dispatch_output: StandardDispatchOutput,
    quant_info: MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    if isinstance(quant_info, FlashInferTrtllmFp4MoeQuantInfo):
        return fused_experts_none_to_flashinfer_trtllm_fp4(
            dispatch_output,
            quant_info,
            runner_config,
            use_routed_topk=True,
        )
    if isinstance(quant_info, FlashInferTrtllmFp8MoeQuantInfo):
        return fused_experts_none_to_flashinfer_trtllm_fp8(
            dispatch_output,
            quant_info,
            runner_config,
            use_routed_topk=True,
        )
    if isinstance(quant_info, FlashInferTrtllmBf16MoeQuantInfo):
        return fused_experts_none_to_flashinfer_trtllm_bf16(
            dispatch_output,
            quant_info,
            runner_config,
            use_routed_topk=True,
        )
    raise TypeError(
        f"Unexpected quant_info type for flashinfer_trtllm_routed: {type(quant_info)}"
    )


@register_fused_func("flashinfer", "flashinfer_trtllm")
@register_fused_func("flashinfer", "flashinfer_trtllm_routed")
def fused_experts_flashinfer_to_flashinfer_trtllm(
    dispatch_output: FlashinferDispatchOutput,
    quant_info: MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> FlashinferCombineInput:
    """Fused function for FlashInfer A2A + TRT-LLM Gen MoE.

    A2A dispatch materializes routing IDs and weights, so the regular and
    explicitly-routed backend names both enter TRT-LLM's routed kernel.
    """
    from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
        FlashinferCombineInput,
    )

    if isinstance(quant_info, FlashInferTrtllmFp4MoeQuantInfo):
        result = fused_experts_none_to_flashinfer_trtllm_fp4(
            dispatch_output,
            quant_info,
            runner_config,
            use_routed_topk=True,
        )
    elif isinstance(quant_info, FlashInferTrtllmFp8MoeQuantInfo):
        mxfp8_dispatch = (
            quant_info.use_mxfp8
            and dispatch_output.hidden_states.dtype == torch.float8_e4m3fn
        )
        if mxfp8_dispatch:
            if dispatch_output.hidden_states_scale is None:
                raise ValueError(
                    "FlashInfer A2A + TRT-LLM Gen MXFP8 MoE requires activation "
                    "scales alongside the FP8 dispatch payload."
                )
        else:
            if dispatch_output.hidden_states.dtype != torch.bfloat16:
                raise TypeError(
                    "FlashInfer A2A + TRT-LLM Gen FP8 MoE requires a BF16 "
                    f"dispatch payload, got {dispatch_output.hidden_states.dtype}."
                )
            if dispatch_output.hidden_states_scale is not None:
                raise ValueError(
                    "FlashInfer A2A + TRT-LLM Gen FP8 MoE quantizes locally; "
                    "the BF16 dispatch payload must not carry activation scales."
                )
        result = fused_experts_none_to_flashinfer_trtllm_fp8(
            dispatch_output,
            quant_info,
            runner_config,
            use_routed_topk=True,
        )
    elif isinstance(quant_info, FlashInferTrtllmBf16MoeQuantInfo):
        result = fused_experts_none_to_flashinfer_trtllm_bf16(
            dispatch_output,
            quant_info,
            runner_config,
            use_routed_topk=True,
        )
    else:
        raise TypeError(
            f"Unexpected quant_info type for flashinfer a2a + flashinfer_trtllm: {type(quant_info)}"
        )
    if (
        isinstance(quant_info, FlashInferTrtllmFp8MoeQuantInfo)
        and result.hidden_states.dtype != torch.bfloat16
    ):
        raise TypeError(
            "FlashInfer A2A + TRT-LLM Gen FP8 MoE must return a BF16 combine "
            f"payload, got {result.hidden_states.dtype}."
        )
    return FlashinferCombineInput(hidden_states=result.hidden_states)
