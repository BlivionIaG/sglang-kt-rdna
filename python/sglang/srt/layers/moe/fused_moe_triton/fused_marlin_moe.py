from __future__ import annotations
import functools
from typing import Optional

import torch

from sglang.srt.utils import is_cuda
import torch.nn.functional as F
import triton
import triton.language as tl
from sglang.srt.runtime_context import get_platform

_is_cuda = is_cuda()

if _is_cuda:
    from sgl_kernel import silu_and_mul
    from sgl_kernel.moe import moe_sum_reduce

    from sglang.jit_kernel.moe_wna16_marlin import moe_wna16_marlin_gemm


def get_scalar_type(num_bits: int, has_zp: bool):
    from sgl_kernel.scalar_type import scalar_types

    if has_zp:
        assert num_bits == 4
        return scalar_types.uint4
    else:
        return scalar_types.uint4b8 if num_bits == 4 else scalar_types.uint8b128


def fused_marlin_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    g_idx1: Optional[torch.Tensor] = None,
    g_idx2: Optional[torch.Tensor] = None,
    sort_indices1: Optional[torch.Tensor] = None,
    sort_indices2: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    num_bits: int = 8,
    is_k_full: bool = True,
    inplace: bool = False,
    routed_scaling_factor: float = None,
    b_q_type: Optional[torch.Tensor] = None,  # really ScalarType; Tensor for infer_schema compat
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - w1_scale (torch.Tensor): Scale to be used for w1.
    - w2_scale (torch.Tensor): Scale to be used for w2.
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - g_idx1 (Optional[torch.Tensor]): The first set of act_order indices.
    - g_idx2 (Optional[torch.Tensor]): The second set of act_order indices.
    - sort_indices1 (Optional[torch.Tensor]): The first act_order input
        permutation.
    - sort_indices2 (Optional[torch.Tensor]): The second act_order input
        permutation.
    - topk_weights (torch.Tensor): Top-k weights.
    - topk_ids (torch.Tensor): Indices of topk-k elements.
    - w1_zeros (Optional[torch.Tensor]): Optional zero points to be used for w1.
    - w2_zeros (Optional[torch.Tensor]): Optional zero points to be used for w2.
    - num_bits (bool): The number of bits in expert weights quantization.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    # Delay the import to avoid circular dependency
    from sglang.srt.layers.moe.fused_moe_triton import (
        moe_align_block_size,
        try_get_optimal_moe_config,
    )

    # Check constraints.
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    assert hidden_states.shape[1] == w1.shape[1] * 16, "Hidden size mismatch w1"
    assert hidden_states.shape[1] == w2.shape[2] // (
        num_bits // 2
    ), "Hidden size mismatch w2"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float16, torch.bfloat16]
    assert (
        hidden_states.dtype == w1_scale.dtype
    ), f"moe_wna16_marlin_gemm assumes hidden_states.dtype ({hidden_states.dtype}) == w1_scale.dtype ({w1_scale.dtype})"
    assert (
        hidden_states.dtype == w2_scale.dtype
    ), f"moe_wna16_marlin_gemm assumes hidden_states.dtype ({hidden_states.dtype}) == w2_scale.dtype ({w2_scale.dtype})"
    assert num_bits in [4, 8]

    M, K = hidden_states.shape
    E = w1.shape[0]
    N = w2.shape[1] * 16
    topk = topk_ids.shape[1]

    # KT hybrid dispatch marks CPU-routed experts as -1 after remapping.
    # Marlin expects expert ids in [0, E), so sanitize invalid ids and
    # zero out their corresponding weights (GPU path should ignore them).
    invalid_topk = (topk_ids < 0) | (topk_ids >= E)
    topk_ids = torch.where(invalid_topk, torch.zeros_like(topk_ids), topk_ids)
    topk_weights = torch.where(invalid_topk, torch.zeros_like(topk_weights), topk_weights)

    # Early return when no experts on GPU (e.g. kt-num-gpu-experts=0)
    # or no tokens to process. Avoids kernel launch failures with empty tensors.
    if E == 0 or M == 0:
        return torch.zeros_like(hidden_states)

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.shape,
        w2.shape,
        topk_ids.shape[1],
        None,
        is_marlin=True,
    )
    config = get_config_func(M)

    block_size_m = config["BLOCK_SIZE_M"]

    if global_num_experts == -1:
        global_num_experts = E
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size_m, global_num_experts
    )

    if workspace is None:
        max_workspace_size = (max(2 * N, K) // 64) * (
            sorted_token_ids.size(0) // block_size_m
        )
        device = hidden_states.device
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        max_workspace_size = min(max_workspace_size, sms * 4)
        workspace = torch.zeros(
            max_workspace_size, dtype=torch.int, device=device, requires_grad=False
        )

    scalar_type1 = b_q_type if b_q_type is not None else get_scalar_type(num_bits, w1_zeros is not None)
    scalar_type2 = b_q_type if b_q_type is not None else get_scalar_type(num_bits, w2_zeros is not None)

    intermediate_cache2 = torch.empty(
        (M * topk_ids.shape[1], N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache13 = torch.empty(
        (M * topk_ids.shape[1] * max(2 * N, K),),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = intermediate_cache13[: M * topk_ids.shape[1] * 2 * N]
    intermediate_cache1 = intermediate_cache1.view(-1, 2 * N)
    intermediate_cache3 = intermediate_cache13[: M * topk_ids.shape[1] * K]
    intermediate_cache3 = intermediate_cache3.view(-1, K)

    use_atomic_add = (
        hidden_states.dtype == torch.half
        or torch.cuda.get_device_capability(hidden_states.device)[0] >= 9
    )

    intermediate_cache1 = moe_wna16_marlin_gemm(
        hidden_states,
        intermediate_cache1,
        w1,
        None,  # b_bias_or_none
        w1_scale,
        None,  # global_scale_or_none
        w1_zeros,
        g_idx1,
        sort_indices1,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=topk,
        mul_topk_weights=False,
        is_ep=expert_map is not None,
        b_q_type=scalar_type1,
        size_m=M,
        size_n=2 * N,
        size_k=K,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=True,
        is_zp_float=False,
    )

    silu_and_mul(intermediate_cache1.view(-1, 2 * N), intermediate_cache2)

    if expert_map is not None:
        intermediate_cache3.zero_()

    intermediate_cache3 = moe_wna16_marlin_gemm(
        intermediate_cache2,
        intermediate_cache3,
        w2,
        None,  # b_bias_or_none
        w2_scale,
        None,  # global_scale_or_none
        w2_zeros,
        g_idx2,
        sort_indices2,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=1,
        mul_topk_weights=True,
        is_ep=expert_map is not None,
        b_q_type=scalar_type2,
        size_m=M * topk,
        size_n=K,
        size_k=N,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=True,
        is_zp_float=False,
    ).view(-1, topk, K)

    if routed_scaling_factor is None:
        routed_scaling_factor = 1.0

    output = hidden_states if inplace else torch.empty_like(hidden_states)
    moe_sum_reduce(
        intermediate_cache3,
        output,
        routed_scaling_factor,
    )
    return output


def fused_marlin_moe_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    g_idx1: Optional[torch.Tensor] = None,
    g_idx2: Optional[torch.Tensor] = None,
    sort_indices1: Optional[torch.Tensor] = None,
    sort_indices2: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    num_bits: int = 8,
    is_k_full: bool = True,
    inplace: bool = False,
    routed_scaling_factor: float = None,
    b_q_type: Optional[torch.Tensor] = None,  # really ScalarType; Tensor for infer_schema compat
) -> torch.Tensor:
    return torch.empty_like(hidden_states)

from sglang.srt.utils import direct_register_custom_op, supports_custom_op

if supports_custom_op():
    direct_register_custom_op(
        op_name="fused_marlin_moe",
        op_func=fused_marlin_moe,
        mutates_args=[],
        fake_impl=fused_marlin_moe_fake,
    )


# --- imported with the qwen4 subsystem (sgl-project/sglang) ---


@triton.jit
def _tl_tanh(x):
    return 2.0 * tl.sigmoid(2.0 * x) - 1.0


@triton.jit
def _situ_and_mul_kernel(
    x_ptr,  # [M, 2N] gate;up halves (non-interleaved)
    out_ptr,  # [M, N]
    N,
    situ_beta,
    linear_beta,
    stride_xm,
    stride_om,
    BLOCK_N: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    base = x_ptr + pid_m * stride_xm
    gate = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(base + N + offs, mask=mask, other=0.0).to(tl.float32)
    gate = situ_beta * _tl_tanh(gate / situ_beta) * tl.sigmoid(gate)
    if HAS_LINEAR_BETA:
        up = linear_beta * _tl_tanh(up / linear_beta)
    out = gate * up
    tl.store(
        out_ptr + pid_m * stride_om + offs,
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def situ_and_mul(
    output: torch.Tensor,
    x: torch.Tensor,
    situ_beta: float,
    linear_beta: Optional[float],
) -> None:
    """SiTU gated activation (Kimi K3), fused into one elementwise kernel:
    out = situ_beta*tanh(gate/situ_beta)*sigmoid(gate) * linear_beta*tanh(up/linear_beta)
    where x = [gate; up] halves along the last dim.
    """
    M, N2 = x.shape
    N = N2 // 2
    assert output.shape == (M, N)
    BLOCK_N = 1024
    grid = (M, triton.cdiv(N, BLOCK_N))
    _situ_and_mul_kernel[grid](
        x,
        output,
        N,
        float(situ_beta),
        float(linear_beta) if linear_beta is not None else 0.0,
        x.stride(0),
        output.stride(0),
        BLOCK_N=BLOCK_N,
        HAS_LINEAR_BETA=linear_beta is not None,
    )


def swiglu_limit_func(
    output: torch.Tensor,
    input: torch.Tensor,  # first half is gate, second half is up
    swiglu_limit: float = 0.0,
) -> None:
    d = input.shape[1] // 2
    if (
        _is_cuda
        and get_platform().is_sm90
        and input.is_cuda
        and input.dtype in (torch.bfloat16, torch.float16)
        and d % 16 == 0
        and input.is_contiguous()
        and output.is_contiguous()
    ):
        silu_and_mul_with_activation_rounding(input, output, clamp_limit=swiglu_limit)
        return
    gate = input[:, :d]
    up = input[:, d:]

    if swiglu_limit > 0:
        gate = torch.clamp(gate, max=swiglu_limit)
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)

    output.copy_(F.silu(gate) * up)


def swiglu_gpt_oss_sigmoid_alpha_contiguous(
    output: torch.Tensor,
    input: torch.Tensor,  # first half is gate, second half is up
    gemm1_alpha: float,
    gemm1_limit: float,
) -> None:
    d = input.shape[1] // 2
    gate = input[:, :d].clamp(max=gemm1_limit)
    up = input[:, d:].clamp(min=-gemm1_limit, max=gemm1_limit)
    output.copy_(gate * torch.sigmoid(gate * gemm1_alpha) * (up + 1))
