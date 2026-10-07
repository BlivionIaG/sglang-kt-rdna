from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    RunnerInput,
    RunnerOutput,
    register_fused_func,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
import triton
import triton.language as tl

if TYPE_CHECKING:
    from sgl_kernel.scalar_type import ScalarType

    from sglang.srt.layers.moe.token_dispatcher import (
        StandardCombineInput,
        StandardDispatchOutput,
    )

MARLIN_MOE_WORKSPACE: Optional[torch.Tensor] = None


@dataclass
class MarlinRunnerInput(RunnerInput):
    """Input bundle passed to the Marlin runner core."""

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.MARLIN


@dataclass
class MarlinRunnerOutput(RunnerOutput):
    """Output bundle returned from the Marlin runner core."""

    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.MARLIN


@dataclass
class MarlinMoeQuantInfo(MoeQuantInfo):
    """Quantization payload consumed by the Marlin backend."""

    w13_qweight: torch.Tensor
    w2_qweight: torch.Tensor
    w13_scales: torch.Tensor
    w2_scales: torch.Tensor
    w13_g_idx_sort_indices: Optional[torch.Tensor]
    w2_g_idx_sort_indices: Optional[torch.Tensor]
    weight_bits: int

    # GPTQ specific (Optional)
    w13_g_idx: Optional[torch.Tensor] = None
    w2_g_idx: Optional[torch.Tensor] = None
    is_k_full: bool = True

    # AWQ specific (Optional)
    w13_qzeros: Optional[torch.Tensor] = None
    w2_qzeros: Optional[torch.Tensor] = None

    # Optional
    expert_map: Optional[torch.Tensor] = None
    # Explicit scalar type override — when set, fused_marlin_moe uses it
    # directly instead of inferring from weight_bits.  Needed for FP8
    # Marlin (float8_e4m3fn) vs INT8 Marlin (uint8b128).
    b_q_type: Optional["ScalarType"] = None


@register_fused_func("none", "marlin")
def fused_experts_none_to_marlin(
    dispatch_output: StandardDispatchOutput,
    quant_info: MarlinMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    global MARLIN_MOE_WORKSPACE
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output

    assert runner_config.activation == "silu", "Only SiLU activation is supported."

    if (
        MARLIN_MOE_WORKSPACE is None
        or MARLIN_MOE_WORKSPACE.device != hidden_states.device
    ):
        MARLIN_MOE_WORKSPACE = marlin_make_workspace(
            hidden_states.device, max_blocks_per_sm=4
        )

    output = fused_marlin_moe(
        hidden_states=hidden_states,
        w1=quant_info.w13_qweight,
        w2=quant_info.w2_qweight,
        w1_scale=quant_info.w13_scales,
        w2_scale=quant_info.w2_scales,
        gating_output=topk_output.router_logits,
        topk_weights=topk_output.topk_weights,
        topk_ids=topk_output.topk_ids,
        expert_map=quant_info.expert_map,
        g_idx1=quant_info.w13_g_idx,
        g_idx2=quant_info.w2_g_idx,
        sort_indices1=quant_info.w13_g_idx_sort_indices,
        sort_indices2=quant_info.w2_g_idx_sort_indices,
        w1_zeros=quant_info.w13_qzeros,
        w2_zeros=quant_info.w2_qzeros,
        workspace=MARLIN_MOE_WORKSPACE,
        num_bits=quant_info.weight_bits,
        is_k_full=quant_info.is_k_full,
        inplace=runner_config.inplace,
        routed_scaling_factor=runner_config.routed_scaling_factor,
        b_q_type=quant_info.b_q_type,
    ).to(hidden_states.dtype)

    return StandardCombineInput(
        hidden_states=output,
    )


# --- imported with the qwen4 subsystem (sgl-project/sglang) ---


@triton.jit
def _unpack_packed_topk_kernel(packed_ptr, ids_ptr, w_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    packed = tl.load(
        packed_ptr + offs, mask=mask
    )  # int32 (id << 16) | bf16-weight-bits
    tl.store(ids_ptr + offs, packed >> 16, mask=mask)  # expert id (high 16 bits)
    wbits = (packed & 0xFFFF).to(tl.int16)  # bf16 weight bits (low 16 bits)
    tl.store(
        w_ptr + offs, wbits.to(tl.bfloat16, bitcast=True).to(tl.float32), mask=mask
    )


def _fused_unpack_packed_topk(packed: torch.Tensor):
    """Single-launch inverse of the fused topk pack ((id << 16) | bf16-weight-bits).

    Returns (topk_ids int32, topk_weights float32). Collapses the ~5 elementwise
    ops (shift / mask / int16 / bitcast / cast) the torch reference emits per call
    into one Triton launch. Mirrors _pack_topk_kernel (ops/lora/moe/trtllm_lora_temp/topk_pack).
    """
    packed = packed.contiguous()
    ids = torch.empty_like(packed, dtype=torch.int32)
    w = torch.empty(packed.shape, dtype=torch.float32, device=packed.device)
    numel = packed.numel()
    if numel:
        BLOCK = 1024
        _unpack_packed_topk_kernel[(triton.cdiv(numel, BLOCK),)](
            packed, ids, w, numel, BLOCK=BLOCK
        )
    return ids, w
