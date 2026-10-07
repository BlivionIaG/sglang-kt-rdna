from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Optional

from sglang.srt.utils.common import rank0_log

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class LinearAttnKernelBackend(Enum):
    TRITON = "triton"
    CUTEDSL = "cutedsl"
    NV_CUTEDSL = "nv_cutedsl"
    FLASHINFER = "flashinfer"
    FLASHKDA = "flashkda"
    NVIDIA_KDA = "nvidia_kda"
    PTX_KDA = "ptx_kda"
    HELION = "helion"
    INTEL_XPU = "intel_xpu"
    CUSTOM = "custom"

    @classmethod
    def _missing_(cls, value):
        # Unknown backend names fall back to CUSTOM rather than raising, so a
        # newer name delivered by a model config cannot abort startup.
        return cls.CUSTOM

    def is_triton(self):
        return self == LinearAttnKernelBackend.TRITON

    def is_cutedsl(self):
        return self == LinearAttnKernelBackend.CUTEDSL

    def is_nv_cutedsl(self):
        return self == LinearAttnKernelBackend.NV_CUTEDSL

    def is_flashinfer(self):
        return self == LinearAttnKernelBackend.FLASHINFER

    def is_flashkda(self):
        return self == LinearAttnKernelBackend.FLASHKDA

    def is_nvidia_kda(self):
        return self == LinearAttnKernelBackend.NVIDIA_KDA

    def is_ptx_kda(self):
        return self == LinearAttnKernelBackend.PTX_KDA

    def is_helion(self):
        return self == LinearAttnKernelBackend.HELION

    def is_intel_xpu(self):
        return self == LinearAttnKernelBackend.INTEL_XPU

    def is_custom(self):
        return self == LinearAttnKernelBackend.CUSTOM


LINEAR_ATTN_DECODE_BACKEND: Optional[LinearAttnKernelBackend] = None
LINEAR_ATTN_PREFILL_BACKEND: Optional[LinearAttnKernelBackend] = None


def initialize_linear_attn_config(server_args: ServerArgs):
    global LINEAR_ATTN_DECODE_BACKEND
    global LINEAR_ATTN_PREFILL_BACKEND

    base = server_args.linear_attn_backend
    decode = server_args.linear_attn_decode_backend or base
    prefill = server_args.linear_attn_prefill_backend or base

    LINEAR_ATTN_DECODE_BACKEND = LinearAttnKernelBackend(decode)
    LINEAR_ATTN_PREFILL_BACKEND = LinearAttnKernelBackend(prefill)
    rank0_log(
        f"Linear attention kernel backend: "
        f"decode={LINEAR_ATTN_DECODE_BACKEND.value}, "
        f"prefill={LINEAR_ATTN_PREFILL_BACKEND.value}"
    )


def get_linear_attn_decode_backend() -> LinearAttnKernelBackend:
    global LINEAR_ATTN_DECODE_BACKEND
    if LINEAR_ATTN_DECODE_BACKEND is None:
        logger.warning(
            "LINEAR_ATTN_DECODE_BACKEND is not initialized, using triton backend"
        )
        LINEAR_ATTN_DECODE_BACKEND = LinearAttnKernelBackend.TRITON
    return LINEAR_ATTN_DECODE_BACKEND


def get_linear_attn_prefill_backend() -> LinearAttnKernelBackend:
    global LINEAR_ATTN_PREFILL_BACKEND
    if LINEAR_ATTN_PREFILL_BACKEND is None:
        logger.warning(
            "LINEAR_ATTN_PREFILL_BACKEND is not initialized, using triton backend"
        )
        LINEAR_ATTN_PREFILL_BACKEND = LinearAttnKernelBackend.TRITON
    return LINEAR_ATTN_PREFILL_BACKEND


# --- imported with the qwen4 subsystem (sgl-project/sglang) -----------------
# The MTP / target_verify intermediate-state row selection. Kept additive: this
# fork's GDN/KDA backends still call initialize_linear_attn_config and the two
# get_linear_attn_*_backend accessors above, so upstream's version of this file
# cannot simply replace ours.


def pp_spec_stable_rows_enabled() -> bool:
    """Whether the PP-stable intermediate-state row table is in use."""
    from sglang.srt.environ import envs

    return envs.SGLANG_ENABLE_PP_SPEC.get()


def select_verify_intermediate_state_indices(
    default_indices, req_pool_indices, valid, pool_size: int
):
    if not pp_spec_stable_rows_enabled():
        return default_indices

    import torch

    req_rows = req_pool_indices[: valid.shape[0]]
    return torch.where(valid, req_rows, torch.full_like(req_rows, pool_size)).to(
        torch.int32
    )
