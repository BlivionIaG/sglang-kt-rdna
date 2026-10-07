from __future__ import annotations
import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import get_device_sm, is_blackwell_supported

logger = logging.getLogger(__name__)


# Capabilities where DeepGEMM's wgmma / tcgen05.mma kernels actually run.
# Hopper (SM_90a) + DC Blackwell (SM_100f, SM_103a). NOT consumer Blackwell
# (SM_120) which lacks both, NOT Ada (SM_89) / Ampere (SM_8x). Outside this
# set, importing DeepGEMM may succeed but invoking its kernels raises
# "Unsupported architecture" (verified on RTX 5090: deep_gemm.attention's
# get_paged_mqa_logits_metadata crashes during CUDA Graph capture).
DEEPGEMM_CAPS = {(9, 0), (10, 0), (10, 3)}


def _compute_enable_deep_gemm():
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability() not in DEEPGEMM_CAPS:
        return False

    try:
        import deep_gemm  # noqa: F401
    except ImportError:
        return False

    return envs.SGLANG_ENABLE_JIT_DEEPGEMM.get()


ENABLE_JIT_DEEPGEMM = _compute_enable_deep_gemm()

DEEPGEMM_BLACKWELL = ENABLE_JIT_DEEPGEMM and is_blackwell_supported()
DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL


# --- imported with the qwen4 subsystem (sgl-project/sglang) ---


def _sm120_deep_gemm_apis_available() -> bool:
    try:
        import deep_gemm
    except (ImportError, OSError, RuntimeError):
        return False
    return all(
        callable(getattr(deep_gemm, name, None))
        for name in (
            "fp8_einsum",
            "m_grouped_fp8_fp4_gemm_nt_contiguous",
            "transform_sf_into_required_layout",
        )
    )


def _supports_paged_sparse_mqa_logits() -> bool:
    if not DEEPGEMM_BLACKWELL:
        return False
    import deep_gemm

    return all(
        callable(getattr(deep_gemm, name, None))
        for name in (
            "get_paged_sparse_mqa_logits_metadata",
            "fp8_fp4_paged_sparse_mqa_logits",
        )
    )
