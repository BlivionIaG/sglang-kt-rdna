# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Hook points for RDNA kernels that live in the sibling repo.

This package does not vendor those sources. gfx1100 WMMA variants must be a
separate object in that repo and must not be imported on gfx1030.

Sibling: https://github.com/BlivionIaG/ktransformers-rdna

Expected optional entry points, once that repo publishes them:

- ``kt_rdna.attention.FaRdna2AttnBackend`` for ``fa_rdna2`` (GQA, D=128/256)
- ``kt_rdna.moe.RDNAW4A16MoEMethod`` for ``moe_q_gemm_rdna2``
- ``kt_rdna.linear.q_gemm_rdna2`` and ``kt_rdna.linear.gemv_f16_rdna2``

Select the attention hook with ``--attention-backend fa_rdna2``. Phase 0 does
not select it: gfx1100 uses Triton attention and gfx1030 uses torch_native.
"""

from __future__ import annotations

import importlib
from typing import Any

SIBLING_REPO = "https://github.com/BlivionIaG/ktransformers-rdna"

# Import path, symbol, and the kernel it stands for.
RDNA_KERNEL_HOOKS = (
    ("kt_rdna.attention", "FaRdna2AttnBackend", "fa_rdna2"),
    ("kt_rdna.moe", "RDNAW4A16MoEMethod", "moe_q_gemm_rdna2"),
    ("kt_rdna.linear", "q_gemm_rdna2", "q_gemm_rdna2"),
    ("kt_rdna.linear", "gemv_f16_rdna2", "gemv_f16_rdna2"),
)


def rdna_kernel_status() -> dict[str, str]:
    """Report which sibling-repo symbols import. Missing is the expected state today."""

    status: dict[str, str] = {}
    for module_name, symbol, kernel in RDNA_KERNEL_HOOKS:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            status[kernel] = f"absent ({exc})"
            continue
        if hasattr(module, symbol):
            status[kernel] = f"present ({module_name}.{symbol})"
        else:
            status[kernel] = f"module imported but {symbol} is missing"
    return status


def load_fa_rdna2_backend(runner: Any) -> Any:
    try:
        from kt_rdna.attention import FaRdna2AttnBackend
    except ImportError as exc:
        raise NotImplementedError(
            "fa_rdna2 is the gfx1030 GQA attention backend from "
            f"{SIBLING_REPO}. This SGLang fork does not embed that kernel. "
            "Phase 0 uses --attention-backend torch_native on gfx1030 and "
            "triton on gfx1100. "
            f"Import failed: {exc}"
        ) from exc
    return FaRdna2AttnBackend(runner)


def load_rdna_w4a16_moe_method(*args: Any, **kwargs: Any) -> Any:
    try:
        from kt_rdna.moe import RDNAW4A16MoEMethod
    except ImportError as exc:
        raise NotImplementedError(
            "moe_q_gemm_rdna2 is the gfx1030 INT4 expert GEMM from "
            f"{SIBLING_REPO}. With --kt-num-gpu-experts 0 every routed expert "
            "stays on kt-kernel and this method is not called. "
            f"Import failed: {exc}"
        ) from exc
    return RDNAW4A16MoEMethod(*args, **kwargs)
