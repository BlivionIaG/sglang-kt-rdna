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
"""Device-generic copy of the fused mamba state scatter.

The Triton kernel in ``mamba_state_scatter_triton`` is the NVIDIA and gfx1100
path. gfx1030 uses this implementation so the GDN / Mamba recurrent state
stays on the same device as attention without requiring Triton to compile
this particular kernel. GDN itself still uses its Triton kernels.
"""

from __future__ import annotations

import torch


def torch_mamba_state_scatter_with_mask(
    dst: torch.Tensor,
    src: torch.Tensor,
    dst_indices_raw: torch.Tensor,
    step_indices_raw: torch.Tensor,
) -> None:
    """dst[:, dst_i] = src[:, req_i, step_i] for entries with step >= 0.

    ``dst`` and ``src`` stay on their own device. Indices are moved there.
    """

    if dst.device != src.device:
        raise ValueError(
            f"dst and src must be on the same device. {dst.device=} {src.device=}"
        )
    if dst_indices_raw.shape != step_indices_raw.shape or dst_indices_raw.ndim != 1:
        raise ValueError(
            f"indices must be 1D and the same length: {dst_indices_raw.shape=} {step_indices_raw.shape=}"
        )
    if dst.ndim < 2 or src.ndim < 3:
        raise ValueError(f"Unexpected tensor ranks: {dst.ndim=} {src.ndim=}")
    if dst.shape[0] != src.shape[0] or dst.shape[2:] != src.shape[3:]:
        raise ValueError(
            f"state shape mismatch: dst {tuple(dst.shape)} src {tuple(src.shape)}"
        )

    device = dst.device
    dst_indices = dst_indices_raw.to(device=device, dtype=torch.long)
    step_indices = step_indices_raw.to(device=device, dtype=torch.long)
    req = torch.arange(step_indices.shape[0], device=device)
    valid = step_indices >= 0
    valid = valid & (dst_indices >= 0) & (dst_indices < dst.shape[1])
    valid = valid & (req < src.shape[1]) & (step_indices < src.shape[2])
    if not torch.any(valid):
        return
    dst[:, dst_indices[valid]] = src[:, req[valid], step_indices[valid]]
