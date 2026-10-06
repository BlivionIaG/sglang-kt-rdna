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
"""Arch policy for the ROCm sgl-kernel build.

gfx1030 (RDNA2) and gfx1100 (RDNA3) are separate modules. gfx1100 may grow
WMMA translation units in https://github.com/BlivionIaG/ktransformers-rdna;
those objects are not part of this package and must never be linked into the
gfx1030 module. CDNA targets gfx942 and gfx950 keep their existing flags.
"""

from __future__ import annotations

CDNA_ARCHS = ("gfx942", "gfx950")
RDNA_ARCHS = ("gfx1030", "gfx1100")
SUPPORTED_ARCHS = CDNA_ARCHS + RDNA_ARCHS

# 64 KiB LDS per workgroup on these parts. Leave room for static shared
# allocations inside the TopK kernel.
TOPK_DYNAMIC_SMEM_48K = 48 * 1024
# gfx950 keeps the budget this tree already used for MI350.
TOPK_DYNAMIC_SMEM_GFX950 = 32 * 1024 * 4


class RdnaArchError(ValueError):
    """Raised when AMDGPU_TARGET is missing or not one this build accepts."""


def normalize_arch(name: str) -> str:
    return name.split(":")[0].strip()


def split_amdgpu_targets(raw: str) -> list[str]:
    return [normalize_arch(part) for part in raw.replace(",", ";").split(";") if part.strip()]


def validate_amdgpu_targets(raw: str) -> list[str]:
    """Return the arch list or raise RdnaArchError with the gate text.

    A single RDNA arch is allowed. gfx942 and gfx950 may still be combined,
    which is the existing CDNA wheel. gfx1030 must not share a module with
    gfx1100 or with a CDNA arch.
    """

    archs = split_amdgpu_targets(raw)
    if not archs:
        raise RdnaArchError(
            "Warning: Unsupported GPU architecture detected ''. "
            f"Expected one of {', '.join(SUPPORTED_ARCHS)}."
        )
    unknown = [arch for arch in archs if arch not in SUPPORTED_ARCHS]
    if unknown:
        shown = unknown[0] if len(archs) == 1 else raw
        raise RdnaArchError(
            f"Warning: Unsupported GPU architecture detected '{shown}'. "
            f"Expected one of {', '.join(SUPPORTED_ARCHS)}."
        )
    rdna = [arch for arch in archs if arch in RDNA_ARCHS]
    if len(rdna) > 1 or (rdna and len(archs) != 1):
        raise RdnaArchError(
            "Refusing to build "
            + " and ".join(archs)
            + " into one sgl-kernel module. "
            "gfx1030 and gfx1100 are separate objects so WMMA code cannot be loaded on gfx1030."
        )
    return archs


def rocm_compile_profile(arch: str) -> dict:
    """Flags for one validated arch. ``arch`` is a single gfx name."""

    archs = validate_amdgpu_targets(arch)
    name = archs[0]
    is_rdna = name in RDNA_ARCHS
    if name == "gfx950":
        topk_bytes = TOPK_DYNAMIC_SMEM_GFX950
    else:
        # gfx942, gfx1030, gfx1100: 64 KiB LDS per workgroup.
        topk_bytes = TOPK_DYNAMIC_SMEM_48K
    fp8_macro = None
    if not is_rdna:
        fp8_macro = "-DHIP_FP8_TYPE_FNUZ" if name == "gfx942" else "-DHIP_FP8_TYPE_E4M3"
    return {
        "arch": name,
        "is_rdna": is_rdna,
        "enable_fp8": not is_rdna,
        "fp8_macro": fp8_macro,
        "topk_dynamic_smem_bytes": topk_bytes,
        "custom_allreduce": not is_rdna,
        "wave32": is_rdna,
    }
