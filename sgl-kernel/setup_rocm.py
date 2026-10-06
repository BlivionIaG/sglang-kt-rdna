# Copyright 2025 SGLang Team. All Rights Reserved.
#
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

import os
import platform
import sys
from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

from rdna_targets import RdnaArchError, rocm_compile_profile

root = Path(__file__).parent.resolve()
arch = platform.machine().lower()


def _get_version():
    with open(root / "pyproject.toml") as f:
        for line in f:
            if line.startswith("version"):
                return line.split("=")[1].strip().strip('"')


operator_namespace = "sgl_kernel"
include_dirs = [
    root / "include",
    root / "include" / "impl",
    root / "csrc",
]

sources = [
    "csrc/common_extension_rocm.cc",
    "csrc/elementwise/activation.cu",
    "csrc/elementwise/topk.cu",
    "csrc/grammar/apply_token_bitmask_inplace_cuda.cu",
    "csrc/moe/moe_align_kernel.cu",
    "csrc/moe/moe_topk_softmax_kernels.cu",
    "csrc/moe/moe_topk_sigmoid_kernels.cu",
    "csrc/speculative/eagle_utils.cu",
    "csrc/kvcacheio/transfer.cu",
    "csrc/memory/weak_ref_tensor.cpp",
    "csrc/elementwise/pos_enc.cu",
]

cxx_flags = ["-O3"]
libraries = ["hiprtc", "amdhip64", "c10", "torch", "torch_python"]
extra_link_args = ["-Wl,-rpath,$ORIGIN/../../torch/lib", f"-L/usr/lib/{arch}-linux-gnu"]

default_target = "gfx942"
amdgpu_target = os.environ.get("AMDGPU_TARGET", default_target)

# Honor AMDGPU_TARGET when it names an RDNA arch even if a CDNA GPU is visible.
# Otherwise a gfx942 box cannot cross-compile the gfx1030/gfx1100 modules.
env_target = os.environ.get("AMDGPU_TARGET", "").split(":")[0].strip()
if env_target in ("gfx1030", "gfx1100"):
    amdgpu_target = env_target
elif torch.cuda.is_available():
    try:
        amdgpu_target = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception as e:
        print(f"Warning: Failed to detect GPU properties: {e}")
else:
    print(f"Warning: torch.cuda not available. Using default target: {amdgpu_target}")

try:
    profile = rocm_compile_profile(amdgpu_target)
except RdnaArchError as exc:
    print(exc)
    sys.exit(1)

amdgpu_target = profile["arch"]
topk_dynamic_smem_bytes = profile["topk_dynamic_smem_bytes"]

if profile["custom_allreduce"]:
    sources = [
        "csrc/allreduce/custom_all_reduce.hip",
        "csrc/allreduce/deterministic_all_reduce.hip",
        "csrc/allreduce/quick_all_reduce.cu",
        *sources,
    ]
else:
    print(
        f"{amdgpu_target}: custom/quick/deterministic all-reduce left out of this module. "
        "Tensor-parallel reductions use RCCL."
    )

rdna_flags = []
if profile["wave32"]:
    # Host launch config and device code must agree. Without this, host
    # compilation of HIP sees WARP_SIZE 64 while the gfx10/gfx11 device
    # compilation sees 32.
    rdna_flags.append("-DSGL_RDNA_WAVE32")
if not profile["enable_fp8"]:
    rdna_flags.append("-DSGL_RDNA_NO_FP8")
    rdna_flags.append("-DSGL_RDNA_NO_CUSTOM_AR")
    if profile["arch"] == "gfx1100":
        rdna_flags.append("-DSGL_ARCH_GFX1100")
    if profile["arch"] == "gfx1030":
        rdna_flags.append("-DSGL_ARCH_GFX1030")

cxx_flags = cxx_flags + rdna_flags

hipcc_flags = [
    "-DNDEBUG",
    f"-DOPERATOR_NAMESPACE={operator_namespace}",
    "-O3",
    "-Xcompiler",
    "-fPIC",
    "-std=c++17",
    f"--amdgpu-target={amdgpu_target}",
    "-DENABLE_BF16",
    *rdna_flags,
    f"-DSGL_TOPK_DYNAMIC_SMEM_BYTES={topk_dynamic_smem_bytes}",
]
if profile["enable_fp8"]:
    hipcc_flags.extend(["-DENABLE_FP8", profile["fp8_macro"]])

ext_modules = [
    CUDAExtension(
        name="sgl_kernel.common_ops",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={
            "nvcc": hipcc_flags,
            "cxx": cxx_flags,
        },
        libraries=libraries,
        extra_link_args=extra_link_args,
        py_limited_api=False,
    ),
]

setup(
    name="sgl-kernel",
    version=_get_version(),
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
