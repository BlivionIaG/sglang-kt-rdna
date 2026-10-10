#!/usr/bin/env bash
# Point a ROCm 7.14 pip-SDK image at rocThrust / hipSPARSE headers.
#
# rocm/pytorch:rocm7.14.x ships hipcc from _rocm_sdk_core. That prefix has
# the HIP runtime but not the math-library headers PyTorch includes
# (thrust/complex.h, hipsparse/hipsparse.h, hipCUB, rocPRIM). Those live
# in rocm-sdk-devel. This script installs the devel wheel that matches
# the image's rocm-sdk-core, expands it, and writes a shell env file.
set -euo pipefail

INDEX_URL="${ROCM_PIP_INDEX:-https://repo.amd.com/rocm/whl-multi-arch/}"
ENV_FILE="${RDNA_ROCM_ENV_FILE:-/tmp/rdna-rocm-env.sh}"

python - "$INDEX_URL" "$ENV_FILE" <<'PY'
import importlib.metadata as md
import shutil
import subprocess
import sys
from pathlib import Path

index_url, env_file = sys.argv[1], sys.argv[2]


def version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


core = version("rocm-sdk-core")
devel = version("rocm-sdk-devel")
meta = version("rocm")
print(f"rocm={meta} rocm-sdk-core={core} rocm-sdk-devel={devel}", flush=True)
if core is None:
    sys.exit("rocm-sdk-core is not installed in this image")


def pip_install(*specs):
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--index-url",
        index_url,
        *specs,
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


if meta is None or shutil.which("rocm-sdk") is None:
    pip_install(f"rocm=={core}")
if devel is None:
    pip_install(f"rocm-sdk-devel=={core}")

root = subprocess.check_output(
    ["rocm-sdk", "path", "--root"], text=True
).strip().splitlines()[-1]
print(f"ROCM_HOME={root}", flush=True)
root_path = Path(root)
required = [
    "include/thrust/complex.h",
    "include/hipsparse/hipsparse.h",
    "include/hipcub/hipcub.hpp",
    "include/rocprim/rocprim.hpp",
    "bin/hipcc",
]
missing = [rel for rel in required if not (root_path / rel).exists()]
lib_dir = root_path / "lib"
hip_libs = list(lib_dir.glob("libamdhip64.so*")) if lib_dir.is_dir() else []
rtc_libs = list(lib_dir.glob("libhiprtc.so*")) if lib_dir.is_dir() else []
if not hip_libs:
    missing.append("lib/libamdhip64.so*")
if not rtc_libs:
    missing.append("lib/libhiprtc.so*")
if missing:
    if lib_dir.is_dir():
        names = sorted(p.name for p in lib_dir.iterdir())
        print("lib entries:", ", ".join(names[:80]), flush=True)
    sys.exit(f"{root} is missing {missing}")

inc = str(root_path / "include")
lib = str(lib_dir)
Path(env_file).write_text(
    "\n".join(
        [
            f'export ROCM_HOME="{root}"',
            f'export ROCM_PATH="{root}"',
            f'export HIP_PATH="{root}"',
            f'export PATH="{root}/bin:$PATH"',
            f'export CPATH="{inc}${{CPATH:+:$CPATH}}"',
            f'export CPLUS_INCLUDE_PATH="{inc}${{CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}}"',
            f'export LIBRARY_PATH="{lib}${{LIBRARY_PATH:+:$LIBRARY_PATH}}"',
            f'export LD_LIBRARY_PATH="{lib}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"',
            "",
        ]
    )
)
print(f"wrote {env_file}", flush=True)
PY
