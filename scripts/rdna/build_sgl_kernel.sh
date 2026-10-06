#!/usr/bin/env bash
# Build the ROCm sgl-kernel module for one arch.
# AMDGPU_TARGET=gfx1030 or gfx1100. A visible CDNA GPU is ignored when the
# variable names an RDNA arch, so a gfx942 box can cross-compile.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARCH="${AMDGPU_TARGET:-}"
if [[ -z "${ARCH}" ]]; then
  if command -v rocminfo >/dev/null 2>&1; then
    ARCH="$(rocminfo | awk '/Name:[[:space:]]+gfx/ {print $2; exit}')"
  fi
fi
if [[ -z "${ARCH}" ]]; then
  echo "Set AMDGPU_TARGET to gfx1030 or gfx1100." >&2
  exit 1
fi
ARCH="${ARCH%%:*}"
export AMDGPU_TARGET="${ARCH}"

cd "${ROOT}/sgl-kernel"
python3 -m pip install ninja setuptools wheel
python3 setup_rocm.py build_ext --inplace
find python -name 'common_ops*.so' -print
