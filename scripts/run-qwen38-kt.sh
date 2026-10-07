#!/usr/bin/env bash
# Serve nvidia/Qwen3.8-Flash-Next-NVFP4 through the kt-vendored sglang fork on par1-llm1.
#
# This is the recipe the port reached its furthest point with: the full 48-layer x 512-expert
# KT mask set built, the staging buffer created, and the CPU expert loader running on 16 AMX
# threads. It stops (on the last observed run) during the per-layer weight walk.
#
# PREREQUISITES on the host -- these are NOT optional, each was needed to get this far:
#   * torch 2.14.1+cu130, flashinfer_python/cubin/jit_cache 0.7.0.post1, transformers 5.17.0,
#     sglang-kernel 0.4.9 (NOT sgl-kernel 0.3.21 -- that wheel links against torch 2.9 and its
#     common_ops.abi3.so fails to import with "undefined symbol c10_cuda_check_implementation")
#   * the flashinfer packages are NOT on PyPI above 0.6.13:
#       pip install flashinfer-cubin==0.7.0.post1     --index-url https://flashinfer.ai/whl
#       pip install flashinfer-jit-cache==0.7.0.post1+cu130 --index-url https://flashinfer.ai/whl/cu130
#   * this fork installed into the venv (python/sglang over site-packages/sglang)
#   * the Qwen3.6 server STOPPED -- it holds ~9 GB of the 16 GB card. Restore recipe lives at
#     ~/serve-backups/serve-stock.sh on the host.
#
# OPEN QUESTION this script exists to settle: the last runs died during the per-layer weight
# walk with SIGKILL from the kernel OOM killer. Two explanations have been RETRACTED already
# (physical RAM exhaustion; vm.overcommit_memory=0). The untested candidate is
# LimitMEMLOCK=8388608 (8 MiB) versus the cudaHostRegister calls kt_ep_wrapper.py:1197 makes.
# Run once as-is and once with `ulimit -l unlimited` in the launching shell, then compare
# which layer each reaches. Do NOT report a root cause until one of those runs completes.
set -u

VENV="${VENV:-$HOME/Projects/kt071-venv}"
KT="${KT:-$HOME/Projects/ktransformers-0.7.1}"
MODEL="${MODEL:-/home/kletorch/models/qwen38-stage}"
PORT="${PORT:-8210}"

source "$VENV/bin/activate"
cd "$KT" || exit 1

export SGLANG_DISABLE_CUDNN_CHECK=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export KT_MXFP4_BACKEND=avx2
export USE_NUMA=1
export TORCH_CUDA_ARCH_LIST="12.0"
export SGLANG_ENABLE_JIT_DEEPGEMM=0

echo "memlock: $(ulimit -l)   (8 = the value under test; unlimited = the fix candidate)"

exec python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --kt-weight-path "$MODEL" \
  --ple-offload-embedding --ple-offload-backend pinned \
  --quantization modelopt_mixed \
  --kt-num-gpu-experts 0 --kt-cpuinfer 16 --kt-threadpool-count 1 --kt-method NVFP4 \
  --attention-backend triton --disable-cuda-graph --mem-fraction-static 0.55 \
  --max-running-requests 2 \
  --host 127.0.0.1 --port "$PORT" \
  --trust-remote-code --disable-radix-cache
