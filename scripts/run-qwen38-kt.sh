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

# ---------------------------------------------------------------------------
# MEMORY ISOLATION -- added 2026-10-07 after the load reproducibly WEDGED the host.
#
# Two launches (13:08, 13:35) each drove par1-llm1 into a state where NOTHING could fork:
# ping and ARP fine, :22 accepting TCP, but the sshd banner never arrived and every service
# port was closed. The box recovered only when the stuck process exited (~24 min later,
# uptime unchanged at 41 days). Earlier launches had been SIGKILLed by the kernel OOM killer
# at ~20 GB anon-rss with 330-345 GB total_vm -- i.e. the machine thrashes on address
# space/commit, not on resident bytes.
#
# That is why this script now offers ISOLATION: a launch that takes the host down gives me
# one data point per 25 minutes and no ability to observe, which is worse than a launch that
# fails cleanly. Keeping sshd alive is worth more than any single flag's effect.
#
# ISOLATE=1  -- run under systemd-run with a hard memory cap, so the load is killed by the
#               cgroup BEFORE the host thrashes and sshd survives.
# LIMIT_V=1  -- set `ulimit -v` so over-large reservations fail at allocation time in Python
#               (MemoryError, with a traceback) instead of by the kernel later.
# FEWER_THREADS=1 -- drop cpuinfer threads from 16 to 4; the KT path allocates pinned buffers
#               per layer, so thread count multiplies the host-side demand.
#
# Run with ISOLATE=1 FIRST. If it dies inside the cap, the traceback says exactly which
# allocation was too large -- which is the measurement I have been missing all along.

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

if [ "${LIMIT_V:-0}" = "1" ]; then
  ulimit -v $((60 * 1024 * 1024))   # 60 GB of address space, well under the 90 GB CommitLimit
  echo "ulimit -v set to 60 GB"
fi
CPUINFER="${CPUINFER:-16}"
# After the FULL load (all 48 MoE layers, 512 experts each) the configurator computes a floor
# from the weights actually resident; 0.55 fails with "Raise --mem-fraction-static above
# 0.692 (minimum viable = 1 - available/pre = 0.6911)". Kept settable so the floor can be
# met without editing the script.
MEMFRAC="${MEMFRAC:-0.55}"
[ "${FEWER_THREADS:-0}" = "1" ] && CPUINFER=4
echo "cpuinfer=$CPUINFER  ISOLATE=${ISOLATE:-0}  LIMIT_V=${LIMIT_V:-0}"

# PLE_BACKEND=pinned|file   -- "file" uses the sparse mmap and REQUIRES the host-side
# gather on this GPU (see SGLANG_QWEN4_PLE_HOST_SIDE_GATHER below), because the default
# Triton gather needs unified memory which an RTX PRO 2000 does not have.
PLE_BACKEND="${PLE_BACKEND:-pinned}"
if [ "$PLE_BACKEND" = "file" ]; then
  # The device check exists to stop the kernel reading garbage; the host-side gather makes
  # that moot, so it is bypassed together with enabling the gather.
  export SGLANG_QWEN4_PLE_HOST_SIDE_GATHER=1
  export SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK=1
  PLE_OFFLOAD=(--ple-offload-embedding --ple-offload-backend file)
  echo "PLE: file backend (sparse mmap) + host-side gather"
else
  PLE_OFFLOAD=(--ple-offload-embedding --ple-offload-backend pinned)
  echo "PLE: pinned backend  (upstream docs say this does NOT boot 126 GiB weights on a 92 GB box)"
fi

ARGS=(
  --model-path "$MODEL"
  --kt-weight-path "$MODEL"
  "${PLE_OFFLOAD[@]}"
  --quantization modelopt_mixed
  --kt-num-gpu-experts 0 --kt-cpuinfer "$CPUINFER" --kt-threadpool-count 1 --kt-method NVFP4
  --attention-backend triton --disable-cuda-graph --mem-fraction-static "${MEMFRAC:-0.55}"
  --max-running-requests 2
  --host 127.0.0.1 --port "$PORT"
  --trust-remote-code --disable-radix-cache
)

if [ "${ISOLATE:-0}" = "1" ]; then
  # Hard memory cap in a transient scope: the cgroup kills the load BEFORE the host
  # thrashes, so sshd keeps working and the log survives to be read.
  MEMCAP="${MEMCAP:-70G}"
  MEMSWAP="${MEMSWAP:-8G}"
  # Swap allowance matters: the first measured run hit 69.30 GiB RSS at layer 40/48 with
  # only 8G of swap permitted, and was killed by the CAP rather than by the machine.
  echo "running under systemd-run --scope -p MemoryMax=$MEMCAP -p MemorySwapMax=$MEMSWAP"
  exec systemd-run --user --scope --collect \
    -p MemoryMax="$MEMCAP" -p MemorySwapMax="$MEMSWAP" \
    python3 -m sglang.launch_server "${ARGS[@]}"
fi

exec python3 -m sglang.launch_server "${ARGS[@]}"

