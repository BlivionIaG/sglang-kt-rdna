#!/usr/bin/env bash
# Phase-0 hardware smoke for Qwen3-30B-A3B-GPTQ-Int4.
#
# The agent VM that added this script has no GPU and did not run it.
# Compile-only coverage is .github/workflows/rdna-sgl-kernel.yml.
#
# gfx1100: --attention-backend triton
# gfx1030: --attention-backend torch_native
# Both: SGLANG_USE_AITER=0, --disable-cuda-graph, --kt-num-gpu-experts 0
#
# Required:
#   KT_WEIGHT_PATH  directory of the GPTQ-Int4 weights (same tree as --model
#                   when the weights are local)
# Optional:
#   AMDGPU_TARGET, MODEL, PORT, KT_CPUINFER, KT_THREADPOOL, PROMPT, MAX_NEW
#   SKIP_BUILD=1    skip the sgl-kernel build
#   OUT             json path (default /tmp/rdna-greedy-${ARCH}.json)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARCH="${AMDGPU_TARGET:-}"
if [[ -z "${ARCH}" ]] && command -v rocminfo >/dev/null 2>&1; then
  ARCH="$(rocminfo | awk '/Name:[[:space:]]+gfx/ {print $2; exit}')"
fi
if [[ -z "${ARCH}" ]]; then
  ARCH="$(python3 - <<'PY'
import torch
print(torch.cuda.get_device_properties(0).gcnArchName.split(":")[0])
PY
)"
fi
ARCH="${ARCH%%:*}"
export AMDGPU_TARGET="${ARCH}"
export SGLANG_USE_AITER=0

case "${ARCH}" in
  gfx103*) ATTN=torch_native ;;
  gfx110*) ATTN=triton ;;
  *)
    echo "unsupported arch ${ARCH}; expected gfx1030 or gfx1100" >&2
    exit 1
    ;;
esac

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  "${ROOT}/scripts/rdna/build_sgl_kernel.sh"
fi

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-GPTQ-Int4}"
PORT="${PORT:-30000}"
: "${KT_WEIGHT_PATH:?set KT_WEIGHT_PATH to the GPTQ-Int4 weight directory}"
OUT="${OUT:-/tmp/rdna-greedy-${ARCH}.json}"
PROMPT="${PROMPT:-The capital of France is}"
MAX_NEW="${MAX_NEW:-32}"

python3 -m sglang.launch_server \
  --model "${MODEL}" \
  --kt-weight-path "${KT_WEIGHT_PATH}" \
  --kt-method GPTQ_INT4 \
  --kt-cpuinfer "${KT_CPUINFER:-16}" \
  --kt-threadpool-count "${KT_THREADPOOL:-1}" \
  --kt-num-gpu-experts 0 \
  --attention-backend "${ATTN}" \
  --disable-cuda-graph \
  --tp 1 \
  --host 127.0.0.1 \
  --port "${PORT}" &
SERVER_PID=$!
trap 'kill ${SERVER_PID} >/dev/null 2>&1 || true' EXIT

python3 - "${PORT}" "${OUT}" "${PROMPT}" "${MAX_NEW}" <<'PY'
import json, sys, time, urllib.request
port, out, prompt, max_new = sys.argv[1:]
url = f"http://127.0.0.1:{port}/generate"
body = json.dumps({
    "text": prompt,
    "sampling_params": {
        "temperature": 0,
        "max_new_tokens": int(max_new),
    },
}).encode()
deadline = time.time() + 1800
last = None
while time.time() < deadline:
    try:
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            payload = resp.read()
        open(out, "wb").write(payload)
        print(out)
        sys.exit(0)
    except Exception as exc:
        last = exc
        time.sleep(5)
print(f"server did not answer /generate: {last}", file=sys.stderr)
sys.exit(1)
PY

echo "wrote ${OUT}"
echo "compare with: python3 ${ROOT}/scripts/rdna/compare_greedy.py /tmp/rdna-greedy-gfx1030.json /tmp/rdna-greedy-gfx1100.json"
