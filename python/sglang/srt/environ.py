from __future__ import annotations
import functools
import json
import os
import subprocess
import warnings
from contextlib import ExitStack, contextmanager
from enum import IntEnum
from typing import Any
import signal
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional


@contextmanager
def temp_set_env(*, allow_sglang: bool = False, **env_vars: Any):
    """Temporarily set environment variables, restoring originals on exit.

    By default, SGLANG_*/SGL_* keys are rejected — use ``Envs`` descriptors
    for those.  Pass ``allow_sglang=True`` only for special env vars that
    intentionally bypass ``environ.py``.
    """
    if not allow_sglang:
        for key in env_vars:
            if key.startswith("SGLANG_") or key.startswith("SGL_"):
                raise ValueError("temp_set_env should not be used for sglang env vars")

    backup = {key: os.environ.get(key) for key in env_vars}
    try:
        for key, value in env_vars.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class EnvField:
    _allow_set_name = True

    def __init__(self, default: Any, secret: bool = False):
        self.default = default
        self.secret = secret
        # NOTE: environ can only accept str values, so we need a flag to indicate
        # whether the env var is explicitly set to None.
        self._set_to_none = False

    def __set_name__(self, owner, name):
        assert EnvField._allow_set_name, "Usage like `a = envs.A` is not allowed"
        self.name = name

    def parse(self, value: str) -> Any:
        raise NotImplementedError()

    def _resolve_default(self) -> Any:
        # Callable defaults are evaluated lazily, only when the env is unset.
        return self.default() if callable(self.default) else self.default

    def get(self) -> Any:
        value = os.getenv(self.name)

        # Explicitly set to None
        if self._set_to_none:
            assert value == str(None)
            return None

        # Not set, return default
        if value is None:
            return self._resolve_default()

        try:
            return self.parse(value)
        except ValueError as e:
            warnings.warn(
                f'Invalid value for {self.name}: {e}, using default "{self.default}"'
            )
            return self.default

    def is_set(self):
        return self.name in os.environ

    def set(self, value: Any):
        self._set_to_none = value is None
        os.environ[self.name] = str(value)

    @contextmanager
    def override(self, value: Any):
        backup_present = self.name in os.environ
        backup_value = os.environ.get(self.name)
        backup_set_to_none = self._set_to_none
        self.set(value)
        yield
        if backup_present:
            os.environ[self.name] = backup_value
        else:
            os.environ.pop(self.name, None)
        self._set_to_none = backup_set_to_none

    def clear(self):
        os.environ.pop(self.name, None)
        self._set_to_none = False

    def __bool__(self):
        raise RuntimeError(
            "Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"
        )

    def __len__(self):
        raise RuntimeError(
            "Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"
        )


class EnvTuple(EnvField):
    def parse(self, value: str) -> tuple[str, ...]:
        return tuple(s.strip() for s in value.split(",") if s.strip())


class EnvStr(EnvField):
    def parse(self, value: str) -> str:
        return value


class EnvBool(EnvField):
    def parse(self, value: str) -> bool:
        value = value.lower()
        if value in ["true", "1", "yes", "y"]:
            return True
        if value in ["false", "0", "no", "n"]:
            return False
        raise ValueError(f'"{value}" is not a valid boolean value')


class EnvInt(EnvField):
    def parse(self, value: str) -> int:
        try:
            return int(value)
        except ValueError:
            raise ValueError(f'"{value}" is not a valid integer value')


class EnvFloat(EnvField):
    def parse(self, value: str) -> float:
        try:
            return float(value)
        except ValueError:
            raise ValueError(f'"{value}" is not a valid float value')


class ToolStrictLevel(IntEnum):
    """
    Defines the strictness levels for tool call parsing and validation.

    OFF: No strict validation
    FUNCTION: Enables structural tag constraints for all tools
    PARAMETER: Enforces strict parameter validation for all tools
    """

    OFF = 0
    FUNCTION = 1
    PARAMETER = 2


class _DeprecatedEnvFallback:
    """Mixin for EnvField subclasses: if the canonical env var is not set,
    check *deprecated_name* and emit DeprecationWarning before reading it.
    """

    def __init__(self, default: Any, deprecated_name: str, secret: bool = False):
        super().__init__(default, secret=secret)
        self.deprecated_name = deprecated_name

    def get(self) -> Any:
        if os.getenv(self.name) is None:
            fallback = os.getenv(self.deprecated_name)
            if fallback is not None:
                warnings.warn(
                    f"Environment variable '{self.deprecated_name}' is deprecated; "
                    f"use '{self.name}' instead. "
                    "The alias will be removed in a future release.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                os.environ[self.name] = fallback
        return super().get()


class EnvBoolWithAlias(_DeprecatedEnvFallback, EnvBool):
    pass


class EnvIntWithAlias(_DeprecatedEnvFallback, EnvInt):
    pass


class EnvJSON(EnvField):
    def parse(self, value: str | None) -> list | dict | None:
        if not value:
            return None
        if os.path.exists(value):
            with open(value) as f:
                return json.load(f)
        return json.loads(value)


@functools.lru_cache(maxsize=1)
class DsparkFoldedSampling(IntEnum):
    """Sampling support in the graph-folded DSpark draft proposal: OFF =
    greedy-only folding, AUTO = on when its buffers fit in free GPU memory,
    FORCE = always."""

    OFF = 0
    AUTO = 1
    FORCE = 2


class GateGemvMode(IntEnum):
    """Small-batch Inkling gate linear implementation.

    OFF: always the cublas GEMM
    PAIR: PDL-chained GEMV and gate JIT kernels
    FUSED: single-launch GEMV + gate epilogue (last-block ticket)
    """

    OFF = 0
    PAIR = 1
    FUSED = 2


class InvariantCheckLevel(IntEnum):
    """Signal level for value/index validity checks (see invariants.py).

    OFF: data layer only (sanitize/containment); no detection, no signal.
    WARN: detect + throttled log/count; degrade, never crash (prod on-demand).
    STRICT: detect + crash on GUARD/FATAL violations (CI default).

    The data layer is unconditional and independent of this level; only the
    detection + signal layer is gated here.
    """

    OFF = 0
    WARN = 1
    STRICT = 2


def _default_cache_subdir(name: str) -> str:
    """A directory under SGLANG_CACHE_DIR, for env defaults that track it.

    Pass as a callable default: SGLANG_CACHE_DIR is declared further down the
    Envs body, and resolving late also lets tests override it.
    """
    return os.path.join(os.path.expanduser(envs.SGLANG_CACHE_DIR.get()), name)


@functools.lru_cache(maxsize=1)
def _default_hip() -> bool:
    """Lazy ROCm/HIP detection for platform-conditional env defaults.

    Avoids importing torch at environ import time (this module is intentionally
    stdlib-only and loaded very early). Resolved on first EnvField.get() that uses
    it as a default, by which point torch is already imported in any real run;
    falls back to False if torch is unavailable.
    """
    try:
        import torch

        return torch.version.hip is not None
    except Exception:
        return False


def _default_tree_cache_sanity_check() -> bool:
    """Enable the expensive tree-cache sanity check by default in CI."""
    return envs.SGLANG_IS_IN_CI.get()


class Envs:
    # fmt: off

    # Model & File Download
    SGLANG_USE_MODELSCOPE = EnvBool(False)
    SGLANG_DISABLED_MODEL_ARCHS = EnvTuple(tuple())
    # "none" = use checkpoint's config.json, "small"/"large" = force the packaged
    # config_backup_{small,large}.json, "auto" = pick small/large based on the
    # checkpoint's num_hidden_layers.
    SGLANG_APPLY_CONFIG_BACKUP = EnvStr("auto")

    # Logging Options
    SGLANG_LOG_GC = EnvBool(False)
    SGLANG_LOG_FORWARD_ITERS = EnvBool(False)
    SGLANG_LOG_MS = EnvBool(False)
    SGLANG_DISABLE_REQUEST_LOGGING = EnvBool(False)
    SGLANG_LOG_REQUEST_EXCEEDED_MS = EnvInt(-1)
    SGLANG_LOG_REQUEST_HEADERS = EnvTuple(tuple())
    SGLANG_LOG_SCHEDULER_STATUS_TARGET = EnvStr("")
    SGLANG_LOG_SCHEDULER_STATUS_INTERVAL = EnvFloat(60.0)

    # SGLang CI
    SGLANG_IS_IN_CI = EnvBool(False)
    SGLANG_IS_IN_CI_AMD = EnvBool(False)
    SGLANG_CUDA_COREDUMP = EnvBool(False)
    SGLANG_CUDA_COREDUMP_DIR = EnvStr("/tmp/sglang_cuda_coredumps")
    SGLANG_TEST_MAX_RETRY = EnvInt(None)

    # Constrained Decoding (Grammar)
    SGLANG_GRAMMAR_POLL_INTERVAL = EnvFloat(0.005)
    SGLANG_GRAMMAR_MAX_POLL_ITERATIONS = EnvInt(10000)
    SGLANG_DISABLE_OUTLINES_DISK_CACHE = EnvBool(False)


    # Test & Debug
    SGLANG_DETECT_SLOW_RANK = EnvBool(False)
    SGLANG_TEST_STUCK_DETOKENIZER = EnvFloat(0)
    SGLANG_TEST_STUCK_DP_CONTROLLER = EnvFloat(0)
    SGLANG_TEST_STUCK_SCHEDULER_INIT = EnvFloat(0)
    SGLANG_TEST_STUCK_TOKENIZER = EnvFloat(0)
    SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS = EnvInt(0)
    IS_BLACKWELL = EnvBool(False)
    IS_H200 = EnvBool(False)
    SGLANG_SET_CPU_AFFINITY = EnvBool(False)
    SGLANG_PROFILE_WITH_STACK = EnvBool(True)
    SGLANG_PROFILE_RECORD_SHAPES = EnvBool(True)
    SGLANG_PROFILE_V2 = EnvBool(False)
    SGLANG_RECORD_STEP_TIME = EnvBool(False)
    SGLANG_FORCE_SHUTDOWN = EnvBool(False)
    SGLANG_DEBUG_MEMORY_POOL = EnvBool(False)
    SGLANG_TEST_REQUEST_TIME_STATS = EnvBool(False)
    SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK = EnvBool(False)
    SGLANG_SIMULATE_ACC_LEN = EnvFloat(-1)
    SGLANG_SIMULATE_ACC_METHOD = EnvStr("multinomial")
    SGLANG_TORCH_PROFILER_DIR = EnvStr("/tmp")
    SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS = EnvInt(500)
    SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE = EnvInt(64)
    SGLANG_NATIVE_MOVE_KV_CACHE = EnvBool(False)
    SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK = EnvBool(True)

    # Scheduler: memory leak test
    SGLANG_TEST_RETRACT = EnvBool(False)
    SGLANG_TEST_RETRACT_INTERVAL = EnvInt(3)
    SGLANG_TEST_RETRACT_NO_PREFILL_BS = EnvInt(2 ** 31)
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY = EnvInt(0)
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE = EnvBool(True)

    # Scheduler: new token ratio hyperparameters
    SGLANG_INIT_NEW_TOKEN_RATIO = EnvFloat(0.7)
    SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR = EnvFloat(0.14)
    SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS = EnvInt(600)
    SGLANG_RETRACT_DECODE_STEPS = EnvInt(20)
    SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION = EnvInt(4096)

    # Scheduler: recv interval
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DEFAULT = EnvInt(1000)
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DECODE = EnvInt(1)
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_TARGET_VERIFY = EnvInt(1)
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_NONE = EnvInt(1)

    # PD Disaggregation (runtime)
    # NOTE: For SGLANG_DISAGGREGATION_THREAD_POOL_SIZE, the effective default is
    # computed dynamically at runtime based on cpu_count; see disaggregation backends.
    SGLANG_DISAGGREGATION_THREAD_POOL_SIZE = EnvInt(None)
    SGLANG_DISAGGREGATION_QUEUE_SIZE = EnvInt(4)
    SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT = EnvInt(300)
    SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL = EnvFloat(5.0)
    SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE = EnvInt(2)
    SGLANG_DISAGGREGATION_WAITING_TIMEOUT = EnvInt(300)
    SGLANG_DISAGGREGATION_NIXL_BACKEND = EnvStr("UCX")

    # Scheduler: others:
    SGLANG_EMPTY_CACHE_INTERVAL = EnvFloat(-1)  # in seconds. Set if you observe high memory accumulation over a long serving period.
    SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP = EnvBool(False)
    SGLANG_SCHEDULER_MAX_RECV_PER_POLL = EnvInt(-1)
    SGLANG_EXPERIMENTAL_CPP_RADIX_TREE = EnvBool(False)
    SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR = EnvFloat(0.75)
    SGLANG_SCHEDULER_SKIP_ALL_GATHER = EnvBool(False)
    SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE = EnvBool(False)
    SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES = EnvInt(None)
    SGLANG_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK = EnvFloat(None)
    SGLANG_DATA_PARALLEL_BUDGET_INTERVAL = EnvInt(1)
    SGLANG_REQ_WAITING_TIMEOUT = EnvFloat(-1)  # in seconds
    SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH = EnvBool(False)
    SGLANG_REQ_RUNNING_TIMEOUT = EnvFloat(-1)  # in seconds
    SGLANG_DISAGGREGATION_BOOTSTRAP_ENTRY_CLEANUP_INTERVAL = EnvInt(120)

    # Test: pd-disaggregation
    SGLANG_TEST_PD_DISAGG_BACKEND = EnvStr("mooncake")
    SGLANG_TEST_PD_DISAGG_DEVICES = EnvStr(None)

    # Model Parallel
    SGLANG_USE_MESSAGE_QUEUE_BROADCASTER = EnvBool(True)
    SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS = EnvBool(False)
    # Override the distributed init method used by torch.distributed.init_process_group.
    # Set to "env://" to use an externally-created TCPStore via MASTER_ADDR/MASTER_PORT.
    SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE = EnvStr(None)

    # Tool Calling
    SGLANG_FORWARD_UNKNOWN_TOOLS = EnvBool(False)

    # Hi-Cache
    SGLANG_HICACHE_HF3FS_CONFIG_PATH = EnvStr(None)
    SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR = EnvStr(None)
    SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR = EnvStr(None)

    # Mooncake KV Transfer
    SGLANG_MOONCAKE_CUSTOM_MEM_POOL = EnvStr(None)
    ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE = EnvBool(False)
    ASCEND_NPU_PHY_ID = EnvInt(-1)
    SGLANG_MOONCAKE_SEND_AUX_TCP = EnvBool(False)

    # Mooncake Store
    SGLANG_HICACHE_MOONCAKE_CONFIG_PATH = EnvStr(None)
    MOONCAKE_MASTER = EnvStr(None)
    MOONCAKE_CLIENT = EnvStr(None)
    MOONCAKE_LOCAL_HOSTNAME = EnvStr("localhost")
    MOONCAKE_TE_META_DATA_SERVER = EnvStr("P2PHANDSHAKE")
    MOONCAKE_GLOBAL_SEGMENT_SIZE = EnvStr("4gb")
    MOONCAKE_PROTOCOL = EnvStr("tcp")
    MOONCAKE_DEVICE = EnvStr("")
    MOONCAKE_MASTER_METRICS_PORT = EnvInt(9003)
    MOONCAKE_CHECK_SERVER = EnvBool(False)
    MOONCAKE_STANDALONE_STORAGE = EnvBool(False)

    # AMD & ROCm
    SGLANG_USE_AITER = EnvBool(False)
    SGLANG_ROCM_FUSED_DECODE_MLA = EnvBool(False)
    SGLANG_ROCM_DISABLE_LINEARQUANT = EnvBool(False)

    # NPU
    SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT = EnvBool(False)
    SGLANG_NPU_USE_MULTI_STREAM = EnvBool(False)
    SGLANG_NPU_USE_MLAPO = EnvBool(False)
    # Forward native implementation for activation gelu tanh for model Skywork-Reward-Gemma-2-27B-v0.2
    SGLANG_NPU_FORWARD_NATIVE_GELUTANH = EnvBool(False)
    # Forward native implementation for gemma rms norm for model Skywork-Reward-Gemma-2-27B-v0.2
    SGLANG_NPU_FORWARD_NATIVE_GEMMA_RMS_NORM = EnvBool(False)

    # Quantization
    SGLANG_INT4_WEIGHT = EnvBool(False)
    SGLANG_CPU_QUANTIZATION = EnvBool(False)
    SGLANG_USE_DYNAMIC_MXFP4_LINEAR = EnvBool(False)
    SGLANG_FORCE_FP8_MARLIN = EnvBool(False)
    SGLANG_MOE_NVFP4_DISPATCH = EnvBool(False)
    SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN = EnvBool(False)
    SGLANG_PER_TOKEN_GROUP_QUANT_8BIT_V2 = EnvBool(False)
    SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE = EnvBool(False)

    # Flashinfer
    SGLANG_IS_FLASHINFER_AVAILABLE = EnvBool(True)
    SGLANG_ENABLE_FLASHINFER_FP8_GEMM = EnvBool(False)
    # Default to the pick from flashinfer
    SGLANG_FLASHINFER_FP4_GEMM_BACKEND = EnvStr("")
    SGLANG_FLASHINFER_WORKSPACE_SIZE = EnvInt(384 * 1024 * 1024)

    # Triton
    SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS = EnvBool(False)
    SGLANG_USE_CUSTOM_TRITON_KERNEL_CACHE = EnvBool(False)

    # Torch Compile
    SGLANG_ENABLE_TORCH_COMPILE = EnvBool(False)

    # EPLB
    SGLANG_EXPERT_LOCATION_UPDATER_LOG_INPUT = EnvBool(False)
    SGLANG_EXPERT_LOCATION_UPDATER_CANARY = EnvBool(False)
    SGLANG_EXPERT_LOCATION_UPDATER_LOG_METRICS = EnvBool(False)
    SGLANG_LOG_EXPERT_LOCATION_METADATA = EnvBool(False)
    SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR = EnvStr("/tmp")
    SGLANG_EPLB_HEATMAP_COLLECTION_INTERVAL = EnvInt(0)
    SGLANG_ENABLE_EPLB_BALANCEDNESS_METRIC = EnvBool(False)

    # TBO
    SGLANG_TBO_DEBUG = EnvBool(False)

    # DeepGemm
    SGLANG_ENABLE_JIT_DEEPGEMM = EnvBool(True)
    SGLANG_JIT_DEEPGEMM_PRECOMPILE = EnvBool(True)
    SGLANG_JIT_DEEPGEMM_FAST_WARMUP = EnvBool(False)
    SGLANG_JIT_DEEPGEMM_COMPILE_WORKERS = EnvInt(4)
    SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE = EnvBool(False)
    SGLANG_DG_CACHE_DIR = EnvStr(os.path.expanduser("~/.cache/deep_gemm"))
    SGLANG_DG_USE_NVRTC = EnvBool(False)
    SGLANG_USE_DEEPGEMM_BMM = EnvBool(False)
    SGLANG_OPT_DEEPGEMM_SCALE_CONVERT_AT_INIT = EnvBool(True)

    # DeepSeek MHA Optimization
    SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD = EnvInt(8192)

    # DeepEP
    SGLANG_DEEPEP_BF16_DISPATCH = EnvBool(False)
    SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(128)
    SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS = EnvInt(32)
    SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO = EnvBool(False)
    SGLANG_HACK_OVERRIDE_TOPK_IDS_RANDOM = EnvBool(False)
    SGLANG_HACK_FORCE_TID2EID_ZERO = EnvBool(False)

    # NSA Backend
    SGLANG_NSA_FUSE_TOPK = EnvBool(True)
    SGLANG_NSA_ENABLE_MTP_PRECOMPUTE_METADATA = EnvBool(True)
    SGLANG_USE_FUSED_METADATA_COPY = EnvBool(True)
    SGLANG_VERIFY_FUSED_METADATA_COPY = EnvBool(False)
    SGLANG_NSA_FORCE_MLA = EnvBool(False)

    # sgl-kernel
    SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK = EnvBool(False)

    # vLLM dependencies (TODO: they have been deprecated, we can remove them safely)
    USE_VLLM_CUTLASS_W8A8_FP8_KERNEL = EnvBool(False)

    USE_TRITON_W8A8_FP8_KERNEL = EnvBool(False)
    SGLANG_RETURN_ORIGINAL_LOGPROB = EnvBool(False)
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN = EnvBool(False)
    SGLANG_MOE_PADDING = EnvBool(False)
    SGLANG_CUTLASS_MOE = EnvBool(False)
    HF_HUB_DISABLE_XET = EnvBool(False)
    DISABLE_OPENAPI_DOC = EnvBool(False)
    SGLANG_ENABLE_TORCH_INFERENCE_MODE = EnvBool(False)
    SGLANG_IS_FIRST_RANK_ON_NODE = EnvBool(True)
    SGLANG_SUPPORT_CUTLASS_BLOCK_FP8 = EnvBool(False)
    SGLANG_SYNC_TOKEN_IDS_ACROSS_TP = EnvBool(False)
    SGLANG_ENABLE_COLOCATED_BATCH_GEN = EnvBool(False)

    # Deterministic inference
    SGLANG_ENABLE_DETERMINISTIC_INFERENCE = EnvBool(False)
    # Use 1-stage all-reduce kernel on AMD (deterministic, fixed accumulation order)
    # If not set: auto (enabled when --enable-deterministic-inference is on)
    # Set to 1: force enable (even without --enable-deterministic-inference)
    # Set to 0: force disable (use default Aiter AR even with --enable-deterministic-inference)
    SGLANG_USE_1STAGE_ALLREDUCE = EnvBool(False)
    SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE = EnvInt(4096)
    SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE = EnvInt(2048)
    SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE = EnvInt(4096)
    SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE = EnvInt(256)

    # RoPE cache configuration
    SGLANG_SPEC_EXPANSION_SAFETY_FACTOR = EnvInt(2)
    SGLANG_ROPE_CACHE_SAFETY_MARGIN = EnvInt(256)
    SGLANG_ROPE_CACHE_ALIGN = EnvInt(128)

    # Overlap Spec V2
    SGLANG_ENABLE_SPEC_V2 = EnvBool(False)
    SGLANG_ENABLE_OVERLAP_PLAN_STREAM = EnvBool(False)

    # Spec Config
    SGLANG_SPEC_ENABLE_STRICT_FILTER_CHECK = EnvBool(True)

    # VLM
    SGLANG_VLM_CACHE_SIZE_MB = EnvInt(100)
    SGLANG_IMAGE_MAX_PIXELS = EnvInt(16384 * 28 * 28)
    SGLANG_RESIZE_RESAMPLE = EnvStr("")
    SGLANG_MM_BUFFER_SIZE_MB = EnvInt(0)
    SGLANG_MM_PRECOMPUTE_HASH = EnvBool(False)
    SGLANG_VIT_ENABLE_CUDA_GRAPH = EnvBool(False)
    SGLANG_MM_SKIP_COMPUTE_HASH = EnvBool(False)


    # VLM Item CUDA IPC Transport
    SGLANG_USE_CUDA_IPC_TRANSPORT = EnvBool(False)
    SGLANG_MM_FEATURE_CACHE_MB = EnvInt(4 * 1024)
    SGLANG_MM_ITEM_MEM_POOL_RECYCLE_INTERVAL_SEC = EnvFloat(0.05)

    # MM splitting behavior control
    SGLANG_ENABLE_MM_SPLITTING = EnvBool(False)

    # Mamba
    SGLANG_MAMBA_CONV_DTYPE = EnvStr("bfloat16")
    SGLANG_MAMBA_SSM_DTYPE = EnvStr(None)

    # Release & Resume Memory
    SGLANG_MEMORY_SAVER_CUDA_GRAPH = EnvBool(False)

    # Sparse Embeddings
    SGLANG_EMBEDDINGS_SPARSE_HEAD = EnvStr(None)

    # Logits processor
    SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK = EnvBool(False)
    SGLANG_LOGITS_PROCESSER_CHUNK_SIZE = EnvInt(2048)

    # Tool-Call behavior
    SGLANG_TOOL_STRICT_LEVEL = EnvInt(ToolStrictLevel.OFF)

    # Ngram
    SGLANG_NGRAM_FORCE_GREEDY_VERIFY = EnvBool(False)

    # Warmup
    SGLANG_WARMUP_TIMEOUT = EnvFloat(-1) # in seconds. If a warmup forward batch takes longer than this, the server will crash to prevent hanging. Recommend to increase warmup timeout to 1800 to accommodate some kernel JIT precache e.g. deep gemm

    # Health Check
    SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION = EnvBool(True)

    # External models
    SGLANG_EXTERNAL_MODEL_PACKAGE = EnvStr("")
    SGLANG_EXTERNAL_MM_MODEL_ARCH = EnvStr("")
    SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE = EnvStr("")

    # Numa
    SGLANG_NUMA_BIND_V2 = EnvBool(True)

    # Metrics
    SGLANG_ENABLE_METRICS_DEVICE_TIMER = EnvBool(False)
    SGLANG_ENABLE_METRICS_DP_ATTENTION = EnvBool(False)

    # Tokenizer
    SGLANG_PATCH_TOKENIZER = EnvBool(False)  # TODO enable by default

    # TokenizerManager
    SGLANG_REQUEST_STATE_WAIT_TIMEOUT = EnvInt(4)

    SGLANG_ENABLE_THINKING = EnvBool(False)
    # Model-specific DeepSeek-V4 effort default. Empty means use the product default.
    SGLANG_DSV4_REASONING_EFFORT = EnvStr("")
    # Deprecated compatibility alias for SGLANG_DSV4_REASONING_EFFORT.
    SGLANG_REASONING_EFFORT = EnvStr("")

    SGLANG_DSV4_MODE = EnvStr("")
    SGLANG_DSV4_2604_SUBMODE = EnvStr("")
    SGLANG_DSV4_FP4_EXPERTS = EnvBool(False)  # Set False when using FP4-to-FP8 converted checkpoint with 2604 config
    SGLANG_OPT_HISPARSE_C4_SHRINK = EnvInt(1)
    SGLANG_OPT_DEEPGEMM_HC_PRENORM = EnvBool(False)
    SGLANG_OPT_USE_TILELANG_MHC_PRE = EnvBool(False)
    SGLANG_OPT_USE_TILELANG_MHC_POST = EnvBool(False)
    SGLANG_FLASHMLA_BACKEND_OVERRIDE = EnvStr("kernel")
    SGLANG_HACK_SKIP_FP4_FP8_GEMM = EnvBool(False)
    SGLANG_OPT_FP8_WO_A_GEMM = EnvBool(False)


    SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK = EnvBool(False)
    SGLANG_OPT_USE_TILELANG_SWA_PREPARE = EnvBool(False)
    SGLANG_OPT_USE_MULTI_STREAM_OVERLAP = EnvBool(False)

    SGLANG_FIX_MTP_HC_HIDDEN = EnvBool(True)
    SGLANG_FIX_ATTN_BACKEND_IDLE = EnvBool(True)
    SGLANG_FIX_PD_IDLE = EnvBool(True)
    SGLANG_FIX_SWA_CHUNKED_REQ_DOUBLE_FREE = EnvBool(True)
    SGLANG_OPT_V4_DRAFT_EXTEND_CUDA_GRAPH = EnvBool(False)  # usually not useful
    SGLANG_OPT_USE_FUSED_STORE_CACHE = EnvBool(False)
    SGLANG_OPT_USE_OVERLAP_STORE_CACHE = EnvBool(False)
    SGLANG_OPT_BF16_FP32_GEMM_ALGO = EnvStr("cublas")
    SGLANG_OPT_USE_FUSED_HASH_TOPK = EnvBool(False)
    SGLANG_OPT_USE_JIT_EP_ACTIVATION = EnvBool(False)
    SGLANG_OPT_ALLOW_SHARED_EXPERT_DUAL_STREAM = EnvBool(False)  # verified in journal 2026-04-21-017
    SGLANG_OPT_CACHE_SWA_TRANSLATION = EnvBool(False)
    SGLANG_OPT_SWA_RADIX_CACHE_COMPACT = EnvBool(False)
    SGLANG_OPT_MXFP4_FUSE_RSF_SHARED_ADD = EnvBool(False)
    SGLANG_OPT_MXFP4_STATIC_SCALE_ONES = EnvBool(False)
    SGLANG_OPT_MXFP4_SKIP_DISPATCHER_MAPPING = EnvBool(False)
    SGLANG_OPT_USE_JIT_INDEXER_METADATA = EnvBool(False)
    SGLANG_OPT_SWIGLU_CLAMP_FUSION = EnvBool(False)
    SGLANG_OPT_DG_PAGED_MQA_LOGITS_CHUNK_SIZE = EnvInt(-1)
    SGLANG_DSV4_FIX_ATTN_PADDING = EnvBool(False)  # verified in journal 2026-04-21-017
    SGLANG_DSV4_FIX_TP_ATTN_A2A_SCATTER = EnvBool(False)
    SGLANG_DEBUG_SANITY_CHECK_CONFIG = EnvBool(False)
    SGLANG_DEBUG_HACK_CP_ASSERT_PURE_EXTEND = EnvBool(False)
    SGLANG_DEBUG_HACK_CP_CHECK_RANK_CONSISTENCY = EnvBool(False)
    SGLANG_OPT_USE_TOPK_V2 = EnvBool(False)
    SGLANG_OPT_FIX_APE_2604 = EnvBool(False)
    SGLANG_OPT_CP_REARRANGE_TRITON = EnvBool(False)
    SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE = EnvBool(False)
    SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK = EnvInt(1024)
    SGLANG_OPT_MEGA_MOE_FUSED_PRE_DISPATCH = EnvBool(False)
    SGLANG_OPT_FUSE_WQA_WKV = EnvBool(False)
    SGLANG_OPT_USE_JIT_NORM = EnvBool(False)
    SGLANG_OPT_FIX_HASH_MEGA_MOE = EnvBool(False)
    SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2 = EnvBool(False)
    SGLANG_OPT_FIX_MEGA_MOE_MEMORY = EnvBool(False)

    # Dangerous untested flagas
    SGLANG_OPT_USE_FAST_MASK_EP = EnvBool(False)
    SGLANG_OPT_USE_FLASHINFER_NORM = EnvBool(False)

    SGLANG_PREP_IN_CUDA_GRAPH = EnvBool(True)

    SGLANG_OPT_USE_TILELANG_INDEXER = EnvBool(False)
    SGLANG_OPT_USE_MINIMAX_FUSED_QKNORM_ROPE = EnvBool(False)
    SGLANG_OPT_USE_BF16_ROUTER_GEMM = EnvBool(False)
    SGLANG_OPT_USE_MINIMAX_DENSE_SPARSE_DECODE = EnvBool(False)
    SGLANG_OPT_USE_MINIMAX_FUSED_KV_INDEX_STORE = EnvBool(False)
    SGLANG_OPT_USE_MINIMAX_DECODE_TOPK_RADIX = EnvBool(False)
    SGLANG_DISABLE_MSA = EnvBool(False)
    SGLANG_TOPK_TRANSFORM_512_TORCH = EnvBool(False)
    SGLANG_FP8_PAGED_MQA_LOGITS_TORCH = EnvBool(False)

    # Symmetric Memory
    SGLANG_SYMM_MEM_PREALLOC_GB_SIZE = EnvInt(-1)

    # Aiter
    SGLANG_USE_AITER_FP8_PER_TOKEN = EnvBool(False)
    # fmt: on

    # EPD
    SGLANG_ENCODER_RECV_TIMEOUT = EnvFloat(180.0)
    SGLANG_ENCODER_SEND_TIMEOUT = EnvFloat(180.0)


    # ---------- merged from sgl-project/sglang: qwen4_exp port ----------
    # runtime_context / layer_boundary / qwen4_exp read these through Envs.
    # Added here instead of replacing this file wholesale: this fork also
    # defines 65 SGLANG_* names upstream lacks (SGLANG_CUTLASS_MOE,
    # SGLANG_DSV4_*, SGLANG_OPT_MEGA_MOE_*, ...). Descriptor semantics are
    # byte-identical between the two files; only the registry differs.
    SGLANG_ROLE_NAMESPACES = EnvStr("off")
    SGLANG_ROLE_NAMESPACES_OUT = EnvStr(None)
    SGLANG_SORT_WEIGHT_FILES = EnvInt(0)
    SGLANG_USE_ATTN_TP_NGRAM = EnvBool(False)
    SGLANG_ENABLE_QWEN4_PLE_FUSION = EnvBool(True)
    SGLANG_QWEN4_PLE_FILE_DIR = EnvStr(lambda: _default_cache_subdir("ple"))
    SGLANG_QWEN4_PLE_FILE_PREFETCH = EnvBool(True)
    SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK = EnvBool(False)
    SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB = EnvFloat(8.0)
    SGLANG_QWEN4_PLE_FILE_RSS_INTERVAL_S = EnvFloat(30.0)
    SGLANG_PREFETCH_BLOCK_SIZE_MB = EnvInt(16)
    SGLANG_GEMMA_OUT_OF_PLACE_POSITION_MUTATION = EnvBool(False)
    SGLANG_ENABLE_WEIGHT_LOADER_V2 = EnvBool(False)
    SGLANG_MOE_COPY_WEIGHT_VIEWS_BEFORE_H2D = EnvBool(False)
    SGLANG_LOAD_SNAPSHOT_USE_ZMQ = EnvBool(False)
    SGLANG_ENABLE_REQUEST_DECOMPRESSION = EnvBool(False)
    SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES = EnvBool(False)
    SGLANG_TIMEOUT_KEEP_ALIVE = EnvInt(5)
    SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT = EnvInt(10)
    SGLANG_EXPOSE_OWN_ENV_VARS = EnvBool(False)
    SGLANG_DIAG_BYPASS_HEALTH_GENERATE = EnvBool(False)
    SGLANG_LOG_DECODE_GRAPH_KEY = EnvBool(False)
    SGLANG_ENABLE_RANK_CONSENSUS_CHECKER = EnvBool(False)
    SGLANG_USE_PICKLE_IPC = EnvBool(True)
    SGLANG_LOG_PICKLE_IPC_OBJECTS = EnvBool(False)
    SGLANG_TCP_STORE_PORT = EnvInt(29600)
    SGLANG_PORT = EnvInt(None)
    SGLANG_BACKUP_PORT_BASE = EnvInt(10000)
    SGLANG_SKIP_RUST_TESTS = EnvBool(False)
    SGLANG_JIT_KERNEL_RUN_FULL_TESTS = EnvBool(False)
    SGLANG_PYSPY_DUMP_BEFORE_CRASH = EnvBool(True)
    SGLANG_CUDA_COREDUMP_BEFORE_CRASH = EnvBool(True)
    SGLANG_CUDA_COREDUMP_BEFORE_CRASH_WAIT_SECS = EnvFloat(60.0)
    SGLANG_TEST_DISAGG_FAILURE_PROB = EnvFloat(0.0)
    SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL = EnvBool(False)
    SGLANG_TEST_SKIP_CACHE_HIT_ASSERT = EnvBool(False)
    SGLANG_TEST_METRICS_FILE = EnvStr(None)
    SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB = EnvFloat(0.0)
    SGLANG_TEST_SCRIPTED_RUNTIME = EnvBool(False)
    SGLANG_TEST_SCRIPTED_RUNTIME_IPC_ADDR = EnvStr(None)
    SGLANG_TEST_SCRIPTED_RUNTIME_OUT_OF_BAND_ERROR_PATH = EnvStr(None)
    SGLANG_TEST_SCRIPTED_RUNTIME_SYS_PATH_ENTRY = EnvStr(None)
    SGLANG_PROFILE_BY_STAGE_DECODE_MIN_BS = EnvInt(0)
    SGLANG_ENABLE_NVTX_SCHEDULER = EnvBool(False)
    SGLANG_ENABLE_NVTX_OPERATIONS = EnvBool(False)
    SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE = EnvBool(False)
    SGLANG_GRAPH_BATCH_CAPTURE = EnvBool(False)
    SGLANG_MEM_PROFILE_MAX_ENTRIES = EnvInt(100000)
    SGLANG_TRACE_ASYNC = EnvBool(False)
    SGLANG_TRACE_ASYNC_FLUSH_THRESHOLD = EnvInt(100)
    SGLANG_TRACE_LOGITS_E2E = EnvBool(False)
    SGLANG_TRACE_LOGITS_E2E_SYNC = EnvBool(False)
    SGLANG_TRACE_SAMPLER_E2E = EnvBool(False)
    SGLANG_TRACE_QWEN_MOE_DEEPEP_E2E = EnvBool(False)
    SGLANG_DEEPEP_V2_TRACE_CONTIG = EnvBool(False)
    SGLANG_DEEPEP_V2_TRACE_MASKED = EnvBool(False)
    SGLANG_VALIDATE_MAMBA_REPLAY_STATE_INDICES = EnvBool(False)
    SGLANG_GDN_DECODE_FUSION_LOG_LAYER_HITS = EnvBool(False)
    SGLANG_GDN_DECODE_FUSION_VERIFY_REAL_TENSORS = EnvBool(False)
    SGLANG_DEBUG_POISON_POOL = EnvBool(False)
    SGLANG_DEBUG_REVERT_PR = EnvInt(0)
    SGLANG_PHASE_CHECKER_DEBUG = EnvBool(False)
    SGLANG_ENABLE_TREE_CACHE_SANITY_CHECK = EnvBool(_default_tree_cache_sanity_check)
    SGLANG_CHECK_KV_PAGE_INVARIANTS = EnvBool(False)
    SGLANG_DEBUG_HISPARSE_SKIP_IO = EnvBool(False)
    SGLANG_ENABLE_ASYNC_ASSERT = EnvBool(False)
    SGLANG_INVARIANT_CHECK = EnvInt(InvariantCheckLevel.OFF)
    SGLANG_SIMULATE_ACC_TOKEN_MODE = EnvStr("fixed")
    SGLANG_SIMULATE_ACC_GREEDY = EnvBool(True)
    SGLANG_SIMULATE_UNIFORM_EXPERTS = EnvBool(False)
    SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS = EnvBool(False)
    SGLANG_DSPARK_DEBUG_CONFIDENCE_PREFIX_SCHEDULER = EnvBool(False)
    SGLANG_DSPARK_DEBUG_CONFIDENCE_METRICS = EnvBool(False)
    SGLANG_DSPARK_DEBUG_DUMP = EnvTuple(tuple())
    SGLANG_DSPARK_LOG_SPS_PRED_INTERVAL = EnvInt(0)
    SGLANG_DSPARK_STS_COLLECT_PATH = EnvStr("")
    SGLANG_DSPARK_BLOCK_ACCEPT_ESTIMATE_PATH = EnvStr("")
    SGLANG_DSPARK_BLOCK_ACCEPT_ONLINE_INTERVAL = EnvInt(0)
    SGLANG_DSPARK_ENABLE_SPS_RECORD = EnvBool(False)
    SGLANG_DSPARK_FAST_KERNEL = EnvBool(True)
    SGLANG_DSPARK_FP32_LM_HEAD = EnvBool(False)
    SGLANG_DSPARK_FAST_SAMPLING = EnvBool(True)
    SGLANG_DSPARK_FOLDED_SAMPLING = EnvInt(DsparkFoldedSampling.AUTO)
    SGLANG_DSPARK_FOLDED_PROPOSAL = EnvBool(True)
    SGLANG_DSPARK_STACKED_CTX_KV = EnvBool(True)
    SGLANG_DSPARK_EMBED_IN_GRAPH = EnvBool(True)
    SGLANG_DSPARK_OPT_MARKOV_W2_BF16 = EnvBool(True)
    SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD = EnvBool(True)
    SGLANG_DSPARK_OPT_FUSED_GREEDY_MARKOV = EnvBool(False)
    SGLANG_DSPARK_NVLINK_VOCAB_GATHER = EnvBool(True)
    SGLANG_DSPARK_ENABLE_MULTI_STREAM = EnvBool(True)
    SGLANG_DSPARK_CONFIDENCE_RELAY_LAG_STEPS = EnvInt(2)
    SGLANG_DISABLE_LAZY_COMPACTION = EnvBool(False)
    SGLANG_LOG_LAZY_COMPACTION_STATS = EnvBool(False)
    SGLANG_LOG_LAZY_COMPACTION_STATS_INTERVAL_SEC = EnvInt(30)
    SGLANG_LAZY_COMPACTION_MAX_MOVES_PER_CALL = EnvInt(4096)
    SGLANG_USE_HND_KVCACHE = EnvBool(False)
    SGLANG_AITER_UNIFIED_DRAFT_EXTEND = EnvBool(True)
    SGLANG_AITER_ASM_PREFILL_HD128 = EnvBool(True)
    SGLANG_AITER_PAGED_PREFILL_ASM = EnvBool(True)
    SGLANG_ENABLE_POST_CAPTURE_KV_SIZING = EnvBool(False)
    SGLANG_MAX_NEW_TOKENS_LIMIT = EnvInt(None)
    SGLANG_CACHE_HIT_RATE_WINDOW_SECONDS = EnvFloat(15.0)
    SGLANG_PREFILL_TILE_BUDGET = EnvInt(0)
    SGLANG_PREFILL_TILE_BUDGET_MODE = EnvStr("compact")
    SGLANG_PREFILL_DELAYER_MAX_PREFILL_BS_WINDOW_SIZE = EnvInt(16)
    SGLANG_EXACT_CHUNK_FILL = EnvBool(True)
    SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION = EnvBool(False)
    SGLANG_FORCE_STREAM_INTERVAL = EnvInt(50)
    SGLANG_ENABLE_DELAY_SAMPLE = EnvBool(False)
    SGLANG_ENABLE_WAR_BARRIER = EnvBool(False)
    SGLANG_FORCE_COARSE_WAR_BARRIER = EnvBool(False)
    SGLANG_ENABLE_PREFILL_WAR_READ_DONE = EnvBool(False)
    SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM = EnvBool(False)
    SGLANG_PP_COMM_OVERLAP = EnvBool(False)
    SGLANG_ENABLE_DISAGG_PREFILL_CONTINUOUS_INPUT_POLLING = EnvBool(False)
    SGLANG_RADIX_FORCE_MISS = EnvBool(False)
    SGLANG_MAX_KV_CHUNK_CAPACITY = EnvInt(128 * 1024)
    SGLANG_DISABLE_HISPARSE_PREFETCH = EnvBool(False)
    SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS = EnvBool(True)
    SGLANG_SWA_EVICTION_INTERVAL = EnvInt(128)
    SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND = EnvStr("rust")
    SGLANG_OPT_RELEASE_PREFILL_SWA = EnvBoolWithAlias(
        False, deprecated_name="SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW"
    )
    SGLANG_ENABLE_DISAGG_SAMPLING_MASK = EnvBool(False)
    SGLANG_DISAGGREGATION_SAMPLING_MASK_MAX_TOKENS = EnvInt(None)
    SGLANG_DISAGGREGATION_ZMQ_SEND_TIMEOUT = EnvInt(1)
    SGLANG_DISAGGREGATION_ENGINE_INIT_TIMEOUT = EnvInt(60)
    SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS = EnvStr("{}")
    SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX = EnvBool(True)
    SGLANG_DISAGGREGATION_ZMQ_MAX_SOCKETS = EnvInt(16384)
    SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER = EnvBool(False)
    SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK = EnvBool(False)
    SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE = EnvBool(True)
    SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE_TIMEOUT = EnvFloat(30.0)
    SGLANG_RAY_BUNDLE_INDICES = EnvStr("")
    SGLANG_SHARED_EXPERT_TP1 = EnvBool(False)
    SGLANG_ENABLE_EMBED_REPLICATION = EnvBool(False)
    SGLANG_EXA_NUM_RESULTS = EnvInt(10)
    SGLANG_EXA_SEARCH_TYPE = EnvStr("auto")
    SGLANG_EXA_INCLUDE_HIGHLIGHTS = EnvBool(True)
    SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB = EnvInt(256)
    SGLANG_HICACHE_TMA_TRANSFER = EnvBool(True)
    SGLANG_MLA_DEDUP_CHUNK_TOKENS = EnvInt(2048)
    SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE = EnvInt(None)
    SGLANG_HICACHE_SKIP_HOST_DUPLICATE_RECLAIM = EnvBool(False)
    SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE = EnvStr(None)
    SGLANG_HICACHE_FILE_BACKEND_EVICTION_RATIO = EnvFloat(0.9)
    SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE = EnvStr("0")
    SGLANG_HICACHE_FILE_BACKEND_ENABLE_METADATA_CACHE = EnvBool(False)
    SGLANG_HICACHE_FILE_BACKEND_METADATA_TTL = EnvFloat(5.0)
    SGLANG_HICACHE_BUFFER_ANCHOR_LOCK_CAP = EnvFloat(0.5)
    SGLANG_HICACHE_NIXL_USE_DIRECT_IO = EnvBool(True)
    SGLANG_HUGEPAGE_SIZE = EnvStr("")
    SGLANG_DISAGG_STAGING_BUFFER = EnvBool(False)
    SGLANG_DISAGG_STAGING_POOL_SIZE_MB = EnvInt(4096)
    SGLANG_STAGING_USE_TORCH = EnvBool(False)
    SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES = EnvInt(0)
    SGLANG_ENABLE_FAILED_SESSION_PROBE = EnvBool(False)
    SGLANG_FAILED_SESSION_PROBE_INTERVAL_S = EnvFloat(30.0)
    SGLANG_HICACHE_MOONCAKE_REUSE_TE = EnvBool(True)
    SGLANG_HICACHE_MEMCACHE_CONFIG_PATH = EnvStr(None)
    SGLANG_NPU_MEMCACHE_ENABLE_WARMUP = EnvBool(False)
    SGLANG_DEEPEP_V2_FORCE_MAX_LEN = EnvBool(False)
    SGLANG_MORI_SEND_AUX_RDMA = EnvBool(False)
    SGLANG_MORI_QP_PER_TRANSFER = EnvInt(4)
    SGLANG_MORI_POST_BATCH_SIZE = EnvInt(-1)
    SGLANG_MORI_NUM_WORKERS = EnvInt(4)
    SGLANG_MORI_TRANSFER_SHARDS = EnvInt(8)
    SGLANG_MORI_WAIT_POLL_MS = EnvInt(1000)
    SGLANG_MORI_TRANSFER_TIMEOUT_MS = EnvInt(0)
    SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(4096)
    SGLANG_M3_USE_AITER_FUSED_QKNORM = EnvBool(True)
    SGLANG_USE_AITER_AG = EnvBool(True)
    SGLANG_DP_USE_REDUCE_SCATTER = EnvBool(_default_hip)
    SGLANG_ENABLE_DP_GATHER_FP8 = EnvBool(False)
    SGLANG_USE_AITER_UNIFIED_ATTN = EnvBool(False)
    SGLANG_USE_AITER_MOE_GU_ITLV = EnvBool(True)
    SGLANG_AITER_MOE_SORTING_DISPATCH_POLICY = EnvInt(2)
    SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED = EnvBool(False)
    SGLANG_AITER_FUSE_RMSNORM_PAD = EnvBool(False)
    SGLANG_AITER_KV_CACHE_LAYOUT = EnvStr("nhd")
    SGLANG_ROCM_USE_MULTI_STREAM = EnvBool(False)
    SGLANG_ROCM_K3_FUSE_KDA_INPROJ = EnvBool(True)
    SGLANG_ROCM_K3_FUSE_KDA_INPROJ_MAX_TOKENS = EnvInt(256)
    SGLANG_HACK_FLASHMLA_BACKEND = EnvStr("auto")
    SGLANG_AMD_USE_FLYDSL_MEGA_MOE = EnvBool(False)
    SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR = EnvInt(8192)
    SGLANG_AMD_FLYDSL_MEGA_QUANT = EnvStr("")
    SGLANG_AITER_MEGA_RANK_SYNC = EnvBool(False)
    SGLANG_AITER_MEGA_EPLB_PREFILL_ONLY = EnvBool(False)
    SGLANG_AITER_MEGA_EPLB_FUSED_MAP_RECORD = EnvBool(False)
    SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION = EnvBool(False)
    SGLANG_AITER_MLA_GLUON = EnvBool(True)
    SGLANG_AITER_MLA_DCP_DECODE_BACKEND = EnvStr("gluon")
    SGLANG_OPT_USE_AITER_SILU_MUL = EnvBool(False)
    SGLANG_OPT_USE_FUSED_QK_NORM_ROPE = EnvBool(True)
    SGLANG_OPT_FUSED_QK_NORM_ROPE_VERIFY = EnvBool(True)
    SGLANG_OPT_USE_AITER_INDEXER = EnvBool(False)
    SGLANG_USE_MLX = EnvBool(False)
    SGLANG_MLX_USE_CUSTOM_ROPE = EnvBool(False)
    SGLANG_MLX_FUSE_SWIGLU = EnvBool(False)
    SGLANG_MLX_CLEAR_CACHE_STEPS = EnvInt(256)
    SGLANG_MLX_CACHE_LIMIT_GB = EnvFloat(None)
    SGLANG_NPU_FINE_GRAINED_MOE_DUAL_STREAM = EnvBool(False)
    SGLANG_NPU_MOE_SITU_MXFP8_FUSED = EnvBool(True)
    SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD = EnvBool(False)
    SGLANG_NPU_USE_FIAS_V2_BSND = EnvBool(False)
    SGLANG_OPT_NPU_BF16_WO_A_GEMM = EnvBool(False)
    SGLANG_USE_AG_AFTER_QLORA = EnvBool(False)
    SGLANG_NPU_W4A4_NEW_PACKING = EnvBool(False)
    SGLANG_NPU_USE_TRITON_PREFIX_KV_CACHE_STORE = EnvBool(False)
    SGLANG_ZBAL_LOCAL_MEM_SIZE = EnvInt(0)
    SGLANG_ZBAL_BOOTSTRAP_URL = EnvStr("")
    SGLANG_MUSA_FA3_FORCE_UPDATE_METADATA = EnvBool(False)
    SGLANG_OPT_HOPPER_BLOCK_FP8_BF16 = EnvBool(True)
    SGLANG_GLM_NEXTN_MOE_PTPC = EnvBool(False)
    SGLANG_QUANT_ALLOW_DOWNCASTING = EnvBool(False)
    SGLANG_FORCE_MXFP8_BLOCK_CONVERT_DENSE = EnvBool(False)
    SGLANG_FP8_IGNORED_LAYERS = EnvStr("")
    SGLANG_FP4_IGNORED_LAYERS = EnvStr("")
    SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE = EnvBool(True)
    SGLANG_HUMMING_ONLINE_QUANT_CONFIG = EnvJSON(None)
    SGLANG_HUMMING_INPUT_QUANT_CONFIG = EnvJSON(None)
    SGLANG_HUMMING_USE_F16_ACCUM = EnvBool(False)
    SGLANG_HUMMING_MOE_GEMM_TYPE = EnvStr("")
    SGLANG_FLASHINFER_USE_PAGED = EnvBool(False)
    SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(None)
    SGLANG_FLASHINFER_MEGAMOE_MAX_TOKENS_PER_RANK = EnvInt(0)
    SGLANG_FLASHINFER_MEGAMOE_IN_KERNEL_FC2_REDUCE = EnvBool(False)
    SGLANG_FLASHINFER_MEGAMOE_COMBINE_DTYPE = EnvStr("bf16")
    SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION = EnvBool(False)
    SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16 = EnvBool(False)
    SGLANG_TRTLLM_MOE_PDL_MAX_TOKENS = EnvInt(8192)
    SGLANG_FLASHINFER_MOE_FUSED_FINALIZE = EnvBool(False)
    SGLANG_EXPERIMENTAL_LORA_OPTI = EnvBool(False)
    SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR = EnvFloat(None)
    SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR = EnvFloat(None)
    SGLANG_TRTLLM_MHA_DECODE_SEQ_LEN_SPLITS = EnvInt(1)
    SGLANG_SM120_FLASHMLA_BACKEND = EnvStr("flashinfer")
    SGLANG_OPT_SM120_DIRECT_SWA_KV = EnvBool(False)
    SGLANG_FLASHINFER_AUTOTUNE_CACHE = EnvBool(True)
    SGLANG_FLASHINFER_AUTOTUNE_EXTEND = EnvBool(False)
    SGLANG_DISABLE_LEAN_ATTENTION = EnvBool(False)
    SGLANG_FORCE_LEAN_GRID_CU_MULT = EnvFloat(1.0)
    SGLANG_TRITON_COMPACT_EXTEND_ATTENTION = EnvBool(True)
    SGLANG_CRASH_ON_TRITON_LOAD_AFTER_READY = EnvBool(False)
    SGLANG_TRITON_SLOW_COMPILE_THRESHOLD_SECS = EnvFloat(1.0)
    SGLANG_TRITON_LOAD_WARNING_THRESHOLD_GB = EnvFloat(1.0)
    SGLANG_MLA_DECODE_TUNE = EnvBool(False)
    SGLANG_TRITON_FP8_PREFILL_ATTN = EnvBool(True)
    SGLANG_TRITON_DENSE_PREFILL_ATTN = EnvBool(True)
    SGLANG_EPLB_P2P_BATCH_CHUNK_SIZE = EnvIntWithAlias(
        32, deprecated_name="SGLANG_EPLB_ROCM_P2P_BATCH_CHUNK_SIZE"
    )
    SGLANG_ENABLE_BF16_SPLITK_GEMM = EnvBool(True)
    SGLANG_DEEPGEMM_STANDARD_LAYOUT = EnvStr("auto")
    SGLANG_DEEPGEMM_MASKED_MEMORY_BUDGET_FRACTION = EnvFloat(0.25)
    SGLANG_OPT_DG_MASKED_M_CAP = EnvBool(False)
    SGLANG_OPT_DG_COMPACT_EAGER = EnvBool(False)
    SGLANG_OPT_MASK_DP_PAD_MOE = EnvBool(False)
    SGLANG_DEEPGEMM_SANITY_CHECK = EnvBool(False)
    SGLANG_DEEPGEMM_PDL = EnvBool(True)
    SGLANG_PP_PARALLEL_DEEPGEMM_WARMUP = EnvBool(False)
    SGLANG_CACHE_DIR = EnvStr(os.path.expanduser("~/.cache/sglang"))
    SGLANG_CUTE_AOT_CACHE_DIR = EnvStr(lambda: _default_cache_subdir("cute_aot"))
    SGLANG_JIT_CACHE_DIR = EnvStr(None)
    SGLANG_JIT_CACHE_DEBUG = EnvBool(False)
    SGLANG_JIT_CACHE_KEEP = EnvInt(None)
    SGLANG_JIT_FORCE_RECOMPILE = EnvBool(False)
    SGLANG_CRASH_ON_JIT_COMPILE = EnvBool(False)
    SGLANG_JIT_LOG_RESOURCE_USAGE = EnvBool(False)
    SGLANG_JIT_BENCHMARK_DISABLE_LOG_BANDWIDTH = EnvBool(False)
    SGLANG_JIT_BENCHMARK_DISABLE_LOG_FLOPS = EnvBool(False)
    SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(128)
    SGLANG_DEEPEP_V2_NUM_SMS = EnvInt(0)
    SGLANG_DEEPEP_V2_ENABLE_PREFILL_EXPAND = EnvBool(None)
    SGLANG_NPU_DSV4_DEEPEP_LL_DISPATCH_QUANT_MODE = EnvStr("mxfp8")
    SGLANG_ENABLE_QWEN_DEEPEP_SHARED_OVERLAP = EnvBool(True)
    SGLANG_DISABLE_STATIC_WATERFILL = EnvBool(False)
    SGLANG_NIXL_EP_BF16_DISPATCH = EnvBool(False)
    SGLANG_NIXL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(128)
    SGLANG_PPLX_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(128)
    SGLANG_ENABLE_MOE_DEFERRED_FINALIZE = EnvBool(True)
    SGLANG_MOE_DEFERRED_FINALIZE_MAX_TOKENS = EnvInt(192)
    SGLANG_OPT_MOE_QUANT_ONCE = EnvBool(False)
    SGLANG_OPT_DEEPGEMM_MEGA_MOE_RESERVED_SMS = EnvInt(2)
    SGLANG_OPT_DEEPGEMM_MEGA_MOE_FUSE_SHARED_EXPERTS = EnvBool(True)
    SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK = EnvBool(False)
    SGLANG_MINICPM_FUSE_TOPK = EnvBool(False)
    SGLANG_MINICPM_DENSE_AS_SPARSE = EnvBool(False)
    SGLANG_MINICPM_FORCE_DENSE = EnvBool(False)
    SGLANG_USE_SGL_FA3_KERNEL = EnvBool(True)
    SGLANG_FORCE_FUSED_OP_BACKEND = EnvStr(None)
    SGLANG_SANITIZE_NAN_LOGITS = EnvBool(False)
    SGLANG_ENABLE_LOGPROB_CHUNK = EnvBool(True)
    SGLANG_LOGPROB_CHUNK_SIZE = EnvInt(2048)
    SGLANG_ENABLE_FAST_INPUT_LOGPROBS = EnvBool(True)
    SGLANG_DETERMINISTIC_NCCL_NCHANNELS = EnvInt(8)
    SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB = EnvInt(16 * 1024)
    SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_PULL_SIZE_KB = EnvInt(None)
    SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_PUSH_SIZE_KB = EnvInt(None)
    SGLANG_ENABLE_PCIE_IPC_ALLREDUCE = EnvBool(False)
    SGLANG_PCIE_IPC_MAX_NUMEL = EnvInt(0)
    SGLANG_ROPE_CACHE_FP32 = EnvBool(False)
    SGLANG_ENABLE_PP_SPEC = EnvBool(False)
    SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION = EnvBool(False)
    SGLANG_ENABLE_METADATA_GLUE_GRAPH = EnvBool(False)
    SGLANG_OPT_FUSED_KDA_VERIFY = EnvBool(False)
    SGLANG_DFLASH_EAGER_DRAFT_SAMPLER = EnvBool(False)
    SGLANG_ENABLE_LILICORR_SAMPLING = EnvBool(False)
    SGLANG_LILICORR_REQUIRE_SAMPLING = EnvBool(False)
    SGLANG_RAGGED_VERIFY_MODE = EnvStr("static")
    SGLANG_TEST_RAGGED_VERIFY_FORCE_UNIFORM_CAPTURE = EnvBool(False)
    SGLANG_SPEC_SKIP_ZERO_STEP_DRAFT_EXTEND = EnvBool(False)
    SGLANG_SPEC_TP_SYNC = EnvStr("all")
    SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH = EnvBool(False)
    SGLANG_ENABLE_SPLITKV_VERIFY = EnvBool(True)
    SGLANG_VIT_ENABLE_VECTORIZED_POS_EMBED = EnvBool(True)
    SGLANG_FORCE_CPU_IMAGE_PREPROCESSING = EnvBool(False)
    SGLANG_MM_AVOID_RETOKENIZE = EnvBool(True)
    SGLANG_USE_IPC_POOL_HANDLE_CACHE = EnvBool(True)
    SGLANG_DISABLE_FUSED_MAMBA_SLOT_OPS = EnvBool(False)
    SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK = EnvBool(False)
    SGLANG_USE_BREAKABLE_CUDA_GRAPH = EnvBool(False)
    SGLANG_ENABLE_CUDA_GRAPH_DEDUP = EnvBool(False)
    SGLANG_ENABLE_GRAPH_POOL_BORROW = EnvBool(False)
    SGLANG_ENABLE_GRAPH_POOL_PRECARVE = EnvBool(False)
    SGLANG_EAGER_INPUT_NO_COPY = EnvBool(False)
    SGLANG_MAX_THINK_TOKENS = EnvInt(-1)
    SGLANG_DEFAULT_THINKING = EnvBool(False)
    SGLANG_ENCODER_GRPC_TIMEOUT_SECS = EnvInt(60)
    SGLANG_ENCODER_MM_RECEIVER_MODE = EnvStr("http")
    SGLANG_ENCODER_HTTP_TIMEOUT = EnvFloat(1800.0)
    SGLANG_ENCODER_REQ_TIMEOUT = EnvFloat(180.0)
    SGLANG_ENCODER_DISPATCH_MIN_ITEMS = EnvInt(2)
    SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU = EnvBool(False)
    SGLANG_ENCODER_MAX_BATCH_SIZE = EnvInt(8)
    SGLANG_ENCODER_PREPROC_WORKERS = EnvInt(8)
    SGLANG_ENCODER_MM_LOAD_WORKERS = EnvInt(4)
    SGLANG_ENCODER_BOOTSTRAP_HEALTH_CHECK_INTERVAL = EnvFloat(10.0)
    SGLANG_ENCODER_BOOTSTRAP_HEALTH_CHECK_TIMEOUT = EnvFloat(2.0)
    SGLANG_ENCODER_BOOTSTRAP_EVICTED_TTL = EnvFloat(600.0)
    SGLANG_EMBEDDING_POOL_SIZE_MB = EnvInt(4096)
    SGLANG_ENCODER_DP_WORKER_MAX_INFLIGHT = EnvInt(64)
    SGLANG_GRPC_PORT = EnvInt(None)
    SGLANG_GRPC_WORKER_THREADS = EnvInt(4)
    SGLANG_AUTO_NUMA_BIND = EnvBool(True)
    SGLANG_CRASH_ON_NUMA_BIND_FAILURE = EnvBool(False)
    SGLANG_DSV4_FP4_DEQUANT = EnvBool(False)
    SGLANG_DSV41_REASONING_EFFORT = EnvStr(None)
    SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE = EnvBool(False)
    SGLANG_DSV4_KV_LAYOUT = EnvStr("v4")
    SGLANG_DSV4_COMPRESSED_KV_LAYOUT = EnvStr("auto")
    SGLANG_DSV4_UNIFIED_KV_FP8 = EnvBool(False)
    SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE = EnvBool(False)
    SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT = EnvStr("shared")
    SGLANG_OPT_USE_FLASHINFER_MHC = EnvBool(False)
    SGLANG_OPT_FUSE_MHC_POST_PRE = EnvBool(True)
    SGLANG_OPT_DSV4_NONPAGED_INDEXER = EnvBool(True)
    SGLANG_OPT_DSV4_NONPAGED_INDEXER_MIN_QUERY_TOKENS = EnvInt(8192)
    SGLANG_OPT_USE_ONLINE_COMPRESS = EnvBool(False)
    SGLANG_EXPERIMENTAL_ONLINE_C128_MTP = EnvBool(False)
    SGLANG_DSV4_COMPRESS_STATE_DTYPE = EnvStr("float32")
    SGLANG_DSV41_TORCH_PREFILL_INDEXER = EnvBool(False)
    SGLANG_OPT_FLASHMLA_SPARSE_PREFILL = EnvBool(True)
    SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL = EnvBool(False)
    SGLANG_HIP_DSPARK_DRAFT_RAW_METADATA = EnvBool(_default_hip)
    SGLANG_HIP_SHARED_ACT_MXFP8 = EnvBool(_default_hip)
    SGLANG_HIP_WO_A_MXFP8 = EnvBool(_default_hip)
    SGLANG_HIP_FFN_NORM_MXFP8 = EnvBool(_default_hip)
    SGLANG_OPT_FP8_WO_A_FUSED_INVROPE = EnvBool(False)
    SGLANG_DSV41_FUSED_WO_A = EnvBool(True)
    SGLANG_OPT_USE_AITER_BATCHED_GEMM = EnvBool(False)
    SGLANG_OPT_USE_FUSED_GATE_TOPK = EnvBool(True)
    SGLANG_OPT_USE_GATE_TOPK_JIT = EnvBool(True)
    SGLANG_OPT_GATE_GEMV_MODE = EnvInt(GateGemvMode.PAIR)
    SGLANG_ENABLE_SINGLE_CG_DRAFT = EnvBool(True)
    SGLANG_OPT_USE_GUMBEL_SAMPLE = EnvBool(True)
    SGLANG_ENABLE_MTP_BOUNDARY_KV_FIX = EnvBool(True)
    SGLANG_OPT_USE_INKLING_MULTI_STREAM_OVERLAP = EnvBool(True)
    SGLANG_OPT_USE_INKLING_SHEARED_BIAS = EnvBool(True)
    SGLANG_OPT_LINEARIZED_SHARED_SINK = EnvBool(True)
    SGLANG_OPT_USE_INKLING_CUSTOM_AR = EnvBool(True)
    SGLANG_OPT_USE_INKLING_FUSED_AR_SCONV_NORM = EnvBool(True)
    SGLANG_OPT_USE_INKLING_FUSED_AR_SCONV = EnvBool(True)
    SGLANG_OPT_USE_INKLING_FUSED_ATTN_PROLOGUE = EnvBool(True)
    SGLANG_OPT_USE_INKLING_SHARED_FUSED_MOE = EnvBool(True)
    SGLANG_OPT_USE_INKLING_FUSED_AR_SHARED = EnvBool(True)
    SGLANG_OPT_USE_INKLING_FUSED_LOG_TAU = EnvBool(True)
    SGLANG_OPT_USE_INKLING_REL_PROJ_DISPATCH = EnvBool(True)
    SGLANG_OPT_INKLING_MXFP8_FUSED_QUANT_STORE = EnvBool(True)
    SGLANG_INKLING_DEFAULT_REASONING_EFFORT = EnvStr("0.9")
    SGLANG_INKLING_RS_MM_PREPROCESS = EnvBool(True)
    SGLANG_DSA_FUSE_TOPK = EnvBool(True)
    SGLANG_EXPERIMENTAL_DSA_KPOOL_METADATA_FUSION = EnvBool(True)
    SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC = EnvBool(False)
    SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK = EnvStr(None)
    SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD = EnvInt(2048)
    SGLANG_DSA_HIP_DISABLE_PRESHUFFLE = EnvBool(False)
    SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION = EnvFloat(0.2)
    SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM = EnvBool(False)
    SGLANG_DSA_TOPK_BROADCAST = EnvBool(False)
    SGLANG_DISABLE_DSA_INDEXER_FUSION = EnvBool(False)
    SGLANG_DISABLE_AITER_FUSED_FP8_DSA_INDEXER = EnvBool(False)
    SGLANG_ENABLE_DSA_Q8KV8_BORN_FP8_Q = EnvBool(False)
    SGLANG_ENABLE_DSA_Q8KV8_TOPK_LENGTH = EnvBool(False)
    SGLANG_ENABLE_DSA_Q8KV8_QPREP_OVERLAP = EnvBool(False)
    SGLANG_ENABLE_DSA_Q8KV8_KV_CAT_FUSION = EnvBool(False)
    SGLANG_OPT_Q8KV8_QPREP_VARIANT = EnvStr("auto")
    SGLANG_OPT_USE_MSA_DECODE_UNDER_GRAPH = EnvBool(False)
    SGLANG_DISABLE_M3_FP8_ATTN_GEMM = EnvBool(False)
    SGLANG_MINIMAX_M3_FUSED_SWIGLU_MXFP8 = EnvBool(False)
    SGLANG_MINIMAX_M3_FUSED_MOE_COMBINE = EnvBool(False)
    SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ = EnvInt(2)
    SGLANG_MINIMAX_M3_INDEXER_CP = EnvBool(False)
    SGLANG_OPT_MINIMAX_M3_FP8_INDEX_CACHE = EnvBool(True)
    SGLANG_OPT_USE_MINIMAX_GLUON_PREFILL = EnvBool(True)
    SGLANG_MINIMAX_NPU_PREFILL_FIA = EnvBool(True)
    SGLANG_MINIMAX_NPU_NATIVE_INDEXER = EnvBool(False)
    SGLANG_MINIMAX_NPU_NATIVE_ATTN = EnvBool(False)
    SGLANG_M3_ALLOW_CUSTOM_AR = EnvBool(False)
    SGLANG_K3_AR_FUSION = EnvBool(False)
    SGLANG_K3_SP_COLLECTIVE = EnvBool(False)
    SGLANG_K3_SP_ATTN_RES = EnvBool(False)
    SGLANG_K3_FUSED_FRONT = EnvBool(True)
    SGLANG_K3_RADIX4_TOPK = EnvBool(False)
    SGLANG_KIMI_K3_VIT_CUDA_GRAPH_CACHE_CAPACITY = EnvInt(2)
    SGLANG_KIMI_K3_VIT_CUDA_GRAPH_MIN_HITS = EnvInt(2)
    SGLANG_KIMI_K3_VIT_CUDA_GRAPH_MAX_SEQLEN = EnvInt(6144)
    SGLANG_DEBUG_SYMM_MEM = EnvBool(False)
    SGLANG_ENABLE_GDN_DECODE_FUSED_PROJ_CONV = EnvBool(True)
    SGLANG_PLATFORM = EnvStr("")
    SGLANG_PLUGINS = EnvStr("")
    SGLANG_KV_CANARY_RING_CAPACITY = EnvInt(1024)
    SGLANG_KV_CANARY_STATS_PRINT_EVERY_N_STEPS = EnvInt(100)
    SGLANG_KV_CANARY_ENABLE_WRITE_INPUT_ASSERT = EnvBool(False)
    SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_PROB = EnvFloat(0.0)
    SGLANG_KV_CANARY_PERTURB_WARMUP_STEPS = EnvInt(50)
    SGLANG_KV_CANARY_PERTURB_REAL_KV_USED_PROB = EnvFloat(0.0)
    SGLANG_KV_CANARY_PERTURB_REAL_KV_UNUSED_CACHE_PROB = EnvFloat(0.0)
    SGLANG_KV_CANARY_PERTURB_REAL_KV_POST_FORWARD_PROB = EnvFloat(0.0)
    SGLANG_KV_CANARY_PERTURB_TARGET_GROUP = EnvStr(None)
    SGLANG_KV_CANARY_PERTURB_NEXT_TOKEN_SWAP_PROB = EnvFloat(0.0)
    SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE = EnvBool(False)
    SGLANG_KV_CANARY_ENABLE_VERIFY_TOKEN_ASSERT = EnvBool(False)
    SGLANG_KV_CANARY_SWA_DIVERGENCE_STATS_INTERVAL = EnvInt(0)
    SGLANG_KV_CANARY_ENABLE_MHA_V = EnvBool(False)
    SGLANG_RUST_SERVER = EnvBool(False)
    SGLANG_RUST_BUILD_MODE = EnvStr("auto")
    SGLANG_MAX_BATCH_REQS_PER_HTTP_REQ = EnvInt(4096)
    SGLANG_WEIGHT_CACHE_SOCKET_TEMPLATE = EnvStr(
        "/tmp/sglang_weight_cache_{device_uuid}.sock"
    )
    SGLANG_WEIGHT_CACHE_READY_TEMPLATE = EnvStr(
        "/tmp/sglang_weight_cache_{device_uuid}.ready"
    )
envs = Envs()
EnvField._allow_set_name = False


from functools import lru_cache


@lru_cache(maxsize=1)
def is_large_dummy_model() -> bool:
    return os.environ.get("SGLANG_HACK_ASSERT_CKPT_VERSION") == "large-dummy"


def _print_deprecated_env(new_name: str, old_name: str):
    if old_name in os.environ:
        warnings.warn(
            f"Environment variable {old_name} will be deprecated, please use {new_name} instead"
        )
        os.environ[new_name] = os.environ[old_name]


def _warn_deprecated_env_to_cli_flag(env_name: str, suggestion: str):
    """Warn when a deprecated environment variable is used.

    This is for env vars that are deprecated in favor of CLI flags.
    """
    if env_name in os.environ:
        warnings.warn(f"Environment variable {env_name} is deprecated. {suggestion}")


def _convert_SGL_to_SGLANG():
    _print_deprecated_env("SGLANG_LOG_GC", "SGLANG_GC_LOG")
    _print_deprecated_env(
        "SGLANG_ENABLE_FLASHINFER_FP8_GEMM", "SGLANG_ENABLE_FLASHINFER_GEMM"
    )
    _print_deprecated_env(
        "SGLANG_MOE_NVFP4_DISPATCH", "SGLANG_CUTEDSL_MOE_NVFP4_DISPATCH"
    )
    _print_deprecated_env(
        "SGLANG_PREP_IN_CUDA_GRAPH", "SGLANG_ADVANCED_CUDA_GRAPH_CAPTURE"
    )
    _deprecated_ms_to_s = {
        "SGLANG_QUEUED_TIMEOUT_MS": "SGLANG_REQ_WAITING_TIMEOUT",
        "SGLANG_FORWARD_TIMEOUT_MS": "SGLANG_REQ_RUNNING_TIMEOUT",
    }
    for old_name, new_name in _deprecated_ms_to_s.items():
        if old_name in os.environ:
            ms_val = os.environ[old_name]
            warnings.warn(
                f"Environment variable {old_name} (in ms) is deprecated, "
                f"please use {new_name} (in seconds) instead"
            )
            os.environ[new_name] = str(float(ms_val) / 1000.0)

    for key, value in os.environ.items():
        if key.startswith("SGL_"):
            new_key = key.replace("SGL_", "SGLANG_", 1)
            warnings.warn(
                f"Environment variable {key} is deprecated, please use {new_key}"
            )
            os.environ[new_key] = value


_convert_SGL_to_SGLANG()

_warn_deprecated_env_to_cli_flag(
    "SGLANG_ENABLE_FLASHINFER_FP8_GEMM",
    "It will be completely removed in 0.5.7. Please use '--fp8-gemm-backend=flashinfer_trtllm' instead.",
)
_warn_deprecated_env_to_cli_flag(
    "SGLANG_ENABLE_FLASHINFER_GEMM",
    "It will be completely removed in 0.5.7. Please use '--fp8-gemm-backend=flashinfer_trtllm' instead.",
)
_warn_deprecated_env_to_cli_flag(
    "SGLANG_SUPPORT_CUTLASS_BLOCK_FP8",
    "It will be completely removed in 0.5.7. Please use '--fp8-gemm-backend=cutlass' instead.",
)
_warn_deprecated_env_to_cli_flag(
    "SGLANG_FLASHINFER_FP4_GEMM_BACKEND",
    "It will be completely removed in 0.5.9. Please use '--fp4-gemm-backend' instead.",
)
_warn_deprecated_env_to_cli_flag(
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE",
    "Please use '--enable-prefill-delayer' instead.",
)
_warn_deprecated_env_to_cli_flag(
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES",
    "Please use '--prefill-delayer-max-delay-passes' instead.",
)
_warn_deprecated_env_to_cli_flag(
    "SGLANG_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK",
    "Please use '--prefill-delayer-token-usage-low-watermark' instead.",
)

# Import cuda_coredump to trigger auto-injection of CUDA env vars
# when SGLANG_CUDA_COREDUMP=1. Best-effort; for strict guarantees,
# set CUDA_* env vars in the shell before launching Python.
import sglang.srt.debug_utils.cuda_coredump  # noqa: F401, E402


def example_with_exit_stack():
    # Use this style of context manager in unit test
    exit_stack = ExitStack()
    exit_stack.enter_context(envs.SGLANG_TEST_RETRACT.override(False))
    assert envs.SGLANG_TEST_RETRACT.get() is False
    exit_stack.close()
    assert envs.SGLANG_TEST_RETRACT.get() is None


def example_with_subprocess():
    command = ["python", "-c", "import os; print(os.getenv('SGLANG_TEST_RETRACT'))"]
    with envs.SGLANG_TEST_RETRACT.override(True):
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        process.wait()
        output = process.stdout.read().decode("utf-8").strip()
        assert output == "True"

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = process.stdout.read().decode("utf-8").strip()
    assert output == "None"


def example_with_implicit_bool_avoidance():
    @contextmanager
    def assert_throws(message_matcher: str):
        try:
            yield
        except Exception as e:
            assert message_matcher in str(e), f"{e=}"
            print(f"assert_throws find expected error: {e}")
            return
        raise AssertionError(f"assert_throws do not see exceptions")

    with assert_throws("Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"):
        if envs.SGLANG_TEST_RETRACT:
            pass

    with assert_throws("Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"):
        if (1 != 1) or envs.SGLANG_TEST_RETRACT:
            pass

    with assert_throws("Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"):
        if envs.SGLANG_TEST_RETRACT or (1 == 1):
            pass


def examples():
    # Example usage for envs
    envs.SGLANG_TEST_RETRACT.clear()
    assert envs.SGLANG_TEST_RETRACT.get() is False

    envs.SGLANG_TEST_RETRACT.set(None)
    assert envs.SGLANG_TEST_RETRACT.is_set() and envs.SGLANG_TEST_RETRACT.get() is None

    envs.SGLANG_TEST_RETRACT.clear()
    assert not envs.SGLANG_TEST_RETRACT.is_set()

    envs.SGLANG_TEST_RETRACT.set(True)
    assert envs.SGLANG_TEST_RETRACT.get() is True

    with envs.SGLANG_TEST_RETRACT.override(None):
        assert (
            envs.SGLANG_TEST_RETRACT.is_set() and envs.SGLANG_TEST_RETRACT.get() is None
        )

    assert envs.SGLANG_TEST_RETRACT.get() is True

    envs.SGLANG_TEST_RETRACT.set(None)
    with envs.SGLANG_TEST_RETRACT.override(True):
        assert envs.SGLANG_TEST_RETRACT.get() is True

    assert envs.SGLANG_TEST_RETRACT.is_set() and envs.SGLANG_TEST_RETRACT.get() is None

    example_with_exit_stack()
    example_with_subprocess()
    example_with_implicit_bool_avoidance()


if __name__ == "__main__":
    examples()


# --- imported with the qwen4 subsystem (sgl-project/sglang) ---


_NON_UTF8_PREFIX = "base64:"


# --- imported with the qwen4 subsystem (sgl-project/sglang) ---


def exportable_env_vars() -> dict[str, str]:
    return {
        field.name: _exportable_value(os.environ[field.name])
        for field in sorted(
            (value for value in vars(Envs).values() if isinstance(value, EnvField)),
            key=lambda field: field.name,
        )
        if not field.secret and field.name in os.environ
    }


# --- imported with the qwen4 subsystem (sgl-project/sglang) ---


def _exportable_value(value: str) -> str:
    try:
        value.encode()
    except UnicodeEncodeError:
        return (
            _NON_UTF8_PREFIX
            + base64.b64encode(value.encode(errors="surrogateescape")).decode()
        )
    return value


# --- imported with the qwen4 subsystem (sgl-project/sglang) ---


class _DeprecatedEnv:
    """One deprecated env var: warn if it is set, and optionally forward its
    (possibly transformed) value to a replacement env var."""

    def __init__(
        self,
        replacement: Optional[str] = None,
        transform: Optional[Callable[[str], str]] = None,
        note: Optional[str] = None,
    ):
        self.replacement = replacement
        self.transform = transform
        self.note = note

    def apply(self, old_name: str):
        if old_name not in os.environ:
            return
        message = f"Environment variable {old_name} is deprecated."
        if self.replacement is not None:
            message += f" Please use {self.replacement} instead."
        if self.note is not None:
            message += f" {self.note}"
        warnings.warn(message)
        if self.replacement is not None:
            value = os.environ[old_name]
            if self.transform is not None:
                value = self.transform(value)
            os.environ[self.replacement] = value


# --- imported with the qwen4 subsystem (sgl-project/sglang) ---


_DEPRECATED_ENVS: Dict[str, _DeprecatedEnv] = {
    "SGLANG_FLASHINFER_MNNVL_CUTEDSL_AR_FUSION": _DeprecatedEnv(
        note=(
            "Pass --flashinfer-allreduce-fusion-backend cutedsl instead. "
            "Without it an eligible model auto-enables the legacy mnnvl "
            "backend rather than the CuTe DSL fusion."
        )
    ),
    "SGLANG_FLASHINFER_MNNVL_CUTEDSL_AR_FUSION_MAX_INSTANCES": _DeprecatedEnv(
        note="One workspace per process is now an invariant, not a limit."
    ),
    # Removed without replacement.
    "SGLANG_ENABLE_CP_V2": _DeprecatedEnv(
        note="Strategy-based prefill context parallelism is now the only generic implementation."
    ),
    "SGLANG_TRACE_QWEN35_FINAL_NORM": _DeprecatedEnv(),
    "SGLANG_QWEN35_NATIVE_FINAL_NORM": _DeprecatedEnv(),
    "SGLANG_ENABLE_HICACHE_BUFFER_ANCHOR_LOCK": _DeprecatedEnv(
        note="Buffer-mode anchor pinning is always on; set "
        "SGLANG_HICACHE_BUFFER_ANCHOR_LOCK_CAP=0 to disable it."
    ),
    # Replaced by CLI flags.
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": _DeprecatedEnv(
        note="Please use '--enable-prefill-delayer' instead."
    ),
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": _DeprecatedEnv(
        note="Please use '--prefill-delayer-max-delay-passes' instead."
    ),
    "SGLANG_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK": _DeprecatedEnv(
        note="Please use '--prefill-delayer-token-usage-low-watermark' instead."
    ),
    "SGLANG_ENABLE_UNIFIED_RADIX_TREE": _DeprecatedEnv(
        note="The unified radix tree is the default tree cache now; unset this env."
    ),
}


# --- imported with the qwen4 subsystem (sgl-project/sglang) ---


def _handle_deprecated_envs():
    for old_name, deprecation in _DEPRECATED_ENVS.items():
        deprecation.apply(old_name)

    # Rewrite the legacy SGL_ prefix to SGLANG_ (names not covered above).
    for key, value in list(os.environ.items()):
        if key.startswith("SGL_") and key not in _DEPRECATED_ENVS:
            new_key = key.replace("SGL_", "SGLANG_", 1)
            warnings.warn(
                f"Environment variable {key} is deprecated, please use {new_key}"
            )
            os.environ[new_key] = value


# --- imported with the qwen4 subsystem (sgl-project/sglang) ---


def third_party_cache_defaults() -> Dict[str, str]:
    base = os.path.expanduser(envs.SGLANG_CACHE_DIR.get())
    return {
        "TRITON_CACHE_DIR": os.path.join(base, "triton"),
        "TORCHINDUCTOR_CACHE_DIR": os.path.join(base, "inductor"),
        "CUDA_CACHE_PATH": os.path.join(base, "nv"),
        # TileLang compiles the DeepSeek-V4 MHC prenorm kernels; left at its own
        # default the burst is invisible to anyone warming, mounting or baking
        # SGLANG_CACHE_DIR, and gets paid again on every cold container.
        "TILELANG_CACHE_DIR": os.path.join(base, "tilelang"),
        # FlashInfer appends ".cache/flashinfer" to this base itself, so this
        # is the base dir rather than the final cache dir.
        "FLASHINFER_WORKSPACE_BASE": base,
    }


# --- imported with the qwen4 subsystem (sgl-project/sglang) ---


def redirect_third_party_caches():
    """Point third-party JIT caches at SGLANG_CACHE_DIR, so a run's compiled
    kernels can be cleaned, warmed or volume-mounted as one directory.

    Must be called early. The redirect silently does nothing if either of
    these has already happened:

    - FlashInfer was imported. It resolves its workspace at import time.
    - Inductor made its first ``cache_dir()`` call. That call setdefaults
      TORCHINDUCTOR_CACHE_DIR itself.
    """
    for key, value in third_party_cache_defaults().items():
        os.environ.setdefault(key, value)
