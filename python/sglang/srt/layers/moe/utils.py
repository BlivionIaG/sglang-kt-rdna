from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import TYPE_CHECKING, NamedTuple, Optional

from sglang.srt.distributed.parallel_state import get_moe_expert_parallel_world_size
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    get_attention_dp_size,
    is_dp_attention_enabled,
)
from sglang.srt.runtime_context import get_exec, get_forward, get_parallel

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


AITER_PADDING_SIZE = 128
TRITON_PADDING_SIZE = 128


def get_moe_padding_size(is_aiter_moe):
    """Per-kernel padding unit (upstream sgl-project/sglang).

    Note: shadowed name with `sglang.srt.configs.update_config.get_moe_padding_size`
    (different signature/semantics — that one takes weight_block_size).
    Callers should import from the module they need.
    """
    if is_aiter_moe:
        return AITER_PADDING_SIZE
    return (
        TRITON_PADDING_SIZE
        if bool(int(os.getenv("SGLANG_MOE_PADDING", "0")))
        else 0
    )


class MoeA2ABackend(Enum):

    NONE = "none"
    DEEPEP = "deepep"
    MOONCAKE = "mooncake"
    MORI = "mori"
    ASCEND_FUSEEP = "ascend_fuseep"
    FLASHINFER = "flashinfer"

    @classmethod
    def _missing_(cls, value):
        if value is None:
            return cls.NONE
        for member in cls:
            if value == member.value:
                return member
        raise ValueError(f"No {cls.__name__} member for value {value}")

    def is_none(self):
        return self == MoeA2ABackend.NONE

    def is_deepep(self):
        return self == MoeA2ABackend.DEEPEP

    def is_mooncake(self):
        return self == MoeA2ABackend.MOONCAKE

    def is_flashinfer(self):
        return self == MoeA2ABackend.FLASHINFER

    def is_ascend_fuseep(self):
        return self == MoeA2ABackend.ASCEND_FUSEEP

    def is_mori(self):
        return self == MoeA2ABackend.MORI


class _MoeRunnerBackendPredicates:
    value: str

    def is_auto(self):
        return self.value == MoeRunnerBackend.AUTO.value

    def is_hpc_ops(self):
        return self.value == MoeRunnerBackend.HPC_OPS.value

    def is_deep_gemm(self):
        return self.value == MoeRunnerBackend.DEEP_GEMM.value

    def is_triton(self):
        return self.value == MoeRunnerBackend.TRITON.value

    def is_ascend(self):
        return self.value == MoeRunnerBackend.ASCEND.value

    def is_triton_kernels(self):
        return self.value == MoeRunnerBackend.TRITON_KERNELS.value

    def is_flashinfer_trtllm(self):
        # experimental_sgl_trtllm shares the TRT-LLM FP8 kernels + layout, so it inherits
        # trtllm weight-prep here; divergent sites check is_experimental_sgl_trtllm() first.
        return self.value in (
            MoeRunnerBackend.FLASHINFER_TRTLLM.value,
            MoeRunnerBackend.EXPERIMENTAL_SGL_TRTLLM.value,
        )

    def is_experimental_sgl_trtllm(self):
        return self.value == MoeRunnerBackend.EXPERIMENTAL_SGL_TRTLLM.value

    def is_flashinfer_trtllm_routed(self):
        return self.value == MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED.value

    def is_flashinfer_cutlass(self):
        return self.value == MoeRunnerBackend.FLASHINFER_CUTLASS.value

    def is_flashinfer_cutedsl(self):
        return self.value == MoeRunnerBackend.FLASHINFER_CUTEDSL.value

    def is_flashinfer_megamoe(self):
        return self.value == MoeRunnerBackend.FLASHINFER_MEGAMOE.value

    def is_flashinfer_mxfp4(self):
        return self.value == MoeRunnerBackend.FLASHINFER_MXFP4.value

    def is_cutlass(self):
        return self.value == MoeRunnerBackend.CUTLASS.value

    def is_marlin(self):
        # experimental_sgl_marlin shares the marlin weight repack, quant-method
        # selection, and base fused path; divergent sites (the LoRA MoE dispatch)
        # check is_experimental_sgl_marlin() first.
        return self.value in (
            MoeRunnerBackend.MARLIN.value,
            MoeRunnerBackend.EXPERIMENTAL_SGL_MARLIN.value,
        )

    def is_experimental_sgl_marlin(self):
        return self.value == MoeRunnerBackend.EXPERIMENTAL_SGL_MARLIN.value

    def is_humming(self):
        return self.value == MoeRunnerBackend.HUMMING.value

    def is_aiter(self):
        return self.value == MoeRunnerBackend.AITER.value

    def is_intel_xpu(self):
        return self.value == MoeRunnerBackend.INTEL_XPU.value


class MoeRunnerBackend(_MoeRunnerBackendPredicates, Enum):
    AUTO = "auto"
    DEEP_GEMM = "deep_gemm"
    TRITON = "triton"
    TRITON_KERNELS = "triton_kernel"
    ASCEND = "ascend"
    FLASHINFER_TRTLLM = "flashinfer_trtllm"
    EXPERIMENTAL_SGL_TRTLLM = "experimental_sgl_trtllm"
    FLASHINFER_TRTLLM_ROUTED = "flashinfer_trtllm_routed"
    FLASHINFER_CUTLASS = "flashinfer_cutlass"
    FLASHINFER_MXFP4 = "flashinfer_mxfp4"
    FLASHINFER_CUTEDSL = "flashinfer_cutedsl"
    FLASHINFER_MEGAMOE = "flashinfer_megamoe"
    CUTLASS = "cutlass"
    MARLIN = "marlin"
    HUMMING = "humming"
    EXPERIMENTAL_SGL_MARLIN = "experimental_sgl_marlin"
    AITER = "aiter"
    HPC_OPS = "hpc_ops"
    INTEL_XPU = "intel_xpu"

    def is_auto(self):
        return self == MoeRunnerBackend.AUTO

    def is_deep_gemm(self):
        return self == MoeRunnerBackend.DEEP_GEMM

    def is_triton(self):
        return self == MoeRunnerBackend.TRITON

    def is_triton_kernels(self):
        return self == MoeRunnerBackend.TRITON_KERNELS

    def is_flashinfer_trtllm(self):
        return self == MoeRunnerBackend.FLASHINFER_TRTLLM

    def is_flashinfer_cutlass(self):
        return self == MoeRunnerBackend.FLASHINFER_CUTLASS

    def is_flashinfer_cutedsl(self):
        return self == MoeRunnerBackend.FLASHINFER_CUTEDSL

    def is_flashinfer_mxfp4(self):
        return self == MoeRunnerBackend.FLASHINFER_MXFP4

    def is_cutlass(self):
        return self == MoeRunnerBackend.CUTLASS

    def is_marlin(self):
        return self == MoeRunnerBackend.MARLIN



# Added with the qwen4 subsystem (sgl-project/sglang): the union alias the
# newer MoE code annotates `get_moe_runner_backend()` with.




# --- imported with the qwen4 subsystem (sgl-project/sglang) -------------
@dataclass(frozen=True)
class RegisteredMoeRunnerBackend(_MoeRunnerBackendPredicates):
    """Identifier for an MoE runner backend supplied by an extension."""

    value: str


MoeRunnerBackendLike = MoeRunnerBackend | RegisteredMoeRunnerBackend
_REGISTERED_MOE_RUNNER_BACKEND_NAMES: set[str] = set()


def register_moe_runner_backend_name(name: str) -> None:
    """Register a backend name supplied by an out-of-tree extension."""

    if not name:
        raise ValueError("MoE runner backend name must not be empty")
    try:
        MoeRunnerBackend(name)
    except ValueError:
        _REGISTERED_MOE_RUNNER_BACKEND_NAMES.add(name)
    else:
        raise ValueError(f"MoE runner backend {name!r} is already built in")


def resolve_moe_runner_backend(
    backend: str | MoeRunnerBackendLike,
) -> MoeRunnerBackendLike:
    """Resolve a built-in or registered backend identifier."""

    if isinstance(backend, (MoeRunnerBackend, RegisteredMoeRunnerBackend)):
        return backend
    try:
        return MoeRunnerBackend(backend)
    except ValueError:
        if backend in _REGISTERED_MOE_RUNNER_BACKEND_NAMES:
            return RegisteredMoeRunnerBackend(backend)
        raise ValueError(
            f"MoE runner backend {backend!r} is neither built in nor registered"
        ) from None


class DeepEPMode(Enum):

    NORMAL = "normal"
    LOW_LATENCY = "low_latency"
    AUTO = "auto"

    def enable_normal(self) -> bool:
        return self in [DeepEPMode.NORMAL, DeepEPMode.AUTO]

    def enable_low_latency(self) -> bool:
        return self in [DeepEPMode.LOW_LATENCY, DeepEPMode.AUTO]

    def resolve(self, is_extend_in_batch: bool) -> DeepEPMode:
        if self != DeepEPMode.AUTO:
            return self

        if is_extend_in_batch:
            return DeepEPMode.NORMAL
        else:
            return DeepEPMode.LOW_LATENCY

    def is_normal(self) -> bool:
        return self == DeepEPMode.NORMAL

    def is_low_latency(self) -> bool:
        return self == DeepEPMode.LOW_LATENCY

    def is_auto(self) -> bool:
        return self == DeepEPMode.AUTO


MOE_A2A_BACKEND: Optional[MoeA2ABackend] = None
MOE_RUNNER_BACKEND: Optional[MoeRunnerBackend] = None
SPECULATIVE_MOE_RUNNER_BACKEND: Optional[MoeRunnerBackend] = None
SPECULATIVE_MOE_A2A_BACKEND: Optional[MoeA2ABackend] = None
DEEPEP_MODE: Optional[DeepEPMode] = None
IS_TBO_ENABLED: Optional[bool] = None
IS_SBO_ENABLED: Optional[bool] = None
TBO_TOKEN_DISTRIBUTION_THRESHOLD: Optional[float] = None
DEEPEP_CONFIG: Optional[str] = None
DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER: Optional[bool] = None
MOE_QUANTIZATION: Optional[str] = None
DISABLE_KT_EP_WRAPPER: bool = False


def initialize_moe_config(server_args: ServerArgs):
    global MOE_A2A_BACKEND
    global MOE_RUNNER_BACKEND
    global SPECULATIVE_MOE_RUNNER_BACKEND
    global SPECULATIVE_MOE_A2A_BACKEND
    global DEEPEP_MODE
    global DEEPEP_CONFIG
    global IS_TBO_ENABLED
    global IS_SBO_ENABLED
    global TBO_TOKEN_DISTRIBUTION_THRESHOLD
    global DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER
    global MOE_QUANTIZATION
    global DISABLE_KT_EP_WRAPPER

    MOE_A2A_BACKEND = MoeA2ABackend(server_args.moe_a2a_backend)
    MOE_RUNNER_BACKEND = MoeRunnerBackend(server_args.moe_runner_backend)
    SPECULATIVE_MOE_RUNNER_BACKEND = (
        MoeRunnerBackend(server_args.speculative_moe_runner_backend)
        if server_args.speculative_moe_runner_backend is not None
        else MOE_RUNNER_BACKEND
    )
    SPECULATIVE_MOE_A2A_BACKEND = (
        MoeA2ABackend(server_args.speculative_moe_a2a_backend)
        if server_args.speculative_moe_a2a_backend is not None
        else MOE_A2A_BACKEND
    )
    DEEPEP_MODE = DeepEPMode(server_args.deepep_mode)
    DEEPEP_CONFIG = server_args.deepep_config or ""
    IS_TBO_ENABLED = server_args.enable_two_batch_overlap
    IS_SBO_ENABLED = server_args.enable_single_batch_overlap
    TBO_TOKEN_DISTRIBUTION_THRESHOLD = server_args.tbo_token_distribution_threshold
    DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER = (
        server_args.disable_flashinfer_cutlass_moe_fp4_allgather
    )
    MOE_QUANTIZATION = server_args.quantization
    DISABLE_KT_EP_WRAPPER = False


def get_moe_a2a_backend() -> MoeA2ABackend:
    global MOE_A2A_BACKEND
    if MOE_A2A_BACKEND is None:
        MOE_A2A_BACKEND = MoeA2ABackend.NONE
    return MOE_A2A_BACKEND


def get_moe_runner_backend() -> MoeRunnerBackend:
    global MOE_RUNNER_BACKEND
    if MOE_RUNNER_BACKEND is None:
        MOE_RUNNER_BACKEND = MoeRunnerBackend.AUTO
    return MOE_RUNNER_BACKEND


def get_speculative_moe_runner_backend() -> MoeRunnerBackend:
    global SPECULATIVE_MOE_RUNNER_BACKEND
    if SPECULATIVE_MOE_RUNNER_BACKEND is None:
        logger.warning(
            "SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend"
        )
        SPECULATIVE_MOE_RUNNER_BACKEND = MoeRunnerBackend.AUTO
    return SPECULATIVE_MOE_RUNNER_BACKEND


def get_speculative_moe_a2a_backend() -> MoeA2ABackend:
    global SPECULATIVE_MOE_A2A_BACKEND
    if SPECULATIVE_MOE_A2A_BACKEND is None:
        logger.warning(
            "SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend"
        )
        SPECULATIVE_MOE_A2A_BACKEND = MoeA2ABackend.NONE
    return SPECULATIVE_MOE_A2A_BACKEND


def get_deepep_mode() -> DeepEPMode:
    global DEEPEP_MODE
    if DEEPEP_MODE is None:
        logger.warning("DEEPEP_MODE is not initialized, using auto mode")
        DEEPEP_MODE = DeepEPMode.AUTO
    return DEEPEP_MODE


def get_deepep_config() -> str:
    global DEEPEP_CONFIG
    if DEEPEP_CONFIG is None:
        logger.warning("DEEPEP_CONFIG is not initialized, using default config")
        DEEPEP_CONFIG = ""
    return DEEPEP_CONFIG


def is_tbo_enabled() -> bool:
    global IS_TBO_ENABLED
    if IS_TBO_ENABLED is None:
        IS_TBO_ENABLED = False
    return IS_TBO_ENABLED


def is_sbo_enabled() -> bool:
    global IS_SBO_ENABLED
    if IS_SBO_ENABLED is None:
        IS_SBO_ENABLED = False
    return IS_SBO_ENABLED


def get_tbo_token_distribution_threshold() -> float:
    global TBO_TOKEN_DISTRIBUTION_THRESHOLD
    if TBO_TOKEN_DISTRIBUTION_THRESHOLD is None:
        logger.warning(
            "TBO_TOKEN_DISTRIBUTION_THRESHOLD is not initialized, using 0.48"
        )
        TBO_TOKEN_DISTRIBUTION_THRESHOLD = 0.48
    return TBO_TOKEN_DISTRIBUTION_THRESHOLD


def filter_moe_weight_param_global_expert(name, x, num_local_experts):
    """
    Filter out for MoE expert parameters that requires global expert.
    """
    return (
        not getattr(x, "_sglang_require_global_experts", False)
        and x.data.ndim > 0
        and x.data.shape[0] == num_local_experts
    )


def should_use_flashinfer_cutlass_moe_fp4_allgather():
    """
    Perform FP4 quantize before all-gather for flashinfer cutlass moe to reduce communication cost for high-throughput serving.
    """
    return (
        not DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER
        and get_moe_a2a_backend().is_none()
        and get_moe_runner_backend().is_flashinfer_cutlass()
        and is_dp_attention_enabled()
        and MOE_QUANTIZATION == "modelopt_fp4"
        and get_moe_expert_parallel_world_size() == get_attention_dp_size()
    )


@contextmanager
def speculative_moe_backend_context():
    """
    Context manager to temporarily use the speculative MoE backend for draft model operations.
    This ensures that draft models in speculative decoding use the configured speculative backend.
    """
    global MOE_RUNNER_BACKEND
    original_backend = MOE_RUNNER_BACKEND
    try:
        MOE_RUNNER_BACKEND = get_speculative_moe_runner_backend()
        yield
    finally:
        MOE_RUNNER_BACKEND = original_backend


@contextmanager
def speculative_moe_a2a_backend_context():
    """
    Context manager to temporarily use the speculative MoE A2A backend for draft model operations.
    This ensures that draft models in speculative decoding use the configured speculative A2A backend.
    """
    global MOE_A2A_BACKEND
    global DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER
    original_backend = MOE_A2A_BACKEND
    original_disable_flashinfer_cutlass_moe_fp4_allgather = (
        DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER
    )
    try:
        MOE_A2A_BACKEND = get_speculative_moe_a2a_backend()
        # Disable FP4 allgather for spec decode since MTP layers are unquantized
        DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER = True
        yield
    finally:
        MOE_A2A_BACKEND = original_backend
        DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER = (
            original_disable_flashinfer_cutlass_moe_fp4_allgather
        )


def is_kt_ep_wrapper_disabled() -> bool:
    """Check if KT EP wrapper is disabled (for draft models in speculative decoding)."""
    global DISABLE_KT_EP_WRAPPER
    return DISABLE_KT_EP_WRAPPER


@contextmanager
def speculative_kt_ep_disabled_context():
    """
    Context manager to disable KT EP wrapper for draft model operations.
    Ensures draft models use pure GPU MoE instead of CPU-GPU hybrid computation
    via kt_ep_wrapper.
    """
    global DISABLE_KT_EP_WRAPPER
    original_value = DISABLE_KT_EP_WRAPPER
    try:
        DISABLE_KT_EP_WRAPPER = True
        yield
    finally:
        DISABLE_KT_EP_WRAPPER = original_value


# The type of method in top-K routing, for use in torch custom op
# Please keep this in sync with the counterpart defined in https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/trtllm/fused_moe/runner.h
class RoutingMethodType(IntEnum):
    # Default: Softmax -> TopK
    Default = (0,)
    # Renormalize: TopK -> Softmax
    Renormalize = (1,)
    # DeepSeekV3: Sigmoid -> RoutingBiasAdd -> Top2 in group -> Top4 groups -> Top8 experts from the Top4 groups
    DeepSeekV3 = (2,)
    # Llama4: Top1 -> Sigmoid
    Llama4 = (3,)
    # Qwen3: Softmax -> TopK -> Renormalize
    RenormalizeNaive = (4,)
    # TopK only (no softmax)
    TopK = (5,)
    # Unspecified
    Unspecified = 6


def is_moe_input_scattered_across_dp_ranks() -> bool:
    """Whether sparse MoE routing runs on a DP-local token shard.

    Upstream's predicate is ``a2a backend != none`` OR
    ``should_use_flashinfer_cutlass_moe_fp4_allgather()`` OR
    ``get_parallel().dwdp_size > 1``. This fork has no ``dwdp_size`` on its
    parallel context, so that third clause is expressed with the MoE
    data-parallel width it does expose -- ``> 1`` is the same condition,
    "there is more than one DP rank to scatter across". The first two clauses
    are taken as-is because both helpers exist here unchanged.
    """
    if not get_moe_a2a_backend().is_none():
        return True
    try:
        if should_use_flashinfer_cutlass_moe_fp4_allgather():
            return True
    except Exception:
        pass
    try:
        from sglang.srt.distributed.parallel_state import (
            get_moe_data_parallel_world_size,
        )

        return get_moe_data_parallel_world_size() > 1
    except Exception:
        return False


# --- imported with the qwen4 subsystem (sgl-project/sglang) -----------------
# `layers/layer_boundary/` (exit.py, layout.py, fusions/allreduce.py) imports
# these four from the `moe` package to decide where an MoE output's all-reduce
# runs. They are the same predicates upstream defines; this fork lacks the
# `post_experts_output_is_complete` / `get_lora` helpers they lean on, so those
# clauses are expressed with the equivalents this fork has. `post_experts_
# reduction_group` returning the wrong group would silently reduce over the
# wrong ranks, so the fallbacks here are deliberately conservative.


def can_merge_post_experts_all_reduce() -> bool:
    """Whether the EP and MoE-TP reductions can collapse into one _TP all-reduce.

    True when moe_dp_size == 1: the two groups are an orthogonal decomposition
    of _TP, so reducing over each in turn equals one _TP reduction.
    """
    parallel = get_parallel()
    moe_dp_size = getattr(parallel, "moe_dp_size", 1)
    return (
        parallel.moe_ep_size > 1
        and parallel.moe_tp_size > 1
        and moe_dp_size == 1
    )


def post_experts_reduction_group():
    """The group one all-reduce of an MoE output runs over: TP when the EP and
    MoE-TP reductions merge, otherwise EP, otherwise MoE-TP."""
    parallel = get_parallel()
    if can_merge_post_experts_all_reduce():
        return parallel.tp_group
    if parallel.moe_ep_size > 1:
        return parallel.moe_ep_group
    return parallel.moe_tp_group


def post_experts_sum_is_one_all_reduce() -> bool:
    """Whether the sum an FFN leaves out when it skips its post-experts (or
    down-projection) all-reduce is one full-precision all-reduce over the TP
    group itself, on a plain partial sum.

    This fork has no `post_experts_output_is_complete` / `get_lora` helpers, so
    those clauses are read from the state this fork does keep: a quantized
    communication path and a LoRA-enabled runner both disqualify the
    optimization, exactly as upstream's clauses do.
    """
    parallel = get_parallel()
    try:
        if get_exec().comm.enable_quant_communications:
            return False
    except Exception:
        pass
    if envs.SGLANG_SHARED_EXPERT_TP1.get():
        return False
    try:
        from sglang.srt.lora.lora_manager import get_lora

        if get_lora().enable_lora:
            return False
    except Exception:
        # No LoRA manager in this fork: the clause is vacuously false, same as
        # a runner with LoRA disabled.
        pass
    # Some MoE blocks reduce EP and MoE-TP in two steps instead of merging.
    if parallel.moe_ep_size > 1 and parallel.moe_tp_size > 1:
        return False
    return post_experts_reduction_group() is parallel.tp_group


def should_use_dp_reduce_scatterv():
    """
    Use reduce_scatterv in the standard dispatcher's combine() for DP attention
    with EP, replacing the default all-reduce + dp_scatter path.

    The reduce_scatterv group is the global TP group, while its variable split
    sizes are one entry per attention-DP rank. Therefore this optimization is
    valid only when each attention-DP shard has a single rank (attention TP=1).
    """
    parallel = get_parallel()
    if not should_use_flashinfer_cutlass_moe_fp4_allgather() \
            and not get_moe_a2a_backend().is_none():
        return False
    try:
        if not is_dp_attention_enabled():
            return False
    except Exception:
        return False
    attn_dp_size = getattr(parallel, "attn_dp_size", 1)
    return (
        attn_dp_size > 1
        and getattr(parallel, "tp_size", attn_dp_size) == attn_dp_size
        and parallel.moe_ep_size == attn_dp_size
    )


def sum_post_experts_output(hidden_states: torch.Tensor) -> torch.Tensor:
    """Complete the sum a MoE output owes over the EP and MoE-TP groups, for the
    boundary that owns it; a path the combine already summed is left alone."""
    parallel = get_parallel()
    return _post_experts_sum(
        hidden_states,
        reduce_ep=parallel.moe_ep_size > 1
        and not post_experts_output_is_complete(is_tp_path=False),
        reduce_tp=parallel.moe_tp_size > 1
        and not post_experts_output_is_complete(is_tp_path=True),
    )


def _post_experts_sum(
    hidden_states: torch.Tensor, *, reduce_ep: bool, reduce_tp: bool
) -> torch.Tensor:
    from sglang.srt.distributed.communication_op import (
        moe_expert_parallel_all_reduce,
        moe_tensor_model_parallel_all_reduce,
        tensor_model_parallel_all_reduce,
    )

    if reduce_ep and reduce_tp and can_merge_post_experts_all_reduce():
        return tensor_model_parallel_all_reduce(hidden_states)

    if reduce_ep:
        hidden_states = moe_expert_parallel_all_reduce(hidden_states)
    if reduce_tp:
        hidden_states = moe_tensor_model_parallel_all_reduce(hidden_states)
    return hidden_states


def post_experts_output_is_complete(*, is_tp_path: bool) -> bool:
    """Whether the experts' output owes no sum over the MoE-TP group
    (``is_tp_path=True``) or the EP group: the combine already summed it, or each
    rank computed its own tokens in full.

    This is a property of the MoE configuration. Whether the MoE block or a later
    step runs a sum that is still owed is decided separately.
    """
    if get_parallel().dwdp_size > 1:
        return True
    if is_tp_path and should_use_flashinfer_cutlass_moe_fp4_allgather():
        # The combine reduce-scatters back to the local tokens.
        return True
    a2a = get_moe_a2a_backend()
    # The flashinfer and pplx combines, and the megamoe kernel's internal
    # combine, sum each token's expert outputs back to its source rank.
    return a2a.is_flashinfer() or a2a.is_pplx() or a2a.is_flashinfer_megamoe()


def post_experts_all_reduce(hidden_states: torch.Tensor) -> torch.Tensor:
    """Reduce the post-experts MoE output across the EP and MoE-TP groups.

    When both are live and mergeable, issues one _TP all-reduce instead of two
    sequential ones, which also restores the invariant the fused residual+LN path
    depends on.
    """
    parallel = get_parallel()
    return _post_experts_sum(
        hidden_states,
        reduce_ep=parallel.moe_ep_size > 1
        and not should_skip_post_experts_all_reduce(is_tp_path=False),
        reduce_tp=parallel.moe_tp_size > 1
        and not should_skip_post_experts_all_reduce(is_tp_path=True),
    )


def deferred_post_experts_all_reduce(hidden_states: torch.Tensor) -> torch.Tensor:
    """Run the post-experts reduction that was deferred to allreduce fusion.

    Called when the fused residual+LN kernel cannot service the shape.
    """
    return post_experts_reduction_group().all_reduce(hidden_states)


def should_skip_mlp_all_reduce() -> bool:
    """Whether dense MLP / row-parallel projections should skip their all-reduce.

    True when the decoder published ``mlp_reduce_scatter`` (postprocess will
    reduce-scatter) on ``get_forward()``.
    """
    return get_forward().mlp_reduce_scatter


def should_skip_post_experts_all_reduce(*, is_tp_path: bool) -> bool:
    """Whether the MoE block should leave out its post-experts all-reduce: a later
    step runs it (fused into the next norm, or as the reduce-scatter back to the
    local tokens), or there is nothing to sum.

    Pass ``is_tp_path=True`` for the TP all-reduce, ``False`` for the EP one.
    """
    return should_skip_mlp_all_reduce() or post_experts_output_is_complete(
        is_tp_path=is_tp_path
    )


def reduce_moe_output(hidden_states: torch.Tensor) -> torch.Tensor:
    """All-reduce a MoE block's output (routed plus shared experts) over TP,
    unless a later step does it or there is nothing to sum."""
    from sglang.srt.distributed.communication_op import (
        tensor_model_parallel_all_reduce,
    )

    if get_parallel().tp_size > 1 and not should_skip_post_experts_all_reduce(
        is_tp_path=True
    ):
        return tensor_model_parallel_all_reduce(hidden_states)
    return hidden_states


def adds_replicated_output_to_partial() -> bool:
    """For a MoE block whose stage boundary completes its sum: whether this rank
    adds an output every TP rank holds in full, such as a shared expert
    replicated with tp_size=1, to its MoE output. While the output still owes a
    TP sum, only TP rank 0 adds it, so the sum counts it once."""
    parallel = get_parallel()
    owes_sum = parallel.tp_size > 1 and not post_experts_output_is_complete(
        is_tp_path=True
    )
    return not owes_sum or parallel.tp_rank == 0


# --- imported with the qwen4 subsystem ---
def xpu_moe_ld_padding_elems(k_dim: int, itemsize: int) -> int:
    """Extra elements to add to an XPU MoE weight's row stride (leading dim).

    The Xe20 grouped GEMM walks B row-by-row over the K dim, so the row stride
    in bytes decides which L3 set each row lands in. The L3 set index is
    derived by XOR-folding address bits; when the row byte size is a multiple
    of 2048 with an odd cofactor >= 3 (K = 3072, 7168, ... in bf16) successive
    rows collapse onto a small number of sets and thrash. Padding the stride
    (without changing the logical shape) breaks the aliasing.

    Returns 0 when the shape is already well distributed, so callers can use
    this to decide whether to allocate a padded buffer at all.
    """
    row_bytes = k_dim * itemsize
    if row_bytes <= 0 or XPU_MOE_LD_PADDING_BYTES % itemsize != 0:
        return 0
    trailing_zeros = (row_bytes & -row_bytes).bit_length() - 1
    odd_cofactor = row_bytes >> trailing_zeros
    if trailing_zeros >= 11 and odd_cofactor >= 3:
        return XPU_MOE_LD_PADDING_BYTES // itemsize
    return 0


# Unit of padding - context dependent


# --- imported with the qwen4 subsystem ---
def get_moe_weight_sizes(inter_dim, is_concat, is_packed, is_aiter_moe):
    """
    Calculate dimensions for MoE weight tensors.

    Args:
        inter_dim: Base intermediate dimension.
        is_concat: If True, fusions W1 (gate) and W3 (up) projections.
        is_packed: If True, uses 4-bit quantization (two FP4 elements per byte).
        is_aiter_moe: If True, applies Aiter-specific kernel padding alignment.
    """
    # w2_down_dim is the packing rank, but w13_up_dim not (of matrix to matmul)
    w13_up_dim = 2 * inter_dim if is_concat else inter_dim
    w2_down_dim = inter_dim // 2 if is_packed else inter_dim

    if is_aiter_moe:
        padding_size = get_moe_padding_size(True)
        align_aiter = lambda n: ((n + padding_size - 1) // padding_size) * padding_size
        is_padded = (w2_down_dim % padding_size) > 0
        if is_padded:
            # w2_down_dim, padding & aligned, unit: parameter dtype
            w2_down_dim = align_aiter(w2_down_dim)
        # up proj + gate fusion : 2x
        if is_concat:
            w13_up_dim = w2_down_dim * 2
        # packed
        if hasattr(torch, "float4_e2m1fn_x2") and is_packed:
            # w13_up_dim (row rank of matmul matrix) is not packing dim, *2 to recover
            w13_up_dim *= 2

    return (w13_up_dim, w2_down_dim, False if not is_aiter_moe else is_padded)


# --- imported with the qwen4 subsystem ---
def has_per_rank_fused_shared_slots(num_fused_shared_experts: int) -> bool:
    """Check whether this layer has fused shared experts in per-rank slots."""
    return num_fused_shared_experts > 0 and uses_per_rank_fused_shared_slots()


# --- imported with the qwen4 subsystem ---
class DispatcherOutputDtype(Enum):
    """
    Describes the dispatch output data type for DeepEP.

    - BF16: dispatch hidden states in bf16
    - FP8: dispatch hidden states in fp8
    - INT8: dispatch hidden states in int8
    - NVFP4: dispatch hidden states in nvfp4
    - MXFP4: dispatch hidden states in mxfp4 (fp4_e2m1 + e8m0 block scale)
    - MXFP8: dispatch hidden states in mxfp8 (fp8_e4m3 + e8m0 block scale)
    """

    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"
    NVFP4 = "nvfp4"
    MXFP4 = "mxfp4"
    MXFP8 = "mxfp8"


# --- imported with the qwen4 subsystem ---
def get_ascend_dispatcher_output_dtype(dispatcher):
    """
    Automatically choose the dispatch output dtype for Ascend.
    """

    # 1. Parse quant config to determine the output dtype of dispatcher
    if dispatcher.quant_config is not None:
        dispatcher_output_dtype = dispatcher.quant_config.get(
            "dispatcher_output_dtype", None
        )
        if dispatcher_output_dtype is not None:
            return DispatcherOutputDtype(dispatcher_output_dtype)

    # 2. Ascend dispatch defaults to BF16
    return DispatcherOutputDtype.BF16


# --- imported with the qwen4 subsystem ---
def get_deepep_output_dtype(self) -> DispatcherOutputDtype:
    """
    Automatically choose the dispatch output dtype for DeepEP.

    The decision follows several checks in priority order:
    0. Parse server argument.
    1. Parse deprecated environment variables.
    2. If quant_config contains input_global_scale → NVFP4 path.
    3. Parse a mode-specific dtype from quant_config.
    4. Parse a generic dtype from quant_config.
    5. If flashinfer_cutedsl or is_cutlass backend is active → BF16 (it quantizes hidden_states internally).
    6. Otherwise default for NPU → BF16 (the default for NPU).
    7. Otherwise → FP8 (the default for most models like DeepSeek-V3).
    """

    # 0. Parse server argument.
    server_args = get_server_args()
    if server_args and get_exec().moe.deepep_dispatcher_output_dtype != "auto":
        return DispatcherOutputDtype(get_exec().moe.deepep_dispatcher_output_dtype)

    # 1. Parse deprecated environment variables.
    if envs.SGLANG_DEEPEP_BF16_DISPATCH.get():
        logger.warning_once(
            "Warning: The env variable SGLANG_DEEPEP_BF16_DISPATCH deprecated "
            "and will be removed in future releases. Please use a new "
            "`--deepep-dispatcher-output-dtype bf16` argument instead."
        )
        return DispatcherOutputDtype.BF16

    # 2. NVFP4 is detected inside dispatch_a / _dispatch_core via quant_config; no need to infer here.
    if self.quant_config is not None:
        input_global_scale = self.quant_config.get("input_global_scale", None)
        if input_global_scale is not None:
            return DispatcherOutputDtype.NVFP4

        # 3. Some MoE kernels require different wire formats for prefill and
        # decode. Prefer a mode-specific override when the dispatcher exposes
        # its concrete mode (normal or low_latency).
        dispatch_mode = getattr(self, "dispatch_mode", None)
        if dispatch_mode is not None:
            mode_dispatcher_output_dtype = self.quant_config.get(
                f"{dispatch_mode.value}_dispatcher_output_dtype", None
            )
            if mode_dispatcher_output_dtype is not None:
                return DispatcherOutputDtype(mode_dispatcher_output_dtype)

        # 4. Parse quant config to determine the output dtype of dispatcher
        dispatcher_output_dtype = self.quant_config.get("dispatcher_output_dtype", None)
        if dispatcher_output_dtype is not None:
            return DispatcherOutputDtype(dispatcher_output_dtype)

    # 5. flashinfer_cutedsl / cutlass / humming expects BF16 dispatch
    if (
        get_moe_runner_backend().is_flashinfer_cutedsl()
        or get_moe_runner_backend().is_cutlass()
        or get_moe_runner_backend().is_humming()
    ):
        return DispatcherOutputDtype.BF16

    # 6. Default on NPU → BF16
    if _is_npu:
        return DispatcherOutputDtype.BF16

    # 7. Default → FP8
    return DispatcherOutputDtype.FP8


# --- imported with the qwen4 subsystem ---
class DeepEPv2Fp8ScaleFormat(NamedTuple):
    """DeepGEMM FP8 activation-scale layout expected from DeepEP v2."""

    tma_aligned: bool
    ue8m0: bool


# --- imported with the qwen4 subsystem ---
def get_deepep_v2_fp8_scale_format() -> DeepEPv2Fp8ScaleFormat:
    """Resolve the FP8 scale layout DeepEP v2 must pre-quantize into."""
    from sglang.srt.layers import deep_gemm_wrapper

    return DeepEPv2Fp8ScaleFormat(
        tma_aligned=(
            deep_gemm_wrapper.DEEPGEMM_NEED_TMA_ALIGNED_SCALES
            or deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
        ),
        ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
    )


# --- imported with the qwen4 subsystem ---
class FlashinferA2ADispatchType(Enum):
    BF16 = "bf16"
    NVFP4 = "nvfp4"
    MXFP8 = "mxfp8"


# --- imported with the qwen4 subsystem ---
def get_flashinfer_a2a_dispatch_type() -> FlashinferA2ADispatchType:
    dispatch_type = get_exec().moe.flashinfer_a2a_dispatch_type

    if dispatch_type is None:
        if envs.SGLANG_MOE_NVFP4_DISPATCH.is_set():
            return (
                FlashinferA2ADispatchType.NVFP4
                if envs.SGLANG_MOE_NVFP4_DISPATCH.get()
                else FlashinferA2ADispatchType.BF16
            )
        return FlashinferA2ADispatchType.BF16

    if dispatch_type != "auto":
        return FlashinferA2ADispatchType(dispatch_type)

    raise RuntimeError(
        "flashinfer_a2a_dispatch_type='auto' reached the published runtime "
        "configuration; ServerArgs must resolve it before publication"
    )
