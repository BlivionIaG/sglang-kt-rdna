from __future__ import annotations

import functools
import logging
from contextlib import contextmanager
from enum import IntEnum, auto
from typing import TYPE_CHECKING, List, Optional, Tuple, Set

import torch
import triton
import triton.language as tl

from sglang.srt.distributed import (
    GroupCoordinator,
    get_attn_context_model_parallel_rank,
    get_attn_context_model_parallel_world_size,
    get_attn_cp_group,
    get_attn_tensor_model_parallel_rank,
    get_attn_tensor_model_parallel_world_size,
    get_attn_tp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.utils import get_bool_env_var, is_hip
from typing import Sequence

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

_ATTN_DP_RANK: Optional[int] = None
_ATTN_DP_SIZE: Optional[int] = None
_LOCAL_ATTN_DP_SIZE: Optional[int] = None
_LOCAL_ATTN_DP_RANK: Optional[int] = None
_ENABLE_DP_ATTENTION_FLAG: bool = False

_is_hip = is_hip()
_USE_ROCM700A_WA = _is_hip and get_bool_env_var("SGLANG_USE_ROCM700A")


class DpPaddingMode(IntEnum):

    # Padding tokens to max length and then gather tokens using `all_gather_into_tensor`
    MAX_LEN = auto()
    # Padding tokens to sum length and then gather tokens using `all_reduce`
    SUM_LEN = auto()

    def is_max_len(self):
        return self == DpPaddingMode.MAX_LEN

    def is_sum_len(self):
        return self == DpPaddingMode.SUM_LEN

    @classmethod
    def get_dp_padding_mode(
        cls, is_extend_in_batch, global_num_tokens: List[int]
    ) -> DpPaddingMode:
        if is_extend_in_batch:
            return DpPaddingMode.SUM_LEN

        # we choose the mode that minimizes the communication cost
        max_len = max(global_num_tokens)
        sum_len = sum(global_num_tokens)
        if sum_len * 2 > max_len * get_attention_dp_size():
            return cls.MAX_LEN
        else:
            return cls.SUM_LEN

    @classmethod
    def get_default_mode_in_cuda_graph(cls) -> DpPaddingMode:
        # TODO(kkhuang-amd): noqa, temporary work-around for rocm 7.0.0 alpha
        # it can be safely removed later, once RCCL fixed
        if _USE_ROCM700A_WA:
            return cls.SUM_LEN
        else:
            return cls.MAX_LEN


class _DpGatheredBufferWrapper:

    _hidden_size: int
    _dtype: torch.dtype
    _device: torch.device
    _global_dp_buffer_len: int
    _local_dp_buffer_len: int
    _dp_max_padding: bool
    _global_num_tokens: Optional[List[int]]
    _is_extend_in_batch: bool

    @classmethod
    def set_metadata(cls, hidden_size: int, dtype: torch.dtype, device: torch.device):
        cls._hidden_size = hidden_size
        cls._dtype = dtype
        cls._device = device

    @classmethod
    def set_dp_buffer_len(
        cls,
        global_dp_buffer_len: int,
        local_dp_buffer_len: int,
        dp_max_padding: bool,
        global_num_tokens: Optional[List[int]] = None,
    ):
        cls._global_dp_buffer_len = global_dp_buffer_len
        cls._local_dp_buffer_len = local_dp_buffer_len
        cls._dp_max_padding = dp_max_padding
        cls._global_num_tokens = global_num_tokens

    @classmethod
    def get_global_dp_buffer(cls) -> torch.Tensor:
        with use_symmetric_memory(get_tp_group(), disabled=not cls._dp_max_padding):
            buffer = torch.empty(
                (cls._global_dp_buffer_len, cls._hidden_size),
                dtype=cls._dtype,
                device=cls._device,
            )
        return buffer

    @classmethod
    def get_local_dp_buffer(cls) -> torch.Tensor:
        with use_symmetric_memory(get_tp_group(), disabled=not cls._dp_max_padding):
            buffer = torch.empty(
                (cls._local_dp_buffer_len, cls._hidden_size),
                dtype=cls._dtype,
                device=cls._device,
            )
        return buffer

    @classmethod
    def get_global_dp_buffer_len(cls) -> int:
        return cls._global_dp_buffer_len

    @classmethod
    def get_local_dp_buffer_len(cls) -> int:
        return cls._local_dp_buffer_len

    @classmethod
    def get_dp_global_num_tokens(cls) -> List[int]:
        return cls._global_num_tokens

    @classmethod
    def get_dp_hidden_size(cls) -> int:
        return cls._hidden_size

    @classmethod
    def get_dp_dtype(cls) -> torch.dtype:
        return cls._dtype

    @classmethod
    def get_dp_device(cls) -> torch.device:
        return cls._device

    @classmethod
    def set_is_extend_in_batch(cls, is_extend_in_batch: bool):
        cls._is_extend_in_batch = is_extend_in_batch

    @classmethod
    def get_is_extend_in_batch(cls) -> bool:
        return cls._is_extend_in_batch

    @classmethod
    def is_dp_max_padding(cls) -> bool:
        return cls._dp_max_padding


def set_dp_buffer_len(
    global_dp_buffer_len: int,
    local_dp_buffer_len: int,
    dp_max_padding: bool,
    global_num_tokens: Optional[List[int]] = None,
):
    _DpGatheredBufferWrapper.set_dp_buffer_len(
        global_dp_buffer_len, local_dp_buffer_len, dp_max_padding, global_num_tokens
    )


def get_global_dp_buffer() -> torch.Tensor:
    return _DpGatheredBufferWrapper.get_global_dp_buffer()


def get_local_dp_buffer() -> torch.Tensor:
    return _DpGatheredBufferWrapper.get_local_dp_buffer()


def get_global_dp_buffer_len() -> int:
    return _DpGatheredBufferWrapper.get_global_dp_buffer_len()


def get_local_dp_buffer_len() -> int:
    return _DpGatheredBufferWrapper.get_local_dp_buffer_len()


def get_dp_global_num_tokens() -> List[int]:
    return _DpGatheredBufferWrapper.get_dp_global_num_tokens()


def get_dp_hidden_size() -> int:
    return _DpGatheredBufferWrapper.get_dp_hidden_size()


def get_dp_dtype() -> torch.dtype:
    return _DpGatheredBufferWrapper.get_dp_dtype()


def get_dp_device() -> torch.device:
    return _DpGatheredBufferWrapper.get_dp_device()


def set_is_extend_in_batch(is_extend_in_batch: bool):
    _DpGatheredBufferWrapper.set_is_extend_in_batch(is_extend_in_batch)


def get_is_extend_in_batch() -> bool:
    return _DpGatheredBufferWrapper.get_is_extend_in_batch()


def is_dp_max_padding() -> bool:
    return _DpGatheredBufferWrapper.is_dp_max_padding()


def compute_dp_attention_world_info(
    enable_dp_attention, tp_rank, tp_size, dp_size, attn_cp_size: int = 1
):
    attn_dp_size = dp_size if enable_dp_attention else 1
    attn_tp_size = tp_size // attn_dp_size // attn_cp_size
    attn_tp_rank = tp_rank % attn_tp_size

    if not enable_dp_attention:
        attn_dp_rank = 0
    else:
        # Rank layout is (dp, cp, tp) where tp is the fastest-changing dim:
        # tp_rank = (attn_dp_rank * attn_cp_size + attn_cp_rank) * attn_tp_size + attn_tp_rank
        attn_dp_rank = tp_rank // (attn_tp_size * attn_cp_size)

    return attn_tp_rank, attn_tp_size, attn_dp_rank


def compute_dp_attention_local_info(
    enable_dp_attention, tp_rank, tp_size, dp_size, moe_dense_tp_size
):
    if not enable_dp_attention:
        return tp_rank, tp_size, 0

    local_tp_size = moe_dense_tp_size if moe_dense_tp_size else tp_size
    local_tp_rank = tp_rank % local_tp_size
    local_dp_size = max(1, dp_size // (tp_size // local_tp_size))

    local_attn_tp_size = local_tp_size // local_dp_size
    local_attn_dp_rank = local_tp_rank // local_attn_tp_size
    local_attn_tp_rank = local_tp_rank % local_attn_tp_size

    return local_attn_tp_rank, local_attn_tp_size, local_attn_dp_rank


def initialize_dp_attention(
    server_args: ServerArgs,
    model_config: ModelConfig,
):
    global _ATTN_DP_RANK, _ATTN_DP_SIZE
    global _LOCAL_ATTN_DP_SIZE, _LOCAL_ATTN_DP_RANK, _ENABLE_DP_ATTENTION_FLAG
    enable_dp_attention = server_args.enable_dp_attention
    dp_size = server_args.dp_size
    moe_dense_tp_size = server_args.moe_dense_tp_size
    attn_cp_size = server_args.attn_cp_size

    _ENABLE_DP_ATTENTION_FLAG = enable_dp_attention

    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()

    _, _, _ATTN_DP_RANK = compute_dp_attention_world_info(
        enable_dp_attention, tp_rank, tp_size, dp_size, attn_cp_size
    )
    _, _, _LOCAL_ATTN_DP_RANK = compute_dp_attention_local_info(
        enable_dp_attention, tp_rank, tp_size, dp_size, moe_dense_tp_size
    )

    if enable_dp_attention:
        _ATTN_DP_SIZE = dp_size
        if moe_dense_tp_size is None:
            _LOCAL_ATTN_DP_SIZE = _ATTN_DP_SIZE
        else:
            _LOCAL_ATTN_DP_SIZE = max(1, dp_size // (tp_size // moe_dense_tp_size))
    else:
        _ATTN_DP_SIZE = 1
        _LOCAL_ATTN_DP_SIZE = 1

    _DpGatheredBufferWrapper.set_metadata(
        hidden_size=model_config.hidden_size,
        dtype=model_config.dtype,
        device=torch.device(server_args.device),
    )


def is_dp_attention_enabled() -> bool:
    return _ENABLE_DP_ATTENTION_FLAG


def is_allocation_symmetric() -> bool:
    return not is_dp_attention_enabled() or is_dp_max_padding()


def get_attention_tp_group() -> GroupCoordinator:
    return get_attn_tp_group()


def get_attention_tp_rank() -> int:
    return get_attn_tensor_model_parallel_rank()


def get_attention_tp_size() -> int:
    return get_attn_tensor_model_parallel_world_size()


def get_attention_cp_group() -> GroupCoordinator:
    return get_attn_cp_group()


def get_attention_cp_rank() -> int:
    return get_attn_context_model_parallel_rank()


def get_attention_cp_size() -> int:
    return get_attn_context_model_parallel_world_size()


def get_attention_dp_rank() -> int:
    assert _ATTN_DP_RANK is not None, "dp attention not initialized!"
    return _ATTN_DP_RANK


def get_attention_dp_size() -> int:
    assert _ATTN_DP_SIZE is not None, "dp attention not initialized!"
    return _ATTN_DP_SIZE


def get_local_attention_dp_rank() -> int:
    assert _LOCAL_ATTN_DP_RANK is not None, "dp attention not initialized!"
    return _LOCAL_ATTN_DP_RANK


def get_local_attention_dp_size() -> int:
    assert _LOCAL_ATTN_DP_SIZE is not None, "dp attention not initialized!"
    return _LOCAL_ATTN_DP_SIZE


@contextmanager
def disable_dp_size():
    """Patch the tp group temporarily until this function ends.

    This method is for draft workers of speculative decoding to run draft model
    with different tp degree from that of target model workers.

    Args:
        tp_group (GroupCoordinator): the tp group coordinator
    """
    global _ATTN_DP_SIZE
    assert _ATTN_DP_SIZE is not None, "dp attention not initialized!"

    old_dp_size = _ATTN_DP_SIZE
    _ATTN_DP_SIZE = 1
    try:
        yield
    finally:
        _ATTN_DP_SIZE = old_dp_size


def get_dp_local_info(forward_batch: ForwardBatch) -> Tuple[torch.Tensor, torch.Tensor]:
    # `get_dp_local_info` is only called in global DP gather and scatter. We use global DP rank here.
    dp_rank = get_attention_dp_rank()

    if forward_batch.dp_local_start_pos is None:
        cumtokens = torch.cumsum(forward_batch.global_num_tokens_gpu, dim=0)
        if dp_rank == 0:
            local_start_pos = torch.zeros_like(cumtokens[0])
        else:
            local_start_pos = cumtokens[dp_rank - 1]
        local_num_tokens = forward_batch.global_num_tokens_gpu[dp_rank]

        forward_batch.dp_local_start_pos = local_start_pos
        forward_batch.dp_local_num_tokens = local_num_tokens

    return forward_batch.dp_local_start_pos, forward_batch.dp_local_num_tokens


@triton.jit
def memcpy_triton_kernel(
    dst_ptr,
    src_ptr,
    offset_ptr,
    sz_ptr,
    offset_src: tl.constexpr,
    chunk_size,  # multiplied for offset and sz
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0).to(tl.int64)
    offset = tl.load(offset_ptr).to(tl.int64) * chunk_size
    sz = tl.load(sz_ptr).to(tl.int64) * chunk_size

    start_index = pid * BLOCK_SIZE
    offs = tl.arange(0, BLOCK_SIZE)
    mask = start_index + offs < sz

    if offset_src:
        data = tl.load(src_ptr + offset + start_index + offs, mask=mask)
        tl.store(dst_ptr + start_index + offs, data, mask=mask)
    else:
        data = tl.load(src_ptr + start_index + offs, mask=mask)
        tl.store(dst_ptr + offset + start_index + offs, data, mask=mask)


def prod(x):
    return functools.reduce(lambda a, b: a * b, x, 1)


def memcpy_triton(dst, src, dim, offset, sz, offset_src):
    max_size = min(src.numel(), dst.numel())
    assert dim == 0, "dim != 0 unsupported"
    assert src.shape[1:] == dst.shape[1:], "src and dst must have same shape"
    chunk_size = prod(src.shape[1:])
    BLOCK_SIZE = 8192
    grid = (triton.cdiv(max_size, BLOCK_SIZE),)

    memcpy_triton_kernel[grid](dst, src, offset, sz, offset_src, chunk_size, BLOCK_SIZE)


def _dp_gather_via_all_reduce(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    is_partial: bool,
):
    local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)

    global_tokens.fill_(0)
    assert local_tokens.is_contiguous()
    assert global_tokens.is_contiguous()

    if local_tokens.shape[0] > 0 and (is_partial or get_attention_tp_rank() == 0):
        assert (
            local_tokens.untyped_storage() is not global_tokens.untyped_storage()
        ), "aliasing between global_tokens and local_tokens not allowed"

        memcpy_triton(
            global_tokens, local_tokens, 0, local_start_pos, local_num_tokens, False
        )

    # Input IDs are in int 32. We should use inplace_all_reduce for local case because of custom all reduce.
    NUM_GPUS_PER_NODE = 8
    if (
        not local_tokens.dtype.is_floating_point
        and get_tensor_model_parallel_world_size() <= NUM_GPUS_PER_NODE
    ):
        from sglang.srt.distributed.parallel_state import inplace_all_reduce

        inplace_all_reduce(global_tokens, group_name=get_tp_group().unique_name)

    else:
        global_tokens[:] = tensor_model_parallel_all_reduce(global_tokens)


def _dp_gather_via_all_gather(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    is_partial: bool,
):
    if get_attention_tp_size() == 1:
        get_tp_group().all_gather_into_tensor(global_tokens, local_tokens)
        return

    if not is_partial:
        if get_attention_tp_rank() != 0:
            local_tokens.fill_(0)
    scattered_local_tokens = local_tokens.tensor_split(get_attention_tp_size())[
        get_attention_tp_rank()
    ]
    get_attention_tp_group().reduce_scatter_tensor(scattered_local_tokens, local_tokens)
    get_tp_group().all_gather_into_tensor(global_tokens, scattered_local_tokens)


def _dp_gather(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    is_partial: bool,
):
    if forward_batch.dp_padding_mode.is_max_len():
        _dp_gather_via_all_gather(
            global_tokens, local_tokens, forward_batch, is_partial
        )
    else:
        _dp_gather_via_all_reduce(
            global_tokens, local_tokens, forward_batch, is_partial
        )


def dp_gather_partial(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
):
    _dp_gather(global_tokens, local_tokens, forward_batch, is_partial=True)


def dp_gather_replicate(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
):
    _dp_gather(global_tokens, local_tokens, forward_batch, is_partial=False)


def dp_scatter(
    local_tokens: torch.Tensor,  # output
    global_tokens: torch.Tensor,  # input
    forward_batch: ForwardBatch,
):
    # local_num_tokens is not necessarily the same as local_tokens.shape[0],
    # since local_tokens may be padded for cuda graph
    local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)

    local_tokens.fill_(0)
    assert local_tokens.is_contiguous()
    assert global_tokens.is_contiguous()
    if local_tokens.shape[0] > 0:
        assert (
            local_tokens.untyped_storage() is not global_tokens.untyped_storage()
        ), "aliasing between local_tokens and global_tokens not allowed"

        memcpy_triton(
            local_tokens, global_tokens, 0, local_start_pos, local_num_tokens, True
        )


def dp_reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor):
    if get_tensor_model_parallel_world_size() == get_attention_dp_size():
        get_tp_group().reduce_scatter_tensor(output, input)
    else:
        scattered_local_tokens = input.tensor_split(
            get_tensor_model_parallel_world_size()
        )[get_tensor_model_parallel_rank()]
        get_tp_group().reduce_scatter_tensor(scattered_local_tokens, input)
        get_attention_tp_group().all_gather_into_tensor(output, scattered_local_tokens)


def attn_tp_reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_attention_tp_group().reduce_scatter_tensor(output, input)


def attn_cp_reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_attention_cp_group().reduce_scatter_tensor(output, input)


def attn_tp_all_reduce(input: torch.Tensor):
    return get_attention_tp_group().all_reduce(input)


def attn_tp_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_attention_tp_group().all_gather_into_tensor(output, input)


def attn_cp_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_attention_cp_group().all_gather_into_tensor(output, input)


def attn_tp_all_gather(output_list: List[torch.Tensor], input: torch.Tensor):
    return get_attention_tp_group().all_gather(input, output_tensor_list=output_list)


# --- imported with the qwen4 subsystem (sgl-project/sglang) -----------------
# `dsa/utils.py` (and the upstream lora / forward-batch paths) call
# `dp_slot_in`, which upstream implements on top of its elastic-scale-up
# gather helpers (`dp_gather_width` / `dp_gather_slot`, reading `get_flags()`
# and `get_parallel()`). This fork predates that machinery, so the same
# contract is expressed with the attention-DP accessors it already has:
# the gather width is the attention-DP replica count, and this process's slot
# is its attention-DP rank. On a single-GPU or non-DP deployment both reduce
# to the length-1 fast path, which is the case this port actually runs.


def dp_gather_width() -> int:
    """Number of entries a per-DP-replica sequence carries (the gather width)."""
    return get_attention_dp_size()


def dp_gather_slot() -> int:
    """This process's index in the DP gather."""
    return get_attention_dp_rank()


def dp_slot_in(per_rank) -> int:
    """Return this process's slot in a per-DP-replica sequence.

    The sequence carries one entry per replica of the gather this process
    takes part in, so its length is the gather width. Length one is the
    all-gather-skipped batch, which carries this process's entry alone.
    """
    if len(per_rank) == 1:
        return 0
    width = dp_gather_width()
    if len(per_rank) != width:
        raise ValueError(
            f"a per-replica sequence of {len(per_rank)} entries does not "
            f"belong to a DP gather of width {width}"
        )
    return dp_gather_slot()


def get_moe_cp_size() -> int:
    """MoE data-parallel group width.

    Upstream reads this from its parallel-context object
    (``get_parallel().moe_dp_group.world_size``). This fork keeps the group on
    ``parallel_state`` instead, so the same value comes from
    ``get_moe_data_parallel_world_size()``. A group that was never initialized
    (a non-MoE or single-rank run) is width 1, which is the answer upstream
    gives too when no MoE CP group exists -- and the case this port runs.
    """
    try:
        from sglang.srt.distributed.parallel_state import (
            get_moe_data_parallel_world_size,
        )

        return get_moe_data_parallel_world_size()
    except Exception:
        return 1


def get_moe_cp_rank() -> int:
    """This process's rank in the MoE data-parallel group (see above)."""
    try:
        from sglang.srt.distributed.parallel_state import get_moe_data_parallel_rank

        return get_moe_data_parallel_rank()
    except Exception:
        return 0


def is_enable_moe_cp_allgather() -> bool:
    """Whether the MoE CP all-gather path is active (single rank: no)."""
    return get_moe_cp_size() > 1


def moe_cp_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):
    from sglang.srt.distributed.parallel_state import get_moe_dp_group

    return get_moe_dp_group().all_gather_into_tensor(output, input)


def can_use_dp_reduce_scatter() -> bool:
    """Whether the fixed TP group tiles the current attention DP x TP layout."""
    if not world_dp_gather_enabled():
        return True

    parallel = get_parallel()
    return parallel.tp_size == parallel.num_dp_ranks * parallel.attn_tp_size


def set_local_dp_buffer_len(local_dp_buffer_len: int) -> None:
    _DpGatheredBufferWrapper.set_local_dp_buffer_len(local_dp_buffer_len)


# --- imported with the qwen4 subsystem ---
def get_dp_local_slice_cpu(
    forward_batch: ForwardBatch,
    can_run_graph: bool,
    cuda_graph_batch: Optional[int],
) -> Tuple[int, int]:
    # CPU (start, length) slice for DP-local data in a rank-padded buffer.
    # Returns Python ints (no D2H sync) and handles the cuda-graph-padded layout.
    global_num_tokens = forward_batch.global_num_tokens_cpu
    dp_rank = dp_slot_in(global_num_tokens)
    local_num_tokens = global_num_tokens[dp_rank]
    if can_run_graph:
        local_start_pos = dp_rank * cuda_graph_batch
    else:
        local_start_pos = sum(global_num_tokens[:dp_rank])
    return local_start_pos, local_num_tokens


from sglang.kernels.ops.memory.memcpy_triton import memcpy_triton
from sglang.srt.distributed.utils import all_gather_single


# TODO: write c++ kernel for cpu


# --- imported with the qwen4 subsystem ---
def mask_dp_pad_moe_topk_ids(topk_ids: torch.Tensor) -> None:
    """Set MAX_LEN pad rows' (post-translation, local) topk_ids to -1 in place.

    Under dp-attention MAX_LEN padding the gathered MoE buffer is
    [num_dp_ranks * max_len, hidden] with rank r's real rows at
    [r*max_len, r*max_len + global_num_tokens[r]); the pad rows carry stale
    hidden values, run the router, and get dispatched into experts whose
    outputs are then discarded by the post-reorder scatter — pure wasted
    compute, and a masked-grouped-GEMM workspace blow-up when they collide
    on the same top-k.  -1 is the drop sentinel both the triton fused_moe
    (filter_expert) and the DeepGEMM EP preprocess honor; it must be applied
    AFTER the local_expert_mapping gather (a pre-translation -1 aliases to
    the mapping table's last entry).  Capture-safe: per-batch state is read
    only from the replay-updated global_num_tokens_gpu tensor.
    """
    counts = _DpGatheredBufferWrapper.get_dp_global_num_tokens_gpu()
    if counts is None:
        return
    max_len = _DpGatheredBufferWrapper.get_local_dp_buffer_len()
    rows, topk = topk_ids.shape
    if max_len <= 0 or rows != counts.shape[0] * max_len:
        # Layout mismatch (e.g. non-DP or logits-path caller): do nothing.
        return
    _mask_dp_pad_topk_ids_kernel[(rows,)](
        topk_ids,
        counts,
        max_len,
        TOPK=topk,
        BLOCK=triton.next_power_of_2(topk),
    )


# --- imported with the qwen4 subsystem (sgl-project/sglang) ---


def deployment_attn_dp_size() -> int:
    """Return the deployment's attention-DP replica count.

    Draft scopes retain this count because their metadata gathers include
    the target's replicas.
    """
    parallel = get_parallel()
    return parallel.num_dp_ranks if parallel.attn_dp_enabled else 1


def world_dp_gather_enabled() -> bool:
    """Whether DP gathers should use expanded WORLD after joiner admission."""
    dp = get_flags().dp
    return dp.use_world_group_for_gather and not dp.joiner_skip_all_gather


def enable_joiner_all_gather():
    get_flags().dp.joiner_skip_all_gather = False


def update_dp_attention_post_scale(new_dp_size: int, new_dp_rank: int):
    """Switch DP gathers to the expanded WORLD.

    The caller updates the configured widths; these arguments identify the
    scale-up in the log.
    """
    get_flags().dp.use_world_group_for_gather = True
    logger.debug(
        "[Elastic EP] dp_attention switched to WORLD: num_dp_ranks=%d dp_rank=%d",
        new_dp_size,
        new_dp_rank,
    )


def set_dp_buffer_len_from_batch(forward_batch: ForwardBatch) -> None:
    """Publish the DP gather sizes ``forward_batch`` carries: the buffer
    length, the per-rank token counts as padded for the gather, this rank's
    entry, and the padding mode. Capture batches carry no separately padded
    list, so the raw counts stand in for it."""
    global_num_tokens = forward_batch.global_num_tokens_padded_cpu
    if global_num_tokens is None:
        global_num_tokens = forward_batch.global_num_tokens_cpu
    dp_rank = get_parallel().attn_dp_rank if len(global_num_tokens) > 1 else 0
    set_dp_buffer_len(
        forward_batch.global_dp_buffer_len,
        global_num_tokens[dp_rank],
        forward_batch.dp_padding_mode.is_max_len(),
        global_num_tokens,
        forward_batch.global_num_tokens_gpu,
    )


def initialize_dp_attention_flags(server_args: ServerArgs):
    """Initialize DP runtime flags without changing the worker's placement."""
    dp = get_flags().dp
    dp.enabled = get_parallel().attn_dp_enabled

    if get_exec().moe.elastic_ep_backend is not None and get_parallel().max_ep_size:
        if ep_scale_joiner_of(resolving_view(server_args)):
            dp.joiner_skip_all_gather = True


def init_dp_gathered_buffer(model_config: ModelConfig):
    """Size the gathered buffer from the model this worker is about to run."""
    get_flags().dp.max_len_with_idle = (
        getattr(model_config.hf_config, "hybrid_override_pattern", None) is not None
    )
    _DpGatheredBufferWrapper.set_metadata(
        hidden_size=model_config.hidden_size,
        dtype=model_config.dtype,
        device=torch.device(get_device().device),
    )


def get_dp_tp_group() -> GroupCoordinator:
    """The TP ranks that run one DP rank's batch: the attention-TP group with
    attention DP, otherwise the whole TP group.

    Attention-CP peers of a DP rank are members only in the second case.
    """
    parallel = get_parallel()
    return parallel.attn_tp_group if parallel.attn_dp_enabled else parallel.tp_group


def multimodal_encoder_runs_here() -> bool:
    """Whether a multimodal tower built on this worker is also forwarded here.

    A tower is built unconditionally by most models, but an encoder-disaggregated
    language instance forwards it on the encoder instance, a language-model-only
    instance rejects multimodal requests, and a PD decode instance embeds nothing
    (`general_mm_embed_routine` skips decode and target-verify forwards).

    Adaptive dispatch is the exception on the first of those: it keeps the
    requests it does not send to the encoder, and forwards the tower for them.
    """
    from sglang.srt.runtime_context import get_disagg

    disagg = get_disagg()
    offloaded_to_an_encoder_instance = (
        disagg.language_only and not disagg.enable_adaptive_dispatch_to_encoder
    )
    return not (
        offloaded_to_an_encoder_instance
        or disagg.language_model_only
        or disagg.disaggregation_mode == "decode"
    )


def reject_attn_tp_shard_with_tp_reduce(
    layer: str,
    *,
    shard_tp_size: int,
    reduces_over_attn_tp: bool,
    multimodal_encoder: bool = False,
    hint: str = "",
) -> None:
    """Reject a layer that shards over attention TP but all-reduces over the TP group.

    The two groups differ only when attention DP or attention CP makes attention
    TP narrower than TP. There the all-reduce mixes ranks that hold other
    requests or replicated inputs, so the output is wrong; a one-rank shard does
    not reduce at all.
    """
    tp_size = get_parallel().tp_size
    if reduces_over_attn_tp or not 1 < shard_tp_size < tp_size:
        return
    if multimodal_encoder and not multimodal_encoder_runs_here():
        return
    # Name only the widths that are actually narrowing the group here, so the
    # remedy is one the operator can apply.
    parallel = get_parallel()
    narrowed_by = []
    if parallel.attn_dp_size > 1:
        narrowed_by.append(f"--attn-dp-size {parallel.attn_dp_size}")
    if parallel.attn_cp_size > 1:
        narrowed_by.append(f"--attn-cp-size {parallel.attn_cp_size}")
    remedy = " and ".join(f"drop {flag}" for flag in narrowed_by) or (
        "widen the attention TP group"
    )
    raise ValueError(
        f"{layer} shards over the attention TP group ({shard_tp_size} ranks) "
        f"but all-reduces over the full TP group ({tp_size} ranks), so it does "
        f"not support {' with '.join(narrowed_by) or 'this layout'} yet. "
        f"Use --attn-dp-size equal to --tp-size, or {remedy}{hint}."
    )


def memcpy_cpu(dst, src, dim, offset, sz, offset_src):
    assert dim == 0, "Only dim=0 supported"
    assert src.shape[1:] == dst.shape[1:], "src and dst must have same trailing shape"

    total_rows_dst, total_rows_src = dst.shape[0], src.shape[0]
    dst_start, src_start = 0, 0

    if offset_src:
        # src[offset:] → dst[0:]
        src_start = offset
        dst_start = 0
    else:
        # src[0:] → dst[offset:]
        src_start = 0
        dst_start = offset

    dst_end = min(dst_start + sz, total_rows_dst)
    src_end = min(src_start + sz, total_rows_src)
    actual_sz = min(dst_end - dst_start, src_end - src_start)

    if actual_sz <= 0:
        return

    dst[dst_start : dst_start + actual_sz].copy_(src[src_start : src_start + actual_sz])


def memcpy(dst, src, dim, offset, sz, offset_src):
    memcpy_func(dst, src, dim, offset, sz, offset_src)


def _cp_shard_rows(
    forward_batch: ForwardBatch, cp_shard_counts: Sequence[int]
) -> Tuple[int, int]:
    """(start, length) of this rank's CP shard in the gathered buffer: the CP ranks'
    shards lie back to back, in CP rank order, at the start of their DP slot."""
    cp_rank = get_parallel().attn_cp_rank
    dp_start = sum(forward_batch.global_num_tokens_cpu[: dp_gather_slot()])
    return dp_start + sum(cp_shard_counts[:cp_rank]), cp_shard_counts[cp_rank]


@functools.lru_cache(maxsize=1)
def _use_dp_gather_fp8() -> bool:
    return envs.SGLANG_ENABLE_DP_GATHER_FP8.get()


def _get_dp_gather_fp8_bufs(rows: int, hidden: int, device: torch.device):
    key = str(device)
    bufs = _dp_gather_fp8_bufs.get(key)
    if bufs is None or bufs[0].shape[0] < rows:
        bufs = (
            torch.empty((rows, hidden), dtype=torch.uint8, device=device),
            torch.empty(
                (rows, hidden // _DP_GATHER_FP8_GROUP),
                dtype=torch.float32,
                device=device,
            ),
        )
        _dp_gather_fp8_bufs[key] = bufs
    return bufs[0][:rows], bufs[1][:rows]


@triton.jit
def _dequant_per_token_group_fp8_kernel(
    q_ptr,
    s_ptr,
    out_ptr,
    HIDDEN: tl.constexpr,
    NGROUPS: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    # HIDDEN may not be a multiple of BLOCK (e.g. DeepSeek 7168 vs BLOCK
    # 2048): the tail iteration must be masked or it reads/writes up to
    # BLOCK-1 elements past the row (cross-row corruption + OOB on the last
    # row).  HIDDEN is constexpr, so the mask folds away when it divides.
    for start in tl.static_range(0, HIDDEN, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < HIDDEN
        qv = tl.load(q_ptr + row * HIDDEN + offs, mask=mask, other=0.0).to(tl.float32)
        sv = tl.load(s_ptr + row * NGROUPS + offs // GROUP, mask=mask, other=0.0)
        tl.store(out_ptr + row * HIDDEN + offs, (qv * sv).to(tl.bfloat16), mask=mask)


@triton.jit
def _mask_dp_pad_topk_ids_kernel(
    topk_ids_ptr,
    counts_ptr,
    max_len,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    rank = row // max_len
    pos = row % max_len
    valid = pos < tl.load(counts_ptr + rank)
    if valid == 0:
        offs = tl.arange(0, BLOCK)
        tl.store(topk_ids_ptr + row * TOPK + offs, -1, mask=offs < TOPK)


def _dp_gather_via_all_gatherv_fp8(
    global_tokens: torch.Tensor,
    local_real: torch.Tensor,
    sizes: List[int],
):
    """fp8 wire format for the variable-length DP gather: quantize the local
    rows per-token-group (the SAME group-128 quantization the MoE expert GEMMs
    apply to their input downstream), gather payload (as uint8 — NCCL has no
    fp8 dtype; the gatherv leg is broadcast-only so a byte view is safe) and
    scales in two output-buffered gatherv calls, then dequantize into the
    bf16 global buffer.  Zero pad rows quantize to (q=0, s=eps) and so
    dequantize back to exact zeros — the MoE-tail invariant is preserved.
    The combine leg (reduce_scatterv) stays bf16: NCCL SUM cannot run on fp8."""
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    rows = global_tokens.shape[0]
    hidden = global_tokens.shape[-1]
    q, s = sglang_per_token_group_quant_fp8(
        local_real.contiguous(), _DP_GATHER_FP8_GROUP
    )
    gq, gs = _get_dp_gather_fp8_bufs(rows, hidden, global_tokens.device)
    tp_group = get_parallel().tp_group
    tp_group.all_gatherv(q.view(torch.uint8), sizes=sizes, output=gq)
    tp_group.all_gatherv(s, sizes=sizes, output=gs)
    _dequant_per_token_group_fp8_kernel[(rows,)](
        gq.view(torch.float8_e4m3fn),
        gs,
        global_tokens,
        HIDDEN=hidden,
        NGROUPS=hidden // _DP_GATHER_FP8_GROUP,
        GROUP=_DP_GATHER_FP8_GROUP,
        BLOCK=2048,
    )


def is_dp_gatherv_active() -> bool:
    """Variable-length DP-MoE gather/scatter (all_gatherv + reduce_scatterv) is
    enabled and applicable to the CURRENT forward. Requires:
      - env SGLANG_DP_USE_GATHERV (default off),
      - supported layout (attn_tp_size==1, tp_size==attn_dp_size),
      - SUM_LEN padding mode. The gatherv pair (all_gatherv + reduce_scatterv) is
        only valid under SUM_LEN; under MAX_LEN the buffer is equal-padded and the
        gather/combine use all_gather / (aiter) reduce_scatter instead. Reading the
        per-forward padding via _DpGatheredBufferWrapper.is_dp_max_padding() (set by
        set_dp_buffer_len) keeps callers that lack a ForwardBatch (e.g.
        dp_reduce_scatter_tensor) consistent."""
    return (
        _USE_DP_GATHERV
        and not world_dp_gather_enabled()
        and get_parallel().attn_tp_size == 1
        and get_parallel().tp_size == get_parallel().attn_dp_size
        and not _DpGatheredBufferWrapper.is_dp_max_padding()
    )


def _dp_gatherv_sizes(forward_batch) -> Optional[List[int]]:
    """Per-rank CPU token counts for the buffer being gathered. The MoE gather
    passes a ForwardBatch (global_num_tokens_cpu); the logits gather passes a
    LogitsMetadata (global_num_tokens_for_logprob_cpu). Return the sizes that
    match the LOCAL tensor for this context, or None to fall back."""
    sizes = getattr(forward_batch, "global_num_tokens_for_logprob_cpu", None)
    if sizes is None:
        sizes = getattr(forward_batch, "global_num_tokens_cpu", None)
    if sizes is None:
        return None
    try:
        return [int(x) for x in sizes]
    except (TypeError, ValueError):
        return None


def _dp_gather_via_all_gatherv(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    is_partial: bool,
    sizes: List[int],
):
    # attn_tp_size == 1: each DP rank contributes exactly `sizes[rank]` rows.
    # CRITICAL: the MoE downstream runs on the WHOLE `global_tokens` buffer
    # (M = global_tokens.shape[0]), so the gather MUST fill every row. We pad
    # each rank's local tensor up to sizes[rank] with zeros (matching the
    # buffer's reserved per-rank slot) so sum(sizes) == buffer rows and there
    # is no uninitialized tail for the MoE to read.
    rank = dp_slot_in(sizes)
    local_rows = sizes[rank]
    if local_tokens.shape[0] == local_rows:
        local_real = local_tokens
    elif local_tokens.shape[0] > local_rows:
        local_real = local_tokens[:local_rows]
    else:
        local_real = local_tokens.new_zeros((local_rows, *local_tokens.shape[1:]))
        local_real[: local_tokens.shape[0]].copy_(local_tokens)
    # sum(sizes) == global_tokens.shape[0] is guaranteed by the caller (else it
    # falls back to all_reduce). Pass global_tokens as the NCCL output buffer so
    # the gather writes directly into it -- avoids the previous extra full-buffer
    # torch.cat + copy_ (two ~sum(sizes)*hidden DtoD copies, ~700us/layer at c512).
    # NOTE: the fp8 branch condition must be identical on EVERY DP rank (all
    # ranks must issue the same NCCL op sequence) — env/dtype/hidden are
    # rank-uniform; never gate on per-rank state like forward_mode (ranks can
    # be extend/idle-mixed within one global forward).  Prefill-only is
    # already structural: the gatherv path runs only under SUM_LEN padding,
    # which decode-only steps and CUDA-graph capture never select.
    if (
        _use_dp_gather_fp8()
        and global_tokens.dtype == torch.bfloat16
        and global_tokens.shape[-1] % _DP_GATHER_FP8_GROUP == 0
    ):
        _dp_gather_via_all_gatherv_fp8(global_tokens, local_real, sizes)
        return
    get_parallel().tp_group.all_gatherv(local_real, sizes=sizes, output=global_tokens)


def _note_dp_gather_in_prefill_graph() -> None:
    dp = get_flags().dp
    if dp.capturing_prefill_graph:
        dp.prefill_graph_has_dp_gather = True


def get_dp_tbo_comm_stream() -> torch.cuda.Stream:

    return get_stream("dp_tbo_comm")


def _tbo_event(key) -> torch.cuda.Event:

    pool = get_resources().tbo_event_pool
    ev = pool.get(key)
    if ev is None:
        ev = torch.cuda.Event()
        pool[key] = ev
    return ev


def dp_gather_partial_async(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    event_key=("gather", 0),
) -> torch.cuda.Event:
    """Launch `dp_gather_partial` (all_gatherv) on the shared DP TBO comm stream;
    re-record + return a PERSISTENT event (keyed by `event_key`) that fires when
    the gather completes. Caller yields, then `compute_stream.wait_event(ev)`
    before reading `global_tokens`."""
    comm = get_dp_tbo_comm_stream()
    compute = torch.cuda.current_stream()
    # Keep buffers alive across streams (caching allocator).
    local_tokens.record_stream(comm)
    global_tokens.record_stream(comm)
    ev = _tbo_event(event_key)
    with torch.cuda.stream(comm):
        comm.wait_stream(compute)  # inputs were produced on the compute stream
        dp_gather_partial(global_tokens, local_tokens, forward_batch)
        ev.record(comm)
    return ev


def get_tbo_persistent_buffer(
    key, rows: int, hidden: int, dtype: torch.dtype, device
) -> torch.Tensor:
    """Return a [rows, hidden] view of a grow-only persistent buffer for `key`.
    Reallocates only when the request exceeds the cached capacity / changes
    dtype|hidden. Caller must treat the returned view as scratch (overwritten)."""
    buf = _TBO_PERSIST_BUF.get(key)
    cap = 0 if buf is None else buf.shape[0]
    if buf is None or rows > cap or buf.shape[1] != hidden or buf.dtype != dtype:
        new_rows = max(rows, cap)
        buf = torch.empty((new_rows, hidden), dtype=dtype, device=device)
        _TBO_PERSIST_BUF[key] = buf
    return buf[:rows]


def dp_reduce_scatterv_async(
    output_local: torch.Tensor,
    global_tokens: torch.Tensor,
    sizes: List[int],
    event_key=("combine", 0),
) -> torch.cuda.Event:
    """Launch the variable-length reduce_scatterv (combine) on the shared DP TBO
    comm stream; re-record + return a PERSISTENT event (keyed by `event_key`).
    Matches the gatherv (SUM_LEN) path."""
    comm = get_dp_tbo_comm_stream()
    compute = torch.cuda.current_stream()
    ev = _tbo_event(event_key)
    with torch.cuda.stream(comm):
        comm.wait_stream(compute)
        get_parallel().tp_group.reduce_scatterv(
            global_tokens, output=output_local, sizes=sizes
        )
        ev.record(comm)
    return ev


def get_moe_cp_group() -> GroupCoordinator:
    """Returns the MOE_DP group, which includes CP partners when attn_cp_size > moe_dp_size."""
    return get_parallel().moe_dp_group
