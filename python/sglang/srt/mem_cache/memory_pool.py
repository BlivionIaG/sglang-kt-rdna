"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

"""
Memory pool.

SGLang has two levels of memory pool.
ReqToTokenPool maps a request to its token locations.
TokenToKVPoolAllocator manages the indices to kv cache data.
KVCache actually holds the physical kv cache.
"""

import abc
import dataclasses
import logging
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import numpy as np
import torch
import triton
import triton.language as tl

from sglang.jit_kernel.kvcache import can_use_store_cache, store_cache
from sglang.srt.configs.mamba_utils import BaseLinearStateParams
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.layers.attention.nsa import index_buf_accessor
from sglang.srt.layers.attention.nsa.quant_k_cache import (
    quantize_k_cache,
    quantize_k_cache_separate,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.utils import (
    get_mla_kv_buffer_triton,
    maybe_init_custom_mem_pool,
    set_mla_kv_buffer_triton,
    set_mla_kv_scale_buffer_triton,
)
from sglang.srt.utils import (
    cpu_has_amx_support,
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
    next_power_of_2,
)
from sglang.srt.utils.custom_op import register_custom_op
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

store_cache = register_custom_op(store_cache, mutates_args=["k_cache", "v_cache"])

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import LayerDoneCounter
    from sglang.srt.managers.schedule_batch import Req


logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024
_is_cuda = is_cuda()
_is_npu = is_npu()
_is_cpu = is_cpu()
_cpu_has_amx_support = cpu_has_amx_support()
_is_hip = is_hip()


def get_tensor_size_bytes(t: Union[torch.Tensor, List[torch.Tensor]]):
    if isinstance(t, list):
        return sum(get_tensor_size_bytes(x) for x in t)
    return np.prod(t.shape) * t.dtype.itemsize


def _set_kv_buffer_impl(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    row_dim: int,  # head_num * head_dim
    store_dtype: torch.dtype,
    device_module: Any,
    alt_stream: Optional[torch.cuda.Stream] = None,
    same_kv_dim: bool = True,
) -> None:
    row_bytes = row_dim * store_dtype.itemsize
    if (_is_cuda or _is_hip) and same_kv_dim and can_use_store_cache(row_bytes):
        return store_cache(
            k.view(-1, row_dim),
            v.view(-1, row_dim),
            k_cache.view(-1, row_dim),
            v_cache.view(-1, row_dim),
            indices,
            row_bytes=row_bytes,
        )

    from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

    if get_is_capture_mode() and alt_stream is not None:
        current_stream = device_module.current_stream()
        alt_stream.wait_stream(current_stream)
        k_cache[indices] = k
        with device_module.stream(alt_stream):
            v_cache[indices] = v
        current_stream.wait_stream(alt_stream)
    else:  # fallback to naive implementation
        k_cache[indices] = k
        v_cache[indices] = v


class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
    ):
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.size = size
        self.max_context_len = max_context_len
        self.device = device
        with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.req_to_token = torch.zeros(
                (size, max_context_len), dtype=torch.int32, device=device
            )
        self.free_slots = list(range(size))

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, reqs: list[Req]) -> Optional[List[int]]:
        chunked = [i for i, r in enumerate(reqs) if r.req_pool_idx is not None]
        if not any(r.is_dllm() for r in reqs):
            assert (
                len(chunked) <= 1
            ), "only one chunked request may reuse req_pool_idx in a batch"
        assert all(
            reqs[i].is_chunked > 0 or reqs[i].kv_committed_len > 0 for i in chunked
        ), "request has req_pool_idx but is not chunked"

        need_size = len(reqs) - len(chunked)
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        offset = 0
        for r in reqs:
            if r.req_pool_idx is None:
                r.req_pool_idx = select_index[offset]
                offset += 1
        return [r.req_pool_idx for r in reqs]

    def free(self, req: Req):
        assert req.req_pool_idx is not None, "request must have req_pool_idx"
        self.free_slots.append(req.req_pool_idx)
        req.req_pool_idx = None

    def clear(self):
        self.free_slots = list(range(self.size))


class MambaPool:
    @dataclass(frozen=True, kw_only=True)
    class State:
        conv: List[torch.Tensor]
        temporal: torch.Tensor

        def at_layer_idx(self, layer: int):
            kwargs = {}
            for k, v in vars(self).items():
                if k == "conv" or k == "intermediate_conv_window":
                    kwargs[k] = [conv[layer] for conv in v]
                else:
                    kwargs[k] = v[layer]
            return type(self)(**kwargs)

        def mem_usage_bytes(self):
            return sum(
                get_tensor_size_bytes(getattr(self, f.name))
                for f in dataclasses.fields(self)
            )

    @dataclass(frozen=True, kw_only=True)
    class SpeculativeState(State):
        intermediate_ssm: torch.Tensor
        intermediate_conv_window: List[torch.Tensor]

    def __init__(
        self,
        *,
        size: int,
        spec_state_size: int,
        cache_params: BaseLinearStateParams,
        device: str,
        enable_memory_saver: bool = False,
        speculative_num_draft_tokens: Optional[int] = None,
    ):
        conv_state_shape = cache_params.shape.conv
        temporal_state_shape = cache_params.shape.temporal
        conv_dtype = cache_params.dtype.conv
        ssm_dtype = cache_params.dtype.temporal
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        num_mamba_layers = len(cache_params.layers)

        self.size = size
        self.device = device

        # for disagg with nvlink
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE), (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.enable_custom_mem_pool
            else nullcontext()
        ):
            conv_state = [
                torch.zeros(
                    size=(num_mamba_layers, size + 1) + conv_shape,
                    dtype=conv_dtype,
                    device=device,
                )
                for conv_shape in conv_state_shape
            ]

            if _is_cpu and _cpu_has_amx_support:
                from sglang.srt.layers.amx_utils import _init_amx_conv_state

                # CPU uses a different layout of conv_state for kernel optimization
                conv_state = _init_amx_conv_state(conv_state)

            temporal_state = torch.zeros(
                size=(num_mamba_layers, size + 1) + temporal_state_shape,
                dtype=ssm_dtype,
                device=device,
            )
            if speculative_num_draft_tokens is not None:
                # Cache intermediate SSM states per draft token during target verify
                # Shape: [num_layers, size + 1, speculative_num_draft_tokens, HV, K, V]
                intermediate_ssm_state_cache = torch.zeros(
                    size=(
                        num_mamba_layers,
                        spec_state_size + 1,
                        speculative_num_draft_tokens,
                        temporal_state_shape[0],
                        temporal_state_shape[1],
                        temporal_state_shape[2],
                    ),
                    dtype=ssm_dtype,
                    device="cuda",
                )
                # Cache intermediate conv windows (last K-1 inputs) per draft token during target verify
                # Shape: [num_layers, size + 1, speculative_num_draft_tokens, dim, K-1]
                intermediate_conv_window_cache = [
                    torch.zeros(
                        size=(
                            num_mamba_layers,
                            spec_state_size + 1,
                            speculative_num_draft_tokens,
                            conv_shape[0],
                            conv_shape[1],
                        ),
                        dtype=conv_dtype,
                        device="cuda",
                    )
                    for conv_shape in conv_state_shape
                ]
                self.mamba_cache = self.SpeculativeState(
                    conv=conv_state,
                    temporal=temporal_state,
                    intermediate_ssm=intermediate_ssm_state_cache,
                    intermediate_conv_window=intermediate_conv_window_cache,
                )
                logger.info(
                    f"Mamba Cache is allocated. "
                    f"max_mamba_cache_size: {size}, "
                    f"conv_state size: {get_tensor_size_bytes(conv_state) / GB:.2f}GB, "
                    f"ssm_state size: {get_tensor_size_bytes(temporal_state) / GB:.2f}GB "
                    f"intermediate_ssm_state_cache size: {get_tensor_size_bytes(intermediate_ssm_state_cache) / GB:.2f}GB "
                    f"intermediate_conv_window_cache size: {get_tensor_size_bytes(intermediate_conv_window_cache) / GB:.2f}GB "
                )
            else:
                self.mamba_cache = self.State(conv=conv_state, temporal=temporal_state)
                logger.info(
                    f"Mamba Cache is allocated. "
                    f"max_mamba_cache_size: {size}, "
                    f"conv_state size: {get_tensor_size_bytes(conv_state) / GB:.2f}GB, "
                    f"ssm_state size: {get_tensor_size_bytes(temporal_state) / GB:.2f}GB "
                )
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            self.free_slots = torch.arange(
                1, self.size + 1, dtype=torch.int64, device=self.device
            )
            self.mem_usage = self.mamba_cache.mem_usage_bytes() / GB
            self.num_mamba_layers = num_mamba_layers

    def get_speculative_mamba2_params_all_layers(self) -> SpeculativeState:
        assert isinstance(self.mamba_cache, self.SpeculativeState)
        return self.mamba_cache

    def mamba2_layer_cache(self, layer_id: int):
        return self.mamba_cache.at_layer_idx(layer_id)

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        # clear at alloc time, fill allocated slots with zeros
        for i in range(len(self.mamba_cache.conv)):
            self.mamba_cache.conv[i][:, select_index] = 0
        self.mamba_cache.temporal[:, select_index] = 0

        return select_index

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        self.free_slots = torch.cat((self.free_slots, free_index))

    def clear(self):
        self.free_slots = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )

    def copy_from(self, src_index: torch.Tensor, dst_index: torch.Tensor):
        for i in range(len(self.mamba_cache.conv)):
            self.mamba_cache.conv[i][:, dst_index] = self.mamba_cache.conv[i][
                :, src_index
            ]
        self.mamba_cache.temporal[:, dst_index] = self.mamba_cache.temporal[
            :, src_index
        ]
        return

    def fork_from(self, src_index: torch.Tensor) -> Optional[torch.Tensor]:
        dst_index = self.alloc(1)
        if dst_index == None:
            return None
        self.copy_from(src_index, dst_index)
        return dst_index

    def get_contiguous_buf_infos(self):
        """
        Get buffer info for RDMA registration.
        Only returns conv and temporal state buffers, excluding intermediate buffers
        used for speculative decoding (intermediate_ssm, intermediate_conv_window).
        """
        state_tensors = []
        for field in vars(self.mamba_cache):
            # Skip intermediate buffers used only for speculative decoding
            # These buffers have different size (spec_state_size + 1) and should not be transferred
            if field in ("intermediate_ssm", "intermediate_conv_window"):
                continue
            value = getattr(self.mamba_cache, field)
            if isinstance(value, list):
                state_tensors.extend(value)
            else:
                state_tensors.append(value)
        data_ptrs, data_lens, item_lens = [], [], []

        for _, state_tensor in enumerate(state_tensors):
            data_ptrs += [
                state_tensor[i].data_ptr() for i in range(self.num_mamba_layers)
            ]
            data_lens += [state_tensor[i].nbytes for i in range(self.num_mamba_layers)]
            item_lens += [
                state_tensor[i][0].nbytes for i in range(self.num_mamba_layers)
            ]
        return data_ptrs, data_lens, item_lens

    def get_state_dim_per_tensor(self):
        """Get the sliceable dimension size for each state tensor.

        For mamba state, the layout is:
        - conv_state: [num_layers, size+1, conv_dim/tp, conv_kernel-1]
        - temporal_state: [num_layers, size+1, num_heads/tp, head_dim, state_size]

        The 3rd dimension (index 2) is the one that gets sliced by TP.
        Returns the size of this dimension for each tensor (repeated for each layer).
        """
        state_tensors = []
        for field in vars(self.mamba_cache):
            value = getattr(self.mamba_cache, field)
            if isinstance(value, list):
                state_tensors.extend(value)
            else:
                state_tensors.append(value)

        dim_per_tensor = []
        for state_tensor in state_tensors:
            # state_tensor shape: [num_layers, size+1, sliceable_dim, ...]
            # The sliceable dimension is at index 2 (after num_layers and size)
            sliceable_dim = state_tensor.shape[2]
            # Repeat for each layer since we have per-layer data_ptrs
            dim_per_tensor += [sliceable_dim] * self.num_mamba_layers
        return dim_per_tensor


class HybridReqToTokenPool(ReqToTokenPool):
    """A memory pool that maps a request to its token locations."""

    def __init__(
        self,
        *,
        size: int,
        mamba_size: int,
        mamba_spec_state_size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        cache_params: BaseLinearStateParams,
        enable_mamba_extra_buffer: bool,
        speculative_num_draft_tokens: int = None,
    ):
        super().__init__(
            size=size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
        )
        self.mamba_ping_pong_track_buffer_size = (
            2 if speculative_num_draft_tokens is None else 1
        )
        self.enable_mamba_extra_buffer = enable_mamba_extra_buffer
        self.enable_memory_saver = enable_memory_saver
        self._init_mamba_pool(
            size=mamba_size,
            mamba_spec_state_size=mamba_spec_state_size,
            cache_params=cache_params,
            device=device,
            enable_mamba_extra_buffer=enable_mamba_extra_buffer,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )

    def _init_mamba_pool(
        self,
        size: int,
        mamba_spec_state_size: int,
        cache_params: BaseLinearStateParams,
        device: str,
        enable_mamba_extra_buffer: bool,
        speculative_num_draft_tokens: int = None,
    ):
        self.mamba_pool = MambaPool(
            size=size,
            spec_state_size=mamba_spec_state_size,
            cache_params=cache_params,
            device=device,
            enable_memory_saver=self.enable_memory_saver,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )
        self.mamba_map = {layer_id: i for i, layer_id in enumerate(cache_params.layers)}

        self.device = device
        self.req_index_to_mamba_index_mapping: torch.Tensor = torch.zeros(
            size, dtype=torch.int32, device=self.device
        )
        if enable_mamba_extra_buffer:
            self.req_index_to_mamba_ping_pong_track_buffer_mapping: torch.Tensor = (
                torch.zeros(
                    (size, self.mamba_ping_pong_track_buffer_size),
                    dtype=torch.int32,
                    device=self.device,
                )
            )

    # For chunk prefill req, we do not need to allocate mamba cache,
    # We could use allocated mamba cache instead.
    def alloc(self, reqs: List["Req"]) -> Optional[List[int]]:
        select_index = super().alloc(reqs)
        if select_index is None:
            return None

        mamba_index = []
        mamba_ping_pong_track_buffer_list = []
        for req in reqs:
            mid = None
            if req.mamba_pool_idx is not None:  # for radix cache
                mid = req.mamba_pool_idx
            else:
                mid = self.mamba_pool.alloc(1)
                assert (
                    mid is not None
                ), f"Not enough space for mamba cache, try to increase --mamba-full-memory-ratio or --max-mamba-cache-size. {mid=}, {self.mamba_pool.size=}, {self.mamba_pool.available_size()=}, {len(reqs)=}"
                mid = mid[0]
                req.mamba_pool_idx = mid
            mamba_index.append(mid)
            if self.enable_mamba_extra_buffer:
                if req.mamba_ping_pong_track_buffer is None:
                    req.mamba_ping_pong_track_buffer = self.mamba_pool.alloc(
                        self.mamba_ping_pong_track_buffer_size
                    )
                    assert (
                        req.mamba_ping_pong_track_buffer is not None
                    ), "Not enough space for mamba ping pong idx, try to increase --mamba-full-memory-ratio."
                    req.mamba_next_track_idx = 0
                mamba_ping_pong_track_buffer_list.append(
                    req.mamba_ping_pong_track_buffer.tolist()
                )
        assert len(select_index) == len(
            mamba_index
        ), f"Not enough space for mamba cache, try to increase --mamba-full-memory-ratio or --max-mamba-cache-size."
        if self.enable_mamba_extra_buffer:
            assert len(select_index) == len(
                mamba_ping_pong_track_buffer_list
            ), f"Not enough space for mamba ping pong idx, try to increase --mamba-full-memory-ratio."
        self.req_index_to_mamba_index_mapping[select_index] = torch.tensor(
            mamba_index, dtype=torch.int32, device=self.device
        )
        if self.enable_mamba_extra_buffer:
            self.req_index_to_mamba_ping_pong_track_buffer_mapping[select_index] = (
                torch.tensor(
                    mamba_ping_pong_track_buffer_list,
                    dtype=torch.int32,
                    device=self.device,
                )
            )
        return select_index

    def get_mamba_indices(self, req_indices: torch.Tensor) -> torch.Tensor:
        return self.req_index_to_mamba_index_mapping[req_indices]

    def mamba2_layer_cache(self, layer_id: int):
        assert layer_id in self.mamba_map
        return self.mamba_pool.mamba2_layer_cache(self.mamba_map[layer_id])

    def get_speculative_mamba2_params_all_layers(self) -> MambaPool.SpeculativeState:
        return self.mamba_pool.get_speculative_mamba2_params_all_layers()

    def get_mamba_ping_pong_other_idx(self, mamba_next_track_idx: int) -> int:
        if self.mamba_ping_pong_track_buffer_size == 2:
            return 1 - mamba_next_track_idx
        else:
            return mamba_next_track_idx

    def free_mamba_cache(
        self, req: "Req", mamba_ping_pong_track_buffer_to_keep: Optional[int] = None
    ):
        mamba_index = req.mamba_pool_idx
        assert mamba_index is not None, "double free? mamba_index is None"
        self.mamba_pool.free(mamba_index.unsqueeze(0))
        req.mamba_pool_idx = None

        if self.enable_mamba_extra_buffer:
            mamba_ping_pong_track_buffer_to_free = (
                self.req_index_to_mamba_ping_pong_track_buffer_mapping[req.req_pool_idx]
            )
            if mamba_ping_pong_track_buffer_to_keep is not None:
                assert mamba_ping_pong_track_buffer_to_keep in [
                    0,
                    1,
                ], f"mamba_ping_pong_track_buffer_to_keep must be 0 or 1, {mamba_ping_pong_track_buffer_to_keep=}"
                idx_to_free = list(range(self.mamba_ping_pong_track_buffer_size))
                idx_to_free.remove(mamba_ping_pong_track_buffer_to_keep)
                mamba_ping_pong_track_buffer_to_free = (
                    mamba_ping_pong_track_buffer_to_free[idx_to_free]
                )
            self.mamba_pool.free(mamba_ping_pong_track_buffer_to_free)

    def clear(self):
        logger.info("Reset HybridReqToTokenPool")
        super().clear()
        self.mamba_pool.clear()
        self.req_index_to_mamba_index_mapping.zero_()
        if self.enable_mamba_extra_buffer:
            self.req_index_to_mamba_ping_pong_track_buffer_mapping.zero_()


class KVCache(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.layer_num = layer_num
        self.start_layer = start_layer or 0
        self.end_layer = end_layer or layer_num - 1
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        self.mem_usage = 0

        # used for chunked cpu-offloading
        self.cpu_offloading_chunk_size = 8192

        # default state for optional layer-wise transfer control
        self.layer_transfer_counter = None

        # for disagg with nvlink
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )

    def _finalize_allocation_log(self, num_tokens: int):
        """Common logging and mem_usage computation for KV cache allocation.
        Supports both tuple (K, V) size returns and single KV size returns.
        """
        kv_size_bytes = self.get_kv_size_bytes()
        if isinstance(kv_size_bytes, tuple):
            k_size, v_size = kv_size_bytes
            k_size_GB = k_size / GB
            v_size_GB = v_size / GB
            logger.info(
                f"KV Cache is allocated. #tokens: {num_tokens}, K size: {k_size_GB:.2f} GB, V size: {v_size_GB:.2f} GB"
            )
            self.mem_usage = k_size_GB + v_size_GB
        else:
            kv_size_GB = kv_size_bytes / GB
            logger.info(
                f"KV Cache is allocated. #tokens: {num_tokens}, KV size: {kv_size_GB:.2f} GB"
            )
            self.mem_usage = kv_size_GB

    @abc.abstractmethod
    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    @abc.abstractmethod
    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        raise NotImplementedError()

    def register_layer_transfer_counter(self, layer_transfer_counter: LayerDoneCounter):
        self.layer_transfer_counter = layer_transfer_counter

    def get_cpu_copy(self, indices):
        raise NotImplementedError()

    def load_cpu_copy(self, kv_cache_cpu, indices):
        raise NotImplementedError()

    def maybe_get_custom_mem_pool(self):
        return self.custom_mem_pool


class MHATokenToKVPool(KVCache):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        v_head_dim: Optional[int] = None,
        swa_head_num: Optional[int] = None,
        swa_head_dim: Optional[int] = None,
        swa_v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
        enable_kv_cache_copy: bool = False,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.head_num = swa_head_num if swa_head_num is not None else head_num
        self.head_dim = swa_head_dim if swa_head_dim is not None else head_dim
        self.v_head_dim = (
            swa_v_head_dim
            if swa_v_head_dim is not None
            else v_head_dim if v_head_dim is not None else head_dim
        )

        self._create_buffers()

        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = (
            self.device_module.Stream() if _is_cuda and enable_alt_stream else None
        )

        if enable_kv_cache_copy:
            self._init_kv_copy_and_warmup()
        else:
            self._kv_copy_config = None

        self._finalize_allocation_log(size)

        # for store_cache JIT kernel
        self.row_dim = self.head_num * self.head_dim
        self.same_kv_dim = self.head_dim == self.v_head_dim

    def _init_kv_copy_and_warmup(self):
        # Heuristics for KV copy tiling
        _KV_COPY_STRIDE_THRESHOLD_LARGE = 8192
        _KV_COPY_STRIDE_THRESHOLD_MEDIUM = 4096
        _KV_COPY_TILE_SIZE_LARGE = 512
        _KV_COPY_TILE_SIZE_MEDIUM = 256
        _KV_COPY_TILE_SIZE_SMALL = 128
        _KV_COPY_NUM_WARPS_LARGE_TILE = 8
        _KV_COPY_NUM_WARPS_SMALL_TILE = 4

        stride_bytes = int(self.data_strides[0].item())
        if stride_bytes >= _KV_COPY_STRIDE_THRESHOLD_LARGE:
            bytes_per_tile = _KV_COPY_TILE_SIZE_LARGE
        elif stride_bytes >= _KV_COPY_STRIDE_THRESHOLD_MEDIUM:
            bytes_per_tile = _KV_COPY_TILE_SIZE_MEDIUM
        else:
            bytes_per_tile = _KV_COPY_TILE_SIZE_SMALL

        # Calculate num_locs_upper to avoid large Triton specialization (e.g. 8192)
        chunk_upper = 128 if bytes_per_tile >= _KV_COPY_TILE_SIZE_LARGE else 256

        self._kv_copy_config = {
            "bytes_per_tile": bytes_per_tile,
            "byte_tiles": (stride_bytes + bytes_per_tile - 1) // bytes_per_tile,
            "num_warps": (
                _KV_COPY_NUM_WARPS_SMALL_TILE
                if bytes_per_tile <= _KV_COPY_TILE_SIZE_MEDIUM
                else _KV_COPY_NUM_WARPS_LARGE_TILE
            ),
            "num_locs_upper": chunk_upper,
        }

        dummy_loc = torch.zeros(chunk_upper, dtype=torch.int64, device=self.device)
        grid = (self.data_ptrs.numel(), self._kv_copy_config["byte_tiles"])

        copy_all_layer_kv_cache_tiled[grid](
            self.data_ptrs,
            self.data_strides,
            dummy_loc,
            dummy_loc,
            1,
            chunk_upper,
            BYTES_PER_TILE=self._kv_copy_config["bytes_per_tile"],
            num_warps=self._kv_copy_config["num_warps"],
            num_stages=2,
        )

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                # [size, head_num, head_dim] for each layer
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                self.k_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, self.head_num, self.head_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, self.head_num, self.v_head_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
        self.data_strides = torch.tensor(
            [
                np.prod(x.shape[1:]) * x.dtype.itemsize
                for x in self.k_buffer + self.v_buffer
            ],
            device=self.device,
        )

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        k_size_bytes = 0
        for k_cache in self.k_buffer:
            k_size_bytes += get_tensor_size_bytes(k_cache)
        v_size_bytes = 0
        for v_cache in self.v_buffer:
            v_size_bytes += get_tensor_size_bytes(v_cache)
        return k_size_bytes, v_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        # layer_num x [seq_len, head_num, head_dim]
        # layer_num x [page_num, page_size, head_num, head_dim]
        kv_data_ptrs = [
            self._get_key_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_data_lens = [
            self._get_key_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_item_lens = [
            self._get_key_buffer(i)[0].nbytes * self.page_size
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i)[0].nbytes * self.page_size
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_cpu_copy(self, indices):
        torch.cuda.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu = self.k_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                v_cpu = self.v_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append([k_cpu, v_cpu])
        torch.cuda.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices):
        torch.cuda.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu, v_cpu = (
                    kv_cache_cpu[layer_id][i // chunk_size][0],
                    kv_cache_cpu[layer_id][i // chunk_size][1],
                )
                assert k_cpu.shape[0] == v_cpu.shape[0] == len(chunk_indices)
                k_chunk = k_cpu.to(self.k_buffer[0].device, non_blocking=True)
                v_chunk = v_cpu.to(self.v_buffer[0].device, non_blocking=True)
                self.k_buffer[layer_id][chunk_indices] = k_chunk
                self.v_buffer[layer_id][chunk_indices] = v_chunk
        torch.cuda.synchronize()

    def _get_key_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.k_buffer[layer_id - self.start_layer]

    def get_key_buffer(self, layer_id: int):
        # note: get_key_buffer is hooked with synchronization for layer-wise KV cache loading
        # it is supposed to be used only by attention backend not for information purpose
        # same applies to get_value_buffer and get_kv_buffer
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_key_buffer(layer_id)

    def _get_value_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.v_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        _set_kv_buffer_impl(
            cache_k,
            cache_v,
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
            loc,
            row_dim=self.row_dim,
            store_dtype=self.store_dtype,
            device_module=self.device_module,
            alt_stream=self.alt_stream,
            same_kv_dim=self.same_kv_dim,
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        if envs.SGLANG_NATIVE_MOVE_KV_CACHE.get():
            move_kv_cache_native(self.k_buffer, self.v_buffer, tgt_loc, src_loc)
            return

        N = tgt_loc.numel()
        if N == 0:
            return

        assert (
            self._kv_copy_config is not None
        ), "KV copy not initialized. Set enable_kv_cache_copy=True in __init__"

        cfg = self._kv_copy_config
        cap = int(cfg.get("num_locs_upper", 256))
        grid = (self.data_ptrs.numel(), cfg["byte_tiles"])

        if N <= cap:
            upper = next_power_of_2(N)
            copy_all_layer_kv_cache_tiled[grid](
                self.data_ptrs,
                self.data_strides,
                tgt_loc,
                src_loc,
                N,
                upper,
                BYTES_PER_TILE=cfg["bytes_per_tile"],
                num_warps=cfg["num_warps"],
                num_stages=2,
            )
            return

        # Huge N: chunk, but each chunk's upper is still pow2(<= cap)
        for start in range(0, N, cap):
            end = min(start + cap, N)
            chunk_len = end - start
            upper = next_power_of_2(chunk_len)
            copy_all_layer_kv_cache_tiled[grid](
                self.data_ptrs,
                self.data_strides,
                tgt_loc[start:end],
                src_loc[start:end],
                chunk_len,
                upper,
                BYTES_PER_TILE=cfg["bytes_per_tile"],
                num_warps=cfg["num_warps"],
                num_stages=2,
            )


class MHATokenToKVPoolFP4(MHATokenToKVPool):

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                # [size, head_num, head_dim] for each layer
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                m = self.size + self.page_size
                n = self.head_num
                k = self.head_dim

                scale_block_size = 16
                self.store_dtype = torch.uint8
                self.k_buffer = [
                    torch.zeros(
                        (m, n, k // 2),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(
                        (m, n, k // 2),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

                self.k_scale_buffer = [
                    torch.zeros(
                        (m, (n * k) // scale_block_size),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_scale_buffer = [
                    torch.zeros(
                        (m, (n * k) // scale_block_size),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer
        del self.k_scale_buffer
        del self.v_scale_buffer

    def _get_key_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            cache_k_nope_fp4 = self.k_buffer[layer_id - self.start_layer].view(
                torch.uint8
            )
            cache_k_nope_fp4_sf = self.k_scale_buffer[layer_id - self.start_layer]

            from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil

            cache_k_nope_fp4_dequant = KVFP4QuantizeUtil.batched_dequantize(
                cache_k_nope_fp4, cache_k_nope_fp4_sf
            )
            return cache_k_nope_fp4_dequant
        return self.k_buffer[layer_id - self.start_layer]

    def _get_value_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            cache_v_nope_fp4 = self.v_buffer[layer_id - self.start_layer].view(
                torch.uint8
            )
            cache_v_nope_fp4_sf = self.v_scale_buffer[layer_id - self.start_layer]

            from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil

            cache_v_nope_fp4_dequant = KVFP4QuantizeUtil.batched_dequantize(
                cache_v_nope_fp4, cache_v_nope_fp4_sf
            )
            return cache_v_nope_fp4_dequant
        return self.v_buffer[layer_id - self.start_layer]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)

            from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil

            cache_k, cache_k_fp4_sf = KVFP4QuantizeUtil.batched_quantize(cache_k)
            cache_v, cache_v_fp4_sf = KVFP4QuantizeUtil.batched_quantize(cache_v)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

            cache_k_fp4_sf = cache_k_fp4_sf.view(self.store_dtype)
            cache_v_fp4_sf = cache_v_fp4_sf.view(self.store_dtype)

        if get_is_capture_mode() and self.alt_stream is not None:
            # Overlap the copy of K and V cache for small batch size
            current_stream = self.device_module.current_stream()
            self.alt_stream.wait_stream(current_stream)
            self.k_buffer[layer_id - self.start_layer][loc] = cache_k

            self.k_scale_buffer[layer_id - self.start_layer][loc] = cache_k_fp4_sf
            with self.device_module.stream(self.alt_stream):
                self.v_buffer[layer_id - self.start_layer][loc] = cache_v

                self.v_scale_buffer[layer_id - self.start_layer][loc] = cache_v_fp4_sf
            current_stream.wait_stream(self.alt_stream)
        else:
            self.k_buffer[layer_id - self.start_layer][loc] = cache_k
            self.v_buffer[layer_id - self.start_layer][loc] = cache_v

            self.k_scale_buffer[layer_id - self.start_layer][loc] = cache_k_fp4_sf
            self.v_scale_buffer[layer_id - self.start_layer][loc] = cache_v_fp4_sf


class HybridLinearKVPool(KVCache):
    """KV cache with separate pools for full and linear attention layers."""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        page_size: int,
        head_num: int,
        head_dim: int,
        full_attention_layer_ids: List[int],
        enable_kvcache_transpose: bool,
        device: str,
        mamba_pool: MambaPool,
        enable_memory_saver: bool = False,
        # TODO: refactor mla related args
        use_mla: bool = False,
        kv_lora_rank: int = None,
        qk_rope_head_dim: int = None,
    ):
        self.size = size
        self.dtype = dtype
        self.device = device
        self.full_layer_nums = len(full_attention_layer_ids)
        self.page_size = page_size
        # TODO support pp?
        self.start_layer = 0
        self.head_num = head_num
        self.head_dim = head_dim
        self.mamba_pool = mamba_pool
        # TODO MHATransposedTokenToKVPool if enable_kvcache_transpose is True
        assert not enable_kvcache_transpose
        self.use_mla = use_mla
        if not use_mla:

            TokenToKVPoolClass = MHATokenToKVPool

            if _is_npu:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMHATokenToKVPool,
                )

                TokenToKVPoolClass = NPUMHATokenToKVPool

            self.full_kv_pool = TokenToKVPoolClass(
                size=size,
                page_size=self.page_size,
                dtype=dtype,
                head_num=head_num,
                head_dim=head_dim,
                layer_num=self.full_layer_nums,
                device=device,
                enable_memory_saver=enable_memory_saver,
            )
        else:

            TokenToKVPoolClass = MLATokenToKVPool

            if _is_npu:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMLATokenToKVPool,
                )

                TokenToKVPoolClass = NPUMLATokenToKVPool

            self.full_kv_pool = TokenToKVPoolClass(
                size=size,
                page_size=self.page_size,
                dtype=dtype,
                layer_num=self.full_layer_nums,
                device=device,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                enable_memory_saver=enable_memory_saver,
            )
        self.full_attention_layer_id_mapping = {
            id: i for i, id in enumerate(full_attention_layer_ids)
        }
        if use_mla:
            self.mem_usage = self.get_kv_size_bytes() / GB
        else:
            k_size, v_size = self.get_kv_size_bytes()
            self.mem_usage = (k_size + v_size) / GB

    def get_kv_size_bytes(self):
        return self.full_kv_pool.get_kv_size_bytes()

    def get_contiguous_buf_infos(self):
        return self.full_kv_pool.get_contiguous_buf_infos()

    def get_state_buf_infos(self):
        mamba_data_ptrs, mamba_data_lens, mamba_item_lens = (
            self.mamba_pool.get_contiguous_buf_infos()
        )
        return mamba_data_ptrs, mamba_data_lens, mamba_item_lens

    def get_state_dim_per_tensor(self):
        """Get the sliceable dimension size for each mamba state tensor."""
        return self.mamba_pool.get_state_dim_per_tensor()

    def maybe_get_custom_mem_pool(self):
        return self.full_kv_pool.maybe_get_custom_mem_pool()

    def _transfer_full_attention_id(self, layer_id: int):
        if layer_id not in self.full_attention_layer_id_mapping:
            raise ValueError(
                f"{layer_id=} not in full attention layers: {self.full_attention_layer_id_mapping.keys()}"
            )
        return self.full_attention_layer_id_mapping[layer_id]

    def get_key_buffer(self, layer_id: int):
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_key_buffer(layer_id)

    def get_value_buffer(self, layer_id: int):
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_kv_buffer(layer_id)

    @contextmanager
    def _transfer_id_context(self, layer: RadixAttention):

        @contextmanager
        def _patch_layer_id(layer):
            original_layer_id = layer.layer_id
            layer.layer_id = self._transfer_full_attention_id(layer.layer_id)
            try:
                yield
            finally:
                layer.layer_id = original_layer_id

        with _patch_layer_id(layer):
            yield

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ):
        layer_id = self._transfer_full_attention_id(layer.layer_id)
        if not self.use_mla:
            self.full_kv_pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=layer_id,
            )
        else:
            with self._transfer_id_context(layer):
                self.full_kv_pool.set_kv_buffer(
                    layer,
                    loc,
                    cache_k,
                    cache_v,
                )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        self.full_kv_pool.move_kv_cache(tgt_loc, src_loc)

    def get_v_head_dim(self):
        return self.full_kv_pool.get_value_buffer(0).shape[-1]

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        assert self.use_mla, "set_mla_kv_buffer called when use_mla is False"
        with self._transfer_id_context(layer):
            self.full_kv_pool.set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope)

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        assert self.use_mla, "get_mla_kv_buffer called when use_mla is False"
        with self._transfer_id_context(layer):
            return self.full_kv_pool.get_mla_kv_buffer(layer, loc, dst_dtype)


class MLATokenToKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        use_nsa: bool = False,
        override_kv_cache_dim: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )

        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.use_nsa = use_nsa
        self.nsa_kv_cache_store_fp8 = (
            use_nsa
            and dtype == torch.float8_e4m3fn
            and override_kv_cache_dim is not None
        )
        # When override_kv_cache_dim is provided with nsa model, we assume the
        # override kv cache dim is correct and use it directly.
        self.kv_cache_dim = (
            override_kv_cache_dim
            if self.nsa_kv_cache_store_fp8
            else (kv_lora_rank + qk_rope_head_dim)
        )

        self._create_buffers()

        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        if not use_nsa:
            # NSA will allocate indexer KV cache later and then log the total size
            self._finalize_allocation_log(size)

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                self.kv_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, 1, self.kv_cache_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def _clear_buffers(self):
        del self.kv_buffer

    def get_kv_size_bytes(self):
        assert hasattr(self, "kv_buffer")
        kv_size_bytes = 0
        for kv_cache in self.kv_buffer:
            kv_size_bytes += get_tensor_size_bytes(kv_cache)
        return kv_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        # MLA has only one kv_buffer, so only the information of this buffer needs to be returned.
        kv_data_ptrs = [self.kv_buffer[i].data_ptr() for i in range(self.layer_num)]
        kv_data_lens = [self.kv_buffer[i].nbytes for i in range(self.layer_num)]
        kv_item_lens = [
            self.kv_buffer[i][0].nbytes * self.page_size for i in range(self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id - self.start_layer].view(self.dtype)

        return self.kv_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id - self.start_layer][
                ..., : self.kv_lora_rank
            ].view(self.dtype)
        return self.kv_buffer[layer_id - self.start_layer][..., : self.kv_lora_rank]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        layer_id = layer.layer_id
        assert not self.nsa_kv_cache_store_fp8
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)

        if self.store_dtype != self.dtype:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k.view(
                self.store_dtype
            )
        else:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        layer_id = layer.layer_id

        if self.nsa_kv_cache_store_fp8:
            # OPTIMIZATION: Quantize k_nope and k_rope separately to avoid concat overhead
            # This also enables reuse of set_mla_kv_buffer_triton two-tensor write path
            # quantize_k_cache_separate returns (nope_part, rope_part) as uint8 bytes
            cache_k_nope_fp8, cache_k_rope_fp8 = quantize_k_cache_separate(
                cache_k_nope, cache_k_rope
            )

            # Reuse existing two-tensor write kernel (works with FP8 byte layout)
            # cache_k_nope_fp8: (num_tokens, 1, 528) uint8 [nope_fp8(512) | scales(16)]
            # cache_k_rope_fp8: (num_tokens, 1, 128) uint8 [rope_bf16_bytes(128)]
            set_mla_kv_buffer_triton(
                self.kv_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope_fp8,
                cache_k_rope_fp8,
            )
        else:
            if cache_k_nope.dtype != self.dtype:
                cache_k_nope = cache_k_nope.to(self.dtype)
                cache_k_rope = cache_k_rope.to(self.dtype)
            if self.store_dtype != self.dtype:
                cache_k_nope = cache_k_nope.view(self.store_dtype)
                cache_k_rope = cache_k_rope.view(self.store_dtype)

            set_mla_kv_buffer_triton(
                self.kv_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope,
                cache_k_rope,
            )

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        # get k nope and k rope from the kv buffer, and optionally cast them to dst_dtype.
        layer_id = layer.layer_id
        kv_buffer = self.get_key_buffer(layer_id)
        dst_dtype = dst_dtype or self.dtype
        cache_k_nope = torch.empty(
            (loc.shape[0], 1, self.kv_lora_rank),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        cache_k_rope = torch.empty(
            (loc.shape[0], 1, self.qk_rope_head_dim),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        get_mla_kv_buffer_triton(kv_buffer, loc, cache_k_nope, cache_k_rope)
        return cache_k_nope, cache_k_rope

    def get_cpu_copy(self, indices):
        torch.cuda.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                kv_cpu = self.kv_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append(kv_cpu)
        torch.cuda.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices):
        torch.cuda.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                kv_cpu = kv_cache_cpu[layer_id][i // chunk_size]
                assert kv_cpu.shape[0] == len(chunk_indices)
                kv_chunk = kv_cpu.to(self.kv_buffer[0].device, non_blocking=True)
                self.kv_buffer[layer_id][chunk_indices] = kv_chunk
        torch.cuda.synchronize()


class MLATokenToKVPoolFP4(MLATokenToKVPool):

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                m = self.size + self.page_size
                n = 1  # head_num
                k = self.kv_cache_dim  # head_dim

                scale_block_size = 16
                self.store_dtype = torch.uint8

                self.kv_buffer = [
                    torch.zeros(
                        (m, n, k // 2),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

                self.kv_scale_buffer = [
                    torch.zeros(
                        (m, k // scale_block_size),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def _clear_buffers(self):
        del self.kv_buffer
        del self.kv_scale_buffer

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            cache_k_nope_fp4 = self.kv_buffer[layer_id - self.start_layer].view(
                torch.uint8
            )
            cache_k_nope_fp4_sf = self.kv_scale_buffer[layer_id - self.start_layer]

            from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil

            cache_k_nope_fp4_dequant = KVFP4QuantizeUtil.batched_dequantize(
                cache_k_nope_fp4, cache_k_nope_fp4_sf
            )
            return cache_k_nope_fp4_dequant

        return self.kv_buffer[layer_id - self.start_layer]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        layer_id = layer.layer_id
        assert not self.nsa_kv_cache_store_fp8
        if cache_k.dtype != self.dtype:
            from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil

            cache_k_fp4, cache_k_fp4_sf = KVFP4QuantizeUtil.batched_quantize(cache_k)

        if self.store_dtype != self.dtype:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k_fp4.view(
                self.store_dtype
            )
            self.kv_scale_buffer[layer_id - self.start_layer][loc] = (
                cache_k_fp4_sf.view(self.store_dtype)
            )
        else:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        layer_id = layer.layer_id

        if self.nsa_kv_cache_store_fp8:
            # original cache_k: (num_tokens, num_heads 1, hidden 576); we unsqueeze the page_size=1 dim here
            # TODO no need to cat
            cache_k = torch.cat([cache_k_nope, cache_k_rope], dim=-1)
            cache_k = quantize_k_cache(cache_k.unsqueeze(1)).squeeze(1)
            cache_k = cache_k.view(self.store_dtype)
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k
        else:
            if cache_k_nope.dtype != self.dtype:
                from sglang.srt.layers.quantization.kvfp4_tensor import (
                    KVFP4QuantizeUtil,
                )

                cache_k_nope_fp4, cache_k_nope_fp4_sf = (
                    KVFP4QuantizeUtil.batched_quantize(cache_k_nope)
                )
                cache_k_rope_fp4, cache_k_rope_fp4_sf = (
                    KVFP4QuantizeUtil.batched_quantize(cache_k_rope)
                )

            if self.store_dtype != self.dtype:
                cache_k_nope = cache_k_nope.view(self.store_dtype)
                cache_k_rope = cache_k_rope.view(self.store_dtype)

            set_mla_kv_buffer_triton(
                self.kv_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope_fp4,
                cache_k_rope_fp4,
            )
            set_mla_kv_scale_buffer_triton(
                self.kv_scale_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope_fp4_sf,
                cache_k_rope_fp4_sf,
            )


class NSATokenToKVPool(MLATokenToKVPool):
    quant_block_size = 128
    index_k_with_scale_buffer_dtype = torch.uint8
    rope_storage_dtype = torch.bfloat16  # rope is always stored in bf16

    def __init__(
        self,
        size: int,
        page_size: int,
        kv_lora_rank: int,
        dtype: torch.dtype,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        index_head_dim: int,
        enable_memory_saver: bool,
        kv_cache_dim: int,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        index_cache_dtype: Optional[torch.dtype] = None,
    ):

        override_dim = (
            kv_cache_dim if kv_cache_dim != kv_lora_rank + qk_rope_head_dim else None
        )

        super().__init__(
            size,
            page_size,
            dtype,
            kv_lora_rank,
            qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
            use_nsa=True,
            override_kv_cache_dim=override_dim,
        )
        # self.index_k_dtype = torch.float8_e4m3fn
        # self.index_k_scale_dtype = torch.float32
        self.index_head_dim = index_head_dim
        self.index_cache_dtype = index_cache_dtype or torch.float8_e4m3fn
        if self.index_cache_dtype not in (torch.float8_e4m3fn, torch.bfloat16):
            raise ValueError(
                "NSA index cache supports only FP8 E4M3 or BF16, got "
                f"{self.index_cache_dtype}"
            )
        self.index_cache_is_bf16 = self.index_cache_dtype == torch.bfloat16
        # num head == 1 and head dim == 128 for index_k in NSA
        assert index_head_dim == 128

        if _is_hip:
            assert self.page_size == 1
        else:
            assert self.page_size == 64
        with (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.custom_mem_pool
            else nullcontext()
        ):
            self.index_k_with_scale_buffer = [
                torch.zeros(
                    # Layout:
                    #     ref: test_attention.py :: kv_cache_cast_to_fp8
                    #     shape: (num_pages, page_size 64 * head_dim 128 + page_size 64 * fp32_nbytes 4)
                    #     data: for page i,
                    #         * buf[i, :page_size * head_dim] for fp8 data
                    #         * buf[i, page_size * head_dim:].view(float32) for scale
                    (
                        (size + page_size + 1) // self.page_size,
                        self.page_size
                        * (
                            index_head_dim
                            if self.index_cache_is_bf16
                            else index_head_dim
                            + index_head_dim // self.quant_block_size * 4
                        ),
                    ),
                    dtype=(
                        torch.bfloat16
                        if self.index_cache_is_bf16
                        else self.index_k_with_scale_buffer_dtype
                    ),
                    device=device,
                )
                for _ in range(layer_num)
            ]
        self._finalize_allocation_log(size)

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.index_k_with_scale_buffer[layer_id - self.start_layer]

    def get_index_k_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        if self.index_cache_is_bf16:
            num_pages = (seq_len + self.page_size - 1) // self.page_size
            pages = buf.index_select(0, page_indices[:num_pages].to(torch.int64))
            return pages.view(-1, self.index_head_dim)[:seq_len]
        return index_buf_accessor.GetK.execute(
            self, buf, seq_len=seq_len, page_indices=page_indices
        )

    def get_index_k_scale_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        if self.index_cache_is_bf16:
            return torch.ones((seq_len,), dtype=torch.float32, device=buf.device)
        return index_buf_accessor.GetS.execute(
            self, buf, seq_len=seq_len, page_indices=page_indices
        )

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        """
        Fused method to get both index K and scale data in a single call using Triton.
        More efficient than calling get_index_k_continuous and get_index_k_scale_continuous separately.

        :param layer_id: Layer index
        :param seq_len: Sequence length
        :param page_indices: Page indices tensor
        :return: tuple of (k_fp8, k_scale) where
                 k_fp8: (seq_len, index_head_dim), uint8
                 k_scale: (seq_len, 4), uint8
        """
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        if self.index_cache_is_bf16:
            return (
                self.get_index_k_continuous(layer_id, seq_len, page_indices),
                torch.ones((seq_len,), dtype=torch.float32, device=buf.device),
            )
        return index_buf_accessor.GetKAndS.execute(
            self, buf, seq_len=seq_len, page_indices=page_indices
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        if self.index_cache_is_bf16:
            if index_k.shape != (loc.numel(), self.index_head_dim):
                raise ValueError(
                    "BF16 NSA index K must have shape "
                    f"{(loc.numel(), self.index_head_dim)}, got {tuple(index_k.shape)}"
                )
            pages = torch.div(loc, self.page_size, rounding_mode="floor").to(torch.long)
            offsets = torch.remainder(loc, self.page_size).to(torch.long)
            buf.view(-1, self.page_size, self.index_head_dim)[pages, offsets] = (
                index_k.to(torch.bfloat16)
            )
            return
        index_buf_accessor.SetKAndS.execute(
            pool=self, buf=buf, loc=loc, index_k=index_k, index_k_scale=index_k_scale
        )

    def get_state_buf_infos(self):
        data_ptrs = [
            self.index_k_with_scale_buffer[i].data_ptr() for i in range(self.layer_num)
        ]
        data_lens = [
            self.index_k_with_scale_buffer[i].nbytes for i in range(self.layer_num)
        ]
        item_lens = [
            self.index_k_with_scale_buffer[i][0].nbytes for i in range(self.layer_num)
        ]
        return data_ptrs, data_lens, item_lens

    def get_kv_size_bytes(self):
        kv_size_bytes = super().get_kv_size_bytes()
        for index_k_cache in self.index_k_with_scale_buffer:
            kv_size_bytes += get_tensor_size_bytes(index_k_cache)
        return kv_size_bytes


class DoubleSparseTokenToKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        heavy_channel_num: int,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                # [size, head_num, head_dim] for each layer
                self.k_buffer = [
                    torch.zeros(
                        (size + page_size, head_num, head_dim),
                        dtype=dtype,
                        device=device,
                    )
                    for _ in range(layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(
                        (size + page_size, head_num, head_dim),
                        dtype=dtype,
                        device=device,
                    )
                    for _ in range(layer_num)
                ]

                # [size, head_num, heavy_channel_num] for each layer
                self.label_buffer = [
                    torch.zeros(
                        (size + 1, head_num, heavy_channel_num),
                        dtype=dtype,
                        device=device,
                    )
                    for _ in range(layer_num)
                ]

    def get_key_buffer(self, layer_id: int):
        return self.k_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        return self.v_buffer[layer_id - self.start_layer]

    def get_label_buffer(self, layer_id: int):
        return self.label_buffer[layer_id - self.start_layer]

    def get_kv_buffer(self, layer_id: int):
        return (
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
        )

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        cache_label: torch.Tensor,
    ):
        # NOTE(Andy): ignore the dtype check
        layer_id = layer.layer_id
        self.k_buffer[layer_id - self.start_layer][loc] = cache_k
        self.v_buffer[layer_id - self.start_layer][loc] = cache_v
        self.label_buffer[layer_id - self.start_layer][loc] = cache_label


def move_kv_cache_native(
    k_buffer: List[torch.Tensor],
    v_buffer: List[torch.Tensor],
    tgt_loc: torch.Tensor,
    src_loc: torch.Tensor,
):
    if tgt_loc.numel() == 0:
        return

    tgt_loc_flat = tgt_loc.view(-1).long()
    src_loc_flat = src_loc.view(-1).long()
    for k_cache, v_cache in zip(k_buffer, v_buffer):
        k_cache[tgt_loc_flat] = k_cache[src_loc_flat]
        v_cache[tgt_loc_flat] = v_cache[src_loc_flat]


@triton.jit
def copy_all_layer_kv_cache_tiled(
    data_ptrs,
    strides,
    tgt_loc_ptr,
    src_loc_ptr,
    num_locs,
    num_locs_upper: tl.constexpr,
    BYTES_PER_TILE: tl.constexpr,
):
    """2D tiled kernel. Safe for in-place copy."""
    bid = tl.program_id(0)
    tid = tl.program_id(1)

    stride = tl.load(strides + bid)
    base_ptr = tl.load(data_ptrs + bid)
    base_ptr = tl.cast(base_ptr, tl.pointer_type(tl.uint8))

    byte_off = tid * BYTES_PER_TILE + tl.arange(0, BYTES_PER_TILE)
    mask_byte = byte_off < stride
    tl.multiple_of(byte_off, 16)

    loc_idx = tl.arange(0, num_locs_upper)
    mask_loc = loc_idx < num_locs

    src = tl.load(src_loc_ptr + loc_idx, mask=mask_loc, other=0)
    tgt = tl.load(tgt_loc_ptr + loc_idx, mask=mask_loc, other=0)

    src_ptr = base_ptr + src[:, None] * stride + byte_off[None, :]
    tgt_ptr = base_ptr + tgt[:, None] * stride + byte_off[None, :]

    mask = mask_loc[:, None] & mask_byte[None, :]
    vals = tl.load(src_ptr, mask=mask)
    tl.store(tgt_ptr, vals, mask=mask)


class MHATokenToKOnlyPool(KVCache):
    """K-only variant of MHATokenToKVPool.

    Used by MiniMax sparse layers whose index branch never reads V (the
    ``sparse_disable_index_value`` flag), so allocating V would just waste
    memory. Exposes the ``k_buffer`` list at the same shape as the K side of
    MHATokenToKVPool, plus ``get_key_buffer`` and accounting hooks.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.head_num = head_num
        self.head_dim = head_dim
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                self.k_buffer = [
                    torch.zeros(
                        (size + page_size, head_num, head_dim),
                        dtype=self.store_dtype,
                        device=device,
                    )
                    for _ in range(layer_num)
                ]
        self._finalize_allocation_log(size)

    def _get_key_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.k_buffer[layer_id - self.start_layer]

    def register_layer_transfer_counter(
        self, layer_transfer_counter: LayerDoneCounter
    ) -> None:
        self.layer_transfer_counter = layer_transfer_counter

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_key_buffer(layer_id)

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError("MHATokenToKOnlyPool does not allocate V")

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("MHATokenToKOnlyPool does not allocate V")

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ) -> None:
        # Routed through MiniMaxSparseKVPool.set_index_k_buffer instead.
        raise NotImplementedError(
            "MHATokenToKOnlyPool: use set_index_k_buffer on the parent "
            "MiniMaxSparseKVPool — this pool does not store V"
        )

    def get_kv_size_bytes(self):
        k_size_bytes = sum(get_tensor_size_bytes(k) for k in self.k_buffer)
        return k_size_bytes, 0


class MiniMaxSparseKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        idx_head_dim: int,
        dense_layer_ids: List[int],
        sparse_layer_ids: List[int],
        device: str,
        disable_value_sparse_layer_ids: Optional[List[int]] = None,
        enable_memory_saver: bool = False,
        index_dtype: Optional[torch.dtype] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        # Do not call super().__init__() — delegate to sub-pools instead.
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device

        local_dense_layer_ids = [
            lid for lid in dense_layer_ids if start_layer <= lid < end_layer
        ]
        local_sparse_layer_ids = [
            lid for lid in sparse_layer_ids if start_layer <= lid < end_layer
        ]

        index_dtype = index_dtype if index_dtype is not None else dtype

        # Split sparse layers by V allocation policy:
        #   * kv_sparse: ``index_kv_pool`` allocates both K and V
        #   * k_only_sparse: ``index_k_pool`` allocates only K (V is never read)
        disable_set = set(disable_value_sparse_layer_ids or [])
        local_kv_sparse_layer_ids = [
            g for g in local_sparse_layer_ids if g not in disable_set
        ]
        local_k_only_sparse_layer_ids = [
            g for g in local_sparse_layer_ids if g in disable_set
        ]

        # Membership check across all sparse layers, regardless of split.
        self.sparse_layer_id_mapping: dict[int, int] = {
            gid: i for i, gid in enumerate(local_sparse_layer_ids)
        }
        # Per-sub-pool local indices.
        self.index_kv_layer_id_mapping: dict[int, int] = {
            gid: i for i, gid in enumerate(local_kv_sparse_layer_ids)
        }
        self.index_k_layer_id_mapping: dict[int, int] = {
            gid: i for i, gid in enumerate(local_k_only_sparse_layer_ids)
        }

        self.main_pool = MHATokenToKVPool(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=len(local_dense_layer_ids) + len(local_sparse_layer_ids),
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
        )

        self.index_kv_pool: Optional[MHATokenToKVPool] = (
            MHATokenToKVPool(
                size=size,
                page_size=page_size,
                dtype=index_dtype,
                head_num=1,
                head_dim=idx_head_dim,
                layer_num=len(local_kv_sparse_layer_ids),
                device=device,
                enable_memory_saver=enable_memory_saver,
            )
            if local_kv_sparse_layer_ids
            else None
        )

        self.index_k_pool: Optional[MHATokenToKOnlyPool] = (
            MHATokenToKOnlyPool(
                size=size,
                page_size=page_size,
                dtype=index_dtype,
                head_num=1,
                head_dim=idx_head_dim,
                layer_num=len(local_k_only_sparse_layer_ids),
                device=device,
                enable_memory_saver=enable_memory_saver,
            )
            if local_k_only_sparse_layer_ids
            else None
        )

        self.mem_usage = self.main_pool.mem_usage
        if self.index_kv_pool is not None:
            self.mem_usage += self.index_kv_pool.mem_usage
        if self.index_k_pool is not None:
            self.mem_usage += self.index_k_pool.mem_usage

        # HiCacheController reads these from the top-level KV pool wrapper.
        self.layer_num = self.main_pool.layer_num
        self.start_layer = self.main_pool.start_layer
        self.end_layer = self.main_pool.end_layer
        # PD disaggregation reads these directly (no fallback) off the wrapper.
        self.head_num = self.main_pool.head_num
        self.head_dim = self.main_pool.head_dim
        self.layer_transfer_counter = None

    def register_layer_transfer_counter(
        self, layer_transfer_counter: LayerDoneCounter
    ) -> None:
        self.layer_transfer_counter = layer_transfer_counter

    def _wait_for_layer(self, layer_id: int) -> None:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        self._wait_for_layer(layer_id)
        return self.main_pool.get_key_buffer(layer_id)

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        self._wait_for_layer(layer_id)
        return self.main_pool.get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._wait_for_layer(layer_id)
        return self.main_pool.get_kv_buffer(layer_id)

    def get_index_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._wait_for_layer(layer_id)
        mapped_id = self.index_kv_layer_id_mapping.get(layer_id)
        if mapped_id is None:
            raise ValueError(
                f"layer_id={layer_id} does not have an index V cache "
                f"(either dense, or in the K-only group). "
                f"index_kv layers: {list(self.index_kv_layer_id_mapping.keys())}"
            )
        return self.index_kv_pool.get_kv_buffer(mapped_id)

    def get_index_k_buffer(self, layer_id: int) -> torch.Tensor:
        self._wait_for_layer(layer_id)
        # First try the K-only pool; fall back to the index_kv pool's K side
        # so callers that just need K work for both sparse subgroups.
        mapped_id = self.index_k_layer_id_mapping.get(layer_id)
        if mapped_id is not None:
            return self.index_k_pool.get_key_buffer(mapped_id)
        mapped_id = self.index_kv_layer_id_mapping.get(layer_id)
        if mapped_id is not None:
            return self.index_kv_pool.get_key_buffer(mapped_id)
        raise ValueError(
            f"layer_id={layer_id} is not a sparse attention layer; "
            f"sparse layers: {list(self.sparse_layer_id_mapping.keys())}"
        )

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ) -> None:
        """Write main K/V at `loc`. Works for any layer (dense or sparse)."""
        self.main_pool.set_kv_buffer(
            layer,
            loc,
            cache_k,
            cache_v,
            k_scale,
            v_scale,
        )

    def set_index_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_idx_k: torch.Tensor,
        cache_idx_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ) -> None:
        mapped_id = self.index_kv_layer_id_mapping.get(layer.layer_id)
        if mapped_id is None:
            raise ValueError(
                f"layer.layer_id={layer.layer_id} does not have an index V "
                f"cache (either dense, or in the K-only group). "
                f"index_kv layers: {list(self.index_kv_layer_id_mapping.keys())}"
            )
        self.index_kv_pool.set_kv_buffer(
            layer,
            loc,
            cache_idx_k,
            cache_idx_v,
            k_scale,
            v_scale,
            layer_id_override=mapped_id,
        )

    def set_index_k_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_idx_k: torch.Tensor,
    ) -> None:
        mapped_id = self.index_k_layer_id_mapping.get(layer.layer_id)
        if mapped_id is None:
            raise ValueError(
                f"layer.layer_id={layer.layer_id} is not in the K-only "
                f"sparse group. K-only layers: "
                f"{list(self.index_k_layer_id_mapping.keys())}"
            )
        sub_pool = self.index_k_pool
        if cache_idx_k.dtype != sub_pool.dtype:
            cache_idx_k = cache_idx_k.to(sub_pool.dtype)
        if sub_pool.store_dtype != sub_pool.dtype:
            cache_idx_k = cache_idx_k.view(sub_pool.store_dtype)
        sub_pool.k_buffer[mapped_id][loc] = cache_idx_k

    def _can_fuse_kv_index_store(
        self,
        index_pool: MHATokenToKVPool,
        cache_k: torch.Tensor,
        cache_idx_k: torch.Tensor,
    ) -> bool:
        """Fast-path precondition: CUDA, no per-store quantization, and a single
        head byte size shared by the main and index caches (see
        ``set_fused_kv_index_buffer``)."""
        main = self.main_pool
        return (
            envs.SGLANG_OPT_USE_MINIMAX_FUSED_KV_INDEX_STORE.get()
            and _is_cuda
            # No dtype conversion / fp8 scaling on either side (the fused kernel
            # is a raw byte copy, it does not quantize).
            and main.store_dtype == main.dtype
            and index_pool.store_dtype == index_pool.dtype
            and cache_k.dtype == main.dtype
            and cache_idx_k.dtype == index_pool.dtype
            # Uniform head byte size collapses head_dim + dtype into one constant.
            and main.dtype == index_pool.dtype
            and main.head_dim == index_pool.head_dim
            # 128-bit vector copy requires a 16-byte-aligned head size.
            and (main.head_dim * main.dtype.itemsize) % 16 == 0
        )

    def set_fused_kv_index_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        cache_idx_k: torch.Tensor,
        cache_idx_v: Optional[torch.Tensor],
    ) -> None:
        """Store main K/V + index K (+ optional index V) for a sparse layer.

        When enabled and applicable, writes all caches in one fused JIT launch;
        otherwise falls back to the separate ``set_kv_buffer`` /
        ``set_index_k_buffer`` / ``set_index_kv_buffer`` calls. ``cache_idx_v``
        is None for value-disabled (block-selector) layers.
        """
        disable_value = cache_idx_v is None
        index_pool = self.index_k_pool if disable_value else self.index_kv_pool

        if index_pool is not None and self._can_fuse_kv_index_store(
            index_pool, cache_k, cache_idx_k
        ):
            from sglang.jit_kernel.minimax_store_kv_index import store_kv_index

            main = self.main_pool
            head_bytes = main.head_dim * main.dtype.itemsize
            if disable_value:
                idx_k_cache = self.get_index_k_buffer(layer.layer_id).flatten(1)
                idx_v_cache = None
            else:
                ik, iv = self.get_index_kv_buffer(layer.layer_id)
                idx_k_cache, idx_v_cache = ik.flatten(1), iv.flatten(1)
            store_kv_index(
                cache_k.flatten(1),
                cache_v.flatten(1),
                main.get_key_buffer(layer.layer_id).flatten(1),
                main.get_value_buffer(layer.layer_id).flatten(1),
                cache_idx_k.flatten(1),
                idx_k_cache,
                None if disable_value else cache_idx_v.flatten(1),
                idx_v_cache,
                loc,
                num_kv_heads=main.head_num,
                head_bytes=head_bytes,
            )
            return

        # Fallback: separate stores (identical semantics).
        self.set_kv_buffer(layer, loc, cache_k, cache_v)
        if disable_value:
            self.set_index_k_buffer(layer, loc, cache_idx_k)
        else:
            self.set_index_kv_buffer(layer, loc, cache_idx_k, cache_idx_v)

    def get_kv_size_bytes(self):
        sub_pools = [self.main_pool, self.index_kv_pool, self.index_k_pool]
        sizes = [p.get_kv_size_bytes() for p in sub_pools if p is not None]
        return sum(k for k, _ in sizes), sum(v for _, v in sizes)

    def get_contiguous_buf_infos(self):
        # Main K/V only; index buffers ride the state-buffer channel.
        return self.main_pool.get_contiguous_buf_infos()

    def get_index_k_state_buf_infos(self):
        # Per-page item_len (MHATokenToKVPool convention); index rows share the
        # main-KV `loc`, so the transfer reuses the same page-ids.
        pool = self.index_k_pool
        n = pool.layer_num
        data_ptrs = [pool.k_buffer[i].data_ptr() for i in range(n)]
        data_lens = [pool.k_buffer[i].nbytes for i in range(n)]
        item_lens = [pool.k_buffer[i][0].nbytes * pool.page_size for i in range(n)]
        return data_ptrs, data_lens, item_lens

    def maybe_get_custom_mem_pool(self):
        return self.main_pool.maybe_get_custom_mem_pool()

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        # TODO: enable speculative decoding by passing
        # `enable_kv_cache_copy=True` to BOTH sub-pools at construction time
        # (so each pool allocates its move-kernel workspace) and then making
        # this delegate to `main_pool.move_kv_cache` + `index_pool.move_kv_cache`.
        # Currently the sub-pools are constructed without that flag, so calling
        # the delegated move would fail; raise explicitly to surface the gap
        # the moment any spec-decode path tries to use this pool.
        raise NotImplementedError(
            "move_kv_cache is not yet supported for MiniMaxSparseKVPool: "
            "sub-pools must be built with enable_kv_cache_copy=True first."
        )

    def get_v_head_dim(self):
        return self.main_pool.get_value_buffer(0).shape[-1]

# --- imported with the qwen4 subsystem (sgl-project/sglang) -----------------
# `layers/cp/zigzag.py` imports this at module level and calls the two
# classmethods when it writes reorganized KV rows. Upstream's version carries
# `physical` / `swa_loc` / `full_loc` because upstream's pool is the "unified"
# one with sub-pools and a write door that requires a physical mark; this fork's
# pool predates that, and its `ForwardBatch` has no
# `out_cache_loc_is_physical`, so those extras are omitted here rather than
# guessed. What remains is the part the CP path actually uses: the write loc for
# a batch or for one layer, plus the per-token slice alignment.


@dataclass
class KVWriteLoc:
    """Write target for ``KVCache.set_kv_buffer``.

    ``loc`` is the generic per-token write location (``out_cache_loc``);
    ``swa_loc`` / ``full_loc`` are optional pre-resolved locations into a
    sliding-window or full-attention sub-pool for hybrid pools, ``None``
    otherwise. Bundling them lets a backend issue one ``set_kv_buffer`` call
    regardless of pool type.
    """

    loc: torch.Tensor
    swa_loc: Optional[torch.Tensor] = None
    full_loc: Optional[torch.Tensor] = None

    @classmethod
    def for_batch(
        cls,
        forward_batch,
        *,
        swa_loc: Optional[torch.Tensor] = None,
        full_loc: Optional[torch.Tensor] = None,
    ) -> "KVWriteLoc":
        """The batch's ``out_cache_loc`` as a write loc."""
        return cls(forward_batch.out_cache_loc, swa_loc, full_loc)

    @classmethod
    def for_layer(
        cls,
        forward_batch,
        layer,
        *,
        swa_loc: Optional[torch.Tensor] = None,
        full_loc: Optional[torch.Tensor] = None,
    ) -> "KVWriteLoc":
        """``layer``'s write loc: the batch's, or for a cross-attention layer
        ``encoder_out_cache_loc``."""
        if getattr(layer, "is_cross_attention", False):
            return cls(forward_batch.encoder_out_cache_loc, swa_loc, full_loc)
        return cls.for_batch(forward_batch, swa_loc=swa_loc, full_loc=full_loc)

    def __post_init__(self):
        # swa_loc / full_loc are resolved once at metadata-init from the full
        # (padded) out_cache_loc; padded paths later narrow loc per layer, so
        # slice these pre-resolved locs to match (same per-token order).
        if self.swa_loc is not None and self.swa_loc.shape[0] != self.loc.shape[0]:
            self.swa_loc = self.swa_loc[: self.loc.shape[0]]
        if self.full_loc is not None and self.full_loc.shape[0] != self.loc.shape[0]:
            self.full_loc = self.full_loc[: self.loc.shape[0]]

    def as_loc(self) -> torch.Tensor:
        """This fork's ``set_kv_buffer`` takes the loc tensor directly."""
        return self.loc


# --- imported with the qwen4 subsystem (sgl-project/sglang), which renamed
# NSATokenToKVPool to DSATokenToKVPool. Both names are bound.
class DSATokenToKVPool(MLATokenToKVPool):
    quant_block_size = 128
    index_k_with_scale_buffer_dtype = torch.uint8
    rope_storage_dtype = torch.bfloat16  # rope is always stored in bf16

    def __init__(
        self,
        size: int,
        page_size: int,
        kv_lora_rank: int,
        dtype: torch.dtype,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        index_head_dim: int,
        enable_memory_saver: bool,
        kv_cache_dim: int,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        index_buf_size: Optional[int] = None,
        index_kpool: int = 1,
        index_kpool_compress: bool = False,
        tail_extra_slots: int = 0,
        max_running_requests: Optional[int] = None,
        skip_topk_layers: Optional[List[bool]] = None,
    ):
        override_dim = (
            kv_cache_dim if kv_cache_dim != kv_lora_rank + qk_rope_head_dim else None
        )

        super().__init__(
            size,
            page_size,
            dtype,
            kv_lora_rank,
            qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
            use_dsa=True,
            override_kv_cache_dim=override_dim,
        )
        # self.index_k_dtype = torch.float8_e4m3fn
        # self.index_k_scale_dtype = torch.float32
        self.index_head_dim = index_head_dim
        self.index_kpool = index_kpool
        self.index_kpool_compress = index_kpool_compress
        self.tail_extra_slots = tail_extra_slots
        assert self.page_size % index_kpool == 0, (
            f"page_size {self.page_size} must be a multiple of index_kpool {index_kpool}"
        )
        self.index_page_size = self.page_size // index_kpool
        if index_buf_size is None:
            index_buf_size = size
        self.index_buf_size = index_buf_size
        # num head == 1 and head dim == 128 for index_k in DSA
        assert index_head_dim == 128

        self.skip_topk_layers = (
            list(skip_topk_layers)
            if skip_topk_layers is not None
            else [False] * layer_num
        )
        assert len(self.skip_topk_layers) == layer_num

        physical_page_size = self.page_size // index_kpool
        if _is_hip:
            if aiter_can_use_preshuffle_paged_mqa():
                assert physical_page_size % 16 == 0, (
                    f"HIP preshuffle requires page_size to be a multiple of 16, got {physical_page_size}"
                )
            else:
                assert physical_page_size == 1, (
                    f"HIP legacy DSA path requires page_size == 1, got {physical_page_size}"
                )
        elif is_xpu():
            assert physical_page_size in (
                64,
                128,
            ), f"XPU DSA requires page_size 64 or 128, got {physical_page_size}"
        else:
            assert physical_page_size == 64, (
                f"DSA requires 64-token physical pages, got page_size={self.page_size} "
                f"with index_kpool={index_kpool}"
            )
        self.index_key_cache = self._create_index_key_cache()
        self._init_kpool_compress_tail_buffers(
            index_kpool=index_kpool,
            index_kpool_compress=index_kpool_compress,
            tail_extra_slots=tail_extra_slots,
            index_head_dim=index_head_dim,
            layer_num=layer_num,
            device=device,
            max_running_requests=max_running_requests,
        )
        self._finalize_allocation_log(size)

    def _create_index_key_cache(self) -> IndexKeyCache:
        return IndexKeyCache(self, self.index_buf_size)

    def _should_allocate_index_layer(self, local_layer_idx: int) -> bool:
        return not self.skip_topk_layers[local_layer_idx]

    def host_pool_decls(self):
        # pool_host imports this module. Resolve the mirror side lazily.
        from sglang.srt.mem_cache.pool_host.dsa import make_dsa_indexer_pool_decl

        kv_decls = super().host_pool_decls()
        # Shared-topk layers own a 0-row placeholder, so a non-empty buffer list
        # is not enough: some layer must actually hold index keys.
        if not self.index_k_with_scale_buffer or all(self.skip_topk_layers):
            return kv_decls
        return (*kv_decls, make_dsa_indexer_pool_decl(self))

    @property
    def index_k_with_scale_buffer(self):
        # Preserve direct HiCache access while storage lives behind the facade.
        return self.index_key_cache.buffer

    def _init_kpool_compress_tail_buffers(
        self,
        index_kpool: int,
        index_kpool_compress: bool,
        tail_extra_slots: int,
        index_head_dim: int,
        layer_num: int,
        device: str,
        max_running_requests: Optional[int],
    ) -> None:
        """Keep request tails on the pool so they follow the index-cache lifecycle."""
        self.kpool_use_compress = index_kpool > 1 and index_kpool_compress

        if not self.kpool_use_compress:
            self._compress_tail_k = None
            self._compress_tail_score = None
            return

        assert max_running_requests is not None, (
            "DSATokenToKVPool with kpool compress requires max_running_requests"
        )
        # +1 mirrors req_to_token_pool.size + 1 used by the indexer to
        # provide an extra slot for invalid / sentinel req indices.
        req_pool_size = max_running_requests + 1
        tail_dtype = torch.bfloat16
        tail_width = index_kpool + tail_extra_slots
        with (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.custom_mem_pool
            else nullcontext()
        ):
            self._compress_tail_k: Optional[List[torch.Tensor]] = [
                torch.zeros(
                    req_pool_size if self._should_allocate_index_layer(i) else 0,
                    tail_width,
                    index_head_dim,
                    dtype=tail_dtype,
                    device=device,
                )
                for i in range(layer_num)
            ]
            self._compress_tail_score: Optional[List[torch.Tensor]] = [
                torch.zeros(
                    req_pool_size if self._should_allocate_index_layer(i) else 0,
                    tail_width,
                    index_head_dim,
                    dtype=tail_dtype,
                    device=device,
                )
                for i in range(layer_num)
            ]

    def get_compress_tail_buffers(
        self, layer_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.kpool_use_compress, (
            "get_compress_tail_buffers called when kpool compress is disabled"
        )
        idx = layer_id - self.start_layer
        return (
            self._compress_tail_k[idx],
            self._compress_tail_score[idx],
        )

    def get_compress_tail_buf_infos(self):
        if not self.kpool_use_compress:
            return [], [], []
        transfer_layer_ids = list(range(self.layer_num))
        # Keep zero-row indexShare entries in the pointer list so layer offsets
        # stay aligned across PD peers; item_len=0 makes transfer backends skip them.
        tail_buffers = [self._compress_tail_k[i] for i in transfer_layer_ids] + [
            self._compress_tail_score[i] for i in transfer_layer_ids
        ]
        data_ptrs = [buf.data_ptr() for buf in tail_buffers]
        data_lens = [buf.nbytes for buf in tail_buffers]
        item_lens = [buf[0].nbytes if buf.shape[0] > 0 else 0 for buf in tail_buffers]
        return data_ptrs, data_lens, item_lens

    def kpool_decode_update_index_cache(
        self,
        layer_id: int,
        key: torch.Tensor,
        slot_score: torch.Tensor,
        ape: torch.Tensor,
        block_tables: torch.Tensor,
        req_pool_indices: torch.Tensor,
        positions: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        round_scale: bool = False,
    ) -> None:
        from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
            kpool_decode_update_and_maybe_write_cache,
        )

        assert self.kpool_use_compress, (
            "kpool_decode_update_index_cache called when kpool compress is disabled"
        )
        idx = layer_id - self.start_layer
        buf = self.get_index_k_with_scale_buffer(layer_id)
        kpool_decode_update_and_maybe_write_cache(
            pool=self,
            buf=buf,
            tail_k=self._compress_tail_k[idx],
            tail_score=self._compress_tail_score[idx],
            key=key,
            slot_score=slot_score,
            ape=ape,
            block_tables=block_tables,
            req_pool_indices=req_pool_indices,
            positions=positions,
            seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            round_scale=round_scale,
        )

    def set_compress_tail_for_request(
        self,
        layer_id: int,
        req_pool_idx: torch.Tensor,
        key_tail: torch.Tensor,
        score_tail: torch.Tensor,
        n_remain: int,
        dst_logical_start: int,
    ) -> None:
        """Leave the ring untouched at a pool boundary; no tail carries over."""
        assert self.kpool_use_compress, (
            "set_compress_tail_for_request called when kpool compress is disabled"
        )
        idx = layer_id - self.start_layer
        if n_remain > 0:
            slots = (
                torch.arange(n_remain, device=key_tail.device, dtype=torch.long)
                + int(dst_logical_start)
            ) % self._compress_tail_k[idx].shape[1]
            self._compress_tail_k[idx][req_pool_idx, slots] = key_tail
            self._compress_tail_score[idx][req_pool_idx, slots] = score_tail

    def _clear_buffers(self):
        super()._clear_buffers()
        self.index_key_cache.clear()
        if hasattr(self, "_compress_tail_k") and self._compress_tail_k is not None:
            del self._compress_tail_k
            del self._compress_tail_score

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        """Move latent KV and the DSA indexer cache (key + scale) in lockstep."""
        super().move_kv_cache(tgt_loc, src_loc)
        self.index_key_cache.move(tgt_loc, src_loc)

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        return self.index_key_cache.get_local_buffer(layer_id)

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ):
        return self.index_key_cache.get_k_and_scale(
            layer_id, seq_len_tensor, page_indices, seq_len_sum, max_seq_len
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        self.index_key_cache.store_quantized(layer_id, loc, index_k, index_k_scale)

    def _get_compress_tail_cpu_copy(self, req_pool_index):
        if not self.kpool_use_compress or req_pool_index is None:
            return None

        tail_k_cpu = []
        tail_score_cpu = []
        for tail_k, tail_score in zip(self._compress_tail_k, self._compress_tail_score):
            if tail_k.shape[0] == 0:
                tail_k_cpu.append(None)
                tail_score_cpu.append(None)
                continue
            tail_k_cpu.append(tail_k[req_pool_index].to("cpu", non_blocking=True))
            tail_score_cpu.append(
                tail_score[req_pool_index].to("cpu", non_blocking=True)
            )
        return tail_k_cpu, tail_score_cpu

    def _load_compress_tail_cpu_copy(self, tail_k_cpu, tail_score_cpu, req_pool_index):
        if (
            not self.kpool_use_compress
            or req_pool_index is None
            or tail_k_cpu is None
            or tail_score_cpu is None
        ):
            return

        for tail_k, tail_score, saved_k, saved_score in zip(
            self._compress_tail_k,
            self._compress_tail_score,
            tail_k_cpu,
            tail_score_cpu,
        ):
            if tail_k.shape[0] == 0 or saved_k is None or saved_score is None:
                continue
            tail_k[req_pool_index] = saved_k.to(tail_k.device, non_blocking=True)
            tail_score[req_pool_index] = saved_score.to(
                tail_score.device, non_blocking=True
            )

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        # Retraction reuses index-cache pages; offload index/scale with KV so resume cannot read another request's entries.
        kv_cache_cpu = super().get_cpu_copy(indices, mamba_indices=mamba_indices)
        cpu_copy = {
            "kv": kv_cache_cpu,
            "index_k": self.index_key_cache.cpu_copy(indices),
        }
        compress_tail = self._get_compress_tail_cpu_copy(req_pool_index)
        if compress_tail is not None:
            cpu_copy["tail_k"], cpu_copy["tail_score"] = compress_tail
        torch.cuda.synchronize()
        return cpu_copy

    def load_cpu_copy(
        self,
        kv_cache_cpu_dict,
        indices,
        mamba_indices=None,
        req_pool_index=None,
    ):
        super().load_cpu_copy(
            kv_cache_cpu_dict["kv"],
            indices,
            mamba_indices=mamba_indices,
            req_pool_index=req_pool_index,
        )
        self.index_key_cache.load_cpu_copy(kv_cache_cpu_dict["index_k"], indices)
        self._load_compress_tail_cpu_copy(
            kv_cache_cpu_dict.get("tail_k"),
            kv_cache_cpu_dict.get("tail_score"),
            req_pool_index,
        )
        torch.cuda.synchronize()

    def get_state_buf_infos(self):
        return self.index_key_cache.state_buf_infos()

    def get_kv_size_bytes(self):
        kv_size_bytes = super().get_kv_size_bytes()
        for index_k_cache in self.index_k_with_scale_buffer:
            kv_size_bytes += get_tensor_size_bytes(index_k_cache)
        return kv_size_bytes


# This fork's name for the same pool; kept so existing references work.
NSATokenToKVPool = DSATokenToKVPool


# --- imported with the qwen4 subsystem (sgl-project/sglang) ---


def conv_window_dedup_enabled(
    is_npu: bool, is_cpu: bool, speculative_eagle_topk: Optional[int], is_kda: bool
) -> bool:
    """Whether the deduplicated sliding-window conv-intermediate layout is safe.

    It is safe for CUDA linear draft chains whose kernels consume the window raw.
    Tree verify, NPU/CPU, and KDA keep dense windows: tree ancestors need independent
    windows, platform kernels expect contiguous steps, and KDA transposes the window
    before conv so the overlapping ``as_strided`` layout would corrupt stores.
    """
    return (
        not is_npu
        and not is_cpu
        and not is_kda
        and (speculative_eagle_topk is None or speculative_eagle_topk <= 1)
    )


def _set_kv_buffer_prefix_valid_impl(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    loc_2d: torch.Tensor,
    commit_lens: torch.Tensor,
    row_dim: int,
    store_dtype: torch.dtype,
) -> None:
    if k.numel() == 0 or loc_2d.numel() == 0 or commit_lens.numel() == 0:
        return

    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()
    if not loc_2d.is_contiguous():
        loc_2d = loc_2d.contiguous()
    if not commit_lens.is_contiguous():
        commit_lens = commit_lens.contiguous()

    row_bytes = row_dim * store_dtype.itemsize
    if row_bytes <= 0:
        return

    if row_bytes >= 8192:
        bytes_per_tile = 512
        num_warps = 8
    elif row_bytes >= 4096:
        bytes_per_tile = 256
        num_warps = 4
    else:
        bytes_per_tile = 128
        num_warps = 4

    grid = (
        int(loc_2d.shape[0]),
        int(loc_2d.shape[1]),
        triton.cdiv(row_bytes, bytes_per_tile),
    )

    set_kv_buffer_prefix_valid_tiled[grid](
        k,
        v,
        k_cache,
        v_cache,
        loc_2d,
        commit_lens,
        int(k.stride(0) * k.element_size()),
        int(v.stride(0) * v.element_size()),
        int(k_cache.stride(0) * k_cache.element_size()),
        int(v_cache.stride(0) * v_cache.element_size()),
        int(loc_2d.shape[1]),
        ROW_BYTES=row_bytes,
        BYTES_PER_TILE=bytes_per_tile,
        num_warps=num_warps,
        num_stages=2,
    )


def _set_kv_buffer_prefix_valid_impl_fp8(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: float,
    v_scale: float,
    loc_2d: torch.Tensor,
    commit_lens: torch.Tensor,
    row_dim: int,
    k_scale_is_tensor: bool = False,
    v_scale_is_tensor: bool = False,
) -> None:
    if k.numel() == 0 or loc_2d.numel() == 0 or commit_lens.numel() == 0:
        return

    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()
    if not loc_2d.is_contiguous():
        loc_2d = loc_2d.contiguous()
    if not commit_lens.is_contiguous():
        commit_lens = commit_lens.contiguous()

    if row_dim <= 0:
        return

    if row_dim >= 4096:
        elems_per_tile = 256
        num_warps = 8
    elif row_dim >= 2048:
        elems_per_tile = 128
        num_warps = 4
    else:
        elems_per_tile = 64
        num_warps = 4
    grid = (
        int(loc_2d.shape[0]),
        int(loc_2d.shape[1]),
        triton.cdiv(row_dim, elems_per_tile),
    )
    set_kv_buffer_prefix_valid_tiled_fp8[grid](
        k,
        v,
        k_cache,
        v_cache,
        loc_2d,
        commit_lens,
        k_scale,
        v_scale,
        int(k.stride(0)),
        int(v.stride(0)),
        int(k_cache.stride(0)),
        int(v_cache.stride(0)),
        int(loc_2d.shape[1]),
        ROW_ELEMS=row_dim,
        ELEMS_PER_TILE=elems_per_tile,
        K_SCALE_IS_TENSOR=k_scale_is_tensor,
        V_SCALE_IS_TENSOR=v_scale_is_tensor,
        num_warps=num_warps,
        num_stages=2,
    )


def _resolve_fused_scale(
    scale,
    layer_scale,
    layer_scale_float,
) -> Optional[float]:
    if isinstance(scale, (float, int)):
        return float(scale)

    if (
        isinstance(scale, torch.Tensor)
        and scale.ndim == 0
        and scale.device.type == "cpu"
    ):
        return float(scale.item())

    if (
        isinstance(scale, torch.Tensor)
        and scale.ndim == 0
        and scale.is_cuda
        and scale is layer_scale
        and isinstance(layer_scale_float, (float, int))
    ):
        return float(layer_scale_float)

    return None


def unwrap_write_loc(loc_info):
    """Return ``(loc, swa_loc, full_loc)`` from a ``KVWriteLoc`` or a bare loc."""
    if isinstance(loc_info, KVWriteLoc):
        return loc_info.loc, loc_info.swa_loc, loc_info.full_loc
    return loc_info, None, None


def write_loc_is_physical(loc_info) -> bool:
    """Whether ``loc_info`` is a ``KVWriteLoc`` marked physical; a bare loc is not."""
    return isinstance(loc_info, KVWriteLoc) and loc_info.physical


class KvBufferDesc:
    """Byte-span math for one KV buffer laid out as rows of ``row_bytes`` holding
    ``tokens_per_row`` tokens each (a row = one token slot, or one whole page)."""

    __slots__ = ("name", "shape", "row_bytes", "tokens_per_row")

    def __init__(self, name: str, shape: tuple, *, row_bytes: int, tokens_per_row: int):
        self.name = name
        self.shape = tuple(shape)
        self.row_bytes = int(row_bytes)
        self.tokens_per_row = int(tokens_per_row)

    def _rows(self, num_tokens: int) -> int:
        n = max(int(num_tokens), 0)
        return (n + self.tokens_per_row - 1) // self.tokens_per_row

    def reserved_span_bytes(self, itemsize: int) -> int:
        """Full upper-bound byte size of the buffer (its whole tensor)."""
        return math.prod(self.shape) * itemsize

    def prefix_span_bytes(self, num_tokens: int, page_size: int) -> int:
        """Bytes to back to make the first ``num_tokens`` tokens usable."""
        return self._rows(num_tokens) * self.row_bytes

    def final_span_bytes(self, num_tokens: int, page_size: int) -> int:
        """Bytes of the final advertised span (adds the padded page). CEIL, not floor:
        an unaligned count must still cover its partial last page (e.g. n=17, page=16
        -> 3 pages, not 2)."""
        return self._rows(max(int(num_tokens), 0) + page_size) * self.row_bytes

    def item_len_bytes(self, page_size: int) -> int:
        """Per-page transfer chunk (one page's worth of this buffer)."""
        return (page_size // self.tokens_per_row) * self.row_bytes


class NoOpMHATokenToKVPool(MHATokenToKVPool):
    """KV cache pool that skips physical K/V buffer allocation.

    Used in embedding-mode prefill-only workloads with the FA
    fa_skip_kv_cache path, where no layer reads or writes KV cache because
    attention uses raw K/V via flash_attn_varlen_func. Other prefill-only paths
    such as scoring/MIS may benefit from the same idea later, but some still
    stage K/V through paged cache today.

    This class keeps the scheduler's view of pool capacity (self.size is
    honored for admission) but allocates only (page_size, head_num, head_dim)
    placeholder tensors per layer to satisfy any code paths that dereference
    the buffers.

    Callers MUST ensure no real set_kv_buffer/get_*_buffer calls happen against
    this pool; those paths raise loudly so misuse is visible.
    """

    def _create_buffers(self):
        # No-op pool keeps tiny NHD placeholders regardless of SGLANG_USE_HND_KVCACHE
        # (no real KV is stored), so force NHD here to keep the store/move fast paths.
        self.use_hnd = False
        self.kv_cache_layout = "nhd"
        # Allocate minimal placeholder buffers. They exist purely so that code
        # paths holding `k_buffer` / `v_buffer` references (pointer tables,
        # layer-transfer counters, stride arithmetic) keep working without
        # None-guards scattered across the codebase. Shape is
        # [page_size, head_num, head_dim] per layer so that the unconditional
        # `key_cache.view(-1, page_size, head_num, head_dim)` in the FA backend
        # at the top of forward_extend succeeds regardless of --page-size.
        # Total footprint is still on the order of KB vs GBs for a real pool.
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.k_buffer = [
                torch.zeros(
                    (self.page_size, self.head_num, self.head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]
            self.v_buffer = [
                torch.zeros(
                    (self.page_size, self.head_num, self.v_head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]

        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
        self.data_strides = torch.tensor(
            [x.stride(0) * x.dtype.itemsize for x in self.k_buffer + self.v_buffer],
            device=self.device,
        )

    def _finalize_allocation_log(self, num_tokens: int):
        self.mem_usage = 0.0
        placeholder_bytes = (
            2
            * self.layer_num
            * self.page_size
            * self.head_num
            * max(self.head_dim, self.v_head_dim)
            * self.store_dtype.itemsize
        )
        logger.info(
            f"KV Cache skipped (no-op pool). Logical #tokens: {num_tokens}, "
            f"physical K/V size: ~{placeholder_bytes / 1024:.1f} KB placeholder"
        )

    def get_kv_size_bytes(self):
        # Report zero so downstream memory accounting matches reality.
        return (0, 0)

    def set_kv_buffer(self, *args, **kwargs):
        raise RuntimeError(
            "NoOpMHATokenToKVPool.set_kv_buffer was called. This pool is only "
            "valid in prefill-only modes (e.g. --is-embedding, scoring) with "
            "the FA backend's fa_skip_kv_cache path active; the attention "
            "backend must never write to it. Check that the workload truly "
            "performs no decode and that the FA backend's fa_skip_kv_cache "
            "preconditions are met."
        )

    def get_key_buffer(self, layer_id: int):
        # Return the placeholder. The FA backend reads this before taking the
        # fa_skip_kv_cache branch (which does not use it); the placeholder shape
        # is (page_size, head_num, head_dim) so downstream .view() calls succeed.
        return self.k_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        return self.v_buffer[layer_id - self.start_layer]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        # no-op; embedding mode has no KV cache to move
        return


class PageMajorMHATokenToKVPool(MHATokenToKVPool):
    """MHA pool with the page-major page-granularity envelope layout.

    NON-CONSTRUCTIBLE: the strided 4-D view builder and its write kernel are
    gone, and ServerArgs rejects the static page-major arm at boot. The class
    stays as the seat for the per-layer-view reimplementation.
    """

    def __init__(
        self,
        *args,
        kv_cache_layout: Optional[str] = None,
        enable_kv_cache_copy: bool = False,
        **kwargs,
    ):
        assert kv_cache_layout in (
            None,
            "page_major_layer_major",
        ), f"PageMajorMHATokenToKVPool fixes its layout; got {kv_cache_layout!r}"
        # The tiled copy kernel assumes stride == row bytes, which the strided 4-D
        # views violate, so the copy path is never available here regardless of
        # what the caller requested (the spec-decode call sites pass
        # enable_kv_cache_copy=True). Always fall back to the native move.
        super().__init__(
            *args,
            kv_cache_layout="page_major_layer_major",
            enable_kv_cache_copy=False,
            **kwargs,
        )

    def _create_buffers(self):
        raise NotImplementedError(
            "PageMajorMHATokenToKVPool: the strided 4-D envelope views were "
            "removed; the static-pool page-major layout is temporarily "
            "unsupported (ServerArgs rejects it at startup). "
            "--enable-unified-memory provides the page-major layout with "
            "per-layer views."
        )

    # The methods below assume the per-layer contiguous 3-D layout. The 4-D
    # strided envelope views have no per-layer contiguous region (their bytes are
    # interleaved layer-major within each page) and index page-major, not
    # token-major. Inheriting them would silently mis-index; fail loudly instead.

    def get_contiguous_buf_infos(self):
        raise NotImplementedError(
            "page-major layout has no per-layer contiguous regions; KV transfer / "
            "disaggregation is unsupported (TODO: expose the single _raw buffer "
            "with a page-aware transfer scheme)."
        )

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        raise NotImplementedError(
            "CPU offloading is unsupported under the page-major layout "
            "(TODO: split token ids into page/slot for the 4-D index)."
        )

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        raise NotImplementedError(
            "CPU offloading is unsupported under the page-major layout "
            "(TODO: split token ids into page/slot for the 4-D index)."
        )

    def set_kv_buffer_prefix_valid(self, *args, **kwargs):
        raise NotImplementedError(
            "prefix-valid commit is unsupported under the page-major layout "
            "(_set_kv_buffer_prefix_valid_impl assumes 3-D contiguous + row_dim)."
        )


class MHATokenToKVPoolMXFP8(MHATokenToKVPool):
    """MHA KV cache pool for MXFP8 block-scaled FP8.

    K/V data is stored as FP8 E4M3. Per-32-element UE8M0 scale factors are
    stored beside it and passed to the FA4 MXFP8 kernel.
    """

    MXFP8_SCALE_BLOCK_SIZE = 32

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                m = self.size + self.page_size
                n = self.head_num
                k = self.head_dim
                v = self.v_head_dim

                if k % self.MXFP8_SCALE_BLOCK_SIZE != 0:
                    raise ValueError(
                        f"MXFP8 KV cache requires head_dim divisible by "
                        f"{self.MXFP8_SCALE_BLOCK_SIZE}, got {k}."
                    )
                if v % self.MXFP8_SCALE_BLOCK_SIZE != 0:
                    raise ValueError(
                        f"MXFP8 KV cache requires v_head_dim divisible by "
                        f"{self.MXFP8_SCALE_BLOCK_SIZE}, got {v}."
                    )
                if not hasattr(torch, "float8_e8m0fnu"):
                    raise RuntimeError(
                        "MXFP8 KV cache requires torch.float8_e8m0fnu support."
                    )
                if self.use_hnd:
                    # Buffers are NHD; the inherited HND move_kv_cache branch
                    # would silently relocate wrong bytes.
                    raise ValueError(
                        "MXFP8 KV cache does not support SGLANG_USE_HND_KVCACHE."
                    )

                self.store_dtype = torch.float8_e4m3fn
                self.k_buffer = [
                    torch.zeros((m, n, k), dtype=self.store_dtype, device=self.device)
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros((m, n, v), dtype=self.store_dtype, device=self.device)
                    for _ in range(self.layer_num)
                ]

                # UE8M0 scales, one per 32-element block. For the production
                # page_size==128 path they are stored interleaved in the FA4
                # BlockScaledBasicChunk atom layout
                # (num_pages, head, 32, page_size//32, sf_dim) and written by
                # the store_sf_interleaved kernel; otherwise flat per slot. Must
                # be zero-initialized (garbage 0xFF is e8m0 NaN).
                k_sf_dim = k // self.MXFP8_SCALE_BLOCK_SIZE
                v_sf_dim = v // self.MXFP8_SCALE_BLOCK_SIZE
                self.mxfp8_sf_interleaved = self.page_size == 128
                if self.mxfp8_sf_interleaved:
                    assert m % self.page_size == 0
                    num_pages = m // self.page_size
                    chunk = self.page_size // self.MXFP8_SCALE_BLOCK_SIZE
                    k_sf_shape = (
                        num_pages,
                        n,
                        self.MXFP8_SCALE_BLOCK_SIZE,
                        chunk,
                        k_sf_dim,
                    )
                    v_sf_shape = (
                        num_pages,
                        n,
                        self.MXFP8_SCALE_BLOCK_SIZE,
                        chunk,
                        v_sf_dim,
                    )
                else:
                    k_sf_shape = (m, n, k_sf_dim)
                    v_sf_shape = (m, n, v_sf_dim)
                self.k_scale_buffer = [
                    torch.zeros(
                        k_sf_shape, dtype=torch.float8_e8m0fnu, device=self.device
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_scale_buffer = [
                    torch.zeros(
                        v_sf_shape, dtype=torch.float8_e8m0fnu, device=self.device
                    )
                    for _ in range(self.layer_num)
                ]

        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
        self.data_strides = torch.tensor(
            [x.stride(0) * x.dtype.itemsize for x in self.k_buffer + self.v_buffer],
            device=self.device,
        )
        # This override replaces the base allocation, so the PD-transfer
        # descriptors for the packed data buffers are built here too.
        self._kv_buffer_descs = self._build_kv_buffer_descs()

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer
        del self.k_scale_buffer
        del self.v_scale_buffer

    def _get_key_buffer(self, layer_id: int):
        return self.k_buffer[layer_id - self.start_layer]

    def _get_value_buffer(self, layer_id: int):
        return self.v_buffer[layer_id - self.start_layer]

    def get_kv_scale_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = layer_id - self.start_layer
        return self.k_scale_buffer[idx], self.v_scale_buffer[idx]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        layer_id_override: Optional[int] = None,
        dcp_kv_mask: Optional[torch.Tensor] = None,
    ):
        if dcp_kv_mask is not None:
            raise NotImplementedError("MXFP8 KV cache does not support DCP KV masks.")
        loc, _, _ = unwrap_write_loc(loc_info)
        maybe_detect_oob(
            loc, 0, self.size + self.page_size, "set_kv_buffer (MHA-MXFP8)"
        )
        layer_id = (
            layer_id_override if layer_id_override is not None else layer.layer_id
        )
        idx = layer_id - self.start_layer

        if k_scale is None or v_scale is None:
            # Fused path (SGLANG_OPT_INKLING_MXFP8_FUSED_QUANT_STORE): the layer
            # hands us bf16 K/V and one kernel quantizes + scatters the fp8
            # payload and the interleaved UE8M0 scales.
            if not self.mxfp8_sf_interleaved or cache_k.dtype == self.store_dtype:
                raise ValueError("MXFP8 KV cache requires K and V scale tensors.")
            from sglang.kernels.ops.quantization.mxfp8_quant import quant_store_kv_mxfp8

            quant_store_kv_mxfp8(
                cache_k,
                cache_v,
                loc,
                self.k_buffer[idx],
                self.v_buffer[idx],
                self.k_scale_buffer[idx],
                self.v_scale_buffer[idx],
                page_size=self.page_size,
            )
            return

        # store_cache and store_sf_interleaved skip the reserved CUDA-graph
        # padding slot 0 in-kernel, matching the bf16 pool.
        row_bytes = self.head_num * self.head_dim * self.store_dtype.itemsize
        v_row_bytes = self.head_num * self.v_head_dim * self.store_dtype.itemsize
        assert _is_cuda and can_use_store_cache(row_bytes, v_row_bytes), (
            f"MXFP8 KV cache requires CUDA and store_cache-compatible rows, "
            f"got _is_cuda={_is_cuda}, {row_bytes=}, {v_row_bytes=}"
        )
        assert self.mxfp8_sf_interleaved, (
            "MXFP8 KV cache requires the page_size=128 interleaved scale layout"
        )
        store_cache(
            cache_k.reshape(loc.shape[0], -1),
            cache_v.reshape(loc.shape[0], -1),
            self.k_buffer[idx].view(-1, row_bytes // self.store_dtype.itemsize),
            self.v_buffer[idx].view(-1, v_row_bytes // self.store_dtype.itemsize),
            loc,
            row_bytes=row_bytes,
            v_row_bytes=v_row_bytes,
            size_limit=self.size + self.page_size,
        )
        self._write_scales(idx, loc, k_scale, v_scale)

    def _write_scales(self, idx, loc, k_scale, v_scale):
        """Write per-token UE8M0 K/V scales — interleaved into the FA4
        BlockScaledBasicChunk layout for page_size==128, flat otherwise."""
        if self.mxfp8_sf_interleaved:
            from sglang.kernels.ops.quantization.mxfp8_interleave_sf import (
                store_sf_interleaved,
            )

            store_sf_interleaved(
                k_scale, self.k_scale_buffer[idx], loc, page_size=self.page_size
            )
            store_sf_interleaved(
                v_scale, self.v_scale_buffer[idx], loc, page_size=self.page_size
            )
        else:
            self.k_scale_buffer[idx][loc] = k_scale
            self.v_scale_buffer[idx][loc] = v_scale

    def _read_sf_interleaved(self, sf_buf: torch.Tensor, loc: torch.Tensor):
        """Inverse of store_sf_interleaved: gather per-slot (T, head, sf_dim)
        UE8M0 scales out of the interleaved BlockScaledBasicChunk buffer."""
        num_pages, n = sf_buf.shape[0], sf_buf.shape[1]
        sf_dim = sf_buf.shape[-1]
        # (num_pages, n, page_size) as u32: 4 packed scales per u32.
        buf_u32 = sf_buf.reshape(num_pages, n, -1).view(torch.int32)
        off = loc % self.page_size
        page = (loc // self.page_size).long()
        chunk = self.page_size // self.MXFP8_SCALE_BLOCK_SIZE
        ipos = (
            (off % self.MXFP8_SCALE_BLOCK_SIZE) * chunk
            + (off // self.MXFP8_SCALE_BLOCK_SIZE)
        ).long()
        heads = torch.arange(n, device=loc.device)
        gathered = buf_u32[page[:, None], heads[None, :], ipos[:, None]]  # (T, n) int32
        return (
            gathered.reshape(loc.shape[0], n, 1)
            .view(torch.uint8)
            .reshape(loc.shape[0], n, sf_dim)
            .view(torch.float8_e8m0fnu)
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        # The mamba extra_buffer allocator relocates KV rows during serving;
        # scale rows must travel with their fp8 payload or dequant reads
        # mismatched exponents.
        if self.mxfp8_sf_interleaved:
            from sglang.kernels.ops.quantization.mxfp8_interleave_sf import (
                store_sf_interleaved,
            )

            for idx in range(self.layer_num):
                self.k_buffer[idx][tgt_loc] = self.k_buffer[idx][src_loc]
                self.v_buffer[idx][tgt_loc] = self.v_buffer[idx][src_loc]
                k_sf = self._read_sf_interleaved(self.k_scale_buffer[idx], src_loc)
                v_sf = self._read_sf_interleaved(self.v_scale_buffer[idx], src_loc)
                store_sf_interleaved(
                    k_sf, self.k_scale_buffer[idx], tgt_loc, page_size=self.page_size
                )
                store_sf_interleaved(
                    v_sf, self.v_scale_buffer[idx], tgt_loc, page_size=self.page_size
                )
        else:
            super().move_kv_cache(tgt_loc, src_loc)
            for idx in range(self.layer_num):
                self.k_scale_buffer[idx][tgt_loc] = self.k_scale_buffer[idx][src_loc]
                self.v_scale_buffer[idx][tgt_loc] = self.v_scale_buffer[idx][src_loc]

    def _read_scales(self, idx, loc):
        """Per-token UE8M0 K/V scales at ``loc``, inverse of ``_write_scales``."""
        if self.mxfp8_sf_interleaved:
            return (
                self._read_sf_interleaved(self.k_scale_buffer[idx], loc),
                self._read_sf_interleaved(self.v_scale_buffer[idx], loc),
            )
        return self.k_scale_buffer[idx][loc], self.v_scale_buffer[idx][loc]

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        # The scales travel with their fp8 payload; a restored slot dequantizes
        # against mismatched exponents without them.
        assert not self.use_hnd, (
            "CPU KV offload indexes by slot (NHD); HND KV cache "
            "(SGLANG_USE_HND_KVCACHE) is not supported with CPU offload yet."
        )
        current_platform.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_scale, v_scale = self._read_scales(layer_id, chunk_indices)
                kv_cache_cpu[-1].append(
                    [
                        self.k_buffer[layer_id][chunk_indices].to(
                            "cpu", non_blocking=True
                        ),
                        self.v_buffer[layer_id][chunk_indices].to(
                            "cpu", non_blocking=True
                        ),
                        k_scale.to("cpu", non_blocking=True),
                        v_scale.to("cpu", non_blocking=True),
                    ]
                )
        current_platform.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        assert not self.use_hnd, (
            "CPU KV offload indexes by slot (NHD); HND KV cache "
            "(SGLANG_USE_HND_KVCACHE) is not supported with CPU offload yet."
        )
        current_platform.synchronize()
        device = self.k_buffer[0].device
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu, v_cpu, k_scale_cpu, v_scale_cpu = kv_cache_cpu[layer_id][
                    i // chunk_size
                ]
                assert k_cpu.shape[0] == v_cpu.shape[0] == len(chunk_indices)
                self.k_buffer[layer_id][chunk_indices] = k_cpu.to(
                    device, non_blocking=True
                )
                self.v_buffer[layer_id][chunk_indices] = v_cpu.to(
                    device, non_blocking=True
                )
                self._write_scales(
                    layer_id,
                    chunk_indices,
                    k_scale_cpu.to(device, non_blocking=True),
                    v_scale_cpu.to(device, non_blocking=True),
                )
        current_platform.synchronize()

    def get_kv_scale_buf_infos(self):
        """(ptrs, lens, item_lens) for the UE8M0 scale buffers, k then v.

        The interleaved layout puts pages on the leading axis, so a page's
        scales are one contiguous row; the flat layout is per slot.
        """
        tensors = self.k_scale_buffer + self.v_scale_buffer
        ptrs = [t.data_ptr() for t in tensors]
        lens = [t.nbytes for t in tensors]
        row_bytes = [t[0].nbytes for t in tensors]
        if self.mxfp8_sf_interleaved:
            item_lens = row_bytes
        else:
            item_lens = [rb * self.page_size for rb in row_bytes]
        return ptrs, lens, item_lens

    def set_kv_buffer_prefix_valid(self, *args, **kwargs):
        raise NotImplementedError(
            "prefix-valid commit is unsupported for MXFP8 KV cache "
            "(it does not carry the scale buffers)."
        )

    def get_kv_size_bytes(self):
        k_size_bytes = 0
        v_size_bytes = 0
        for k_cache in self.k_buffer:
            k_size_bytes += get_tensor_size_bytes(k_cache)
        for k_scale in self.k_scale_buffer:
            k_size_bytes += get_tensor_size_bytes(k_scale)
        for v_cache in self.v_buffer:
            v_size_bytes += get_tensor_size_bytes(v_cache)
        for v_scale in self.v_scale_buffer:
            v_size_bytes += get_tensor_size_bytes(v_scale)
        return k_size_bytes, v_size_bytes


@triton.jit
def masked_set_kv_buffer_kernel(
    k_ptr,
    v_ptr,
    k_buffer_ptr,
    v_buffer_ptr,
    loc_ptr,
    mask_ptr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    CHUNK: tl.constexpr,
    k_stride_B: tl.constexpr,
    k_stride_H: tl.constexpr,
    v_stride_B: tl.constexpr,
    v_stride_H: tl.constexpr,
    k_buffer_stride: tl.constexpr,
    v_buffer_stride: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    do_write = tl.load(mask_ptr + pid) != 0
    if not do_write:
        return

    loc = tl.load(loc_ptr + pid)
    total = H * D
    num_chunks = tl.cdiv(total, CHUNK)

    for c in range(num_chunks):
        offs = tl.arange(0, CHUNK)
        idx = c * CHUNK + offs
        mask = idx < total
        row = idx // D
        col = idx % D

        key = tl.load(k_ptr + pid * k_stride_B + row * k_stride_H + col, mask=mask)
        tl.store(k_buffer_ptr + loc * k_buffer_stride + idx, key, mask=mask)

        value = tl.load(v_ptr + pid * v_stride_B + row * v_stride_H + col, mask=mask)
        tl.store(v_buffer_ptr + loc * v_buffer_stride + idx, value, mask=mask)


def get_minimax_sparse_index_dtype(
    *, fp8_attn_gemm: bool, kv_cache_dtype: torch.dtype, model_dtype: torch.dtype
) -> torch.dtype:
    """Return the index-K cache dtype; the pool's cell-size estimate must match it."""
    # fp8 attn-GEMM mode runs the indexer GEMMs in fp8 too; plain fp8 KV keeps bf16.
    if fp8_attn_gemm:
        return kv_cache_dtype
    if _is_gfx95_supported and envs.SGLANG_OPT_MINIMAX_M3_FP8_INDEX_CACHE.get():
        return torch.float8_e4m3fn
    return model_dtype
