from __future__ import annotations
import concurrent.futures
import logging
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from sglang.srt.disaggregation.ascend.transfer_engine import AscendTransferEngine
from sglang.srt.disaggregation.common.utils import group_concurrent_contiguous
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVBootstrapServer,
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
)
from sglang.srt.utils import get_local_ip_auto
import enum
from typing import List, Optional, Tuple
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.utils.network import get_local_ip_auto

logger = logging.getLogger(__name__)


class AscendKVManager(MooncakeKVManager):
    def init_engine(self):
        # TransferEngine initialized on ascend.
        local_ip = get_local_ip_auto()
        self.engine = AscendTransferEngine(
            hostname=local_ip,
            npu_id=self.kv_args.gpu_id,
            disaggregation_mode=self.disaggregation_mode,
        )

    def register_buffer_to_engine(self):
        self.engine.batch_register(self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens)
        # The Ascend backend optimize batch registration for small memory blocks.
        self.engine.batch_register(
            self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens
        )

    def send_kvcache(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
    ):
        # Group by indices
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_kv_indices, dst_kv_indices
        )

        num_layers = len(self.kv_args.kv_data_ptrs)
        layers_params = [
            (
                self.kv_args.kv_data_ptrs[layer_id],
                dst_kv_ptrs[layer_id],
                self.kv_args.kv_item_lens[layer_id],
            )
            for layer_id in range(num_layers)
        ]

        def set_transfer_blocks(
            src_ptr: int, dst_ptr: int, item_len: int
        ) -> List[Tuple[int, int, int]]:
            transfer_blocks = []
            for prefill_index, decode_index in zip(prefill_kv_blocks, dst_kv_blocks):
                src_addr = src_ptr + int(prefill_index[0]) * item_len
                dst_addr = dst_ptr + int(decode_index[0]) * item_len
                length = item_len * len(prefill_index)
                transfer_blocks.append((src_addr, dst_addr, length))
            return transfer_blocks

        # Worker function for processing a single layer
        def process_layer(src_ptr: int, dst_ptr: int, item_len: int) -> int:
            transfer_blocks = set_transfer_blocks(src_ptr, dst_ptr, item_len)
            return self._transfer_data(mooncake_session_id, transfer_blocks)

        # Worker function for processing all layers in a batch
        def process_layers(layers_params: List[Tuple[int, int, int]]) -> int:
            transfer_blocks = []
            for src_ptr, dst_ptr, item_len in layers_params:
                transfer_blocks.extend(set_transfer_blocks(src_ptr, dst_ptr, item_len))
            return self._transfer_data(mooncake_session_id, transfer_blocks)

        if self.enable_custom_mem_pool:
            futures = [
                executor.submit(
                    process_layer,
                    src_ptr,
                    dst_ptr,
                    item_len,
                )
                for (src_ptr, dst_ptr, item_len) in layers_params
            ]
            for future in concurrent.futures.as_completed(futures):
                status = future.result()
                if status != 0:
                    for f in futures:
                        f.cancel()
                    return status
        else:
            # Combining all layers' params in one batch transfer is more efficient
            # compared to using multiple threads
            return process_layers(layers_params)

        return 0


class AscendKVSender(MooncakeKVSender):
    pass


class AscendKVReceiver(MooncakeKVReceiver):
    pass


class AscendKVBootstrapServer(MooncakeKVBootstrapServer):
    pass


# --- imported with the qwen4 subsystem (sgl-project/sglang) ---


class AscendStateType(str, enum.Enum):
    """DSV4-on-NPU PD components without a cross-hardware equivalent."""

    DSV4_C128 = "dsv4_c128"
    # C4 compress-state rows (attention + indexer) addressed within each
    # req_pool_idx bank on A5 (CYCLE cache_mode).  Separate from StateType.SWA
    # because each peer maps logical positions into its own local ring.
    DSV4_C4_STATE = "dsv4_c4_state"


def _build_page_interleaved_dcp_plan(
    src_page_indices: npt.NDArray[np.int32],
    dst_page_indices: npt.NDArray[np.int32],
    *,
    page_size: int,
    dcp_size: int,
    dcp_rank: int,
    src_page_offset: int,
    decode_prefix_len: int,
    num_kv_tokens: int,
) -> Tuple[npt.NDArray[np.int64], ...]:
    """Map physical prefill pages to local/global decode page slots."""
    if not 0 <= dcp_rank < dcp_size:
        raise ValueError(f"Invalid DCP rank {dcp_rank} for size {dcp_size}")
    virtual_page_size = page_size * dcp_size
    if decode_prefix_len % virtual_page_size:
        raise ValueError(
            "Ascend PD DCP requires decode_prefix_len to align to the virtual "
            f"page size ({virtual_page_size}), got {decode_prefix_len}"
        )
    if src_page_offset < 0 or num_kv_tokens < 0:
        raise ValueError(
            "Ascend PD DCP page offset and token count must be nonnegative"
        )

    src_pages = np.asarray(src_page_indices, dtype=np.int64)
    dst_pages = np.asarray(dst_page_indices, dtype=np.int64)
    max_src_pages = (num_kv_tokens + page_size - 1) // page_size
    if src_pages.size > max_src_pages:
        raise ValueError(
            "Ascend PD DCP source page count exceeds the token count: "
            f"pages={src_pages.size}, tokens={num_kv_tokens}, page_size={page_size}"
        )
    if src_pages.size == 0:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty.copy(), empty.copy(), empty.copy()

    # CP may assign this sender only a contiguous subset of the chunk pages;
    # index_slice.start still carries that subset's suffix-relative offset.
    relative_pages = src_page_offset + np.arange(src_pages.size, dtype=np.int64)
    dst_positions = relative_pages // dcp_size
    if dst_positions[-1] >= dst_pages.size:
        raise ValueError(
            "Ascend PD DCP destination does not contain enough virtual pages: "
            f"required={dst_positions[-1] + 1}, available={dst_pages.size}"
        )
    dst_super_pages = dst_pages[dst_positions]
    owners = (decode_prefix_len // page_size + relative_pages) % dcp_size
    local = owners == dcp_rank

    return (
        src_pages[local],
        dst_super_pages[local],
        src_pages,
        dst_super_pages * dcp_size + owners,
    )
