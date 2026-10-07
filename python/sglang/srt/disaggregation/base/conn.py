from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Set

import numpy as np
import numpy.typing as npt

from sglang.srt.server_args import ServerArgs
import enum
import dataclasses

if TYPE_CHECKING:
    from sglang.srt.disaggregation.utils import DisaggregationMode


class KVArgs:
    engine_rank: int
    kv_data_ptrs: List[int]
    kv_data_lens: List[int]
    kv_item_lens: List[int]
    aux_data_ptrs: List[int]
    aux_data_lens: List[int]
    aux_item_lens: List[int]
    state_data_ptrs: List[int]
    state_data_lens: List[int]
    state_item_lens: List[int]
    state_type: str  # "none", "mamba", "swa"
    # for mamba state different tp slice transfer
    state_dim_per_tensor: List[int]  # dimension to slice for each state tensor
    ib_device: str
    ib_traffic_class: str
    gpu_id: int
    # for different tp
    decode_tp_size: int
    kv_head_num: int
    total_kv_head_num: int
    page_size: int
    # for pp prefill
    prefill_pp_size: int
    pp_rank: int
    prefill_start_layer: int
    # for system dp
    system_dp_rank: int


class KVPoll:
    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4


class BaseKVManager(ABC):
    """Base class for managing transfer states"""

    @abstractmethod
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ): ...

    @abstractmethod
    def register_to_bootstrap(self):
        """Register to the bootstrap server."""
        ...


class BaseKVSender(ABC):

    @abstractmethod
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ): ...

    @abstractmethod
    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        """
        Set req's index metadata locally or notify the decoder server about the kv indices length and aux index.
        """
        ...

    @abstractmethod
    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List[int]] = None,
    ):
        """
        Send the kv cache at the given kv indices and the extra cache/state at the given indices to the decoder server.
        """
        ...

    @abstractmethod
    def poll(self) -> KVPoll:
        """
        Check the status of the kv cache transfer.
        """
        ...

    @abstractmethod
    def failure_exception(self):
        """
        Raise an exception if the kv cache transfer fails.
        """
        ...


class BaseKVReceiver(ABC):

    @abstractmethod
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ): ...

    @abstractmethod
    def init(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
    ):
        """
        Set req's index metadata locally or notify the prefill server about the kv indices, aux index, and state_indices.
        """
        ...

    @abstractmethod
    def poll(self) -> KVPoll:
        """
        Check the status of the kv cache transfer.
        """
        ...

    @abstractmethod
    def failure_exception(self):
        """
        Raise an exception if the kv cache transfer fails.
        """
        ...

    def clear(self):
        """
        Clear any internal states.
        """
        pass

    def abort(self):
        """
        Abort the current transfer.
        """
        pass


class BaseKVBootstrapServer(ABC):
    @abstractmethod
    def __init__(self, host: str, port: int, dp_size: int = 1): ...


# --- imported with the qwen4 subsystem (sgl-project/sglang) ---


class StateType(str, enum.Enum):
    MAMBA = "mamba"
    QSA_PENDING = "qsa_pending"
    QSA_COMPRESSED = "qsa_compressed"
    SWA = "swa"
    DSA = "dsa"
    # DSA kpool-compress tail: one per-request ring row. The indices encode
    # only the live subrange of that row for the current open pool.
    DSA_TAIL = "dsa_tail"
    MINIMAX_INDEX_K = "minimax_index_k"
    MINIMAX_DENSE_KV = "minimax_dense_kv"
    # DeepSeek-V4 unified_kv SWA ring: addressed per-row by ring slot
    # (req_pool_idx * ring_stride + pos % ring_stride), needs its own component.
    SWA_RING = "swa_ring"
    # DeepSeek-V4 request-scoped compression state; preserve the legacy wire value.
    DSV4_REQUEST_STATE = "c128_state"
    # A block-scaled KV dtype keeps its per-block scales in buffers parallel to
    # K/V, one component per sub-pool so each carries the index payload of the
    # KV it describes (whole sequence for full attention, window for SWA).
    BLOCK_SCALE = "block_scale"
    BLOCK_SCALE_SWA = "block_scale_swa"


@dataclasses.dataclass
class KVTransferMetric:
    # Backends that cannot isolate transfer latency can leave this as None.
    transfer_latency_s: Optional[float] = None
    # Backends that cannot isolate allocation wait latency can leave this as None.
    alloc_latency_s: Optional[float] = None
    transfer_total_bytes: Optional[int] = None


class KVTransferDestination(str, enum.Enum):
    DEVICE = "device"
    HOST = "host"
