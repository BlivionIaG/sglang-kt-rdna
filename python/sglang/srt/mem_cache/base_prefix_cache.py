from __future__ import annotations

import dataclasses
import time
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    TYPE_CHECKING,
    Tuple,
    runtime_checkable,
)

import torch
from enum import Enum, auto

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.observability.metrics_collector import RadixCacheMetricsCollector
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
import types

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.radix_cache import RadixKey


@runtime_checkable
class PrefixCacheTrait(Protocol):
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    page_size: int
    disable: bool


@dataclasses.dataclass
class MatchPrefixParams:
    """Unified parameters for match_prefix across different cache types"""

    key: RadixKey

    # Mamba specific
    cow_mamba: bool = False
    req: Optional[Req] = None


@dataclasses.dataclass
class InsertParams:
    """Unified parameters for insert across different cache types"""

    key: RadixKey
    value: Optional[torch.Tensor] = None

    # Mamba specific
    mamba_value: Optional[torch.Tensor] = None

    # SWA specific
    prev_prefix_len: int = 0
    swa_evicted_seqlen: int = 0

    # General
    chunked: bool = False
    priority: int = 0


@dataclasses.dataclass
class InsertResult:
    """Result of an insert operation"""

    prefix_len: int
    mamba_exist: bool = False


@dataclasses.dataclass
class EvictParams:
    """Unified parameters for evict across different cache types"""

    num_tokens: int
    swa_num_tokens: int = 0
    mamba_num: int = 0


@dataclasses.dataclass
class EvictResult:
    """Result of an evict operation"""

    num_tokens_evicted: int = 0
    swa_num_tokens_evicted: int = 0
    mamba_num_evicted: int = 0


class MatchResult(NamedTuple):
    """Result of a prefix match operation.

    Attributes:
        device_indices  :   Indices of the KV cache on the device matched by common prefix.
        last_device_node:   The last TreeNode on the device that was matched.
        last_host_node  :   The last TreeNode on the host that was matched.
                            Note that if HiCache is not enabled,
                            this **must** be the same as `last_device_node`.
        host_hit_length :   Length of the KV cache hit on the host, if applicable.
                            0 if HiCache is not enabled.
        mamba_branching_seqlen: The mamba radix cache branching point, which is the longest
                                page-aligned position that could've been cache hit if there
                                exists a mamba state.
    """

    device_indices: torch.Tensor
    last_device_node: Any
    last_host_node: Any
    host_hit_length: int = 0
    mamba_branching_seqlen: Optional[int] = None


class BasePrefixCache(ABC, PrefixCacheTrait):
    """Cache can be indexed by either rid or key."""

    metrics_collector: Optional[RadixCacheMetricsCollector] = (
        None  # metrics collector for the cache
    )

    def init_metrics_collector(self):
        from sglang.srt.server_args import get_global_server_args

        server_args = get_global_server_args()
        labels = {"cache_type": self.__class__.__name__}
        if server_args.extra_metric_labels:
            labels.update(server_args.extra_metric_labels)
        self.metrics_collector = RadixCacheMetricsCollector(labels=labels)

    def update_eviction_metrics(self, num_evicted: int, start_time: float):
        if self.metrics_collector is not None and num_evicted > 0:
            self.metrics_collector.observe_eviction_duration(
                time.perf_counter() - start_time
            )
            self.metrics_collector.increment_eviction_num_tokens(num_evicted)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        pass

    @abstractmethod
    def cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs):
        pass

    @abstractmethod
    def cache_unfinished_req(self, req: Req, **kwargs):
        pass

    @abstractmethod
    def evict(self, params: EvictParams) -> EvictResult:
        pass

    @abstractmethod
    def inc_lock_ref(self, node: Any):
        pass

    @abstractmethod
    def dec_lock_ref(self, node: Any, swa_uuid_for_lock: Optional[str] = None):
        pass

    def evictable_size(self):
        return 0

    def full_evictable_size(self):
        return 0

    def swa_evictable_size(self):
        return 0

    def protected_size(self):
        return 0

    def full_protected_size(self):
        return 0

    def swa_protected_size(self):
        return 0

    def total_size(self):
        raise NotImplementedError()

    def pretty_print(self):
        raise NotImplementedError()

    def init_load_back(
        self,
        last_host_node: Any,
        host_hit_length: int,
    ) -> Tuple[torch.Tensor, Any]:
        """
        Preparing KV cache loading from host to device.
        """
        raise NotImplementedError()

    def ready_to_load_host_cache(self) -> Any:
        """
        Notify the cache controller to start the KV cache loading
        """
        raise NotImplementedError()

    def check_hicache_events(self) -> Any:
        """
        Check HiCache related activities to update radix tree and synchronize across TP workers if needed
        """
        raise NotImplementedError()

    def take_events(self):
        return []

    def supports_swa(self) -> bool:
        return False

    def supports_mamba(self) -> bool:
        return False

    def is_chunk_cache(self) -> bool:
        return False

    def is_tree_cache(self) -> bool:
        return not self.is_chunk_cache()

    def available_and_evictable_str(self) -> str:
        available_size = self.token_to_kv_pool_allocator.available_size()
        evictable_size = self.evictable_size()
        return f"Available tokens: {available_size + evictable_size} ({available_size=} + {evictable_size=})\n"


# --- imported with the qwen4 subsystem (sgl-project/sglang) ---


@dataclasses.dataclass(frozen=True)
class CacheRequestHandle:
    rid: str
    attempt_id: int


class CacheRequestOutcome(Enum):
    SUCCESS = auto()
    ABORT = auto()


@dataclasses.dataclass
class IncLockRefResult:
    """Receipt returned by ``inc_lock_ref``.

    ``node_id`` is the anchor the lock was taken on; a release replays the
    receipt on that node only. A recorded UUID marks a segment boundary;
    ``None`` means root, while an absent entry means no receipt.
    ``skipped_lock_components`` records the components the acquire left
    untaken, so the release leaves them untouched.
    """

    delta: Optional[int] = None
    node_id: Optional[int] = None
    skipped_lock_components: tuple[ComponentType, ...] = ()
    component_lock_uuids: dict[ComponentType, Optional[int]] = dataclasses.field(
        default_factory=dict
    )
    component_host_lock_uuids: dict[ComponentType, Optional[int]] = dataclasses.field(
        default_factory=dict
    )

    def set_lock_uuid(
        self,
        component_type: ComponentType,
        uuid: Optional[int],
        *,
        lock_host: bool = False,
    ) -> None:
        uuids = (
            self.component_host_lock_uuids if lock_host else self.component_lock_uuids
        )
        uuids[component_type] = uuid

    def to_dec_params(self) -> DecLockRefParams:
        """Convert to the corresponding DecLockRefParams for dec_lock_ref."""
        return DecLockRefParams(
            node_id=self.node_id,
            skipped_lock_components=tuple(self.skipped_lock_components),
            component_lock_uuids=dict(self.component_lock_uuids),
            component_host_lock_uuids=dict(self.component_host_lock_uuids),
        )


@dataclasses.dataclass
class DecLockRefParams:
    """Receipt required by unified-tree ``dec_lock_ref``.

    A segment release requires its component's boundary entry; a missing
    entry must not be treated as a lock reaching the root. ``node_id`` is
    ``None`` only for receipts that never came from a unified-tree acquire
    (legacy caches, session sentinels).
    """

    node_id: Optional[int] = None
    skipped_lock_components: tuple[ComponentType, ...] = ()
    component_lock_uuids: dict[ComponentType, Optional[int]] = dataclasses.field(
        default_factory=dict
    )
    component_host_lock_uuids: dict[ComponentType, Optional[int]] = dataclasses.field(
        default_factory=dict
    )

    def get_lock_uuid(
        self, component_type: ComponentType, *, lock_host: bool = False
    ) -> Optional[int]:
        uuids = (
            self.component_host_lock_uuids if lock_host else self.component_lock_uuids
        )
        return uuids[component_type]


@dataclasses.dataclass
class TreeLock:
    """``receipt`` replays the acquire on release; ``swa_released`` marks the
    SWA part released early, so neither release takes it twice."""

    node: Any
    receipt: DecLockRefParams
    swa_released: bool = False


@dataclasses.dataclass
class DecLockRefResult:
    """Result of an dec_lock_ref operation."""

    delta: Optional[int] = None


@dataclasses.dataclass
class InitLoadBackParams:
    """Unified parameters for init_load_back across different cache types."""

    best_match_node: Any
    host_hit_length: int
    mem_quota: Optional[int] = None
    req: Optional[Req] = None


def zero_match_result(
    tree_cache, match_result: MatchResult, extra_key: Optional[str] = None
) -> MatchResult:
    if not tree_cache.supports_prefix_sharing():
        # match_prefix already returns a miss; no root_node to walk back to.
        return match_result
    root = tree_cache.root_node_handle(extra_key=extra_key)
    return match_result._replace(
        # [:0] keeps dtype and device of the original tensor (e.g. CUDA int64)
        # without allocating a fresh empty tensor.
        device_indices=match_result.device_indices[:0],
        last_device_node=root,
        last_host_node=root,
        best_match_node=root,
        host_hit_length=0,
        swa_host_hit_length=0,
        swa_branching_seqlen=None,
        mamba_host_hit_length=0,
        full_kv_hit_length=0,
    )


def _dfs_weight_order(
    root_node: Any,
    node_handles: Sequence[Any],
    resolve_node_handle: Callable[[Any], Any],
) -> list[int]:
    last_node_to_indices: dict[Any, list[int]] = {}
    for index, node_handle in enumerate(node_handles):
        node = resolve_node_handle(node_handle)
        last_node_to_indices.setdefault(node, []).append(index)

    node_to_weight: dict[Any, int] = {
        node: len(indices) for node, indices in last_node_to_indices.items()
    }

    stack: list[tuple[Any, bool]] = [(root_node, False)]
    while stack:
        node, visited = stack.pop()
        if visited:
            weight = node_to_weight.get(node, 0)
            for child in node.children.values():
                weight += node_to_weight.get(child, 0)
            node_to_weight[node] = weight
            continue
        stack.append((node, True))
        for child in reversed(list(node.children.values())):
            stack.append((child, False))

    order: list[int] = []

    stack = [(root_node, False)]
    while stack:
        node, visited = stack.pop()
        if visited:
            order.extend(last_node_to_indices.get(node, ()))
            continue
        children = list(node.children.values())
        children.sort(key=lambda child: -node_to_weight.get(child, 0))
        stack.append((node, True))
        for child in reversed(children):
            stack.append((child, False))
    return order
