from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass
from enum import Enum
from itertools import chain
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
import torch
from sglang.srt.runtime_context import get_disagg, get_exec, get_memory, get_schedule

if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker


class BaseDraftWorker(ABC):
    @abstractmethod
    def draft():
        pass

    @abstractmethod
    def draft_extend():
        pass


class BaseSpecWorker(ABC):
    @property
    @abstractmethod
    def target_worker(self) -> TpModelWorker:
        pass

    @property
    @abstractmethod
    def draft_worker(self) -> BaseDraftWorker:
        pass

    @abstractmethod
    def clear_cache_pool(self):
        # TODO: move this abstract method to BaseTpWorker and call through self.model_runner
        pass


# --- imported with the qwen4 subsystem (sgl-project/sglang) ---


class HiCacheDraftMode(str, Enum):
    NONE = "none"
    PACKED = "packed"
    SIDECAR = "sidecar"


@dataclass(frozen=True, slots=True)
class HiCacheDraftPlan:
    mode: HiCacheDraftMode = HiCacheDraftMode.NONE
    device_pools: tuple[object, ...] = ()


def _can_pack_hicache_mtp(
    spec_algorithm: SpeculativeAlgorithm,
    draft_runners: tuple[ModelRunner, ...],
) -> bool:
    is_nextn_mtp = (
        spec_algorithm.is_eagle()
        and not spec_algorithm.is_eagle3()
        and all(
            runner.model_config.num_nextn_predict_layers for runner in draft_runners
        )
    )
    is_dspark_dsv4 = (
        spec_algorithm.is_dspark()
        and draft_runners[0].model_config.hf_config.architectures[0]
        == "DeepseekV4ForCausalLMDSpark"
    )
    return is_nextn_mtp or is_dspark_dsv4


class EagleDraftWorkerBase(ABC):
    # topk=1 chain constants for draft_forward's fast path; None when topk > 1.
    _topk1_parents_prealloc: Optional[torch.Tensor] = None
    _topk1_score_indices_prealloc: Optional[torch.Tensor] = None

    def __init__(self) -> None:
        self._specialized_graph_memory_usage: dict[str, float] = {}
        self._specialized_graph_time_usage: dict[str, float] = {}

    @abstractmethod
    def draft():
        pass

    @abstractmethod
    def draft_extend():
        pass

    @property
    def draft_runners(self) -> list[ModelRunner]:
        """All draft model runners; multi-layer eagle overrides with its
        per-step runner list."""
        return [self.draft_runner]

    @property
    def graph_memory_usage(self) -> dict[str, float]:
        return merge_graph_memory_usage(
            *(runner.graph_memory_usage for runner in self.draft_runners),
            self._specialized_graph_memory_usage,
        )

    @property
    def graph_time_usage(self) -> dict[str, float]:
        return merge_graph_time_usage(
            *(runner.graph_time_usage for runner in self.draft_runners),
            self._specialized_graph_time_usage,
        )

    @property
    def weight_load_time(self) -> float:
        return sum(runner.weight_load_time for runner in self.draft_runners)

    @property
    def preloaded_weights_bytes(self) -> int:
        return sum(runner.preloaded_weights_bytes for runner in self.draft_runners)

    def alloc_memory_pool(self, **kwargs):
        pass

    def init_attention_backends(self):
        """Subclasses wrap this with their context managers (draft_tp_context,
        speculative_moe_backend_context, etc.) rather than reimplementing it."""
        self.draft_worker.init_attention_backends()
        self.init_attention_backend()

    def init_cuda_graphs(self):
        """Capture draft graphs (decode disabled on the draft TpModelWorker)."""
        self.draft_worker.init_cuda_graphs(capture_decode_cuda_graph=False)
        self._capture_cuda_graphs()

    def _rebuild_topk1_chain_buffers(self) -> None:
        # For topk=1 the draft tree degenerates to a chain, so parent_list and
        # top_scores_index are runtime-invariant. Must be rebuilt after any
        # change to speculative_num_steps / speculative_num_draft_tokens.
        if self.topk != 1:
            return
        # _override_worker_state can set both directly, bypassing the hook that
        # pins this relation; the fast path is only valid when it holds.
        assert self.speculative_num_draft_tokens == self.speculative_num_steps + 1, (
            "topk=1 requires speculative_num_draft_tokens == speculative_num_steps + 1, "
            f"got {self.speculative_num_draft_tokens} and {self.speculative_num_steps}"
        )
        num_steps = self.speculative_num_steps
        sa = self.server_args
        decode_max_bs = (
            get_exec().graph.cuda_graph_config.decode.max_bs
            if get_exec().graph.cuda_graph_config is not None
            else None
        )
        max_bs = max(
            decode_max_bs or 0,
            get_schedule().max_running_requests or 0,
            1,
        )
        # A single-step chain has no parent entries (slow path drops the last
        # step). repeat (not expand): the kernel reads these as contiguous.
        parent_width = num_steps if num_steps > 1 else 0
        self._topk1_parents_prealloc = torch.arange(
            -1, parent_width - 1, dtype=torch.long, device=self.device
        ).repeat(max_bs, 1)
        self._topk1_score_indices_prealloc = torch.arange(
            num_steps, dtype=torch.long, device=self.device
        ).repeat(max_bs, 1)
