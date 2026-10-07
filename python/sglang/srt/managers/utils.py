from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, List, Optional, Union, Any

import torch
from dataclasses import dataclass, field
from enum import auto
from copy import copy

from sglang.srt.eplb.expert_distribution import ExpertDistributionMetrics
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.overlap_utils import FutureIndices
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.server_args import ServerArgs
import msgspec

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.speculative.eagle_info import EagleDraftInput


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class GenerationBatchResult:
    logits_output: Optional[LogitsProcessorOutput] = None
    pp_hidden_states_proxy_tensors: Optional[PPProxyTensors] = None
    next_token_ids: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None
    num_accepted_tokens: int = 0
    accept_length_per_req_cpu: Optional[List[int]] = None
    can_run_cuda_graph: bool = False

    # For output processing
    extend_input_len_per_req: Optional[List[int]] = None
    extend_logprob_start_len_per_req: Optional[List[int]] = None

    # For overlap scheduling
    copy_done: Optional[torch.cuda.Event] = None
    delay_sample_func: Optional[callable] = None
    future_indices: Optional[FutureIndices] = None

    # FIXME(lsyin): maybe move to a better place?
    # sync path: forward stream -> output processor
    accept_lens: Optional[torch.Tensor] = None

    # relay path: forward stream -> next step forward
    next_draft_input: Optional[EagleDraftInput] = None

    # metrics
    expert_distribution_metrics: Optional[ExpertDistributionMetrics] = None

    def copy_to_cpu(self, return_logprob: bool):
        """Copy tensors to CPU in overlap scheduling.
        Only the tensors which are needed for processing results are copied,
        e.g., next_token_ids, logits outputs
        """
        if return_logprob:
            if self.logits_output.next_token_logprobs is not None:
                self.logits_output.next_token_logprobs = (
                    self.logits_output.next_token_logprobs.to("cpu", non_blocking=True)
                )
            if self.logits_output.input_token_logprobs is not None:
                self.logits_output.input_token_logprobs = (
                    self.logits_output.input_token_logprobs.to("cpu", non_blocking=True)
                )
        if self.logits_output.hidden_states is not None:
            self.logits_output.hidden_states = self.logits_output.hidden_states.to(
                "cpu", non_blocking=True
            )
        self.next_token_ids = self.next_token_ids.to("cpu", non_blocking=True)

        if self.accept_lens is not None:
            self.accept_lens = self.accept_lens.to("cpu", non_blocking=True)

        if (x := self.expert_distribution_metrics) is not None:
            x.copy_to_cpu()

        self.copy_done.record()

    @classmethod
    def from_pp_proxy(
        cls, logits_output, next_pp_outputs: PPProxyTensors, can_run_cuda_graph
    ):
        # TODO(lsyin): refactor PP and avoid using dict
        proxy_dict = next_pp_outputs.tensors
        return cls(
            logits_output=logits_output,
            pp_hidden_states_proxy_tensors=None,
            next_token_ids=next_pp_outputs["next_token_ids"],
            extend_input_len_per_req=proxy_dict.get("extend_input_len_per_req", None),
            extend_logprob_start_len_per_req=proxy_dict.get(
                "extend_logprob_start_len_per_req", None
            ),
            can_run_cuda_graph=can_run_cuda_graph,
        )


def validate_input_length(
    req: Req, max_req_input_len: int, allow_auto_truncate: bool
) -> Optional[str]:
    """Validate and potentially truncate input length.

    Args:
        req: The request containing input_ids to validate
        max_req_input_len: Maximum allowed input length
        allow_auto_truncate: Whether to truncate long inputs

    Returns:
        Error message if validation fails, None if successful
    """
    if len(req.origin_input_ids) >= max_req_input_len:
        if allow_auto_truncate:
            logger.warning(
                "Request length is longer than the KV cache pool size or "
                "the max context length. Truncated. "
                f"{len(req.origin_input_ids)=}, {max_req_input_len=}."
            )
            req.origin_input_ids = req.origin_input_ids[:max_req_input_len]
            return None
        else:
            error_msg = (
                f"Input length ({len(req.origin_input_ids)} tokens) exceeds "
                f"the maximum allowed length ({max_req_input_len} tokens). "
                f"Use a shorter input or enable --allow-auto-truncate."
            )
            return error_msg

    return None


def get_logprob_dict_from_result(result: GenerationBatchResult) -> dict:

    logits_output = result.logits_output
    assert logits_output is not None

    return {
        "extend_input_len_per_req": result.extend_input_len_per_req,
        "extend_logprob_start_len_per_req": result.extend_logprob_start_len_per_req,
        "next_token_logprobs": result.logits_output.next_token_logprobs,
        "next_token_top_logprobs_val": result.logits_output.next_token_top_logprobs_val,
        "next_token_top_logprobs_idx": result.logits_output.next_token_top_logprobs_idx,
        "next_token_token_ids_logprobs_val": result.logits_output.next_token_token_ids_logprobs_val,
        "next_token_token_ids_logprobs_idx": result.logits_output.next_token_token_ids_logprobs_idx,
        "input_token_logprobs": result.logits_output.input_token_logprobs,
        "input_top_logprobs_val": result.logits_output.input_top_logprobs_val,
        "input_top_logprobs_idx": result.logits_output.input_top_logprobs_idx,
        "input_token_ids_logprobs_val": result.logits_output.input_token_ids_logprobs_val,
        "input_token_ids_logprobs_idx": result.logits_output.input_token_ids_logprobs_idx,
    }


def get_logprob_from_pp_outputs(
    next_pp_outputs: PPProxyTensors,
) -> tuple[LogitsProcessorOutput, list[int], list[int]]:
    logits_output = LogitsProcessorOutput(
        # Do not send logits and hidden states because they are large
        next_token_logits=None,
        hidden_states=None,
        next_token_logprobs=next_pp_outputs["next_token_logprobs"],
        next_token_top_logprobs_val=next_pp_outputs["next_token_top_logprobs_val"],
        next_token_top_logprobs_idx=next_pp_outputs["next_token_top_logprobs_idx"],
        next_token_token_ids_logprobs_val=next_pp_outputs[
            "next_token_token_ids_logprobs_val"
        ],
        next_token_token_ids_logprobs_idx=next_pp_outputs[
            "next_token_token_ids_logprobs_idx"
        ],
        input_token_logprobs=next_pp_outputs["input_token_logprobs"],
        input_top_logprobs_val=next_pp_outputs["input_top_logprobs_val"],
        input_top_logprobs_idx=next_pp_outputs["input_top_logprobs_idx"],
        input_token_ids_logprobs_val=next_pp_outputs["input_token_ids_logprobs_val"],
        input_token_ids_logprobs_idx=next_pp_outputs["input_token_ids_logprobs_idx"],
    )
    extend_input_len_per_req = next_pp_outputs["extend_input_len_per_req"]
    extend_logprob_start_len_per_req = next_pp_outputs[
        "extend_logprob_start_len_per_req"
    ]

    return logits_output, extend_input_len_per_req, extend_logprob_start_len_per_req


def get_alloc_len_per_decode(server_args: Optional[ServerArgs] = None) -> int:
    if server_args is None:
        from sglang.srt.server_args import get_global_server_args

        server_args = get_global_server_args()

    if server_args.speculative_algorithm is None:
        return 1

    # Spec v1:
    # 1) alloc topk * num_steps when draft decoding and then restore the allocation
    # 2) alloc num_draft_tokens when verifying the drafts
    # Sepc v2: allocate max(topk * num_steps, num_draft_tokens)

    spec_steps = server_args.speculative_num_steps or 1
    spec_topk = server_args.speculative_eagle_topk or 1
    spec_tokens = server_args.speculative_num_draft_tokens
    page_size = server_args.page_size

    if page_size == 1 or spec_topk == 1:
        return max(spec_steps * spec_topk, spec_tokens)
    else:
        raise NotImplementedError(
            "get_alloc_len_per_decode not implemented for page_size > 1 and spec_topk > 1"
        )


# --- imported with the qwen4 subsystem (sgl-project/sglang) ---


def allocate_distinct_stream(device_module, avoid_streams):
    """Draw a stream that aliases none of ``avoid_streams``.

    CUDA/HIP streams come from a fixed round-robin pool, so a fresh ``Stream()``
    may hand back one that is already in use.
    """
    avoid = {stream.cuda_stream for stream in avoid_streams}
    for _ in range(65):
        stream = device_module.Stream(priority=0)
        if stream.cuda_stream not in avoid:
            return stream
    raise RuntimeError("Unable to allocate a distinct stream")


@dataclass
class EmbeddingBatchResult:
    """Result from an embedding/classification forward pass.

    Attributes:
        embeddings: Model output — pooled embeddings or classification logits.
        pooled_hidden_states: Raw hidden states before the task head.  Present
            only when the batch contained ``return_pooled_hidden_states=True``
            requests.  Tensor (uniform shapes) or list of tensors (MIS).
        copy_done: CUDA event recorded after the async CPU copy completes.
    """

    embeddings: torch.Tensor
    pooled_hidden_states: Optional[torch.Tensor] = None
    copy_done: Optional[torch.cuda.Event] = None
    can_run_cuda_graph: bool = False

    @torch.profiler.record_function("copy_embedding_to_cpu")
    def copy_to_cpu(self):
        """Copy embeddings and pooled hidden states to CPU for overlap scheduling."""
        if isinstance(self.embeddings, torch.Tensor):
            self.copy_done = torch.get_device_module(self.embeddings.device).Event()
            self.embeddings = _async_d2h(self.embeddings)
        else:
            assert isinstance(self.embeddings, list)
            if len(self.embeddings) == 0:
                return

            self.copy_done = torch.get_device_module(self.embeddings[0].device).Event()
            self.embeddings = [_async_d2h(emb) for emb in self.embeddings]

        if self.pooled_hidden_states is not None:
            if isinstance(self.pooled_hidden_states, list):
                self.pooled_hidden_states = [
                    _async_d2h(t) for t in self.pooled_hidden_states
                ]
            else:
                self.pooled_hidden_states = _async_d2h(self.pooled_hidden_states)

        self.copy_done.record()


def is_health_check_generate_req(recv_req):
    rid = getattr(recv_req, "rid", None)
    return rid is not None and rid.startswith(HEALTH_CHECK_RID_PREFIX)


class MsgpackDecodeError(ValueError):
    """A msgpack frame the typed decoder rejected, with the failure explained:
    ``rid`` (when recoverable from the raw tagged array) and a human-readable
    ``reason`` whose leading ``$[<n>]`` array index is resolved to the struct
    field name.
    """

    def __init__(self, rid: Optional[str], reason: str):
        super().__init__(reason)
        self.rid = rid
        self.reason = reason


def msgpack_decode_explained(data: bytes) -> Any:
    """`io_struct.msgpack_decode`, but a rejected frame raises
    `MsgpackDecodeError` carrying the rid (recovered via an untyped re-decode of
    the tagged array) and a reason with the failing field named — for callers
    that must report the failure back to a client (e.g. the rust ingress)
    instead of just crashing."""
    # TODO: the hook_custom_types() currently only apply for unit tests, once it
    # esclate to the main code, we can provide a function to access the _all_types

    try:
        return io_struct.msgpack_decode(data)
    except Exception as e:
        msg = str(e)
        try:
            arr = msgspec.msgpack.decode(data)
        except Exception:
            arr = None
        if not (isinstance(arr, (list, tuple)) and arr):
            raise MsgpackDecodeError(None, msg) from e
        # Tagged array_like layout is [tag, *fields]; rid is the first field of
        # every BaseReq struct.
        rid = str(arr[1]) if len(arr) > 1 and arr[1] is not None else None
        tag_to_fields = {
            cls.__struct_config__.tag: cls.__struct_fields__
            for cls in io_struct._all_types
            if isinstance(cls, type) and issubclass(cls, msgspec.Struct)
        }
        fields = tag_to_fields.get(arr[0])
        if fields is not None:
            # Leading ``$[<n>]`` in a msgspec ValidationError path, e.g.
            # ``$[12][0]``.
            m = re.search(r"\$\[(\d+)\]", msg)
            if m is not None:
                idx = int(m.group(1))
                if 1 <= idx <= len(fields):
                    msg = f"{msg[: m.start()]}$.{fields[idx - 1]}{msg[m.end() :]}"
        raise MsgpackDecodeError(rid, msg) from e


def compute_num_reserved_tokens() -> int:
    """Output token slots reserved per request, on top of its input.

    The current eagle implementation stores draft tokens in the output token
    slots, so the context budget has to account for them; every other algorithm
    reserves nothing. Shared by `TokenizerManager` and the rust server's
    `server_args` handoff (`RustServer._build_server_args`), which needs the same
    number to run the total-token check in Rust.
    """
    spec = get_spec()
    algorithm = SpeculativeAlgorithm.from_string(spec.speculative_algorithm)
    if not algorithm.is_eagle():
        return 0
    return max(
        spec.speculative_eagle_topk * spec.speculative_num_steps,
        max_speculative_num_draft_tokens(),
    )
