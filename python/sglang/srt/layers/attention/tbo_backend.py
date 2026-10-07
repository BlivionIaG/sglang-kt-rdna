from __future__ import annotations
from typing import TYPE_CHECKING, Callable, List, Optional

import torch

from sglang.srt.batch_overlap import two_batch_overlap
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.speculative.spec_info import SpecInput
from types import SimpleNamespace

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


class TboAttnBackend(AttentionBackend):
    def __init__(self, primary: AttentionBackend, children: List[AttentionBackend]):
        super().__init__()
        self.primary = primary
        self.children = children

    @classmethod
    def init_new(cls, creator: Callable[[], AttentionBackend]):
        return cls(
            primary=creator(),
            children=[creator() for _ in range(2)],
        )

    def init_forward_metadata(self, forward_batch: "ForwardBatch"):
        self.primary.init_forward_metadata(forward_batch=forward_batch)
        if forward_batch.tbo_children is not None:
            for child, forward_batch_child in zip(
                self.children, forward_batch.tbo_children, strict=True
            ):
                if forward_batch_child.batch_size > 0:
                    child.init_forward_metadata(forward_batch=forward_batch_child)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.primary.init_cuda_graph_state(max_bs=max_bs, max_num_tokens=max_num_tokens)
        for item in self.children:
            # TODO for children, maybe can provide *smaller* max_bs to optimize
            item.init_cuda_graph_state(max_bs=max_bs, max_num_tokens=max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: "ForwardMode",
        spec_info: Optional[SpecInput],
    ):
        self.primary.init_forward_metadata_capture_cuda_graph(
            bs=bs,
            num_tokens=num_tokens,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            encoder_lens=encoder_lens,
            forward_mode=forward_mode,
            spec_info=spec_info,
        )

        self._init_forward_metadata_cuda_graph_children(
            fn_name="init_forward_metadata_capture_cuda_graph",
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            encoder_lens=encoder_lens,
            forward_mode=forward_mode,
            spec_info=spec_info,
            capture_num_tokens=num_tokens,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: "ForwardMode",
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        self.primary.init_forward_metadata_replay_cuda_graph(
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_sum=seq_lens_sum,
            encoder_lens=encoder_lens,
            forward_mode=forward_mode,
            spec_info=spec_info,
            seq_lens_cpu=seq_lens_cpu,
        )

        self._init_forward_metadata_cuda_graph_children(
            fn_name="init_forward_metadata_replay_cuda_graph",
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            encoder_lens=encoder_lens,
            forward_mode=forward_mode,
            spec_info=spec_info,
            replay_seq_lens_sum=seq_lens_sum,
            replay_seq_lens_cpu=seq_lens_cpu,
        )

    def _init_forward_metadata_cuda_graph_children(
        self,
        fn_name: str,
        # common args
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: "ForwardMode",
        spec_info: Optional[SpecInput],
        # capture args
        capture_num_tokens: int = None,
        # replay args
        replay_seq_lens_sum: int = None,
        replay_seq_lens_cpu: Optional[torch.Tensor] = None,
    ):
        token_num_per_seq = two_batch_overlap.get_token_num_per_seq(
            forward_mode=forward_mode, spec_info=spec_info
        )
        if fn_name == "init_forward_metadata_capture_cuda_graph":
            assert (
                capture_num_tokens == bs * token_num_per_seq
            ), "For target-verify or decode mode, num_tokens should be equal to token_num_per_seq * bs"
        num_tokens = bs * token_num_per_seq

        tbo_split_seq_index, tbo_split_token_index = (
            two_batch_overlap.compute_split_indices_for_cuda_graph_replay(
                forward_mode=forward_mode,
                cuda_graph_num_tokens=num_tokens,
                spec_info=spec_info,
            )
        )

        num_tokens_child_left = tbo_split_token_index
        num_tokens_child_right = num_tokens - tbo_split_token_index
        bs_child_left = tbo_split_seq_index
        bs_child_right = bs - bs_child_left

        assert (
            num_tokens_child_left > 0 and num_tokens_child_right > 0
        ), f"{num_tokens_child_left=} {num_tokens_child_right=} {forward_mode=} {num_tokens=}"

        common_pre_split_args = dict(
            fn_name=fn_name,
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            encoder_lens=encoder_lens,
            forward_mode=forward_mode,
            spec_info=spec_info,
            capture_num_tokens=capture_num_tokens,
            replay_seq_lens_sum=replay_seq_lens_sum,
            replay_seq_lens_cpu=replay_seq_lens_cpu,
        )

        args_left = _init_forward_metadata_cuda_graph_split(
            output_bs=bs_child_left,
            seq_slice=slice(None, tbo_split_seq_index),
            **common_pre_split_args,
        )
        args_right = _init_forward_metadata_cuda_graph_split(
            output_bs=bs_child_right,
            seq_slice=slice(tbo_split_seq_index, None),
            **common_pre_split_args,
        )

        child_left, child_right = self.children
        getattr(child_left, fn_name)(**args_left)
        getattr(child_right, fn_name)(**args_right)

    def get_cuda_graph_seq_len_fill_value(self):
        ans = self.primary.get_cuda_graph_seq_len_fill_value()
        for child in self.children:
            assert ans == child.get_cuda_graph_seq_len_fill_value()
        return ans

    def forward_extend(self, *args, **kwargs):
        return self.primary.forward_extend(*args, **kwargs)

    def forward_decode(self, *args, **kwargs):
        return self.primary.forward_decode(*args, **kwargs)

    def get_indexer_metadata(self, layer_id: int, forward_batch: "ForwardBatch"):
        return self.primary.get_indexer_metadata(layer_id, forward_batch)


def _init_forward_metadata_cuda_graph_split(
    fn_name: str,
    seq_slice: slice,
    output_bs: int,
    # common args
    bs: int,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    encoder_lens: Optional[torch.Tensor],
    forward_mode: "ForwardMode",
    spec_info: Optional[SpecInput],
    # capture args
    capture_num_tokens: int = None,
    # replay args
    replay_seq_lens_sum: int = None,
    replay_seq_lens_cpu: Optional[torch.Tensor] = None,
):
    token_num_per_seq = two_batch_overlap.get_token_num_per_seq(
        forward_mode=forward_mode, spec_info=spec_info
    )
    assert encoder_lens is None, "encoder_lens is not supported yet"
    if spec_info is not None:
        output_spec_info = two_batch_overlap.split_spec_info(
            spec_info=spec_info,
            start_seq_index=seq_slice.start if seq_slice.start is not None else 0,
            end_seq_index=seq_slice.stop if seq_slice.stop is not None else bs,
            start_token_index=(
                seq_slice.start * token_num_per_seq
                if seq_slice.start is not None
                else 0
            ),
            end_token_index=(
                seq_slice.stop * token_num_per_seq
                if seq_slice.stop is not None
                else bs * token_num_per_seq
            ),
        )

    else:
        output_spec_info = None
    ans = dict(
        bs=output_bs,
        req_pool_indices=req_pool_indices[seq_slice],
        seq_lens=seq_lens[seq_slice],
        # directly forward
        forward_mode=forward_mode,
        # ignore
        encoder_lens=None,
        spec_info=output_spec_info,
    )

    if fn_name == "init_forward_metadata_capture_cuda_graph":
        assert (
            capture_num_tokens == bs * token_num_per_seq
        ), "Only support num_tokens==bs * token_num_per_seq for target-verify or decode mode"
        ans.update(
            dict(
                num_tokens=output_bs * token_num_per_seq,
            )
        )
    elif fn_name == "init_forward_metadata_replay_cuda_graph":
        output_seq_lens_cpu = replay_seq_lens_cpu[seq_slice]
        ans.update(
            dict(
                seq_lens_sum=output_seq_lens_cpu.sum().item(),
                seq_lens_cpu=output_seq_lens_cpu,
            )
        )
    else:
        raise NotImplementedError

    return ans


# --- imported with the qwen4 subsystem (sgl-project/sglang) ---


def _build_tbo_child_replay_fb_view(
    fb_view,
    *,
    child_bs: int,
    seq_slice: slice,
    tok_slice: slice,
    token_num_per_seq: int,
) -> SimpleNamespace:
    """Slice a parent replay fb_view into a per-child view.

    Mirrors the legacy ``_init_forward_metadata_cuda_graph_split`` (deleted
    along with the cuda_graph variants) for the new
    ``init_forward_metadata_out_graph(fb_view)`` contract: padded
    capture-time buffers are sliced per child, spec_info is split, and
    seq_lens_sum is recomputed from the sliced ``seq_lens_cpu``.
    """
    assert getattr(fb_view, "encoder_lens", None) is None, (
        "TBO replay split does not support encoder_lens yet"
    )
    spec_info = getattr(fb_view, "spec_info", None)
    if spec_info is not None:
        start_seq = seq_slice.start or 0
        end_seq = seq_slice.stop if seq_slice.stop is not None else start_seq + child_bs
        child_spec_info = two_batch_overlap.split_spec_info(
            spec_info=spec_info,
            start_seq_index=start_seq,
            end_seq_index=end_seq,
            start_token_index=start_seq * token_num_per_seq,
            end_token_index=end_seq * token_num_per_seq,
        )
    else:
        child_spec_info = None
    child_seq_lens_cpu = fb_view.seq_lens_cpu[seq_slice]
    parent_input_ids = getattr(fb_view, "input_ids", None)
    parent_out_cache_loc = getattr(fb_view, "out_cache_loc", None)
    return SimpleNamespace(
        batch_size=child_bs,
        forward_mode=fb_view.forward_mode,
        actual_forward_mode=getattr(
            fb_view, "actual_forward_mode", fb_view.forward_mode
        ),
        input_ids=(
            parent_input_ids[tok_slice] if parent_input_ids is not None else None
        ),
        req_pool_indices=fb_view.req_pool_indices[seq_slice],
        seq_lens=fb_view.seq_lens[seq_slice],
        seq_lens_sum=int(child_seq_lens_cpu.sum()),
        seq_lens_cpu=child_seq_lens_cpu,
        encoder_lens=None,
        out_cache_loc=(
            parent_out_cache_loc[tok_slice]
            if parent_out_cache_loc is not None
            else None
        ),
        spec_info=child_spec_info,
        max_seq_len_override=fb_view.max_seq_len_override,
    )
