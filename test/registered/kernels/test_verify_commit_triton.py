import os
import sys

import pytest
import torch

from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
    fused_commit_track_indices,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def _reference(accept_index, accept_lens, seq_lens, draft_token_num, track_interval):
    """Mirrors the eager branch of spec_utils._verify_commit_step_indices."""
    bs = accept_lens.shape[0]
    offset = torch.arange(
        0,
        bs * draft_token_num,
        step=draft_token_num,
        dtype=accept_lens.dtype,
        device=accept_lens.device,
    )
    req_idx = torch.arange(bs, dtype=torch.int64, device=accept_lens.device)
    last = accept_index[req_idx, (accept_lens - 1).to(torch.int64)] - offset
    if track_interval <= 0:
        return last, None
    pre = seq_lens
    post = seq_lens + accept_lens
    mask = pre // track_interval != post // track_interval
    point = post // track_interval * track_interval
    # Verify step i caches the state after pre + i + 1 tokens, so the exact
    # boundary lives at step point - pre - 1, bounded by the last accepted step
    # (issue #17: the bound must apply to the shifted step index).
    ith = torch.clamp(
        torch.minimum(point - pre - 1, accept_lens - 1), min=0
    ).to(torch.int64)
    cand = accept_index[req_idx, ith] - offset
    track = torch.where(mask, cand, torch.full_like(cand, -1))
    return last, track


@pytest.mark.parametrize("bs", [1, 3, 48, 257])
@pytest.mark.parametrize("track_interval", [0, 64])
def test_verify_commit_steps_matches_eager(bs, track_interval):
    """Regression guard: the eager commit-step math launched ~12 tiny kernels
    on [bs] tensors per verify; the fused kernel must match both outputs,
    including interval-crossing selection near tracking boundaries."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    torch.manual_seed(bs + track_interval)
    device = "cuda"
    draft_token_num = 4
    accept_lens = torch.randint(
        1, draft_token_num + 1, (bs,), device=device, dtype=torch.int64
    )
    accept_index = torch.arange(bs, device=device, dtype=torch.int64).unsqueeze(
        1
    ) * draft_token_num + torch.arange(
        draft_token_num, device=device, dtype=torch.int64
    )
    # Cluster seq lens around tracking boundaries to exercise the crossing.
    seq_lens = torch.randint(60, 70, (bs,), device=device, dtype=torch.int64)

    exp_last, exp_track = _reference(
        accept_index, accept_lens, seq_lens, draft_token_num, track_interval
    )
    got_last, got_track = fused_commit_track_indices(
        accept_index,
        accept_lens,
        seq_lens if track_interval > 0 else None,
        draft_token_num,
        track_interval,
    )
    assert torch.equal(got_last, exp_last)
    if track_interval > 0:
        assert torch.equal(got_track, exp_track)
    else:
        assert got_track is None


def test_verify_commit_steps_clamp_to_accepted_path():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    draft_token_num = 4
    accept_lens = torch.full((4,), 4, dtype=torch.int64, device="cuda")
    accept_index = torch.arange(16, dtype=torch.int64, device="cuda").reshape(4, 4)
    seq_lens = torch.tensor([63, 62, 61, 60], dtype=torch.int64, device="cuda")

    _, track = fused_commit_track_indices(
        accept_index,
        accept_lens,
        seq_lens,
        draft_token_num,
        64,
    )

    # Step i holds the state after seq_pre + i + 1 tokens, so the checkpoint for
    # the 64-token boundary is step 64 - seq_pre - 1: [0, 1, 2, 3]. The last
    # row (pre=60) is the case where the boundary falls on the final accepted
    # step and the accepted-path bound selects it.
    assert torch.equal(track, torch.tensor([0, 1, 2, 3], device="cuda"))


def _track_indices_cpu(accept_index, accept_lens, seq_lens, draft_token_num, interval):
    """Run the fused kernel on CPU tensors under Triton's interpreter so the
    boundary selection is checkable on hosts without a GPU (issue #17).

    ``@triton.jit`` resolves TRITON_INTERPRET at decoration time, so the module
    is re-imported once with the knob on and the original module is restored
    afterwards (the interpreter never touches the CUDA driver, which is absent
    on CPU-only hosts).
    """
    import importlib
    import sys

    mod_name = "sglang.kernels.ops.mamba.mamba_state_scatter_triton"
    parent_name, leaf_name = mod_name.rsplit(".", 1)
    saved = os.environ.get("TRITON_INTERPRET")
    os.environ["TRITON_INTERPRET"] = "1"
    orig = sys.modules.get(mod_name)
    orig_parent_attr = getattr(sys.modules.get(parent_name), leaf_name, None)
    try:
        sys.modules.pop(mod_name, None)
        module = importlib.import_module(mod_name)
        return module.fused_commit_track_indices(
            accept_index, accept_lens, seq_lens, draft_token_num, interval
        )
    finally:
        if saved is None:
            os.environ.pop("TRITON_INTERPRET", None)
        else:
            os.environ["TRITON_INTERPRET"] = saved
        if orig is not None:
            sys.modules[mod_name] = orig
        if orig_parent_attr is not None:
            setattr(sys.modules[parent_name], leaf_name, orig_parent_attr)


def test_verify_commit_steps_boundary_cpu():
    """CPU-runnable boundary expectations for the fused kernel (no CUDA skip).

    Expectations are hand-derived from the verify-kernel cache semantics: step i
    of a req's block holds the state after ``seq_pre + i + 1`` tokens, so the
    checkpoint for a boundary at ``tracking_point`` is the node
    ``accept_index[tracking_point - seq_pre - 1] - b * draft_token_num``; rows
    that cross no interval are -1. On exact base the pre-fix
    ``min(tp - pre, al - 1)`` selection returned [1, 2, -1, 5] here.
    """
    draft_token_num = 8
    # Each req gets one draft_token_num-wide node block; row 3 is a tree
    # (topk > 1) row whose accepted path (positions 0..3) maps to
    # non-sequential in-block nodes 0, 2, 5, 7, so the selected position must
    # be gathered through accept_index rather than equalled. Unaccepted tree
    # siblings sit at positions 4..7 and are never read for al == 4.
    accept_index = torch.tensor(
        [
            [0, 1, 2, 3, -1, -1, -1, -1],
            [8, 9, 10, 11, -1, -1, -1, -1],
            [16, -1, -1, -1, -1, -1, -1, -1],
            [24, 26, 29, 31, 25, 27, 28, 30],
        ],
        dtype=torch.int64,
    )
    accept_lens = torch.tensor([4, 4, 1, 4], dtype=torch.int64)
    seq_lens = torch.tensor([63, 62, 61, 62], dtype=torch.int64)
    # req0: pre=63, post=67, boundary 64 -> position 64-63-1=0 -> node 0-0=0
    #       (pre-fix min(64-63, 3)=1 -> 1).
    # req1: pre=62, post=66, boundary 64 -> position 1 -> node 9-8=1 (the #17
    #       worked example; pre-fix min(2, 3)=2 -> 10-8=2).
    # req2: pre=61, post=62, no crossing (61//64 == 62//64) -> -1.
    # req3: pre=62, post=66, boundary -> position 1 -> tree node 26-24=2
    #       (pre-fix position 2 -> 29-24=5); last accepted node 31-24=7.
    last, track = _track_indices_cpu(
        accept_index, accept_lens, seq_lens, draft_token_num, 64
    )
    assert torch.equal(last, torch.tensor([3, 3, 0, 7], dtype=torch.int64))
    assert torch.equal(track, torch.tensor([0, 1, -1, 2], dtype=torch.int64))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
