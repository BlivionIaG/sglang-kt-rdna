"""Op-level checks of the RDNA sgl-kernel module against torch.

Skip the whole module when no HIP/CUDA device is visible. This file is the
hardware correctness test; the agent VM that added it did not run it.
rmsnorm is not registered on the ROCm module (FlashInfer norm.cuh); the
Python layernorm path is forward_native, so that case is skipped with that
reason when the op is absent.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="RDNA op reference needs a visible HIP or CUDA device",
)


def _has_op(name: str) -> bool:
    return hasattr(torch.ops, "sgl_kernel") and hasattr(torch.ops.sgl_kernel, name)


def test_silu_gelu_against_torch():
    from sgl_kernel import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul

    x = torch.randn(2, 4, 256, device="cuda", dtype=torch.float16)
    dim = x.shape[-1] // 2
    torch.testing.assert_close(
        silu_and_mul(x),
        x[..., dim:] * torch.nn.functional.silu(x[..., :dim]),
        rtol=1e-3,
        atol=1e-3,
    )
    torch.testing.assert_close(
        gelu_and_mul(x),
        x[..., dim:] * torch.nn.functional.gelu(x[..., :dim], approximate="none"),
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        gelu_tanh_and_mul(x),
        x[..., dim:] * torch.nn.functional.gelu(x[..., :dim], approximate="tanh"),
        rtol=1e-2,
        atol=1e-2,
    )


def _torch_topk_softmax(logits: torch.Tensor, k: int):
    probs = torch.softmax(logits, dim=-1)
    return torch.topk(probs, k=k, dim=-1)


def test_topk_softmax_and_sigmoid():
    from sgl_kernel import topk_sigmoid, topk_softmax

    logits = torch.tensor(
        [[0.2, 1.5, -0.4, 0.7], [2.0, -1.0, 0.3, 0.1]],
        device="cuda",
        dtype=torch.float32,
    )
    k = 2
    weights = torch.empty(logits.shape[0], k, device="cuda", dtype=torch.float32)
    ids = torch.empty(logits.shape[0], k, device="cuda", dtype=torch.int32)
    topk_softmax(weights, ids, logits, False, 0.0, None)
    ref_w, ref_i = _torch_topk_softmax(logits, k)
    torch.testing.assert_close(weights, ref_w, rtol=1e-4, atol=1e-4)
    assert torch.equal(ids, ref_i.to(torch.int32))

    sig_w = torch.empty_like(weights)
    sig_i = torch.empty_like(ids)
    topk_sigmoid(sig_w, sig_i, logits, False, None)
    sig = torch.sigmoid(logits)
    ref_sw, ref_si = torch.topk(sig, k=k, dim=-1)
    torch.testing.assert_close(sig_w, ref_sw, rtol=1e-4, atol=1e-4)
    assert torch.equal(sig_i, ref_si.to(torch.int32))


def _align_sets(topk_ids: torch.Tensor, block_size: int, num_experts: int):
    """Kernel contract: slot = id + 1, expert_ids = slot - 1, pad id = numel."""

    flat = topk_ids.reshape(-1).to(torch.int64)
    numel = flat.numel()
    slot = flat + 1
    counts = torch.bincount(slot.cpu(), minlength=num_experts)
    padded = ((counts + block_size - 1) // block_size) * block_size
    total = int(padded.sum().item())
    order = torch.argsort(slot.cpu(), stable=True)
    per_expert = {}
    for expert in range(num_experts):
        ids = order[slot.cpu()[order] == expert]
        per_expert[expert - 1] = set(int(i) for i in ids)
    return per_expert, total, numel


def test_moe_align_per_expert_sets():
    from sgl_kernel import moe_align_block_size

    topk_ids = torch.tensor(
        [[0, 1], [1, -1], [0, 2], [2, 0]], device="cuda", dtype=torch.int32
    )
    logical_experts = 3
    block_size = 4
    kernel_experts = logical_experts + 1
    numel = topk_ids.numel()
    max_num_tokens_padded = max(numel * block_size, numel + kernel_experts * block_size)
    sorted_ids = torch.full(
        (max_num_tokens_padded,), numel, dtype=torch.int32, device="cuda"
    )
    expert_ids = torch.full(
        ((max_num_tokens_padded + block_size - 1) // block_size,),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    num_tokens_post_pad = torch.empty(1, dtype=torch.int32, device="cuda")
    cumsum = torch.empty(kernel_experts + 1, dtype=torch.int32, device="cuda")
    moe_align_block_size(
        topk_ids,
        kernel_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum,
        True,
    )
    per_expert, total, pad = _align_sets(topk_ids, block_size, kernel_experts)
    assert int(num_tokens_post_pad.item()) == total
    got = {expert: set() for expert in per_expert}
    for block, expert in enumerate(expert_ids.tolist()):
        if block * block_size >= total:
            break
        chunk = sorted_ids[block * block_size : (block + 1) * block_size].tolist()
        got.setdefault(int(expert), set()).update(i for i in chunk if i != pad)
    for expert, ids in per_expert.items():
        assert got.get(expert, set()) == ids


def test_rotary_embedding_against_native():
    from sgl_kernel.testing.rotary_embedding import (
        RotaryEmbedding,
        SglKernelRotaryEmbedding,
    )

    head_size = 64
    batch, heads = 4, 4
    positions = torch.arange(batch, device="cuda")
    query = torch.randn(batch, heads * head_size, device="cuda", dtype=torch.float16)
    key = torch.randn(batch, heads * head_size, device="cuda", dtype=torch.float16)
    native = RotaryEmbedding(head_size, head_size, 128, 10000, True, torch.float16).cuda()
    kernel = SglKernelRotaryEmbedding(
        head_size, head_size, 128, 10000, True, torch.float16
    ).cuda()
    kernel.cos_sin_cache = native.cos_sin_cache
    q_ref, k_ref = native.forward_native(positions, query.clone(), key.clone())
    q_out, k_out = kernel.forward_cuda(positions, query.clone(), key.clone())
    torch.testing.assert_close(q_out, q_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k_out, k_ref, rtol=1e-2, atol=1e-2)


def test_grammar_bitmask():
    from sgl_kernel import apply_token_bitmask_inplace_cuda

    logits = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device="cuda", dtype=torch.float32
    )
    bitmask = torch.tensor([0b01010101], dtype=torch.int32, device="cuda")
    apply_token_bitmask_inplace_cuda(logits, bitmask)
    allowed = torch.tensor(
        [True, False, True, False, True, False, True, False], device="cuda"
    )
    assert torch.all(torch.isfinite(logits[allowed]))
    assert torch.all(logits[~allowed] < -1e20)


def test_kv_transfer_per_layer():
    from sgl_kernel.kvcacheio import transfer_kv_per_layer

    dtype = torch.float16
    item = 32
    src_k = torch.randn(8, item, dtype=dtype).pin_memory()
    src_v = torch.randn(8, item, dtype=dtype).pin_memory()
    dst_k = torch.zeros(8, item, device="cuda", dtype=dtype)
    dst_v = torch.zeros(8, item, device="cuda", dtype=dtype)
    src_indices = torch.tensor([1, 3, 5], dtype=torch.int64)
    dst_indices = torch.tensor([2, 4, 6], device="cuda", dtype=torch.int64)
    transfer_kv_per_layer(
        src_k,
        dst_k,
        src_v,
        dst_v,
        src_indices.to("cuda"),
        dst_indices,
        item_size=item * dtype.itemsize,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(dst_k[dst_indices], src_k[src_indices].to("cuda"))
    torch.testing.assert_close(dst_v[dst_indices], src_v[src_indices].to("cuda"))


def test_rmsnorm_registered_or_documented_fallback():
    if not _has_op("rmsnorm"):
        pytest.skip(
            "rmsnorm is not in the RDNA ROCm module; "
            "layernorm uses forward_native"
        )
    hidden = torch.randn(2, 32, device="cuda", dtype=torch.float16)
    weight = torch.ones(32, device="cuda", dtype=torch.float16)
    out = torch.empty_like(hidden)
    torch.ops.sgl_kernel.rmsnorm(out, hidden, weight, 1e-6)
    variance = hidden.float().pow(2).mean(-1, keepdim=True)
    ref = (hidden.float() * torch.rsqrt(variance + 1e-6) * weight.float()).to(
        hidden.dtype
    )
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
