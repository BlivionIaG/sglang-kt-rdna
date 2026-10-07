from __future__ import annotations
from typing import Tuple, Union

import torch

from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.utils import (
    LinearAttnKernelBackend,
    get_linear_attn_decode_backend,
    get_linear_attn_prefill_backend,
)
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.utils import is_cpu, is_cuda, is_npu
from sglang.srt.utils.common import rank0_log
from typing import Optional, Tuple, Union
from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.srt.utils import is_cpu, is_cuda, is_hip, is_npu, is_xpu
import msgspec
from sglang.srt.configs.hybrid_arch import hybrid_gdn_config
from sglang.srt.runtime_context import get_exec, get_memory, get_schedule

if not is_cpu():
    from sglang.srt.layers.attention.fla.chunk_delta_h import (
        CHUNK_SIZE as FLA_CHUNK_SIZE,
    )

if is_cuda():
    from sglang.srt.layers.attention.mamba.causal_conv1d import (
        causal_conv1d_fn as causal_conv1d_fn_cuda,
    )

    causal_conv1d_fn = causal_conv1d_fn_cuda
elif is_npu():
    from sgl_kernel_npu.mamba.causal_conv1d import (
        causal_conv1d_fn_npu,
        causal_conv1d_update_npu,
    )

    causal_conv1d_fn = causal_conv1d_fn_npu
    causal_conv1d_update = causal_conv1d_update_npu
elif is_cpu():
    from sgl_kernel.mamba import causal_conv1d_fn_cpu, causal_conv1d_update_cpu

    causal_conv1d_fn = causal_conv1d_fn_cpu
    causal_conv1d_update = causal_conv1d_update_cpu
    fused_gdn_gating = torch.ops.sgl_kernel.fused_gdn_gating_cpu


class GDNKernelDispatcher:
    """Dispatches GDN kernel calls to the appropriate backend per mode."""

    def __init__(
        self,
        decode_backend: LinearAttnKernelBackend,
        prefill_backend: LinearAttnKernelBackend,
    ):
        triton_kernel = TritonGDNKernel()

        if decode_backend.is_triton():
            self.decode_kernel = triton_kernel
        elif decode_backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("CuTe DSL backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.gdn_cutedsl import (
                CuteDSLGDNKernel,
            )

            self.decode_kernel = CuteDSLGDNKernel()
        else:
            raise ValueError(f"Unsupported GDN decode backend: {decode_backend}")

        if prefill_backend.is_triton():
            self.extend_kernel = triton_kernel
        elif prefill_backend.is_cutedsl():
            raise ValueError(
                "CuTe DSL backend only supports decode, not prefill. "
                "Use --linear-attn-prefill-backend triton instead."
            )
        else:
            raise ValueError(f"Unsupported GDN prefill backend: {prefill_backend}")

        self.verify_kernel = triton_kernel

        rank0_log(
            f"GDN kernel dispatcher: decode={self.decode_kernel.__class__.__name__}, "
            f"extend={self.extend_kernel.__class__.__name__}, "
            f"verify={self.verify_kernel.__class__.__name__}"
        )

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.decode_kernel.decode(
            q,
            k,
            v,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> tuple:
        return self.extend_kernel.extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def target_verify(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.verify_kernel.target_verify(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )


class GDNAttnBackend(MambaAttnBackendBase):
    """Attention backend for GDN (Gated Delta Network) linear attention."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self.conv_states_shape = (
            model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape
        )
        if not is_cpu() and not is_npu():
            assert (
                self.conv_states_shape[-1] < FLA_CHUNK_SIZE
            ), f"{self.conv_states_shape[-1]=} should be less than {FLA_CHUNK_SIZE}"

        decode_backend = get_linear_attn_decode_backend()
        prefill_backend = get_linear_attn_prefill_backend()
        self.kernel_dispatcher = GDNKernelDispatcher(decode_backend, prefill_backend)

    def forward_decode(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        assert isinstance(mixed_qkv, torch.Tensor)
        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            layer.conv_weights,
            layer.bias,
            layer.activation,
            conv_state_indices=cache_indices,
        )

        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )
        # Reshape from [bs, h*d] to [1, bs, h, d]
        bs = forward_batch.batch_size
        query = query.view(1, bs, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, bs, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, bs, layer.num_v_heads, layer.head_v_dim)

        core_attn_out = self.kernel_dispatcher.decode(
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )

        self._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices
        )

        return core_attn_out

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        assert isinstance(mixed_qkv, torch.Tensor)
        seq_len = mixed_qkv.shape[0]

        is_target_verify = forward_batch.forward_mode.is_target_verify()
        forward_metadata = self.forward_metadata

        query_start_loc = forward_metadata.query_start_loc
        cache_indices = forward_metadata.mamba_cache_indices
        retrieve_next_token = forward_metadata.retrieve_next_token
        retrieve_next_sibling = forward_metadata.retrieve_next_sibling
        retrieve_parent_token = forward_metadata.retrieve_parent_token

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal
        if is_target_verify:
            assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)
            intermediate_state_cache = mamba_cache_params.intermediate_ssm
            intermediate_conv_window_cache = (
                mamba_cache_params.intermediate_conv_window[0]
            )
            has_initial_states = torch.ones(
                seq_len // forward_batch.spec_info.draft_token_num,
                dtype=torch.bool,
                device=forward_batch.input_ids.device,
            )
            intermediate_state_indices = torch.arange(
                cache_indices.shape[0], dtype=torch.int32, device=cache_indices.device
            )
        else:
            has_initial_states = forward_batch.extend_prefix_lens > 0

        if is_target_verify:
            batch_size = seq_len // forward_batch.spec_info.draft_token_num
            draft_token_num = forward_batch.spec_info.draft_token_num
            mixed_qkv_reshaped = mixed_qkv.view(
                batch_size, draft_token_num, -1
            ).transpose(1, 2)
            mixed_qkv_processed = causal_conv1d_update(
                mixed_qkv_reshaped,
                conv_states,
                layer.conv_weights,
                layer.bias,
                layer.activation,
                conv_state_indices=cache_indices[:batch_size],
                intermediate_conv_window=intermediate_conv_window_cache,
                intermediate_state_indices=intermediate_state_indices[:batch_size],
                retrieve_next_token=retrieve_next_token,
                retrieve_next_sibling=retrieve_next_sibling,
                retrieve_parent_token=retrieve_parent_token,
            )
            mixed_qkv = mixed_qkv_processed.transpose(1, 2).view(seq_len, -1)
        else:
            mixed_qkv = mixed_qkv.transpose(0, 1)
            if (
                forward_batch.mamba_track_mask is not None
                and forward_batch.mamba_track_mask.any()
            ):
                conv_dst = forward_batch.mamba_track_indices
                mixed_qkv_to_track = mixed_qkv[
                    :, forward_metadata.track_conv_indices
                ].transpose(0, 1)
                mask_indices = forward_batch.mamba_track_mask.nonzero(as_tuple=True)[0]
                conv_states[conv_dst[mask_indices]] = mixed_qkv_to_track

            mixed_qkv = causal_conv1d_fn(
                mixed_qkv,
                layer.conv_weights,
                layer.bias,
                activation=layer.activation,
                conv_states=conv_states,
                has_initial_state=has_initial_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            ).transpose(0, 1)[:seq_len]

        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )

        actual_seq_len = query.shape[0]
        query = query.view(1, actual_seq_len, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, actual_seq_len, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, actual_seq_len, layer.num_v_heads, layer.head_v_dim)

        g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)

        if is_target_verify:
            core_attn_out = self.kernel_dispatcher.target_verify(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                intermediate_states_buffer=intermediate_state_cache,
                intermediate_state_indices=intermediate_state_indices,
                cache_steps=forward_batch.spec_info.draft_token_num,
                retrieve_parent_token=retrieve_parent_token,
            )
        else:
            core_attn_out, last_recurrent_state, h = self.kernel_dispatcher.extend(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
            )
            if is_npu() or is_cpu():
                last_recurrent_state = last_recurrent_state.to(
                    ssm_states.dtype, copy=False
                )
                ssm_states[cache_indices] = last_recurrent_state

            self._track_mamba_state_extend(
                forward_batch, h, ssm_states, forward_metadata
            )

        return core_attn_out


# --- imported with the qwen4 subsystem (sgl-project/sglang) ---


class GDNMISMetadata(msgspec.Struct, frozen=True):
    query_token_indices: torch.Tensor
    query_cu_seqlens: torch.Tensor
    query_seq_lens_cpu: list[int]
    query_request_indices: torch.Tensor
    item_token_indices: torch.Tensor
    item_cu_seqlens: torch.Tensor
    item_seq_lens_cpu: list[int]
    item_request_indices: torch.Tensor


def build_gdn_mis_metadata(forward_batch: ForwardBatch) -> GDNMISMetadata:
    """Build compact query/item segments from request-local MIS delimiters."""
    if not forward_batch.is_prefill_only:
        raise ValueError("GDN MIS is only supported for prefill-only requests")

    prefix_lens = forward_batch.extend_prefix_lens_cpu
    if isinstance(prefix_lens, torch.Tensor):
        prefix_lens = prefix_lens.tolist()
    if any(int(prefix_len) != 0 for prefix_len in prefix_lens):
        raise ValueError("GDN MIS does not support cached prefixes")

    seq_lens = forward_batch.extend_seq_lens_cpu
    if isinstance(seq_lens, torch.Tensor):
        seq_lens = seq_lens.tolist()
    seq_lens = [int(seq_len) for seq_len in seq_lens]
    delimiter_indices = forward_batch.multi_item_delimiter_indices
    if delimiter_indices is None or len(delimiter_indices) != len(seq_lens):
        raise ValueError("GDN MIS requires delimiter indices for every request")
    if sum(seq_lens) > forward_batch.input_ids.numel():
        raise ValueError("GDN MIS sequence lengths exceed the input tokens")

    query_token_indices: list[int] = []
    query_seq_lens_cpu: list[int] = []
    query_request_indices: list[int] = []
    item_token_indices: list[int] = []
    item_seq_lens_cpu: list[int] = []
    item_request_indices: list[int] = []

    request_start = 0
    for request_idx, (seq_len, request_delimiters) in enumerate(
        zip(seq_lens, delimiter_indices)
    ):
        delimiters = [int(index) for index in request_delimiters.tolist()]
        if len(delimiters) < 2:
            raise ValueError("GDN MIS requires at least two delimiters per request")
        if any(
            current >= following
            for current, following in zip(delimiters, delimiters[1:])
        ):
            raise ValueError("GDN MIS delimiter indices must be strictly increasing")
        if delimiters[0] < 0 or delimiters[-1] >= seq_len:
            raise ValueError("GDN MIS delimiter index is outside the request")
        if delimiters[-1] != seq_len - 1:
            raise ValueError("GDN MIS final delimiter must be the last request token")

        query_len = delimiters[0]
        if query_len > 0:
            query_token_indices.extend(range(request_start, request_start + query_len))
            query_seq_lens_cpu.append(query_len)
            query_request_indices.append(request_idx)

        branch_ends = delimiters[1:] + [seq_len]
        for branch_start, branch_end in zip(delimiters, branch_ends):
            branch_len = branch_end - branch_start
            item_token_indices.extend(
                range(request_start + branch_start, request_start + branch_end)
            )
            item_seq_lens_cpu.append(branch_len)
            item_request_indices.append(request_idx)

        request_start += seq_len

    device = forward_batch.input_ids.device

    def _indices(values: list[int], dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(values, dtype=dtype, device=device)

    def _cu_seqlens(lengths: list[int]) -> torch.Tensor:
        result = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
        if lengths:
            result[1:] = torch.tensor(lengths, dtype=torch.int32, device=device).cumsum(
                dim=0
            )
        return result

    return GDNMISMetadata(
        query_token_indices=_indices(query_token_indices, torch.int64),
        query_cu_seqlens=_cu_seqlens(query_seq_lens_cpu),
        query_seq_lens_cpu=query_seq_lens_cpu,
        query_request_indices=_indices(query_request_indices, torch.int64),
        item_token_indices=_indices(item_token_indices, torch.int64),
        item_cu_seqlens=_cu_seqlens(item_seq_lens_cpu),
        item_seq_lens_cpu=item_seq_lens_cpu,
        item_request_indices=_indices(item_request_indices, torch.int64),
    )


def validate_gdn_mis_backend(prefill_backend: LinearAttnKernelBackend) -> None:
    if not get_exec().features.enable_mis:
        return
    if not prefill_backend.is_triton():
        raise ValueError(
            "GDN multi-item scoring requires the Triton linear-attention prefill "
            "backend. Set --linear-attn-prefill-backend triton."
        )
    if get_memory().enable_page_major_kv_layout:
        raise ValueError("GDN multi-item scoring does not support page-major layout")


def flashinfer_gdn_prefill_default(model_runner: ModelRunner) -> Optional[str]:
    """FlashInfer for the narrow SM90/SM100 GDN prefill domains we validated, else None."""
    sm_major = torch.cuda.get_device_capability()[0] if is_cuda() else 0
    if (
        get_exec().mamba.linear_attn_prefill_backend is not None
        or get_exec().mamba.linear_attn_backend != "triton"
        or get_exec().deterministic.enable_deterministic_inference
        or get_memory().enable_page_major_kv_layout
        or sm_major not in (9, 10)
    ):
        return None

    # SM100 runs the CUDA>=13 CuTe-DSL chunk kernel on a bf16 state pool;
    # SM90 runs the fused Hopper kernel on an fp32 state pool and tolerates
    # larger chunks. Everything outside these validated domains keeps Triton.
    cuda_version = torch.version.cuda
    if sm_major == 10:
        if cuda_version is None or int(cuda_version.split(".", 1)[0]) < 13:
            return None
        max_chunk = 8192
        expected_state_dtype = torch.bfloat16
    else:
        max_chunk = 32768
        expected_state_dtype = torch.float32

    chunk_size = get_schedule().chunked_prefill_size
    config = hybrid_gdn_config(model_runner.model_config)
    if (
        get_schedule().enable_dynamic_chunking
        or chunk_size is None
        or not 1 <= chunk_size <= max_chunk
        or getattr(config, "linear_key_head_dim", None) != 128
        or getattr(config, "linear_value_head_dim", None) != 128
        or model_runner.req_to_token_pool.mamba_pool.mamba_cache.temporal.dtype
        != expected_state_dtype
    ):
        return None

    from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
        is_flashinfer_gdn_prefill_available,
    )

    if not is_flashinfer_gdn_prefill_available():
        return None

    rank0_log(f"Defaulting SM{sm_major}0 GDN prefill backend to FlashInfer.")
    return "flashinfer"


def _validate_gdn_linear_attn_backends(backends: LinearAttnBackends) -> None:
    if (
        get_exec().deterministic.enable_deterministic_inference
        and backends.prefill.is_flashinfer()
    ):
        raise ValueError(
            "FlashInfer GDN prefill is not supported with "
            "--enable-deterministic-inference. Use "
            "--linear-attn-prefill-backend triton."
        )
