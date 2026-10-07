from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from sglang.srt.layers.amx_utils import (
    CPUQuantMethod,
    _amx_process_weight_after_loading,
)
from sglang.srt.layers.moe import (
    MoeRunner,
    MoeRunnerBackend,
    MoeRunnerConfig,
    get_moe_runner_backend,
)
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizeMethodBase,
)
from sglang.srt.layers.utils import MultiPlatformOp
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_hip,
    is_npu,
    next_power_of_2,
    set_weight_attrs,
    use_intel_amx_backend,
    use_intel_xpu_backend,
)
from enum import Enum
from sglang.srt.utils.custom_op import register_custom_op
from sglang.srt.environ import envs
from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
from sglang.srt.layers.moe.utils import xpu_moe_ld_padding_elems

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )


_is_cpu_amx_available = cpu_has_amx_support()
_is_hip = is_hip()
_is_cpu = is_cpu()
_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter import ActivationType
    from aiter.fused_moe import fused_moe
    from aiter.ops.shuffle import shuffle_weight

if _is_npu:
    from sglang.srt.hardware_backend.npu.utils import npu_format_cast

try:
    from flashinfer.fused_moe import cutlass_fused_moe as flashinfer_cutlass_fused_moe
except ImportError:
    flashinfer_cutlass_fused_moe = None


class UnquantizedEmbeddingMethod(QuantizeMethodBase):
    """Unquantized method for embeddings."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for embedding layer."""
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_, layer.weight)


class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _is_cpu and _is_cpu_amx_available:
            _amx_process_weight_after_loading(layer, ["weight"])

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if use_intel_amx_backend(layer):
            x_shapes = x.shape
            if len(x_shapes) == 3:
                x = x.view(-1, x.shape[-1])
            output = torch.ops.sgl_kernel.weight_packed_linear(
                x,
                layer.weight,
                bias,
                True,  # is_vnni
            )
            if len(x_shapes) == 3:
                output = output.view(x_shapes[0], x_shapes[1], -1)
            return output

        return F.linear(x, layer.weight, bias)


class UnquantizedFusedMoEMethod(FusedMoEMethodBase, MultiPlatformOp):
    """MoE method without quantization."""

    def __init__(
        self, use_triton_kernels: bool = False, use_flashinfer_trtllm_moe: bool = False
    ):
        super().__init__()
        self.use_flashinfer_cutlass = get_moe_runner_backend().is_flashinfer_cutlass()
        self.use_triton_kernels = use_triton_kernels
        self.with_bias = False
        self.use_flashinfer_trtllm_moe = use_flashinfer_trtllm_moe
        self._cache_permute_indices = dict({})

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        with_bias: bool = False,
        **extra_weight_attrs,
    ):
        self.with_bias = with_bias

        # Fused gate_up_proj (column parallel)
        w13_up_dim = (
            2 * intermediate_size_per_partition
            if layer.moe_runner_config.is_gated
            else intermediate_size_per_partition
        )
        w13_weight_n, w13_weight_k = (w13_up_dim, hidden_size)
        if self.use_triton_kernels:
            w13_weight_n, w13_weight_k = w13_weight_k, w13_weight_n
        w13_weight = torch.nn.Parameter(
            torch.empty(num_experts, w13_weight_n, w13_weight_k, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        if self.with_bias:
            w13_weight_bias = torch.nn.Parameter(
                torch.empty(num_experts, w13_up_dim, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_bias", w13_weight_bias)
            set_weight_attrs(w13_weight_bias, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight_n, w2_weight_k = (
            hidden_size,
            intermediate_size_per_partition,
        )
        if self.use_triton_kernels:
            w2_weight_n, w2_weight_k = w2_weight_k, w2_weight_n
        w2_weight = torch.nn.Parameter(
            torch.empty(num_experts, w2_weight_n, w2_weight_k, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        if self.with_bias:
            w2_weight_bias = torch.nn.Parameter(
                torch.empty(num_experts, hidden_size, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_bias", w2_weight_bias)
            set_weight_attrs(w2_weight_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Skip aiter weight shuffle when using non-auto MoE backend (e.g., triton, triton_kernels)
        # because aiter CK kernels don't support all GEMM dimensions
        _should_use_aiter_moe = _use_aiter and get_moe_runner_backend().is_auto()
        if _should_use_aiter_moe:
            layer.w13_weight = torch.nn.Parameter(
                shuffle_weight(layer.w13_weight.data, (16, 16)),
                requires_grad=False,
            )
            torch.cuda.empty_cache()
            layer.w2_weight = torch.nn.Parameter(
                shuffle_weight(layer.w2_weight.data, (16, 16)),
                requires_grad=False,
            )
            torch.cuda.empty_cache()

        # Pack weight for get better performance on CPU
        if _is_cpu and _is_cpu_amx_available:
            _amx_process_weight_after_loading(layer, ["w13_weight", "w2_weight"])

        # Reorder rows of W1 for fused gated activation
        if self.use_flashinfer_trtllm_moe:
            from flashinfer.fused_moe.core import (
                _maybe_get_cached_w3_w1_permute_indices,
                convert_to_block_layout,
                get_w2_permute_indices_with_cache,
            )

            # w1 and w3 have been swapped, so we don't need do that here
            epilogue_tile_m = 128
            block_k = 128
            old_shape_w13 = layer.w13_weight.data[0].shape
            old_shape_w2 = layer.w2_weight.data[0].shape
            new_shape_w13 = None
            new_shape_w2 = None
            for i in range(layer.num_local_experts):
                permute_indices = _maybe_get_cached_w3_w1_permute_indices(
                    self._cache_permute_indices,
                    layer.w13_weight.data[i].view(torch.uint8),
                    epilogue_tile_m,
                )
                tmp_weights1 = (
                    layer.w13_weight.data[i]
                    .clone()
                    .view(torch.uint8)[permute_indices.to(layer.w13_weight.data.device)]
                    .contiguous()
                )

                permute_indices = get_w2_permute_indices_with_cache(
                    self._cache_permute_indices,
                    layer.w2_weight.data[i].view(torch.uint8),
                    epilogue_tile_m,
                )
                tmp_weights2 = (
                    layer.w2_weight.data[i]
                    .clone()
                    .view(torch.uint8)[permute_indices.to(layer.w2_weight.data.device)]
                    .contiguous()
                )

                tmp_weights1 = convert_to_block_layout(
                    tmp_weights1.view(torch.uint8), block_k
                )
                tmp_weights2 = convert_to_block_layout(
                    tmp_weights2.view(torch.uint8), block_k
                )

                new_shape_w13 = tmp_weights1.view(torch.bfloat16).shape
                new_shape_w2 = tmp_weights2.view(torch.bfloat16).shape
                layer.w13_weight.data[i] = (
                    tmp_weights1.view(torch.bfloat16)
                    .contiguous()
                    .reshape(old_shape_w13)
                )
                layer.w2_weight.data[i] = (
                    tmp_weights2.view(torch.bfloat16).contiguous().reshape(old_shape_w2)
                )

            layer.w13_weight.data = layer.w13_weight.data.reshape(
                layer.num_local_experts, *new_shape_w13
            )
            layer.w2_weight.data = layer.w2_weight.data.reshape(
                layer.num_local_experts, *new_shape_w2
            )

        if _is_npu:
            for weight_name in ["w13_weight", "w2_weight"]:
                weight = getattr(layer, weight_name)
                weight.data = weight.data.transpose(1, 2)
                weight.data = npu_format_cast(
                    weight.data,
                )

        return

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        if self.use_flashinfer_trtllm_moe:
            backend = MoeRunnerBackend.FLASHINFER_TRTLLM
        elif self.use_triton_kernels:
            backend = MoeRunnerBackend.TRITON_KERNELS
        else:
            backend = MoeRunnerBackend.TRITON
        self.runner = MoeRunner(backend, moe_runner_config)

    @property
    def load_up_proj_weight_first(self) -> bool:
        # FlashInfer CUTLASS kernel assumes [Up, Gate] Proj as W13
        return self.use_flashinfer_cutlass

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        return self.forward(
            layer=layer,
            dispatch_output=dispatch_output,
        )

    def forward_cuda(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config

        backend = self.runner.runner_backend
        if backend.is_triton_kernels():
            from sglang.srt.layers.moe.moe_runner.triton_kernels import (
                TritonKernelsQuantInfo,
            )

            quant_info = TritonKernelsQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                w13_bias=getattr(layer, "w13_weight_bias", None),
                w2_bias=getattr(layer, "w2_weight_bias", None),
            )
            return self.runner.run(dispatch_output, quant_info)
        elif self.use_flashinfer_cutlass:
            output = flashinfer_cutlass_fused_moe(
                input=x,
                token_selected_experts=topk_output.topk_ids,
                token_final_scales=topk_output.topk_weights,
                fc1_expert_weights=layer.w13_weight,
                fc2_expert_weights=layer.w2_weight,
                output_dtype=x.dtype,
                quant_scales=None,
                ep_size=layer.moe_ep_size,
                ep_rank=layer.moe_ep_rank,
                tp_size=layer.moe_tp_size,
                tp_rank=layer.moe_tp_rank,
                tune_max_num_tokens=next_power_of_2(x.shape[0]),
            )[0]
            return StandardCombineInput(hidden_states=output)
        elif self.use_flashinfer_trtllm_moe:
            from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                FlashInferTrtllmBf16MoeQuantInfo,
            )

            quant_info = FlashInferTrtllmBf16MoeQuantInfo(
                gemm1_weights=layer.w13_weight,
                gemm2_weights=layer.w2_weight,
                global_num_experts=layer.num_experts,
                local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,
            )
            return self.runner.run(dispatch_output, quant_info)
        else:
            # Skip aiter fused_moe when using non-auto MoE backend (e.g., triton, triton_kernels)
            # because aiter CK kernels don't support all GEMM dimensions
            _should_use_aiter_moe = _use_aiter and get_moe_runner_backend().is_auto()
            if _should_use_aiter_moe:
                assert not moe_runner_config.no_combine, "unsupported"
                topk_weights, topk_ids, _ = topk_output
                if moe_runner_config.apply_router_weight_on_input:
                    assert (
                        topk_weights.dim() == 2
                    ), "`topk_weights` should be in shape (num_tokens, topk)"
                    _, topk = topk_weights.shape
                    assert (
                        topk == 1
                    ), "Only support topk=1 when `apply_router_weight_on_input` is True"
                    x = x * topk_weights.to(x.dtype)
                    topk_weights = torch.ones_like(
                        topk_weights, dtype=torch.float32
                    )  # topk_weights must be FP32 (float32)
                output = fused_moe(
                    x,
                    layer.w13_weight,
                    layer.w2_weight,
                    topk_weights,
                    topk_ids,
                    activation=(
                        ActivationType.Silu
                        if moe_runner_config.activation == "silu"
                        else ActivationType.Gelu
                    ),
                    expert_mask=layer.expert_mask_gpu,
                )
                return StandardCombineInput(hidden_states=output)
            else:
                quant_info = TritonMoeQuantInfo(
                    w13_weight=layer.w13_weight,
                    w2_weight=layer.w2_weight,
                    b13=getattr(layer, "w13_weight_bias", None),
                    b2=getattr(layer, "w2_weight_bias", None),
                )
                return self.runner.run(dispatch_output, quant_info)

    def forward_cpu(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config

        assert (
            moe_runner_config.activation == "silu"
        ), f"activation = {moe_runner_config.activation} is not supported."

        if use_intel_amx_backend(layer):
            from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

            topk_weights, topk_ids, _ = topk_output
            x, topk_weights = apply_topk_weights_cpu(
                moe_runner_config.apply_router_weight_on_input, topk_weights, x
            )
            output = torch.ops.sgl_kernel.fused_experts_cpu(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                False,  # inplace # See [Note] inplace should be False in fused_experts.
                CPUQuantMethod.UNQUANT,
                None,  # w1_scale
                None,  # w2_scale
                None,  # w1_zp
                None,  # w2_zp
                None,  # block_size
                True,  # is_vnni
            )
            return StandardCombineInput(hidden_states=output)
        else:
            from sglang.srt.layers.moe.fused_moe_native import moe_forward_native

            output = moe_forward_native(
                layer,
                x,
                topk_output,
                moe_runner_config,
            )
            return StandardCombineInput(hidden_states=output)

    def forward_xpu(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config
        assert moe_runner_config.activation in [
            "silu",
            "gelu",
        ], f"activation = {moe_runner_config.activation} is not supported."

        backend = self.runner.runner_backend
        if use_intel_xpu_backend():
            # sgl-kernel-xpu path
            from sgl_kernel import fused_experts

            topk_weights, topk_ids, _ = topk_output
            output = fused_experts(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                b1=getattr(layer, "w13_weight_bias", None),
                b2=getattr(layer, "w2_weight_bias", None),
                activation=moe_runner_config.activation,
            )
            return StandardCombineInput(hidden_states=output)
        else:
            assert backend.is_triton()
            assert (
                moe_runner_config.activation == "silu"
            ), f"activation = {moe_runner_config.activation} is not supported \
            for Triton PATH, please set ENV SGLANG_USE_SGL_XPU=1."

            quant_info = TritonMoeQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                b13=getattr(layer, "w13_weight_bias", None),
                b2=getattr(layer, "w2_weight_bias", None),
            )
            return self.runner.run(dispatch_output, quant_info)

    def forward_npu(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        # x.shape = [B*S, H]
        x = dispatch_output.hidden_states
        # topk_weights.shape = [B*S, K]; topk_ids.shape = [B*S, K]
        topk_weights, topk_ids, _ = dispatch_output.topk_output

        original_dtype = x.dtype
        num_tokens = x.shape[0]
        topk_weights = topk_weights.to(x.dtype)
        topk_ids = topk_ids.to(torch.int32)
        num_experts = layer.num_experts
        top_k = layer.top_k or topk_ids.shape[1]  # in case layer.top_k is not set

        hidden_states, expanded_row_idx, expert_tokens, _ = (
            torch.ops.npu.npu_moe_init_routing_v2(
                x,
                topk_ids,
                active_num=num_tokens * top_k,
                expert_num=num_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[0, num_experts],
                quant_mode=-1,
            )
        )
        expert_tokens = expert_tokens.to(torch.int64)
        w13_bias = [layer.w13_weight_bias] if self.with_bias else None
        w2_bias = [layer.w2_weight_bias] if self.with_bias else None

        # gmm1: gate_up_proj
        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w13_weight],
            bias=w13_bias,
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=expert_tokens,
            output_dtype=original_dtype,
        )[0]

        # act_fn:
        if self.moe_runner_config.activation == "npu_swiglu_oai":
            from sgl_kernel_npu.activation.swiglu_oai import swiglu_oai

            hidden_states = swiglu_oai(layer, hidden_states)
        elif self.moe_runner_config.activation == "silu":
            hidden_states = torch.ops.npu.npu_swiglu(hidden_states)
        else:
            from sglang.srt.layers.activation import GeluAndMul

            hidden_states = GeluAndMul()(hidden_states)

        # gmm2: down_proj
        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w2_weight],
            bias=w2_bias,
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=expert_tokens,
            output_dtype=original_dtype,
        )[0]

        final_hidden_states = torch.ops.npu.npu_moe_finalize_routing(
            hidden_states,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=topk_ids,
            drop_pad_mode=2,
        )

        return StandardCombineInput(hidden_states=final_hidden_states)

    def forward_tpu(self, *args, **kwargs) -> CombineInput:
        raise NotImplementedError("The TPU backend currently does not support MoE.")

    forward_native = forward_cpu


# --- imported with the qwen4 subsystem (sgl-project/sglang) ---


def _prefer_triton_a16w16(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Return whether the Triton BF16 path supports this decode shape."""
    if x.dim() != 2 or weight.dim() != 2:
        return False
    k = weight.shape[1]
    max_m = (
        _A16W16_TRITON_NARROW_K_MAX_M
        if k <= _A16W16_TRITON_NARROW_K
        else _A16W16_TRITON_MAX_M
    )
    return (
        x.is_cuda
        and weight.is_cuda
        and x.device == weight.device
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.shape[1] == k
        and 0 < k <= _A16W16_TRITON_MAX_K
        and 0 < x.shape[0] <= max_m
        and weight.shape[0] > 0
        and x.is_contiguous()
        and weight.is_contiguous()
    )


class Bf16GemmBackend(Enum):
    AUTO = "auto"
    CUTEDSL = "cutedsl"
    GEMV = "gemv"
    TORCH = "torch"

    def is_auto(self) -> bool:
        return self == Bf16GemmBackend.AUTO

    def is_cutedsl(self) -> bool:
        return self == Bf16GemmBackend.CUTEDSL

    def is_gemv(self) -> bool:
        return self == Bf16GemmBackend.GEMV


def use_bf16_splitk_gemm(m: int, n: int, k: int) -> bool:
    return (m, n, k) in _BF16_SPLITK_TUNED_TACTICS


def precompile_splitk_tactics() -> bool:
    """JIT-compile every tuned tactic through the real dispatch,
    so CUDA graph capture never hits a cold kernel."""
    if not _enable_bf16_splitk_gemm:
        return False
    device = torch.cuda.current_device()
    for m, n, k in _BF16_SPLITK_TUNED_TACTICS:
        x = torch.zeros(m, k, dtype=torch.bfloat16, device=device)
        weight = torch.zeros(n, k, dtype=torch.bfloat16, device=device)
        out = torch.empty(m, n, dtype=torch.bfloat16, device=device)
        _bf16_splitk_gemm_out(x, weight, None, out)
    torch.cuda.synchronize()
    return True


def should_enable_bf16_splitk_gemm(backend: Bf16GemmBackend) -> bool:
    """Return whether the optional Split-K path should be initialized."""
    return backend.is_cutedsl() and envs.SGLANG_ENABLE_BF16_SPLITK_GEMM.get()


def initialize_bf16_gemm_config() -> None:
    global _BF16_GEMM_BACKEND
    global _cutedsl_bf16_gemm, _use_cutedsl_bf16_gemm
    global _splitk_tactic
    global _run_splitk_dense
    global _direct_default_tactic
    global _prefer_direct
    global _run_direct_dense
    global _enable_bf16_splitk_gemm

    backend_str = get_exec().kernel.bf16_gemm_backend
    if backend_str == "auto" and get_platform().is_sm100:
        backend_str = (
            "torch"
            if get_exec().deterministic.enable_deterministic_inference
            else "cutedsl"
        )

    backend = Bf16GemmBackend(backend_str)

    if backend.is_gemv():
        if torch.cuda.get_device_capability()[0] != 9:
            raise ValueError("--bf16-gemm-backend gemv requires SM90 (Hopper)")

        global _hopper_bf16_gemv, _use_hopper_bf16_gemv
        from sglang.kernels.ops.gemm.hopper_bf16_gemv import (
            hopper_bf16_gemv,
            use_hopper_bf16_gemv,
        )

        _hopper_bf16_gemv = hopper_bf16_gemv
        _use_hopper_bf16_gemv = use_hopper_bf16_gemv
    elif backend.is_cutedsl():
        if get_exec().deterministic.enable_deterministic_inference:
            raise ValueError(
                "--bf16-gemm-backend cutedsl is batch-size dependent and cannot "
                "be combined with --enable-deterministic-inference"
            )
        if not get_platform().is_sm100:
            raise ValueError(
                f"--bf16-gemm-backend {backend.value} requires SM100/SM103 (Blackwell)"
            )

        from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import (
            cutedsl_bf16_gemm,
            use_cutedsl_bf16_gemm,
        )

        _cutedsl_bf16_gemm = cutedsl_bf16_gemm
        _use_cutedsl_bf16_gemm = use_cutedsl_bf16_gemm

    _enable_bf16_splitk_gemm = False
    if should_enable_bf16_splitk_gemm(backend):
        from flashinfer.gemm.kernels.dense_bf16_gemm_direct import (
            default_tactic,
            prefer_direct_bf16_gemm_sm100,
            run_direct_dense,
        )
        from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import (
            SplitKTactic,
            run_splitk_dense,
        )

        _splitk_tactic = SplitKTactic
        _run_splitk_dense = run_splitk_dense
        _direct_default_tactic = default_tactic
        _prefer_direct = prefer_direct_bf16_gemm_sm100
        _run_direct_dense = run_direct_dense
        _enable_bf16_splitk_gemm = True

    _BF16_GEMM_BACKEND = backend


def _bf16_gemm_dispatch_fake(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def _bf16_splitk_gemm_out(
    x_2d: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
) -> torch.Tensor:
    m, n, k = x_2d.shape[0], weight.shape[0], weight.shape[1]
    if bias is None and _prefer_direct(m, n, k):
        tactic = _direct_default_tactic(m, n, k)
        _run_direct_dense(x_2d, weight.T, out, True, tactic)
    else:
        tactic = _splitk_tactic(*_BF16_SPLITK_TUNED_TACTICS[(m, n, k)])
        _run_splitk_dense(
            x_2d,
            weight.T,
            bias,
            out,
            True,
            tactic,
        )
    return out


def _bf16_splitk_gemm(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    x_2d = x.view(-1, x.shape[-1])
    out = torch.empty((x_2d.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    _bf16_splitk_gemm_out(x_2d, weight, bias, out)
    return out.view(*x.shape[:-1], weight.shape[0])


def _bf16_gemm_dispatch_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    addend: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    m = x.numel() // x.shape[-1]
    if _enable_bf16_splitk_gemm and use_bf16_splitk_gemm(
        m, weight.shape[0], weight.shape[1]
    ):
        output = _bf16_splitk_gemm(x, weight, bias)
    elif (
        _use_hopper_bf16_gemv is not None
        and bias is None
        and _use_hopper_bf16_gemv(m, weight.shape[0], weight.shape[1])
    ):
        output = _hopper_bf16_gemv(x.view(-1, x.shape[-1]), weight).view(
            *x.shape[:-1], -1
        )
    elif _use_cutedsl_bf16_gemm is not None and _use_cutedsl_bf16_gemm(
        m, weight.shape[0], weight.shape[1]
    ):
        output = _cutedsl_bf16_gemm(x.view(-1, x.shape[-1]), weight, bias).view(
            *x.shape[:-1], -1
        )
    elif addend is not None:
        # cuBLAS folds the addend in through the GEMM beta input;
        # a bias would need a third operand, so callers must exclude it.
        assert bias is None
        return torch.addmm(addend, x, weight.t(), out=addend)
    else:
        return F.linear(x, weight, bias)

    if addend is not None:
        output.add_(addend)
    return output


@register_custom_op(fake_impl=_bf16_gemm_dispatch_fake)
def bf16_gemm_dispatch(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    return _bf16_gemm_dispatch_impl(x, weight, bias)


def _can_accumulate_into_addend(
    *,
    weight: torch.Tensor,
    x: torch.Tensor,
    addend: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> bool:
    if not _is_cuda or torch.compiler.is_compiling():
        return False
    # Batch-invariant mode overrides aten::mm and aten::addmm,
    # but not aten::addmm.out, so deterministic inference keeps a separate add.
    if is_batch_invariant_mode_enabled():
        return False
    # x.is_cuda also keeps the CPU AMX route in apply().
    if bias is not None or x.ndim != 2 or not x.is_cuda:
        return False
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return False
    return (
        addend.dtype == torch.bfloat16
        and addend.is_contiguous()
        and addend.shape == (x.shape[0], weight.shape[0])
        and not (x.requires_grad or addend.requires_grad or weight.requires_grad)
    )


def get_bf16_gemm_backend() -> Bf16GemmBackend:
    global _BF16_GEMM_BACKEND
    if _BF16_GEMM_BACKEND is None:
        _BF16_GEMM_BACKEND = Bf16GemmBackend.AUTO
    return _BF16_GEMM_BACKEND


def _use_xpu_moe_ld_padding(use_triton_kernels: bool) -> bool:
    """Whether MoE expert weights should get a padded row stride for XPU.

    is_xpu() only tells us an XPU exists on this machine, not that the weights
    being created land on it -- this can be true while serving on CPU/CUDA.
    create_weights takes no device argument and allocates under the model
    loader's ambient device context, so check that context too: padding a
    non-XPU weight would make it non-contiguous for no benefit, and other
    backends' MoE kernels expect contiguous expert tensors.

    The Triton path stores B transposed and does not read a row stride, so it
    is excluded even on XPU (either via --moe-runner-backend triton or the
    triton_kernels build).
    """
    return (
        is_xpu()
        and not get_moe_runner_backend().is_triton()
        and torch.get_default_device().type == "xpu"
        and not use_triton_kernels
    )


def _empty_xpu_moe_expert_weight(
    num_experts: int,
    n_dim: int,
    k_dim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Allocate an [E, N, K] XPU expert weight, over-allocating K when padding
    its row stride would avoid L3 set aliasing.

    Some K dims (3072, 7168 in bf16) put every weight row in the same handful
    of L3 sets, which throttles the grouped GEMM's B loads. Over-allocating K
    and returning a narrowed view keeps the logical [E, N, K] shape (so the
    weight loader is unchanged) while giving the rows a non-aliasing stride.
    The Xe20 grouped GEMM reads B's row stride from the tensor, so the padding
    is transparent to it.

    Callers must have checked _use_xpu_moe_ld_padding() first. K dims that are
    already well distributed get no padding and allocate normally.
    """
    pad = xpu_moe_ld_padding_elems(k_dim, dtype.itemsize)
    if pad == 0:
        return torch.empty(num_experts, n_dim, k_dim, dtype=dtype)
    # The view is non-contiguous; only the K slice is ever read or written.
    return torch.empty(num_experts, n_dim, k_dim + pad, dtype=dtype)[:, :, :k_dim]
