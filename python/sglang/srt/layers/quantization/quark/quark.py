from __future__ import annotations
# SPDX-License-Identifier: Apache-2.0

import fnmatch
import logging
from typing import TYPE_CHECKING, Any, List, Optional, cast

import torch

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.quantization.base_config import (  # noqa: E501
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod
from sglang.srt.layers.quantization.quark.schemes import (
    QuarkLinearScheme,
    QuarkMoEScheme,
    QuarkW4A4MXFP4,
    QuarkW4A4MXFp4MoE,
    QuarkW8A8Fp8,
    QuarkW8A8FP8MoE,
)
from sglang.srt.layers.quantization.quark.utils import deep_compare, should_ignore_layer
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.utils import get_device_capability
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast
import re
import time

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput

__all__ = ["QuarkLinearMethod", "QuarkFusedMoEMethod"]

logger = logging.getLogger(__name__)


class QuarkConfig(QuantizationConfig):

    def __init__(
        self,
        quant_config: dict[str, Any],
        kv_cache_group: Optional[list[str]] = None,
        kv_cache_config: Optional[dict[str, Any]] = None,
        pack_method: str = "reorder",
    ):
        super().__init__()
        if kv_cache_group is None:
            kv_cache_group = []
        self.quant_config = quant_config
        self.kv_cache_group = kv_cache_group
        self.kv_cache_config = kv_cache_config
        self.pack_method = pack_method

        self.packed_modules_mapping = self.quant_config["packed_modules_mapping"]

    def get_linear_method(self) -> "QuarkLinearMethod":
        return QuarkLinearMethod(self)

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def get_name(self) -> str:
        return "quark"

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        # Check if the layer is skipped for quantization.
        exclude_layers = cast(list[str], self.quant_config.get("exclude"))
        if should_ignore_layer(
            prefix, ignore=exclude_layers, fused_mapping=self.packed_modules_mapping
        ):
            if isinstance(layer, LinearBase):
                return UnquantizedLinearMethod()
            elif isinstance(layer, RadixAttention):
                return QuarkKVCacheMethod(self)
            return None

        if isinstance(layer, LinearBase):
            scheme = self.get_linear_scheme(layer=layer, layer_name=prefix)
            layer.scheme = scheme
            return QuarkLinearMethod(self)

        if isinstance(layer, RadixAttention):
            return QuarkKVCacheMethod(self)

        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        if isinstance(layer, FusedMoE):
            layer.scheme = self.get_moe_scheme(layer, prefix)
            return QuarkFusedMoEMethod(self)

        return None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "QuarkConfig":
        export_config = config.get("export")
        if export_config is None:
            raise ValueError(
                "The export key should be included in "
                "the configurations of Quark quantized model"
            )

        kv_cache_group = cast(list[str], export_config.get("kv_cache_group"))
        pack_method = cast(str, export_config.get("pack_method"))

        # In the export model of quark, the quantization configuration
        # of kv_cache is stored in layer_quant_config. First, it is
        # judged whether kv_cache_group exists, and then it is judged
        # whether layer_quant_config has a quantization configuration
        # that matches kv_cache.
        if len(kv_cache_group) == 0:
            kv_cache_config = None
        else:
            kv_cache_set = set(kv_cache_group)
            layer_quant_config = cast(dict[str, Any], config.get("layer_quant_config"))
            layer_quant_names = list(layer_quant_config.keys())
            layer_quant_set = set(layer_quant_names)

            if not kv_cache_set.issubset(layer_quant_set):
                raise ValueError(
                    "The Quark quantized model has the "
                    "kv_cache_group parameter setting, "
                    "but no kv_cache quantization settings "
                    "were found in the quantization "
                    "configuration."
                )

            q_configs = [
                cast(dict[str, Any], layer_quant_config.get(name))
                for name in kv_cache_group
            ]
            if not all(deep_compare(q_config, q_configs[0]) for q_config in q_configs):
                raise ValueError(
                    "The quantization method used for kv_cache should "
                    "be the same, but the quantization method for the "
                    "kv_cache layer in the config is different."
                )
            kv_cache_config = q_configs[0].get("output_tensors")
            if kv_cache_config is None:
                raise ValueError("The kv_cache quantization configuration is empty.")

            # Since we have already set kv_cache quantization configurations,
            # we will remove the quantization configuration for the
            # output_tensors corresponding to the kv_cache layer.
            for q_config in q_configs:
                q_config["output_tensors"] = None

            # In case q_proj output is also quantized, remove the configuration
            # to keep qkv consistency.
            q_proj_q_config = cast(dict[str, Any], layer_quant_config.get("*q_proj"))
            if q_proj_q_config is not None:
                q_proj_q_config["output_tensors"] = None

        return cls(
            quant_config=config,
            kv_cache_group=kv_cache_group,
            kv_cache_config=kv_cache_config,
            pack_method=pack_method,
        )

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def _check_scheme_supported(self, min_capability: int, error: bool = True) -> bool:
        capability_tuple = get_device_capability()

        if capability_tuple is not None:
            assert 0 <= capability_tuple[1] < 10
            capability = capability_tuple[0] * 10 + capability_tuple[1]

            supported = capability >= min_capability
            if error and not supported:
                raise RuntimeError(
                    "Quantization scheme is not supported for ",
                    f"the current GPU. Min capability: {min_capability}. ",
                    f"Current capability: {capability}.",
                )
            return supported
        else:
            return False

    def _is_fp8_w8a8(
        self,
        weight_quant: Optional[dict[str, Any]],
        input_quant: Optional[dict[str, Any]],
    ) -> bool:
        # Confirm weights and input quantized.
        if weight_quant is None or input_quant is None:
            return False

        # Confirm weight scheme is supported
        is_fp8_dtype = (
            weight_quant.get("dtype") == "fp8_e4m3"
            and input_quant.get("dtype") == "fp8_e4m3"
        )
        is_static_weight = not weight_quant.get("is_dynamic")
        is_per_tensor_or_channel_weight = weight_quant.get("qscheme") in [
            "per_tensor",
            "per_channel",
        ]

        if not (is_fp8_dtype and is_static_weight and is_per_tensor_or_channel_weight):
            return False

        # Dynamic quantization is always supported if weights supported.
        if input_quant.get("is_dynamic"):
            return True

        # Confirm activation scheme is supported.
        is_per_tensor_activation = input_quant.get("qscheme") == "per_tensor"
        return is_per_tensor_activation

    def _is_mx_fp4(
        self,
        weight_quant: Optional[dict[str, Any]],
        input_quant: Optional[dict[str, Any]],
    ) -> bool:
        # Confirm weights and input quantized.
        if weight_quant is None or input_quant is None:
            logger.debug(
                "Quark model is not in MX-FP4 format: "
                "weight_quant or input_quant not set"
            )
            return False

        # Input and weight dtype needs to be fp4.
        if weight_quant.get("dtype") != "fp4" or input_quant.get("dtype") != "fp4":
            logger.debug("Quark model is not in MX-FP4 format: dtype not fp4")
            return False

        # Input and weight qscheme needs to be per group.
        if (
            weight_quant.get("qscheme") != "per_group"
            or input_quant.get("qscheme") != "per_group"
        ):
            logger.debug("Quark model is not in MX-FP4 format: not per_group")
            return False

        # Input and weight group size needs to be 32.
        if weight_quant.get("group_size") != 32 or input_quant.get("group_size") != 32:
            logger.debug("Quark model is not in MX-FP4 format: not group_size=32")
            return False

        # Weights need to use static quantization.
        if weight_quant.get("is_dynamic") is True:
            logger.debug("Quark model is not in MX-FP4 format: not weight static")
            return False

        # Activations need to use dynamic quantization.
        if input_quant.get("is_dynamic") is False:
            logger.debug("Quark model is not in MX-FP4 format: not activation dynamic")
            return False

        # Activations and weight scales need to be in e8m0 format.
        if (
            weight_quant.get("scale_format") != "e8m0"
            or input_quant.get("scale_format") != "e8m0"
        ):
            logger.debug("Quark model is not in MX-FP4 format: not scale_format e8m0")
            return False

        return True

    def _find_matched_config(
        self, layer_name: str, module: torch.nn.Module
    ) -> dict[str, Any]:

        proj_name = layer_name.split(".")[-1]
        if proj_name in self.packed_modules_mapping:
            shard_proj_names = self.packed_modules_mapping[proj_name]

            # Convert fused_name --> [shard_names]
            shard_names = [
                layer_name.replace(proj_name, shard_proj_name)
                for shard_proj_name in shard_proj_names
            ]
            shard_configs = [
                self._find_matched_config(shard_name, module)
                for shard_name in shard_names
            ]
            if not all(
                deep_compare(q_config, shard_configs[0]) for q_config in shard_configs
            ):
                raise ValueError(
                    f"Found a different quantization configuration for "
                    f"{shard_proj_names} in {layer_name}. vLLM "
                    "requires all to use the same scheme."
                )
            return shard_configs[0]
        else:
            layer_quant_config = cast(
                dict[str, Any], self.quant_config.get("layer_quant_config")
            )
            for name_pattern in layer_quant_config:
                if fnmatch.fnmatch(layer_name, name_pattern):
                    return layer_quant_config[name_pattern]

            layer_type = type(module).__name__
            layer_type_quant_config = cast(
                dict[str, Any], self.quant_config.get("layer_type_quant_config")
            )
            if layer_type in layer_type_quant_config:
                return layer_type_quant_config[layer_type]

            global_quant_config = cast(
                dict[str, Any], self.quant_config.get("global_quant_config")
            )
            return global_quant_config

    def _get_scheme_from_config(self, config: dict[str, Any]) -> "QuarkLinearScheme":
        if config.get("output_tensors") or config.get("bias"):
            raise NotImplementedError(
                "Currently, Quark models with output_tensors "
                "and bias quantized are not supported"
            )
        weight_config = cast(dict[str, Any], config.get("weight"))
        input_config = cast(dict[str, Any], config.get("input_tensors"))

        if self._is_mx_fp4(weight_config, input_config):
            return QuarkW4A4MXFP4(weight_config, input_config)
        if self._is_fp8_w8a8(weight_config, input_config):
            is_fp8_w8a8_supported = self._check_scheme_supported(
                QuarkW8A8Fp8.get_min_capability(), error=False
            )
            if is_fp8_w8a8_supported:
                return QuarkW8A8Fp8(weight_config, input_config)

        raise NotImplementedError(
            "No quark compatible scheme was found. "
            f"Weight config: {weight_config}, "
            f"Input config: {input_config}"
        )

    def get_linear_scheme(
        self, layer: torch.nn.Module, layer_name: str
    ) -> "QuarkLinearScheme":

        layer_quant_config = self._find_matched_config(layer_name, layer)

        # Find the quant_scheme
        scheme = self._get_scheme_from_config(layer_quant_config)

        # Raise error if device does not support the scheme
        # (e.g. fp8 needs ada lovelace)
        self._check_scheme_supported(scheme.get_min_capability())

        return scheme

    def get_moe_scheme(
        self,
        module: torch.nn.Module,
        layer_name: str,
    ) -> "QuarkMoEScheme":
        layer_quant_config = self._find_matched_config(layer_name, module)

        if layer_quant_config.get("output_tensors") or layer_quant_config.get("bias"):
            raise NotImplementedError(
                "Currently, Quark models with "
                "output_tensors and bias "
                "quantized are not supported"
            )
        weight_config = layer_quant_config.get("weight")
        input_config = layer_quant_config.get("input_tensors")

        if self._is_mx_fp4(weight_config, input_config):
            return QuarkW4A4MXFp4MoE(weight_config, input_config)
        elif self._is_fp8_w8a8(weight_config, input_config):
            return QuarkW8A8FP8MoE(weight_config, input_config)
        else:
            raise RuntimeError("Unsupported FusedMoe scheme")

    def get_scaled_act_names(self) -> List[str]:
        return []


class QuarkLinearMethod(LinearMethodBase):

    def __init__(self, quantization_config: QuarkConfig):
        self.quantization_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scheme.process_weights_after_loading(layer)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """
        Use the QuarkLinearScheme associated with the layer to create
        the necessary parameters for the layer. See LinearMethodBase for param
        details
        """
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.scheme.create_weights(
            layer=layer,
            input_size=input_size,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            output_size=output_size,
            params_dtype=params_dtype,
            weight_loader=weight_loader,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        """
        Use the output of create_weights and the QuarkLinearScheme
        associated with the layer to apply the forward pass with the
        layer input.  See LinearMethodBase for param details

        """
        scheme = layer.scheme
        if scheme is None:
            raise ValueError("A scheme must be defined for each layer")
        return scheme.apply_weights(layer, x, bias=bias)


class QuarkFusedMoEMethod(FusedMoEMethodBase):

    def __init__(self, quantization_config: QuarkConfig):
        self.quantization_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scheme.process_weights_after_loading(layer)

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """
        Use the QuarkMoEScheme associated with the layer to create
        the necessary parameters for the layer. See FusedMoEMethodBase for param
        details
        """
        layer.scheme.create_weights(
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        layer.scheme.create_moe_runner(layer, moe_runner_config)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ):
        """
        Use the output of create_weights and the QuarkMoEScheme
        associated with the layer to apply the forward pass with the
        fused MoE layer. See FusedMoEMethodBase for param details

        """
        scheme = layer.scheme
        if scheme is None:
            raise ValueError("A scheme must be defined for each layer")
        return scheme.apply_weights(layer, dispatch_output)


class QuarkKVCacheMethod(BaseKVCacheMethod):
    """
    Supports loading kv-cache scaling factors from quark checkpoints.
    """

    def __init__(self, quant_config: QuarkConfig):
        self.validate_kv_cache_config(quant_config.kv_cache_config)
        super().__init__(quant_config)

    @staticmethod
    def validate_kv_cache_config(kv_cache_config: Optional[dict[str, Any]]):
        """
        Validator for the kv cache configuration. Useful for controlling the
        kv cache quantization schemes, that are being supported in vLLM
        :param kv_cache_config: the quark kv cache scheme
        """
        if kv_cache_config is None:
            return

        dtype = kv_cache_config.get("dtype")
        if dtype != "fp8_e4m3":
            raise NotImplementedError(
                "Currently supported kv cache quantization is "
                f"dtype=fp8_e4m3, however received {dtype}"
            )

        qscheme = kv_cache_config.get("qscheme")
        if qscheme != "per_tensor":
            raise NotImplementedError(
                "Only support per-tensor scaling factor "
                "for quark KV cache. "
                f"Expected qscheme: per_tensor, found qscheme: {qscheme}"
            )


# --- imported with the qwen4 subsystem (sgl-project/sglang) ---


def _parse_nvfp4_excludes(hf_quant_config: Dict[str, Any]) -> List[str]:
    """Extract NVFP4 producer-declared excludes as `re:` patterns.

    Reads the producer-specific key:
      - `ignore`          - ModelOpt (config.json)
      - `exclude_modules` - ModelOpt hf_quant_config.json
      - `exclude`         - AMD Quark export

    Entries are usually fnmatch-style (literal strings work too), but ModelOpt
    `ignore` lists may already carry `re:`-prefixed regexes (e.g.
    `re:.*linear_attn\\.in_proj_a$`); those are passed through untouched.
    Wrapping an already-`re:` entry with another `re:` + `fnmatch.translate`
    yields a pattern that never matches, silently un-excluding the layer.
    Returns [] if no key present.
    """
    pats = (
        hf_quant_config.get("ignore")
        or hf_quant_config.get("exclude_modules")
        or hf_quant_config.get("exclude")
        or []
    )
    return [p if p.startswith("re:") else "re:" + fnmatch.translate(p) for p in pats]


def _detect_nvfp4_source(config: Dict[str, Any]) -> Optional["Nvfp4SourceConfig"]:
    """Return an Nvfp4SourceConfig if `config` (the checkpoint's
    quantization_config dict) describes a supported NVFP4 source, else None.

    Handles two producers:
      - ModelOpt:  quant_method in {modelopt, modelopt_fp4, nvfp4}
                   with quant_algo NVFP4/FP4 (or unspecified).
      - AMD Quark: quant_method == "quark". global_quant_config.weight is a
                   2-element list [fp4_per_group_gs16, fp8_e4m3_per_tensor].

    compressed-tensors NVFP4 is not supported at this time.
    """
    from sglang.srt.layers.quantization.quark.utils import Nvfp4SourceConfig

    quant_method = config.get("quant_method", "")
    quant_algo = (config.get("quant_algo") or "").upper()

    if quant_method in ("modelopt", "modelopt_fp4", "nvfp4") and quant_algo in (
        "",
        "NVFP4",
        "FP4",
    ):
        return Nvfp4SourceConfig()
    if quant_method == "quark":
        gqc = config.get("global_quant_config", {})
        weight = gqc.get("weight")
        if not (isinstance(weight, list) and len(weight) == 2):
            return None
        w0, w1 = weight
        is_nvfp4_weight = (
            isinstance(w0, dict)
            and w0.get("dtype") == "fp4"
            and w0.get("qscheme") == "per_group"
            and w0.get("group_size") == 16
            and not w0.get("is_dynamic")
        )
        is_nvfp4_scale_2 = (
            isinstance(w1, dict)
            and w1.get("dtype") == "fp8_e4m3"
            and w1.get("qscheme") == "per_tensor"
            and not w1.get("is_dynamic")
        )
        if is_nvfp4_weight and is_nvfp4_scale_2:
            return Nvfp4SourceConfig()
        return None
    if quant_method in ("compressed-tensors", "compressed_tensors"):
        raise NotImplementedError(
            "Online MXFP4 requantization from compressed-tensors NVFP4 "
            "checkpoints is not supported at this time."
        )
    return None


def _fp8_per_tensor_spec(is_dynamic_input: bool) -> Dict[str, Any]:
    return {
        "weight": {
            "dtype": "fp8_e4m3",
            "qscheme": "per_tensor",
            "is_dynamic": False,
        },
        "input_tensors": {
            "dtype": "fp8_e4m3",
            "qscheme": "per_tensor",
            "is_dynamic": is_dynamic_input,
        },
        "output_tensors": None,
        "bias": None,
    }


def _fp8_is_dynamic_from_config_groups(
    config_groups: Any,
) -> bool:
    """Return whether FP8 activation quantization is dynamic, from config_groups.

    Reads the `input_activations.dynamic` field of the first config_group whose
    `num_bits` is 8, and falls back to True (dynamic) when none exists or the
    format is not a recognised dict-of-dicts.
    """
    if not isinstance(config_groups, dict):
        return True
    for group in config_groups.values():
        if not isinstance(group, dict):
            continue
        input_act = group.get("input_activations") or {}
        if input_act.get("num_bits") == 8:
            return bool(input_act.get("dynamic", True))
    return True


def _mixed_precision_layer_map(config: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Return {layer_name: quant_algo} for a MIXED_PRECISION source, else None.

    Reads ModelOpt's per-layer `quantized_layers` map (from
    hf_quant_config.json or config.json's quantization_config). Only the
    quant_algo string per layer is needed;
    """
    if (config.get("quant_algo") or "").upper() != "MIXED_PRECISION":
        return None
    quantized_layers = config.get("quantized_layers")
    if not isinstance(quantized_layers, dict) or not quantized_layers:
        return None
    layer_map: Dict[str, str] = {}
    for name, info in quantized_layers.items():
        if isinstance(info, dict):
            layer_map[name] = str(info.get("quant_algo", "")).upper()
    return layer_map


def _build_mixed_precision_layer_quant_config(
    layer_map: Dict[str, str],
    config_groups: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], bool]:
    """Collapse a per-layer {name: quant_algo} map into a compact
    `layer_quant_config` keyed by fnmatch glob patterns.
    """
    # suffix tail -> set of algos seen (to detect inconsistency)
    tail_algos: Dict[str, set] = {}
    for name, algo in layer_map.items():
        # Suffix after the last `.layers.<idx>.` (or the whole name if
        # unindexed); this is the part shared across all layer indices.
        tail = re.split(r"\.layers\.\d+\.", name, maxsplit=1)[-1]
        tail_algos.setdefault(tail, set()).add(algo)

    fp8_is_dynamic = _fp8_is_dynamic_from_config_groups(config_groups or {})
    fp8_spec = _fp8_per_tensor_spec(is_dynamic_input=fp8_is_dynamic)

    layer_quant_config: Dict[str, Any] = {}
    has_nvfp4 = False
    for tail, algos in tail_algos.items():
        if len(algos) != 1:
            raise NotImplementedError(
                f"MIXED_PRECISION layer group {tail!r} has inconsistent "
                f"quant algos across layers: {sorted(algos)}. SGLang requires "
                "all layers in a group to share one algo."
            )
        algo = next(iter(algos))
        pattern = "*" + tail
        if algo in ("NVFP4", "W4A16_NVFP4"):
            layer_quant_config[pattern] = _MXFP4_TARGET_SPEC
            has_nvfp4 = True
        elif algo == "FP8":
            layer_quant_config[pattern] = fp8_spec
        else:
            raise NotImplementedError(
                f"MIXED_PRECISION layer group {tail!r} uses unsupported "
                f"quant algo {algo!r}; online requantization supports NVFP4 "
                "(-> MXFP4) and FP8 (kept as-is) only."
            )
    return layer_quant_config, has_nvfp4


def _build_excluded_fp8_config(config: Dict[str, Any]) -> Optional["Fp8Config"]:
    """Build a load-as-is `Fp8Config` for the excluded layers of a
    mixed-precision NVFP4 source, or None if excluded layers are bf16.

    Two producer conventions are handled:

    - FP8-serialized base (``quant_method == "fp8"``, e.g.
      DeepSeek-V4-Pro-NVFP4): the routed experts are NVFP4 (requantized to
      MXFP4) while attn / shared_experts stay FP8 and are listed in the
      excludes. Those FP8 layers load through `Fp8LinearMethod`;
      ``weight_block_size`` selects block (e.g. ``[128, 128]``) vs per-tensor
      (``None``), so a single config covers either granularity - and a
      checkpoint carrying only per-tensor or only block layers is handled
      without any per-layer probing.

    - ModelOpt mixed base (``quant_method`` in {modelopt, modelopt_mixed},
      e.g. Qwen3.5-397B-A17B-NVFP4-V2): FP8 layers are enumerated in the
      per-layer ``quantized_layers`` map (loaded via `QuarkW8A8Fp8`), not in
      the excludes, so the excludes are genuinely bf16 -> None.
    """
    if config.get("quant_method") != "fp8":
        return None
    # Fp8Config.from_config reads quant_method/activation_scheme/
    # weight_block_size/packed_modules_mapping straight off the checkpoint's
    # quantization_config dict, which is exactly what `config` carries here.
    return Fp8Config.from_config(config)
