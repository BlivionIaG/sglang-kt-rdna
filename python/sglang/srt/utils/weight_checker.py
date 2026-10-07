from __future__ import annotations
import logging
from typing import Dict, Iterable, Tuple

import torch

from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_dequant,
    inverse_transform_scale_ue8m0,
)
from typing import NamedTuple
from pydantic import BaseModel
from pydantic import ConfigDict
from typing import Optional
from typing import Set
from sglang.srt.mem_cache.storage.mmap.mmap_allocator import alloc_mmap
from sglang.srt.managers.mm_utils import tensor_hash

logger = logging.getLogger(__name__)


class WeightChecker:
    def __init__(self, model_runner):
        self._model_runner = model_runner
        self._snapshot_tensors = None

    def handle(self, action: str):
        logger.info(f"[WeightChecker] handle action={action}")
        if action == "snapshot":
            self._snapshot()
        elif action == "reset_tensors":
            self._reset_tensors()
        elif action == "compare":
            self._compare()
        else:
            raise Exception(f"Unsupported {action=}")

    def _snapshot(self):
        named_tensors = [
            (name, param.data.detach().cpu()) for name, param in self._model_state()
        ]
        self._snapshot_tensors = dict(named_tensors)
        assert len(self._snapshot_tensors) == len(
            named_tensors
        ), f"should not have duplicated tensor name"

    def _reset_tensors(self):
        for name, param in self._model_state():
            param.copy_(_random_like(param))

    def _compare(self):
        assert self._snapshot_tensors is not None

        _check_tensors(
            expect_tensors=_postprocess_tensors(self._snapshot_tensors),
            actual_tensors=_postprocess_tensors(dict(self._model_state())),
        )

    def _model_state(self):
        # TODO: support EAGLE etc (e.g. yield from both main model and draft model)
        yield from self._model_runner.model.named_parameters()
        yield from self._model_runner.model.named_buffers()


def _check_tensors(
    expect_tensors: Iterable[Tuple[str, bool, torch.Tensor]],
    actual_tensors: Iterable[Tuple[str, bool, torch.Tensor]],
):
    from sglang.srt.debug_utils.dumper import get_tensor_info

    good_names = []
    error_messages = []
    info_messages = []

    for (expect_name, expect_should_compare, expect), (
        actual_name,
        actual_should_compare,
        actual,
    ) in zip(expect_tensors, actual_tensors, strict=True):
        assert expect_name == actual_name, f"{expect_name=} {actual_name=}"
        assert (
            expect_should_compare == actual_should_compare
        ), f"{expect_should_compare=} {actual_should_compare=}"
        name = expect_name
        should_compare = expect_should_compare

        expect = expect.cuda()
        actual = actual.cuda()

        if torch.all(expect == actual):
            good_names.append(name)
        else:
            abs_diff = (actual.float() - expect.float()).abs()
            msg = (
                f"name={name} "
                f"max_abs_err={abs_diff.max()} "
                f"mean_abs_err={abs_diff.mean()} "
                f"{get_tensor_info(expect)=} "
                f"{get_tensor_info(actual)=} "
            )
            (error_messages if should_compare else info_messages).append(msg)

    logger.info(f"[check_tensors] equal tensors: {good_names}")
    if len(info_messages) > 0:
        logger.info(f"[check_tensors] info: {info_messages}")
    if len(error_messages) > 0:
        raise Exception(f"check tensor equality failed:\n" + "\n".join(error_messages))


def _random_like(t: torch.Tensor):
    device = t.device
    shape = t.shape
    dtype = t.dtype

    if dtype.is_floating_point or "float" in str(dtype):
        return torch.rand(shape, device=device, dtype=torch.float32).to(dtype)

    if dtype == torch.bool:
        return torch.rand(shape, device=device) > 0.5

    if dtype.is_complex:
        return torch.randn(shape, device=device, dtype=dtype)

    info = torch.iinfo(dtype)
    return torch.randint(
        low=int(info.min), high=int(info.max), size=shape, device=device, dtype=dtype
    )


def _postprocess_tensors(
    raw: Dict[str, torch.Tensor],
) -> Iterable[Tuple[str, bool, torch.Tensor]]:
    from sglang.srt.debug_utils.dumper import get_tensor_info

    skip_compare_names = [
        name
        for name in raw
        if any(pattern in name for pattern in ["attn_mqa.k_scale", "attn_mqa.v_scale"])
    ]
    skip_compare_names += [
        name
        for name in raw
        if any(pattern in name for pattern in ["freqs_cis", "cos_sin_cache"])
    ]

    # dequant fp8
    quant_names = [
        name
        for name in raw
        # Match: `something.weight`, `something.experts.w2_weight`
        if name.endswith("weight") and name.replace("weight", "weight_scale_inv") in raw
    ]
    quant_scale_names = [
        name.replace("weight", "weight_scale_inv") for name in quant_names
    ]
    skip_compare_names += quant_names
    skip_compare_names += quant_scale_names
    for name in quant_names:
        w_q = raw[name]
        w_s = raw[name.replace("weight", "weight_scale_inv")]

        try:
            if w_s.dtype == torch.int32:
                w_s_for_dequant = inverse_transform_scale_ue8m0(w_s, mn=w_q.shape[-2])
            else:
                w_s_for_dequant = w_s

            w_dequant = block_quant_dequant(
                w_q,
                w_s_for_dequant,
                # TODO do not hardcode
                block_size=[128, 128],
                dtype=torch.bfloat16,
            )
            yield name, True, w_dequant
        except Exception as e:
            e.add_note(
                f"when handling {name=} {get_tensor_info(w_q)=} {get_tensor_info(w_s)=}"
            )
            raise

    for name in raw:
        should_compare = name not in skip_compare_names
        yield name, should_compare, raw[name]


# --- imported with the qwen4 subsystem (sgl-project/sglang) ---


class _StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ParallelismInfo(_StrictBaseModel):
    # "target", or a draft role such as "draft" / "draft_step_0"
    role: str
    tp_rank: int
    tp_size: int
    dp_rank: int
    dp_size: int
    pp_rank: int
    pp_size: int
    rank: int
    size: int


class ChecksumInfo(_StrictBaseModel):
    checksums: Dict[str, str]
    per_gpu_checksum: str
    parallelism_info: ParallelismInfo


class CheckEntry(NamedTuple):
    name: str
    should_compare: bool
    comparable: ComparableWeight


class QuantizedWeight(NamedTuple):
    comparable_cls: type[ComparableWeight]
    scale_name: str
    is_shuffled: bool = False


def _is_non_persistent_buffer_name(name: str) -> bool:
    return any(pat in name for pat in _NON_PERSISTENT_BUFFER_PATTERNS)


def _is_skip_weight_check(name, param, skip_tensor_list=None) -> bool:
    # one skip set shared by reset / compare / checksum
    return (
        _is_non_persistent_buffer_name(name)
        or getattr(param, "_skip_weight_check", False)
        or any(pat in name for pat in (skip_tensor_list or ()))
    )


def overall_checksum(checksums: Dict[str, str]) -> str:
    h = hashlib.sha256()
    for name in sorted(checksums):
        h.update(name.encode())
        h.update(checksums[name].encode())
    return h.hexdigest()


def _padded(nbytes: int, align: int) -> int:
    return (nbytes + align - 1) // align * align


class _ArenaAllocator:
    """Bump-allocates aligned views out of one mmap arena, so the whole snapshot is one munmap."""

    def __init__(self, total_bytes: int, align: int):
        self.arena = alloc_mmap((max(total_bytes, align),), torch.uint8)
        self._align = align
        self._pointer = 0

    def allocate(self, like: torch.Tensor) -> torch.Tensor:
        start = self._pointer
        self._pointer += _padded(like.nbytes, self._align)
        assert self._pointer <= len(self.arena)
        return self.arena[start : start + like.nbytes].view(like.dtype).view(like.shape)


def _hash_tensor(t: torch.Tensor) -> str:
    return f"{tensor_hash(t):016x}"


def _build_quantized_set(model) -> Dict[str, QuantizedWeight]:
    """Run the router over the model: {weight_name: QuantizedWeight} for each
    quantized weight; weights absent from the set compare raw."""
    quantized_set = {}
    for module_name, module in model.named_modules():
        comparable_cls = select_comparable_weight(getattr(module, "quant_method", None))
        if comparable_cls is None:
            continue
        prefix = f"{module_name}." if module_name else ""
        own = dict(module.named_parameters(recurse=False))
        for name, parameter in own.items():
            scale = name.replace("weight", "weight_scale_inv")
            if name.endswith("weight") and scale in own:
                quantized_set[prefix + name] = QuantizedWeight(
                    comparable_cls,
                    prefix + scale,
                    getattr(parameter, "is_shuffled", False),
                )
    return quantized_set


def _build_check_entries(
    raw: Dict[str, torch.Tensor],
    skip_compare_names: Set[str],
    quantized_set: Optional[Dict[str, QuantizedWeight]] = None,
) -> Iterable[CheckEntry]:
    """Yields a CheckEntry per weight; quantized weights consume their scale, everything
    else is raw."""
    skip_compare_names = set(skip_compare_names)
    quantized_set = quantized_set or {}
    scale_names = {qw.scale_name for qw in quantized_set.values()}

    for name, tensor in raw.items():
        if name in scale_names:
            continue  # compared via its weight's comparable
        if name in quantized_set:
            qw = quantized_set[name]
            yield CheckEntry(
                name,
                True,
                qw.comparable_cls(
                    tensor, raw[qw.scale_name], is_shuffled=qw.is_shuffled
                ),
            )
        else:
            should_compare = name not in skip_compare_names and (
                not _is_non_persistent_buffer_name(name)
            )
            yield CheckEntry(name, should_compare, RawComparable(tensor))
