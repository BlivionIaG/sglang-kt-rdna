from __future__ import annotations
import bisect
import logging
import math
import os
from contextlib import contextmanager
from enum import IntEnum
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

import sglang.srt.distributed.device_communicators.custom_all_reduce_ops as ops
from sglang.srt.utils import is_hip
from typing import Any, Callable, ClassVar, Optional, Union
from dataclasses import replace as dataclass_replace
import msgspec

logger = logging.getLogger(__name__)

_is_hip = is_hip()


class MscclContextSelection(IntEnum):
    MSCCL1SHOT1NODELL = 1
    MSCCL1SHOT2NODELL = 2


def mscclpp_is_weak_contiguous(inp: torch.Tensor):
    return inp.is_contiguous() or (
        inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )


def mscclpp_convert_to_bytes(size_str):
    """
    Converts a human-readable size string (e.g., "1MB", "2.5kb", "3 GB")
    into the equivalent number of bytes using binary units.

    Args:
        size_str (str): A string representing size with unit (KB, MB, GB).

    Returns:
        int: Number of bytes.
    """
    size_str = size_str.strip().lower()

    if not size_str:
        raise ValueError("Empty input string")

    # Extract numeric part and unit
    for i in range(len(size_str)):
        if not size_str[i].isdigit() and size_str[i] != ".":
            break
    num_str = size_str[:i]
    unit = size_str[i:].strip()

    try:
        num = float(num_str)
    except ValueError:
        raise ValueError(f"Invalid numeric value in '{size_str}'")

    # Conversion factors
    if unit == "b":
        return int(num)
    elif unit == "kb":
        return int(num * 1024)
    elif unit == "mb":
        return int(num * 1024 * 1024)
    elif unit == "gb":
        return int(num * 1024 * 1024 * 1024)
    else:
        raise ValueError(f"Unsupported unit: {unit}, support B, KB, MB, GB only")


def mscclpp_bench_time(func, test_niter: int = 10, warmup_niter: int = 2):
    # warmup
    for _ in range(warmup_niter):
        func()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    dist.barrier()
    start_event.record()
    for _ in range(test_niter):
        func()
    end_event.record()
    end_event.synchronize()
    func_cost_us = start_event.elapsed_time(end_event) / test_niter * 1000
    return func_cost_us


class PyMscclppCommunicator:
    _SUPPORTED_WORLD_SIZES = [8, 16]
    _MAX_BYTES = mscclpp_convert_to_bytes(os.getenv("SGLANG_MSCCLPP_MAX_BYTES", "1MB"))
    _SUPPORTED_DTYPE = [torch.float, torch.float16, torch.bfloat16]

    # max_bytes: max supported mscclpp allreduce size
    # in A100 mscclpp is faster than nccl only under condition of msg size smaller than1MB
    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
        max_bytes=_MAX_BYTES,
    ) -> None:
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the CustomAllreduce to. If None,
                it will be bind to f"cuda:{local_rank}".
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device, and all communicators in this group
        are in the same node.
        """
        self._IS_CAPTURING = False
        self.disabled = True

        if not ops.IS_MSCCLPP_AR_AVAILABLE:
            # disable because of missing mscclpp library
            # e.g. in a non-cuda environment
            return

        self.group = group

        assert (
            dist.get_backend(group) != dist.Backend.NCCL
        ), "CustomAllreduce should be attached to a non-NCCL group."

        rank = dist.get_rank(group=self.group)
        world_size = dist.get_world_size(group=self.group)
        if world_size == 1:
            # No need to initialize mscclpp for single GPU case.
            return

        if world_size not in PyMscclppCommunicator._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "PyMscclpp is disabled due to an unsupported world"
                " size: %d. Supported world sizes: %s. To silence this "
                "warning, specify disable_mscclpp=True explicitly.",
                world_size,
                str(PyMscclppCommunicator._SUPPORTED_WORLD_SIZES),
            )
            return

        self.ranks = torch.distributed.get_process_group_ranks(group)
        self.nranks_per_node = torch.cuda.device_count()
        # for now mscclpp with stride in the communicator is not tested
        if not (abs(self.ranks[-1] - self.ranks[0]) == world_size - 1):
            logger.warning(
                "PyMscclpp is disabled due to an unsupported group %s."
                "Please ensure all ranks in the group are consecutive."
                "To silence this warning, specify disable_mscclpp=True explicitly.",
                str(self.ranks),
            )
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device

        self.max_bytes = max_bytes
        self.rank = rank
        self.world_size = world_size

        if dist.get_rank(group) == 0:
            unique_id = [ops.mscclpp_generate_unique_id()]
        else:
            unique_id = [None]
        dist.broadcast_object_list(unique_id, src=self.ranks[0], group=self.group)
        self.unique_id = unique_id[0]
        self.rank_to_node, self.rank_to_ib = list(range(world_size)), list(
            range(world_size)
        )
        for r in range(world_size):
            self.rank_to_node[r] = r // 8
            self.rank_to_ib[r] = self.rank % 8

        self._context = None
        self.context_selection = None
        self.msg_size_for_finetune = [
            2**i for i in range(10, math.floor(math.log2(self.max_bytes)) + 1)
        ]
        self.msg_size2best_config = {}
        if world_size == 8:
            self.context_selection = MscclContextSelection.MSCCL1SHOT1NODELL
        elif world_size == 16:
            self.context_selection = MscclContextSelection.MSCCL1SHOT2NODELL
        if not _is_hip:
            self.scratch = torch.empty(
                self.max_bytes * 8,
                dtype=torch.uint8,
                device=self.device,
            )
            self.put_buffer = torch.empty(
                self.max_bytes * 8 // self.nranks_per_node,
                dtype=torch.uint8,
                device=self.device,
            )
            self._context = ops.mscclpp_init_context(
                self.unique_id,
                self.rank,
                self.world_size,
                self.scratch,
                self.put_buffer,
                self.nranks_per_node,
                self.rank_to_node,
                self.rank_to_ib,
                int(self.context_selection),
            )
        else:
            raise NotImplementedError("HIP Mscclpp is not supported yet.")

        self.msg_size2best_config = {}
        self.pre_tune_config()
        if dist.get_rank(group) == 0:
            msg_size2best_config = [self.msg_size2best_config]
        else:
            msg_size2best_config = [None]
        dist.broadcast_object_list(
            msg_size2best_config, src=self.ranks[0], group=self.group
        )
        self.msg_size2best_config = msg_size2best_config[0]

        # PyMscclpp is enabled only in cuda graph
        self.disabled = True

    def pre_tune_config(self, dtype=torch.bfloat16) -> bool:
        logger.debug(f"start to pre-tune configs for rank {self.rank}")
        nthreads_to_try = [256, 512, 1024]
        nblocks_to_try = [21, 42, 84]
        inp_randn = torch.ones(
            self.msg_size_for_finetune[-1] // dtype.itemsize, dtype=dtype, device="cuda"
        )
        oup_randn = torch.empty_like(inp_randn)
        for msg_size in self.msg_size_for_finetune:
            mock_inp, mock_outp = (
                inp_randn[: msg_size // dtype.itemsize],
                oup_randn[: msg_size // dtype.itemsize],
            )
            best_config, best_time = None, None
            for nthreads in nthreads_to_try:
                for nblocks in nblocks_to_try:
                    cur_cost = mscclpp_bench_time(
                        lambda: ops.mscclpp_allreduce(
                            self._context, mock_inp, mock_outp, nthreads, nblocks
                        )
                    )
                    if best_time is None or cur_cost < best_time:
                        best_config = (nthreads, nblocks)
                        best_time = cur_cost
            self.msg_size2best_config[msg_size] = best_config
            if self.rank == 0:
                logger.debug(
                    f"for msg_size {msg_size}, best_config: {best_config}, best_time: {best_time}us"
                )

    def should_mscclpp_allreduce(
        self, inp: torch.Tensor, op: ReduceOp = ReduceOp.SUM
    ) -> bool:
        if self.disabled or self._context is None:
            return False
        if inp.dtype not in PyMscclppCommunicator._SUPPORTED_DTYPE:
            return False
        if not mscclpp_is_weak_contiguous(inp):
            return False
        # only support sum op
        if op != ReduceOp.SUM:
            return False
        if inp.numel() * inp.element_size() > self.max_bytes:
            return False
        return True

    def all_reduce(self, tensor: torch.Tensor, op: ReduceOp = ReduceOp.SUM):
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                self.graph_input_set.add((tensor.dtype, tensor.numel()))
        msg_size = tensor.numel() * tensor.itemsize
        index = bisect.bisect_left(self.msg_size_for_finetune, msg_size)
        msg_size_finetune = self.msg_size_for_finetune[index]
        nthreads, nblocks = self.msg_size2best_config[msg_size_finetune]
        result = torch.empty_like(tensor)
        ops.mscclpp_allreduce(self._context, tensor, result, nthreads, nblocks)
        return result

    @contextmanager
    def change_state(
        self,
        enable: Optional[bool] = None,
    ):
        if enable is None:
            # guess a default value when not specified
            enable = self.available

        old_disable = self.disabled
        self.disabled = not enable

        yield

        self.disabled = old_disable


# --- imported with the qwen4 subsystem (sgl-project/sglang) ---


class _MessageSizeRange(msgspec.Struct, frozen=True):
    minimum: int
    maximum: int

    def __post_init__(self):
        if self.minimum < 0 or self.maximum < self.minimum:
            raise ValueError(f"Invalid message size range: {self}")

    def contains(self, size: int) -> bool:
        return self.minimum <= size <= self.maximum


class _AlgorithmConfig(msgspec.Struct, frozen=True, kw_only=True):
    implementation: ClassVar[str]
    name: str
    collective: str
    world_sizes: tuple[int, ...]
    ipc_domain_counts: tuple[int, ...]
    threads_per_block: tuple[int, ...]
    message_size_range: _MessageSizeRange
    reduce_op: str
    parameter_adapter: Optional[_ParameterAdapter] = None
    supported_dtypes: tuple[torch.dtype, ...] = _DEFAULT_SUPPORTED_DTYPES
    in_place: bool = True
    requires_nvls: bool = False
    requires_symmetric_memory: bool = False
    algorithm: Any = None

    def __post_init__(self):
        if self.collective not in _SUPPORTED_COLLECTIVES:
            raise ValueError(f"Unsupported collective: {self.collective}")
        if self.reduce_op not in {"SUM", "NOP"}:
            raise ValueError(f"Unsupported reduction operation: {self.reduce_op}")
        if not self.world_sizes or not self.ipc_domain_counts:
            raise ValueError("Algorithm topology constraints cannot be empty")
        if not self.threads_per_block:
            raise ValueError("Algorithm topology constraints cannot be empty")
        if not self.supported_dtypes:
            raise ValueError("Algorithm supported dtypes cannot be empty")

    def supports_topology(self, world_size: int, ipc_domain_count: int) -> bool:
        return (
            world_size in self.world_sizes
            and ipc_domain_count in self.ipc_domain_counts
        )

    def requirements_satisfied(
        self,
        world_size: int,
        ipc_domain_count: int,
        nvls_supported: bool,
        symmetric_memory: bool,
    ) -> bool:
        return (
            self.supports_topology(world_size, ipc_domain_count)
            and self.support_nvls(nvls_supported)
            and self.support_symmetric_memory(symmetric_memory)
        )

    def support_nvls(self, nvls_supported: bool) -> bool:
        return not self.requires_nvls or nvls_supported

    def support_symmetric_memory(self, symmetric_memory: bool) -> bool:
        return not self.requires_symmetric_memory or symmetric_memory

    def supports_message_size(self, message_size: int) -> bool:
        return self.message_size_range.contains(message_size)

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        return dtype in self.supported_dtypes

    def resolve_reduce_op(self, reduce_ops):
        return getattr(reduce_ops, self.reduce_op)

    def adapt_to_topology(
        self, world_size: int, nranks_per_ipc_domain: int
    ) -> "_AlgorithmConfig":
        if self.parameter_adapter is None:
            return self
        return self.parameter_adapter(
            self,
            world_size=world_size,
            nranks_per_ipc_domain=nranks_per_ipc_domain,
        )

    def input_buffer_size(self, message_size: int, world_size: int) -> int:
        if self.collective == "allgather":
            if message_size % world_size != 0:
                raise ValueError(
                    f"All-gather output size {message_size} must be divisible "
                    f"by world size {world_size}"
                )
            return message_size // world_size
        return message_size

    def output_buffer_size(self, message_size: int, world_size: int) -> int:
        if self.collective == "reducescatter":
            if message_size % world_size != 0:
                raise ValueError(
                    f"Reduce-scatter input size {message_size} must be divisible "
                    f"by world size {world_size}"
                )
            return message_size // world_size
        return message_size

    def tuning_launches(self) -> tuple[tuple[int, int], ...]:
        raise NotImplementedError

    def bind(self, algorithm) -> "_AlgorithmConfig":
        return msgspec.structs.replace(self, algorithm=algorithm)

    def select(self, nblocks: int, threads_per_block: int) -> "_AlgorithmConfig":
        if self.algorithm is None:
            raise RuntimeError(f"Algorithm {self.name} has not been bound")
        if (nblocks, threads_per_block) not in self.tuning_launches():
            raise ValueError(f"Invalid launch selection for algorithm {self.name}")
        return self

    def selected_launch(self) -> tuple[int, int]:
        if self.algorithm is None:
            raise RuntimeError(f"Algorithm {self.name} has not been bound")
        return self.tuning_launches()[0]

    def reset(self):
        if self.algorithm is None:
            raise RuntimeError(f"Algorithm {self.name} has not been bound")
        self.algorithm.reset()

    @staticmethod
    def compose_rsag(
        reduce_scatter: "_AlgorithmConfig",
        allgather: "_AlgorithmConfig",
        *,
        rank: int,
        world_size: int,
        ipc_domain_count: int,
        reduce_ops: Any,
    ) -> "_CompositeAlgorithmConfig":
        if (
            reduce_scatter.collective != "reducescatter"
            or allgather.collective != "allgather"
        ):
            raise ValueError("RSAG requires reduce-scatter and all-gather algorithms")
        if reduce_scatter.algorithm is None or allgather.algorithm is None:
            raise RuntimeError("RSAG composition requires bound algorithms")
        message_size_range = _MessageSizeRange(
            minimum=max(
                reduce_scatter.message_size_range.minimum,
                allgather.message_size_range.minimum,
            ),
            maximum=min(
                reduce_scatter.message_size_range.maximum,
                allgather.message_size_range.maximum,
            ),
        )
        algorithm = _TwoKernelAllReduce(
            name=(
                f"allreduce_rsag_{reduce_scatter.algorithm.name}_"
                f"{allgather.algorithm.name}"
            ),
            reduce_scatter=reduce_scatter.algorithm,
            allgather=allgather.algorithm,
            allgather_op=allgather.resolve_reduce_op(reduce_ops),
            rank=rank,
            world_size=world_size,
        )
        return _CompositeAlgorithmConfig(
            name=algorithm.name,
            collective="allreduce",
            world_sizes=(world_size,),
            ipc_domain_counts=(ipc_domain_count,),
            threads_per_block=(0,),
            message_size_range=message_size_range,
            reduce_op="SUM",
            supported_dtypes=tuple(
                dtype
                for dtype in reduce_scatter.supported_dtypes
                if dtype in allgather.supported_dtypes
            ),
            algorithm=algorithm,
        )


class _DslAlgorithmConfig(_AlgorithmConfig):
    implementation: ClassVar[str] = "dsl"
    algo_spec: Any
    algorithm_kwargs: dict[str, tuple[Any, ...]] = msgspec.field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        if not self.threads_per_block:
            raise ValueError("DSL compile thread candidates cannot be empty")
        if any(threads <= 0 for threads in self.threads_per_block):
            raise ValueError("DSL compile thread candidates must be positive")
        if any(not values for values in self.algorithm_kwargs.values()):
            raise ValueError("DSL algorithm kwarg candidates cannot be empty")
        if self.algorithm is None:
            if self.algo_spec.world_size != 0 or self.algo_spec.nranks_per_node != 0:
                raise ValueError("DSL AlgoSpec templates require zero topology values")
            if self.algo_spec.name != self.name:
                raise ValueError("DSL AlgoSpec and algorithm names must match")
        elif self.algo_spec.world_size <= 0 or self.algo_spec.nranks_per_node <= 0:
            raise ValueError("Bound DSL algorithms require a materialized AlgoSpec")
        if self.algo_spec.collective.name != self.collective:
            raise ValueError("DSL AlgoSpec and algorithm collectives must match")
        if self.algo_spec.in_place != self.in_place:
            raise ValueError("DSL AlgoSpec and algorithm buffer modes must match")

    def tuning_launches(self) -> tuple[tuple[int, int], ...]:
        return ((0, 0),)

    def adapt_to_topology(
        self,
        world_size: int,
        nranks_per_ipc_domain: int,
        algorithm_kwargs: Optional[dict[str, Any]] = None,
    ) -> "_AlgorithmConfig":
        if self.parameter_adapter is None:
            return self
        return self.parameter_adapter(
            self,
            world_size=world_size,
            nranks_per_ipc_domain=nranks_per_ipc_domain,
            instances=self.algo_spec.instances,
            algorithm_kwargs=algorithm_kwargs or {},
        )

    def bind(self, algorithm, algo_spec) -> "_AlgorithmConfig":
        return msgspec.structs.replace(self, algorithm=algorithm, algo_spec=algo_spec)

    def dsl_name(
        self,
        ipc_domain_count: int,
        threads_per_block: int,
        algorithm_kwargs: dict[str, Any],
    ) -> str:
        variant_parts = [
            f"{key}_{value}" for key, value in sorted(algorithm_kwargs.items())
        ]
        variant = f"{'_'.join(variant_parts)}_" if variant_parts else ""
        return f"{self.name}_{ipc_domain_count}node_{variant}{threads_per_block}TPB"


class _NativeAlgorithmConfig(_AlgorithmConfig):
    implementation: ClassVar[str] = "native"
    nblocks: tuple[int, ...]
    selected_launch_parameters: Optional[tuple[int, int]] = None

    def __post_init__(self):
        super().__post_init__()
        if not self.nblocks:
            raise ValueError("Native launch candidates cannot be empty")
        if self.selected_launch_parameters is not None:
            if self.algorithm is None:
                raise ValueError("Only bound native algorithms can select a launch")
            if self.selected_launch_parameters not in self.tuning_launches():
                raise ValueError("Invalid native launch selection")

    def tuning_launches(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (nblocks, threads_per_block)
            for nblocks in self.nblocks
            for threads_per_block in self.threads_per_block
        )

    def select(self, nblocks: int, threads_per_block: int) -> "_AlgorithmConfig":
        if self.algorithm is None:
            raise RuntimeError(f"Algorithm {self.name} has not been bound")
        launch = (nblocks, threads_per_block)
        if launch not in self.tuning_launches():
            raise ValueError(f"Invalid launch selection for algorithm {self.name}")
        return msgspec.structs.replace(self, selected_launch_parameters=launch)

    def selected_launch(self) -> tuple[int, int]:
        if self.selected_launch_parameters is None:
            raise RuntimeError(f"Algorithm {self.name} has not been tuned")
        return self.selected_launch_parameters


class _CompositeAlgorithmConfig(_AlgorithmConfig):
    implementation: ClassVar[str] = "composite"

    def __post_init__(self):
        super().__post_init__()
        if self.algorithm is None:
            raise ValueError("Composite algorithms must be bound when constructed")

    def tuning_launches(self) -> tuple[tuple[int, int], ...]:
        return ((0, 0),)


def _adapt_allreduce_packet(
    config: _AlgorithmConfig,
    *,
    world_size: int,
    nranks_per_ipc_domain: int,
) -> _AlgorithmConfig:
    if not isinstance(config, _NativeAlgorithmConfig):
        raise TypeError("AllReduce packet adaptation requires a native config")
    if world_size != nranks_per_ipc_domain:
        return config
    min_blocks = nranks_per_ipc_domain - 1
    nblocks = tuple(nblocks for nblocks in config.nblocks if nblocks >= min_blocks)
    if not nblocks:
        return config
    return msgspec.structs.replace(config, nblocks=nblocks)


def _adapt_dsl_message_size_range(
    config: _AlgorithmConfig,
    *,
    world_size: int,
    nranks_per_ipc_domain: int,
    algorithm_kwargs: dict[str, Any],
    **_: Any,
) -> _AlgorithmConfig:
    if not isinstance(config, _DslAlgorithmConfig):
        raise TypeError("DSL message size adaptation requires a DSL config")
    ipc_domain_count = world_size // nranks_per_ipc_domain
    thread_block_group_size = algorithm_kwargs.get("thread_block_group_size", 1)
    return msgspec.structs.replace(
        config,
        message_size_range=_MessageSizeRange(
            minimum=config.message_size_range.minimum * thread_block_group_size,
            maximum=config.message_size_range.maximum * ipc_domain_count,
        ),
    )


def _create_native_algorithm_configs() -> tuple[_NativeAlgorithmConfig, ...]:
    return (
        _NativeAlgorithmConfig(
            name="default_allreduce_nvls_packet",
            collective="allreduce",
            world_sizes=_NATIVE_SUPPORTED_WORLD_SIZES,
            ipc_domain_counts=_NATIVE_IPC_DOMAIN_COUNTS,
            message_size_range=_MessageSizeRange(0, 512 << 10),
            nblocks=(4, 8, 12, 16),
            threads_per_block=_DEFAULT_THREADS_PER_BLOCK,
            reduce_op="SUM",
            requires_nvls=True,
        ),
        _NativeAlgorithmConfig(
            name="default_allreduce_packet",
            collective="allreduce",
            world_sizes=_NATIVE_SUPPORTED_WORLD_SIZES,
            ipc_domain_counts=_NATIVE_IPC_DOMAIN_COUNTS,
            message_size_range=_MessageSizeRange(0, 2 << 20),
            nblocks=(14, 21, 28, 42, 56),
            threads_per_block=_DEFAULT_THREADS_PER_BLOCK,
            reduce_op="SUM",
            parameter_adapter=_adapt_allreduce_packet,
        ),
        _NativeAlgorithmConfig(
            name="default_allreduce_rsag_zero_copy",
            collective="allreduce",
            world_sizes=_NATIVE_SUPPORTED_WORLD_SIZES,
            ipc_domain_counts=_NATIVE_IPC_DOMAIN_COUNTS,
            message_size_range=_MessageSizeRange(512 << 10, 4 << 30),
            nblocks=(32, 48, 64, 128),
            threads_per_block=_DEFAULT_THREADS_PER_BLOCK,
            reduce_op="SUM",
        ),
        _NativeAlgorithmConfig(
            name="default_allreduce_nvls_zero_copy",
            collective="allreduce",
            world_sizes=_NATIVE_SUPPORTED_WORLD_SIZES,
            ipc_domain_counts=_NATIVE_IPC_DOMAIN_COUNTS,
            message_size_range=_MessageSizeRange(1 << 10, 4 << 30),
            nblocks=(4, 8, 12, 16, 32),
            threads_per_block=_DEFAULT_THREADS_PER_BLOCK,
            reduce_op="SUM",
            requires_nvls=True,
            requires_symmetric_memory=True,
        ),
    )


def _create_algorithm_configs(language) -> tuple[_AlgorithmConfig, ...]:
    default_spec = language.AlgoSpec(
        name="allreduce_multi_nodes",
        collective=language.collectives.AllReduce(0, 1, True),
        nranks_per_node=0,
        world_size=0,
        in_place=True,
        instances=1,
        protocol="LL",
        instr_fusion=True,
        auto_sync=False,
        replication_policy=language.ReplicationPolicy.interleaved,
        reuse_resources=True,
        use_double_scratch_buffer=True,
        buffer_alignment=16,
    )
    allgather_spec = dataclass_replace(
        default_spec,
        name="allgather_multi_nodes",
        collective=language.collectives.AllGather(0, 1, False),
        in_place=False,
    )
    reduce_scatter_spec = dataclass_replace(
        default_spec,
        name="reducescatter_multi_nodes",
        collective=language.collectives.ReduceScatter(0, 1, True),
        instr_fusion=False,
    )
    dsl_configs = (
        _DslAlgorithmConfig(
            name="allreduce_multi_nodes",
            collective="allreduce",
            world_sizes=_SUPPORTED_WORLD_SIZES,
            ipc_domain_counts=_MULTI_NODE_IPC_DOMAIN_COUNTS,
            message_size_range=_MessageSizeRange(1 << 10, 1 << 20),
            reduce_op="SUM",
            algo_spec=default_spec,
            threads_per_block=_DEFAULT_THREADS_PER_BLOCK,
            algorithm_kwargs={
                "thread_block_group_size": _DEFAULT_THREAD_BLOCK_GROUP_SIZES
            },
            parameter_adapter=_adapt_dsl_message_size_range,
        ),
        _DslAlgorithmConfig(
            name="allgather_multi_nodes",
            collective="allgather",
            world_sizes=_SUPPORTED_WORLD_SIZES,
            ipc_domain_counts=_MULTI_NODE_IPC_DOMAIN_COUNTS,
            message_size_range=_MessageSizeRange(1 << 10, 1 << 20),
            reduce_op="NOP",
            in_place=False,
            algo_spec=allgather_spec,
            threads_per_block=_DEFAULT_THREADS_PER_BLOCK,
            parameter_adapter=_adapt_dsl_message_size_range,
        ),
        _DslAlgorithmConfig(
            name="reducescatter_multi_nodes",
            collective="reducescatter",
            world_sizes=_SUPPORTED_WORLD_SIZES,
            ipc_domain_counts=_MULTI_NODE_IPC_DOMAIN_COUNTS,
            message_size_range=_MessageSizeRange(1 << 10, 1 << 20),
            reduce_op="SUM",
            algo_spec=reduce_scatter_spec,
            threads_per_block=_DEFAULT_THREADS_PER_BLOCK,
            algorithm_kwargs={
                "thread_block_group_size": _DEFAULT_THREAD_BLOCK_GROUP_SIZES
            },
            parameter_adapter=_adapt_dsl_message_size_range,
        ),
    )
    return (*dsl_configs, *_create_native_algorithm_configs())


class _TwoKernelAllReduce(msgspec.Struct, frozen=True):
    name: str
    reduce_scatter: Any
    allgather: Any
    allgather_op: Any
    rank: int
    world_size: int

    def execute(
        self,
        *,
        comm,
        executor,
        input_buffer,
        output_buffer,
        input_size,
        output_size,
        dtype,
        op,
        stream,
        nblocks,
        nthreads_per_block,
        symmetric_memory,
    ):
        shard_size = input_size // self.world_size

        result = self.reduce_scatter.execute(
            comm=comm,
            executor=executor,
            input_buffer=input_buffer,
            output_buffer=output_buffer,
            input_size=input_size,
            output_size=shard_size,
            dtype=dtype,
            op=op,
            stream=stream,
            nblocks=nblocks,
            nthreads_per_block=nthreads_per_block,
            symmetric_memory=symmetric_memory,
        )
        if result != 0:
            return result
        return self.allgather.execute(
            comm=comm,
            executor=executor,
            input_buffer=output_buffer + self.rank * shard_size,
            output_buffer=output_buffer,
            input_size=shard_size,
            output_size=output_size,
            dtype=dtype,
            op=self.allgather_op,
            stream=stream,
            nblocks=nblocks,
            nthreads_per_block=nthreads_per_block,
            symmetric_memory=symmetric_memory,
        )

    def reset(self):
        self.reduce_scatter.reset()
        self.allgather.reset()
