"""Opt-in, stall-only diagnostics for QSA chunk prefill.

The diagnostic deliberately keeps its host heartbeat independent of CUDA.  A
driver call made while the device is unhealthy can block, so event readiness is
queried only after the host-only line has been emitted.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Callable, Iterator, Optional

import torch

logger = logging.getLogger(__name__)

_STALL_TIMEOUT_S = 30.0
_MAX_STAGE_EVENTS = 128


@dataclass(frozen=True)
class QSAStallMetadata:
    layer_id: int
    forward_mode: str
    request_shape: tuple[int, ...]
    hidden_shape: tuple[int, ...]
    sequence_count: int
    prefix_sequence_count: int
    max_sequence_length: int
    max_extend_length: int


@dataclass
class _ActiveSpan:
    generation: int
    metadata: QSAStallMetadata
    started_at: float
    phase: str = "before_qkv_or_indexer"
    reported: bool = False
    indexer_event: object = None
    indexer_event_error: Optional[str] = None
    transfer_event: object = None
    transfer_event_error: Optional[str] = None
    transfer_counter: object = None
    transfer_consumer_slot: Optional[int] = None
    transfer_layer_index: Optional[int] = None
    # Immutable snapshots prevent the monitor from iterating a mutating list.
    stage_events: tuple = ()
    omitted_stage_events: int = 0


def _cpu_ints(values) -> Optional[list[int]]:
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        # Diagnostics must never introduce a device-to-host readback.
        if values.device.type != "cpu":
            return None
        values = values.flatten()
    try:
        return [int(value) for value in values]
    except (TypeError, ValueError):
        return None


def snapshot_qsa_stall_metadata(
    *, layer_id: int, hidden_states: torch.Tensor, forward_batch
) -> Optional[QSAStallMetadata]:
    """Return host-only metadata for a genuine prefix-bearing chunk prefill."""

    mode = getattr(forward_batch, "forward_mode", None)
    is_chunk_prefill = getattr(
        mode, "is_extend_or_draft_extend_or_mixed", lambda: False
    )()
    if not is_chunk_prefill:
        return None

    sequence_lengths = _cpu_ints(getattr(forward_batch, "seq_lens_cpu", None))
    extend_lengths = _cpu_ints(getattr(forward_batch, "extend_seq_lens_cpu", None))
    if (
        sequence_lengths is None
        or extend_lengths is None
        or len(sequence_lengths) != len(extend_lengths)
        or not sequence_lengths
    ):
        return None
    prefix_count = sum(
        sequence_length > extend_length
        for sequence_length, extend_length in zip(sequence_lengths, extend_lengths)
    )
    if prefix_count == 0:
        return None

    req_pool_indices = getattr(forward_batch, "req_pool_indices", None)
    request_shape = tuple(int(size) for size in getattr(req_pool_indices, "shape", ()))
    hidden_shape = tuple(int(size) for size in hidden_states.shape)
    return QSAStallMetadata(
        layer_id=int(layer_id),
        forward_mode=getattr(mode, "name", type(mode).__name__),
        request_shape=request_shape,
        hidden_shape=hidden_shape,
        sequence_count=len(sequence_lengths),
        prefix_sequence_count=prefix_count,
        max_sequence_length=max(sequence_lengths),
        max_extend_length=max(extend_lengths),
    )


class QSAStallDiagnostics:
    """Monitor the one QSA layer span owned by a model execution thread."""

    def __init__(
        self,
        *,
        timeout_s: float = _STALL_TIMEOUT_S,
        clock: Callable[[], float] = time.monotonic,
        start_thread: bool = True,
    ) -> None:
        self._timeout_s = timeout_s
        self._clock = clock
        self._lock = threading.Lock()
        self._active: Optional[_ActiveSpan] = None
        self._generation = 0
        if start_thread:
            threading.Thread(
                target=self._monitor_loop,
                daemon=True,
                name="qsa-stall-diagnostics",
            ).start()

    @contextmanager
    def track(
        self, *, layer_id: int, hidden_states: torch.Tensor, forward_batch
    ) -> Iterator[bool]:
        metadata = snapshot_qsa_stall_metadata(
            layer_id=layer_id,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        if metadata is None:
            yield False
            return
        with self._lock:
            self._generation += 1
            generation = self._generation
            self._active = _ActiveSpan(
                generation=generation,
                metadata=metadata,
                started_at=self._clock(),
            )
        try:
            yield True
        finally:
            self.finish(layer_id, generation=generation)

    def set_phase(self, layer_id: int, phase: str) -> None:
        with self._lock:
            span = self._matching_span(layer_id)
            if span is not None:
                span.phase = phase

    def is_tracking(self, layer_id: int) -> bool:
        with self._lock:
            return self._matching_span(layer_id) is not None

    def mark_stage(self, layer_id: int, stage: str) -> None:
        """Record progress on the caller's stream, never synchronize it.

        A pending marker bounds unfinished stream work, not a faulty kernel:
        dependencies or work preceding this layer can also keep it pending.
        Each span owns fresh events so a later forward cannot re-record an
        event while the monitor is inspecting the previous generation.
        """
        with self._lock:
            span = self._matching_span(layer_id)
            if span is None:
                return
            span.phase = "recording_stage_event:" + stage
            if len(span.stage_events) >= _MAX_STAGE_EVENTS:
                span.omitted_stage_events += 1
                span.phase = stage
                return
            generation = span.generation
        try:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
            error = None
        except Exception as exc:
            event = None
            error = type(exc).__name__
        with self._lock:
            span = self._matching_span(layer_id, generation)
            if span is not None:
                span.stage_events += ((stage, event, error),)
                span.phase = stage

    def mark_indexer_enqueued(self, layer_id: int) -> None:
        # Set the phase before any CUDA API call: event creation/record can
        # itself become unresponsive when the driver is unhealthy.
        self.set_phase(layer_id, "recording_indexer_stream_event")
        with self._lock:
            span = self._matching_span(layer_id)
            if span is None:
                return
            generation = span.generation
        try:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
            error = None
        except Exception as exc:  # diagnostic failure must not change serving
            event = None
            error = type(exc).__name__
        with self._lock:
            span = self._matching_span(layer_id, generation)
            if span is not None:
                span.indexer_event = event
                span.indexer_event_error = error
                # This event covers all earlier work on the same stream. It is
                # not proof about the indexer in isolation.
                span.phase = "pre_attention_work_enqueued"

    def mark_before_kv_getters(self, layer_id: int, pool) -> None:
        consumer_slot = None
        layer_index = None
        transfer_event = None
        error = None
        counter = getattr(pool, "layer_transfer_counter", None)
        if counter is not None:
            try:
                consumer_slot = int(counter.consumer_index)
                layer_index = int(layer_id) - int(pool.start_layer)
                if consumer_slot >= 0:
                    transfer_event = counter.events[consumer_slot].load_events[
                        layer_index
                    ]
                    if int(counter.consumer_index) != consumer_slot:
                        error = "consumer_slot_changed"
                        transfer_event = None
            except (AttributeError, IndexError, TypeError, ValueError) as exc:
                error = type(exc).__name__
                transfer_event = None
        with self._lock:
            span = self._matching_span(layer_id)
            if span is None:
                return
            span.phase = "before_kv_getters"
            span.transfer_consumer_slot = consumer_slot
            span.transfer_layer_index = layer_index
            span.transfer_event = transfer_event
            span.transfer_event_error = error
            span.transfer_counter = counter

    def finish(self, layer_id: int, *, generation: Optional[int] = None) -> None:
        with self._lock:
            span = self._matching_span(layer_id, generation)
            if span is not None:
                self._active = None

    def _matching_span(
        self, layer_id: int, generation: Optional[int] = None
    ) -> Optional[_ActiveSpan]:
        span = self._active
        if span is None or span.metadata.layer_id != int(layer_id):
            return None
        if generation is not None and span.generation != generation:
            return None
        return span

    def _monitor_loop(self) -> None:
        interval = min(1.0, max(0.05, self._timeout_s / 2))
        while True:
            time.sleep(interval)
            try:
                self._check_once()
            except Exception:
                logger.exception("QSA stall diagnostic monitor failed")

    def _check_once(self, now: Optional[float] = None) -> bool:
        now = self._clock() if now is None else now
        with self._lock:
            span = self._active
            if span is None or span.reported or now - span.started_at < self._timeout_s:
                return False
            span.reported = True
            report = replace(span)

        metadata = report.metadata
        pending_s = now - report.started_at
        transfer_event = report.transfer_event
        transfer_error = report.transfer_event_error
        if report.transfer_counter is not None:
            try:
                current_slot = int(report.transfer_counter.consumer_index)
                if current_slot != report.transfer_consumer_slot:
                    transfer_event = None
                    transfer_error = "consumer_slot_changed"
            except (AttributeError, TypeError, ValueError) as exc:
                transfer_event = None
                transfer_error = type(exc).__name__
        logger.error(
            "QSA stall host heartbeat: pending_s=%.1f layer=%d phase=%s "
            "mode=%s request_shape=%s hidden_shape=%s sequences=%d "
            "prefix_sequences=%d max_sequence_length=%d max_extend_length=%d "
            "transfer_consumer_slot=%s transfer_layer_index=%s",
            pending_s,
            metadata.layer_id,
            report.phase,
            metadata.forward_mode,
            metadata.request_shape,
            metadata.hidden_shape,
            metadata.sequence_count,
            metadata.prefix_sequence_count,
            metadata.max_sequence_length,
            metadata.max_extend_length,
            report.transfer_consumer_slot,
            report.transfer_layer_index,
        )

        # Completion can race the host log. Do not query event objects after
        # their span has ended; the HiCache ring may then be reused by a newer
        # forward and its readiness would be misattributed.
        with self._lock:
            active = self._active
            if active is None or active.generation != report.generation:
                logger.error(
                    "QSA stall CUDA readiness: generation=%d state=stale",
                    report.generation,
                )
                return True

        stage_statuses = [
            (stage, self._query_event(event, error))
            for stage, event, error in report.stage_events
        ]
        indexer_status = self._query_event(
            report.indexer_event, report.indexer_event_error
        )
        transfer_status = self._query_event(transfer_event, transfer_error)
        with self._lock:
            active = self._active
            stale = active is None or active.generation != report.generation
            if not stale and report.transfer_counter is not None:
                try:
                    stale = (
                        int(report.transfer_counter.consumer_index)
                        != report.transfer_consumer_slot
                    )
                except (AttributeError, TypeError, ValueError):
                    stale = True
        if stale:
            logger.error(
                "QSA stall CUDA readiness: generation=%d state=stale "
                "probe_results=discarded",
                report.generation,
            )
        else:
            logger.error(
                "QSA stall CUDA readiness: generation=%d "
                "pre_attention_stream_event=%s transfer_event=%s",
                report.generation,
                indexer_status,
                transfer_status,
            )
            if stage_statuses or report.omitted_stage_events:
                logger.error(
                    "QSA stall stage readiness: generation=%d stages=%s "
                    "omitted_stage_events=%d",
                    report.generation,
                    ", ".join(f"{stage}={status}" for stage, status in stage_statuses),
                    report.omitted_stage_events,
                )
        return True

    @staticmethod
    def _query_event(event, prior_error: Optional[str]) -> str:
        if prior_error is not None:
            return f"unavailable({prior_error})"
        if event is None:
            return "unavailable"
        try:
            return "ready" if event.query() else "pending"
        except Exception as exc:
            return f"error({type(exc).__name__})"


__all__ = [
    "QSAStallDiagnostics",
    "QSAStallMetadata",
    "snapshot_qsa_stall_metadata",
]
