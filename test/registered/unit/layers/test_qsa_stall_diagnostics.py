import logging
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa.stall_diagnostics import (
    _MAX_STAGE_EVENTS,
    QSAStallDiagnostics,
    snapshot_qsa_stall_metadata,
)
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QwenSparseAttnBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class _ChunkPrefillMode:
    name = "EXTEND"

    @staticmethod
    def is_extend_or_draft_extend_or_mixed():
        return True


def _forward_batch(*, seq_lens=(9, 4), extend_lens=(3, 4)):
    return SimpleNamespace(
        forward_mode=_ChunkPrefillMode(),
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32),
        extend_seq_lens_cpu=list(extend_lens),
        req_pool_indices=torch.empty(len(seq_lens), dtype=torch.int32),
    )


def test_qsa_stall_diagnostics_default_off_has_no_monitor():
    assert envs.SGLANG_QSA_STALL_DIAGNOSTICS.default is False
    with envs.SGLANG_QSA_STALL_DIAGNOSTICS.override(False):
        backend = QwenSparseAttnBackend()
    assert backend.qsa_stall_diagnostics is None


def test_qsa_stall_metadata_uses_cpu_batch_facts_only():
    metadata = snapshot_qsa_stall_metadata(
        layer_id=7,
        hidden_states=torch.empty((7, 128)),
        forward_batch=_forward_batch(),
    )

    assert metadata is not None
    assert metadata.layer_id == 7
    assert metadata.forward_mode == "EXTEND"
    assert metadata.request_shape == (2,)
    assert metadata.hidden_shape == (7, 128)
    assert metadata.sequence_count == 2
    assert metadata.prefix_sequence_count == 1
    assert metadata.max_sequence_length == 9
    assert metadata.max_extend_length == 4


def test_qsa_stall_reports_once_and_completion_cleans_up(caplog):
    now = [10.0]
    diagnostics = QSAStallDiagnostics(
        timeout_s=30.0, clock=lambda: now[0], start_thread=False
    )

    with caplog.at_level(logging.ERROR):
        with diagnostics.track(
            layer_id=7,
            hidden_states=torch.empty((7, 128)),
            forward_batch=_forward_batch(),
        ) as tracked:
            assert tracked
            now[0] = 40.0
            assert diagnostics._check_once()
            assert not diagnostics._check_once()

        now[0] = 100.0
        assert not diagnostics._check_once()

    host_lines = [
        record.message
        for record in caplog.records
        if record.message.startswith("QSA stall host heartbeat:")
    ]
    assert len(host_lines) == 1
    assert "layer=7" in host_lines[0]
    assert "phase=before_qkv_or_indexer" in host_lines[0]


def test_qsa_stall_cuda_query_failure_is_reported_after_host_line(caplog):
    class _BrokenEvent:
        def query(self):
            raise RuntimeError("driver unavailable")

    now = [0.0]
    diagnostics = QSAStallDiagnostics(
        timeout_s=30.0, clock=lambda: now[0], start_thread=False
    )
    with diagnostics.track(
        layer_id=3,
        hidden_states=torch.empty((7, 64)),
        forward_batch=_forward_batch(),
    ):
        with diagnostics._lock:
            diagnostics._active.indexer_event = _BrokenEvent()
        now[0] = 31.0
        with caplog.at_level(logging.ERROR):
            assert diagnostics._check_once()

    messages = [record.message for record in caplog.records]
    assert messages[0].startswith("QSA stall host heartbeat:")
    assert messages[1].startswith("QSA stall CUDA readiness:")
    assert "pre_attention_stream_event=error(RuntimeError)" in messages[1]


def test_qsa_stall_discards_event_results_if_span_finishes_during_query(caplog):
    now = [0.0]
    diagnostics = QSAStallDiagnostics(
        timeout_s=30.0, clock=lambda: now[0], start_thread=False
    )

    class _CompletingEvent:
        def query(self):
            diagnostics.finish(5)
            return True

    with diagnostics.track(
        layer_id=5,
        hidden_states=torch.empty((7, 64)),
        forward_batch=_forward_batch(),
    ):
        with diagnostics._lock:
            diagnostics._active.indexer_event = _CompletingEvent()
        now[0] = 31.0
        with caplog.at_level(logging.ERROR):
            assert diagnostics._check_once()

    messages = [record.message for record in caplog.records]
    assert messages[0].startswith("QSA stall host heartbeat:")
    assert messages[1].endswith("state=stale probe_results=discarded")


class _Event:
    def __init__(self, ready=True):
        self.ready = ready
        self.stream = None

    def record(self, stream):
        self.stream = stream

    def query(self):
        return self.ready


def test_stage_events_record_caller_stream_and_report_after_host(monkeypatch, caplog):
    events = [_Event(True), _Event(False)]
    pending = iter(events)
    stream = object()
    monkeypatch.setattr(torch.cuda, "Event", lambda: next(pending))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    now = [0.0]
    diagnostics = QSAStallDiagnostics(clock=lambda: now[0], start_thread=False)
    with diagnostics.track(
        layer_id=43, hidden_states=torch.empty((7, 64)), forward_batch=_forward_batch()
    ):
        assert diagnostics.is_tracking(43)
        diagnostics.mark_stage(43, "compressed_keys_updated")
        diagnostics.mark_stage(43, "scores_ready[0:2]")
        snapshot = diagnostics._active.stage_events
        now[0] = 31.0
        with caplog.at_level(logging.ERROR):
            assert diagnostics._check_once()
    assert all(event.stream is stream for event in events)
    assert len(snapshot) == 2
    assert not diagnostics.is_tracking(43)
    messages = [record.message for record in caplog.records]
    assert messages[0].startswith("QSA stall host heartbeat:")
    assert "compressed_keys_updated=ready" in messages[2]
    assert "scores_ready[0:2]=pending" in messages[2]


def test_stage_events_are_bounded_and_nonqualifying_spans_allocate_none(monkeypatch):
    events = []

    def make_event():
        event = _Event()
        events.append(event)
        return event

    monkeypatch.setattr(torch.cuda, "Event", make_event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    diagnostics = QSAStallDiagnostics(start_thread=False)
    diagnostics.mark_stage(43, "no_span")
    with diagnostics.track(
        layer_id=43,
        hidden_states=torch.empty((4, 64)),
        forward_batch=_forward_batch(seq_lens=(4,), extend_lens=(4,)),
    ) as tracked:
        assert not tracked
        diagnostics.mark_stage(43, "no_prefix")
    assert not events
    with diagnostics.track(
        layer_id=43, hidden_states=torch.empty((7, 64)), forward_batch=_forward_batch()
    ):
        for i in range(_MAX_STAGE_EVENTS + 3):
            diagnostics.mark_stage(43, f"stage_{i}")
        assert len(events) == _MAX_STAGE_EVENTS
        assert diagnostics._active.omitted_stage_events == 3
    assert diagnostics._active is None


def test_stage_query_discarded_when_transfer_slot_changes(monkeypatch, caplog):
    now = [0.0]
    diagnostics = QSAStallDiagnostics(clock=lambda: now[0], start_thread=False)
    counter = SimpleNamespace(
        consumer_index=0, events=[SimpleNamespace(load_events=[_Event()])]
    )

    class _ReusedSlotEvent(_Event):
        def query(self):
            counter.consumer_index = 1
            return True

    monkeypatch.setattr(torch.cuda, "Event", _ReusedSlotEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    with diagnostics.track(
        layer_id=43, hidden_states=torch.empty((7, 64)), forward_batch=_forward_batch()
    ):
        diagnostics.mark_stage(43, "scores_ready")
        diagnostics.mark_before_kv_getters(
            43, SimpleNamespace(start_layer=43, layer_transfer_counter=counter)
        )
        now[0] = 31.0
        with caplog.at_level(logging.ERROR):
            assert diagnostics._check_once()
    messages = [record.message for record in caplog.records]
    assert messages[1].endswith("state=stale probe_results=discarded")
    assert not any("QSA stall stage readiness:" in line for line in messages)


def test_stage_record_failure_is_diagnostic_only(monkeypatch, caplog):
    def broken_event():
        raise RuntimeError("no driver")

    monkeypatch.setattr(torch.cuda, "Event", broken_event)
    now = [0.0]
    diagnostics = QSAStallDiagnostics(clock=lambda: now[0], start_thread=False)
    with diagnostics.track(
        layer_id=43, hidden_states=torch.empty((7, 64)), forward_batch=_forward_batch()
    ):
        diagnostics.mark_stage(43, "compressed_keys_updated")
        now[0] = 31.0
        with caplog.at_level(logging.ERROR):
            assert diagnostics._check_once()
    assert "compressed_keys_updated=unavailable(RuntimeError)" in caplog.text


def test_prefill_scoring_topk_and_expansion_stage_order(monkeypatch):
    from sglang.srt.layers.attention.qsa import qsa_indexer

    calls = []
    diagnostics = SimpleNamespace(
        set_phase=lambda layer, phase: calls.append(("phase", phase)),
        mark_stage=lambda layer, phase: calls.append(("event", phase)),
    )
    monkeypatch.setattr(qsa_indexer, "_qsa_prefill_row_chunk_size", lambda *args: 2)
    monkeypatch.setattr(
        qsa_indexer, "qsa_mqa_prefill", lambda q, k, *args: torch.zeros((q.shape[0], 2))
    )
    monkeypatch.setattr(
        qsa_indexer,
        "qsa_fast_topk",
        lambda logits, *args, **kwargs: torch.zeros(
            (logits.shape[0], 1), dtype=torch.int32
        ),
    )
    monkeypatch.setattr(
        qsa_indexer,
        "expand_qsa_block_indices",
        lambda blocks, *args, **kwargs: torch.ones(
            (blocks.shape[0], 7), dtype=torch.int32
        ),
    )
    indexer = SimpleNamespace(layer_id=43, token_topk=4, compress_ratio=4, block_topk=1)
    inputs = (
        torch.ones((3, 2, 4)),
        torch.ones((2, 1, 4)),
        *[torch.zeros(3, dtype=torch.int32)] * 4,
    )
    result = qsa_indexer.QSAIndexer.select_prefill_tokens(
        indexer, *inputs, stall_diagnostics=diagnostics
    )
    assert torch.equal(result, torch.ones((3, 7), dtype=torch.int32))
    assert [phase for kind, phase in calls if kind == "event"] == [
        "scores_ready[0:2]",
        "topk_ready[0:2]",
        "expanded_rows_ready[0:2]",
        "scores_ready[2:3]",
        "topk_ready[2:3]",
        "expanded_rows_ready[2:3]",
    ]
    calls.clear()
    without = qsa_indexer.QSAIndexer.select_prefill_tokens(indexer, *inputs)
    assert torch.equal(result, without)
    assert not calls
