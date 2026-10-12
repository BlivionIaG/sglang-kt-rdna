"""CPU regressions for write-through persistence and in-flight host ownership.

Both Pennyroyal recipes serve with ``--hicache-write-policy write_through``, a
page_first host tier and the NIXL storage backend, so two things must hold:

* a chunked prefill insert still schedules the host backup for the fresh
  prefix it creates. Skipping hit-count bookkeeping for chunked requests must
  not also skip the backup, or the unbacked prefix can be evicted before L3
  ever sees it and a later restore comes back partially (sglang#39444). That
  eager zero-hit backup belongs to write_through alone: under
  write_through_selective a node must still earn persistence from real repeat
  hits, so a chunked insert fires nothing.
* a storage SET that reads host blocks zero-copy keeps every fragment created
  by a later radix split pinned until the SET acks. The prefix fragment does
  not inherit the parent's host lock, so duplicate-host reclamation can hand
  the same blocks to a new backup while the backend still reads them
  (sglang#38480).

No device or host pools exist here: ``BackupKV`` actions are observed rather
than executed, and host values are plain index tensors.
"""

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams, InsertParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.cache_action import BackupKV
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_cache.components.full_component import FullComponent
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    EvictLayer,
    TreeComponent,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

FULL = ComponentType.FULL
MAMBA = ComponentType.MAMBA


class _AuxStateComponent(TreeComponent):
    """Mamba-shaped stand-in: a single-node host lock, no device state."""

    component_type = MAMBA

    def create_match_validator(self, match_device_only: bool = False):
        return lambda node: True

    def redistribute_on_node_split(self, new_parent, child):
        new_parent.component_data[MAMBA].host_lock_ref = 0

    def evict_component(
        self, node, device_frees, host_frees, target: EvictLayer = EvictLayer.DEVICE
    ) -> tuple[int, int]:
        return 0, 0

    def acquire_component_lock(self, node, result, lock_host: bool = False):
        if lock_host:
            node.component_data[MAMBA].host_lock_ref += 1
        return result

    def release_component_lock(self, node, params, lock_host: bool = False):
        if lock_host:
            cd = node.component_data[MAMBA]
            if cd.host_lock_ref:
                cd.host_lock_ref -= 1

    def _evict_device_start(self, request_cnt) -> None:
        pass

    def _evict_device_next_node(self, tracker, device_frees, host_frees):
        return None

    def _evict_device_end(self) -> None:
        pass

    def _dec_session_coverage(self, session_id, leaf) -> None:
        pass

    def _advance_session_coverage(self, session_id, leaf, old_ancestor) -> None:
        pass

    def _recede_session_coverage(self, session_id, leaf, fallback) -> None:
        pass


def _tree(write_policy="write_through", with_aux=False):
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=1,
    )
    cache = SimpleNamespace(enable_session_radix_cache=False)
    components = {FULL: FullComponent(cache, params)}
    if with_aux:
        components[MAMBA] = _AuxStateComponent(cache, params)
    core = UnifiedTreeCore(params, components)
    core.enable_hicache = True
    core.enable_storage = True
    core.is_write_back = write_policy == "write_back"
    core.write_through_threshold = 1 if write_policy == "write_through" else 2
    return core


def _insert(core, tokens, chunked=False):
    value = torch.arange(1, len(tokens) + 1, dtype=torch.int64)
    step = core.begin_insert(
        InsertParams(
            key=RadixKey(array("q", tokens)), value=value, chunked=chunked
        )
    )
    actions = list(step.actions)
    while core.has_ongoing_insert():
        actions += list(core.resume_insert().actions)
    core.end_insert()
    return actions


def _backed_up(actions):
    return {nid for a in actions if isinstance(a, BackupKV) for nid in a.node_ids}


def _node_for(core, tokens):
    return next(
        n
        for n in core._node_arena.values()
        if list(n.key.token_ids) == list(tokens) and n is not core.root_node
    )


class TestChunkedWriteThroughBackup(CustomTestCase):
    def test_chunked_insert_backs_up_fresh_prefix(self):
        core = _tree()
        first = _insert(core, [1, 2, 3, 4], chunked=True)
        self.assertEqual(
            _backed_up(first),
            {_node_for(core, [1, 2, 3, 4]).id},
            "a chunked insert must schedule the host backup it just created",
        )

        second = _insert(core, [1, 2, 3, 4, 5, 6], chunked=True)
        self.assertIn(
            _node_for(core, [5, 6]).id,
            _backed_up(second),
            "the second chunk's new segment must be backed up too",
        )

    def test_chunked_insert_does_not_count_itself_as_a_hit(self):
        core = _tree()
        _insert(core, [1, 2, 3, 4], chunked=True)
        _insert(core, [1, 2, 3, 4, 5, 6], chunked=True)
        for tokens in ([1, 2, 3, 4], [5, 6]):
            self.assertEqual(_node_for(core, tokens).hit_count, 0)

    def test_write_back_policy_still_skips_chunked_backup(self):
        core = _tree(write_policy="write_back")
        _insert(core, [1, 2, 3, 4], chunked=True)
        self.assertEqual(_backed_up(_insert(core, [1, 2, 3, 4, 5, 6], True)), set())

    def test_chunked_insert_respects_the_selective_threshold(self):
        """write_through_selective (threshold 2) must not be front-run by the
        zero-hit chunked backup, and a chunk never counts as a hit."""
        core = _tree(write_policy="write_through_selective")
        _insert(core, [1, 2, 3, 4], chunked=True)
        self.assertEqual(
            _backed_up(_insert(core, [1, 2, 3, 4, 5, 6], chunked=True)),
            set(),
            "a chunked insert must not bypass the selective admission threshold",
        )
        for tokens in ([1, 2, 3, 4], [5, 6]):
            self.assertEqual(_node_for(core, tokens).hit_count, 0)

        # Two separate non-chunked requests reach the threshold of 2.
        _insert(core, [1, 2, 3, 4, 5, 6])
        self.assertEqual(
            _backed_up(_insert(core, [1, 2, 3, 4, 5, 6])),
            {_node_for(core, [1, 2, 3, 4]).id, _node_for(core, [5, 6]).id},
        )

    def test_non_chunked_backup_still_needs_the_threshold(self):
        core = _tree(write_policy="write_through_selective")
        self.assertEqual(_backed_up(_insert(core, [1, 2, 3])), set())
        # The second, separate request reaches the threshold of 2.
        self.assertEqual(
            _backed_up(_insert(core, [1, 2, 3])),
            {_node_for(core, [1, 2, 3]).id},
        )


class TestInFlightBackupDedup(CustomTestCase):
    """An overlapping BackupKV chain must not allocate a second host copy for
    a node whose backup is already in flight: the ack bookkeeping tracks one
    pending backup per node, so a second executor would orphan the first."""

    def _cache(self, core):
        cache = object.__new__(UnifiedRadixCache)
        cache.buffer_pipeline = None
        cache.tree_core = core
        core.commit_backup = lambda *args: None
        executed = []
        cache._build_backup_sidecar = lambda *args: []

        def _execute(node_id, *args):
            executed.append(node_id)
            return torch.arange(len(core.node_by_id(node_id).key), dtype=torch.int64)

        cache._execute_kv_backup = _execute
        cache.inc_lock_ref = lambda node_id: SimpleNamespace(
            to_dec_params=lambda: DecLockRefParams()
        )
        cache._track_write_through_node = lambda node_id, params: (
            core.mark_write_through_pending(node_id)
        )
        return cache, executed

    def test_pending_node_is_backed_up_only_once(self):
        core = _tree()
        cache, executed = self._cache(core)
        _insert(core, [1, 2, 3, 4], chunked=True)
        _insert(core, [5, 6], chunked=True)
        parent = _node_for(core, [1, 2, 3, 4])
        child = _node_for(core, [5, 6])

        self.assertEqual(
            cache._execute_and_commit_kv_backup(BackupKV([parent.id, child.id])),
            2,
        )
        self.assertEqual(executed, [parent.id, child.id])

        # Both are pending now (in flight, host copy not committed): an
        # overlapping chain that reaches them again executes nothing.
        executed.clear()
        self.assertEqual(
            cache._execute_and_commit_kv_backup(BackupKV([parent.id, child.id])),
            0,
        )
        self.assertEqual(executed, [])


class TestSplitHostLockOwnership(CustomTestCase):
    def _host_backed_leaf(self, core, tokens):
        _insert(core, tokens)
        leaf = _node_for(core, tokens)
        cd = leaf.component_data[FULL]
        cd.host_value = torch.arange(0, len(tokens), dtype=torch.int64)
        return leaf, cd

    def test_split_keeps_the_in_flight_set_source_pinned(self):
        core = _tree()
        tokens = list(range(100, 110))
        leaf, cd = self._host_backed_leaf(core, tokens)
        receipt = core.inc_host_lock_ref(leaf.id)
        # The SET reads the whole pre-split span; the lock sits on the anchor.
        self.assertEqual(cd.host_lock_ref, 1)

        _insert(core, tokens[:4] + [999])
        fragment = _node_for(core, tokens[:4])
        self.assertEqual(
            fragment.component_data[FULL].host_lock_ref,
            1,
            "the prefix fragment must inherit the in-flight SET's host lock",
        )
        self.assertEqual(leaf.component_data[FULL].host_lock_ref, 1)

        core.dec_host_lock_ref(leaf.id, receipt.to_dec_params())
        self.assertEqual(fragment.component_data[FULL].host_lock_ref, 0)
        self.assertEqual(leaf.component_data[FULL].host_lock_ref, 0)

    def test_release_walks_every_split_fragment(self):
        core = _tree()
        tokens = list(range(200, 212))
        leaf, _ = self._host_backed_leaf(core, tokens)
        receipt = core.inc_host_lock_ref(leaf.id).to_dec_params()

        _insert(core, tokens[:3] + [888])
        _insert(core, tokens[:6] + [777])
        fragments = [
            _node_for(core, tokens[:3]),
            _node_for(core, tokens[3:6]),
            leaf,
        ]
        for node in fragments:
            self.assertEqual(node.component_data[FULL].host_lock_ref, 1)

        core.dec_host_lock_ref(leaf.id, receipt)
        for node in fragments:
            self.assertEqual(
                node.component_data[FULL].host_lock_ref,
                0,
                "one receipt must release exactly one lock per fragment it covered",
            )

    def test_overlapping_receipts_release_only_what_they_took(self):
        """The pre-split lock must not consume the post-split one."""
        core = _tree()
        tokens = list(range(300, 310))
        leaf, cd = self._host_backed_leaf(core, tokens)
        before_split = core.inc_host_lock_ref(leaf.id).to_dec_params()

        _insert(core, tokens[:4] + [555])
        fragment = _node_for(core, tokens[:4])
        after_split = core.inc_host_lock_ref(leaf.id).to_dec_params()
        self.assertEqual(cd.host_lock_ref, 2)

        core.dec_host_lock_ref(leaf.id, after_split)
        self.assertEqual(cd.host_lock_ref, 1)
        self.assertEqual(
            fragment.component_data[FULL].host_lock_ref,
            1,
            "releasing the anchor-local receipt must not reach the fragment",
        )

        core.dec_host_lock_ref(leaf.id, before_split)
        self.assertEqual(cd.host_lock_ref, 0)
        self.assertEqual(fragment.component_data[FULL].host_lock_ref, 0)

    def test_release_without_a_receipt_does_not_guess(self):
        core = _tree()
        tokens = list(range(400, 406))
        leaf, cd = self._host_backed_leaf(core, tokens)
        core.inc_host_lock_ref(leaf.id)
        core.dec_host_lock_ref(leaf.id, DecLockRefParams())
        self.assertEqual(
            cd.host_lock_ref,
            1,
            "a release without the acquisition result leaves the lock in place",
        )

    def test_aux_component_host_lock_still_releases_alone(self):
        """A node with no Full host copy pins only the aux pool; the same
        receipt must still release it."""
        core = _tree(with_aux=True)
        tokens = list(range(500, 506))
        _insert(core, tokens)
        leaf = _node_for(core, tokens)
        self.assertIsNone(leaf.component_data[FULL].host_value)

        receipt = core.inc_host_lock_ref(leaf.id).to_dec_params()
        self.assertIsNone(receipt.full_uuid_for_host_lock)
        self.assertEqual(leaf.component_data[MAMBA].host_lock_ref, 1)
        self.assertEqual(leaf.component_data[FULL].host_lock_ref, 0)

        core.dec_host_lock_ref(leaf.id, receipt)
        self.assertEqual(leaf.component_data[MAMBA].host_lock_ref, 0)
        self.assertEqual(leaf.component_data[FULL].host_lock_ref, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
