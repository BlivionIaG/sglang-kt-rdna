"""Focused Hugging Face model counter archive tests; no network calls."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

scripts = Path(__file__).parent
spec = importlib.util.spec_from_file_location('hf_archive', scripts / 'archive_hf_downloads.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def meta(all_time=1044, rolling30=1000, likes=8, created_at='2026-09-17T03:42:44.000Z'):
    return {'id': h.HF_MODEL_ID,
            'createdAt': created_at,
            'lastModified': '2026-09-20T10:00:00.000Z',
            'downloads': rolling30,
            'downloadsAllTime': all_time,
            'likes': likes}


class FakeAPI:
    """Fake Git Data API supporting tree/blob requests for HF tests."""

    def __init__(self, tree_entries=None, blobs=None):
        self.tree_entries = tree_entries if tree_entries is not None else [
            {'path': 'daily.json', 'type': 'blob', 'sha': 'daily-sha'},
        ]
        self.blobs = blobs or {}

    def request(self, path, method='GET', data=None, raw=False):
        if path.startswith('git/ref/heads/'):
            return {'object': {'sha': 'headsha'}}
        if path == 'git/commits/headsha':
            return {'tree': {'sha': 'treesha'}}
        if path == 'git/trees/treesha':
            return {'truncated': False, 'tree': self.tree_entries}
        if path.startswith('git/trees/'):
            sha = path[len('git/trees/'):]
            return {'truncated': False, 'tree': self.blobs.get(f'tree:{sha}', [])}
        if path.startswith('git/blobs/'):
            sha = path[len('git/blobs/'):]
            if sha not in self.blobs:
                raise AssertionError(f'unexpected blob request: {sha}')
            return self.blobs[sha]
        raise AssertionError(f'unexpected API call: {path}')


def b64_history(history):
    import base64
    return {'encoding': 'base64',
            'content': base64.b64encode(json.dumps(history).encode()).decode()}


class HFArchiveTests(unittest.TestCase):
    def test_validate_api_response_accepts_valid(self):
        h.validate_api_response(meta())

    def test_validate_rejects_bad_fields(self):
        cases = []
        r = meta(); r['id'] = 'wrong/model'; cases.append(r)
        r = meta(); r['downloadsAllTime'] = 'x'; cases.append(r)
        r = meta(); r['downloads'] = -1; cases.append(r)
        r = meta(); r['likes'] = True; cases.append(r)
        r = meta(); del r['createdAt']; cases.append(r)
        r = meta(); r['createdAt'] = 'not-a-date'; cases.append(r)
        for raw in cases:
            with self.subTest(id=raw.get('id'), dat=raw.get('downloadsAllTime')), \
                    self.assertRaises(h.ArchiveError):
                h.validate_api_response(raw)

    def test_rolling_exceeding_lifetime_rejected(self):
        r = meta(all_time=100, rolling30=200)
        with self.assertRaises(h.ArchiveError):
            h.validate_api_response(r)

    def test_merge_seed_preserves_first_observed(self):
        raw = json.dumps(meta(1044, 1000, 8)).encode()
        counts = h.extract_counts(meta(1044, 1000, 8))
        history, obs = h.merge(None, counts, raw, '2026-09-17T12:00:00Z', '2026-09-17T03:42:44.000Z')
        self.assertEqual(history['first_observed']['downloads_all_time'], 1044)
        self.assertEqual(history['first_observed']['likes'], 8)
        self.assertEqual(history['first_collected_at'], '2026-09-17T12:00:00Z')
        self.assertEqual(obs['source_sha256'], h.hashlib.sha256(raw).hexdigest())

    def test_merge_preserves_older_days_replaces_same_day(self):
        raw1 = json.dumps(meta(100, 90, 5)).encode()
        history, _ = h.merge(None, h.extract_counts(meta(100, 90, 5)), raw1,
                             '2026-09-20T01:00:00Z', '2026-09-17T03:42:44.000Z')
        raw2 = json.dumps(meta(110, 95, 6)).encode()
        history, _ = h.merge(history, h.extract_counts(meta(110, 95, 6)), raw2,
                             '2026-09-20T12:00:00Z', '2026-09-17T03:42:44.000Z')
        self.assertEqual(len(history['days']), 1)
        self.assertEqual(history['days']['2026-09-20']['downloads_all_time'], 110)
        raw3 = json.dumps(meta(120, 100, 7)).encode()
        history, _ = h.merge(history, h.extract_counts(meta(120, 100, 7)), raw3,
                             '2026-09-21T01:00:00Z', '2026-09-17T03:42:44.000Z')
        self.assertEqual(len(history['days']), 2)
        self.assertEqual(history['days']['2026-09-20']['downloads_all_time'], 110)
        self.assertEqual(history['first_collected_at'], '2026-09-20T01:00:00Z')

    def test_merge_rejects_stale_collection(self):
        raw = json.dumps(meta()).encode()
        history, _ = h.merge(None, h.extract_counts(meta()), raw,
                             '2026-09-20T12:00:00Z', '2026-09-17T03:42:44.000Z')
        with self.assertRaises(h.ArchiveError):
            h.merge(history, h.extract_counts(meta()), raw,
                    '2026-09-19T12:00:00Z', '2026-09-17T03:42:44.000Z')

    def test_merge_preserves_corrections_decrease(self):
        raw1 = json.dumps(meta(1044, 1000, 8)).encode()
        history, _ = h.merge(None, h.extract_counts(meta(1044, 1000, 8)), raw1,
                             '2026-09-20T01:00:00Z', '2026-09-17T03:42:44.000Z')
        raw2 = json.dumps(meta(1040, 900, 7)).encode()
        history, obs = h.merge(history, h.extract_counts(meta(1040, 900, 7)), raw2,
                               '2026-09-21T01:00:00Z', '2026-09-17T03:42:44.000Z')
        self.assertEqual(obs['downloads_all_time'], 1040)
        self.assertEqual(history['days']['2026-09-20']['downloads_all_time'], 1044)

    def test_merge_rejects_suspicious_zero_lifetime_replacing_positive(self):
        raw1 = json.dumps(meta(1044, 1000, 8)).encode()
        history, _ = h.merge(None, h.extract_counts(meta(1044, 1000, 8)), raw1,
                             '2026-09-20T01:00:00Z', '2026-09-17T03:42:44.000Z')
        raw2 = json.dumps(meta(0, 0, 0)).encode()
        with self.assertRaises(h.ArchiveError):
            h.merge(history, h.extract_counts(meta(0, 0, 0)), raw2,
                    '2026-09-21T01:00:00Z', '2026-09-17T03:42:44.000Z')

    def test_merge_allows_likes_dropping_to_zero(self):
        """Likes may legitimately drop from 8 to 0 (unlikes); only lifetime zero is suspicious."""
        raw1 = json.dumps(meta(1044, 1000, 8)).encode()
        history, _ = h.merge(None, h.extract_counts(meta(1044, 1000, 8)), raw1,
                             '2026-09-20T01:00:00Z', '2026-09-17T03:42:44.000Z')
        raw2 = json.dumps(meta(1044, 990, 0)).encode()
        history, obs = h.merge(history, h.extract_counts(meta(1044, 990, 0)), raw2,
                               '2026-09-21T01:00:00Z', '2026-09-17T03:42:44.000Z')
        self.assertEqual(obs['likes'], 0)
        self.assertEqual(obs['downloads_all_time'], 1044)

    def test_merge_rejects_created_at_change(self):
        """Same model ID reporting a different createdAt must fail, never reset provenance."""
        raw1 = json.dumps(meta(1044, 1000, 8)).encode()
        history, _ = h.merge(None, h.extract_counts(meta(1044, 1000, 8)), raw1,
                             '2026-09-20T01:00:00Z', '2026-09-17T03:42:44.000Z')
        raw2 = json.dumps(meta(1050, 1000, 8)).encode()
        with self.assertRaises(h.ArchiveError):
            h.merge(history, h.extract_counts(meta(1050, 1000, 8)), raw2,
                    '2026-09-21T01:00:00Z', '2026-01-01T00:00:00.000Z')

    def test_merge_accepts_first_observation_zero(self):
        raw = json.dumps(meta(0, 0, 0)).encode()
        counts = h.extract_counts(meta(0, 0, 0))
        history, _ = h.merge(None, counts, raw, '2026-09-17T12:00:00Z',
                             '2026-09-17T03:42:44.000Z')
        self.assertEqual(history['first_observed']['downloads_all_time'], 0)

    def test_validate_history_rejects_corrupt(self):
        raw = json.dumps(meta()).encode()
        history, _ = h.merge(None, h.extract_counts(meta()), raw,
                             '2026-09-17T12:00:00Z', '2026-09-17T03:42:44.000Z')
        corrupt = copy.deepcopy(history)
        corrupt['model_id'] = 'wrong'
        with self.assertRaises(h.ArchiveError):
            h.validate_hf_history(corrupt)
        corrupt = copy.deepcopy(history)
        corrupt['days'] = {}
        with self.assertRaises(h.ArchiveError):
            h.validate_hf_history(corrupt)
        corrupt = copy.deepcopy(history)
        corrupt['days']['2026-09-17']['downloads_all_time'] = 'x'
        with self.assertRaises(h.ArchiveError):
            h.validate_hf_history(corrupt)

    def test_validate_rejects_future_row_beyond_collected_at(self):
        """A day beyond top-level collected_at must not sort latest and mispublish."""
        raw = json.dumps(meta()).encode()
        history, _ = h.merge(None, h.extract_counts(meta()), raw,
                            '2026-10-04T01:00:00Z', '2026-09-17T03:42:44.000Z')
        # Simulate corrupt archive: an Oct 6 row inside an Oct 4 collection.
        history['days']['2026-10-06'] = {'downloads_all_time': 1,
                                         'downloads_last_30_days': 1, 'likes': 1,
                                         'collected_at': '2026-10-06T01:00:00Z',
                                         'source_sha256': history['days']['2026-10-04']['source_sha256']}
        with self.assertRaises(h.ArchiveError):
            h.validate_hf_history(history)
        # A later fetch (Oct 5) must reject this previous history instead of
        # accepting and publishing the Oct 6 row as latest.
        with self.assertRaises(h.ArchiveError):
            h.merge(history, h.extract_counts(meta(1050, 1000, 8)), raw,
                    '2026-10-05T01:00:00Z', '2026-09-17T03:42:44.000Z')

    def test_validate_rejects_canonical_key_row_date_mismatch(self):
        raw = json.dumps(meta()).encode()
        history, _ = h.merge(None, h.extract_counts(meta()), raw,
                             '2026-09-17T12:00:00Z', '2026-09-17T03:42:44.000Z')
        history['days']['2026-09-17']['collected_at'] = '2026-09-18T12:00:00Z'
        history['collected_at'] = '2026-09-18T12:00:00Z'
        with self.assertRaises(h.ArchiveError):
            h.validate_hf_history(history)

    def test_validate_rejects_impossible_timestamps(self):
        raw = json.dumps(meta()).encode()
        history, _ = h.merge(None, h.extract_counts(meta()), raw,
                             '2026-09-17T12:00:00Z', '2026-09-17T03:42:44.000Z')
        for value in ('2026-02-30T12:00:00Z', '9999-99-99T99:99:99Z', '2026-09-17T25:00:00Z'):
            corrupt = copy.deepcopy(history)
            corrupt['collected_at'] = value
            with self.subTest(value=value), self.assertRaises(h.ArchiveError):
                h.validate_hf_history(corrupt)
        # Observation timestamps are validated as real times too.
        corrupt = copy.deepcopy(history)
        corrupt['days']['2026-09-17']['collected_at'] = '2026-02-30T12:00:00Z'
        with self.assertRaises(h.ArchiveError):
            h.validate_hf_history(corrupt)

    def test_valid_same_day_replacement_and_history_unaffected(self):
        raw1 = json.dumps(meta(100, 90, 5)).encode()
        history, _ = h.merge(None, h.extract_counts(meta(100, 90, 5)), raw1,
                             '2026-09-20T01:00:00Z', '2026-09-17T03:42:44.000Z')
        raw2 = json.dumps(meta(110, 95, 6)).encode()
        history, _ = h.merge(history, h.extract_counts(meta(110, 95, 6)), raw2,
                             '2026-09-20T12:00:00Z', '2026-09-17T03:42:44.000Z')
        h.validate_hf_history(history)  # same-day replacement stays valid
        self.assertEqual(history['days']['2026-09-20']['collected_at'], '2026-09-20T12:00:00Z')

    def test_render_shows_separate_deltas(self):
        raw = json.dumps(meta(1044, 1000, 8)).encode()
        history, _ = h.merge(None, h.extract_counts(meta(1044, 1000, 8)), raw,
                             '2026-09-17T12:00:00Z', '2026-09-17T03:42:44.000Z')
        raw2 = json.dumps(meta(1050, 920, 9)).encode()
        history, _ = h.merge(history, h.extract_counts(meta(1050, 920, 9)), raw2,
                             '2026-09-18T12:00:00Z', '2026-09-17T03:42:44.000Z')
        text = h.render(history)
        self.assertIn('1,050', text)
        self.assertIn('+6', text)
        self.assertIn('920', text)
        self.assertIn('9', text)
        self.assertIn('+1', text)

    # --- read_hf_history path tests (correction 1) ---

    def test_read_hf_history_genuinely_uninitialized_returns_none(self):
        """No HF artifacts at all — seed allowed, history=None."""
        api = FakeAPI(tree_entries=[
            {'path': 'daily.json', 'type': 'blob', 'sha': 'daily-sha'},
            {'path': 'snapshots', 'type': 'tree', 'sha': 'snap-sha'},
            {'path': 'raw', 'type': 'tree', 'sha': 'raw-sha'},
        ], blobs={'tree:raw-sha': [
            {'path': 'first-run', 'type': 'tree', 'sha': 'fr-sha'},
            {'path': 'packages', 'type': 'tree', 'sha': 'pkg-sha'},
        ]})
        result = h.read_hf_history(api, 'treesha')
        self.assertIsNone(result)

    def test_read_hf_history_missing_canonical_with_existing_md_fails(self):
        """HUGGINGFACE-DOWNLOADS.md exists but canonical JSON missing — refuse reset."""
        api = FakeAPI(tree_entries=[
            {'path': 'daily.json', 'type': 'blob', 'sha': 'daily-sha'},
            {'path': 'HUGGINGFACE-DOWNLOADS.md', 'type': 'blob', 'sha': 'hf-md-sha'},
        ])
        with self.assertRaises(h.ArchiveError):
            h.read_hf_history(api, 'treesha')

    def test_read_hf_history_missing_canonical_with_existing_snapshots_fails(self):
        """huggingface-snapshots tree exists but canonical JSON missing — refuse reset."""
        api = FakeAPI(tree_entries=[
            {'path': 'daily.json', 'type': 'blob', 'sha': 'daily-sha'},
            {'path': 'huggingface-snapshots', 'type': 'tree', 'sha': 'hf-snap-sha'},
        ])
        with self.assertRaises(h.ArchiveError):
            h.read_hf_history(api, 'treesha')

    def test_read_hf_history_missing_canonical_with_raw_hf_subtree_fails(self):
        """raw/huggingface/ exists but canonical JSON missing — refuse reset."""
        api = FakeAPI(tree_entries=[
            {'path': 'daily.json', 'type': 'blob', 'sha': 'daily-sha'},
            {'path': 'raw', 'type': 'tree', 'sha': 'raw-sha'},
        ], blobs={'tree:raw-sha': [
            {'path': 'first-run', 'type': 'tree', 'sha': 'fr-sha'},
            {'path': 'huggingface', 'type': 'tree', 'sha': 'hf-raw-sha'},
        ]})
        with self.assertRaises(h.ArchiveError):
            h.read_hf_history(api, 'treesha')

    def test_read_hf_history_valid_canonical_returns_history(self):
        """Canonical JSON present and valid — returns history dict."""
        raw = json.dumps(meta()).encode()
        expected, _ = h.merge(None, h.extract_counts(meta()), raw,
                              '2026-09-17T12:00:00Z', '2026-09-17T03:42:44.000Z')
        api = FakeAPI(tree_entries=[
            {'path': 'huggingface-downloads.json', 'type': 'blob', 'sha': 'hf-json-sha'},
        ], blobs={'hf-json-sha': b64_history(expected)})
        result = h.read_hf_history(api, 'treesha')
        self.assertEqual(result, expected)

    def test_read_hf_history_corrupt_canonical_fails(self):
        """Canonical JSON present but corrupt — raises."""
        import base64
        api = FakeAPI(tree_entries=[
            {'path': 'huggingface-downloads.json', 'type': 'blob', 'sha': 'hf-json-sha'},
        ], blobs={'hf-json-sha': {'encoding': 'base64',
                                   'content': base64.b64encode(b'{bad json').decode()}})
        with self.assertRaises(h.ArchiveError):
            h.read_hf_history(api, 'treesha')

    # --- archive integration tests (corrections 1, 4, 5) ---

    def test_first_run_writes_raw_and_future_runs_do_not_replace(self):
        captured = []
        original_read, original_publish = h.read_hf_history, h.publish
        h.fetch_hf_meta = lambda: (json.dumps(meta()).encode(), meta())
        h.read_hf_history = lambda api, tree: None  # genuinely uninitialized
        h.publish = lambda api, head, tree, files, msg: captured.append(files) or 'sha'
        try:
            h.archive(FakeAPI(), '2026-09-17T12:00:00Z')
            self.assertIn('raw/huggingface/first-run/model.json', captured[0])
            self.assertIn('raw/huggingface/first-run/metadata.json', captured[0])
            self.assertIn('huggingface-downloads.json', captured[0])
            self.assertEqual(len([p for p in captured[0] if p.startswith('huggingface-snapshots/')]), 1)
            # Correction 4: snapshot must contain complete raw_response
            snap_path = next(p for p in captured[0] if p.startswith('huggingface-snapshots/'))
            snap = json.loads(captured[0][snap_path])
            self.assertIn('raw_response', snap)
            self.assertEqual(snap['raw_response'], json.dumps(meta()))
        finally:
            h.read_hf_history, h.publish = original_read, original_publish

    def test_later_run_no_raw_first_run(self):
        captured = []
        original_read, original_publish = h.read_hf_history, h.publish
        raw = json.dumps(meta()).encode()
        existing, _ = h.merge(None, h.extract_counts(meta()), raw,
                              '2026-09-17T12:00:00Z', '2026-09-17T03:42:44.000Z')
        h.fetch_hf_meta = lambda: (json.dumps(meta(1050, 1000, 9)).encode(), meta(1050, 1000, 9))
        h.read_hf_history = lambda api, tree: existing
        h.publish = lambda api, head, tree, files, msg: captured.append(files) or 'sha'
        try:
            h.archive(FakeAPI(), '2026-09-18T12:00:00Z')
            self.assertNotIn('raw/huggingface/first-run/model.json', captured[0])
            self.assertIn('huggingface-downloads.json', captured[0])
            # Snapshot has raw_response of the CURRENT run
            snap_path = next(p for p in captured[0] if p.startswith('huggingface-snapshots/'))
            snap = json.loads(captured[0][snap_path])
            self.assertIn('raw_response', snap)
            self.assertEqual(json.loads(snap['raw_response'])['downloadsAllTime'], 1050)
        finally:
            h.read_hf_history, h.publish = original_read, original_publish

    def test_established_but_lost_canonical_fails_not_resets(self):
        """read_hf_history with existing artifacts but missing canonical raises before publish."""
        raw = json.dumps(meta()).encode()
        h.fetch_hf_meta = lambda: (raw, meta())
        original_publish = h.publish
        published = []
        h.publish = lambda *args, **kw: published.append(args) or 'sha'
        try:
            # Fake API: no canonical but HUGGINGFACE-DOWNLOADS.md exists
            api = FakeAPI(tree_entries=[
                {'path': 'daily.json', 'type': 'blob', 'sha': 'daily-sha'},
                {'path': 'HUGGINGFACE-DOWNLOADS.md', 'type': 'blob', 'sha': 'md-sha'},
            ])
            with self.assertRaises(h.ArchiveError):
                h.archive(api, '2026-09-18T12:00:00Z')
            self.assertEqual(published, [], 'publish must not be called on failure')
        finally:
            h.publish = original_publish


class WorkflowPathTests(unittest.TestCase):
    """Verify the workflow uses Contents API (no checkout) for the HF collector."""

    def test_no_checkout_in_workflow(self):
        workflow = (scripts.parent / 'workflows' / 'archive-traffic.yml').read_text()
        self.assertNotIn('actions/checkout', workflow)
        self.assertIn('archive_hf_downloads.py', workflow)
        self.assertIn('repos/${GITHUB_REPOSITORY}/contents/.github/scripts/archive_hf_downloads.py?ref=${GITHUB_SHA}',
                      workflow)
        # HF script retrieval must be in the HF step, not the initial retrieval step
        # Verify by checking that the first retrieval step does NOT contain the HF script
        first_retrieval = workflow.split('- name: Collect, merge')[0]
        self.assertNotIn('archive_hf_downloads.py', first_retrieval,
                         'HF retrieval must not block traffic/package stages')
        # HF step ordering: after package
        hf_pos = workflow.index('Archive Hugging Face')
        pkg_pos = workflow.index('Archive GHCR package')
        traffic_pos = workflow.index('Collect, merge and atomically archive traffic')
        self.assertLess(traffic_pos, pkg_pos)
        self.assertLess(pkg_pos, hf_pos)


if __name__ == '__main__':
    unittest.main()
