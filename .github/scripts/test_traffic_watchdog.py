import datetime as dt
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('watchdog', Path(__file__).with_name('traffic_watchdog.py'))
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
T = dt.datetime(2026, 9, 7, 2, 0, tzinfo=dt.timezone.utc)


class FakeGitHub:
    def __init__(self, collected='2026-09-06T01:35:52+00:00', status='completed',
                 conclusion='success', package_collected=None, hf_collected=None):
        self.collected = collected
        self.package_collected = package_collected or collected
        self.hf_collected = hf_collected or collected
        self.status = status
        self.conclusion = conclusion
        self.dispatch_count = 0
        self.active = []
        self.latest = []
        self.update = True
        self.update_package = True
        self.update_hf = True
        self.dispatch_failure = False

    def archive(self, branch):
        return {'head': 'history', 'collected_at': self.collected,
                'package_collected_at': self.package_collected,
                'hf_collected_at': self.hf_collected,
                'through_date': '2026-09-05'}

    def runs(self):
        return self.active + self.latest

    def dispatch(self):
        self.dispatch_count += 1
        if self.dispatch_failure:
            raise RuntimeError('connection failed')
        return 123

    def run(self, run_id):
        if self.status == 'completed' and self.conclusion == 'success' and self.update:
            self.collected = T.isoformat()
        if self.status == 'completed' and self.conclusion == 'success' and self.update_package:
            self.package_collected = T.isoformat()
        if self.status == 'completed' and self.conclusion == 'success' and self.update_hf:
            self.hf_collected = T.isoformat()
        return {'id': run_id, 'status': self.status, 'conclusion': self.conclusion,
                'html_url': 'https://github.com/test/repo/actions/runs/' + str(run_id)}


class WatchdogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = Path(self.tmp.name)
        self.clock = patch.object(w, 'now', return_value=T)
        self.clock.start()
        self.addCleanup(self.clock.stop)

    def verified(self):
        w.write_json(self.state / 'verified.json', {'run_id': 1})

    def test_deadline_uses_utc_calendar_day(self):
        self.assertEqual(w.required_cutoff(T.replace(hour=0, minute=16)).isoformat(), '2026-09-06T00:17:00+00:00')
        self.assertEqual(w.required_cutoff(T.replace(hour=0, minute=17)).isoformat(), '2026-09-07T00:17:00+00:00')

    def test_stale_archive_dispatches_and_verifies(self):
        self.verified()
        g = FakeGitHub()
        w.reconcile(g, self.state, 'traffic-history', wait_seconds=0)
        self.assertEqual(g.dispatch_count, 1)
        self.assertEqual(w.load_json(self.state / 'verified.json')['run_id'], 123)
        self.assertFalse((self.state / 'pending.json').exists())

    def test_fresh_archive_is_noop_after_commissioning(self):
        self.verified()
        g = FakeGitHub(T.isoformat())
        w.reconcile(g, self.state, 'traffic-history')
        self.assertEqual(g.dispatch_count, 0)

    def test_daily_chain_fails_if_another_watchdog_holds_the_lock(self):
        args = ['traffic_watchdog.py', '--repository', 'jpezzulli/sglang-rtxpro6000',
                '--state-dir', str(self.state), '--fail-if-locked']
        with patch.object(sys, 'argv', args), \
             patch.object(w.fcntl, 'flock', side_effect=BlockingIOError):
            with self.assertRaises(RuntimeError):
                w.main()

    def test_same_day_pre_deadline_capture_does_not_suppress_due_run(self):
        self.verified()
        g = FakeGitHub('2026-09-07T00:05:00+00:00')
        w.reconcile(g, self.state, 'traffic-history')
        self.assertEqual(g.dispatch_count, 1)

    def test_stale_package_archive_dispatches_when_traffic_is_fresh(self):
        self.verified()
        g = FakeGitHub(T.isoformat(), package_collected='2026-09-06T01:35:52+00:00')
        w.reconcile(g, self.state, 'traffic-history')
        self.assertEqual(g.dispatch_count, 1)

    def test_old_adopted_run_cannot_verify_stale_archive(self):
        g = FakeGitHub();g.update = False
        g.active = [{'id': 77, 'created_at': '2026-09-06T00:00:00Z', 'status': 'in_progress'}]
        with self.assertRaises(RuntimeError):
            w.reconcile(g, self.state, 'traffic-history')
        self.assertFalse((self.state / 'verified.json').exists())

    def test_first_timer_commissions_even_if_capture_fresh(self):
        g = FakeGitHub(T.isoformat())
        w.reconcile(g, self.state, 'traffic-history')
        self.assertEqual(g.dispatch_count, 1)

    def test_active_workflow_adopted_without_duplicate_dispatch(self):
        g = FakeGitHub()
        g.active = [{'id': 77, 'created_at': T.isoformat(), 'status': 'in_progress'}]
        w.reconcile(g, self.state, 'traffic-history')
        self.assertEqual(g.dispatch_count, 0)
        self.assertEqual(w.load_json(self.state / 'verified.json')['run_id'], 77)

    def test_timeout_retains_pending_and_resumes_same_run(self):
        g = FakeGitHub(status='queued')
        with self.assertRaises(RuntimeError):
            w.reconcile(g, self.state, 'traffic-history', wait_seconds=0)
        self.assertEqual(w.load_json(self.state / 'pending.json')['run_id'], 123)
        g.status = 'completed'
        w.reconcile(g, self.state, 'traffic-history', wait_seconds=0)
        self.assertEqual(g.dispatch_count, 1)

    def test_failure_preserves_verified_state_and_retries_next_tick(self):
        self.verified()
        g = FakeGitHub(conclusion='failure')
        with self.assertRaises(RuntimeError):
            w.reconcile(g, self.state, 'traffic-history')
        self.assertEqual(w.load_json(self.state / 'verified.json')['run_id'], 1)
        self.assertFalse((self.state / 'pending.json').exists())
        g.conclusion = 'success'
        w.reconcile(g, self.state, 'traffic-history')
        self.assertEqual(g.dispatch_count, 2)

    def test_success_without_archive_update_is_failure(self):
        g = FakeGitHub();g.update = False
        with self.assertRaises(RuntimeError):
            w.reconcile(g, self.state, 'traffic-history')
        self.assertFalse((self.state / 'verified.json').exists())

    def test_success_without_package_archive_update_is_failure(self):
        g = FakeGitHub();g.update_package = False
        with self.assertRaises(RuntimeError):
            w.reconcile(g, self.state, 'traffic-history')
        self.assertFalse((self.state / 'verified.json').exists())

    def test_stale_hf_archive_dispatches_when_traffic_and_package_fresh(self):
        self.verified()
        g = FakeGitHub(T.isoformat(), hf_collected='2026-09-06T01:35:52+00:00')
        w.reconcile(g, self.state, 'traffic-history')
        self.assertEqual(g.dispatch_count, 1)

    def test_success_without_hf_archive_update_is_failure(self):
        g = FakeGitHub();g.update_hf = False
        with self.assertRaises(RuntimeError):
            w.reconcile(g, self.state, 'traffic-history')
        self.assertFalse((self.state / 'verified.json').exists())

    def test_unknown_dispatch_persists_intent_and_adopts_registered_run(self):
        g = FakeGitHub();g.dispatch_failure = True
        with self.assertRaises(RuntimeError):
            w.reconcile(g, self.state, 'traffic-history')
        self.assertIsNone(w.load_json(self.state / 'pending.json')['run_id'])
        g.latest = [{'id': 88, 'status': 'completed', 'event': 'workflow_dispatch', 'created_at': T.isoformat()}]
        w.reconcile(g, self.state, 'traffic-history')
        self.assertEqual(g.dispatch_count, 1)
        self.assertEqual(w.load_json(self.state / 'verified.json')['run_id'], 88)

    def test_deleted_pending_run_retires_reference_and_recovers(self):
        g = FakeGitHub()
        w.write_json(self.state / 'pending.json', {'run_id': 99, 'requested_at': T.isoformat()})
        with patch.object(g, 'run', side_effect=w.MissingRunResource('404')):
            with self.assertRaises(RuntimeError):w.reconcile(g, self.state, 'traffic-history')
        self.assertFalse((self.state / 'pending.json').exists())
        w.reconcile(g, self.state, 'traffic-history')
        self.assertEqual(g.dispatch_count, 1)

    def test_daily_job_retries_retired_terminal_run_immediately(self):
        g = FakeGitHub()
        w.write_json(self.state / 'pending.json',
                     {'run_id': 77, 'requested_at': (T-dt.timedelta(days=1)).isoformat()})
        original_run = g.run
        def run(run_id):
            if run_id == 77:
                return {'id': 77, 'status': 'completed', 'conclusion': 'failure',
                        'html_url': 'https://github.com/test/repo/actions/runs/77'}
            return original_run(run_id)
        g.run = run
        w.reconcile(g, self.state, 'traffic-history', wait_seconds=0,
                    retry_retired_pending=True)
        self.assertEqual(g.dispatch_count, 1)
        self.assertEqual(w.load_json(self.state / 'verified.json')['run_id'], 123)

    def test_daily_job_does_not_retry_its_own_failed_dispatch(self):
        g = FakeGitHub(conclusion='failure')
        with self.assertRaises(RuntimeError):
            w.reconcile(g, self.state, 'traffic-history', wait_seconds=0,
                        retry_retired_pending=True)
        self.assertEqual(g.dispatch_count, 1)

    def test_lookup_authentication_failure_keeps_pending(self):
        g = FakeGitHub()
        w.write_json(self.state / 'pending.json', {'run_id': 99, 'requested_at': T.isoformat()})
        with patch.object(g, 'run', side_effect=RuntimeError('HTTP 403')):
            with self.assertRaises(RuntimeError):w.reconcile(g, self.state, 'traffic-history')
        self.assertEqual(w.load_json(self.state / 'pending.json')['run_id'], 99)
        self.assertEqual(g.dispatch_count, 0)

    def test_404_without_accessible_run_listing_keeps_pending(self):
        g = FakeGitHub()
        w.write_json(self.state / 'pending.json', {'run_id': 99, 'requested_at': T.isoformat()})
        with patch.object(g, 'run', side_effect=w.MissingRunResource('404')), patch.object(g, 'runs', side_effect=RuntimeError('HTTP 403')):
            with self.assertRaises(RuntimeError):w.reconcile(g, self.state, 'traffic-history')
        self.assertEqual(w.load_json(self.state / 'pending.json')['run_id'], 99)

    def test_api_distinguishes_not_found_from_auth_errors(self):
        from types import SimpleNamespace
        g = w.GitHub('test/repo', 'archive-traffic.yml')
        for status, error in [('404', w.MissingRunResource), ('403', RuntimeError), ('401', RuntimeError)]:
            result = SimpleNamespace(returncode=1, stdout='HTTP/2.0 ' + status + ' Error\n\n{}')
            with patch.object(g, 'command', return_value=result), self.assertRaises(error):
                g.api('actions/runs/99')
        result = SimpleNamespace(returncode=0, stdout='HTTP/2.0 200 OK\nX-Example: value\n\n{"id": 99}')
        with patch.object(g, 'command', return_value=result):
            self.assertEqual(g.api('actions/runs/99'), {'id': 99})

    def test_unknown_dispatch_does_not_retry_immediately(self):
        g = FakeGitHub();g.dispatch_failure = True
        with self.assertRaises(RuntimeError):w.reconcile(g, self.state, 'traffic-history')
        with self.assertRaises(RuntimeError):w.reconcile(g, self.state, 'traffic-history')
        self.assertEqual(g.dispatch_count, 1)

    def test_unknown_dispatch_expires_after_registration_grace(self):
        g = FakeGitHub()
        w.write_json(self.state / 'pending.json', {'run_id': None, 'requested_at': (T-dt.timedelta(minutes=20)).isoformat()})
        with self.assertRaises(RuntimeError):w.reconcile(g, self.state, 'traffic-history')
        self.assertFalse((self.state / 'pending.json').exists())
        self.assertEqual(g.dispatch_count, 0)


def _traffic_blob():
    import base64
    doc = {'repository': 'test/repo', 'schema_version': 1,
           'collected_at': T.isoformat(),
           'coverage': {'through_date': '2026-09-05'},
           'days': {'2026-09-05': {'views': {'count': 1, 'uniques': 1},
                                    'clones': {'count': 1, 'uniques': 1}}}}
    return {'encoding': 'base64', 'content': base64.b64encode(json.dumps(doc).encode()).decode()}


def _package_blob():
    import base64
    doc = {'repository': 'test/repo', 'schema_version': 1,
           'collected_at': T.isoformat(),
           'days': {'2026-09-05': {'total_downloads': 1}}}
    return {'encoding': 'base64', 'content': base64.b64encode(json.dumps(doc).encode()).decode()}


def _hf_blob(stamp=None, day=None):
    import base64
    moment = stamp or T.isoformat()
    day = day or moment[:10]
    doc = {'repository': 'test/repo', 'schema_version': 1,
           'first_collected_at': moment, 'collected_at': moment,
           'days': {day: {'downloads_all_time': 100, 'downloads_last_30_days': 90,
                          'likes': 5, 'collected_at': moment}}}
    return {'encoding': 'base64', 'content': base64.b64encode(json.dumps(doc).encode()).decode()}


class RealArchivePathTests(unittest.TestCase):
    """Exercise the actual GitHub.archive read path via a fake api (no subprocess)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = Path(self.tmp.name)

    def gh(self, tree_entries, blobs):
        g = w.GitHub('test/repo', 'archive-traffic.yml')
        def api(path):
            if path.startswith('git/ref/heads/'):
                return {'object': {'sha': 'head'}}
            if path == 'git/trees/head':
                return {'truncated': False, 'tree': tree_entries}
            if path.startswith('git/trees/'):
                return {'truncated': False, 'tree': blobs.get(path, [])}
            if path.startswith('git/blobs/'):
                sha = path[len('git/blobs/'):]
                if sha not in blobs:
                    raise AssertionError(f'unexpected blob: {sha}')
                return blobs[sha]
            raise AssertionError(path)
        g.api = api
        return g

    def test_uninitialized_hf_represents_as_needing_collection(self):
        """HF genuinely absent (no artifacts) → epoch timestamp, no raise."""
        entries = [
            {'path': 'daily.json', 'type': 'blob', 'sha': 'daily'},
            {'path': 'package-downloads.json', 'type': 'blob', 'sha': 'pkg'},
        ]
        blobs = {'daily': _traffic_blob(), 'pkg': _package_blob()}
        result = self.gh(entries, blobs).archive('traffic-history')
        self.assertEqual(result['hf_collected_at'], w._EPOCH)
        # Epoch is always older than any cutoff, so the collector must dispatch
        self.assertLess(w.timestamp(result['hf_collected_at']),
                        w.required_cutoff(T))

    def test_uninitialized_hf_dispatches_then_verifies_recovered_archive(self):
        """Reconcile dispatches when HF is epoch; verifies only after HF becomes fresh."""
        class HFPhaseGitHub(FakeGitHub):
            def __init__(self):
                super().__init__(T.isoformat())
                self.archive_calls = 0

            def archive(self, branch):
                self.archive_calls += 1
                if self.archive_calls == 1:
                    # First read: HF uninitialized (epoch), others fresh
                    return {'head': 'history', 'collected_at': T.isoformat(),
                            'package_collected_at': T.isoformat(),
                            'hf_collected_at': w._EPOCH,
                            'through_date': '2026-09-05'}
                # Post-dispatch read: HF recovered
                return {'head': 'history', 'collected_at': T.isoformat(),
                        'package_collected_at': T.isoformat(),
                        'hf_collected_at': T.isoformat(),
                        'through_date': '2026-09-05'}

        clock = patch.object(w, 'now', return_value=T)
        clock.start()
        self.addCleanup(clock.stop)
        g = HFPhaseGitHub()
        w.reconcile(g, self.state, 'traffic-history', wait_seconds=0)
        self.assertEqual(g.dispatch_count, 1)
        self.assertEqual(g.archive_calls, 2)
        self.assertTrue((self.state / 'verified.json').exists())

    def test_recovered_hf_never_silently_verified_without_fresh_hf(self):
        """Zero silent success: if HF stays epoch after run, verified.json is not written."""
        class HFStuckGitHub(FakeGitHub):
            def archive(self, branch):
                return {'head': 'history', 'collected_at': T.isoformat(),
                        'package_collected_at': T.isoformat(),
                        'hf_collected_at': w._EPOCH,
                        'through_date': '2026-09-05'}

        clock = patch.object(w, 'now', return_value=T)
        clock.start()
        self.addCleanup(clock.stop)
        w.write_json(self.state / 'verified.json', {'run_id': 1})
        g = HFStuckGitHub()
        with self.assertRaises(RuntimeError):
            w.reconcile(g, self.state, 'traffic-history', wait_seconds=0)
        # The stale-HF run must never be recorded as verified:
        # verified.json keeps its previous run_id, never the new dispatch.
        self.assertEqual(w.load_json(self.state / 'verified.json')['run_id'], 1)

    def test_established_but_lost_hf_fails_loud(self):
        """HF artifacts exist but canonical JSON gone: archive raises, never epoch."""
        entries = [
            {'path': 'daily.json', 'type': 'blob', 'sha': 'daily'},
            {'path': 'package-downloads.json', 'type': 'blob', 'sha': 'pkg'},
            {'path': 'HUGGINGFACE-DOWNLOADS.md', 'type': 'blob', 'sha': 'hf-md'},
        ]
        blobs = {'daily': _traffic_blob(), 'pkg': _package_blob()}
        with self.assertRaises(RuntimeError):
            self.gh(entries, blobs).archive('traffic-history')

    def test_corrupt_hf_canonical_fails_loud(self):
        """HF canonical present but invalid schema: archive raises."""
        import base64
        entries = [
            {'path': 'daily.json', 'type': 'blob', 'sha': 'daily'},
            {'path': 'package-downloads.json', 'type': 'blob', 'sha': 'pkg'},
            {'path': 'huggingface-downloads.json', 'type': 'blob', 'sha': 'hf'},
        ]
        blobs = {'daily': _traffic_blob(), 'pkg': _package_blob(),
                 'hf': {'encoding': 'base64',
                        'content': base64.b64encode(b'{"repository": "wrong"}').decode()}}
        with self.assertRaises(RuntimeError):
            self.gh(entries, blobs).archive('traffic-history')

    def test_fresh_hf_verified_normally(self):
        """All three archives fresh: normal return."""
        entries = [
            {'path': 'daily.json', 'type': 'blob', 'sha': 'daily'},
            {'path': 'package-downloads.json', 'type': 'blob', 'sha': 'pkg'},
            {'path': 'huggingface-downloads.json', 'type': 'blob', 'sha': 'hf'},
        ]
        blobs = {'daily': _traffic_blob(), 'pkg': _package_blob(), 'hf': _hf_blob()}
        result = self.gh(entries, blobs).archive('traffic-history')
        self.assertEqual(result['hf_collected_at'], w.timestamp(T.isoformat()).isoformat())

    def test_hf_future_row_cannot_be_marked_current(self):
        """Observation dated beyond the archive's collected_at fails the read path."""
        entries = [
            {'path': 'daily.json', 'type': 'blob', 'sha': 'daily'},
            {'path': 'package-downloads.json', 'type': 'blob', 'sha': 'pkg'},
            {'path': 'huggingface-downloads.json', 'type': 'blob', 'sha': 'hf'},
        ]
        blobs = {'daily': _traffic_blob(), 'pkg': _package_blob(),
                 'hf': _hf_blob(day='2099-01-01')}
        with self.assertRaises(RuntimeError):
            self.gh(entries, blobs).archive('traffic-history')

    def test_hf_missing_first_collected_at_cannot_be_marked_current(self):
        import base64
        entries = [
            {'path': 'daily.json', 'type': 'blob', 'sha': 'daily'},
            {'path': 'package-downloads.json', 'type': 'blob', 'sha': 'pkg'},
            {'path': 'huggingface-downloads.json', 'type': 'blob', 'sha': 'hf'},
        ]
        corrupt = json.loads(base64.b64decode(_hf_blob()['content']))
        del corrupt['first_collected_at']
        blobs = {'daily': _traffic_blob(), 'pkg': _package_blob(),
                 'hf': {'encoding': 'base64',
                        'content': base64.b64encode(json.dumps(corrupt).encode()).decode()}}
        with self.assertRaises(RuntimeError):
            self.gh(entries, blobs).archive('traffic-history')


if __name__ == '__main__':
    unittest.main()
