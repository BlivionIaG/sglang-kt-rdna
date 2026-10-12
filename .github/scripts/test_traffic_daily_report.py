"""Focused daily-email report tests; no network, no live traffic calls, no mail delivery."""
import base64
import contextlib
import importlib.util
import datetime as dt
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

scripts = Path(__file__).parent
spec = importlib.util.spec_from_file_location('report', scripts / 'traffic_daily_report.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)

REPOSITORY = 'jpezzulli/sglang-rtxpro6000'
NOW = dt.datetime(2026, 10, 3, 10, tzinfo=dt.timezone.utc)
CURRENT_STAMP = '2026-10-03T01:20:00Z'
BASELINE_STAMP = '2026-10-02T01:15:00Z'
SNAPSHOT_PATH = f'snapshots/{CURRENT_STAMP[:10]}/{CURRENT_STAMP.replace(":", "")}-1-1.json'


class FakeGitHub:
    repository = REPOSITORY

    def __init__(self, commits=None):
        self.trees, self.blobs = {}, {}
        self.commits = [{'sha': 'base'}] if commits is None else commits
        self.calls = []

    def api(self, path):
        self.calls.append(path)
        if path.startswith('traffic/'):
            raise AssertionError(f'the report must never call a live traffic endpoint: {path}')
        if path.startswith('git/ref/heads/'):
            return {'object': {'sha': 'head'}}
        if path.startswith('git/trees/'):
            name = path[len('git/trees/'):]
            if name not in self.trees:
                raise AssertionError(path)
            value = self.trees[name]
            return value if isinstance(value, dict) else {'truncated': False, 'tree': value}
        if path.startswith('git/blobs/'):
            sha = path[len('git/blobs/'):]
            if sha not in self.blobs:
                raise AssertionError(path)
            document = self.blobs[sha]
            if isinstance(document, dict) and '_raw' in document:
                return document['_raw']
            return {'encoding': 'base64',
                    'content': base64.b64encode(json.dumps(document).encode()).decode()}
        if path.startswith('commits?'):
            return self.commits
        raise AssertionError(path)


def pair(count, uniques):
    return {'count': count, 'uniques': uniques}


def day_row(views, clones, clone_uniques, views_uniques=None):
    return {'views': pair(views, views_uniques if views_uniques is not None else views // 8),
            'clones': pair(clones, clone_uniques)}


def daily_document(stamp, days, through):
    return {'repository': REPOSITORY, 'schema_version': 1, 'collected_at': stamp,
            'launch_date': '2026-09-20',
            'coverage': {'complete_through_latest_exposed': True,
                         'missing_dates_through_latest_exposed': [], 'through_date': through},
            'days': {day: {**row, 'last_collected_at': stamp} for day, row in days.items()}}


def metrics_document(stamp, days):
    return {'repository': REPOSITORY, 'schema_version': 1, 'collected_at': stamp, 'days': days}


def package_document(stamp, days):
    return {'repository': REPOSITORY, 'schema_version': 1, 'collected_at': stamp, 'days': days}


def hf_document(stamp, days, model_id='jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4'):
    return {'repository': REPOSITORY, 'schema_version': 1, 'collected_at': stamp,
            'model_id': model_id, 'days': days}


def hf_obs(all_time, rolling30, likes):
    return {'downloads_all_time': all_time, 'downloads_last_30_days': rolling30, 'likes': likes}


def snapshot_document(daily, window=None, stamp=None, extra_rows=()):
    """A coherent snapshot: its dated rows mirror the canonical rows collected with it."""
    days = daily['days']
    window = sorted(days)[-2:] if window is None else list(window)
    responses = {}
    for kind in ('views', 'clones'):
        rows, count, uniques = [], 0, 0
        for day in window:
            rows.append({'timestamp': f'{day}T00:00:00Z', **days[day][kind]})
            count += days[day][kind]['count']
            uniques += days[day][kind]['uniques']
        for day, value in extra_rows:
            rows.append({'timestamp': f'{day}T00:00:00Z', **value[kind]})
            count += value[kind]['count']
            uniques += value[kind]['uniques']
        responses[kind] = {'count': count, 'uniques': uniques, kind: rows}
    return {'repository': REPOSITORY, 'schema_version': 1, 'collected_at':
            daily['collected_at'] if stamp is None else stamp,
            'launch_date': daily['launch_date'],
            'traffic': {**responses,
                        'referrers': [{'referrer': 'google.com', 'count': 400, 'uniques': 200},
                                      {'referrer': 'reddit.com', 'count': 100, 'uniques': 90}],
                        'paths': [{'path': f'/{REPOSITORY}', 'count': 300, 'uniques': 150}]}}


def baseline_rows():
    rows = {day: day_row(800, 2640, 180, 100) for day in r.dates('2026-09-20', '2026-09-29')}
    rows['2026-09-30'] = day_row(1020, 2876, 259, 150)
    return rows  # totals: 9020 views / 29276 clones / 2059 daily unique cloners


def current_rows():
    rows = baseline_rows()
    rows['2026-10-01'] = day_row(159, 962, 82, 79)
    return rows  # totals: 9179 / 30238 / 2141 — the Oct 3 vs Oct 2 acceptance case


def head_documents(days=None, through='2026-10-01', stamp=CURRENT_STAMP):
    return {'daily.json': daily_document(stamp, current_rows() if days is None else days, through),
            'package-downloads.json': package_document(
                stamp, {'2026-10-02': {'total_downloads': 42}, '2026-10-03': {'total_downloads': 45}}),
            'repository-metrics.json': metrics_document(
                stamp, {'2026-10-02': {'stars': 11, 'forks': 3},
                        '2026-10-03': {'stars': 12, 'forks': 3}}),
            'huggingface-downloads.json': hf_document(
                stamp, {'2026-10-02': hf_obs(1044, 900, 8),
                        '2026-10-03': hf_obs(1050, 920, 9)})}


def base_documents(names=('daily.json', 'package-downloads.json', 'repository-metrics.json',
                          'huggingface-downloads.json')):
    documents = {'daily.json': daily_document(BASELINE_STAMP, baseline_rows(), '2026-09-30'),
                 'package-downloads.json': package_document(
                     BASELINE_STAMP, {'2026-10-02': {'total_downloads': 42}}),
                 'repository-metrics.json': metrics_document(
                     BASELINE_STAMP, {'2026-10-02': {'stars': 11, 'forks': 3}}),
                 'huggingface-downloads.json': hf_document(
                     BASELINE_STAMP, {'2026-10-02': hf_obs(1044, 900, 8)})}
    return {key: documents[key] for key in names}


def github_fixture(head=None, snapshots=None, base=None, commits=None):
    github = FakeGitHub(commits=commits)

    def add(label, documents):
        entries = []
        for path, document in documents.items():
            sha = f'{label}/{path}'
            github.blobs[sha] = document
            entries.append({'path': path, 'type': 'blob', 'sha': sha})
        return entries

    head_entries = add('head', head if head is not None else head_documents())
    github.trees['head'] = head_entries
    github.trees['head?recursive=1'] = head_entries + add('snapshot', snapshots or {})
    if base is not None:
        github.trees['base'] = add('base', base)
    return github


def standard_github(**kwargs):
    head = kwargs.get('head')
    if head is None:
        head = kwargs['head'] = head_documents()
    snapshots = kwargs.pop('snapshots', None)
    if snapshots is None:
        snapshots = {SNAPSHOT_PATH: snapshot_document(head['daily.json'])}
    return github_fixture(head=head, snapshots=snapshots,
                          base=kwargs.pop('base', base_documents()),
                          commits=kwargs.pop('commits', None))


class DailyReportTests(unittest.TestCase):
    def setUp(self):
        clock = patch.object(r, 'now', return_value=NOW)
        clock.start()
        self.addCleanup(clock.stop)

    def test_oct3_over_oct2_baseline_advances_with_pending_day(self):
        body, values, status = r.report(standard_github(), 'traffic-history')
        self.assertEqual(values, {'stars': 12, 'forks': 3, 'views': 9179, 'clones': 30238,
                                  'daily_unique_cloners': 2141, 'package_downloads': 45,
                                  'hf_all_time': 1050, 'hf_rolling30': 920, 'hf_likes': 9})
        self.assertIn('Last recorded traffic day: 2026-10-01 UTC.', body)
        self.assertIn('Completed UTC days still pending from GitHub: 2026-10-02.', body)
        self.assertIn('New traffic dates: 2026-10-01.', body)
        self.assertIn('Changes since the 2026-10-02 collection', body)
        self.assertIn('- Views: 9,179 (+159)', body)
        self.assertIn('- Clones: 30,238 (+962)', body)
        self.assertIn('- Sum of daily unique cloners: 2,141 (+82)', body)
        self.assertIn('- Stars: 12 (+1), recorded 2026-10-03 UTC', body)
        self.assertIn('- Forks: 3 (+0), recorded 2026-10-03 UTC', body)
        self.assertIn('- Container downloads: 45 (+3), recorded 2026-10-03 UTC', body)
        self.assertIn('Hugging Face model', body)
        self.assertIn('- Model: jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4', body)
        self.assertIn('- All-time downloads: 1,050 (+6), observed 2026-10-03 UTC', body)
        self.assertIn('- Rolling 30-day downloads: 920 (separate window, not cumulative)', body)
        self.assertIn('- Likes: 9 (+1), observed 2026-10-03 UTC', body)
        self.assertIn('2026-10-01 activity (last recorded completed UTC day)', body)
        self.assertIn('- Views: 159 from 79 daily unique visitors', body)
        self.assertIn('- Clones: 962 from 82 daily unique cloners', body)
        self.assertIn('Rolling window returned by this collection: 2026-09-30 through 2026-10-01', body)
        self.assertIn('- Views: 1,179 total / 229 unique visitors', body)
        self.assertIn('- Clones: 3,838 total / 341 unique cloners', body)
        self.assertIn('- google.com: 400 views / 200 unique visitors', body)
        self.assertIn('- /jpezzulli/sglang-rtxpro6000: 300 views / 150 unique visitors', body)
        self.assertIn('Summed daily uniques re-count the same people across days', body)
        self.assertEqual(body.count('not distinct users'), 1)
        self.assertEqual(status, {'through_date': '2026-10-01', 'pending': ['2026-10-02'],
                                  'baseline_date': '2026-10-02', 'new_dates': ['2026-10-01'],
                                  'revised': False, 'partial_today': False})
        self.assertEqual(r.subject_line('2026-10-03', status),
                         'Pennyroyal GitHub report — 2026-10-03')
        for noise in ('WARNING', 'stale', 'frozen', 'Archive commit', 'schema', 'API'):
            self.assertNotIn(noise, body)

    def test_snapshot_prelaunch_rows_stay_outside_lifetime_totals(self):
        daily = head_documents()['daily.json']
        snapshots = {SNAPSHOT_PATH: snapshot_document(
            daily, extra_rows=[('2026-08-22', {'views': pair(7, 3), 'clones': pair(9, 4)})])}
        body, values, _ = r.report(standard_github(snapshots=snapshots), 'traffic-history')
        self.assertEqual(values['views'], 9179)  # the prelaunch day never joins lifetime sums
        self.assertIn('- Views: 9,179 (+159)', body)
        self.assertIn('Rolling window returned by this collection: 2026-08-22 through 2026-10-01', body)

    def test_regressed_snapshot_window_is_allowed_and_accurately_dated(self):
        daily = head_documents()['daily.json']
        snapshots = {SNAPSHOT_PATH: snapshot_document(daily, window=['2026-09-29', '2026-09-30'])}
        body, values, _ = r.report(standard_github(snapshots=snapshots), 'traffic-history')
        self.assertIn('Rolling window returned by this collection: 2026-09-29 through 2026-09-30', body)
        self.assertIn('- Views: 1,820 total / 250 unique visitors', body)  # 800+1020 / 100+150
        self.assertIn('- Clones: 5,516 total / 439 unique cloners', body)  # 2640+2876 / 180+259
        self.assertIn('Last recorded traffic day: 2026-10-01 UTC.', body)
        self.assertIn('- Views: 9,179 (+159)', body)  # the rolling window is never a lifetime total

    def test_two_reads_of_one_pinned_archive_are_deterministic(self):
        github = standard_github()
        first = r.report(github, 'traffic-history')
        second = r.report(github, 'traffic-history')
        self.assertEqual(first, second)
        self.assertIn('commits?sha=head&until=2026-10-02T23:59:59Z&per_page=1', github.calls)
        self.assertTrue(all(not call.startswith('traffic/') for call in github.calls))

    def test_identical_repeat_reports_no_newer_traffic_date(self):
        github = standard_github(head=head_documents(days=baseline_rows(), through='2026-09-30'))
        body, values, _ = r.report(github, 'traffic-history')
        self.assertIn('No newer traffic date since the 2026-10-02 collection.', body)
        self.assertNotIn('New traffic dates', body)
        self.assertNotIn('revised', body)
        self.assertIn('- Views: 9,020 (+0)', body)
        self.assertIn('- Clones: 29,276 (+0)', body)
        self.assertIn('- Sum of daily unique cloners: 2,059 (+0)', body)
        self.assertIn('Completed UTC days still pending from GitHub: 2026-10-01, 2026-10-02.', body)

    def test_same_date_github_correction_says_counts_revised(self):
        days = baseline_rows()
        days['2026-09-30'] = day_row(1031, 2870, 255, 160)
        github = standard_github(head=head_documents(days=days, through='2026-09-30'))
        body, values, status = r.report(github, 'traffic-history')
        self.assertEqual(values['views'], 9031)
        self.assertEqual(values['clones'], 29270)
        self.assertEqual(values['daily_unique_cloners'], 2055)
        self.assertTrue(status['revised'])
        self.assertIn('No newer traffic date since the 2026-10-02 collection; GitHub revised '
                      'recorded counts for already-recorded dates.', body)
        self.assertIn('- Views: 9,031 (+11)', body)
        self.assertIn('- Clones: 29,270 (-6)', body)  # negative corrections are honest deltas
        self.assertIn('- Sum of daily unique cloners: 2,055 (-4)', body)
        self.assertIn('2026-09-30 activity (last recorded completed UTC day)', body)
        self.assertIn('- Views: 1,031 from 160 daily unique visitors', body)

    def test_multi_day_backfill_lists_every_new_date_once(self):
        days = {day: row for day, row in baseline_rows().items() if day <= '2026-09-28'}
        for day in ('2026-09-29', '2026-09-30', '2026-10-01'):
            days[day] = day_row(159, 962, 82, 60)
        base = base_documents()
        base['daily.json'] = daily_document(BASELINE_STAMP,
                                            {day: row for day, row in days.items() if day <= '2026-09-28'},
                                            '2026-09-28')
        github = standard_github(head=head_documents(days=days), base=base)
        body, values, status = r.report(github, 'traffic-history')
        self.assertEqual(status['new_dates'], ['2026-09-29', '2026-09-30', '2026-10-01'])
        self.assertIn('New traffic dates: 2026-09-29 through 2026-10-01.', body)
        self.assertEqual(values['views'], 7677)  # 9 * 800 + 3 * 159
        self.assertIn('- Views: 7,677 (+477)', body)
        self.assertIn('- Sum of daily unique cloners: 1,866 (+246)', body)  # 9*180+3*82 vs 9*180
        self.assertIn('Completed UTC days still pending from GitHub: 2026-10-02.', body)

    def test_current_partial_today_is_dated_and_labelled_partial(self):
        days = current_rows()
        days['2026-10-02'] = day_row(240, 900, 70, 90)
        days['2026-10-03'] = day_row(12, 20, 7, 5)
        github = standard_github(head=head_documents(days=days, through='2026-10-03'))
        body, values, status = r.report(github, 'traffic-history')
        self.assertIn('Last recorded traffic day: 2026-10-03 UTC.', body)
        self.assertIn('2026-10-03 activity (partial UTC day so far; GitHub can still revise it)', body)
        self.assertIn('Completed UTC days still pending from GitHub: none.', body)
        self.assertIn('New traffic dates: 2026-10-01 through 2026-10-03.', body)
        self.assertTrue(status['partial_today'])
        self.assertIn('- Views: 9,431 (+411)', body)
        self.assertIn('- Sum of daily unique cloners: 2,218 (+159)', body)

    def test_no_baseline_never_guesses_growth(self):
        github = standard_github(commits=[])
        body, values, status = r.report(github, 'traffic-history')
        self.assertIn('Baseline unavailable', body)
        self.assertNotIn('New traffic dates', body)
        self.assertNotIn('No newer traffic date', body)
        self.assertIn('- Views: 9,179 (—)', body)
        self.assertIn('- Stars: 12 (—), recorded 2026-10-03 UTC', body)
        self.assertNotIn('(+', body)
        self.assertIsNone(status['baseline_date'])
        self.assertIn('Completed UTC days still pending from GitHub: 2026-10-02.', body)

    def test_baseline_without_optional_documents_still_compares_traffic(self):
        github = standard_github(base=base_documents(('daily.json',)))
        body, _, _ = r.report(github, 'traffic-history')
        self.assertIn('- Views: 9,179 (+159)', body)
        self.assertIn('- Stars: 12 (—)', body)
        self.assertIn('- Container downloads: 45 (—)', body)
        self.assertIn('- All-time downloads: 1,050 (—)', body)
        self.assertIn('- Likes: 9 (—)', body)

    def test_pending_dates_are_exact_and_never_zero_filled(self):
        days = {day: row for day, row in baseline_rows().items() if day <= '2026-09-29'}
        base = base_documents()
        base['daily.json'] = daily_document(BASELINE_STAMP, days, '2026-09-29')
        github = standard_github(head=head_documents(days=days, through='2026-09-29'), base=base)
        body, _, status = r.report(github, 'traffic-history')
        self.assertEqual(status['pending'], ['2026-09-30', '2026-10-01', '2026-10-02'])
        self.assertIn('Completed UTC days still pending from GitHub: 2026-09-30, 2026-10-01, '
                      '2026-10-02.', body)

    def test_matching_row_collection_timestamp_is_accepted(self):
        github = standard_github()
        github.blobs['head/repository-metrics.json']['days']['2026-10-03'][
            'collected_at'] = CURRENT_STAMP
        body, values, _ = r.report(github, 'traffic-history')
        self.assertIn('- Stars: 12 (+1), recorded 2026-10-03 UTC', body)

    def test_fail_loud_rejections(self):
        def mutation(name):
            if name == 'missing_daily':
                head = {key: value for key, value in head_documents().items() if key != 'daily.json'}
                return standard_github(head=head,
                                       snapshots={SNAPSHOT_PATH: snapshot_document(
                                           head_documents()['daily.json'])})
            if name == 'malformed_blob':
                github = standard_github()
                github.blobs['head/daily.json'] = {'_raw': {'encoding': 'json', 'content': '{}'}}
                return github
            if name == 'corrupt_json':
                github = standard_github()
                github.blobs['head/daily.json'] = {'_raw': {
                    'encoding': 'base64', 'content': base64.b64encode(b'{not json').decode()}}
                return github
            if name == 'wrong_repository':
                github = standard_github()
                github.blobs['head/daily.json'] = {**github.blobs['head/daily.json'],
                                                   'repository': 'other/repo'}
                return github
            if name == 'incomplete_coverage':
                github = standard_github()
                document = github.blobs['head/daily.json']
                document['coverage']['complete_through_latest_exposed'] = False
                return github
            if name == 'coverage_date_mismatch':
                github = standard_github()
                github.blobs['head/daily.json']['coverage']['through_date'] = '2026-10-02'
                return github
            if name == 'pre_cutoff_collection':
                github = standard_github()
                github.blobs['head/daily.json']['collected_at'] = '2026-10-02T23:00:00Z'
                return github
            if name == 'future_day':
                days = current_rows()
                days['2026-10-04'] = day_row(1, 1, 1, 1)
                return standard_github(head=head_documents(days=days, through='2026-10-04'))
            if name == 'missing_snapshot':
                return standard_github(snapshots={})
            if name == 'mismatched_snapshot':
                github = standard_github()
                github.blobs[f'snapshot/{SNAPSHOT_PATH}']['collected_at'] = BASELINE_STAMP
                return github
            if name == 'snapshot_row_disagrees_with_canonical':
                github = standard_github()
                responses = github.blobs[f'snapshot/{SNAPSHOT_PATH}']['traffic']
                for row in responses['views']['views']:
                    if row['timestamp'].startswith('2026-10-01'):
                        row['count'] = 999
                responses['views']['count'] = sum(row['count'] for row in responses['views']['views'])
                return github
            if name == 'snapshot_future_row':
                daily = head_documents()['daily.json']
                snapshots = {SNAPSHOT_PATH: snapshot_document(
                    daily, extra_rows=[('2026-10-04', {'views': pair(5, 1), 'clones': pair(6, 2)})])}
                return standard_github(snapshots=snapshots)
            if name == 'rolling_total_mismatch':
                github = standard_github()
                github.blobs[f'snapshot/{SNAPSHOT_PATH}']['traffic']['views']['count'] += 1
                return github
            if name == 'truncated_head_tree':
                github = standard_github()
                github.trees['head'] = {'truncated': True, 'tree': github.trees['head']}
                return github
            if name == 'truncated_snapshot_tree':
                github = standard_github()
                github.trees['head?recursive=1'] = {
                    'truncated': True, 'tree': github.trees['head?recursive=1']}
                return github
            if name == 'missing_baseline_daily':
                return standard_github(base=base_documents(('package-downloads.json',
                                                             'repository-metrics.json')))
            if name == 'malformed_baseline':
                github = standard_github()
                github.blobs['base/daily.json'] = {**github.blobs['base/daily.json'],
                                                   'schema_version': 2}
                return github
            if name == 'baseline_not_older':
                github = standard_github()
                github.blobs['base/daily.json']['collected_at'] = CURRENT_STAMP
                return github
            if name == 'baseline_missing_interior_date':  # a lost row must fail, not fabricate growth
                github = standard_github()
                del github.blobs['base/daily.json']['days']['2026-09-25']
                return github
            if name == 'baseline_incomplete_coverage':
                github = standard_github()
                github.blobs['base/daily.json']['coverage'][
                    'complete_through_latest_exposed'] = False
                return github
            if name == 'baseline_launch_mismatch':
                github = standard_github()
                github.blobs['base/daily.json']['launch_date'] = '2026-08-24'
                return github
            if name == 'current_dropped_baseline_dates':
                base = base_documents()
                base['daily.json'] = daily_document(BASELINE_STAMP, current_rows(), '2026-10-01')
                return standard_github(head=head_documents(days=baseline_rows(), through='2026-09-30'),
                                       base=base)
            if name == 'rolling_uniques_below_daily_max':
                github = standard_github()
                github.blobs[f'snapshot/{SNAPSHOT_PATH}']['traffic']['views']['uniques'] = 40
                return github
            if name == 'rolling_uniques_above_daily_sum':
                github = standard_github()
                github.blobs[f'snapshot/{SNAPSHOT_PATH}']['traffic']['views']['uniques'] = 300
                return github
            if name == 'metrics_row_timestamp_conflicts':
                github = standard_github()
                github.blobs['head/repository-metrics.json']['days']['2026-10-03'][
                    'collected_at'] = '2026-10-01T05:00:00Z'
                return github
            if name == 'current_adoption_row_after_collection':
                github = standard_github()
                github.blobs['head/repository-metrics.json']['days']['2026-10-04'] = {
                    'stars': 13, 'forks': 3}
                return github
            if name == 'baseline_package_row_after_collection':
                github = standard_github()
                github.blobs['base/package-downloads.json']['days']['2026-10-03'] = {
                    'total_downloads': 44}
                return github
            if name == 'missing_hf':
                head = {key: value for key, value in head_documents().items() if key != 'huggingface-downloads.json'}
                return standard_github(head=head,
                                       snapshots={SNAPSHOT_PATH: snapshot_document(
                                           head_documents()['daily.json'])})
            if name == 'stale_hf':
                github = standard_github()
                github.blobs['head/huggingface-downloads.json']['collected_at'] = '2026-10-02T23:00:00Z'
                return github
            if name == 'hf_wrong_repository':
                github = standard_github()
                github.blobs['head/huggingface-downloads.json'] = {**github.blobs['head/huggingface-downloads.json'],
                                                                     'repository': 'other/repo'}
                return github
            if name == 'hf_empty_days':
                github = standard_github()
                github.blobs['head/huggingface-downloads.json']['days'] = {}
                return github
            raise AssertionError(name)

        for name in ('missing_daily', 'malformed_blob', 'corrupt_json', 'wrong_repository',
                     'incomplete_coverage', 'coverage_date_mismatch', 'pre_cutoff_collection',
                     'future_day', 'missing_snapshot', 'mismatched_snapshot',
                     'rolling_total_mismatch', 'truncated_head_tree', 'truncated_snapshot_tree',
                     'missing_baseline_daily', 'malformed_baseline', 'baseline_not_older',
                     'snapshot_row_disagrees_with_canonical', 'snapshot_future_row',
                     'baseline_missing_interior_date', 'baseline_incomplete_coverage',
                     'baseline_launch_mismatch', 'current_dropped_baseline_dates',
                     'rolling_uniques_below_daily_max', 'rolling_uniques_above_daily_sum',
                     'metrics_row_timestamp_conflicts', 'current_adoption_row_after_collection',
                     'baseline_package_row_after_collection',
                     'missing_hf', 'stale_hf', 'hf_wrong_repository', 'hf_empty_days'):
            with self.subTest(name), self.assertRaises(RuntimeError):
                r.report(mutation(name), 'traffic-history')

    def test_main_emails_stable_subject_without_sending(self):
        run = patch.object(r.subprocess, 'run', return_value=Mock(returncode=0))
        argv = patch.object(sys, 'argv', ['traffic_daily_report.py', '--repository',
                                          REPOSITORY, '--recipient', 'j@example.com',
                                          '--mail-command', '/usr/bin/mail'])
        with patch.object(r, 'GitHub', return_value=standard_github()), run as sent, argv, \
                contextlib.redirect_stdout(io.StringIO()):
            r.main()
        args, kwargs = sent.call_args
        self.assertEqual(['/usr/bin/mail', '-s', 'Pennyroyal GitHub report — 2026-10-03',
                          'j@example.com'], args[0])
        self.assertIn('New traffic dates: 2026-10-01.', kwargs['input'])

    def test_main_dry_run_prints_subject_and_sends_nothing(self):
        run = patch.object(r.subprocess, 'run')
        argv = patch.object(sys, 'argv', ['traffic_daily_report.py', '--repository',
                                          REPOSITORY, '--recipient', 'j@example.com', '--dry-run'])
        with patch.object(r, 'GitHub', return_value=standard_github()), run as sent, argv, \
                contextlib.redirect_stdout(io.StringIO()) as out:
            r.main()
        text = out.getvalue()
        sent.assert_not_called()
        self.assertIn('Subject: Pennyroyal GitHub report — 2026-10-03', text)
        self.assertIn('Changes since the 2026-10-02 collection', text)

    def test_hf_no_baseline_shows_unavailable_not_initial_value(self):
        """First baseline unavailable: no prior HF doc means delta is unavailable, never +1044."""
        github = standard_github(base=base_documents(('daily.json',)))
        body, values, _ = r.report(github, 'traffic-history')
        self.assertIn('- All-time downloads: 1,050 (—)', body)
        self.assertIn('- Likes: 9 (—)', body)
        self.assertNotIn('+1044', body)
        self.assertNotIn('+1050', body)

    def test_hf_separate_from_github_clones_no_combined_people_count(self):
        body, values, _ = r.report(standard_github(), 'traffic-history')
        self.assertIn('Hugging Face model', body)
        self.assertIn('- All-time downloads: 1,050 (+6)', body)
        self.assertIn('- Rolling 30-day downloads: 920 (separate window, not cumulative)', body)
        self.assertIn('- Likes: 9 (+1)', body)
        self.assertIn('- Clones: 30,238', body)
        self.assertNotIn('combined', body.lower())


if __name__ == '__main__':
    unittest.main()
