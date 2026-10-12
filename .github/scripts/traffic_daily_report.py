#!/usr/bin/env python3
"""Email one daily Pennyroyal traffic report as a read-only view of one pinned traffic-history commit."""
import argparse
import base64
import datetime as dt
import json
import subprocess
import sys
from urllib.parse import quote
from zoneinfo import ZoneInfo

from traffic_watchdog import GitHub, now, required_cutoff, timestamp


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def dates(first, last):
    current = dt.date.fromisoformat(first)
    stop = dt.date.fromisoformat(last)
    while current <= stop:
        yield current.isoformat()
        current += dt.timedelta(days=1)


def counts(value, label):
    require(isinstance(value, dict), f'Invalid {label}')
    result = {}
    for field in ('count', 'uniques'):
        require(type(value.get(field)) is int and value[field] >= 0,
                f'Invalid {label} {field}')
        result[field] = value[field]
    require(result['uniques'] <= result['count'], f'Invalid {label} uniques')
    return result


def observed(row):
    """Comparable daily counts only: last_collected_at churn is not a GitHub revision."""
    require(isinstance(row, dict), 'Invalid daily record')
    return {kind: counts(row.get(kind), f'daily {kind}') for kind in ('views', 'clones')}


def daily_totals(traffic, label, today=None):
    totals = {'views': 0, 'clones': 0, 'daily_unique_visitors': 0, 'daily_unique_cloners': 0}
    for day, row in traffic['days'].items():
        require(dt.date.fromisoformat(day).isoformat() == day, f'Invalid {label} date {day}')
        require(today is None or dt.date.fromisoformat(day) <= today,
                f'Traffic data is dated in the future: {day}')
        observed_row = observed(row)
        totals['views'] += observed_row['views']['count']
        totals['clones'] += observed_row['clones']['count']
        totals['daily_unique_visitors'] += observed_row['views']['uniques']
        totals['daily_unique_cloners'] += observed_row['clones']['uniques']
    return totals


def top_rows(rows, key, label):
    require(isinstance(rows, list), f'Invalid {label} list')
    result = []
    for row in rows[:5]:
        require(isinstance(row.get(key), str) and row[key], f'Invalid {label} name')
        value = counts(row, label)
        result.append((row[key], value['count'], value['uniques']))
    return result


def decode_blob(github, sha, label):
    blob = github.api('git/blobs/' + sha)
    require(blob.get('encoding') == 'base64' and blob.get('content'),
            f'Malformed traffic-history blob: {label}')
    try:
        return json.loads(base64.b64decode(blob['content']))
    except (ValueError, UnicodeError):
        raise RuntimeError(f'Malformed traffic-history JSON: {label}') from None


def read_documents(github, tree_sha, required_paths):
    # A commit SHA also resolves its root tree, so baseline and head read identically.
    tree = github.api('git/trees/' + tree_sha)
    require(not tree.get('truncated'), f'Truncated traffic-history tree: {tree_sha}')
    entries = {entry['path']: entry for entry in tree['tree'] if entry['type'] == 'blob'}
    documents = {}
    for path, is_required in required_paths.items():
        if path not in entries:
            require(not is_required, f'Required traffic-history file missing: {path}')
            documents[path] = None
        else:
            documents[path] = decode_blob(github, entries[path]['sha'], path)
    return documents


def validate_document(value, repository, label):
    require(isinstance(value, dict), f'Invalid {label} document')
    require(value.get('repository') == repository and value.get('schema_version') == 1,
            f'Invalid {label} archive identity/schema')
    require(isinstance(value.get('days'), dict) and value['days'], f'Empty {label} archive')
    timestamp(value.get('collected_at'))


def latest_record(document, fields, label, today=None):
    if document is None:
        return None, None
    day = max(document['days'])
    require(dt.date.fromisoformat(day).isoformat() == day, f'Invalid {label} date {day}')
    require(dt.date.fromisoformat(day) <= dt.date.fromisoformat(document['collected_at'][:10]),
            f'{label} record {day} is later than its own collection date')
    require(today is None or dt.date.fromisoformat(day) <= today,
            f'{label} record {day} is dated in the future')
    row = document['days'][day]
    require(isinstance(row, dict), f'Invalid {label} record')
    for field in fields:
        require(type(row.get(field)) is int and row[field] >= 0, f'Invalid {label} {field}')
    stamp = row.get('collected_at')
    if stamp is not None:
        parsed = None
        if isinstance(stamp, str):
            try:
                parsed = timestamp(stamp).date().isoformat()
            except ValueError:
                pass
        require(parsed == day, f'Invalid {label} row collection timestamp')
    return day, row


def rolling(value, kind, label):
    total = counts(value, label)
    rows = value.get(kind)
    require(isinstance(rows, list) and rows, f'Empty {label} series')
    days, count = {}, 0
    for row in rows:
        stamp = row.get('timestamp')
        require(isinstance(stamp, str) and stamp.endswith('T00:00:00Z'),
                f'Invalid {label} timestamp')
        day = stamp[:10]
        dt.date.fromisoformat(day)
        require(day not in days, f'Duplicate {label} date')
        days[day] = counts(row, f'{label} row')
        count += days[day]['count']
    require(count == total['count'], f'Inconsistent {label} total')
    require(max(row['uniques'] for row in days.values()) <= total['uniques'] <=
            sum(row['uniques'] for row in days.values()), f'Inconsistent {label} uniques')
    return total, days


def read_snapshot(github, head, traffic, today):
    """The one snapshot of the current collection: rolling totals, referrers, paths.

    Every snapshot date at or after launch must agree with the canonical archive row;
    legitimate prelaunch days GitHub exposes stay outside lifetime totals.
    """
    collected_at = traffic['collected_at']
    tree = github.api('git/trees/' + head + '?recursive=1')
    require(not tree.get('truncated'), 'Truncated traffic-history snapshot tree')
    prefix = f"snapshots/{collected_at[:10]}/{collected_at.replace(':', '')}-"
    candidates = sorted((entry for entry in tree['tree'] if entry['type'] == 'blob'
                         and entry['path'].startswith(prefix)), key=lambda entry: entry['path'])
    require(candidates, f'No traffic snapshot matches collection {collected_at}')
    snapshot = decode_blob(github, candidates[-1]['sha'], 'traffic snapshot')
    require(isinstance(snapshot, dict) and snapshot.get('repository') == github.repository
            and snapshot.get('schema_version') == 1, 'Invalid traffic snapshot identity/schema')
    require(snapshot.get('collected_at') == collected_at,
            'Traffic snapshot does not match the current collection')
    responses = snapshot.get('traffic')
    require(isinstance(responses, dict), 'Malformed traffic snapshot')
    views, view_days = rolling(responses.get('views'), 'views', 'snapshot rolling views')
    clones, clone_days = rolling(responses.get('clones'), 'clones', 'snapshot rolling clones')
    require(view_days.keys() == clone_days.keys(), 'Snapshot rolling views/clones windows differ')
    launch = dt.date.fromisoformat(traffic['launch_date'])
    for kind, days in (('views', view_days), ('clones', clone_days)):
        for day, row in days.items():
            require(dt.date.fromisoformat(day) <= today, f'Snapshot data is dated in the future: {day}')
            if dt.date.fromisoformat(day) < launch:
                continue
            require(day in traffic['days'] and
                    {field: traffic['days'][day][kind][field] for field in ('count', 'uniques')} == row,
                    f'Snapshot {kind} for {day} disagrees with the canonical archive')
    return {'views': views, 'clones': clones, 'window': sorted(view_days),
            'referrers': top_rows(responses.get('referrers'), 'referrer', 'referrer'),
            'paths': top_rows(responses.get('paths'), 'path', 'path')}


def read_current(github, branch):
    head = github.api('git/ref/heads/' + quote(branch, safe=''))['object']['sha']
    documents = read_documents(github, head, {path: True for path in (
        'daily.json', 'package-downloads.json', 'repository-metrics.json',
        'huggingface-downloads.json')})
    traffic = documents['daily.json']
    package = documents['package-downloads.json']
    metrics = documents['repository-metrics.json']
    hf = documents['huggingface-downloads.json']
    for name, value in [('traffic', traffic), ('package', package), ('repository metrics', metrics),
                        ('Hugging Face', hf)]:
        validate_document(value, github.repository, name)
    coverage = traffic.get('coverage')
    require(isinstance(coverage, dict) and coverage.get('complete_through_latest_exposed') is True and
            coverage.get('missing_dates_through_latest_exposed') == [],
            'Traffic archive coverage is incomplete')
    launch, through = traffic.get('launch_date'), coverage.get('through_date')
    require(isinstance(launch, str) and isinstance(through, str) and
            set(dates(launch, through)) == set(traffic['days']),
            'Traffic archive dates are incomplete or inconsistent')
    today = now().date()
    require(dt.date.fromisoformat(through) <= today, 'Traffic archive is dated in the future')
    cutoff = required_cutoff(now())
    for name, value in [('traffic', traffic), ('package', package), ('repository metrics', metrics),
                        ('Hugging Face', hf)]:
        require(timestamp(value['collected_at']) >= cutoff,
                f'Stale {name} archive; refusing to email a success report')
    return head, traffic, package, metrics, hf, read_snapshot(github, head, traffic, today)


def read_baseline(github, head, collected_at, launch, days):
    """The last archive commit before the current collection's UTC day, read as full prior state."""
    start = timestamp(collected_at).replace(hour=0, minute=0, second=0, microsecond=0)
    until = (start - dt.timedelta(seconds=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
    commits = github.api(f'commits?sha={head}&until={until}&per_page=1')
    require(isinstance(commits, list) and all(isinstance(commit, dict) and
            isinstance(commit.get('sha'), str) for commit in commits),
            'Malformed traffic-history commit listing')
    if not commits:
        return None
    documents = read_documents(github, commits[0]['sha'], {
        'daily.json': True, 'package-downloads.json': False, 'repository-metrics.json': False,
        'huggingface-downloads.json': False})
    daily = documents['daily.json']
    validate_document(daily, github.repository, 'baseline traffic')
    require(timestamp(daily['collected_at']) < timestamp(collected_at),
            'Baseline collection is not older than the current collection')
    coverage = daily.get('coverage')
    require(isinstance(coverage, dict) and coverage.get('complete_through_latest_exposed') is True and
            coverage.get('missing_dates_through_latest_exposed') == [],
            'Baseline traffic archive coverage is incomplete')
    through = coverage.get('through_date')
    require(daily.get('launch_date') == launch and isinstance(through, str) and
            set(dates(launch, through)) == set(daily['days']),
            'Baseline traffic archive dates are incomplete or inconsistent')
    require(set(daily['days']) <= set(days),
            'Current archive dropped dates retained by the baseline commit')
    for name in ('package-downloads.json', 'repository-metrics.json', 'huggingface-downloads.json'):
        if documents[name] is not None:
            validate_document(documents[name], github.repository, f'baseline {name}')
    return {'collected_at': daily['collected_at'], 'daily': daily,
            'package': documents['package-downloads.json'],
            'metrics': documents['repository-metrics.json'],
            'hf': documents['huggingface-downloads.json']}


def delta(current, previous):
    return '—' if previous is None else f'{current - previous:+,}'


def span(days):
    return days[0] if len(days) == 1 else f'{days[0]} through {days[-1]}'


def report(github, branch):
    head, traffic, package, metrics, hf, snapshot = read_current(github, branch)
    today = now().date()
    through = traffic['coverage']['through_date']
    totals = daily_totals(traffic, 'archive', today)
    yesterday = (today - dt.timedelta(days=1)).isoformat()
    pending = list(dates(through, yesterday))[1:] if through <= yesterday else []

    baseline = read_baseline(github, head, traffic['collected_at'], traffic['launch_date'],
                             traffic['days'])
    baseline_day = baseline['collected_at'][:10] if baseline else None
    baseline_totals = daily_totals(baseline['daily'], 'baseline archive') if baseline else None
    new_dates, revised = [], False
    if baseline:
        new_dates = sorted(set(traffic['days']) - set(baseline['daily']['days']))
        revised = any(observed(traffic['days'][day]) != observed(baseline['daily']['days'][day])
                      for day in set(traffic['days']) & set(baseline['daily']['days']))
    adoption_day, adoption = latest_record(metrics, ('stars', 'forks'), 'adoption', today)
    package_day, package_latest = latest_record(package, ('total_downloads',), 'package', today)
    hf_day, hf_latest = latest_record(hf, ('downloads_all_time', 'downloads_last_30_days', 'likes'),
                                       'Hugging Face', today)
    _, baseline_adoption = latest_record(baseline['metrics'] if baseline else None,
                                         ('stars', 'forks'), 'baseline adoption', today)
    _, baseline_package = latest_record(baseline['package'] if baseline else None,
                                        ('total_downloads',), 'baseline package', today)
    _, baseline_hf = latest_record(baseline['hf'] if baseline else None,
                                   ('downloads_all_time', 'downloads_last_30_days', 'likes'),
                                   'baseline Hugging Face', today)

    values = {
        'stars': adoption['stars'], 'forks': adoption['forks'],
        'views': totals['views'], 'clones': totals['clones'],
        'daily_unique_cloners': totals['daily_unique_cloners'],
        'package_downloads': package_latest['total_downloads'],
        'hf_all_time': hf_latest['downloads_all_time'],
        'hf_rolling30': hf_latest['downloads_last_30_days'],
        'hf_likes': hf_latest['likes'],
    }
    status = {'through_date': through, 'pending': pending, 'baseline_date': baseline_day,
              'new_dates': new_dates, 'revised': revised, 'partial_today': through == today.isoformat()}

    lines = [
        f'Last recorded traffic day: {through} UTC.',
        'Completed UTC days still pending from GitHub: ' + (', '.join(pending) or 'none') + '.',
    ]
    if baseline is None:
        lines.append('Baseline unavailable: no archive commit predates this collection, '
                     'so change figures are not available.')
    elif new_dates:
        lines.append('New traffic dates: ' + span(new_dates) + '.' +
                     (' Already-recorded dates were also revised by GitHub.' if revised else ''))
    elif revised:
        lines.append(f'No newer traffic date since the {baseline_day} collection; GitHub revised '
                     'recorded counts for already-recorded dates.')
    else:
        lines.append(f'No newer traffic date since the {baseline_day} collection.')
    lines.append('')
    if baseline:
        lines += [f'Changes since the {baseline_day} collection', '']
    lines += [
        'Adoption',
        f'- Stars: {values["stars"]:,} ({delta(values["stars"], baseline_adoption["stars"] if baseline_adoption else None)}), '
        f'recorded {adoption_day} UTC',
        f'- Forks: {values["forks"]:,} ({delta(values["forks"], baseline_adoption["forks"] if baseline_adoption else None)}), '
        f'recorded {adoption_day} UTC',
        f'- Container downloads: {values["package_downloads"]:,} '
        f'({delta(values["package_downloads"], baseline_package["total_downloads"] if baseline_package else None)}), '
        f'recorded {package_day} UTC', '',
        'Hugging Face model',
        f'- Model: {hf["model_id"]}',
        f'- All-time downloads: {values["hf_all_time"]:,} '
        f'({delta(values["hf_all_time"], baseline_hf["downloads_all_time"] if baseline_hf else None)}), '
        f'observed {hf_day} UTC',
        f'- Rolling 30-day downloads: {values["hf_rolling30"]:,} (separate window, not cumulative)',
        f'- Likes: {values["hf_likes"]:,} '
        f'({delta(values["hf_likes"], baseline_hf["likes"] if baseline_hf else None)}), '
        f'observed {hf_day} UTC', '',
        'Traffic since launch',
        f'- Views: {values["views"]:,} '
        f'({delta(values["views"], baseline_totals["views"] if baseline_totals else None)})',
        f'- Clones: {values["clones"]:,} '
        f'({delta(values["clones"], baseline_totals["clones"] if baseline_totals else None)})',
        f'- Sum of daily unique cloners: {values["daily_unique_cloners"]:,} '
        f'({delta(values["daily_unique_cloners"], baseline_totals["daily_unique_cloners"] if baseline_totals else None)})', '',
        f'{through} activity' + (' (partial UTC day so far; GitHub can still revise it)' if status['partial_today']
                                 else ' (last recorded completed UTC day)'),
        f'- Views: {traffic["days"][through]["views"]["count"]:,} from '
        f'{traffic["days"][through]["views"]["uniques"]:,} daily unique visitors',
        f'- Clones: {traffic["days"][through]["clones"]["count"]:,} from '
        f'{traffic["days"][through]["clones"]["uniques"]:,} daily unique cloners', '',
        'Rolling window returned by this collection: ' + span(snapshot['window']),
        f'- Views: {snapshot["views"]["count"]:,} total / {snapshot["views"]["uniques"]:,} unique visitors',
        f'- Clones: {snapshot["clones"]["count"]:,} total / {snapshot["clones"]["uniques"]:,} unique cloners', '',
        'Top referrers (same rolling window)',
    ]
    lines += [f'- {name}: {count:,} views / {uniques:,} unique visitors'
              for name, count, uniques in snapshot['referrers']] or ['- None returned']
    lines += ['', 'Top paths (same rolling window)']
    lines += [f'- {name}: {count:,} views / {uniques:,} unique visitors'
              for name, count, uniques in snapshot['paths']] or ['- None returned']
    lines += ['', 'Summed daily uniques re-count the same people across days; clones and '
              'container downloads are not distinct users.', '',
              f'https://github.com/{github.repository}/tree/{branch}']
    return '\n'.join(lines) + '\n', values, status


def subject_line(local_date, status):
    return f'Pennyroyal GitHub report — {local_date}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', required=True)
    parser.add_argument('--history-branch', default='traffic-history')
    parser.add_argument('--recipient', required=True)
    parser.add_argument('--timezone', default='America/New_York')
    parser.add_argument('--mail-command', default='/usr/bin/mail')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    local_date = now().astimezone(ZoneInfo(args.timezone)).date().isoformat()
    body, _, status = report(GitHub(args.repository, 'archive-traffic.yml'), args.history_branch)
    subject = subject_line(local_date, status)
    if args.dry_run:
        print(f'Subject: {subject}')
        print(body, end='')
        return
    result = subprocess.run([args.mail_command, '-s', subject, args.recipient], input=body,
                            text=True, timeout=60, check=False)
    if result.returncode:
        raise RuntimeError(f'Mail sender failed with exit {result.returncode}')
    print(f'Emailed daily traffic report for {local_date}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'TRAFFIC REPORT ERROR: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
