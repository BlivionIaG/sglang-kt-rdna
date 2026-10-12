#!/usr/bin/env python3
"""Archive public Hugging Face model download counters using only the public metadata API."""
import base64
import copy
import datetime as dt
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request

from archive_traffic import (BRANCH, REPOSITORY, API, ArchiveError, encode,
                             publish, require)

HF_MODEL_ID = 'jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4'
HF_API_URL = (f'https://huggingface.co/api/models/{HF_MODEL_ID}'
              '?expand%5B%5D=createdAt&expand%5B%5D=lastModified'
              '&expand%5B%5D=downloads&expand%5B%5D=downloadsAllTime'
              '&expand%5B%5D=likes')
HF_HTML_URL = f'https://huggingface.co/{HF_MODEL_ID}'
HF_SCHEMA_VERSION = 1
HF_USER_AGENT = 'sglang-rtxpro6000-hf-archive'


def fetch_hf_meta():
    """Return (raw_bytes, parsed_dict) from a single unauthenticated HF API metadata request."""
    req = urllib.request.Request(HF_API_URL, headers={'User-Agent': HF_USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        raise ArchiveError(f'Hugging Face API HTTP {exc.code}') from None
    except (urllib.error.URLError, TimeoutError):
        raise ArchiveError('Hugging Face API transport failure') from None
    require(body, 'Empty Hugging Face API response')
    try:
        return body, json.loads(body)
    except (ValueError, UnicodeError):
        raise ArchiveError('Malformed Hugging Face API JSON') from None


def _meta_timestamp(value, label):
    require(isinstance(value, str) and
            re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z', value),
            f'Invalid {label}')
    dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    return value


def validate_api_response(raw):
    """Validate identity, dates, and numeric counter fields from an HF API response."""
    require(isinstance(raw, dict), 'Malformed Hugging Face API response')
    require(raw.get('id') == HF_MODEL_ID, f'Hugging Face model ID mismatch: {raw.get("id")}')
    _meta_timestamp(raw.get('createdAt'), 'Hugging Face createdAt')
    if 'lastModified' in raw:
        _meta_timestamp(raw['lastModified'], 'Hugging Face lastModified')
    for field in ('downloads', 'downloadsAllTime', 'likes'):
        value = raw.get(field)
        require(type(value) is int and value >= 0,
                f'Invalid Hugging Face {field}: {value!r}')
    require(raw['downloads'] <= raw['downloadsAllTime'],
            f'rolling30 ({raw["downloads"]}) exceeds lifetime ({raw["downloadsAllTime"]})')


def extract_counts(raw):
    return {'downloads_all_time': raw['downloadsAllTime'],
            'downloads_last_30_days': raw['downloads'],
            'likes': raw['likes']}


def validate_observation(obs):
    require(isinstance(obs, dict), 'Malformed Hugging Face observation')
    for field in ('downloads_all_time', 'downloads_last_30_days', 'likes'):
        value = obs.get(field)
        require(type(value) is int and value >= 0,
                f'Invalid Hugging Face observation {field}')
    require(obs['downloads_last_30_days'] <= obs['downloads_all_time'],
            'rolling30 exceeds lifetime in observation')
    require(isinstance(obs.get('source_sha256'), str) and
            re.fullmatch(r'[0-9a-f]{64}', obs['source_sha256']),
            'Invalid Hugging Face observation source hash')


def _strict_stamp(value, label):
    """Parse a real UTC timestamp; regex alone cannot reject impossible calendar dates."""
    require(isinstance(value, str) and
            re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z', value),
            f'Invalid {label}')
    try:
        parsed = dt.datetime.strptime(value, '%Y-%m-%dT%H:%M:%SZ')
    except ValueError:
        raise ArchiveError(f'Invalid {label}') from None
    return parsed.replace(tzinfo=dt.timezone.utc)


def validate_hf_history(history):
    require(isinstance(history, dict), 'Malformed Hugging Face history')
    required = {
        'schema_version': HF_SCHEMA_VERSION,
        'repository': REPOSITORY,
        'model_id': HF_MODEL_ID,
        'api_url': HF_API_URL,
        'html_url': HF_HTML_URL,
    }
    for key, expected in required.items():
        require(history.get(key) == expected,
                f'Hugging Face history metadata mismatch: {key}')
    require(isinstance(history.get('created_at'), str) and
            re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z',
                         history['created_at']),
            'Invalid Hugging Face model created_at')
    first = _strict_stamp(history.get('first_collected_at'),
                          'Hugging Face history first_collected_at')
    collected = _strict_stamp(history.get('collected_at'),
                              'Hugging Face history collected_at')
    require(first <= collected, 'Hugging Face timestamps out of order')
    require(isinstance(history.get('days'), dict) and history['days'],
            'Empty Hugging Face history')
    for day, obs in history['days'].items():
        require(dt.date.fromisoformat(day).isoformat() == day,
                'Invalid Hugging Face history date')
        validate_observation(obs)
        obs_at = _strict_stamp(obs.get('collected_at'),
                               f'Hugging Face observation {day} collected_at')
        require(obs_at.date().isoformat() == day,
                'Hugging Face canonical day does not match observation UTC date')
        require(first <= obs_at <= collected,
                'Hugging Face observation is outside the archive collection window')


def merge(previous, current_counts, raw_body, stamp, created_at):
    """Merge observed counters into history, preserving all prior days."""
    require(re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z', stamp),
            'Invalid collection stamp')
    for field in ('downloads_all_time', 'downloads_last_30_days', 'likes'):
        require(type(current_counts.get(field)) is int and current_counts[field] >= 0,
                f'Invalid counter: {field}')
    require(current_counts['downloads_last_30_days'] <= current_counts['downloads_all_time'],
            'rolling30 exceeds lifetime')
    if previous is not None:
        validate_hf_history(previous)
        require(stamp >= previous['collected_at'],
                'Hugging Face collection is older than latest archive')
        require(previous['created_at'] == created_at,
                'Hugging Face API reports different createdAt for the same model; '
                'preserving original provenance and refusing to reset')
        first = previous['first_collected_at']
        first_obs = copy.deepcopy(previous['first_observed'])
        days = copy.deepcopy(previous['days'])
        prev_obs = days.get(stamp[:10], previous['days'][max(previous['days'])])
        # Reject suspicious zero lifetime that would replace established positive history.
        # Likes may legitimately drop to zero (unlikes), so only guard lifetime.
        if (current_counts['downloads_all_time'] == 0
                and prev_obs.get('downloads_all_time', 0) > 0):
            raise ArchiveError(
                'Suspicious zero downloads_all_time would replace established positive history')
    else:
        first = stamp
        first_obs = {**current_counts, 'collected_at': stamp}
        days = {}
    observation = {
        'collected_at': stamp,
        'downloads_all_time': current_counts['downloads_all_time'],
        'downloads_last_30_days': current_counts['downloads_last_30_days'],
        'likes': current_counts['likes'],
        'source_sha256': hashlib.sha256(raw_body).hexdigest(),
    }
    validate_observation(observation)
    days[stamp[:10]] = observation
    result = {
        'schema_version': HF_SCHEMA_VERSION,
        'repository': REPOSITORY,
        'model_id': HF_MODEL_ID,
        'api_url': HF_API_URL,
        'html_url': HF_HTML_URL,
        'created_at': created_at,
        'first_collected_at': first,
        'collected_at': stamp,
        'first_observed': first_obs,
        'days': dict(sorted(days.items())),
    }
    return result, observation


HF_FIRST_RUN_PREFIX = 'raw/huggingface/first-run/'


def _hf_artifact_paths(roots):
    """Return True if any HF-related artifact exists in the tree."""
    return ('huggingface-downloads.json' in roots
            or 'HUGGINGFACE-DOWNLOADS.md' in roots
            or 'huggingface-snapshots' in roots
            or 'raw' in roots)  # raw subtree may contain huggingface/


def read_hf_history(api, tree_sha):
    """Return (history_or_None). None means genuinely uninitialized (seed allowed).

    Raises ArchiveError when established HF artifacts exist but the canonical
    JSON is missing or corrupt — this prevents silently resetting history.
    """
    tree = api.request(f'git/trees/{tree_sha}')
    require(not tree.get('truncated'), 'Truncated history root tree')
    roots = {entry['path']: entry for entry in tree['tree']}
    entry = roots.get('huggingface-downloads.json')
    if entry is None:
        # Seed is allowed ONLY when there are no prior HF artifacts at all.
        has_artifacts = ('HUGGINGFACE-DOWNLOADS.md' in roots
                         or 'huggingface-snapshots' in roots)
        # Check raw subtree for huggingface/ directory
        if not has_artifacts and 'raw' in roots:
            raw_tree = api.request(f'git/trees/{roots["raw"]["sha"]}')
            has_artifacts = any(e['path'] == 'huggingface' for e in raw_tree.get('tree', []))
        require(not has_artifacts,
                'Existing Hugging Face artifacts found but canonical JSON missing; '
                'refusing to reset history')
        return None
    blob = api.request('git/blobs/' + entry['sha'])
    require(blob['encoding'] == 'base64', 'Unsupported Hugging Face history blob encoding')
    try:
        history = json.loads(base64.b64decode(blob['content'], validate=False))
    except (ValueError, UnicodeError):
        raise ArchiveError('Malformed Hugging Face history JSON') from None
    validate_hf_history(history)
    return history


def delta(current, previous):
    return current - previous if previous is not None else None


def render(history):
    days = list(history['days'].items())
    _, latest = days[-1]
    previous_obs = days[-2][1] if len(days) > 1 else None
    lines = [
        '# Hugging Face model download archive', '',
        f'Model: [`{history["model_id"]}`]({history["html_url"]}).',
        f'Model created: **{history["created_at"]}**. First captured: **{history["first_collected_at"]}**.',
        f'Latest capture: **{latest["collected_at"]}**.', '',
        '| Metric | Latest reported | Change from previous archived day |',
        '| --- | ---: | ---: |',
    ]
    for field, label in [('downloads_all_time', 'All-time downloads (lifetime)'),
                          ('downloads_last_30_days', 'Rolling 30-day downloads'),
                          ('likes', 'Likes')]:
        change = delta(latest[field], previous_obs[field] if previous_obs else None)
        change_str = f'{change:+,}' if change is not None else '—'
        lines.append(f'| {label} | {latest[field]:,} | {change_str} |')
    lines += [
        '',
        'These are Hugging Face public API counters, not unique users or verified installations. '
        'The `downloads` field represents a rolling 30-day window; `downloadsAllTime` is cumulative '
        'since model creation. Exact daily download history before the first archived observation '
        'is unavailable and must never be reconstructed by summing rolling windows.', '',
        '## Daily observations', '',
        '| UTC date | Collected at | All-time | 30-day | Likes |',
        '| --- | --- | ---: | ---: | ---: |',
    ]
    for day, obs in days:
        lines.append(f'| {day} | {obs["collected_at"]} | '
                     f'{obs["downloads_all_time"]:,} | '
                     f'{obs["downloads_last_30_days"]:,} | '
                     f'{obs["likes"]:,} |')
    lines += ['', 'Provenance: unauthenticated public API metadata from',
              f'`{history["api_url"]}`',
              'No model weights, config files, or HEAD query files are fetched.', '',
              ]
    return '\n'.join(lines)


def archive(api, stamp):
    raw_body, raw_meta = fetch_hf_meta()
    validate_api_response(raw_meta)
    current_counts = extract_counts(raw_meta)
    created_at = raw_meta['createdAt']
    head = api.request(f'git/ref/heads/{BRANCH}')['object']['sha']
    commit = api.request(f'git/commits/{head}')
    tree_sha = commit['tree']['sha']
    previous = read_hf_history(api, tree_sha)
    seeding = previous is None
    history, observation = merge(previous, current_counts, raw_body, stamp, created_at)
    run_id = os.environ.get('GITHUB_RUN_ID', 'hf-seed')
    attempt = os.environ.get('GITHUB_RUN_ATTEMPT', '1')
    snapshot_path = (f'huggingface-snapshots/{stamp[:10]}/'
                     f'{stamp.replace(":", "")}-{run_id}-{attempt}.json')
    snapshot = {key: history[key] for key in (
        'schema_version', 'repository', 'model_id', 'api_url', 'html_url',
        'created_at', 'first_collected_at', 'collected_at')}
    snapshot.update(observation)
    snapshot['raw_response'] = raw_body.decode('utf-8')
    files = {
        'huggingface-downloads.json': encode(history),
        'HUGGINGFACE-DOWNLOADS.md': render(history),
        snapshot_path: encode(snapshot),
    }
    if seeding:
        files['raw/huggingface/first-run/model.json'] = raw_body.decode('utf-8')
        files['raw/huggingface/first-run/metadata.json'] = encode({
            'collected_at': stamp,
            'source_url': HF_API_URL,
            'sha256': hashlib.sha256(raw_body).hexdigest(),
            'model_id': HF_MODEL_ID,
            'created_at': created_at,
            'historical_daily_downloads_before_first_collection_available': False,
        })
    sha = publish(api, head, tree_sha, files,
                  f'Archive Hugging Face counters collected {stamp}')
    print(f'Archived Hugging Face model {HF_MODEL_ID}: '
          f'all_time={observation["downloads_all_time"]} '
          f'rolling30={observation["downloads_last_30_days"]} '
          f'likes={observation["likes"]}; history commit {sha}')
    if previous is not None:
        old = previous['days'][max(previous['days'])]
        if observation['downloads_all_time'] < old['downloads_all_time']:
            print('::warning::Hugging Face all-time download counter decreased')
    return sha


def main():
    require(os.environ.get('GITHUB_REPOSITORY') == REPOSITORY, 'Unexpected repository')
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    archive(API(os.environ.get('GITHUB_TOKEN')), stamp)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'::error::{type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)
