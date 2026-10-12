# Permanent GitHub traffic archive

The workflow `.github/workflows/archive-traffic.yml` runs at **00:17 UTC daily**
(`17 0 * * *`) on `pennyroyal-main-sm120-final` and supports manual dispatch.
Scheduled execution can be delayed or dropped by GitHub. An independent persistent
timer on thegrid now ensures a daily capture (details below). Run manually with:

```sh
gh workflow run archive-traffic.yml --repo jpezzulli/sglang-rtxpro6000 --ref pennyroyal-main-sm120-final
```

Read the [archive summary](https://github.com/jpezzulli/sglang-rtxpro6000/tree/traffic-history).
The independent `traffic-history` branch contains only archive data, no runtime
source or workflows. Every update is one atomic Git Data API commit. No checkout,
clone, Git fetch, package installation, or third-party Action is used. The script
itself is retrieved using Contents API at the workflow's exact commit SHA.

## Files and data semantics

- `daily.json`: date-keyed daily views/clones with GitHub's original `count` and
  `uniques`, plus the last collection time for each date. Starts at 2026-08-24.
- `snapshots/YYYY-MM-DD/<UTC timestamp>-<run ID>-<attempt>.json`: immutable per-run
  snapshot of all four traffic responses, including rolling counts, rolling
  uniques, referrers and popular paths. Multiple runs on a day are retained.
- `raw/first-run/{views,clones,referrers,paths}.json`: original response bodies,
  byte-for-byte, including any prelaunch days exposed by GitHub.
- `raw/first-run/metadata.json`: first capture provenance, SHA-256 checksums,
  endpoint query strings and exact coverage/missing dates.
- `README.md`: cumulative daily counts, daily table, current rolling uniques,
  current stars/forks, and current referrer/path tables. Previous snapshots remain
  in the archive.
- `repository-metrics.json`: date-keyed daily observations of public star and
  fork counts, beginning **2026-09-19** when this metric was added. Existing
  traffic history is not backfilled with invented repository-metric values.
- `package-downloads.json`: date-keyed observations of the public GHCR package
  total and every discovered version's download counter.
- `package-snapshots/YYYY-MM-DD/<UTC timestamp>-<run ID>-<attempt>.json`:
  immutable parsed package observations for every successful collection.
- `PACKAGE-DOWNLOADS.md`: readable package totals, per-version counters and
  day-over-day changes. Package downloads are never added to repository clones.
- `raw/packages/first-run/package.html` and `metadata.json`: immutable public
  package-page baseline, its hash, extracted counters and provenance.
- `huggingface-downloads.json`: date-keyed observations of public Hugging Face
  model download counters for `jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4`
  (all-time downloads, rolling 30-day downloads, likes). The collector reads only
  the unauthenticated public API metadata endpoint; no model files, config, or
  HEAD query files are downloaded. First captured **2026-09-17**.
- `huggingface-snapshots/YYYY-MM-DD/<UTC timestamp>-<run ID>-<attempt>.json`:
  immutable per-run parsed Hugging Face observations including the complete
  original `raw_response` API body and its SHA-256 hash, unique per run attempt.
- `HUGGINGFACE-DOWNLOADS.md`: readable Hugging Face model counters and day-over-day
  changes. Hugging Face downloads are never added to repository clones or GHCR
  package downloads.
- `raw/huggingface/first-run/model.json` and `metadata.json`: immutable first-run
  API response, its hash, and provenance. Written only on first initialization.

All structured archive documents contain repository, launch date, UTC collection
time, schema version `1`, and API version `2022-11-28`. Unmodified raw API bodies
are associated with their metadata sidecar. Referrer/path endpoints return top
10 rolling-window lists, not daily breakdowns; `per=day` is supplied on all four
requests but is meaningful only for views/clones.

Each run retrieves and merges the entire returned daily window, replacing both
counts and daily uniques for overlapping dates. Older dates and snapshots are
preserved. Exact cumulative counts apply only through the latest exposed date
when every date since launch is present. Missing dates are listed without zero
filling; recent unexposed dates are labeled separately. Counts are GitHub-reported
activity, including any bots GitHub counts, not verified human activity.

Stars and forks are collected from the repository REST response using the same
run timestamp as the traffic snapshot. The canonical daily record is replaced
only by a later successful collection on that same UTC day; each traffic snapshot
retains its own timestamped observed star/fork values.

The public GitHub Packages page reports cumulative package and version download
counters. They are not unique users or successful installations: OCI clients can
fetch an index, platform manifests and layers, and builds/tests can contribute.
GitHub's documented Packages REST responses expose metadata but do not include
download counters. The package collector therefore validates the public HTML,
tracks versions by stable GitHub version ID after discovery, and fails loudly if
the representation becomes malformed. Package collection runs after repository
traffic is safely committed, so a package-page failure cannot cost the expiring
14-day traffic window. The failed workflow remains visible and is retried by the
existing thegrid watchdog.

The public Hugging Face model API (`https://huggingface.co/api/models/{id}?expand[]=...`)
returns `downloads` (a rolling 30-day count representing GET/HEAD on designated
query files) and `downloadsAllTime` (cumulative since model creation, confirmed
by HuggingFace documentation). The HF collector reads only this public metadata
endpoint unauthenticated — no model weights, config files, HEAD query files, or
self-manufactured downloads. Exact daily download history before the first
archived observation is unavailable and must never be reconstructed by summing
rolling-window values. The collector preserves the first-run raw API response
as an immutable artifact, embeds the complete original response in every
per-run snapshot, and pins the model `createdAt` provenance — the same model
ID reporting a different creation time fails rather than resetting lifetime
provenance. A suspicious zero is rejected only for `downloadsAllTime` against
established positive history; likes may legitimately fall to zero (unlikes).
Like the package collector, the HF step runs after
traffic is committed and is retried by the watchdog on failure or staleness.

Never add rolling totals or rolling unique values across snapshots. The summary
labels daily-unique sums as `sum_of_daily_uniques`; these are not a deduplicated
lifetime person count. GitHub exposes no identities for historical deduplication.

## Authentication and permissions

The job grants only `contents: write` to its built-in `GITHUB_TOKEN`. It exercises
that token against all four traffic endpoints and logs HTTP status only. If those
reads succeed, no separate token is needed. All archive writes always use
`GITHUB_TOKEN` so history updates do not trigger other workflows.

If GitHub returns HTTP 403, repository Actions secret **`TRAFFIC_READ_TOKEN`**
provides traffic read access. John explicitly selected reuse of his existing
GitHub CLI OAuth credential rather than creating a fine-grained token. The
credential is transferred directly from `gh auth token` into `gh secret set`
through a pipe; it must never appear in command arguments, logs, files or commits.
All archive writes continue to use `GITHUB_TOKEN`.

The current operator OAuth credential has `repo` access and the separate
`workflow` scope needed for the one-time workflow installation. Revoking this
OAuth authorization requires replacing the Actions secret. Authentication errors
fail the collector without changing history.

For a future separately scoped credential, GitHub documents **Administration:
read** for all four traffic endpoints; Contents write is not required on that
traffic-read credential. This is documentation, not a requirement to change
John's selected authentication method.

GitHub traffic permission reference:
https://docs.github.com/en/rest/metrics/traffic

## Integrity and failure behavior

A single repository-wide concurrency group serializes scheduled/manual runs;
`cancel-in-progress: false` lets active collections finish. The collector reads
history from an immutable commit SHA, creates a tree based on that history, checks
that the branch has not moved, and updates the ref with `force: false`. A concurrent
writer causes a loud failure; rerun to merge the new head. Never force-push history.

Authentication/HTTP/transport errors, malformed JSON, invalid or duplicate daily
records, empty daily arrays, all-zero windows that would erase positive overlapping
history, mismatched daily windows or totals, corrupt history,
and branch conflicts fail before replacing the branch. Missing history cannot be
silently recreated during a scheduled run. Empty referrer/path lists can be valid;
they are stored as new snapshots and never delete previous snapshots. Original
raw capture files are written only during the explicit seed operation.

The initial branch is an orphan archive-only commit. History commit messages also
carry `[skip ci]`. Combined with built-in-token writes and no workflows on the
history branch, archive updates do not start runtime build/test jobs. Runtime
source, existing workflows, tags, releases and local candidate work are untouched.

To test archive merge/integrity logic without GitHub access:

```sh
python3 .github/scripts/test_archive_traffic.py
python3 .github/scripts/test_archive_package_downloads.py
python3 .github/scripts/test_archive_hf_downloads.py
```

The first capture exposed 2026-08-22 through 2026-09-04, including all dates from
launch on 2026-08-24. No launch-period dates had been discarded. August 24 contains
explicit GitHub-reported zero counts. September 5 and 6 were not yet exposed at
the initial 2026-09-06T01:14:03Z collection; they were not invented or counted.

## Independent timer and recovery on thegrid

GitHub's first 00:17 UTC schedule event produced no run by 2026-09-07T02:03Z,
although the workflow was active on the correct default branch. Manual dispatch
continued to succeed. The underlying GitHub scheduler cause is unconfirmed;
manual success alone does not verify GitHub's schedule delivery.

Thegrid runs one `traffic-daily-report.timer` job at **06:15 America/New_York**.
It first invokes the watchdog as `mrkaos`, then emails the report only after the
watchdog verifies fresh traffic and package archives. User lingering was already
enabled, and `Persistent=true` catches up after downtime. A fresh archive requires
a capture at or after **00:17 UTC**; otherwise the watchdog dispatches the existing
GitHub workflow, waits for completion, and verifies its writes before the email step.

The watchdog dispatches the existing GitHub workflow if repository traffic,
package-download collection, or Hugging Face model counter collection is overdue.
A genuinely uninitialized Hugging Face archive (no HF artifacts at all in the
history tree) counts as overdue and triggers the same dispatch/recovery path;
once established, a missing or corrupt canonical `huggingface-downloads.json`
fails loudly instead of silently re-initializing. It adopts a queued/running
collector instead of dispatching another, waits up to
eight minutes, and verifies workflow success plus fresh `daily.json`,
`package-downloads.json`, and `huggingface-downloads.json` on `traffic-history`.
Authentication, transport, archive, or workflow failures leave history untouched,
skip the email, and mark the single daily service failed for the existing journal
alert path. The existing GitHub workflow concurrency group serializes collectors.

The watchdog reuses the operator's `gh` login directly and does not copy or print
a credential. It makes no Git clone/fetch requests and never writes history
itself. GitHub retains collection and all archive writes, using the existing
Actions secret and built-in write token. GitHub's original daily schedule remains
active as a second trigger; its delivery is not relied on for the guarantee.
Both schedulers still depend on GitHub API/Actions availability; recovery resumes
when the service is available. A thegrid outage is caught up after user-manager
startup; data absent from GitHub's retention window cannot be reconstructed.

The report uses the existing host `mail` → `msmtp` sender. It keeps no local
report state or report history and never calls the live `traffic/*` endpoints:
the report is a read-only view of one pinned `traffic-history` commit, reading the
current canonical `daily.json`, `package-downloads.json`,
`repository-metrics.json`, and `huggingface-downloads.json` plus the one traffic
snapshot whose `collected_at` equals `daily.json`'s, which supplies the rolling
totals, referrers, and popular paths. The snapshot's dated rows must agree with the canonical archive for
every date at or after launch (legitimate prelaunch exposure stays out of
lifetime totals), and the email names the actual first/last dates of the
returned rolling window rather than assuming fourteen days, so a regressed
window is never misdated. Two reads of the same commit render identically.
Daily unique sums are
labeled honestly and never presented as a deduplicated lifetime-person count;
clones and container downloads are never presented as distinct users.

Change figures are true net differences against the last archive commit before
the current collection's UTC day, located with the commits API pinned to the
same head (`commits?sha=<head>&until=<one second before that day>&per_page=1`);
that commit's canonical histories are read without the current-capture freshness
requirement but with the same identity, coverage, and contiguity validation as
the current archive, including an identical launch date and no baseline dates
missing from the current archive, so lost history fails loudly instead of
inventing growth or negative corrections. Cumulative views, clones, and summed daily unique cloners are
current archive sums minus baseline sums, so delayed GitHub corrections to
older dates and newly exposed backfilled dates are included; stars, forks, and
container downloads compare that baseline's latest canonical records. All
deltas are labeled once as `Changes since the <date> collection`, using the
baseline's actual collection date and never claiming a previous email. With no
earlier commit the report says `Baseline unavailable` and shows no change
figures instead of guessing `+0`. The body leads with the last recorded traffic
day, the exact completed UTC dates still pending from GitHub (kept unknown,
never zero-filled, never treated as an outage because age alone cannot tell a
delayed API from a slow day), and either `New traffic dates: <range>`, `No newer
traffic date since <baseline collection>`, or `Baseline unavailable`;
same-date GitHub corrections are reported as revised recorded counts, not as a
new date. Today's bucket is labeled partial. The subject is the stable
`Pennyroyal GitHub report — <local date>` with no stale WARNING prefix.
Collection failures (a capture before the 00:17 UTC cutoff, malformed or
incomplete archives, a missing or mismatched current snapshot, future-dated
days) still fail loudly and skip the email. `--dry-run` prints the subject and
the body and sends nothing.

Installed locations:

- `/opt/sglang/traffic-archive/traffic_watchdog.py`: deployed watchdog.
- `/opt/sglang/traffic-archive/state/verified.json`: last run verified by watchdog.
- `/opt/sglang/traffic-archive/state/pending.json`: in-flight dispatch, when present.
- `~/.config/systemd/user/traffic-archive-watchdog.service`: callable archive verifier.
- `~/.config/systemd/user/traffic-daily-report.{service,timer}`: daily verify-then-email job.
- `.github/traffic-archive/`: portable unit templates; substitute `@INSTALL_DIR@`,
  `@REPOSITORY@`, and `@RECIPIENT@` in the daily-report service template during installation.

Inspect without changing anything:

```sh
systemctl --user list-timers traffic-daily-report.timer
systemctl --user status traffic-daily-report.service
journalctl --user -u traffic-daily-report.service --since today
cat /opt/sglang/traffic-archive/state/verified.json
```

A successful oneshot service normally shows `inactive (dead)` with exit status 0;
the timer remains `active (waiting)`. Check that combination, the journal, the
verified state, and the remote archive timestamp. Do not interpret a completed
oneshot's inactive state as a failed scheduler.

Focused watchdog tests: `python3 .github/scripts/test_traffic_watchdog.py`.
No runtime code or inference service participates in this mechanism.
