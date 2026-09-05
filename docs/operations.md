# Production operations

This runbook describes the process contract implemented by the three server
binaries (`skeg`, the native protocol, and `skeg-resp3`, which serves both
the single-tenant and the authenticated multi-tenant profile). It does not
turn a
single node into a replicated service: skeg has no quorum, automatic failover,
or built-in backup scheduler.

## Memory and maintenance

On Linux the process reads the tightest cgroup v2 headroom and retains a
reserve before admitting request buffers, vector delta growth, and maintenance
working sets. `SKEG_MEMORY_LIMIT_BYTES` may make that headroom smaller, never
larger. `SKEG_MEMORY_RESERVE_BYTES` overrides the retained margin; the default
is `max(64 MiB, limit/10)`.

Every flush, runs merge, delete patch, full fold, and IVF rebuild estimates its
peak transient heap before its snapshot is materialised. The job holds both a
byte reservation and, for graph-heavy work, a concurrency permit through its
finish phase. An automatic job that cannot reserve returns its ladder turn and
lets cheaper work run; an explicit consolidate returns retryable
`BACKPRESSURE`. Watch:

- `skeg_memory_headroom_bytes` and `skeg_memory_reserved_bytes`;
- `skeg_maintenance_budget_skips_total` for deferred jobs;
- `skeg_maintenance_failures_total` for jobs that ran and failed.

The release gate runs concurrent ingest plus a real fold inside independent
256 MiB and 512 MiB cgroups:

```sh
scripts/check-maintenance-cgroup.sh
```

It is intentionally Linux-only and requires Docker. A test run outside a real
`memory.max` is not equivalent evidence.

## Graceful shutdown

`SIGTERM` and Ctrl-C use one lifecycle on both wire protocols:

1. close the listener, so no new connection can be accepted;
2. drain accepted connections for `SKEG_SHUTDOWN_TIMEOUT_MS` (default 30000);
3. abort only connection tasks that exceeded that deadline and record failure;
4. close every shard inbox and drain requests already accepted by a shard;
5. stop and join compaction, snapshot, and maintenance loops cooperatively;
6. synchronise every persistent vector index's mutable delta WAL;
7. flush every shard VLog, join every worker, and aggregate all failures.

Exit status zero means every step completed and every file barrier succeeded.
An expired connection deadline, a task panic, `EIO`, `ENOSPC`, or another sync
failure produces a non-zero exit after the remaining shards have still been
attempted. `SIGKILL` cannot run this sequence and carries no durability
promise.

The counters are `skeg_shutdown_started_total`,
`skeg_shutdown_completed_total`, `skeg_shutdown_failures_total`, and
`skeg_shutdown_connection_timeouts_total`. Because a successful shutdown ends
the process immediately after incrementing them, retain the final scrape or
use the supervisor's exit status and logs as the authoritative outcome.

### Docker and Compose

The image declares `STOPSIGNAL SIGTERM`. Allow longer than the internal
connection deadline for maintenance already in flight and filesystem syncs:

```sh
docker stop --time 120 skeg
```

The supplied Compose example sets `stop_grace_period: 2m`. Do not set the
container runtime's grace period equal to `SKEG_SHUTDOWN_TIMEOUT_MS`; that
would leave no time for the shard barriers after connection draining.

### systemd

Use an ordinary foreground service and let systemd observe the exit code:

```ini
[Service]
ExecStart=/usr/local/bin/skeg-resp3 --data-dir /var/lib/skeg
Environment=SKEG_SHUTDOWN_TIMEOUT_MS=30000
KillSignal=SIGTERM
TimeoutStopSec=120
Restart=on-failure
```

An unclean stop should be investigated before an automatic restart loop hides
it. In particular, any increase in `skeg_vlog_flush_failures_total` or a log
line containing `vector WAL sync` means the last durability barrier failed.

## Backup boundary

Stopping the process successfully gives a quiescent, durable directory that
can be copied as a unit. A live filesystem copy is not a snapshot protocol:
use a storage-level atomic snapshot or stop the process first. Restore into an
empty directory and validate it with the same binary version before exposing
the listener.
