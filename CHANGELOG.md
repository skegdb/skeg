# Changelog

All notable changes to the engine are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file tracks **only the engine** (this repository). Multi-tenant
implementation details, auth store internals, and tenant API surface
live in a separate (private) repo and are documented there.

## [0.2.0] — 2026-05-28

### Added

- **`skeg-telemetry` dynamic registry.** Downstream consumers
  (`skeg-kv-cache`, `skeg-tenant`, applications on top of skeg) can now
  register their own counters, histograms, and gauges without patching
  the engine's closed enums.
  - `register_counter(name) -> &'static [AtomicU64; MAX_SHARDS]`
  - `register_histogram(name) -> &'static DynHistogram`
  - `register_gauge(name) -> &'static AtomicU64`
  - `register_op!("base_name")` macro: derives `<name>_total` (sharded
    counter) and `<name>_duration_seconds` (histogram) from a single
    base, returns a `DynOp` bundling both.
  - Idempotent: repeated calls with the same name return the same
    `&'static` handle, so `OnceLock::get_or_init(|| register_*(…))`
    patterns are safe.
  - Pool sizing: 256 sharded counter slots (64 KiB), 64 histogram
    slots (14 KiB), 64 gauge slots (512 B). All `static` `AtomicU64`s,
    no allocator on the hot path.
  - Hot-path cost: same shape as the closed-enum path (one
    `OnceLock::get` branch + one or two `fetch_add(Relaxed)`). Measured
    ~2 ns on Apple M1.
- Histogram buckets extended from 22 to 26 (1 µs → 16.78 s upper
  bound, then `+Inf`). The previous range clipped any observation
  ≥ 1.05 s into the sentinel; downstream consumers with longer-tail
  operations (`skeg-kv-cache` blob restore, tenant quota probes) now
  observe the real distribution out to ~16 s.
- `SKEG.STATS` and `/metrics` output now appends the dynamic registry
  contents (sorted by name) after the engine's static metric block.
  Engine schema is grep-stable; downstream metrics are added below the
  blank-line separator.

### Changed

- **Per-crate versioning** (workspace-level): each crate now carries
  its own `version` field instead of inheriting from
  `[workspace.package]`. The release workflow diffs each crate's tree
  against the previous tag and skips `cargo publish` for unchanged
  crates, so a release that touches only `skeg-telemetry` no longer
  republishes the seven other crates as identical no-op bumps on
  crates.io.
- `skeg-server` bumped to `0.2.0` to track the new telemetry dep
  surface and stay aligned with the user-facing release tag (the
  Homebrew / GitHub Release naming follows the tag).

### Notes

- `skeg-core` keeps version `0.1.2` on crates.io: its source did not
  change in this release. The local workspace dep requirement was
  bumped to `skeg-telemetry = "0.2"` so the bench/dev path build
  works; a future skeg-core release will carry that requirement out.
- The seven unwired gauges from v0.1.2 (`VlogSegmentsLive`,
  `VlogSegmentsCompacting`, `VlogTotalBytes`, `CompactionInProgress`,
  `VindexSizeBytes`, `VindexVectors`) are still TODO. The schema is
  stable; dashboards written against v0.1.2 keep working.

## [0.1.2] — 2026-05-28

### Added

- **Telemetry.** New `skeg-telemetry` crate provides zero-overhead
  `AtomicU64` counters and exponential histograms for the hot path.
  - Per-op counters (`skeg_ops_total{op}`) and duration histograms
    (`skeg_op_duration_seconds_*`) for `get`, `set`, `del`, `vset`,
    `vsearch`, `vdel`, `ping`.
  - Cache counters: `skeg_cache_hits_total`, `_misses_total`,
    `_evictions_total` (wired in `skeg-core/cache`).
  - vLog counters: `skeg_vlog_syncs_total`,
    `_group_commit_batches_total`, `_compaction_runs_total`,
    `_compaction_bytes_total`. Gauge `skeg_vlog_live_bytes`
    refreshes on each `STATS` call.
  - Measured hot-path cost: `record_op` ≈ 4.7 ns on Apple M1;
    `criterion` gate in `crates/skeg-telemetry/benches/overhead.rs`.
  - Three exposure modes:
    - default: counters compiled in, dump via RESP3 `SKEG.STATS`.
    - `--no-default-features` on the crate: every public function
      is a compile-out `#[inline(always)]` no-op.
    - `--features metrics-http` on `skeg-server`: tiny HTTP
      exporter on a dedicated thread, surfaces `/metrics` in
      Prometheus text format.
- **`--metrics-port <PORT>` CLI flag** on `skeg` (and the matching
  `SKEG_METRICS_PORT` env). Spawns the Prometheus exporter on
  `127.0.0.1:PORT` when the binary is built with `metrics-http`.
- `SKEG.STATS` response is now extended with the full telemetry
  dump after the legacy `cache_bytes=…` summary line, separated by
  a blank line. Old clients that grep for `cache_bytes=` keep
  working unchanged.

### Notes

- Five gauges remain unwired in this release
  (`VlogSegmentsLive`, `VlogSegmentsCompacting`, `VlogTotalBytes`,
  `CompactionInProgress`, `VindexSizeBytes`, `VindexVectors`). They
  read `0` from `STATS` and `/metrics`. Wiring sites are marked
  with `TODO(telemetry):` comments in `vlog.rs` and `shard.rs` and
  will land in a follow-up; the schema is stable and dashboards
  written today will not need to change.

## [0.1.1] — 2026-05-26

### Added

- `skeg --help`, `skeg -h`, `skeg --version`, `skeg -V` (and the same on
  `skeg-resp3`). The binaries now print a usage block and exit
  cleanly instead of starting the server when these flags are
  passed. Unblocks the canonical `brew install` smoke test.

### Fixed

- README quickstart used the wrong vector command syntax. All vector
  operations are namespaced under `SKEG.*` and take positional args
  (`SKEG.VINDEX.CREATE <name> <dim> <kind> <backend>`), not the
  `VINDEX.CREATE docs DIM 1024 METRIC cosine` form that the previous
  README implied.

## [Unreleased] — pre-release v0.1.0

### Added

- New `tenant` Cargo feature (default off) in `skeg-server`. When
  enabled at compile time, the server accepts a `TenantContext`
  provided by an external crate and scopes KV and vector ops per
  tenant. Without the feature the engine ships as a pure single-tenant
  store with byte-identical wire and disk layout to pre-tenancy code.
- `tune_socket` applies `TCP_NODELAY` and `SO_KEEPALIVE` (60s idle, 10s
  probe interval) on every accepted server connection and every client
  connection from `SkegClient`. Catches half-open TCP states that
  otherwise leak file descriptors in long-running deployments.
- `--workers <N>` (env `SKEG_WORKERS`) dispatches `VSEARCH` requests to
  `tokio::task::spawn_blocking` so KV ops on the same shard do not
  queue behind multi-ms searches. Default `0` keeps the inline
  behaviour that matches the public benchmark numbers.
- `--tier-mmap` (env `SKEG_TIER_MMAP`): the TurboQuant `codes` buffer
  is persisted to `tier.cache.bin` at open and memory-mapped. The OS
  page cache can reclaim tier pages under pressure instead of pushing
  anonymous memory to swap.
- `--graph-mmap` (env `SKEG_GRAPH_MMAP`): `graph.vmn` is opened as a
  memory map and the `Node` array is reinterpreted directly from the
  mmap'd bytes. No per-`Node` parsing at open, OS page cache reclaim
  on the graph. Combines with `--tier-mmap` to make the whole disk
  index paginable.
- New RESP3 verbs in the `SKEG.*` namespace for vector ops:
  `SKEG.VINDEX.CREATE`, `SKEG.VINDEX.DROP`, `SKEG.VINDEX.LIST`,
  `SKEG.VSET`, `SKEG.VDEL`, `SKEG.VSEARCH`. Vector payloads are RESP
  bulk-strings carrying raw little-endian `f32` bytes; length must be
  exactly `dim * 4`. When the `tenant` feature is on, names are scoped
  per tenant; otherwise they pass through unchanged.
- Workspace metadata: root `Cargo.toml` declares `[workspace.package]`
  with `edition = "2024"` and `rust-version = "1.86"`. Older
  toolchains get a clear error instead of a cryptic build failure.

### Changed

- `set_speed_enabled` returns `Result<(), SpeedAlreadySet>` instead of
  `Result<(), ()>`. Caller can format or log a meaningful message;
  behaviour is unchanged.

### Documentation

- `README.md`, `CHANGELOG.md`, `LICENSE`, `NOTICE`, and `SECURITY.md`
  added at the repo root for the public release.
