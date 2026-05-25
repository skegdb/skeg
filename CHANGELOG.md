# Changelog

All notable changes to the engine are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file tracks **only the engine** (this repository). Multi-tenant
implementation details, auth store internals, and tenant API surface
live in a separate (private) repo and are documented there.

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
