# Changelog

All notable changes to the engine are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file tracks the engine and the multi-tenant server, both in this
repository.

## [0.5.0] - unreleased

### Added

- **Filtered vector search.** A vector can carry an optional payload, and a
  search can restrict its results to vectors whose payload matches a filter.
  `SKEG.VSET <name> <id> <vector> [PAYLOAD <blob>]` stores an opaque blob beside
  the vector (in the KV vLog under a reserved, tenant-scoped key, so the
  quantized graph stays dense). `SKEG.VSEARCH <name> <k> <l_search> <query>
  [WITHPAYLOAD] [FILTER <expr>]` returns the blob with each hit (`WITHPAYLOAD`)
  and/or applies a payload filter (`FILTER`). The blob's `key=value` fields are
  parsed into a per-index payload index; the filter grammar supports `field =
  value`, `field IN (...)`, the ranges `>= > <= < BETWEEN a AND b`, `field
  EXISTS`, and `AND` / `OR` / `NOT` with parentheses. A field repeated in a
  payload is multi-valued (matches any of its values).

- **Adaptive filtered-search planner.** A selective filter (small matching set)
  is scored exactly over just the matching vectors. A broad filter runs a
  filtered graph search: two complementary walks merged (one that explores only
  the matching subgraph, one that navigates the whole graph and filters at
  re-rank), so recall holds whether the matching vectors cluster together (real
  metadata) or scatter. Validated on real 1024-dim embeddings at 100k and 500k:
  recall@10 0.98 to 1.00 across selectivities and metadata shapes, at query time
  with no extra build cost. The payload index is rebuilt from the stored blobs
  on the first filtered search after a restart.

- **TurboQuant tiers on the read-write disk path.** `SKEG.VINDEX.CREATE name dim
  tq1|tq2|tq4 disk` builds a live-writable disk index whose resident tier is
  TurboQuant (`dim*bits/8` bytes/vector) instead of int8. No trained codebook
  (unlike PQ, which stays serve-only), so it works under streaming writes. The
  tier kind persists in a sidecar and is rebuilt on open and consolidate. tq2 is
  the recommended sweet spot (recall ~1.0, sub-int8 RAM, latency ~int8); tq1 is
  the leanest but best-effort below 512d.

- **`SKEG.VMSET` bulk insert.** One `name` followed by `(id, vector, payload)`
  triples; the server fans the items out concurrently so durable payload writes
  batch in the group committer. Combined with relaxed payload durability and a
  geometric delta rebuild, 100k ingest dropped from 1928s to 28s.

- **`SKEG.VINDEX.CONSOLIDATE name`** force-folds a disk index's delta into the
  graph after a bulk load. Idle indexes also self-consolidate: a per-shard
  background task folds a delta that has been stable across ticks, so an index
  is lean by default without an explicit call.

### Changed

- **Vindex RAM accounting (`VindexSizeBytes` gauge) is now tier-accurate.** Disk
  indexes report their own `resident_bytes()` (graph + quantized tier + delta)
  instead of an `n*dim` int8 estimate, so the gauge tracks tq1/tq2/tq4 correctly.

- **Plain (non-filtered) search and writes are unchanged.** A VSET without
  `PAYLOAD` does no extra work; a VSEARCH without `FILTER`/`WITHPAYLOAD` takes
  the existing path byte-for-byte. The native binary protocol stays
  payload/filter-free; RESP3 is the surface for the new feature.

- **`ShardSet::vset` / `ShardSet::vsearch` gained parameters** (payload, the
  tenant, `want_payload`, and an optional filter), and `skeg-vector` gained
  `DiskVamanaIndex::search_filtered` / `score_ids` / `live_ids`. Library code on
  these APIs must update its call sites, which is why `skeg-server` takes a minor
  version bump.

### Versions bumped

- `skeg-vector` 0.1.5, `skeg-resp3` 0.2.1, `skeg-server` 0.5.0,
  `skeg-server-tenant` 0.2.1

## [0.4.1] - 2026-06-15

### Added

- **Fair per-tenant cache eviction.** Per-tenant cache accounting already
  shipped; eviction itself was tenant-blind, so under a noisy neighbour a
  tenant with a large hot set could evict another tenant's small hot set in
  FIFO order. The Main queue's victim selection is now share-aware: it
  computes an equal share (`cache budget / active tenants`) once per
  eviction and, when some tenant is over its share, briefly skips
  under-share victims (bounded to 16 re-queues) to evict an over-share
  tenant instead. The Small queue stays tenant-blind on purpose: it already
  absorbs scan floods, so no fairness is needed there. A flooding tenant can
  no longer starve a quiet tenant's working set out of the cache.

### Changed

- **Single-tenant and anonymous traffic is unchanged.** Fairness activates
  only when more than one tenant is resident in a cache shard; with a single
  tenant the eviction path is byte-identical to before and adds no
  measurable overhead. The over-share scan is `O(active tenants)` once per
  eviction but short-circuits on the first over-share tenant; a scaling
  bench (1 to 10000 resident tenants) shows eviction cost stays flat.

### Versions bumped

- `skeg-core` 0.3.1, `skeg-server` 0.4.1

## [0.4.0] - 2026-06-14

### Added

- **Per-tenant resource accounting and hard quotas.** The engine now
  tracks, per tenant, the hot-key cache bytes and the live on-disk KV
  bytes each tenant holds, and can enforce optional hard limits at
  admission. `VLog` gains a `tenant(id)` view that scopes cache
  residency and disk accounting to a tenant; the per-tenant disk total
  is rebuilt from the index on restart. A `TenantBackend::limits(tenant)`
  hook lets a deployment cap a tenant's vector count (`max_vectors`,
  checked on `SKEG.VSET`) and its on-disk KV bytes (`max_disk_bytes`,
  checked on `SET`); an over-limit write is rejected before anything is
  stored. The vector quota is enforced under the index write lock so an
  insert is counted exactly once and overwrites stay free; the disk
  quota counter is shared across shards so the limit is global per
  tenant. New public surface: `VLog::tenant`, `TenantView`,
  `SharedTenantDisk`, `new_shared_disk` (skeg-core); `TenantLimits`,
  `TenantVectorQuota` (skeg-server).

- **`SKEG.QUOTA.SET` / `SKEG.QUOTA.GET` admin commands.** An operator can
  set a tenant's quotas at runtime over RESP3:
  `SKEG.QUOTA.SET <tenant> <max_vectors> <max_disk_bytes>` (`*` = unlimited),
  and read them back with `SKEG.QUOTA.GET <tenant>`. The commands require
  an admin connection; the multi-tenant binary designates the admin tenant
  with `--admin-tenant <name>` and persists the limits in a sidecar next to
  `auth.kdb`, so they survive a restart.

### Changed

- **Single-tenant and anonymous traffic is unchanged.** With no limit
  configured nothing is counted and the write path is byte-identical to
  before; the per-tenant accounting adds no measurable overhead on the
  single-tenant path.

- **`skeg_core::Error` and `skeg_resp3::Command` gained variants**
  (`Error::DiskQuota`; `Command::SkegQuotaSet` / `SkegQuotaGet`). Code that
  matches these enums with a wildcard arm is unaffected; an exhaustive
  match must add the new arms. This is why `skeg-core` and `skeg-resp3`
  take a breaking version bump.

### Versions bumped

- `skeg-core` 0.3.0, `skeg-resp3` 0.2.0, `skeg-server` 0.4.0,
  `skeg-server-tenant` 0.2.0, `skeg-vector` 0.1.4

## [0.3.8] - 2026-06-09

### Added

- **`skeg-multi-tenant` first crates.io publish.** The crate moves
  back into the engine workspace now that all of its sibling rigging
  transports (notably `skeg-rigging-net-resp3`) are on the registry.
  Path overrides on the rigging deps were dropped in favour of plain
  version requirements so the local copy of `skeg-rigging` and the
  one pulled in transitively by `skeg-rigging-net-resp3` resolve to a
  single shared instance; without it the `live-attach` feature would
  refuse to compile because `TenantId` would surface as two distinct
  types.

### Changed

- **`Cargo.toml` workspace lint config** gains
  `unknown_lints = "allow"` so the MSRV CI rustc no longer screams
  when it meets the `clippy::manual_is_multiple_of` allow line we ship
  for the dev toolchain.

### Versions bumped

- `skeg-multi-tenant` 0.1.0 (first publish)

## [0.3.7] - 2026-06-08

### Added

- **Child spans inside `VSEARCH`.** `DiskVamanaIndex::search_with_l`
  now emits `vsearch.walk` (fields: `list_size`, `rerank`, `early`,
  `visited`, `returned`) and `vsearch.rerank` (fields: `candidates`,
  `disk_reads`, `skipped`) nested under the `vsearch` parent the
  handler creates. Traces shipped through the OTLP exporter carry the
  full hierarchy without extra configuration.

- **`compat-tests/redis_py_compat.py`** — end-to-end smoke against a
  live `skeg-resp3` driven through `redis-py` 5+. Exercises every
  typed command from v0.3.5, asserts byte-exact error strings, and
  validates RESP2/RESP3 negotiation. Manual gate (not part of `cargo
  test`); runs in ~3 seconds and exits non-zero on any divergence.

### Changed

- **`skeg-server::resp3_handler`** folds the v0.3.5 `_typed` wrapper
  layer into the `dispatch_command` match arms and drops the legacy
  `dispatch_unknown` / `dispatch_skeg` arms that the typed parse path
  made unreachable. Net: about 190 lines removed, identical wire
  behaviour preserved.

### Versions bumped

- `skeg-server` 0.3.2 -> 0.3.3 (handler simplification, span
  hierarchy; no new public API surface)

## [0.3.6] - 2026-06-07

### Added

- **Default-on Prometheus exporter.** The `metrics-http` cargo feature
  is now part of `skeg-server`'s default set, so released binaries
  serve `/metrics` out of the box when `--metrics-port <PORT>` is
  passed. Drop it with `--no-default-features` to get a slim build;
  the `--metrics-port` flag stays parseable and logs a warning when the
  feature is off.

- **OTLP/gRPC tracing exporter** (`tracing-otlp` feature, default-on).
  When `SKEG_TRACE_OTLP_ENDPOINT=<url>` is set, spans flow to an
  OpenTelemetry collector through a `tracing-opentelemetry` bridge with
  head-based sampling. Env knobs:

  - `SKEG_TRACE_OTLP_ENDPOINT`: OTLP/gRPC URL (unset disables export).
  - `SKEG_TRACE_SAMPLE_RATE`: `[0.0, 1.0]` head sampler, default `1.0`.
  - `SKEG_TRACE_RESOURCE_ATTRS`: `k1=v1,k2=v2` resource labels.

  Both binaries (`skeg`, `skeg-resp3`) install the layer; the exporter
  is shut down cleanly on exit so in-flight spans flush.

- **`vsearch` span** emitted from both protocol handlers (binary +
  RESP3), with structured fields: tenant, vindex, k, l_search,
  vector_dim, hits. Resource attributes include `service.name=skeg`
  and `service.version=<crate version>`.

- **Documentation and operator assets.** New `OBSERVABILITY.md` covers
  the metric schema, a Prometheus scrape config, OTel collector
  integration, and the tracing overhead numbers. `assets/grafana/`
  ships an overview dashboard JSON, `prometheus.yml`, and a starter
  `otel-collector-config.yaml`.

- **Tracing overhead microbench** at
  `crates/skeg-server/benches/tracing_overhead.rs`. Measured on M1 Pro:
  1.54 ns/span with subscriber installed and filter dropping; 1116
  ns/span when fully serialised. The G-O3.1 gate (under 5% relative to
  the no-subscriber baseline) clears with a ~70x margin.

### Changed

- **`skeg-vector` sort sites** switched from `sort_unstable_by(|a, b|
  b.0.cmp(&a.0))` to `sort_unstable_by_key(|x|
  std::cmp::Reverse(x.0))` where the key type is `Copy`. Identical
  ordering; quiets the new clippy 1.96 `unnecessary_sort_by` lint
  without changing behaviour.

### Versions bumped

- `skeg-server` 0.3.1 -> 0.3.2 (new default features, span
  instrumentation; depends on the same skeg-vector / skeg-resp3 as
  v0.3.5)

## [0.3.5] - 2026-06-04

### Added

- **`skeg-resp3` typed `Command` variants** for the 21 KV and `SKEG.*`
  verbs the server already exposed via the legacy untyped fallback. KV
  (11): `GET`, `SET`, `DEL`, `EXISTS`, `MGET`, `MSET`, `INCR`, `DECR`,
  `INCRBY`, `DECRBY`, `SELECT`. `SKEG.*` (10): `STATS`, `SHARDS`,
  `WHOAMI`, `AUTH`, `VINDEX.LIST`, `VINDEX.CREATE`, `VINDEX.DROP`,
  `VSET`, `VDEL`, `VSEARCH`. The new `CommandError` variants
  (`WrongArity`, `WrongAritySkeg`, `SelectDbOutOfRange`,
  `SelectInvalidIndex`, `NotAnInteger`) render the same error strings
  the server emitted byte-for-byte, so existing `redis-cli` scripts and
  Redis client libraries that match on `ERR ...` text keep working.

  Effect on consumers:

  - `skeg-resp3` is now the source of truth for command arity + simple
    arg parsing. Downstream crates (`skeg-server`, future
    `skeg-client-rs` / `skeg-py`) get the same typed dispatch instead
    of duplicating the validation logic.
  - 180 unit tests in `skeg-resp3` (43 new typed-command tests + 5
    proptests covering KV byte-preservation across random inputs).
  - Verified end-to-end with `redis-cli` on the live binary: 14
    commands round-trip with identical error strings.

### Changed

- **`skeg-server::resp3_handler` dispatches the new typed variants
  directly**, bypassing the legacy `dispatch_unknown` for KV and
  `SKEG.*`. The legacy untyped path is preserved for genuinely unknown
  commands (`FOO bar` -> `ERR unknown command 'FOO'`).

### Versions bumped

- `skeg-resp3` 0.1.2 -> 0.1.3 (added public `Command` and
  `CommandError` variants; backwards-compatible additive change)
- `skeg-server` 0.3.0 -> 0.3.1 (depends on `skeg-resp3` 0.1.3; dispatch
  rewired)

## [0.3.4] - 2026-06-04

### Added

- **`skeg-simd` block-32 SIMD scoring for TurboQuant 4-bit codes.**
  New public API: `build_tq4_lut_f32`, `interleave_tq4_codes`,
  `quantize_tq4_lut_u8`, `tq4_block32_score_scalar`,
  `tq4_block32_score_u8_scalar`, and (on aarch64)
  `tq4_block32_score_u8_neon`. The block kernel scores 32 vectors in
  parallel via `vqtbl1q_u8` lookups into a per-query u8 LUT with a
  periodic widening flush to f32. Measured on M1 Pro, 100k synthetic
  vectors, single-thread flat scan:

  - dim=384: row 275 QPS -> block 1359 QPS (4.93x)
  - dim=1024: row 107 QPS -> block 515 QPS (4.80x)
  - dim=1536: row 73 QPS -> block 342 QPS (4.71x)

- **`skeg-vector::FlatIndex::search_block_tq4`**: opt-in entry point
  routing the flat scan through the block-32 kernel for the
  `TurboQuant { bits: 4 }` tier. Returns `None` for other tiers so the
  caller falls back to the row-major `search` path. Recall is
  equivalent to row scoring (proptest verified, |delta| < 1e-3).

- **`skeg-vector` pq32 / pq64 aliases**: cache-fit experiments for
  PQ with smaller `m` values.

- **Diagnostic benches**: `flat_block_throughput`,
  `flat_block_pareto`, `vamana_rerank_pipe`.

### Changed

- **`skeg-simd::tq4_adc_i8_neon`**: further +100% via 8-accumulator
  unroll + native nibble unpack (on top of v0.3.3's +35-66%). 32 coords
  per outer iteration over 8 independent f32x4 accumulator chains,
  replacing the SWAR+stack-roundtrip nibble unpack with a NEON-native
  `vand_u8 + vshr_n_u8 + vzip1_u8 + vzip2_u8 + vcombine_u8` sequence
  (~6 cycles/chunk saved). Bit-equivalent to the previous kernel within
  the existing equivalence test tolerance.

- **`skeg-simd::tq2_adc_i8_neon`**: +70-74% via the same NEON-native
  nibble unpack pattern adapted to 2-bit codes.

### Versions bumped

- `skeg-simd` 0.1.3 -> 0.1.4 (added block-32 API; row kernels still
  binary-compatible)
- `skeg-vector` 0.1.2 -> 0.1.3 (added `FlatIndex::search_block_tq4`)

## [0.3.3] - 2026-06-02

### Changed

- **`skeg-simd::tq4_adc_i8_neon` and `tq2_adc_i8_neon`**: refactor the
  TurboQuant ADC NEON kernels to use multiple independent f32x4
  accumulators (4 for tq4, 8 for tq2) and defer the `i8_scale`
  multiply until after the horizontal sum. Breaks the serial-FMA
  dependency chain that previously throttled per-row throughput.

  Measured on M1 Pro, 100k synthetic vectors, single-thread flat
  scan via `cargo bench -p skeg-vector --bench flat_throughput`:

  - tq2 at dim=1024: 32 QPS -> 55 QPS (+72%)
  - tq2 at dim=1536: 21 QPS -> 38 QPS (+81%)
  - tq4 at dim=1024: 32 QPS -> 43-53 QPS (+35-66%)
  - tq4 at dim=1536: 21 QPS -> 35 QPS (+67%)

  Result is bit-equivalent to the previous kernel within the
  existing equivalence test tolerance (float multiply distributes
  over the scale factor; per-test 1e-5 budget).

### Versions bumped

- `skeg-simd` 0.1.2 -> 0.1.3 (kernel refactor, behaviour unchanged)

## [0.3.2] - 2026-06-02

### Fixed

- **`release.yml`**: the publish-crates loop hard-coded the list of
  crates and was missing `skeg-tenant` and `skeg-server-tenant`. They
  shipped to GitHub Releases in v0.3.1 but never reached crates.io.
  The loop is now extended to include them, and both crates are
  bumped to 0.1.1 so the diff-based skip logic re-publishes them.

### Versions bumped

- `skeg-tenant` 0.1.0 -> 0.1.1
- `skeg-server-tenant` 0.1.0 -> 0.1.1

## [0.3.1] - 2026-06-01

### Added

- **`skeg-tenant` crate.** Multi-tenant primitives shipped as a
  first-class workspace member: tenant id (xxh3_128), argon2id
  password hashing, HMAC-SHA256 tokens, on-disk auth store
  (`auth.kdb`), quota tracker. Apache-2.0, same as the engine.
- **`skeg-server-tenant` crate.** Multi-tenant server binary
  that wraps `skeg-server` and installs `skeg-tenant` as the
  `TenantBackend`. Ships a binary called `skeg-server` (same name
  as the OSS one; different package). Two extra flags vs the OSS
  server: `--tenant-auth <path>` (enable tenant resolution against
  an `auth.kdb`) and `--tenant-strict` (reject anonymous `HELLO 3`).
  Apache-2.0.

### Versions bumped

- `skeg-tenant` (new) 0.1.0
- `skeg-server-tenant` (new) 0.1.0

## [0.3.0] - 2026-06-01

### Added

- **`SharedCommitter` for multi-shard write throughput on Apple Silicon.**
  `F_FULLFSYNC` (the macOS power-loss durability call) is a device-wide
  barrier: concurrent fsyncs on different files serialize at the
  hardware. Previously, multi-shard write throughput on macOS regressed
  going from 1 shard to 4 shards because each shard issued its own
  barrier. The new process-wide `SharedCommitter` aggregates pending
  writes from every shard into a single fsync per batch, amortising
  the barrier across all shards.

  Measured on a MacBook Pro M1 (1000 power-durable appends per shard,
  128 byte records, 5 runs each variant, median wall clock):

  | variant       | durability model | shards | median  | ops/s   |
  | ------------- | ---------------- | ------ | ------- | ------- |
  | 1sh_perfile   | per-file fsync   | 1      | 6.60 s  | 152     |
  | 4sh_perfile   | per-file fsync   | 4      | 20.52 s | 195     |
  | 1sh_devglobal | shared committer | 1      | 6.93 s  | 144     |
  | 4sh_devglobal | shared committer | 4      | 7.87 s  | **508** |

  4-shard shared committer recovers to within 1.19x of the 1-shard
  baseline (cap 1.5x), a 2.61x improvement over the per-shard fsync
  regression (floor 1.5x). A 100-iteration random-seed crash-recovery
  test passes with zero data loss.

- **Per-platform durability dispatch via `skeg-platform::DurabilityModel`.**
  Linux keeps the per-file group committer (per-file `fdatasync`
  parallelism is already efficient there); macOS routes through the
  shared committer. The model is detected at runtime via
  `resolve_durability_model()`. Tests can override the model via the
  `testing` feature of `skeg-platform`.

### Changed

- **`skeg-core::group_commit::GroupCommitter::start` is now `async`.**
  Required because the shared-committer arm attaches the file to its
  internal registry before the first append. In-tree callers
  (`VLog::open`, `VLog::maybe_rotate`) are already updated to `.await`.
  **Breaking** for any out-of-tree user of `GroupCommitter`.

### Versions bumped

- `skeg-core` 0.1.3 -> 0.2.0 (breaking: `GroupCommitter::start` async)
- `skeg-platform` 0.1.2 -> 0.1.3 (additive: `DurabilityModel`)
- `skeg-server` 0.2.2 -> 0.3.0

## [0.2.2] — 2026-05-29

### Changed

- **Per-vindex locks (Q11 phase 2).** The shard's vindex set was
  previously wrapped in a single `RwLock<HashMap<String,
  VectorBackend>>`, so any `VSET` / `VSEARCH` held the outer write
  lock for its entire duration and blocked operations on every
  other vindex on the same shard. Each entry is now its own
  `Arc<RwLock<VectorBackend>>`:
  - The outer map is held only for the lookup, then released.
  - The per-vindex lock serialises operations on the **same**
    vindex (still required: `VectorBackend::search` mutates the
    working-set cache and the streaming-insert buffer).
  - The worker-pool path (`--workers N` since v0.1) now lifts two
    concurrent `VSEARCH` calls on different vindexes to the
    blocking pool without contention.
  - SoL gate (`test_per_vindex_locks_concurrency_gate`): two-thread
    workload at 2,000-vector, 256-dim flat indexes, `workers=2`,
    requires `baseline / concurrent >= 1.4x`. Measured 1.99x on
    Apple M1 (theoretical max 2.0x). Floor sits below the measurement
    to absorb noise on slower CI runners.

### Notes

- Wire format unchanged, public `ShardSet` API unchanged. Existing
  callers see no behavioural change beyond the new parallelism on
  multi-vindex workloads.
- `VINDEX DROP` keeps its previous semantics: the entry is popped
  from the map before the data directory is removed. In-flight
  operations on the dropped vindex hold their own `Arc` clone and
  finish their inner lock window before dropping it; on POSIX the
  directory deletion is decoupled from the file-handle lifetime.

## [0.2.1] — 2026-05-29

### Added

- All seven engine gauges now report live values; the five that read
  `0` in v0.2.0 are wired:
  - `skeg_vlog_segments_live`     refreshed on each `SKEG.STATS` call
    from `VLog::segment_count()`.
  - `skeg_vlog_total_bytes`       refreshed similarly via the new
    `VLog::disk_bytes_total()` helper (sealed × max_seg_size + active
    write offset, no `stat()`).
  - `skeg_compaction_in_progress` and
    `skeg_vlog_segments_compacting` use an RAII guard in
    `VLog::compact_segment` so every return path (including `?` and
    early no-ops) leaves the gauges balanced.
  - `skeg_vindex_vectors` and `skeg_vindex_size_bytes` are aggregated
    on `STATS` from the shard's vindex set; the size approximation is
    `n * dim * 4` for flat indexes and `n * dim` for disk (tier-1
    int8 codes only — the graph and full f32 vectors live on disk and
    are not counted as RAM).
- `skeg_telemetry::incr_gauge` / `decr_gauge` / `add_gauge` for
  delta-style updates on gauges (used by the compaction RAII guard).
  The closed-enum and dynamic registry APIs both gain the new
  helpers; same `#[inline(always)]` / no-op compile-out story.

### Notes

- v0.2.1 is an additive release: no engine behaviour change, no wire
  format change, no schema change. Dashboards built against v0.2.0
  start displaying real values on the previously-empty gauges with
  no rewrite.

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
