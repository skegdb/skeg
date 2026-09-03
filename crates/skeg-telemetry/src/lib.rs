//! Zero-overhead telemetry for skeg.
//!
//! All public API entry points are `#[inline(always)]`. When neither the
//! `stats` nor `http` feature is enabled, every call collapses to a no-op
//! the compiler eliminates (verified with `cargo asm`).
//!
//! When `stats` is enabled (default), the static counters and histograms
//! tick on the hot path with a single atomic fetch_add each. Reading the
//! values is done by [`stats::dump_text`] (or the helper accessors on
//! [`metrics`] / [`histograms`]); reading does not lock, and never blocks
//! the hot path.
//!
//! When `http` is also enabled, [`http::serve_blocking`] runs a tiny
//! HTTP server on a dedicated thread that serves `/metrics` in Prometheus
//! text format. The server is purely a reader - it never writes through
//! the hot path.
//!
//! # Hot-path cost budget
//!
//! - per-op counter tick: `AtomicU64::fetch_add(1, Relaxed)` ≈ 1–2 ns
//! - per-op histogram tick: leading-zeros bucket pick + one `fetch_add` ≈ 3–5 ns
//!
//! The crate's `benches/overhead.rs` gates these with criterion; CI fails
//! the build if any record path exceeds 50 ns.

#![cfg_attr(not(any(feature = "stats", feature = "http")), allow(dead_code))]

#[cfg(any(feature = "stats", feature = "http"))]
pub mod dynamic;
#[cfg(any(feature = "stats", feature = "http"))]
pub mod histograms;
#[cfg(any(feature = "stats", feature = "http"))]
pub mod metrics;
#[cfg(any(feature = "stats", feature = "http"))]
pub mod sources;
#[cfg(any(feature = "stats", feature = "http"))]
pub mod stats;

#[cfg(feature = "http")]
pub mod http;

// ───────────────────────────────────────────────────────────────────────────
// Re-exports for the dynamic registry (v0.2.0). Downstream crates that need
// their own metrics should reach for these instead of patching the closed
// enums below; see [`dynamic`] for the design rationale and pool sizing.
// ───────────────────────────────────────────────────────────────────────────

#[cfg(any(feature = "stats", feature = "http"))]
pub use dynamic::{DynHistogram, DynOp, register_counter, register_gauge, register_histogram};

#[cfg(any(feature = "stats", feature = "http"))]
pub use metrics::MAX_SHARDS;

#[cfg(any(feature = "stats", feature = "http"))]
pub use sources::{GaugeSample, GaugeSource, register_gauge_source};

/// Enumeration of operations tracked on the hot path.
///
/// Kept small and `repr(usize)` so it indexes directly into the static
/// metric arrays. Add variants here when a new hot-path operation needs
/// counting; the array sizes in [`metrics`] track this enum.
#[repr(usize)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Op {
    /// `GET key` (scalar lookup).
    Get = 0,
    /// `SET key val` (scalar store; group-committed downstream).
    Set = 1,
    /// `DEL key` (tombstone).
    Del = 2,
    /// `VSET name vec` (vector store).
    VSet = 3,
    /// `VSEARCH name vec k` (vector top-k search).
    VSearch = 4,
    /// `VDEL name id` (vector tombstone).
    VDel = 5,
    /// `PING` (round-trip probe).
    Ping = 6,
}

impl Op {
    /// Number of variants. Update array sizes in [`metrics`] if this grows.
    pub const COUNT: usize = 7;

    /// All variants in declaration order. Used by the dumpers to iterate
    /// without unsafe transmutes.
    pub const ALL: [Op; Self::COUNT] = [
        Op::Get,
        Op::Set,
        Op::Del,
        Op::VSet,
        Op::VSearch,
        Op::VDel,
        Op::Ping,
    ];

    /// Compact textual name used in metric labels.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Op::Get => "get",
            Op::Set => "set",
            Op::Del => "del",
            Op::VSet => "vset",
            Op::VSearch => "vsearch",
            Op::VDel => "vdel",
            Op::Ping => "ping",
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Public hot-path API.
// Every function is `#[inline(always)]`. With no telemetry feature enabled
// the body is empty and the parameters are forced into `let _ = …` sinks
// so the compiler treats them as side-effect-free and removes the calls.
// ───────────────────────────────────────────────────────────────────────────

/// Record completion of one operation, with its observed duration.
///
/// `shard_id` is the worker shard that handled the request (used to
/// partition counters and avoid cross-core cache-line contention).
#[inline(always)]
pub fn record_op(op: Op, shard_id: u16, duration: core::time::Duration) {
    #[cfg(any(feature = "stats", feature = "http"))]
    {
        metrics::tick_op(op, shard_id);
        histograms::observe_us(op, duration.as_micros() as u64);
    }
    #[cfg(not(any(feature = "stats", feature = "http")))]
    {
        let _ = (op, shard_id, duration);
    }
}

/// Operations of one kind recorded so far.
///
/// The read side of [`record_op`], next to [`counter_value`]: callers can
/// assert on either without reaching into the `stats` internals. Returns 0
/// when metrics are compiled out.
pub fn op_total(op: Op) -> u64 {
    #[cfg(any(feature = "stats", feature = "http"))]
    {
        metrics::op_total(op)
    }
    #[cfg(not(any(feature = "stats", feature = "http")))]
    {
        let _ = op;
        0
    }
}

/// Current value of one counter.
///
/// Exposed so callers can assert on a counter without reaching into the
/// `stats` internals; returns 0 when metrics are compiled out.
pub fn counter_value(c: Counter) -> u64 {
    #[cfg(any(feature = "stats", feature = "http"))]
    {
        metrics::counter(c)
    }
    #[cfg(not(any(feature = "stats", feature = "http")))]
    {
        let _ = c;
        0
    }
}

/// Set the current value of a gauge metric (overwrites; not a counter).
#[inline(always)]
pub fn set_gauge(g: Gauge, value: u64) {
    #[cfg(any(feature = "stats", feature = "http"))]
    {
        metrics::set_gauge(g, value);
    }
    #[cfg(not(any(feature = "stats", feature = "http")))]
    {
        let _ = (g, value);
    }
}

/// Increment a gauge by one. Pair with [`decr_gauge`] for "in flight"
/// counters where the natural API is `incr` at the start of an
/// operation and `decr` at the end.
#[inline(always)]
pub fn incr_gauge(g: Gauge) {
    #[cfg(any(feature = "stats", feature = "http"))]
    {
        metrics::incr_gauge(g);
    }
    #[cfg(not(any(feature = "stats", feature = "http")))]
    {
        let _ = g;
    }
}

/// Decrement a gauge by one. Safe to call when the gauge is already
/// zero (wraps; pair calls correctly with [`incr_gauge`] for symmetry).
#[inline(always)]
pub fn decr_gauge(g: Gauge) {
    #[cfg(any(feature = "stats", feature = "http"))]
    {
        metrics::decr_gauge(g);
    }
    #[cfg(not(any(feature = "stats", feature = "http")))]
    {
        let _ = g;
    }
}

/// Increment a counter that is not tied to a specific operation.
#[inline(always)]
pub fn tick_counter(c: Counter) {
    #[cfg(any(feature = "stats", feature = "http"))]
    {
        metrics::tick_counter(c, 1);
    }
    #[cfg(not(any(feature = "stats", feature = "http")))]
    {
        let _ = c;
    }
}

/// Add a delta to a counter (for batch / amortised paths).
#[inline(always)]
pub fn add_counter(c: Counter, delta: u64) {
    #[cfg(any(feature = "stats", feature = "http"))]
    {
        metrics::tick_counter(c, delta);
    }
    #[cfg(not(any(feature = "stats", feature = "http")))]
    {
        let _ = (c, delta);
    }
}

/// Counters that exist outside the per-op hot path.
#[repr(usize)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Counter {
    CacheHits = 0,
    CacheMisses = 1,
    CacheEvictions = 2,
    CompactionRunsTotal = 3,
    CompactionBytesTotal = 4,
    VlogSyncs = 5,
    VlogGroupCommitBatches = 6,
    /// `Durability::Power` flushes that took the Linux `fdatasync` fast path
    /// (the file's length was fixed by `PlatformFile::preallocate`) rather
    /// than the full `fsync`/`F_FULLFSYNC`. Compare against `VlogSyncs` to
    /// see how much of the durability traffic is on the cheap path.
    VlogFdatasyncFastPath = 7,
    /// Segment files preallocated to the rotation cap: fresh
    /// stores, rotations, and the active segment re-armed on recovery.
    VlogPreallocations = 8,
    /// Bytes written through `pwritev` instead of a
    /// combined-buffer copy - one flush_batch's total payload per tick.
    VlogPwritevBytesTotal = 9,
    /// `sync_file_range` writeback hints issued, one per
    /// destination segment per compaction run - paces dirty-page writeback
    /// ahead of the durability call that follows, instead of letting a
    /// whole compaction's worth of dirty pages pile up unflushed.
    VlogWritebackHints = 10,
    /// Records decoded while replaying the log at open. This is what makes a
    /// restart slow: a snapshot is supposed to cover most of them, and a value
    /// close to the total key count means it is not doing its job.
    VlogRecoveryRecords = 11,
    /// Payload indexes rebuilt from stored blobs. The rebuild reads every live
    /// id's payload, so it belongs at open, not on a user's query: a non-zero
    /// value while serving means some search paid for it.
    PayloadIndexRebuilds = 12,
    /// Ids whose payload index came from `payload.idx` and was used as it
    /// stood. Zero after a restart that had the file means it was refused, and
    /// the reason is worth knowing: a stamp mismatch, a damaged file, or a log
    /// tail long enough that nothing in it was usable.
    ///
    /// Counts ids taken, not ids covered: an id the log tail touched is read
    /// back from the log instead, and counting it here would hide exactly the
    /// protection that makes the file safe to trust.
    PayloadIndexFromDisk = 13,
    /// Ids the file covered but that had to be read from the log anyway,
    /// because the tail touched them after the file was stamped. This is what
    /// a stale snapshot costs at open.
    PayloadIndexRefreshed = 14,
    /// Maintenance operations run, by kind. Which one fires decides whether a
    /// store scales: a fold rebuilds the whole base and costs O(live), the
    /// other three are proportional to what changed. Without these you cannot
    /// tell a healthy store from one folding itself to death.
    MaintenanceFlush = 15,
    MaintenanceConsolidate = 16,
    MaintenanceRunsMerge = 17,
    MaintenanceDeletePatch = 18,
    /// Maintenance ops that found the fold budget full and skipped their tick
    /// instead of parking. A few are the budget working; a sustained climb
    /// means heavy folds are starving the cheap maintenance of its turns.
    MaintenanceBudgetSkips = 19,
    /// Re-rank row served from the per-segment cache instead of a positioned
    /// read.
    RerankCacheHits = 20,
    /// Re-rank row read from disk (and inserted into the cache).
    RerankCacheMisses = 21,
    /// Nanoseconds spent in the graph walks of vsearch (all segments).
    VsearchWalkNanos = 22,
    /// Nanoseconds spent in the bounded disk re-rank of vsearch.
    VsearchRerankNanos = 23,
    /// Disk rows read by the re-rank.
    VsearchRerankReads = 24,
    /// Nanoseconds spent flat-scanning the delta and flush staging.
    VsearchDeltaNanos = 25,
    /// Re-rank candidates skipped by the adaptive bound.
    RerankAdaptiveSkips = 26,
    /// Vsearches that carried a payload filter (the oversampled walk and the
    /// selectivity-scaled re-rank budget).
    VsearchFiltered = 27,
    /// Filtered/hybrid searches (score_ids path), by count.
    VsearchHybrid = 28,
    /// Nanoseconds spent in the hybrid path (proxy scoring of the id set).
    VsearchHybridScoreNanos = 29,
    /// Nanoseconds the hybrid path spent in its f32 disk re-rank.
    VsearchHybridRerankNanos = 30,
    /// Disk rows the hybrid path re-ranked.
    VsearchHybridReads = 31,
    /// Ids the hybrid path proxy-scored (the |S| or shortlist actually paid).
    VsearchHybridScored = 32,
    /// Walk seeded from the semantic entry cache (query-sketch hit).
    EntryCacheHits = 33,
    /// Entry-cache lookups that found no sketch match.
    EntryCacheMisses = 34,
    /// Graph nodes expanded by search walks (sum; divide by VsearchInner for
    /// the average hops a query pays to reach its answers).
    VsearchWalkHops = 35,
    /// Searches that took the graph-walk route (the denominator for
    /// per-search walk averages; the hybrid route has its own counters).
    VsearchInner = 36,
    /// Filtered search, phase 1: turning the match set into base rows
    /// (external id -> row lookup). O(|S|) and invisible until now - the IVF
    /// router cuts scoring and rerank, never this.
    VsearchHybridMapNanos = 37,
    /// Filtered search, phase 2: the IVF router bucketing the rows it was
    /// handed and picking the shortlist. Also O(|S|).
    VsearchHybridRouteNanos = 38,
    /// Run segments scanned instead of walked because the run count is over
    /// the ceiling: the visible symptom of maintenance falling behind.
    RunScanFallback = 39,
    /// Maintenance jobs that ran and failed. Previously indistinguishable
    /// from "nothing to do", which hid real failures from every signal.
    MaintenanceFailures = 40,
    /// Vacuum attempts declined: not enough garbage, or already done for
    /// this generation of runs. A rising count next to a flat reclaimed
    /// count is what an infinite rewrite loop looks like from outside.
    VacuumSkipped = 41,
    /// Maintenance jobs that COMMITTED but could not reclaim what they
    /// replaced: an old run directory left on disk, a router sidecar not
    /// written, a WAL not re-encoded. The work happened - this is not a
    /// failure count - but each one leaves something behind, and a rising
    /// number means disk that nothing will ever free on its own.
    MaintenanceCleanupFailures = 42,
    /// Writes refused because the memory budget was gone. Zero on a store with
    /// no ceiling; a climb is the engine telling a client to slow down, which
    /// is the signal an operator needs BEFORE the alternative, which is the
    /// kernel telling the process to stop.
    MemoryRefused = 43,
    /// Payload blobs written AHEAD of the commit that publishes their row.
    ///
    /// The denominator for the two below, and the only place a staged blob is
    /// visible at all: until the WAL record lands, nothing can read one.
    PayloadBlobsStaged = 44,
    /// Staged blobs that are a COPY of the row's existing payload, made
    /// because an overwrite carried no payload of its own and the row's new
    /// version needs its blob under a new key. The cost of keeping a
    /// payload-less overwrite's payload; a climb here against a flat
    /// `PayloadBlobsStaged` is a client sending vectors without their
    /// payloads.
    PayloadBlobsCarriedForward = 45,
    /// Steps AFTER a commit point that did not complete: the payload postings
    /// not indexed, a superseded blob not reclaimed, a deleted row's blob not
    /// reclaimed.
    ///
    /// None of them can be reported to the client - the write is durable and
    /// saying otherwise invites a retry of something that has happened - so
    /// without this number they are invisible outside the log. Each one leaves
    /// either a filter that misses a row until its postings are rebuilt, or a
    /// blob the next open collects.
    PayloadPostCommitFailures = 46,
    /// Payload blobs collected at open because no live row named them.
    ///
    /// A steady non-zero value across restarts means writes are dying between
    /// their staging and their commit, or post-commit reclamation keeps
    /// failing - both of which have their own counter above. A one-off after a
    /// crash is the mechanism working.
    PayloadBlobsReclaimedAtOpen = 47,
    /// Connections refused at accept because the ingress class was full. This
    /// is the budget doing its job in the only place it can be done cheaply;
    /// a sustained climb means the cap is too small for the client count, not
    /// that a client is misbehaving.
    IngressRefusedAccept = 48,
    /// Frames refused because the connection could not be granted the buffer
    /// they needed. Separate from the accept counter on purpose: one is about
    /// how many peers there are, the other about how big their frames are.
    IngressRefusedGrowth = 49,
    /// Reads paused because a growth was refused and the connection waited to
    /// see whether the room came back. A stall that ends in room is invisible
    /// to the client except as latency, so without this it is invisible full
    /// stop.
    IngressStalls = 50,
    /// Connections accepted while the ingress budget could not be established
    /// (a cgroup limit applies and its headroom is unreadable) and were
    /// therefore given their floor and no growth. Non-zero means the server is
    /// serving small frames only, which a client experiences as refusals it
    /// cannot explain.
    IngressBudgetUnreadable = 51,
    /// Replies whose buffer the ingress class could not charge for.
    ///
    /// "Could not charge", not "was large". The reply is still written - it is
    /// the answer to work that has already committed, and withdrawing it would
    /// make a client retry something that has happened - so this is the one
    /// place the budget is knowingly exceeded. Counted rather than enforced,
    /// and the buffer is handed back immediately afterwards.
    ///
    /// Under a full class it therefore ticks for ANY reply, including a
    /// seven-byte `+PONG`: what failed is the charge, not the reply. Read it
    /// next to `IngressRefusedGrowth` as a signal that the class is full, and
    /// not as a measure of how big replies are getting.
    IngressReplyOverBudget = 52,
    /// Writes refused because the tenant was at its vector limit.
    ///
    /// The quota was the one admission decision with no number attached: it
    /// refused, the client saw a line of text, and nothing counted it - so a
    /// tenant hitting its ceiling looked from outside exactly like a tenant
    /// that had stopped writing.
    QuotaRefused = 53,
    /// Tenant-backend refusals whose message carried no code word this build
    /// knows.
    ///
    /// The backend trait asks for a leading uppercase code and the engine
    /// classifies the refusal from it. A backend that sends prose, or a word
    /// this build has never heard of, gets the safe answer - permanent - and
    /// this counter, because the alternative is a rate limit that silently
    /// reads as "give up" to every client of that deployment.
    BackendRefusalUnclassified = 54,
    /// Bytes of KV VALUE materialised to answer a `GET`/`MGET`, on either
    /// wire.
    ///
    /// The denominator of the read-side admission story, and the only way to
    /// see the order from outside: a read whose reply was refused before the
    /// store was touched leaves this number exactly where it was, and a
    /// refusal that arrived after the values had already been fetched does
    /// not. Counted at the shard worker, next to the fetch itself, so it
    /// cannot be true by construction in the handler that is being tested.
    KvReadBytesFetched = 55,
    /// `GET`/`MGET` requests refused by the read preflight: the summed value
    /// lengths did not fit the connection's allowance, or a value grew past
    /// the size that was reserved for it between the measurement and the
    /// read.
    ///
    /// Separate from `IngressRefusedGrowth`, which counts a connection
    /// refused a BUFFER. This one counts a request refused an ANSWER, which
    /// an operator reads differently: it means clients are asking for more
    /// than the class can hand back at once, not that they are sending more
    /// than it can take in.
    KvReadRefused = 56,
}

impl Counter {
    pub const COUNT: usize = 57;
    pub const ALL: [Counter; Self::COUNT] = [
        Counter::CacheHits,
        Counter::CacheMisses,
        Counter::CacheEvictions,
        Counter::CompactionRunsTotal,
        Counter::CompactionBytesTotal,
        Counter::VlogSyncs,
        Counter::VlogGroupCommitBatches,
        Counter::VlogFdatasyncFastPath,
        Counter::VlogPreallocations,
        Counter::VlogPwritevBytesTotal,
        Counter::VlogWritebackHints,
        Counter::VlogRecoveryRecords,
        Counter::PayloadIndexRebuilds,
        Counter::PayloadIndexFromDisk,
        Counter::PayloadIndexRefreshed,
        Counter::MaintenanceFlush,
        Counter::MaintenanceConsolidate,
        Counter::MaintenanceRunsMerge,
        Counter::MaintenanceDeletePatch,
        Counter::MaintenanceBudgetSkips,
        Counter::RerankCacheHits,
        Counter::RerankCacheMisses,
        Counter::VsearchWalkNanos,
        Counter::VsearchRerankNanos,
        Counter::VsearchRerankReads,
        Counter::VsearchDeltaNanos,
        Counter::RerankAdaptiveSkips,
        Counter::VsearchFiltered,
        Counter::VsearchHybrid,
        Counter::VsearchHybridScoreNanos,
        Counter::VsearchHybridRerankNanos,
        Counter::VsearchHybridReads,
        Counter::VsearchHybridScored,
        Counter::EntryCacheHits,
        Counter::EntryCacheMisses,
        Counter::VsearchWalkHops,
        Counter::VsearchInner,
        Counter::VsearchHybridMapNanos,
        Counter::VsearchHybridRouteNanos,
        Counter::RunScanFallback,
        Counter::MaintenanceFailures,
        Counter::MaintenanceCleanupFailures,
        Counter::MemoryRefused,
        Counter::VacuumSkipped,
        Counter::PayloadBlobsStaged,
        Counter::PayloadBlobsCarriedForward,
        Counter::PayloadPostCommitFailures,
        Counter::PayloadBlobsReclaimedAtOpen,
        Counter::IngressRefusedAccept,
        Counter::IngressRefusedGrowth,
        Counter::IngressStalls,
        Counter::IngressBudgetUnreadable,
        Counter::IngressReplyOverBudget,
        Counter::QuotaRefused,
        Counter::BackendRefusalUnclassified,
        Counter::KvReadBytesFetched,
        Counter::KvReadRefused,
    ];

    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Counter::CacheHits => "skeg_cache_hits_total",
            Counter::CacheMisses => "skeg_cache_misses_total",
            Counter::CacheEvictions => "skeg_cache_evictions_total",
            Counter::CompactionRunsTotal => "skeg_compaction_runs_total",
            Counter::CompactionBytesTotal => "skeg_compaction_bytes_total",
            Counter::VlogSyncs => "skeg_vlog_syncs_total",
            Counter::VlogGroupCommitBatches => "skeg_vlog_group_commit_batches_total",
            Counter::VlogFdatasyncFastPath => "skeg_vlog_fdatasync_fastpath_total",
            Counter::VlogPreallocations => "skeg_vlog_preallocations_total",
            Counter::VlogPwritevBytesTotal => "skeg_vlog_pwritev_bytes_total",
            Counter::VlogWritebackHints => "skeg_vlog_writeback_hints_total",
            Counter::VlogRecoveryRecords => "skeg_vlog_recovery_records_total",
            Counter::PayloadIndexRebuilds => "skeg_payload_index_rebuilds_total",
            Counter::PayloadIndexFromDisk => "skeg_payload_index_from_disk_total",
            Counter::PayloadIndexRefreshed => "skeg_payload_index_refreshed_total",
            Counter::MaintenanceFlush => "skeg_maintenance_flush_total",
            Counter::MaintenanceConsolidate => "skeg_maintenance_consolidate_total",
            Counter::MaintenanceRunsMerge => "skeg_maintenance_runs_merge_total",
            Counter::MaintenanceDeletePatch => "skeg_maintenance_delete_patch_total",
            Counter::MaintenanceBudgetSkips => "skeg_maintenance_budget_skips_total",
            Counter::RerankCacheHits => "skeg_rerank_cache_hits_total",
            Counter::RerankCacheMisses => "skeg_rerank_cache_misses_total",
            Counter::VsearchWalkNanos => "skeg_vsearch_walk_nanos_total",
            Counter::VsearchRerankNanos => "skeg_vsearch_rerank_nanos_total",
            Counter::VsearchRerankReads => "skeg_vsearch_rerank_reads_total",
            Counter::VsearchDeltaNanos => "skeg_vsearch_delta_nanos_total",
            Counter::RerankAdaptiveSkips => "skeg_rerank_adaptive_skips_total",
            Counter::VsearchFiltered => "skeg_vsearch_filtered_total",
            Counter::VsearchHybrid => "skeg_vsearch_hybrid_total",
            Counter::VsearchHybridScoreNanos => "skeg_vsearch_hybrid_score_nanos_total",
            Counter::VsearchHybridRerankNanos => "skeg_vsearch_hybrid_rerank_nanos_total",
            Counter::VsearchHybridReads => "skeg_vsearch_hybrid_reads_total",
            Counter::VsearchHybridScored => "skeg_vsearch_hybrid_scored_total",
            Counter::EntryCacheHits => "skeg_entry_cache_hits_total",
            Counter::EntryCacheMisses => "skeg_entry_cache_misses_total",
            Counter::VsearchWalkHops => "skeg_vsearch_walk_hops_total",
            Counter::VsearchInner => "skeg_vsearch_inner_total",
            Counter::VsearchHybridMapNanos => "skeg_vsearch_hybrid_map_nanos_total",
            Counter::VsearchHybridRouteNanos => "skeg_vsearch_hybrid_route_nanos_total",
            Counter::RunScanFallback => "skeg_run_scan_fallback_total",
            Counter::MaintenanceFailures => "skeg_maintenance_failures_total",
            Counter::MaintenanceCleanupFailures => "skeg_maintenance_cleanup_failures_total",
            Counter::MemoryRefused => "skeg_memory_refused_total",
            Counter::PayloadBlobsStaged => "skeg_payload_blobs_staged_total",
            Counter::PayloadBlobsCarriedForward => "skeg_payload_blobs_carried_forward_total",
            Counter::PayloadPostCommitFailures => "skeg_payload_post_commit_failures_total",
            Counter::PayloadBlobsReclaimedAtOpen => "skeg_payload_blobs_reclaimed_at_open_total",
            Counter::VacuumSkipped => "skeg_vacuum_skipped_total",
            Counter::IngressRefusedAccept => "skeg_ingress_refused_accept_total",
            Counter::IngressRefusedGrowth => "skeg_ingress_refused_growth_total",
            Counter::IngressStalls => "skeg_ingress_stalls_total",
            Counter::IngressBudgetUnreadable => "skeg_ingress_budget_unreadable_total",
            Counter::IngressReplyOverBudget => "skeg_ingress_reply_over_budget_total",
            Counter::QuotaRefused => "skeg_quota_refused_total",
            Counter::BackendRefusalUnclassified => "skeg_backend_refusal_unclassified_total",
            Counter::KvReadBytesFetched => "skeg_kv_read_bytes_total",
            Counter::KvReadRefused => "skeg_kv_read_refused_total",
        }
    }
}

/// Gauges (current value, not monotonic).
///
/// Wiring status (as of v0.2.1):
/// - `VlogLiveBytes`          wired in `skeg-server` `STATS` handler
/// - `VlogSegmentsLive`       wired in `skeg-server` `STATS` handler
/// - `VlogTotalBytes`         wired in `skeg-server` `STATS` handler
/// - `CompactionInProgress`   wired by RAII guard in `vlog::compact_segment`
/// - `VlogSegmentsCompacting` wired by RAII guard in `vlog::compact_segment`
/// - `VindexSizeBytes`        wired in `skeg-server` `STATS` handler
/// - `VindexVectors`          wired in `skeg-server` `STATS` handler
///
/// The vlog-segment and vindex gauges refresh on every `STATS` call
/// (cheap arithmetic, no allocation). The compaction gauges use
/// `incr`/`decr` so the count is accurate between polls.
#[repr(usize)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Gauge {
    VlogSegmentsLive = 0,
    VlogSegmentsCompacting = 1,
    VlogLiveBytes = 2,
    VlogTotalBytes = 3,
    CompactionInProgress = 4,
    VindexSizeBytes = 5,
    VindexVectors = 6,
    /// Heavy maintenance builds parked at the process-wide fold budget. A
    /// non-zero value while latency is fine means the budget is doing its job;
    /// a persistently high one means folds are being produced faster than the
    /// budget lets them retire.
    FoldsWaiting = 7,
}

impl Gauge {
    pub const COUNT: usize = 8;
    pub const ALL: [Gauge; Self::COUNT] = [
        Gauge::VlogSegmentsLive,
        Gauge::VlogSegmentsCompacting,
        Gauge::VlogLiveBytes,
        Gauge::VlogTotalBytes,
        Gauge::CompactionInProgress,
        Gauge::VindexSizeBytes,
        Gauge::VindexVectors,
        Gauge::FoldsWaiting,
    ];

    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Gauge::VlogSegmentsLive => "skeg_vlog_segments_live",
            Gauge::VlogSegmentsCompacting => "skeg_vlog_segments_compacting",
            Gauge::VlogLiveBytes => "skeg_vlog_live_bytes",
            Gauge::VlogTotalBytes => "skeg_vlog_total_bytes",
            Gauge::CompactionInProgress => "skeg_compaction_in_progress",
            Gauge::VindexSizeBytes => "skeg_vindex_size_bytes",
            Gauge::VindexVectors => "skeg_vindex_vectors",
            Gauge::FoldsWaiting => "skeg_folds_waiting",
        }
    }
}
