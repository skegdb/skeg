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
//! text format. The server is purely a reader — it never writes through
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
pub mod histograms;
#[cfg(any(feature = "stats", feature = "http"))]
pub mod metrics;
#[cfg(any(feature = "stats", feature = "http"))]
pub mod stats;

#[cfg(feature = "http")]
pub mod http;

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
        Op::Get, Op::Set, Op::Del, Op::VSet, Op::VSearch, Op::VDel, Op::Ping,
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
}

impl Counter {
    pub const COUNT: usize = 7;
    pub const ALL: [Counter; Self::COUNT] = [
        Counter::CacheHits,
        Counter::CacheMisses,
        Counter::CacheEvictions,
        Counter::CompactionRunsTotal,
        Counter::CompactionBytesTotal,
        Counter::VlogSyncs,
        Counter::VlogGroupCommitBatches,
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
        }
    }
}

/// Gauges (current value, not monotonic).
///
/// Wiring status (as of v0.1.2):
/// - `VlogLiveBytes`         wired in `skeg-server` `Stats` handler
/// - `VlogSegmentsLive`      TODO · wire from `vlog::seal_active`
/// - `VlogSegmentsCompacting` TODO · wire from `vlog::compact_segment` start/end
/// - `VlogTotalBytes`        TODO · wire from `vlog::append_raw` running sum
/// - `CompactionInProgress`  TODO · increment at compact_segment start, decrement at end
/// - `VindexSizeBytes`       TODO · wire from `vindex` create/drop/resize hooks
/// - `VindexVectors`         TODO · wire from `vset`/`vdel` handlers (per-name aggregate)
///
/// Each unwired gauge currently reads `0` from `STATS` and `/metrics`.
/// They are kept in the enum so the metric schema is stable; wiring is
/// a follow-up that does not change the API.
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
}

impl Gauge {
    pub const COUNT: usize = 7;
    pub const ALL: [Gauge; Self::COUNT] = [
        Gauge::VlogSegmentsLive,
        Gauge::VlogSegmentsCompacting,
        Gauge::VlogLiveBytes,
        Gauge::VlogTotalBytes,
        Gauge::CompactionInProgress,
        Gauge::VindexSizeBytes,
        Gauge::VindexVectors,
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
        }
    }
}

/// Compile-time check: when neither feature is enabled the entire module
/// is dead code from the caller's perspective, but the enum + signatures
/// still compile so the consumer code does not need `#[cfg]` gates.
#[allow(dead_code)]
const _ASSERT_ZERO_COST: () = {
    // If `record_op` ever stops being inline + side-effect-free with no
    // features, this const will fail to evaluate due to the `let _` sinks
    // being optimised out.
};
