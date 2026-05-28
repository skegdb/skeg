//! Dynamic metric registry — extensibility without giving up zero overhead.
//!
//! The closed enums [`crate::Op`], [`crate::Counter`], [`crate::Gauge`] are
//! great for the engine itself but they freeze the metric set at compile
//! time: a downstream consumer (`skeg-kv-cache`, `skeg-tenant`, an
//! application built on top of skeg) cannot add its own counters without
//! patching this crate.
//!
//! This module fixes that without compromising hot-path cost. The trick:
//!
//! 1. A fixed-size **static pool** of atomics is allocated up front
//!    (`COUNTER_POOL`, `HIST_POOL`, `GAUGE_POOL`). Each entry is a slot
//!    a downstream metric can claim.
//! 2. [`register_counter`], [`register_histogram`], [`register_gauge`]
//!    map a `&'static str` name to a slot index through a one-time
//!    `Mutex<BTreeMap>` lookup. The returned handle is a `&'static`
//!    reference into the static pool — the same kind of pointer the
//!    closed-enum API uses internally.
//! 3. The downstream caller is expected to cache the handle in an
//!    `OnceLock` (or a static initialiser) so the hot path is one
//!    `OnceLock::get()` branch + one `AtomicU64::fetch_add(_, Relaxed)`.
//!    Total cost: ~2 ns — same order as the closed-enum path.
//!
//! Idempotency: repeated calls with the same name return the same slot,
//! so `OnceLock::get_or_init(|| register_*(…))` patterns are safe even
//! if the same code path runs from multiple translation units.
//!
//! # Pool sizing
//!
//! | Pool      | Slots | Per-slot bytes                | Total static |
//! |-----------|------:|-------------------------------|-------------:|
//! | counters  |   256 | [AtomicU64; MAX_SHARDS] = 256 |       64 KiB |
//! | histograms|    64 | DynHistogram ≈ 224 B          |     14.3 KiB |
//! | gauges    |    64 | AtomicU64 = 8 B               |        512 B |
//!
//! Counters are sharded along `MAX_SHARDS` to mirror the closed-enum
//! `OP_COUNTERS` layout (consistency, and ready for multi-shard use).
//! Histograms are *not* sharded yet (matches the closed-enum
//! `HISTOGRAMS`). Gauges are scalar (`store` semantics, sharding is
//! ill-defined for set-style values).
//!
//! # When the pool fills
//!
//! Exhausting a pool is a programming error (the engine ships with
//! O(10) metrics; a downstream that registers thousands of distinct
//! names is doing something wrong). The registration functions panic
//! in that case with the slot kind and current cap in the message.

use core::sync::atomic::{AtomicU64, Ordering};
use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::histograms::{BUCKET_BOUNDS_US, BUCKETS, bucket_idx};
use crate::metrics::MAX_SHARDS;

const COUNTER_POOL_SIZE: usize = 256;
const HIST_POOL_SIZE: usize = 64;
const GAUGE_POOL_SIZE: usize = 64;

/// Per-name sharded counter array. The caller picks a slot with the
/// current shard id; cross-shard reads are summed in [`dump_text`].
type CounterRow = [AtomicU64; MAX_SHARDS];

static COUNTER_POOL: [CounterRow; COUNTER_POOL_SIZE] =
    [const { [const { AtomicU64::new(0) }; MAX_SHARDS] }; COUNTER_POOL_SIZE];

static HIST_POOL: [DynHistogram; HIST_POOL_SIZE] = [const { DynHistogram::new() }; HIST_POOL_SIZE];

static GAUGE_POOL: [AtomicU64; GAUGE_POOL_SIZE] = [const { AtomicU64::new(0) }; GAUGE_POOL_SIZE];

/// Three independent registries — name → slot index. Iterated in
/// sorted order by [`dump_text`], so `BTreeMap` (not `HashMap`) gives
/// deterministic output for free.
static COUNTER_REGISTRY: Mutex<BTreeMap<&'static str, usize>> = Mutex::new(BTreeMap::new());
static HIST_REGISTRY: Mutex<BTreeMap<&'static str, usize>> = Mutex::new(BTreeMap::new());
static GAUGE_REGISTRY: Mutex<BTreeMap<&'static str, usize>> = Mutex::new(BTreeMap::new());

/// A histogram exposed to dynamic consumers. Layout mirrors the
/// closed-enum [`crate::histograms`] static: per-bucket counts plus a
/// total observation count and microsecond sum (Prometheus
/// `_count` / `_sum` lines).
#[repr(C)]
pub struct DynHistogram {
    buckets: [AtomicU64; BUCKETS],
    count: AtomicU64,
    sum_us: AtomicU64,
}

impl DynHistogram {
    /// `const`-constructible so it can live in a static array.
    pub const fn new() -> Self {
        Self {
            buckets: [const { AtomicU64::new(0) }; BUCKETS],
            count: AtomicU64::new(0),
            sum_us: AtomicU64::new(0),
        }
    }

    /// Record one observation in microseconds.
    ///
    /// Same cost profile as [`crate::histograms::observe_us`]: 1 branch,
    /// 1 `leading_zeros`, 3 `fetch_add(Relaxed)`.
    #[inline(always)]
    pub fn observe(&self, us: u64) {
        let b = bucket_idx(us);
        self.buckets[b].fetch_add(1, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum_us.fetch_add(us, Ordering::Relaxed);
    }
}

impl Default for DynHistogram {
    fn default() -> Self {
        Self::new()
    }
}

/// A counter + histogram pair, the idiomatic "operation" handle.
///
/// Construct via the [`register_op!`] macro, which derives the two
/// metric names from a single base (`<base>_total` and
/// `<base>_duration_seconds`) at compile time.
#[derive(Copy, Clone)]
pub struct DynOp {
    pub counter: &'static CounterRow,
    pub histogram: &'static DynHistogram,
}

impl DynOp {
    /// Record one operation completing on `shard_id` with `us`
    /// microseconds elapsed.
    #[inline(always)]
    pub fn record(&self, shard_id: u16, us: u64) {
        let s = (shard_id as usize) & (MAX_SHARDS - 1);
        self.counter[s].fetch_add(1, Ordering::Relaxed);
        self.histogram.observe(us);
    }
}

/// Register (or look up) a sharded counter by name.
///
/// The returned handle is `&'static [AtomicU64; MAX_SHARDS]`. The hot
/// path is `handle[shard_id & (MAX_SHARDS - 1)].fetch_add(1, Relaxed)`.
/// If you don't have a shard concept, pass `0` — the cost is the same.
pub fn register_counter(name: &'static str) -> &'static CounterRow {
    let mut reg = COUNTER_REGISTRY.lock().unwrap_or_else(|p| p.into_inner());
    let idx = match reg.get(name) {
        Some(&idx) => idx,
        None => {
            let idx = reg.len();
            assert!(
                idx < COUNTER_POOL_SIZE,
                "skeg-telemetry counter pool exhausted (cap {COUNTER_POOL_SIZE}); registering {name:?}"
            );
            reg.insert(name, idx);
            idx
        }
    };
    drop(reg);
    &COUNTER_POOL[idx]
}

/// Register (or look up) a histogram by name.
pub fn register_histogram(name: &'static str) -> &'static DynHistogram {
    let mut reg = HIST_REGISTRY.lock().unwrap_or_else(|p| p.into_inner());
    let idx = match reg.get(name) {
        Some(&idx) => idx,
        None => {
            let idx = reg.len();
            assert!(
                idx < HIST_POOL_SIZE,
                "skeg-telemetry histogram pool exhausted (cap {HIST_POOL_SIZE}); registering {name:?}"
            );
            reg.insert(name, idx);
            idx
        }
    };
    drop(reg);
    &HIST_POOL[idx]
}

/// Register (or look up) a scalar gauge by name.
///
/// Gauges use `store` semantics; per-shard sharding is intentionally
/// not provided because aggregating `store` across shards has no
/// well-defined meaning (last-write? max? sum?). Pick "counter
/// stored / counter deleted" instead if you want a process-wide
/// invariant computed in dashboards.
pub fn register_gauge(name: &'static str) -> &'static AtomicU64 {
    let mut reg = GAUGE_REGISTRY.lock().unwrap_or_else(|p| p.into_inner());
    let idx = match reg.get(name) {
        Some(&idx) => idx,
        None => {
            let idx = reg.len();
            assert!(
                idx < GAUGE_POOL_SIZE,
                "skeg-telemetry gauge pool exhausted (cap {GAUGE_POOL_SIZE}); registering {name:?}"
            );
            reg.insert(name, idx);
            idx
        }
    };
    drop(reg);
    &GAUGE_POOL[idx]
}

/// Sum a sharded counter row.
pub fn counter_total(row: &CounterRow) -> u64 {
    row.iter().map(|a| a.load(Ordering::Relaxed)).sum()
}

/// Snapshot helpers used by [`dump_text`].
pub fn hist_bucket(h: &DynHistogram, idx: usize) -> u64 {
    h.buckets[idx].load(Ordering::Relaxed)
}

pub fn hist_count(h: &DynHistogram) -> u64 {
    h.count.load(Ordering::Relaxed)
}

pub fn hist_sum_us(h: &DynHistogram) -> u64 {
    h.sum_us.load(Ordering::Relaxed)
}

/// Serialise the entire dynamic registry into `out` in Prometheus
/// text format. Output is sorted alphabetically (a `BTreeMap` walk),
/// so the dump is stable across runs regardless of init order.
pub(crate) fn dump_text(out: &mut String) {
    use core::fmt::Write;

    // Counters.
    let counters = COUNTER_REGISTRY.lock().unwrap_or_else(|p| p.into_inner());
    if !counters.is_empty() {
        out.push('\n');
        for (name, &idx) in counters.iter() {
            let _ = writeln!(out, "# TYPE {name} counter");
            let _ = writeln!(out, "{} {}", name, counter_total(&COUNTER_POOL[idx]));
        }
    }
    drop(counters);

    // Histograms.
    let hists = HIST_REGISTRY.lock().unwrap_or_else(|p| p.into_inner());
    if !hists.is_empty() {
        out.push('\n');
        for (name, &idx) in hists.iter() {
            let h = &HIST_POOL[idx];
            let _ = writeln!(out, "# TYPE {name} histogram");
            let mut cumulative: u64 = 0;
            for (b, &bound_us) in BUCKET_BOUNDS_US.iter().enumerate() {
                cumulative += hist_bucket(h, b);
                let le_str = if b == BUCKETS - 1 {
                    String::from("+Inf")
                } else {
                    let secs = bound_us as f64 / 1_000_000.0;
                    format!("{secs:.6}")
                };
                let _ = writeln!(out, "{name}_bucket{{le=\"{le_str}\"}} {cumulative}");
            }
            let total = hist_count(h);
            let sum_secs = hist_sum_us(h) as f64 / 1_000_000.0;
            let _ = writeln!(out, "{name}_count {total}");
            let _ = writeln!(out, "{name}_sum {sum_secs}");
        }
    }
    drop(hists);

    // Gauges.
    let gauges = GAUGE_REGISTRY.lock().unwrap_or_else(|p| p.into_inner());
    if !gauges.is_empty() {
        out.push('\n');
        for (name, &idx) in gauges.iter() {
            let v = GAUGE_POOL[idx].load(Ordering::Relaxed);
            let _ = writeln!(out, "# TYPE {name} gauge");
            let _ = writeln!(out, "{name} {v}");
        }
    }
}

/// Register a `DynOp` (sharded counter + histogram) from a single base
/// name. Expands at compile time to two calls into the dynamic
/// registries with the canonical Prometheus suffixes.
///
/// ```ignore
/// use std::sync::OnceLock;
/// use skeg_telemetry::DynOp;
///
/// static LOOKUP: OnceLock<DynOp> = OnceLock::new();
///
/// fn observe(shard_id: u16, us: u64) {
///     LOOKUP
///         .get_or_init(|| skeg_telemetry::register_op!("kv_cache_lookup"))
///         .record(shard_id, us);
/// }
/// ```
///
/// Emits `kv_cache_lookup_total` (sharded counter) and
/// `kv_cache_lookup_duration_seconds` (histogram).
#[macro_export]
macro_rules! register_op {
    ($base:literal) => {{
        let counter = $crate::register_counter(concat!($base, "_total"));
        let histogram = $crate::register_histogram(concat!($base, "_duration_seconds"));
        $crate::DynOp { counter, histogram }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn counter_idempotent_and_ticks() {
        let a = register_counter("test_counter_idem");
        let b = register_counter("test_counter_idem");
        assert!(core::ptr::eq(a, b));
        a[0].fetch_add(3, Ordering::Relaxed);
        a[5].fetch_add(2, Ordering::Relaxed);
        assert_eq!(counter_total(a), 5);
    }

    #[test]
    fn histogram_idempotent_and_observes() {
        let a = register_histogram("test_hist_idem");
        let b = register_histogram("test_hist_idem");
        assert!(core::ptr::eq(a, b));
        a.observe(50);
        a.observe(1500);
        assert_eq!(hist_count(a), 2);
        assert_eq!(hist_sum_us(a), 1550);
    }

    #[test]
    fn gauge_idempotent_and_stores() {
        let a = register_gauge("test_gauge_idem");
        let b = register_gauge("test_gauge_idem");
        assert!(core::ptr::eq(a, b));
        a.store(42, Ordering::Relaxed);
        assert_eq!(a.load(Ordering::Relaxed), 42);
    }

    #[test]
    fn register_op_macro_expands() {
        let op: DynOp = register_op!("test_macro_op");
        op.record(0, 100);
        op.record(1, 200);
        assert_eq!(counter_total(op.counter), 2);
        assert_eq!(hist_count(op.histogram), 2);
        assert_eq!(hist_sum_us(op.histogram), 300);
    }

    #[test]
    fn dump_includes_registered_metrics() {
        let _ = register_counter("test_dump_counter");
        let _ = register_gauge("test_dump_gauge");
        let _ = register_histogram("test_dump_hist");
        let mut out = String::new();
        dump_text(&mut out);
        assert!(out.contains("test_dump_counter"));
        assert!(out.contains("test_dump_gauge"));
        assert!(out.contains("test_dump_hist_bucket"));
        assert!(out.contains("test_dump_hist_count"));
        assert!(out.contains("test_dump_hist_sum"));
    }
}
