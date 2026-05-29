//! Static atomic metric storage.
//!
//! Per-op counters are sharded along `MAX_SHARDS` so the M-series P-cores
//! that dominate the bench (4 or 8 of them) hit independent cache lines.
//! The shard dimension is `[u16; MAX_SHARDS]`; the op dimension is
//! `[AtomicU64; Op::COUNT]`. Layout chosen to keep each shard's counters
//! adjacent (one cache line per shard ≤ 8 ops × 8 bytes = 64 B).

use core::sync::atomic::{AtomicU64, Ordering};

use crate::{Counter, Gauge, Op};

/// Cap on tracked shards. Going above this just falls back to shard 0.
/// 32 covers M1/M3/M4 + small servers without inflating the static.
pub const MAX_SHARDS: usize = 32;

/// Per-op counters, partitioned by shard to avoid false sharing.
///
/// `OP_COUNTERS[shard][op]` is one `AtomicU64`. With `MAX_SHARDS = 32` and
/// `Op::COUNT = 7`, that's 32 × 7 × 8 = 1792 bytes ≈ 28 cache lines.
static OP_COUNTERS: [[AtomicU64; Op::COUNT]; MAX_SHARDS] =
    [const { [const { AtomicU64::new(0) }; Op::COUNT] }; MAX_SHARDS];

/// Global counters not tied to a shard.
static COUNTERS: [AtomicU64; Counter::COUNT] = [const { AtomicU64::new(0) }; Counter::COUNT];

/// Gauges (overwriteable current values).
static GAUGES: [AtomicU64; Gauge::COUNT] = [const { AtomicU64::new(0) }; Gauge::COUNT];

/// Tick a per-op counter on the requesting shard.
///
/// Cost: one `AtomicU64::fetch_add(1, Relaxed)`. On Apple Silicon that's
/// ~1–2 ns on uncontended lines, with no fence (`Relaxed` is enough — we
/// never read a counter to make a decision on the hot path).
#[inline(always)]
pub fn tick_op(op: Op, shard_id: u16) {
    let s = (shard_id as usize) & (MAX_SHARDS - 1);
    OP_COUNTERS[s][op as usize].fetch_add(1, Ordering::Relaxed);
}

/// Add `delta` to a global counter.
#[inline(always)]
pub fn tick_counter(c: Counter, delta: u64) {
    COUNTERS[c as usize].fetch_add(delta, Ordering::Relaxed);
}

/// Overwrite a gauge's current value.
#[inline(always)]
pub fn set_gauge(g: Gauge, value: u64) {
    GAUGES[g as usize].store(value, Ordering::Relaxed);
}

/// Add `delta` to a gauge (signed, via wrapping `fetch_add`).
///
/// Useful for "in flight" counts where the natural API is
/// `incr` at the start of an operation and `decr` at the end. Returns
/// the previous value to let the caller validate symmetry in debug
/// builds if they want to.
#[inline(always)]
#[allow(clippy::cast_sign_loss)]
pub fn add_gauge(g: Gauge, delta: i64) -> u64 {
    GAUGES[g as usize].fetch_add(delta as u64, Ordering::Relaxed)
}

/// Convenience: `add_gauge(g, +1)`.
#[inline(always)]
pub fn incr_gauge(g: Gauge) {
    let _ = add_gauge(g, 1);
}

/// Convenience: `add_gauge(g, -1)`. Safe to call when the gauge is
/// already zero (the wrapping semantics still produce a defined value;
/// the next `incr_gauge` returns it to the expected level).
#[inline(always)]
pub fn decr_gauge(g: Gauge) {
    let _ = add_gauge(g, -1);
}

// ───────────────────────────────────────────────────────────────────────────
// Read helpers (used by the dumpers, never the hot path).
// ───────────────────────────────────────────────────────────────────────────

/// Sum the per-op counter across all shards for one op.
pub fn op_total(op: Op) -> u64 {
    OP_COUNTERS
        .iter()
        .map(|row| row[op as usize].load(Ordering::Relaxed))
        .sum()
}

/// Snapshot a single shard's counter.
pub fn op_shard(op: Op, shard_id: u16) -> u64 {
    let s = (shard_id as usize) & (MAX_SHARDS - 1);
    OP_COUNTERS[s][op as usize].load(Ordering::Relaxed)
}

/// Snapshot a global counter.
pub fn counter(c: Counter) -> u64 {
    COUNTERS[c as usize].load(Ordering::Relaxed)
}

/// Snapshot a gauge.
pub fn gauge(g: Gauge) -> u64 {
    GAUGES[g as usize].load(Ordering::Relaxed)
}
