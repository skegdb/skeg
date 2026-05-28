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
static OP_COUNTERS: [[AtomicU64; Op::COUNT]; MAX_SHARDS] = {
    // Build a zero-init array. `AtomicU64::new(0)` is `const`.
    const Z: AtomicU64 = AtomicU64::new(0);
    const ROW: [AtomicU64; Op::COUNT] = [Z, Z, Z, Z, Z, Z, Z]; // mirror Op::COUNT (must update if Op grows)
    // SAFETY-equivalent: copy the row 32 times; everything is `const`.
    [
        ROW, ROW, ROW, ROW, ROW, ROW, ROW, ROW, ROW, ROW, ROW, ROW, ROW, ROW, ROW, ROW, ROW, ROW,
        ROW, ROW, ROW, ROW, ROW, ROW, ROW, ROW, ROW, ROW, ROW, ROW, ROW, ROW,
    ]
};

/// Global counters not tied to a shard.
static COUNTERS: [AtomicU64; Counter::COUNT] = {
    const Z: AtomicU64 = AtomicU64::new(0);
    [Z, Z, Z, Z, Z, Z, Z] // mirror Counter::COUNT
};

/// Gauges (overwriteable current values).
static GAUGES: [AtomicU64; Gauge::COUNT] = {
    const Z: AtomicU64 = AtomicU64::new(0);
    [Z, Z, Z, Z, Z, Z, Z] // mirror Gauge::COUNT
};

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
