//! Fixed-bucket exponential histograms.
//!
//! Bucket edges are powers of two from 1 µs to ~16.8 s, plus a final
//! `+Inf` bucket. 26 buckets per op, [`Op::COUNT`] ops, one
//! `AtomicU64` per bucket. Total static: 26 × 7 × 8 = 1456 bytes ≈ 23
//! cache lines.
//!
//! The range was extended from 22 to 26 buckets in v0.2.0 so downstream
//! consumers with longer-tail operations (`skeg-kv-cache` blob
//! restore, future tenant-quota checks) can still observe outliers up
//! to ~16 s without clipping to `+Inf`.
//!
//! Bucket index for a duration of `us` microseconds:
//!
//! ```text
//! bucket(us) = if us == 0       { 0 }
//!              else if us >= LAST { LAST_IDX }
//!              else              { 64 - leading_zeros(us) }
//! ```
//!
//! Cost on hot path: 1 branch + 1 `leading_zeros` + 1 `fetch_add`
//! ≈ 3–5 ns on Apple Silicon.

use core::sync::atomic::{AtomicU64, Ordering};

use crate::Op;

/// Number of buckets per op. Increment if extending the time range; the
/// underlying array literal below must be extended too.
pub const BUCKETS: usize = 26;

/// Upper bound (exclusive) for each bucket, in microseconds. Last entry
/// is the sentinel "+Inf" bucket. Bounds form a power-of-two ladder so
/// `bucket_idx` can use `64 - leading_zeros(us)` to pick the slot.
pub const BUCKET_BOUNDS_US: [u64; BUCKETS] = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1_024,
    2_048,
    4_096,
    8_192,
    16_384,
    32_768,
    65_536,
    131_072,
    262_144,
    524_288,
    1_048_576,
    2_097_152,
    4_194_304,
    8_388_608,
    16_777_216,
    u64::MAX,
];

/// Per-op buckets. One independent `AtomicU64` per (op, bucket) slot.
static HISTOGRAMS: [[AtomicU64; BUCKETS]; Op::COUNT] =
    [const { [const { AtomicU64::new(0) }; BUCKETS] }; Op::COUNT];

/// Cumulative count of all observations per op (for the `_count` line
/// in the Prometheus output).
static OP_COUNT: [AtomicU64; Op::COUNT] = [const { AtomicU64::new(0) }; Op::COUNT];

/// Cumulative sum of microseconds per op (for the `_sum` line and for
/// computing average latency).
static OP_SUM_US: [AtomicU64; Op::COUNT] = [const { AtomicU64::new(0) }; Op::COUNT];

/// Pick the right bucket index for an observed microsecond value.
///
/// `us = 0` lands in bucket 0 (the `< 1 µs` bucket). `us ≥ 16 777 216`
/// (i.e. >= 16.78 s) lands in the `+Inf` bucket.
#[inline(always)]
pub fn bucket_idx(us: u64) -> usize {
    if us == 0 {
        return 0;
    }
    if us >= BUCKET_BOUNDS_US[BUCKETS - 2] {
        return BUCKETS - 1;
    }
    // 64 − leading_zeros(us) is `ceil(log2(us+1))`. For powers of two we
    // bump the bucket index because the bounds are exclusive upper
    // edges.
    let lz = us.leading_zeros() as usize;
    64 - lz
}

/// Record one observation in microseconds.
///
/// Cost: 1 branch (sentinel check) + 1 `leading_zeros` + 3 `fetch_add`.
/// The three increments are the bucket itself, the cumulative count,
/// and the cumulative sum.
#[inline(always)]
pub fn observe_us(op: Op, us: u64) {
    let b = bucket_idx(us);
    HISTOGRAMS[op as usize][b].fetch_add(1, Ordering::Relaxed);
    OP_COUNT[op as usize].fetch_add(1, Ordering::Relaxed);
    OP_SUM_US[op as usize].fetch_add(us, Ordering::Relaxed);
}

// ───────────────────────────────────────────────────────────────────────────
// Read helpers (dumpers only — never hot path).
// ───────────────────────────────────────────────────────────────────────────

/// Snapshot one bucket of one op.
pub fn bucket(op: Op, idx: usize) -> u64 {
    HISTOGRAMS[op as usize][idx].load(Ordering::Relaxed)
}

/// Snapshot the cumulative count for one op.
pub fn count(op: Op) -> u64 {
    OP_COUNT[op as usize].load(Ordering::Relaxed)
}

/// Snapshot the cumulative microsecond sum for one op.
pub fn sum_us(op: Op) -> u64 {
    OP_SUM_US[op as usize].load(Ordering::Relaxed)
}

/// Compute the `q` quantile (0.0..=1.0) for one op from the bucket
/// counts. Returns `None` if no observations have been recorded.
///
/// Bucket midpoints are used as the value for that bucket. This is a
/// log-linear approximation: accurate to within a power of two, which
/// is what the bucket layout already implies.
pub fn quantile_us(op: Op, q: f64) -> Option<u64> {
    let total = count(op);
    if total == 0 {
        return None;
    }
    let target = ((total as f64) * q).ceil() as u64;
    let mut running: u64 = 0;
    for i in 0..BUCKETS {
        running += bucket(op, i);
        if running >= target {
            // Midpoint of the bucket (or its lower edge for the +Inf one).
            if i == 0 {
                return Some(0);
            }
            if i == BUCKETS - 1 {
                return Some(BUCKET_BOUNDS_US[BUCKETS - 2]);
            }
            let lo = BUCKET_BOUNDS_US[i - 1];
            let hi = BUCKET_BOUNDS_US[i];
            return Some((lo + hi) / 2);
        }
    }
    Some(BUCKET_BOUNDS_US[BUCKETS - 2])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_idx_endpoints() {
        assert_eq!(bucket_idx(0), 0);
        assert_eq!(bucket_idx(1), 1);
        assert_eq!(bucket_idx(2), 2);
        assert_eq!(bucket_idx(3), 2);
        assert_eq!(bucket_idx(4), 3);
        // 10 s sits below 2^24 (16.78 s) so it now falls in a real bucket
        // rather than the +Inf sentinel that v0.1.x clipped it to.
        assert!(bucket_idx(10_000_000) < BUCKETS - 1);
        assert_eq!(bucket_idx(16_777_216), BUCKETS - 1);
        assert_eq!(bucket_idx(u64::MAX), BUCKETS - 1);
    }
}
