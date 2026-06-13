//! P0a gate G-P0a-2 / G-P0a-3: per-tenant accounting must not regress the
//! single-tenant cache path. SOL reference = the unmodified cache (run this same
//! bench against the parent commit's `cache.rs` to get the baseline).
//!
//! `harness = false`, custom main: we report median ns/op (wall-clock medians are
//! the figure of merit, matching the `sharded_commit` bench convention), not a
//! criterion mean+CI.
//!
//! Mono-tenant only: drives `insert`/`get`, which exist identically in baseline
//! and S1, so the delta between the two builds is exactly the accounting overhead.

use std::time::Instant;

use skeg_core::S3Fifo;

const ENTRY: usize = 8 + 8 + 64; // key + u64 value + ENTRY_OVERHEAD

/// Min and median of a slice of nanos-per-op samples. Min is the closest proxy
/// to speed-of-light (filters scheduler/thermal noise); median is the typical.
fn min_median(mut xs: Vec<f64>) -> (f64, f64) {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    let med = if n % 2 == 1 {
        xs[n / 2]
    } else {
        (xs[n / 2 - 1] + xs[n / 2]) / 2.0
    };
    (xs[0], med)
}

/// Insert `n` distinct keys into a cache holding `capacity_entries` entries.
/// `capacity_entries < n` forces eviction; `>= n` is the no-evict path.
fn bench_insert(n: u64, capacity_entries: usize, rounds: usize) -> (f64, f64) {
    let budget = capacity_entries * ENTRY;
    let keys: Vec<[u8; 8]> = (0..n).map(u64::to_le_bytes).collect();
    let mut samples = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let mut c: S3Fifo<u64> = S3Fifo::new(budget);
        let t = Instant::now();
        for (i, k) in keys.iter().enumerate() {
            c.insert(k, i as u64, 8);
        }
        let elapsed = t.elapsed().as_nanos() as f64;
        samples.push(elapsed / n as f64);
        std::hint::black_box(&c);
    }
    min_median(samples)
}

/// Lookups against a populated, fully-resident cache (get is unchanged by S1;
/// this is a control that should show ~zero delta).
fn bench_get(n: u64, rounds: usize) -> (f64, f64) {
    let budget = (n as usize + 16) * ENTRY;
    let keys: Vec<[u8; 8]> = (0..n).map(u64::to_le_bytes).collect();
    let mut c: S3Fifo<u64> = S3Fifo::new(budget);
    for (i, k) in keys.iter().enumerate() {
        c.insert(k, i as u64, 8);
    }
    let mut samples = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let t = Instant::now();
        for k in &keys {
            std::hint::black_box(c.get(k));
        }
        let elapsed = t.elapsed().as_nanos() as f64;
        samples.push(elapsed / n as f64);
    }
    min_median(samples)
}

fn main() {
    let rounds = 15;
    let n = 1_000_000;

    // No-eviction insert: capacity holds the whole keyset.
    let (ins_noevict_min, ins_noevict_med) = bench_insert(n, n as usize + 16, rounds);
    // Heavy-eviction insert: capacity = 10% of the keyset.
    let (ins_evict_min, ins_evict_med) = bench_insert(n, n as usize / 10, rounds);
    let (get_min, get_med) = bench_get(n, rounds);

    println!("cache_tenant_accounting (mono-tenant, ns/op, n={n}, rounds={rounds})");
    println!("                    min       median");
    println!("  insert_no_evict : {ins_noevict_min:8.2}  {ins_noevict_med:8.2}");
    println!("  insert_evict    : {ins_evict_min:8.2}  {ins_evict_med:8.2}");
    println!("  get             : {get_min:8.2}  {get_med:8.2}");
}
