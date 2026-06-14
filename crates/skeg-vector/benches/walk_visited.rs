//! Primitive gate microbench: VisitedBitset vs AHashSet on the
//! Vamana greedy walk access pattern.
//!
//! Simulated per-query pattern: ~100 expansions x ~64 neighbours = ~6400
//! `test_and_set` operations plus a few `is_set`. The VecId sequence is
//! pseudo-random but seeded for reproducibility, so both the bitset and the
//! hashset bench see the same access trace.
//!
//! Pre-registered primitive gate: bitset >= 3x hashset throughput.
//!
//! Measured 2026-05-21: bitset 3.90 us / 6400 ops vs hashset 24.19 us /
//! 6400 ops = **6.20x** speedup; integration gate (skeg-pq128 100K,
//! dual-distribution mxbai + MiniLM) passed at +6-8% QPS, -7-9% latency,
//! recall identical.

use ahash::AHashSet;
use criterion::{Criterion, criterion_group, criterion_main};
use rand::{Rng, SeedableRng, rngs::StdRng};
use skeg_vector::VisitedBitset;
use std::hint::black_box;

const N: u32 = 100_000;
const EXPANSIONS: usize = 100;
const NEIGHBORS_PER_EXPANSION: usize = 64;

/// The id sequence the greedy walk visits: one packet of
/// `NEIGHBORS_PER_EXPANSION` pseudo-random ids per "expansion". Generated
/// once, replayed identically by both bench cases for a fair comparison.
fn build_access_pattern() -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let total = EXPANSIONS * NEIGHBORS_PER_EXPANSION;
    let mut ids = Vec::with_capacity(total);
    for _ in 0..total {
        ids.push(rng.random_range(0..N));
    }
    ids
}

fn bench_visited(c: &mut Criterion) {
    let pattern = build_access_pattern();

    let mut g = c.benchmark_group("visited_tracking");
    g.throughput(criterion::Throughput::Elements(pattern.len() as u64));

    // VisitedBitset: allocated once, cleared between queries.
    g.bench_function("bitset_100k_query", |b| {
        let mut bs = VisitedBitset::new(N as usize);
        b.iter(|| {
            bs.clear();
            let mut new_count = 0u32;
            for &id in &pattern {
                if !bs.test_and_set(black_box(id)) {
                    new_count += 1;
                }
            }
            black_box(new_count);
        });
    });

    // AHashSet: `clear` preserves capacity (no re-allocation between queries).
    g.bench_function("hashset_100k_query", |b| {
        let mut hs: AHashSet<u32> = AHashSet::with_capacity(pattern.len());
        b.iter(|| {
            hs.clear();
            let mut new_count = 0u32;
            for &id in &pattern {
                if hs.insert(black_box(id)) {
                    new_count += 1;
                }
            }
            black_box(new_count);
        });
    });

    g.finish();
}

criterion_group!(benches, bench_visited);
criterion_main!(benches);
