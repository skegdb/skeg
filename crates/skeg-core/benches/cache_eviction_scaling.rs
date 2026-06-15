//! Does fair eviction's per-eviction over-share scan (O(resident tenants)) bottleneck
//! as the number of resident tenants grows? Fills the cache with N tenants (one
//! entry each), then times inserts that each force one Main eviction. If the
//! `has_over_share` scan dominates, ns/op grows ~linearly with N.
//!
//! `harness = false`, custom main, min-of-rounds ns/op.

use std::time::Instant;

use skeg_core::S3Fifo;

const ENTRY: usize = 8 + 8 + 64; // 8-byte key + u64 value + ENTRY_OVERHEAD

fn bench(tenants: u64, evicting_ops: u64, rounds: usize) -> f64 {
    let budget = tenants as usize * ENTRY; // ~one resident entry per tenant
    let mut best = f64::MAX;
    for _ in 0..rounds {
        let mut c: S3Fifo<u64> = S3Fifo::new(budget);
        // Fill: one key per tenant, all resident -> per_tenant has `tenants` keys.
        for t in 0..tenants {
            c.insert_for(&t.to_le_bytes(), t, 8, u128::from(t) + 1);
        }
        // Time inserts that each evict one entry (cache is full). Each eviction
        // runs has_over_share over the ~`tenants`-sized per_tenant map.
        let t0 = Instant::now();
        for i in 0..evicting_ops {
            let key = (tenants + i).to_le_bytes();
            c.insert_for(&key, i, 8, u128::from(i % tenants) + 1);
        }
        let ns = t0.elapsed().as_nanos() as f64 / evicting_ops as f64;
        best = best.min(ns);
        std::hint::black_box(&c);
    }
    best
}

fn main() {
    let ops = 200_000;
    let rounds = 5;
    println!("eviction scaling (insert-with-evict, min ns/op, ops={ops}, rounds={rounds})");
    for &t in &[1u64, 10, 100, 1_000, 10_000] {
        println!(
            "  resident tenants={t:6}: {:8.1} ns/op",
            bench(t, ops, rounds)
        );
    }
}
