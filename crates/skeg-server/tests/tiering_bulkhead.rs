//! Bench + regression guard for the tiering cold-start bulkhead (`get_or_reopen`).
//!
//! The claim: when one vindex is evicted and a query lazily reopens it, that
//! reopen runs OFF the shard thread (`spawn_blocking`), so other vindexes on the
//! same shard keep being served. Before that fix the reopen ran synchronously on
//! the shard's single-threaded runtime and starved every queued request until it
//! finished.
//!
//! This test makes the difference observable: index A is large (slow to reopen),
//! index B is small (fast to search). We evict A, fire one search on A (triggers
//! the reopen) and hammer B concurrently, then measure how B fares during A's
//! reopen window.
//!
//!   - bulkhead working (async reopen): B is served many times while A reopens,
//!     each B search far faster than the reopen itself.
//!   - bulkhead broken (sync reopen): B's first search blocks behind A's whole
//!     reopen, so we see ~1 B search with latency ~= the reopen time.
//!
//! Heavy-ish (builds A), so `#[ignore]`. Run with:
//!
//! ```sh
//! cargo test -p skeg-server --test tiering_bulkhead --release -- --ignored --nocapture
//! ```
//!
//! Tune A's size with `SKEG_BULKHEAD_NA` (default 40000); bigger = slower reopen
//! = larger margin.

use std::time::{Duration, Instant};

use skeg_server::shard::ShardSet;

const DIM: usize = 64;
const N_B: u64 = 1_000;

/// Deterministic pseudo-random unit-ish vector from a seed (no rand dep).
fn vec_for(seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..DIM)
        .map(|_| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            // map to [-1, 1)
            ((s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx]
}

async fn load(shards: &ShardSet, name: &str, n: u64) {
    // 1 = int8 tier, 1 = disk backend (evictable + reopenable).
    shards.vindex_create(name, DIM as u32, 1, 1).await.unwrap();
    let mut id = 0u64;
    while id < n {
        let end = (id + 4096).min(n);
        // `None` infers to `Option<Bytes>` from vmset's signature.
        shards
            .vmset(name, (id..end).map(|i| (i, vec_for(i), None)).collect(), 0, None)
            .await
            .unwrap();
        id = end;
    }
    // Fold the delta into the on-disk graph so a reopen reads a real index.
    shards.vindex_consolidate(name).await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "heavy: builds a large index; run with --ignored --nocapture"]
async fn reopen_does_not_stall_other_indexes() {
    let n_a: u64 = std::env::var("SKEG_BULKHEAD_NA")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(40_000);

    let dir = tempfile::TempDir::new().unwrap();
    let shards = ShardSet::open(dir.path(), 1).unwrap();

    load(&shards, "A", n_a).await;
    load(&shards, "B", N_B).await;

    let qa = vec_for(7);
    let qb = vec_for(11);

    // Warm both, then evict only A.
    shards.vsearch("A", qa.clone(), 10, 0, 0, false, None).await.unwrap();
    shards.vsearch("B", qb.clone(), 10, 0, 0, false, None).await.unwrap();
    let evicted = shards.control_handle().evict(0, "A").await.unwrap();
    assert!(evicted, "A must have been resident before eviction");

    // Fire the A search (triggers the reopen) concurrently with a tight B loop.
    let a_shards = shards.clone();
    let a_task = tokio::spawn(async move {
        let s = Instant::now();
        a_shards.vsearch("A", qa, 10, 0, 0, false, None).await.unwrap();
        s.elapsed()
    });

    // Hammer B until A's reopen completes; record each B latency.
    let mut b_lat = Vec::new();
    loop {
        let s = Instant::now();
        shards.vsearch("B", qb.clone(), 10, 0, 0, false, None).await.unwrap();
        b_lat.push(s.elapsed());
        if a_task.is_finished() {
            break;
        }
    }
    let t_reopen = a_task.await.unwrap();

    b_lat.sort_unstable();
    let b_p50 = percentile(&b_lat, 0.50);
    let b_p99 = percentile(&b_lat, 0.99);
    let b_max = *b_lat.last().unwrap();

    println!("\n== tiering bulkhead ==");
    println!("A reopen (+search): {t_reopen:?}  (N_A={n_a})");
    println!(
        "B served during reopen: {} ops  p50={b_p50:?} p99={b_p99:?} max={b_max:?}",
        b_lat.len()
    );

    // Solid checks - both fail under a synchronous reopen:
    //
    // 1. Reopen must be the slow thing (otherwise the test proves nothing). If
    //    this trips, raise SKEG_BULKHEAD_NA.
    assert!(
        t_reopen > b_p50 * 4,
        "reopen ({t_reopen:?}) not clearly slower than a B search (p50 {b_p50:?}); raise SKEG_BULKHEAD_NA"
    );
    // 2. B made real progress during the reopen window. Sync reopen serves ~1.
    assert!(
        b_lat.len() >= 5,
        "only {} B ops served during A's reopen - shard was stalled",
        b_lat.len()
    );
    // 3. No B op waited for the whole reopen. Sync reopen makes B's first op
    //    latency ~= t_reopen; the bulkhead keeps every B op well under it.
    assert!(
        b_p99 * 2 < t_reopen,
        "B p99 ({b_p99:?}) not well under reopen ({t_reopen:?}) - B serialized behind the reopen"
    );
}
