//! What do the admin paths on a vindex cost, and what do those costs scale
//! with? DROP, and the LIST that an operator's TUI polls.
//!
//! The blob reclamation inside a DROP used to enumerate the index's own
//! `live_ids`: O(vectors), no walk. It now selects blobs by walking the shard's
//! key index and matching the key shape exactly - which is what lets a DROP
//! reclaim the blobs of an index that will not OPEN, and what stops a name from
//! eating the blobs of a name it prefixes. The walk visits every live key on
//! the shard, so the question this bench exists to answer is whether that made
//! DROP O(whole keyspace).
//!
//! Two sweeps pin it down:
//!   A. index size fixed, unrelated keyspace grows -> isolates the walk term.
//!   B. keyspace fixed, index size grows           -> isolates the delete term.
//!
//! Measured 2026-09-01, aarch64, 4 shards, dim 8. TWO runs are recorded because
//! the first alone would have overstated the effect - run-to-run spread here is
//! 10-20%, and a single run's slope is not a measurement.
//!
//!   A  bystanders      0   25k    100k    250k     (index fixed at 2k vectors)
//!      drop ms      37.2  36.6    52.2    54.3     run 1
//!      drop ms      39.0  46.7    44.3    48.1     run 2
//!
//!   B  vectors       500    2k      8k     20k     (keyspace fixed at 50k)
//!      drop ms      68.7  40.1    65.0    70.4     run 1
//!      drop ms      53.0  59.3    57.3    63.1     run 2
//!
//!   C  indexes        1    10      50     200      (all evicted)
//!      us/LIST     106.6 120.4   155.4   283.6
//!
//! A climbs in both runs, by +17 ms and +9 ms across 250k keys: the walk is
//! real, somewhere around 36-68 ns per key, and the honest statement is the
//! range and not either endpoint. Extrapolated, ~+10-70 ms at 1M keys and
//! ~+0.4-0.7 s at 10M. B is flat, so below ~20k vectors a DROP is dominated by
//! fixed per-shard costs (registry rewrite, directory removal) rather than by
//! the deletes; its first point is cold-start noise, not a real inversion.
//!
//! C answers the cost of the registry read that LIST gained so that a
//! committed-but-evicted index still appears: 27-71 us per shard including the
//! round trip, growing about 0.9 us per index. A TUI polling once a second does
//! not notice; at 10 Hz over 200 indexes it is 2.8 ms/s, 0.3% of one shard
//! thread.
//!
//! Verdict: DROP is now O(whole keyspace), the same property `count_tenant_keys`
//! and the erase sweep already declare, on a path an operator invokes rarely.
//! Recorded so the ceiling is known rather than discovered. If a store ever
//! makes that walk hurt, the answer is an index on blob keys, not a return to
//! `live_ids` - that path cannot serve an index which will not open.
//!
//! `harness = false`, custom main, wall-clock ms. The store is rebuilt for every
//! point (a drop is destructive), so populate time dominates the run; only the
//! drop itself is timed.

use std::time::Instant;

use bytes::Bytes;
use skeg_core::group_commit::Durability;
use skeg_server::shard::ShardSet;
use tempfile::TempDir;

const SHARDS: usize = 4;
const DIM: u32 = 8;

fn bystander_key(i: u32) -> Vec<u8> {
    format!("bystander:{i:010}").into_bytes()
}

/// Build a store holding one disk-backed vindex of `vectors` rows (each with a
/// payload blob) plus `bystanders` unrelated KV keys, then time the DROP.
async fn drop_once(dir: &TempDir, vectors: u64, bystanders: u32) -> f64 {
    let shards = ShardSet::open(dir.path(), SHARDS).unwrap();
    shards.vindex_create("v", DIM, 1, 1).await.unwrap();
    for id in 0..vectors {
        let vec = vec![id as f32; DIM as usize];
        shards
            .vset(
                "v",
                id,
                vec,
                0,
                None,
                Some(Bytes::from_static(b"payload-blob")),
            )
            .await
            .unwrap();
    }
    for i in 0..bystanders {
        shards
            .set(&bystander_key(i), b"v", Durability::Relaxed)
            .await
            .unwrap();
    }

    let t0 = Instant::now();
    shards.vindex_drop("v", 0).await.unwrap();
    t0.elapsed().as_secs_f64() * 1e3
}

/// Build a store with `indexes` disk-backed vindexes, all EVICTED, then time
/// repeated LISTs. Evicted is the case that matters: LIST now reads each
/// shard's registry so a committed-but-not-resident index still appears, and
/// that read is blocking I/O on the shard thread which also serves searches.
/// Returns microseconds per LIST call.
async fn list_cost(dir: &TempDir, indexes: usize) -> f64 {
    let shards = ShardSet::open(dir.path(), SHARDS).unwrap();
    let ctl = shards.control_handle();
    for i in 0..indexes {
        let name = format!("idx{i:04}");
        shards.vindex_create(&name, DIM, 1, 1).await.unwrap();
        shards
            .vset(&name, 0, vec![0.0; DIM as usize], 0, None, None)
            .await
            .unwrap();
        shards.vindex_consolidate(&name).await.unwrap();
        ctl.evict(0, &name).await.unwrap();
    }
    // Warm the page cache: this measures the steady state a polling TUI sees,
    // not the first read after a build.
    for _ in 0..5 {
        shards.vindex_list().await.unwrap();
    }
    const ROUNDS: usize = 50;
    let t0 = Instant::now();
    for _ in 0..ROUNDS {
        let rows = shards.vindex_list().await.unwrap();
        assert_eq!(rows.len(), indexes, "every committed index must be listed");
    }
    t0.elapsed().as_secs_f64() * 1e6 / ROUNDS as f64
}

fn main() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    println!("vindex_drop: what does the cost scale with? ({SHARDS} shards)\n");

    println!("A. index fixed at 2k vectors, unrelated keyspace grows  (the walk)");
    println!(
        "   {:>10}  {:>8}  {:>9}  {:>12}",
        "bystanders", "vectors", "drop ms", "us/vector"
    );
    for &bystanders in &[0u32, 25_000, 100_000, 250_000] {
        let vectors = 2_000u64;
        let dir = TempDir::new().unwrap();
        let ms = rt.block_on(drop_once(&dir, vectors, bystanders));
        let us_each = ms * 1e3 / vectors as f64;
        println!("   {bystanders:>10}  {vectors:>8}  {ms:>9.1}  {us_each:>12.2}");
    }

    println!("\nB. keyspace fixed at 50k bystanders, index grows  (the deletes)");
    println!(
        "   {:>10}  {:>8}  {:>9}  {:>12}",
        "bystanders", "vectors", "drop ms", "us/vector"
    );
    for &vectors in &[500u64, 2_000, 8_000, 20_000] {
        let dir = TempDir::new().unwrap();
        let ms = rt.block_on(drop_once(&dir, vectors, 50_000));
        let us_each = ms * 1e3 / vectors as f64;
        println!(
            "   {:>10}  {vectors:>8}  {ms:>9.1}  {us_each:>12.2}",
            50_000
        );
    }

    println!("\nC. LIST over evicted indexes  (the registry read added to it)");
    println!(
        "   {:>10}  {:>12}  {:>14}",
        "indexes", "us/LIST", "us/shard-read"
    );
    for &indexes in &[1usize, 10, 50, 200] {
        let dir = TempDir::new().unwrap();
        let us = rt.block_on(list_cost(&dir, indexes));
        println!("   {indexes:>10}  {us:>12.1}  {:>14.1}", us / SHARDS as f64);
    }
}
