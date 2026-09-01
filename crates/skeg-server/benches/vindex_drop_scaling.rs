//! What does dropping a vindex cost, and what does that cost scale with?
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
//! Measured 2026-09-01, aarch64, 4 shards, dim 8, debug-built bench in release:
//!
//!   A  bystanders      0   25k    100k    250k
//!      drop ms      37.2  36.6    52.2    54.3     (index fixed at 2k vectors)
//!
//!   B  vectors       500    2k      8k     20k
//!      drop ms      68.7  40.1    65.0    70.4     (keyspace fixed at 50k)
//!
//! A climbs: the walk is real, about 68 ns per key, so +46% on a small index in
//! a store with 250k unrelated keys - and by extrapolation ~+68 ms at 1M keys,
//! ~+0.7 s at 10M. B is flat, so below ~20k vectors a DROP is dominated by
//! fixed per-shard costs (registry rewrite, directory removal), not by the
//! deletes; its first point is cold-start noise, not a real inversion.
//!
//! Verdict: DROP is now O(whole keyspace), the same property `count_tenant_keys`
//! and the erase sweep already declare. Acceptable on an admin path at these
//! sizes and recorded here so the ceiling is known rather than discovered. If a
//! store ever makes that walk hurt, the answer is an index on blob keys, not a
//! return to `live_ids` - that path cannot serve an index which will not open.
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
}
