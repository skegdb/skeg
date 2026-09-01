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
//! If A is flat, the walk is not the price and the change is free at these
//! sizes. If A climbs with the keyspace, a DROP in a large store pays for keys
//! that have nothing to do with it, and the reclamation needs an index.
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
