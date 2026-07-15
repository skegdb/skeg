//! What does erasing a tenant cost, and what does that cost scale with?
//!
//! `erase_tenant` sweeps each shard's index with a prefix-filtered
//! `for_each_key` walk. The walk visits every live key on the shard, not just
//! the victim's, so the suspicion is that erasure is O(whole keyspace) and only
//! the delete half is O(victim). If that holds, erasing a small tenant out of a
//! large store costs the same as erasing a large one, and the price is paid in
//! the walk.
//!
//! Two sweeps pin that down:
//!   A. victim size fixed, keyspace grows  -> isolates the walk term.
//!   B. keyspace fixed, victim size grows  -> isolates the delete term.
//!
//! `harness = false`, custom main, wall-clock ms. The store is rebuilt for every
//! point (erasure is destructive), so populate time dominates the run; only the
//! erase itself is timed.

use std::time::Instant;

use skeg_core::group_commit::Durability;
use skeg_server::shard::ShardSet;
use tempfile::TempDir;

const SHARDS: usize = 4;
const VICTIM: u128 = 1;
const BYSTANDER: u128 = 2;

/// Tenant-scoped key: the tenant's 16 bytes little-endian, then the raw key.
fn scoped(tenant: u128, i: u32) -> Vec<u8> {
    let mut k = tenant.to_le_bytes().to_vec();
    k.extend_from_slice(format!("k:{i:010}").as_bytes());
    k
}

/// Build a store holding `victim_keys` keys for the victim tenant and
/// `total_keys - victim_keys` for a bystander, then time the victim's erasure.
/// Returns `(erase_ms, keys_deleted)`.
async fn erase_once(dir: &TempDir, total_keys: u32, victim_keys: u32) -> (f64, u64) {
    let shards = ShardSet::open(dir.path(), SHARDS).unwrap();
    for i in 0..victim_keys {
        shards
            .set(&scoped(VICTIM, i), b"v", Durability::Relaxed)
            .await
            .unwrap();
    }
    for i in 0..(total_keys - victim_keys) {
        shards
            .set(&scoped(BYSTANDER, i), b"v", Durability::Relaxed)
            .await
            .unwrap();
    }

    let t0 = Instant::now();
    let (_, deleted) = shards
        .erase_tenant(VICTIM, Durability::Relaxed)
        .await
        .unwrap();
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    assert_eq!(deleted, u64::from(victim_keys), "erased the victim's keys");
    (ms, deleted)
}

/// Populate `total_keys` for the victim under two subject prefixes (`keep/` and
/// `drop/`, half each), erase the `drop/` subject, then reclaim. Returns
/// `(erase_ms, keys_erased, reclaim_ms, bytes_freed)`. Exercises the selective
/// erase (`erase_prefix`) and the physical reclaim together.
async fn erase_prefix_then_reclaim(dir: &TempDir, total_keys: u32) -> (f64, u64, f64, u64) {
    let shards = ShardSet::open(dir.path(), SHARDS).unwrap();
    let key = |subject: &str, i: u32| {
        let mut k = VICTIM.to_le_bytes().to_vec();
        k.extend_from_slice(format!("{subject}/{i}").as_bytes());
        k
    };
    for i in 0..total_keys {
        let subject = if i % 2 == 0 { "keep" } else { "drop" };
        shards
            .set(&key(subject, i), b"v", Durability::Relaxed)
            .await
            .unwrap();
    }

    let t0 = Instant::now();
    let erased = shards
        .erase_prefix(VICTIM, b"drop/", Durability::Relaxed)
        .await
        .unwrap();
    let erase_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t1 = Instant::now();
    let freed = shards.reclaim().await.unwrap();
    let reclaim_ms = t1.elapsed().as_secs_f64() * 1e3;
    (erase_ms, erased, reclaim_ms, freed)
}

fn main() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    println!("erase_tenant: what does the cost scale with? ({SHARDS} shards)\n");

    println!("A. victim fixed at 10k keys, keyspace grows  (isolates the index walk)");
    println!(
        "   {:>10}  {:>10}  {:>9}  {:>12}",
        "keyspace", "victim", "erase ms", "us/victim"
    );
    for &total in &[50_000u32, 100_000, 250_000, 500_000] {
        let victim = 10_000u32;
        let dir = TempDir::new().unwrap();
        let (ms, n) = rt.block_on(erase_once(&dir, total, victim));
        let us_each = ms * 1e3 / f64::from(victim);
        println!("   {total:>10}  {victim:>10}  {ms:>9.1}  {us_each:>12.2}");
        std::hint::black_box(n);
    }

    println!("\nB. keyspace fixed at 250k keys, victim grows  (isolates the deletes)");
    println!(
        "   {:>10}  {:>10}  {:>9}  {:>12}",
        "keyspace", "victim", "erase ms", "us/victim"
    );
    for &victim in &[1_000u32, 10_000, 50_000, 200_000] {
        let total = 250_000u32;
        let dir = TempDir::new().unwrap();
        let (ms, n) = rt.block_on(erase_once(&dir, total, victim));
        let us_each = ms * 1e3 / f64::from(victim);
        println!("   {total:>10}  {victim:>10}  {ms:>9.1}  {us_each:>12.2}");
        std::hint::black_box(n);
    }

    println!("\nC. selective erase (drop/ subject = half) + physical reclaim");
    println!(
        "   {:>10}  {:>9}  {:>9}  {:>11}  {:>10}",
        "keyspace", "erased", "erase ms", "reclaim ms", "MiB freed"
    );
    for &total in &[50_000u32, 100_000, 250_000, 500_000] {
        let dir = TempDir::new().unwrap();
        let (ems, erased, rms, freed) = rt.block_on(erase_prefix_then_reclaim(&dir, total));
        let mib = freed as f64 / (1024.0 * 1024.0);
        println!("   {total:>10}  {erased:>9}  {ems:>9.1}  {rms:>11.1}  {mib:>10.1}");
    }
}
