//! Recovery (cold reopen) time vs key count — the regression guard for the
//! batch-aware recovery scan. Every reopened record now passes through the
//! `BatchGate`; this measures whether that one extra branch per record costs
//! anything on the normal (no-batch) path that every store pays on open.
//!
//! Populate N plain keys, flush, drop, then time `VLog::open` (full scan, no
//! snapshot). A/B this binary across commits to compare with/without the gate.
//!
//! `harness = false`, custom main, min-of-rounds ms.

use std::time::Instant;

use skeg_core::VLog;
use skeg_core::group_commit::Durability;
use tempfile::TempDir;

fn main() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let rounds = 3;

    println!("VLog recovery (full-scan reopen), min of {rounds} rounds");
    println!("  {:>9}  {:>12}  {:>14}", "keys", "reopen ms", "us/key");
    for &n in &[100_000u32, 500_000] {
        let mut best = f64::MAX;
        for _ in 0..rounds {
            let dir = TempDir::new().unwrap();
            rt.block_on(async {
                let v = VLog::open(dir.path()).await.unwrap();
                for i in 0..n {
                    v.set(
                        format!("k:{i:08}").as_bytes(),
                        b"value-bytes",
                        Durability::Relaxed,
                    )
                    .await
                    .unwrap();
                }
                v.flush().await.unwrap();
            });
            let ms = rt.block_on(async {
                let t0 = Instant::now();
                let v = VLog::open(dir.path()).await.unwrap();
                let ms = t0.elapsed().as_secs_f64() * 1e3;
                assert_eq!(v.len(), n as usize, "recovered every key");
                ms
            });
            best = best.min(ms);
        }
        let us = best * 1e3 / f64::from(n);
        println!("  {n:>9}  {best:>12.1}  {us:>14.3}");
    }
}
