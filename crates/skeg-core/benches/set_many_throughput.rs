//! Does `set_many` earn its keep over writing keys one at a time?
//!
//! Writing a multi-key MSET as N separate `set` calls pays one group-commit
//! flush PER KEY. `set_many` puts the whole batch in one append -> one flush,
//! and it is atomic on top. Two honest baselines:
//!   - sequential: N awaited `set`s (one flush each) — what MSET used to do.
//!   - concurrent: N `set`s in flight via buffer_unordered (the committer
//!     batches them, so flushes are already shared) — the fair rival.
//! `set_many` should match or beat concurrent while ALSO being atomic, and
//! crush sequential.
//!
//! `harness = false`, custom main, wall-clock. Kernel durability so the flush
//! cost (the thing being amortised) is real, not elided.

use std::time::Instant;

use futures_util::stream::{self, StreamExt};
use skeg_core::VLog;
use skeg_core::group_commit::Durability;
use tempfile::TempDir;

const DUR: Durability = Durability::Kernel;
const ROUNDS: usize = 3;

fn pairs(n: usize, seed: u32) -> Vec<(Vec<u8>, Vec<u8>)> {
    (0..n)
        .map(|i| {
            (
                format!("k:{seed}:{i:08}").into_bytes(),
                format!("v:{i}").into_bytes(),
            )
        })
        .collect()
}

async fn seq(v: &VLog, ps: &[(Vec<u8>, Vec<u8>)]) {
    for (k, val) in ps {
        v.set(k, val, DUR).await.unwrap();
    }
}

async fn concurrent(v: &VLog, ps: &[(Vec<u8>, Vec<u8>)]) {
    stream::iter(ps)
        .map(|(k, val)| v.set(k, val, DUR))
        .buffer_unordered(256)
        .for_each(|r| async { r.unwrap() })
        .await;
}

async fn batch(v: &VLog, ps: &[(Vec<u8>, Vec<u8>)]) {
    let refs: Vec<(&[u8], &[u8])> = ps
        .iter()
        .map(|(k, val)| (k.as_slice(), val.as_slice()))
        .collect();
    v.set_many(&refs, DUR).await.unwrap();
}

fn main() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    println!("MSET of N keys: sequential vs concurrent vs set_many (Kernel durability)");
    println!("  min of {ROUNDS} rounds, one fresh store per round\n");
    println!(
        "  {:>7}  {:>12}  {:>12}  {:>12}   {:>10}",
        "N", "seq ms", "concurrent ms", "set_many ms", "seq/batch"
    );

    for &n in &[10usize, 100, 1_000, 10_000] {
        let mut best = [f64::MAX; 3];
        for round in 0..ROUNDS {
            for (mode, f) in [0, 1, 2].into_iter().zip([0u8, 1, 2]) {
                let dir = TempDir::new().unwrap();
                let ps = pairs(n, round as u32 * 3 + u32::from(f));
                let ms = rt.block_on(async {
                    let v = VLog::open(dir.path()).await.unwrap();
                    let t0 = Instant::now();
                    match f {
                        0 => seq(&v, &ps).await,
                        1 => concurrent(&v, &ps).await,
                        _ => batch(&v, &ps).await,
                    }
                    v.flush().await.unwrap();
                    t0.elapsed().as_secs_f64() * 1e3
                });
                best[mode] = best[mode].min(ms);
            }
        }
        let speedup = best[0] / best[2];
        println!(
            "  {n:>7}  {:>12.2}  {:>12.2}  {:>12.2}   {speedup:>9.1}x",
            best[0], best[1], best[2]
        );
    }
}
