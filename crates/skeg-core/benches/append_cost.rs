//! What does the naive read-modify-write `append` cost as a value grows?
//!
//! Each `append` reads the whole current value and rewrites it, so appending
//! the k-th delta copies ~k*delta bytes: building a list of N deltas is O(N^2).
//! This bench makes that curve visible, so the boundary between "fine" (bounded
//! values, e.g. a fan-out-capped adjacency list) and "needs the chunked
//! variant" (unbounded values, e.g. hot BM25 postings) is a measurement, not a
//! guess.
//!
//! For contrast it also builds the same final bytes with ONE `set`, the floor a
//! chunked/delta append could approach.
//!
//! `harness = false`, custom main, wall-clock. Relaxed durability so the copy
//! cost, not the flush, is what shows.

use std::time::Instant;

use skeg_core::VLog;
use skeg_core::group_commit::Durability;
use tempfile::TempDir;

const DUR: Durability = Durability::Relaxed;
const DELTA: &[u8] = b"node:00000000,"; // ~14 bytes, one adjacency entry

fn main() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    println!(
        "append: naive RMW cost of building an N-entry list (delta = {} B)",
        DELTA.len()
    );
    println!(
        "  {:>7}  {:>12}  {:>12}  {:>14}  {:>12}",
        "N", "append ms", "us/append", "final KiB", "one set ms"
    );
    for &n in &[100usize, 1_000, 5_000, 20_000] {
        let dir = TempDir::new().unwrap();
        let (append_ms, set_ms) = rt.block_on(async {
            let v = VLog::open(dir.path()).await.unwrap();
            let t0 = Instant::now();
            for _ in 0..n {
                v.append(b"adj", DELTA, DUR).await.unwrap();
            }
            let append_ms = t0.elapsed().as_secs_f64() * 1e3;

            // The same final value written once, for the floor.
            let whole: Vec<u8> = DELTA.repeat(n);
            let t1 = Instant::now();
            v.set(b"adj2", &whole, DUR).await.unwrap();
            let set_ms = t1.elapsed().as_secs_f64() * 1e3;
            (append_ms, set_ms)
        });
        let per = append_ms * 1e3 / n as f64;
        let kib = (n * DELTA.len()) as f64 / 1024.0;
        println!("  {n:>7}  {append_ms:>12.1}  {per:>12.2}  {kib:>14.1}  {set_ms:>12.3}");
    }
    // At these (KiB-range) sizes us/append is near-flat: it is dominated by the
    // per-append group-commit round-trip (~1.6ms), and the O(value) copy is a
    // rounding error next to it. The O(N^2) copy only overtakes the flush once
    // the value reaches MB scale — the regime where the chunked/delta variant
    // would pay off. For bounded values (a fan-out-capped adjacency list) naive
    // RMW is flush-bound like any single write.
    println!(
        "\n  near-flat us/append = flush-bound; O(N^2) copy only overtakes at MB-scale values."
    );
}
