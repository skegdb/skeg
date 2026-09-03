//! The per-request fan-out of `SKEG.VMSET`, which the ingress budget does not
//! charge and nothing else bounded.
//!
//! Task 2 made a VMSET answer per item, and to do it the coordinator spawns one
//! task per item. That is the right shape - the per-vector blob writes
//! accumulate in the group committer and flush in batches instead of one
//! barrier per vector - but it was unbounded: one maximum batch is 4096
//! concurrent tasks on ONE connection, about 98,000 across 24, and 4.2 million
//! at the default connection limit. Measured at 24 connections it cost
//! +284.9 MiB resident, roughly 12 MiB per connection, in no budget at all.
//!
//! Its own test target on purpose: the peak counter is process-wide, and each
//! integration target is its own process, so nothing else can be running a
//! VMSET while this measures one.

use std::sync::Arc;
use std::time::Instant;

use skeg_server::shard::{ShardSet, VMSET_INFLIGHT, vmset_inflight};

const DIM: u32 = 8;

fn items(n: usize) -> Vec<(u64, Vec<f32>, Option<bytes::Bytes>)> {
    (0..n as u64)
        .map(|id| (id, vec![0.5f32; DIM as usize], None))
        .collect()
}

/// The bound, watched rather than asserted from the constant: a test that
/// checked `VMSET_INFLIGHT == 64` would pass against code that ignored it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "opens in the next commit (shard: bound the VMSET fan-out)"]
async fn a_maximum_vmset_never_runs_more_than_the_inflight_bound() {
    let dir = tempfile::tempdir().expect("tempdir");
    let shards = ShardSet::open(dir.path(), 2).expect("a store");
    shards
        .vindex_create("fanout", DIM, 0, 1)
        .await
        .expect("index");

    vmset_inflight::reset();
    let results = shards.vmset("fanout", items(4096), 0, None).await;
    let peak = vmset_inflight::peak();

    assert_eq!(results.len(), 4096, "one answer per item, in request order");
    assert!(
        results.iter().all(Result::is_ok),
        "the batch itself must succeed, or the fan-out was never reached"
    );
    assert!(
        peak > 1,
        "nothing ran concurrently, so this measured a serial loop: {peak}"
    );
    assert!(
        peak <= VMSET_INFLIGHT,
        "a 4096-item batch ran {peak} item writes at once against a bound of \
         {VMSET_INFLIGHT}"
    );
}

/// Order and the never-abort-siblings property of Task 2, under the bound.
/// A batch bigger than the window has to come back in REQUEST order, not in
/// completion order, and one item's outcome must not decide another's.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_batch_larger_than_the_window_still_answers_in_request_order() {
    let dir = tempfile::tempdir().expect("tempdir");
    let shards = ShardSet::open(dir.path(), 2).expect("a store");
    shards
        .vindex_create("order", DIM, 0, 1)
        .await
        .expect("index");

    // Every third item carries a vector of the wrong dimension, so its own
    // write fails while its neighbours succeed. The pattern is what pins the
    // order: a reply ordered by completion would scramble it.
    let batch: Vec<_> = (0..1000u64)
        .map(|id| {
            let dim = if id % 3 == 0 {
                DIM as usize + 1
            } else {
                DIM as usize
            };
            (id, vec![0.25f32; dim], None)
        })
        .collect();
    let results = shards.vmset("order", batch, 0, None).await;
    assert_eq!(results.len(), 1000);
    for (id, r) in results.iter().enumerate() {
        if id % 3 == 0 {
            assert!(r.is_err(), "item {id} had a bad dimension and succeeded");
        } else {
            assert!(r.is_ok(), "item {id} was failed by a neighbour: {r:?}");
        }
    }
}

/// Resident cost of the fan-out: K connections, one maximum batch each.
///
/// A number, not a gate. The server runs in this process, so `rss_bytes` is
/// the server's own, and the figure the audit measured (+284.9 MiB at K=24)
/// is what this reproduces.
#[test]
#[ignore = "measurement probe: run with --ignored --nocapture"]
fn vmset_fanout_resident_cost() {
    const K: usize = 24;
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    let dir = tempfile::tempdir().expect("tempdir");
    let shards = rt.block_on(async {
        let shards = ShardSet::open(dir.path(), 2).expect("a store");
        shards.vindex_create("rss", DIM, 0, 1).await.expect("index");
        shards
    });

    let before = skeg_platform::rss_bytes();
    let peak = Arc::new(std::sync::atomic::AtomicU64::new(before));
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let sampler = {
        let peak = Arc::clone(&peak);
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            while !stop.load(std::sync::atomic::Ordering::Acquire) {
                peak.fetch_max(
                    skeg_platform::rss_bytes(),
                    std::sync::atomic::Ordering::AcqRel,
                );
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        })
    };

    rt.block_on(async {
        let mut calls = Vec::new();
        for k in 0..K {
            let shards = shards.clone();
            calls.push(tokio::spawn(async move {
                let base = (k * 4096) as u64;
                let batch: Vec<_> = (0..4096u64)
                    .map(|i| (base + i, vec![0.5f32; DIM as usize], None))
                    .collect();
                shards.vmset("rss", batch, 0, None).await
            }));
        }
        for c in calls {
            let r = c.await.expect("call");
            assert_eq!(r.len(), 4096);
        }
    });
    stop.store(true, std::sync::atomic::Ordering::Release);
    sampler.join().expect("sampler");

    let peak = peak.load(std::sync::atomic::Ordering::Acquire);
    println!(
        "vmset fan-out: K={K} batches of 4096, RSS {before} -> peak {peak}, \
         delta {:.1} MiB, {:.1} MiB per connection, bound {VMSET_INFLIGHT}",
        (peak - before) as f64 / (1024.0 * 1024.0),
        (peak - before) as f64 / (1024.0 * 1024.0) / K as f64,
    );
}

/// Throughput of one maximum batch, so the bound can be chosen against a
/// number instead of a hunch.
#[test]
#[ignore = "measurement probe: run with --ignored --nocapture"]
fn vmset_batch_throughput() {
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    let dir = tempfile::tempdir().expect("tempdir");
    rt.block_on(async {
        let shards = ShardSet::open(dir.path(), 2).expect("a store");
        shards.vindex_create("tp", DIM, 0, 1).await.expect("index");
        // One warm batch first: the first write of an index pays for opening
        // it, which is not what this measures.
        let _ = shards.vmset("tp", items(4096), 0, None).await;
        let mut best = 0f64;
        for round in 0..3 {
            let batch: Vec<_> = (0..4096u64)
                .map(|i| {
                    (
                        1_000_000 + round * 4096 + i,
                        vec![0.5f32; DIM as usize],
                        None,
                    )
                })
                .collect();
            let start = Instant::now();
            let r = shards.vmset("tp", batch, 0, None).await;
            let secs = start.elapsed().as_secs_f64();
            assert!(r.iter().all(Result::is_ok));
            best = best.max(4096.0 / secs);
        }
        println!("vmset throughput: {best:.0} items/s (best of 3, bound {VMSET_INFLIGHT})");
    });
}
