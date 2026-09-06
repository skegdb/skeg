//! Performance gates for the DiskVamana-backed Tenant adapter.
//!
//! Run with:
//!   cargo test --release --test gates
//!
//! Skip in debug (the DiskVamana code path is ~10× slower without
//! optimisations and thresholds would be meaningless). Mirrors the
//! gates policy in `hansa/private/gates.md`.

use std::time::{Duration, Instant};

use skeg_rigging::{
    EventFilter, Filter, IterVectors, QueryFiltered, RecordId, RecordMeta, TenantEvents, TenantId,
    TenantLifecycle, TenantStats,
};
use skeg_rigging_skeg::Tenant;

// ─────────────────────────────────────────────────────────────────────
// Thresholds (M-series Apple Silicon dev laptop, release mode)
// ─────────────────────────────────────────────────────────────────────

/// Latency for one `query_filtered(top_k=10)` at 1000 records, dim=32.
/// Baseline ~20 µs; gate at 200 µs (10× headroom).
const GATE_QUERY_1K_DIM32_US: u128 = 200;

/// Snapshot 1000 records to a fresh dir. Baseline ~1.4 ms; gate at 50 ms.
const GATE_SNAPSHOT_1K_MS: u128 = 50;

/// Restore from a snapshot of 1000 records. Baseline ~2.3 ms; gate at 100 ms.
const GATE_RESTORE_1K_MS: u128 = 100;

/// Wall-clock to compute `bytes_on_disk` over a 1000-record tenant.
/// Baseline ~50 µs; gate at 5 ms (100× headroom - fs::metadata is
/// the dominant cost and varies wildly with kernel cache state).
const GATE_BYTES_ON_DISK_1K_US: u128 = 5_000;

/// Broadcast latency for one event to one subscriber. Baseline ~3 µs;
/// gate at 100 µs.
const GATE_EVENT_ONE_SUB_US: u128 = 100;

/// Broadcast latency for one event to 16 subscribers. Baseline ~10 µs;
/// gate at 500 µs.
const GATE_EVENT_16_SUBS_US: u128 = 500;

fn skip_unless_release() -> bool {
    if cfg!(debug_assertions) {
        eprintln!(
            "[gates] skipping in debug mode; run `cargo test --release --test gates` to enforce"
        );
        true
    } else {
        false
    }
}

const DIM: u32 = 32;

fn synth_vector(seed: u64) -> Vec<f32> {
    (0..DIM)
        .map(|d| {
            let h = ((seed as u32).wrapping_mul(2654435761) ^ d.wrapping_mul(40503)) as f32;
            (h.sin() + 1.0) * 0.5
        })
        .collect()
}

struct AcceptAll;
impl Filter for AcceptAll {
    fn accept(&self, _m: &RecordMeta<'_>) -> bool {
        true
    }
}

fn populate(path: &std::path::Path, n: u64) -> Tenant {
    let t = Tenant::create_new(path, TenantId::ZERO, DIM).expect("create");
    for i in 0..n {
        t.insert(
            RecordId(i),
            synth_vector(i),
            true,
            vec![],
            format!("p{i}").into_bytes(),
        )
        .expect("insert");
    }
    t.flush().expect("flush");
    t
}

#[test]
fn gate_query_1k_dim32_under_threshold() {
    if skip_unless_release() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let tenant = populate(dir.path(), 1_000);
    let q = synth_vector(0);
    // Warm-up.
    for _ in 0..5 {
        let _ = tenant.query_filtered(&q, 10, &AcceptAll).unwrap();
    }
    let mut best_us = u128::MAX;
    for _ in 0..50 {
        let t = Instant::now();
        let hits = tenant.query_filtered(&q, 10, &AcceptAll).unwrap();
        best_us = best_us.min(t.elapsed().as_micros());
        assert!(!hits.is_empty());
    }
    assert!(
        best_us <= GATE_QUERY_1K_DIM32_US,
        "query best-of-50 = {best_us} µs, gate {GATE_QUERY_1K_DIM32_US} µs"
    );
}

#[test]
fn gate_snapshot_1k_under_threshold() {
    if skip_unless_release() {
        return;
    }
    let src = tempfile::tempdir().unwrap();
    let snap = tempfile::tempdir().unwrap();
    let tenant = populate(src.path(), 1_000);
    let boxed: Box<dyn TenantLifecycle> = Box::new(tenant);
    let t = Instant::now();
    boxed.snapshot(snap.path()).expect("snapshot");
    let elapsed_ms = t.elapsed().as_millis();
    assert!(
        elapsed_ms <= GATE_SNAPSHOT_1K_MS,
        "snapshot took {elapsed_ms} ms, gate {GATE_SNAPSHOT_1K_MS} ms"
    );
}

#[test]
fn gate_restore_1k_under_threshold() {
    if skip_unless_release() {
        return;
    }
    let src = tempfile::tempdir().unwrap();
    let snap = tempfile::tempdir().unwrap();
    let dest = tempfile::tempdir().unwrap();
    let tenant = populate(src.path(), 1_000);
    let boxed: Box<dyn TenantLifecycle> = Box::new(tenant);
    boxed.snapshot(snap.path()).expect("snapshot");
    let t = Instant::now();
    let restored = Tenant::restore_from(snap.path(), dest.path()).expect("restore");
    let elapsed_ms = t.elapsed().as_millis();
    assert!(
        elapsed_ms <= GATE_RESTORE_1K_MS,
        "restore took {elapsed_ms} ms, gate {GATE_RESTORE_1K_MS} ms"
    );
    // Correctness check: restored holds the same N records.
    assert_eq!(
        <Tenant as IterVectors>::record_count(&restored),
        1_000,
        "record count differs after restore"
    );
}

#[test]
fn gate_bytes_on_disk_1k_under_threshold() {
    if skip_unless_release() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let tenant = populate(dir.path(), 1_000);
    // Warm-up: prime the FS cache.
    for _ in 0..3 {
        let _ = tenant.bytes_on_disk();
    }
    let mut best_us = u128::MAX;
    for _ in 0..20 {
        let t = Instant::now();
        let n = tenant.bytes_on_disk();
        best_us = best_us.min(t.elapsed().as_micros());
        assert!(n > 0, "bytes_on_disk returned zero on populated tenant");
    }
    assert!(
        best_us <= GATE_BYTES_ON_DISK_1K_US,
        "bytes_on_disk best-of-20 = {best_us} µs, gate {GATE_BYTES_ON_DISK_1K_US} µs"
    );
}

#[test]
fn gate_event_broadcast_one_subscriber_under_threshold() {
    if skip_unless_release() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let tenant = Tenant::create_new(dir.path(), TenantId::ZERO, DIM).unwrap();
    let stream = tenant.subscribe(EventFilter::ALL);
    // Warm-up.
    for i in 0..5 {
        tenant
            .insert(RecordId(i), synth_vector(i), false, vec![], b"x".to_vec())
            .unwrap();
        let _ = stream.recv_timeout(Duration::from_millis(50));
    }
    let mut best_us = u128::MAX;
    for i in 100..150u64 {
        let t = Instant::now();
        tenant
            .insert(RecordId(i), synth_vector(i), false, vec![], b"x".to_vec())
            .unwrap();
        let _ = stream.recv_timeout(Duration::from_millis(50)).unwrap();
        best_us = best_us.min(t.elapsed().as_micros());
    }
    assert!(
        best_us <= GATE_EVENT_ONE_SUB_US,
        "1-sub broadcast best-of-50 = {best_us} µs, gate {GATE_EVENT_ONE_SUB_US} µs"
    );
}

#[test]
fn gate_event_broadcast_16_subscribers_under_threshold() {
    if skip_unless_release() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let tenant = Tenant::create_new(dir.path(), TenantId::ZERO, DIM).unwrap();
    let streams: Vec<_> = (0..16)
        .map(|_| tenant.subscribe(EventFilter::ALL))
        .collect();
    // Warm-up.
    for i in 0..5u64 {
        tenant
            .insert(RecordId(i), synth_vector(i), false, vec![], b"x".to_vec())
            .unwrap();
        for s in &streams {
            let _ = s.recv_timeout(Duration::from_millis(50));
        }
    }
    let mut best_us = u128::MAX;
    for i in 100..150u64 {
        let t = Instant::now();
        tenant
            .insert(RecordId(i), synth_vector(i), false, vec![], b"x".to_vec())
            .unwrap();
        for s in &streams {
            let _ = s.recv_timeout(Duration::from_millis(50)).unwrap();
        }
        best_us = best_us.min(t.elapsed().as_micros());
    }
    assert!(
        best_us <= GATE_EVENT_16_SUBS_US,
        "16-sub broadcast best-of-50 = {best_us} µs, gate {GATE_EVENT_16_SUBS_US} µs"
    );
}

#[test]
fn gate_snapshot_is_byte_equal_for_meta_json() {
    if skip_unless_release() {
        return;
    }
    let src = tempfile::tempdir().unwrap();
    let snap = tempfile::tempdir().unwrap();
    let tenant = populate(src.path(), 50);
    let original = std::fs::read(Tenant::meta_path(src.path())).unwrap();
    let boxed: Box<dyn TenantLifecycle> = Box::new(tenant);
    boxed.snapshot(snap.path()).expect("snapshot");
    let snapshot = std::fs::read(Tenant::meta_path(snap.path())).unwrap();
    assert_eq!(original, snapshot, "snapshot meta.json differs from source");
}
