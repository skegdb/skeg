//! Performance gates for the multi-tenant orchestration layer.
//!
//! Run with:
//!   cargo test --release --test gates -p skeg-multi-tenant
//!
//! Gates are skipped in debug mode (write-path latency would inflate
//! ~10x without `--release`, making thresholds meaningless). Each gate
//! exercises one hot operation and asserts a best-of-N timing.

use std::sync::Arc;
use std::time::Instant;

use skeg_multi_tenant::tenant_primitives::QuotaTracker;
use skeg_multi_tenant::{MultiTenantRoot, SkegTenantId, TenantHandle};
use skeg_rigging::{Quota, RecordId, TenantQuota};

const DIM: u32 = 32;

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

// ── Thresholds ──────────────────────────────────────────────────────

/// `open_scoped` is one tracker hash-map lookup + one adapter `open`.
/// Best-of-20 below 5 ms (adapter open dominates).
const GATE_OPEN_SCOPED_MS: u128 = 5;

/// `TenantQuota::set_quota` is two relaxed atomic stores. Best-of-100
/// below 1 us.
const GATE_SET_QUOTA_US: u128 = 1;

/// `TenantQuota::quota` (read) + `current_usage` (read). Same shape —
/// two atomic loads each. Best-of-100 below 1 us.
const GATE_READ_USAGE_US: u128 = 1;

/// `TenantHandle::insert` with an empty quota (CAS loop runs but never
/// trips a cap). Includes the underlying adapter insert. Best-of-50
/// below 1 ms.
const GATE_INSERT_UNCAPPED_MS: u128 = 1;

/// `TenantHandle::insert` against a cap that has just been hit. Quota
/// path short-circuits before the adapter touches disk. Best-of-100
/// below 5 us.
const GATE_INSERT_REJECTED_US: u128 = 5;

// ── Helpers ─────────────────────────────────────────────────────────

fn unit(at: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM as usize];
    v[at] = 1.0;
    v
}

fn build_handle(root_dir: &std::path::Path, tid_byte: u8) -> (Arc<MultiTenantRoot>, TenantHandle) {
    let tracker = Arc::new(QuotaTracker::new());
    let root = Arc::new(MultiTenantRoot::new(root_dir).with_quota_tracker(tracker.clone()));
    let tid = SkegTenantId::from_bytes([tid_byte; 16]);
    let handle = root.open_scoped(tid, DIM).unwrap();
    (root, handle)
}

// ── Gates ───────────────────────────────────────────────────────────

#[test]
fn gate_open_scoped_under_threshold() {
    if skip_unless_release() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let tracker = Arc::new(QuotaTracker::new());
    let root = MultiTenantRoot::new(dir.path()).with_quota_tracker(tracker);
    // Warm-up: first open builds the adapter dir.
    let _ = root
        .open_scoped(SkegTenantId::from_bytes([0xff; 16]), DIM)
        .unwrap();
    let mut best_ms = u128::MAX;
    for i in 0..20u8 {
        let tid = SkegTenantId::from_bytes([i; 16]);
        let t = Instant::now();
        let _ = root.open_scoped(tid, DIM).unwrap();
        best_ms = best_ms.min(t.elapsed().as_millis());
    }
    assert!(
        best_ms <= GATE_OPEN_SCOPED_MS,
        "open_scoped best-of-20 = {best_ms} ms, gate {GATE_OPEN_SCOPED_MS} ms"
    );
}

#[test]
fn gate_set_quota_under_threshold() {
    if skip_unless_release() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_root, handle) = build_handle(dir.path(), 0x11);
    // Warm-up.
    for _ in 0..16 {
        handle
            .set_quota(Quota {
                max_records: Some(1_000),
                max_bytes: Some(1_000_000),
            })
            .unwrap();
    }
    let mut best_us = u128::MAX;
    for i in 0..100u64 {
        let t = Instant::now();
        handle
            .set_quota(Quota {
                max_records: Some(1_000 + i),
                max_bytes: Some(1_000_000),
            })
            .unwrap();
        best_us = best_us.min(t.elapsed().as_micros());
    }
    assert!(
        best_us <= GATE_SET_QUOTA_US,
        "set_quota best-of-100 = {best_us} us, gate {GATE_SET_QUOTA_US} us"
    );
}

#[test]
fn gate_read_usage_under_threshold() {
    if skip_unless_release() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_root, handle) = build_handle(dir.path(), 0x22);
    for i in 0..5 {
        handle
            .insert(RecordId(i), unit(0), false, vec![], b"x".to_vec())
            .unwrap();
    }
    // Warm-up.
    for _ in 0..16 {
        let _ = handle.current_usage();
        let _ = handle.quota();
    }
    let mut best_us = u128::MAX;
    for _ in 0..100 {
        let t = Instant::now();
        let _ = handle.current_usage();
        best_us = best_us.min(t.elapsed().as_micros());
    }
    assert!(
        best_us <= GATE_READ_USAGE_US,
        "current_usage best-of-100 = {best_us} us, gate {GATE_READ_USAGE_US} us"
    );
}

#[test]
fn gate_insert_uncapped_under_threshold() {
    if skip_unless_release() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_root, handle) = build_handle(dir.path(), 0x33);
    // Warm-up.
    for i in 0..10u64 {
        handle
            .insert(RecordId(i), unit(0), false, vec![], b"x".to_vec())
            .unwrap();
    }
    let mut best_ms = u128::MAX;
    for i in 100..150u64 {
        let t = Instant::now();
        handle
            .insert(RecordId(i), unit(0), false, vec![], b"x".to_vec())
            .unwrap();
        best_ms = best_ms.min(t.elapsed().as_millis());
    }
    assert!(
        best_ms <= GATE_INSERT_UNCAPPED_MS,
        "insert(uncapped) best-of-50 = {best_ms} ms, gate \
         {GATE_INSERT_UNCAPPED_MS} ms"
    );
}

#[test]
fn gate_insert_rejected_under_threshold() {
    if skip_unless_release() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (_root, handle) = build_handle(dir.path(), 0x44);
    // Cap at 0 — every insert is rejected before the adapter sees it.
    handle
        .set_quota(Quota {
            max_records: Some(0),
            ..Quota::UNLIMITED
        })
        .unwrap();
    // Warm-up.
    for i in 0..16u64 {
        let _ = handle.insert(RecordId(i), unit(0), false, vec![], b"x".to_vec());
    }
    let mut best_us = u128::MAX;
    for i in 100..200u64 {
        let t = Instant::now();
        let _ = handle.insert(RecordId(i), unit(0), false, vec![], b"x".to_vec());
        best_us = best_us.min(t.elapsed().as_micros());
    }
    assert!(
        best_us <= GATE_INSERT_REJECTED_US,
        "insert(rejected) best-of-100 = {best_us} us, gate \
         {GATE_INSERT_REJECTED_US} us"
    );
}
