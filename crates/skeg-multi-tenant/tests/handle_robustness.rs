//! Robustness coverage for `TenantHandle`:
//!
//! - trait-object dispatch via every forwarded rigging trait
//! - concurrent inserts across two handles to the same tenant id
//! - quota refund on delete
//! - empty embedding rejection routed through the quota path
//! - lifecycle round-trip (snapshot via handle, restore via adapter)

use std::sync::Arc;
use std::thread;

use skeg_multi_tenant::{MultiTenantRoot, RiggingTenant, SkegTenantId, TenantHandle};
use skeg_rigging::{
    EventFilter, IterVectors, QueryFiltered, Quota, ReadOnlyView, RecordId, RecordMeta,
    TenantEvents, TenantInfo, TenantQuota, TenantStats,
};
use skeg_tenant::quota::QuotaTracker;

const DIM: u32 = 4;

fn unit(at: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM as usize];
    v[at] = 1.0;
    v
}

fn open_handle(root_dir: &std::path::Path, tid_byte: u8) -> (TenantHandle, Arc<QuotaTracker>) {
    let tracker = Arc::new(QuotaTracker::new());
    let root = MultiTenantRoot::new(root_dir).with_quota_tracker(tracker.clone());
    let tid = SkegTenantId::from_bytes([tid_byte; 16]);
    let handle = root.open_scoped(tid, DIM).unwrap();
    (handle, tracker)
}

// ─── Trait-object dispatch ────────────────────────────────────────────

#[test]
fn handle_acts_as_tenant_info_trait_object() {
    let dir = tempfile::tempdir().unwrap();
    let (h, _) = open_handle(dir.path(), 0x11);
    h.insert(
        RecordId(1),
        unit(0),
        true,
        vec!["topic".into()],
        b"x".to_vec(),
    )
    .unwrap();
    let info: &dyn TenantInfo = &h;
    assert_eq!(info.embedding_dim(), DIM);
    assert_eq!(info.record_count(), 1);
    assert!(!info.capabilities().is_empty());
}

#[test]
fn handle_acts_as_tenant_stats_trait_object() {
    let dir = tempfile::tempdir().unwrap();
    let (h, _) = open_handle(dir.path(), 0x12);
    h.insert(RecordId(1), unit(0), true, vec![], b"hello".to_vec())
        .unwrap();
    h.flush().unwrap();
    let stats: &dyn TenantStats = &h;
    assert_eq!(stats.record_count(), 1);
    assert!(stats.bytes_on_disk() > 0);
    assert_eq!(stats.memory_resident(), 0);
}

#[test]
fn handle_acts_as_iter_and_query_trait_objects() {
    let dir = tempfile::tempdir().unwrap();
    let (h, _) = open_handle(dir.path(), 0x13);
    for i in 0..5u64 {
        h.insert(
            RecordId(i),
            unit((i % 4) as usize),
            true,
            vec![],
            format!("p{i}").into_bytes(),
        )
        .unwrap();
    }
    let iv: &dyn IterVectors = &h;
    assert_eq!(iv.record_count(), 5);

    let qf: &dyn QueryFiltered = &h;
    let hits = qf
        .query_filtered(&unit(0), 10, &|m: &RecordMeta<'_>| m.shareable)
        .unwrap();
    assert!(!hits.is_empty());
}

#[test]
fn boxed_handle_is_readonly_view() {
    let dir = tempfile::tempdir().unwrap();
    let (h, _) = open_handle(dir.path(), 0x14);
    h.insert(RecordId(1), unit(0), true, vec![], b"x".to_vec())
        .unwrap();
    let view: Box<dyn ReadOnlyView> = Box::new(h);
    assert_eq!(view.record_count(), 1);
    view.close().unwrap();
}

#[test]
fn handle_emits_events_through_trait_object() {
    let dir = tempfile::tempdir().unwrap();
    let (h, _) = open_handle(dir.path(), 0x15);
    let stream = (&h as &dyn TenantEvents).subscribe(EventFilter::ALL);
    h.insert(RecordId(7), unit(0), true, vec![], b"x".to_vec())
        .unwrap();
    let ev = stream
        .recv_timeout(std::time::Duration::from_millis(200))
        .expect("event should arrive");
    let _ = ev; // shape covered in mock + adapter tests
}

// ─── Quota refund on delete ───────────────────────────────────────────

#[test]
fn delete_refunds_vector_counter() {
    let dir = tempfile::tempdir().unwrap();
    let (h, _) = open_handle(dir.path(), 0x21);
    h.set_quota(Quota {
        max_records: Some(2),
        ..Quota::UNLIMITED
    })
    .unwrap();
    h.insert(RecordId(1), unit(0), false, vec![], b"x".to_vec())
        .unwrap();
    h.insert(RecordId(2), unit(1), false, vec![], b"y".to_vec())
        .unwrap();
    assert_eq!(h.current_usage().records, 2);
    // Capacity full - third insert blocked.
    assert!(
        h.insert(RecordId(3), unit(2), false, vec![], b"z".to_vec())
            .is_err()
    );
    // Delete one row; the counter refunds and the third insert
    // now lands within the cap.
    assert!(h.delete(RecordId(1)).unwrap());
    assert_eq!(h.current_usage().records, 1);
    h.insert(RecordId(3), unit(2), false, vec![], b"z".to_vec())
        .unwrap();
    assert_eq!(h.current_usage().records, 2);
}

#[test]
fn delete_of_missing_record_does_not_refund() {
    let dir = tempfile::tempdir().unwrap();
    let (h, _) = open_handle(dir.path(), 0x22);
    h.insert(RecordId(1), unit(0), false, vec![], b"x".to_vec())
        .unwrap();
    let before = h.current_usage().records;
    assert!(!h.delete(RecordId(999)).unwrap());
    assert_eq!(h.current_usage().records, before);
}

// ─── Concurrency ──────────────────────────────────────────────────────

#[test]
fn two_handles_same_tenant_share_quota_counters() {
    let dir = tempfile::tempdir().unwrap();
    let tracker = Arc::new(QuotaTracker::new());
    let root = MultiTenantRoot::new(dir.path()).with_quota_tracker(tracker.clone());
    let tid = SkegTenantId::from_bytes([0x31; 16]);

    // Two independent handles for the same tenant id pick up the
    // SAME quota entry from the tracker. (Adapter side: they each
    // get their own `Tenant`, which is fine - Vamana coordinates
    // through DiskVamana's own locking.)
    let h1 = root.open_scoped(tid, DIM).unwrap();
    let h2 = root.open_scoped(tid, DIM).unwrap();
    h1.set_quota(Quota {
        max_records: Some(4),
        ..Quota::UNLIMITED
    })
    .unwrap();
    // Both handles see the same cap.
    assert_eq!(h2.quota().max_records, Some(4));

    h1.insert(RecordId(1), unit(0), false, vec![], b"x".to_vec())
        .unwrap();
    h2.insert(RecordId(2), unit(1), false, vec![], b"y".to_vec())
        .unwrap();
    // Combined view through either handle.
    assert_eq!(h1.current_usage().records, 2);
    assert_eq!(h2.current_usage().records, 2);
}

#[test]
fn concurrent_inserts_respect_shared_quota_cap() {
    let dir = tempfile::tempdir().unwrap();
    let tracker = Arc::new(QuotaTracker::new());
    let root = MultiTenantRoot::new(dir.path()).with_quota_tracker(tracker.clone());
    let tid = SkegTenantId::from_bytes([0x41; 16]);
    let h = Arc::new(root.open_scoped(tid, DIM).unwrap());
    h.set_quota(Quota {
        max_records: Some(50),
        ..Quota::UNLIMITED
    })
    .unwrap();

    // 4 threads each try to insert 30 records (120 total against a
    // cap of 50). Exactly 50 must succeed; the rest must fail with
    // QuotaExceeded. The tracker's CAS guarantees no over-admission.
    let mut threads = Vec::new();
    let counter = Arc::new(std::sync::atomic::AtomicU64::new(0));
    for t in 0..4u64 {
        let h = h.clone();
        let counter = counter.clone();
        threads.push(thread::spawn(move || {
            let mut local_ok = 0u64;
            for i in 0..30u64 {
                let id = RecordId(t * 1000 + i);
                if h.insert(id, unit(0), false, vec![], b"x".to_vec()).is_ok() {
                    local_ok += 1;
                }
            }
            counter.fetch_add(local_ok, std::sync::atomic::Ordering::Relaxed);
        }));
    }
    for t in threads {
        t.join().unwrap();
    }
    let total_ok = counter.load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(
        total_ok, 50,
        "expected exactly 50 admissions, got {total_ok}"
    );
    assert_eq!(h.current_usage().records, 50);
}

// ─── Lifecycle round-trip ─────────────────────────────────────────────

#[test]
fn snapshot_from_handle_then_restore_via_adapter() {
    let workdir = tempfile::tempdir().unwrap();
    let root_dir = workdir.path().join("root");
    let snap_dir = workdir.path().join("snap");
    let restored_dir = workdir.path().join("restored");

    let tracker = Arc::new(QuotaTracker::new());
    let root = MultiTenantRoot::new(&root_dir).with_quota_tracker(tracker);
    let tid = SkegTenantId::from_bytes([0x51; 16]);
    let h = root.open_scoped(tid, DIM).unwrap();
    for i in 0..7u64 {
        h.insert(
            RecordId(i),
            unit((i % 4) as usize),
            i % 2 == 0,
            vec![],
            format!("p{i}").into_bytes(),
        )
        .unwrap();
    }
    h.flush().unwrap();

    // Snapshot via the handle's forwarded API.
    h.snapshot(&snap_dir).expect("snapshot");

    // Restore via the adapter directly.
    let restored = RiggingTenant::restore_from(&snap_dir, &restored_dir).expect("restore");
    assert_eq!(<RiggingTenant as IterVectors>::record_count(&restored), 7);
}

// ─── Adapter rejection rolls quota back ───────────────────────────────

#[test]
fn dim_mismatch_rolls_quota_back() {
    let dir = tempfile::tempdir().unwrap();
    let (h, _) = open_handle(dir.path(), 0x61);
    let before = h.current_usage();
    // Wrong-dim embedding: tracker accepts, adapter rejects. Counters
    // must be restored.
    let bad = vec![1.0f32, 0.0, 0.0]; // dim 3, expected 4
    let _ = h.insert(RecordId(1), bad, false, vec![], b"x".to_vec());
    let after = h.current_usage();
    assert_eq!(
        before, after,
        "quota counters drifted on adapter rejection: before={before:?} after={after:?}"
    );
}
