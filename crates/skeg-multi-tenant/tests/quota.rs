//! `TenantQuota` enforcement via the multi-tenant root.

use std::sync::Arc;

use skeg_multi_tenant::{MultiTenantRoot, SkegTenantId, TenantHandle};
use skeg_rigging::{Quota, RecordId, TenantQuota};
use skeg_tenant::quota::QuotaTracker;

const DIM: u32 = 4;

fn unit(at: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM as usize];
    v[at] = 1.0;
    v
}

#[test]
fn default_quota_is_unlimited() {
    let dir = tempfile::tempdir().unwrap();
    let tracker = Arc::new(QuotaTracker::new());
    let root = MultiTenantRoot::new(dir.path()).with_quota_tracker(tracker);
    let tid = SkegTenantId::from_bytes([0x11; 16]);
    let scoped = root.open_scoped(tid, DIM).unwrap();
    assert!(scoped.quota().is_unlimited());
    assert_eq!(scoped.current_usage().records, 0);
}

#[test]
fn vectors_limit_blocks_inserts_past_cap() {
    let dir = tempfile::tempdir().unwrap();
    let tracker = Arc::new(QuotaTracker::new());
    let root = MultiTenantRoot::new(dir.path()).with_quota_tracker(tracker);
    let tid = SkegTenantId::from_bytes([0x22; 16]);
    let scoped = root.open_scoped(tid, DIM).unwrap();
    scoped
        .set_quota(Quota {
            max_records: Some(2),
            ..Quota::UNLIMITED
        })
        .unwrap();
    scoped
        .insert(RecordId(1), unit(0), false, vec![], b"x".to_vec())
        .unwrap();
    scoped
        .insert(RecordId(2), unit(1), false, vec![], b"y".to_vec())
        .unwrap();
    let err = scoped
        .insert(RecordId(3), unit(2), false, vec![], b"z".to_vec())
        .expect_err("third insert must hit the quota");
    let msg = format!("{err}");
    assert!(
        msg.contains("quota") || msg.contains("exceeded"),
        "got {msg}"
    );
    assert_eq!(scoped.current_usage().records, 2);
}

#[test]
fn bytes_limit_blocks_oversized_payload() {
    let dir = tempfile::tempdir().unwrap();
    let tracker = Arc::new(QuotaTracker::new());
    let root = MultiTenantRoot::new(dir.path()).with_quota_tracker(tracker);
    let tid = SkegTenantId::from_bytes([0x33; 16]);
    let scoped = root.open_scoped(tid, DIM).unwrap();
    scoped
        .set_quota(Quota {
            max_bytes: Some(64),
            ..Quota::UNLIMITED
        })
        .unwrap();
    // 4 floats * 4 B + 10-byte payload = 26 per record.
    scoped
        .insert(RecordId(1), unit(0), false, vec![], vec![b'p'; 10])
        .unwrap();
    scoped
        .insert(RecordId(2), unit(1), false, vec![], vec![b'q'; 10])
        .unwrap();
    // Third record (another 26 B) would push past 64 B.
    let err = scoped
        .insert(RecordId(3), unit(2), false, vec![], vec![b'r'; 10])
        .expect_err("third insert must hit the cache_bytes cap");
    assert!(format!("{err}").to_lowercase().contains("exceeded"));
}

#[test]
fn tracker_shared_across_tenants_isolates_counts() {
    let dir = tempfile::tempdir().unwrap();
    let tracker = Arc::new(QuotaTracker::new());
    let root = MultiTenantRoot::new(dir.path()).with_quota_tracker(tracker.clone());
    let a_id = SkegTenantId::from_bytes([0x44; 16]);
    let b_id = SkegTenantId::from_bytes([0x55; 16]);
    let a = root.open_scoped(a_id, DIM).unwrap();
    let b = root.open_scoped(b_id, DIM).unwrap();
    a.set_quota(Quota {
        max_records: Some(2),
        ..Quota::UNLIMITED
    })
    .unwrap();
    a.insert(RecordId(1), unit(0), false, vec![], vec![])
        .unwrap();
    a.insert(RecordId(2), unit(1), false, vec![], vec![])
        .unwrap();
    assert!(
        a.insert(RecordId(3), unit(2), false, vec![], vec![])
            .is_err()
    );
    for i in 0..5 {
        b.insert(RecordId(i), unit(0), false, vec![], vec![])
            .unwrap();
    }
    assert_eq!(b.current_usage().records, 5);
}

#[test]
fn open_scoped_requires_tracker() {
    let dir = tempfile::tempdir().unwrap();
    let root = MultiTenantRoot::new(dir.path());
    let tid = SkegTenantId::from_bytes([0x66; 16]);
    let res: Result<TenantHandle, _> = root.open_scoped(tid, DIM);
    assert!(res.is_err(), "open_scoped without tracker must fail");
}
