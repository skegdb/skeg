//! F.14 - TenantStats on the DiskVamana-backed adapter.
//!
//! `TenantQuota` lives in `skeg-multi-tenant` (a sister crate to
//! `skeg-tenant`) because enforcement requires the auth context that
//! skeg-tenant holds; this adapter only exposes observation counters.

use skeg_rigging::{RecordId, TenantId, TenantStats};
use skeg_rigging_skeg::Tenant;

const DIM: u32 = 4;

fn unit(at: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM as usize];
    v[at] = 1.0;
    v
}

#[test]
fn stats_bytes_on_disk_grows_with_records() {
    let dir = tempfile::tempdir().unwrap();
    let t = Tenant::create_new(dir.path(), TenantId::ZERO, DIM).unwrap();
    let empty_bytes = t.bytes_on_disk();
    for i in 0..50u64 {
        t.insert(
            RecordId(i),
            unit((i % 4) as usize),
            true,
            vec!["topic".into()],
            format!("payload-{i}").into_bytes(),
        )
        .unwrap();
    }
    t.flush().unwrap();
    let full_bytes = t.bytes_on_disk();
    assert!(
        full_bytes > empty_bytes,
        "bytes_on_disk should grow: empty={empty_bytes} full={full_bytes}"
    );
    // meta.json alone is hundreds of bytes after flush.
    assert!(full_bytes >= 100);
}

#[test]
fn stats_record_count_matches_inserts() {
    let dir = tempfile::tempdir().unwrap();
    let t = Tenant::create_new(dir.path(), TenantId::ZERO, DIM).unwrap();
    for i in 0..23u64 {
        t.insert(
            RecordId(i),
            unit((i % 4) as usize),
            true,
            vec![],
            format!("p-{i}").into_bytes(),
        )
        .unwrap();
    }
    assert_eq!(<Tenant as TenantStats>::record_count(&t), 23);
}

#[test]
fn stats_memory_resident_returns_zero_in_v01() {
    // The adapter mmaps via the OS page cache; per-tenant RSS is not
    // tracked. v0.1 contract: always 0 here. If future skeg-vector
    // exposes a residency counter, this test should be updated.
    let dir = tempfile::tempdir().unwrap();
    let t = Tenant::create_new(dir.path(), TenantId::ZERO, DIM).unwrap();
    t.insert(RecordId(1), unit(0), false, vec![], b"x".to_vec())
        .unwrap();
    assert_eq!(t.memory_resident(), 0);
}
