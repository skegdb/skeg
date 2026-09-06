//! Lifecycle / info trait tests on the skeg-rigging-skeg adapter.

use skeg_rigging::prelude::*;
use skeg_rigging::{CAP_VECTOR_KV, TenantInfo, TenantLifecycle};
use skeg_rigging_skeg::Tenant;

const DIM: u32 = 4;

fn unit(at: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM as usize];
    v[at] = 1.0;
    v
}

#[test]
fn create_new_rejects_existing_path() {
    let dir = tempfile::tempdir().unwrap();
    let tid = TenantId::from_bytes([1; 16]);
    {
        let t = Tenant::create_new(dir.path(), tid, DIM).expect("first create");
        t.insert(RecordId(1), unit(0), true, vec![], b"x".to_vec())
            .unwrap();
        t.flush().unwrap();
    }
    // Second create_new must fail with AlreadyExists.
    match Tenant::create_new(dir.path(), tid, DIM) {
        Ok(_) => panic!("expected AlreadyExists"),
        Err(e) => {
            let msg = format!("{e}");
            assert!(msg.contains("already exists"), "got {msg}");
        }
    }
}

#[test]
fn tenant_info_reports_capabilities_and_counts() {
    let dir = tempfile::tempdir().unwrap();
    let tid = TenantId::from_bytes([2; 16]);
    let t = Tenant::create_new(dir.path(), tid, DIM).unwrap();
    for i in 0..5 {
        t.insert(
            RecordId(i),
            unit((i % 4) as usize),
            true,
            vec![],
            format!("p{i}").into_bytes(),
        )
        .unwrap();
    }
    t.flush().unwrap();
    assert_eq!(t.tenant_id(), tid);
    assert_eq!(<Tenant as TenantInfo>::embedding_dim(&t), DIM);
    assert_eq!(<Tenant as TenantInfo>::record_count(&t), 5);
    let caps = t.capabilities();
    assert_eq!(caps, vec![CAP_VECTOR_KV]);
}

#[test]
fn snapshot_then_restore_round_trips_records() {
    let workdir = tempfile::tempdir().unwrap();
    let src = workdir.path().join("source");
    let snap = workdir.path().join("snapshot");
    let restored = workdir.path().join("restored");

    let tid = TenantId::from_bytes([3; 16]);
    {
        let t = Tenant::create_new(&src, tid, DIM).unwrap();
        t.insert(
            RecordId(1),
            unit(0),
            true,
            vec!["a".into()],
            b"hello".to_vec(),
        )
        .unwrap();
        t.insert(
            RecordId(2),
            unit(1),
            false,
            vec!["b".into()],
            b"world".to_vec(),
        )
        .unwrap();
        t.flush().unwrap();
        let boxed: Box<dyn TenantLifecycle> = Box::new(t);
        boxed.snapshot(&snap).expect("snapshot");
    }

    let restored_t = Tenant::restore_from(&snap, &restored).expect("restore");
    assert_eq!(restored_t.tenant_id(), tid);
    assert_eq!(<Tenant as IterVectors>::record_count(&restored_t), 2);

    let hits = restored_t
        .query_filtered(&unit(0), 10, &|m: &RecordMeta<'_>| m.shareable)
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].record_id, RecordId(1));
    assert_eq!(&hits[0].payload[..], b"hello");
}

#[test]
fn destroy_removes_dir() {
    let workdir = tempfile::tempdir().unwrap();
    let tdir = workdir.path().join("to-kill");
    let tid = TenantId::from_bytes([4; 16]);
    let t = Tenant::create_new(&tdir, tid, DIM).unwrap();
    t.insert(RecordId(1), unit(0), true, vec![], vec![])
        .unwrap();
    t.flush().unwrap();
    assert!(tdir.exists());

    let boxed: Box<dyn TenantLifecycle> = Box::new(t);
    boxed.destroy().expect("destroy");
    assert!(!tdir.exists(), "directory still present after destroy");
}

#[test]
fn restore_into_populated_dest_fails() {
    let workdir = tempfile::tempdir().unwrap();
    let src = workdir.path().join("source");
    let dest = workdir.path().join("dest");

    let tid = TenantId::from_bytes([5; 16]);
    {
        let t = Tenant::create_new(&src, tid, DIM).unwrap();
        t.insert(RecordId(1), unit(0), true, vec![], vec![])
            .unwrap();
        t.flush().unwrap();
    }
    // Populate dest with its own tenant.
    {
        let t = Tenant::create_new(&dest, TenantId::from_bytes([6; 16]), DIM).unwrap();
        t.insert(RecordId(99), unit(1), true, vec![], vec![])
            .unwrap();
        t.flush().unwrap();
    }
    // Restore must refuse.
    match Tenant::restore_from(&src, &dest) {
        Ok(_) => panic!("expected AlreadyExists"),
        Err(e) => {
            let msg = format!("{e}");
            assert!(msg.contains("already populated"), "got {msg}");
        }
    }
}
