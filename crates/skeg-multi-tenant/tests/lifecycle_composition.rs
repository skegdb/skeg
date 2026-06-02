//! The multi-tenant root's `open()` returns a `RiggingTenant`
//! (alias of `skeg_rigging_skeg::Tenant`) which implements
//! `TenantInfo` + `TenantLifecycle`. This file asserts that the
//! composition works: an orchestrator holding only a
//! `MultiTenantRoot` can introspect + snapshot + destroy each tenant
//! via the trait surface, without depending on the concrete type.

use std::sync::Arc;
use std::time::Duration;

use skeg_multi_tenant::{MultiTenantRoot, RiggingTenant, SkegTenantId};
use skeg_rigging::prelude::*;
use skeg_rigging::{CAP_VECTOR_KV, TenantInfo, TenantLifecycle};
use skeg_tenant::auth::TokenStore;

const DIM: u32 = 4;

fn unit(at: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM as usize];
    v[at] = 1.0;
    v
}

#[test]
fn tenant_via_root_exposes_info_capabilities() {
    let dir = tempfile::tempdir().unwrap();
    let root = MultiTenantRoot::new(dir.path());
    let tid = SkegTenantId::from_bytes([0x55; 16]);

    let tenant = root.open(tid, DIM).expect("open");
    tenant
        .insert(
            RecordId(1),
            unit(0),
            true,
            vec!["topic".into()],
            b"hello".to_vec(),
        )
        .unwrap();
    tenant.flush().unwrap();

    let info: &dyn TenantInfo = &tenant;
    assert_eq!(info.embedding_dim(), DIM);
    assert_eq!(info.record_count(), 1);
    assert_eq!(info.capabilities(), vec![CAP_VECTOR_KV]);
}

#[test]
fn snapshot_destroy_round_trip_via_root() {
    let workdir = tempfile::tempdir().unwrap();
    let bridge_root_dir = workdir.path().join("root");
    let snap_dir = workdir.path().join("snap");

    let root = MultiTenantRoot::new(&bridge_root_dir);
    let tid = SkegTenantId::from_bytes([0x77; 16]);

    let tenant = root.open(tid, DIM).unwrap();
    tenant
        .insert(RecordId(1), unit(0), true, vec![], b"x".to_vec())
        .unwrap();
    tenant
        .insert(RecordId(2), unit(1), false, vec![], b"y".to_vec())
        .unwrap();
    tenant.flush().unwrap();

    let lifecycle: Box<dyn TenantLifecycle> = Box::new(tenant);
    lifecycle.snapshot(&snap_dir).expect("snapshot");

    let restored_dir = workdir.path().join("restored");
    let restored = RiggingTenant::restore_from(&snap_dir, &restored_dir).expect("restore");
    assert_eq!(
        restored.tenant_id(),
        skeg_rigging::TenantId(*tid.as_bytes())
    );
    assert_eq!(<RiggingTenant as IterVectors>::record_count(&restored), 2);
}

#[test]
fn open_with_token_then_destroy_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(TokenStore::from_key([1u8; 32], Duration::from_secs(60)));
    let root = MultiTenantRoot::new(dir.path()).with_tokens(store.clone());

    let tid = SkegTenantId::from_bytes([0xaa; 16]);
    let token = store.issue(tid).unwrap();
    {
        let t = root.open_with_token(&token, DIM).unwrap();
        t.insert(RecordId(1), unit(0), true, vec![], vec![])
            .unwrap();
        t.flush().unwrap();
    }
    let t2 = root.open_with_token(&token, DIM).unwrap();
    let boxed: Box<dyn TenantLifecycle> = Box::new(t2);
    boxed.destroy().expect("destroy");
    assert!(!root.tenant_dir(tid).exists());
    let listed = root.list_tenants().unwrap();
    assert!(
        !listed.contains(&tid),
        "destroyed tenant still listed: {listed:?}"
    );
}
