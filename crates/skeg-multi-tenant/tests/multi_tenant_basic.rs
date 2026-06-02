//! Sanity tests for the multi-tenant root: open / list / auth / read-only.

use std::sync::Arc;
use std::time::Duration;

use skeg_multi_tenant::{MultiTenantError, MultiTenantRoot, RiggingTenant, SkegTenantId};
use skeg_rigging::prelude::*;
use skeg_tenant::auth::TokenStore;

const DIM: u32 = 4;

fn unit(at: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM as usize];
    v[at] = 1.0;
    v
}

#[test]
fn open_and_list_tenants() {
    let dir = tempfile::tempdir().unwrap();
    let root = MultiTenantRoot::new(dir.path());

    let t1 = SkegTenantId::from_bytes([0x11; 16]);
    let t2 = SkegTenantId::from_bytes([0x22; 16]);

    {
        let tenant = root.open(t1, DIM).unwrap();
        tenant
            .insert(RecordId(1), unit(0), true, vec![], b"hello".to_vec())
            .unwrap();
        tenant.flush().unwrap();
    }
    {
        let tenant = root.open(t2, DIM).unwrap();
        tenant
            .insert(RecordId(1), unit(1), false, vec![], b"world".to_vec())
            .unwrap();
        tenant.flush().unwrap();
    }

    let listed = root.list_tenants().unwrap();
    assert_eq!(listed.len(), 2);
    assert!(listed.contains(&t1) && listed.contains(&t2));
}

#[test]
fn open_with_token_validates_against_store() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(TokenStore::from_key([7u8; 32], Duration::from_secs(60)));
    let root = MultiTenantRoot::new(dir.path()).with_tokens(store.clone());

    let tenant_id = SkegTenantId::from_bytes([0x42; 16]);

    let token = store.issue(tenant_id).unwrap();
    let tenant = root.open_with_token(&token, DIM).unwrap();
    tenant
        .insert(
            RecordId(1),
            unit(0),
            true,
            vec!["topic".into()],
            b"payload".to_vec(),
        )
        .unwrap();
    tenant.flush().unwrap();
    assert_eq!(<RiggingTenant as IterVectors>::record_count(&tenant), 1);

    // A garbage token must fail.
    let mut bad = token;
    bad[0] ^= 0xff;
    match root.open_with_token(&bad, DIM) {
        Ok(_) => panic!("expected TokenError on garbage token"),
        Err(e) => assert!(matches!(e, MultiTenantError::TokenError(_))),
    }
}

#[test]
fn open_without_token_store_errors() {
    let dir = tempfile::tempdir().unwrap();
    let root = MultiTenantRoot::new(dir.path());
    let dummy = [0u8; 47];
    match root.open_with_token(&dummy, DIM) {
        Ok(_) => panic!("expected NoTokenStore"),
        Err(e) => assert!(matches!(e, MultiTenantError::NoTokenStore)),
    }
}

#[test]
fn open_readonly_returns_box_dyn_readonly_view() {
    let dir = tempfile::tempdir().unwrap();
    let root = MultiTenantRoot::new(dir.path());
    let tid = SkegTenantId::from_bytes([0x33; 16]);

    {
        let tenant = root.open(tid, DIM).unwrap();
        tenant
            .insert(RecordId(1), unit(0), true, vec!["x".into()], b"hi".to_vec())
            .unwrap();
        tenant.flush().unwrap();
    }

    let view = root.open_readonly(tid).expect("open ro");
    assert_eq!(view.record_count(), 1);
    let hits = view
        .query_filtered(&unit(0), 5, &|m: &RecordMeta<'_>| m.shareable)
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].record_id, RecordId(1));
    let _ = view.close();
}

#[test]
fn open_readonly_missing_tenant_errors() {
    let dir = tempfile::tempdir().unwrap();
    let root = MultiTenantRoot::new(dir.path());
    let tid = SkegTenantId::from_bytes([0xff; 16]);
    match root.open_readonly(tid) {
        Ok(_) => panic!("expected NotFound"),
        Err(e) => assert!(matches!(e, OpenError::NotFound)),
    }
}
