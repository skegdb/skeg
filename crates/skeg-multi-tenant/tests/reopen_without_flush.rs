//! A tenant that inserted but never called `flush` has its vectors in the
//! engine's WAL; reopening it must find them. The adapter (`skeg-rigging-skeg`)
//! keys "does this tenant exist" on its metadata sidecar, which only `flush`
//! writes, so a reopen without a prior flush went down the create path and
//! the engine's `create_empty` wiped the index underneath it.

use skeg_multi_tenant::{MultiTenantRoot, SkegTenantId};
use skeg_rigging::prelude::*;

const DIM: u32 = 4;

#[test]
#[ignore = "needs skeg-rigging-skeg >= 0.1.4 (Tenant::open reopens an index without a sidecar); passes with the fix path-patched"]
fn reopen_without_flush_keeps_inserted_vectors() {
    let dir = tempfile::tempdir().unwrap();
    let root = MultiTenantRoot::new(dir.path());
    let id = SkegTenantId::from_bytes([0x11; 16]);
    {
        let tenant = root.open(id, DIM).unwrap();
        tenant
            .insert(RecordId(1), vec![0.0, 0.0, 1.0, 0.0], true, vec![], vec![])
            .unwrap();
        // no flush
    }
    let idx = skeg_vector::DiskVamanaIndex::open(&root.tenant_dir(id)).unwrap();
    assert_eq!(idx.len(), 1, "the WAL-durable insert survives on disk");
    let tenant = root.open(id, DIM).expect("reopen without flush");
    let idx = skeg_vector::DiskVamanaIndex::open(&root.tenant_dir(id)).unwrap();
    assert_eq!(idx.len(), 1, "reopen must not recreate the index");
    drop(tenant);
}
