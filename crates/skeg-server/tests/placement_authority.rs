//! One placement authority per index: a point op's decision and its commit
//! must not straddle an owner-map publish.
//!
//! `rebuild_owner_maps` scans every shard and then REPLACES an index's whole
//! owner map. Point ops read that map to decide which shard holds a row, and
//! write it back to record where they put one. Nothing used to order the two,
//! so:
//!
//!   - a VDEL that removed the row AND its map entry had the entry restored by
//!     a publish whose scan predated it - a row the tenant no longer has, still
//!     named by the map, so the next write of that id is billed as an overwrite
//!     and the tenant's count stays one short for ever;
//!   - a VSET that committed on a new owner and published the entry had that
//!     entry overwritten by the pre-scan copy, which points at the shard the
//!     VSET's own cleanup just emptied: an acknowledged write nobody can read.
//!
//! Audit 18 reproduced both by racing a reshard against point ops, in 30-70% of
//! runs. The window is microseconds wide, so the tests here do not race it:
//! `PlacementFailpoint::OwnerMapPublish` PARKS the publish inside the exclusive
//! section, immediately before the wholesale insert, and the test decides what
//! runs during it.
//!
//! SD1 (an acked VSET is never lost), SD2 (a VDEL never reports `false` for a
//! row that stays live), SD4 (a DROP during a reshard credits every logical
//! row) and SD5 (one tenant's reshard does not block another's point ops) are
//! stated in `docs/adr-placement-authority.md`; each test below names the one
//! it stands for.

use skeg_server::failpoint::{
    PlacementFailpoint, arm_gate_at, fired_gate_at, release_gate_at, wait_reached,
};
use skeg_server::shard::ShardSet;
use skeg_vector::QuantKind;
use std::time::Duration;

const TIER: QuantKind = QuantKind::TurboQuant { bits: 2 };
const DIM: usize = 16;
const FP: PlacementFailpoint = PlacementFailpoint::OwnerMapPublish;
/// Long enough that a point op which is NOT waiting for the authority has
/// finished several times over, short enough to keep the suite quick.
const PENDING: Duration = Duration::from_millis(200);
/// The tenant every test writes as. NOT zero: `TenantVectorQuota` only tracks
/// tenants that write under a limit, and the open-time rebuild attributes rows
/// by the `<32 hex>::` prefix of the index name - so the accounting assertions
/// need a real tenant and a scoped name.
const T: u128 = 7;
/// Far above anything here: the limit is what turns the counter on, not what
/// is being tested.
const LIMIT: Option<u64> = Some(1_000_000);

/// Two orthogonal clusters, so a semantic reshard splits them across shards,
/// with every row unique inside its cluster (same construction as
/// `tests/semantic_overwrite.rs`: identical vectors make a hundred rows tie at
/// 1.0, and an assertion about which copy came back cannot fail when it
/// should).
fn cluster_vec(id: u64, cluster: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    v[cluster] = 1.0;
    let mut s = (id << 1) | 1;
    for slot in v.iter_mut().skip(2) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *slot = if s % 2 == 0 { 0.5 } else { -0.5 };
    }
    v
}

fn vec_for(id: u64) -> Vec<f32> {
    cluster_vec(id, (id % 2) as usize)
}

/// The same row's fingerprint, in the OTHER cluster - so a routed VSET of it
/// moves the row to the other shard.
fn other_cluster(id: u64) -> Vec<f32> {
    cluster_vec(id, ((id + 1) % 2) as usize)
}

/// The scoped key the server builds for a tenant's index, mirroring
/// `shard.rs::scope_key`: 32 hex digits of `to_le_bytes`, then `::`.
fn scoped(tenant: u128, index: &str) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(32 + 2 + index.len());
    for b in tenant.to_le_bytes() {
        let _ = write!(s, "{b:02x}");
    }
    s.push_str("::");
    s.push_str(index);
    s
}

fn open(dir: &std::path::Path) -> ShardSet {
    ShardSet::open_mode_with_workers(dir, 2, false, TIER, 1).unwrap()
}

/// A routed index with `n` rows on 2 shards, already resharded once - so its
/// owner map exists and names every row, which is the state a publish has to
/// replace atomically.
async fn routed_index(shards: &ShardSet, name: &str, n: u64) {
    shards
        .vindex_create_scoped(name, DIM as u32, 4, 1)
        .await
        .unwrap();
    for id in 0..n {
        shards
            .vset(name, id, vec_for(id), T, LIMIT, None)
            .await
            .unwrap();
    }
    let moved = shards.reshard(name, 0.25, 10, T).await.unwrap();
    assert!(
        moved > 0,
        "the two clusters must actually split across shards"
    );
}

/// SD2. A VDEL of a LIVE row, issued while the owner map of its index is being
/// published, must wait for the authority and then really remove the row -
/// including its map entry, which the publish must not bring back.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_vdel_of_a_live_id_during_the_owner_map_publish_returns_true() {
    let dir = tempfile::TempDir::new().unwrap();
    const N: u64 = 200;
    let name = scoped(T, "pa-vdel");
    let shards = open(dir.path());
    routed_index(&shards, &name, N).await;
    assert_eq!(shards.tenant_vector_count(T), N);

    let victim = 7u64;
    assert!(shards.vget(&name, victim).await.unwrap().is_some());

    arm_gate_at(FP, &name);
    let publisher = {
        let (s, n) = (shards.clone(), name.clone());
        tokio::spawn(async move { s.reshard(&n, 0.25, 10, T).await })
    };
    wait_reached(FP, &name).await;

    let deleter = {
        let (s, n) = (shards.clone(), name.clone());
        tokio::spawn(async move { s.vdel(&n, victim, T).await })
    };
    tokio::time::sleep(PENDING).await;
    // Checked at the END: what a reader needs first is what the publish did to
    // the delete, not the mechanism.
    let decided_during_the_publish = deleter.is_finished();

    release_gate_at(FP, &name);
    let existed = deleter.await.unwrap().expect("vdel");
    publisher.await.unwrap().expect("reshard");
    assert!(
        fired_gate_at(FP, &name),
        "the gate never fired: this test proved nothing"
    );

    assert!(existed, "a live row's delete must report true");
    assert!(
        shards.vget(&name, victim).await.unwrap().is_none(),
        "the row is gone from the point read"
    );
    let hits = shards
        .vsearch(&name, vec_for(victim), 10, 64, T, false, None)
        .await
        .unwrap();
    assert!(
        !hits.iter().any(|(id, _, _)| *id == victim),
        "the deleted row is still searchable"
    );
    assert_eq!(shards.tenant_vector_count(T), N - 1);

    // The row LEFT. Writing it again is an INSERT. When the publish restores
    // the map entry the delete removed, the write is billed as an overwrite
    // instead, and the tenant stays one row short of what it holds - for ever.
    shards
        .vset(&name, victim, vec_for(victim), T, LIMIT, None)
        .await
        .unwrap();
    assert_eq!(
        shards.tenant_vector_count(T),
        N,
        "the owner map still named a row the VDEL had removed, so rewriting it \
         was billed as an overwrite"
    );
    assert!(
        !decided_during_the_publish,
        "the VDEL decided where the row lives while its placement was being \
         replaced: there is more than one authority"
    );

    drop(shards);
    let shards = open(dir.path());
    assert!(shards.vget(&name, victim).await.unwrap().is_some());
    assert_eq!(shards.tenant_vector_count(T), N);
}

/// SD1. A VSET acknowledged while the owner map is being published must be
/// readable afterwards: the publish must not restore a placement that points at
/// the shard the VSET's own cleanup emptied.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_vset_during_the_owner_map_publish_survives_the_publish() {
    let dir = tempfile::TempDir::new().unwrap();
    const N: u64 = 200;
    let name = scoped(T, "pa-vset");
    let shards = open(dir.path());
    routed_index(&shards, &name, N).await;

    let moved_row = 7u64;
    let before = shards.owners_of(&name, &[moved_row]).await.unwrap()[0].0;

    arm_gate_at(FP, &name);
    let publisher = {
        let (s, n) = (shards.clone(), name.clone());
        tokio::spawn(async move { s.reshard(&n, 0.25, 10, T).await })
    };
    wait_reached(FP, &name).await;

    let writer = {
        let (s, n) = (shards.clone(), name.clone());
        tokio::spawn(async move {
            s.vset(&n, moved_row, other_cluster(moved_row), T, LIMIT, None)
                .await
        })
    };
    tokio::time::sleep(PENDING).await;
    let decided_during_the_publish = writer.is_finished();

    release_gate_at(FP, &name);
    writer.await.unwrap().expect("vset");
    publisher.await.unwrap().expect("reshard");
    assert!(
        fired_gate_at(FP, &name),
        "the gate never fired: this test proved nothing"
    );

    let got = shards
        .vget(&name, moved_row)
        .await
        .unwrap()
        .expect("an acknowledged VSET must not be lost by a placement publish");
    assert_eq!(
        got,
        other_cluster(moved_row),
        "and it must be the NEW value"
    );
    let after = shards.owners_of(&name, &[moved_row]).await.unwrap()[0].0;
    assert_ne!(
        after, before,
        "the row moved to the other cluster's shard; the map must say so"
    );
    assert_eq!(shards.tenant_vector_count(T), N, "an overwrite is free");
    assert!(
        !decided_during_the_publish,
        "the VSET chose and published a placement while the map was being \
         replaced: there is more than one authority"
    );

    drop(shards);
    let shards = open(dir.path());
    assert_eq!(
        shards.vget(&name, moved_row).await.unwrap().as_deref(),
        Some(other_cluster(moved_row).as_slice())
    );
}
