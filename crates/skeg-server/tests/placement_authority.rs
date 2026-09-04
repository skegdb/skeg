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
    // The copy the map names is the copy a search returns: the write moved the
    // row, and nothing here may still prefer the one its own cleanup deleted.
    let hits = shards
        .vsearch(&name, other_cluster(moved_row), 5, 64, T, false, None)
        .await
        .unwrap();
    assert_eq!(
        hits.first().map(|(id, _, _)| *id),
        Some(moved_row),
        "the new copy must be the one search returns"
    );
    assert_eq!(shards.tenant_vector_count(T), N, "an overwrite is free");
    assert!(
        shards.check(&name).await.unwrap().is_empty(),
        "the owner map must be internally consistent after the publish"
    );
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

/// SD4. A DROP that lands during a reshard must credit the tenant every
/// LOGICAL row, not the ones the reshard loop happened to have moved.
///
/// `vindex_drop` counts rows as `owners[name].len()`. A first reshard used to
/// publish one entry per row it moved, into a map nothing had built - so the
/// count was the migration's progress, and the tenant stayed charged for the
/// rest. Under the authority the map is built before the first row moves and
/// the count cannot be read halfway through a publish.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_drop_during_a_reshard_credits_every_logical_row() {
    let dir = tempfile::TempDir::new().unwrap();
    const N: u64 = 200;
    let name = scoped(T, "pa-drop");
    let shards = open(dir.path());
    // NOT `routed_index`: this one must be the FIRST reshard the index ever
    // sees, which is the run with no owner map to start from.
    shards
        .vindex_create_scoped(&name, DIM as u32, 4, 1)
        .await
        .unwrap();
    for id in 0..N {
        shards
            .vset(&name, id, vec_for(id), T, LIMIT, None)
            .await
            .unwrap();
    }
    assert_eq!(shards.tenant_vector_count(T), N);

    arm_gate_at(FP, &name);
    let resharder = {
        let (s, n) = (shards.clone(), name.clone());
        tokio::spawn(async move { s.reshard(&n, 0.25, 10, T).await })
    };
    wait_reached(FP, &name).await;

    let dropper = {
        let (s, n) = (shards.clone(), name.clone());
        tokio::spawn(async move { s.vindex_drop(&n, T).await })
    };
    tokio::time::sleep(PENDING).await;
    let counted_during_the_publish = dropper.is_finished();

    release_gate_at(FP, &name);
    dropper.await.unwrap().expect("vindex_drop");
    // The reshard is writing to an index that is being dropped underneath it,
    // so either answer is legitimate; what is not is the accounting.
    let _ = resharder.await.unwrap();
    assert!(
        fired_gate_at(FP, &name),
        "the gate never fired: this test proved nothing"
    );
    assert_eq!(
        shards.tenant_vector_count(T),
        0,
        "the drop credited fewer rows than the index held: the count came off a \
         map the reshard had only partly built"
    );
    assert!(
        !counted_during_the_publish,
        "the drop counted the index while its placement was being published"
    );

    drop(shards);
    let shards = open(dir.path());
    assert_eq!(shards.tenant_vector_count(T), 0, "and it stays credited");
}

/// What the maintenance loop under test is.
#[derive(Clone, Copy)]
enum Maintenance {
    /// Moves rows AND republishes the whole map at the end.
    Reshard,
    /// Adds replica slots to existing entries.
    Overlap,
}

/// The state one id must be in when the dust settles, decided by the LAST
/// operation the client got an ack for.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Expect {
    Untouched,
    Rewritten,
    Deleted,
}

/// One run of the oracle: every id gets at most one point op, concurrently with
/// a maintenance pass, and afterwards every id must hold exactly what its last
/// acknowledged op said - in RAM and again after a reopen.
async fn last_acked_op_wins(kind: Maintenance, run: usize) {
    const N: u64 = 300;
    let dir = tempfile::TempDir::new().unwrap();
    let tag = match kind {
        Maintenance::Reshard => "reshard",
        Maintenance::Overlap => "overlap",
    };
    let name = scoped(T, &format!("pa-oracle-{tag}-{run}"));
    let shards = open(dir.path());
    match kind {
        // The concurrent pass is this index's FIRST reshard, so it really
        // MOVES rows while the point ops run. A second reshard of an index
        // already in place moves nothing and races nothing.
        Maintenance::Reshard => {
            shards
                .vindex_create_scoped(&name, DIM as u32, 4, 1)
                .await
                .unwrap();
            for id in 0..N {
                shards
                    .vset(&name, id, vec_for(id), T, LIMIT, None)
                    .await
                    .unwrap();
            }
        }
        // `overlap` needs a router, so this one starts from a resharded index.
        Maintenance::Overlap => routed_index(&shards, &name, N).await,
    }

    // One op per id, fixed up front so the oracle is a function of the id and
    // not of who won a race.
    let plan: Vec<(u64, Expect)> = (0..N)
        .map(|id| {
            (
                id,
                match id % 3 {
                    0 => Expect::Rewritten,
                    1 => Expect::Deleted,
                    _ => Expect::Untouched,
                },
            )
        })
        .collect();

    let maintenance = {
        let (s, n) = (shards.clone(), name.clone());
        tokio::spawn(async move {
            match kind {
                Maintenance::Reshard => s.reshard(&n, 0.25, 10, T).await,
                Maintenance::Overlap => s.overlap(&n, 4.0, T).await,
            }
        })
    };
    let ops = {
        let (s, n, plan) = (shards.clone(), name.clone(), plan.clone());
        tokio::spawn(async move {
            for (id, want) in plan {
                match want {
                    Expect::Rewritten => s
                        .vset(&n, id, other_cluster(id), T, LIMIT, None)
                        .await
                        .unwrap(),
                    // A live row. `false` here IS the B0 defect: the delete
                    // addressed a shard the row had already left.
                    Expect::Deleted => assert!(
                        s.vdel(&n, id, T).await.unwrap(),
                        "VDEL of a live id {id} answered false"
                    ),
                    Expect::Untouched => {}
                }
            }
        })
    };
    ops.await.unwrap();
    let touched = maintenance.await.unwrap().expect("maintenance pass");
    assert!(
        touched > 0,
        "the maintenance pass moved/replicated nothing: this run raced nothing"
    );

    let verify = async |shards: &ShardSet, when: &str| {
        for &(id, want) in &plan {
            let got = shards.vget(&name, id).await.unwrap();
            match want {
                Expect::Untouched => assert_eq!(
                    got.as_deref(),
                    Some(vec_for(id).as_slice()),
                    "{when}: id {id} was not touched and must be unchanged"
                ),
                Expect::Rewritten => assert_eq!(
                    got.as_deref(),
                    Some(other_cluster(id).as_slice()),
                    "{when}: id {id} must hold the value its acknowledged VSET wrote"
                ),
                Expect::Deleted => assert_eq!(
                    got, None,
                    "{when}: id {id} was deleted and acknowledged, and came back"
                ),
            }
        }
        // A deleted row must not be findable either: a copy the map stopped
        // naming is still reachable by search.
        for &(id, want) in &plan {
            if want != Expect::Deleted {
                continue;
            }
            let hits = shards
                .vsearch(&name, vec_for(id), 5, 64, T, false, None)
                .await
                .unwrap();
            assert!(
                !hits.iter().any(|(h, _, _)| *h == id),
                "{when}: deleted id {id} is still searchable"
            );
        }
        assert!(
            shards.check(&name).await.unwrap().is_empty(),
            "{when}: the index reports its own owner map as inconsistent"
        );
    };

    verify(&shards, "in RAM").await;
    let live = plan.iter().filter(|(_, w)| *w != Expect::Deleted).count() as u64;
    assert_eq!(shards.tenant_vector_count(T), live);

    drop(shards);
    let shards = open(dir.path());
    verify(&shards, "after a reopen").await;
    assert_eq!(shards.tenant_vector_count(T), live);
}

/// SD1 + SD2, on every id of an index rather than on one chosen row: a reshard
/// running underneath a full round of point ops must leave each id holding what
/// its own last acknowledged op said. Five runs - this one is a race, not a
/// gate, so a single green run says very little.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reshard_and_concurrent_point_ops_agree_with_the_last_acked_op() {
    for run in 0..5 {
        last_acked_op_wins(Maintenance::Reshard, run).await;
    }
}

/// The same oracle against `overlap`, which writes the replica slot of an entry
/// a concurrent VDEL may be removing, plus `check()` on the way out.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn overlap_and_concurrent_point_ops_agree_with_the_last_acked_op() {
    for run in 0..5 {
        last_acked_op_wins(Maintenance::Overlap, run).await;
    }
}

/// SD5. The authority is per SCOPED NAME, so freezing one tenant's placement
/// must leave another tenant's point ops running. A global lock passes every
/// other test in this file and fails this one.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_tenants_do_not_block_each_other_across_a_reshard() {
    const A: u128 = 11;
    const B: u128 = 22;
    const N: u64 = 120;
    let dir = tempfile::TempDir::new().unwrap();
    let shards = open(dir.path());
    // The same raw name under two tenants: two scoped keys, two authorities.
    let (a, b) = (scoped(A, "pa-iso"), scoped(B, "pa-iso"));
    for (name, tenant) in [(&a, A), (&b, B)] {
        shards
            .vindex_create_scoped(name, DIM as u32, 4, 1)
            .await
            .unwrap();
        for id in 0..N {
            shards
                .vset(name, id, vec_for(id), tenant, LIMIT, None)
                .await
                .unwrap();
        }
        assert!(shards.reshard(name, 0.25, 10, tenant).await.unwrap() > 0);
    }

    arm_gate_at(FP, &a);
    let frozen = {
        let (s, n) = (shards.clone(), a.clone());
        tokio::spawn(async move { s.reshard(&n, 0.25, 10, A).await })
    };
    wait_reached(FP, &a).await;

    // Tenant A's placement is now held exclusive and parked. Tenant B must get
    // through a full round of point ops regardless.
    let work = async {
        for id in 0..25 {
            shards
                .vset(&b, id, other_cluster(id), B, LIMIT, None)
                .await
                .unwrap();
        }
        for id in 25..50 {
            assert!(shards.vdel(&b, id, B).await.unwrap());
        }
        shards.vget(&b, 0).await.unwrap()
    };
    let got = tokio::time::timeout(Duration::from_secs(10), work)
        .await
        .expect(
            "tenant B's point ops blocked behind tenant A's frozen reshard: the \
             placement authority is not per index",
        );
    assert_eq!(got.as_deref(), Some(other_cluster(0).as_slice()));
    assert!(
        !frozen.is_finished(),
        "tenant A's reshard was supposed to still be parked"
    );

    release_gate_at(FP, &a);
    frozen.await.unwrap().expect("reshard");
    assert!(
        fired_gate_at(FP, &a),
        "the gate never fired: this test proved nothing"
    );
    // And A came through its own reshard untouched by any of that.
    for id in 0..N {
        assert_eq!(
            shards.vget(&a, id).await.unwrap().as_deref(),
            Some(vec_for(id).as_slice()),
            "tenant A lost id {id} across its own reshard"
        );
    }
    assert_eq!(shards.tenant_vector_count(A), N);
    assert_eq!(shards.tenant_vector_count(B), N - 25);
}
