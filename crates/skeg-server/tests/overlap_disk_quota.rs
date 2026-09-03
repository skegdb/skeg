//! `max_disk_bytes` is documented as a HARD tenant quota, and every client
//! write now goes through it - but one internal path did not: the boundary
//! replica `SKEG.VINDEX.OVERLAP` writes.
//!
//! A replica is a second physical copy of the row's payload blob, on a second
//! shard's vLog, and nothing ever deletes the first: the two stand side by
//! side for as long as the row's margin keeps it replicated. The shared
//! per-tenant disk counter SEES that copy (a blob key carries the same
//! 16-byte tenant prefix a KV key does), but the write passed
//! `disk_limit: None`, so nothing REFUSED it - and a tenant sitting exactly
//! at its ceiling could be pushed one blob past it per replicated row, by an
//! operation it triggers itself.
//!
//! The fix is not "refuse the overlap". A boundary replica is an
//! optimisation - the row is reachable through its primary either way - so
//! refusing would strand a maintenance run for a tenant whose only offence is
//! being at its own limit. The replica is SKIPPED instead: the tenant's
//! limit rides into the replica's `Vset`, the destination's disk-quota check
//! refuses before anything is staged, and `overlap` treats that one refusal
//! as "this row does not get a replica today", counts it, and carries on.
//!
//! What these tests pin: the tenant never ends above its limit, the run still
//! completes, the primary is untouched, the skipped rows are exactly the ones
//! that would have been replicated, a partially-full budget replicates what
//! fits, nothing is left as a ghost the owner map does not name, the counter
//! moves once per skip, and the counter rebuilt at reopen agrees. Plus the
//! control the ADR asserted in prose and never tested: a reshard MOVE does
//! not duplicate a blob.
//!
//! See `docs/adr-payload-transaction.md`, "Disk quota".

use skeg_server::shard::ShardSet;
use skeg_telemetry::{Counter, counter_value};

/// `skeg_overlap_replicas_skipped_quota_total` is process-wide, and a test
/// binary runs its `#[tokio::test]` functions concurrently by default - so
/// the tests below that assert a DELTA on it would otherwise measure each
/// other's skips. Take turns instead, the same way `ingress_budget.rs` does
/// with its own shared counter.
///
/// Held by every test that TICKS the counter as well as every test that
/// reads it: a test that only skips replicas still moves the number another
/// test is measuring, which is exactly how this file failed once under a
/// full-package run before the two silent ones took their turn too.
static SKIPPED_COUNTER_TESTS: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

const DIM: usize = 16;
/// Big enough that one blob dominates the record and the arithmetic below is
/// about payload bytes rather than key framing.
const BLOB: usize = 4096;
/// Enough rows that a 2-way split leaves a populated boundary.
const N: u64 = 60;

/// Two orthogonal clusters so a semantic reshard splits them across two
/// shards, every row unique within its cluster (a tie would make "which copy
/// came back" unfalsifiable).
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

fn blob(id: u64) -> bytes::Bytes {
    bytes::Bytes::from(vec![(id % 251) as u8; BLOB])
}

/// The registry key of `name` under `tenant`, as the RESP3 handler scopes it.
fn scoped_name(tenant: u128, name: &str) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    for b in tenant.to_le_bytes() {
        let _ = write!(s, "{b:02x}");
    }
    s.push_str("::");
    s.push_str(name);
    s
}

/// A two-shard index of `N` payload-carrying rows for `tenant`, already
/// resharded, so the next step is an overlap. Returns the shard set, the
/// index's registry key, and the tenant's disk bytes after the reshard.
async fn seeded(dir: &std::path::Path, tenant: u128, name: &str) -> (ShardSet, String, u64) {
    let shards = ShardSet::open(dir, 2).expect("shard set opens");
    let scoped = scoped_name(tenant, name);
    shards
        .vindex_create_scoped(&scoped, DIM as u32, 1, 1)
        .await
        .expect("scoped vindex create");
    for id in 0..N {
        shards
            .vset(&scoped, id, vec_for(id), tenant, None, Some(blob(id)))
            .await
            .expect("seed row");
    }
    let moved = shards
        .reshard(&scoped, 0.25, 10, tenant)
        .await
        .expect("reshard");
    assert!(
        moved > 0,
        "fixture: the clusters have to split across shards"
    );
    let bytes = shards.tenant_disk_bytes(tenant);
    (shards, scoped, bytes)
}

/// The physical cost of one row's blob for this tenant, from a fixture whose
/// only vLog content is `N` such blobs. Every blob key here has the same
/// length (fixed-width generation, id and version, one index name), so every
/// record pads to the same size.
fn unit(total: u64) -> u64 {
    assert_eq!(
        total % N,
        0,
        "fixture: {N} identical blobs must divide evenly"
    );
    total / N
}

// ── The invariant ─────────────────────────────────────────────────────────

/// A tenant exactly at `max_disk_bytes` runs an overlap: the run completes,
/// every replica is skipped, the counter says how many, and the tenant's
/// physical bytes did not move.
#[tokio::test]
async fn an_overlap_at_the_tenant_disk_limit_writes_no_replica_and_completes() {
    let _turn = SKIPPED_COUNTER_TESTS.lock().await;
    const T: u128 = 0xB3_01;
    let dir = tempfile::TempDir::new().unwrap();
    let (shards, name, at_limit) = seeded(dir.path(), T, "ov").await;

    let before = counter_value(Counter::OverlapReplicasSkippedQuota);
    let replicated = shards
        .overlap_with_disk_limit(&name, 4.0, T, Some(at_limit))
        .await
        .expect("an overlap must not be stranded by the tenant being at its limit");
    let skipped = counter_value(Counter::OverlapReplicasSkippedQuota) - before;

    assert_eq!(replicated, 0, "no replica fits in zero headroom");
    assert!(
        skipped > 0,
        "the fixture must have boundary rows to skip; the counter moved by {skipped}"
    );
    assert_eq!(
        shards.tenant_disk_bytes(T),
        at_limit,
        "an internal operation must not take a tenant past max_disk_bytes"
    );
}

/// The rows it skipped are exactly the rows it would have replicated with
/// room: same fixture, same tau, one with headroom and one without.
#[tokio::test]
async fn an_overlap_at_the_limit_skips_exactly_the_replicas_it_would_have_written() {
    let _turn = SKIPPED_COUNTER_TESTS.lock().await;
    const T: u128 = 0xB3_02;
    let roomy = tempfile::TempDir::new().unwrap();
    let (with_room, name, base) = seeded(roomy.path(), T, "ov").await;
    let replicated = with_room
        .overlap_with_disk_limit(&name, 4.0, T, Some(base * 4))
        .await
        .expect("overlap with headroom");
    assert!(replicated > 0, "fixture: the boundary must be non-empty");

    const U: u128 = 0xB3_03;
    let tight = tempfile::TempDir::new().unwrap();
    let (at_limit, uname, ubase) = seeded(tight.path(), U, "ov").await;
    let before = counter_value(Counter::OverlapReplicasSkippedQuota);
    let none = at_limit
        .overlap_with_disk_limit(&uname, 4.0, U, Some(ubase))
        .await
        .expect("overlap at the limit");
    let skipped = counter_value(Counter::OverlapReplicasSkippedQuota) - before;

    assert_eq!(none, 0);
    assert_eq!(
        skipped, replicated,
        "the quota must skip the same rows the same overlap replicates with room, \
         not stop the scan early"
    );
}

/// Headroom for exactly one replica: one is written, the rest are skipped,
/// and the tenant lands on its limit rather than past it. The partial case is
/// where an "abort the run on the first refusal" fix would look right and be
/// wrong.
#[tokio::test]
async fn an_overlap_with_room_for_one_replica_writes_one_and_skips_the_rest() {
    let _turn = SKIPPED_COUNTER_TESTS.lock().await;
    const T: u128 = 0xB3_04;
    let dir = tempfile::TempDir::new().unwrap();
    let (shards, name, base) = seeded(dir.path(), T, "ov").await;
    let one = unit(base);

    let before = counter_value(Counter::OverlapReplicasSkippedQuota);
    let replicated = shards
        .overlap_with_disk_limit(&name, 4.0, T, Some(base + one))
        .await
        .expect("overlap with one blob of headroom");
    let skipped = counter_value(Counter::OverlapReplicasSkippedQuota) - before;

    assert_eq!(replicated, 1, "exactly one replica fits");
    assert!(skipped > 0, "and the rest are skipped, not written");
    assert_eq!(
        shards.tenant_disk_bytes(T),
        base + one,
        "the one replica that fit is charged, and nothing else is"
    );
}

/// The primary is untouched by a skip: every row still reads back, with its
/// payload, and still answers a search.
#[tokio::test]
async fn a_skipped_replica_leaves_the_primary_readable() {
    let _turn = SKIPPED_COUNTER_TESTS.lock().await;
    const T: u128 = 0xB3_05;
    let dir = tempfile::TempDir::new().unwrap();
    let (shards, name, at_limit) = seeded(dir.path(), T, "ov").await;
    shards
        .overlap_with_disk_limit(&name, 4.0, T, Some(at_limit))
        .await
        .expect("overlap at the limit");

    for id in 0..N {
        assert_eq!(
            shards.vget(&name, id).await.unwrap(),
            Some(vec_for(id)),
            "row {id} must survive an overlap that could not replicate it"
        );
        let hits = shards
            .vsearch(&name, vec_for(id), 5, 0, T, true, None)
            .await
            .unwrap();
        assert!(
            hits.iter()
                .any(|(hit, _, payload)| *hit == id && payload.as_deref() == Some(&blob(id)[..])),
            "row {id} must still be found, with its payload"
        );
    }
}

/// A skipped replica is a replica that was never written, so there is nothing
/// for the owner map to fail to name: deleting every row leaves nothing
/// searchable and nothing on disk. (A replica written and then not recorded
/// is the ghost `an_overlap_that_fails_after_writing_a_replica_does_not_leave_a_ghost`
/// is about; this is the same property for the refusal path.)
#[tokio::test]
async fn a_skipped_replica_leaves_no_copy_the_owner_map_does_not_name() {
    let _turn = SKIPPED_COUNTER_TESTS.lock().await;
    const T: u128 = 0xB3_06;
    let dir = tempfile::TempDir::new().unwrap();
    let (shards, name, at_limit) = seeded(dir.path(), T, "ov").await;
    shards
        .overlap_with_disk_limit(&name, 4.0, T, Some(at_limit))
        .await
        .expect("overlap at the limit");

    for id in 0..N {
        assert!(shards.vdel(&name, id, T).await.expect("delete"));
    }
    for id in 0..N {
        let hits = shards
            .vsearch(&name, vec_for(id), 5, 0, T, false, None)
            .await
            .unwrap();
        assert!(
            hits.is_empty(),
            "row {id} came back after every row was deleted: a copy nothing names"
        );
    }
    assert_eq!(
        shards.tenant_disk_bytes(T),
        0,
        "every blob the tenant held must be reclaimed by its row's delete"
    );
}

/// The counter is rebuilt from the recovered index at open. After a run that
/// skipped replicas, the rebuilt number must be the same one the refusals
/// were measured against - otherwise the limit binds differently before and
/// after a restart.
#[tokio::test]
async fn the_disk_counter_after_a_skipped_overlap_is_the_same_after_a_reopen() {
    let _turn = SKIPPED_COUNTER_TESTS.lock().await;
    const T: u128 = 0xB3_07;
    let dir = tempfile::TempDir::new().unwrap();
    let live = {
        let (shards, name, at_limit) = seeded(dir.path(), T, "ov").await;
        shards
            .overlap_with_disk_limit(&name, 4.0, T, Some(at_limit))
            .await
            .expect("overlap at the limit");
        let live = shards.tenant_disk_bytes(T);
        assert_eq!(live, at_limit);
        live
    };

    let shards = ShardSet::open(dir.path(), 2).expect("reopen");
    assert_eq!(
        shards.tenant_disk_bytes(T),
        live,
        "the counter rebuilt at open must agree with the one the quota used"
    );

    // And the limit still binds against the rebuilt number, without another
    // restart being what makes it true.
    let name = scoped_name(T, "ov");
    let before = counter_value(Counter::OverlapReplicasSkippedQuota);
    let replicated = shards
        .overlap_with_disk_limit(&name, 4.0, T, Some(live))
        .await
        .expect("overlap after reopen");
    assert_eq!(replicated, 0);
    assert!(counter_value(Counter::OverlapReplicasSkippedQuota) > before);
    assert_eq!(shards.tenant_disk_bytes(T), live);
}

// ── The other side: nothing changes below the limit ───────────────────────

/// A tenant with room replicates exactly as it did before, and every replica
/// is charged for - the quota is the same physical counter, so "written as
/// before" and "counted" are one assertion.
#[tokio::test]
async fn an_overlap_below_the_tenant_disk_limit_writes_and_charges_its_replicas() {
    let _turn = SKIPPED_COUNTER_TESTS.lock().await;
    const T: u128 = 0xB3_08;
    let dir = tempfile::TempDir::new().unwrap();
    let (shards, name, base) = seeded(dir.path(), T, "ov").await;
    let one = unit(base);

    let before = counter_value(Counter::OverlapReplicasSkippedQuota);
    let replicated = shards
        .overlap_with_disk_limit(&name, 4.0, T, Some(base * 4))
        .await
        .expect("overlap with headroom");
    assert!(replicated > 0, "fixture: the boundary must be non-empty");
    assert_eq!(
        counter_value(Counter::OverlapReplicasSkippedQuota),
        before,
        "nothing was skipped, so nothing may be counted as skipped"
    );
    assert_eq!(
        shards.tenant_disk_bytes(T),
        base + replicated * one,
        "each replica is one more physical blob, and the counter sees every one"
    );
}

/// And with no limit at all the path is what it always was: the same number
/// of replicas, the same bytes.
#[tokio::test]
async fn an_overlap_with_no_limit_replicates_exactly_as_before() {
    const T: u128 = 0xB3_09;
    let dir = tempfile::TempDir::new().unwrap();
    let (shards, name, base) = seeded(dir.path(), T, "ov").await;
    let one = unit(base);
    let replicated = shards.overlap(&name, 4.0, T).await.expect("overlap");
    assert!(replicated > 0);
    assert_eq!(shards.tenant_disk_bytes(T), base + replicated * one);
}

// ── The control the ADR claimed in prose ──────────────────────────────────

/// The OTHER internal relocation. A reshard move writes the destination copy
/// and deletes the source, so a tenant's physical bytes are the same on both
/// sides of it - which is why the move keeps no disk limit and the replica
/// now takes one. Asserted here rather than argued in the ADR.
#[tokio::test]
async fn a_reshard_move_does_not_duplicate_a_blob() {
    const T: u128 = 0xB3_0A;
    let dir = tempfile::TempDir::new().unwrap();
    let shards = ShardSet::open(dir.path(), 2).expect("shard set opens");
    let name = scoped_name(T, "mv");
    shards
        .vindex_create_scoped(&name, DIM as u32, 1, 1)
        .await
        .expect("scoped vindex create");
    for id in 0..N {
        shards
            .vset(&name, id, vec_for(id), T, None, Some(blob(id)))
            .await
            .expect("seed row");
    }
    let before = shards.tenant_disk_bytes(T);
    let moved = shards.reshard(&name, 0.25, 10, T).await.expect("reshard");
    assert!(moved > 0, "fixture: the reshard has to move rows");
    assert_eq!(
        shards.tenant_disk_bytes(T),
        before,
        "a move relocates a blob; it must not leave the source copy behind"
    );
}
