//! `max_disk_bytes` is documented as a hard tenant quota, and a KV `SET`
//! enforces it - but `stage_payload_blob` wrote a vector's payload straight to
//! the vLog with no limit attached. An authenticated tenant could fill the
//! shared disk through `VSET`/`VMSET` payloads alone, unbounded by the same
//! number that already bounds its `SET`s.
//!
//! `vset_with_disk_limit`/`vmset_with_disk_limit` close that: the tenant's
//! `max_disk_bytes` is threaded into the blob's own `VLog::set`, the same
//! call and the same physical counter (live KV bytes plus live blob bytes,
//! keyed by the 16-byte tenant prefix every scoped key carries) a KV `SET`
//! already goes through. See `docs/adr-payload-transaction.md`, "Disk quota".
//!
//! These tests exercise: plain enforcement, a VMSET item refused without its
//! siblings paying for it, an overwrite whose old blob is released only after
//! the commit, the disk counter's refund at every payload-transaction
//! failpoint window (staged-but-uncommitted and superseded-but-unreclaimed
//! blobs are physical bytes until the next open collects them - the
//! "temporary margin" the ADR names), the counter's rebuild at reopen, and
//! that one tenant at its limit cannot touch another's budget.

use skeg_server::failpoint::{WriteFailpoint, arm_at, disarm_at, fired_at};
use skeg_server::shard::ShardSet;

const DIM: usize = 8;

fn vec_at(id: u64, phase: u64) -> Vec<f32> {
    let mut s = ((id << 3) | phase) | 1;
    let mut v = vec![0f32; DIM];
    for slot in v.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *slot = if s % 2 == 0 { 0.5 } else { -0.5 };
    }
    v[(id % DIM as u64) as usize] = 1.0;
    v
}

fn vec_for(id: u64) -> Vec<f32> {
    vec_at(id, 0)
}

fn open(dir: &std::path::Path) -> ShardSet {
    ShardSet::open(dir, 1).expect("shard set opens")
}

async fn create_scoped(shards: &ShardSet, name: &str) {
    shards
        .vindex_create_scoped(name, DIM as u32, 0, 1)
        .await
        .expect("scoped vindex create");
}

fn blob(n: usize, fill: u8) -> bytes::Bytes {
    bytes::Bytes::from(vec![fill; n])
}

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

// ── Plain enforcement ─────────────────────────────────────────────────────

/// A payload big enough to push the tenant over its `max_disk_bytes` is
/// refused, and nothing it would have written is visible: no vector, no
/// payload, no growth of the counter the refusal was measured against.
#[tokio::test]
#[ignore = "opens in 'server: enforce the tenant disk limit when staging a payload blob'"]
async fn a_vset_whose_payload_would_exceed_the_tenant_disk_limit_is_refused() {
    const T: u128 = 0x111;
    let dir = tempfile::TempDir::new().unwrap();
    let shards = open(dir.path());
    let name = scoped_name(T, "dq1");
    create_scoped(&shards, &name).await;

    // Probe: one small payload costs how many physical bytes for this
    // tenant? Written, measured, deleted - so the real test below starts
    // from a clean, known baseline instead of guessing padding.
    let before = shards.tenant_disk_bytes(T);
    let outcome = shards
        .vset_with_disk_limit(
            &name,
            1,
            vec_for(1),
            T,
            None,
            Some(1), // a one-byte budget: nothing this size can fit
            Some(blob(4096, b'x')),
        )
        .await;

    assert!(
        outcome.is_err(),
        "a payload far over a one-byte disk budget must be refused"
    );
    assert_eq!(
        shards.tenant_disk_bytes(T),
        before,
        "a refused write must not grow the tenant's disk usage"
    );
    assert!(
        shards.vget(&name, 1).await.unwrap().is_none(),
        "the vector of a write refused by the disk quota must not be stored"
    );
}

/// The same limit, applied per tenant: a limit generous enough for the
/// payload admits the write, and the counter grows by exactly its physical
/// cost.
#[tokio::test]
async fn a_vset_within_the_tenant_disk_limit_is_admitted_and_charged() {
    const T: u128 = 0x112;
    let dir = tempfile::TempDir::new().unwrap();
    let shards = open(dir.path());
    let name = scoped_name(T, "dq2");
    create_scoped(&shards, &name).await;

    let before = shards.tenant_disk_bytes(T);
    shards
        .vset_with_disk_limit(
            &name,
            1,
            vec_for(1),
            T,
            None,
            Some(1 << 20),
            Some(blob(256, b'y')),
        )
        .await
        .expect("a payload well under a 1 MiB budget must be admitted");
    assert!(
        shards.tenant_disk_bytes(T) > before,
        "an admitted payload must be charged to the tenant"
    );
}

// ── VMSET: one refusal does not cost its siblings ─────────────────────────

/// A VMSET batch where one item's payload alone would blow the tenant's disk
/// budget must still store every OTHER item: per-item admission, same as
/// quota and dimension checks on this path.
#[tokio::test]
#[ignore = "opens in 'server: enforce the tenant disk limit when staging a payload blob'"]
async fn a_vmset_item_refused_by_disk_quota_does_not_abort_its_siblings() {
    const T: u128 = 0x113;
    let dir = tempfile::TempDir::new().unwrap();
    let shards = open(dir.path());
    let name = scoped_name(T, "dq3");
    create_scoped(&shards, &name).await;

    // Room for a handful of small payloads, not for one large one.
    let items = vec![
        (1u64, vec_for(1), Some(blob(64, 1))),
        (2u64, vec_for(2), Some(blob(1 << 20, 2))), // the oversized one
        (3u64, vec_for(3), Some(blob(64, 3))),
    ];
    let results = shards
        .vmset_with_disk_limit(&name, items, T, None, Some(4096))
        .await;
    assert_eq!(results.len(), 3);
    assert!(
        results[0].is_ok(),
        "item 1 (small) must be stored: {:?}",
        results[0]
    );
    assert!(
        results[1].is_err(),
        "item 2 (oversized) must be refused by the disk quota"
    );
    assert!(
        results[2].is_ok(),
        "item 3 (small) must be stored: {:?}",
        results[2]
    );

    assert!(shards.vget(&name, 1).await.unwrap().is_some());
    assert!(
        shards.vget(&name, 2).await.unwrap().is_none(),
        "the refused item's vector must not be stored either - it never staged"
    );
    assert!(shards.vget(&name, 3).await.unwrap().is_some());
}

// ── Overwrite: the old blob is released after the commit, not before ──────

/// An overwrite stages the NEW payload at a new key while the OLD one is
/// still resident - so for the width of that window the tenant is charged
/// for both. That is the documented physical margin, not a bypass: a limit
/// with room for only one blob refuses the overwrite; a limit with room for
/// both admits it, and once it commits the superseded blob is reclaimed and
/// the counter comes back down to one blob's worth - proving the release
/// happens AFTER the commit, not before it (staging the new blob at the old
/// one's key would have destroyed the old payload on a write that could
/// still fail).
#[tokio::test]
#[ignore = "opens in 'server: enforce the tenant disk limit when staging a payload blob'"]
async fn an_overwrite_needs_room_for_both_blobs_and_releases_the_old_one_after_commit() {
    const T: u128 = 0x114;
    let dir = tempfile::TempDir::new().unwrap();
    let shards = open(dir.path());
    let name = scoped_name(T, "dq4");
    create_scoped(&shards, &name).await;

    shards
        .vset_with_disk_limit(&name, 1, vec_for(1), T, None, None, Some(blob(512, b'a')))
        .await
        .expect("the first, unconstrained write");
    let one_blob = shards.tenant_disk_bytes(T);
    assert!(one_blob > 0);

    // Room for one blob, not two: the overwrite's new key cannot land while
    // the old one is still charged.
    let tight = shards
        .vset_with_disk_limit(
            &name,
            1,
            vec_at(1, 1),
            T,
            None,
            Some(one_blob), // exactly what the FIRST blob alone costs
            Some(blob(512, b'b')),
        )
        .await;
    assert!(
        tight.is_err(),
        "an overwrite needing room for two blobs must be refused by a \
         one-blob budget"
    );
    assert_eq!(
        shards.tenant_disk_bytes(T),
        one_blob,
        "a refused overwrite must leave the original blob's charge exactly \
         where it was"
    );

    // Room for both, transiently: the overwrite is admitted, and after it
    // commits the counter is back to ONE blob's worth, not two - the old key
    // was reclaimed, not left charged forever.
    shards
        .vset_with_disk_limit(
            &name,
            1,
            vec_at(1, 1),
            T,
            None,
            Some(one_blob * 3), // headroom for the transient double write
            Some(blob(512, b'b')),
        )
        .await
        .expect("an overwrite with room for both blobs must be admitted");
    assert_eq!(
        shards.tenant_disk_bytes(T),
        one_blob,
        "the superseded blob must be released after the commit, so the \
         counter returns to one blob's worth"
    );
}

// ── Failpoint windows: the counter is refunded, never left stuck ──────────

/// W1: staging itself refused (not by the quota - an arbitrary I/O-shaped
/// failure). Nothing was written, so nothing is charged.
#[tokio::test]
async fn a_failpoint_at_payload_prepare_leaves_the_disk_counter_untouched() {
    const T: u128 = 0x115;
    let dir = tempfile::TempDir::new().unwrap();
    let shards = open(dir.path());
    let name = scoped_name(T, "dqw1");
    create_scoped(&shards, &name).await;
    let before = shards.tenant_disk_bytes(T);

    arm_at(WriteFailpoint::PayloadPrepare, &name);
    let outcome = shards
        .vset_with_disk_limit(
            &name,
            1,
            vec_for(1),
            T,
            None,
            Some(1 << 20),
            Some(blob(64, 1)),
        )
        .await;
    disarm_at(WriteFailpoint::PayloadPrepare, &name);
    assert!(
        fired_at(WriteFailpoint::PayloadPrepare, &name),
        "the failpoint never fired, so this test proved nothing"
    );
    assert!(outcome.is_err());
    assert_eq!(
        shards.tenant_disk_bytes(T),
        before,
        "a write that never reached the vLog must not move the counter"
    );
}

/// W2: the blob staged, but the WAL commit that would have published it
/// failed. The blob is an orphan - physically present, no live row names it -
/// and stays charged (the documented margin) until the next open's
/// reclamation collects it and the counter comes back to where it was.
#[tokio::test]
async fn a_failpoint_at_vector_commit_leaves_an_orphan_reclaimed_at_the_next_open() {
    const T: u128 = 0x116;
    let dir = tempfile::TempDir::new().unwrap();
    let name = scoped_name(T, "dqw2");
    let before;
    {
        let shards = open(dir.path());
        create_scoped(&shards, &name).await;
        before = shards.tenant_disk_bytes(T);

        arm_at(WriteFailpoint::VectorCommit, &name);
        let outcome = shards
            .vset_with_disk_limit(
                &name,
                1,
                vec_for(1),
                T,
                None,
                Some(1 << 20),
                Some(blob(256, 2)),
            )
            .await;
        disarm_at(WriteFailpoint::VectorCommit, &name);
        assert!(
            fired_at(WriteFailpoint::VectorCommit, &name),
            "the failpoint never fired, so this test proved nothing"
        );
        assert!(outcome.is_err(), "an uncommitted write must be reported");
        assert!(
            shards.tenant_disk_bytes(T) > before,
            "the staged-but-uncommitted blob is a physical byte on disk \
             until the next open reclaims it - that is the margin, not a \
             leak"
        );
    }

    let shards = open(dir.path());
    assert_eq!(
        shards.tenant_disk_bytes(T),
        before,
        "the orphaned blob must be reclaimed at open and the counter must \
         land back exactly where it started"
    );
}

/// W4: the commit landed, but the cleanup that reclaims the SUPERSEDED blob
/// failed. Same shape as W2's orphan, reached from the other side of an
/// overwrite: charged until the next open, then refunded exactly.
#[tokio::test]
async fn a_failpoint_at_post_commit_cleanup_leaves_the_superseded_blob_reclaimed_at_the_next_open()
{
    const T: u128 = 0x117;
    let dir = tempfile::TempDir::new().unwrap();
    let name = scoped_name(T, "dqw4");
    let one_blob;
    {
        let shards = open(dir.path());
        create_scoped(&shards, &name).await;
        shards
            .vset_with_disk_limit(&name, 1, vec_for(1), T, None, None, Some(blob(256, 3)))
            .await
            .expect("the first, unconstrained write");
        one_blob = shards.tenant_disk_bytes(T);

        arm_at(WriteFailpoint::PayloadPostCommitCleanup, &name);
        shards
            .vset_with_disk_limit(&name, 1, vec_at(1, 1), T, None, None, Some(blob(256, 4)))
            .await
            .expect("the commit itself must still succeed - only cleanup failed");
        disarm_at(WriteFailpoint::PayloadPostCommitCleanup, &name);
        assert!(
            fired_at(WriteFailpoint::PayloadPostCommitCleanup, &name),
            "the failpoint never fired, so this test proved nothing"
        );
        assert!(
            shards.tenant_disk_bytes(T) > one_blob,
            "the superseded blob the cleanup failed to reclaim is still a \
             physical byte on disk - both keys are charged until the next \
             open"
        );
    }

    let shards = open(dir.path());
    assert_eq!(
        shards.tenant_disk_bytes(T),
        one_blob,
        "the next open must reclaim the superseded blob and land the \
         counter back at exactly one blob's worth"
    );
}

/// W8: a drop whose blob sweep failed leaves every blob of the dropped index
/// behind, still charged - the next open's reclamation is what finally frees
/// them, because a dropped index's name no longer resolves to any live row.
#[tokio::test]
async fn a_failpoint_at_drop_blob_sweep_leaves_its_blobs_reclaimed_at_the_next_open() {
    const T: u128 = 0x118;
    let dir = tempfile::TempDir::new().unwrap();
    let name = scoped_name(T, "dqw8");
    let before;
    {
        let shards = open(dir.path());
        before = shards.tenant_disk_bytes(T);
        create_scoped(&shards, &name).await;
        shards
            .vset_with_disk_limit(&name, 1, vec_for(1), T, None, None, Some(blob(256, 5)))
            .await
            .expect("the write");
        assert!(shards.tenant_disk_bytes(T) > before);

        arm_at(WriteFailpoint::DropBlobSweep, &name);
        shards
            .vindex_drop(&name, T)
            .await
            .expect("the drop itself must still succeed - only the sweep failed");
        disarm_at(WriteFailpoint::DropBlobSweep, &name);
        assert!(
            fired_at(WriteFailpoint::DropBlobSweep, &name),
            "the failpoint never fired, so this test proved nothing"
        );
        assert!(
            shards.tenant_disk_bytes(T) > before,
            "a drop whose sweep failed leaves its blobs on disk, still \
             charged"
        );
    }

    let shards = open(dir.path());
    assert_eq!(
        shards.tenant_disk_bytes(T),
        before,
        "the next open must reclaim a dropped index's abandoned blobs and \
         land the counter back at its pre-write baseline"
    );
}

// ── Reopen: the counter is rebuilt, not merely preserved in RAM ───────────

/// The disk counter lives in RAM (`SharedTenantDisk`); a real restart starts
/// it empty. What has to reproduce the pre-close number is `VLog::open`'s own
/// recovery scan plus the readiness-barrier reclamation, both of which already
/// run before `ShardSet::open` returns - this pins that they land on the same
/// answer, including a payload blob's bytes.
#[tokio::test]
#[ignore = "opens in 'server: enforce the tenant disk limit when staging a payload blob'"]
async fn the_disk_counter_is_rebuilt_from_payload_blobs_at_reopen() {
    const T: u128 = 0x119;
    let dir = tempfile::TempDir::new().unwrap();
    let name = scoped_name(T, "dqr");
    let before;
    {
        let shards = open(dir.path());
        create_scoped(&shards, &name).await;
        for id in 0..20u64 {
            shards
                .vset_with_disk_limit(
                    &name,
                    id,
                    vec_for(id),
                    T,
                    None,
                    None,
                    Some(blob(128, id as u8)),
                )
                .await
                .expect("an unconstrained write");
        }
        before = shards.tenant_disk_bytes(T);
        assert!(before > 0);
    }

    let shards = open(dir.path());
    assert_eq!(
        shards.tenant_disk_bytes(T),
        before,
        "a reopen must rebuild the same physical total the writes left \
         behind, payload blobs included"
    );
    // And the rebuilt number is not a fossil: a limit set exactly at it
    // refuses the next write.
    let refused = shards
        .vset_with_disk_limit(
            &name,
            999,
            vec_for(999),
            T,
            None,
            Some(before),
            Some(blob(128, 9)),
        )
        .await;
    assert!(
        refused.is_err(),
        "the rebuilt counter must be live enough to enforce a limit \
         against, not just readable"
    );
}

// ── Two tenants, one disk: isolation ───────────────────────────────────────

/// A tenant pinned at its own disk limit must not be able to touch another
/// tenant's budget, and the reverse: neither tenant's admission decision may
/// read the other's usage.
#[tokio::test]
#[ignore = "opens in 'server: enforce the tenant disk limit when staging a payload blob'"]
async fn two_tenants_on_one_disk_a_at_its_limit_cannot_touch_b() {
    const A: u128 = 0xA0A0;
    const B: u128 = 0xB0B0;
    let dir = tempfile::TempDir::new().unwrap();
    let shards = open(dir.path());
    let name_a = scoped_name(A, "dqa");
    let name_b = scoped_name(B, "dqb");
    create_scoped(&shards, &name_a).await;
    create_scoped(&shards, &name_b).await;

    shards
        .vset_with_disk_limit(&name_a, 1, vec_for(1), A, None, None, Some(blob(256, 1)))
        .await
        .expect("A's first write, unconstrained, to learn its cost");
    let a_used = shards.tenant_disk_bytes(A);
    assert!(a_used > 0);
    assert_eq!(shards.tenant_disk_bytes(B), 0, "B must start uncharged");

    // A is now pinned exactly at its own usage: nothing more fits.
    let a_refused = shards
        .vset_with_disk_limit(
            &name_a,
            2,
            vec_for(2),
            A,
            None,
            Some(a_used),
            Some(blob(256, 2)),
        )
        .await;
    assert!(a_refused.is_err(), "A must be refused at its own limit");

    // B, meanwhile, has no limit of its own and is untouched by A's.
    shards
        .vset_with_disk_limit(&name_b, 1, vec_for(1), B, None, None, Some(blob(4096, 3)))
        .await
        .expect("B has its own budget and A's limit must not apply to it");
    assert!(
        shards.tenant_disk_bytes(B) > 0,
        "B's write must have been charged to B"
    );
    assert_eq!(
        shards.tenant_disk_bytes(A),
        a_used,
        "B's write must not have moved A's counter"
    );
}
