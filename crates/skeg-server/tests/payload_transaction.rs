//! A vector and its payload blob are one write, or they are a split brain.
//!
//! `VSET name id vector payload` today lands in three places, in this order:
//!
//!     1. the engine's delta + WAL append      (the vector becomes visible)
//!     2. the payload postings, under the lock (the filter can see it)
//!     3. the blob in the KV vLog, after it    (the payload becomes readable)
//!
//! Every step after 1 can fail on its own, and each failure is reported to the
//! client as a failed write while the vector it wrote stays visible. A search
//! then returns a row the client was told does not exist, with the payload of
//! the value it replaced - or with none at all.
//!
//! The fix is an ORDER, not a lock: stage the blob where nothing can reach it,
//! then make the WAL record that names it the single commit point, then clean
//! up what it superseded WITHOUT being able to fail the call. These tests pin
//! the windows that order creates, one test per window, plus the reopen that
//! has to agree with whatever the client was told.
//!
//! Windows, as the failpoints name them:
//!
//! | window | failpoint                     | after it, the client is told |
//! |--------|-------------------------------|------------------------------|
//! | W1     | `PayloadPrepare`              | failed, and nothing is there |
//! | W2     | `VectorCommit`                | failed, and nothing is there |
//! | W3     | `PayloadApply`                | SUCCEEDED (post-commit)      |
//! | W4     | `PayloadPostCommitCleanup`    | SUCCEEDED (post-commit)      |
//! | W6     | `OverwriteOldCopyDelete`      | SUCCEEDED (already pinned)   |
//! | W8     | `DropBlobSweep`               | the drop succeeded           |

use bytes::Bytes;
use skeg_server::failpoint::{WriteFailpoint, arm_at, disarm_at, fired_at};
use skeg_server::shard::ShardSet;

const DIM: usize = 8;

/// Unique per id, and far enough apart that a search for one row never
/// mistakes its neighbour for it.
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

/// One shard: the hash-placed path, where an id maps to exactly one shard for
/// ever. The subject here is the vector/payload pair, not placement, and a
/// second shard would only add a router to reason about.
fn open(dir: &std::path::Path) -> ShardSet {
    ShardSet::open(dir, 1).expect("shard set opens")
}

async fn create(shards: &ShardSet, name: &str) {
    shards
        .vindex_create(name, DIM as u32, 0, 1)
        .await
        .expect("vindex create");
}

/// The payload a search reports for `id`, or `None`.
///
/// Read back through the SEARCH path rather than the KV key, deliberately: the
/// key is exactly what these commits change, and a test that computed it would
/// be checking the implementation against itself.
async fn payload_of(shards: &ShardSet, name: &str, id: u64) -> Option<Bytes> {
    let hits = shards
        .vsearch(name, vec_for(id), 64, 0, 0, true, None)
        .await
        .expect("vsearch");
    hits.into_iter().find(|h| h.0 == id).and_then(|h| h.2)
}

async fn vector_of(shards: &ShardSet, name: &str, id: u64) -> Option<Vec<f32>> {
    shards.vget(name, id).await.expect("vget")
}

/// What one id looks like from outside: its vector and its payload.
type Row = (Option<Vec<f32>>, Option<Bytes>);

fn blob(text: &str) -> Bytes {
    Bytes::copy_from_slice(text.as_bytes())
}

// ── W1: the blob is staged before anything is published ──────────────────────

/// A staged blob is not a write. If staging fails, the client is told the
/// write failed - and there must be no vector, no payload and no posting to
/// contradict that.
#[tokio::test]
async fn vset_error_at_payload_prepare_leaves_nothing_visible() {
    let dir = tempfile::TempDir::new().unwrap();
    let shards = open(dir.path());
    create(&shards, "w1").await;

    arm_at(WriteFailpoint::PayloadPrepare, "w1");
    let outcome = shards
        .vset("w1", 1, vec_for(1), 0, None, Some(blob("colour=red")))
        .await;
    disarm_at(WriteFailpoint::PayloadPrepare, "w1");
    assert!(
        fired_at(WriteFailpoint::PayloadPrepare, "w1"),
        "the failpoint never fired, so this test proved nothing"
    );

    assert!(
        outcome.is_err(),
        "a write whose payload could not be staged has not happened"
    );
    assert_eq!(
        vector_of(&shards, "w1", 1).await,
        None,
        "the vector of a refused write is visible"
    );
    assert_eq!(
        payload_of(&shards, "w1", 1).await,
        None,
        "the payload of a refused write is visible"
    );
}

// ── W2: the WAL record is the commit point ───────────────────────────────────

/// The append that publishes the vector is the commit point. If IT fails,
/// nothing is committed - the staged blob is unreachable because no row names
/// it, and the client is told the truth.
#[tokio::test]
async fn vset_error_at_wal_commit_leaves_nothing_visible() {
    let dir = tempfile::TempDir::new().unwrap();
    let shards = open(dir.path());
    create(&shards, "w2").await;

    arm_at(WriteFailpoint::VectorCommit, "w2");
    let outcome = shards
        .vset("w2", 1, vec_for(1), 0, None, Some(blob("colour=red")))
        .await;
    disarm_at(WriteFailpoint::VectorCommit, "w2");
    assert!(
        fired_at(WriteFailpoint::VectorCommit, "w2"),
        "the failpoint never fired, so this test proved nothing"
    );

    assert!(outcome.is_err(), "an uncommitted write must be reported");
    assert_eq!(vector_of(&shards, "w2", 1).await, None);
    assert_eq!(
        payload_of(&shards, "w2", 1).await,
        None,
        "a staged blob no row names must not be readable"
    );
}

// ── W3 / W4: after the commit point, nothing may fail the call ───────────────

/// Everything after the commit point is cleanup. Failing it reports a write
/// that succeeded as failed, which is the same lie as the opposite one: the
/// client retries, or gives up, on a row that is already durable.
#[tokio::test]
async fn vset_error_after_commit_is_not_reported_as_failure() {
    let dir = tempfile::TempDir::new().unwrap();
    let shards = open(dir.path());
    create(&shards, "w4").await;
    shards
        .vset("w4", 1, vec_for(1), 0, None, Some(blob("colour=red")))
        .await
        .expect("the first write is ordinary");

    // The overwrite is what leaves a superseded blob for the cleanup to
    // reclaim, so this is the write whose post-commit half can fail.
    arm_at(WriteFailpoint::PayloadPostCommitCleanup, "w4");
    let outcome = shards
        .vset("w4", 1, vec_at(1, 1), 0, None, Some(blob("colour=blue")))
        .await;
    disarm_at(WriteFailpoint::PayloadPostCommitCleanup, "w4");
    assert!(
        fired_at(WriteFailpoint::PayloadPostCommitCleanup, "w4"),
        "the failpoint never fired, so this test proved nothing"
    );

    assert!(
        outcome.is_ok(),
        "a committed write reported as failed: {outcome:?}"
    );
    assert_eq!(vector_of(&shards, "w4", 1).await, Some(vec_at(1, 1)));
    assert_eq!(
        payload_of(&shards, "w4", 1).await,
        Some(blob("colour=blue")),
        "the committed payload must be the one that is readable"
    );
}

/// The same rule at the other post-commit site: the payload postings.
#[tokio::test]
async fn vset_error_at_payload_apply_is_not_reported_as_failure() {
    let dir = tempfile::TempDir::new().unwrap();
    let shards = open(dir.path());
    create(&shards, "w3").await;

    arm_at(WriteFailpoint::PayloadApply, "w3");
    let outcome = shards
        .vset("w3", 1, vec_for(1), 0, None, Some(blob("colour=red")))
        .await;
    disarm_at(WriteFailpoint::PayloadApply, "w3");
    assert!(
        fired_at(WriteFailpoint::PayloadApply, "w3"),
        "the failpoint never fired, so this test proved nothing"
    );

    assert!(
        outcome.is_ok(),
        "a committed write reported as failed: {outcome:?}"
    );
    assert_eq!(vector_of(&shards, "w3", 1).await, Some(vec_for(1)));
    assert_eq!(payload_of(&shards, "w3", 1).await, Some(blob("colour=red")));
}

// ── The overwrite that fails must not destroy what it was replacing ──────────

/// Staging the new blob at the key the OLD one occupies is the obvious
/// implementation and the wrong one: a write that then fails has already
/// destroyed the payload it was overwriting, and the client is told nothing
/// happened.
#[tokio::test]
async fn vset_overwrite_that_fails_keeps_the_previous_payload() {
    let dir = tempfile::TempDir::new().unwrap();
    let shards = open(dir.path());
    create(&shards, "ov").await;
    shards
        .vset("ov", 1, vec_for(1), 0, None, Some(blob("colour=red")))
        .await
        .expect("the committed write");

    arm_at(WriteFailpoint::VectorCommit, "ov");
    let outcome = shards
        .vset("ov", 1, vec_at(1, 1), 0, None, Some(blob("colour=blue")))
        .await;
    disarm_at(WriteFailpoint::VectorCommit, "ov");
    assert!(
        fired_at(WriteFailpoint::VectorCommit, "ov"),
        "the failpoint never fired, so this test proved nothing"
    );

    assert!(outcome.is_err(), "the overwrite must be reported as failed");
    assert_eq!(
        vector_of(&shards, "ov", 1).await,
        Some(vec_for(1)),
        "the committed vector was destroyed by a write that failed"
    );
    assert_eq!(
        payload_of(&shards, "ov", 1).await,
        Some(blob("colour=red")),
        "the committed payload was destroyed by a write that failed"
    );
}

// ── The headline: a kill in any window reopens on ONE logical state ──────────

/// Whatever window the process dies in, the store must come back holding a
/// vector and a payload that belong to the SAME write - and the pair must
/// agree with what the client was told.
///
/// Table-driven over W1..W4 because the windows are the point: a fix that
/// closes one of them and opens another passes any single-window test.
#[tokio::test]
async fn reopen_after_kill_in_every_window_has_one_logical_state() {
    for (label, fp) in [
        ("W1 prepare", WriteFailpoint::PayloadPrepare),
        ("W2 commit", WriteFailpoint::VectorCommit),
        ("W3 apply", WriteFailpoint::PayloadApply),
        ("W4 cleanup", WriteFailpoint::PayloadPostCommitCleanup),
    ] {
        let dir = tempfile::TempDir::new().unwrap();
        let name = "kw";
        {
            let shards = open(dir.path());
            create(&shards, name).await;
            shards
                .vset(name, 1, vec_for(1), 0, None, Some(blob("colour=red")))
                .await
                .expect("the committed write");

            arm_at(fp, name);
            let outcome = shards
                .vset(name, 1, vec_at(1, 1), 0, None, Some(blob("colour=blue")))
                .await;
            disarm_at(fp, name);
            assert!(
                fired_at(fp, name),
                "{label}: the failpoint never fired, so this case proved nothing"
            );
            let committed = outcome.is_ok();
            // The process dies here: no consolidate, no flush, no clean drop
            // of the vindex - only whatever reached the WAL and the vLog.
            drop(shards);

            let shards = open(dir.path());
            let v = vector_of(&shards, name, 1).await;
            let p = payload_of(&shards, name, 1).await;
            let old = (Some(vec_for(1)), Some(blob("colour=red")));
            let new = (Some(vec_at(1, 1)), Some(blob("colour=blue")));
            assert!(
                (v.clone(), p.clone()) == old || (v.clone(), p.clone()) == new,
                "{label}: reopened on a mixed state: vector {v:?}, payload {p:?}"
            );
            if committed {
                assert_eq!(
                    (v, p),
                    new,
                    "{label}: the write was acknowledged and did not survive"
                );
            } else {
                assert_eq!(
                    (v, p),
                    old,
                    "{label}: the write was refused and happened anyway"
                );
            }
        }
    }
}

/// Replaying the same WAL twice must reach the same place. Idempotence is what
/// makes a crash recoverable at all; a payload reference in the record is one
/// more thing it has to hold for.
#[tokio::test]
async fn replay_is_idempotent() {
    let dir = tempfile::TempDir::new().unwrap();
    const N: u64 = 24;
    {
        let shards = open(dir.path());
        create(&shards, "rp").await;
        for id in 0..N {
            shards
                .vset(
                    "rp",
                    id,
                    vec_for(id),
                    0,
                    None,
                    Some(blob(&format!("i={id}"))),
                )
                .await
                .unwrap();
        }
    }
    let mut previous: Option<Vec<Row>> = None;
    for round in 0..3 {
        let shards = open(dir.path());
        let mut state = Vec::new();
        for id in 0..N {
            state.push((
                vector_of(&shards, "rp", id).await,
                payload_of(&shards, "rp", id).await,
            ));
        }
        for (id, (v, p)) in state.iter().enumerate() {
            assert_eq!(v.as_deref(), Some(&vec_for(id as u64)[..]), "round {round}");
            assert_eq!(
                p.as_ref(),
                Some(&blob(&format!("i={id}"))),
                "round {round}: id {id}"
            );
        }
        if let Some(before) = &previous {
            assert_eq!(&state, before, "round {round} differs from the one before");
        }
        previous = Some(state);
    }
}

// ── A name is reusable; an incarnation is not ────────────────────────────────

/// `VINDEX.DROP` sweeps the blobs, and the sweep runs after the catalogue has
/// already stopped naming the index - so a failure there leaves blobs behind
/// with nothing to remove them. If the next index of the same name keys its
/// blobs the same way, it serves the dead one's payloads as its own.
#[tokio::test]
async fn a_recreated_index_does_not_inherit_the_old_generations_blobs() {
    let dir = tempfile::TempDir::new().unwrap();
    let shards = open(dir.path());
    create(&shards, "re").await;
    for id in 1..=8u64 {
        shards
            .vset(
                "re",
                id,
                vec_for(id),
                0,
                None,
                Some(blob(&format!("old={id}"))),
            )
            .await
            .unwrap();
    }

    arm_at(WriteFailpoint::DropBlobSweep, "re");
    shards.vindex_drop("re", 0).await.expect("the drop commits");
    disarm_at(WriteFailpoint::DropBlobSweep, "re");
    assert!(
        fired_at(WriteFailpoint::DropBlobSweep, "re"),
        "the failpoint never fired, so this test proved nothing"
    );

    create(&shards, "re").await;
    for id in 1..=8u64 {
        // No payload this time: the new index has nothing to say about these
        // ids' payloads, so a search must report nothing.
        shards
            .vset("re", id, vec_for(id), 0, None, None)
            .await
            .unwrap();
    }
    for id in 1..=8u64 {
        assert_eq!(
            payload_of(&shards, "re", id).await,
            None,
            "id {id}: the new index inherited a dropped index's payload"
        );
    }
}

// ── A delete is durable before its blob is reclaimed ─────────────────────────

/// The tombstone is the commit point of a delete. Reclaiming the blob comes
/// after it, so a failure there is cleanup that did not happen - not a delete
/// that did not happen.
#[tokio::test]
async fn vdel_error_at_blob_delete_still_reports_the_delete() {
    let dir = tempfile::TempDir::new().unwrap();
    let shards = open(dir.path());
    create(&shards, "dl").await;
    shards
        .vset("dl", 1, vec_for(1), 0, None, Some(blob("colour=red")))
        .await
        .unwrap();

    arm_at(WriteFailpoint::VdelBlobDelete, "dl");
    let outcome = shards.vdel("dl", 1, 0).await;
    disarm_at(WriteFailpoint::VdelBlobDelete, "dl");
    assert!(
        fired_at(WriteFailpoint::VdelBlobDelete, "dl"),
        "the failpoint never fired, so this test proved nothing"
    );

    assert_eq!(
        outcome.ok(),
        Some(true),
        "the row is gone; the delete must say so"
    );
    assert_eq!(vector_of(&shards, "dl", 1).await, None);
    assert_eq!(
        payload_of(&shards, "dl", 1).await,
        None,
        "a deleted row must not answer with a payload"
    );
}

// ── The blob a failed commit staged is garbage, and must be collected ────────

/// A commit that never happened leaves a staged blob no row names. It is not
/// wrong - nothing reads it - but nothing removes it either, so it is disk a
/// store pays for for ever. The open that follows is where it goes.
#[tokio::test]
async fn a_blob_left_by_a_failed_commit_is_reclaimed_at_the_next_open() {
    let dir = tempfile::TempDir::new().unwrap();
    let name = "or";
    {
        let shards = open(dir.path());
        create(&shards, name).await;
        for id in 1..=8u64 {
            arm_at(WriteFailpoint::VectorCommit, name);
            let outcome = shards
                .vset(
                    name,
                    id,
                    vec_for(id),
                    0,
                    None,
                    Some(blob(&format!("x={id}"))),
                )
                .await;
            disarm_at(WriteFailpoint::VectorCommit, name);
            assert!(outcome.is_err());
        }
        assert!(fired_at(WriteFailpoint::VectorCommit, name));
        let held = shards
            .payload_blobs_held(0, name)
            .await
            .expect("blob count");
        assert_eq!(
            held, 8,
            "fixture: the staged blobs must actually be on disk, or the \
             reclamation below has nothing to prove"
        );
    }
    let shards = open(dir.path());
    assert_eq!(
        shards
            .payload_blobs_held(0, name)
            .await
            .expect("blob count"),
        0,
        "blobs no row names survived the open that was supposed to reclaim them"
    );
}

// ── VMSET answers per item ───────────────────────────────────────────────────

/// A bulk write of n items is n writes. One reply for all of them can only say
/// "some prefix worked", and the client cannot tell which prefix.
#[tokio::test]
async fn vmset_reports_one_result_per_item() {
    let dir = tempfile::TempDir::new().unwrap();
    let shards = open(dir.path());
    create(&shards, "mi").await;

    let results = shards
        .vmset(
            "mi",
            vec![
                (1, vec_for(1), Some(blob("a"))),
                (2, vec![0.0; DIM + 3], Some(blob("b"))), // wrong dimension
                (3, vec_for(3), Some(blob("c"))),
            ],
            0,
            None,
        )
        .await;

    assert_eq!(results.len(), 3, "one result per item: {results:?}");
    assert!(results[0].is_ok(), "item 0: {:?}", results[0]);
    assert!(
        results[1].is_err(),
        "item 1 has the wrong dimension and must say so"
    );
    assert!(results[2].is_ok(), "item 2: {:?}", results[2]);
}

/// And a failing item must not take its siblings with it. The `JoinSet` this
/// replaces aborted every outstanding task on the first error - including
/// tasks whose write had already committed but whose owner map had not been
/// published, which loses an acknowledged row silently.
#[tokio::test]
async fn vmset_does_not_abort_its_siblings() {
    let dir = tempfile::TempDir::new().unwrap();
    let shards = open(dir.path());
    create(&shards, "ms").await;

    let mut items: Vec<(u64, Vec<f32>, Option<Bytes>)> = Vec::new();
    for id in 0..64u64 {
        // Every eighth item is malformed, so the failures are spread through
        // the batch rather than sitting at one end of it.
        let vector = if id % 8 == 3 {
            vec![0.0; DIM + 1]
        } else {
            vec_for(id)
        };
        items.push((id, vector, Some(blob(&format!("i={id}")))));
    }
    let results = shards.vmset("ms", items, 0, None).await;
    assert_eq!(results.len(), 64);
    for id in 0..64u64 {
        let i = id as usize;
        if id % 8 == 3 {
            assert!(results[i].is_err(), "id {id} is malformed");
            assert_eq!(
                vector_of(&shards, "ms", id).await,
                None,
                "id {id} was refused and stored anyway"
            );
        } else {
            assert!(results[i].is_ok(), "id {id}: {:?}", results[i]);
            assert_eq!(
                vector_of(&shards, "ms", id).await,
                Some(vec_for(id)),
                "id {id} was lost because a sibling failed"
            );
            assert_eq!(
                payload_of(&shards, "ms", id).await,
                Some(blob(&format!("i={id}"))),
                "id {id}'s payload was lost because a sibling failed"
            );
        }
    }
}

// ── Whose blobs are these? ───────────────────────────────────────────────────

/// The scoped map key the RESP3 handler builds for `(tenant, index)`: the
/// tenant's 16 bytes in `to_le_bytes` order as hex, `::`, then the name.
/// Written out here rather than imported because the point of these two tests
/// is that a NAME can spell one of these without meaning it.
fn scope_key_of(tenant: u128, index: &str) -> String {
    let mut s = String::with_capacity(32 + 2 + index.len());
    for b in tenant.to_le_bytes() {
        s.push_str(&format!("{b:02x}"));
    }
    s.push_str("::");
    s.push_str(index);
    s
}

/// `payload_of`, for a tenant other than zero.
async fn payload_of_as(shards: &ShardSet, tenant: u128, name: &str, id: u64) -> Option<Bytes> {
    let hits = shards
        .vsearch(name, vec_for(id), 64, 0, tenant, true, None)
        .await
        .expect("vsearch");
    hits.into_iter().find(|h| h.0 == id).and_then(|h| h.2)
}

/// A vindex name is a client-chosen string, and `[A-Za-z0-9._:-]` is legal - so
/// a client on tenant 0 can name an index exactly like the scoped map key of
/// some OTHER tenant. It is still tenant 0's index: it was created by tenant 0,
/// its blobs are written under tenant 0, and every read of them uses tenant 0.
///
/// Anything that recovers the owner by RE-READING the name disagrees, and the
/// disagreement is not an error - it is a lookup that misses. The reclamation
/// at open then finds blobs it believes belong to no index here and deletes
/// them, leaving rows whose payload is gone with nothing said about it.
///
/// Deterministic, no interleaving, one restart. The native protocol uses tenant
/// 0 and the raw name, so it is exposed by default.
#[tokio::test]
async fn a_name_shaped_like_a_scope_key_keeps_its_blobs_at_open() {
    let dir = tempfile::TempDir::new().unwrap();
    // Tenant 0's index, named like tenant 1's scope key.
    let name = scope_key_of(1, "x");
    {
        let shards = open(dir.path());
        create(&shards, &name).await;
        for id in 1..=5u64 {
            shards
                .vset(
                    &name,
                    id,
                    vec_for(id),
                    0,
                    None,
                    Some(blob(&format!("p={id}"))),
                )
                .await
                .unwrap();
        }
        assert_eq!(
            shards.payload_blobs_held(0, &name).await.unwrap(),
            5,
            "fixture: the blobs must be stored before the reopen"
        );
    }

    let shards = open(dir.path());
    assert_eq!(
        shards.payload_blobs_held(0, &name).await.unwrap(),
        5,
        "the open reclaimed the blobs of a live index because its NAME parses \
         as another tenant's"
    );
    for id in 1..=5u64 {
        assert_eq!(
            vector_of(&shards, &name, id).await,
            Some(vec_for(id)),
            "id {id}: the vector"
        );
        assert_eq!(
            payload_of(&shards, &name, id).await,
            Some(blob(&format!("p={id}"))),
            "id {id}: a live row lost its payload, silently"
        );
    }
}

/// The control, and the reason the one above is a bug rather than a rule: a
/// REAL tenant-scoped index - created by that tenant, read by that tenant -
/// keeps its blobs across the same restart. Without this the fix could be
/// "stop reclaiming anything" and both tests would pass.
#[tokio::test]
async fn a_tenant_scoped_index_keeps_its_blobs_at_open() {
    const TENANT: u128 = 0x2a;
    let dir = tempfile::TempDir::new().unwrap();
    let name = scope_key_of(TENANT, "ti");
    {
        let shards = open(dir.path());
        create(&shards, &name).await;
        for id in 1..=5u64 {
            shards
                .vset(
                    &name,
                    id,
                    vec_for(id),
                    TENANT,
                    None,
                    Some(blob(&format!("p={id}"))),
                )
                .await
                .unwrap();
        }
        assert_eq!(
            shards.payload_blobs_held(TENANT, &name).await.unwrap(),
            5,
            "fixture: the blobs must be stored before the reopen"
        );
    }

    let shards = open(dir.path());
    assert_eq!(
        shards.payload_blobs_held(TENANT, &name).await.unwrap(),
        5,
        "a tenant's own index lost its blobs at open"
    );
    for id in 1..=5u64 {
        assert_eq!(
            payload_of_as(&shards, TENANT, &name, id).await,
            Some(blob(&format!("p={id}"))),
            "id {id}"
        );
    }
}
