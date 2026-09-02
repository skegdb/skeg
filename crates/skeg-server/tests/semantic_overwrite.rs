//! An overwrite that moves a row between shards must not lose it when the
//! write to the new owner fails.
//!
//! The routed `vset` deletes the old copies BEFORE writing the new one:
//!
//!     1. pick the new owner from the vector
//!     2. VDEL the old copies on the other shards   <- awaited, propagates
//!     3. VSET on the new owner                     <- can fail
//!     4. update the owner map
//!
//! If 3 fails, 2 has already happened and the previously ACKNOWLEDGED version
//! is gone, while the client is told the write failed. A failed write that
//! destroys the value it was overwriting is worse than either outcome the
//! client can plan for.
//!
//! `reshard` in the same file already uses the opposite order - write the new
//! copy, then delete the source - precisely so a crash leaves a recoverable
//! duplicate rather than a hole. The search path dedups duplicates; nothing
//! recovers a hole.
//!
//! The failpoint here needs no new machinery: a `limit` below the tenant's
//! current vector count. The VDEL frees one slot, and the VSET on the new
//! owner is still over quota, so it fails deterministically at exactly the
//! moment the old copy is gone.

use skeg_server::shard::ShardSet;
use skeg_vector::QuantKind;

const TIER: QuantKind = QuantKind::TurboQuant { bits: 2 };
const DIM: usize = 16;

/// Two orthogonal clusters, so a semantic reshard splits them across shards -
/// but every row UNIQUE within its cluster.
///
/// The uniqueness matters for the search tests: with identical vectors a
/// hundred rows tie at 1.0 and the one being watched drowns among them, so an
/// assertion about which copy came back cannot fail even when it should. The
/// dominant component carries the cluster; a small per-id fingerprint in the
/// remaining dimensions makes each row its own nearest neighbour.
fn cluster_vec(id: u64, cluster: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    v[cluster] = 1.0;
    // Xorshift rather than bits of the id: the first attempt used
    // `(id >> i) & 1`, which gave ids 5 and 7 the identical fingerprint, and
    // the row under test then tied with a sibling instead of standing alone.
    let mut s = (id << 1) | 1;
    for slot in v.iter_mut().skip(2) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        // DISCRETE +/- 0.5, and 14 dimensions of it.
        //
        // Two earlier attempts got this wrong in ways worth recording. A
        // +/- 0.05 fingerprint left the runner-up at 0.9986, no room for any
        // assertion. Then `(s % 200 - 100) / 200` looked like +/- 0.5 but is
        // CONTINUOUS around zero, so most components landed near 0 and the
        // separation stayed weak. And 6 binary dimensions give 64 patterns for
        // 100 rows per cluster, so collisions are guaranteed by pigeonhole -
        // hence DIM 16.
        //
        // With this: an exact self-match is 1.0, the nearest sibling differs
        // in one dimension and scores ~0.86, and the moved copy sits around
        // 0.6. Room to tell them apart.
        *slot = if s % 2 == 0 { 0.5 } else { -0.5 };
    }
    v
}

fn vec_for(id: u64) -> Vec<f32> {
    cluster_vec(id, (id % 2) as usize)
}

/// The same row's fingerprint, in the OTHER cluster.
fn other_cluster(id: u64) -> Vec<f32> {
    cluster_vec(id, ((id + 1) % 2) as usize)
}

#[tokio::test]
async fn a_failed_overwrite_must_not_destroy_the_committed_version() {
    let dir = tempfile::TempDir::new().unwrap();
    let shards = ShardSet::open_mode_with_workers(dir.path(), 2, false, TIER, 1).unwrap();
    shards.vindex_create("ow", DIM as u32, 4, 1).await.unwrap();

    const N: u64 = 200;
    for id in 0..N {
        shards
            .vset("ow", id, vec_for(id), 0, None, None)
            .await
            .unwrap();
    }
    let moved = shards.reshard("ow", 0.25, 10, 0).await.expect("reshard");
    assert!(moved > 0, "the clusters must actually split across shards");

    let victim = 7u64;
    let owner = shards.owners_of("ow", &[victim]).await.unwrap()[0].0;
    let target = 1 - u32::from(owner) as usize; // two shards
    let before = shards
        .vget("ow", victim)
        .await
        .unwrap()
        .expect("the row is there before the overwrite");

    // The failpoint: make the DESTINATION shard unable to serve this vindex.
    // Evict it everywhere, then take the permissions off the destination's
    // directory so its lazy reopen fails. Deterministic, and it uses nothing
    // but the public API and the filesystem.
    //
    // The quota was the first failpoint here, and stopped being one once a
    // cross-shard overwrite became quota-neutral - which is the point: the
    // window has to hold against ANY failure of the write, not one of them.
    shards.control_handle().evict(0, "ow").await.unwrap();
    let blocked = dir.path().join(format!("shard-{target}")).join("vindex-ow");
    let saved = std::fs::metadata(&blocked).unwrap().permissions();
    std::fs::set_permissions(
        &blocked,
        std::os::unix::fs::PermissionsExt::from_mode(0o000),
    )
    .unwrap();

    let err = shards
        .vset("ow", victim, other_cluster(victim), 0, None, None)
        .await
        .expect_err("the destination shard cannot serve the index, so the write must fail");

    // THE ASSERTION. The write failed, so the committed value must still be
    // readable. Anything else means a refused operation destroyed data.
    std::fs::set_permissions(&blocked, saved).unwrap();
    let after =
        shards.vget("ow", victim).await.unwrap().unwrap_or_else(|| {
            panic!("a REFUSED overwrite destroyed the committed version ({err})")
        });
    assert_eq!(
        before, after,
        "a refused overwrite must not change the stored vector"
    );
}

#[tokio::test]
async fn an_overwrite_that_changes_shard_does_not_consume_a_quota_slot() {
    // The quota counts a tenant's LOGICAL cardinality, and an overwrite does
    // not change it - the doc on the check says so: "an overwrite never
    // touches the quota".
    //
    // Across shards it did. The new owner has never seen the id, so it looks
    // new and takes a slot, and at exactly the limit the write is refused for
    // a row the tenant already owns. Reordering to write-before-delete made
    // this worse rather than better: the delete used to free the slot first,
    // by accident.
    let dir = tempfile::TempDir::new().unwrap();
    let shards = ShardSet::open_mode_with_workers(dir.path(), 2, false, TIER, 1).unwrap();
    shards.vindex_create("q", DIM as u32, 4, 1).await.unwrap();

    const N: u64 = 200;
    for id in 0..N {
        shards
            .vset("q", id, vec_for(id), 0, Some(N), None)
            .await
            .unwrap();
    }
    shards.reshard("q", 0.25, 10, 0).await.expect("reshard");

    // Exactly at the limit: N rows, limit N. Move one across shards.
    let victim = 7u64;
    shards
        .vset("q", victim, other_cluster(victim), 0, Some(N), None)
        .await
        .expect(
            "an overwrite at exactly the quota must be allowed: the tenant already owns this id",
        );

    // And the accounting must not have drifted: a NEW id is still refused.
    let err = shards
        .vset("q", N + 1, vec_for(0), 0, Some(N), None)
        .await
        .expect_err("a genuinely new id at the limit must still be refused");
    assert!(format!("{err}").contains("quota"), "got: {err}");
}

#[tokio::test]
async fn search_returns_the_committed_copy_not_the_best_scoring_stale_one() {
    // The cleanup after a committed overwrite is best-effort, so a stale copy
    // can survive on the shard the row moved away from. VGET is safe - it
    // routes through the owner map - but the search merge deduped by BEST
    // SCORE, and the two copies hold different vectors: the old one and the
    // new one. A query resembling the old vector scores the stale copy higher,
    // so search handed back the value the write had replaced, with a confident
    // score, while VGET returned the new one. The same shape as the
    // stale-vector defects found earlier, one level up.
    let dir = tempfile::TempDir::new().unwrap();
    let shards = ShardSet::open_mode_with_workers(dir.path(), 2, false, TIER, 1).unwrap();
    shards.vindex_create("dup", DIM as u32, 4, 1).await.unwrap();

    const N: u64 = 200;
    for id in 0..N {
        shards
            .vset("dup", id, vec_for(id), 0, None, None)
            .await
            .unwrap();
    }
    shards.reshard("dup", 0.25, 10, 0).await.expect("reshard");

    let victim = 7u64;
    let old_owner = usize::from(shards.owners_of("dup", &[victim]).await.unwrap()[0].0);
    let old_vector = shards.vget("dup", victim).await.unwrap().unwrap();

    // Prove the fixture before relying on it: the victim's own vector must
    // find the victim, alone at the top. Without that, the assertion at the
    // end cannot fail even when the bug is present - which is exactly what the
    // first version of this test did.
    let baseline = shards
        .vsearch("dup", old_vector.clone(), 3, 0, 0, false, None)
        .await
        .unwrap();
    assert_eq!(
        baseline[0].0, victim,
        "the fixture must make {victim} unique"
    );
    assert!(
        baseline[0].1 > 0.999,
        "and an exact self-match: {baseline:?}"
    );
    assert!(
        baseline[1].1 < 0.9,
        "with daylight to the next: {baseline:?}"
    );

    // Commit the overwrite, then make the cleanup fail: the old shard cannot
    // serve the index, so its copy survives.
    shards.control_handle().evict(0, "dup").await.unwrap();
    let blocked = dir
        .path()
        .join(format!("shard-{old_owner}"))
        .join("vindex-dup");
    let saved = std::fs::metadata(&blocked).unwrap().permissions();
    std::fs::set_permissions(
        &blocked,
        std::os::unix::fs::PermissionsExt::from_mode(0o000),
    )
    .unwrap();
    let new_vector = other_cluster(victim);
    shards
        .vset("dup", victim, new_vector.clone(), 0, None, None)
        .await
        .expect("the write to the NEW owner must still succeed");
    std::fs::set_permissions(&blocked, saved).unwrap();

    // Both copies are now live on disk. VGET must see the new one...
    assert_eq!(
        shards.vget("dup", victim).await.unwrap().unwrap(),
        new_vector,
        "the point read must see the committed copy"
    );

    // ...and so must the search, even when the query is the OLD vector, which
    // is exactly the case where the stale copy scores best.
    let hits = shards
        .vsearch("dup", old_vector.clone(), 10, 0, 0, false, None)
        .await
        .unwrap();
    let &(_, score, _) = hits
        .iter()
        .find(|h| h.0 == victim)
        .unwrap_or_else(|| panic!("id {victim} must be in the results at all: {hits:?}"));
    // The stale copy is an EXACT match for this query and would score ~1.0.
    // The committed copy sits in the other cluster and cannot.
    assert!(
        score < 0.9,
        "search returned the STALE copy for id {victim} (score {score}): the \
         committed copy holds a different vector and cannot score that high"
    );
}

// ── the stale-copy interleavings ─────────────────────────────────────────────
//
// A row can exist in more than one place at once, and which copy is LIVE was
// decided by position: the shard the owner map happened to name, and on a
// reopen the lowest shard number that answered. Position is not identity.
// `reshard` and `overlap` relocate rows without taking the stripe lock that
// serialises point ops on an id, so both can publish a copy they read before a
// concurrent write replaced it - and an acknowledged write then disappears.
//
// Each test here reproduces one interleaving through the public API only.

/// The bigger fixture the concurrency tests need: enough rows that a reshard
/// takes many batches, so a concurrent overwrite lands inside one.
async fn seeded(dir: &std::path::Path, name: &str, n: u64) -> ShardSet {
    let shards = ShardSet::open_mode_with_workers(dir, 2, false, TIER, 1).unwrap();
    shards.vindex_create(name, DIM as u32, 4, 1).await.unwrap();
    for id in 0..n {
        shards
            .vset(name, id, vec_for(id), 0, None, None)
            .await
            .unwrap();
    }
    shards
}

/// The vindex directory of one shard, used as a failure injector.
fn vindex_dir(dir: &std::path::Path, shard: usize, name: &str) -> std::path::PathBuf {
    dir.join(format!("shard-{shard}"))
        .join(format!("vindex-{name}"))
}

fn block(path: &std::path::Path) -> std::fs::Permissions {
    let saved = std::fs::metadata(path).unwrap().permissions();
    std::fs::set_permissions(path, std::os::unix::fs::PermissionsExt::from_mode(0o000)).unwrap();
    saved
}

/// Interleaving A. `reshard` collects a batch of rows, then moves them one at
/// a time with an await between each. A `vset` that commits inside that window
/// replaces the row - the client is told so - and the reshard then writes the
/// copy IT read over the top and points the owner map at it. The acknowledged
/// version is gone, and nothing reports a failure.
#[tokio::test]
async fn a_reshard_must_not_republish_a_vector_a_concurrent_vset_replaced() {
    let dir = tempfile::TempDir::new().unwrap();
    const N: u64 = 1200;
    let shards = seeded(dir.path(), "ow", N).await;

    // Spread the victims across the id space so they land in different
    // batches, and across BOTH clusters: a reshard batch from one shard holds
    // only the rows that shard has to give away, and a victim set drawn from
    // the other half is never in the batch that is in flight.
    let victims: Vec<u64> = (0..N).step_by(5).collect();
    let writer = {
        let shards = shards.clone();
        let victims = victims.clone();
        async move {
            // The routed path only exists once the router does; before that a
            // vset is hash-placed and never enters the owner map.
            while shards.router("ow").is_none() {
                tokio::task::yield_now().await;
            }
            for id in victims {
                shards
                    .vset("ow", id, other_cluster(id), 0, None, None)
                    .await
                    .expect("the overwrite is acknowledged");
            }
        }
    };
    let (resharded, ()) = tokio::join!(shards.reshard("ow", 0.25, 10, 0), writer);
    resharded.expect("reshard");

    // THE ASSERTION. Every acknowledged overwrite must still be the value the
    // store holds, and the owner map must name the shard that holds it.
    for id in victims {
        let want = other_cluster(id);
        assert_eq!(
            shards.vget("ow", id).await.unwrap().as_deref(),
            Some(&want[..]),
            "id {id}: the reshard republished the copy the overwrite replaced"
        );
    }
}

/// Interleaving B. `overlap` reads a batch of boundary rows, checks the owner
/// map, then writes a replica - with awaits in between. A `vdel` that commits
/// inside that window removes both copies and the map entry; the replica then
/// lands afterwards on a shard nobody will clean up, and the deleted row is
/// searchable again.
#[tokio::test]
async fn an_overlap_must_not_replicate_a_row_a_concurrent_vdel_removed() {
    let dir = tempfile::TempDir::new().unwrap();
    const N: u64 = 800;
    let shards = seeded(dir.path(), "ov", N).await;
    shards.reshard("ov", 0.25, 10, 0).await.expect("reshard");

    // The deletes have to WALK WITH the replication, not race past it: the
    // window is one await wide, so a deleter that sweeps the id space at a
    // different rate crosses it once and mostly misses. `overlap` replicates
    // the rows of shard 0 in ascending id, one await each; deleting exactly
    // those ids in the same order keeps the two in step.
    let all: Vec<u64> = (0..N).collect();
    let placement = shards.owners_of("ov", &all).await.unwrap();
    let victims: Vec<u64> = all
        .iter()
        .copied()
        .filter(|&id| placement[id as usize].0 == 0)
        .collect();
    assert!(
        victims.len() > N as usize / 4,
        "the fixture must split the set"
    );
    let deleter = {
        let shards = shards.clone();
        let victims = victims.clone();
        async move {
            for id in victims {
                shards
                    .vdel("ov", id, 0)
                    .await
                    .expect("the delete is acknowledged");
            }
        }
    };
    // tau above any achievable margin: every row is a boundary row, so the
    // replication loop is long enough to interleave with the deletes.
    let (replicated, ()) = tokio::join!(shards.overlap("ov", 4.0, 0), deleter);
    replicated.expect("overlap");

    for id in victims {
        assert!(
            shards.vget("ov", id).await.unwrap().is_none(),
            "id {id}: the overlap replicated a row a concurrent delete removed"
        );
        let hits = shards
            .vsearch("ov", vec_for(id), 5, 0, 0, false, None)
            .await
            .unwrap();
        assert!(
            !hits.iter().any(|h| h.0 == id),
            "id {id}: a deleted row came back through its replica: {hits:?}"
        );
    }
}

/// Pick a live id whose primary is `from`, and whose overwritten form the
/// router would place elsewhere.
async fn victim_on(shards: &ShardSet, name: &str, n: u64, from: u8) -> u64 {
    for id in 0..n {
        if shards.owners_of(name, &[id]).await.unwrap()[0].0 == from {
            return id;
        }
    }
    panic!("no row is placed on shard {from}");
}

/// The fixture the reopen tests share: an overwrite that moves a row UP a
/// shard number, with the old copy left behind because the cleanup could not
/// run. Returns `(victim, new owner, the vector now committed)`.
async fn a_committed_move_with_a_surviving_old_copy(
    dir: &std::path::Path,
    shards: &ShardSet,
    name: &str,
    n: u64,
) -> (u64, u8, Vec<f32>) {
    // The old copy has to sit on the LOWER shard, or the positional rule and
    // the correct answer coincide and the test cannot fail.
    let victim = victim_on(shards, name, n, 0).await;
    shards.control_handle().evict(0, name).await.unwrap();
    let blocked = vindex_dir(dir, 0, name);
    let saved = block(&blocked);
    let committed = other_cluster(victim);
    shards
        .vset(name, victim, committed.clone(), 0, None, None)
        .await
        .expect("the write to the new owner must succeed");
    std::fs::set_permissions(&blocked, saved).unwrap();
    let owner = shards.owners_of(name, &[victim]).await.unwrap()[0].0;
    assert_eq!(owner, 1, "the overwrite must have moved the row up a shard");
    (victim, owner, committed)
}

/// Interleaving C, without any concurrency at all. Two copies of a row, the
/// stale one on the lower shard. `rebuild_owner_maps` picks a primary by
/// iteration order - lowest shard first - so a restart promotes the copy the
/// overwrite replaced, and every read after it returns the old value with a
/// confident face.
#[tokio::test]
async fn a_reopened_set_names_the_newest_copy_primary_not_the_lowest_shard() {
    let dir = tempfile::TempDir::new().unwrap();
    const N: u64 = 200;
    let shards = seeded(dir.path(), "rp", N).await;
    shards.reshard("rp", 0.25, 10, 0).await.expect("reshard");
    let (victim, owner, committed) =
        a_committed_move_with_a_surviving_old_copy(dir.path(), &shards, "rp", N).await;
    drop(shards);

    let shards = ShardSet::open_mode_with_workers(dir.path(), 2, false, TIER, 1).unwrap();
    assert_eq!(
        shards.owners_of("rp", &[victim]).await.unwrap()[0].0,
        owner,
        "the reopened set named the stale copy primary"
    );
    assert_eq!(
        shards.vget("rp", victim).await.unwrap().unwrap(),
        committed,
        "a restart handed back the value the overwrite replaced"
    );
}

/// The same state, asked the two ways a client can ask. A point read and a
/// search that disagree about which copy of a row is live is a wrong answer
/// that cannot be recognised as one from either side.
#[tokio::test]
async fn vget_and_vsearch_agree_on_which_copy_is_live_after_a_reopen() {
    let dir = tempfile::TempDir::new().unwrap();
    const N: u64 = 200;
    let shards = seeded(dir.path(), "ag", N).await;
    shards.reshard("ag", 0.25, 10, 0).await.expect("reshard");
    let stale = shards.vget("ag", 0).await.unwrap(); // touch, keeps the map warm
    let _ = stale;
    let (victim, _, committed) =
        a_committed_move_with_a_surviving_old_copy(dir.path(), &shards, "ag", N).await;
    let replaced = vec_for(victim);
    drop(shards);

    let shards = ShardSet::open_mode_with_workers(dir.path(), 2, false, TIER, 1).unwrap();
    assert_eq!(
        shards.vget("ag", victim).await.unwrap().unwrap(),
        committed,
        "the point read must see the committed copy"
    );
    // The query the stale copy is an EXACT match for: the case where scoring
    // alone hands back the replaced value.
    let hits = shards
        .vsearch("ag", replaced, 10, 0, 0, false, None)
        .await
        .unwrap();
    let &(_, score, _) = hits
        .iter()
        .find(|h| h.0 == victim)
        .unwrap_or_else(|| panic!("id {victim} must be in the results at all: {hits:?}"));
    assert!(
        score < 0.9,
        "search returned the STALE copy for id {victim} (score {score}) while \
         vget returned the committed one"
    );
}

/// A move whose destination write fails must leave the source authoritative:
/// nothing removed, nothing renamed, and the failure reported. The failpoint
/// makes exactly that one step fail - which permissions cannot, since the
/// source and the destination of a reshard are the same kind of directory and
/// both are needed.
#[tokio::test]
async fn a_reshard_that_cannot_write_the_destination_leaves_the_source_authoritative() {
    use skeg_server::failpoint::{WriteFailpoint, arm, disarm_all, fired};
    let dir = tempfile::TempDir::new().unwrap();
    const N: u64 = 200;
    let shards = seeded(dir.path(), "fd", N).await;

    arm(WriteFailpoint::ReshardDestinationWrite);
    let outcome = shards.reshard("fd", 0.25, 10, 0).await;
    disarm_all();
    assert!(
        fired(WriteFailpoint::ReshardDestinationWrite),
        "the failpoint never fired, so this test proved nothing"
    );
    assert!(
        outcome.is_err(),
        "a move whose destination write failed must be reported, not counted"
    );
    for id in 0..N {
        assert_eq!(
            shards.vget("fd", id).await.unwrap().as_deref(),
            Some(&vec_for(id)[..]),
            "id {id}: a failed move must leave the source copy readable"
        );
    }
}

/// The crash window a move exists to survive: the destination copy is written
/// and the source delete never happens. Both copies hold the same vector, so
/// there is no wrong value to return - but there must be exactly ONE live
/// copy after a restart, and it must be found the same way by every reader.
#[tokio::test]
async fn a_reshard_that_crashes_after_copy_before_delete_reopens_with_one_winner() {
    use skeg_server::failpoint::{WriteFailpoint, arm, disarm_all, fired};
    let dir = tempfile::TempDir::new().unwrap();
    const N: u64 = 200;
    let shards = seeded(dir.path(), "fs", N).await;

    arm(WriteFailpoint::ReshardSourceDelete);
    let outcome = shards.reshard("fs", 0.25, 10, 0).await;
    disarm_all();
    assert!(
        fired(WriteFailpoint::ReshardSourceDelete),
        "the failpoint never fired, so this test proved nothing"
    );
    assert!(
        outcome.is_err(),
        "a move whose source delete failed must be reported"
    );
    drop(shards);

    let shards = ShardSet::open_mode_with_workers(dir.path(), 2, false, TIER, 1).unwrap();
    for id in 0..N {
        assert_eq!(
            shards.vget("fs", id).await.unwrap().as_deref(),
            Some(&vec_for(id)[..]),
            "id {id}: the surviving copy must hold the right vector"
        );
        let hits = shards
            .vsearch("fs", vec_for(id), 5, 0, 0, false, None)
            .await
            .unwrap();
        assert_eq!(
            hits.iter().filter(|h| h.0 == id).count(),
            1,
            "id {id}: exactly one winner, {hits:?}"
        );
    }
    // The crash left a real second copy of the row it had already moved, and
    // the reopened map is right to name it as a replica - what matters is that
    // it is TRACKED, so a delete still reaches it. An untracked copy is a
    // ghost: invisible to the map, findable by search.
    for id in 0..N {
        shards.vdel("fs", id, 0).await.unwrap();
    }
    for id in 0..N {
        let hits = shards
            .vsearch("fs", vec_for(id), 5, 0, 0, false, None)
            .await
            .unwrap();
        assert!(
            !hits.iter().any(|h| h.0 == id),
            "id {id}: a copy the crash left behind outlived its row: {hits:?}"
        );
    }
}

/// The cleanup after a committed overwrite is best-effort by design, so a
/// surviving old copy is an expected state, not a corrupt one. What must not
/// happen is a restart promoting it.
#[tokio::test]
async fn a_reopen_after_a_failed_old_copy_cleanup_still_names_the_new_copy_primary() {
    use skeg_server::failpoint::{WriteFailpoint, arm, disarm_all, fired};
    let dir = tempfile::TempDir::new().unwrap();
    const N: u64 = 200;
    let shards = seeded(dir.path(), "cl", N).await;
    shards.reshard("cl", 0.25, 10, 0).await.expect("reshard");
    let victim = victim_on(&shards, "cl", N, 0).await;
    let committed = other_cluster(victim);

    arm(WriteFailpoint::OverwriteOldCopyDelete);
    shards
        .vset("cl", victim, committed.clone(), 0, None, None)
        .await
        .expect("the overwrite commits: the cleanup is post-commit");
    disarm_all();
    assert!(
        fired(WriteFailpoint::OverwriteOldCopyDelete),
        "the failpoint never fired, so this test proved nothing"
    );

    // The failpoint must actually have left the duplicate behind, or the rest
    // of this test proves nothing.
    let held: usize = shards
        .control_handle()
        .open_indices()
        .await
        .iter()
        .filter(|s| s.index == "cl")
        .map(|s| s.vectors)
        .sum();
    assert_eq!(
        held,
        N as usize + 1,
        "the old copy must have survived the failed cleanup"
    );
    let owner = shards.owners_of("cl", &[victim]).await.unwrap()[0].0;
    assert_eq!(owner, 1, "the overwrite must have moved the row up a shard");
    drop(shards);

    let shards = ShardSet::open_mode_with_workers(dir.path(), 2, false, TIER, 1).unwrap();
    assert_eq!(
        shards.owners_of("cl", &[victim]).await.unwrap()[0].0,
        owner,
        "the reopened set named the stale copy primary"
    );
    assert_eq!(
        shards.vget("cl", victim).await.unwrap().unwrap(),
        committed,
        "a restart handed back the value the overwrite replaced"
    );
}

/// The owner map is derived state, and there is a window on every restart
/// where it does not exist yet: it is rebuilt on the first point op, and a
/// search does not perform one. Until then the merge had nothing to prefer by
/// and fell back to the score - so a query resembling the copy an overwrite
/// replaced got that copy back, at 1.0, from a store that had committed the
/// replacement.
///
/// The version comes from the shard holding the row, so it cannot be behind
/// the row the way the map can.
#[tokio::test]
async fn a_search_before_the_owner_map_is_rebuilt_still_returns_the_newest_copy() {
    let dir = tempfile::TempDir::new().unwrap();
    const N: u64 = 200;
    let shards = seeded(dir.path(), "nm", N).await;
    shards.reshard("nm", 0.25, 10, 0).await.expect("reshard");
    let (victim, _, committed) =
        a_committed_move_with_a_surviving_old_copy(dir.path(), &shards, "nm", N).await;
    let replaced = vec_for(victim);
    drop(shards);

    // Reopened and untouched: no point op has run, so no owner map exists.
    let shards = ShardSet::open_mode_with_workers(dir.path(), 2, false, TIER, 1).unwrap();
    let hits = shards
        .vsearch("nm", replaced.clone(), 10, 0, 0, false, None)
        .await
        .unwrap();
    let &(_, score, _) = hits
        .iter()
        .find(|h| h.0 == victim)
        .unwrap_or_else(|| panic!("id {victim} must be in the results at all: {hits:?}"));
    assert!(
        score < 0.9,
        "search returned the STALE copy for id {victim} (score {score}): the \
         committed copy is in the other cluster and cannot score that high"
    );
    // And the point read, which does rebuild the map, agrees.
    assert_eq!(shards.vget("nm", victim).await.unwrap().unwrap(), committed);
}

/// The version allocator is derived state, seeded at every owner-map rebuild
/// from what the shards report. A DELETED row is not in that report, and its
/// tombstone carries the highest version in the index whenever the last thing
/// anyone did was delete the row they had just written.
///
/// The allocator then restarts below that tombstone, the next write to that id
/// is handed a version the tombstone beats, and the engine drops it - correctly
/// by its own rule, and catastrophically from outside: `vset` returns success
/// and the row is not there.
#[tokio::test]
async fn a_reinsert_after_a_restart_is_not_dropped_as_stale() {
    let dir = tempfile::TempDir::new().unwrap();
    const N: u64 = 200;
    let shards = seeded(dir.path(), "rs", N).await;
    shards.reshard("rs", 0.25, 10, 0).await.expect("reshard");

    // Give one row the highest version in the index, then delete it: the
    // delete allocates a version above that again, and it now lives only in a
    // tombstone.
    let victim = 7u64;
    for _ in 0..3 {
        shards
            .vset("rs", victim, other_cluster(victim), 0, None, None)
            .await
            .unwrap();
        shards
            .vset("rs", victim, vec_for(victim), 0, None, None)
            .await
            .unwrap();
    }
    assert!(shards.vdel("rs", victim, 0).await.unwrap());
    drop(shards);

    let shards = ShardSet::open_mode_with_workers(dir.path(), 2, false, TIER, 1).unwrap();
    let back = other_cluster(victim);
    shards
        .vset("rs", victim, back.clone(), 0, None, None)
        .await
        .expect("the re-insert is acknowledged");
    assert_eq!(
        shards.vget("rs", victim).await.unwrap().as_deref(),
        Some(&back[..]),
        "an acknowledged re-insert after a restart was dropped as stale"
    );
    let hits = shards
        .vsearch("rs", back.clone(), 3, 0, 0, false, None)
        .await
        .unwrap();
    assert!(
        hits.iter().any(|h| h.0 == victim),
        "and it must be searchable: {hits:?}"
    );
}

/// Interleaving B again, on the other relocation. `overlap` skips a row in TWO
/// cases - the owner map has no entry for it, or its version has advanced -
/// and `reshard` skipped only the second. A successful `vdel` REMOVES the map
/// entry, so `current` comes back `None`, the version guard does not fire, and
/// the row the client was told was deleted is written to its new owner and
/// published as live.
///
/// The missing half cannot be an `else` on "entry absent": for `reshard` that
/// is also the normal state of a row that has never moved, which is most of
/// them on a first reshard. The tombstone is on the SOURCE shard, and the
/// source is the only thing that can tell the two apart.
///
/// The fixture is built rather than hoped for. Training the router first makes
/// the move ORDER computable - source 0's rows in ascending id, then source
/// 1's - so the deletes can walk exactly that sequence at exactly that rate,
/// which is what keeps them inside the one-await window instead of crossing it
/// once. Started on the router epoch changing, so they begin when `reshard`'s
/// own retrain finishes and its move loop starts.
#[tokio::test]
async fn a_reshard_must_not_republish_a_row_a_concurrent_vdel_removed() {
    let dir = tempfile::TempDir::new().unwrap();
    const N: u64 = 1200;
    let shards = seeded(dir.path(), "rd", N).await;
    shards.train_router("rd", 0.25, 10).await.expect("train");
    let router = shards.router("rd").expect("router just trained");
    let epoch = router.epoch;

    let all: Vec<u64> = (0..N).collect();
    let placed = shards.owners_of("rd", &all).await.unwrap();
    let mut victims: Vec<u64> = Vec::new();
    for source in 0..2u8 {
        for id in 0..N {
            if placed[id as usize].0 == source && router.assign(&vec_for(id)) as u8 != source {
                victims.push(id);
            }
        }
    }
    assert!(
        victims.len() > N as usize / 4,
        "the fixture must give the reshard real work: {}",
        victims.len()
    );

    let deleter = {
        let shards = shards.clone();
        let victims = victims.clone();
        async move {
            // `reshard` retrains before it moves anything; the epoch bump is
            // the moment its move loop starts.
            while shards.router("rd").is_none_or(|r| r.epoch == epoch) {
                tokio::task::yield_now().await;
            }
            for id in victims {
                shards
                    .vdel("rd", id, 0)
                    .await
                    .expect("the delete is acknowledged");
            }
        }
    };
    let (moved, ()) = tokio::join!(shards.reshard("rd", 0.25, 10, 0), deleter);
    moved.expect("reshard");

    let mut undone: Vec<u64> = Vec::new();
    for &id in &victims {
        if shards.vget("rd", id).await.unwrap().is_some() {
            undone.push(id);
            continue;
        }
        let hits = shards
            .vsearch("rd", vec_for(id), 5, 0, 0, false, None)
            .await
            .unwrap();
        if hits.iter().any(|h| h.0 == id) {
            undone.push(id);
        }
    }
    assert!(
        undone.is_empty(),
        "{} of {} acknowledged deletes were undone by the reshard: {:?}",
        undone.len(),
        victims.len(),
        &undone[..undone.len().min(8)]
    );
}

/// The version allocator is only as good as the rebuild that seeds it, and the
/// rebuild used to treat a shard that could not open its index as a shard that
/// did not have it. `get_or_reopen` returns `None` for both - it logs the I/O
/// error and swallows it - so `LiveIds` answered "not found" either way.
///
/// The consequence is worse than an incomplete map. That shard contributes no
/// high-water version either, so the allocator is seeded BELOW the versions it
/// holds, and the next user write to one of its rows gets a version the copy
/// there beats: the destination refuses it as superseded, which is correct,
/// and the client is told `+OK` for a write that never happened.
///
/// Absent and unreadable are different states and only one of them has a safe
/// answer.
#[tokio::test]
async fn a_shard_that_cannot_open_its_index_fails_the_owner_map_rebuild() {
    let dir = tempfile::TempDir::new().unwrap();
    const N: u64 = 200;
    let shards = seeded(dir.path(), "un", N).await;
    shards.reshard("un", 0.25, 10, 0).await.expect("reshard");
    let victim = victim_on(&shards, "un", N, 1).await;
    drop(shards);

    // A reopen with the directory already blocked is refused at startup -
    // `recover_vindexes` fails the whole open, which is the right answer and
    // not the window here. The reachable one is an index that becomes
    // unreadable AFTER the shard is serving: evicted from RAM, so the next
    // access goes through `get_or_reopen`, and that is the path that logs the
    // I/O error and returns the same `None` as a name nobody ever created.
    let shards = ShardSet::open_mode_with_workers(dir.path(), 2, false, TIER, 1).unwrap();
    shards.control_handle().evict(0, "un").await.unwrap();
    let blocked = vindex_dir(dir.path(), 1, "un");
    let saved = block(&blocked);
    let err = shards
        .vset("un", victim, other_cluster(victim), 0, None, None)
        .await
        .expect_err(
            "a write whose index cannot be enumerated on one shard must be refused, \
             not acknowledged against a map built from the shards that answered",
        );
    let err = format!("{err}");
    assert!(
        err.contains("un"),
        "the error must name the index it is about: {err}"
    );
    // A read is refused for the same reason, rather than answering from a map
    // that is missing a shard's rows.
    shards
        .vget("un", victim)
        .await
        .expect_err("and so is a read that routes through the same map");

    // Restored, it serves again - the refusal is about the state, not a latch.
    std::fs::set_permissions(&blocked, saved).unwrap();
    let want = other_cluster(victim);
    shards
        .vset("un", victim, want.clone(), 0, None, None)
        .await
        .expect("with the shard readable the write goes through");
    assert_eq!(shards.vget("un", victim).await.unwrap().unwrap(), want);
}

/// A replica the owner map never names is a ghost. `vdel` finds the second
/// copy of a row by reading the replica slot, so a copy that is on disk and
/// not in the map is one nothing will ever remove: the deleted row stays
/// searchable until the next rebuild.
///
/// `overlap` returned before updating the map on every failure of the replica
/// write - including the ones where the write LANDED. The destination stores
/// the vector and then writes the payload blob, and a blob failure comes back
/// as an error after the row is already in; a lost reply does the same.
#[tokio::test]
async fn an_overlap_that_fails_after_writing_a_replica_does_not_leave_a_ghost() {
    use skeg_server::failpoint::{WriteFailpoint, arm, disarm_all, fired};
    let dir = tempfile::TempDir::new().unwrap();
    const N: u64 = 200;
    let shards = seeded(dir.path(), "gh", N).await;
    shards.reshard("gh", 0.25, 10, 0).await.expect("reshard");

    arm(WriteFailpoint::OverlapReplicaWrite);
    let err = shards.overlap("gh", 4.0, 0).await;
    disarm_all();
    assert!(
        fired(WriteFailpoint::OverlapReplicaWrite),
        "the failpoint never fired, so this test proved nothing"
    );
    err.expect_err("the replica write failed, so the overlap must report it");

    // Every row deleted; nothing may survive. A copy the map does not name is
    // exactly the copy this cannot reach.
    for id in 0..N {
        shards.vdel("gh", id, 0).await.expect("delete");
    }
    for id in 0..N {
        let hits = shards
            .vsearch("gh", vec_for(id), 5, 0, 0, false, None)
            .await
            .unwrap();
        assert!(
            !hits.iter().any(|h| h.0 == id),
            "id {id} outlived its delete: the failed overlap left a copy the \
             owner map never named, so nothing went looking for it: {hits:?}"
        );
    }
    let held: usize = shards
        .control_handle()
        .open_indices()
        .await
        .iter()
        .filter(|s| s.index == "gh")
        .map(|s| s.vectors)
        .sum();
    assert_eq!(held, 0, "and no copy of any row is left on any shard");
}
