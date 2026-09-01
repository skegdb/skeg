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
    // stale-vector P0s, one level up.
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
