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
const DIM: usize = 8;

/// Two orthogonal clusters, so a semantic reshard splits them across shards.
fn vec_for(id: u64) -> Vec<f32> {
    let mut v = vec![0.05f32; DIM];
    v[(id % 2) as usize] = 1.0;
    v
}

/// A vector belonging to the OTHER cluster than `vec_for(id)`.
fn other_cluster(id: u64) -> Vec<f32> {
    let mut v = vec![0.05f32; DIM];
    v[((id + 1) % 2) as usize] = 1.0;
    v
}

#[tokio::test]
async fn a_failed_overwrite_must_not_destroy_the_committed_version() {
    let dir = tempfile::TempDir::new().unwrap();
    let shards = ShardSet::open_mode_with_workers(dir.path(), 2, false, TIER, 1).unwrap();
    shards.vindex_create("ow", DIM as u32, 4, 1).await.unwrap();

    // A generous limit on the way IN, because the tenant counter is only
    // maintained when a limit is passed: with `None` it stays at zero and the
    // failpoint below would never fire.
    const N: u64 = 200;
    for id in 0..N {
        shards
            .vset("ow", id, vec_for(id), 0, Some(10_000), None)
            .await
            .unwrap();
    }
    let moved = shards.reshard("ow", 0.25, 10, 0).await.expect("reshard");
    assert!(moved > 0, "the clusters must actually split across shards");

    // The row we are about to overwrite, and what it holds right now.
    let victim = 7u64;
    let before = shards
        .vget("ow", victim)
        .await
        .unwrap()
        .expect("the row is there before the overwrite");

    // Overwrite it with a vector from the OTHER cluster, so the owner moves,
    // and with a quota below the current count so the write to the new owner
    // is refused after the old copy has been deleted.
    let err = shards
        .vset("ow", victim, other_cluster(victim), 0, Some(1), None)
        .await
        .expect_err("the quota must refuse this write");
    assert!(
        format!("{err}").contains("quota"),
        "the failpoint must be the quota, not something else: {err}"
    );

    // THE ASSERTION. The write failed, so the committed value must still be
    // readable. Anything else means a refused operation destroyed data.
    let after = shards
        .vget("ow", victim)
        .await
        .unwrap()
        .expect("a REFUSED overwrite must leave the committed version readable");
    assert_eq!(
        before, after,
        "a refused overwrite must not change the stored vector"
    );
}
