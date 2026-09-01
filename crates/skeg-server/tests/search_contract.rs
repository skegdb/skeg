//! What a VSEARCH answer means when a shard is unavailable.
//!
//! A partial top-k is not a slightly worse answer - it is a wrong answer the
//! client cannot recognise as one. The shape is identical to a complete
//! result: the same field layout, plausible scores, and a count that proves
//! nothing, since `k` is a maximum and a genuine query can legitimately return
//! fewer. Whatever is built on top inherits the error silently: a RAG context
//! missing its best passage, a dedup that misses the duplicate.
//!
//! So: consistency over availability. Any shard failing fails the search.
//! A best-effort mode is a reasonable thing to want, but it has to say so in
//! the response - a partial flag and the failed shards - rather than being the
//! silent default.

use skeg_server::shard::ShardSet;
use skeg_vector::QuantKind;
use std::os::unix::fs::PermissionsExt;

const TIER: QuantKind = QuantKind::TurboQuant { bits: 2 };
const DIM: usize = 8;

fn vec_for(id: u64) -> Vec<f32> {
    let mut v = vec![0.05f32; DIM];
    v[(id % 4) as usize] = 1.0;
    v
}

#[tokio::test]
async fn a_search_fails_when_any_shard_fails_even_if_others_returned_hits() {
    let dir = tempfile::TempDir::new().unwrap();
    let shards = ShardSet::open_mode_with_workers(dir.path(), 4, false, TIER, 1).unwrap();
    shards.vindex_create("s", DIM as u32, 4, 1).await.unwrap();
    for id in 0..400u64 {
        shards
            .vset("s", id, vec_for(id), 0, None, None)
            .await
            .unwrap();
    }

    // Complete answer first, as the baseline.
    let full = shards
        .vsearch("s", vec_for(1), 20, 0, 0, false, None)
        .await
        .expect("the healthy search must succeed");
    assert_eq!(full.len(), 20, "the baseline must fill k");

    // Take one shard out: evict everywhere, then make shard 2 unable to
    // reopen the index. The other three still hold rows and will answer.
    shards.control_handle().evict(0, "s").await.unwrap();
    let blocked = dir.path().join("shard-2").join("vindex-s");
    let saved = std::fs::metadata(&blocked).unwrap().permissions();
    std::fs::set_permissions(&blocked, PermissionsExt::from_mode(0o000)).unwrap();

    let result = shards.vsearch("s", vec_for(1), 20, 0, 0, false, None).await;
    std::fs::set_permissions(&blocked, saved).unwrap();

    let err = result.expect_err(
        "a search missing a shard must FAIL: a shorter top-k is indistinguishable \
         from a complete one",
    );
    assert!(
        format!("{err}").contains('s'),
        "the error should name what went wrong: {err}"
    );

    // And it recovers once the shard is readable again - the refusal is about
    // this attempt, not a latched state.
    let again = shards
        .vsearch("s", vec_for(1), 20, 0, 0, false, None)
        .await
        .expect("the search works again once every shard can answer");
    assert_eq!(again.len(), 20);
}
