//! A vector quota set while the server runs must count what the tenant
//! already holds.
//!
//! Task 3 made the counter survive a restart: the readiness barrier rebuilds
//! it from the store, so an operator who sets `max_vectors` and restarts gets
//! the ceiling they asked for. What it did not cover is the transition
//! *without* a restart: a tenant that wrote while it had no limit was never
//! counted at all, because the charge was taken only when a limit existed.
//! `SKEG.QUOTA.SET tenant N` then found a counter at zero and let that tenant
//! write another N rows on top of the ones it already had - a hard quota that
//! was not one until the next restart.
//!
//! The rule these tests pin: the counter tracks the tenant's live rows
//! whether or not a limit exists, and the limit only decides refusal.

use skeg_server::shard::ShardSet;
use skeg_vector::QuantKind;

const DIM: usize = 4;
const TENANT: u128 = 0x2a;

fn vec_at(i: u64) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    v[(i as usize) % DIM] = 1.0 + (i as f32);
    v
}

async fn set(dir: &std::path::Path) -> ShardSet {
    let shards = ShardSet::open_mode_with_workers(dir, 1, false, QuantKind::Int8, 1).unwrap();
    shards.vindex_create("q", DIM as u32, 0, 1).await.unwrap();
    shards
}

/// The tenant wrote 5 rows with no ceiling; a ceiling of 5 applied afterwards
/// is already reached, so the sixth row is refused - without a restart.
#[tokio::test]
async fn a_limit_set_after_the_writes_counts_the_writes_that_came_before() {
    let dir = tempfile::tempdir().unwrap();
    let shards = set(dir.path()).await;
    for id in 0..5u64 {
        shards
            .vset("q", id, vec_at(id), TENANT, None, None)
            .await
            .expect("unlimited writes are admitted");
    }
    assert_eq!(
        shards.tenant_vector_count(TENANT),
        5,
        "rows written without a ceiling must still be counted"
    );
    let refused = shards
        .vset("q", 99, vec_at(99), TENANT, Some(5), None)
        .await;
    assert!(
        refused.is_err(),
        "a tenant at its new ceiling must be refused, not given another 5 rows: {refused:?}"
    );
}

/// The ceiling applies to new rows only: an overwrite of a row the tenant
/// already holds is admitted at the ceiling, because it changes no
/// cardinality.
#[tokio::test]
async fn an_overwrite_at_the_new_limit_is_still_admitted() {
    let dir = tempfile::tempdir().unwrap();
    let shards = set(dir.path()).await;
    for id in 0..5u64 {
        shards
            .vset("q", id, vec_at(id), TENANT, None, None)
            .await
            .unwrap();
    }
    shards
        .vset("q", 3, vec_at(300), TENANT, Some(5), None)
        .await
        .expect("an overwrite adds no row and must pass at the ceiling");
    assert_eq!(shards.tenant_vector_count(TENANT), 5);
    assert_eq!(shards.vget("q", 3).await.unwrap(), Some(vec_at(300)));
}

/// A delete credits the counter whether or not a limit existed when the row
/// was written, so raising and lowering a ceiling cannot strand a tenant
/// above it for ever.
#[tokio::test]
async fn a_delete_of_an_unlimited_write_frees_a_slot_at_the_new_limit() {
    let dir = tempfile::tempdir().unwrap();
    let shards = set(dir.path()).await;
    for id in 0..5u64 {
        shards
            .vset("q", id, vec_at(id), TENANT, None, None)
            .await
            .unwrap();
    }
    assert!(
        shards.vdel("q", 0, TENANT).await.unwrap(),
        "the row was there"
    );
    assert_eq!(shards.tenant_vector_count(TENANT), 4);
    shards
        .vset("q", 100, vec_at(100), TENANT, Some(5), None)
        .await
        .expect("the freed slot is usable under the ceiling");
    assert_eq!(shards.tenant_vector_count(TENANT), 5);
}
