//! Release gate for the composition of ingest and graph maintenance.
//!
//! This test is ignored because its assertion is environmental: run it under
//! a real Linux cgroup, via `scripts/check-maintenance-cgroup.sh`. Running it
//! on a developer shell without `memory.max` would prove nothing.

use std::time::Duration;

use skeg_server::memory::Budget;
use skeg_server::shard::{ShardError, ShardSet};

fn setting(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn row(id: usize, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|column| ((id.wrapping_mul(31) ^ column.wrapping_mul(17)) % 997) as f32 / 997.0)
        .collect()
}

#[tokio::test]
#[ignore = "must run through scripts/check-maintenance-cgroup.sh"]
async fn ingest_and_fold_stay_alive_under_the_cgroup_budget() {
    let dim = setting("SKEG_CGROUP_DIM", 128);
    let declared_limit = setting("SKEG_CGROUP_LIMIT_MIB", 0);
    assert!(
        matches!(declared_limit, 256 | 512),
        "runner must name the real 256/512 MiB cgroup limit"
    );
    let base_rows = setting(
        "SKEG_CGROUP_BASE_ROWS",
        if declared_limit == 256 {
            20_000
        } else {
            40_000
        },
    );
    let ingest_rows = setting("SKEG_CGROUP_INGEST_ROWS", 5_000);

    let dir = tempfile::TempDir::new().unwrap();
    let shards = ShardSet::open(dir.path(), 1).unwrap();
    let Budget::Room(room) = shards.memory().budget() else {
        panic!("the process is not observing a finite readable cgroup budget");
    };
    assert!(
        room < declared_limit as u64 * 1024 * 1024,
        "the governor did not subtract live usage and its reserve: room={room}"
    );

    shards
        .vindex_create("bounded", dim as u32, 0, 1)
        .await
        .unwrap();
    for id in 0..base_rows {
        shards
            .vset("bounded", id as u64, row(id, dim), 0, None, None)
            .await
            .unwrap();
    }

    let fold_shards = shards.clone();
    let fold = tokio::spawn(async move { fold_shards.vindex_consolidate("bounded").await });

    // Do not merely race two futures and call that concurrency. The byte
    // reservation is acquired before the fold snapshot and lives through the
    // blocking build, so a non-zero value proves ingest starts inside that
    // window. A conservative estimate may refuse the fold under 256 MiB; that
    // is also a valid bounded outcome, but this release gate deliberately uses
    // a shape which must fit so it exercises the composed peak.
    tokio::time::timeout(Duration::from_secs(30), async {
        while shards.memory().reserved_bytes() == 0 && !fold.is_finished() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("fold did not enter admission in 30 seconds");
    assert!(
        shards.memory().reserved_bytes() > 0,
        "fold was refused or completed before the overlap window"
    );

    let mut accepted = Vec::new();
    for id in base_rows..base_rows + ingest_rows {
        match shards
            .vset("bounded", id as u64, row(id, dim), 0, None, None)
            .await
        {
            Ok(()) => accepted.push(id),
            Err(ShardError::Admission(_)) => {
                // A typed refusal is the governor protecting the cgroup. The
                // process remaining alive and the previously accepted rows
                // remaining readable are the contract under pressure.
            }
            Err(other) => panic!("ingest failed outside admission: {other}"),
        }
    }
    fold.await.unwrap().expect("admitted fold failed");
    assert!(!accepted.is_empty(), "fold starved every concurrent write");
    assert!(
        shards.vget("bounded", 0).await.unwrap().is_some(),
        "fold lost a row accepted before its snapshot"
    );
    let last = *accepted.last().unwrap();
    assert!(
        shards.vget("bounded", last as u64).await.unwrap().is_some(),
        "fold lost a row accepted during its build"
    );

    shards.shutdown().await.expect("durable shutdown barrier");
    let reopened = ShardSet::open(dir.path(), 1).unwrap();
    assert!(reopened.vget("bounded", 0).await.unwrap().is_some());
    assert!(
        reopened
            .vget("bounded", last as u64)
            .await
            .unwrap()
            .is_some()
    );
    reopened.shutdown().await.unwrap();
}
