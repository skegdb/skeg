//! The commit windows, counted.
//!
//! A staged blob is unreadable until its commit lands, a post-commit failure
//! cannot be reported to the client, and a blob collected at open was already
//! invisible to every reader. All three are therefore states with NO observable
//! behaviour - which is exactly why they need a number, and why the number is
//! the only thing that can tell an operator the difference between "this
//! mechanism is idle" and "this mechanism is firing constantly".
//!
//! ONE test in this file, deliberately. The counters are process-wide, so a
//! second test in the same binary would run beside this one and move them; the
//! exact deltas below are only assertable because nothing else in this process
//! writes a vector. (That is the flaw already recorded against the existing
//! `PayloadIndexRebuilds` test, and this file is how it is avoided rather than
//! repeated.)

use bytes::Bytes;
use skeg_server::failpoint::{WriteFailpoint, arm_at, disarm_at, fired_at};
use skeg_server::shard::ShardSet;
use skeg_telemetry::{Counter, counter_value};

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

fn blob(text: &str) -> Bytes {
    Bytes::copy_from_slice(text.as_bytes())
}

#[tokio::test]
async fn every_commit_window_is_counted() {
    let dir = tempfile::TempDir::new().unwrap();
    let name = "ct";

    let staged0 = counter_value(Counter::PayloadBlobsStaged);
    let carried0 = counter_value(Counter::PayloadBlobsCarriedForward);
    let failed0 = counter_value(Counter::PayloadPostCommitFailures);
    let reclaimed0 = counter_value(Counter::PayloadBlobsReclaimedAtOpen);
    assert_eq!(
        (staged0, carried0, failed0, reclaimed0),
        (0, 0, 0, 0),
        "fixture: this binary must hold exactly one test, or the deltas below \
         are somebody else's writes"
    );

    {
        let shards = ShardSet::open(dir.path(), 1).unwrap();
        shards.vindex_create(name, DIM as u32, 0, 1).await.unwrap();

        // One write with a payload: one blob staged.
        shards
            .vset(name, 1, vec_at(1, 0), 0, None, Some(blob("colour=red")))
            .await
            .unwrap();
        assert_eq!(counter_value(Counter::PayloadBlobsStaged), 1);
        assert_eq!(counter_value(Counter::PayloadBlobsCarriedForward), 0);

        // A payload-less overwrite: the row's blob is carried to the new
        // version's key, so it is staged AND carried.
        shards
            .vset(name, 1, vec_at(1, 1), 0, None, None)
            .await
            .unwrap();
        assert_eq!(counter_value(Counter::PayloadBlobsStaged), 2);
        assert_eq!(counter_value(Counter::PayloadBlobsCarriedForward), 1);

        // A write with no payload on a row that has none: nothing to stage,
        // nothing to carry.
        shards
            .vset(name, 2, vec_at(2, 0), 0, None, None)
            .await
            .unwrap();
        assert_eq!(counter_value(Counter::PayloadBlobsStaged), 2);
        assert_eq!(counter_value(Counter::PayloadBlobsCarriedForward), 1);

        // A post-commit failure: reported as success, counted here.
        arm_at(WriteFailpoint::PayloadPostCommitCleanup, name);
        shards
            .vset(name, 1, vec_at(1, 2), 0, None, Some(blob("colour=blue")))
            .await
            .expect("post-commit failures do not fail the call");
        disarm_at(WriteFailpoint::PayloadPostCommitCleanup, name);
        assert!(
            fired_at(WriteFailpoint::PayloadPostCommitCleanup, name),
            "the failpoint never fired, so this test proved nothing"
        );
        assert_eq!(
            counter_value(Counter::PayloadPostCommitFailures),
            1,
            "a step after the commit point did not run and said nothing"
        );

        // And one commit that never landed, which leaves its staged blob for
        // the open below.
        arm_at(WriteFailpoint::VectorCommit, name);
        assert!(
            shards
                .vset(name, 3, vec_at(3, 0), 0, None, Some(blob("colour=green")))
                .await
                .is_err()
        );
        disarm_at(WriteFailpoint::VectorCommit, name);
        assert_eq!(counter_value(Counter::PayloadBlobsStaged), 4);
        assert_eq!(counter_value(Counter::PayloadBlobsReclaimedAtOpen), 0);
    }

    // The blob of the failed commit, and the one the post-commit cleanup did
    // not reclaim.
    let _shards = ShardSet::open(dir.path(), 1).unwrap();
    assert_eq!(
        counter_value(Counter::PayloadBlobsReclaimedAtOpen),
        2,
        "the open collected a different number of blobs than the windows above left"
    );
}
