//! A backup nobody has restored is not a backup.
//!
//! The procedure this pins is the file-level one an operator actually has:
//! quiesce, snapshot, copy the data directory, open the copy elsewhere. What
//! must hold on the copy is not "it starts" but everything a user would
//! notice: the same rows, the same vectors, a clean integrity report, and the
//! same answers to the same queries.
//!
//! The hot case is pinned too, and deliberately NOT as "it works": a copy
//! taken while writes are in flight may miss the newest rows, and the test
//! records which guarantee actually holds so nobody has to guess later.

use skeg_server::shard::ShardSet;
use skeg_vector::QuantKind;

const TIER: QuantKind = QuantKind::TurboQuant { bits: 2 };
const DIM: u32 = 32;
const ROWS: u64 = 3000;
const SHARDS: usize = 4;

fn vec_for(id: u64, generation: u64) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM as usize];
    for (i, s) in v.iter_mut().enumerate() {
        let t = (id as f32 * 0.37 + i as f32 * 0.11 + generation as f32 * 1.7).sin();
        *s = t;
    }
    v
}

/// Recursive directory copy - what `cp -R` does, which is what an operator
/// (or a snapshotting filesystem) does.
///
/// A file the writer removed between `read_dir` and `copy` (a segment
/// rotated away, a WAL replaced) is not part of the copy: the hot-copy test
/// is about what the copy CONTAINS being consistent, not about racing the
/// writer for a file that no longer exists. `cp -R` would print a warning
/// and go on; so does this. Any other error is real and propagates.
fn copy_dir(src: &std::path::Path, dst: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let to = dst.join(entry.file_name());
        let gone = |e: &std::io::Error| e.kind() == std::io::ErrorKind::NotFound;
        let is_dir = match entry.file_type() {
            Ok(t) => t.is_dir(),
            Err(e) if gone(&e) => continue,
            Err(e) => return Err(e),
        };
        let r = if is_dir {
            copy_dir(&entry.path(), &to)
        } else {
            std::fs::copy(entry.path(), &to).map(|_| ())
        };
        match r {
            Ok(()) => {}
            Err(e) if gone(&e) => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

async fn write_corpus(dir: &std::path::Path, generation: u64) {
    let shards = ShardSet::open_mode_with_workers(dir, SHARDS, false, TIER, 1).unwrap();
    shards.vindex_create("bk", DIM, 4, 1).await.unwrap();
    for id in 0..ROWS {
        shards
            .vset("bk", id, vec_for(id, generation), 0, None, None)
            .await
            .unwrap();
    }
    shards.vindex_consolidate("bk").await.unwrap();
    // The operator's quiesce step: flush what is in memory to disk.
    shards.write_snapshot_and_payload_indexes().await;
}

/// The whole procedure end to end: quiesce, copy, open the copy, CHECK, and
/// compare answers - not just row counts, which a half-restored index can
/// match by accident.
#[tokio::test]
async fn a_quiesced_copy_restores_with_identical_answers() {
    let src = tempfile::TempDir::new().unwrap();
    write_corpus(src.path(), 1).await;

    // Read the reference answers from the original.
    let origin = ShardSet::open_mode_with_workers(src.path(), SHARDS, false, TIER, 1).unwrap();
    let mut want = Vec::new();
    for q in 0..25u64 {
        let hits = origin
            .vsearch("bk", vec_for(q * 41, 1), 10, 0, 0, false, None)
            .await
            .unwrap();
        want.push(hits.into_iter().map(|(id, _, _)| id).collect::<Vec<_>>());
    }
    let want_rows: u64 = origin
        .vindex_list()
        .await
        .unwrap()
        .iter()
        .filter(|r| r.name == "bk")
        .map(|r| r.n_vectors)
        .sum();
    drop(origin);

    let dst = tempfile::TempDir::new().unwrap();
    let restored_dir = dst.path().join("restored");
    copy_dir(src.path(), &restored_dir).expect("copy the data directory");

    let restored = ShardSet::open_mode_with_workers(&restored_dir, SHARDS, false, TIER, 1).unwrap();

    // 1. Integrity, by the engine's own reckoning.
    let problems = restored
        .check("bk")
        .await
        .expect("CHECK on the restored copy");
    assert!(
        problems.is_empty(),
        "restored copy is not clean: {problems:?}"
    );

    // 2. Every row is there.
    let got_rows: u64 = restored
        .vindex_list()
        .await
        .unwrap()
        .iter()
        .filter(|r| r.name == "bk")
        .map(|r| r.n_vectors)
        .sum();
    assert_eq!(
        got_rows, want_rows,
        "restored copy has {got_rows} of {want_rows} rows"
    );

    // 3. Point reads return the same vectors.
    for id in [0u64, 1, ROWS / 2, ROWS - 1] {
        let v = restored
            .vget("bk", id)
            .await
            .unwrap()
            .expect("row present after restore");
        let expect = vec_for(id, 1);
        let dot: f32 = v.iter().zip(&expect).map(|(a, b)| a * b).sum();
        let norm = |x: &[f32]| x.iter().map(|a| a * a).sum::<f32>().sqrt();
        assert!(
            dot / (norm(&v) * norm(&expect)) > 0.999,
            "id {id} came back different after restore"
        );
    }

    // 4. The same queries give the same answers - the check that a merely
    //    openable copy cannot pass.
    for (i, expected) in want.iter().enumerate() {
        let hits = restored
            .vsearch("bk", vec_for(i as u64 * 41, 1), 10, 0, 0, false, None)
            .await
            .unwrap();
        let got: Vec<u64> = hits.into_iter().map(|(id, _, _)| id).collect();
        assert_eq!(
            &got, expected,
            "query {i} answers differently after restore"
        );
    }
}

/// A copy taken WHILE writes are in flight. This does not assert that the
/// newest rows survive - they may not, and pretending otherwise is how a
/// backup procedure lies. What it does assert is that the copy is never
/// CORRUPT: it opens, it passes CHECK, and every row it does hold is a row
/// that was really written. That is the guarantee an operator can rely on
/// without stopping the writer.
#[tokio::test]
async fn a_hot_copy_is_never_corrupt_even_if_it_is_behind() {
    let src = tempfile::TempDir::new().unwrap();
    write_corpus(src.path(), 1).await;

    let shards = ShardSet::open_mode_with_workers(src.path(), SHARDS, false, TIER, 1).unwrap();
    let dst = tempfile::TempDir::new().unwrap();
    let hot = dst.path().join("hot");

    // Copy in the middle of a second generation of writes.
    let writer = async {
        for id in 0..ROWS {
            shards
                .vset("bk", id, vec_for(id, 2), 0, None, None)
                .await
                .unwrap();
        }
    };
    let copier = async {
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        copy_dir(src.path(), &hot)
    };
    let (_, copied) = tokio::join!(writer, copier);
    copied.expect("copy while writing");

    let restored = ShardSet::open_mode_with_workers(&hot, SHARDS, false, TIER, 1).unwrap();
    let problems = restored.check("bk").await.expect("CHECK on the hot copy");
    assert!(problems.is_empty(), "hot copy is corrupt: {problems:?}");

    // Every row present must be one of the two generations actually written -
    // never a torn mixture of both.
    let norm = |x: &[f32]| x.iter().map(|a: &f32| a * a).sum::<f32>().sqrt();
    for id in [0u64, 7, ROWS / 3, ROWS - 1] {
        if let Some(v) = restored.vget("bk", id).await.unwrap() {
            let matches_a_generation = [1u64, 2].iter().any(|g| {
                let e = vec_for(id, *g);
                let dot: f32 = v.iter().zip(&e).map(|(a, b)| a * b).sum();
                dot / (norm(&v) * norm(&e)) > 0.999
            });
            assert!(
                matches_a_generation,
                "id {id} is neither generation: torn write"
            );
        }
    }
}
