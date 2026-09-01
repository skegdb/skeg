//! Adversarial pass: each test states an invariant a user would assume without
//! being told, and tries to break it.
//!
//! Not a feature list. The question is only ever "can I make this engine give
//! a wrong answer with a confident face", so every assertion below is about
//! what a client observes, never about internal structure.

use skeg_vector::{DiskVamanaIndex, QuantKind};

const TIER: QuantKind = QuantKind::TurboQuant { bits: 2 };
const DIM: usize = 16;

/// Unique per id, and deliberately so: a fixture whose vectors collide makes
/// the row under test tie with its twins, and an assertion about which one
/// came back then cannot fail. That has bitten this suite twice.
fn vec_at(id: u64, phase: u64) -> Vec<f32> {
    let mut s = ((id << 3) | phase) | 1;
    let mut x = vec![0f32; DIM];
    for slot in x.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *slot = if s % 2 == 0 { 0.5 } else { -0.5 };
    }
    // A strong component so the neighbourhoods are distinguishable.
    x[(id % DIM as u64) as usize] = 1.0;
    x
}

fn v(id: u64) -> Vec<f32> {
    vec_at(id, 0)
}

fn idx(dir: &std::path::Path) -> DiskVamanaIndex {
    let mut i = DiskVamanaIndex::create_empty_with_tier(dir, DIM, 64, TIER).unwrap();
    i.set_auto_flush(false);
    i
}

fn flush(i: &mut DiskVamanaIndex, dir: &std::path::Path) {
    if let Some(j) = i.flush_begin().unwrap() {
        let b = j.build(dir).unwrap();
        i.flush_finish(b).unwrap().expect_clean();
    }
}

fn fold(i: &mut DiskVamanaIndex, dir: &std::path::Path) {
    if let Some(j) = i.consolidate_begin().unwrap() {
        let b = j.build(dir).unwrap();
        let _ = i.consolidate_finish(b).unwrap();
    }
}

/// A delete must survive every ordering of the maintenance operations that
/// can follow it. Twelve orderings, one deleted id, one question.
#[test]
fn a_deleted_row_never_comes_back_whatever_maintenance_runs_next() {
    for (n, ops) in [
        "flush",
        "fold",
        "flush,fold",
        "fold,flush",
        "flush,flush,fold",
        "fold,fold",
        "flush,fold,flush",
        "flush,fold,fold",
    ]
    .iter()
    .enumerate()
    {
        let tmp = tempfile::TempDir::new().unwrap();
        let mut i = idx(tmp.path());
        for id in 0..200u64 {
            i.insert(id, &v(id)).unwrap();
        }
        flush(&mut i, tmp.path());
        let doomed: Vec<u64> = (0..200).step_by(17).collect();
        for &id in &doomed {
            i.delete(id).unwrap();
        }
        // More writes after the delete, so the delete is not the last word in
        // the WAL and has to survive being replayed alongside them.
        for id in 500..600u64 {
            i.insert(id, &v(id)).unwrap();
        }
        for op in ops.split(',') {
            match op {
                "flush" => flush(&mut i, tmp.path()),
                "fold" => fold(&mut i, tmp.path()),
                _ => unreachable!(),
            }
        }
        for &id in &doomed {
            assert!(
                i.get(id).unwrap().is_none(),
                "case {n} ({ops}): id {id} came back live"
            );
        }
        drop(i);
        let re = DiskVamanaIndex::open_with_tier(tmp.path(), TIER).unwrap();
        for &id in &doomed {
            assert!(
                re.get(id).unwrap().is_none(),
                "case {n} ({ops}): id {id} came back after a reopen"
            );
        }
    }
}

/// A search must never return an id whose row was deleted, at any layer.
#[test]
fn search_never_returns_a_deleted_id() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut i = idx(tmp.path());
    for id in 0..400u64 {
        i.insert(id, &v(id)).unwrap();
    }
    flush(&mut i, tmp.path());
    fold(&mut i, tmp.path());
    // Delete across all three layers: base (folded), a fresh run, the delta.
    for id in 400..500u64 {
        i.insert(id, &v(id)).unwrap();
    }
    flush(&mut i, tmp.path());
    for id in 500..600u64 {
        i.insert(id, &v(id)).unwrap();
    }
    let doomed: Vec<u64> = (0..600).step_by(7).collect();
    for &id in &doomed {
        i.delete(id).unwrap();
    }
    let gone: std::collections::HashSet<u64> = doomed.iter().copied().collect();
    for probe in (0..600).step_by(3) {
        for (hit, _) in i.search(&v(probe), 20).unwrap() {
            assert!(
                !gone.contains(&hit),
                "search for {probe} returned deleted id {hit}"
            );
        }
    }
}

/// An overwrite must be all-or-nothing to a reader: the value a search scores
/// and the value a point read returns are the same version.
#[test]
fn an_overwrite_is_never_half_visible() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut i = idx(tmp.path());
    for id in 0..300u64 {
        i.insert(id, &v(id)).unwrap();
    }
    flush(&mut i, tmp.path());
    fold(&mut i, tmp.path());

    // Rewrite every id with a vector from a different neighbourhood.
    // Phase 1: a different vector for the same id, still unique.
    let new = |id: u64| vec_at(id, 1);
    for id in 0..300u64 {
        i.insert(id, &new(id)).unwrap();
    }
    flush(&mut i, tmp.path());

    // The fixture must make each row its own nearest neighbour, or the
    // assertion below cannot fail.
    for id in (0..300u64).step_by(11) {
        let top = i.search(&new(id), 1).unwrap();
        assert_eq!(top[0].0, id, "fixture: {id} is not unique, got {top:?}");
    }
    for id in (0..300u64).step_by(11) {
        let stored = i.get(id).unwrap().expect("still there");
        assert_eq!(
            stored,
            new(id),
            "point read returned the old version of {id}"
        );
        // And the search must score it as the NEW vector: querying with the
        // new one must find it at the top.
        let hits = i.search(&new(id), 5).unwrap();
        assert!(
            hits.iter().any(|h| h.0 == id),
            "search with the new vector of {id} does not find it: {hits:?}"
        );
    }
}

/// A run whose data file is truncated must be refused at open, not read short.
/// A short read is a silently smaller index.
#[test]
fn a_truncated_run_is_refused_not_read_short() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut i = idx(tmp.path());
    for id in 0..300u64 {
        i.insert(id, &v(id)).unwrap();
    }
    flush(&mut i, tmp.path());
    let before = i.len();
    drop(i);

    let run = std::fs::read_dir(tmp.path())
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .find(|p| {
            p.is_dir()
                && p.file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with("run-"))
        })
        .expect("a run to damage");
    let vbin = run.join("vectors.bin");
    let bytes = std::fs::read(&vbin).unwrap();
    std::fs::write(&vbin, &bytes[..bytes.len() - DIM * 4 * 10]).unwrap();

    match DiskVamanaIndex::open_with_tier(tmp.path(), TIER) {
        Err(_) => {} // refused: correct
        Ok(re) => panic!(
            "a truncated run opened anyway, reporting {} rows of {before}",
            re.len()
        ),
    }
}

/// Deleting an id and inserting it again must give the NEW value, not the
/// tombstone and not the old row. The tombstone has to stop applying to a row
/// that is deliberately back.
#[test]
fn a_reinserted_id_is_alive_again_with_the_new_value() {
    for (n, ops) in ["", "flush", "fold", "flush,fold", "fold,flush"]
        .iter()
        .enumerate()
    {
        let tmp = tempfile::TempDir::new().unwrap();
        let mut i = idx(tmp.path());
        for id in 0..200u64 {
            i.insert(id, &v(id)).unwrap();
        }
        flush(&mut i, tmp.path());
        fold(&mut i, tmp.path());

        let back: Vec<u64> = (0..200).step_by(13).collect();
        for &id in &back {
            i.delete(id).unwrap();
        }
        for &id in &back {
            i.insert(id, &vec_at(id, 2)).unwrap();
        }
        for op in ops.split(',').filter(|s| !s.is_empty()) {
            match op {
                "flush" => flush(&mut i, tmp.path()),
                "fold" => fold(&mut i, tmp.path()),
                _ => unreachable!(),
            }
        }
        for &id in &back {
            let got = i.get(id).unwrap();
            assert_eq!(
                got.as_deref(),
                Some(vec_at(id, 2).as_slice()),
                "case {n} ({ops}): reinserted id {id} reads back wrong"
            );
        }
        drop(i);
        let re = DiskVamanaIndex::open_with_tier(tmp.path(), TIER).unwrap();
        for &id in &back {
            assert_eq!(
                re.get(id).unwrap().as_deref(),
                Some(vec_at(id, 2).as_slice()),
                "case {n} ({ops}): reinserted id {id} lost across a reopen"
            );
        }
    }
}

/// Degenerate k must not panic, over-return, or invent hits.
#[test]
fn degenerate_k_is_answered_not_crashed() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut i = idx(tmp.path());

    // Empty index first: every k is legal and the answer is nothing.
    for k in [0usize, 1, 10] {
        assert!(i.search(&v(1), k).unwrap().is_empty(), "empty index, k={k}");
    }

    for id in 0..7u64 {
        i.insert(id, &v(id)).unwrap();
    }
    flush(&mut i, tmp.path());
    assert!(
        i.search(&v(1), 0).unwrap().is_empty(),
        "k=0 must return nothing"
    );
    let all = i.search(&v(1), 1000).unwrap();
    assert!(
        all.len() <= 7,
        "k above the row count returned {} hits for 7 rows",
        all.len()
    );
    let mut ids: Vec<u64> = all.iter().map(|h| h.0).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), all.len(), "the same id came back twice: {all:?}");
}

/// Dimensions that are not a comfortable multiple of anything.
///
/// Two answers are acceptable and one is not. A tier that cannot PACK the
/// dimension must refuse it with a reason; one that can must then store and
/// find the row. What must never happen is an assertion: the crate builds
/// with `panic = "abort"`, so a dimension chosen by a caller was able to end
/// the process rather than return an error.
#[test]
fn awkward_dimensions_are_refused_or_work_but_never_abort() {
    for dim in [1usize, 2, 3, 4, 5, 7, 8, 16, 17, 31, 32, 33] {
        let tmp = tempfile::TempDir::new().unwrap();
        let mut i = match DiskVamanaIndex::create_empty_with_tier(tmp.path(), dim, 64, TIER) {
            Ok(i) => i,
            Err(e) => {
                assert!(
                    e.to_string().contains("divisible"),
                    "dim {dim} refused without saying why: {e}"
                );
                continue;
            }
        };
        i.set_auto_flush(false);
        let mk = |id: u64| -> Vec<f32> {
            let mut s = (id << 1) | 1;
            (0..dim)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    if s % 2 == 0 { 0.5 } else { -0.5 }
                })
                .collect()
        };
        for id in 0..64u64 {
            i.insert(id, &mk(id)).unwrap();
        }
        if let Some(j) = i.flush_begin().unwrap() {
            let b = j.build(tmp.path()).unwrap();
            i.flush_finish(b).unwrap().expect_clean();
        }
        for id in [0u64, 31, 63] {
            assert_eq!(
                i.get(id).unwrap().as_deref(),
                Some(mk(id).as_slice()),
                "dim {dim}: point read of {id} came back different"
            );
            let hits = i.search(&mk(id), 8).unwrap();
            assert!(
                !hits.is_empty(),
                "dim {dim}: search for a stored row returned nothing"
            );
        }
    }
}

/// A run built but never marked durable is what a crash between `build` and
/// `flush_finish` leaves. Its rows are still in the WAL, so a reopen must
/// bring them back through the replay - not lose them, and not read the
/// half-made run as if it were complete.
#[test]
fn an_unmarked_run_is_discarded_and_its_rows_replayed() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut i = idx(tmp.path());
    for id in 0..300u64 {
        i.insert(id, &v(id)).unwrap();
    }
    // Build the run but never finish: exactly the crash window.
    let job = i.flush_begin().unwrap().unwrap();
    let _built = job.build(tmp.path()).unwrap();
    drop(i);

    let re = DiskVamanaIndex::open_with_tier(tmp.path(), TIER).unwrap();
    assert_eq!(re.len(), 300, "rows lost across the unfinished flush");
    for id in (0..300u64).step_by(29) {
        assert_eq!(
            re.get(id).unwrap().as_deref(),
            Some(v(id).as_slice()),
            "id {id} did not come back through the WAL replay"
        );
    }
}

/// A merge whose directory removal fails must not leave the old runs as
/// LAYERS after a restart.
///
/// The fold was fixed to retire `run.ok` before unlinking; the merge was not,
/// so a directory it could not remove came back as a live run holding rows the
/// merged run already contains.
///
/// Honest about its own strength: this pins the property and does NOT
/// discriminate against the previous code. A merge does not compact the WAL,
/// so the tombstones survive it and mask the resurrected rows anyway - the
/// exposure needs a LATER compaction to drop them, and the fold now retires
/// those markers itself on its way past. The retire here is defence in depth
/// for the window between the two, not a fix for something this test can
/// currently make fail.
#[test]
fn a_merge_that_cannot_unlink_does_not_leave_the_old_runs_as_layers() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::TempDir::new().unwrap();
    let mut i = idx(tmp.path());

    // Two runs to merge, with a deleted row so a resurrection is visible.
    for id in 0..200u64 {
        i.insert(id, &v(id)).unwrap();
    }
    flush(&mut i, tmp.path());
    for id in 200..400u64 {
        i.insert(id, &v(id)).unwrap();
    }
    flush(&mut i, tmp.path());
    let doomed: Vec<u64> = (0..400).step_by(37).collect();
    for &id in &doomed {
        i.delete(id).unwrap();
    }
    let live = i.len();

    let old = tmp.path().join("run-0");
    let saved = std::fs::metadata(&old).unwrap().permissions();
    std::fs::set_permissions(&old, PermissionsExt::from_mode(0o555)).unwrap();
    let job = i.merge_runs_begin().unwrap().expect("two runs to merge");
    let b = job.build(tmp.path()).unwrap();
    let outcome = i.merge_runs_finish(b).unwrap();
    std::fs::set_permissions(&old, saved).unwrap();
    assert!(
        outcome.cleanup_error().is_some(),
        "the fixture must actually fail the cleanup"
    );

    drop(i);
    let re = DiskVamanaIndex::open_with_tier(tmp.path(), TIER).unwrap();
    assert_eq!(
        re.len(),
        live,
        "the merge changed the live set across a reopen"
    );
    for &id in &doomed {
        assert!(
            re.get(id).unwrap().is_none(),
            "id {id} was deleted before the merge and came back"
        );
    }
}

/// The INLINE fold must publish a new generation, not overwrite the live one.
///
/// `consolidate()` calls `save()`, which writes graph.vmn and then
/// vectors.bin straight into the slot CURRENT names. A failure between the
/// two leaves a base whose graph is the new one and whose vectors are the
/// old: torn, with no previous generation left to fall back to. The
/// background fold builds into a temp directory and installs it with one
/// atomic rename, which is what the generation slots exist for.
///
/// The failpoint writes nothing of its own: `vectors.bin` in the live slot is
/// made read-only, so the graph write succeeds and the vector write does not.
#[test]
fn the_inline_fold_does_not_tear_the_live_base() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::TempDir::new().unwrap();
    let mut i = idx(tmp.path());
    for id in 0..300u64 {
        i.insert(id, &v(id)).unwrap();
    }
    flush(&mut i, tmp.path());
    fold(&mut i, tmp.path());
    let live = i.len();

    // More to fold, so the next consolidate has real work.
    for id in 300..600u64 {
        i.insert(id, &v(id)).unwrap();
    }
    flush(&mut i, tmp.path());

    // Freeze the vectors file of whichever slot is live.
    let slot = std::fs::read_to_string(tmp.path().join("CURRENT")).unwrap();
    let vbin = tmp
        .path()
        .join(format!("g{}", slot.trim()))
        .join("vectors.bin");
    assert!(vbin.exists(), "the fixture must find the live vectors file");
    let saved = std::fs::metadata(&vbin).unwrap().permissions();
    std::fs::set_permissions(&vbin, PermissionsExt::from_mode(0o444)).unwrap();

    // Deliberately NOT asserting that the fold fails. Freezing the live
    // vectors file is a failpoint only for a fold that writes there; one that
    // builds beside the base and installs it never touches this file and
    // simply succeeds. Requiring the failure would have made this test stop
    // discriminating the moment it was fixed - which it did, on the first try.
    //
    // The invariant is the same either way: whatever the fold does, the base
    // that was committed before it must still be readable.
    let _ = i.consolidate();
    std::fs::set_permissions(&vbin, saved).unwrap();
    drop(i);

    // THE ASSERTION: a fold that could not complete must leave a base that
    // still opens, and still holds everything that was committed before it.
    let re = DiskVamanaIndex::open_with_tier(tmp.path(), TIER)
        .expect("a failed fold must leave a readable base behind");
    assert!(
        re.len() >= live,
        "the failed fold lost committed rows: {} of at least {live}",
        re.len()
    );
    for id in (0..300u64).step_by(23) {
        assert_eq!(
            re.get(id).unwrap().as_deref(),
            Some(v(id).as_slice()),
            "id {id} did not survive a failed inline fold"
        );
    }
}
