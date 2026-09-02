//! Adversarial pass: each test states an invariant a user would assume without
//! being told, and tries to break it.
//!
//! Not a feature list. The question is only ever "can I make this engine give
//! a wrong answer with a confident face", so every assertion below is about
//! what a client observes, never about internal structure.

use skeg_vector::{DiskVamanaIndex, QuantKind, VectorVersion};

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
    // Best effort: a fold that succeeded installed a NEW generation and
    // reclaimed the superseded slot, so this file is legitimately gone. This
    // step only exists so the frozen file cannot outlive the test; it is not
    // the invariant, which is asserted below.
    let _ = std::fs::set_permissions(&vbin, saved);
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

/// `create_empty` on a directory that already holds an index used to succeed
/// and write an empty graph, vectors, CURRENT and WAL over it: an embedder
/// that "creates" on every open (the rigging adapter does, whenever its
/// sidecar is missing) silently wiped a populated index. A directory with a
/// tier file or a CURRENT pointer is an index, and creating over it is an
/// error, not a reset.
#[test]
fn create_empty_refuses_a_dir_that_already_holds_an_index() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("idx");
    let mut i = idx(&dir);
    i.insert(7, &v(7)).unwrap();
    drop(i);

    let err = match DiskVamanaIndex::create_empty_with_tier(&dir, DIM, 64, TIER) {
        Ok(_) => panic!("create over an index must fail"),
        Err(e) => e,
    };
    assert_eq!(err.kind(), std::io::ErrorKind::AlreadyExists, "{err}");
    let reopened = DiskVamanaIndex::open(&dir).unwrap();
    assert_eq!(reopened.len(), 1, "the existing index must be untouched");
}

// ── the vector version invariant ──────────────────────────────────────────────
//
// A row can exist in more than one place at once, and until now "which copy is
// live" was inferred from position: a higher LSM layer, a lower shard number,
// the order a map happened to be iterated in. Position is not identity. These
// pin the fact that replaces it.

/// A straggler must lose. An older copy of a row arriving after a newer one -
/// which is exactly what a reshard move that read before an overwrite and
/// wrote after looks like from here - must not become the value the index
/// serves, and must not become it at the next restart either.
#[test]
fn a_wal_replay_of_an_older_insert_does_not_overwrite_a_newer_one() {
    let d = tempfile::TempDir::new().unwrap();
    let mut i = idx(d.path());
    let newer = vec_at(7, 0);
    let older = vec_at(7, 1);
    i.insert_versioned(7, &newer, VectorVersion::new(5))
        .unwrap();
    i.insert_versioned(7, &older, VectorVersion::new(3))
        .unwrap();
    assert_eq!(
        i.get(7).unwrap().unwrap(),
        newer,
        "an older version must not overwrite a newer one"
    );
    assert_eq!(i.version_of(7), VectorVersion::new(5));
    drop(i);

    // And the WAL holds both records, so the replay decides it again.
    let i = DiskVamanaIndex::open_with_tier(d.path(), TIER).unwrap();
    assert_eq!(
        i.get(7).unwrap().unwrap(),
        newer,
        "the replay must reach the same verdict as the live write path"
    );
    assert_eq!(i.version_of(7), VectorVersion::new(5));
}

/// The same rule with the operations swapped: a delete is a write like any
/// other and an older insert cannot undo it. This is the shape that resurrects
/// a deleted row - an overlap replicating a row a concurrent delete removed.
#[test]
fn a_delete_at_version_n_is_not_undone_by_an_insert_at_version_n_minus_one() {
    let d = tempfile::TempDir::new().unwrap();
    let mut i = idx(d.path());
    i.insert_versioned(9, &v(9), VectorVersion::new(4)).unwrap();
    assert!(i.delete_versioned(9, VectorVersion::new(9)).unwrap());
    // The straggler.
    i.insert_versioned(9, &v(9), VectorVersion::new(8)).unwrap();
    assert!(i.get(9).unwrap().is_none(), "the delete must stand");
    assert_eq!(i.len(), 0);
    drop(i);

    let i = DiskVamanaIndex::open_with_tier(d.path(), TIER).unwrap();
    assert!(
        i.get(9).unwrap().is_none(),
        "and must still stand after a restart"
    );
    assert_eq!(i.len(), 0);
}

/// The version is a property of the ROW, so every operation that rewrites a
/// segment has to carry it: a flush into a run, a runs-only merge, a fold into
/// a new base, and the reopen that follows. A version dropped anywhere along
/// that chain silently reverts the row to legacy and hands the tie-break back
/// to position.
#[test]
fn versions_survive_a_flush_a_run_merge_and_a_fold() {
    let d = tempfile::TempDir::new().unwrap();
    let mut i = idx(d.path());
    let ver = |id: u64| VectorVersion::new(1000 + id);
    for id in 0..64 {
        i.insert_versioned(id, &v(id), ver(id)).unwrap();
    }
    flush(&mut i, d.path());
    for id in 64..128 {
        i.insert_versioned(id, &v(id), ver(id)).unwrap();
    }
    flush(&mut i, d.path());
    assert!(i.run_count() >= 2, "two runs, so the merge has work");

    if let Some(j) = i.merge_runs_begin().unwrap() {
        let b = j.build(d.path()).unwrap();
        i.merge_runs_finish(b).unwrap().expect_clean();
    }
    for id in 0..128 {
        assert_eq!(i.version_of(id), ver(id), "id {id} after the run merge");
    }

    fold(&mut i, d.path());
    for id in 0..128 {
        assert_eq!(i.version_of(id), ver(id), "id {id} after the fold");
    }
    drop(i);

    let i = DiskVamanaIndex::open_with_tier(d.path(), TIER).unwrap();
    for id in 0..128 {
        assert_eq!(i.version_of(id), ver(id), "id {id} after the reopen");
        assert_eq!(i.get(id).unwrap().unwrap(), v(id), "id {id} value");
    }
}

/// Write one record of each legacy WAL encoding by hand and make the engine
/// read it. A format bump that cannot open what is already on disk is not a
/// bump, it is a wipe - and the rows come back as legacy versions, which is
/// what makes the upgrade a no-op on data at rest.
///
/// The V3 promotion happens at the fold, never at open: opening must not
/// rewrite a file it was only asked to read.
#[test]
fn a_v2_store_opens_and_upgrades_to_v3_on_the_first_fold() {
    for (name, magic, framed) in [("v1", &b""[..], false), ("v2", &b"SKWL\x02"[..], true)] {
        let d = tempfile::TempDir::new().unwrap();
        {
            let _ = idx(d.path()); // graph, vectors, CURRENT
        }
        // Hand-built legacy WAL: [op=0][id u64][dim f32] (+ crc32c for V2).
        let mut wal = magic.to_vec();
        for id in 0..8u64 {
            let mut body = vec![0u8];
            body.extend_from_slice(&id.to_le_bytes());
            for x in v(id) {
                body.extend_from_slice(&x.to_le_bytes());
            }
            if framed {
                let c = crc32c::crc32c(&body);
                body.extend_from_slice(&c.to_le_bytes());
            }
            wal.extend_from_slice(&body);
        }
        std::fs::write(d.path().join("delta.log"), &wal).unwrap();

        let mut i = DiskVamanaIndex::open_with_tier(d.path(), TIER).unwrap();
        assert_eq!(i.len(), 8, "{name}: every legacy record must replay");
        for id in 0..8 {
            assert_eq!(i.get(id).unwrap().unwrap(), v(id), "{name}: id {id}");
            assert_eq!(
                i.version_of(id),
                VectorVersion::LEGACY,
                "{name}: a legacy record carries no version"
            );
        }
        // Opening READ the file; it must not have rewritten it.
        assert_eq!(
            std::fs::read(d.path().join("delta.log")).unwrap(),
            wal,
            "{name}: open must not rewrite the WAL"
        );

        fold(&mut i, d.path());
        let after = std::fs::read(d.path().join("delta.log")).unwrap();
        assert_eq!(
            &after[..5],
            b"SKWL\x03",
            "{name}: the fold is where the WAL becomes versioned"
        );
        for id in 0..8 {
            assert_eq!(i.get(id).unwrap().unwrap(), v(id), "{name}: id {id} folded");
        }
    }
}

/// A crash during the final append leaves a partial record. The decoder must
/// stop at it - the records before it are complete and durable, the partial
/// one never happened. Half-applying it (an id with no vector, a version with
/// no id) would be worse than losing it.
#[test]
fn a_truncated_v3_record_is_ignored_not_half_applied() {
    // Every prefix length of the second record, so the cut lands inside the id,
    // inside the version, inside the payload_ref tag and inside the vector.
    let full_len = 1 + 8 + 8 + 1 + DIM * 4 + 4;
    for cut in 1..full_len {
        let d = tempfile::TempDir::new().unwrap();
        {
            let _ = idx(d.path());
        }
        let rec = |id: u64| {
            let mut body = vec![0u8];
            body.extend_from_slice(&id.to_le_bytes());
            body.extend_from_slice(&(100 + id).to_le_bytes());
            body.push(0); // payload_ref: Unchanged
            for x in v(id) {
                body.extend_from_slice(&x.to_le_bytes());
            }
            let c = crc32c::crc32c(&body);
            body.extend_from_slice(&c.to_le_bytes());
            body
        };
        let mut wal = b"SKWL\x03".to_vec();
        wal.extend_from_slice(&rec(1));
        wal.extend_from_slice(&rec(2)[..cut]);
        std::fs::write(d.path().join("delta.log"), &wal).unwrap();

        let i = DiskVamanaIndex::open_with_tier(d.path(), TIER).unwrap();
        assert_eq!(
            i.get(1).unwrap().unwrap(),
            v(1),
            "cut {cut}: the complete record must apply"
        );
        assert_eq!(i.version_of(1), VectorVersion::new(101), "cut {cut}");
        assert!(
            i.get(2).unwrap().is_none(),
            "cut {cut}: the torn record must not apply at all"
        );
        assert_eq!(i.len(), 1, "cut {cut}");
    }
}

/// The version column is part of the generation, not a sidecar bolted on
/// beside it: a build that cannot write it must not publish the generation.
/// The trap being avoided is the one `load_attr` fell into - a sidecar that is
/// silently dropped when it does not fit, so the index comes back serving
/// zeros with a clean bill of health.
#[test]
fn a_fold_that_cannot_write_versions_bin_does_not_commit_a_new_generation() {
    use skeg_vector::failpoint::{WriteFailpoint, arm, disarm_all, fired};

    let d = tempfile::TempDir::new().unwrap();
    let mut i = idx(d.path());
    for id in 0..64 {
        i.insert_versioned(id, &v(id), VectorVersion::new(500 + id))
            .unwrap();
    }
    let job = i
        .consolidate_begin()
        .unwrap()
        .expect("there is work to fold");
    arm(WriteFailpoint::VersionsSidecarWrite);
    let built = job.build(d.path());
    disarm_all();
    assert!(
        fired(WriteFailpoint::VersionsSidecarWrite),
        "the failpoint never fired, so this test proved nothing"
    );
    assert!(
        built.is_err(),
        "a fold that cannot write the version column must fail, not publish"
    );

    // Nothing committed: the live index is untouched and every row is readable.
    for id in 0..64 {
        assert_eq!(i.get(id).unwrap().unwrap(), v(id), "id {id}");
        assert_eq!(i.version_of(id), VectorVersion::new(500 + id), "id {id}");
    }
    // And it comes back the same way.
    drop(i);
    let i = DiskVamanaIndex::open_with_tier(d.path(), TIER).unwrap();
    for id in 0..64 {
        assert_eq!(i.get(id).unwrap().unwrap(), v(id), "id {id} after reopen");
        assert_eq!(
            i.version_of(id),
            VectorVersion::new(500 + id),
            "id {id} after reopen"
        );
    }
}

/// `live_ids_with_versions` has two paths - a direct read of the base's own
/// column in the folded steady state, and the general one across every layer -
/// and they have to give the same answer. Two paths that agree today and
/// diverge later is how a cold start starts naming the wrong primary.
#[test]
fn live_ids_with_versions_agrees_across_the_layer_shapes() {
    let d = tempfile::TempDir::new().unwrap();
    let mut i = idx(d.path());
    let general = |i: &DiskVamanaIndex| {
        let mut want: Vec<(u64, VectorVersion)> = i
            .live_ids()
            .into_iter()
            .map(|id| (id, i.version_of(id)))
            .collect();
        want.sort_unstable();
        want
    };
    let taken = |i: &DiskVamanaIndex| {
        let mut got = i.live_ids_with_versions();
        got.sort_unstable();
        got
    };

    for id in 0..40 {
        i.insert_versioned(id, &v(id), VectorVersion::new(70 + id))
            .unwrap();
    }
    // Delta only.
    assert_eq!(taken(&i), general(&i));
    flush(&mut i, d.path());
    for id in 40..60 {
        i.insert_versioned(id, &v(id), VectorVersion::new(70 + id))
            .unwrap();
    }
    i.delete_versioned(3, VectorVersion::new(500)).unwrap();
    // A run, a delta and a tombstone at once.
    assert_eq!(taken(&i), general(&i));
    assert!(!taken(&i).iter().any(|&(id, _)| id == 3));

    fold(&mut i, d.path());
    // Folded: the direct path.
    assert_eq!(taken(&i), general(&i));
    assert_eq!(taken(&i).len(), 59);
}

/// The insert record says what the write did to the row's payload blob, and
/// the replay reads it back from the RECORD - not from the shape of a key, and
/// not by guessing from the row's version. A reference the reopen cannot see
/// is a reference the recovery cannot act on.
#[test]
fn wal_v3_round_trips_a_payload_ref() {
    use skeg_vector::PayloadRef;
    let d = tempfile::TempDir::new().unwrap();
    let refs = [
        (1u64, PayloadRef::Blob(7)),
        (2, PayloadRef::Cleared),
        (3, PayloadRef::Unchanged),
        // A sequence that needs all eight bytes, so a truncated field shows.
        (4, PayloadRef::Blob(0xfedc_ba98_7654_3210)),
    ];
    {
        let mut i = idx(d.path());
        for (id, r) in refs {
            i.insert_with_payload(id, &v(id), VectorVersion::new(id + 10), r)
                .unwrap();
            assert_eq!(i.payload_ref_of(id), r, "id {id}: before the reopen");
        }
    }
    let i = DiskVamanaIndex::open_with_tier(d.path(), TIER).unwrap();
    for (id, r) in refs {
        assert_eq!(i.get(id).unwrap().unwrap(), v(id), "id {id}");
        assert_eq!(i.payload_ref_of(id), r, "id {id}: after the reopen");
    }
    assert_eq!(
        i.payload_ref_of(99),
        PayloadRef::Unchanged,
        "a row this index has never seen says nothing about a payload"
    );
}

/// A record written before the field existed says nothing about the payload -
/// which is not the same as saying the row has none. `Unchanged` is the only
/// honest answer, and inventing `Cleared` there would delete a blob a legacy
/// store still serves.
#[test]
fn wal_v2_reads_as_payload_ref_unchanged() {
    use skeg_vector::PayloadRef;
    for (name, magic, framed) in [("v1", &b""[..], false), ("v2", &b"SKWL\x02"[..], true)] {
        let d = tempfile::TempDir::new().unwrap();
        {
            let _ = idx(d.path());
        }
        let mut wal = magic.to_vec();
        for id in 0..4u64 {
            let mut body = vec![0u8];
            body.extend_from_slice(&id.to_le_bytes());
            for x in v(id) {
                body.extend_from_slice(&x.to_le_bytes());
            }
            if framed {
                let c = crc32c::crc32c(&body);
                body.extend_from_slice(&c.to_le_bytes());
            }
            wal.extend_from_slice(&body);
        }
        std::fs::write(d.path().join("delta.log"), &wal).unwrap();

        let i = DiskVamanaIndex::open_with_tier(d.path(), TIER).unwrap();
        for id in 0..4u64 {
            assert_eq!(i.get(id).unwrap().unwrap(), v(id), "{name}: id {id}");
            assert_eq!(
                i.payload_ref_of(id),
                PayloadRef::Unchanged,
                "{name}: id {id} - a legacy record makes no claim about a payload"
            );
        }
    }
}

/// The WAL append is the commit point of a vector and the blob staged for it.
/// A failure there must leave the row absent - not present in RAM and missing
/// from the file it will be recovered from.
#[test]
fn a_wal_append_that_fails_leaves_the_row_absent() {
    use skeg_vector::failpoint::{WriteFailpoint, arm, disarm_all, fired};
    let d = tempfile::TempDir::new().unwrap();
    let mut i = idx(d.path());
    i.insert_versioned(1, &v(1), VectorVersion::new(1)).unwrap();

    arm(WriteFailpoint::DeltaWalAppend);
    let outcome = i.insert_versioned(2, &v(2), VectorVersion::new(1));
    disarm_all();
    assert!(
        fired(WriteFailpoint::DeltaWalAppend),
        "the failpoint never fired, so this test proved nothing"
    );
    assert!(outcome.is_err(), "an unappended write must be reported");
    assert!(i.get(2).unwrap().is_none(), "and must not be in RAM");
    assert_eq!(i.len(), 1);
    drop(i);

    let i = DiskVamanaIndex::open_with_tier(d.path(), TIER).unwrap();
    assert_eq!(
        i.get(1).unwrap().unwrap(),
        v(1),
        "the committed row survives"
    );
    assert!(
        i.get(2).unwrap().is_none(),
        "a row whose append failed came back from the WAL"
    );
}
