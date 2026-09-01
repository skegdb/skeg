//! The registry is the COMMIT RECORD, and this pins what that means.
//!
//! A vindex directory that no registry entry names is UNCOMMITTED: the store
//! must open without it, and it must not be silently reused. Both halves
//! matter. Refusing to start would turn an ordinary crash window into an
//! unbootable shard; reusing the directory would let a create destroy a fully
//! built index whose catalogue entry was lost.
//!
//! The order these operations publish in is what makes a crash survivable:
//!
//!   CREATE:  build the directory -> publish the registry -> acknowledge
//!   DROP:    publish the registry without the entry -> remove the directory
//!
//! Either way a crash in the middle leaves an orphan directory, which is
//! reclaimable. The reverse orders leave an acknowledged create with no
//! commit record, or a registry naming a directory that no longer exists -
//! and recovery opens every entry it lists, so that one does not start at all.

use skeg_server::shard::ShardSet;
use skeg_vector::QuantKind;

const TIER: QuantKind = QuantKind::TurboQuant { bits: 2 };

fn vec_for(id: u64) -> Vec<f32> {
    let mut v = vec![0.05f32; 8];
    v[(id % 4) as usize] = 1.0;
    v
}

/// A directory built but never committed: exactly what a crash between
/// `create_empty_with_tier` and the registry publish leaves behind.
async fn make_orphan(root: &std::path::Path, name: &str) {
    let staging = tempfile::TempDir::new().unwrap();
    {
        let shards = ShardSet::open_mode_with_workers(staging.path(), 1, false, TIER, 1).unwrap();
        shards.vindex_create(name, 8, 4, 1).await.unwrap();
        for id in 0..40u64 {
            shards
                .vset(name, id, vec_for(id), 0, None, None)
                .await
                .unwrap();
        }
        shards.write_snapshot_and_payload_indexes().await;
    }
    // Move the built directory across, leaving the destination's registry
    // untouched - it never learns about it.
    let from = staging
        .path()
        .join("shard-0")
        .join(format!("vindex-{name}"));
    let to = root.join("shard-0").join(format!("vindex-{name}"));
    std::fs::create_dir_all(to.parent().unwrap()).unwrap();
    let mut opts = fs_extra_copy(&from, &to);
    assert!(opts, "the orphan directory must be in place");
    opts = to.join("graph.vmn").exists() || to.join("g0").join("graph.vmn").exists();
    assert!(opts, "the orphan must look like a real index");
}

/// Minimal recursive copy - no dev-dependency for four lines.
fn fs_extra_copy(src: &std::path::Path, dst: &std::path::Path) -> bool {
    if !src.is_dir() {
        return false;
    }
    std::fs::create_dir_all(dst).unwrap();
    for e in std::fs::read_dir(src).unwrap() {
        let e = e.unwrap();
        let to = dst.join(e.file_name());
        if e.file_type().unwrap().is_dir() {
            fs_extra_copy(&e.path(), &to);
        } else {
            std::fs::copy(e.path(), &to).unwrap();
        }
    }
    true
}

#[tokio::test]
async fn an_orphan_directory_does_not_stop_the_store_from_opening() {
    let dir = tempfile::TempDir::new().unwrap();
    {
        let shards = ShardSet::open_mode_with_workers(dir.path(), 1, false, TIER, 1).unwrap();
        shards.vindex_create("live", 8, 4, 1).await.unwrap();
        shards
            .vset("live", 1, vec_for(1), 0, None, None)
            .await
            .unwrap();
        shards.write_snapshot_and_payload_indexes().await;
    }
    make_orphan(dir.path(), "ghost").await;

    // The store opens, and serves the committed index.
    let shards = ShardSet::open_mode_with_workers(dir.path(), 1, false, TIER, 1).unwrap();
    let rows = shards.vindex_list().await.unwrap();
    assert!(
        rows.iter().any(|r| r.name == "live"),
        "the committed index must still be there"
    );
    assert!(
        !rows.iter().any(|r| r.name == "ghost"),
        "an uncommitted directory is not an index"
    );
}

#[tokio::test]
async fn creating_over_an_orphan_refuses_instead_of_destroying_it() {
    // The orphan may be a fully built index whose registry entry was lost.
    // `create_empty_with_tier` calls `create_dir_all`, which happily succeeds
    // on an existing directory and then writes over graph, vectors, CURRENT
    // and the WAL. "Not committed" must not come to mean "free space".
    let dir = tempfile::TempDir::new().unwrap();
    {
        let shards = ShardSet::open_mode_with_workers(dir.path(), 1, false, TIER, 1).unwrap();
        shards.vindex_create("keep", 8, 4, 1).await.unwrap();
        shards.write_snapshot_and_payload_indexes().await;
    }
    make_orphan(dir.path(), "ghost").await;
    let vdir = dir.path().join("shard-0").join("vindex-ghost");
    let before: Vec<_> = std::fs::read_dir(&vdir)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();

    let shards = ShardSet::open_mode_with_workers(dir.path(), 1, false, TIER, 1).unwrap();
    let err = shards
        .vindex_create("ghost", 8, 4, 1)
        .await
        .expect_err("a create must not overwrite an unregistered directory");
    let msg = format!("{err}");
    assert!(
        msg.contains("registry") && msg.contains("ghost"),
        "the refusal must explain what it found, got: {msg}"
    );

    let after: Vec<_> = std::fs::read_dir(&vdir)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert_eq!(before.len(), after.len(), "the orphan must be untouched");
}

#[tokio::test]
async fn a_dropped_index_leaves_nothing_behind_and_the_name_is_reusable() {
    // The ordinary path, which the commit-first ordering must not break:
    // after a successful DROP the directory is gone and the name is free.
    let dir = tempfile::TempDir::new().unwrap();
    let shards = ShardSet::open_mode_with_workers(dir.path(), 1, false, TIER, 1).unwrap();
    shards.vindex_create("tmp", 8, 4, 1).await.unwrap();
    shards
        .vset("tmp", 1, vec_for(1), 0, None, None)
        .await
        .unwrap();
    shards.vindex_drop("tmp", 0).await.unwrap();

    assert!(
        !dir.path().join("shard-0").join("vindex-tmp").exists(),
        "DROP must remove the directory after committing"
    );
    shards
        .vindex_create("tmp", 8, 4, 1)
        .await
        .expect("the name is free again after a complete DROP");
}

/// Make the registry publish fail deterministically: `write_registry` creates
/// `<VINDEX_REGISTRY>.tmp`, and `File::create` cannot create a file where a
/// directory already sits.
fn block_registry_writes(shard_dir: &std::path::Path) {
    std::fs::create_dir_all(shard_dir.join("vindexes.registry.tmp")).unwrap();
}

fn unblock_registry_writes(shard_dir: &std::path::Path) {
    std::fs::remove_dir_all(shard_dir.join("vindexes.registry.tmp")).unwrap();
}

#[tokio::test]
async fn a_drop_whose_commit_fails_changes_nothing_the_client_can_see() {
    // The drop did the visible work FIRST - out of the resident map, quota
    // returned - and only then tried to commit. A failed commit therefore left
    // the index gone from this process while the registry still listed it, so
    // it reappeared at the next open with its quota already given back.
    //
    // Nothing observable may move until the commit lands.
    let dir = tempfile::TempDir::new().unwrap();
    let shard0 = dir.path().join("shard-0");
    let shards = ShardSet::open_mode_with_workers(dir.path(), 1, false, TIER, 1).unwrap();
    shards.vindex_create("keep", 8, 4, 1).await.unwrap();
    for id in 0..20u64 {
        shards
            .vset("keep", id, vec_for(id), 0, None, None)
            .await
            .unwrap();
    }

    block_registry_writes(&shard0);
    let err = shards
        .vindex_drop("keep", 0)
        .await
        .expect_err("a drop that cannot commit must fail");
    assert!(
        format!("{err}").contains("registry"),
        "the error must name the commit record, got: {err}"
    );

    // Still there, still readable, still listed.
    let rows = shards.vindex_list().await.unwrap();
    assert!(
        rows.iter().any(|r| r.name == "keep"),
        "a refused drop must leave the index listed"
    );
    assert!(
        shards.vget("keep", 7).await.unwrap().is_some(),
        "and readable"
    );
    assert!(
        shard0.join("vindex-keep").exists(),
        "and its directory intact"
    );

    // And it survives a restart, since the registry was never rewritten.
    unblock_registry_writes(&shard0);
    drop(shards);
    let again = ShardSet::open_mode_with_workers(dir.path(), 1, false, TIER, 1).unwrap();
    let rows = again.vindex_list().await.unwrap();
    let row = rows
        .iter()
        .find(|r| r.name == "keep")
        .expect("still there after restart");
    assert_eq!(row.n_vectors, 20, "with every row it had");
}

#[tokio::test]
async fn a_create_whose_commit_fails_is_not_acknowledged() {
    // The registry is the commit record, so a create that cannot publish one
    // has not happened. Reporting Done would make an orphan directory the
    // trace of a CONFIRMED operation, which is exactly what "not in the
    // registry means not committed" cannot survive.
    let dir = tempfile::TempDir::new().unwrap();
    let shard0 = dir.path().join("shard-0");
    let shards = ShardSet::open_mode_with_workers(dir.path(), 1, false, TIER, 1).unwrap();

    block_registry_writes(&shard0);
    let err = shards
        .vindex_create("ghost", 8, 4, 1)
        .await
        .expect_err("a create that cannot commit must fail");
    assert!(
        format!("{err}").contains("registry"),
        "the error must name the commit record, got: {err}"
    );

    // Not visible in this process either: the in-memory entry is rolled back.
    let rows = shards.vindex_list().await.unwrap();
    assert!(
        !rows.iter().any(|r| r.name == "ghost"),
        "a create that did not commit must not be servable"
    );

    // And after a restart it is still not an index - just an orphan directory,
    // which the create guard then refuses to overwrite.
    unblock_registry_writes(&shard0);
    drop(shards);
    let again = ShardSet::open_mode_with_workers(dir.path(), 1, false, TIER, 1).unwrap();
    assert!(
        !again
            .vindex_list()
            .await
            .unwrap()
            .iter()
            .any(|r| r.name == "ghost")
    );
}

#[tokio::test]
async fn a_dropped_index_stays_dropped_across_a_restart() {
    // The gap the other DROP tests walked straight past, because they either
    // recreated the same name immediately (which overwrites the stale entry)
    // or never restarted at all.
    //
    // `persist_registry` decides its content by starting from the existing
    // registry and keeping every entry whose DIRECTORY still exists - a rule
    // written when the directory was deleted BEFORE the publish. Committing
    // first inverted that: at publish time the directory is still there, the
    // entry is retained, and the "commit" republishes the index it was meant
    // to remove. Then the directory goes, and the registry names something
    // that no longer exists. Recovery opens every entry it lists, so the next
    // start fails outright - the exact failure the commit-first ordering was
    // introduced to prevent.
    let dir = tempfile::TempDir::new().unwrap();
    {
        let shards = ShardSet::open_mode_with_workers(dir.path(), 1, false, TIER, 1).unwrap();
        shards.vindex_create("keep", 8, 4, 1).await.unwrap();
        shards.vindex_create("gone", 8, 4, 1).await.unwrap();
        for id in 0..10u64 {
            shards
                .vset("gone", id, vec_for(id), 0, None, None)
                .await
                .unwrap();
            shards
                .vset("keep", id, vec_for(id), 0, None, None)
                .await
                .unwrap();
        }
        shards.vindex_drop("gone", 0).await.unwrap();
        shards.write_snapshot_and_payload_indexes().await;
    }

    // Reopen WITHOUT recreating the name.
    let shards = ShardSet::open_mode_with_workers(dir.path(), 1, false, TIER, 1)
        .expect("a shard whose index was dropped must still open");
    let rows = shards.vindex_list().await.unwrap();
    assert!(
        !rows.iter().any(|r| r.name == "gone"),
        "the dropped index must not come back: {rows:?}"
    );
    assert!(
        rows.iter().any(|r| r.name == "keep"),
        "and the one that was kept must still be there"
    );
}
