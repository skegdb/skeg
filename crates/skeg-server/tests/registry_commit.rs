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
