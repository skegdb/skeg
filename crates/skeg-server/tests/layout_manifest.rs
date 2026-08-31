//! The on-disk layout must be DECLARED, not inferred.
//!
//! Discovery-by-directory-scan is a guess that happens to be right. It was
//! wrong once already, in the worst possible way: serve mode assumed one shard
//! over an eight-shard store and answered every query with full confidence
//! over an eighth of the index. The scan is now fail-closed, which turns that
//! class of bug into a refusal - but a refusal still requires the engine to
//! deduce something the writer already knew.
//!
//! A manifest states it: format version, shard count, store identity. Anything
//! that disagrees with it is a startup error, and the scan survives only as a
//! one-way migration for stores written before this file existed.

use skeg_server::layout_manifest::{LayoutManifest, MAX_SHARDS, OpenMode, ShardCount};

/// A checked count for the test's own use. Panics on a bad one, which is what
/// a test wants: a fixture that cannot be built is a broken test, not a case.
fn sc(n: usize) -> ShardCount {
    ShardCount::checked(n, "test").unwrap()
}

fn rw(n: usize) -> OpenMode {
    OpenMode::ReadWrite {
        requested_shards: sc(n),
    }
}

/// A legacy store: shard directories and nothing declaring them.
fn legacy_store(root: &std::path::Path, shards: usize) {
    for i in 0..shards {
        std::fs::create_dir_all(root.join(format!("shard-{i}"))).unwrap();
    }
}

#[test]
fn a_new_store_declares_its_layout() {
    let dir = tempfile::TempDir::new().unwrap();
    let m = LayoutManifest::open_or_migrate(dir.path(), rw(8)).unwrap();
    assert_eq!(m.shard_count(), sc(8));
    assert!(
        dir.path().join("LAYOUT").exists(),
        "the manifest must be on disk before any shard is written"
    );
}

#[test]
fn the_manifest_round_trips_identity_and_shape() {
    let dir = tempfile::TempDir::new().unwrap();
    let first = LayoutManifest::open_or_migrate(dir.path(), rw(4)).unwrap();
    let again = LayoutManifest::open_or_migrate(dir.path(), rw(4)).unwrap();
    assert_eq!(first.shard_count(), again.shard_count());
    assert_eq!(
        first.store_uuid(),
        again.store_uuid(),
        "the store identity must survive a reopen"
    );
    // And it is an identity, not a constant: a different store differs.
    let other = tempfile::TempDir::new().unwrap();
    let elsewhere = LayoutManifest::open_or_migrate(other.path(), rw(4)).unwrap();
    assert_ne!(first.store_uuid(), elsewhere.store_uuid());
}

#[test]
fn a_shard_count_mismatch_refuses_to_open() {
    let dir = tempfile::TempDir::new().unwrap();
    LayoutManifest::open_or_migrate(dir.path(), rw(8)).unwrap();
    let err = LayoutManifest::open_or_migrate(dir.path(), rw(4))
        .expect_err("opening an 8-shard store as 4 must refuse");
    let msg = err.to_string();
    assert!(
        msg.contains('8') && msg.contains('4'),
        "the refusal must name both counts, got: {msg}"
    );
}

#[test]
fn a_corrupt_checksum_refuses_to_open() {
    let dir = tempfile::TempDir::new().unwrap();
    LayoutManifest::open_or_migrate(dir.path(), rw(2)).unwrap();
    let path = dir.path().join("LAYOUT");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[12] ^= 0xFF; // inside the shard count, checksum untouched
    std::fs::write(&path, &bytes).unwrap();

    let err = LayoutManifest::open_or_migrate(dir.path(), OpenMode::ReadOnly)
        .expect_err("a manifest that fails its checksum must not be trusted");
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
}

#[test]
fn a_truncated_manifest_refuses_to_open() {
    let dir = tempfile::TempDir::new().unwrap();
    LayoutManifest::open_or_migrate(dir.path(), rw(2)).unwrap();
    let path = dir.path().join("LAYOUT");
    let bytes = std::fs::read(&path).unwrap();
    std::fs::write(&path, &bytes[..bytes.len() - 3]).unwrap();
    let err = LayoutManifest::open_or_migrate(dir.path(), OpenMode::ReadOnly).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
}

#[test]
fn trailing_bytes_refuse_to_open() {
    // A longer file is not a newer format - the version field says the format.
    // Accepting a tail is how a future writer's extra data gets silently
    // ignored by an older reader.
    let dir = tempfile::TempDir::new().unwrap();
    LayoutManifest::open_or_migrate(dir.path(), rw(2)).unwrap();
    let path = dir.path().join("LAYOUT");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.extend_from_slice(b"more");
    std::fs::write(&path, &bytes).unwrap();
    let err = LayoutManifest::open_or_migrate(dir.path(), OpenMode::ReadOnly).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
}

#[test]
fn an_unknown_format_version_refuses_to_open() {
    // Forward compatibility is a refusal, not a guess. A store written by a
    // newer skeg must not be opened by an older one on the assumption that the
    // parts it recognises are still the whole truth.
    let dir = tempfile::TempDir::new().unwrap();
    LayoutManifest::open_or_migrate(dir.path(), rw(2)).unwrap();
    let path = dir.path().join("LAYOUT");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[8] = 99; // format_version
    let sum = crc32c::crc32c(&bytes[..40]);
    bytes[40..44].copy_from_slice(&sum.to_le_bytes()); // re-sign: a VALID v99
    std::fs::write(&path, &bytes).unwrap();

    let err = LayoutManifest::open_or_migrate(dir.path(), OpenMode::ReadOnly).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(
        err.to_string().contains("99"),
        "the refusal must name the version it cannot read"
    );
}

#[test]
fn a_wrong_magic_refuses_to_open() {
    let dir = tempfile::TempDir::new().unwrap();
    LayoutManifest::open_or_migrate(dir.path(), rw(2)).unwrap();
    let path = dir.path().join("LAYOUT");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[..8].copy_from_slice(b"NOTSKEG!");
    let sum = crc32c::crc32c(&bytes[..40]);
    bytes[40..44].copy_from_slice(&sum.to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();
    let err = LayoutManifest::open_or_migrate(dir.path(), OpenMode::ReadOnly).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
}

#[test]
fn a_legacy_store_migrates_once_when_writable() {
    let dir = tempfile::TempDir::new().unwrap();
    legacy_store(dir.path(), 8);
    let m = LayoutManifest::open_or_migrate(dir.path(), rw(8)).unwrap();
    assert_eq!(m.shard_count(), sc(8));
    assert!(
        dir.path().join("LAYOUT").exists(),
        "a writable legacy store adopts a manifest"
    );
    // And the identity it minted is now stable.
    let again = LayoutManifest::open_or_migrate(dir.path(), rw(8)).unwrap();
    assert_eq!(m.store_uuid(), again.store_uuid());
}

#[test]
fn a_legacy_store_is_readable_without_being_written_to() {
    // Serve mode runs over a copy an operator may have mounted read-only, and
    // over stores this version did not write. Reading one must not mutate it.
    let dir = tempfile::TempDir::new().unwrap();
    legacy_store(dir.path(), 8);
    let m = LayoutManifest::open_or_migrate(dir.path(), OpenMode::ReadOnly).unwrap();
    assert_eq!(m.shard_count(), sc(8));
    assert!(
        !dir.path().join("LAYOUT").exists(),
        "a read-only open must leave the store exactly as it found it"
    );
}

#[test]
fn a_legacy_store_with_a_hole_still_refuses() {
    // The migration path inherits the scan's fail-closed rule; it does not get
    // a weaker one for being a migration.
    let dir = tempfile::TempDir::new().unwrap();
    legacy_store(dir.path(), 8);
    std::fs::remove_dir_all(dir.path().join("shard-3")).unwrap();
    let err = LayoutManifest::open_or_migrate(dir.path(), rw(8)).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
}

#[test]
fn a_legacy_count_that_contradicts_the_request_refuses() {
    let dir = tempfile::TempDir::new().unwrap();
    legacy_store(dir.path(), 8);
    let err = LayoutManifest::open_or_migrate(dir.path(), rw(2))
        .expect_err("migrating an 8-shard store as 2 would strand six shards");
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
}

#[test]
fn a_manifest_that_contradicts_the_directories_refuses() {
    // The two sources of truth must agree. If they do not, the one thing that
    // must NOT happen is picking a winner silently.
    let dir = tempfile::TempDir::new().unwrap();
    LayoutManifest::open_or_migrate(dir.path(), rw(4)).unwrap();
    for i in 0..4 {
        std::fs::create_dir_all(dir.path().join(format!("shard-{i}"))).unwrap();
    }
    std::fs::remove_dir_all(dir.path().join("shard-2")).unwrap();
    let err = LayoutManifest::open_or_migrate(dir.path(), rw(4))
        .expect_err("a missing shard directory contradicts the manifest");
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
}

#[test]
fn an_empty_read_only_directory_refuses() {
    // Nothing to serve, and nothing to deduce. The old code called this one
    // shard and started.
    let dir = tempfile::TempDir::new().unwrap();
    let err = LayoutManifest::open_or_migrate(dir.path(), OpenMode::ReadOnly).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
}

#[test]
fn a_zero_shard_manifest_refuses() {
    let dir = tempfile::TempDir::new().unwrap();
    LayoutManifest::open_or_migrate(dir.path(), rw(2)).unwrap();
    let path = dir.path().join("LAYOUT");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[12..16].copy_from_slice(&0u32.to_le_bytes());
    let sum = crc32c::crc32c(&bytes[..40]);
    bytes[40..44].copy_from_slice(&sum.to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();
    let err = LayoutManifest::open_or_migrate(dir.path(), OpenMode::ReadOnly).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
}

#[test]
fn unknown_feature_flags_refuse_to_open() {
    // A flag this build does not know about means the store uses something
    // this build cannot honour. Opening it anyway is the silent-wrong-answer
    // failure mode all over again.
    let dir = tempfile::TempDir::new().unwrap();
    LayoutManifest::open_or_migrate(dir.path(), rw(2)).unwrap();
    let path = dir.path().join("LAYOUT");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[32..40].copy_from_slice(&(1u64 << 17).to_le_bytes());
    let sum = crc32c::crc32c(&bytes[..40]);
    bytes[40..44].copy_from_slice(&sum.to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();
    let err = LayoutManifest::open_or_migrate(dir.path(), OpenMode::ReadOnly).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
}

// ---- what the parser must survive ----
//
// `LAYOUT` is bytes on disk, and bytes on disk are not trusted input just
// because this process wrote them last time. A restored backup, a shared
// volume, a corrupt sector, an operator with a hex editor: the parser is the
// boundary, and its job is to refuse rather than to cope.
//
// The checksum is NOT a defence against any of this. It detects accidental
// corruption; anyone who can write the file can recompute it, which is
// exactly what these tests do.

#[test]
fn an_enormous_manifest_is_refused_without_reading_it_all() {
    // `std::fs::read` allocates the whole file BEFORE anything can check its
    // length. A 512 MiB LAYOUT is a 512 MiB allocation at startup, and the
    // file is 44 bytes by construction - there is never a reason to hold more
    // than that in memory.
    let dir = tempfile::TempDir::new().unwrap();
    LayoutManifest::open_or_migrate(dir.path(), rw(2)).unwrap();
    let path = dir.path().join("LAYOUT");
    let big = vec![0u8; 512 * 1024 * 1024];
    std::fs::write(&path, &big).unwrap();

    let err = LayoutManifest::open_or_migrate(dir.path(), OpenMode::ReadOnly).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
}

#[test]
fn an_absurd_shard_count_is_refused() {
    // The count is a u32 read from the file and handed to a loop that spawns
    // one OS THREAD per shard. Four billion of them is not a layout, it is a
    // fork bomb with a checksum. Re-signed here, because a valid checksum is
    // exactly what an attacker with write access produces.
    let dir = tempfile::TempDir::new().unwrap();
    LayoutManifest::open_or_migrate(dir.path(), rw(2)).unwrap();
    let path = dir.path().join("LAYOUT");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
    let sum = crc32c::crc32c(&bytes[..40]);
    bytes[40..44].copy_from_slice(&sum.to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();

    let err = LayoutManifest::open_or_migrate(dir.path(), OpenMode::ReadOnly)
        .expect_err("a four-billion-shard layout must not be honoured");
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(
        err.to_string().contains("4294967295"),
        "the refusal must name the count it rejected, got: {err}"
    );
}

#[test]
fn a_legacy_scan_cannot_be_talked_into_an_absurd_count() {
    // The migration path reaches the same loop. A directory named for a huge
    // index is refused by the contiguity rule, but the bound is asserted here
    // too so the two paths cannot drift apart - which is how every P0 in this
    // engine happened.
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(dir.path().join("shard-0")).unwrap();
    std::fs::create_dir_all(dir.path().join("shard-4294967294")).unwrap();
    let err = LayoutManifest::open_or_migrate(dir.path(), OpenMode::ReadOnly).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
}

#[test]
fn a_shard_count_at_the_bound_is_still_accepted() {
    // The bound must be a bound, not a fence around the useful range. This
    // pins that a legitimate large deployment is not refused by it.
    let dir = tempfile::TempDir::new().unwrap();
    LayoutManifest::open_or_migrate(dir.path(), rw(2)).unwrap();
    let path = dir.path().join("LAYOUT");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[12..16].copy_from_slice(&(MAX_SHARDS as u32).to_le_bytes());
    let sum = crc32c::crc32c(&bytes[..40]);
    bytes[40..44].copy_from_slice(&sum.to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();

    let m = LayoutManifest::open_or_migrate(dir.path(), OpenMode::ReadOnly).unwrap();
    assert_eq!(m.shard_count().get(), MAX_SHARDS);
}

#[test]
fn a_layout_that_is_a_directory_is_an_error_not_a_migration() {
    // The absent-manifest branch is reached on NotFound. Any other I/O error
    // must propagate: treating "cannot read this" as "there is nothing here"
    // is how a store gets silently re-migrated on top of itself.
    let dir = tempfile::TempDir::new().unwrap();
    legacy_store(dir.path(), 2);
    std::fs::create_dir(dir.path().join("LAYOUT")).unwrap();
    let err = LayoutManifest::open_or_migrate(dir.path(), OpenMode::ReadOnly)
        .expect_err("an unreadable LAYOUT is not an absent one");
    assert_ne!(err.kind(), std::io::ErrorKind::NotFound);
}

#[test]
fn a_caller_asking_for_an_absurd_count_is_refused_too() {
    // The bound is not about untrusted FILES, it is about what reaches the
    // loop that spawns one OS thread and one channel per shard. A caller's
    // number reaches the same loop, so it goes through the same check - and
    // the type makes that structural rather than remembered.
    let dir = tempfile::TempDir::new().unwrap();
    let Err(err) = skeg_server::shard::ShardSet::open(dir.path(), MAX_SHARDS + 1) else {
        panic!("a caller must not be able to ask for more than the bound");
    };
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
}

#[test]
fn zero_shards_from_a_caller_is_an_error_not_a_panic() {
    // This used to be `assert!(n_shards >= 1)`: a panic, and under
    // `panic = "abort"` a dead process, for what is an ordinary bad argument.
    let dir = tempfile::TempDir::new().unwrap();
    let Err(err) = skeg_server::shard::ShardSet::open(dir.path(), 0) else {
        panic!("zero shards is not a store");
    };
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
}
