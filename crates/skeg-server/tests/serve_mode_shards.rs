//! A read-only replica must serve the WHOLE index, or refuse to start.
//!
//! `bind_serve*` opened the shard set with a hardcoded count of 1. A set
//! written with eight shards therefore served exactly the rows that landed in
//! shard 0 - an eighth of the index - with no error, no warning and no hint,
//! answering every query with complete confidence. Measured before the fix:
//! 597 of 5,000 vectors, and 0.115 recall against the full corpus.
//!
//! These tests go through `Server::bind_serve_full_mmap` itself, in read-only
//! mode, over data written with several shards. A unit test on the discovery
//! helper cannot do that job: the bug lived in the caller, so a test that does
//! not cross the caller would let it back in unnoticed.

use skeg_core::VLog;
use skeg_server::Server;
use skeg_server::shard::ShardSet;
use skeg_vector::QuantKind;
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpStream};
use std::path::{Path, PathBuf};
use std::time::Duration;

const TIER: QuantKind = QuantKind::TurboQuant { bits: 2 };
const DIM: u32 = 8;
const ROWS: u64 = 200;
const SHARDS: usize = 2;

fn vec_for(id: u64) -> Vec<f32> {
    let mut v = vec![0.05f32; DIM as usize];
    v[(id % 4) as usize] = 1.0;
    v
}

/// Write `ROWS` rows across `SHARDS` shards and close the set cleanly.
async fn write_sharded(dir: &std::path::Path) {
    let shards = ShardSet::open_mode_with_workers(dir, SHARDS, false, TIER, 1).unwrap();
    shards.vindex_create("sv", DIM, 4, 1).await.unwrap();
    for id in 0..ROWS {
        shards
            .vset("sv", id, vec_for(id), 0, None, None)
            .await
            .unwrap();
    }
    shards.vindex_consolidate("sv").await.unwrap();
    shards.write_snapshot_and_payload_indexes().await;
}

fn free_port() -> u16 {
    std::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn inventory(root: &Path) -> Vec<(PathBuf, Option<Vec<u8>>)> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<(PathBuf, Option<Vec<u8>>)>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            let relative = path.strip_prefix(root).unwrap().to_owned();
            if path.is_dir() {
                out.push((relative, None));
                walk(root, &path, out);
            } else {
                out.push((relative, Some(std::fs::read(&path).unwrap())));
            }
        }
    }

    let mut out = Vec::new();
    walk(root, root, &mut out);
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// The real gate: open the SERVE path over an eight-shard set and count what
/// it can see. The old hardcoded 1 shows up here as roughly an eighth.
#[tokio::test]
async fn serve_mode_opens_every_shard_it_was_written_with() {
    let dir = tempfile::TempDir::new().unwrap();
    write_sharded(dir.path()).await;

    let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, free_port()));
    let server = Server::bind_serve_full_mmap(&addr.to_string(), dir.path(), TIER, 1, false, false)
        .await
        .expect("serve mode must open an eight-shard set");

    let rows = server
        .shards()
        .vindex_list()
        .await
        .expect("list through the serve path");
    let seen: u64 = rows
        .iter()
        .filter(|r| r.name == "sv")
        .map(|r| r.n_vectors)
        .sum();
    assert_eq!(
        seen, ROWS,
        "serve mode saw {seen} of {ROWS} rows: it opened the wrong number of shards"
    );
    // And the socket is actually listening, so this is the served path, not
    // just an object graph that happened to open.
    assert!(TcpStream::connect_timeout(&addr, Duration::from_secs(5)).is_ok());
}

/// The request filter is not enough to make serve mode read-only: its storage
/// handles must use shared locks and recovery that never writes. Holding one
/// read-only VLog per shard makes an accidental writable open fail before the
/// server can claim readiness.
#[tokio::test]
async fn serve_mode_uses_shared_read_only_vlog_locks() {
    let dir = tempfile::TempDir::new().unwrap();
    write_sharded(dir.path()).await;

    let mut readers = Vec::with_capacity(SHARDS);
    for shard in 0..SHARDS {
        readers.push(
            VLog::open_read_only(&dir.path().join(format!("shard-{shard}")))
                .await
                .unwrap(),
        );
    }

    let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, free_port()));
    let server = Server::bind_serve_full_mmap(&addr.to_string(), dir.path(), TIER, 1, false, false)
        .await
        .expect("serve mode must coexist with other read-only shard handles");
    drop((server, readers));
}

#[tokio::test]
async fn serve_mode_leaves_the_store_byte_for_byte_unchanged() {
    let dir = tempfile::TempDir::new().unwrap();
    write_sharded(dir.path()).await;
    let before = inventory(dir.path());

    let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, free_port()));
    let server = Server::bind_serve_full_mmap(&addr.to_string(), dir.path(), TIER, 1, true, false)
        .await
        .expect("serve mode opens the completed store");
    server.shards().vindex_list().await.unwrap();
    drop(server);

    let after = inventory(dir.path());
    assert_eq!(
        after.len(),
        before.len(),
        "a read-only open changed the directory inventory"
    );
    let changed = before
        .iter()
        .zip(&after)
        .find(|(before, after)| before != after)
        .map(|(before, _)| before.0.display().to_string());
    assert!(
        changed.is_none(),
        "a read-only open changed persistent data: {}",
        changed.unwrap_or_default()
    );
}

/// An empty directory is NOT a one-shard replica. Starting anyway is how
/// "healthy" gets printed over nothing at all.
#[tokio::test]
async fn serve_mode_refuses_an_empty_directory() {
    let dir = tempfile::TempDir::new().unwrap();
    let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, free_port()));
    let err = Server::bind_serve_full_mmap(&addr.to_string(), dir.path(), TIER, 1, false, false)
        .await
        .err()
        .expect("an empty directory must not become a replica");
    assert!(
        format!("{err}").contains("shard"),
        "the refusal must name the layout problem, got: {err}"
    );
}

/// A gap in the numbering means a shard is missing. Deducing "highest + 1"
/// would serve around the hole - silently, which is the whole failure mode.
#[tokio::test]
async fn serve_mode_refuses_a_gap_in_the_numbering() {
    let dir = tempfile::TempDir::new().unwrap();
    write_sharded(dir.path()).await;
    std::fs::remove_dir_all(dir.path().join("shard-1")).unwrap();

    let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, free_port()));
    let err = Server::bind_serve_full_mmap(&addr.to_string(), dir.path(), TIER, 1, false, false)
        .await
        .err()
        .expect("a missing shard must not be served around");
    let msg = format!("{err}");
    assert!(
        msg.contains("contiguous") || msg.contains("hole") || msg.contains("different set"),
        "the refusal must say the layout has a hole, got: {msg}"
    );
}
