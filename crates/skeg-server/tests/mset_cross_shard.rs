//! B2 (audit/20): an `MSET` that spans shards must be refused BEFORE the
//! first write, not committed shard by shard.
//!
//! `MSET` is exposed as a Redis-compatible command, and what that name means
//! in public is all-or-nothing. The engine only ever promised it per shard:
//! keys route by `xxh3_64(key) % n_shards`, each shard's portion is its own
//! `VLog::set_many` commit, and the portions run in sequence. So a batch
//! whose keys land on two shards could have the first shard's keys already
//! durable when the second shard's disk quota - or its I/O - refused, and the
//! client was told `ERR`. A durable write acknowledged as a failure is the
//! one answer a store must never give.
//!
//! The fix is the prudent semantics of a sharded system, and the one Redis
//! Cluster itself uses: refuse the whole command up front, with `CROSSSLOT`,
//! before any of it runs. No two-phase commit, no cross-shard coordination,
//! and nothing that can be half-applied. A batch whose keys all route to one
//! shard keeps exactly what it had - one atomic `set_many`, its disk quota
//! reserved before the first byte.
//!
//! There are no hash tags: `{tag}` does not change where a key routes, so a
//! client cannot force two keys onto one shard the way a Redis Cluster client
//! can. Making `shard_for` tag-aware would move every existing key that
//! contains braces to a different shard, which for a store written by an
//! earlier version means data that is still on disk and no longer reachable.
//! See the report and `docs/adr-payload-transaction.md`.

use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use skeg_server::Server;
use skeg_server::admission::{AdmissionError, Retryability};
use skeg_server::shard::{ShardError, ShardSet, shard_for};

/// Enough shards that a search for a key on a given one terminates fast, and
/// few enough that the fan-out of an open is cheap in a test.
const N: usize = 4;

/// The first key of `prefix0, prefix1, ...` that routes to `shard`.
///
/// Searched rather than hard-coded: the hash is an implementation detail and
/// a literal that happens to land right today is a test that silently stops
/// testing anything the day it does not.
fn key_on_shard(prefix: &str, shard: usize, n: usize) -> Vec<u8> {
    for i in 0u32..100_000 {
        let k = format!("{prefix}{i}").into_bytes();
        if shard_for(&k, n) == shard {
            return k;
        }
    }
    panic!("no key with prefix {prefix:?} routes to shard {shard} of {n}");
}

fn scoped(tenant: u128, raw: &[u8]) -> Vec<u8> {
    let mut k = tenant.to_le_bytes().to_vec();
    k.extend_from_slice(raw);
    k
}

async fn n_keys_per_shard(shards: &ShardSet) -> Vec<u64> {
    shards
        .stats_per_shard()
        .await
        .expect("per-shard stats")
        .into_iter()
        .map(|s| s.n_keys)
        .collect()
}

// ── the refusal itself ──────────────────────────────────────────────────

/// The whole point: a batch whose keys span two shards writes NOTHING, on
/// EITHER shard, and says so with a typed refusal rather than an `ERR`
/// string.
///
/// The first key is deliberately the one that would have been committed
/// first: under the old code its shard's portion was already durable by the
/// time the second shard's was even looked at.
#[tokio::test]
async fn a_cross_shard_mset_writes_nothing_on_either_shard() {
    let dir = tempfile::TempDir::new().unwrap();
    let shards = ShardSet::open(dir.path(), N).unwrap();

    let a = key_on_shard("csa", 0, N);
    let b = key_on_shard("csb", 1, N);
    assert_ne!(
        shard_for(&a, N),
        shard_for(&b, N),
        "the premise of the test"
    );

    let before = n_keys_per_shard(&shards).await;
    let value = vec![b'v'; 64];
    let pairs: Vec<(&[u8], &[u8])> = vec![
        (a.as_slice(), value.as_slice()),
        (b.as_slice(), value.as_slice()),
    ];

    let err = shards
        .mset(&pairs, skeg_core::Durability::Kernel)
        .await
        .expect_err("a batch that spans shards must be refused, not half written");

    match err {
        ShardError::Admission(AdmissionError::CrossSlot) => {}
        other => panic!(
            "a cross-shard MSET must be refused as \
             ShardError::Admission(AdmissionError::CrossSlot), not {other:?}"
        ),
    }

    assert_eq!(
        shards.get(&a).await.unwrap(),
        None,
        "the key on the shard that used to commit FIRST must not exist"
    );
    assert_eq!(shards.get(&b).await.unwrap(), None, "nor the other one");
    assert_eq!(
        n_keys_per_shard(&shards).await,
        before,
        "no shard's key count moved"
    );
}

/// The refusal is decided before the batch is copied and before any shard is
/// spoken to, so an attacker who sends deliberately cross-shard batches at
/// line rate cannot make the server do the work of one.
///
/// Asserted where it is observable: the tenant's disk charge, which every
/// admitted batch moves and this one must not, plus the key counts above.
#[tokio::test]
async fn a_refused_cross_shard_mset_costs_the_tenant_nothing() {
    const T: u128 = 0x4001;
    let dir = tempfile::TempDir::new().unwrap();
    let shards = ShardSet::open(dir.path(), N).unwrap();

    let a = scoped(T, &key_on_shard("csq", 0, N));
    let b = scoped(T, &key_on_shard("csr", 1, N));
    assert_ne!(shard_for(&a, N), shard_for(&b, N));

    let before = shards.tenant_disk_bytes(T);
    let value = vec![0u8; 512];
    let pairs: Vec<(&[u8], &[u8])> = vec![
        (a.as_slice(), value.as_slice()),
        (b.as_slice(), value.as_slice()),
    ];

    assert!(
        shards
            .mset_with_disk_limit(&pairs, skeg_core::Durability::Kernel, T, Some(1 << 30))
            .await
            .is_err(),
        "refused even with room to spare: the shard span decides, not the quota"
    );
    assert_eq!(
        shards.tenant_disk_bytes(T),
        before,
        "a refused batch must not move the tenant's disk charge"
    );
    assert_eq!(shards.get(&a).await.unwrap(), None);
    assert_eq!(shards.get(&b).await.unwrap(), None);
}

/// The classification, on both wires, decided once.
#[test]
fn the_cross_slot_refusal_is_permanent_and_spelled_the_way_redis_spells_it() {
    let e = AdmissionError::CrossSlot;
    assert_eq!(
        e.retryability(),
        Retryability::Permanent,
        "the same keys hash the same way every time: telling a client to \
         retry is telling it to loop"
    );
    assert_eq!(
        e.wire_message(),
        "CROSSSLOT Keys in request don't hash to the same slot",
        "byte for byte what Redis says, so a client that already routes on \
         CROSSSLOT needs no change"
    );
    assert_eq!(e.resp3_code(), "CROSSSLOT");
    assert_eq!(
        e.code(),
        skeg_proto::ErrCode::InvalidRequest,
        "the request is what is wrong with it, not the server"
    );
}

// ── the same-shard batch keeps everything it had ────────────────────────

/// A multi-key batch whose keys all route to ONE shard of a multi-shard
/// store is still admitted, still atomic, and still durable across a reopen.
/// The refusal is about the span, not about having more than one key.
#[tokio::test]
async fn a_same_shard_mset_on_a_multi_shard_store_is_admitted_and_survives_reopen() {
    let dir = tempfile::TempDir::new().unwrap();
    let owned: Vec<(Vec<u8>, Vec<u8>)> = (0u32..8)
        .map(|i| {
            (
                key_on_shard(&format!("ss{i}_"), 2, N),
                format!("v{i}").into_bytes(),
            )
        })
        .collect();
    {
        let shards = ShardSet::open(dir.path(), N).unwrap();
        let pairs: Vec<(&[u8], &[u8])> = owned
            .iter()
            .map(|(k, v)| (k.as_slice(), v.as_slice()))
            .collect();
        shards
            .mset(&pairs, skeg_core::Durability::Kernel)
            .await
            .expect("keys that all route to one shard are one atomic batch");
        for (k, v) in &owned {
            assert_eq!(shards.get(k).await.unwrap().as_deref(), Some(v.as_slice()));
        }
    }
    let shards = ShardSet::open(dir.path(), N).unwrap();
    for (k, v) in &owned {
        assert_eq!(
            shards.get(k).await.unwrap().as_deref(),
            Some(v.as_slice()),
            "every member survived the reopen"
        );
    }
}

/// A same-shard batch refused by the tenant's disk quota still writes none of
/// its members - the property audit/17 closed, re-asserted on a MULTI-shard
/// store so it is the shard-grouping path being tested and not a store with
/// one shard where the two are the same thing.
#[tokio::test]
async fn a_same_shard_mset_refused_by_the_disk_quota_writes_nothing() {
    const T: u128 = 0x4002;
    let dir = tempfile::TempDir::new().unwrap();
    let shards = ShardSet::open(dir.path(), N).unwrap();

    // Two keys on one shard - of the SCOPED key, which is what routes.
    let mut a = None;
    let mut b = None;
    for i in 0u32..100_000 {
        let k = scoped(T, format!("dq{i}").as_bytes());
        if shard_for(&k, N) == 3 {
            if a.is_none() {
                a = Some(k);
            } else {
                b = Some(k);
                break;
            }
        }
    }
    let (a, b) = (a.unwrap(), b.unwrap());
    assert_eq!(shard_for(&a, N), shard_for(&b, N));

    // What one such pair costs, then undone: the limit below is expressed in
    // those units rather than guessed.
    let big = vec![0u8; 512];
    shards
        .tenant(T)
        .with_disk_limit(None)
        .set(&a, &big, skeg_core::Durability::Kernel)
        .await
        .unwrap();
    let unit = shards.tenant_disk_bytes(T);
    assert!(shards.del(&a, skeg_core::Durability::Kernel).await.unwrap());
    assert_eq!(shards.tenant_disk_bytes(T), 0);

    let limit = unit + unit / 2; // room for 1.5; the batch asks for 2
    let pairs: Vec<(&[u8], &[u8])> = vec![
        (a.as_slice(), big.as_slice()),
        (b.as_slice(), big.as_slice()),
    ];
    let err = shards
        .mset_with_disk_limit(&pairs, skeg_core::Durability::Kernel, T, Some(limit))
        .await
        .expect_err("the batch's sum does not fit");
    assert!(
        matches!(err, ShardError::Admission(AdmissionError::DiskQuota { .. })),
        "the quota refusal must stay a quota refusal, not become CROSSSLOT: {err:?}"
    );
    assert_eq!(shards.get(&a).await.unwrap(), None, "neither key written");
    assert_eq!(shards.get(&b).await.unwrap(), None, "neither key written");
    assert_eq!(shards.tenant_disk_bytes(T), 0, "the counter did not move");
}

/// One key is one shard, and a batch that repeats the same key is still one
/// shard. Neither may be caught by a rule about spanning.
#[tokio::test]
async fn a_single_key_batch_and_a_repeated_key_batch_are_not_cross_slot() {
    let dir = tempfile::TempDir::new().unwrap();
    let shards = ShardSet::open(dir.path(), N).unwrap();
    let k = key_on_shard("one", 1, N);

    shards
        .mset(
            &[(k.as_slice(), b"1".as_slice())],
            skeg_core::Durability::Kernel,
        )
        .await
        .expect("one key spans one shard");
    shards
        .mset(
            &[
                (k.as_slice(), b"1".as_slice()),
                (k.as_slice(), b"2".as_slice()),
            ],
            skeg_core::Durability::Kernel,
        )
        .await
        .expect("the same key twice is still one shard");
    assert_eq!(
        shards.get(&k).await.unwrap().as_deref(),
        Some(b"2".as_slice()),
        "last write wins, unchanged"
    );
}

// ── the RESP3 wire ──────────────────────────────────────────────────────

async fn send_command(stream: &mut TcpStream, args: &[&[u8]]) {
    let mut out = format!("*{}\r\n", args.len()).into_bytes();
    for a in args {
        out.extend_from_slice(format!("${}\r\n", a.len()).as_bytes());
        out.extend_from_slice(a);
        out.extend_from_slice(b"\r\n");
    }
    stream.write_all(&out).await.expect("send");
}

async fn read_line(stream: &mut TcpStream) -> String {
    let mut buf = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        let mut byte = [0u8; 1];
        match tokio::time::timeout_at(deadline, stream.read(&mut byte)).await {
            Ok(Ok(0)) | Err(_) | Ok(Err(_)) => break,
            Ok(Ok(_)) => {
                buf.push(byte[0]);
                if buf.ends_with(b"\r\n") {
                    break;
                }
            }
        }
    }
    String::from_utf8_lossy(&buf).trim_end().to_owned()
}

/// End to end on the wire a client actually speaks: the exact Redis line, and
/// neither key readable afterwards.
#[tokio::test]
async fn the_resp3_wire_answers_a_cross_shard_mset_with_the_redis_crossslot_line() {
    let dir = tempfile::tempdir().expect("tempdir");
    let server = Server::bind_with_shards("127.0.0.1:0", dir.path(), N, 0)
        .await
        .expect("bind");
    let addr = server.local_addr().expect("addr");
    tokio::spawn(async move {
        let _ = server.run_resp3().await;
    });

    let a = key_on_shard("wa", 0, N);
    let b = key_on_shard("wb", 1, N);

    let mut stream = TcpStream::connect(addr).await.expect("connect");
    send_command(&mut stream, &[b"MSET", &a, b"1", &b, b"2"]).await;
    let line = read_line(&mut stream).await;
    assert_eq!(
        line, "-CROSSSLOT Keys in request don't hash to the same slot",
        "the wire line is Redis's, byte for byte"
    );

    // No HELLO, so the connection is RESP2 and an absent key is `$-1`.
    send_command(&mut stream, &[b"GET", &a]).await;
    assert_eq!(
        read_line(&mut stream).await,
        "$-1",
        "the first key is absent"
    );
    send_command(&mut stream, &[b"GET", &b]).await;
    assert_eq!(
        read_line(&mut stream).await,
        "$-1",
        "the second key is absent"
    );

    // A same-shard MSET on the same connection still works.
    let c = key_on_shard("wc", 2, N);
    let d = key_on_shard("wd", 2, N);
    send_command(&mut stream, &[b"MSET", &c, b"1", &d, b"2"]).await;
    assert_eq!(read_line(&mut stream).await, "+OK");
}

// ── the conformance cases stay honest ───────────────────────────────────

/// `conformance/validate.py` pins the server's shard count so the CROSSSLOT
/// cases mean the same thing on every machine. The keys in those cases are
/// literals, so this asserts what the literals rely on: with that many
/// shards, one pair spans shards and the other does not.
#[test]
fn the_conformance_case_keys_route_the_way_the_cases_assume() {
    const CONFORMANCE_SHARDS: usize = 4;
    assert_ne!(
        shard_for(b"cf:xs:1", CONFORMANCE_SHARDS),
        shard_for(b"cf:xs:2", CONFORMANCE_SHARDS),
        "kv.mset.crossslot sends these two and expects CROSSSLOT"
    );
    assert_eq!(
        shard_for(b"cf:ss:1", CONFORMANCE_SHARDS),
        shard_for(b"cf:ss:2", CONFORMANCE_SHARDS),
        "kv.mset.same.slot sends these two and expects +OK"
    );
}
