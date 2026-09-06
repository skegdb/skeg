//! End-to-end test for `Resp3Tenant::from_pool` + concurrent queries.
//!
//! Verifies that:
//! - A tenant built via `from_pool` resolves dim / record_count via
//!   the pool (one conn acquired, released back).
//! - Concurrent `query_filtered` calls on a single tenant open
//!   multiple connections from the pool (up to `max_total`) and
//!   serve in parallel.
//! - After all queries return, idle queue contains a connection.
//!
//! The mock server here is a smaller variant of the one in
//! `mock_roundtrip.rs`. Duplicating ~80 lines is cheaper than a
//! shared `tests/common/` module for now.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Arc;
use std::thread;

use bytes::{Bytes, BytesMut};
use skeg_resp3::{Frame, FrameDecoder, ProtoVersion, encode_frame};
use skeg_rigging::prelude::*;
use skeg_rigging_net::RecordEnvelope;
use skeg_rigging_net_resp3::{PoolConfig, Resp3Pool, Resp3Tenant};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

const DIM: u32 = 4;

fn unit(at: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM as usize];
    v[at] = 1.0;
    v
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0, 0.0, 0.0);
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

struct MockRecord {
    id: u64,
    vector: Vec<f32>,
}

/// Spawns a multi-accept mock RESP3 server and returns (port, connect_count).
fn run_mock(records: Vec<MockRecord>) -> (u16, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().unwrap().port();
    let records = Arc::new(records);
    let connects = Arc::new(AtomicUsize::new(0));
    let conn_counter = connects.clone();
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            conn_counter.fetch_add(1, Ordering::SeqCst);
            let records = records.clone();
            thread::spawn(move || {
                let mut decoder = FrameDecoder::new();
                let mut readbuf = [0u8; 4096];
                loop {
                    let frame = loop {
                        if let Some(f) = decoder.decode().expect("decode") {
                            break Some(f);
                        }
                        let n = match stream.read(&mut readbuf) {
                            Ok(0) | Err(_) => break None,
                            Ok(n) => n,
                        };
                        decoder.feed(&readbuf[..n]);
                    };
                    let Some(frame) = frame else { break };
                    let reply = dispatch(frame, &records);
                    let mut out = BytesMut::new();
                    encode_frame(&reply, ProtoVersion::Resp3, &mut out);
                    if stream.write_all(&out).is_err() {
                        break;
                    }
                }
            });
        }
    });
    (port, connects)
}

fn dispatch(frame: Frame, records: &[MockRecord]) -> Frame {
    let Frame::Array(items) = frame else {
        return Frame::Error("expected Array".into());
    };
    let mut iter = items.into_iter();
    let cmd = match iter.next() {
        Some(Frame::Bulk(b)) => String::from_utf8_lossy(&b).to_ascii_uppercase(),
        _ => return Frame::Error("missing cmd".into()),
    };
    let args: Vec<Frame> = iter.collect();
    match cmd.as_str() {
        "HELLO" => Frame::Map(vec![(
            Frame::Bulk(Bytes::from_static(b"proto")),
            Frame::Integer(3),
        )]),
        "SKEG.VINDEX.LIST" => Frame::Array(vec![Frame::Bulk(Bytes::from(format!(
            "name=hansa dim={DIM} kind=f32 backend=flat n_vectors={}",
            records.len()
        )))]),
        "SKEG.VSEARCH" => {
            let k: usize = match &args[1] {
                Frame::Bulk(b) => std::str::from_utf8(b)
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0),
                _ => 0,
            };
            let vec_bytes = match &args[3] {
                Frame::Bulk(b) => b.clone(),
                _ => return Frame::Error("bad vector arg".into()),
            };
            let mut query = Vec::with_capacity(vec_bytes.len() / 4);
            for chunk in vec_bytes.chunks_exact(4) {
                query.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            let mut scored: Vec<(u64, f32)> = records
                .iter()
                .map(|r| (r.id, cosine(&query, &r.vector)))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            scored.truncate(k);
            let mut out: Vec<Frame> = Vec::with_capacity(scored.len() * 2);
            for (id, score) in scored {
                out.push(Frame::Bulk(Bytes::from(id.to_string())));
                out.push(Frame::Double(score as f64));
            }
            Frame::Array(out)
        }
        "MGET" => {
            let mut out = Vec::with_capacity(args.len());
            for arg in args {
                let key = match arg {
                    Frame::Bulk(b) => String::from_utf8_lossy(&b).into_owned(),
                    _ => return Frame::Error("MGET arg not Bulk".into()),
                };
                let id: Option<u64> = key.strip_prefix("hansa:rec:").and_then(|n| n.parse().ok());
                let rec = id.and_then(|i| records.iter().find(|r| r.id == i));
                match rec {
                    Some(_) => {
                        let env = RecordEnvelope::new(
                            true,
                            vec!["topic".into()],
                            b"shareable payload".to_vec(),
                        );
                        out.push(Frame::Bulk(Bytes::from(env.encode())));
                    }
                    None => out.push(Frame::Null),
                }
            }
            Frame::Array(out)
        }
        other => Frame::Error(format!("MOCK unknown cmd {other}")),
    }
}

fn fixture() -> Vec<MockRecord> {
    (1..=4u64)
        .map(|i| MockRecord {
            id: i,
            vector: unit((i as usize - 1) % DIM as usize),
        })
        .collect()
}

#[test]
fn from_pool_resolves_dim_and_count() {
    let (port, _) = run_mock(fixture());
    let pool = Arc::new(Resp3Pool::new(format!("127.0.0.1:{port}")));
    let tenant = Resp3Tenant::from_pool(pool.clone(), TenantId::ZERO, "hansa").expect("from_pool");
    assert_eq!(tenant.embedding_dim(), DIM);
    assert_eq!(tenant.record_count(), 4);
    // The conn used for VINDEX.LIST must have been returned to the pool.
    assert!(pool.idle_count() >= 1);
    assert_eq!(pool.in_use_count(), 0);
}

#[test]
fn pooled_tenant_query_returns_hits() {
    let (port, _) = run_mock(fixture());
    let pool = Arc::new(Resp3Pool::new(format!("127.0.0.1:{port}")));
    let tenant = Resp3Tenant::from_pool(pool, TenantId::from_bytes([0x42; 16]), "hansa").unwrap();

    let hits = tenant
        .query_filtered(&unit(0), 5, &|_m: &RecordMeta<'_>| true)
        .expect("query");
    assert!(!hits.is_empty(), "no hits");
}

#[test]
fn concurrent_queries_open_multiple_connections() {
    // Pool max_total=4. Spawn 4 concurrent threads making blocking
    // queries. Each thread holds its acquired conn for the duration
    // of the (VSEARCH + MGET) pair; we should see >= 2 distinct
    // connections opened on the server side (CAS may interleave).
    let (port, server_connects) = run_mock(fixture());
    let pool = Arc::new(Resp3Pool::with_config(
        format!("127.0.0.1:{port}"),
        None,
        PoolConfig {
            max_idle: 4,
            max_total: 4,
            idle_timeout: Duration::from_secs(60),
        },
    ));
    let tenant = Arc::new(Resp3Tenant::from_pool(pool.clone(), TenantId::ZERO, "hansa").unwrap());

    let connects_before = server_connects.load(Ordering::SeqCst);
    let mut handles = vec![];
    for _ in 0..4 {
        let t = tenant.clone();
        handles.push(thread::spawn(move || {
            t.query_filtered(&unit(0), 5, &|_: &RecordMeta<'_>| true)
                .expect("query")
        }));
    }
    let mut total = 0;
    for h in handles {
        total += h.join().unwrap().len();
    }
    assert!(total > 0);

    // The first vindex_info opened 1 conn. The 4 concurrent queries
    // collectively need 1..=4 more. At minimum we must have opened
    // 2 total (the initial + at least one query worker).
    let total_connects = server_connects.load(Ordering::SeqCst);
    assert!(
        total_connects > connects_before + 1,
        "expected pool to open multiple connections, server saw {total_connects} total \
         (before workers: {connects_before})"
    );

    // After workers finish, all conns must be back in the idle queue
    // (or evicted past max_idle). in_use_count == 0.
    assert_eq!(pool.in_use_count(), 0);
}

#[test]
fn pooled_tenant_as_readonly_view_trait_object() {
    let (port, _) = run_mock(fixture());
    let pool = Arc::new(Resp3Pool::new(format!("127.0.0.1:{port}")));
    let tenant = Resp3Tenant::from_pool(pool, TenantId::from_bytes([0xaa; 16]), "hansa").unwrap();
    let view: Box<dyn ReadOnlyView> = Box::new(tenant);
    assert_eq!(view.tenant_id(), TenantId::from_bytes([0xaa; 16]));
    let _ = view.close();
}
