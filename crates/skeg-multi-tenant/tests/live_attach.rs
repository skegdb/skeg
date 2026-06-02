//! F.41 - live attach mode: bridge routes ops to a running
//! skeg-server-tenant via RESP3 instead of opening on-disk tenants.
//!
//! Uses a tiny mock RESP3 server (copied from the resp3 crate's own
//! mock_roundtrip, trimmed) so this test runs without a real
//! skeg-server-tenant binary.

#![cfg(feature = "live-attach")]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use bytes::{Bytes, BytesMut};
use skeg_multi_tenant::{LiveAttachError, LiveAttachRoot, SkegTenantId};
use skeg_resp3::{Frame, FrameDecoder, ProtoVersion, encode_frame};
use skeg_rigging::prelude::*;
use skeg_rigging_net::RecordEnvelope;
use skeg_tenant::auth::TokenStore;

const DIM: u32 = 4;

fn unit(at: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM as usize];
    v[at] = 1.0;
    v
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
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
    shareable: bool,
}

fn fixture() -> Vec<MockRecord> {
    vec![
        MockRecord {
            id: 1,
            vector: unit(0),
            shareable: true,
        },
        MockRecord {
            id: 2,
            vector: unit(1),
            shareable: false,
        },
        MockRecord {
            id: 3,
            vector: unit(0),
            shareable: true,
        },
    ]
}

fn run_mock(records: Vec<MockRecord>, expected_index: String) -> (u16, Arc<AtomicUsize>) {
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
            let expected_index = expected_index.clone();
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
                    let reply = dispatch(frame, &records, &expected_index);
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

fn dispatch(frame: Frame, records: &[MockRecord], expected_index: &str) -> Frame {
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
            "name={expected_index} dim={DIM} kind=f32 backend=flat n_vectors={}",
            records.len()
        )))]),
        "SKEG.VSEARCH" => {
            // args[0] is the index name — verify it matches expected.
            let actual_index = match &args[0] {
                Frame::Bulk(b) => String::from_utf8_lossy(b).into_owned(),
                _ => return Frame::Error("bad index arg".into()),
            };
            if actual_index != expected_index {
                return Frame::Error(format!(
                    "wrong index: got {actual_index}, expected {expected_index}"
                ));
            }
            let k = match &args[1] {
                Frame::Bulk(b) => std::str::from_utf8(b)
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0usize),
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
                    Some(r) => {
                        let env = RecordEnvelope::new(
                            r.shareable,
                            vec!["topic".into()],
                            format!("rec-{}", r.id).into_bytes(),
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

// ─── Tests ──────────────────────────────────────────────────────────

#[test]
fn live_root_open_resolves_dim() {
    let tid = SkegTenantId::from_bytes([0x11; 16]);
    // Bridge uses a per-tenant index name by default (so two tenants
    // on the same skeg-server don't collide).
    let expected_idx = format!("tenant_{}", hex_of(tid));
    let (port, _) = run_mock(fixture(), expected_idx);

    let store = Arc::new(TokenStore::from_key([7u8; 32], Duration::from_secs(60)));
    let root = LiveAttachRoot::new(format!("127.0.0.1:{port}"), store);
    let tenant = root.open(tid, DIM).expect("open");
    assert_eq!(tenant.embedding_dim(), DIM);
    assert_eq!(tenant.record_count(), 3);
}

#[test]
fn live_root_query_returns_hits_through_resp3() {
    let tid = SkegTenantId::from_bytes([0x22; 16]);
    let expected_idx = format!("tenant_{}", hex_of(tid));
    let (port, _) = run_mock(fixture(), expected_idx);

    let store = Arc::new(TokenStore::from_key([7u8; 32], Duration::from_secs(60)));
    let root = LiveAttachRoot::new(format!("127.0.0.1:{port}"), store);
    let tenant = root.open(tid, DIM).expect("open");

    let hits = tenant
        .query_filtered(&unit(0), 5, &|m: &RecordMeta<'_>| m.shareable)
        .expect("query");
    // ids 1 and 3 are shareable + match unit(0); id 2 is filtered out.
    let ids: Vec<u64> = hits.iter().map(|h| h.record_id.0).collect();
    for &id in &ids {
        assert!(id == 1 || id == 3, "leaked non-shareable id {id}");
    }
    assert!(!ids.is_empty());
}

#[test]
fn live_root_open_with_token_validates() {
    let tid = SkegTenantId::from_bytes([0x33; 16]);
    let expected_idx = format!("tenant_{}", hex_of(tid));
    let (port, _) = run_mock(fixture(), expected_idx);

    let store = Arc::new(TokenStore::from_key([9u8; 32], Duration::from_secs(60)));
    let token = store.issue(tid).unwrap();
    let root = LiveAttachRoot::new(format!("127.0.0.1:{port}"), store);

    let tenant = root.open_with_token(&token, DIM).expect("open via token");
    assert_eq!(tenant.embedding_dim(), DIM);
}

#[test]
fn live_root_bad_token_rejects() {
    let (port, _) = run_mock(fixture(), "tenant_anything".into());
    let store = Arc::new(TokenStore::from_key([1u8; 32], Duration::from_secs(60)));
    let root = LiveAttachRoot::new(format!("127.0.0.1:{port}"), store);

    let garbage = [0u8; 47];
    match root.open_with_token(&garbage, DIM) {
        Ok(_) => panic!("garbage token must fail"),
        Err(e) => assert!(matches!(e, LiveAttachError::TokenError(_))),
    }
}

#[test]
fn live_root_endpoint_unreachable_surfaces_io_error() {
    let store = Arc::new(TokenStore::from_key([3u8; 32], Duration::from_secs(60)));
    // Port 1 should reject on every system.
    let root = LiveAttachRoot::new("127.0.0.1:1", store);
    match root.open(SkegTenantId::from_bytes([0x44; 16]), DIM) {
        Ok(_) => panic!("unreachable endpoint must fail"),
        Err(e) => assert!(matches!(e, LiveAttachError::Net(_))),
    }
}

#[test]
fn live_root_custom_index_name_is_used() {
    let custom_idx = "my-fixed-index";
    let (port, _) = run_mock(fixture(), custom_idx.into());
    let store = Arc::new(TokenStore::from_key([5u8; 32], Duration::from_secs(60)));
    let root = LiveAttachRoot::new(format!("127.0.0.1:{port}"), store).with_index_name(custom_idx);
    let tenant = root
        .open(SkegTenantId::from_bytes([0x55; 16]), DIM)
        .expect("open");
    // Issue a query — mock verifies the index name matches.
    let hits = tenant
        .query_filtered(&unit(0), 3, &|_m: &RecordMeta<'_>| true)
        .expect("query");
    assert!(!hits.is_empty());
}

#[test]
fn live_root_concurrent_queries_share_pool() {
    let tid = SkegTenantId::from_bytes([0x66; 16]);
    let expected_idx = format!("tenant_{}", hex_of(tid));
    let (port, connects) = run_mock(fixture(), expected_idx);

    let store = Arc::new(TokenStore::from_key([2u8; 32], Duration::from_secs(60)));
    let root = LiveAttachRoot::new(format!("127.0.0.1:{port}"), store);
    let tenant = Arc::new(root.open(tid, DIM).expect("open"));

    let before = connects.load(Ordering::SeqCst);
    let mut handles = vec![];
    for _ in 0..4 {
        let t = tenant.clone();
        handles.push(thread::spawn(move || {
            t.query_filtered(&unit(0), 3, &|_: &RecordMeta<'_>| true)
                .expect("query")
        }));
    }
    for h in handles {
        let hits = h.join().unwrap();
        assert!(!hits.is_empty());
    }
    let after = connects.load(Ordering::SeqCst);
    // Each query opens its own conn (max_total default = 16 in the
    // pool). Server should see more than the single conn used at
    // construction.
    assert!(
        after > before + 1,
        "pool didn't multiplex: server saw {after} total (before workers: {before})"
    );
}

fn hex_of(tid: SkegTenantId) -> String {
    let mut s = String::with_capacity(32);
    for b in tid.as_bytes() {
        s.push_str(&format!("{b:02x}"));
    }
    s
}
