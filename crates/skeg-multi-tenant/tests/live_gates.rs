//! F.41 release gates — live-attach open + concurrent throughput.
//!
//! Run with:
//!   cargo test --release --test live_gates -p skeg-multi-tenant --features live-attach

#![cfg(feature = "live-attach")]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use bytes::{Bytes, BytesMut};
use skeg_multi_tenant::{LiveAttachRoot, SkegTenantId};
use skeg_resp3::{Frame, FrameDecoder, ProtoVersion, encode_frame};
use skeg_rigging::prelude::*;
use skeg_rigging_net::RecordEnvelope;
use skeg_tenant::auth::TokenStore;

const DIM: u32 = 4;

fn skip_unless_release() -> bool {
    if cfg!(debug_assertions) {
        eprintln!(
            "[gates] skipping in debug mode; run `cargo test --release --test live_gates --features live-attach` to enforce"
        );
        true
    } else {
        false
    }
}

// ── Thresholds ──────────────────────────────────────────────────────

/// `LiveAttachRoot::open` over loopback = TCP connect + HELLO +
/// VINDEX.LIST + return. Best-of-20 below 10 ms.
const GATE_LIVE_OPEN_MS: u128 = 10;

/// 8 concurrent loopback queries via shared pool. Best-of-3 below
/// 100 ms (16 round-trips serialised over a few connections).
const GATE_LIVE_CONCURRENT_MS: u128 = 100;

// ── Mock server (single-tenant, minimal) ────────────────────────────

fn unit(at: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM as usize];
    v[at] = 1.0;
    v
}

fn run_mock(expected_index: String) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().unwrap().port();
    let connects = Arc::new(AtomicUsize::new(0));
    let conn_counter = connects.clone();
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            conn_counter.fetch_add(1, Ordering::SeqCst);
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
                    let reply = dispatch(frame, &expected_index);
                    let mut out = BytesMut::new();
                    encode_frame(&reply, ProtoVersion::Resp3, &mut out);
                    if stream.write_all(&out).is_err() {
                        break;
                    }
                }
            });
        }
    });
    port
}

fn dispatch(frame: Frame, expected_index: &str) -> Frame {
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
            "name={expected_index} dim={DIM} kind=f32 backend=flat n_vectors=1"
        )))]),
        "SKEG.VSEARCH" => {
            let _k = &args[1];
            // Return a single hit so MGET has work.
            Frame::Array(vec![
                Frame::Bulk(Bytes::from_static(b"1")),
                Frame::Double(1.0),
            ])
        }
        "MGET" => {
            let env = RecordEnvelope::new(true, vec!["topic".into()], b"hit".to_vec());
            Frame::Array(vec![Frame::Bulk(Bytes::from(env.encode()))])
        }
        other => Frame::Error(format!("MOCK unknown cmd {other}")),
    }
}

fn hex_of(tid: SkegTenantId) -> String {
    let mut s = String::with_capacity(32);
    for b in tid.as_bytes() {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

// ── Gates ───────────────────────────────────────────────────────────

#[test]
fn gate_live_open_under_threshold() {
    if skip_unless_release() {
        return;
    }
    let tid = SkegTenantId::from_bytes([0x11; 16]);
    let port = run_mock(format!("tenant_{}", hex_of(tid)));
    let store = Arc::new(TokenStore::from_key([7u8; 32], Duration::from_secs(60)));
    let root = LiveAttachRoot::new(format!("127.0.0.1:{port}"), store);

    // Warm-up.
    let _ = root.open(tid, DIM).unwrap();
    let mut best_ms = u128::MAX;
    for _ in 0..20 {
        let t = Instant::now();
        let _ = root.open(tid, DIM).unwrap();
        best_ms = best_ms.min(t.elapsed().as_millis());
    }
    eprintln!("[gate] live_open best-of-20 = {best_ms} ms (cap {GATE_LIVE_OPEN_MS})");
    assert!(
        best_ms <= GATE_LIVE_OPEN_MS,
        "live_open best-of-20 = {best_ms} ms, gate {GATE_LIVE_OPEN_MS} ms"
    );
}

#[test]
fn gate_live_concurrent_queries_throughput() {
    if skip_unless_release() {
        return;
    }
    let tid = SkegTenantId::from_bytes([0x22; 16]);
    let port = run_mock(format!("tenant_{}", hex_of(tid)));
    let store = Arc::new(TokenStore::from_key([2u8; 32], Duration::from_secs(60)));
    let root = LiveAttachRoot::new(format!("127.0.0.1:{port}"), store);
    let tenant = Arc::new(root.open(tid, DIM).unwrap());

    // Warm-up the pool.
    for _ in 0..2 {
        let _ = tenant
            .query_filtered(&unit(0), 3, &|_: &RecordMeta<'_>| true)
            .unwrap();
    }

    let mut best_ms = u128::MAX;
    for _ in 0..3 {
        let t = Instant::now();
        let mut handles = vec![];
        for _ in 0..8 {
            let tt = tenant.clone();
            handles.push(thread::spawn(move || {
                tt.query_filtered(&unit(0), 3, &|_: &RecordMeta<'_>| true)
                    .expect("query")
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        best_ms = best_ms.min(t.elapsed().as_millis());
    }
    eprintln!(
        "[gate] live_concurrent(8 queries) best-of-3 = {best_ms} ms (cap {GATE_LIVE_CONCURRENT_MS})"
    );
    assert!(
        best_ms <= GATE_LIVE_CONCURRENT_MS,
        "live_concurrent best-of-3 = {best_ms} ms, gate {GATE_LIVE_CONCURRENT_MS} ms"
    );
}
