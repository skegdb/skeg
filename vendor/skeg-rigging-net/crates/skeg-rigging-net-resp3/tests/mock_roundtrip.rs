//! Mock-server roundtrip for `Resp3Tenant`.
//!
//! Spawns a tiny RESP3 server on loopback that replies to the exact
//! command sequence `Resp3Tenant::connect` + `query_filtered` issues.
//! This validates wire encoding, parser handling of all the reply
//! shapes (Map, Array, Bulk, Double, Null), and the post-filter logic
//! without needing a real skeg-server.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

use bytes::{Bytes, BytesMut};
use skeg_resp3::{Frame, FrameDecoder, ProtoVersion, encode_frame};
use skeg_rigging::prelude::*;
use skeg_rigging_net::RecordEnvelope;
use skeg_rigging_net_resp3::{Resp3Connection, Resp3Tenant};

const DIM: u32 = 4;

fn unit(at: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM as usize];
    v[at] = 1.0;
    v
}

struct MockRecord {
    id: u64,
    vector: Vec<f32>,
    shareable: bool,
    payload: &'static [u8],
}

fn fixture() -> Vec<MockRecord> {
    vec![
        MockRecord {
            id: 1,
            vector: unit(0),
            shareable: true,
            payload: b"alpha (shareable)",
        },
        MockRecord {
            id: 2,
            vector: unit(1),
            shareable: false,
            payload: b"beta (private)",
        },
        MockRecord {
            id: 3,
            vector: unit(0),
            shareable: true,
            payload: b"gamma (shareable copy)",
        },
        MockRecord {
            id: 4,
            vector: unit(2),
            shareable: false,
            payload: b"delta (private)",
        },
    ]
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
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

fn run_mock(records: Vec<MockRecord>) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
    let port = listener.local_addr().unwrap().port();
    let records = std::sync::Arc::new(records);
    thread::spawn(move || {
        // Loop-accept: a single client makes one connection (old
        // tests); a pool can open several to the same endpoint
        // (pooled test). Spawn a worker per connection so they
        // can be served in parallel.
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let records = records.clone();
            thread::spawn(move || {
                let mut decoder = FrameDecoder::new();
                let mut readbuf = [0u8; 4096];
                loop {
                    let frame = loop {
                        if let Some(f) = decoder.decode().expect("decode frame") {
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
    port
}

fn dispatch(frame: Frame, records: &[MockRecord]) -> Frame {
    let Frame::Array(items) = frame else {
        return Frame::Error("expected Array".into());
    };
    let mut iter = items.into_iter();
    let name_bytes = match iter.next() {
        Some(Frame::Bulk(b)) => b,
        _ => return Frame::Error("missing cmd".into()),
    };
    let name = String::from_utf8_lossy(&name_bytes).to_ascii_uppercase();
    let args: Vec<Frame> = iter.collect();
    match name.as_str() {
        "HELLO" => Frame::Map(vec![
            (
                Frame::Bulk(Bytes::from_static(b"server")),
                Frame::Bulk(Bytes::from_static(b"skeg-mock")),
            ),
            (Frame::Bulk(Bytes::from_static(b"proto")), Frame::Integer(3)),
        ]),
        "SKEG.VINDEX.LIST" => Frame::Array(vec![Frame::Bulk(Bytes::from(format!(
            "name=hansa dim={DIM} kind=f32 backend=flat n_vectors={}",
            records.len()
        )))]),
        "SKEG.VSEARCH" => {
            // args: [index, k, l_search, vector_bytes]
            let _index = &args[0];
            let k = parse_usize(&args[1]).unwrap_or(0);
            let vec_bytes = match &args[3] {
                Frame::Bulk(b) => b.clone(),
                _ => return Frame::Error("bad vector arg".into()),
            };
            if vec_bytes.len() % 4 != 0 {
                return Frame::Error("vector length not multiple of 4".into());
            }
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
            // args: hansa:rec:<id> ...
            let mut out = Vec::with_capacity(args.len());
            for arg in args {
                let key = match arg {
                    Frame::Bulk(b) => String::from_utf8_lossy(&b).into_owned(),
                    _ => return Frame::Error("MGET arg not Bulk".into()),
                };
                let id: Option<u64> = key.strip_prefix("hansa:rec:").and_then(|n| n.parse().ok());
                let env = id.and_then(|i| records.iter().find(|r| r.id == i));
                match env {
                    Some(rec) => {
                        let env = RecordEnvelope::new(
                            rec.shareable,
                            vec!["topic".into()],
                            rec.payload.to_vec(),
                        );
                        out.push(Frame::Bulk(Bytes::from(env.encode())));
                    }
                    None => out.push(Frame::Null),
                }
            }
            Frame::Array(out)
        }
        "SKEG.VINDEX.CREATE" | "SKEG.VSET" | "SET" => Frame::Simple("OK".into()),
        "TEST.BACKPRESSURE" => Frame::Error("BACKPRESSURE memory budget exhausted".into()),
        other => Frame::Error(format!("MOCK unknown cmd {other}")),
    }
}

fn parse_usize(f: &Frame) -> Option<usize> {
    match f {
        Frame::Bulk(b) => std::str::from_utf8(b).ok()?.parse().ok(),
        Frame::Integer(i) => Some(*i as usize),
        _ => None,
    }
}

#[test]
fn connect_and_resolve_dim() {
    let port = run_mock(fixture());
    let endpoint = format!("127.0.0.1:{port}");
    let tenant =
        Resp3Tenant::connect(&endpoint, TenantId::from_bytes([1; 16]), None).expect("connect");
    assert_eq!(tenant.embedding_dim(), DIM);
    assert_eq!(tenant.record_count(), 4);
}

#[test]
fn query_filtered_drops_non_shareable() {
    let port = run_mock(fixture());
    let endpoint = format!("127.0.0.1:{port}");
    let tenant =
        Resp3Tenant::connect(&endpoint, TenantId::from_bytes([2; 16]), None).expect("connect");

    let hits = tenant
        .query_filtered(&unit(0), 5, &|m: &RecordMeta<'_>| m.shareable)
        .expect("query");

    // Only ids 1 and 3 are shareable AND match the query (unit-x).
    // Ids 2 and 4 are non-shareable, must be excluded.
    let ids: Vec<u64> = hits.iter().map(|h| h.record_id.0).collect();
    for &id in &ids {
        assert!(id == 1 || id == 3, "leaked non-shareable record {id}");
    }
    assert!(!ids.is_empty(), "no shareable hits");
}

#[test]
fn query_filtered_accept_all_returns_top_k_in_score_order() {
    let port = run_mock(fixture());
    let endpoint = format!("127.0.0.1:{port}");
    let tenant = Resp3Tenant::connect(&endpoint, TenantId::ZERO, None).expect("connect");

    let hits = tenant
        .query_filtered(&unit(0), 4, &|_m: &RecordMeta<'_>| true)
        .expect("query");
    let ids: Vec<u64> = hits.iter().map(|h| h.record_id.0).collect();
    // Top by cosine(unit-x): {1, 3} share sim=1.0, others 0.0.
    assert!(ids.contains(&1) && ids.contains(&3));
    // Hits sorted desc by similarity from skeg-server's side; ensure
    // monotone non-increasing.
    let mut sims = hits.iter().map(|h| h.similarity).collect::<Vec<_>>();
    sims.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let original: Vec<f32> = hits.iter().map(|h| h.similarity).collect();
    assert_eq!(original, sims, "scores not in descending order");
}

#[test]
fn read_only_view_object_safety() {
    let port = run_mock(fixture());
    let endpoint = format!("127.0.0.1:{port}");
    let tenant =
        Resp3Tenant::connect(&endpoint, TenantId::from_bytes([9; 16]), None).expect("connect");
    let view: Box<dyn ReadOnlyView> = Box::new(tenant);
    assert_eq!(view.tenant_id(), TenantId::from_bytes([9; 16]));
    let _ = view.close();
}

#[test]
fn iter_vectors_is_empty_on_resp3() {
    let port = run_mock(fixture());
    let endpoint = format!("127.0.0.1:{port}");
    let tenant = Resp3Tenant::connect(&endpoint, TenantId::ZERO, None).expect("connect");
    let v: Vec<_> = tenant.iter_vectors().collect();
    assert!(v.is_empty(), "iter_vectors should be empty over RESP3");
}

#[test]
fn backpressure_remains_retryable_after_resp3_roundtrip() {
    let port = run_mock(fixture());
    let endpoint = format!("127.0.0.1:{port}");
    let mut connection = Resp3Connection::connect(&endpoint, None).expect("connect");

    let error = connection
        .call("TEST.BACKPRESSURE", &[])
        .expect_err("mock must reject the command");

    assert!(error.is_retryable());
}
