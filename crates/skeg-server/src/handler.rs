use bytes::Bytes;
use bytes::BytesMut;
use skeg_proto::{
    ErrCode, Flags, Frame, FrameParser, decode_key_payload, decode_mget_payload,
    decode_set_payload, decode_vindex_create_payload, decode_vname_id_payload,
    decode_vname_payload, decode_vsearch_payload, decode_vset_payload, encode_err, encode_ok,
    encode_ok_bool, encode_ok_mget, encode_ok_shards, encode_ok_stats, encode_ok_value,
    encode_ok_vindex_list, encode_ok_vsearch, f32_vec_to_bytes,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tracing::{debug, warn};

use skeg_core::Durability;

use crate::shard::{ShardError, ShardSet};

/// Durability applied to writes that do not request one explicitly.
/// `Kernel` survives a process crash and kernel panic without paying the
/// `F_FULLFSYNC` cost on every write - the right default for AI workloads
/// (see `design-write-perf.md` §2.4).
const DEFAULT_DURABILITY: Durability = Durability::Kernel;

pub async fn handle_connection(mut stream: TcpStream, shards: ShardSet) {
    let peer = stream.peer_addr().ok();
    debug!(?peer, "connection accepted");

    let mut parser = FrameParser::new();
    let mut buf = BytesMut::with_capacity(64 * 1024);

    loop {
        match parser.feed(&mut buf) {
            Ok(Some(frame)) => {
                if let Some(response) = dispatch(&frame, &shards).await
                    && stream.write_all(&response).await.is_err()
                {
                    break;
                }
            }
            Ok(None) => match stream.read_buf(&mut buf).await {
                Ok(0) => break,
                Ok(_) => {}
                Err(e) => {
                    warn!(?peer, "read error: {e}");
                    break;
                }
            },
            Err(e) => {
                warn!(?peer, "protocol error: {e}");
                break;
            }
        }
    }

    debug!(?peer, "connection closed");
}

fn shard_err_to_response(req_id: u64, e: &ShardError) -> Bytes {
    warn!("shard error: {e}");
    encode_err(req_id, ErrCode::Internal, &e.to_string())
}

/// Dispatch a parsed frame to the shard set and return an optional response.
#[allow(clippy::too_many_lines)] // one arm per protocol op; splitting hurts readability
async fn dispatch(frame: &Frame, shards: &ShardSet) -> Option<Bytes> {
    let req_id = frame.header.req_id;
    let payload = &frame.payload;

    match frame.header.op {
        skeg_proto::Op::Ping => Some(encode_ok(req_id)),

        skeg_proto::Op::Stats => match shards.stats().await {
            Ok(stats) => Some(encode_ok_stats(req_id, stats)),
            Err(e) => Some(shard_err_to_response(req_id, &e)),
        },

        skeg_proto::Op::Shards => match shards.stats_per_shard().await {
            Ok(rows) => Some(encode_ok_shards(req_id, &rows)),
            Err(e) => Some(shard_err_to_response(req_id, &e)),
        },

        skeg_proto::Op::VindexList => match shards.vindex_list().await {
            Ok(rows) => {
                let info: Vec<skeg_proto::VindexInfo> = rows
                    .into_iter()
                    .map(
                        |(name, dim, kind, backend, n_vectors)| skeg_proto::VindexInfo {
                            name,
                            dim,
                            kind,
                            backend,
                            n_vectors,
                        },
                    )
                    .collect();
                Some(encode_ok_vindex_list(req_id, &info))
            }
            Err(e) => Some(shard_err_to_response(req_id, &e)),
        },

        skeg_proto::Op::Get => match decode_key_payload(payload) {
            Ok(key) => match shards.get(&key).await {
                Ok(Some(val)) => Some(encode_ok_value(req_id, &val)),
                Ok(None) => Some(encode_err(req_id, ErrCode::NotFound, "key not found")),
                Err(e) => Some(shard_err_to_response(req_id, &e)),
            },
            Err(e) => Some(encode_err(req_id, ErrCode::InvalidRequest, &e.to_string())),
        },

        skeg_proto::Op::Set => match decode_set_payload(payload) {
            Ok((key, val)) => match shards.set(&key, &val, DEFAULT_DURABILITY).await {
                Ok(()) => {
                    if frame.header.flags.contains(Flags::NO_REPLY) {
                        None
                    } else {
                        Some(encode_ok(req_id))
                    }
                }
                Err(e) => Some(shard_err_to_response(req_id, &e)),
            },
            Err(e) => Some(encode_err(req_id, ErrCode::InvalidRequest, &e.to_string())),
        },

        skeg_proto::Op::Del => match decode_key_payload(payload) {
            Ok(key) => match shards.del(&key, DEFAULT_DURABILITY).await {
                Ok(existed) => Some(encode_ok_bool(req_id, existed)),
                Err(e) => Some(shard_err_to_response(req_id, &e)),
            },
            Err(e) => Some(encode_err(req_id, ErrCode::InvalidRequest, &e.to_string())),
        },

        skeg_proto::Op::Mget => match decode_mget_payload(payload) {
            Ok(keys) => match shards.mget(&keys).await {
                Ok(results) => Some(encode_ok_mget(req_id, &results)),
                Err(e) => Some(shard_err_to_response(req_id, &e)),
            },
            Err(e) => Some(encode_err(req_id, ErrCode::InvalidRequest, &e.to_string())),
        },

        skeg_proto::Op::VindexCreate => {
            let (name, dim, kind, backend) = match decode_vindex_create_payload(payload) {
                Ok(v) => v,
                Err(e) => return Some(encode_err(req_id, ErrCode::InvalidRequest, &e.to_string())),
            };
            let Ok(name) = std::str::from_utf8(&name) else {
                return Some(encode_err(
                    req_id,
                    ErrCode::InvalidRequest,
                    "index name not utf-8",
                ));
            };
            match shards.vindex_create(name, dim, kind, backend).await {
                Ok(()) => Some(encode_ok(req_id)),
                Err(e) => Some(shard_err_to_response(req_id, &e)),
            }
        }

        skeg_proto::Op::VindexDrop => {
            let name = match decode_vname_payload(payload) {
                Ok(v) => v,
                Err(e) => return Some(encode_err(req_id, ErrCode::InvalidRequest, &e.to_string())),
            };
            let Ok(name) = std::str::from_utf8(&name) else {
                return Some(encode_err(
                    req_id,
                    ErrCode::InvalidRequest,
                    "index name not utf-8",
                ));
            };
            match shards.vindex_drop(name, 0).await {
                Ok(()) => Some(encode_ok(req_id)),
                Err(e) => Some(shard_err_to_response(req_id, &e)),
            }
        }

        skeg_proto::Op::Vset => {
            let (name, id, vector) = match decode_vset_payload(payload) {
                Ok(v) => v,
                Err(e) => return Some(encode_err(req_id, ErrCode::InvalidRequest, &e.to_string())),
            };
            let Ok(name) = std::str::from_utf8(&name) else {
                return Some(encode_err(
                    req_id,
                    ErrCode::InvalidRequest,
                    "index name not utf-8",
                ));
            };
            match shards.vset(name, id, vector, 0, None).await {
                Ok(()) => {
                    if frame.header.flags.contains(Flags::NO_REPLY) {
                        None
                    } else {
                        Some(encode_ok(req_id))
                    }
                }
                Err(e) => Some(shard_err_to_response(req_id, &e)),
            }
        }

        skeg_proto::Op::Vget => {
            let (name, id) = match decode_vname_id_payload(payload) {
                Ok(v) => v,
                Err(e) => return Some(encode_err(req_id, ErrCode::InvalidRequest, &e.to_string())),
            };
            let Ok(name) = std::str::from_utf8(&name) else {
                return Some(encode_err(
                    req_id,
                    ErrCode::InvalidRequest,
                    "index name not utf-8",
                ));
            };
            match shards.vget(name, id).await {
                Ok(Some(v)) => Some(encode_ok_value(req_id, &f32_vec_to_bytes(&v))),
                Ok(None) => Some(encode_err(req_id, ErrCode::NotFound, "vector not found")),
                Err(e) => Some(shard_err_to_response(req_id, &e)),
            }
        }

        skeg_proto::Op::Vdel => {
            let (name, id) = match decode_vname_id_payload(payload) {
                Ok(v) => v,
                Err(e) => return Some(encode_err(req_id, ErrCode::InvalidRequest, &e.to_string())),
            };
            let Ok(name) = std::str::from_utf8(&name) else {
                return Some(encode_err(
                    req_id,
                    ErrCode::InvalidRequest,
                    "index name not utf-8",
                ));
            };
            match shards.vdel(name, id, 0).await {
                Ok(existed) => Some(encode_ok_bool(req_id, existed)),
                Err(e) => Some(shard_err_to_response(req_id, &e)),
            }
        }

        skeg_proto::Op::Vsearch => {
            let (name, k, query, l_search) = match decode_vsearch_payload(payload) {
                Ok(v) => v,
                Err(e) => return Some(encode_err(req_id, ErrCode::InvalidRequest, &e.to_string())),
            };
            let Ok(name) = std::str::from_utf8(&name) else {
                return Some(encode_err(
                    req_id,
                    ErrCode::InvalidRequest,
                    "index name not utf-8",
                ));
            };
            let span = tracing::info_span!(
                "vsearch",
                protocol = "binary",
                vindex = name,
                k,
                l_search,
                vector_dim = query.len(),
                hits = tracing::field::Empty,
            );
            let _guard = span.enter();
            match shards.vsearch(name, query, k as usize, l_search).await {
                Ok(hits) => {
                    span.record("hits", hits.len());
                    Some(encode_ok_vsearch(req_id, &hits))
                }
                Err(e) => Some(shard_err_to_response(req_id, &e)),
            }
        }

        op => Some(encode_err(
            req_id,
            ErrCode::InvalidRequest,
            &format!("op {op:?} not implemented"),
        )),
    }
}
