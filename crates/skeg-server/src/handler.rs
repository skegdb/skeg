use bytes::Bytes;
use bytes::BytesMut;
use skeg_proto::{
    ErrCode, Flags, Frame, FrameParser, NativeCapabilities, NativeVectorKindV2, VERSION_V1,
    VERSION_V2, decode_key_payload, decode_mget_payload, decode_set_payload,
    decode_vindex_create_payload, decode_vname_id_payload, decode_vname_payload,
    decode_vsearch_payload, decode_vset_payload, encode_err, encode_ok, encode_ok_bool,
    encode_ok_mget, encode_ok_native_capabilities, encode_ok_shards, encode_ok_stats,
    encode_ok_value, encode_ok_vindex_list, encode_ok_vsearch, f32_vec_to_bytes,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tracing::{debug, warn};

use skeg_core::Durability;

use crate::shard::{ShardError, ShardSet};

/// Durability applied to writes that do not request one explicitly.
/// `Kernel` survives a process crash and kernel panic without paying the
/// `F_FULLFSYNC` cost on every write - the right default for AI workloads
/// (see `design-write-perf.md`).
const DEFAULT_DURABILITY: Durability = Durability::Kernel;

pub async fn handle_connection(mut stream: TcpStream, shards: ShardSet) {
    let peer = stream.peer_addr().ok();
    debug!(?peer, "connection accepted");

    let mut parser = FrameParser::new();
    let mut buf = BytesMut::with_capacity(64 * 1024);

    loop {
        match parser.feed(&mut buf) {
            Ok(Some(frame)) => {
                if let Some(response) = dispatch(&frame, &shards)
                    .await
                    .map(|response| response_for_version(response, frame.header.version))
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

/// Responses must stay in the request's native protocol version. The typed
/// response encoders intentionally default to v1 for legacy callers, so this
/// is the sole server-side point that applies the connection's frame version.
fn response_for_version(response: Bytes, version: u8) -> Bytes {
    if version == VERSION_V1 {
        return response;
    }
    let mut response = BytesMut::from(response.as_ref());
    response[2] = version;
    response.freeze()
}

/// Validate the raw `VINDEX.CREATE` kind byte against the frame version.
///
/// Native v1 is deliberately limited to its original three kinds. Its code 3
/// meant PQ in released Python clients, so accepting it as today's TQ1 would
/// silently create a different index than the caller requested.
fn native_vindex_kind_is_allowed(version: u8, kind: u8) -> Result<(), &'static str> {
    match version {
        VERSION_V1 => match kind {
            0..=2 => Ok(()),
            3 => Err("native v1 kind 3 is PQ; use RESP3 or native v2"),
            _ => Err("native v1 supports only f32, int8, and binary"),
        },
        VERSION_V2 => NativeVectorKindV2::try_from(kind)
            .map(|_| ())
            .map_err(|_| "native v2 kind must be f32, int8, binary, tq1, tq2, or tq4"),
        _ => Err("unsupported native protocol version"),
    }
}

fn shard_err_to_response(req_id: u64, e: &ShardError) -> Bytes {
    warn!("shard error: {e}");
    encode_err(req_id, ErrCode::Internal, &e.to_string())
}

/// The index name of a native request: UTF-8, and free of the tenant-scope
/// separator.
///
/// EVERY arm that takes an index name goes through this, and that is the whole
/// point of it existing. The native protocol has no tenant: it calls with
/// tenant `0` and the client's raw name. Closing the create door stops a
/// client MAKING a key that reads as another tenant's, but not one NAMING a
/// key that already exists - a tenant's index, created over RESP3 from an id
/// the server authenticated, is just a string here. Without this, VGET reads
/// that tenant's vectors, VSET writes into its index, and VDEL and
/// VINDEX.DROP destroy them, from a connection that authenticated as nobody.
///
/// Returns the encoded error frame to send back, so a new arm that forgets the
/// check is a name it cannot use rather than a hole it cannot see.
fn native_index_name(req_id: u64, raw: &Bytes) -> Result<&str, Bytes> {
    let Ok(name) = std::str::from_utf8(raw) else {
        return Err(encode_err(
            req_id,
            ErrCode::InvalidRequest,
            "index name not utf-8",
        ));
    };
    if let Err(e) = crate::shard::reject_scope_separator(name) {
        return Err(encode_err(req_id, ErrCode::InvalidRequest, &e.to_string()));
    }
    Ok(name)
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
                // Hide every tenant-scoped name, the same rule RESP3 applies
                // to an anonymous connection (`skeg_vindex_list`). This
                // listener has no tenant - it is always tenant 0 - so the
                // names it may see are exactly the unscoped ones. Without
                // this it handed out the scoped name, dim, kind and vector
                // count of every index of every tenant, and a scoped name is
                // the tenant id in hex, so the listing enumerated the tenants
                // as well.
                //
                // Filtered BEFORE the v1 tier check below: that check refuses
                // the call because the ROWS CANNOT BE REPRESENTED to this
                // client, so it must weigh the rows this client receives, not
                // ones it is not allowed to know exist.
                let rows: Vec<_> = rows
                    .into_iter()
                    .filter(|row| !row.name.contains(crate::shard::SCOPE_SEP))
                    .collect();
                if frame.header.version == VERSION_V1 && rows.iter().any(|row| row.kind > 2) {
                    return Some(encode_err(
                        req_id,
                        ErrCode::InvalidRequest,
                        "native v1 cannot represent TurboQuant indexes; use RESP3 or native v2",
                    ));
                }
                // The binary proto's VindexInfo predates the LSM-debt fields;
                // it keeps its wire shape and drops them.
                let info: Vec<skeg_proto::VindexInfo> = rows
                    .into_iter()
                    .map(|row| skeg_proto::VindexInfo {
                        name: row.name,
                        dim: row.dim,
                        kind: row.kind,
                        backend: row.backend,
                        n_vectors: row.n_vectors,
                    })
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
            let name = match native_index_name(req_id, &name) {
                Ok(n) => n,
                Err(frame) => return Some(frame),
            };
            if let Err(message) = native_vindex_kind_is_allowed(frame.header.version, kind) {
                return Some(encode_err(req_id, ErrCode::InvalidRequest, message));
            }
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
            let name = match native_index_name(req_id, &name) {
                Ok(n) => n,
                Err(frame) => return Some(frame),
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
            let name = match native_index_name(req_id, &name) {
                Ok(n) => n,
                Err(frame) => return Some(frame),
            };
            match shards.vset(name, id, vector, 0, None, None).await {
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
            let name = match native_index_name(req_id, &name) {
                Ok(n) => n,
                Err(frame) => return Some(frame),
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
            let name = match native_index_name(req_id, &name) {
                Ok(n) => n,
                Err(frame) => return Some(frame),
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
            let name = match native_index_name(req_id, &name) {
                Ok(n) => n,
                Err(frame) => return Some(frame),
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
            // Native wire stays payload-less in P1a; drop the (always-None)
            // blob and encode (id, score) pairs as before.
            match shards
                .vsearch(name, query, k as usize, l_search, 0, false, None)
                .await
            {
                Ok(hits) => {
                    span.record("hits", hits.len());
                    let pairs: Vec<(u64, f32)> =
                        hits.into_iter().map(|(id, score, _)| (id, score)).collect();
                    Some(encode_ok_vsearch(req_id, &pairs))
                }
                Err(e) => Some(shard_err_to_response(req_id, &e)),
            }
        }

        skeg_proto::Op::NativeHello => {
            if frame.header.version != VERSION_V2 {
                Some(encode_err(
                    req_id,
                    ErrCode::InvalidRequest,
                    "native hello requires protocol version 2",
                ))
            } else if !payload.is_empty() {
                Some(encode_err(
                    req_id,
                    ErrCode::InvalidRequest,
                    "native hello payload must be empty",
                ))
            } else {
                Some(encode_ok_native_capabilities(
                    req_id,
                    NativeCapabilities {
                        protocol_version: VERSION_V2,
                        vector_kind_mask: 0b0011_1111,
                    },
                ))
            }
        }

        op => Some(encode_err(
            req_id,
            ErrCode::InvalidRequest,
            &format!("op {op:?} not implemented"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skeg_proto::{VERSION_V1, VERSION_V2};

    #[test]
    fn v1_pq_discriminator_is_rejected_instead_of_becoming_tq1() {
        assert_eq!(
            native_vindex_kind_is_allowed(VERSION_V1, 3),
            Err("native v1 kind 3 is PQ; use RESP3 or native v2")
        );
    }

    #[test]
    fn v2_turboquant_discriminators_are_accepted() {
        assert_eq!(native_vindex_kind_is_allowed(VERSION_V2, 3), Ok(()));
        assert_eq!(native_vindex_kind_is_allowed(VERSION_V2, 5), Ok(()));
    }

    /// Feed one encoded request through the real dispatcher and return the
    /// response frame. The native path has no tenant of its own - it is always
    /// tenant 0 with the client's raw name - so this is the door where a
    /// crafted scope prefix used to walk straight in.
    async fn native_roundtrip(shards: &ShardSet, request: Bytes) -> Frame {
        let mut parser = FrameParser::new();
        let mut buf = BytesMut::from(&request[..]);
        let frame = parser
            .feed(&mut buf)
            .expect("the request must parse")
            .expect("one whole frame");
        let response = dispatch(&frame, shards).await.expect("a response");
        let mut parser = FrameParser::new();
        let mut buf = BytesMut::from(&response[..]);
        parser
            .feed(&mut buf)
            .expect("the response must parse")
            .expect("one whole frame")
    }

    /// `[u8 code][u8 len][msg]` - the shape `encode_err` writes.
    fn err_message(frame: &Frame) -> String {
        assert_eq!(frame.header.op, skeg_proto::Op::Err, "expected an error");
        let n = frame.payload[1] as usize;
        String::from_utf8_lossy(&frame.payload[2..2 + n]).into_owned()
    }

    #[tokio::test]
    async fn a_native_create_cannot_spell_another_tenants_scope() {
        // `VINDEX.CREATE "<32 hex of B>::x"` over the binary protocol. There is
        // no AUTH here and never was: the handler passes the raw name to
        // `vindex_create` with tenant 0, so the refusal has to live at the
        // ShardSet door or it does not exist for this protocol at all.
        let dir = tempfile::TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 1).unwrap();
        let squat = format!("{}::x", "2a".repeat(16));

        let frame = native_roundtrip(
            &shards,
            skeg_proto::encode_vindex_create(7, &squat, 8, 0, 0),
        )
        .await;
        let msg = err_message(&frame);
        assert!(
            msg.contains("must not contain '::'"),
            "the native door must refuse the scope separator, got: {msg}"
        );

        let rows = shards.vindex_list().await.unwrap();
        assert!(
            !rows.iter().any(|r| r.name == squat),
            "the refused name must not exist: {rows:?}"
        );
    }

    #[tokio::test]
    async fn no_native_op_can_address_a_tenant_scoped_index() {
        // Closing the CREATE door stops a client MAKING a key that reads as
        // another tenant's. It does not stop one NAMING a key that already
        // exists: a tenant's index, created over RESP3 from an authenticated
        // id, is a plain string to the native listener, which has no tenant of
        // its own and passes the name straight through. So VGET reads that
        // tenant's vectors, VSET writes into its index, and VDEL / VINDEX.DROP
        // destroy them - from a connection that never authenticated as anyone.
        //
        // Every native arm that takes an index name refuses the separator, and
        // this walks all of them: one arm left out is one whole op still open.
        let dir = tempfile::TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 1).unwrap();
        // The victim: created the way the RESP3 layer creates it, with a
        // prefix the server wrote from an id it authenticated.
        let theirs = format!("{}::idx", "2a".repeat(16));
        shards.vindex_create_scoped(&theirs, 4, 0, 0).await.unwrap();
        shards
            .vset(&theirs, 1, vec![1.0, 0.0, 0.0, 0.0], 0x2a2a, None, None)
            .await
            .unwrap();

        for (op, request) in [
            ("VGET", skeg_proto::encode_vget(1, &theirs, 1)),
            (
                "VSET",
                skeg_proto::encode_vset(2, &theirs, 1, &[0.0, 1.0, 0.0, 0.0], Flags::empty()),
            ),
            ("VDEL", skeg_proto::encode_vdel(3, &theirs, 1)),
            (
                "VSEARCH",
                skeg_proto::encode_vsearch(4, &theirs, 1, &[1.0, 0.0, 0.0, 0.0]),
            ),
            ("VINDEX.DROP", skeg_proto::encode_vindex_drop(5, &theirs)),
        ] {
            let frame = native_roundtrip(&shards, request).await;
            let msg = err_message(&frame);
            assert!(
                msg.contains("must not contain '::'"),
                "native {op} reached another tenant's index; answered: {msg}"
            );
        }

        // Nothing moved: the index is still catalogued and the row still holds
        // the vector it was written with.
        let rows = shards.vindex_list().await.unwrap();
        assert!(
            rows.iter().any(|r| r.name == theirs),
            "the tenant's index was dropped from the native listener: {rows:?}"
        );
        assert_eq!(
            shards.vget(&theirs, 1).await.unwrap(),
            Some(vec![1.0, 0.0, 0.0, 0.0]),
            "the tenant's row was overwritten or deleted from the native listener"
        );
    }

    #[tokio::test]
    async fn native_vindex_list_does_not_name_other_tenants_indexes() {
        // The last asymmetry between the two handlers over one store. RESP3
        // hides every `::` name from an anonymous connection; the native arm
        // listed the lot - scoped name, dim, kind and vector count for every
        // index of every tenant, to a connection that authenticated as nobody.
        // The scoped name IS the tenant id in hex, so the listing enumerates
        // the tenants as well as their indexes.
        //
        // Metadata only - the ops themselves refuse `::` - but two handlers on
        // one store answering differently is a defect, not a decision.
        let dir = tempfile::TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 1).unwrap();
        let theirs = format!("{}::idx", "2a".repeat(16));
        shards.vindex_create_scoped(&theirs, 4, 0, 0).await.unwrap();
        shards.vindex_create("mine", 4, 0, 0).await.unwrap();

        let frame = native_roundtrip(&shards, skeg_proto::encode_vindex_list(1)).await;
        assert_eq!(frame.header.op, skeg_proto::Op::Ok, "expected a listing");
        let rows = skeg_proto::decode_vindex_list_response(&frame.payload);
        let names: Vec<&str> = rows.iter().map(|r| r.name.as_str()).collect();
        assert!(
            !names.iter().any(|n| n.contains("::")),
            "the native listing names another tenant's index: {names:?}"
        );
        assert!(
            names.contains(&"mine"),
            "hiding scoped names must not hide this listener's own: {names:?}"
        );
    }
}
