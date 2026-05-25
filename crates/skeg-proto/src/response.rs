//! Response payload encoding (server → client) and decoding (client-side).

use bytes::{BufMut, Bytes, BytesMut};

use crate::{Flags, Op, frame::encode_frame};

/// Error codes sent in `Err` response frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ErrCode {
    NotFound = 0x01,
    InvalidRequest = 0x02,
    Internal = 0x03,
}

// ── Encoding ────────────────────────────────────────────────────────────────

/// Encode a plain Ok response (PING reply, etc.). Empty payload.
#[must_use]
pub fn encode_ok(req_id: u64) -> Bytes {
    encode_frame(Op::Ok, Flags::empty(), req_id, &[])
}

/// Encode Ok with a value payload: `[u32 value_len][value]`.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn encode_ok_value(req_id: u64, value: &[u8]) -> Bytes {
    let mut p = BytesMut::with_capacity(4 + value.len());
    p.put_u32_le(value.len() as u32);
    p.put_slice(value);
    encode_frame(Op::Ok, Flags::empty(), req_id, &p)
}

/// Encode Ok with a u64 payload (e.g., timestamp after SET): `[u64 val]`.
#[must_use]
pub fn encode_ok_u64(req_id: u64, val: u64) -> Bytes {
    encode_frame(Op::Ok, Flags::empty(), req_id, &val.to_le_bytes())
}

/// Encode Ok with a bool payload (1 byte): used for DEL.
#[must_use]
pub fn encode_ok_bool(req_id: u64, val: bool) -> Bytes {
    encode_frame(Op::Ok, Flags::empty(), req_id, &[u8::from(val)])
}

/// Encode Ok with MGET results.
///
/// Payload: `[u32 n][ [u8 status: 0=found|1=missing] [u32 vlen][value]? ] × n`.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn encode_ok_mget(req_id: u64, results: &[Option<Bytes>]) -> Bytes {
    let body_len: usize = results
        .iter()
        .map(|r| 1 + r.as_ref().map_or(0, |v| 4 + v.len()))
        .sum();
    let mut p = BytesMut::with_capacity(4 + body_len);
    p.put_u32_le(results.len() as u32);
    for result in results {
        match result {
            Some(v) => {
                p.put_u8(0); // found
                p.put_u32_le(v.len() as u32);
                p.put_slice(v);
            }
            None => {
                p.put_u8(1); // not found
            }
        }
    }
    encode_frame(Op::Ok, Flags::empty(), req_id, &p)
}

/// Aggregate server statistics, summed across shards.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ServerStats {
    /// Bytes currently held by the hot-key caches.
    pub cache_bytes: u64,
    /// Cache entries evicted over the server's lifetime.
    pub cache_evictions: u64,
    /// Live keys in the index.
    pub n_keys: u64,
    /// Configured cache byte budget.
    pub cache_budget: u64,
}

/// Per-shard statistics. The aggregate `ServerStats` is the sum of these
/// over all shards; clients that want a breakdown (TUI dashboards, ops
/// tooling) ask for the raw per-shard form via `Op::Shards`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ShardStats {
    pub shard_id: u32,
    pub cache_bytes: u64,
    pub cache_evictions: u64,
    pub n_keys: u64,
    pub cache_budget: u64,
}

/// One row of `Op::VindexList` response. `kind` and `backend` are the
/// same wire bytes accepted by `VINDEX CREATE` (0=f32 / 1=int8 / 2=binary
/// for `kind`; 0=flat / 1=disk Vamana for `backend`). `n_vectors` is the
/// live count across shards (summed by the client).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VindexInfo {
    pub name: String,
    pub dim: u32,
    pub kind: u8,
    pub backend: u8,
    pub n_vectors: u64,
}

/// Encode an Ok response carrying [`ServerStats`]: four little-endian u64s.
#[must_use]
pub fn encode_ok_stats(req_id: u64, stats: ServerStats) -> Bytes {
    let mut p = BytesMut::with_capacity(32);
    p.put_u64_le(stats.cache_bytes);
    p.put_u64_le(stats.cache_evictions);
    p.put_u64_le(stats.n_keys);
    p.put_u64_le(stats.cache_budget);
    encode_frame(Op::Ok, Flags::empty(), req_id, &p)
}

/// Encode an Ok response carrying VINDEX list rows.
///
/// Payload: `[u32 n][[u16 nlen][name][u32 dim][u8 kind][u8 backend]
/// [u64 n_vectors]] * n`.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn encode_ok_vindex_list(req_id: u64, rows: &[VindexInfo]) -> Bytes {
    // Worst-case size: 4 + per-row (2 + 64 + 4 + 1 + 1 + 8 = 80) = generous.
    let cap = 4 + rows
        .iter()
        .map(|r| 2 + r.name.len() + 4 + 1 + 1 + 8)
        .sum::<usize>();
    let mut p = BytesMut::with_capacity(cap);
    p.put_u32_le(rows.len() as u32);
    for r in rows {
        let bytes = r.name.as_bytes();
        p.put_u16_le(bytes.len() as u16);
        p.put_slice(bytes);
        p.put_u32_le(r.dim);
        p.put_u8(r.kind);
        p.put_u8(r.backend);
        p.put_u64_le(r.n_vectors);
    }
    encode_frame(Op::Ok, Flags::empty(), req_id, &p)
}

/// Encode an Ok response carrying per-shard stats: `[u32 n][[u32 shard_id]
/// [u64 cache_bytes][u64 cache_evictions][u64 n_keys][u64 cache_budget]] * n`.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn encode_ok_shards(req_id: u64, rows: &[ShardStats]) -> Bytes {
    let mut p = BytesMut::with_capacity(4 + rows.len() * 36);
    p.put_u32_le(rows.len() as u32);
    for r in rows {
        p.put_u32_le(r.shard_id);
        p.put_u64_le(r.cache_bytes);
        p.put_u64_le(r.cache_evictions);
        p.put_u64_le(r.n_keys);
        p.put_u64_le(r.cache_budget);
    }
    encode_frame(Op::Ok, Flags::empty(), req_id, &p)
}

/// Encode an Err response. Payload: `[u8 code][u8 msg_len][msg bytes]`.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn encode_err(req_id: u64, code: ErrCode, msg: &str) -> Bytes {
    let msg_bytes = msg.as_bytes();
    let msg_len = msg_bytes.len().min(255) as u8;
    let mut p = BytesMut::with_capacity(2 + msg_len as usize);
    p.put_u8(code as u8);
    p.put_u8(msg_len);
    p.put_slice(&msg_bytes[..msg_len as usize]);
    encode_frame(Op::Err, Flags::empty(), req_id, &p)
}

// ── Decoding ────────────────────────────────────────────────────────────────

/// Decode a value response payload `[u32 vlen][value]`. Returns `None` on malformed input.
#[must_use]
pub fn decode_value_response(payload: &Bytes) -> Option<Bytes> {
    if payload.len() < 4 {
        return None;
    }
    let val_len = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
    if payload.len() < 4 + val_len {
        return None;
    }
    Some(payload.slice(4..4 + val_len))
}

/// Decode a bool response (1-byte payload). Returns `false` on missing/malformed.
#[must_use]
pub fn decode_bool_response(payload: &Bytes) -> bool {
    payload.first().copied().unwrap_or(0) != 0
}

/// Decode a u64 response (8-byte LE payload). Returns `0` on malformed.
#[must_use]
pub fn decode_u64_response(payload: &Bytes) -> u64 {
    if payload.len() < 8 {
        return 0;
    }
    u64::from_le_bytes([
        payload[0], payload[1], payload[2], payload[3], payload[4], payload[5], payload[6],
        payload[7],
    ])
}

/// Decode a VINDEX list response. Returns an empty vec on malformed input.
#[must_use]
pub fn decode_vindex_list_response(payload: &Bytes) -> Vec<VindexInfo> {
    if payload.len() < 4 {
        return Vec::new();
    }
    let n = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
    let mut rows = Vec::with_capacity(n);
    let mut pos = 4usize;
    for _ in 0..n {
        if pos + 2 > payload.len() {
            break;
        }
        let nlen = u16::from_le_bytes([payload[pos], payload[pos + 1]]) as usize;
        pos += 2;
        if pos + nlen + 4 + 1 + 1 + 8 > payload.len() {
            break;
        }
        let name = String::from_utf8_lossy(&payload[pos..pos + nlen]).into_owned();
        pos += nlen;
        let dim = u32::from_le_bytes([
            payload[pos],
            payload[pos + 1],
            payload[pos + 2],
            payload[pos + 3],
        ]);
        pos += 4;
        let kind = payload[pos];
        pos += 1;
        let backend = payload[pos];
        pos += 1;
        let n_vectors = u64::from_le_bytes([
            payload[pos],
            payload[pos + 1],
            payload[pos + 2],
            payload[pos + 3],
            payload[pos + 4],
            payload[pos + 5],
            payload[pos + 6],
            payload[pos + 7],
        ]);
        pos += 8;
        rows.push(VindexInfo {
            name,
            dim,
            kind,
            backend,
            n_vectors,
        });
    }
    rows
}

/// Decode a per-shard stats response. Returns an empty vec on malformed input.
#[must_use]
pub fn decode_shards_response(payload: &Bytes) -> Vec<ShardStats> {
    if payload.len() < 4 {
        return Vec::new();
    }
    let n = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
    let row_size = 4 + 4 * 8; // 4-byte shard_id + 4 * u64 counters
    let mut rows = Vec::with_capacity(n);
    let mut pos = 4usize;
    let u32_at =
        |p: usize| u32::from_le_bytes([payload[p], payload[p + 1], payload[p + 2], payload[p + 3]]);
    let u64_at = |p: usize| {
        u64::from_le_bytes([
            payload[p],
            payload[p + 1],
            payload[p + 2],
            payload[p + 3],
            payload[p + 4],
            payload[p + 5],
            payload[p + 6],
            payload[p + 7],
        ])
    };
    for _ in 0..n {
        if pos + row_size > payload.len() {
            break;
        }
        rows.push(ShardStats {
            shard_id: u32_at(pos),
            cache_bytes: u64_at(pos + 4),
            cache_evictions: u64_at(pos + 12),
            n_keys: u64_at(pos + 20),
            cache_budget: u64_at(pos + 28),
        });
        pos += row_size;
    }
    rows
}

/// Decode a [`ServerStats`] response payload. Returns `None` if malformed.
#[must_use]
pub fn decode_stats_response(payload: &Bytes) -> Option<ServerStats> {
    if payload.len() < 32 {
        return None;
    }
    let u64_at = |i: usize| {
        u64::from_le_bytes([
            payload[i],
            payload[i + 1],
            payload[i + 2],
            payload[i + 3],
            payload[i + 4],
            payload[i + 5],
            payload[i + 6],
            payload[i + 7],
        ])
    };
    Some(ServerStats {
        cache_bytes: u64_at(0),
        cache_evictions: u64_at(8),
        n_keys: u64_at(16),
        cache_budget: u64_at(24),
    })
}

/// Decode a MGET response payload into `Vec<Option<Bytes>>`.
#[must_use]
pub fn decode_mget_response(payload: &Bytes) -> Vec<Option<Bytes>> {
    if payload.len() < 4 {
        return vec![];
    }
    let n = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
    let mut results = Vec::with_capacity(n);
    let mut pos = 4usize;
    for _ in 0..n {
        if pos >= payload.len() {
            break;
        }
        let status = payload[pos];
        pos += 1;
        if status == 0 {
            if pos + 4 > payload.len() {
                break;
            }
            let val_len = u32::from_le_bytes([
                payload[pos],
                payload[pos + 1],
                payload[pos + 2],
                payload[pos + 3],
            ]) as usize;
            pos += 4;
            if pos + val_len > payload.len() {
                break;
            }
            results.push(Some(payload.slice(pos..pos + val_len)));
            pos += val_len;
        } else {
            results.push(None);
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FrameParser;
    use bytes::BytesMut;

    fn parse_one(b: Bytes) -> crate::Frame {
        FrameParser::new()
            .feed(&mut BytesMut::from_iter(b))
            .unwrap()
            .unwrap()
    }

    #[test]
    fn encode_decode_ok_empty() {
        let frame = parse_one(encode_ok(1));
        assert_eq!(frame.header.op, Op::Ok);
        assert!(frame.payload.is_empty());
    }

    #[test]
    fn encode_decode_ok_value() {
        let frame = parse_one(encode_ok_value(1, b"hello"));
        let val = decode_value_response(&frame.payload).unwrap();
        assert_eq!(val.as_ref(), b"hello");
    }

    #[test]
    fn encode_decode_ok_value_empty() {
        let frame = parse_one(encode_ok_value(1, b""));
        let val = decode_value_response(&frame.payload).unwrap();
        assert!(val.is_empty());
    }

    #[test]
    fn encode_decode_ok_u64() {
        let frame = parse_one(encode_ok_u64(1, 0xDEAD_BEEF_1234_5678));
        let val = decode_u64_response(&frame.payload);
        assert_eq!(val, 0xDEAD_BEEF_1234_5678);
    }

    #[test]
    fn encode_decode_ok_bool_true() {
        let frame = parse_one(encode_ok_bool(1, true));
        assert!(decode_bool_response(&frame.payload));
    }

    #[test]
    fn encode_decode_ok_bool_false() {
        let frame = parse_one(encode_ok_bool(1, false));
        assert!(!decode_bool_response(&frame.payload));
    }

    #[test]
    fn encode_decode_ok_mget_mixed() {
        let results: Vec<Option<Bytes>> = vec![
            Some(Bytes::from_static(b"val1")),
            None,
            Some(Bytes::from_static(b"val3")),
            None,
        ];
        let frame = parse_one(encode_ok_mget(1, &results));
        let decoded = decode_mget_response(&frame.payload);
        assert_eq!(decoded.len(), 4);
        assert_eq!(decoded[0].as_deref(), Some(b"val1".as_ref()));
        assert!(decoded[1].is_none());
        assert_eq!(decoded[2].as_deref(), Some(b"val3".as_ref()));
        assert!(decoded[3].is_none());
    }

    #[test]
    fn encode_decode_ok_mget_empty() {
        let frame = parse_one(encode_ok_mget(1, &[]));
        assert!(decode_mget_response(&frame.payload).is_empty());
    }

    #[test]
    fn encode_decode_err_not_found() {
        let frame = parse_one(encode_err(1, ErrCode::NotFound, "key not found"));
        assert_eq!(frame.header.op, Op::Err);
        assert_eq!(frame.payload[0], ErrCode::NotFound as u8);
    }

    #[test]
    fn encode_err_long_message_truncated_to_255() {
        let long_msg = "x".repeat(300);
        let frame = parse_one(encode_err(1, ErrCode::Internal, &long_msg));
        let msg_len = frame.payload[1] as usize;
        assert!(msg_len <= 255);
    }

    #[test]
    fn decode_value_response_too_short_returns_none() {
        assert!(decode_value_response(&Bytes::from_static(&[0, 0, 0])).is_none());
    }

    #[test]
    fn decode_bool_empty_payload_returns_false() {
        assert!(!decode_bool_response(&Bytes::new()));
    }

    #[test]
    fn decode_u64_short_payload_returns_zero() {
        assert_eq!(decode_u64_response(&Bytes::from_static(&[1, 2, 3])), 0);
    }
}
