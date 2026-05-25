//! Wire codecs for the vector ops: VINDEX CREATE/DROP, VSET, VGET, VDEL,
//! VSEARCH.
//!
//! Vectors travel as little-endian f32. The `kind` byte is raw on the wire
//! (0 = f32, 1 = int8, 2 = binary); the server maps it to its quantizer type.

use bytes::{BufMut, Bytes, BytesMut};

use crate::{Flags, Op, ParseError, frame::encode_frame};

// ── helpers ───────────────────────────────────────────────────────────────────

#[allow(clippy::cast_possible_truncation)] // index names are short by construction
fn put_name(p: &mut BytesMut, name: &str) {
    p.put_u16_le(name.len() as u16);
    p.put_slice(name.as_bytes());
}

#[allow(clippy::cast_possible_truncation)] // dims fit u32 for any real embedding
fn put_f32_vec(p: &mut BytesMut, v: &[f32]) {
    p.put_u32_le(v.len() as u32);
    for &x in v {
        p.put_f32_le(x);
    }
}

/// Read a `[u16 len][bytes]` name at `pos`; returns `(name, next_pos)`.
fn read_name(payload: &Bytes, pos: usize) -> Result<(Bytes, usize), ParseError> {
    if pos + 2 > payload.len() {
        return Err(ParseError::InvalidPayload {
            msg: "vector name header truncated",
        });
    }
    let len = u16::from_le_bytes([payload[pos], payload[pos + 1]]) as usize;
    let start = pos + 2;
    if start + len > payload.len() {
        return Err(ParseError::InvalidPayload {
            msg: "vector name truncated",
        });
    }
    Ok((payload.slice(start..start + len), start + len))
}

/// Read a `[u32 dim][f32 x dim]` vector at `pos`; returns `(vec, next_pos)`.
fn read_f32_vec(payload: &Bytes, pos: usize) -> Result<(Vec<f32>, usize), ParseError> {
    if pos + 4 > payload.len() {
        return Err(ParseError::InvalidPayload {
            msg: "vector dim header truncated",
        });
    }
    let dim = u32::from_le_bytes([
        payload[pos],
        payload[pos + 1],
        payload[pos + 2],
        payload[pos + 3],
    ]) as usize;
    let start = pos + 4;
    let end = start + dim * 4;
    if end > payload.len() {
        return Err(ParseError::InvalidPayload {
            msg: "vector body truncated",
        });
    }
    let mut v = Vec::with_capacity(dim);
    for chunk in payload[start..end].chunks_exact(4) {
        v.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok((v, end))
}

fn read_u64(payload: &Bytes, pos: usize) -> Result<u64, ParseError> {
    if pos + 8 > payload.len() {
        return Err(ParseError::InvalidPayload {
            msg: "vector u64 field truncated",
        });
    }
    Ok(u64::from_le_bytes([
        payload[pos],
        payload[pos + 1],
        payload[pos + 2],
        payload[pos + 3],
        payload[pos + 4],
        payload[pos + 5],
        payload[pos + 6],
        payload[pos + 7],
    ]))
}

fn read_u32(payload: &Bytes, pos: usize) -> Result<u32, ParseError> {
    if pos + 4 > payload.len() {
        return Err(ParseError::InvalidPayload {
            msg: "vector u32 field truncated",
        });
    }
    Ok(u32::from_le_bytes([
        payload[pos],
        payload[pos + 1],
        payload[pos + 2],
        payload[pos + 3],
    ]))
}

// ── encoding (client side) ────────────────────────────────────────────────────

/// Encode VINDEX CREATE. Payload: `[u16 nlen][name][u32 dim][u8 kind][u8 backend]`.
///
/// `backend`: 0 = flat (in-RAM `FlatIndex`), 1 = disk Vamana graph.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn encode_vindex_create(req_id: u64, name: &str, dim: u32, kind: u8, backend: u8) -> Bytes {
    let mut p = BytesMut::with_capacity(2 + name.len() + 6);
    put_name(&mut p, name);
    p.put_u32_le(dim);
    p.put_u8(kind);
    p.put_u8(backend);
    encode_frame(Op::VindexCreate, Flags::empty(), req_id, &p)
}

/// Encode VINDEX DROP. Payload: `[u16 nlen][name]`.
#[must_use]
pub fn encode_vindex_drop(req_id: u64, name: &str) -> Bytes {
    let mut p = BytesMut::with_capacity(2 + name.len());
    put_name(&mut p, name);
    encode_frame(Op::VindexDrop, Flags::empty(), req_id, &p)
}

/// Encode VINDEX LIST request. Empty payload.
#[must_use]
pub fn encode_vindex_list(req_id: u64) -> Bytes {
    encode_frame(Op::VindexList, Flags::empty(), req_id, &[])
}

/// Encode VSET. Payload: `[u16 nlen][name][u64 vec_id][u32 dim][f32 x dim]`.
#[must_use]
pub fn encode_vset(req_id: u64, name: &str, vec_id: u64, vector: &[f32], flags: Flags) -> Bytes {
    let mut p = BytesMut::with_capacity(2 + name.len() + 12 + vector.len() * 4);
    put_name(&mut p, name);
    p.put_u64_le(vec_id);
    put_f32_vec(&mut p, vector);
    encode_frame(Op::Vset, flags, req_id, &p)
}

/// Encode VGET. Payload: `[u16 nlen][name][u64 vec_id]`.
#[must_use]
pub fn encode_vget(req_id: u64, name: &str, vec_id: u64) -> Bytes {
    let mut p = BytesMut::with_capacity(2 + name.len() + 8);
    put_name(&mut p, name);
    p.put_u64_le(vec_id);
    encode_frame(Op::Vget, Flags::empty(), req_id, &p)
}

/// Encode VDEL. Payload: `[u16 nlen][name][u64 vec_id]`.
#[must_use]
pub fn encode_vdel(req_id: u64, name: &str, vec_id: u64) -> Bytes {
    let mut p = BytesMut::with_capacity(2 + name.len() + 8);
    put_name(&mut p, name);
    p.put_u64_le(vec_id);
    encode_frame(Op::Vdel, Flags::empty(), req_id, &p)
}

/// Encode VSEARCH. Payload: `[u16 nlen][name][u32 k][u32 dim][f32 x dim]`.
#[must_use]
pub fn encode_vsearch(req_id: u64, name: &str, k: u32, query: &[f32]) -> Bytes {
    let mut p = BytesMut::with_capacity(2 + name.len() + 8 + query.len() * 4);
    put_name(&mut p, name);
    p.put_u32_le(k);
    put_f32_vec(&mut p, query);
    encode_frame(Op::Vsearch, Flags::empty(), req_id, &p)
}

/// Encode an Ok VSEARCH result: `[u32 n][[u64 id][f32 score]] x n`.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn encode_ok_vsearch(req_id: u64, hits: &[(u64, f32)]) -> Bytes {
    let mut p = BytesMut::with_capacity(4 + hits.len() * 12);
    p.put_u32_le(hits.len() as u32);
    for &(id, score) in hits {
        p.put_u64_le(id);
        p.put_f32_le(score);
    }
    encode_frame(Op::Ok, Flags::empty(), req_id, &p)
}

// ── decoding (server side) ────────────────────────────────────────────────────

/// Decoded VINDEX CREATE: `(name, dim, kind, backend)`.
///
/// # Errors
///
/// Returns `ParseError::InvalidPayload` if the payload is malformed.
pub fn decode_vindex_create_payload(payload: &Bytes) -> Result<(Bytes, u32, u8, u8), ParseError> {
    let (name, pos) = read_name(payload, 0)?;
    let dim = read_u32(payload, pos)?;
    let kind = *payload.get(pos + 4).ok_or(ParseError::InvalidPayload {
        msg: "vindex kind byte missing",
    })?;
    let backend = *payload.get(pos + 5).ok_or(ParseError::InvalidPayload {
        msg: "vindex backend byte missing",
    })?;
    Ok((name, dim, kind, backend))
}

/// Decode a payload carrying only an index name (VINDEX DROP).
///
/// # Errors
///
/// Returns `ParseError::InvalidPayload` if the payload is malformed.
pub fn decode_vname_payload(payload: &Bytes) -> Result<Bytes, ParseError> {
    let (name, _) = read_name(payload, 0)?;
    Ok(name)
}

/// Decode a `(name, vec_id)` payload (VGET, VDEL).
///
/// # Errors
///
/// Returns `ParseError::InvalidPayload` if the payload is malformed.
pub fn decode_vname_id_payload(payload: &Bytes) -> Result<(Bytes, u64), ParseError> {
    let (name, pos) = read_name(payload, 0)?;
    let id = read_u64(payload, pos)?;
    Ok((name, id))
}

/// Decode a VSET payload: `(name, vec_id, vector)`.
///
/// # Errors
///
/// Returns `ParseError::InvalidPayload` if the payload is malformed.
pub fn decode_vset_payload(payload: &Bytes) -> Result<(Bytes, u64, Vec<f32>), ParseError> {
    let (name, pos) = read_name(payload, 0)?;
    let id = read_u64(payload, pos)?;
    let (vector, _) = read_f32_vec(payload, pos + 8)?;
    Ok((name, id, vector))
}

/// Decode a VSEARCH payload: `(name, k, query, l_search)`.
///
/// The trailing `[u32 l_search]` is optional: a frame without it decodes to
/// `l_search = 0` ("use the index default"); a non-zero value overrides the
/// query-time search-list size. Old clients (no trailing field) keep working.
///
/// # Errors
///
/// Returns `ParseError::InvalidPayload` if the payload is malformed.
pub fn decode_vsearch_payload(payload: &Bytes) -> Result<(Bytes, u32, Vec<f32>, u32), ParseError> {
    let (name, pos) = read_name(payload, 0)?;
    let k = read_u32(payload, pos)?;
    let (query, end) = read_f32_vec(payload, pos + 4)?;
    let l_search = read_u32(payload, end).unwrap_or(0);
    Ok((name, k, query, l_search))
}

/// Decode a VSEARCH Ok response into `(id, score)` hits.
#[must_use]
pub fn decode_vsearch_response(payload: &Bytes) -> Vec<(u64, f32)> {
    if payload.len() < 4 {
        return Vec::new();
    }
    let n = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
    let mut hits = Vec::with_capacity(n);
    let mut pos = 4;
    for _ in 0..n {
        if pos + 12 > payload.len() {
            break;
        }
        let id = u64::from_le_bytes([
            payload[pos],
            payload[pos + 1],
            payload[pos + 2],
            payload[pos + 3],
            payload[pos + 4],
            payload[pos + 5],
            payload[pos + 6],
            payload[pos + 7],
        ]);
        let score = f32::from_le_bytes([
            payload[pos + 8],
            payload[pos + 9],
            payload[pos + 10],
            payload[pos + 11],
        ]);
        hits.push((id, score));
        pos += 12;
    }
    hits
}

/// Encode an f32 vector as little-endian bytes (VGET response value body).
#[must_use]
pub fn f32_vec_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Decode little-endian f32 bytes back into a vector (VGET response value).
#[must_use]
pub fn bytes_to_f32_vec(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FrameParser;

    fn parse_one(b: Bytes) -> crate::Frame {
        FrameParser::new()
            .feed(&mut BytesMut::from_iter(b))
            .unwrap()
            .unwrap()
    }

    #[test]
    fn vindex_create_roundtrip() {
        let frame = parse_one(encode_vindex_create(1, "embeddings", 1536, 1, 1));
        assert_eq!(frame.header.op, Op::VindexCreate);
        let (name, dim, kind, backend) = decode_vindex_create_payload(&frame.payload).unwrap();
        assert_eq!(name.as_ref(), b"embeddings");
        assert_eq!(dim, 1536);
        assert_eq!(kind, 1);
        assert_eq!(backend, 1);
    }

    #[test]
    fn vindex_drop_roundtrip() {
        let frame = parse_one(encode_vindex_drop(2, "idx"));
        assert_eq!(frame.header.op, Op::VindexDrop);
        assert_eq!(
            decode_vname_payload(&frame.payload).unwrap().as_ref(),
            b"idx"
        );
    }

    #[test]
    fn vset_roundtrip() {
        let vector = [0.5f32, -1.0, 2.25, 0.0, 7.5];
        let frame = parse_one(encode_vset(3, "idx", 99, &vector, Flags::empty()));
        assert_eq!(frame.header.op, Op::Vset);
        let (name, id, v) = decode_vset_payload(&frame.payload).unwrap();
        assert_eq!(name.as_ref(), b"idx");
        assert_eq!(id, 99);
        assert_eq!(v, vector);
    }

    #[test]
    fn vget_vdel_roundtrip() {
        let g = parse_one(encode_vget(4, "idx", 7));
        assert_eq!(g.header.op, Op::Vget);
        assert_eq!(
            decode_vname_id_payload(&g.payload).unwrap(),
            (Bytes::from_static(b"idx"), 7)
        );
        let d = parse_one(encode_vdel(5, "idx", 8));
        assert_eq!(d.header.op, Op::Vdel);
        assert_eq!(
            decode_vname_id_payload(&d.payload).unwrap(),
            (Bytes::from_static(b"idx"), 8)
        );
    }

    #[test]
    fn vsearch_roundtrip() {
        let query = [1.0f32, 2.0, 3.0, 4.0];
        let frame = parse_one(encode_vsearch(6, "idx", 10, &query));
        assert_eq!(frame.header.op, Op::Vsearch);
        let (name, k, q, l_search) = decode_vsearch_payload(&frame.payload).unwrap();
        assert_eq!(name.as_ref(), b"idx");
        assert_eq!(k, 10);
        assert_eq!(q, query);
        assert_eq!(l_search, 0, "frame without trailing l_search decodes to 0");
    }

    #[test]
    fn vsearch_response_roundtrip() {
        let hits = [(3u64, 0.99f32), (17, 0.81), (4, 0.5)];
        let frame = parse_one(encode_ok_vsearch(7, &hits));
        let decoded = decode_vsearch_response(&frame.payload);
        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[0], (3, 0.99));
        assert_eq!(decoded[1], (17, 0.81));
        assert_eq!(decoded[2], (4, 0.5));
    }

    #[test]
    fn vsearch_response_empty() {
        let frame = parse_one(encode_ok_vsearch(1, &[]));
        assert!(decode_vsearch_response(&frame.payload).is_empty());
    }

    #[test]
    fn f32_vec_byte_roundtrip() {
        let v = [0.0f32, -3.5, 100.25, 1e-9];
        assert_eq!(bytes_to_f32_vec(&f32_vec_to_bytes(&v)), v);
    }

    #[test]
    fn decode_truncated_payloads_error() {
        assert!(decode_vname_payload(&Bytes::from_static(&[0x05])).is_err());
        assert!(decode_vset_payload(&Bytes::from_static(&[0, 0, 1, 2])).is_err());
        assert!(decode_vsearch_payload(&Bytes::from_static(&[0, 0])).is_err());
        // name says 3 bytes but only 1 follows
        assert!(decode_vindex_create_payload(&Bytes::from_static(&[3, 0, b'x'])).is_err());
    }
}
