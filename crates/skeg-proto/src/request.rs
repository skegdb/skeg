//! Request payload encoding (client → server) and decoding (server-side).

use bytes::{BufMut, Bytes, BytesMut};

use crate::{Flags, Op, ParseError, frame::encode_frame};

// ── Encoding ────────────────────────────────────────────────────────────────

/// Encode a PING request (empty payload).
#[must_use]
pub fn encode_ping(req_id: u64) -> Bytes {
    encode_frame(Op::Ping, Flags::empty(), req_id, &[])
}

/// Encode a STATS request (empty payload).
#[must_use]
pub fn encode_stats(req_id: u64) -> Bytes {
    encode_frame(Op::Stats, Flags::empty(), req_id, &[])
}

/// Encode a SHARDS request (empty payload). Server returns per-shard
/// stats; aggregate `STATS` is the sum across these.
#[must_use]
pub fn encode_shards(req_id: u64) -> Bytes {
    encode_frame(Op::Shards, Flags::empty(), req_id, &[])
}

/// Encode a GET request. Payload: `[u16 key_len][key]`.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn encode_get(req_id: u64, key: &[u8]) -> Bytes {
    let mut p = BytesMut::with_capacity(2 + key.len());
    p.put_u16_le(key.len() as u16);
    p.put_slice(key);
    encode_frame(Op::Get, Flags::empty(), req_id, &p)
}

/// Encode a SET request. Payload: `[u16 key_len][u32 value_len][key][value]`.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn encode_set(req_id: u64, key: &[u8], value: &[u8], flags: Flags) -> Bytes {
    let mut p = BytesMut::with_capacity(6 + key.len() + value.len());
    p.put_u16_le(key.len() as u16);
    p.put_u32_le(value.len() as u32);
    p.put_slice(key);
    p.put_slice(value);
    encode_frame(Op::Set, flags, req_id, &p)
}

/// Encode a DEL request. Payload: `[u16 key_len][key]`.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn encode_del(req_id: u64, key: &[u8]) -> Bytes {
    let mut p = BytesMut::with_capacity(2 + key.len());
    p.put_u16_le(key.len() as u16);
    p.put_slice(key);
    encode_frame(Op::Del, Flags::empty(), req_id, &p)
}

/// Encode a MGET request. Payload: `[u32 n_keys][[u16 key_len][key]] × n`.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn encode_mget(req_id: u64, keys: &[&[u8]]) -> Bytes {
    let total = 4 + keys.iter().map(|k| 2 + k.len()).sum::<usize>();
    let mut p = BytesMut::with_capacity(total);
    p.put_u32_le(keys.len() as u32);
    for key in keys {
        p.put_u16_le(key.len() as u16);
        p.put_slice(key);
    }
    encode_frame(Op::Mget, Flags::empty(), req_id, &p)
}

// ── Decoding ────────────────────────────────────────────────────────────────

/// Decode a single-key payload (GET, DEL): `[u16 key_len][key]` → key slice.
///
/// # Errors
///
/// Returns `Err(ParseError::InvalidPayload)` if the payload is too short or truncated.
pub fn decode_key_payload(payload: &Bytes) -> Result<Bytes, ParseError> {
    if payload.len() < 2 {
        return Err(ParseError::InvalidPayload {
            msg: "key payload too short",
        });
    }
    let key_len = u16::from_le_bytes([payload[0], payload[1]]) as usize;
    if payload.len() < 2 + key_len {
        return Err(ParseError::InvalidPayload {
            msg: "key payload truncated",
        });
    }
    Ok(payload.slice(2..2 + key_len))
}

/// Decode a SET payload: `[u16 key_len][u32 val_len][key][value]` → `(key, value)`.
///
/// # Errors
///
/// Returns `Err(ParseError::InvalidPayload)` if the payload is too short or truncated.
pub fn decode_set_payload(payload: &Bytes) -> Result<(Bytes, Bytes), ParseError> {
    if payload.len() < 6 {
        return Err(ParseError::InvalidPayload {
            msg: "set payload too short",
        });
    }
    let key_len = u16::from_le_bytes([payload[0], payload[1]]) as usize;
    let val_len = u32::from_le_bytes([payload[2], payload[3], payload[4], payload[5]]) as usize;
    let needed = 6 + key_len + val_len;
    if payload.len() < needed {
        return Err(ParseError::InvalidPayload {
            msg: "set payload truncated",
        });
    }
    Ok((
        payload.slice(6..6 + key_len),
        payload.slice(6 + key_len..needed),
    ))
}

/// Decode a MGET payload: `[u32 n][[u16 key_len][key]] × n` → Vec of key slices.
///
/// # Errors
///
/// Returns `Err(ParseError::InvalidPayload)` if the payload is malformed or truncated.
pub fn decode_mget_payload(payload: &Bytes) -> Result<Vec<Bytes>, ParseError> {
    if payload.len() < 4 {
        return Err(ParseError::InvalidPayload {
            msg: "mget payload too short",
        });
    }
    let n = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
    let mut keys = Vec::with_capacity(n);
    let mut pos = 4usize;
    for _ in 0..n {
        if pos + 2 > payload.len() {
            return Err(ParseError::InvalidPayload {
                msg: "mget key header truncated",
            });
        }
        let key_len = u16::from_le_bytes([payload[pos], payload[pos + 1]]) as usize;
        pos += 2;
        if pos + key_len > payload.len() {
            return Err(ParseError::InvalidPayload {
                msg: "mget key truncated",
            });
        }
        keys.push(payload.slice(pos..pos + key_len));
        pos += key_len;
    }
    Ok(keys)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FrameParser, Op};
    use bytes::BytesMut;

    fn parse_one(b: Bytes) -> crate::Frame {
        let mut buf = BytesMut::from_iter(b);
        FrameParser::new().feed(&mut buf).unwrap().unwrap()
    }

    #[test]
    fn encode_decode_ping() {
        let frame = parse_one(encode_ping(42));
        assert_eq!(frame.header.op, Op::Ping);
        assert_eq!(frame.header.req_id, 42);
        assert!(frame.payload.is_empty());
    }

    #[test]
    fn encode_decode_get() {
        let frame = parse_one(encode_get(7, b"mykey"));
        assert_eq!(frame.header.op, Op::Get);
        let key = decode_key_payload(&frame.payload).unwrap();
        assert_eq!(key.as_ref(), b"mykey");
    }

    #[test]
    fn encode_decode_get_empty_key() {
        let frame = parse_one(encode_get(1, b""));
        let key = decode_key_payload(&frame.payload).unwrap();
        assert!(key.is_empty());
    }

    #[test]
    fn encode_decode_set() {
        let frame = parse_one(encode_set(3, b"k", b"v_value", Flags::empty()));
        assert_eq!(frame.header.op, Op::Set);
        let (key, val) = decode_set_payload(&frame.payload).unwrap();
        assert_eq!(key.as_ref(), b"k");
        assert_eq!(val.as_ref(), b"v_value");
    }

    #[test]
    fn encode_decode_set_no_reply_flag() {
        let frame = parse_one(encode_set(1, b"k", b"v", Flags::NO_REPLY));
        assert!(frame.header.flags.contains(Flags::NO_REPLY));
    }

    #[test]
    fn encode_decode_set_empty_value() {
        let frame = parse_one(encode_set(1, b"k", b"", Flags::empty()));
        let (key, val) = decode_set_payload(&frame.payload).unwrap();
        assert_eq!(key.as_ref(), b"k");
        assert!(val.is_empty());
    }

    #[test]
    fn encode_decode_del() {
        let frame = parse_one(encode_del(5, b"delkey"));
        assert_eq!(frame.header.op, Op::Del);
        let key = decode_key_payload(&frame.payload).unwrap();
        assert_eq!(key.as_ref(), b"delkey");
    }

    #[test]
    fn encode_decode_mget_multiple_keys() {
        let keys: &[&[u8]] = &[b"a", b"bb", b"ccc"];
        let frame = parse_one(encode_mget(9, keys));
        assert_eq!(frame.header.op, Op::Mget);
        let decoded = decode_mget_payload(&frame.payload).unwrap();
        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[0].as_ref(), b"a");
        assert_eq!(decoded[1].as_ref(), b"bb");
        assert_eq!(decoded[2].as_ref(), b"ccc");
    }

    #[test]
    fn encode_decode_mget_empty_list() {
        let frame = parse_one(encode_mget(1, &[]));
        let decoded = decode_mget_payload(&frame.payload).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn decode_key_payload_too_short_errors() {
        let p = Bytes::from_static(&[0x05]); // only 1 byte, needs 2 for len
        assert!(matches!(
            decode_key_payload(&p),
            Err(ParseError::InvalidPayload { .. })
        ));
    }

    #[test]
    fn decode_set_payload_too_short_errors() {
        let p = Bytes::from_static(&[0, 0, 0, 0]); // only 4 bytes, needs 6
        assert!(matches!(
            decode_set_payload(&p),
            Err(ParseError::InvalidPayload { .. })
        ));
    }

    #[test]
    fn binary_key_and_value_roundtrip() {
        let key = &[0u8, 1, 2, 128, 255];
        let val = &[10u8, 20, 0, 255, 99];
        let frame = parse_one(encode_set(1, key, val, Flags::empty()));
        let (k, v) = decode_set_payload(&frame.payload).unwrap();
        assert_eq!(k.as_ref(), key);
        assert_eq!(v.as_ref(), val);
    }
}
