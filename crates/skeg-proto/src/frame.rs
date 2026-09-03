//! Binary frame format.
//!
//! Header layout (24 bytes, all little-endian):
//! ```text
//! [magic u16][version u8][op u8][flags u32][req_id u64][payload_len u32][reserved u32]
//!  2B         1B          1B    4B          8B           4B               4B
//! ```

use bytes::{Buf, BufMut, Bytes, BytesMut};

use crate::{Flags, Op, ParseError};

pub const MAGIC: u16 = 0x564B; // "KV" in little-endian
/// Native protocol version emitted by the compatibility encoder.
pub const VERSION_V1: u8 = 1;
/// Native protocol version with explicit TurboQuant kind discriminants.
pub const VERSION_V2: u8 = 2;
/// Backwards-compatible spelling for the v1 encoder version.
pub const VERSION: u8 = VERSION_V1;
pub const HEADER_LEN: usize = 24;
pub const MAX_FRAME_SIZE: u32 = 16 * 1024 * 1024; // 16 MiB

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameHeader {
    /// Native protocol version carried by this frame.
    pub version: u8,
    pub op: Op,
    pub flags: Flags,
    pub req_id: u64,
    pub payload_len: u32,
}

#[derive(Debug, Clone)]
pub struct Frame {
    pub header: FrameHeader,
    pub payload: Bytes,
}

enum ParseState {
    Header {
        buf: [u8; HEADER_LEN],
        filled: usize,
    },
    Payload {
        header: FrameHeader,
        buf: BytesMut,
    },
}

/// Streaming frame parser. Feed bytes incrementally; get `Frame`s out.
///
/// The parser is a state machine: it accumulates bytes until a full header
/// (24 bytes) is parsed, then accumulates `payload_len` more bytes.
pub struct FrameParser {
    state: ParseState,
    /// The largest payload this parser will accept, checked at the HEADER.
    ///
    /// Defaults to [`MAX_FRAME_SIZE`]. A server that gives its connections a
    /// smaller allowance passes it here so the refusal happens on the 24 bytes
    /// that declared the length, not after the bytes have been buffered.
    limit: u32,
}

impl FrameParser {
    #[must_use]
    pub fn new() -> Self {
        Self::with_limit(MAX_FRAME_SIZE)
    }

    /// A parser that refuses any frame declaring more than `limit` payload
    /// bytes.
    ///
    /// `limit` is clamped to [`MAX_FRAME_SIZE`]: a caller cannot raise the
    /// protocol ceiling, only lower it for itself.
    #[must_use]
    pub fn with_limit(limit: u32) -> Self {
        Self {
            state: ParseState::Header {
                buf: [0u8; HEADER_LEN],
                filled: 0,
            },
            limit: limit.min(MAX_FRAME_SIZE),
        }
    }

    /// The largest payload this parser accepts.
    #[must_use]
    pub fn limit(&self) -> u32 {
        self.limit
    }

    /// Bytes the parser has actually ALLOCATED for the frame in flight.
    ///
    /// Reported, not just held: the whole hazard of a length-prefixed protocol
    /// is a declared length that allocates before anything arrives, and a
    /// number nobody can read is a promise nobody can check.
    #[must_use]
    pub fn buffered_capacity(&self) -> usize {
        match &self.state {
            ParseState::Header { .. } => 0,
            ParseState::Payload { buf, .. } => buf.capacity(),
        }
    }

    /// Feed bytes from the network into the parser.
    ///
    /// Returns `Ok(Some(frame))` when a full frame is ready.
    /// Returns `Ok(None)` when more bytes are needed.
    ///
    /// `input` is advanced past the bytes consumed.
    ///
    /// # Errors
    ///
    /// Returns `Err(ParseError)` on a protocol violation - caller should drop the connection.
    pub fn feed(&mut self, input: &mut BytesMut) -> Result<Option<Frame>, ParseError> {
        loop {
            match &mut self.state {
                ParseState::Header { buf, filled } => {
                    let need = HEADER_LEN - *filled;
                    let take = input.len().min(need);
                    buf[*filled..*filled + take].copy_from_slice(&input[..take]);
                    *filled += take;
                    input.advance(take);

                    if *filled < HEADER_LEN {
                        return Ok(None);
                    }

                    let header = parse_header(buf)?;
                    let cap = header.payload_len as usize;
                    self.state = ParseState::Payload {
                        header,
                        buf: BytesMut::with_capacity(cap),
                    };
                    // fall through to try payload on the same call
                }

                ParseState::Payload { header, buf } => {
                    let need = header.payload_len as usize - buf.len();
                    let take = input.len().min(need);
                    buf.extend_from_slice(&input[..take]);
                    input.advance(take);

                    if buf.len() < header.payload_len as usize {
                        return Ok(None);
                    }

                    let frame = Frame {
                        header: header.clone(),
                        payload: std::mem::take(buf).freeze(),
                    };
                    self.state = ParseState::Header {
                        buf: [0u8; HEADER_LEN],
                        filled: 0,
                    };
                    return Ok(Some(frame));
                }
            }
        }
    }
}

impl Default for FrameParser {
    fn default() -> Self {
        Self::new()
    }
}

fn parse_header(buf: &[u8; HEADER_LEN]) -> Result<FrameHeader, ParseError> {
    let magic = u16::from_le_bytes([buf[0], buf[1]]);
    if magic != MAGIC {
        return Err(ParseError::InvalidMagic {
            expected: MAGIC,
            got: magic,
        });
    }

    let version = buf[2];
    if !matches!(version, VERSION_V1 | VERSION_V2) {
        return Err(ParseError::InvalidVersion { got: version });
    }

    let op_byte = buf[3];
    let op = Op::from_u8(op_byte).ok_or(ParseError::UnknownOp { got: op_byte })?;

    let flags_raw = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
    let flags = Flags::from_bits(flags_raw).ok_or(ParseError::InvalidFlags { bits: flags_raw })?;

    let req_id = u64::from_le_bytes([
        buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15],
    ]);

    let payload_len = u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]);
    if payload_len > MAX_FRAME_SIZE {
        return Err(ParseError::FrameTooLarge {
            len: payload_len,
            max: MAX_FRAME_SIZE,
        });
    }

    // buf[20..24] = reserved, ignored on read

    Ok(FrameHeader {
        version,
        op,
        flags,
        req_id,
        payload_len,
    })
}

/// Encode a frame into a `Bytes` buffer (zero-copy after construction).
///
/// `payload.len()` must fit in `u32` (≤ 4 GiB); in practice callers must
/// ensure it is ≤ `MAX_FRAME_SIZE` (16 MiB).
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn encode_frame(op: Op, flags: Flags, req_id: u64, payload: &[u8]) -> Bytes {
    encode_frame_versioned(VERSION_V1, op, flags, req_id, payload)
}

/// Encode a frame using a supported native protocol version.
///
/// # Panics
///
/// Panics in debug builds if `version` is not supported. Callers choosing a
/// version dynamically must validate it before encoding.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn encode_frame_versioned(
    version: u8,
    op: Op,
    flags: Flags,
    req_id: u64,
    payload: &[u8],
) -> Bytes {
    debug_assert!(matches!(version, VERSION_V1 | VERSION_V2));
    let mut buf = BytesMut::with_capacity(HEADER_LEN + payload.len());
    buf.put_u16_le(MAGIC);
    buf.put_u8(version);
    buf.put_u8(op as u8);
    buf.put_u32_le(flags.bits());
    buf.put_u64_le(req_id);
    buf.put_u32_le(payload.len() as u32);
    buf.put_u32_le(0u32); // reserved
    buf.put_slice(payload);
    buf.freeze()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::encode_get;

    fn make_raw_header(magic: u16, version: u8, op: u8, payload_len: u32) -> [u8; HEADER_LEN] {
        let mut buf = [0u8; HEADER_LEN];
        buf[0..2].copy_from_slice(&magic.to_le_bytes());
        buf[2] = version;
        buf[3] = op;
        // flags = 0, req_id = 0
        buf[16..20].copy_from_slice(&payload_len.to_le_bytes());
        buf
    }

    #[test]
    fn encode_decode_empty_payload() {
        let encoded = encode_frame(Op::Ping, Flags::empty(), 99, &[]);
        let mut buf = BytesMut::from(encoded.as_ref());
        let frame = FrameParser::new().feed(&mut buf).unwrap().unwrap();
        assert_eq!(frame.header.op, Op::Ping);
        assert_eq!(frame.header.req_id, 99);
        assert_eq!(frame.header.flags, Flags::empty());
        assert!(frame.payload.is_empty());
        assert!(buf.is_empty()); // all bytes consumed
    }

    #[test]
    fn v2_frame_roundtrips_and_preserves_its_version() {
        let encoded = encode_frame_versioned(VERSION_V2, Op::Ping, Flags::empty(), 9, &[]);
        let mut buf = BytesMut::from(encoded.as_ref());
        let frame = FrameParser::new().feed(&mut buf).unwrap().unwrap();
        assert_eq!(frame.header.version, VERSION_V2);
    }

    #[test]
    fn encode_decode_with_payload() {
        let payload = b"hello world";
        let encoded = encode_frame(Op::Ok, Flags::NO_REPLY, 7, payload);
        let mut buf = BytesMut::from(encoded.as_ref());
        let frame = FrameParser::new().feed(&mut buf).unwrap().unwrap();
        assert_eq!(frame.header.op, Op::Ok);
        assert_eq!(frame.header.req_id, 7);
        assert!(frame.header.flags.contains(Flags::NO_REPLY));
        assert_eq!(frame.payload.as_ref(), payload);
    }

    #[test]
    fn parse_partial_header_returns_none() {
        let encoded = encode_frame(Op::Ping, Flags::empty(), 1, &[]);
        let mut buf = BytesMut::from(&encoded[..10]); // partial header
        let mut parser = FrameParser::new();
        assert!(parser.feed(&mut buf).unwrap().is_none());

        let mut rest = BytesMut::from(&encoded[10..]);
        let frame = parser.feed(&mut rest).unwrap().unwrap();
        assert_eq!(frame.header.op, Op::Ping);
    }

    #[test]
    fn parse_partial_payload_returns_none() {
        let payload = b"abcdefghij";
        let encoded = encode_frame(Op::Ok, Flags::empty(), 1, payload);
        let split = HEADER_LEN + 5; // header + half payload
        let mut buf = BytesMut::from(&encoded[..split]);
        let mut parser = FrameParser::new();
        assert!(parser.feed(&mut buf).unwrap().is_none());

        let mut rest = BytesMut::from(&encoded[split..]);
        let frame = parser.feed(&mut rest).unwrap().unwrap();
        assert_eq!(frame.payload.as_ref(), payload);
    }

    #[test]
    fn parse_two_frames_concatenated() {
        let f1 = encode_frame(Op::Ping, Flags::empty(), 1, &[]);
        let f2 = encode_frame(Op::Ping, Flags::empty(), 2, &[]);
        let mut buf = BytesMut::new();
        buf.extend_from_slice(&f1);
        buf.extend_from_slice(&f2);
        let mut parser = FrameParser::new();
        let frame1 = parser.feed(&mut buf).unwrap().unwrap();
        let frame2 = parser.feed(&mut buf).unwrap().unwrap();
        assert_eq!(frame1.header.req_id, 1);
        assert_eq!(frame2.header.req_id, 2);
        assert!(buf.is_empty());
    }

    #[test]
    fn parse_invalid_magic_returns_error() {
        let raw = make_raw_header(0x0000, VERSION, Op::Ping as u8, 0);
        let mut buf = BytesMut::from(&raw[..]);
        let err = FrameParser::new().feed(&mut buf).unwrap_err();
        assert!(matches!(err, ParseError::InvalidMagic { got: 0, .. }));
    }

    #[test]
    fn parse_invalid_version_returns_error() {
        let raw = make_raw_header(MAGIC, 0x99, Op::Ping as u8, 0);
        let mut buf = BytesMut::from(&raw[..]);
        let err = FrameParser::new().feed(&mut buf).unwrap_err();
        assert!(matches!(err, ParseError::InvalidVersion { got: 0x99, .. }));
    }

    #[test]
    fn parse_unknown_op_returns_error() {
        let raw = make_raw_header(MAGIC, VERSION, 0xFF, 0);
        let mut buf = BytesMut::from(&raw[..]);
        let err = FrameParser::new().feed(&mut buf).unwrap_err();
        assert!(matches!(err, ParseError::UnknownOp { got: 0xFF }));
    }

    #[test]
    fn parse_frame_too_large_returns_error() {
        let raw = make_raw_header(MAGIC, VERSION, Op::Ping as u8, MAX_FRAME_SIZE + 1);
        let mut buf = BytesMut::from(&raw[..]);
        let err = FrameParser::new().feed(&mut buf).unwrap_err();
        assert!(matches!(err, ParseError::FrameTooLarge { .. }));
    }

    #[test]
    fn parse_invalid_flags_returns_error() {
        let mut raw = make_raw_header(MAGIC, VERSION, Op::Ping as u8, 0);
        // Set bits 28-31 which are not defined flags
        raw[4..8].copy_from_slice(&(0xF000_0000u32).to_le_bytes());
        let mut buf = BytesMut::from(&raw[..]);
        let err = FrameParser::new().feed(&mut buf).unwrap_err();
        assert!(matches!(err, ParseError::InvalidFlags { .. }));
    }

    #[test]
    fn req_id_max_value_roundtrip() {
        let encoded = encode_frame(Op::Ping, Flags::empty(), u64::MAX, &[]);
        let mut buf = BytesMut::from(encoded.as_ref());
        let frame = FrameParser::new().feed(&mut buf).unwrap().unwrap();
        assert_eq!(frame.header.req_id, u64::MAX);
    }

    /// The hazard of every length-prefixed protocol: 24 bytes of header
    /// declare 16 MiB, and the parser believes them before a single payload
    /// byte has arrived. Sixteen connections doing that pin 256 MiB from 384
    /// bytes of input.
    ///
    /// The parser must allocate for what ARRIVED, and grow as more does.
    #[test]
    #[ignore = "opens in commit 2 (proto: allocate what arrived)"]
    fn a_declared_native_payload_length_allocates_nothing_before_bytes_arrive() {
        let declared = MAX_FRAME_SIZE; // 16 MiB, the largest a frame may claim
        let raw = make_raw_header(MAGIC, VERSION, Op::Ok as u8, declared);
        let mut parser = FrameParser::new();
        let mut buf = BytesMut::from(&raw[..]);
        assert!(parser.feed(&mut buf).unwrap().is_none(), "header only");
        assert_eq!(
            parser.buffered_capacity(),
            0,
            "the declared length was believed before any payload arrived"
        );

        // A kilobyte arrives. The parser may hold that, and not a megabyte.
        let mut some = BytesMut::from(&vec![7u8; 1024][..]);
        assert!(parser.feed(&mut some).unwrap().is_none(), "still partial");
        let held = parser.buffered_capacity();
        assert!(held >= 1024, "the bytes that arrived must be kept: {held}");
        assert!(
            held < declared as usize / 16,
            "still allocating for what was declared, not for what arrived: {held}"
        );
    }

    /// A connection given a smaller allowance than the protocol ceiling
    /// refuses on the HEADER, which is the only place the refusal is free.
    #[test]
    #[ignore = "opens in commit 2 (proto: allocate what arrived)"]
    fn a_payload_length_over_the_parser_limit_is_refused_at_the_header() {
        let raw = make_raw_header(MAGIC, VERSION, Op::Ok as u8, 5000);
        let mut buf = BytesMut::from(&raw[..]);
        let err = FrameParser::with_limit(4096).feed(&mut buf).unwrap_err();
        assert!(
            matches!(
                err,
                ParseError::FrameTooLarge {
                    len: 5000,
                    max: 4096
                }
            ),
            "wrong refusal: {err:?}"
        );
        // And the protocol ceiling is still a ceiling: a limit above it does
        // not raise it.
        let raw = make_raw_header(MAGIC, VERSION, Op::Ok as u8, MAX_FRAME_SIZE + 1);
        let mut buf = BytesMut::from(&raw[..]);
        let err = FrameParser::with_limit(u32::MAX)
            .feed(&mut buf)
            .unwrap_err();
        assert!(matches!(err, ParseError::FrameTooLarge { .. }), "{err:?}");
    }

    #[test]
    fn leftover_bytes_not_consumed() {
        let encoded = encode_get(1, b"key");
        let extra = b"leftover";
        let mut buf = BytesMut::new();
        buf.extend_from_slice(&encoded);
        buf.extend_from_slice(extra);
        let mut parser = FrameParser::new();
        parser.feed(&mut buf).unwrap().unwrap();
        assert_eq!(buf.as_ref(), extra);
    }
}
