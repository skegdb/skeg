//! Streaming-friendly RESP parser.
//!
//! `parse_frame` returns `Ok(Some((frame, n)))` when a full frame has been
//! decoded from the front of `input` (n is the number of bytes consumed),
//! `Ok(None)` if more bytes are needed, and `Err(...)` on malformed input.
//!
//! Covers RESP2 (`+`, `-`, `:`, `$`, `*`) and RESP3 (`_`, `#`, `,`, `%`, `~`,
//! `>`, `!`, `=`, `(`). Streaming aggregates (`*?`, `%?`) are not implemented:
//! skeg does not produce them and the spec allows servers to ignore them.

use bytes::Bytes;

use crate::frame::Frame;

/// Max child count for an aggregate (array/map/set/push). Caps the worst-case
/// pre-allocation a malicious client can force. Redis uses similar limits.
pub const MAX_AGGREGATE_LEN: usize = 1_048_576;
/// Max bulk/verbatim payload size. Matches Redis `proto-max-bulk-len` default.
pub const MAX_BULK_LEN: usize = 512 * 1024 * 1024;
/// Max aggregate nesting depth. Bounds the recursive descent so a stream of
/// nested aggregate headers (e.g. `*1\r\n` repeated) cannot overflow the stack
/// and abort the process. Redis uses 128.
pub const MAX_NESTING_DEPTH: usize = 128;

pub type ParseResult = Result<Option<(Frame, usize)>, ParseError>;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ParseError {
    #[error("unknown type marker: 0x{0:02X}")]
    UnknownTypeMarker(u8),
    #[error("invalid utf-8 in line")]
    InvalidUtf8,
    #[error("invalid integer: {0}")]
    InvalidInteger(String),
    #[error("invalid length: {0}")]
    InvalidLength(i64),
    #[error("aggregate too large: {0}")]
    AggregateTooLarge(usize),
    #[error("aggregate nesting too deep: {0}")]
    NestingTooDeep(usize),
    #[error("bulk too large: {0}")]
    BulkTooLarge(usize),
    #[error("invalid boolean payload: {0:?}")]
    InvalidBoolean(Vec<u8>),
    #[error("invalid double: {0}")]
    InvalidDouble(String),
    #[error("invalid verbatim string: missing 3-char fmt + ':' prefix")]
    InvalidVerbatim,
    #[error("unexpected payload between type marker and CRLF")]
    UnexpectedPayload,
}

/// Parse one frame from the front of `input`.
///
/// # Errors
///
/// Returns `ParseError::UnknownTypeMarker` if the first byte is not a
/// recognised RESP marker, and the various specific variants when a length
/// prefix, integer, or aggregate exceeds limits or fails to parse.
pub fn parse_frame(input: &[u8]) -> ParseResult {
    parse_frame_depth(input, 0)
}

/// Inner parser that tracks aggregate nesting `depth`. Each level of array/map/
/// set/push recurses with `depth + 1`; exceeding [`MAX_NESTING_DEPTH`] is a
/// `NestingTooDeep` error rather than unbounded recursion (stack-overflow DoS).
fn parse_frame_depth(input: &[u8], depth: usize) -> ParseResult {
    if input.is_empty() {
        return Ok(None);
    }
    if depth > MAX_NESTING_DEPTH {
        return Err(ParseError::NestingTooDeep(depth));
    }
    let body = &input[1..];
    match input[0] {
        // RESP2
        b'+' => Ok(parse_line(body)?.map(|(s, n)| (Frame::Simple(s), n + 1))),
        b'-' => Ok(parse_line(body)?.map(|(s, n)| (Frame::Error(s), n + 1))),
        b':' => Ok(parse_integer_line(body)?.map(|(i, n)| (Frame::Integer(i), n + 1))),
        b'$' => Ok(shift(parse_bulk(body)?)),
        b'*' => Ok(shift(parse_aggregate(body, AggKind::Array, depth)?)),
        // RESP3
        b'_' => Ok(shift(parse_null_resp3(body)?)),
        b'#' => Ok(shift(parse_boolean(body)?)),
        b',' => Ok(shift(parse_double(body)?)),
        b'%' => Ok(shift(parse_aggregate(body, AggKind::Map, depth)?)),
        b'~' => Ok(shift(parse_aggregate(body, AggKind::Set, depth)?)),
        b'>' => Ok(shift(parse_aggregate(body, AggKind::Push, depth)?)),
        b'!' => Ok(shift(parse_blob_error(body)?)),
        b'=' => Ok(shift(parse_verbatim(body)?)),
        b'(' => Ok(shift(parse_big_number(body)?)),
        c => Err(ParseError::UnknownTypeMarker(c)),
    }
}

/// Advance the consumed-bytes counter by 1 to account for the type marker
/// stripped before dispatch.
fn shift(opt: Option<(Frame, usize)>) -> Option<(Frame, usize)> {
    opt.map(|(f, n)| (f, n + 1))
}

fn find_crlf(input: &[u8]) -> Option<usize> {
    input.windows(2).position(|w| w == b"\r\n")
}

/// Parse `<line>\r\n`. Returns the line text (without CRLF) and the total
/// number of bytes consumed (line length + 2).
fn parse_line(body: &[u8]) -> Result<Option<(String, usize)>, ParseError> {
    let Some(end) = find_crlf(body) else {
        return Ok(None);
    };
    let line = std::str::from_utf8(&body[..end]).map_err(|_| ParseError::InvalidUtf8)?;
    Ok(Some((line.to_string(), end + 2)))
}

fn parse_integer_line(body: &[u8]) -> Result<Option<(i64, usize)>, ParseError> {
    let Some(end) = find_crlf(body) else {
        return Ok(None);
    };
    let s = std::str::from_utf8(&body[..end]).map_err(|_| ParseError::InvalidUtf8)?;
    let n: i64 = s
        .parse()
        .map_err(|_| ParseError::InvalidInteger(s.to_string()))?;
    Ok(Some((n, end + 2)))
}

fn parse_bulk(body: &[u8]) -> ParseResult {
    let Some(len_end) = find_crlf(body) else {
        return Ok(None);
    };
    let len_str = std::str::from_utf8(&body[..len_end]).map_err(|_| ParseError::InvalidUtf8)?;
    let len: i64 = len_str
        .parse()
        .map_err(|_| ParseError::InvalidInteger(len_str.to_string()))?;
    if len == -1 {
        // RESP2 null bulk.
        return Ok(Some((Frame::Null, len_end + 2)));
    }
    let Ok(len) = usize::try_from(len) else {
        return Err(ParseError::InvalidLength(len));
    };
    if len > MAX_BULK_LEN {
        return Err(ParseError::BulkTooLarge(len));
    }
    let start = len_end + 2;
    let end = start + len;
    if body.len() < end + 2 {
        return Ok(None);
    }
    // Trailing CRLF is mandatory per the spec. Treat the absence as
    // "incomplete" rather than malformed; the parser does not currently
    // distinguish a truncated stream from a corrupt one for the trailing
    // bytes, and treating both as incomplete is safe.
    if &body[end..end + 2] != b"\r\n" {
        return Ok(None);
    }
    Ok(Some((
        Frame::Bulk(Bytes::copy_from_slice(&body[start..end])),
        end + 2,
    )))
}

#[derive(Clone, Copy)]
enum AggKind {
    Array,
    Set,
    Push,
    Map,
}

fn parse_aggregate(body: &[u8], kind: AggKind, depth: usize) -> ParseResult {
    let Some(len_end) = find_crlf(body) else {
        return Ok(None);
    };
    let len_str = std::str::from_utf8(&body[..len_end]).map_err(|_| ParseError::InvalidUtf8)?;
    let count: i64 = len_str
        .parse()
        .map_err(|_| ParseError::InvalidInteger(len_str.to_string()))?;
    if count == -1 {
        // Only Array has a RESP2 null form (*-1). Map/Set/Push don't.
        if matches!(kind, AggKind::Array) {
            return Ok(Some((Frame::Null, len_end + 2)));
        }
        return Err(ParseError::InvalidLength(-1));
    }
    let Ok(count) = usize::try_from(count) else {
        return Err(ParseError::InvalidLength(count));
    };
    if count > MAX_AGGREGATE_LEN {
        return Err(ParseError::AggregateTooLarge(count));
    }
    // Map encodes N key-value pairs as 2N frames on the wire.
    let frame_count = if matches!(kind, AggKind::Map) {
        count * 2
    } else {
        count
    };
    let mut items = Vec::with_capacity(frame_count);
    let mut consumed = len_end + 2;
    for _ in 0..frame_count {
        match parse_frame_depth(&body[consumed..], depth + 1)? {
            None => return Ok(None),
            Some((frame, n)) => {
                items.push(frame);
                consumed += n;
            }
        }
    }
    let frame = match kind {
        AggKind::Array => Frame::Array(items),
        AggKind::Set => Frame::Set(items),
        AggKind::Push => Frame::Push(items),
        AggKind::Map => {
            let mut pairs = Vec::with_capacity(count);
            let mut iter = items.into_iter();
            while let Some(k) = iter.next() {
                // frame_count = 2 * count, so the value is always present.
                let v = iter.next().expect("map produces 2N frames by construction");
                pairs.push((k, v));
            }
            Frame::Map(pairs)
        }
    };
    Ok(Some((frame, consumed)))
}

fn parse_null_resp3(body: &[u8]) -> ParseResult {
    let Some(end) = find_crlf(body) else {
        return Ok(None);
    };
    if end != 0 {
        return Err(ParseError::UnexpectedPayload);
    }
    Ok(Some((Frame::Null, 2)))
}

fn parse_boolean(body: &[u8]) -> ParseResult {
    let Some(end) = find_crlf(body) else {
        return Ok(None);
    };
    let value = match &body[..end] {
        b"t" => true,
        b"f" => false,
        other => return Err(ParseError::InvalidBoolean(other.to_vec())),
    };
    Ok(Some((Frame::Boolean(value), end + 2)))
}

fn parse_double(body: &[u8]) -> ParseResult {
    let Some(end) = find_crlf(body) else {
        return Ok(None);
    };
    let s = std::str::from_utf8(&body[..end]).map_err(|_| ParseError::InvalidUtf8)?;
    // f64::from_str accepts decimal, scientific, "inf", "-inf", "nan" - all
    // valid per the RESP3 spec for the double type.
    let v: f64 = s
        .parse()
        .map_err(|_| ParseError::InvalidDouble(s.to_string()))?;
    Ok(Some((Frame::Double(v), end + 2)))
}

fn parse_blob_error(body: &[u8]) -> ParseResult {
    let Some(len_end) = find_crlf(body) else {
        return Ok(None);
    };
    let len_str = std::str::from_utf8(&body[..len_end]).map_err(|_| ParseError::InvalidUtf8)?;
    let len: i64 = len_str
        .parse()
        .map_err(|_| ParseError::InvalidInteger(len_str.to_string()))?;
    let Ok(len) = usize::try_from(len) else {
        return Err(ParseError::InvalidLength(len));
    };
    if len > MAX_BULK_LEN {
        return Err(ParseError::BulkTooLarge(len));
    }
    let start = len_end + 2;
    let end = start + len;
    if body.len() < end + 2 {
        return Ok(None);
    }
    if &body[end..end + 2] != b"\r\n" {
        return Ok(None);
    }
    let payload = std::str::from_utf8(&body[start..end]).map_err(|_| ParseError::InvalidUtf8)?;
    // Convention: "<CODE> <message>". Empty message if no space.
    let (code, message) = payload.find(' ').map_or_else(
        || (payload.to_string(), String::new()),
        |idx| (payload[..idx].to_string(), payload[idx + 1..].to_string()),
    );
    Ok(Some((Frame::BlobError { code, message }, end + 2)))
}

fn parse_verbatim(body: &[u8]) -> ParseResult {
    let Some(len_end) = find_crlf(body) else {
        return Ok(None);
    };
    let len_str = std::str::from_utf8(&body[..len_end]).map_err(|_| ParseError::InvalidUtf8)?;
    let len: i64 = len_str
        .parse()
        .map_err(|_| ParseError::InvalidInteger(len_str.to_string()))?;
    let Ok(len) = usize::try_from(len) else {
        return Err(ParseError::InvalidLength(len));
    };
    if len > MAX_BULK_LEN {
        return Err(ParseError::BulkTooLarge(len));
    }
    if len < 4 {
        // Need at least 3-char format + ':' separator.
        return Err(ParseError::InvalidVerbatim);
    }
    let start = len_end + 2;
    let end = start + len;
    if body.len() < end + 2 {
        return Ok(None);
    }
    if &body[end..end + 2] != b"\r\n" {
        return Ok(None);
    }
    if body[start + 3] != b':' {
        return Err(ParseError::InvalidVerbatim);
    }
    let fmt = [body[start], body[start + 1], body[start + 2]];
    let data = Bytes::copy_from_slice(&body[start + 4..end]);
    Ok(Some((Frame::Verbatim { fmt, data }, end + 2)))
}

fn parse_big_number(body: &[u8]) -> ParseResult {
    let Some(end) = find_crlf(body) else {
        return Ok(None);
    };
    let s = std::str::from_utf8(&body[..end]).map_err(|_| ParseError::InvalidUtf8)?;
    let bytes = s.as_bytes();
    let valid = match bytes.first() {
        Some(b'-' | b'+') => bytes.len() > 1 && bytes[1..].iter().all(u8::is_ascii_digit),
        Some(_) => bytes.iter().all(u8::is_ascii_digit),
        None => false,
    };
    if !valid {
        return Err(ParseError::InvalidInteger(s.to_string()));
    }
    Ok(Some((Frame::BigNumber(s.to_string()), end + 2)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn done(input: &[u8]) -> (Frame, usize) {
        parse_frame(input)
            .expect("parse must succeed")
            .expect("frame must be complete")
    }

    #[test]
    fn simple_string_ok() {
        let (f, n) = done(b"+OK\r\n");
        assert_eq!(f, Frame::Simple("OK".into()));
        assert_eq!(n, 5);
    }

    #[test]
    fn simple_string_with_payload() {
        let (f, _) = done(b"+PONG message body\r\n");
        assert_eq!(f, Frame::Simple("PONG message body".into()));
    }

    #[test]
    fn error_line() {
        let (f, _) = done(b"-ERR unknown command\r\n");
        assert_eq!(f, Frame::Error("ERR unknown command".into()));
    }

    #[test]
    fn integer_positive() {
        let (f, n) = done(b":1234\r\n");
        assert_eq!(f, Frame::Integer(1234));
        assert_eq!(n, 7);
    }

    #[test]
    fn integer_negative() {
        let (f, _) = done(b":-42\r\n");
        assert_eq!(f, Frame::Integer(-42));
    }

    #[test]
    fn integer_zero() {
        let (f, _) = done(b":0\r\n");
        assert_eq!(f, Frame::Integer(0));
    }

    #[test]
    fn bulk_string() {
        let (f, n) = done(b"$5\r\nhello\r\n");
        assert_eq!(f, Frame::Bulk(Bytes::from_static(b"hello")));
        assert_eq!(n, 11);
    }

    #[test]
    fn bulk_empty() {
        let (f, n) = done(b"$0\r\n\r\n");
        assert_eq!(f, Frame::Bulk(Bytes::new()));
        assert_eq!(n, 6);
    }

    #[test]
    fn bulk_with_binary_payload() {
        let raw: &[u8] = b"$5\r\n\x00\x01\x02\x03\x04\r\n";
        let (f, _) = done(raw);
        assert_eq!(f, Frame::Bulk(Bytes::from_static(&[0u8, 1, 2, 3, 4])));
    }

    #[test]
    fn null_bulk_resp2() {
        let (f, n) = done(b"$-1\r\n");
        assert_eq!(f, Frame::Null);
        assert_eq!(n, 5);
    }

    #[test]
    fn array_simple_two_items() {
        let (f, _) = done(b"*2\r\n+OK\r\n:7\r\n");
        assert_eq!(
            f,
            Frame::Array(vec![Frame::Simple("OK".into()), Frame::Integer(7)])
        );
    }

    #[test]
    fn array_empty() {
        let (f, _) = done(b"*0\r\n");
        assert_eq!(f, Frame::Array(vec![]));
    }

    #[test]
    fn array_null_resp2() {
        let (f, _) = done(b"*-1\r\n");
        assert_eq!(f, Frame::Null);
    }

    #[test]
    fn array_of_bulks() {
        let raw = b"*3\r\n$3\r\nGET\r\n$3\r\nfoo\r\n$0\r\n\r\n";
        let (f, _) = done(raw);
        let Frame::Array(items) = f else {
            panic!("expected Array");
        };
        assert_eq!(items.len(), 3);
        assert_eq!(items[0], Frame::Bulk(Bytes::from_static(b"GET")));
        assert_eq!(items[1], Frame::Bulk(Bytes::from_static(b"foo")));
        assert_eq!(items[2], Frame::Bulk(Bytes::new()));
    }

    #[test]
    fn array_nested() {
        let (f, _) = done(b"*2\r\n*1\r\n:1\r\n+OK\r\n");
        let Frame::Array(items) = f else {
            panic!("expected Array");
        };
        assert_eq!(items[0], Frame::Array(vec![Frame::Integer(1)]));
        assert_eq!(items[1], Frame::Simple("OK".into()));
    }

    #[test]
    fn empty_input_returns_none() {
        assert_eq!(parse_frame(b"").unwrap(), None);
    }

    #[test]
    fn partial_simple_returns_none() {
        assert_eq!(parse_frame(b"+OK\r").unwrap(), None);
    }

    #[test]
    fn partial_bulk_payload_returns_none() {
        assert_eq!(parse_frame(b"$5\r\nhel").unwrap(), None);
    }

    #[test]
    fn partial_bulk_missing_trailing_crlf_returns_none() {
        assert_eq!(parse_frame(b"$5\r\nhello").unwrap(), None);
    }

    #[test]
    fn partial_array_with_only_one_child_returns_none() {
        assert_eq!(parse_frame(b"*2\r\n+OK\r\n").unwrap(), None);
    }

    #[test]
    fn malformed_integer_errors() {
        assert!(matches!(
            parse_frame(b":notanumber\r\n").unwrap_err(),
            ParseError::InvalidInteger(_)
        ));
    }

    #[test]
    fn invalid_bulk_length_negative_errors() {
        assert!(matches!(
            parse_frame(b"$-2\r\n").unwrap_err(),
            ParseError::InvalidLength(-2)
        ));
    }

    #[test]
    fn aggregate_too_large_errors() {
        let big = format!("*{}\r\n", MAX_AGGREGATE_LEN + 1);
        assert!(matches!(
            parse_frame(big.as_bytes()).unwrap_err(),
            ParseError::AggregateTooLarge(_)
        ));
    }

    #[test]
    fn consumed_bytes_track_correctly_across_frames() {
        // The caller uses the returned `n` to advance into the next frame.
        let buf: &[u8] = b"+OK\r\n:5\r\n";
        let (f1, n1) = done(buf);
        assert_eq!(f1, Frame::Simple("OK".into()));
        assert_eq!(n1, 5);
        let (f2, n2) = done(&buf[n1..]);
        assert_eq!(f2, Frame::Integer(5));
        assert_eq!(n2, 4);
        assert_eq!(n1 + n2, buf.len());
    }

    // ── RESP3 ──────────────────────────────────────────────────────────────

    #[test]
    fn null_resp3() {
        let (f, n) = done(b"_\r\n");
        assert_eq!(f, Frame::Null);
        assert_eq!(n, 3);
    }

    #[test]
    fn null_resp3_with_junk_errors() {
        assert_eq!(
            parse_frame(b"_x\r\n").unwrap_err(),
            ParseError::UnexpectedPayload
        );
    }

    #[test]
    fn boolean_true() {
        let (f, n) = done(b"#t\r\n");
        assert_eq!(f, Frame::Boolean(true));
        assert_eq!(n, 4);
    }

    #[test]
    fn boolean_false() {
        let (f, _) = done(b"#f\r\n");
        assert_eq!(f, Frame::Boolean(false));
    }

    #[test]
    fn boolean_invalid_payload_errors() {
        assert!(matches!(
            parse_frame(b"#x\r\n").unwrap_err(),
            ParseError::InvalidBoolean(_)
        ));
    }

    #[test]
    fn double_decimal() {
        let (f, _) = done(b",2.5\r\n");
        let Frame::Double(v) = f else {
            panic!("expected Double");
        };
        assert!((v - 2.5).abs() < 1e-9);
    }

    #[test]
    fn double_integer_like() {
        let (f, _) = done(b",42\r\n");
        assert_eq!(f, Frame::Double(42.0));
    }

    #[test]
    fn double_negative() {
        let (f, _) = done(b",-1.5\r\n");
        assert_eq!(f, Frame::Double(-1.5));
    }

    #[test]
    fn double_inf() {
        let (f, _) = done(b",inf\r\n");
        let Frame::Double(v) = f else {
            panic!("expected Double");
        };
        assert!(v.is_infinite() && v > 0.0);
    }

    #[test]
    fn double_neg_inf() {
        let (f, _) = done(b",-inf\r\n");
        let Frame::Double(v) = f else {
            panic!("expected Double");
        };
        assert!(v.is_infinite() && v < 0.0);
    }

    #[test]
    fn double_nan() {
        let (f, _) = done(b",nan\r\n");
        let Frame::Double(v) = f else {
            panic!("expected Double");
        };
        assert!(v.is_nan());
    }

    #[test]
    fn double_invalid_errors() {
        assert!(matches!(
            parse_frame(b",abc\r\n").unwrap_err(),
            ParseError::InvalidDouble(_)
        ));
    }

    #[test]
    fn map_two_entries() {
        let raw = b"%2\r\n+k1\r\n+v1\r\n+k2\r\n:42\r\n";
        let (f, _) = done(raw);
        let Frame::Map(pairs) = f else {
            panic!("expected Map");
        };
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].0, Frame::Simple("k1".into()));
        assert_eq!(pairs[0].1, Frame::Simple("v1".into()));
        assert_eq!(pairs[1].0, Frame::Simple("k2".into()));
        assert_eq!(pairs[1].1, Frame::Integer(42));
    }

    #[test]
    fn map_empty() {
        let (f, _) = done(b"%0\r\n");
        assert_eq!(f, Frame::Map(vec![]));
    }

    #[test]
    fn map_rejects_resp2_null() {
        // %-1 is not valid: Map has no null form in RESP3.
        assert_eq!(
            parse_frame(b"%-1\r\n").unwrap_err(),
            ParseError::InvalidLength(-1)
        );
    }

    #[test]
    fn partial_map_returns_none() {
        // 2 pairs declared (4 frames), only 3 fully present.
        assert_eq!(parse_frame(b"%2\r\n+k1\r\n+v1\r\n+k2\r\n").unwrap(), None);
    }

    #[test]
    fn set_three_items() {
        let raw = b"~3\r\n+a\r\n+b\r\n+c\r\n";
        let (f, _) = done(raw);
        assert_eq!(
            f,
            Frame::Set(vec![
                Frame::Simple("a".into()),
                Frame::Simple("b".into()),
                Frame::Simple("c".into()),
            ])
        );
    }

    #[test]
    fn set_rejects_resp2_null() {
        assert_eq!(
            parse_frame(b"~-1\r\n").unwrap_err(),
            ParseError::InvalidLength(-1)
        );
    }

    #[test]
    fn push_notification() {
        let raw = b">2\r\n+job\r\n+done\r\n";
        let (f, _) = done(raw);
        assert_eq!(
            f,
            Frame::Push(vec![
                Frame::Simple("job".into()),
                Frame::Simple("done".into()),
            ])
        );
    }

    #[test]
    fn blob_error_with_code_and_message() {
        let raw = b"!21\r\nSYNTAX invalid syntax\r\n";
        let (f, _) = done(raw);
        assert_eq!(
            f,
            Frame::BlobError {
                code: "SYNTAX".into(),
                message: "invalid syntax".into(),
            }
        );
    }

    #[test]
    fn blob_error_code_only() {
        let raw = b"!4\r\nFAIL\r\n";
        let (f, _) = done(raw);
        assert_eq!(
            f,
            Frame::BlobError {
                code: "FAIL".into(),
                message: String::new(),
            }
        );
    }

    #[test]
    fn verbatim_string() {
        let raw = b"=15\r\ntxt:Some string\r\n";
        let (f, _) = done(raw);
        assert_eq!(
            f,
            Frame::Verbatim {
                fmt: *b"txt",
                data: Bytes::from_static(b"Some string"),
            }
        );
    }

    #[test]
    fn verbatim_missing_colon_errors() {
        // "abc!body" has no ':' at position 3.
        let raw = b"=8\r\nabc!body\r\n";
        assert_eq!(parse_frame(raw).unwrap_err(), ParseError::InvalidVerbatim);
    }

    #[test]
    fn verbatim_too_short_errors() {
        // Needs at least 4 bytes (3-char fmt + ':').
        let raw = b"=2\r\nab\r\n";
        assert_eq!(parse_frame(raw).unwrap_err(), ParseError::InvalidVerbatim);
    }

    #[test]
    fn big_number_positive() {
        let (f, _) = done(b"(31337\r\n");
        assert_eq!(f, Frame::BigNumber("31337".into()));
    }

    #[test]
    fn big_number_negative() {
        let (f, _) = done(b"(-100\r\n");
        assert_eq!(f, Frame::BigNumber("-100".into()));
    }

    #[test]
    fn big_number_beyond_i64() {
        // Beyond i64::MAX: must still parse as a string.
        let (f, _) = done(b"(99999999999999999999999999999\r\n");
        assert_eq!(f, Frame::BigNumber("99999999999999999999999999999".into()));
    }

    #[test]
    fn big_number_invalid_errors() {
        assert!(matches!(
            parse_frame(b"(abc\r\n").unwrap_err(),
            ParseError::InvalidInteger(_)
        ));
    }

    #[test]
    fn nested_resp3_in_map() {
        // VSEARCH-style result: map with double scores and set of payloads.
        let raw = b"%2\r\n+score\r\n,0.95\r\n+ok\r\n#t\r\n";
        let (f, _) = done(raw);
        let Frame::Map(pairs) = f else {
            panic!("expected Map");
        };
        assert_eq!(pairs[0].0, Frame::Simple("score".into()));
        let Frame::Double(v) = pairs[0].1 else {
            panic!("expected Double");
        };
        assert!((v - 0.95).abs() < 1e-9);
        assert_eq!(pairs[1].0, Frame::Simple("ok".into()));
        assert_eq!(pairs[1].1, Frame::Boolean(true));
    }

    #[test]
    fn deep_nesting_errors_instead_of_overflowing() {
        // A stream of nested array headers (`*1` = "array of one element", which
        // is itself an array...) would recurse without bound. The depth limit
        // must turn this into an error, not a stack overflow / process abort.
        let mut raw = Vec::new();
        for _ in 0..(MAX_NESTING_DEPTH + 16) {
            raw.extend_from_slice(b"*1\r\n");
        }
        assert!(matches!(
            parse_frame(&raw).unwrap_err(),
            ParseError::NestingTooDeep(_)
        ));
        // A legitimately deep-but-bounded frame still parses.
        let mut ok = Vec::new();
        for _ in 0..8 {
            ok.extend_from_slice(b"*1\r\n");
        }
        ok.extend_from_slice(b":7\r\n");
        assert!(parse_frame(&ok).unwrap().is_some());
    }
}
