//! RESP2 + RESP3 wire encoder.
//!
//! `encode_frame(frame, version, out)` appends the wire encoding of `frame`
//! to `out`. The `version` controls how RESP3-only types are downgraded for
//! RESP2 clients:
//!
//! | RESP3 type    | RESP2 downgrade                             |
//! |---------------|---------------------------------------------|
//! | `_\r\n` null  | `$-1\r\n`                                   |
//! | `#t / #f`     | `:1 / :0`                                   |
//! | `,<f64>`      | bulk string of formatted value              |
//! | `%<n>` map    | flat `*<2n>` array, alternating key/value   |
//! | `~<n>` set    | `*<n>` array                                |
//! | `><n>` push   | `*<n>` array (server pushes are RESP3-only) |
//! | `!<n>` blob   | `-<code> <message>\r\n` error line          |
//! | `=<n>` verbat | bulk string (3-char format tag dropped)     |
//! | `(<digits>`   | bulk string                                 |
//!
//! The type table and downgrade decisions follow the RESP3 protocol spec.

use bytes::{BufMut, BytesMut};

use crate::frame::Frame;
use crate::version::ProtoVersion;

/// Append the wire encoding of `frame` to `out`, using `version` to decide
/// how RESP3-only variants are downgraded for RESP2 clients.
pub fn encode_frame(frame: &Frame, version: ProtoVersion, out: &mut BytesMut) {
    match frame {
        Frame::Simple(s) => write_line(b'+', s.as_bytes(), out),
        Frame::Error(s) => write_line(b'-', s.as_bytes(), out),
        Frame::Integer(n) => {
            out.put_u8(b':');
            write_int(*n, out);
            out.put_slice(b"\r\n");
        }
        Frame::Bulk(b) => write_bulk(b.as_ref(), out),
        Frame::Null => {
            if version.is_resp3() {
                out.put_slice(b"_\r\n");
            } else {
                out.put_slice(b"$-1\r\n");
            }
        }
        Frame::Array(items) => write_seq(b'*', items, version, out),
        Frame::Boolean(v) => {
            if version.is_resp3() {
                out.put_slice(if *v { b"#t\r\n" } else { b"#f\r\n" });
            } else {
                out.put_slice(if *v { b":1\r\n" } else { b":0\r\n" });
            }
        }
        Frame::Double(d) => {
            let s = format_double(*d);
            if version.is_resp3() {
                out.put_u8(b',');
                out.put_slice(s.as_bytes());
                out.put_slice(b"\r\n");
            } else {
                write_bulk(s.as_bytes(), out);
            }
        }
        Frame::Map(pairs) => write_map(pairs, version, out),
        Frame::Set(items) => {
            let marker = if version.is_resp3() { b'~' } else { b'*' };
            write_seq(marker, items, version, out);
        }
        Frame::Push(items) => {
            // Push is server-initiated and only meaningful to RESP3 clients
            // that opted in. For RESP2 we downgrade to plain array - the
            // server-side dispatch is expected to gate push emission on
            // RESP3, so this branch is mostly defensive.
            let marker = if version.is_resp3() { b'>' } else { b'*' };
            write_seq(marker, items, version, out);
        }
        Frame::BlobError { code, message } => write_blob_error(code, message, version, out),
        Frame::Verbatim { fmt, data } => write_verbatim(*fmt, data.as_ref(), version, out),
        Frame::BigNumber(s) => {
            if version.is_resp3() {
                out.put_u8(b'(');
                out.put_slice(s.as_bytes());
                out.put_slice(b"\r\n");
            } else {
                write_bulk(s.as_bytes(), out);
            }
        }
    }
}

fn write_line(marker: u8, line: &[u8], out: &mut BytesMut) {
    out.put_u8(marker);
    out.put_slice(line);
    out.put_slice(b"\r\n");
}

fn write_int(n: i64, out: &mut BytesMut) {
    out.put_slice(n.to_string().as_bytes());
}

fn write_usize(n: usize, out: &mut BytesMut) {
    out.put_slice(n.to_string().as_bytes());
}

fn write_bulk(data: &[u8], out: &mut BytesMut) {
    out.put_u8(b'$');
    write_usize(data.len(), out);
    out.put_slice(b"\r\n");
    out.put_slice(data);
    out.put_slice(b"\r\n");
}

fn write_seq(marker: u8, items: &[Frame], version: ProtoVersion, out: &mut BytesMut) {
    out.put_u8(marker);
    write_usize(items.len(), out);
    out.put_slice(b"\r\n");
    for f in items {
        encode_frame(f, version, out);
    }
}

fn write_map(pairs: &[(Frame, Frame)], version: ProtoVersion, out: &mut BytesMut) {
    if version.is_resp3() {
        out.put_u8(b'%');
        write_usize(pairs.len(), out);
    } else {
        // Downgrade: flat array of 2N elements (key, value, key, value, ...).
        out.put_u8(b'*');
        write_usize(pairs.len() * 2, out);
    }
    out.put_slice(b"\r\n");
    for (k, v) in pairs {
        encode_frame(k, version, out);
        encode_frame(v, version, out);
    }
}

fn write_blob_error(code: &str, message: &str, version: ProtoVersion, out: &mut BytesMut) {
    let payload_len = if message.is_empty() {
        code.len()
    } else {
        code.len() + 1 + message.len()
    };
    if version.is_resp3() {
        out.put_u8(b'!');
        write_usize(payload_len, out);
        out.put_slice(b"\r\n");
    } else {
        // RESP2 has no blob error; collapse to a single error line.
        out.put_u8(b'-');
    }
    out.put_slice(code.as_bytes());
    if !message.is_empty() {
        out.put_u8(b' ');
        out.put_slice(message.as_bytes());
    }
    out.put_slice(b"\r\n");
}

fn write_verbatim(fmt: [u8; 3], data: &[u8], version: ProtoVersion, out: &mut BytesMut) {
    if version.is_resp3() {
        let payload_len = 3 + 1 + data.len(); // fmt + ':' + data
        out.put_u8(b'=');
        write_usize(payload_len, out);
        out.put_slice(b"\r\n");
        out.put_slice(&fmt);
        out.put_u8(b':');
        out.put_slice(data);
        out.put_slice(b"\r\n");
    } else {
        // RESP2 has no verbatim type; drop the format tag and emit a bulk.
        write_bulk(data, out);
    }
}

fn format_double(d: f64) -> String {
    if d.is_nan() {
        "nan".to_string()
    } else if d.is_infinite() {
        if d.is_sign_positive() {
            "inf".to_string()
        } else {
            "-inf".to_string()
        }
    } else {
        format!("{d}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_frame;
    use bytes::Bytes;

    fn resp2(f: &Frame) -> BytesMut {
        let mut out = BytesMut::new();
        encode_frame(f, ProtoVersion::Resp2, &mut out);
        out
    }

    fn resp3(f: &Frame) -> BytesMut {
        let mut out = BytesMut::new();
        encode_frame(f, ProtoVersion::Resp3, &mut out);
        out
    }

    fn roundtrip(f: &Frame, version: ProtoVersion) {
        let mut out = BytesMut::new();
        encode_frame(f, version, &mut out);
        let (parsed, n) = parse_frame(&out)
            .expect("encoded frame must parse")
            .expect("encoded frame must be complete");
        assert_eq!(&parsed, f, "roundtrip mismatch under {version:?}");
        assert_eq!(n, out.len(), "n must equal encoded length");
    }

    // ── Wire-level golden tests (RESP2) ────────────────────────────────────

    #[test]
    fn simple_string_wire() {
        assert_eq!(&*resp2(&Frame::Simple("OK".into())), b"+OK\r\n");
    }

    #[test]
    fn error_wire() {
        assert_eq!(&*resp2(&Frame::Error("ERR x".into())), b"-ERR x\r\n");
    }

    #[test]
    fn integer_wire() {
        assert_eq!(&*resp2(&Frame::Integer(-42)), b":-42\r\n");
    }

    #[test]
    fn bulk_wire() {
        assert_eq!(
            &*resp2(&Frame::Bulk(Bytes::from_static(b"hi"))),
            b"$2\r\nhi\r\n"
        );
    }

    #[test]
    fn bulk_empty_wire() {
        assert_eq!(&*resp2(&Frame::Bulk(Bytes::new())), b"$0\r\n\r\n");
    }

    #[test]
    fn null_resp2_wire() {
        assert_eq!(&*resp2(&Frame::Null), b"$-1\r\n");
    }

    #[test]
    fn array_wire() {
        let f = Frame::Array(vec![Frame::Integer(1), Frame::Simple("ok".into())]);
        assert_eq!(&*resp2(&f), b"*2\r\n:1\r\n+ok\r\n");
    }

    // ── Wire-level golden tests (RESP3) ────────────────────────────────────

    #[test]
    fn null_resp3_wire() {
        assert_eq!(&*resp3(&Frame::Null), b"_\r\n");
    }

    #[test]
    fn boolean_resp3_wire() {
        assert_eq!(&*resp3(&Frame::Boolean(true)), b"#t\r\n");
        assert_eq!(&*resp3(&Frame::Boolean(false)), b"#f\r\n");
    }

    #[test]
    fn double_resp3_wire() {
        assert_eq!(&*resp3(&Frame::Double(2.5)), b",2.5\r\n");
    }

    #[test]
    fn double_inf_resp3_wire() {
        assert_eq!(&*resp3(&Frame::Double(f64::INFINITY)), b",inf\r\n");
        assert_eq!(&*resp3(&Frame::Double(f64::NEG_INFINITY)), b",-inf\r\n");
    }

    #[test]
    fn double_nan_resp3_wire() {
        assert_eq!(&*resp3(&Frame::Double(f64::NAN)), b",nan\r\n");
    }

    #[test]
    fn map_resp3_wire() {
        let f = Frame::Map(vec![(Frame::Simple("k".into()), Frame::Integer(1))]);
        assert_eq!(&*resp3(&f), b"%1\r\n+k\r\n:1\r\n");
    }

    #[test]
    fn set_resp3_wire() {
        let f = Frame::Set(vec![Frame::Simple("a".into())]);
        assert_eq!(&*resp3(&f), b"~1\r\n+a\r\n");
    }

    #[test]
    fn push_resp3_wire() {
        let f = Frame::Push(vec![Frame::Simple("event".into())]);
        assert_eq!(&*resp3(&f), b">1\r\n+event\r\n");
    }

    #[test]
    fn blob_error_resp3_wire() {
        let f = Frame::BlobError {
            code: "ERR".into(),
            message: "bad".into(),
        };
        // 7 bytes: "ERR bad"
        assert_eq!(&*resp3(&f), b"!7\r\nERR bad\r\n");
    }

    #[test]
    fn blob_error_code_only_resp3_wire() {
        let f = Frame::BlobError {
            code: "FAIL".into(),
            message: String::new(),
        };
        assert_eq!(&*resp3(&f), b"!4\r\nFAIL\r\n");
    }

    #[test]
    fn verbatim_resp3_wire() {
        let f = Frame::Verbatim {
            fmt: *b"txt",
            data: Bytes::from_static(b"hello"),
        };
        // 9 bytes: "txt:hello"
        assert_eq!(&*resp3(&f), b"=9\r\ntxt:hello\r\n");
    }

    #[test]
    fn big_number_resp3_wire() {
        let f = Frame::BigNumber("99999999999999999999".into());
        assert_eq!(&*resp3(&f), b"(99999999999999999999\r\n");
    }

    // ── RESP3 -> RESP2 downgrades ──────────────────────────────────────────

    #[test]
    fn boolean_downgrades_to_integer() {
        assert_eq!(&*resp2(&Frame::Boolean(true)), b":1\r\n");
        assert_eq!(&*resp2(&Frame::Boolean(false)), b":0\r\n");
    }

    #[test]
    fn double_downgrades_to_bulk() {
        let out = resp2(&Frame::Double(2.5));
        assert_eq!(&*out, b"$3\r\n2.5\r\n");
    }

    #[test]
    fn map_downgrades_to_flat_array() {
        let f = Frame::Map(vec![(Frame::Simple("k".into()), Frame::Integer(1))]);
        // 1 pair becomes 2-element array.
        assert_eq!(&*resp2(&f), b"*2\r\n+k\r\n:1\r\n");
    }

    #[test]
    fn set_downgrades_to_array() {
        let f = Frame::Set(vec![Frame::Simple("a".into())]);
        assert_eq!(&*resp2(&f), b"*1\r\n+a\r\n");
    }

    #[test]
    fn push_downgrades_to_array() {
        let f = Frame::Push(vec![Frame::Simple("e".into())]);
        assert_eq!(&*resp2(&f), b"*1\r\n+e\r\n");
    }

    #[test]
    fn blob_error_downgrades_to_error_line() {
        let f = Frame::BlobError {
            code: "ERR".into(),
            message: "bad".into(),
        };
        assert_eq!(&*resp2(&f), b"-ERR bad\r\n");
    }

    #[test]
    fn verbatim_downgrades_to_bulk_dropping_fmt() {
        let f = Frame::Verbatim {
            fmt: *b"txt",
            data: Bytes::from_static(b"hello"),
        };
        // RESP2 has no verbatim, so the fmt tag is dropped on the wire.
        assert_eq!(&*resp2(&f), b"$5\r\nhello\r\n");
    }

    #[test]
    fn big_number_downgrades_to_bulk() {
        let f = Frame::BigNumber("999".into());
        assert_eq!(&*resp2(&f), b"$3\r\n999\r\n");
    }

    // ── parse(encode(x)) == x roundtrips ───────────────────────────────────

    #[test]
    fn roundtrip_resp2_basic_types() {
        let cases = [
            Frame::Simple("OK".into()),
            Frame::Error("ERR x".into()),
            Frame::Integer(0),
            Frame::Integer(-9999),
            Frame::Integer(i64::MAX),
            Frame::Bulk(Bytes::from_static(b"foo")),
            Frame::Bulk(Bytes::new()),
            Frame::Null,
            Frame::Array(vec![Frame::Integer(1), Frame::Simple("ok".into())]),
            Frame::Array(vec![]),
        ];
        for f in &cases {
            roundtrip(f, ProtoVersion::Resp2);
        }
    }

    #[test]
    fn roundtrip_resp3_full_surface() {
        let cases = vec![
            Frame::Simple("OK".into()),
            Frame::Error("E".into()),
            Frame::Integer(42),
            Frame::Bulk(Bytes::from_static(b"data")),
            Frame::Null,
            Frame::Array(vec![Frame::Integer(1), Frame::Boolean(true)]),
            Frame::Boolean(true),
            Frame::Boolean(false),
            Frame::Double(2.5),
            Frame::Double(-1.0),
            Frame::Double(0.0),
            Frame::Map(vec![(Frame::Simple("k".into()), Frame::Integer(1))]),
            Frame::Set(vec![Frame::Simple("a".into()), Frame::Simple("b".into())]),
            Frame::Push(vec![Frame::Simple("notif".into())]),
            Frame::BlobError {
                code: "ERR".into(),
                message: "bad".into(),
            },
            Frame::BlobError {
                code: "FAIL".into(),
                message: String::new(),
            },
            Frame::Verbatim {
                fmt: *b"txt",
                data: Bytes::from_static(b"hello world"),
            },
            Frame::BigNumber("31337".into()),
        ];
        for f in &cases {
            roundtrip(f, ProtoVersion::Resp3);
        }
    }

    #[test]
    fn roundtrip_double_specials_resp3() {
        // NaN cannot be compared with PartialEq, so we check its bit pattern.
        let mut out = BytesMut::new();
        encode_frame(&Frame::Double(f64::NAN), ProtoVersion::Resp3, &mut out);
        let (parsed, _) = parse_frame(&out).unwrap().unwrap();
        let Frame::Double(v) = parsed else {
            panic!("expected Double");
        };
        assert!(v.is_nan());

        roundtrip(&Frame::Double(f64::INFINITY), ProtoVersion::Resp3);
        roundtrip(&Frame::Double(f64::NEG_INFINITY), ProtoVersion::Resp3);
    }

    #[test]
    fn roundtrip_nested_aggregate() {
        // VSEARCH-style result: map with hits = array of (id, score) maps.
        let f = Frame::Map(vec![(
            Frame::Simple("hits".into()),
            Frame::Array(vec![Frame::Map(vec![
                (Frame::Simple("id".into()), Frame::Simple("doc-1".into())),
                (Frame::Simple("score".into()), Frame::Double(0.95)),
            ])]),
        )]);
        roundtrip(&f, ProtoVersion::Resp3);
    }
}
