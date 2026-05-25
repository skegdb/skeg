//! Stateless command handlers.
//!
//! Pure functions: `Command::Variant(args) -> Frame`. No state mutation, no
//! storage access. HELLO lives in `connection.rs` because it mutates the
//! per-connection state; the KV layer (GET/SET/DEL/EXISTS) will live in
//! `skeg-server` because it touches `ShardSet`. What stays here is the
//! protocol-only commands.

use bytes::Bytes;

use crate::frame::Frame;

/// `PING [msg]` -> `+PONG` when msg is absent, bulk-echo otherwise.
#[must_use]
pub fn handle_ping(msg: Option<Bytes>) -> Frame {
    match msg {
        None => Frame::pong(),
        Some(b) => Frame::Bulk(b),
    }
}

/// `ECHO msg` -> bulk-echo of msg.
#[must_use]
pub fn handle_echo(msg: Bytes) -> Frame {
    Frame::Bulk(msg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::{Command, parse_command};
    use crate::encoder::encode_frame;
    use crate::parser::parse_frame;
    use crate::version::ProtoVersion;
    use bytes::BytesMut;

    #[test]
    fn ping_no_msg_is_pong() {
        assert_eq!(handle_ping(None), Frame::Simple("PONG".into()));
    }

    #[test]
    fn ping_with_msg_echoes_as_bulk() {
        let m = Bytes::from_static(b"hello");
        assert_eq!(handle_ping(Some(m.clone())), Frame::Bulk(m));
    }

    #[test]
    fn echo_returns_bulk() {
        let m = Bytes::from_static(b"world");
        assert_eq!(handle_echo(m.clone()), Frame::Bulk(m));
    }

    /// End-to-end smoke: wire bytes -> parsed frame -> typed command ->
    /// handler -> response frame -> encoded wire bytes. Proves the modules
    /// compose without needing a TCP server (that wiring lands in step 8).
    #[test]
    fn ping_full_pipeline_resp2() {
        let wire = b"*1\r\n$4\r\nPING\r\n";
        let (frame, n) = parse_frame(wire).unwrap().unwrap();
        assert_eq!(n, wire.len());
        let resp = match parse_command(frame).unwrap() {
            Command::Ping(msg) => handle_ping(msg),
            _ => panic!("expected Ping"),
        };
        let mut out = BytesMut::new();
        encode_frame(&resp, ProtoVersion::Resp2, &mut out);
        assert_eq!(&*out, b"+PONG\r\n");
    }

    #[test]
    fn ping_with_msg_full_pipeline() {
        let wire = b"*2\r\n$4\r\nPING\r\n$2\r\nhi\r\n";
        let (frame, _) = parse_frame(wire).unwrap().unwrap();
        let resp = match parse_command(frame).unwrap() {
            Command::Ping(msg) => handle_ping(msg),
            _ => panic!("expected Ping"),
        };
        let mut out = BytesMut::new();
        encode_frame(&resp, ProtoVersion::Resp2, &mut out);
        assert_eq!(&*out, b"$2\r\nhi\r\n");
    }

    #[test]
    fn echo_full_pipeline() {
        let wire = b"*2\r\n$4\r\nECHO\r\n$3\r\nfoo\r\n";
        let (frame, _) = parse_frame(wire).unwrap().unwrap();
        let resp = match parse_command(frame).unwrap() {
            Command::Echo(msg) => handle_echo(msg),
            _ => panic!("expected Echo"),
        };
        let mut out = BytesMut::new();
        encode_frame(&resp, ProtoVersion::Resp2, &mut out);
        assert_eq!(&*out, b"$3\r\nfoo\r\n");
    }
}
