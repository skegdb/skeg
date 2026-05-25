//! Per-connection state and the HELLO handler.
//!
//! A `ConnectionState` is owned by the connection task (one per client
//! socket). It tracks the negotiated `ProtoVersion`, a server-assigned id,
//! and an optional client-set name. `apply_hello` is the one mutation site:
//! it folds a parsed `HelloArgs` into the state and returns the response
//! frame the server should send back.
//!
//! Auth credentials in `HELLO ... AUTH user pass` are parsed but ignored in
//! the pivot - design §5.7. When the v0.1 auth model lands (§7.2) it will
//! be wired through here without changing the parsing layer.

use crate::command::HelloArgs;
use crate::frame::Frame;
use crate::version::ProtoVersion;

#[derive(Debug, Clone)]
pub struct ConnectionState {
    /// Server-assigned connection id, surfaced in `HELLO` and `CLIENT ID`.
    pub id: i64,
    /// Negotiated protocol version. New connections default to `Resp2`.
    pub version: ProtoVersion,
    /// Client name from `HELLO ... SETNAME` (or future `CLIENT SETNAME`).
    pub name: Option<String>,
}

impl ConnectionState {
    #[must_use]
    pub fn new(id: i64) -> Self {
        Self {
            id,
            version: ProtoVersion::default(),
            name: None,
        }
    }

    /// Fold a parsed `HELLO` into the state and produce the response frame.
    ///
    /// Order matters: `protover` is applied before we read `self.version`
    /// for the response, so `HELLO 3` already returns the RESP3-encoded
    /// shape (the encoder dispatches on `self.version` at write time).
    pub fn apply_hello(&mut self, args: &HelloArgs, server_version: &str) -> Frame {
        if let Some(v) = args.protover {
            self.version = if v == 3 {
                ProtoVersion::Resp3
            } else {
                ProtoVersion::Resp2
            };
        }
        if let Some(name) = &args.setname {
            self.name = Some(name.clone());
        }
        // Auth: parsed for parity with the spec, ignored here. Recording
        // it now would lock in a schema before §7.2 picks an auth model.
        let _ = args.auth.as_ref();

        let proto = if self.version.is_resp3() { 3 } else { 2 };
        // Keys + string values as Bulk (REDIS_REPLY_STRING in hiredis), not
        // Simple (REDIS_REPLY_STATUS). redis-cli's cliSwitchProto asserts
        // REDIS_REPLY_STRING on every HELLO-response key when negotiating
        // RESP3 - real Redis sends bulk strings for this reason.
        // Modules field added for parity with real Redis HELLO response.
        Frame::Map(vec![
            (
                Frame::Bulk(bytes::Bytes::from_static(b"server")),
                Frame::Bulk(bytes::Bytes::from_static(b"skeg")),
            ),
            (
                Frame::Bulk(bytes::Bytes::from_static(b"version")),
                Frame::Bulk(bytes::Bytes::copy_from_slice(server_version.as_bytes())),
            ),
            (
                Frame::Bulk(bytes::Bytes::from_static(b"proto")),
                Frame::Integer(proto),
            ),
            (
                Frame::Bulk(bytes::Bytes::from_static(b"id")),
                Frame::Integer(self.id),
            ),
            (
                Frame::Bulk(bytes::Bytes::from_static(b"mode")),
                Frame::Bulk(bytes::Bytes::from_static(b"standalone")),
            ),
            (
                Frame::Bulk(bytes::Bytes::from_static(b"role")),
                Frame::Bulk(bytes::Bytes::from_static(b"master")),
            ),
            (
                Frame::Bulk(bytes::Bytes::from_static(b"modules")),
                Frame::Array(vec![]),
            ),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::encode_frame;
    use bytes::BytesMut;

    fn hello(protover: Option<u8>) -> HelloArgs {
        HelloArgs {
            protover,
            auth: None,
            setname: None,
        }
    }

    #[test]
    fn new_defaults_to_resp2_no_name() {
        let s = ConnectionState::new(7);
        assert_eq!(s.id, 7);
        assert_eq!(s.version, ProtoVersion::Resp2);
        assert!(s.name.is_none());
    }

    #[test]
    fn hello_3_upgrades_to_resp3() {
        let mut s = ConnectionState::new(7);
        let frame = s.apply_hello(&hello(Some(3)), "0.1.0-dev");
        assert_eq!(s.version, ProtoVersion::Resp3);
        // Response is a Map; check the proto field reflects the new state.
        let Frame::Map(pairs) = frame else {
            panic!("expected Map");
        };
        let proto = pairs
            .iter()
            .find_map(|(k, v)| {
                if let (Frame::Bulk(name), Frame::Integer(n)) = (k, v)
                    && name.as_ref() == b"proto"
                {
                    Some(*n)
                } else {
                    None
                }
            })
            .expect("proto field present");
        assert_eq!(proto, 3);
    }

    #[test]
    fn hello_2_downgrades_from_resp3() {
        let mut s = ConnectionState::new(1);
        let _ = s.apply_hello(&hello(Some(3)), "v");
        assert_eq!(s.version, ProtoVersion::Resp3);
        let _ = s.apply_hello(&hello(Some(2)), "v");
        assert_eq!(s.version, ProtoVersion::Resp2);
    }

    #[test]
    fn hello_no_protover_preserves_version() {
        let mut s = ConnectionState::new(1);
        let _ = s.apply_hello(&hello(Some(3)), "v");
        let _ = s.apply_hello(&hello(None), "v");
        assert_eq!(s.version, ProtoVersion::Resp3);
    }

    #[test]
    fn hello_setname_recorded() {
        let mut s = ConnectionState::new(1);
        let _ = s.apply_hello(
            &HelloArgs {
                protover: None,
                auth: None,
                setname: Some("worker-42".into()),
            },
            "v",
        );
        assert_eq!(s.name.as_deref(), Some("worker-42"));
    }

    #[test]
    fn hello_auth_ignored_for_now() {
        // Parser accepts AUTH; state must NOT record credentials in pivot.
        // NOTE: this locks in the §5.7 placeholder behaviour.
        // If this starts failing, AUTH model wiring likely landed and the
        // design docs must be updated in the same change.
        let mut s = ConnectionState::new(1);
        let _ = s.apply_hello(
            &HelloArgs {
                protover: Some(3),
                auth: Some(("alice".into(), "hunter2".into())),
                setname: None,
            },
            "v",
        );
        // No public field for auth; assert behaviour by checking nothing
        // else changed that shouldn't have.
        assert_eq!(s.version, ProtoVersion::Resp3);
        assert!(s.name.is_none());
    }

    #[test]
    fn hello_response_has_required_fields() {
        let mut s = ConnectionState::new(42);
        let frame = s.apply_hello(&hello(Some(3)), "0.1.0-dev");
        let Frame::Map(pairs) = frame else {
            panic!("expected Map");
        };
        let names: Vec<_> = pairs
            .iter()
            .map(|(k, _)| match k {
                Frame::Bulk(b) => String::from_utf8_lossy(b.as_ref()).into_owned(),
                _ => unreachable!(),
            })
            .collect();
        for required in [
            "server", "version", "proto", "id", "mode", "role", "modules",
        ] {
            assert!(names.iter().any(|n| n == required), "missing {required}");
        }
    }

    #[test]
    fn hello_response_encodes_as_resp3_map() {
        // Apply HELLO 3, then encode with the post-apply version. The wire
        // bytes must start with `%` (RESP3 map marker), not `*` (array).
        let mut s = ConnectionState::new(7);
        let frame = s.apply_hello(&hello(Some(3)), "0.1.0-dev");
        let mut out = BytesMut::new();
        encode_frame(&frame, s.version, &mut out);
        assert_eq!(out[0], b'%');
    }

    #[test]
    fn hello_response_encodes_as_resp2_flat_array_when_pre_negotiated() {
        // Client sends HELLO 2 (or no protover, default Resp2). The encoder
        // downgrades the Map to a flat 2N-element Array on the wire.
        let mut s = ConnectionState::new(7);
        let frame = s.apply_hello(&hello(Some(2)), "0.1.0-dev");
        let mut out = BytesMut::new();
        encode_frame(&frame, s.version, &mut out);
        assert_eq!(out[0], b'*');
        // 7 fields = 14 elements when flattened (server/version/proto/id/
        // mode/role/modules, last with empty array as value).
        assert!(out.starts_with(b"*14\r\n"));
    }
}
