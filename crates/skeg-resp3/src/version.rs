//! Protocol version negotiated via HELLO.
//!
//! New connections default to RESP2.
//! A client opts in to RESP3 by sending `HELLO 3`; the server then encodes
//! responses with RESP3-only types (map, double, boolean, push, etc).

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProtoVersion {
    #[default]
    Resp2,
    Resp3,
}

impl ProtoVersion {
    #[must_use]
    pub fn is_resp3(self) -> bool {
        matches!(self, Self::Resp3)
    }
}
