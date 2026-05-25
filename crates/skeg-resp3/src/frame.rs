//! `Frame` is the in-memory representation of one RESP message.
//!
//! Variants cover both RESP2 and RESP3. The parser/encoder use `ProtoVersion`
//! to decide how to encode `Null` (RESP2 uses `$-1` or `*-1`, RESP3 uses `_`)
//! and which extra variants to emit (Map/Set/Double/Boolean/Push are RESP3
//! only on the wire, but always valid in the Frame type so server code can
//! construct them uniformly).

use bytes::Bytes;

#[derive(Debug, Clone, PartialEq)]
pub enum Frame {
    /// `+<line>\r\n`. Short status string (no CR/LF in payload).
    Simple(String),
    /// `-<line>\r\n`. RESP2 error.
    Error(String),
    /// `:<int>\r\n`.
    Integer(i64),
    /// `$<len>\r\n<bytes>\r\n`. Binary-safe.
    Bulk(Bytes),
    /// `$-1\r\n` / `*-1\r\n` (RESP2) or `_\r\n` (RESP3).
    Null,
    /// `*<n>\r\n<n frames>`.
    Array(Vec<Frame>),

    // ── RESP3 only on the wire ───────────────────────────────────────────
    /// `!<len>\r\n<code> <message>\r\n`. Structured error.
    BlobError { code: String, message: String },
    /// `#t\r\n` or `#f\r\n`.
    Boolean(bool),
    /// `,<f64>\r\n`. Also accepts `,inf`, `,-inf`, `,nan`.
    Double(f64),
    /// `%<n>\r\n<2n frames>`. Order-preserving for serialization.
    Map(Vec<(Frame, Frame)>),
    /// `~<n>\r\n<n frames>`. Unordered set semantics.
    Set(Vec<Frame>),
    /// `><n>\r\n<n frames>`. Server-initiated push (job completion, pub/sub).
    Push(Vec<Frame>),
    /// `=<len>\r\n<3 chars>:<bytes>\r\n`. `fmt` is the 3-char format (e.g.
    /// `txt`, `mkd`); `data` is the payload without the colon.
    Verbatim { fmt: [u8; 3], data: Bytes },
    /// `(<digits>\r\n`. Arbitrary-precision integer (not produced by skeg).
    BigNumber(String),
}

impl Frame {
    #[must_use]
    pub fn ok() -> Self {
        Self::Simple("OK".into())
    }

    #[must_use]
    pub fn pong() -> Self {
        Self::Simple("PONG".into())
    }
}
