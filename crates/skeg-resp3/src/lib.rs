#![deny(unsafe_code)]

//! `skeg-resp3` - RESP2 + RESP3 protocol parser/encoder for skeg.
//!
//! Implements the wire format described in `resp3/design-protocol.md`.
//!
//! Current scope (pivot, step 0-6):
//! - Frame enum covering RESP2 (5 types) + RESP3 (8 extra types + null).
//! - Parser: all 14 markers (`+`, `-`, `:`, `$`, `*`, `_`, `#`, `,`, `%`, `~`,
//!   `>`, `!`, `=`, `(`).
//! - Encoder: emits any `Frame` in either RESP2 or RESP3, downgrading
//!   RESP3-only types when targeting a RESP2 client.
//! - `FrameDecoder`: stateful wrapper over the parser that drives a
//!   streaming byte source (TCP socket, pipe, etc).
//! - `Command` parsing: HELLO, PING, ECHO typed; the rest fall through to
//!   Unknown.
//! - `ConnectionState` + `apply_hello`: per-connection version/name, and
//!   the HELLO handler that mutates state and produces the response frame.
//! - `handlers`: stateless handlers for PING and ECHO.
//!
//! New connections speak RESP2 until `HELLO 3` upgrades them.

pub mod command;
pub mod connection;
pub mod decoder;
pub mod encoder;
pub mod frame;
pub mod handlers;
pub mod parser;
pub mod version;

pub use command::{Command, CommandError, HelloArgs, parse_command};
pub use connection::ConnectionState;
pub use decoder::FrameDecoder;
pub use encoder::encode_frame;
pub use frame::Frame;
pub use handlers::{handle_echo, handle_ping};
pub use parser::{MAX_AGGREGATE_LEN, MAX_BULK_LEN, ParseError, ParseResult, parse_frame};
pub use version::ProtoVersion;
