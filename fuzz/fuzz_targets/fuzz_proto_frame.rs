//! Fuzz the binary protocol frame parser.
//!
//! Entry point: `skeg_proto::FrameParser::feed`. The parser is a state
//! machine over a `BytesMut` byte stream and is exercised on every TCP
//! connection. A malformed input must either return `Ok(None)` (need
//! more bytes), `Ok(Some(_))` (a parsed frame), or `Err(ParseError)` -
//! never panic or read past the buffer.
//!
//! Adversarial inputs to discover: malformed magic, oversized
//! payload_len, partial header fragments, payloads that span multiple
//! `feed` calls in unfortunate ways.

#![no_main]

use bytes::BytesMut;
use libfuzzer_sys::fuzz_target;
use skeg_proto::FrameParser;

fuzz_target!(|data: &[u8]| {
    let mut parser = FrameParser::new();
    let mut buf = BytesMut::from(data);
    // Drive the parser to completion on the input. The contract is
    // "never panic"; any Result is acceptable.
    let _ = parser.feed(&mut buf);
});
