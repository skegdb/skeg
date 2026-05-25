//! Stateful frame decoder over a streamed byte source.
//!
//! `parse_frame` is per-shot: it sees a `&[u8]` and reports complete /
//! incomplete / malformed. A real connection delivers bytes a chunk at a time
//! and may interleave multiple frames in one read. `FrameDecoder` wraps an
//! internal `BytesMut`: callers append received bytes via [`feed`] (or write
//! directly through [`buf_mut`] for `AsyncReadExt::read_buf`), then call
//! [`decode`] in a loop until it returns `Ok(None)`.
//!
//! [`feed`]: FrameDecoder::feed
//! [`buf_mut`]: FrameDecoder::buf_mut
//! [`decode`]: FrameDecoder::decode

use bytes::BytesMut;

use crate::frame::Frame;
use crate::parser::{ParseError, parse_frame};

/// Default initial capacity. One TCP read fits comfortably and a typical
/// short command fits without ever growing.
const DEFAULT_CAPACITY: usize = 4096;

#[derive(Debug)]
pub struct FrameDecoder {
    buf: BytesMut,
}

impl FrameDecoder {
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            buf: BytesMut::with_capacity(cap),
        }
    }

    /// Append `bytes` to the decoder's internal buffer.
    pub fn feed(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    /// Mutable handle on the internal buffer. Lets async I/O write into the
    /// decoder without an intermediate copy: e.g.
    /// `socket.read_buf(decoder.buf_mut()).await`.
    pub fn buf_mut(&mut self) -> &mut BytesMut {
        &mut self.buf
    }

    /// Bytes currently buffered (debug/metrics).
    #[must_use]
    pub fn buffered(&self) -> usize {
        self.buf.len()
    }

    /// Try to decode one frame.
    ///
    /// # Returns
    /// - `Ok(Some(frame))`: a complete frame was decoded; its bytes have been
    ///   removed from the buffer.
    /// - `Ok(None)`: the buffer holds an incomplete frame; feed more bytes.
    /// - `Err(...)`: the buffered bytes do not form a valid RESP frame.
    ///
    /// # Errors
    ///
    /// Surfaces any `ParseError` raised by [`crate::parse_frame`]. The bad
    /// bytes remain in the buffer: callers that want to recover should
    /// discard the decoder and close the connection (RESP has no resync
    /// marker, so a malformed prefix is generally fatal).
    pub fn decode(&mut self) -> Result<Option<Frame>, ParseError> {
        match parse_frame(&self.buf)? {
            None => Ok(None),
            Some((frame, n)) => {
                let _ = self.buf.split_to(n);
                Ok(Some(frame))
            }
        }
    }
}

impl Default for FrameDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::encode_frame;
    use crate::version::ProtoVersion;
    use bytes::Bytes;

    #[test]
    fn empty_decoder_returns_none() {
        let mut d = FrameDecoder::new();
        assert!(d.decode().unwrap().is_none());
        assert_eq!(d.buffered(), 0);
    }

    #[test]
    fn one_complete_frame_in_one_feed() {
        let mut d = FrameDecoder::new();
        d.feed(b"+OK\r\n");
        assert_eq!(d.decode().unwrap(), Some(Frame::Simple("OK".into())));
        assert_eq!(d.buffered(), 0);
        assert!(d.decode().unwrap().is_none());
    }

    #[test]
    fn partial_then_complete() {
        let mut d = FrameDecoder::new();
        d.feed(b"+O");
        assert!(d.decode().unwrap().is_none());
        d.feed(b"K\r\n");
        assert_eq!(d.decode().unwrap(), Some(Frame::Simple("OK".into())));
    }

    #[test]
    fn two_frames_in_one_feed_decode_in_order() {
        let mut d = FrameDecoder::new();
        d.feed(b"+OK\r\n:42\r\n");
        assert_eq!(d.decode().unwrap(), Some(Frame::Simple("OK".into())));
        assert_eq!(d.decode().unwrap(), Some(Frame::Integer(42)));
        assert!(d.decode().unwrap().is_none());
    }

    #[test]
    fn second_frame_partially_buffered() {
        let mut d = FrameDecoder::new();
        d.feed(b"+OK\r\n:4");
        assert_eq!(d.decode().unwrap(), Some(Frame::Simple("OK".into())));
        // Second frame is incomplete.
        assert!(d.decode().unwrap().is_none());
        d.feed(b"2\r\n");
        assert_eq!(d.decode().unwrap(), Some(Frame::Integer(42)));
    }

    #[test]
    fn malformed_marker_propagates_error() {
        let mut d = FrameDecoder::new();
        d.feed(b"X\r\n");
        let err = d.decode().unwrap_err();
        assert!(matches!(err, ParseError::UnknownTypeMarker(b'X')));
    }

    #[test]
    fn buf_mut_extends_internal_buffer() {
        // Simulates the AsyncReadExt::read_buf integration: the I/O layer
        // writes into our buffer; we then decode.
        let mut d = FrameDecoder::new();
        d.buf_mut().extend_from_slice(b"+ack\r\n");
        assert_eq!(d.decode().unwrap(), Some(Frame::Simple("ack".into())));
    }

    /// Feed `bytes` to the decoder one byte at a time. Each step calls
    /// `decode` and asserts the frame appears at the exact step where the
    /// final byte lands. This is the boundary-fuzz that proves the state is
    /// preserved across arbitrarily small chunks.
    fn one_byte_at_a_time(bytes: &[u8], expected: &Frame) {
        let mut d = FrameDecoder::new();
        let last = bytes.len() - 1;
        for (i, &b) in bytes.iter().enumerate() {
            d.feed(&[b]);
            let got = d.decode().unwrap();
            if i == last {
                assert_eq!(
                    got.as_ref(),
                    Some(expected),
                    "frame should emerge on the final byte; bytes={bytes:?}"
                );
            } else {
                assert!(
                    got.is_none(),
                    "frame appeared too early at byte {i} of {}",
                    bytes.len()
                );
            }
        }
    }

    #[test]
    fn boundary_fuzz_simple_string() {
        one_byte_at_a_time(b"+OK\r\n", &Frame::Simple("OK".into()));
    }

    #[test]
    fn boundary_fuzz_bulk() {
        one_byte_at_a_time(
            b"$5\r\nhello\r\n",
            &Frame::Bulk(Bytes::from_static(b"hello")),
        );
    }

    #[test]
    fn boundary_fuzz_array_of_bulks() {
        one_byte_at_a_time(
            b"*2\r\n$3\r\nGET\r\n$3\r\nfoo\r\n",
            &Frame::Array(vec![
                Frame::Bulk(Bytes::from_static(b"GET")),
                Frame::Bulk(Bytes::from_static(b"foo")),
            ]),
        );
    }

    #[test]
    fn boundary_fuzz_resp3_map_with_double() {
        // Simulates a VSEARCH-style response: map { score => Double }.
        one_byte_at_a_time(
            b"%1\r\n+score\r\n,0.95\r\n",
            &Frame::Map(vec![(Frame::Simple("score".into()), Frame::Double(0.95))]),
        );
    }

    /// Round-trip: encode each Frame, feed the encoded bytes split at every
    /// possible position, decode, assert equality. Exhaustive boundary check
    /// across the full surface of RESP3 types.
    #[test]
    fn encode_then_decode_at_every_split() {
        let cases = vec![
            Frame::Simple("OK".into()),
            Frame::Integer(0),
            Frame::Integer(-9999),
            Frame::Bulk(Bytes::from_static(b"hello world")),
            Frame::Null,
            Frame::Array(vec![Frame::Integer(1), Frame::Simple("ok".into())]),
            Frame::Boolean(true),
            Frame::Double(2.5),
            Frame::Map(vec![(Frame::Simple("k".into()), Frame::Integer(1))]),
            Frame::Set(vec![Frame::Simple("a".into())]),
        ];
        for frame in cases {
            let mut encoded = BytesMut::new();
            encode_frame(&frame, ProtoVersion::Resp3, &mut encoded);
            let total = encoded.len();
            for split in 0..=total {
                let mut d = FrameDecoder::new();
                d.feed(&encoded[..split]);
                let first = d.decode().unwrap();
                if split < total {
                    assert!(
                        first.is_none(),
                        "frame emerged early at split {split}/{total} for {frame:?}"
                    );
                }
                d.feed(&encoded[split..]);
                let second = d.decode().unwrap();
                let decoded = first.or(second);
                assert_eq!(
                    decoded.as_ref(),
                    Some(&frame),
                    "split {split}/{total} failed for {frame:?}"
                );
            }
        }
    }

    #[test]
    fn back_to_back_frames_via_byte_stream() {
        let mut a = BytesMut::new();
        encode_frame(&Frame::Simple("first".into()), ProtoVersion::Resp3, &mut a);
        encode_frame(&Frame::Integer(7), ProtoVersion::Resp3, &mut a);
        encode_frame(
            &Frame::Bulk(Bytes::from_static(b"third")),
            ProtoVersion::Resp3,
            &mut a,
        );

        let mut d = FrameDecoder::new();
        // Feed in arbitrary 3-byte chunks, draining decode after each feed.
        let mut decoded: Vec<Frame> = Vec::new();
        for chunk in a.chunks(3) {
            d.feed(chunk);
            while let Some(f) = d.decode().unwrap() {
                decoded.push(f);
            }
        }
        assert_eq!(
            decoded,
            vec![
                Frame::Simple("first".into()),
                Frame::Integer(7),
                Frame::Bulk(Bytes::from_static(b"third")),
            ]
        );
        assert_eq!(d.buffered(), 0);
    }
}
