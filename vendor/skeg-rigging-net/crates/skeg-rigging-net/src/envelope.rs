//! Record envelope for transports that don't model `shareable` / tags.
//!
//! skeg-server's RESP3 surface stores `(vector_id, f32_bytes)` for
//! vectors and `(key, value_bytes)` for KV records. There is no slot
//! for `shareable` or `tags`. To preserve hansa's filter semantics
//! across the network, the hansa-side writer wraps the record's
//! payload in an envelope and stores it under
//! `hansa:rec:<vector_id>`. The reader fetches the envelope, decodes
//! it, applies the filter client-side, and returns the inner payload
//! up the `skeg-rigging` stack.
//!
//! ## Wire format
//!
//! Two encodings, discriminated by the first byte:
//!
//! - **JSON (`{`, `0x7B`)** — back-compat. Original v0.1 encoding,
//!   readable with `cat`.
//! - **Binary (`0xB0`, "B" magic)** — F.55. Section-based layout,
//!   5–15× smaller than JSON for text payloads.
//!
//! [`RecordEnvelope::decode`] auto-detects the form. [`encode`] keeps
//! JSON as the default to stay byte-compatible with anything writing
//! envelopes before F.55; [`encode_binary`] is opt-in for new
//! producers that want to save bandwidth.
//!
//! ### Binary layout
//!
//! ```text
//! ┌──────────────────────────────────────┐
//! │ 1B  magic = 0xB0                     │
//! │ 1B  flags (bit 0: shareable,         │
//! │           bit 1: zstd-payload,       │
//! │           bits 2-7: reserved=0)      │
//! │ 2B  tag count, little-endian u16     │
//! │ for each tag:                        │
//! │   2B  length, LE u16                 │
//! │   N   UTF-8 bytes                    │
//! │ 4B  payload length, LE u32           │
//! │ N   payload bytes                    │
//! │ 4B  CRC32C, LE u32                   │
//! └──────────────────────────────────────┘
//! ```
//!
//! The CRC32C covers every byte from `magic` through the end of the
//! payload (i.e. everything before the CRC field itself). zstd
//! compression is reserved for a follow-up (F.20); v0.1 always
//! writes the bit as 0 and rejects 1 on read.
//!
//! This convention lives **outside** skeg so the engine stays
//! engine-neutral.

use serde::{Deserialize, Serialize};

/// Magic byte identifying the binary v1 envelope encoding. Chosen so
/// that it cannot collide with a JSON envelope's first byte (`{` = 0x7B).
pub const BINARY_MAGIC: u8 = 0xB0;

/// First byte of a JSON envelope: a `{` (a serde-json encoding always
/// starts with the object opening brace).
pub const JSON_MAGIC: u8 = b'{';

/// Bit positions inside the binary envelope's flags byte.
const FLAG_SHAREABLE: u8 = 0b0000_0001;
const FLAG_ZSTD_PAYLOAD: u8 = 0b0000_0010;
const FLAG_RESERVED_MASK: u8 = !(FLAG_SHAREABLE | FLAG_ZSTD_PAYLOAD);

/// Default zstd compression level for [`RecordEnvelope::encode_binary_zstd`].
/// Level 3 is zstd's own default; trades roughly 2x decode speed for
/// 5-10% better ratio compared to level 1. Good for one-off envelope
/// writes where decode is the hot path.
pub const DEFAULT_ZSTD_LEVEL: i32 = 3;

/// JSON / binary envelope wrapping a record's payload, shareable
/// flag, and tags.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RecordEnvelope {
    /// Whether peers in a hansa may see this record.
    pub shareable: bool,
    /// Tag strings attached to the record.
    pub tags: Vec<String>,
    /// Raw payload bytes.
    pub payload: Vec<u8>,
}

/// Errors from envelope decoding.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum EnvelopeError {
    /// JSON encoding was malformed.
    #[error("invalid JSON envelope: {0}")]
    Json(#[from] serde_json::Error),

    /// Buffer was too short to contain a valid envelope.
    #[error("truncated envelope: expected {expected} bytes, got {got}")]
    Truncated {
        /// Bytes required to advance to the next field.
        expected: usize,
        /// Bytes available in the buffer.
        got: usize,
    },

    /// Magic byte didn't match either JSON `{` or binary `0xB0`.
    #[error("invalid envelope magic: 0x{0:02x}")]
    InvalidMagic(u8),

    /// CRC32C check failed.
    #[error("binary envelope CRC mismatch: header {expected:08x}, computed {got:08x}")]
    CrcMismatch {
        /// CRC stored in the envelope trailer.
        expected: u32,
        /// CRC the decoder computed over the read bytes.
        got: u32,
    },

    /// Reserved flag bits were set; refuse rather than risk
    /// misinterpreting a future format.
    #[error("reserved binary envelope flags set: 0x{0:02x}")]
    ReservedFlagsSet(u8),

    /// zstd decompression of a flagged payload failed (frame
    /// truncated, corrupt, or malformed).
    #[error("zstd payload decompression failed: {0}")]
    ZstdDecompress(String),

    /// A tag string was not valid UTF-8.
    #[error("binary envelope: tag {index} is not UTF-8: {source}")]
    TagNotUtf8 {
        /// Index of the offending tag in the tag array.
        index: usize,
        /// Underlying UTF-8 error.
        source: std::str::Utf8Error,
    },
}

impl RecordEnvelope {
    /// Convenience constructor.
    pub fn new(shareable: bool, tags: Vec<String>, payload: Vec<u8>) -> Self {
        Self {
            shareable,
            tags,
            payload,
        }
    }

    /// Encode as JSON bytes. The historical default; preserved for
    /// back-compat with envelopes written before F.55.
    pub fn encode(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("RecordEnvelope is always serialisable")
    }

    /// Encode as the F.55 binary envelope with the payload stored
    /// uncompressed. ~3-4x smaller than JSON for text payloads.
    pub fn encode_binary(&self) -> Vec<u8> {
        self.encode_binary_inner(None)
    }

    /// Encode as the F.55 binary envelope with the payload zstd-
    /// compressed (F.20). Sets the `FLAG_ZSTD_PAYLOAD` bit; readers
    /// transparently decompress on [`Self::decode`] /
    /// [`Self::decode_binary`].
    ///
    /// `level` follows zstd conventions (1 = fast, 22 = max). Use
    /// [`DEFAULT_ZSTD_LEVEL`] (= 3) unless you have a specific reason.
    ///
    /// Worth using when payloads exceed a few hundred bytes of
    /// compressible text (markdown, code, prose). For small or
    /// already-compressed payloads (images, encrypted blobs) the
    /// zstd frame overhead can make the output bigger than the
    /// uncompressed form; callers that don't know their corpus should
    /// compare both and keep the smaller. [`Self::encode_binary_smallest`]
    /// does exactly that.
    pub fn encode_binary_zstd(&self, level: i32) -> Vec<u8> {
        self.encode_binary_inner(Some(level))
    }

    /// Encode and return whichever of [`Self::encode_binary`] and
    /// [`Self::encode_binary_zstd`] (at [`DEFAULT_ZSTD_LEVEL`])
    /// produces a shorter output. Safe default for mixed payloads.
    pub fn encode_binary_smallest(&self) -> Vec<u8> {
        let plain = self.encode_binary();
        let compressed = self.encode_binary_zstd(DEFAULT_ZSTD_LEVEL);
        if compressed.len() < plain.len() {
            compressed
        } else {
            plain
        }
    }

    fn encode_binary_inner(&self, zstd_level: Option<i32>) -> Vec<u8> {
        let (payload_bytes, zstd_flag) = match zstd_level {
            Some(level) => match zstd::bulk::compress(&self.payload, level) {
                Ok(compressed) => (std::borrow::Cow::Owned(compressed), FLAG_ZSTD_PAYLOAD),
                // zstd never fails for reasonable inputs; degrade to
                // uncompressed rather than panic if it ever does.
                Err(_) => (std::borrow::Cow::Borrowed(self.payload.as_slice()), 0),
            },
            None => (std::borrow::Cow::Borrowed(self.payload.as_slice()), 0),
        };

        let mut tag_bytes_total = 0usize;
        for t in &self.tags {
            tag_bytes_total += 2 + t.len();
        }
        let total = 1 + 1 + 2 + tag_bytes_total + 4 + payload_bytes.len() + 4;
        let mut buf = Vec::with_capacity(total);

        buf.push(BINARY_MAGIC);
        let flags = if self.shareable { FLAG_SHAREABLE } else { 0 } | zstd_flag;
        buf.push(flags);

        let tag_count = self.tags.len() as u16;
        buf.extend_from_slice(&tag_count.to_le_bytes());
        for t in &self.tags {
            let tag_bytes = t.as_bytes();
            let tag_len = tag_bytes.len() as u16;
            buf.extend_from_slice(&tag_len.to_le_bytes());
            buf.extend_from_slice(tag_bytes);
        }

        let payload_len = payload_bytes.len() as u32;
        buf.extend_from_slice(&payload_len.to_le_bytes());
        buf.extend_from_slice(&payload_bytes);

        let crc = crc32c::crc32c(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());
        buf
    }

    /// Decode a record envelope, auto-detecting JSON vs binary by
    /// looking at the first byte. Returns
    /// [`EnvelopeError::InvalidMagic`] for buffers that don't start
    /// with `{` or `0xB0`.
    pub fn decode(buf: &[u8]) -> Result<Self, EnvelopeError> {
        match buf.first().copied() {
            None => Err(EnvelopeError::Truncated {
                expected: 1,
                got: 0,
            }),
            Some(JSON_MAGIC) => Ok(serde_json::from_slice(buf)?),
            Some(BINARY_MAGIC) => Self::decode_binary(buf),
            Some(other) => Err(EnvelopeError::InvalidMagic(other)),
        }
    }

    /// Decode strictly as the binary form. Useful for callers that
    /// already negotiated the encoding and want a precise error type
    /// instead of "could be JSON or binary".
    pub fn decode_binary(buf: &[u8]) -> Result<Self, EnvelopeError> {
        let mut cur = Cursor::new(buf);
        let magic = cur.read_u8()?;
        if magic != BINARY_MAGIC {
            return Err(EnvelopeError::InvalidMagic(magic));
        }
        let flags = cur.read_u8()?;
        if flags & FLAG_RESERVED_MASK != 0 {
            return Err(EnvelopeError::ReservedFlagsSet(flags));
        }
        let shareable = flags & FLAG_SHAREABLE != 0;
        let zstd_payload = flags & FLAG_ZSTD_PAYLOAD != 0;

        let tag_count = cur.read_u16_le()? as usize;
        let mut tags = Vec::with_capacity(tag_count);
        for i in 0..tag_count {
            let tag_len = cur.read_u16_le()? as usize;
            let bytes = cur.read_bytes(tag_len)?;
            let s = std::str::from_utf8(bytes).map_err(|e| EnvelopeError::TagNotUtf8 {
                index: i,
                source: e,
            })?;
            tags.push(s.to_string());
        }

        let payload_len = cur.read_u32_le()? as usize;
        let payload_bytes = cur.read_bytes(payload_len)?.to_vec();

        // CRC covers everything from magic through the end of the
        // payload bytes on the wire (the on-wire form is the
        // compressed bytes when the zstd flag is set).
        let body_end = cur.pos;
        let stored_crc = cur.read_u32_le()?;
        let computed_crc = crc32c::crc32c(&buf[..body_end]);
        if stored_crc != computed_crc {
            return Err(EnvelopeError::CrcMismatch {
                expected: stored_crc,
                got: computed_crc,
            });
        }

        let payload = if zstd_payload {
            zstd::bulk::decompress(&payload_bytes, ZSTD_MAX_DECOMPRESSED_SIZE)
                .map_err(|e| EnvelopeError::ZstdDecompress(e.to_string()))?
        } else {
            payload_bytes
        };

        Ok(Self {
            shareable,
            tags,
            payload,
        })
    }
}

/// Upper bound for zstd-decompressed payload size: 64 MiB. Caps the
/// damage from a maliciously crafted envelope that advertises a
/// massive decompressed size. Adjust if your application legitimately
/// stores larger payloads per record (rare in hansa's
/// embedding-centric model).
const ZSTD_MAX_DECOMPRESSED_SIZE: usize = 64 * 1024 * 1024;

/// Tiny forward-only cursor over a byte slice. Internal; could
/// switch to `bytes::Buf` later but the current API is shorter and
/// the binary envelope is short enough that the difference doesn't
/// matter.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn ensure(&self, n: usize) -> Result<(), EnvelopeError> {
        if self.pos + n > self.buf.len() {
            return Err(EnvelopeError::Truncated {
                expected: self.pos + n,
                got: self.buf.len(),
            });
        }
        Ok(())
    }

    fn read_u8(&mut self) -> Result<u8, EnvelopeError> {
        self.ensure(1)?;
        let v = self.buf[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn read_u16_le(&mut self) -> Result<u16, EnvelopeError> {
        self.ensure(2)?;
        let v = u16::from_le_bytes([self.buf[self.pos], self.buf[self.pos + 1]]);
        self.pos += 2;
        Ok(v)
    }

    fn read_u32_le(&mut self) -> Result<u32, EnvelopeError> {
        self.ensure(4)?;
        let v = u32::from_le_bytes([
            self.buf[self.pos],
            self.buf[self.pos + 1],
            self.buf[self.pos + 2],
            self.buf[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(v)
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], EnvelopeError> {
        self.ensure(n)?;
        let slice = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }
}

/// KV key prefix the bridge uses to store envelopes. Composed with a
/// record's vector id: `hansa:rec:<id>`.
pub const ENVELOPE_KEY_PREFIX: &str = "hansa:rec:";

/// Compute the KV key for a given record id under this convention.
pub fn envelope_key_for(record_id: u64) -> String {
    format!("{ENVELOPE_KEY_PREFIX}{record_id}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(payload: &str) -> RecordEnvelope {
        RecordEnvelope {
            shareable: true,
            tags: vec!["topic".into(), "crypto".into()],
            payload: payload.as_bytes().to_vec(),
        }
    }

    #[test]
    fn json_round_trip_back_compat() {
        let env = sample("hello world");
        let bytes = env.encode();
        assert_eq!(bytes[0], JSON_MAGIC);
        let back = RecordEnvelope::decode(&bytes).unwrap();
        assert_eq!(env, back);
    }

    #[test]
    fn binary_round_trip() {
        let env = sample("hello world");
        let bytes = env.encode_binary();
        assert_eq!(bytes[0], BINARY_MAGIC);
        let back = RecordEnvelope::decode(&bytes).unwrap();
        assert_eq!(env, back);
    }

    #[test]
    fn binary_round_trip_empty_tags_empty_payload() {
        let env = RecordEnvelope {
            shareable: false,
            tags: vec![],
            payload: vec![],
        };
        let back = RecordEnvelope::decode(&env.encode_binary()).unwrap();
        assert_eq!(env, back);
    }

    #[test]
    fn binary_round_trip_unicode_tags() {
        let env = RecordEnvelope {
            shareable: true,
            tags: vec!["café".into(), "日本語".into(), "🦀".into()],
            payload: b"unicode payload".to_vec(),
        };
        let back = RecordEnvelope::decode(&env.encode_binary()).unwrap();
        assert_eq!(env, back);
    }

    #[test]
    fn binary_round_trip_large_payload() {
        let env = RecordEnvelope {
            shareable: false,
            tags: vec!["x".into()],
            payload: vec![0xAB; 100_000],
        };
        let back = RecordEnvelope::decode(&env.encode_binary()).unwrap();
        assert_eq!(env, back);
    }

    #[test]
    fn binary_is_smaller_than_json_for_text_payloads() {
        // A realistic record: short shareable, a couple of tags, a
        // 1 KB text payload. Binary must beat JSON noticeably.
        let payload: Vec<u8> = (0..1024).map(|i| ((i % 26) as u8) + b'a').collect();
        let env = RecordEnvelope {
            shareable: true,
            tags: vec!["topic".into(), "skill:python".into()],
            payload,
        };
        let json_bytes = env.encode();
        let bin_bytes = env.encode_binary();
        // Binary is ~3.8x smaller than JSON for byte-array payloads
        // (JSON serialises each byte as a decimal + separator).
        // Assert at least 3x to leave headroom for future tweaks.
        assert!(
            bin_bytes.len() * 3 < json_bytes.len(),
            "expected binary < 1/3 JSON, got binary={} json={} ratio={:.2}x",
            bin_bytes.len(),
            json_bytes.len(),
            json_bytes.len() as f32 / bin_bytes.len() as f32,
        );
    }

    #[test]
    fn decode_rejects_garbage_first_byte() {
        let err = RecordEnvelope::decode(b"xyz garbage").unwrap_err();
        assert!(matches!(err, EnvelopeError::InvalidMagic(b'x')));
    }

    #[test]
    fn decode_rejects_empty_buffer() {
        let err = RecordEnvelope::decode(b"").unwrap_err();
        assert!(matches!(err, EnvelopeError::Truncated { .. }));
    }

    #[test]
    fn decode_rejects_truncated_binary() {
        let env = sample("hello");
        let bytes = env.encode_binary();
        let err = RecordEnvelope::decode(&bytes[..bytes.len() - 2]).unwrap_err();
        assert!(matches!(err, EnvelopeError::Truncated { .. }));
    }

    #[test]
    fn decode_detects_crc_corruption() {
        let env = sample("hello");
        let mut bytes = env.encode_binary();
        // Flip one byte in the payload region; CRC must catch it.
        let payload_start = bytes.len() - env.payload.len() - 4;
        bytes[payload_start] ^= 0xFF;
        let err = RecordEnvelope::decode(&bytes).unwrap_err();
        assert!(matches!(err, EnvelopeError::CrcMismatch { .. }));
    }

    #[test]
    fn decode_rejects_reserved_flags() {
        let env = sample("hi");
        let mut bytes = env.encode_binary();
        // Set bit 7 in flags (reserved). Will also break CRC but the
        // reserved-flag check fires first.
        bytes[1] |= 0b1000_0000;
        let err = RecordEnvelope::decode(&bytes).unwrap_err();
        assert!(matches!(err, EnvelopeError::ReservedFlagsSet(_)));
    }

    // F.20 ─ zstd payload compression ───────────────────────────────

    #[test]
    fn binary_zstd_round_trip() {
        let env = sample("hello world this is a longer payload to be compressed");
        let bytes = env.encode_binary_zstd(DEFAULT_ZSTD_LEVEL);
        assert_eq!(bytes[0], BINARY_MAGIC);
        assert!(bytes[1] & FLAG_ZSTD_PAYLOAD != 0, "zstd flag missing");
        let back = RecordEnvelope::decode(&bytes).unwrap();
        assert_eq!(env, back);
    }

    #[test]
    fn binary_zstd_round_trip_large_compressible() {
        let env = RecordEnvelope {
            shareable: true,
            tags: vec!["doc".into()],
            payload: "lorem ipsum dolor sit amet ".repeat(2_000).into_bytes(),
        };
        let bytes = env.encode_binary_zstd(DEFAULT_ZSTD_LEVEL);
        let back = RecordEnvelope::decode(&bytes).unwrap();
        assert_eq!(env, back);
    }

    #[test]
    fn binary_zstd_beats_plain_on_text_payload() {
        let env = RecordEnvelope {
            shareable: false,
            tags: vec!["x".into()],
            // Highly compressible: repeating English prose.
            payload: "the quick brown fox jumps over the lazy dog. "
                .repeat(500)
                .into_bytes(),
        };
        let plain = env.encode_binary();
        let zstd = env.encode_binary_zstd(DEFAULT_ZSTD_LEVEL);
        assert!(
            zstd.len() * 4 < plain.len(),
            "expected zstd < 1/4 plain, got zstd={} plain={}",
            zstd.len(),
            plain.len()
        );
    }

    #[test]
    fn binary_zstd_smallest_picks_shorter_per_payload() {
        // Compressible text -> zstd wins.
        let text_env = RecordEnvelope::new(true, vec![], "aaaaaaaaaa ".repeat(300).into_bytes());
        let text_smallest = text_env.encode_binary_smallest();
        let text_zstd = text_env.encode_binary_zstd(DEFAULT_ZSTD_LEVEL);
        assert_eq!(text_smallest, text_zstd);

        // Tiny payload -> plain wins (zstd frame overhead > savings).
        let tiny_env = RecordEnvelope::new(true, vec![], b"hi".to_vec());
        let tiny_smallest = tiny_env.encode_binary_smallest();
        let tiny_plain = tiny_env.encode_binary();
        assert_eq!(tiny_smallest, tiny_plain);
    }

    #[test]
    fn binary_zstd_corrupt_frame_returns_error() {
        let env = sample("hello world payload");
        let mut bytes = env.encode_binary_zstd(DEFAULT_ZSTD_LEVEL);
        // Corrupt one byte inside the zstd-compressed payload.
        let payload_start = bytes.len() - 4 - 1; // CRC + one byte back into payload
        bytes[payload_start] ^= 0xFF;
        // CRC catches this first — but if we fix the CRC it should
        // then fail with ZstdDecompress. Easier to just assert it's
        // an error of either kind.
        let err = RecordEnvelope::decode(&bytes).unwrap_err();
        assert!(matches!(
            err,
            EnvelopeError::CrcMismatch { .. } | EnvelopeError::ZstdDecompress(_)
        ));
    }

    #[test]
    fn binary_zstd_empty_payload_round_trip() {
        // Edge: zstd-flag with empty payload. zstd's empty-frame
        // round-trips fine; this guards against future regressions.
        let env = RecordEnvelope {
            shareable: false,
            tags: vec!["e".into()],
            payload: vec![],
        };
        let bytes = env.encode_binary_zstd(DEFAULT_ZSTD_LEVEL);
        let back = RecordEnvelope::decode(&bytes).unwrap();
        assert_eq!(env, back);
    }

    #[test]
    fn envelope_key_is_predictable() {
        assert_eq!(envelope_key_for(42), "hansa:rec:42");
        assert_eq!(envelope_key_for(0), "hansa:rec:0");
    }

    #[test]
    fn binary_format_round_trip_through_decode_binary() {
        // decode_binary refuses a JSON buffer outright.
        let env = sample("a");
        let json = env.encode();
        let err = RecordEnvelope::decode_binary(&json).unwrap_err();
        assert!(matches!(err, EnvelopeError::InvalidMagic(_)));
    }
}
