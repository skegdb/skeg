#![deny(unsafe_code)]

use crc32c::crc32c;

/// Kind of value stored in a vLog record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RecordKind {
    Scalar = 0,
    Tombstone = 1,
    VecF32 = 2,
    VecInt8 = 3,
    VecBinary = 4,
    /// Header of an atomic multi-key write (see `VLog::set_many`). Its value is
    /// the `u32` LE count `N` of the records that immediately follow it as one
    /// contiguous group. Recovery buffers those `N` records and applies them
    /// only if all `N` are intact; a batch torn by a crash (fewer than `N`
    /// durable) is dropped whole, giving all-or-nothing semantics. Carries no
    /// key and is never indexed; compaction drops it like any unreferenced
    /// record, and the members it introduced live on as ordinary `Scalar`s.
    BatchBegin = 5,
}

impl RecordKind {
    /// Parse kind byte.
    ///
    /// # Errors
    ///
    /// Returns `UnknownKind` if `k` is not a recognized discriminant.
    pub fn from_u8(k: u8) -> Result<Self, crate::Error> {
        match k {
            0 => Ok(Self::Scalar),
            1 => Ok(Self::Tombstone),
            2 => Ok(Self::VecF32),
            3 => Ok(Self::VecInt8),
            4 => Ok(Self::VecBinary),
            5 => Ok(Self::BatchBegin),
            _ => Err(crate::Error::UnknownKind { kind: k }),
        }
    }
}

/// Decoded vLog record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    pub ts: u64,
    pub kind: RecordKind,
    pub key: Vec<u8>,
    pub value: Vec<u8>,
}

/// Fixed bytes in a record header: crc(4) + ts(8) + ksz(4) + vsz(4) + kind(1) = 21.
pub const HEADER_SIZE: usize = 21;

/// On-disk size of a record with the given key/value lengths,
/// padded to a 128-byte boundary (M1 cache line).
#[must_use]
pub fn padded_record_size(ksz: usize, vsz: usize) -> usize {
    (HEADER_SIZE + ksz + vsz + 127) & !127
}

/// Encode `key`/`value` into a zero-padded record buffer.
///
/// Layout: `[crc32c 4B][ts 8B][ksz 4B][vsz 4B][kind 1B][key][value][zero padding]`
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn encode_record(key: &[u8], value: &[u8], kind: RecordKind, ts: u64) -> Vec<u8> {
    let padded = padded_record_size(key.len(), value.len());
    let mut buf = vec![0u8; padded];

    buf[4..12].copy_from_slice(&ts.to_le_bytes());
    buf[12..16].copy_from_slice(&(key.len() as u32).to_le_bytes());
    buf[16..20].copy_from_slice(&(value.len() as u32).to_le_bytes());
    buf[20] = kind as u8;
    buf[21..21 + key.len()].copy_from_slice(key);
    buf[21 + key.len()..21 + key.len() + value.len()].copy_from_slice(value);

    let crc_end = 21 + key.len() + value.len();
    let crc = crc32c(&buf[4..crc_end]);
    buf[0..4].copy_from_slice(&crc.to_le_bytes());

    buf
}

/// Decode a record from `buf`. Verifies CRC.
///
/// # Errors
///
/// Returns an error if the buffer is too short, the CRC does not match,
/// or the kind byte is unrecognized.
///
/// # Panics
///
/// Panics if `buf.len() >= HEADER_SIZE` but slicing invariants are violated
/// (cannot happen given the length checks above).
pub fn decode_record(buf: &[u8]) -> Result<Record, crate::Error> {
    if buf.len() < HEADER_SIZE {
        return Err(crate::Error::InvalidRecord {
            msg: "buffer too short for header",
        });
    }

    let stored_crc = u32::from_le_bytes(buf[0..4].try_into().expect("4 bytes"));
    let ts = u64::from_le_bytes(buf[4..12].try_into().expect("8 bytes"));
    let ksz = u32::from_le_bytes(buf[12..16].try_into().expect("4 bytes")) as usize;
    let vsz = u32::from_le_bytes(buf[16..20].try_into().expect("4 bytes")) as usize;
    let kind = RecordKind::from_u8(buf[20])?;

    let payload_end = HEADER_SIZE + ksz + vsz;
    if buf.len() < payload_end {
        return Err(crate::Error::InvalidRecord {
            msg: "buffer too short for key/value",
        });
    }

    let computed = crc32c(&buf[4..payload_end]);
    if computed != stored_crc {
        return Err(crate::Error::CrcMismatch {
            expected: computed,
            got: stored_crc,
        });
    }

    let key_start = HEADER_SIZE;
    let val_start = key_start + ksz;
    Ok(Record {
        ts,
        kind,
        key: buf[key_start..key_start + ksz].to_vec(),
        value: buf[val_start..val_start + vsz].to_vec(),
    })
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padded_size_aligns_to_128() {
        // HEADER_SIZE = 21; key=0, val=0 → raw=21, padded=128
        assert_eq!(padded_record_size(0, 0), 128);
        // raw=128 → already aligned
        assert_eq!(padded_record_size(107, 0), 128);
        // raw=129 → next multiple of 128
        assert_eq!(padded_record_size(108, 0), 256);
    }

    #[test]
    fn test_vlog_record_roundtrip() {
        let key = b"hello";
        let value = b"world";
        let buf = encode_record(key, value, RecordKind::Scalar, 42);
        let rec = decode_record(&buf).expect("decode");
        assert_eq!(rec.ts, 42);
        assert_eq!(rec.kind, RecordKind::Scalar);
        assert_eq!(rec.key, key);
        assert_eq!(rec.value, value);
    }

    #[test]
    fn test_vlog_record_tombstone_roundtrip() {
        let buf = encode_record(b"k", b"", RecordKind::Tombstone, 99);
        let rec = decode_record(&buf).expect("decode tombstone");
        assert_eq!(rec.kind, RecordKind::Tombstone);
        assert!(rec.value.is_empty());
    }

    #[test]
    fn test_vlog_crc_corruption_detected() {
        let mut buf = encode_record(b"key", b"val", RecordKind::Scalar, 1);
        // Flip a bit in the payload (after the CRC field)
        buf[10] ^= 0xFF;
        let err = decode_record(&buf).expect_err("should fail");
        assert!(matches!(err, crate::Error::CrcMismatch { .. }));
    }

    #[test]
    fn test_decode_too_short_returns_error() {
        let err = decode_record(&[0u8; 10]).expect_err("too short");
        assert!(matches!(err, crate::Error::InvalidRecord { .. }));
    }

    #[test]
    fn test_decode_truncated_payload_returns_error() {
        let buf = encode_record(b"key", b"value", RecordKind::Scalar, 1);
        // Trim to just the header - no key/value bytes
        let err = decode_record(&buf[..HEADER_SIZE]).expect_err("truncated payload");
        assert!(matches!(err, crate::Error::InvalidRecord { .. }));
    }

    #[test]
    fn test_unknown_kind_returns_error() {
        let mut buf = encode_record(b"k", b"v", RecordKind::Scalar, 1);
        // Overwrite kind byte with unknown value, then fix CRC
        buf[20] = 0xFF;
        let crc_end = HEADER_SIZE + 1 + 1;
        let crc = crc32c::crc32c(&buf[4..crc_end]);
        buf[0..4].copy_from_slice(&crc.to_le_bytes());
        let err = decode_record(&buf).expect_err("unknown kind");
        assert!(matches!(err, crate::Error::UnknownKind { kind: 0xFF }));
    }

    #[test]
    fn test_zero_key_zero_value() {
        let buf = encode_record(b"", b"", RecordKind::Scalar, 0);
        assert_eq!(buf.len(), 128);
        let rec = decode_record(&buf).expect("decode empty");
        assert!(rec.key.is_empty());
        assert!(rec.value.is_empty());
    }

    // ── proptest ─────────────────────────────────────────────────────────────

    proptest::proptest! {
        #[test]
        fn prop_record_encode_decode(
            key   in proptest::collection::vec(proptest::num::u8::ANY, 0..256),
            value in proptest::collection::vec(proptest::num::u8::ANY, 0..256),
            ts    in proptest::num::u64::ANY,
        ) {
            let buf = encode_record(&key, &value, RecordKind::Scalar, ts);
            let rec = decode_record(&buf).expect("roundtrip");
            proptest::prop_assert_eq!(rec.key,   key);
            proptest::prop_assert_eq!(rec.value, value);
            proptest::prop_assert_eq!(rec.ts,    ts);
        }
    }
}
