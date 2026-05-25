#![deny(unsafe_code)]

//! Index snapshot: a serialized copy of the in-RAM index for fast recovery.
//!
//! Without a snapshot, opening a `VLog` rescans every segment - O(n) in the
//! whole dataset (~110 us/record). With a snapshot, recovery loads the index
//! directly and rescans only the segments written since the snapshot was
//! taken (those with id >= the high-water mark).
//!
//! File layout (`index.snapshot`), little-endian:
//! ```text
//!   magic       u32   = 0x534B5350 ("SKSP")
//!   version     u8    = 1
//!   hwm         u16   active segment id when the snapshot was written
//!   max_ts      u64   clock value to resume from
//!   n_entries   u32
//!   entries     n_entries x { klen u32, key[klen], fingerprint u32,
//!                             segment_id u16, offset u32, size u32 }
//!   crc32c      u32   over every preceding byte
//! ```
//!
//! Written atomically: `index.snapshot.tmp` -> fsync -> rename.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use crc32c::crc32c;

use crate::index::IndexEntry;

const MAGIC: u32 = 0x534B_5350;
const VERSION: u8 = 1;
const HEADER_LEN: usize = 4 + 1 + 2 + 8 + 4; // magic+version+hwm+max_ts+n_entries

/// Path of the committed snapshot file.
#[must_use]
pub fn snapshot_path(dir: &Path) -> PathBuf {
    dir.join("index.snapshot")
}

fn snapshot_tmp_path(dir: &Path) -> PathBuf {
    dir.join("index.snapshot.tmp")
}

/// A decoded index snapshot.
pub struct Snapshot {
    /// Active segment id when the snapshot was taken; segments with a lower id
    /// are fully captured here and need not be rescanned.
    pub hwm: u16,
    /// Logical clock value to resume from.
    pub max_ts: u64,
    /// `(key, entry)` pairs of the index at snapshot time.
    pub entries: Vec<(Vec<u8>, IndexEntry)>,
}

/// Serialize `entries` into the snapshot wire format.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn encode(hwm: u16, max_ts: u64, entries: &[(Vec<u8>, IndexEntry)]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(HEADER_LEN + entries.len() * 32 + 4);
    buf.extend_from_slice(&MAGIC.to_le_bytes());
    buf.push(VERSION);
    buf.extend_from_slice(&hwm.to_le_bytes());
    buf.extend_from_slice(&max_ts.to_le_bytes());
    buf.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for (key, e) in entries {
        buf.extend_from_slice(&(key.len() as u32).to_le_bytes());
        buf.extend_from_slice(key);
        buf.extend_from_slice(&e.fingerprint.to_le_bytes());
        buf.extend_from_slice(&e.segment_id.to_le_bytes());
        buf.extend_from_slice(&e.offset.to_le_bytes());
        buf.extend_from_slice(&e.size.to_le_bytes());
    }
    let crc = crc32c(&buf);
    buf.extend_from_slice(&crc.to_le_bytes());
    buf
}

/// Decode a snapshot buffer. Returns `None` if the buffer is malformed,
/// truncated, has a bad magic/version, or fails the CRC check.
#[must_use]
pub fn decode(buf: &[u8]) -> Option<Snapshot> {
    if buf.len() < HEADER_LEN + 4 {
        return None;
    }
    let crc_stored = u32::from_le_bytes(buf[buf.len() - 4..].try_into().ok()?);
    if crc32c(&buf[..buf.len() - 4]) != crc_stored {
        return None;
    }
    if u32::from_le_bytes(buf[0..4].try_into().ok()?) != MAGIC {
        return None;
    }
    if buf[4] != VERSION {
        return None;
    }
    let hwm = u16::from_le_bytes(buf[5..7].try_into().ok()?);
    let max_ts = u64::from_le_bytes(buf[7..15].try_into().ok()?);
    let n = u32::from_le_bytes(buf[15..19].try_into().ok()?) as usize;

    let mut entries = Vec::with_capacity(n);
    let mut pos = HEADER_LEN;
    let end = buf.len() - 4;
    for _ in 0..n {
        if pos + 4 > end {
            return None;
        }
        let klen = u32::from_le_bytes(buf[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;
        if pos + klen + 14 > end {
            return None;
        }
        let key = buf[pos..pos + klen].to_vec();
        pos += klen;
        let fingerprint = u32::from_le_bytes(buf[pos..pos + 4].try_into().ok()?);
        let segment_id = u16::from_le_bytes(buf[pos + 4..pos + 6].try_into().ok()?);
        let offset = u32::from_le_bytes(buf[pos + 6..pos + 10].try_into().ok()?);
        let size = u32::from_le_bytes(buf[pos + 10..pos + 14].try_into().ok()?);
        pos += 14;
        entries.push((
            key,
            IndexEntry {
                fingerprint,
                segment_id,
                _pad: 0,
                offset,
                size,
            },
        ));
    }
    Some(Snapshot {
        hwm,
        max_ts,
        entries,
    })
}

/// Write a snapshot atomically to `dir/index.snapshot`.
///
/// Writes to a `.tmp` file, fsyncs it, then renames over the old snapshot.
/// A crash before the rename leaves the previous snapshot (or none) intact.
///
/// # Errors
///
/// Returns an IO error if the file cannot be written.
pub fn write(
    dir: &Path,
    hwm: u16,
    max_ts: u64,
    entries: &[(Vec<u8>, IndexEntry)],
) -> std::io::Result<()> {
    let buf = encode(hwm, max_ts, entries);
    let tmp = snapshot_tmp_path(dir);
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&buf)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, snapshot_path(dir))?;
    Ok(())
}

/// Read and decode `dir/index.snapshot`, or `None` if absent or corrupt.
#[must_use]
pub fn read(dir: &Path) -> Option<Snapshot> {
    let raw = std::fs::read(snapshot_path(dir)).ok()?;
    decode(&raw)
}

/// Remove the snapshot file, if present. Used to invalidate a snapshot that a
/// compaction has made stale.
///
/// # Errors
///
/// Returns an IO error other than "not found".
pub fn remove(dir: &Path) -> std::io::Result<()> {
    match std::fs::remove_file(snapshot_path(dir)) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn entry(seg: u16, off: u32) -> IndexEntry {
        IndexEntry {
            fingerprint: off ^ 0xABCD,
            segment_id: seg,
            _pad: 0,
            offset: off,
            size: 128,
        }
    }

    #[test]
    fn test_snapshot_encode_decode_roundtrip() {
        let entries = vec![
            (b"alpha".to_vec(), entry(0, 0)),
            (b"beta".to_vec(), entry(1, 256)),
            (b"".to_vec(), entry(2, 512)),
        ];
        let buf = encode(7, 9999, &entries);
        let snap = decode(&buf).expect("decode");
        assert_eq!(snap.hwm, 7);
        assert_eq!(snap.max_ts, 9999);
        assert_eq!(snap.entries.len(), 3);
        assert_eq!(snap.entries[0].0, b"alpha");
        assert_eq!(snap.entries[1].1.segment_id, 1);
        assert_eq!(snap.entries[1].1.offset, 256);
    }

    #[test]
    fn test_snapshot_crc_corruption_detected() {
        let entries = vec![(b"k".to_vec(), entry(0, 0))];
        let mut buf = encode(0, 1, &entries);
        buf[HEADER_LEN + 2] ^= 0xFF; // flip a byte inside the payload
        assert!(decode(&buf).is_none(), "corruption must fail the CRC check");
    }

    #[test]
    fn test_snapshot_truncated_returns_none() {
        let entries = vec![(b"k".to_vec(), entry(0, 0))];
        let buf = encode(0, 1, &entries);
        assert!(decode(&buf[..buf.len() / 2]).is_none());
        assert!(decode(&[]).is_none());
        assert!(decode(&[0u8; 8]).is_none());
    }

    #[test]
    fn test_snapshot_bad_magic_returns_none() {
        let mut buf = encode(0, 1, &[(b"k".to_vec(), entry(0, 0))]);
        buf[0] = 0xEE;
        // Recompute the CRC so only the magic is wrong.
        let crc = crc32c(&buf[..buf.len() - 4]);
        let n = buf.len();
        buf[n - 4..].copy_from_slice(&crc.to_le_bytes());
        assert!(decode(&buf).is_none(), "bad magic must be rejected");
    }

    #[test]
    fn test_snapshot_write_read_atomic() {
        let dir = TempDir::new().unwrap();
        let entries = vec![
            (b"one".to_vec(), entry(0, 0)),
            (b"two".to_vec(), entry(0, 128)),
        ];
        write(dir.path(), 3, 42, &entries).unwrap();
        // The tmp file must not survive a successful write.
        assert!(!snapshot_tmp_path(dir.path()).exists());

        let snap = read(dir.path()).expect("read back");
        assert_eq!(snap.hwm, 3);
        assert_eq!(snap.max_ts, 42);
        assert_eq!(snap.entries.len(), 2);
    }

    #[test]
    fn test_snapshot_read_absent_returns_none() {
        let dir = TempDir::new().unwrap();
        assert!(read(dir.path()).is_none());
    }

    #[test]
    fn test_snapshot_remove() {
        let dir = TempDir::new().unwrap();
        write(dir.path(), 0, 1, &[(b"k".to_vec(), entry(0, 0))]).unwrap();
        assert!(read(dir.path()).is_some());
        remove(dir.path()).unwrap();
        assert!(read(dir.path()).is_none());
        // Removing an absent snapshot is not an error.
        remove(dir.path()).unwrap();
    }
}
