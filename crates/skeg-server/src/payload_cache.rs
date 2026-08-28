//! Persisted payload index for a recovered vindex.
//!
//! Warming a vindex's payload index reads every live id's blob from the vlog,
//! one random read each: 471.918 of them cost ~10s on a real corpus, and that
//! is the whole of a restart. The blobs are already on disk in a known order,
//! so the same content read sequentially from one file is a different order of
//! cost.
//!
//! # Why this cannot go stale
//!
//! Payload blobs live in the vlog, which keeps changing after this file is
//! written, so a cache validated only against the vindex directory would serve
//! values that were overwritten since. Two things prevent that:
//!
//! - the file is stamped with the vlog snapshot's `(hwm, hwm_offset)` and is
//!   only considered at all when recovery seeded from that exact snapshot;
//! - an id whose payload key appears in the replayed log tail is refused, and
//!   the caller reads it from the vlog instead.
//!
//! Anything not covered by both is read the old way. A wrong payload index
//! makes a filtered search silently drop results, which is far worse than a
//! slow open, so every uncertainty resolves towards re-reading.

use std::io::{Read, Write};
use std::path::Path;

/// `"SPLC"`, payload cache.
const MAGIC: u32 = 0x5350_4C43;
const VERSION: u32 = 1;
/// magic + version + hwm + hwm_offset + count + crc32c(body).
const HEADER: usize = 4 + 4 + 8 + 8 + 8 + 4;

/// File name inside a `vindex-<name>/` directory.
pub const FILE: &str = "payload.cache.bin";

/// Serialise `entries` (`(id, blob)`) stamped with the vlog snapshot position
/// they reflect.
///
/// Written to a temporary file and renamed, both fsynced, so a crash midway
/// leaves the previous file or none, never a half one. Created `0600` rather
/// than inheriting the umask: the blobs are user payloads and this file sits
/// next to the store's other owner-only data.
///
/// # Errors
///
/// Returns an error if the file cannot be written, synced or renamed.
pub fn write(dir: &Path, stamp: (u64, u64), entries: &[(u64, Vec<u8>)]) -> std::io::Result<()> {
    let mut body = Vec::with_capacity(entries.iter().map(|(_, b)| 12 + b.len()).sum());
    for (id, blob) in entries {
        body.extend_from_slice(&id.to_le_bytes());
        body.extend_from_slice(&(blob.len() as u32).to_le_bytes());
        body.extend_from_slice(blob);
    }
    let mut out = Vec::with_capacity(HEADER + body.len());
    out.extend_from_slice(&MAGIC.to_le_bytes());
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&stamp.0.to_le_bytes());
    out.extend_from_slice(&stamp.1.to_le_bytes());
    out.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    out.extend_from_slice(&crc32c::crc32c(&body).to_le_bytes());
    out.extend_from_slice(&body);

    let path = dir.join(FILE);
    let tmp = path.with_extension("bin.tmp");
    {
        let mut opts = std::fs::OpenOptions::new();
        opts.create(true).truncate(true).write(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let mut f = opts.open(&tmp)?;
        f.write_all(&out)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, &path)?;
    skeg_platform::sync_dir(dir)
}

/// Read the cache if it exists, is intact, and carries `stamp`.
///
/// `max_entries` is how many the caller can actually use (the vindex's live id
/// count). A file claiming more than that was not written for this index, and
/// is refused before anything is allocated for it.
///
/// Returns `None` for every other case too: absent, wrong magic or version, a
/// different stamp, a failed checksum, truncated, or a length field that runs
/// past the end. A caller that gets `None` rebuilds from the log, which is
/// always correct.
///
/// # Order of checks
///
/// The 32-byte header is read and validated on its own, and the body is only
/// read after `count` has been bounded. A crafted or corrupt file therefore
/// cannot make this allocate from a length field it supplied: the same rule
/// the record and snapshot decoders follow.
#[must_use]
pub fn read(dir: &Path, stamp: (u64, u64), max_entries: usize) -> Option<Vec<(u64, Vec<u8>)>> {
    let path = dir.join(FILE);
    let mut f = std::fs::File::open(&path).ok()?;
    let mut head = [0u8; HEADER];
    f.read_exact(&mut head).ok()?;
    let u32_at = |o: usize| u32::from_le_bytes([head[o], head[o + 1], head[o + 2], head[o + 3]]);
    let u64_at = |o: usize| {
        let mut b = [0u8; 8];
        b.copy_from_slice(&head[o..o + 8]);
        u64::from_le_bytes(b)
    };
    if u32_at(0) != MAGIC || u32_at(4) != VERSION {
        return None;
    }
    if (u64_at(8), u64_at(16)) != stamp {
        return None;
    }
    let count = usize::try_from(u64_at(24)).ok()?;
    if count > max_entries {
        return None;
    }
    let expected_crc = u32_at(32);

    // Even bounded by `count`, the body length is still whatever is on disk;
    // read it and let the checksum and the parse below judge it.
    let body_len = f.metadata().ok()?.len().checked_sub(HEADER as u64)?;
    let mut body = Vec::new();
    body.try_reserve_exact(usize::try_from(body_len).ok()?).ok()?;
    f.read_to_end(&mut body).ok()?;
    if crc32c::crc32c(&body) != expected_crc {
        return None;
    }

    let u32_b = |o: usize| u32::from_le_bytes([body[o], body[o + 1], body[o + 2], body[o + 3]]);
    let u64_b = |o: usize| {
        let mut b = [0u8; 8];
        b.copy_from_slice(&body[o..o + 8]);
        u64::from_le_bytes(b)
    };
    let mut out = Vec::with_capacity(count.min(body.len() / 12));
    let mut o = 0usize;
    for _ in 0..count {
        if o + 12 > body.len() {
            return None;
        }
        let id = u64_b(o);
        let len = u32_b(o + 8) as usize;
        o += 12;
        if o + len > body.len() {
            return None;
        }
        out.push((id, body[o..o + len].to_vec()));
        o += len;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn sample() -> Vec<(u64, Vec<u8>)> {
        vec![
            (1, b"dl=10 lic=mit".to_vec()),
            (2, b"dl=20 lic=apache-2.0 params_m=7000".to_vec()),
            (9, Vec::new()), // an empty payload is a real case, not a missing one
        ]
    }

    #[test]
    fn round_trips() {
        let d = TempDir::new().unwrap();
        write(d.path(), (3, 4096), &sample()).unwrap();
        assert_eq!(read(d.path(), (3, 4096), 8), Some(sample()));
    }

    #[test]
    fn a_different_stamp_is_refused() {
        let d = TempDir::new().unwrap();
        write(d.path(), (3, 4096), &sample()).unwrap();
        // Same segment, log has moved on: the payloads may have changed.
        assert_eq!(read(d.path(), (3, 8192), 8), None);
        // Same offset, different segment.
        assert_eq!(read(d.path(), (4, 4096), 8), None);
    }

    #[test]
    fn absent_file_is_not_an_error() {
        let d = TempDir::new().unwrap();
        assert_eq!(read(d.path(), (0, 0), 8), None);
    }

    #[test]
    fn truncation_at_every_length_is_refused_and_never_panics() {
        let d = TempDir::new().unwrap();
        write(d.path(), (1, 2), &sample()).unwrap();
        let full = std::fs::read(d.path().join(FILE)).unwrap();
        for cut in 0..full.len() {
            std::fs::write(d.path().join(FILE), &full[..cut]).unwrap();
            assert_eq!(
                read(d.path(), (1, 2), 8),
                None,
                "a file cut at {cut} of {} was accepted",
                full.len()
            );
        }
    }

    #[test]
    fn a_lying_length_field_is_refused() {
        let d = TempDir::new().unwrap();
        write(d.path(), (1, 2), &sample()).unwrap();
        let mut buf = std::fs::read(d.path().join(FILE)).unwrap();
        // Claim far more entries than the bytes can hold.
        buf[24..32].copy_from_slice(&u64::MAX.to_le_bytes());
        std::fs::write(d.path().join(FILE), &buf).unwrap();
        assert_eq!(read(d.path(), (1, 2), 8), None);

        // Claim one entry whose blob runs past the end.
        let mut buf = std::fs::read(d.path().join(FILE)).unwrap();
        buf[24..32].copy_from_slice(&1u64.to_le_bytes());
        buf[HEADER + 8..HEADER + 12].copy_from_slice(&u32::MAX.to_le_bytes());
        std::fs::write(d.path().join(FILE), &buf).unwrap();
        assert_eq!(read(d.path(), (1, 2), 8), None);
    }

    #[test]
    fn a_foreign_file_is_refused() {
        let d = TempDir::new().unwrap();
        std::fs::write(d.path().join(FILE), vec![0xAB; 4096]).unwrap();
        assert_eq!(read(d.path(), (0, 0), 8), None);
    }

    #[test]
    fn rewriting_replaces_the_previous_file() {
        let d = TempDir::new().unwrap();
        write(d.path(), (1, 1), &sample()).unwrap();
        let newer = vec![(5u64, b"dl=1".to_vec())];
        write(d.path(), (2, 2), &newer).unwrap();
        assert_eq!(read(d.path(), (1, 1), 8), None, "the old stamp must stop matching");
        assert_eq!(read(d.path(), (2, 2), 8), Some(newer));
        assert!(!d.path().join("payload.cache.bin.tmp").exists(), "the temporary file must be renamed, not left behind");
    }

    #[test]
    fn a_flipped_bit_in_the_body_is_refused() {
        let d = TempDir::new().unwrap();
        write(d.path(), (1, 2), &sample()).unwrap();
        let mut buf = std::fs::read(d.path().join(FILE)).unwrap();
        let last = buf.len() - 1;
        buf[last] ^= 0x01;
        std::fs::write(d.path().join(FILE), &buf).unwrap();
        assert_eq!(
            read(d.path(), (1, 2), 8),
            None,
            "a corrupt payload cache must be refused, not fed into the index"
        );
    }

    #[test]
    fn more_entries_than_the_caller_can_use_is_refused() {
        let d = TempDir::new().unwrap();
        write(d.path(), (1, 2), &sample()).unwrap();
        assert_eq!(read(d.path(), (1, 2), 2), None, "3 entries for an index holding 2");
        assert!(read(d.path(), (1, 2), 3).is_some());
    }

    #[cfg(unix)]
    #[test]
    fn the_file_is_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let d = TempDir::new().unwrap();
        write(d.path(), (1, 2), &sample()).unwrap();
        let mode = std::fs::metadata(d.path().join(FILE)).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600, "payload blobs must not inherit the umask");
    }
}
