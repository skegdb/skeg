#![deny(unsafe_code)]

//! Segment files: naming, listing, and sequential recovery scan.
//!
//! A segment is a plain file of back-to-back padded records. Writes go through
//! [`crate::group_commit::GroupCommitter`]; reads use `PlatformFile::pread`.
//! This module only owns the parts that are not write-path: filename layout,
//! directory listing, and the recovery scan.

use std::io;
use std::path::{Path, PathBuf};

use skeg_platform::PlatformFile;

use crate::record::{HEADER_SIZE, Record, decode_record, padded_record_size};

/// Maximum segment file size before rotation (512 MiB).
pub const MAX_SEGMENT_SIZE: u64 = 512 * 1024 * 1024;

/// Path of a segment file given its directory and ID.
#[must_use]
pub fn segment_path(dir: &Path, id: u16) -> PathBuf {
    dir.join(format!("{id:010}.seg"))
}

/// List all segment IDs present in `dir`, sorted ascending.
///
/// # Errors
///
/// Returns an IO error if the directory cannot be read.
pub fn list_segments(dir: &Path) -> io::Result<Vec<u16>> {
    let mut ids = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let s = name.to_string_lossy();
        if let Some(stem) = s.strip_suffix(".seg")
            && let Ok(id) = stem.parse::<u16>()
        {
            ids.push(id);
        }
    }
    ids.sort_unstable();
    Ok(ids)
}

/// Scan a segment file from offset 0, invoking `f(offset, record)` for every
/// valid record. Stops at the first record that cannot be read - a truncated
/// tail or a CRC failure - which is the expected crash-recovery boundary.
///
/// Returns the offset just past the last valid record (the truncation point).
///
/// # Errors
///
/// Returns an IO error on read failure. CRC mismatches are *not* errors; they
/// terminate the scan cleanly.
///
/// # Panics
///
/// Panics only if header slicing invariants are violated after a length check
/// (cannot happen in practice).
pub fn scan_file(pf: &PlatformFile, mut f: impl FnMut(u64, Record)) -> io::Result<u64> {
    let mut offset = 0u64;
    let mut header = [0u8; HEADER_SIZE];

    loop {
        let n = pf.pread_sync(offset, &mut header)?;
        if n < HEADER_SIZE {
            break; // EOF or partial header
        }

        let ksz = u32::from_le_bytes(header[12..16].try_into().expect("4 bytes")) as usize;
        let vsz = u32::from_le_bytes(header[16..20].try_into().expect("4 bytes")) as usize;
        let padded = padded_record_size(ksz, vsz);

        let mut buf = vec![0u8; padded];
        let n2 = pf.pread_sync(offset, &mut buf)?;
        if n2 < padded {
            break; // truncated record
        }

        match decode_record(&buf) {
            Ok(rec) => {
                f(offset, rec);
                offset += u64::try_from(padded).expect("padded fits u64");
            }
            Err(_) => break, // CRC failure / corrupt kind → recovery boundary
        }
    }

    Ok(offset)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{RecordKind, encode_record};
    use std::io::Write as _;
    use tempfile::TempDir;

    #[test]
    fn test_segment_path_format() {
        let path = segment_path(Path::new("/data"), 1);
        assert_eq!(
            path.file_name().unwrap().to_str().unwrap(),
            "0000000001.seg"
        );
    }

    #[test]
    fn test_list_segments_sorted() {
        let dir = TempDir::new().unwrap();
        for id in [3u16, 1, 7] {
            std::fs::File::create(segment_path(dir.path(), id)).unwrap();
        }
        assert_eq!(list_segments(dir.path()).unwrap(), vec![1, 3, 7]);
    }

    #[test]
    fn test_scan_file_reads_all_records() {
        let dir = TempDir::new().unwrap();
        let path = segment_path(dir.path(), 0);
        {
            let mut file = std::fs::File::create(&path).unwrap();
            for i in 0u64..5 {
                let rec = encode_record(
                    format!("key{i}").as_bytes(),
                    format!("val{i}").as_bytes(),
                    RecordKind::Scalar,
                    i,
                );
                file.write_all(&rec).unwrap();
            }
        }

        let pf = PlatformFile::open(&path).unwrap();
        let mut records = Vec::new();
        let last = scan_file(&pf, |_, r| records.push(r)).unwrap();
        assert_eq!(records.len(), 5);
        assert_eq!(last, pf.size().unwrap());
        for (i, r) in records.iter().enumerate() {
            assert_eq!(r.key, format!("key{i}").as_bytes());
        }
    }

    #[test]
    fn test_scan_file_stops_at_corruption() {
        let dir = TempDir::new().unwrap();
        let path = segment_path(dir.path(), 0);
        let good_len;
        {
            let mut file = std::fs::File::create(&path).unwrap();
            let rec = encode_record(b"good", b"record", RecordKind::Scalar, 1);
            good_len = rec.len() as u64;
            file.write_all(&rec).unwrap();
            // Partial garbage tail - simulates a crash mid-write.
            file.write_all(&[0xDE, 0xAD, 0xBE, 0xEF]).unwrap();
        }

        let pf = PlatformFile::open(&path).unwrap();
        let mut records = Vec::new();
        let last = scan_file(&pf, |_, r| records.push(r)).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(last, good_len, "scan stops before the partial tail");
    }

    #[test]
    fn test_scan_file_empty() {
        let dir = TempDir::new().unwrap();
        let path = segment_path(dir.path(), 0);
        std::fs::File::create(&path).unwrap();
        let pf = PlatformFile::open(&path).unwrap();
        let mut count = 0;
        let last = scan_file(&pf, |_, _| count += 1).unwrap();
        assert_eq!(count, 0);
        assert_eq!(last, 0);
    }
}
