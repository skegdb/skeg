//! Names whose catalogue fan-out is in flight.
//!
//! `VINDEX CREATE` and `VINDEX DROP` reach every shard, and each shard commits
//! into its own registry independently. Nothing coordinates them: a shard that
//! fails leaves the shards that succeeded committed, and the caller gets an
//! error describing a store it no longer has. Measured, on four shards with one
//! of them unable to write:
//!
//! * a failed CREATE left the index on three shards, accepting writes the
//!   caller believed impossible - and recreating the name with a different dim
//!   then took on the failed shard while the others refused it, leaving one
//!   name with two dims that LIST reported as a single agreed row;
//! * a failed DROP had already removed the data from three shards, so the
//!   remainder was unreachable through search, still on disk, still catalogued,
//!   and back at the next open.
//!
//! The fix is one durable fact recorded before the fan-out starts: this NAME is
//! in flight, under this OPERATION. The file's presence means exactly "not
//! acknowledged as successful", which is why it is cleared BEFORE a success is
//! returned: if the clear fails the caller is told the operation failed, and
//! the next open makes that true.
//!
//! The operation is recorded because the two do not resolve the same way, and
//! they differ in exactly one case. Read `k` as the number of shards whose
//! registry still lists the name:
//!
//! * `k < n` - the fan-out reached some shards and not others. Undoing a create
//!   and finishing a drop are the same action here: remove it everywhere.
//! * `k == n` - nothing was removed anywhere. A create in that state succeeded
//!   without being acknowledged, so it is undone; a DROP in that state never
//!   started, the caller was told it failed, and the index is whole. Removing
//!   it would turn a refused drop into a silent deletion.
//!
//! That second case is why this is not just a list of names. It was, until an
//! existing single-shard test failed on it: one shard means every failure is
//! `k == n`, so the ambiguity is not a corner, it is the common case.
//!
//! Read-only opens never resolve and never write; they decline to serve an
//! in-flight name instead, which is the same refusal to publish a half-state.

use std::io;
use std::path::{Path, PathBuf};

use crate::shard::{MAX_VINDEXES_PER_SHARD, validate_vindex_name};

const FILE: &str = "catalog-intents";
const MAGIC: &str = "SCI1";
/// Same bound the registry uses. A store with more names in flight than it can
/// hold indexes is not in a state worth parsing further.
const MAX_NAMES: usize = MAX_VINDEXES_PER_SHARD;
/// magic line + one 255-byte name per line + the crc line.
const MAX_BYTES: u64 = (5 + MAX_NAMES * 256 + 14) as u64;

fn path(root: &Path) -> PathBuf {
    root.join(FILE)
}

/// Text, one name per line, because a vindex name is `[A-Za-z0-9._:-]` and can
/// hold neither a newline nor a space - so the framing needs no lengths to be
/// unambiguous, and the parser has no arithmetic to get wrong. Every name is
/// re-validated on the way in, which is the property that matters: a corrupted
/// file cannot produce a name that escapes the directory it will be joined to.
/// What a recorded name was in the middle of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Create,
    Drop,
}

impl Op {
    fn as_str(self) -> &'static str {
        match self {
            Op::Create => "create",
            Op::Drop => "drop",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s {
            "create" => Some(Op::Create),
            "drop" => Some(Op::Drop),
            _ => None,
        }
    }
}

fn encode(entries: &[(Op, String)]) -> String {
    let body: String = entries
        .iter()
        .map(|(op, n)| format!("{} {n}\n", op.as_str()))
        .collect();
    format!(
        "{MAGIC}\n{body}crc={:08x}\n",
        crc32c::crc32c(body.as_bytes())
    )
}

/// Names currently in flight. A missing file is an empty list; anything else
/// that will not parse is an error, because guessing here means either dropping
/// a live index or leaving a half-built one.
pub fn pending(root: &Path) -> io::Result<Vec<(Op, String)>> {
    let p = path(root);
    let meta = match std::fs::metadata(&p) {
        Ok(m) => m,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    if meta.len() > MAX_BYTES {
        return Err(io::Error::other(format!(
            "{} is {} bytes, over the {MAX_BYTES} a catalogue intent file can be",
            p.display(),
            meta.len()
        )));
    }
    let text = std::fs::read_to_string(&p)?;
    let bad = |why: &str| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{}: {why}", p.display()),
        )
    };
    let body = text
        .strip_prefix(MAGIC)
        .and_then(|r| r.strip_prefix('\n'))
        .ok_or_else(|| bad("not a catalogue intent file"))?;
    let (body, crc_line) = body
        .rsplit_once("crc=")
        .ok_or_else(|| bad("has no checksum line"))?;
    let want = u32::from_str_radix(crc_line.trim_end_matches('\n'), 16)
        .map_err(|_| bad("checksum is not hex"))?;
    if crc32c::crc32c(body.as_bytes()) != want {
        return Err(bad("checksum does not match"));
    }
    let lines: Vec<&str> = body.lines().collect();
    if lines.len() > MAX_NAMES {
        return Err(bad(&format!(
            "lists {} names, over the {MAX_NAMES} allowed",
            lines.len()
        )));
    }
    let mut entries = Vec::with_capacity(lines.len());
    for line in lines {
        let (op, name) = line
            .split_once(' ')
            .ok_or_else(|| bad(&format!("'{line}' is not an operation and a name")))?;
        let op = Op::parse(op).ok_or_else(|| bad(&format!("'{op}' is not an operation")))?;
        validate_vindex_name(name).map_err(|_| bad(&format!("'{name}' is not a vindex name")))?;
        entries.push((op, name.to_owned()));
    }
    Ok(entries)
}

/// Publish by rename, and fsync the directory: a rename that has not reached
/// the disk is a fact the next open will not see.
fn write(root: &Path, entries: &[(Op, String)]) -> io::Result<()> {
    if entries.is_empty() {
        match std::fs::remove_file(path(root)) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e),
        }
    } else {
        let tmp = root.join(format!("{FILE}.tmp"));
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(encode(entries).as_bytes())?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, path(root))?;
    }
    std::fs::File::open(root)?.sync_all()
}

/// Record `name` as in flight under `op`. Idempotent for the same pair; a
/// later operation on a name already in flight replaces the earlier one,
/// because the later is what the store is now in the middle of.
pub fn record(root: &Path, op: Op, name: &str) -> io::Result<()> {
    let mut entries = pending(root)?;
    if entries.iter().any(|(o, n)| *o == op && n == name) {
        return Ok(());
    }
    entries.retain(|(_, n)| n != name);
    if entries.len() >= MAX_NAMES {
        return Err(io::Error::other(format!(
            "{MAX_NAMES} catalogue operations already in flight"
        )));
    }
    entries.push((op, name.to_owned()));
    entries.sort_by(|a, b| a.1.cmp(&b.1));
    write(root, &entries)
}

/// Forget `name`, whatever it was doing. Idempotent.
pub fn clear(root: &Path, name: &str) -> io::Result<()> {
    let mut entries = pending(root)?;
    let before = entries.len();
    entries.retain(|(_, n)| n != name);
    if entries.len() == before {
        return Ok(());
    }
    write(root, &entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn a_missing_file_is_an_empty_list_and_not_an_error() {
        let dir = TempDir::new().unwrap();
        assert!(pending(dir.path()).unwrap().is_empty());
    }

    #[test]
    fn names_round_trip_and_clearing_the_last_one_removes_the_file() {
        let dir = TempDir::new().unwrap();
        record(dir.path(), Op::Drop, "b").unwrap();
        record(dir.path(), Op::Create, "a").unwrap();
        record(dir.path(), Op::Create, "a").unwrap(); // idempotent
        assert_eq!(
            pending(dir.path()).unwrap(),
            vec![(Op::Create, "a".to_owned()), (Op::Drop, "b".to_owned())]
        );
        clear(dir.path(), "a").unwrap();
        assert_eq!(
            pending(dir.path()).unwrap(),
            vec![(Op::Drop, "b".to_owned())]
        );
        clear(dir.path(), "b").unwrap();
        assert!(pending(dir.path()).unwrap().is_empty());
        assert!(!path(dir.path()).exists(), "an empty list leaves no file");
    }

    #[test]
    fn a_scoped_name_survives_the_round_trip() {
        // Tenant-scoped names carry `::` and hex; they are the map key, and the
        // intent has to name exactly what a drop will be given.
        let dir = TempDir::new().unwrap();
        let name = "0000000000000000000000000000002a::idx";
        record(dir.path(), Op::Drop, name).unwrap();
        assert_eq!(
            pending(dir.path()).unwrap(),
            vec![(Op::Drop, name.to_owned())]
        );
    }

    #[test]
    fn a_corrupt_file_is_an_error_and_not_an_empty_list() {
        // Reading it as empty would silently abandon a half-finished fan-out.
        let dir = TempDir::new().unwrap();
        record(dir.path(), Op::Create, "a").unwrap();
        let mut bytes = std::fs::read(path(dir.path())).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        std::fs::write(path(dir.path()), &bytes).unwrap();
        assert_eq!(
            pending(dir.path()).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn a_truncated_file_is_an_error() {
        // Losing the tail loses the checksum line, which is the whole point of
        // putting it last: a file cut short cannot look complete.
        let dir = TempDir::new().unwrap();
        record(dir.path(), Op::Create, "abcdef").unwrap();
        let text = std::fs::read_to_string(path(dir.path())).unwrap();
        std::fs::write(path(dir.path()), &text[..text.len() - 8]).unwrap();
        assert_eq!(
            pending(dir.path()).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn a_name_that_is_not_a_vindex_name_is_refused() {
        // The property the text format buys: every name is re-validated, so a
        // corrupted file cannot hand back something that escapes the directory
        // it is about to be joined to. Checksum recomputed, so the ONLY thing
        // refusing this is the name check.
        let dir = TempDir::new().unwrap();
        for evil in ["../../etc/passwd", "..", "with space", "a/b"] {
            let body = format!("create {evil}\n");
            let text = format!(
                "{MAGIC}\n{body}crc={:08x}\n",
                crc32c::crc32c(body.as_bytes())
            );
            std::fs::write(path(dir.path()), text).unwrap();
            assert_eq!(
                pending(dir.path()).unwrap_err().kind(),
                io::ErrorKind::InvalidData,
                "accepted {evil:?} as a vindex name"
            );
        }
    }

    #[test]
    fn a_later_operation_on_the_same_name_replaces_the_earlier_one() {
        // A create left in flight and then dropped is, now, a drop: resolving
        // it as the create would undo something the caller asked to remove.
        let dir = TempDir::new().unwrap();
        record(dir.path(), Op::Create, "a").unwrap();
        record(dir.path(), Op::Drop, "a").unwrap();
        assert_eq!(
            pending(dir.path()).unwrap(),
            vec![(Op::Drop, "a".to_owned())]
        );
    }

    #[test]
    fn an_unknown_operation_is_refused() {
        let dir = TempDir::new().unwrap();
        let body = "rename a\n";
        let text = format!(
            "{MAGIC}\n{body}crc={:08x}\n",
            crc32c::crc32c(body.as_bytes())
        );
        std::fs::write(path(dir.path()), text).unwrap();
        assert_eq!(
            pending(dir.path()).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn an_enormous_file_is_refused_without_reading_it() {
        let dir = TempDir::new().unwrap();
        std::fs::write(path(dir.path()), vec![0u8; (MAX_BYTES + 1) as usize]).unwrap();
        assert!(pending(dir.path()).is_err());
    }
}
