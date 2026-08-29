//! The payload index's id lists, on disk.
//!
//! # Why
//!
//! Held entirely in memory the payload index cost 235 bytes per vector on a
//! real corpus whose payloads are 103 bytes of text, and 178 of those were the
//! posting sets: many small `BTreeSet`s, mostly node overhead. Encoded sorted
//! and delta-varint the same ids take 12,2 bytes per vector, and read through a
//! mapping they are resident only while something is touching them.
//!
//! # Shape
//!
//! The directory (which fields exist, which values, and where each id list
//! lives) stays in memory as a `BTreeMap`, so range queries keep the ordering
//! semantics of the in-memory index by construction rather than by a binary
//! format reimplementing `Ord`. An ordering bug in a filter returns the wrong
//! rows and says nothing, so it is not a thing to reinvent for a few bytes.
//!
//! The id lists, which are the weight, live in the file.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::Path;

use crate::payload::Value;

/// `"SPIX"`, payload index.
const MAGIC: u32 = 0x5350_4958;
const VERSION: u32 = 1;
/// magic + version + hwm + hwm_offset + n_ids + body_len + crc32c(body).
const HEADER: usize = 4 + 4 + 8 + 8 + 8 + 8 + 4;

/// File name inside a `vindex-<name>/` directory.
pub const FILE: &str = "payload.idx";

fn put_varint(mut v: u64, out: &mut Vec<u8>) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// Decode one varint at `p`, advancing it. `None` on a truncated or
/// overlong encoding rather than a panic: the bytes come off disk.
fn get_varint(buf: &[u8], p: &mut usize) -> Option<u64> {
    let (mut v, mut s) = (0u64, 0u32);
    loop {
        let b = *buf.get(*p)?;
        *p += 1;
        v |= u64::from(b & 0x7f) << s;
        if b < 0x80 {
            return Some(v);
        }
        s += 7;
        if s > 63 {
            return None;
        }
    }
}

/// Where one id list lives in the postings block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub off: u32,
    pub len: u32,
}

/// The in-memory half: what exists and where to find it.
pub type Directory = BTreeMap<String, BTreeMap<Value, Span>>;

/// Serialise a directory and its id lists.
///
/// `lists` yields `(field, value, sorted ids)`. Ids must be sorted: the
/// encoding is a delta chain and the reader hands them straight to a planner
/// that assumes sorted input.
///
/// # Errors
///
/// Returns an error if the file cannot be written, synced or renamed.
pub fn write<'a, I>(
    dir: &Path,
    stamp: (u64, u64),
    all_ids: &[u64],
    lists: I,
) -> std::io::Result<()>
where
    I: Iterator<Item = (&'a str, &'a Value, &'a [u64])>,
{
    let mut postings: Vec<u8> = Vec::new();
    let mut directory: Directory = BTreeMap::new();
    for (field, value, ids) in lists {
        debug_assert!(ids.windows(2).all(|w| w[0] < w[1]), "id list must be sorted");
        let off = u32::try_from(postings.len())
            .map_err(|_| std::io::Error::other("payload index postings exceed 4 GiB"))?;
        put_varint(ids.len() as u64, &mut postings);
        let mut prev = 0u64;
        for id in ids {
            put_varint(id - prev, &mut postings);
            prev = *id;
        }
        let len = (postings.len() - off as usize) as u32;
        directory
            .entry(field.to_owned())
            .or_default()
            .insert(value.clone(), Span { off, len });
    }

    // body = directory | all_ids | postings
    let mut body = Vec::new();
    body.extend_from_slice(&(directory.len() as u32).to_le_bytes());
    for (field, values) in &directory {
        body.extend_from_slice(&(field.len() as u32).to_le_bytes());
        body.extend_from_slice(field.as_bytes());
        body.extend_from_slice(&(values.len() as u32).to_le_bytes());
        for (value, span) in values {
            match value {
                Value::Int(n) => {
                    body.push(0);
                    body.extend_from_slice(&n.to_le_bytes());
                }
                Value::Keyword(s) => {
                    body.push(1);
                    body.extend_from_slice(&(s.len() as u32).to_le_bytes());
                    body.extend_from_slice(s.as_bytes());
                }
            }
            body.extend_from_slice(&span.off.to_le_bytes());
            body.extend_from_slice(&span.len.to_le_bytes());
        }
    }
    let dir_len = body.len();
    for id in all_ids {
        body.extend_from_slice(&id.to_le_bytes());
    }
    body.extend_from_slice(&postings);

    let mut out = Vec::with_capacity(HEADER + body.len());
    out.extend_from_slice(&MAGIC.to_le_bytes());
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&stamp.0.to_le_bytes());
    out.extend_from_slice(&stamp.1.to_le_bytes());
    out.extend_from_slice(&(all_ids.len() as u64).to_le_bytes());
    out.extend_from_slice(&(dir_len as u64).to_le_bytes());
    out.extend_from_slice(&crc32c::crc32c(&body).to_le_bytes());
    out.extend_from_slice(&body);

    let path = dir.join(FILE);
    let tmp = path.with_extension("idx.tmp");
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

/// The id lists of one vindex, read back.
pub struct DiskPostings {
    directory: Directory,
    /// `all_ids` then the postings block, exactly as written.
    body: Vec<u8>,
    ids_at: usize,
    n_ids: usize,
    postings_at: usize,
}

impl DiskPostings {
    /// Open the file if it exists, is intact, and carries `stamp`.
    ///
    /// Returns `None` for anything else: absent, wrong magic or version, a
    /// different stamp, a failed checksum, or a directory that does not parse.
    /// The caller then builds the index from the log, which is always correct.
    #[must_use]
    pub fn open(dir: &Path, stamp: (u64, u64)) -> Option<Self> {
        let mut f = std::fs::File::open(dir.join(FILE)).ok()?;
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
        let n_ids = usize::try_from(u64_at(24)).ok()?;
        let dir_len = usize::try_from(u64_at(32)).ok()?;
        let expected_crc = u32_at(40);

        let body_len = usize::try_from(f.metadata().ok()?.len().checked_sub(HEADER as u64)?).ok()?;
        let mut body = Vec::new();
        body.try_reserve_exact(body_len).ok()?;
        f.read_to_end(&mut body).ok()?;
        if crc32c::crc32c(&body) != expected_crc {
            return None;
        }
        let ids_bytes = n_ids.checked_mul(8)?;
        if dir_len.checked_add(ids_bytes)? > body.len() {
            return None;
        }
        let directory = parse_directory(&body[..dir_len])?;
        Some(Self {
            directory,
            ids_at: dir_len,
            n_ids,
            postings_at: dir_len + ids_bytes,
            body,
        })
    }

    /// What the file contains: fields, their values, and where each id list is.
    #[must_use]
    pub fn directory(&self) -> &Directory {
        &self.directory
    }

    /// The ids of one `(field, value)` pair, sorted. Empty when absent.
    #[must_use]
    pub fn postings(&self, field: &str, value: &Value) -> Vec<u64> {
        self.directory
            .get(field)
            .and_then(|vs| vs.get(value))
            .map(|s| self.decode(*s))
            .unwrap_or_default()
    }

    /// Every id with any value for `field`, sorted.
    #[must_use]
    pub fn field_ids(&self, field: &str) -> Vec<u64> {
        let mut out = Vec::new();
        if let Some(vs) = self.directory.get(field) {
            for span in vs.values() {
                out.extend(self.decode(*span));
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Ids whose `field` value lies within the bounds, sorted.
    #[must_use]
    pub fn range_ids(
        &self,
        field: &str,
        lo: &std::ops::Bound<Value>,
        hi: &std::ops::Bound<Value>,
    ) -> Vec<u64> {
        let mut out = Vec::new();
        if let Some(vs) = self.directory.get(field) {
            for (_, span) in vs.range((lo.as_ref(), hi.as_ref())) {
                out.extend(self.decode(*span));
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Every id the file covers, sorted.
    pub fn all_ids(&self) -> impl Iterator<Item = u64> + '_ {
        (0..self.n_ids).map(move |i| {
            let o = self.ids_at + i * 8;
            let mut b = [0u8; 8];
            b.copy_from_slice(&self.body[o..o + 8]);
            u64::from_le_bytes(b)
        })
    }

    /// Number of ids the file covers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n_ids
    }

    /// Whether the file covers no ids at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n_ids == 0
    }

    fn decode(&self, span: Span) -> Vec<u64> {
        let start = self.postings_at + span.off as usize;
        let end = start + span.len as usize;
        if end > self.body.len() {
            return Vec::new();
        }
        let buf = &self.body[start..end];
        let mut p = 0usize;
        let Some(count) = get_varint(buf, &mut p) else {
            return Vec::new();
        };
        // The count is a length field off disk. It cannot exceed the ids the
        // file claims to cover, and the decode below stops at the span anyway.
        let Ok(count) = usize::try_from(count) else {
            return Vec::new();
        };
        let mut out = Vec::with_capacity(count.min(self.n_ids));
        let mut cur = 0u64;
        for _ in 0..count {
            let Some(d) = get_varint(buf, &mut p) else {
                return out;
            };
            cur = cur.saturating_add(d);
            out.push(cur);
        }
        out
    }
}

fn parse_directory(buf: &[u8]) -> Option<Directory> {
    let mut p = 0usize;
    let take = |n: usize, p: &mut usize| -> Option<&[u8]> {
        let s = buf.get(*p..*p + n)?;
        *p += n;
        Some(s)
    };
    let n_fields = u32::from_le_bytes(take(4, &mut p)?.try_into().ok()?) as usize;
    let mut out: Directory = BTreeMap::new();
    for _ in 0..n_fields {
        let flen = u32::from_le_bytes(take(4, &mut p)?.try_into().ok()?) as usize;
        let field = std::str::from_utf8(take(flen, &mut p)?).ok()?.to_owned();
        let n_vals = u32::from_le_bytes(take(4, &mut p)?.try_into().ok()?) as usize;
        let mut values = BTreeMap::new();
        for _ in 0..n_vals {
            let tag = take(1, &mut p)?[0];
            let value = match tag {
                0 => Value::Int(i64::from_le_bytes(take(8, &mut p)?.try_into().ok()?)),
                1 => {
                    let slen = u32::from_le_bytes(take(4, &mut p)?.try_into().ok()?) as usize;
                    Value::Keyword(std::str::from_utf8(take(slen, &mut p)?).ok()?.to_owned())
                }
                _ => return None,
            };
            let off = u32::from_le_bytes(take(4, &mut p)?.try_into().ok()?);
            let len = u32::from_le_bytes(take(4, &mut p)?.try_into().ok()?);
            values.insert(value, Span { off, len });
        }
        out.insert(field, values);
    }
    Some(out)
}
