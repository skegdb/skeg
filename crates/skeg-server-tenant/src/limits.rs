//! Persisted per-tenant quota limits: a small sidecar next to `auth.kdb`.
//!
//! The engine enforces `max_vectors` and `max_disk_bytes` per tenant; this
//! store is where an admin's `SKEG.QUOTA.SET` writes them so they survive a
//! restart. Kept separate from `auth.kdb` so the security-critical auth format
//! is untouched. Each field is `Option<u64>` (`None` = unlimited).

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"SKEGQUOT";
const VERSION: u32 = 1;

/// One tenant's limits: `(max_vectors, max_disk_bytes)`, `None` = unlimited.
pub type Limits = (Option<u64>, Option<u64>);

/// On-disk map of tenant id (16 bytes) to limits. Writes are atomic
/// (temp file + rename).
#[derive(Debug, Default)]
pub struct LimitsStore {
    path: PathBuf,
    by_tenant: HashMap<[u8; 16], Limits>,
}

impl LimitsStore {
    /// Open the store at `path`, loading it if the file exists. A missing
    /// file is an empty store (no limits configured yet).
    ///
    /// # Errors
    ///
    /// Returns an IO error on read failure or a malformed file.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut store = Self {
            path: path.clone(),
            by_tenant: HashMap::new(),
        };
        if path.exists() {
            let bytes = std::fs::read(&path)?;
            store.parse_into(&bytes)?;
        }
        Ok(store)
    }

    /// Limits for `tenant` (both `None` if untracked).
    #[must_use]
    pub fn get(&self, tenant: [u8; 16]) -> Limits {
        self.by_tenant.get(&tenant).copied().unwrap_or((None, None))
    }

    /// Set `tenant`'s limits and persist atomically.
    ///
    /// # Errors
    ///
    /// Returns an IO error if the file cannot be written.
    pub fn set(&mut self, tenant: [u8; 16], limits: Limits) -> io::Result<()> {
        self.by_tenant.insert(tenant, limits);
        self.save()
    }

    fn save(&self) -> io::Result<()> {
        let mut buf = Vec::with_capacity(16 + self.by_tenant.len() * 34);
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&(self.by_tenant.len() as u32).to_le_bytes());
        for (tid, (mv, md)) in &self.by_tenant {
            buf.extend_from_slice(tid);
            push_opt(&mut buf, *mv);
            push_opt(&mut buf, *md);
        }
        let tmp = self.path.with_extension("quotas.tmp");
        std::fs::write(&tmp, &buf)?;
        std::fs::rename(&tmp, &self.path)
    }

    fn parse_into(&mut self, bytes: &[u8]) -> io::Result<()> {
        let bad = || io::Error::new(io::ErrorKind::InvalidData, "malformed quota store");
        if bytes.len() < 16 || &bytes[0..8] != MAGIC {
            return Err(bad());
        }
        if u32::from_le_bytes(bytes[8..12].try_into().unwrap()) != VERSION {
            return Err(bad());
        }
        let n = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let mut off = 16usize;
        for _ in 0..n {
            if off + 16 + 18 > bytes.len() {
                return Err(bad());
            }
            let mut tid = [0u8; 16];
            tid.copy_from_slice(&bytes[off..off + 16]);
            off += 16;
            let mv = read_opt(&bytes[off..off + 9]);
            off += 9;
            let md = read_opt(&bytes[off..off + 9]);
            off += 9;
            self.by_tenant.insert(tid, (mv, md));
        }
        Ok(())
    }
}

/// Encode an `Option<u64>` as a 1-byte present flag plus 8 LE bytes.
fn push_opt(buf: &mut Vec<u8>, v: Option<u64>) {
    buf.push(u8::from(v.is_some()));
    buf.extend_from_slice(&v.unwrap_or(0).to_le_bytes());
}

/// Decode a 9-byte `[flag][u64 LE]` into an `Option<u64>`.
fn read_opt(b: &[u8]) -> Option<u64> {
    if b[0] == 0 {
        None
    } else {
        Some(u64::from_le_bytes(b[1..9].try_into().unwrap()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn set_get_roundtrip() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("auth.kdb.quotas");
        let mut s = LimitsStore::open(&p).unwrap();
        assert_eq!(s.get([1u8; 16]), (None, None));
        s.set([1u8; 16], (Some(100), None)).unwrap();
        s.set([2u8; 16], (None, Some(4096))).unwrap();
        assert_eq!(s.get([1u8; 16]), (Some(100), None));
        assert_eq!(s.get([2u8; 16]), (None, Some(4096)));
    }

    #[test]
    fn persists_across_reopen() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("auth.kdb.quotas");
        {
            let mut s = LimitsStore::open(&p).unwrap();
            s.set([7u8; 16], (Some(5), Some(9))).unwrap();
        }
        let s = LimitsStore::open(&p).unwrap();
        assert_eq!(s.get([7u8; 16]), (Some(5), Some(9)));
        assert_eq!(s.get([8u8; 16]), (None, None));
    }

    #[test]
    fn missing_file_is_empty() {
        let dir = TempDir::new().unwrap();
        let s = LimitsStore::open(dir.path().join("nope.quotas")).unwrap();
        assert_eq!(s.get([1u8; 16]), (None, None));
    }
}
