//! Persisted per-tenant quota limits: a small text sidecar next to `auth.kdb`.
//!
//! One line per tenant: `<tenant-id-u128> <max_vectors|*> <max_disk_bytes|*>`.
//! `*` means unlimited. Kept separate from `auth.kdb` so the security-critical
//! auth format is untouched. Text so an operator can read/edit it.

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};

/// One tenant's limits: `(max_vectors, max_disk_bytes)`, `None` = unlimited.
pub type Limits = (Option<u64>, Option<u64>);

/// On-disk map of tenant id to limits. Writes are atomic (temp + rename).
#[derive(Debug, Default)]
pub struct LimitsStore {
    path: PathBuf,
    by_tenant: HashMap<[u8; 16], Limits>,
}

impl LimitsStore {
    /// Open the store at `path`. A missing file is an empty store.
    ///
    /// # Errors
    ///
    /// Returns an IO error on read failure or a malformed line.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut by_tenant = HashMap::new();
        if let Ok(text) = std::fs::read_to_string(&path) {
            for line in text.lines().filter(|l| !l.trim().is_empty()) {
                let mut f = line.split_whitespace();
                let (Some(t), Some(mv), Some(md), None) = (f.next(), f.next(), f.next(), f.next())
                else {
                    return Err(bad("malformed quota line"));
                };
                let tid = t
                    .parse::<u128>()
                    .map_err(|_| bad("bad tenant id"))?
                    .to_le_bytes();
                by_tenant.insert(tid, (parse_opt(mv)?, parse_opt(md)?));
            }
        }
        Ok(Self { path, by_tenant })
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
        let body: String = self
            .by_tenant
            .iter()
            .map(|(t, (mv, md))| {
                format!("{} {} {}\n", u128::from_le_bytes(*t), opt(*mv), opt(*md))
            })
            .collect();
        let tmp = self.path.with_extension("quotas.tmp");
        std::fs::write(&tmp, body)?;
        std::fs::rename(&tmp, &self.path)
    }
}

fn bad(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

fn opt(v: Option<u64>) -> String {
    v.map_or_else(|| "*".to_string(), |x| x.to_string())
}

fn parse_opt(s: &str) -> io::Result<Option<u64>> {
    if s == "*" {
        return Ok(None);
    }
    s.parse::<u64>().map(Some).map_err(|_| bad("bad limit"))
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
