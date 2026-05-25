//! Durable user→tenant mapping.
//!
//! `AuthStore` keeps an in-memory `HashMap<String, AuthRecord>` and
//! persists by writing a full snapshot to `auth.kdb.tmp` and renaming
//! atomically over `auth.kdb`. Snapshot-on-write keeps the format
//! trivial and recovery boils down to "open, parse, done". This is
//! deliberate: the user table is small (tens to thousands of entries),
//! so paying for an append-only log + compactor here would be over-design.
//!
//! Format:
//!
//! ```text
//! header: [magic 8B "SKEGAUTH"][version u32 LE][n_records u32 LE]
//! body:   per record: [crc32c 4B][tenant 16B][user_len u32][hash_len u32][user][hash]
//! ```
//!
//! `crc32c` is computed over `tenant || user_len || hash_len || user || hash`.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use thiserror::Error;
use tracing::warn;

use crate::auth::password::{PasswordError, PasswordHash, verify_password};
use crate::id::TenantId;

const MAGIC: &[u8; 8] = b"SKEGAUTH";
const VERSION: u32 = 1;

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("bad magic: file is not an auth.kdb")]
    BadMagic,
    #[error("unsupported version: {0}")]
    BadVersion(u32),
    #[error("truncated record at offset {0}")]
    Truncated(usize),
    #[error("crc mismatch at record {0}")]
    BadCrc(usize),
    #[error("user already exists: {0}")]
    DuplicateUser(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthRecord {
    pub tenant: TenantId,
    pub hash: PasswordHash,
}

#[derive(Debug)]
pub struct AuthStore {
    path: PathBuf,
    by_user: HashMap<String, AuthRecord>,
}

impl AuthStore {
    /// Open or create a store at `path`. Missing file is treated as an
    /// empty store; callers must call `save` to persist.
    ///
    /// # Errors
    ///
    /// Returns `StoreError::Io` on read failure, or one of the format
    /// errors on a corrupt file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let path = path.as_ref().to_path_buf();
        let mut store = Self {
            path: path.clone(),
            by_user: HashMap::new(),
        };
        if path.exists() {
            let bytes = std::fs::read(&path)?;
            store.parse_into(&bytes)?;
        }
        Ok(store)
    }

    fn parse_into(&mut self, bytes: &[u8]) -> Result<(), StoreError> {
        if bytes.len() < 16 || &bytes[0..8] != MAGIC {
            return Err(StoreError::BadMagic);
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if version != VERSION {
            return Err(StoreError::BadVersion(version));
        }
        let n = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let mut off = 16usize;
        for idx in 0..n {
            // Each record: 4B crc + 16B tenant + 4B user_len + 4B hash_len + user + hash
            if off + 4 + 16 + 4 + 4 > bytes.len() {
                return Err(StoreError::Truncated(off));
            }
            let crc_stored = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            off += 4;
            let body_start = off;

            let mut tid = [0u8; 16];
            tid.copy_from_slice(&bytes[off..off + 16]);
            off += 16;
            let user_len = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
            off += 4;
            let hash_len = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
            off += 4;
            if off + user_len + hash_len > bytes.len() {
                return Err(StoreError::Truncated(off));
            }
            let user = std::str::from_utf8(&bytes[off..off + user_len])
                .map_err(|_| StoreError::Truncated(off))?
                .to_string();
            off += user_len;
            let hash = std::str::from_utf8(&bytes[off..off + hash_len])
                .map_err(|_| StoreError::Truncated(off))?
                .to_string();
            off += hash_len;

            let crc_calc = crc32c::crc32c(&bytes[body_start..off]);
            if crc_calc != crc_stored {
                return Err(StoreError::BadCrc(idx));
            }

            self.by_user.insert(
                user,
                AuthRecord {
                    tenant: TenantId::from_bytes(tid),
                    hash: PasswordHash(hash),
                },
            );
        }
        Ok(())
    }

    /// Add or update a user. Returns the prior record if present.
    pub fn upsert(
        &mut self,
        user: impl Into<String>,
        tenant: TenantId,
        hash: PasswordHash,
    ) -> Option<AuthRecord> {
        self.by_user
            .insert(user.into(), AuthRecord { tenant, hash })
    }

    /// Insert a user, refusing to overwrite an existing one.
    ///
    /// # Errors
    ///
    /// Returns `StoreError::DuplicateUser` if `user` is already present.
    pub fn insert(
        &mut self,
        user: impl Into<String>,
        tenant: TenantId,
        hash: PasswordHash,
    ) -> Result<(), StoreError> {
        let user = user.into();
        if self.by_user.contains_key(&user) {
            return Err(StoreError::DuplicateUser(user));
        }
        self.by_user.insert(user, AuthRecord { tenant, hash });
        Ok(())
    }

    /// Drop a user.
    pub fn remove(&mut self, user: &str) -> Option<AuthRecord> {
        self.by_user.remove(user)
    }

    #[must_use]
    pub fn get(&self, user: &str) -> Option<&AuthRecord> {
        self.by_user.get(user)
    }

    /// True if any record is bound to `tenant`. Linear in the user count;
    /// the auth store is small (tens to thousands of entries) so the cost
    /// is irrelevant.
    #[must_use]
    pub fn has_tenant(&self, tenant: TenantId) -> bool {
        self.by_user.values().any(|r| r.tenant == tenant)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.by_user.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_user.is_empty()
    }

    /// Verify a username + password pair, returning the bound tenant.
    /// Constant-time when the user exists; when the user is missing we
    /// still pay the verify cost against a throwaway hash to avoid a
    /// user-enumeration side channel.
    ///
    /// # Errors
    ///
    /// Returns `PasswordError::VerifyFailed` for any mismatch (wrong
    /// password, unknown user, malformed stored hash) so callers cannot
    /// distinguish these cases from each other.
    pub fn verify_login(
        &self,
        user: &str,
        password: &[u8],
        decoy: &PasswordHash,
    ) -> Result<TenantId, PasswordError> {
        if let Some(rec) = self.by_user.get(user) {
            verify_password(password, &rec.hash).map(|()| rec.tenant)
        } else {
            // Pay the cost anyway against a known-good hash, then return
            // a generic failure. Without this the timing of an unknown
            // user is shorter than a wrong password.
            let _ = verify_password(password, decoy);
            Err(PasswordError::VerifyFailed)
        }
    }

    /// Persist a full snapshot atomically. Uses tmp + rename. The
    /// caller is responsible for any `F_FULLFSYNC` it wants on the
    /// containing directory.
    ///
    /// # Errors
    ///
    /// Returns `StoreError::Io` if any filesystem operation fails.
    pub fn save(&self) -> Result<(), StoreError> {
        let bytes = self.encode();
        let tmp = self.path.with_extension("kdb.tmp");
        {
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&tmp)?;
            f.write_all(&bytes)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, &self.path)?;
        Ok(())
    }

    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(16 + self.by_user.len() * 64);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        let n = u32::try_from(self.by_user.len()).unwrap_or_else(|_| {
            warn!(
                n = self.by_user.len(),
                "auth store size truncated for serialization"
            );
            u32::MAX
        });
        out.extend_from_slice(&n.to_le_bytes());

        for (user, rec) in &self.by_user {
            let user_bytes = user.as_bytes();
            let hash_bytes = rec.hash.0.as_bytes();
            // Skip records whose lengths overflow u32 to keep the file
            // internally consistent. Application policy keeps them well
            // under 4 GiB so this should never fire.
            let Ok(user_len) = u32::try_from(user_bytes.len()) else {
                warn!(
                    user,
                    len = user_bytes.len(),
                    "username > u32::MAX, skipping"
                );
                continue;
            };
            let Ok(hash_len) = u32::try_from(hash_bytes.len()) else {
                warn!(user, len = hash_bytes.len(), "hash > u32::MAX, skipping");
                continue;
            };
            let mut body = Vec::with_capacity(16 + 8 + user_bytes.len() + hash_bytes.len());
            body.extend_from_slice(rec.tenant.as_bytes());
            body.extend_from_slice(&user_len.to_le_bytes());
            body.extend_from_slice(&hash_len.to_le_bytes());
            body.extend_from_slice(user_bytes);
            body.extend_from_slice(hash_bytes);
            let crc = crc32c::crc32c(&body);
            out.extend_from_slice(&crc.to_le_bytes());
            out.extend_from_slice(&body);
        }
        out
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::password::{hash_password_with, test_cost};
    use tempfile::tempdir;

    fn mk_hash(p: &[u8]) -> PasswordHash {
        hash_password_with(p, test_cost()).unwrap()
    }

    #[test]
    fn open_missing_file_yields_empty_store() {
        let d = tempdir().unwrap();
        let s = AuthStore::open(d.path().join("auth.kdb")).unwrap();
        assert!(s.is_empty());
    }

    #[test]
    fn upsert_save_reopen_roundtrip() {
        let d = tempdir().unwrap();
        let path = d.path().join("auth.kdb");
        let alice = TenantId::from_name("alice");
        {
            let mut s = AuthStore::open(&path).unwrap();
            s.upsert("alice", alice, mk_hash(b"hunter2"));
            s.save().unwrap();
        }
        let s2 = AuthStore::open(&path).unwrap();
        let rec = s2.get("alice").unwrap();
        assert_eq!(rec.tenant, alice);
    }

    #[test]
    fn insert_rejects_duplicate() {
        let mut s = AuthStore::open("/tmp/skeg_auth_test_dup.kdb").unwrap_or_else(|_| {
            // path may have content from a previous run; tolerate
            AuthStore {
                path: "/tmp/skeg_auth_test_dup.kdb".into(),
                by_user: HashMap::new(),
            }
        });
        let alice = TenantId::from_name("alice");
        s.insert("alice", alice, mk_hash(b"x")).unwrap();
        let e = s.insert("alice", alice, mk_hash(b"x")).unwrap_err();
        matches!(e, StoreError::DuplicateUser(_));
    }

    #[test]
    fn verify_login_succeeds_with_right_password() {
        let alice = TenantId::from_name("alice");
        let mut s = AuthStore {
            path: PathBuf::from("/tmp/skeg_auth_unused.kdb"),
            by_user: HashMap::new(),
        };
        s.upsert("alice", alice, mk_hash(b"hunter2"));
        let decoy = mk_hash(b"decoy-not-used");
        let t = s.verify_login("alice", b"hunter2", &decoy).unwrap();
        assert_eq!(t, alice);
    }

    #[test]
    fn verify_login_rejects_wrong_password() {
        let alice = TenantId::from_name("alice");
        let mut s = AuthStore {
            path: PathBuf::from("/tmp/skeg_auth_unused.kdb"),
            by_user: HashMap::new(),
        };
        s.upsert("alice", alice, mk_hash(b"hunter2"));
        let decoy = mk_hash(b"decoy-not-used");
        let e = s.verify_login("alice", b"wrong", &decoy).unwrap_err();
        assert!(matches!(e, PasswordError::VerifyFailed));
    }

    #[test]
    fn verify_login_rejects_unknown_user() {
        let s = AuthStore {
            path: PathBuf::from("/tmp/skeg_auth_unused.kdb"),
            by_user: HashMap::new(),
        };
        let decoy = mk_hash(b"decoy-not-used");
        let e = s.verify_login("ghost", b"any", &decoy).unwrap_err();
        assert!(matches!(e, PasswordError::VerifyFailed));
    }

    #[test]
    fn corrupted_crc_rejected_on_open() {
        let d = tempdir().unwrap();
        let path = d.path().join("auth.kdb");
        {
            let mut s = AuthStore::open(&path).unwrap();
            s.upsert("alice", TenantId::from_name("alice"), mk_hash(b"x"));
            s.save().unwrap();
        }
        // Flip a byte inside the record body (after header) to corrupt CRC.
        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        std::fs::write(&path, &bytes).unwrap();
        let e = AuthStore::open(&path).unwrap_err();
        match e {
            StoreError::BadCrc(_) | StoreError::Truncated(_) => {}
            other => panic!("expected BadCrc or Truncated, got {other:?}"),
        }
    }

    #[test]
    fn bad_magic_rejected() {
        let d = tempdir().unwrap();
        let path = d.path().join("auth.kdb");
        std::fs::write(&path, b"NOTSKEGA\x01\x00\x00\x00\x00\x00\x00\x00").unwrap();
        let e = AuthStore::open(&path).unwrap_err();
        assert!(matches!(e, StoreError::BadMagic));
    }
}
