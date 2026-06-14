//! Multi-tenant wrapper for `skeg-server`.
//!
//! Implements [`skeg_server::TenantBackend`] on top of the `skeg-tenant`
//! primitives (auth store, tenant ids, argon2 password hashing), plus a
//! persisted per-tenant quota store an admin writes via `SKEG.QUOTA.SET`.

#![deny(unsafe_code)]

mod limits;

use std::error::Error;
use std::path::Path;
use std::sync::Arc;

use parking_lot::RwLock;
use skeg_server::{AnonymousPolicy, QuotaAdminError, TenantBackend, TenantId, TenantLimits};
use skeg_tenant::auth::{Argon2Params, PasswordHash, hash_password_with};
use skeg_tenant::{AuthStore, TenantId as TenantTenantId};

use crate::limits::LimitsStore;

/// `TenantBackend` implementation backed by an on-disk `auth.kdb` (identity)
/// and a sidecar quota store (per-tenant limits).
pub struct AuthStoreBackend {
    auth: Arc<RwLock<AuthStore>>,
    limits: RwLock<LimitsStore>,
    decoy: PasswordHash,
    strict: bool,
    /// Tenant allowed to run admin commands, if any (`--admin-tenant`).
    admin: Option<TenantTenantId>,
}

impl AuthStoreBackend {
    /// Open `auth.kdb` at `path` plus its `<path>.quotas` sidecar.
    ///
    /// `strict = true` rejects anonymous `HELLO 3`. `admin_tenant` names the
    /// tenant permitted to run `SKEG.QUOTA.SET/GET`; `None` means no admin.
    ///
    /// # Errors
    ///
    /// Returns the underlying store / hashing error.
    pub fn open(
        path: impl AsRef<Path>,
        strict: bool,
        admin_tenant: Option<&str>,
    ) -> Result<Arc<Self>, Box<dyn Error>> {
        let path = path.as_ref();
        let store = AuthStore::open(path)?;
        let quotas_path: std::path::PathBuf = format!("{}.quotas", path.to_string_lossy()).into();
        let limits = LimitsStore::open(quotas_path)?;
        // Precomputed decoy hash used when verifying an unknown user, so the
        // timing of "wrong password" and "unknown user" is the same.
        let decoy = hash_password_with(b"skeg-decoy", Argon2Params::default())?;
        Ok(Arc::new(Self {
            auth: Arc::new(RwLock::new(store)),
            limits: RwLock::new(limits),
            decoy,
            strict,
            admin: admin_tenant.map(TenantTenantId::from_name),
        }))
    }
}

fn tid_to_engine(t: TenantTenantId) -> TenantId {
    TenantId::from_bytes(*t.as_bytes())
}

fn tid_from_engine(t: TenantId) -> TenantTenantId {
    TenantTenantId::from_bytes(*t.as_bytes())
}

impl TenantBackend for AuthStoreBackend {
    fn verify_login(&self, user: &str, password: &[u8]) -> Option<TenantId> {
        self.auth
            .read()
            .verify_login(user, password, &self.decoy)
            .ok()
            .map(tid_to_engine)
    }

    fn has_tenant(&self, id: TenantId) -> bool {
        self.auth.read().has_tenant(tid_from_engine(id))
    }

    fn anonymous_policy(&self) -> AnonymousPolicy {
        if self.strict {
            AnonymousPolicy::Strict
        } else {
            AnonymousPolicy::Lenient
        }
    }

    fn limits(&self, id: TenantId) -> TenantLimits {
        let (max_vectors, max_disk_bytes) = self.limits.read().get(*tid_from_engine(id).as_bytes());
        TenantLimits {
            max_vectors,
            max_disk_bytes,
        }
    }

    fn is_admin(&self, id: TenantId) -> bool {
        !id.is_zero() && self.admin == Some(tid_from_engine(id))
    }

    fn resolve_tenant(&self, name: &str) -> Option<TenantId> {
        let t = TenantTenantId::from_name(name);
        self.auth.read().has_tenant(t).then(|| tid_to_engine(t))
    }

    fn set_limits(&self, id: TenantId, limits: TenantLimits) -> Result<(), QuotaAdminError> {
        self.limits
            .write()
            .set(
                *tid_from_engine(id).as_bytes(),
                (limits.max_vectors, limits.max_disk_bytes),
            )
            .map_err(|_| QuotaAdminError::Unsupported)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Write an `auth.kdb` binding each `(user, tenant_name)`.
    fn write_auth(dir: &Path, users: &[(&str, &str)]) -> std::path::PathBuf {
        let path = dir.join("auth.kdb");
        let mut store = AuthStore::open(&path).unwrap();
        let hash = hash_password_with(b"pw", Argon2Params::default()).unwrap();
        for (user, tenant) in users {
            // upsert returns the prior record (None for a new user); ignore it.
            store.upsert(*user, TenantTenantId::from_name(tenant), hash.clone());
        }
        store.save().unwrap();
        path
    }

    #[test]
    fn admin_sets_and_persists_tenant_limits() {
        let dir = TempDir::new().unwrap();
        let path = write_auth(dir.path(), &[("admin", "admin"), ("u", "acme")]);
        let be = AuthStoreBackend::open(&path, false, Some("admin")).unwrap();

        let admin_id = tid_to_engine(TenantTenantId::from_name("admin"));
        let acme_id = tid_to_engine(TenantTenantId::from_name("acme"));

        // admin gating
        assert!(be.is_admin(admin_id));
        assert!(!be.is_admin(acme_id));
        assert!(!be.is_admin(TenantId::ZERO));

        // name resolution
        assert_eq!(be.resolve_tenant("acme"), Some(acme_id));
        assert_eq!(be.resolve_tenant("nobody"), None);

        // default unlimited, then a set the engine can read back
        assert_eq!(be.limits(acme_id), TenantLimits::default());
        be.set_limits(
            acme_id,
            TenantLimits {
                max_vectors: Some(1000),
                max_disk_bytes: Some(1 << 20),
            },
        )
        .unwrap();
        assert_eq!(be.limits(acme_id).max_vectors, Some(1000));
        assert_eq!(be.limits(acme_id).max_disk_bytes, Some(1 << 20));

        // persisted: a fresh backend over the same files sees it
        drop(be);
        let be2 = AuthStoreBackend::open(&path, false, Some("admin")).unwrap();
        assert_eq!(be2.limits(acme_id).max_vectors, Some(1000));
    }

    #[test]
    fn no_admin_configured_means_no_admin() {
        let dir = TempDir::new().unwrap();
        let path = write_auth(dir.path(), &[("u", "acme")]);
        let be = AuthStoreBackend::open(&path, false, None).unwrap();
        let acme_id = tid_to_engine(TenantTenantId::from_name("acme"));
        assert!(!be.is_admin(acme_id));
        assert!(be.set_limits(acme_id, TenantLimits::default()).is_ok());
    }
}
