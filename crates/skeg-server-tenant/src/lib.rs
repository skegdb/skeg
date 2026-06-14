//! Multi-tenant wrapper for `skeg-server`.
//!
//! Implements [`skeg_server::TenantBackend`] on top of the `skeg-tenant`
//! primitives (auth store, tenant ids, argon2 password hashing).

#![deny(unsafe_code)]

use std::path::Path;
use std::sync::Arc;

use parking_lot::RwLock;
use skeg_server::{AnonymousPolicy, TenantBackend, TenantId};
use skeg_tenant::auth::{Argon2Params, PasswordHash, hash_password_with};
use skeg_tenant::{AuthStore, TenantId as TenantTenantId};

/// `TenantBackend` implementation backed by an on-disk `auth.kdb`.
pub struct AuthStoreBackend {
    auth: Arc<RwLock<AuthStore>>,
    decoy: PasswordHash,
    strict: bool,
}

impl AuthStoreBackend {
    /// Open `auth.kdb` at `path`. `strict = true` makes the server
    /// reject anonymous HELLO 3 (no AUTH); `false` keeps the lenient
    /// behaviour (anonymous maps to `TenantId::ZERO`).
    ///
    /// # Errors
    ///
    /// Returns the underlying `skeg-tenant` error on store open or
    /// decoy hashing.
    pub fn open(path: impl AsRef<Path>, strict: bool) -> Result<Arc<Self>, skeg_tenant::AuthError> {
        let store = AuthStore::open(path)?;
        // Precomputed decoy hash used when verifying an unknown user,
        // so the timing of "wrong password" and "unknown user" is the
        // same.
        let decoy = hash_password_with(b"skeg-decoy", Argon2Params::default())
            .map_err(skeg_tenant::AuthError::from)?;
        Ok(Arc::new(Self {
            auth: Arc::new(RwLock::new(store)),
            decoy,
            strict,
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
}
