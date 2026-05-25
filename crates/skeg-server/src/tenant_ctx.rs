//! Server-side wiring around `skeg-tenant`.
//!
//! `TenantContext` is the bundle the server hands to each connection
//! task: an auth store (user→tenant lookup), a token store (HMAC bearer
//! tokens), a resolver chain (decides which tenant a request belongs
//! to), and a quota tracker. The struct is `Arc`-wrapped at the call
//! sites; `clone()` is cheap.
//!
//! When the server is configured without tenancy, `TenantContext` is
//! absent (`Option::None`) and every connection resolves to
//! `TenantId::ZERO`. This keeps the single-tenant path byte-identical
//! to the pre-tenancy code.

use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;

#[cfg(test)]
use skeg_tenant::auth::cheap_test_cost;
use skeg_tenant::auth::{Argon2Params, hash_password_with};
use skeg_tenant::{
    AuthBoundResolver, AuthStore, NullResolver, PerRequestResolver, QuotaTracker, ResolverChain,
    TenantResolver, TokenStore,
};
pub use skeg_tenant::{TenantId, scoped_vindex_name};

/// Top-level tenant configuration. Shared (Arc) across all connections.
pub struct TenantContext {
    pub auth: RwLock<AuthStore>,
    pub tokens: TokenStore,
    pub quotas: QuotaTracker,
    pub resolver: Box<dyn TenantResolver>,
    /// A precomputed "decoy" password hash. We verify against this
    /// when an unknown user logs in, so timing does not leak whether
    /// the user exists. Built once at startup.
    pub decoy: skeg_tenant::auth::PasswordHash,
}

impl TenantContext {
    /// Build a default context: in-memory `AuthStore` at `path`,
    /// 1h token TTL, `AuthBoundResolver` + `PerRequestResolver`
    /// (lenient, falls back to `ZERO`).
    ///
    /// # Errors
    ///
    /// Returns the underlying `skeg-tenant` error on store open or
    /// decoy hashing.
    pub fn open_lenient(
        auth_path: impl AsRef<std::path::Path>,
    ) -> Result<Arc<Self>, skeg_tenant::AuthError> {
        let store = AuthStore::open(auth_path)?;
        let tokens = TokenStore::fresh(Duration::from_secs(3600));
        let decoy = hash_password_with(b"skeg-decoy", Argon2Params::default())
            .map_err(skeg_tenant::AuthError::from)?;
        let resolver: Box<dyn TenantResolver> = Box::new(
            ResolverChain::new()
                .with(AuthBoundResolver)
                .with(PerRequestResolver),
        );
        Ok(Arc::new(Self {
            auth: RwLock::new(store),
            tokens,
            quotas: QuotaTracker::new(),
            resolver,
            decoy,
        }))
    }

    /// Strict variant: anonymous traffic is rejected (resolver chain
    /// without the `ZERO` fallback). Use this for public-facing
    /// multi-tenant servers.
    ///
    /// # Errors
    ///
    /// As `open_lenient`.
    pub fn open_strict(
        auth_path: impl AsRef<std::path::Path>,
    ) -> Result<Arc<Self>, skeg_tenant::AuthError> {
        let store = AuthStore::open(auth_path)?;
        let tokens = TokenStore::fresh(Duration::from_secs(3600));
        let decoy = hash_password_with(b"skeg-decoy", Argon2Params::default())
            .map_err(skeg_tenant::AuthError::from)?;
        let resolver: Box<dyn TenantResolver> = Box::new(
            ResolverChain::new()
                .with(AuthBoundResolver)
                .with(PerRequestResolver)
                .strict(),
        );
        Ok(Arc::new(Self {
            auth: RwLock::new(store),
            tokens,
            quotas: QuotaTracker::new(),
            resolver,
            decoy,
        }))
    }

    /// Variant used by tests: in-memory store, fast password params,
    /// shared key on the `TokenStore` for determinism.
    #[cfg(test)]
    #[must_use]
    pub fn for_tests() -> Arc<Self> {
        let store = AuthStore::open(std::env::temp_dir().join(format!(
            "skeg-tenant-ctx-{}-{}.kdb",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        )))
        .expect("open in-memory store");
        let tokens = TokenStore::from_key([0xAB; 32], Duration::from_secs(3600));
        let decoy = hash_password_with(b"decoy", cheap_test_cost()).unwrap();
        let resolver: Box<dyn TenantResolver> = Box::new(
            ResolverChain::new()
                .with(AuthBoundResolver)
                .with(PerRequestResolver),
        );
        Arc::new(Self {
            auth: RwLock::new(store),
            tokens,
            quotas: QuotaTracker::new(),
            resolver,
            decoy,
        })
    }

    /// Run the resolver chain on an externally-built context.
    ///
    /// # Errors
    ///
    /// Propagates the resolver's error.
    pub fn resolve(
        &self,
        ctx: &skeg_tenant::resolver::ResolveContext,
    ) -> Result<TenantId, skeg_tenant::ResolveError> {
        self.resolver.resolve(ctx)
    }

    /// Verify `user` / `pass` against the auth store, with a constant-time
    /// decoy hash path for unknown users.
    pub fn verify_login(
        &self,
        user: &str,
        pass: &[u8],
    ) -> Result<TenantId, skeg_tenant::AuthError> {
        self.auth
            .read()
            .verify_login(user, pass, &self.decoy)
            .map_err(Into::into)
    }

    /// Whether this auth store contains at least one user bound to
    /// `candidate`.
    #[must_use]
    pub fn has_tenant(&self, candidate: TenantId) -> bool {
        self.auth.read().has_tenant(candidate)
    }
}

/// A `NullResolver`-equivalent constant. Use this in places that need
/// a resolver value where tenancy is disabled.
#[must_use]
pub fn null_resolver() -> NullResolver {
    NullResolver
}

#[cfg(test)]
mod tests {
    use super::*;
    use skeg_tenant::auth::PasswordHash;
    use skeg_tenant::resolver::ResolveContext;

    #[test]
    fn open_lenient_resolves_anonymous_to_zero() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = TenantContext::open_lenient(dir.path().join("auth.kdb")).unwrap();
        let t = ctx.resolve(&ResolveContext::empty()).unwrap();
        assert_eq!(t, TenantId::ZERO);
    }

    #[test]
    fn open_strict_rejects_anonymous() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = TenantContext::open_strict(dir.path().join("auth.kdb")).unwrap();
        let e = ctx.resolve(&ResolveContext::empty()).unwrap_err();
        matches!(e, skeg_tenant::ResolveError::Missing);
    }

    #[test]
    fn auth_login_sets_tenant_via_chain() {
        let ctx = TenantContext::for_tests();
        let alice = TenantId::from_name("alice");
        let hash = PasswordHash(hash_password_with(b"hunter2", cheap_test_cost()).unwrap().0);
        ctx.auth.write().upsert("alice", alice, hash);

        let tid = ctx
            .auth
            .read()
            .verify_login("alice", b"hunter2", &ctx.decoy)
            .unwrap();
        let resolved = ctx
            .resolve(&ResolveContext::empty().with_auth(tid))
            .unwrap();
        assert_eq!(resolved, alice);
    }
}
