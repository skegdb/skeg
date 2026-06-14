//! Public extension points for an external multi-tenant layer.
//!
//! `skeg-server` ships single-tenant by default. A separate crate
//! (typically `skeg-server-tenant` in the BUSL-1.1 tenant repo) can
//! install an implementation of [`TenantBackend`] via
//! [`Server::with_tenant_backend`](crate::Server::with_tenant_backend),
//! at which point the RESP3 handler honours `HELLO 3 AUTH` and scopes
//! KV / vector ops per tenant.
//!
//! The interface lives here so the public engine has no compile-time
//! dependency on any specific tenant implementation. The trait is
//! object-safe; consumers pass `Arc<dyn TenantBackend>`.

/// Fixed-width tenant identifier. 16 bytes is enough to embed any
/// 128-bit hash (we use `xxh3_128` of the tenant name in the standard
/// implementation, but the trait does not require it). The all-zero
/// id is reserved as the anonymous / single-tenant sentinel.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TenantId(pub [u8; 16]);

impl TenantId {
    /// The anonymous / single-tenant sentinel.
    pub const ZERO: Self = Self([0; 16]);
    /// Byte length of the identifier.
    pub const LEN: usize = 16;

    /// True for the `ZERO` sentinel.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.0 == [0; 16]
    }

    /// Raw bytes view.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    /// Construct from raw bytes.
    #[must_use]
    pub fn from_bytes(b: [u8; 16]) -> Self {
        Self(b)
    }
}

impl std::fmt::Display for TenantId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for b in self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

/// What to do when a RESP3 client sends `HELLO 3` without AUTH.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum AnonymousPolicy {
    /// Anonymous connections are accepted and resolved to
    /// [`TenantId::ZERO`]. Single-tenant deployments behave this way.
    #[default]
    Lenient,
    /// Anonymous connections are rejected with `-NOAUTH`.
    Strict,
}

/// External hook for the multi-tenant layer.
///
/// Implementations must be `Send + Sync` so they can be shared across
/// per-connection async tasks. The trait is object-safe.
pub trait TenantBackend: Send + Sync {
    /// Verify a `(user, password)` pair. `Some(id)` on success,
    /// `None` on any failure (wrong password, unknown user, malformed
    /// hash). Implementations are expected to be constant-time wrt
    /// user existence, to avoid leaking valid usernames via timing.
    fn verify_login(&self, user: &str, password: &[u8]) -> Option<TenantId>;

    /// True if any record in the backing store is bound to `id`. Used
    /// by the anonymous-prefix forgery defense in the RESP3 handler:
    /// a `TenantId::ZERO` client cannot forge a key whose first 16
    /// bytes match a real tenant id.
    fn has_tenant(&self, id: TenantId) -> bool;

    /// Strict or lenient handling of `HELLO 3` without AUTH.
    fn anonymous_policy(&self) -> AnonymousPolicy {
        AnonymousPolicy::Lenient
    }

    /// Hard resource limits for `id`. Default is unlimited, so existing
    /// backends and single-tenant deployments are unaffected. The server
    /// enforces these at admission (e.g. `max_vectors` on VSET).
    fn limits(&self, id: TenantId) -> crate::quota::TenantLimits {
        let _ = id;
        crate::quota::TenantLimits::default()
    }
}
