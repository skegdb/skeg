//! Errors surfaced by the multi-tenant orchestrator.

use std::path::PathBuf;

use skeg_tenant::auth::TokenError;

/// Errors from [`crate::MultiTenantRoot`] and [`crate::TenantHandle`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum MultiTenantError {
    /// Underlying I/O failure.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Tenant directory missing.
    #[error("tenant not found at {0}")]
    TenantNotFound(PathBuf),

    /// Caller requested token-based open but the root has no
    /// [`skeg_tenant::auth::TokenStore`] configured.
    #[error("token verification unavailable: no TokenStore configured")]
    NoTokenStore,

    /// Caller asked for quota-scoped operations but the root has no
    /// [`skeg_tenant::quota::QuotaTracker`] configured.
    #[error("quota enforcement unavailable: no QuotaTracker configured")]
    NoQuotaTracker,

    /// Token verification failed.
    #[error("token verification failed: {0}")]
    TokenError(#[from] TokenError),

    /// Embedding dim mismatch when opening a pre-existing tenant.
    #[error("embedding dim mismatch: on-disk {on_disk}, requested {requested}")]
    DimMismatch {
        /// Dim from the on-disk sidecar.
        on_disk: u32,
        /// Dim the caller asked for.
        requested: u32,
    },

    /// Surfaced from the underlying `skeg-rigging-skeg::Tenant`.
    #[error("tenant adapter: {0}")]
    Tenant(#[from] skeg_rigging_skeg::TenantError),

    /// skeg-tenant's quota tracker rejected the write.
    #[error("quota exceeded: {0}")]
    Quota(#[from] skeg_tenant::quota::QuotaError),

    /// The rigging-side quota surface was used incorrectly.
    #[error("rigging quota error: {0}")]
    RiggingQuota(#[from] skeg_rigging::QuotaError),
}
