#![deny(unsafe_code)]
#![warn(missing_docs)]

//! `skeg-multi-tenant` - multi-tenant orchestration layer on top of
//! `skeg-tenant`.
//!
//! Sister crate to `skeg-tenant`. Combines the per-tenant primitives
//! (auth, quota tracker, namespaces) with on-disk tenant directories
//! into a single orchestrator surface - [`MultiTenantRoot`] - so a
//! caller can:
//!
//! - Open / create a tenant by [`SkegTenantId`].
//! - Open a tenant by verifying a bearer token (no manual id lookup).
//! - List the tenants currently materialised on disk.
//! - Get a [`TenantHandle`] that charges the shared
//!   [`skeg_tenant::quota::QuotaTracker`] before every write.
//!
//! ## License
//!
//! Apache-2.0, matching the rest of the engine.
//!
//! ## Where the rigging traits live
//!
//! [`TenantHandle`] implements [`skeg_rigging::TenantQuota`],
//! [`skeg_rigging::TenantStats`] (via the wrapped adapter), and the
//! rest of the rigging trait set. The intent is that hansa (or any
//! other rigging consumer) can take a `TenantHandle` and pass it
//! anywhere a rigging tenant is expected. The mapping between
//! `skeg-tenant`'s atomic quota counters and rigging's
//! [`skeg_rigging::Quota`] / [`skeg_rigging::Usage`] types lives in
//! the `handle` module.
//!
//! ## Scope in v0.1
//!
//! - On-disk root with one subdir per tenant.
//! - Open / create / read-only open.
//! - Token-verified open via `skeg_tenant::auth::TokenStore`.
//! - Quota enforcement via `skeg_tenant::quota::QuotaTracker`.
//! - Lifecycle (snapshot, destroy) inherited from the wrapped adapter.
//!
//! Out of scope: cross-machine federation (lives in hansa), event
//! multiplexing.
//!
//! Opt-in: live `skeg-server-tenant` attach is available behind the
//! `live-attach` feature. See [`LiveAttachRoot`] when enabled.

use skeg_rigging::TenantId as RiggingTenantId;

pub use skeg_tenant::TenantId as SkegTenantId;
pub use skeg_tenant::TenantId;

mod error;
mod handle;
#[cfg(feature = "live-attach")]
mod live;
mod root;

pub use error::MultiTenantError;
pub use handle::TenantHandle;
#[cfg(feature = "live-attach")]
pub use live::{LiveAttachError, LiveAttachRoot};
pub use root::MultiTenantRoot;

/// Re-export of the concrete adapter tenant the handles wrap.
pub use skeg_rigging_skeg::Tenant as RiggingTenant;

/// Re-exports of the skeg-tenant primitives that consumers wire up to
/// the root. Pulling these through `skeg-multi-tenant` means
/// orchestrator authors don't have to take a direct dep on
/// `skeg-tenant`.
pub mod tenant_primitives {
    pub use skeg_tenant::auth::{TokenError, TokenStore};
    pub use skeg_tenant::quota::{QuotaError as SkegQuotaError, QuotaTracker, TenantQuota};
}

/// Convert a `skeg_tenant::TenantId` to a `skeg_rigging::TenantId`.
/// Both are 16-byte newtypes; we keep them as distinct types because
/// their semantic role differs (auth scoping vs memory unit scoping)
/// but the wire form matches.
pub fn rigging_tenant_id(id: SkegTenantId) -> RiggingTenantId {
    RiggingTenantId::from_bytes(*id.as_bytes())
}

/// The inverse of [`rigging_tenant_id`].
pub fn skeg_tenant_id_from_rigging(id: RiggingTenantId) -> SkegTenantId {
    SkegTenantId::from_bytes(*id.as_bytes())
}
