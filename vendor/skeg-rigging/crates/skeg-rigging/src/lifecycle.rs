//! Tenant lifecycle: discovery, snapshot, destroy.
//!
//! M3 roadmap item. These two traits are the minimum a programmatic
//! orchestrator needs in order to manage tenants of mixed engines
//! without knowing each engine's concrete type:
//!
//! - [`TenantInfo`] - read-side discovery. What is this tenant, what
//!   can it do?
//! - [`TenantLifecycle`] - write-side management. Snapshot to a path
//!   for backup; destroy when no longer needed.
//!
//! Factory methods (`create` / `restore`) deliberately stay on the
//! adapter's concrete type because their argument shape varies by
//! engine - a graph engine's "create" takes graph parameters, a
//! vector engine's takes embedding dim, etc. Orchestrators that
//! create tenants do so via an engine-specific constructor; the
//! traits below cover everything that comes *after*.

use std::path::Path;

use crate::OpenError;
use crate::ids::TenantId;

/// Engine + plugin capability namespace.
///
/// Conventions:
///
/// - `vector.kv` - skeg-core baseline (KV + vector).
/// - `vector.quantized` - skeg-core with int8 or PQ tier active.
/// - `graph.traverse` - hypothetical graph engine.
/// - `temporal.windowed` - hypothetical temporal engine.
/// - `hansa.member` - tenant participates in a hansa.
/// - `hansa.membrane` - tenant can serve membrane queries to peers.
///
/// The string is the source of truth; comparison is byte-for-byte.
/// Plugins declare their own `CapabilityId` constants.
///
/// Stability: Stable.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Ord, PartialOrd)]
pub struct CapabilityId(pub &'static str);

impl CapabilityId {
    /// The underlying string slice.
    pub fn as_str(&self) -> &'static str {
        self.0
    }
}

impl std::fmt::Display for CapabilityId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

/// Canonical capability for the skeg-core KV + vector engine.
pub const CAP_VECTOR_KV: CapabilityId = CapabilityId("vector.kv");
/// Canonical capability for quantised vector tiers (int8 / PQ).
pub const CAP_VECTOR_QUANTIZED: CapabilityId = CapabilityId("vector.quantized");

/// Discovery surface for a tenant. Engine-neutral.
///
/// Stability: Stable.
pub trait TenantInfo {
    /// Stable id of this tenant.
    fn tenant_id(&self) -> TenantId;
    /// Embedding dimension. Mirrors `IterVectors::embedding_dim` so
    /// `TenantInfo` is usable standalone.
    fn embedding_dim(&self) -> u32;
    /// Records currently live in this tenant.
    fn record_count(&self) -> u64;
    /// Capabilities this tenant exposes. Used by orchestrators to
    /// decide what queries / operations are valid. See
    /// [`CapabilityId`] for the namespace convention.
    fn capabilities(&self) -> Vec<CapabilityId>;
}

/// Management surface for a tenant. Object-safe.
///
/// Stability: Stable. Adapters and engines should implement this on
/// the same type that implements [`crate::ReadOnlyView`] so an
/// orchestrator can hold one `Box<dyn>` and dispatch both read and
/// management ops.
pub trait TenantLifecycle: Send + Sync {
    /// Snapshot the tenant's persistent state into `dest`. The
    /// destination directory is created if missing. The contents are
    /// engine-defined; an orchestrator can later pass `dest` to a
    /// matching engine's restore method.
    ///
    /// Implementations must be safe to call while the tenant has open
    /// readers - they typically copy underlying files atomically or
    /// rely on the engine's snapshot mechanism.
    fn snapshot(&self, dest: &Path) -> Result<(), OpenError>;

    /// Destroy the tenant. Consumes the boxed handle and removes
    /// on-disk state. After this returns the tenant directory is
    /// gone; opening it again is `OpenError::NotFound`.
    fn destroy(self: Box<Self>) -> Result<(), OpenError>;
}
