//! Read-only access to a tenant opened from a path.

use std::path::Path;

use crate::error::OpenError;
use crate::filter::QueryFiltered;
use crate::ids::TenantId;
use crate::iter::IterVectors;

/// Read-only view onto a tenant on disk.
///
/// Multiple processes may hold `ReadOnlyView` handles to the same
/// tenant concurrently. Writers in the owning process keep working
/// subject to the engine's atomic-rename update strategy; readers see
/// either the pre-rename or post-rename snapshot, never a partial state.
///
/// Stability: **Stable** (promoted from Provisional in v0.1.1).
/// Cross-process snapshot semantics validated by hansa's
/// `cross_process` and `concurrent_populate` integration tests against
/// the reference `skeg-rigging-skeg` adapter.
pub trait ReadOnlyView: IterVectors + QueryFiltered + Send + Sync {
    /// Stable identifier of the tenant behind this view.
    fn tenant_id(&self) -> TenantId;

    /// Explicitly release file handles. Equivalent to dropping the
    /// boxed view, but returns errors that would otherwise be silently
    /// swallowed by `Drop`.
    fn close(self: Box<Self>) -> Result<(), OpenError>;
}

/// Open a tenant at `path` in read-only mode.
///
/// This is a free function - there is no `self` to dispatch on at the
/// call site, only a path. The concrete reader implementation lives in
/// adapter crates (see `skeg-rigging-skeg`); rigging itself does not
/// know how to interpret an engine-specific file tree.
///
/// In v0.1 this function is a placeholder that always returns
/// `OpenError::NotFound`. The implementation is provided by adapter
/// crates that wrap it in their own constructor (see
/// `skeg_rigging_skeg::Tenant::open_readonly_at`). A future v0.2 may
/// add a registry of adapter handlers selected by filesystem markers.
pub fn open_readonly(_path: &Path) -> Result<Box<dyn ReadOnlyView>, OpenError> {
    Err(OpenError::NotFound)
}
