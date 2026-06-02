//! `TenantHandle`: a quota-scoped tenant managed by the multi-tenant root.
//!
//! The handle pairs an on-disk [`skeg_rigging_skeg::Tenant`] with a
//! shared [`skeg_tenant::quota::TenantQuota`] entry. Every write is
//! pre-charged against the tracker; admission failures roll the
//! counters back so the in-memory state stays consistent with the
//! tenant on disk.
//!
//! ## Quota mapping
//!
//! `skeg-tenant` tracks two counters:
//!
//! - `n_vectors` - every accepted record `+1`.
//! - `cache_bytes` - best effort estimate of payload + tag + embedding
//!   bytes per record.
//!
//! `skeg-rigging`'s [`Quota`] is dimensioned the same way; this
//! handle bridges the two:
//!
//! - [`Quota::max_records`] ↔ skeg-tenant `vectors_limit`
//! - [`Quota::max_bytes`]   ↔ skeg-tenant `cache_bytes_limit`
//!
//! `None` on either rigging dimension means unlimited; the tracker
//! uses `0` for the same meaning. The mapping is bi-directional so
//! [`TenantQuota::quota`] round-trips cleanly.
//!
//! ## What we charge
//!
//! Every successful insert charges both counters. Overwrites of the
//! same `record_id` count too - the handle treats the tracker as a
//! billing/admission gate rather than a live-row accounting mirror.
//! Callers that need strict live-row semantics maintain their own
//! row index and call [`TenantHandle::sub_usage_on_delete`].

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use skeg_rigging::{
    CapabilityId, EventFilter, EventStream, Filter, Hit, IterVectors, OpenError, QueryError,
    QueryFiltered, Quota, QuotaError as RiggingQuotaError, ReadOnlyView, RecordId, TenantEvents,
    TenantId as RiggingTenantId, TenantInfo, TenantLifecycle, TenantQuota, TenantStats, Usage,
};
use skeg_rigging_skeg::Tenant;
use skeg_tenant::TenantId as SkegTenantId;
use skeg_tenant::quota::TenantQuota as SkegTenantQuota;

use crate::error::MultiTenantError;

/// A tenant opened through the multi-tenant root, with its quota
/// counters wired up.
///
/// Holds the wrapped adapter plus an `Arc` to the skeg-tenant quota
/// entry. Cheap to share between threads via the inner `Arc`.
pub struct TenantHandle {
    inner: Tenant,
    skeg_quota: Arc<SkegTenantQuota>,
    skeg_tenant_id: SkegTenantId,
}

impl TenantHandle {
    pub(crate) fn new(
        inner: Tenant,
        skeg_quota: Arc<SkegTenantQuota>,
        skeg_tenant_id: SkegTenantId,
    ) -> Self {
        Self {
            inner,
            skeg_quota,
            skeg_tenant_id,
        }
    }

    /// Borrow the wrapped adapter tenant. Use this for read-side
    /// operations ([`skeg_rigging::QueryFiltered`], iteration, etc.) -
    /// the handle only intercepts the *write* path.
    pub fn inner(&self) -> &Tenant {
        &self.inner
    }

    /// The skeg-tenant id this handle was opened for.
    pub fn tenant_id(&self) -> SkegTenantId {
        self.skeg_tenant_id
    }

    /// Convenience: forward to the rigging-side tenant id.
    pub fn rigging_tenant_id(&self) -> RiggingTenantId {
        crate::rigging_tenant_id(self.skeg_tenant_id)
    }

    /// Insert a record, charging the quota first. If the tracker
    /// rejects the write the adapter is not touched. If the adapter
    /// rejects after the tracker accepted, both counters roll back.
    pub fn insert(
        &self,
        record_id: RecordId,
        embedding: Vec<f32>,
        shareable: bool,
        tags: Vec<String>,
        payload: Vec<u8>,
    ) -> Result<(), MultiTenantError> {
        let bytes_delta = std::mem::size_of_val(embedding.as_slice()) as u64
            + payload.len() as u64
            + tags.iter().map(|t| t.len() as u64).sum::<u64>();

        self.skeg_quota.try_add_vectors(self.skeg_tenant_id, 1)?;
        if let Err(e) = self
            .skeg_quota
            .try_add_cache_bytes(self.skeg_tenant_id, bytes_delta)
        {
            self.skeg_quota.sub_vectors(1);
            return Err(e.into());
        }
        if let Err(e) = self
            .inner
            .insert(record_id, embedding, shareable, tags, payload)
        {
            self.skeg_quota.sub_vectors(1);
            self.skeg_quota.sub_cache_bytes(bytes_delta);
            return Err(MultiTenantError::Tenant(e));
        }
        Ok(())
    }

    /// Reclaim quota counters on delete. The handle does not remember
    /// per-record sizes, so the caller passes them explicitly.
    pub fn sub_usage_on_delete(&self, vector_count: u64, byte_count: u64) {
        self.skeg_quota.sub_vectors(vector_count);
        self.skeg_quota.sub_cache_bytes(byte_count);
    }

    /// Delete a record. Refunds the vector counter automatically
    /// (the tracker doesn't know per-record byte sizes, so byte
    /// reclaim still requires [`Self::sub_usage_on_delete`]). Returns
    /// `true` if the row was present.
    pub fn delete(&self, record_id: RecordId) -> Result<bool, MultiTenantError> {
        let removed = self.inner.delete(record_id)?;
        if removed {
            self.skeg_quota.sub_vectors(1);
        }
        Ok(removed)
    }

    /// Persist the sidecar JSON. Forwards to the wrapped adapter.
    pub fn flush(&self) -> Result<(), MultiTenantError> {
        self.inner.flush().map_err(MultiTenantError::from)
    }

    /// Trigger a DiskVamana consolidation. Forwards to the wrapped
    /// adapter.
    pub fn consolidate(&self) -> Result<(), MultiTenantError> {
        self.inner.consolidate().map_err(MultiTenantError::from)
    }

    /// Snapshot this tenant to `dest`. Forwards to the adapter's
    /// `TenantLifecycle::snapshot`.
    pub fn snapshot(&self, dest: &Path) -> Result<(), OpenError> {
        TenantLifecycle::snapshot(&self.inner, dest)
    }

    /// Borrow the underlying skeg-tenant quota entry. Useful when a
    /// metrics exporter needs the raw atomics without going through
    /// the rigging trait.
    pub fn skeg_quota(&self) -> &Arc<SkegTenantQuota> {
        &self.skeg_quota
    }
}

impl TenantQuota for TenantHandle {
    fn set_quota(&self, quota: Quota) -> Result<(), RiggingQuotaError> {
        self.skeg_quota
            .set_vectors_limit(quota.max_records.unwrap_or(0));
        self.skeg_quota
            .set_cache_limit(quota.max_bytes.unwrap_or(0));
        Ok(())
    }

    fn quota(&self) -> Quota {
        Quota {
            max_records: nonzero(self.skeg_quota.vectors_limit.load(Ordering::Relaxed)),
            max_bytes: nonzero(self.skeg_quota.cache_limit.load(Ordering::Relaxed)),
        }
    }

    fn current_usage(&self) -> Usage {
        Usage {
            records: self.skeg_quota.n_vectors(),
            bytes: self.skeg_quota.cache_bytes(),
        }
    }
}

fn nonzero(n: u64) -> Option<u64> {
    if n == 0 { None } else { Some(n) }
}

// ─── Forwarded rigging traits ────────────────────────────────────────
//
// `TenantHandle` is a full rigging tenant: every read-side trait
// forwards to the wrapped `Tenant`, every write-side trait that
// affects accounting goes through the handle's own methods (above).
// Forwarding rather than re-exposing `inner()` means consumers can
// take a `Box<dyn ReadOnlyView>` (or any other rigging trait object)
// directly from a handle without unwrapping.

impl TenantInfo for TenantHandle {
    fn tenant_id(&self) -> RiggingTenantId {
        TenantInfo::tenant_id(&self.inner)
    }
    fn embedding_dim(&self) -> u32 {
        TenantInfo::embedding_dim(&self.inner)
    }
    fn record_count(&self) -> u64 {
        TenantInfo::record_count(&self.inner)
    }
    fn capabilities(&self) -> Vec<CapabilityId> {
        TenantInfo::capabilities(&self.inner)
    }
}

impl TenantStats for TenantHandle {
    fn bytes_on_disk(&self) -> u64 {
        TenantStats::bytes_on_disk(&self.inner)
    }
    fn record_count(&self) -> u64 {
        TenantStats::record_count(&self.inner)
    }
    fn memory_resident(&self) -> u64 {
        TenantStats::memory_resident(&self.inner)
    }
}

impl IterVectors for TenantHandle {
    fn iter_vectors(&self) -> Box<dyn Iterator<Item = (RecordId, Vec<f32>)> + '_> {
        self.inner.iter_vectors()
    }
    fn record_count(&self) -> u64 {
        IterVectors::record_count(&self.inner)
    }
    fn embedding_dim(&self) -> u32 {
        IterVectors::embedding_dim(&self.inner)
    }
}

impl QueryFiltered for TenantHandle {
    fn query_filtered(
        &self,
        embedding: &[f32],
        top_k: u32,
        filter: &dyn Filter,
    ) -> Result<Vec<Hit>, QueryError> {
        self.inner.query_filtered(embedding, top_k, filter)
    }
}

impl ReadOnlyView for TenantHandle {
    fn tenant_id(&self) -> RiggingTenantId {
        ReadOnlyView::tenant_id(&self.inner)
    }
    fn close(self: Box<Self>) -> Result<(), OpenError> {
        Ok(())
    }
}

impl TenantEvents for TenantHandle {
    fn subscribe(&self, filter: EventFilter) -> EventStream {
        self.inner.subscribe(filter)
    }
}

// Lifecycle is **not** forwarded on the handle itself because
// `destroy(self: Box<Self>)` would consume the handle and leave the
// quota counters frozen at their pre-destroy values. Callers that
// want lifecycle ops should:
//   - call `handle.snapshot(dest)` (forwarded above), or
//   - move the inner adapter out via `handle.inner()` and box it
//     into a `Box<dyn TenantLifecycle>` for destroy, then drop the
//     handle so the tracker entry can be reclaimed by the caller.
//
// Discriminating between "quota is meaningful past destroy" and "it
// isn't" is a v0.2 decision; for now we keep the handle Lifecycle-
// agnostic and document the workaround.

impl AsRef<Tenant> for TenantHandle {
    fn as_ref(&self) -> &Tenant {
        &self.inner
    }
}

/// Event helper to consume the handle and return the wrapped adapter.
/// Lets callers move into a `Box<dyn TenantLifecycle>` for destroy.
impl TenantHandle {
    /// Consume the handle, returning the wrapped adapter. The quota
    /// entry stays in the [`skeg_tenant::quota::QuotaTracker`] until
    /// the orchestrator forgets it explicitly. Useful before
    /// invoking [`TenantLifecycle::destroy`].
    pub fn into_inner(self) -> Tenant {
        self.inner
    }
}
