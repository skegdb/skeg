//! Per-tenant quota accounting.
//!
//! Two counters are tracked by default:
//!
//! - `cache_bytes`: live size of the S3-FIFO partition for the tenant.
//!   Tracking this means an evict path will trim *this tenant's*
//!   partition, not a global LRU shared across tenants.
//! - `n_vectors`: total vectors across all VINDEX owned by the tenant.
//!   Cheap to maintain (one counter per VSET / VDEL) and immediately
//!   gives operators a knob.
//!
//! Disk-byte accounting is deliberately omitted; the vLog append path
//! is hot and we want a real load profile before paying for it.
//! See FEATURES.md §multi-tenant for the rationale.
//!
//! The implementation is lock-free for reads and uses atomic CAS for
//! updates. There is one `TenantQuota` per active tenant; the
//! `QuotaTracker` is a sharded map keyed on `TenantId`. We use
//! `parking_lot::RwLock` to mediate add/remove of the entries
//! themselves (a per-tenant counter map mutation), not the increments.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use ahash::AHashMap;
use parking_lot::RwLock;

use crate::id::TenantId;

#[derive(Debug, thiserror::Error, PartialEq, Eq, Clone)]
pub enum QuotaError {
    #[error("tenant {tenant} cache budget exceeded: would be {would_be}/{limit}")]
    CacheBudgetExceeded {
        tenant: TenantId,
        would_be: u64,
        limit: u64,
    },
    #[error("tenant {tenant} vector count exceeded: would be {would_be}/{limit}")]
    VectorCountExceeded {
        tenant: TenantId,
        would_be: u64,
        limit: u64,
    },
}

/// Per-tenant live counters + limits. `limit == 0` means unlimited.
#[derive(Debug)]
pub struct TenantQuota {
    pub cache_bytes: AtomicU64,
    pub cache_limit: AtomicU64,
    pub n_vectors: AtomicU64,
    pub vectors_limit: AtomicU64,
}

impl TenantQuota {
    #[must_use]
    pub fn unlimited() -> Self {
        Self {
            cache_bytes: AtomicU64::new(0),
            cache_limit: AtomicU64::new(0),
            n_vectors: AtomicU64::new(0),
            vectors_limit: AtomicU64::new(0),
        }
    }

    #[must_use]
    pub fn with_limits(cache_limit: u64, vectors_limit: u64) -> Self {
        let q = Self::unlimited();
        q.cache_limit.store(cache_limit, Ordering::Relaxed);
        q.vectors_limit.store(vectors_limit, Ordering::Relaxed);
        q
    }

    pub fn set_cache_limit(&self, limit: u64) {
        self.cache_limit.store(limit, Ordering::Relaxed);
    }

    pub fn set_vectors_limit(&self, limit: u64) {
        self.vectors_limit.store(limit, Ordering::Relaxed);
    }

    /// Try to add `delta` bytes to `cache_bytes`. Fails if it would push
    /// the counter past `cache_limit` (when the limit is non-zero).
    ///
    /// # Errors
    ///
    /// Returns `QuotaError::CacheBudgetExceeded` if the proposed value
    /// would exceed the configured limit.
    pub fn try_add_cache_bytes(&self, tenant: TenantId, delta: u64) -> Result<(), QuotaError> {
        let limit = self.cache_limit.load(Ordering::Relaxed);
        // Compare-and-set loop. We need the result to be observably
        // consistent with the limit: a parallel grant for the same tenant
        // must see the freshly bumped counter before deciding.
        loop {
            let cur = self.cache_bytes.load(Ordering::Acquire);
            let next = cur.saturating_add(delta);
            if limit != 0 && next > limit {
                return Err(QuotaError::CacheBudgetExceeded {
                    tenant,
                    would_be: next,
                    limit,
                });
            }
            if self
                .cache_bytes
                .compare_exchange_weak(cur, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Ok(());
            }
        }
    }

    pub fn sub_cache_bytes(&self, delta: u64) {
        // Saturating: we never want this to wrap if accounting drifts.
        loop {
            let cur = self.cache_bytes.load(Ordering::Acquire);
            let next = cur.saturating_sub(delta);
            if self
                .cache_bytes
                .compare_exchange_weak(cur, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return;
            }
        }
    }

    /// Try to add `delta` vectors to the per-tenant counter.
    ///
    /// # Errors
    ///
    /// Returns `QuotaError::VectorCountExceeded` if the proposed value
    /// would exceed the configured limit.
    pub fn try_add_vectors(&self, tenant: TenantId, delta: u64) -> Result<(), QuotaError> {
        let limit = self.vectors_limit.load(Ordering::Relaxed);
        loop {
            let cur = self.n_vectors.load(Ordering::Acquire);
            let next = cur.saturating_add(delta);
            if limit != 0 && next > limit {
                return Err(QuotaError::VectorCountExceeded {
                    tenant,
                    would_be: next,
                    limit,
                });
            }
            if self
                .n_vectors
                .compare_exchange_weak(cur, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Ok(());
            }
        }
    }

    pub fn sub_vectors(&self, delta: u64) {
        loop {
            let cur = self.n_vectors.load(Ordering::Acquire);
            let next = cur.saturating_sub(delta);
            if self
                .n_vectors
                .compare_exchange_weak(cur, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return;
            }
        }
    }

    #[must_use]
    pub fn cache_bytes(&self) -> u64 {
        self.cache_bytes.load(Ordering::Relaxed)
    }
    #[must_use]
    pub fn n_vectors(&self) -> u64 {
        self.n_vectors.load(Ordering::Relaxed)
    }
}

/// Process-wide quota registry. Cloning gives a cheap handle (Arc inside).
#[derive(Debug, Default, Clone)]
pub struct QuotaTracker {
    inner: Arc<RwLock<AHashMap<TenantId, Arc<TenantQuota>>>>,
    default_cache_limit: Arc<AtomicU64>,
    default_vectors_limit: Arc<AtomicU64>,
}

impl QuotaTracker {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Limit every newly-registered tenant inherits. Existing entries
    /// are left untouched.
    pub fn set_defaults(&self, cache_limit: u64, vectors_limit: u64) {
        self.default_cache_limit
            .store(cache_limit, Ordering::Relaxed);
        self.default_vectors_limit
            .store(vectors_limit, Ordering::Relaxed);
    }

    /// Register a tenant with explicit limits, replacing any prior entry.
    pub fn register(&self, tenant: TenantId, q: TenantQuota) -> Arc<TenantQuota> {
        let q = Arc::new(q);
        self.inner.write().insert(tenant, q.clone());
        q
    }

    /// Lookup or insert with default limits.
    #[must_use]
    pub fn entry(&self, tenant: TenantId) -> Arc<TenantQuota> {
        if let Some(q) = self.inner.read().get(&tenant) {
            return q.clone();
        }
        let mut w = self.inner.write();
        if let Some(q) = w.get(&tenant) {
            return q.clone();
        }
        let q = Arc::new(TenantQuota::with_limits(
            self.default_cache_limit.load(Ordering::Relaxed),
            self.default_vectors_limit.load(Ordering::Relaxed),
        ));
        w.insert(tenant, q.clone());
        q
    }

    /// Remove a tenant from the tracker. Returns the dropped record so
    /// callers can observe the final counts (useful for billing flushes).
    #[must_use]
    pub fn forget(&self, tenant: TenantId) -> Option<Arc<TenantQuota>> {
        self.inner.write().remove(&tenant)
    }

    #[must_use]
    pub fn snapshot(&self) -> Vec<(TenantId, u64, u64)> {
        self.inner
            .read()
            .iter()
            .map(|(t, q)| (*t, q.cache_bytes(), q.n_vectors()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_bytes_within_limit_succeeds() {
        let t = TenantId::from_name("alice");
        let q = TenantQuota::with_limits(1024, 0);
        q.try_add_cache_bytes(t, 512).unwrap();
        q.try_add_cache_bytes(t, 512).unwrap();
        assert_eq!(q.cache_bytes(), 1024);
    }

    #[test]
    fn cache_bytes_over_limit_rejected() {
        let t = TenantId::from_name("alice");
        let q = TenantQuota::with_limits(1024, 0);
        q.try_add_cache_bytes(t, 1024).unwrap();
        let e = q.try_add_cache_bytes(t, 1).unwrap_err();
        match e {
            QuotaError::CacheBudgetExceeded {
                would_be, limit, ..
            } => {
                assert_eq!(would_be, 1025);
                assert_eq!(limit, 1024);
            }
            QuotaError::VectorCountExceeded { .. } => panic!("wrong variant"),
        }
        // Counter must be unchanged by the failed admission.
        assert_eq!(q.cache_bytes(), 1024);
    }

    #[test]
    fn cache_bytes_sub_saturates_at_zero() {
        let q = TenantQuota::unlimited();
        q.sub_cache_bytes(100); // already zero
        assert_eq!(q.cache_bytes(), 0);
    }

    #[test]
    fn vectors_limit_enforced() {
        let t = TenantId::from_name("alice");
        let q = TenantQuota::with_limits(0, 10);
        q.try_add_vectors(t, 10).unwrap();
        assert!(q.try_add_vectors(t, 1).is_err());
        q.sub_vectors(5);
        q.try_add_vectors(t, 5).unwrap();
    }

    #[test]
    fn zero_limit_means_unlimited() {
        let t = TenantId::from_name("alice");
        let q = TenantQuota::unlimited();
        q.try_add_cache_bytes(t, u64::MAX / 2).unwrap();
        q.try_add_cache_bytes(t, u64::MAX / 2).unwrap();
        // Saturating prevented overflow, no error raised.
    }

    #[test]
    fn tracker_inherits_defaults() {
        let tk = QuotaTracker::new();
        tk.set_defaults(2048, 100);
        let t = TenantId::from_name("alice");
        let q = tk.entry(t);
        assert!(q.try_add_cache_bytes(t, 2048).is_ok());
        assert!(q.try_add_cache_bytes(t, 1).is_err());
    }

    #[test]
    fn tracker_isolates_per_tenant() {
        let tk = QuotaTracker::new();
        tk.set_defaults(100, 10);
        let a = TenantId::from_name("alice");
        let b = TenantId::from_name("bob");
        let qa = tk.entry(a);
        let qb = tk.entry(b);
        qa.try_add_cache_bytes(a, 100).unwrap();
        // Bob is unaffected.
        qb.try_add_cache_bytes(b, 100).unwrap();
    }

    #[test]
    fn tracker_explicit_register_overrides_defaults() {
        let tk = QuotaTracker::new();
        tk.set_defaults(100, 10);
        let a = TenantId::from_name("alice");
        tk.register(a, TenantQuota::with_limits(10_000, 1_000));
        let qa = tk.entry(a);
        qa.try_add_cache_bytes(a, 9_000).unwrap();
    }

    #[test]
    fn snapshot_lists_active_tenants() {
        let tk = QuotaTracker::new();
        tk.set_defaults(1024, 100);
        let a = TenantId::from_name("alice");
        let qa = tk.entry(a);
        qa.try_add_cache_bytes(a, 256).unwrap();
        qa.try_add_vectors(a, 4).unwrap();
        let snap = tk.snapshot();
        assert_eq!(snap.len(), 1);
        let (tid, bytes, n) = snap[0];
        assert_eq!(tid, a);
        assert_eq!(bytes, 256);
        assert_eq!(n, 4);
    }
}
