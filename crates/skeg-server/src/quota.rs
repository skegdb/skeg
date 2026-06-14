//! Per-tenant resource limits and the server-side usage counters that enforce
//! them. P0a S3: hard `n_vectors` quota with admission rejection.
//!
//! The LIMITS come from the pluggable [`crate::tenant::TenantBackend`] (so the
//! server stays decoupled from any concrete tenant store); the USAGE lives here,
//! server-side, because it is derived from what the shards actually hold. A
//! tenant with no limit is never counted, so single-tenant deployments pay
//! nothing.

use std::collections::HashMap;

use parking_lot::RwLock;

/// Hard limits for one tenant. `None` on a field means unlimited.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TenantLimits {
    /// Maximum number of vectors the tenant may store. `None` = unlimited.
    pub max_vectors: Option<u64>,
    /// Maximum live on-disk KV bytes for the tenant. `None` = unlimited.
    pub max_disk_bytes: Option<u64>,
}

/// Admission rejected because it would exceed a tenant's quota.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuotaExceeded;

/// Concurrent per-tenant vector counter. Only tenants with an active limit are
/// tracked (the caller checks `limit.is_some()` before touching this), so the
/// map holds nothing for unlimited / single-tenant traffic.
#[derive(Debug, Default)]
pub struct TenantVectorQuota {
    counts: RwLock<HashMap<u128, u64>>,
}

impl TenantVectorQuota {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Reserve `delta` vectors for `tenant`, capped at `limit`. On success the
    /// count is incremented and `Ok` returned; if it would exceed `limit` (or
    /// overflow), the count is left unchanged and `Err(QuotaExceeded)` returned.
    pub fn try_add(&self, tenant: u128, delta: u64, limit: u64) -> Result<(), QuotaExceeded> {
        let mut g = self.counts.write();
        let cur = g.get(&tenant).copied().unwrap_or(0);
        let new = cur
            .checked_add(delta)
            .filter(|n| *n <= limit)
            .ok_or(QuotaExceeded)?;
        g.insert(tenant, new);
        Ok(())
    }

    /// Release `delta` vectors for `tenant`. Saturates at zero; drops the map
    /// entry when it reaches zero so the map tracks only tenants with usage.
    pub fn sub(&self, tenant: u128, delta: u64) {
        let mut g = self.counts.write();
        if let Some(c) = g.get_mut(&tenant) {
            *c = c.saturating_sub(delta);
            if *c == 0 {
                g.remove(&tenant);
            }
        }
    }

    /// Current reserved vector count for `tenant` (0 if untracked).
    #[must_use]
    pub fn count(&self, tenant: u128) -> u64 {
        self.counts.read().get(&tenant).copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_try_add_within_limit() {
        let q = TenantVectorQuota::new();
        assert!(q.try_add(7, 1, 3).is_ok());
        assert!(q.try_add(7, 2, 3).is_ok());
        assert_eq!(q.count(7), 3);
    }

    #[test]
    fn test_try_add_rejects_at_limit() {
        let q = TenantVectorQuota::new();
        q.try_add(7, 3, 3).unwrap();
        assert_eq!(
            q.try_add(7, 1, 3),
            Err(QuotaExceeded),
            "exactly at limit rejects"
        );
        assert_eq!(q.count(7), 3, "rejected add must not change the count");
    }

    #[test]
    fn test_sub_frees_a_slot() {
        let q = TenantVectorQuota::new();
        q.try_add(7, 3, 3).unwrap();
        q.sub(7, 1);
        assert_eq!(q.count(7), 2);
        assert!(q.try_add(7, 1, 3).is_ok(), "freed slot is reusable");
    }

    #[test]
    fn test_sub_saturates_and_drops_entry() {
        let q = TenantVectorQuota::new();
        q.try_add(7, 1, 10).unwrap();
        q.sub(7, 5); // over-subtract
        assert_eq!(q.count(7), 0);
        q.sub(7, 1); // already absent
        assert_eq!(q.count(7), 0);
    }

    #[test]
    fn test_tenants_isolated() {
        let q = TenantVectorQuota::new();
        q.try_add(7, 2, 2).unwrap();
        assert!(q.try_add(9, 2, 2).is_ok(), "tenant 9 has its own budget");
        assert_eq!(q.try_add(7, 1, 2), Err(QuotaExceeded));
        assert_eq!(q.count(9), 2);
    }

    #[test]
    fn test_overflow_rejected() {
        let q = TenantVectorQuota::new();
        q.try_add(7, 5, u64::MAX).unwrap();
        assert_eq!(q.try_add(7, u64::MAX, u64::MAX), Err(QuotaExceeded));
        assert_eq!(q.count(7), 5);
    }
}
