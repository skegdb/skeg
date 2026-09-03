//! Per-tenant resource limits and the server-side usage counters that enforce
//! them. Hard `n_vectors` quota with admission rejection.
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

/// A tenant's QoS limits as plain numbers. `None` on a field means unlimited.
/// The engine only carries these between the admin command and the backend; the
/// meaning (token-bucket rate / concurrency cap) lives in the backend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TenantQos {
    /// Sustained compute credits per second (token-bucket refill). A flat
    /// command costs 1 credit; a vector search costs more. `None` = unlimited.
    pub rate: Option<u32>,
    /// Token-bucket burst allowance, in credits. `None` = unlimited.
    pub burst: Option<u32>,
    /// Maximum concurrent in-flight commands. `None` = unlimited.
    pub max_concurrent: Option<u32>,
}

/// Admission rejected because it would exceed a tenant's quota.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuotaExceeded;

/// What one physical vector write or delete means for the tenant's LOGICAL
/// cardinality - which is the quantity the quota counts.
///
/// It replaced a `bool` called `internal`, and the reason is that "internal"
/// answered the wrong question. The quota counts a tenant's distinct rows; a
/// single shard cannot see that, because it decides "is this id new" from its
/// own contents and a row arriving from another shard looks new every time.
/// Two opposite bugs came out of a boolean: a cross-shard overwrite took a
/// slot for a row the tenant already had, and was refused at exactly the
/// limit; and a reshard move wrote with no limit (so no increment) while its
/// source delete credited one back, walking the counter downward once per
/// moved row until a tenant was counted below its own contents.
///
/// The variant says what the operation IS, and the decision is derived here,
/// in one place, by matches that name every variant - so a sixth kind of
/// write cannot be added without someone deciding what it costs.
///
/// One enum for both directions, because both questions are the same
/// question. A `Vset` never carries [`Delete`](Self::Delete) and a `Vdel`
/// never carries [`Insert`](Self::Insert) or [`Overwrite`](Self::Overwrite);
/// those arms are still spelled out below rather than folded into a default,
/// since "unreachable" and "free" are different claims and only one of them
/// is safe to guess.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuotaEffect {
    /// A user write of a row the tenant does not have. The only write that
    /// can consume a slot - and only if the shard taking it holds no copy of
    /// the id already, which is what makes an overwrite on the hash-placed
    /// path free without the coordinator having to know it is one.
    Insert,
    /// A user write over a row the tenant already owns, wherever the old copy
    /// happens to live. Cardinality unchanged.
    Overwrite,
    /// One side of a relocation: the copy written to a row's new owner, or
    /// the copy removed from its old one. The row is not arriving and not
    /// leaving, it is moving.
    Move,
    /// A second physical copy of one logical row - the boundary replica an
    /// overlap writes, or that same copy being removed. It never had a slot
    /// of its own, so it neither takes nor gives one back.
    Replica,
    /// The user delete: the row really is leaving the tenant. The one
    /// operation that credits the counter.
    Delete,
}

impl QuotaEffect {
    /// Does this write consume a quota slot? `absent_here` is the receiving
    /// shard's own answer to "did I hold this id already".
    #[must_use]
    pub fn charges(self, absent_here: bool) -> bool {
        match self {
            QuotaEffect::Insert => absent_here,
            // The tenant already has this row: an overwrite of it, a move of
            // it, or a second copy of it changes no cardinality.
            QuotaEffect::Overwrite | QuotaEffect::Move | QuotaEffect::Replica => false,
            // Not reachable on a write. Charging for it would be the worse
            // guess of the two.
            QuotaEffect::Delete => false,
        }
    }

    /// Does this delete give a slot back?
    #[must_use]
    pub fn credits(self) -> bool {
        match self {
            QuotaEffect::Delete => true,
            // The far side of a move, or a replica whose primary was already
            // credited. Crediting here drops the counter for a row that still
            // exists.
            QuotaEffect::Move | QuotaEffect::Replica => false,
            // Not reachable on a delete, and nothing was taken to give back.
            QuotaEffect::Insert | QuotaEffect::Overwrite => false,
        }
    }
}

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

    /// STATE that `tenant` holds `count` vectors. Startup only.
    ///
    /// This is the one writer that is not an admission decision, and the
    /// difference is the whole point of it existing separately from
    /// [`try_add`](Self::try_add):
    ///
    /// - it SETS rather than adds, because the caller has just counted what
    ///   the store holds and the number it arrived with is the answer, not a
    ///   delta on top of whatever a previous open left behind;
    /// - it NEVER REJECTS, because the rows are already on disk. A count over
    ///   the tenant's limit is a fact about the store - reached by lowering a
    ///   limit under a tenant, or by an earlier build that did not count at
    ///   all - and refusing to record it would leave the counter at zero,
    ///   which is exactly the state this exists to end. The limit is applied
    ///   afterwards, by the next `try_add`, which then refuses: an over-quota
    ///   tenant may not grow, and its existing rows stay readable.
    ///
    /// Zero removes the entry, keeping the same invariant [`sub`](Self::sub)
    /// keeps: the map tracks only tenants with usage.
    ///
    /// It must run before any write is admitted. A `rebuild` racing a
    /// `try_add` would overwrite a reservation the write is relying on; the
    /// readiness barrier in `ShardSet::open` is what makes that impossible,
    /// not anything here.
    pub fn rebuild(&self, tenant: u128, count: u64) {
        let mut g = self.counts.write();
        if count == 0 {
            g.remove(&tenant);
        } else {
            g.insert(tenant, count);
        }
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

    /// The startup setter states what the store holds. It is not an
    /// admission decision and has nothing to refuse: the rows are already on
    /// disk, and a count over the tenant's limit is a fact about the store,
    /// not a request to be denied. Refusing it would leave the counter at
    /// zero, which is the bug it exists to close.
    #[test]
    fn test_rebuild_sets_a_count_nothing_reserved() {
        let q = TenantVectorQuota::new();
        q.rebuild(7, 42);
        assert_eq!(q.count(7), 42);
        assert_eq!(q.count(9), 0, "another tenant is untouched");
    }

    #[test]
    fn test_rebuild_overwrites_a_stale_count() {
        let q = TenantVectorQuota::new();
        q.try_add(7, 5, 100).unwrap();
        q.rebuild(7, 2);
        assert_eq!(q.count(7), 2, "rebuild SETS, it does not add");
        q.rebuild(7, 9);
        assert_eq!(q.count(7), 9, "and it can go back up");
    }

    #[test]
    fn test_rebuild_to_zero_removes_the_entry() {
        let q = TenantVectorQuota::new();
        q.try_add(7, 3, 100).unwrap();
        q.rebuild(7, 0);
        assert_eq!(q.count(7), 0);
        // Same invariant `sub` keeps: the map tracks only tenants with usage.
        q.rebuild(9, 0);
        assert_eq!(q.count(9), 0);
    }

    /// A count above every limit the tenant could have still lands: the
    /// setter reports, it does not admit. What follows is that the tenant's
    /// next write is refused, which is the correct answer for a tenant over
    /// its quota - and the opposite of what a zeroed counter would say.
    #[test]
    fn test_rebuild_never_rejects() {
        let q = TenantVectorQuota::new();
        q.rebuild(7, u64::MAX);
        assert_eq!(q.count(7), u64::MAX);
        assert_eq!(q.try_add(7, 1, 10), Err(QuotaExceeded));
    }

    #[test]
    fn test_overflow_rejected() {
        let q = TenantVectorQuota::new();
        q.try_add(7, 5, u64::MAX).unwrap();
        assert_eq!(q.try_add(7, u64::MAX, u64::MAX), Err(QuotaExceeded));
        assert_eq!(q.count(7), 5);
    }
}
