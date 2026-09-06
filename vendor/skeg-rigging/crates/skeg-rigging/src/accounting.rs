//! Tenant accounting: resource counters + quotas.
//!
//! M5 roadmap item. Orchestrators need two things to make routing
//! and billing decisions: a snapshot of what each tenant is currently
//! consuming ([`TenantStats`]) and a knob to cap that consumption
//! ([`TenantQuota`]).
//!
//! ## Stability
//!
//! - [`TenantStats`] - Stable. Counter shape is locked.
//! - [`TenantQuota`], [`Quota`], [`Usage`] - Provisional. May gain
//!   fields (per-stream quotas, rate limits) in a 0.x minor release.
//!
//! ## What's deferred
//!
//! `query_rate` (sliding-window QPS counters) is in the design but
//! ships in a later slice - it needs an interceptor around
//! [`crate::QueryFiltered`] which the adapter does not yet have. v0.1
//! of accounting therefore covers disk + record counters + memory
//! residency, plus the quota surface.

use crate::error::OpenError;

/// Resource counters. Snapshot semantics: every getter returns the
/// value at call time without locking the underlying tenant for the
/// duration of a multi-getter read. Implementations are expected to
/// be cheap; callers can poll at 1 Hz for dashboards without
/// regressing the data plane.
///
/// Stability: Stable.
pub trait TenantStats {
    /// Total bytes the tenant currently occupies on persistent
    /// storage. For directory-based engines, the sum of file sizes
    /// under the tenant root. Mmap-backed bytes count once (on disk),
    /// not also against [`Self::memory_resident`].
    fn bytes_on_disk(&self) -> u64;

    /// Number of records visible to a fresh query right now.
    /// Mirrors [`crate::TenantInfo::record_count`]; redeclared here so
    /// `TenantStats` is usable standalone.
    fn record_count(&self) -> u64;

    /// Best effort estimate of the bytes the tenant currently keeps
    /// resident in process memory. Engines that mmap from disk and
    /// rely on the OS page cache return `0` here (the cache is
    /// shared and not directly attributable). Engines that maintain
    /// their own resident structures (graph adjacency, posting lists,
    /// etc.) return their tracked size.
    fn memory_resident(&self) -> u64;
}

/// Caps on tenant resource consumption. v0.1 is informational +
/// adapter-enforced on inserts: the adapter rejects a write that
/// would push usage over the cap with [`QuotaError::Exceeded`].
/// Query-side and rate-side quotas are deferred.
///
/// Stability: Provisional.
pub trait TenantQuota {
    /// Replace the current quota. Passing [`Quota::UNLIMITED`] lifts
    /// every cap; passing a stricter quota than the current usage is
    /// allowed (the next write that violates the new cap will fail -
    /// existing rows are untouched).
    fn set_quota(&self, quota: Quota) -> Result<(), QuotaError>;

    /// Current quota in force. Default after open is
    /// [`Quota::UNLIMITED`].
    fn quota(&self) -> Quota;

    /// What the tenant is actually consuming right now. Independent
    /// snapshot of the [`TenantStats`] counters relevant to the quota
    /// dimensions - kept separate so callers can compare without
    /// mixing concerns.
    fn current_usage(&self) -> Usage;
}

/// Resource caps. `None` on a dimension means "unlimited on that
/// dimension".
///
/// Stability: Provisional.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Quota {
    /// Maximum bytes the tenant may occupy on disk. The check fires
    /// at insert time on a best effort estimate of the post-write
    /// size; small overshoots are possible (e.g. when a write
    /// triggers a flush that aligns up to a page).
    pub max_bytes: Option<u64>,
    /// Maximum live record count.
    pub max_records: Option<u64>,
}

impl Quota {
    /// No caps. Default after open.
    pub const UNLIMITED: Quota = Quota {
        max_bytes: None,
        max_records: None,
    };

    /// True when no dimension is capped.
    pub fn is_unlimited(&self) -> bool {
        self.max_bytes.is_none() && self.max_records.is_none()
    }
}

/// Current consumption snapshot. Mirrors the dimensions of [`Quota`]
/// so `usage_vs_quota` comparisons stay obvious.
///
/// Stability: Provisional.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    /// Bytes currently on disk.
    pub bytes: u64,
    /// Records currently live.
    pub records: u64,
}

/// Errors from the quota surface.
#[derive(Debug, thiserror::Error)]
pub enum QuotaError {
    /// A write would push the tenant past its quota. The caller can
    /// retry after raising the quota or deleting records.
    #[error("quota exceeded: {dimension} (cap {cap}, attempted {attempted})")]
    Exceeded {
        /// Which dimension hit the cap. One of `"bytes"` or `"records"`.
        dimension: &'static str,
        /// The cap currently in force.
        cap: u64,
        /// The value the write would have brought the dimension to.
        attempted: u64,
    },
    /// An I/O failure while sampling or persisting the quota
    /// (relevant when the quota is durable; v0.1's quota is in-memory
    /// so this variant exists for forward-compat).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<QuotaError> for OpenError {
    fn from(e: QuotaError) -> Self {
        match e {
            QuotaError::Io(io) => OpenError::Io(io),
            QuotaError::Exceeded { .. } => OpenError::Io(std::io::Error::other(e.to_string())),
        }
    }
}
