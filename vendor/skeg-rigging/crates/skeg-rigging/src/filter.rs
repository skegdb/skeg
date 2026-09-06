//! Filtered similarity search.

use bytes::Bytes;

use crate::error::QueryError;
use crate::ids::RecordId;

/// Metadata view of a candidate record during a filtered search.
///
/// Borrows the underlying tag strings; the filter must not retain them.
///
/// Stability: Stable.
#[derive(Debug, Clone, Copy)]
pub struct RecordMeta<'a> {
    /// Record identifier.
    pub record_id: RecordId,
    /// Whether this record opts into being shared with peers in a hansa.
    pub shareable: bool,
    /// Tag strings attached to the record.
    pub tags: &'a [&'a str],
}

/// Predicate decided by the caller, evaluated per candidate during a
/// filtered search.
///
/// Filters must be cheap. Expensive filters degrade search throughput
/// proportionally because the engine evaluates the filter for every
/// visited candidate, not just the survivors.
///
/// Stability: Stable.
pub trait Filter: Send + Sync {
    /// Accept or reject a candidate based on its metadata.
    fn accept(&self, meta: &RecordMeta<'_>) -> bool;
}

impl<F> Filter for F
where
    F: Fn(&RecordMeta<'_>) -> bool + Send + Sync,
{
    fn accept(&self, meta: &RecordMeta<'_>) -> bool {
        self(meta)
    }
}

/// A search hit.
///
/// Stability: Stable.
#[derive(Debug, Clone)]
pub struct Hit {
    /// Hit record identifier.
    pub record_id: RecordId,
    /// Similarity score, higher = closer. Range is implementation-defined
    /// (typically cosine similarity in [-1, 1]).
    pub similarity: f32,
    /// Record payload, refcounted to avoid copies from mmap pages.
    pub payload: Bytes,
    /// Vector that produced this hit, when the backend can return it
    /// cheaply. Local on-disk backends (e.g. `skeg-rigging-skeg`) can
    /// surface the stored embedding for downstream semantic dedup;
    /// remote transports that don't ship the raw vector back (RESP3
    /// `VSEARCH`, HTTP saga summaries) leave this `None`. Consumers
    /// must treat `None` as "not available", not "no vector exists".
    pub embedding: Option<Vec<f32>>,
}

/// Filtered similarity search over a vault.
///
/// Stability: Stable.
pub trait QueryFiltered {
    /// Return the top-`top_k` hits whose metadata passes `filter`.
    ///
    /// The filter is evaluated during traversal where the implementation
    /// allows it; the v0.1 reference implementation in
    /// `skeg-rigging-skeg` oversamples and post-filters, since
    /// filter-during-traversal in skeg-vector is a v0.2 optimisation.
    fn query_filtered(
        &self,
        embedding: &[f32],
        top_k: u32,
        filter: &dyn Filter,
    ) -> Result<Vec<Hit>, QueryError>;
}
