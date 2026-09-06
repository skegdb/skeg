//! Streaming raw vector access.

use crate::ids::RecordId;

/// Streaming access to the raw vectors stored in a tenant.
///
/// Plugins use this for building digests, summaries, or auxiliary
/// indexes - anything that requires walking every vector.
///
/// **Performance contract.** `iter_vectors` may be slow (it reads from
/// disk). Implementations *must not* allocate buffers proportional to
/// `record_count`; they should stream pages lazily.
///
/// **Engine note.** The trait talks about "vectors" by name because the
/// initial cluster of engines is vector-shaped, but implementors are
/// free to interpret the f32 slice however suits their model (a graph
/// engine may return a structural fingerprint, a multimodal engine may
/// return a fused embedding). What matters is that one `Vec<f32>` per
/// record is yielded consistently with `embedding_dim`.
///
/// Stability: Stable.
pub trait IterVectors {
    /// Iterate over every `(record_id, vector)` pair in the tenant.
    fn iter_vectors(&self) -> Box<dyn Iterator<Item = (RecordId, Vec<f32>)> + '_>;

    /// Total record count in this tenant.
    fn record_count(&self) -> u64;

    /// Embedding dimension. All vectors in a tenant share the same dim.
    fn embedding_dim(&self) -> u32;
}
