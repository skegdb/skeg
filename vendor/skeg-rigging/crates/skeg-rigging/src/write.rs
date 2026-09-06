//! Record insertion — the write side of a tenant.
//!
//! [`IterVectors`](crate::IterVectors) and [`ReadOnlyView`](crate::ReadOnlyView)
//! cover reading a tenant; `TenantWrite` is the matching capability for
//! *putting records in*. It exists so that a single ingest routine
//! (chunk → embed → insert) can target **whatever store a tenant is
//! backed by** without knowing which one it is:
//!
//! - an in-process engine (the reference adapter `skeg-rigging-skeg`
//!   writes to a local on-disk index), or
//! - a remote engine reached over the wire (the `skeg-rigging-net-resp3`
//!   bridge issues `SKEG.VSET` / `SET` against a skeg server).
//!
//! Both expose the identical logical operation — *insert one record*: a
//! vector, a `shareable` flag, tag strings, and an opaque payload (the
//! original text, typically). Capturing it as a trait lets consumers
//! like `hansa` and bulk importers write once and run against either
//! location, which is the whole point of rigging being a capability
//! layer rather than one engine's API.
//!
//! Stability: Provisional (v0.2). The shape mirrors the two reference
//! writers; a future minor release may add batch insertion as a
//! default-method extension.

use crate::ids::RecordId;

/// Insert records into a tenant, regardless of where the tenant lives.
///
/// The five fields of [`Self::insert`] are the common denominator of the
/// reference writers and are interpreted consistently:
///
/// - `record_id`: caller-assigned u64 identity; re-inserting the same id
///   overwrites.
/// - `embedding`: the vector, length must equal [`Self::embedding_dim`].
/// - `shareable`: whether the record is visible to federation peers.
///   Engines that do not model sharing may ignore it.
/// - `tags`: free-form labels (provenance, source path, category).
/// - `payload`: opaque bytes the engine stores verbatim and returns on
///   read — usually the source text the embedding was computed from.
///
/// **Durability.** `insert` may buffer; callers must invoke
/// [`Self::flush`] to guarantee the records are persisted. Remote
/// implementations where every insert is a synchronous round-trip may
/// treat `flush` as a no-op (the default).
pub trait TenantWrite {
    /// Error type surfaced by this writer's backend.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Insert (or overwrite) one record. See the trait docs for the
    /// meaning of each field.
    fn insert(
        &mut self,
        record_id: RecordId,
        embedding: &[f32],
        shareable: bool,
        tags: Vec<String>,
        payload: Vec<u8>,
    ) -> Result<(), Self::Error>;

    /// Persist any buffered inserts. Default: nothing to do.
    fn flush(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Embedding dimension this writer expects. Inserts whose vector
    /// length differs should be rejected by the implementation.
    fn embedding_dim(&self) -> u32;
}
