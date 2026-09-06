#![deny(unsafe_code)]
#![warn(missing_docs)]

//! `skeg-rigging` - the public contract between memory engines and the
//! plugins that build on top of them.
//!
//! Rigging defines *what* a tenant - one isolated unit of memory inside
//! some engine - must be able to do for plugins like
//! [`hansa`](https://github.com/skegdb/hansa) to work. It does **not**
//! implement those capabilities. The reference adapter
//! `skeg-rigging-skeg` is one implementation that wraps skeg's public
//! engine API; other engines (graph, multimodal, temporal, in-memory
//! mocks) can implement the same surface and be drop-in interchangeable
//! to consumers.
//!
//! ## Stability tiers
//!
//! Traits in v0.1 carry one of two tiers:
//!
//! - `Stable`: signature locked for v0.x. Patch and minor releases will
//!   not break consumers; only additive default-method changes go in
//!   minor releases.
//! - `Provisional`: shape may shift in minor releases. Plugins that
//!   consume Provisional traits should expect a small migration at the
//!   next minor bump. In v0.1, only [`ReadOnlyView`] is Provisional.
//!
//! Stability commitments apply to *every* implementor of a rigging
//! trait, not just the reference engine. A streaming method must
//! stream; a filter must be applied during traversal where the trait
//! says so; a `ReadOnlyView` must produce a consistent snapshot.
//!
//! ## Engine pluralism
//!
//! Rigging is designed as a capability layer for memory engines in
//! general, not as the proprietary API of any single engine. The trait
//! set is named for *capabilities* (iter, filter, read-only), not for
//! the data model underneath. A vector + KV engine like skeg-core is
//! the reference implementor; a graph engine that interprets
//! "embedding" as a structural fingerprint, a temporal engine that
//! returns time-weighted similarity, or an in-memory mock for tests all
//! satisfy the contract.
//!
//! This is a *position*: nothing in rigging's design will prevent
//! alternative engines from existing, and consumers will not need to be
//! rewritten when they appear. The pluralism implies discipline in the
//! shape of the trait set - no vector-specific assumptions in shared
//! types, no engine-specific internals in method signatures - and a
//! capability-discovery convention (forthcoming in v0.2 via
//! `TenantInfo::capabilities()`) so consumers can discriminate engines
//! at runtime.
//!
//! Capability identifiers follow the convention `<engine>.<capability>`:
//! `vector.kv` is the skeg-core baseline, `graph.traverse`,
//! `media.multimodal`, `temporal.windowed` are reserved namespaces for
//! hypothetical engines, and consumer-side capabilities like
//! `hansa.member` and `hansa.membrane` extend the set without colliding.
//!
//! ## Vocabulary
//!
//! A **tenant** is one engine's isolated memory unit. [`TenantId`] is
//! its stable 16-byte identifier. The auth/quota scoping `TenantId` in
//! the separate `skeg-tenant` crate is orthogonal: different namespace,
//! different concern, the two coexist.

mod accounting;
#[cfg(feature = "async")]
pub mod asyncs;
mod error;
mod events;
mod filter;
mod ids;
mod iter;
mod lifecycle;
#[cfg(feature = "mock")]
pub mod mock;
mod readonly;
mod write;

pub use accounting::{Quota, QuotaError, TenantQuota, TenantStats, Usage};
pub use error::{OpenError, QueryError};
pub use events::{Event, EventFilter, EventKind, EventStream, TenantEvents};
pub use filter::{Filter, Hit, QueryFiltered, RecordMeta};
pub use ids::{RecordId, TenantId};
pub use iter::IterVectors;
pub use lifecycle::{
    CAP_VECTOR_KV, CAP_VECTOR_QUANTIZED, CapabilityId, TenantInfo, TenantLifecycle,
};
pub use readonly::{ReadOnlyView, open_readonly};
pub use write::TenantWrite;

/// Common imports for plugin authors.
///
/// ```
/// use skeg_rigging::prelude::*;
/// ```
pub mod prelude {
    pub use crate::{
        CapabilityId, Event, EventFilter, EventKind, EventStream, Filter, Hit, IterVectors,
        OpenError, QueryError, QueryFiltered, Quota, QuotaError, ReadOnlyView, RecordId,
        RecordMeta, TenantEvents, TenantId, TenantInfo, TenantLifecycle, TenantQuota, TenantStats,
        TenantWrite, Usage, open_readonly,
    };
}
