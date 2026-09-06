//! Async trait variants (F.10).
//!
//! Mirrors the sync trait set under the `async` feature flag. Same
//! semantics, async method signatures. Adapters that already manage
//! async I/O implement these natively; adapters that only do sync
//! work can use [`SyncToAsync`] to bridge.
//!
//! ## Feature flag naming
//!
//! This crate calls the flag `async` because the trait surface is
//! runtime-agnostic at the type level — every method is just
//! `async fn`. The bridge [`SyncToAsync`] does pull in
//! `tokio::task::spawn_blocking`, so in practice the runtime is
//! Tokio; consumer crates that wire a specific implementation
//! (`hansa`'s `tokio` feature, `skeg-rigging-net-http`'s `async`
//! feature) name it more directly. This is intentional drift: the
//! trait crate names the *capability*, the adapter crate names the
//! *runtime it commits to*.
//!
//! ## Design
//!
//! - `async_trait` macro for object-safety. Native `async fn in trait`
//!   landed in Rust 1.75 but is not yet `dyn`-compatible; async-trait
//!   produces `Pin<Box<dyn Future + Send>>` returns that are.
//! - All async methods are `Send` so they can run on a multi-threaded
//!   Tokio runtime. The trait bounds reflect that.
//! - Sync traits are NOT auto-imported into the async ones. A type
//!   that wants both impls both. This keeps the bound-set explicit
//!   per trait and avoids the "async fn that blocks" footgun.
//!
//! ## Bridging a sync impl
//!
//! Wrap any sync tenant in [`SyncToAsync`] and you get the async
//! traits for free, backed by `tokio::task::spawn_blocking`. Useful
//! for adapters that haven't been ported yet, or for engines that
//! are inherently blocking (file-backed Vamana, RocksDB, etc).
//!
//! ```rust,ignore
//! use skeg_rigging::asyncs::{AsyncQueryFiltered, SyncToAsync};
//!
//! let sync_tenant: MyTenant = build_it();
//! let async_view = SyncToAsync::new(sync_tenant);
//! let hits = async_view.query_filtered_async(&q, 10, &filter).await?;
//! ```

use std::sync::Arc;

use async_trait::async_trait;

use crate::error::{OpenError, QueryError};
use crate::filter::{Filter, Hit};
use crate::ids::{RecordId, TenantId};
use crate::{IterVectors, QueryFiltered, ReadOnlyView};

/// Async counterpart of [`crate::IterVectors`].
///
/// Stability: Provisional. The exact return type for `iter_vectors_async`
/// may shift to a `Stream` when `futures-core::Stream` stabilises as
/// part of `std`.
#[async_trait]
pub trait AsyncIterVectors: Send + Sync {
    /// Async version of [`IterVectors::iter_vectors`]. Returns the
    /// full list materialised — streaming over a real `Stream`
    /// follows in a later slice.
    async fn iter_vectors_async(&self) -> Vec<(RecordId, Vec<f32>)>;

    /// Async version of [`IterVectors::record_count`].
    async fn record_count_async(&self) -> u64;

    /// Async version of [`IterVectors::embedding_dim`].
    async fn embedding_dim_async(&self) -> u32;
}

/// Async counterpart of [`crate::QueryFiltered`].
///
/// The filter is passed as `Arc<dyn Filter>` so the call can move
/// ownership into `tokio::task::spawn_blocking` (or any other
/// detached task) without lifetime gymnastics. Async callers
/// typically build a filter once and reuse it across queries; an
/// `Arc` is the natural shape.
///
/// A filter that internally awaits is out of scope for v0.1.
///
/// Stability: Provisional.
#[async_trait]
pub trait AsyncQueryFiltered: Send + Sync {
    /// Async version of [`QueryFiltered::query_filtered`].
    async fn query_filtered_async(
        &self,
        embedding: &[f32],
        top_k: u32,
        filter: Arc<dyn Filter>,
    ) -> Result<Vec<Hit>, QueryError>;
}

/// Async counterpart of [`crate::ReadOnlyView`]. Composes the two
/// async traits above plus identity / close semantics, matching the
/// sync trait shape.
///
/// Stability: Provisional.
#[async_trait]
pub trait AsyncReadOnlyView: AsyncIterVectors + AsyncQueryFiltered + Send + Sync {
    /// Async version of [`ReadOnlyView::tenant_id`]. Sync in practice
    /// (no I/O), kept async to preserve trait symmetry.
    async fn tenant_id_async(&self) -> TenantId;

    /// Async close. Implementations typically drop kept resources
    /// (sockets, file handles) here.
    ///
    /// Caveat for [`SyncToAsync`] consumers: that bridge cannot run
    /// the wrapped sync `ReadOnlyView::close` because it holds the
    /// inner tenant in an `Arc` (so other handles can keep using it
    /// concurrently). `SyncToAsync::close_async` only releases the
    /// bridge's Arc reference. Callers that need *deterministic*
    /// close of the underlying tenant should consume the inner
    /// `ReadOnlyView` directly before wrapping, or build their own
    /// bridge that owns the tenant outright.
    async fn close_async(self: Box<Self>) -> Result<(), OpenError>;
}

/// Bridge that lifts any sync tenant into the async trait set by
/// running each call on `tokio::task::spawn_blocking`.
///
/// The wrapped value is held in `Arc<T>` so calls can clone the
/// handle into the blocking task without moving the original.
///
/// Suitable for adapters whose underlying engine is inherently
/// blocking (file-backed Vamana, RocksDB, SQLite). Not suitable for
/// hot-path async work where the blocking pool would dominate latency:
/// rewrite the adapter natively async instead.
pub struct SyncToAsync<T> {
    inner: Arc<T>,
}

impl<T> SyncToAsync<T>
where
    T: Send + Sync + 'static,
{
    /// Wrap a sync tenant.
    pub fn new(inner: T) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Wrap an already-shared sync tenant. Useful when the caller is
    /// holding the tenant in an `Arc` for other reasons.
    pub fn from_arc(inner: Arc<T>) -> Self {
        Self { inner }
    }

    /// Borrow the wrapped sync tenant.
    pub fn inner(&self) -> &T {
        &self.inner
    }
}

#[async_trait]
impl<T> AsyncIterVectors for SyncToAsync<T>
where
    T: IterVectors + Send + Sync + 'static,
{
    async fn iter_vectors_async(&self) -> Vec<(RecordId, Vec<f32>)> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || inner.iter_vectors().collect::<Vec<_>>())
            .await
            .expect("spawn_blocking join")
    }

    async fn record_count_async(&self) -> u64 {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || inner.record_count())
            .await
            .expect("spawn_blocking join")
    }

    async fn embedding_dim_async(&self) -> u32 {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || inner.embedding_dim())
            .await
            .expect("spawn_blocking join")
    }
}

#[async_trait]
impl<T> AsyncQueryFiltered for SyncToAsync<T>
where
    T: QueryFiltered + Send + Sync + 'static,
{
    async fn query_filtered_async(
        &self,
        embedding: &[f32],
        top_k: u32,
        filter: Arc<dyn Filter>,
    ) -> Result<Vec<Hit>, QueryError> {
        let inner = self.inner.clone();
        let embedding = embedding.to_vec();
        tokio::task::spawn_blocking(move || {
            inner.query_filtered(&embedding, top_k, filter.as_ref())
        })
        .await
        .expect("spawn_blocking join")
    }
}

#[async_trait]
impl<T> AsyncReadOnlyView for SyncToAsync<T>
where
    T: ReadOnlyView + Send + Sync + 'static,
{
    async fn tenant_id_async(&self) -> TenantId {
        ReadOnlyView::tenant_id(&*self.inner)
    }

    async fn close_async(self: Box<Self>) -> Result<(), OpenError> {
        // The sync `close` consumes a `Box<Self>`; we can't move out
        // of an `Arc`. v0.1 bridge: just drop the Arc reference. The
        // wrapped tenant's own `Drop` runs when the last reference
        // goes; consumers that need explicit close should use the
        // sync `ReadOnlyView::close` on the inner tenant before
        // wrapping it.
        let _ = self;
        Ok(())
    }
}

/// Async counterpart of [`crate::open_readonly`]. Placeholder for
/// v0.1: returns `NotFound` so the trait set is reachable from a
/// hansa-side caller that wants to pass `open_readonly_async` as
/// part of a `PeerOpenerAsync`.
pub async fn open_readonly_async(
    _path: &std::path::Path,
) -> Result<Box<dyn AsyncReadOnlyView>, OpenError> {
    Err(OpenError::NotFound)
}

/// Used by tests to assert the trait set is `Send + Sync` enough to
/// run on a multi-threaded Tokio runtime.
#[cfg(test)]
fn _assert_send_sync<T: Send + Sync>() {}

#[cfg(test)]
#[allow(dead_code)]
fn _check_bounds() {
    _assert_send_sync::<Box<dyn AsyncQueryFiltered>>();
    _assert_send_sync::<Box<dyn AsyncReadOnlyView>>();
    _assert_send_sync::<Box<dyn AsyncIterVectors>>();
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    // A trivially-async tenant for trait-shape tests. Real-engine
    // bridges and the mock variant live downstream of this module.
    struct StubAsync;

    #[async_trait]
    impl AsyncIterVectors for StubAsync {
        async fn iter_vectors_async(&self) -> Vec<(RecordId, Vec<f32>)> {
            vec![]
        }
        async fn record_count_async(&self) -> u64 {
            0
        }
        async fn embedding_dim_async(&self) -> u32 {
            4
        }
    }

    #[async_trait]
    impl AsyncQueryFiltered for StubAsync {
        async fn query_filtered_async(
            &self,
            _embedding: &[f32],
            _top_k: u32,
            _filter: Arc<dyn Filter>,
        ) -> Result<Vec<Hit>, QueryError> {
            Ok(vec![Hit {
                record_id: RecordId(1),
                similarity: 0.9,
                payload: Bytes::from_static(b"stub"),
                embedding: None,
            }])
        }
    }

    #[async_trait]
    impl AsyncReadOnlyView for StubAsync {
        async fn tenant_id_async(&self) -> TenantId {
            TenantId::ZERO
        }
        async fn close_async(self: Box<Self>) -> Result<(), OpenError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn stub_returns_one_hit() {
        let t = StubAsync;
        struct True;
        impl Filter for True {
            fn accept(&self, _: &crate::RecordMeta<'_>) -> bool {
                true
            }
        }
        let hits = t
            .query_filtered_async(&[1.0; 4], 5, Arc::new(True))
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[tokio::test]
    async fn boxed_trait_object_is_usable() {
        let v: Box<dyn AsyncReadOnlyView> = Box::new(StubAsync);
        assert_eq!(v.tenant_id_async().await, TenantId::ZERO);
        v.close_async().await.unwrap();
    }

    #[tokio::test]
    async fn open_readonly_async_placeholder() {
        let result = open_readonly_async(std::path::Path::new("/nonexistent")).await;
        assert!(matches!(result, Err(OpenError::NotFound)));
    }

    // ─── SyncToAsync bridge tests ────────────────────────────────────

    #[cfg(feature = "mock")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sync_to_async_bridges_mock_tenant() {
        use crate::RecordMeta;
        use crate::mock::MockTenant;

        let mut t = MockTenant::new(TenantId::ZERO, 3);
        t.insert(
            RecordId(1),
            vec![1.0, 0.0, 0.0],
            true,
            vec!["a".into()],
            Bytes::from_static(b"hello"),
        )
        .unwrap();
        t.insert(
            RecordId(2),
            vec![0.0, 1.0, 0.0],
            false,
            vec![],
            Bytes::from_static(b"world"),
        )
        .unwrap();

        let async_view = SyncToAsync::new(t);

        // IterVectors → AsyncIterVectors
        assert_eq!(async_view.record_count_async().await, 2);
        assert_eq!(async_view.embedding_dim_async().await, 3);
        let vecs = async_view.iter_vectors_async().await;
        assert_eq!(vecs.len(), 2);

        // QueryFiltered → AsyncQueryFiltered (shareable only)
        struct Shareable;
        impl Filter for Shareable {
            fn accept(&self, m: &RecordMeta<'_>) -> bool {
                m.shareable
            }
        }
        let hits = async_view
            .query_filtered_async(&[1.0, 0.0, 0.0], 5, Arc::new(Shareable))
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record_id, RecordId(1));
    }

    #[cfg(feature = "mock")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sync_to_async_as_readonly_trait_object() {
        use crate::mock::MockTenant;

        let t = MockTenant::new(TenantId::from_bytes([0x42; 16]), 2);
        let view: Box<dyn AsyncReadOnlyView> = Box::new(SyncToAsync::new(t));
        assert_eq!(
            view.tenant_id_async().await,
            TenantId::from_bytes([0x42; 16])
        );
        view.close_async().await.unwrap();
    }

    #[cfg(feature = "mock")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sync_to_async_runs_concurrent_queries() {
        // Concurrency smoke: SyncToAsync should be safely callable
        // from multiple async tasks; spawn_blocking dispatches each
        // sync call onto the blocking pool independently.
        use crate::RecordMeta;
        use crate::mock::MockTenant;

        let mut t = MockTenant::new(TenantId::ZERO, 3);
        for i in 0..32u64 {
            t.insert(
                RecordId(i),
                vec![1.0, 0.0, 0.0],
                true,
                vec![],
                Bytes::from_static(b"x"),
            )
            .unwrap();
        }
        let view = Arc::new(SyncToAsync::new(t));

        struct All;
        impl Filter for All {
            fn accept(&self, _: &RecordMeta<'_>) -> bool {
                true
            }
        }
        let filter: Arc<dyn Filter> = Arc::new(All);
        let mut handles = vec![];
        for _ in 0..8 {
            let v = view.clone();
            let f = filter.clone();
            handles.push(tokio::spawn(async move {
                v.query_filtered_async(&[1.0, 0.0, 0.0], 4, f).await
            }));
        }
        for h in handles {
            let hits = h.await.unwrap().unwrap();
            assert_eq!(hits.len(), 4);
        }
    }
}
