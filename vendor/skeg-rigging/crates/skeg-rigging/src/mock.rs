//! In-memory tenant implementation for testing plugins.
//!
//! Enabled by the `mock` feature flag. Provides [`MockTenant`], a
//! brute-force vector store that implements every Stable trait in this
//! crate plus [`ReadOnlyView`]. It is not optimised: O(N) on every
//! query. Use it to exercise plugin code paths without dragging in a
//! real engine.
//!
//! ```
//! # #[cfg(feature = "mock")] {
//! use bytes::Bytes;
//! use skeg_rigging::mock::MockTenant;
//! use skeg_rigging::prelude::*;
//!
//! let mut t = MockTenant::new(TenantId::ZERO, 3);
//! t.insert(RecordId(1), vec![1.0, 0.0, 0.0], true, vec!["x".into()], Bytes::from_static(b"hello"))
//!     .unwrap();
//! let hits = t
//!     .query_filtered(&[1.0, 0.0, 0.0], 5, &|_: &RecordMeta<'_>| true)
//!     .unwrap();
//! assert_eq!(hits.len(), 1);
//! # }
//! ```

use std::sync::Mutex;
use std::sync::mpsc;

use bytes::Bytes;

use crate::{
    Event, EventFilter, EventStream, Filter, Hit, IterVectors, OpenError, QueryError,
    QueryFiltered, Quota, QuotaError, ReadOnlyView, RecordId, RecordMeta, TenantEvents, TenantId,
    TenantQuota, TenantStats, Usage,
};

/// One record in a [`MockTenant`].
#[derive(Debug, Clone)]
pub struct MockRecord {
    /// Record identifier.
    pub record_id: RecordId,
    /// Full-precision embedding.
    pub embedding: Vec<f32>,
    /// Visibility flag - peers in a hansa never see records with
    /// `shareable == false`.
    pub shareable: bool,
    /// Tag strings.
    pub tags: Vec<String>,
    /// Record payload, refcounted.
    pub payload: Bytes,
}

/// In-memory tenant. Cheap to construct.
///
/// Quota is held behind an internal mutex so [`TenantQuota`] methods
/// take `&self`, matching the trait. The records vector stays plain
/// (no contention with quota reads) so existing tests keep their
/// O(1) borrows.
#[derive(Debug)]
pub struct MockTenant {
    tenant_id: TenantId,
    dim: u32,
    records: Vec<MockRecord>,
    quota: Mutex<Quota>,
    subscribers: Mutex<Vec<mpsc::Sender<Event>>>,
}

impl MockTenant {
    /// Construct an empty tenant.
    pub fn new(tenant_id: TenantId, embedding_dim: u32) -> Self {
        Self {
            tenant_id,
            dim: embedding_dim,
            records: Vec::new(),
            quota: Mutex::new(Quota::UNLIMITED),
            subscribers: Mutex::new(Vec::new()),
        }
    }

    /// Broadcast `event` to every live subscriber. Best effort:
    /// senders whose receivers have been dropped are pruned silently.
    fn emit(&self, event: Event) {
        let mut subs = self.subscribers.lock().expect("subscribers mutex poisoned");
        subs.retain(|s| s.send(event.clone()).is_ok());
    }

    /// Insert (or overwrite) a record.
    ///
    /// Panics if the embedding length does not match the configured
    /// dim - caller is expected to know what they are doing.
    ///
    /// Returns [`QuotaError::Exceeded`] when the write would push
    /// the tenant past a cap currently set via [`TenantQuota::set_quota`].
    /// Overwrites of an existing `record_id` are always allowed (the
    /// row already counts towards the cap).
    pub fn insert(
        &mut self,
        record_id: RecordId,
        embedding: Vec<f32>,
        shareable: bool,
        tags: Vec<String>,
        payload: Bytes,
    ) -> Result<(), QuotaError> {
        assert_eq!(
            embedding.len() as u32,
            self.dim,
            "MockTenant insert: dim mismatch"
        );
        let is_overwrite = self.records.iter().any(|r| r.record_id == record_id);
        // Quota check (only when growing the tenant - overwrites
        // never increase the live count).
        let quota = *self.quota.lock().expect("quota mutex poisoned");
        if !is_overwrite {
            let next_records = self.records.len() as u64 + 1;
            if let Some(cap) = quota.max_records
                && next_records > cap
            {
                return Err(QuotaError::Exceeded {
                    dimension: "records",
                    cap,
                    attempted: next_records,
                });
            }
            if let Some(cap) = quota.max_bytes {
                let next_bytes = self.bytes_on_disk() + estimate_size(&embedding, &tags, &payload);
                if next_bytes > cap {
                    return Err(QuotaError::Exceeded {
                        dimension: "bytes",
                        cap,
                        attempted: next_bytes,
                    });
                }
            }
        }
        let rec = MockRecord {
            record_id,
            embedding,
            shareable,
            tags,
            payload,
        };
        if let Some(slot) = self.records.iter_mut().find(|r| r.record_id == record_id) {
            *slot = rec;
        } else {
            self.records.push(rec);
        }
        self.emit(Event::RecordInserted {
            record_id,
            shareable,
        });
        Ok(())
    }

    /// Remove a record. Returns `true` if it was present (emits
    /// [`Event::RecordDeleted`] in that case); no-op on missing ids.
    pub fn delete(&mut self, record_id: RecordId) -> bool {
        let before = self.records.len();
        self.records.retain(|r| r.record_id != record_id);
        let removed = self.records.len() != before;
        if removed {
            self.emit(Event::RecordDeleted { record_id });
        }
        removed
    }

    /// Borrow the records vector.
    pub fn records(&self) -> &[MockRecord] {
        &self.records
    }
}

impl IterVectors for MockTenant {
    fn iter_vectors(&self) -> Box<dyn Iterator<Item = (RecordId, Vec<f32>)> + '_> {
        Box::new(
            self.records
                .iter()
                .map(|r| (r.record_id, r.embedding.clone())),
        )
    }

    fn record_count(&self) -> u64 {
        self.records.len() as u64
    }

    fn embedding_dim(&self) -> u32 {
        self.dim
    }
}

impl QueryFiltered for MockTenant {
    fn query_filtered(
        &self,
        embedding: &[f32],
        top_k: u32,
        filter: &dyn Filter,
    ) -> Result<Vec<Hit>, QueryError> {
        if embedding.len() as u32 != self.dim {
            return Err(QueryError::EmbeddingDimMismatch {
                expected: self.dim,
                got: embedding.len() as u32,
            });
        }
        let mut hits: Vec<Hit> = self
            .records
            .iter()
            .filter_map(|r| {
                let tags: Vec<&str> = r.tags.iter().map(String::as_str).collect();
                let meta = RecordMeta {
                    record_id: r.record_id,
                    shareable: r.shareable,
                    tags: &tags,
                };
                if !filter.accept(&meta) {
                    return None;
                }
                Some(Hit {
                    record_id: r.record_id,
                    similarity: cosine(embedding, &r.embedding),
                    payload: r.payload.clone(),
                    embedding: Some(r.embedding.clone()),
                })
            })
            .collect();
        hits.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(top_k as usize);
        Ok(hits)
    }
}

impl ReadOnlyView for MockTenant {
    fn tenant_id(&self) -> TenantId {
        self.tenant_id
    }
    fn close(self: Box<Self>) -> Result<(), OpenError> {
        Ok(())
    }
}

impl TenantStats for MockTenant {
    fn bytes_on_disk(&self) -> u64 {
        self.records
            .iter()
            .map(|r| estimate_size(&r.embedding, &r.tags, &r.payload))
            .sum()
    }
    fn record_count(&self) -> u64 {
        self.records.len() as u64
    }
    fn memory_resident(&self) -> u64 {
        // Mock is purely in-RAM, so on-disk and resident are the same.
        self.bytes_on_disk()
    }
}

impl TenantQuota for MockTenant {
    fn set_quota(&self, quota: Quota) -> Result<(), QuotaError> {
        *self.quota.lock().expect("quota mutex poisoned") = quota;
        Ok(())
    }
    fn quota(&self) -> Quota {
        *self.quota.lock().expect("quota mutex poisoned")
    }
    fn current_usage(&self) -> Usage {
        Usage {
            bytes: self.bytes_on_disk(),
            records: self.records.len() as u64,
        }
    }
}

impl TenantEvents for MockTenant {
    fn subscribe(&self, filter: EventFilter) -> EventStream {
        let (tx, rx) = mpsc::channel();
        self.subscribers
            .lock()
            .expect("subscribers mutex poisoned")
            .push(tx);
        EventStream::new(rx, filter)
    }
}

/// Best effort byte estimate of one record. Used by both the
/// `TenantStats::bytes_on_disk` count and the quota pre-check on
/// insert. Counts the embedding (f32 * dim), tag strings (UTF-8
/// length), and the payload. Mirrors the on-disk footprint of the
/// reference adapter closely enough for orchestrator decisions.
fn estimate_size(embedding: &[f32], tags: &[String], payload: &Bytes) -> u64 {
    let emb_bytes = std::mem::size_of_val(embedding) as u64;
    let tag_bytes: u64 = tags.iter().map(|t| t.len() as u64).sum();
    emb_bytes + tag_bytes + payload.len() as u64
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_mock_tenant_has_zero_count() {
        let t = MockTenant::new(TenantId::ZERO, 3);
        assert_eq!(<MockTenant as IterVectors>::record_count(&t), 0);
        assert_eq!(t.embedding_dim(), 3);
    }

    #[test]
    fn insert_and_query() {
        let mut t = MockTenant::new(TenantId::from_bytes([1; 16]), 3);
        t.insert(
            RecordId(1),
            vec![1.0, 0.0, 0.0],
            true,
            vec!["a".into()],
            Bytes::from_static(b"x"),
        )
        .unwrap();
        t.insert(
            RecordId(2),
            vec![0.0, 1.0, 0.0],
            false,
            vec!["b".into()],
            Bytes::from_static(b"y"),
        )
        .unwrap();
        let hits = t
            .query_filtered(&[1.0, 0.0, 0.0], 5, &|m: &RecordMeta<'_>| m.shareable)
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record_id, RecordId(1));
    }

    #[test]
    fn dim_mismatch_errors() {
        let t = MockTenant::new(TenantId::ZERO, 4);
        let err = t
            .query_filtered(&[1.0, 0.0], 5, &|_: &RecordMeta<'_>| true)
            .unwrap_err();
        assert!(matches!(err, QueryError::EmbeddingDimMismatch { .. }));
    }

    #[test]
    fn delete_round_trip() {
        let mut t = MockTenant::new(TenantId::ZERO, 2);
        t.insert(
            RecordId(7),
            vec![1.0, 0.0],
            true,
            vec![],
            Bytes::from_static(b""),
        )
        .unwrap();
        assert_eq!(<MockTenant as IterVectors>::record_count(&t), 1);
        assert!(t.delete(RecordId(7)));
        assert_eq!(<MockTenant as IterVectors>::record_count(&t), 0);
        assert!(!t.delete(RecordId(7)));
    }

    // ─── F.14 - TenantStats + TenantQuota ─────────────────────────

    #[test]
    fn stats_bytes_grow_with_records() {
        let mut t = MockTenant::new(TenantId::ZERO, 3);
        assert_eq!(t.bytes_on_disk(), 0);
        t.insert(
            RecordId(1),
            vec![1.0, 0.0, 0.0],
            true,
            vec!["tag".into()],
            Bytes::from_static(b"payload-1"),
        )
        .unwrap();
        let after_one = t.bytes_on_disk();
        assert!(after_one > 0);
        t.insert(
            RecordId(2),
            vec![0.0, 1.0, 0.0],
            true,
            vec!["another".into()],
            Bytes::from_static(b"payload-2"),
        )
        .unwrap();
        assert!(t.bytes_on_disk() > after_one);
    }

    #[test]
    fn stats_record_count_matches_inserts() {
        let mut t = MockTenant::new(TenantId::ZERO, 2);
        for i in 0..7 {
            t.insert(
                RecordId(i),
                vec![1.0, 0.0],
                false,
                vec![],
                Bytes::from_static(b""),
            )
            .unwrap();
        }
        assert_eq!(<MockTenant as TenantStats>::record_count(&t), 7);
    }

    #[test]
    fn default_quota_is_unlimited() {
        let t = MockTenant::new(TenantId::ZERO, 2);
        assert_eq!(t.quota(), Quota::UNLIMITED);
        assert!(t.quota().is_unlimited());
    }

    #[test]
    fn quota_records_blocks_inserts_past_cap() {
        let mut t = MockTenant::new(TenantId::ZERO, 2);
        t.set_quota(Quota {
            max_records: Some(2),
            ..Quota::UNLIMITED
        })
        .unwrap();
        t.insert(RecordId(1), vec![1.0, 0.0], false, vec![], Bytes::new())
            .unwrap();
        t.insert(RecordId(2), vec![0.0, 1.0], false, vec![], Bytes::new())
            .unwrap();
        let err = t
            .insert(RecordId(3), vec![1.0, 1.0], false, vec![], Bytes::new())
            .unwrap_err();
        match err {
            QuotaError::Exceeded {
                dimension,
                cap,
                attempted,
            } => {
                assert_eq!(dimension, "records");
                assert_eq!(cap, 2);
                assert_eq!(attempted, 3);
            }
            other => panic!("expected Exceeded, got {other:?}"),
        }
        assert_eq!(t.current_usage().records, 2);
    }

    #[test]
    fn quota_bytes_blocks_inserts_past_cap() {
        let mut t = MockTenant::new(TenantId::ZERO, 2);
        // 2 floats = 8 B; payload "x" = 1 B; tag "" = 0; per record = 9 B.
        t.set_quota(Quota {
            max_bytes: Some(20),
            ..Quota::UNLIMITED
        })
        .unwrap();
        t.insert(
            RecordId(1),
            vec![1.0, 0.0],
            false,
            vec![],
            Bytes::from_static(b"x"),
        )
        .unwrap();
        t.insert(
            RecordId(2),
            vec![0.0, 1.0],
            false,
            vec![],
            Bytes::from_static(b"y"),
        )
        .unwrap();
        // Third record would push past 20 B.
        let err = t
            .insert(
                RecordId(3),
                vec![1.0, 1.0],
                false,
                vec![],
                Bytes::from_static(b"z"),
            )
            .unwrap_err();
        assert!(matches!(
            err,
            QuotaError::Exceeded {
                dimension: "bytes",
                ..
            }
        ));
    }

    #[test]
    fn quota_overwrite_existing_record_does_not_trigger_check() {
        let mut t = MockTenant::new(TenantId::ZERO, 2);
        t.insert(
            RecordId(1),
            vec![1.0, 0.0],
            false,
            vec![],
            Bytes::from_static(b"x"),
        )
        .unwrap();
        // Cap at the current count: a new insert would fail, but an
        // overwrite of an existing row must still succeed.
        t.set_quota(Quota {
            max_records: Some(1),
            ..Quota::UNLIMITED
        })
        .unwrap();
        t.insert(
            RecordId(1),
            vec![0.0, 1.0],
            true,
            vec![],
            Bytes::from_static(b"x-updated"),
        )
        .unwrap();
        assert_eq!(t.current_usage().records, 1);
    }

    // ─── F.13 - TenantEvents ──────────────────────────────────────

    use std::time::Duration;

    #[test]
    fn subscribe_receives_record_inserted() {
        let mut t = MockTenant::new(TenantId::ZERO, 2);
        let stream = t.subscribe(EventFilter::ALL);
        t.insert(RecordId(1), vec![1.0, 0.0], true, vec![], Bytes::new())
            .unwrap();
        let ev = stream
            .recv_timeout(Duration::from_millis(100))
            .expect("event should arrive");
        match ev {
            Event::RecordInserted {
                record_id,
                shareable,
            } => {
                assert_eq!(record_id, RecordId(1));
                assert!(shareable);
            }
            other => panic!("expected RecordInserted, got {other:?}"),
        }
    }

    #[test]
    fn subscribe_receives_record_deleted() {
        let mut t = MockTenant::new(TenantId::ZERO, 2);
        t.insert(RecordId(7), vec![1.0, 0.0], false, vec![], Bytes::new())
            .unwrap();
        let stream = t.subscribe(EventFilter::ALL);
        assert!(t.delete(RecordId(7)));
        let ev = stream
            .recv_timeout(Duration::from_millis(100))
            .expect("event should arrive");
        assert!(matches!(ev, Event::RecordDeleted { record_id } if record_id == RecordId(7)));
    }

    #[test]
    fn filter_drops_unwanted_events() {
        let mut t = MockTenant::new(TenantId::ZERO, 2);
        let stream = t.subscribe(EventFilter {
            record_inserted: false,
            record_deleted: true,
            ..EventFilter::NONE
        });
        t.insert(RecordId(1), vec![1.0, 0.0], false, vec![], Bytes::new())
            .unwrap();
        // Insert is filtered out - should time out.
        assert!(stream.recv_timeout(Duration::from_millis(30)).is_err());
        // Deletes pass.
        t.delete(RecordId(1));
        let ev = stream
            .recv_timeout(Duration::from_millis(100))
            .expect("delete should pass filter");
        assert!(matches!(ev, Event::RecordDeleted { .. }));
    }

    #[test]
    fn multiple_subscribers_each_get_their_own_copy() {
        let mut t = MockTenant::new(TenantId::ZERO, 2);
        let s1 = t.subscribe(EventFilter::ALL);
        let s2 = t.subscribe(EventFilter::ALL);
        t.insert(RecordId(42), vec![1.0, 0.0], true, vec![], Bytes::new())
            .unwrap();
        let e1 = s1.recv_timeout(Duration::from_millis(100)).unwrap();
        let e2 = s2.recv_timeout(Duration::from_millis(100)).unwrap();
        assert!(matches!(e1, Event::RecordInserted { record_id, .. } if record_id == RecordId(42)));
        assert!(matches!(e2, Event::RecordInserted { record_id, .. } if record_id == RecordId(42)));
    }

    #[test]
    fn dropping_subscription_prunes_sender() {
        let mut t = MockTenant::new(TenantId::ZERO, 2);
        {
            let _stream = t.subscribe(EventFilter::ALL);
            t.insert(RecordId(1), vec![1.0, 0.0], false, vec![], Bytes::new())
                .unwrap();
            // _stream dropped at end of block, sender becomes invalid.
        }
        // Next emit prunes the dead sender; no panic, len -> 0.
        t.insert(RecordId(2), vec![0.0, 1.0], false, vec![], Bytes::new())
            .unwrap();
        let live = t.subscribers.lock().unwrap().len();
        assert_eq!(live, 0, "dead sender should have been pruned");
    }

    #[test]
    fn delete_of_missing_record_is_silent() {
        let mut t = MockTenant::new(TenantId::ZERO, 2);
        let stream = t.subscribe(EventFilter::ALL);
        assert!(!t.delete(RecordId(99)));
        assert!(stream.recv_timeout(Duration::from_millis(30)).is_err());
    }

    #[test]
    fn set_quota_below_current_usage_is_allowed_but_blocks_growth() {
        let mut t = MockTenant::new(TenantId::ZERO, 2);
        for i in 0..5 {
            t.insert(RecordId(i), vec![1.0, 0.0], false, vec![], Bytes::new())
                .unwrap();
        }
        // Existing rows untouched; further growth blocked.
        t.set_quota(Quota {
            max_records: Some(3),
            ..Quota::UNLIMITED
        })
        .unwrap();
        assert_eq!(t.current_usage().records, 5);
        let err = t
            .insert(RecordId(99), vec![1.0, 1.0], false, vec![], Bytes::new())
            .unwrap_err();
        assert!(matches!(err, QuotaError::Exceeded { .. }));
    }
}
