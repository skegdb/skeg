//! Compile-time check: third parties can implement the rigging traits
//! on their own types without depending on any adapter crate. If this
//! test compiles, the trait set is consumable by an alternative engine
//! (in-memory mock here; a graph or temporal engine would look similar).

use bytes::Bytes;
use skeg_rigging::prelude::*;

struct InMemTenant {
    id: TenantId,
    dim: u32,
    vectors: Vec<(RecordId, Vec<f32>, RecordMetaOwned)>,
}

struct RecordMetaOwned {
    shareable: bool,
    tags: Vec<String>,
    payload: Bytes,
}

impl IterVectors for InMemTenant {
    fn iter_vectors(&self) -> Box<dyn Iterator<Item = (RecordId, Vec<f32>)> + '_> {
        Box::new(self.vectors.iter().map(|(id, v, _)| (*id, v.clone())))
    }

    fn record_count(&self) -> u64 {
        self.vectors.len() as u64
    }

    fn embedding_dim(&self) -> u32 {
        self.dim
    }
}

impl QueryFiltered for InMemTenant {
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
            .vectors
            .iter()
            .filter_map(|(id, v, m)| {
                let tags: Vec<&str> = m.tags.iter().map(String::as_str).collect();
                let meta = RecordMeta {
                    record_id: *id,
                    shareable: m.shareable,
                    tags: &tags,
                };
                if !filter.accept(&meta) {
                    return None;
                }
                let sim = dot(embedding, v);
                Some(Hit {
                    record_id: *id,
                    similarity: sim,
                    payload: m.payload.clone(),
                    embedding: Some(v.clone()),
                })
            })
            .collect();
        hits.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        hits.truncate(top_k as usize);
        Ok(hits)
    }
}

impl ReadOnlyView for InMemTenant {
    fn tenant_id(&self) -> TenantId {
        self.id
    }
    fn close(self: Box<Self>) -> Result<(), OpenError> {
        Ok(())
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[test]
fn third_party_impl_satisfies_traits() {
    let v = InMemTenant {
        id: TenantId::from_bytes([9; 16]),
        dim: 3,
        vectors: vec![
            (
                RecordId(1),
                vec![1.0, 0.0, 0.0],
                RecordMetaOwned {
                    shareable: true,
                    tags: vec!["a".into()],
                    payload: Bytes::from_static(b"one"),
                },
            ),
            (
                RecordId(2),
                vec![0.0, 1.0, 0.0],
                RecordMetaOwned {
                    shareable: false,
                    tags: vec!["b".into()],
                    payload: Bytes::from_static(b"two"),
                },
            ),
        ],
    };
    assert_eq!(v.record_count(), 2);
    assert_eq!(v.embedding_dim(), 3);

    let hits = v
        .query_filtered(&[1.0, 0.0, 0.0], 10, &|m: &RecordMeta<'_>| m.shareable)
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].record_id, RecordId(1));

    // ReadOnlyView object safety.
    let view: Box<dyn ReadOnlyView> = Box::new(v);
    assert_ne!(view.tenant_id(), TenantId::ZERO);
    view.close().unwrap();
}

#[test]
fn open_readonly_placeholder_returns_not_found() {
    match open_readonly(std::path::Path::new("/nonexistent")) {
        Ok(_) => panic!("expected error from placeholder open_readonly"),
        Err(e) => assert!(matches!(e, OpenError::NotFound)),
    }
}
