//! Flat-scan vector index: the M7 vector tier.
//!
//! A [`FlatIndex`] keeps every vector at full f32 precision (the source of
//! truth, used for exact re-ranking) plus, for the int8 and binary kinds, a
//! compact [`QuantizedVectors`] form rebuilt lazily after mutations. A search
//! scans the quantized form for a generous candidate set, then re-ranks those
//! candidates with exact f32 cosine. With no quantization the scan is exact
//! and no re-rank is needed.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use ahash::AHashMap;
use fixedbitset::FixedBitSet;
use ordered_float::OrderedFloat;
use skeg_simd::cosine_f32;

use crate::quant::{QuantKind, QuantizedVectors};

/// Candidate fan-out: re-rank `max(k * FANOUT, MIN_RERANK)` quantized hits.
const RERANK_FANOUT: usize = 8;
const MIN_RERANK: usize = 64;

fn rerank_width(k: usize) -> usize {
    k.saturating_mul(RERANK_FANOUT).max(MIN_RERANK)
}

/// A flat (brute-force) vector index over a single `dim` and quantization.
#[derive(Debug)]
pub struct FlatIndex {
    dim: usize,
    kind: QuantKind,
    /// Row-major f32 vectors, the precise source of truth. `n * dim` floats.
    f32_data: Vec<f32>,
    /// External id per row.
    ids: Vec<u64>,
    /// Id -> row, for overwrite and delete.
    id_to_row: AHashMap<u64, usize>,
    /// Row liveness; a cleared bit is a tombstone.
    live: FixedBitSet,
    live_count: usize,
    /// Quantized scan form; `None` means dirty (rebuilt on next search).
    quant: Option<QuantizedVectors>,
}

impl FlatIndex {
    /// Create an empty index for `dim`-dimensional vectors.
    ///
    /// # Panics
    ///
    /// Panics if `dim == 0`.
    #[must_use]
    pub fn new(dim: usize, kind: QuantKind) -> FlatIndex {
        assert!(dim > 0, "dim must be positive");
        FlatIndex {
            dim,
            kind,
            f32_data: Vec::new(),
            ids: Vec::new(),
            id_to_row: AHashMap::new(),
            live: FixedBitSet::new(),
            live_count: 0,
            quant: None,
        }
    }

    /// Vector dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Quantization kind.
    #[must_use]
    pub fn kind(&self) -> QuantKind {
        self.kind
    }

    /// Number of live vectors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.live_count
    }

    /// True if there are no live vectors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.live_count == 0
    }

    /// Insert `vector` under `id`, overwriting any existing vector for `id`.
    ///
    /// # Panics
    ///
    /// Panics if `vector.len()` does not equal the index dimension.
    pub fn insert(&mut self, id: u64, vector: &[f32]) {
        assert_eq!(vector.len(), self.dim, "vector dim mismatch");
        if let Some(&row) = self.id_to_row.get(&id) {
            self.f32_data[row * self.dim..(row + 1) * self.dim].copy_from_slice(vector);
            if !self.live.contains(row) {
                self.live.insert(row);
                self.live_count += 1;
            }
        } else {
            let row = self.ids.len();
            self.f32_data.extend_from_slice(vector);
            self.ids.push(id);
            self.id_to_row.insert(id, row);
            self.live.grow(row + 1);
            self.live.insert(row);
            self.live_count += 1;
        }
        self.quant = None; // f32 data changed: quantized form is now stale
    }

    /// Tombstone the vector for `id`. Returns `true` if it was live.
    ///
    /// The quantized form is left intact: the scan skips dead rows, so a
    /// delete needs no rebuild.
    pub fn delete(&mut self, id: u64) -> bool {
        match self.id_to_row.get(&id) {
            Some(&row) if self.live.contains(row) => {
                self.live.set(row, false);
                self.live_count -= 1;
                true
            }
            _ => false,
        }
    }

    /// True if `id` has a live vector.
    #[must_use]
    pub fn contains(&self, id: u64) -> bool {
        self.id_to_row
            .get(&id)
            .is_some_and(|&row| self.live.contains(row))
    }

    /// The full-precision f32 vector stored for `id`, if it is live.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<Vec<f32>> {
        let &row = self.id_to_row.get(&id)?;
        if self.live.contains(row) {
            Some(self.f32_data[row * self.dim..(row + 1) * self.dim].to_vec())
        } else {
            None
        }
    }

    /// Top-`k` `(id, cosine)` matches for `query`, highest cosine first.
    ///
    /// # Panics
    ///
    /// Panics if `query.len()` does not equal the index dimension.
    pub fn search(&mut self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        assert_eq!(query.len(), self.dim, "query dim mismatch");
        if k == 0 || self.live_count == 0 {
            return Vec::new();
        }
        match self.kind {
            QuantKind::F32 => self.search_exact(query, k),
            QuantKind::Int8
            | QuantKind::Binary
            | QuantKind::Pq { .. }
            | QuantKind::TurboQuant { .. } => self.search_quantized(query, k),
        }
    }

    /// Exact path: f32 cosine over every live row, no quantization.
    fn search_exact(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let top = self.select_top_k(k, |row| OrderedFloat(self.cosine(query, row)));
        top.into_iter()
            .map(|(score, row)| (self.ids[row], score.into_inner()))
            .collect()
    }

    /// Quantized path: scan the quantized proxy for candidates, re-rank exact.
    fn search_quantized(&mut self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        if self.quant.is_none() {
            self.quant = Some(QuantizedVectors::build(&self.f32_data, self.dim, self.kind));
        }
        let quant = self.quant.as_ref().expect("quant just built");
        let code = quant.quantize_query(query);

        let width = rerank_width(k).min(self.live_count);
        let candidates = self.select_top_k(width, |row| quant.proxy(row, &code));

        let mut scored: Vec<(OrderedFloat<f32>, u64)> = candidates
            .into_iter()
            .map(|(_, row)| (OrderedFloat(self.cosine(query, row)), self.ids[row]))
            .collect();
        scored.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        scored.truncate(k);
        scored
            .into_iter()
            .map(|(score, id)| (id, score.into_inner()))
            .collect()
    }

    /// Exact cosine between `query` and the f32 vector stored at `row`.
    fn cosine(&self, query: &[f32], row: usize) -> f32 {
        cosine_f32(query, &self.f32_data[row * self.dim..(row + 1) * self.dim])
    }

    /// The `k` live rows with the greatest `key`, returned best-first.
    fn select_top_k<K, F>(&self, k: usize, mut key: F) -> Vec<(K, usize)>
    where
        K: Ord,
        F: FnMut(usize) -> K,
    {
        // Bounded min-heap: the root is the weakest of the current best `k`.
        let mut heap: BinaryHeap<Reverse<(K, usize)>> = BinaryHeap::with_capacity(k + 1);
        for row in self.live.ones() {
            let entry = Reverse((key(row), row));
            if heap.len() < k {
                heap.push(entry);
            } else if heap.peek().is_some_and(|weakest| entry.0 > weakest.0) {
                heap.pop();
                heap.push(entry);
            }
        }
        let mut out: Vec<(K, usize)> = heap.into_iter().map(|Reverse(pair)| pair).collect();
        out.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Deterministic random unit-ish vectors.
    fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect())
            .collect()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na * nb == 0.0 { 0.0 } else { dot / (na * nb) }
    }

    /// Brute-force exact top-k ids by cosine, highest first.
    fn brute_force(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<u64> {
        let mut scored: Vec<(OrderedFloat<f32>, u64)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (OrderedFloat(cosine(query, v)), i as u64))
            .collect();
        scored.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        scored.into_iter().take(k).map(|(_, id)| id).collect()
    }

    #[test]
    fn vset_vsearch_k1_returns_the_query_vector() {
        let dim = 64;
        let vectors = random_vectors(50, dim, 1);
        let mut index = FlatIndex::new(dim, QuantKind::F32);
        for (i, v) in vectors.iter().enumerate() {
            index.insert(i as u64, v);
        }
        // Query equals stored vector #7: it must come back first, cosine ~1.
        let hits = index.search(&vectors[7], 1);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 7);
        assert!((hits[0].1 - 1.0).abs() < 1e-4, "cosine {}", hits[0].1);
    }

    #[test]
    fn search_excludes_tombstoned() {
        let dim = 32;
        let vectors = random_vectors(40, dim, 2);
        let mut index = FlatIndex::new(dim, QuantKind::F32);
        for (i, v) in vectors.iter().enumerate() {
            index.insert(i as u64, v);
        }
        // Vector #11 is the exact match for its own query; delete it.
        assert!(index.delete(11));
        assert!(!index.contains(11));
        let hits = index.search(&vectors[11], 10);
        assert!(
            hits.iter().all(|&(id, _)| id != 11),
            "tombstoned id returned"
        );
        assert_eq!(index.len(), 39);
    }

    #[test]
    fn overwrite_updates_vector_in_place() {
        let dim = 16;
        let mut index = FlatIndex::new(dim, QuantKind::F32);
        let v1 = vec![1.0f32; dim];
        let mut v2 = vec![-1.0f32; dim];
        v2[0] = 1.0;
        index.insert(7, &v1);
        index.insert(9, &v2);
        index.insert(7, &v2); // overwrite id 7 with v2
        assert_eq!(index.len(), 2);
        // Querying v2 must now find both id 7 and id 9 at cosine ~1.
        let hits = index.search(&v2, 2);
        let ids: Vec<u64> = hits.iter().map(|&(id, _)| id).collect();
        assert!(ids.contains(&7) && ids.contains(&9), "ids {ids:?}");
    }

    #[test]
    fn get_returns_stored_vector_or_none() {
        let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut index = FlatIndex::new(v.len(), QuantKind::F32);
        index.insert(5, &v);
        assert_eq!(index.get(5), Some(v));
        assert_eq!(index.get(404), None);
        index.delete(5);
        assert_eq!(index.get(5), None);
    }

    #[test]
    fn empty_index_and_zero_k_return_nothing() {
        let mut index = FlatIndex::new(8, QuantKind::F32);
        assert!(index.search(&[0.0; 8], 5).is_empty());
        index.insert(1, &[1.0; 8]);
        assert!(index.search(&[1.0; 8], 0).is_empty());
    }

    proptest::proptest! {
        /// The F32 flat scan must reproduce the brute-force cosine ranking.
        #[test]
        fn flat_scan_f32_matches_brute_force(
            seed in 0u64..256,
            n in 5usize..60,
            k in 1usize..10,
        ) {
            let dim = 48;
            let vectors = random_vectors(n, dim, seed);
            let query = random_vectors(1, dim, seed ^ 0xABCD).pop().unwrap();
            let mut index = FlatIndex::new(dim, QuantKind::F32);
            for (i, v) in vectors.iter().enumerate() {
                index.insert(i as u64, v);
            }
            let got: Vec<u64> = index.search(&query, k).into_iter().map(|(id, _)| id).collect();
            let want = brute_force(&vectors, &query, k.min(n));
            proptest::prop_assert_eq!(got, want);
        }
    }

    /// Int8 quantization with f32 re-rank keeps recall@10 effectively perfect.
    #[test]
    #[allow(clippy::cast_precision_loss)] // small test counts, well within f64
    fn int8_recall_at_10() {
        let dim = 128;
        let n = 1_000;
        let vectors = random_vectors(n, dim, 42);
        let mut int8 = FlatIndex::new(dim, QuantKind::Int8);
        for (i, v) in vectors.iter().enumerate() {
            int8.insert(i as u64, v);
        }
        let queries = random_vectors(50, dim, 777);
        let mut hits = 0usize;
        let mut total = 0usize;
        for q in &queries {
            let want: Vec<u64> = brute_force(&vectors, q, 10);
            let got: Vec<u64> = int8.search(q, 10).into_iter().map(|(id, _)| id).collect();
            hits += got.iter().filter(|id| want.contains(id)).count();
            total += want.len();
        }
        let recall = hits as f64 / total as f64;
        assert!(recall >= 0.99, "int8 recall@10 = {recall:.4}");
    }

    /// Binary quantization is coarse but the exact-match vector still ranks
    /// first after re-rank (its Hamming distance to itself is zero).
    #[test]
    fn binary_finds_exact_match() {
        let dim = 256;
        let vectors = random_vectors(500, dim, 9);
        let mut index = FlatIndex::new(dim, QuantKind::Binary);
        for (i, v) in vectors.iter().enumerate() {
            index.insert(i as u64, v);
        }
        for probe in [0usize, 123, 499] {
            let hits = index.search(&vectors[probe], 5);
            assert_eq!(hits[0].0, probe as u64, "binary search missed exact match");
            assert!((hits[0].1 - 1.0).abs() < 1e-4);
        }
    }
}
