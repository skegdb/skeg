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
    /// Block-interleaved TurboQuant 4-bit codes, built lazily by
    /// [`FlatIndex::search_block_tq4`] and reused across subsequent
    /// queries. Invalidated alongside `quant` whenever the index
    /// mutates.
    block_codes_tq4: Option<Vec<u8>>,
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
            block_codes_tq4: None,
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
        self.block_codes_tq4 = None; // and the interleaved cache too
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
        scored.sort_unstable_by_key(|x| std::cmp::Reverse(x.0));
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

    /// Search using the block-32 SIMD scoring path for TurboQuant
    /// 4-bit. Returns the top-`k` `(id, cosine)` pairs after a
    /// candidate-and-rerank pass: the block kernel preselects
    /// `rerank_width(k)` candidates by their proxy score, then each
    /// candidate is reranked with the exact f32 cosine against the
    /// stored f32 vectors. Equivalent semantics to
    /// [`search_quantized`](Self::search_quantized) for `bits = 4`,
    /// just routed through the block-32 layout.
    ///
    /// Returns `None` (delegating to the row-major path is the
    /// caller's responsibility) when the tier is not
    /// `TurboQuant { bits = 4 }`.
    ///
    /// # Panics
    ///
    /// Panics if `query.len() != dim`.
    pub fn search_block_tq4(&mut self, query: &[f32], k: usize) -> Option<Vec<(u64, f32)>> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        #[cfg(target_arch = "aarch64")]
        use skeg_simd::tq4_block32_score_u8_neon;
        #[cfg(not(target_arch = "aarch64"))]
        use skeg_simd::tq4_block32_score_u8_scalar;
        use skeg_simd::{
            BLOCK as TQ4_BLOCK, build_tq4_lut_f32, interleave_tq4_codes, quantize_tq4_lut_u8,
        };

        assert_eq!(query.len(), self.dim, "query dim mismatch");
        // Block path is TurboQuant-bits-4-only. Pre-empt the build
        // call for the F32 tier, which would panic in
        // `QuantizedVectors::build`.
        if !matches!(self.kind, QuantKind::TurboQuant { bits: 4 }) {
            return None;
        }
        if k == 0 || self.live_count == 0 {
            return Some(Vec::new());
        }

        if self.quant.is_none() {
            self.quant = Some(QuantizedVectors::build(&self.f32_data, self.dim, self.kind));
        }
        let quant = self.quant.as_ref().expect("quant just built");
        if !quant.supports_tq4_block() {
            return None;
        }

        // Lazy interleave: rebuild only after `insert` invalidates
        // the cache. The block kernel iterates byte-group-major so
        // every byte is touched exactly once per query - sequential
        // memory pattern at minimal compute cost.
        if self.block_codes_tq4.is_none() {
            let n = quant.len();
            let n_blocks = n / TQ4_BLOCK;
            let n_groups = self.dim / 2;
            let row_codes = quant.tq4_codes().expect("guarded above");
            let mut blocks = vec![0u8; n_blocks * n_groups * TQ4_BLOCK];
            for b in 0..n_blocks {
                let row_refs: Vec<&[u8]> = (0..TQ4_BLOCK)
                    .map(|v| {
                        let row = b * TQ4_BLOCK + v;
                        &row_codes[row * n_groups..(row + 1) * n_groups]
                    })
                    .collect();
                let block_slice =
                    &mut blocks[b * n_groups * TQ4_BLOCK..(b + 1) * n_groups * TQ4_BLOCK];
                interleave_tq4_codes(&row_refs, self.dim, block_slice);
            }
            self.block_codes_tq4 = Some(blocks);
        }
        let blocks = self.block_codes_tq4.as_ref().expect("just built");
        let n = quant.len();
        let n_blocks = n / TQ4_BLOCK;
        let n_groups = self.dim / 2;
        let block_stride = n_groups * TQ4_BLOCK;

        // Per-query LUT pre-compute: unit-normalise + rotate the
        // query so the inner product over the rotated codes tracks
        // cosine.
        let mut unit = vec![0.0f32; self.dim];
        let norm = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for (o, &x) in unit.iter_mut().zip(query) {
                *o = x / norm;
            }
        }
        let q_rot = quant.tq4_rotate_query(&unit).expect("guarded above");
        let centroids_slice = quant.tq4_centroids().expect("guarded above");
        let mut centroids: [f32; 16] = [0.0; 16];
        centroids.copy_from_slice(&centroids_slice[..16]);
        let scales = quant.tq4_scales().expect("guarded above");

        let mut lut_f32 = vec![0.0f32; n_groups * 32];
        let mut lut_u8 = vec![0u8; n_groups * 32];
        build_tq4_lut_f32(&q_rot, &centroids, self.dim, &mut lut_f32);
        let (inv_scale, bias_per_group) = quantize_tq4_lut_u8(&lut_f32, &mut lut_u8);

        // Pre-select with the block kernel; the candidate pool size
        // matches the row-major path's `rerank_width` so recall is
        // comparable. Min-heap keyed on fixed-point score: `peek`
        // returns the weakest kept candidate, replaced when a new
        // candidate scores higher.
        let rerank = rerank_width(k).min(n);
        let mut cands: BinaryHeap<Reverse<(i64, usize)>> = BinaryHeap::with_capacity(rerank + 1);
        let scale_to_i64 = 1_000_000.0_f32;
        let mut block_out = [0.0f32; TQ4_BLOCK];
        let push_cand =
            |cands: &mut BinaryHeap<Reverse<(i64, usize)>>, score_i64: i64, row: usize| {
                let entry = Reverse((score_i64, row));
                if cands.len() < rerank {
                    cands.push(entry);
                } else if let Some(Reverse((min_score, _))) = cands.peek()
                    && score_i64 > *min_score
                {
                    cands.pop();
                    cands.push(entry);
                }
            };
        for b in 0..n_blocks {
            let block_slice = &blocks[b * block_stride..(b + 1) * block_stride];
            #[cfg(target_arch = "aarch64")]
            tq4_block32_score_u8_neon(
                block_slice,
                &lut_u8,
                inv_scale,
                bias_per_group,
                self.dim,
                &mut block_out,
            );
            #[cfg(not(target_arch = "aarch64"))]
            tq4_block32_score_u8_scalar(
                block_slice,
                &lut_u8,
                inv_scale,
                bias_per_group,
                self.dim,
                &mut block_out,
            );
            for lane in 0..TQ4_BLOCK {
                let row = b * TQ4_BLOCK + lane;
                if !self.live.contains(row) {
                    continue;
                }
                let score = block_out[lane] * scales[row];
                let score_i64 = (score * scale_to_i64) as i64;
                push_cand(&mut cands, score_i64, row);
            }
        }
        // Tail: rows in `[n_blocks * BLOCK, n)` scored via the
        // existing row-major proxy so the candidate set covers the
        // full corpus when N % 32 != 0.
        let tail_start = n_blocks * TQ4_BLOCK;
        if tail_start < n {
            let code = quant.quantize_query(query);
            for row in tail_start..n {
                if !self.live.contains(row) {
                    continue;
                }
                let proxy = quant.proxy(row, &code);
                push_cand(&mut cands, i64::from(proxy), row);
            }
        }

        // Rerank with exact f32 cosine against the source-of-truth
        // vectors. Same pattern as the row-major search path.
        let mut scored: Vec<(OrderedFloat<f32>, u64)> = cands
            .into_iter()
            .map(|Reverse((_, row))| (OrderedFloat(self.cosine(query, row)), self.ids[row]))
            .collect();
        scored.sort_unstable_by_key(|x| std::cmp::Reverse(x.0));
        scored.truncate(k);
        Some(
            scored
                .into_iter()
                .map(|(score, id)| (id, score.into_inner()))
                .collect(),
        )
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
        // `K` is generic and only `Ord`-bound; `sort_unstable_by_key`
        // would require an extra `Copy` or `Clone` bound. Direct
        // comparator is the cleanest option.
        #[allow(clippy::unnecessary_sort_by)]
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
        scored.sort_unstable_by_key(|x| std::cmp::Reverse(x.0));
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

    /// TurboQuant 4-bit block-32 path: same semantic guarantee as the
    /// row-major search (exact-match query returns its own id at top-1)
    /// and recall@10 stays >= 0.95 across a 256-vector corpus.
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn tq4_block_search_recovers_exact_match_and_high_recall() {
        let dim = 64;
        let n = 256;
        let vectors = random_vectors(n, dim, 4242);
        let mut index = FlatIndex::new(dim, QuantKind::TurboQuant { bits: 4 });
        for (i, v) in vectors.iter().enumerate() {
            index.insert(i as u64, v);
        }
        for probe in [0usize, 33, 128, 200] {
            let hits = index
                .search_block_tq4(&vectors[probe], 5)
                .expect("tq4 path available");
            assert_eq!(hits[0].0, probe as u64, "block tq4 missed exact match");
            assert!((hits[0].1 - 1.0).abs() < 1e-3, "cosine self {}", hits[0].1);
        }
        // recall@10 vs brute force across a query set.
        let queries = random_vectors(20, dim, 8888);
        let mut hits = 0usize;
        let mut total = 0usize;
        for q in &queries {
            let want = brute_force(&vectors, q, 10);
            let got: Vec<u64> = index
                .search_block_tq4(q, 10)
                .expect("tq4 path available")
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            hits += got.iter().filter(|id| want.contains(id)).count();
            total += want.len();
        }
        let recall = hits as f64 / total as f64;
        assert!(recall >= 0.95, "tq4 block recall@10 = {recall:.4}");
    }

    /// `search_block_tq4` must refuse to handle non-TurboQuant tiers
    /// (returns None so the caller falls back to row-major).
    #[test]
    fn tq4_block_returns_none_for_non_tq4_tier() {
        let dim = 32;
        let vectors = random_vectors(64, dim, 1);
        let mut index = FlatIndex::new(dim, QuantKind::F32);
        for (i, v) in vectors.iter().enumerate() {
            index.insert(i as u64, v);
        }
        assert!(index.search_block_tq4(&vectors[0], 5).is_none());
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
