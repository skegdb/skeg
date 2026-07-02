//! Coarse IVF router: the "cells" branch of the hybrid filtered search. A cheap
//! k-means partition (a few thousand centroids, no per-vector graph) that turns a
//! sparse filtered search from "score every match" (O(|S|)) into "score only the
//! matches in the query-nearest cells that CONTAIN matches" - sub-linear scoring.
//!
//! It is a smart candidate NARROWER, not a full index: given the filter's sorted
//! matching id list `s`, [`probe`](IvfRouter::probe) returns a shortlist ⊂ s that
//! the caller then proxy-scores + f32-reranks (reusing `score_ids_quantized`).
//!
//! Predicate-aware by construction: it ranks the cells that actually hold `s`
//! members (bucketed from `s`, O(|s|) cheap lookups), not the globally nearest
//! cells - so a filter whose matches cluster AWAY from the query is still found
//! (validated: correlated 1% recall 0.69 -> ~1.0 vs query-centric probing).

use ahash::AHashMap;
use rayon::prelude::*;
use skeg_simd::cosine_f32;

/// A coarse k-means partition + per-vector cell assignment. RAM-resident and
/// small: `n_cells * dim` f32 centroids + one u32 per vector.
pub struct IvfRouter {
    centroids: Vec<f32>, // n_cells * dim, row-major
    dim: usize,
    n_cells: usize,
    /// Cell id for each vector row (index = vector row).
    cell_of: Vec<u32>,
}

impl IvfRouter {
    /// A sensible cell count for `n` vectors: ~√n, clamped to [64, 65536].
    #[must_use]
    pub fn cells_for(n: usize) -> usize {
        (n as f64).sqrt().round().clamp(64.0, 65536.0) as usize
    }

    /// Build the router over `n` row-major unit vectors read via `row`. Runs
    /// `iters` Lloyd iterations (cosine assignment) seeded from evenly-spaced
    /// samples, then assigns every vector to its nearest centroid.
    ///
    /// # Panics
    /// Panics if `n == 0` or `dim == 0`.
    #[must_use]
    pub fn build(data: &[f32], n: u32, dim: usize, n_cells: usize, iters: usize) -> IvfRouter {
        assert!(n > 0 && dim > 0, "ivf build needs vectors");
        assert_eq!(data.len(), n as usize * dim, "data/n/dim mismatch");
        let n_cells = n_cells.min(n as usize).max(1);
        let vec_at = |r: usize| &data[r * dim..r * dim + dim];
        // k-means on a bounded SAMPLE (Lloyd iters); a full-corpus refit adds
        // little for coarse routing and would dominate the build. The final
        // assignment (below) still covers every vector.
        let sample: usize = (n as usize).min(50_000).max(n_cells);
        let sstep = (n as usize / sample).max(1);
        let step = (n as usize / n_cells).max(1);
        let mut cent = vec![0.0f32; n_cells * dim];
        for c in 0..n_cells {
            cent[c * dim..c * dim + dim].copy_from_slice(vec_at((c * step) % n as usize));
        }
        for _ in 0..iters {
            // Parallel assign+accumulate over the sample.
            let (sums, counts) = (0..sample)
                .into_par_iter()
                .map(|si| {
                    let r = (si * sstep) % n as usize;
                    let c = nearest(&cent, n_cells, dim, vec_at(r));
                    (c, r)
                })
                .fold(
                    || (vec![0.0f32; n_cells * dim], vec![0u32; n_cells]),
                    |(mut s, mut cnt), (c, r)| {
                        let v = vec_at(r);
                        for j in 0..dim {
                            s[c * dim + j] += v[j];
                        }
                        cnt[c] += 1;
                        (s, cnt)
                    },
                )
                .reduce(
                    || (vec![0.0f32; n_cells * dim], vec![0u32; n_cells]),
                    |(mut sa, mut ca), (sb, cb)| {
                        for i in 0..n_cells * dim {
                            sa[i] += sb[i];
                        }
                        for c in 0..n_cells {
                            ca[c] += cb[c];
                        }
                        (sa, ca)
                    },
                );
            for c in 0..n_cells {
                if counts[c] > 0 {
                    let inv = 1.0 / counts[c] as f32;
                    for j in 0..dim {
                        cent[c * dim + j] = sums[c * dim + j] * inv;
                    }
                }
            }
        }
        // Final assignment of EVERY vector, in parallel.
        let cell_of: Vec<u32> = (0..n as usize)
            .into_par_iter()
            .map(|r| nearest(&cent, n_cells, dim, vec_at(r)) as u32)
            .collect();
        IvfRouter {
            centroids: cent,
            dim,
            n_cells,
            cell_of,
        }
    }

    /// Predicate-aware probe. `s` = the filter's SORTED matching ids (external =
    /// vector rows here). Returns a shortlist ⊂ s: the `s` members that live in
    /// the query-nearest cells CONTAINING `s`, gathered until `budget` is reached.
    /// The caller proxy-scores + reranks the shortlist. Falls back to all of `s`
    /// if `budget >= |s|`.
    #[must_use]
    pub fn probe(&self, query: &[f32], s: &[u64], budget: usize) -> Vec<u64> {
        if s.len() <= budget {
            return s.to_vec();
        }
        // Bucket s into the cells that hold it (O(|s|) cheap lookups).
        let mut by_cell: AHashMap<u32, Vec<u64>> = AHashMap::new();
        for &id in s {
            let c = self.cell_of[id as usize];
            by_cell.entry(c).or_default().push(id);
        }
        // Rank the S-cells by query-centroid cosine (highest = nearest).
        let mut cells: Vec<(f32, u32)> = by_cell
            .keys()
            .map(|&c| {
                let ce = &self.centroids[c as usize * self.dim..c as usize * self.dim + self.dim];
                (cosine_f32(query, ce), c)
            })
            .collect();
        cells.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        // Gather members from nearest S-cells until the budget is met.
        let mut out: Vec<u64> = Vec::with_capacity(budget + s.len() / self.n_cells.max(1));
        for (_, c) in cells {
            out.extend_from_slice(&by_cell[&c]);
            if out.len() >= budget {
                break;
            }
        }
        out
    }

    /// Cells in the partition.
    #[must_use]
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }

    /// Number of assigned vectors (rows).
    #[must_use]
    pub fn len(&self) -> usize {
        self.cell_of.len()
    }

    /// True if no vectors are assigned.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cell_of.is_empty()
    }

    /// Serialise to bytes: `[n_cells u32][dim u32][n u32][centroids f32...][cell_of u32...]`.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let n = self.cell_of.len();
        let mut b = Vec::with_capacity(12 + self.centroids.len() * 4 + n * 4);
        b.extend_from_slice(&(self.n_cells as u32).to_le_bytes());
        b.extend_from_slice(&(self.dim as u32).to_le_bytes());
        b.extend_from_slice(&(n as u32).to_le_bytes());
        for &x in &self.centroids {
            b.extend_from_slice(&x.to_le_bytes());
        }
        for &c in &self.cell_of {
            b.extend_from_slice(&c.to_le_bytes());
        }
        b
    }

    /// Inverse of [`to_bytes`](Self::to_bytes). Returns `None` on a malformed or
    /// truncated buffer.
    #[must_use]
    pub fn from_bytes(b: &[u8]) -> Option<IvfRouter> {
        if b.len() < 12 {
            return None;
        }
        let rd = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as usize;
        let (n_cells, dim, n) = (rd(0), rd(4), rd(8));
        let cent_len = n_cells * dim;
        let need = 12 + cent_len * 4 + n * 4;
        if b.len() != need {
            return None;
        }
        let centroids: Vec<f32> = (0..cent_len)
            .map(|i| {
                let o = 12 + i * 4;
                f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
            })
            .collect();
        let base = 12 + cent_len * 4;
        let cell_of: Vec<u32> = (0..n)
            .map(|i| {
                let o = base + i * 4;
                u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
            })
            .collect();
        Some(IvfRouter {
            centroids,
            dim,
            n_cells,
            cell_of,
        })
    }
}

fn nearest(cent: &[f32], n_cells: usize, dim: usize, v: &[f32]) -> usize {
    let mut best = 0;
    let mut bd = f32::NEG_INFINITY;
    for c in 0..n_cells {
        let d = cosine_f32(v, &cent[c * dim..c * dim + dim]);
        if d > bd {
            bd = d;
            best = c;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spread(r: u32, dim: usize) -> Vec<f32> {
        // Deterministic pseudo-random unit vector (no rng dep).
        let mut v: Vec<f32> = (0..dim)
            .map(|j| (((r as usize * 131 + j * 977) % 1000) as f32) / 1000.0 - 0.5)
            .collect();
        let nrm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        v.iter_mut().for_each(|x| *x /= nrm);
        v
    }

    #[test]
    fn probe_shortlist_recovers_filtered_top_k() {
        // Spread vectors + a scattered filter (every 4th id). The router's
        // shortlist (⊂ s) must recover most of the brute filtered top-10.
        let dim = 16;
        let n = 4000u32;
        let row = |r: u32| spread(r, dim);
        let data: Vec<f32> = (0..n).flat_map(|r| spread(r, dim)).collect();
        let ivf = IvfRouter::build(&data, n, dim, IvfRouter::cells_for(n as usize), 6);
        let s: Vec<u64> = (0..n as u64).filter(|id| id % 4 == 0).collect(); // 25%
        let query = spread(123_456, dim);
        let shortlist = ivf.probe(&query, &s, 120);
        assert!(shortlist.iter().all(|id| id % 4 == 0), "shortlist ⊂ s");

        let mut truth: Vec<(f32, u64)> = s
            .iter()
            .map(|&id| (cosine_f32(&query, &row(id as u32)), id))
            .collect();
        truth.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        let top10: std::collections::HashSet<u64> =
            truth.iter().take(10).map(|&(_, id)| id).collect();
        let sl: std::collections::HashSet<u64> = shortlist.into_iter().collect();
        let hit = top10.iter().filter(|id| sl.contains(id)).count();
        assert!(
            hit >= 7,
            "recovered {hit}/10 (shortlist should hold the near cells' matches)"
        );
    }

    #[test]
    fn probe_returns_all_when_budget_exceeds_s() {
        let data: Vec<f32> = (0..500u32).flat_map(|r| spread(r, 8)).collect();
        let ivf = IvfRouter::build(&data, 500, 8, 64, 4);
        let s: Vec<u64> = (0..50).collect();
        assert_eq!(ivf.probe(&spread(1, 8), &s, 1000).len(), 50);
    }
}
