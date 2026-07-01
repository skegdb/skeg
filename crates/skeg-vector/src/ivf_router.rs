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
    pub fn build(
        row: &dyn Fn(u32) -> Vec<f32>,
        n: u32,
        dim: usize,
        n_cells: usize,
        iters: usize,
    ) -> IvfRouter {
        assert!(n > 0 && dim > 0, "ivf build needs vectors");
        let n_cells = n_cells.min(n as usize).max(1);
        let step = (n as usize / n_cells).max(1);
        let mut cent = vec![0.0f32; n_cells * dim];
        for c in 0..n_cells {
            let src = row(((c * step) as u32) % n);
            cent[c * dim..c * dim + dim].copy_from_slice(&src);
        }
        let mut cell_of = vec![0u32; n as usize];
        for it in 0..iters {
            let mut sums = vec![0.0f32; n_cells * dim];
            let mut counts = vec![0u32; n_cells];
            for r in 0..n {
                let v = row(r);
                let c = nearest(&cent, n_cells, dim, &v);
                cell_of[r as usize] = c as u32;
                let s = &mut sums[c * dim..c * dim + dim];
                for j in 0..dim {
                    s[j] += v[j];
                }
                counts[c] += 1;
            }
            // Update centroids (skip empty cells). Last iter only assigns.
            if it + 1 < iters {
                for c in 0..n_cells {
                    if counts[c] > 0 {
                        let inv = 1.0 / counts[c] as f32;
                        for j in 0..dim {
                            cent[c * dim + j] = sums[c * dim + j] * inv;
                        }
                    }
                }
            }
        }
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
        let ivf = IvfRouter::build(&row, n, dim, IvfRouter::cells_for(n as usize), 6);
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
        let row = |r: u32| spread(r, 8);
        let ivf = IvfRouter::build(&row, 500, 8, 64, 4);
        let s: Vec<u64> = (0..50).collect();
        assert_eq!(ivf.probe(&spread(1, 8), &s, 1000).len(), 50);
    }
}
