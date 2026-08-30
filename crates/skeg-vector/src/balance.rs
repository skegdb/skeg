//! Balanced cosine k-means for semantic shard assignment (SPANN-style):
//! Lloyd iterations where the assignment cost is `-cos(x, c_j) + lambda *
//! size_j / target`, so a growing cluster prices itself out and the
//! partition stays foldable-sized everywhere. Pure and deterministic
//! (seeded): the reshard job feeds it a sample, ships the centroids.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use skeg_simd::cosine_f32;

/// Train `k` balanced centroids over `n` row-major unit vectors.
///
/// `lambda` prices cluster growth: 0 is plain k-means, 0.1-0.3 measured
/// (SPANN) to hold skew near 1 with little reconstruction loss. Returns
/// row-major centroids (`k * dim`), unit-normalised.
///
/// # Panics
///
/// Panics if `n == 0`, `dim == 0`, `k == 0` or `k > n`.
#[must_use]
pub fn balanced_kmeans(
    data: &[f32],
    n: usize,
    dim: usize,
    k: usize,
    lambda: f32,
    iters: usize,
    seed: u64,
) -> Vec<f32> {
    assert!(n > 0 && dim > 0 && k > 0 && k <= n, "bad kmeans shape");
    assert_eq!(data.len(), n * dim, "data/n/dim mismatch");
    let row = |i: usize| &data[i * dim..(i + 1) * dim];

    // Seed from a shuffled sample so two nearby duplicates rarely both seed.
    let mut order: Vec<usize> = (0..n).collect();
    order.shuffle(&mut StdRng::seed_from_u64(seed));
    let mut centroids: Vec<f32> = Vec::with_capacity(k * dim);
    for &i in order.iter().take(k) {
        centroids.extend_from_slice(row(i));
    }
    normalize_rows(&mut centroids, k, dim);

    let target = n as f32 / k as f32;
    let mut assign = vec![0usize; n];
    for _ in 0..iters {
        // Sequential greedy assignment with live sizes: the balance term
        // sees the sizes as they grow, which is what actually balances
        // (a post-hoc penalty on final sizes does not).
        let mut sizes = vec![0f32; k];
        for (i, slot) in assign.iter_mut().enumerate() {
            let x = row(i);
            let mut best = 0usize;
            let mut best_cost = f32::INFINITY;
            for j in 0..k {
                let c = &centroids[j * dim..(j + 1) * dim];
                let cost = -cosine_f32(x, c) + lambda * sizes[j] / target;
                if cost < best_cost {
                    best_cost = cost;
                    best = j;
                }
            }
            *slot = best;
            sizes[best] += 1.0;
        }
        // Recentre.
        let mut sums = vec![0f32; k * dim];
        let mut counts = vec![0usize; k];
        for (i, &j) in assign.iter().enumerate() {
            counts[j] += 1;
            for (s, &x) in sums[j * dim..(j + 1) * dim].iter_mut().zip(row(i)) {
                *s += x;
            }
        }
        for j in 0..k {
            if counts[j] > 0 {
                centroids[j * dim..(j + 1) * dim].copy_from_slice(&sums[j * dim..(j + 1) * dim]);
            }
        }
        normalize_rows(&mut centroids, k, dim);
    }
    centroids
}

/// Assign one vector: the nearest centroid by cosine (no balance term - at
/// query/write time the partition is what it is).
#[must_use]
pub fn nearest_centroid(x: &[f32], centroids: &[f32], k: usize, dim: usize) -> usize {
    let mut best = 0usize;
    let mut best_cos = f32::NEG_INFINITY;
    for j in 0..k {
        let c = cosine_f32(x, &centroids[j * dim..(j + 1) * dim]);
        if c > best_cos {
            best_cos = c;
            best = j;
        }
    }
    best
}

fn normalize_rows(v: &mut [f32], k: usize, dim: usize) {
    for j in 0..k {
        let r = &mut v[j * dim..(j + 1) * dim];
        let norm = r.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in r {
                *x /= norm;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `masses[c]` rows around orthogonal centre `e_c`, unit-normalised.
    fn imbalanced_clusters(masses: &[usize], dim: usize, spread: f32, seed: u64) -> Vec<f32> {
        use rand::Rng;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut data = Vec::new();
        for (c, &m) in masses.iter().enumerate() {
            for _ in 0..m {
                let mut v: Vec<f32> =
                    (0..dim).map(|_| rng.random_range(-spread..spread)).collect();
                v[c] += 1.0;
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                for x in &mut v {
                    *x /= norm;
                }
                data.extend_from_slice(&v);
            }
        }
        data
    }

    /// On deliberately imbalanced natural clusters, lambda holds the
    /// partition near even while plain k-means follows the imbalance.
    #[test]
    fn lambda_holds_skew_near_one_on_imbalanced_clusters() {
        let dim = 16;
        let masses = [600usize, 200, 100];
        let n: usize = masses.iter().sum();
        let data = imbalanced_clusters(&masses, dim, 0.25, 5);
        let k = 4;
        let skew = |cent: &[f32]| {
            let mut sizes = vec![0usize; k];
            for i in 0..n {
                sizes[nearest_centroid(&data[i * dim..(i + 1) * dim], cent, k, dim)] += 1;
            }
            *sizes.iter().max().unwrap() as f32 / (n as f32 / k as f32)
        };
        let plain = balanced_kmeans(&data, n, dim, k, 0.0, 12, 42);
        let balanced = balanced_kmeans(&data, n, dim, k, 0.4, 12, 42);
        assert!(
            skew(&balanced) < skew(&plain),
            "lambda must reduce skew: balanced {} vs plain {}",
            skew(&balanced),
            skew(&plain)
        );
        assert!(
            skew(&balanced) <= 1.6,
            "balanced partition skew {} too high",
            skew(&balanced)
        );
    }

    /// Deterministic: same inputs, same centroids.
    #[test]
    fn training_is_deterministic() {
        let data = imbalanced_clusters(&[100, 100, 100, 100], 8, 0.1, 7);
        let a = balanced_kmeans(&data, 400, 8, 4, 0.2, 8, 9);
        let b = balanced_kmeans(&data, 400, 8, 4, 0.2, 8, 9);
        assert_eq!(a, b);
    }
}
