//! Degree histogram of the built Vamana graph.
//!
//! Not a Criterion bench: a reporting harness (`harness = false`).
//!
//! Compacting the graph layout is "guaranteed" as a mechanism, but its
//! magnitude depends on the real degree distribution. The current node is
//! fixed-width: `degree: u32 + [VecId; 64]` = 260 bytes, independent of the
//! actual degree. If most nodes are close to R=64 the packing yields little;
//! the real saving comes from 24-bit neighbor ids.
//!
//! This harness builds graphs (real mxbai 10K + synthetic at various scales),
//! prints the degree histogram and the bytes/node of alternative layouts.

#![allow(clippy::cast_precision_loss)]

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use skeg_vector::{VamanaConfig, VamanaIndex};

const CORPUS_NPY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../bench-compare/embeddings_cache/corpus_mxbai-embed-large_10000.npy"
);

/// Current node: `degree: u32` + `[VecId; 64]` = 260 bytes, fixed-width.
const FIXED_NODE_BYTES: usize = 260;

fn load_npy(path: &str) -> Option<(Vec<f32>, usize, usize)> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        return None;
    }
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + header_len]).ok()?;
    let sh = header.find("'shape':")?;
    let lp = header[sh..].find('(')? + sh + 1;
    let rp = header[lp..].find(')')? + lp;
    let dims: Vec<usize> = header[lp..rp]
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if dims.len() != 2 {
        return None;
    }
    let data: Vec<f32> = bytes[10 + header_len..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Some((data, dims[0], dims[1]))
}

/// One standard-normal sample (Box-Muller).
fn gaussian(rng: &mut StdRng) -> f32 {
    let u1 = rng.random::<f32>().max(1e-9);
    let u2 = rng.random::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// `n` vectors uniform on the unit sphere - the validated proxy for real
/// isotropic embeddings (real mxbai behaves like uniform, not clustered).
fn uniform_sphere(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n * dim);
    for _ in 0..n {
        let mut v: Vec<f32> = (0..dim).map(|_| gaussian(&mut rng)).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for x in &mut v {
            *x /= norm;
        }
        out.extend(v);
    }
    out
}

/// `n` clustered vectors: ~n/100 centres, each point a centre plus noise.
fn clustered(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let nc = (n / 100).max(8);
    let centers: Vec<Vec<f32>> = (0..nc)
        .map(|_| (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect())
        .collect();
    let mut out = Vec::with_capacity(n * dim);
    for _ in 0..n {
        let c = &centers[rng.random_range(0..nc)];
        for &x in c {
            out.push(x + rng.random_range(-0.15..0.15));
        }
    }
    out
}

/// Print the degree histogram of `index` plus the packed-layout byte models.
fn report(label: &str, index: &VamanaIndex) {
    let hist = index.degree_histogram();
    let n = index.len();
    let max_r = hist.len() - 1;

    let total_edges: u64 = hist
        .iter()
        .enumerate()
        .map(|(d, &c)| d as u64 * u64::from(c))
        .sum();
    let mean = total_edges as f64 / n as f64;
    let at_max = hist[max_r];
    let isolated = hist[0];

    // median degree
    let mut cum = 0u32;
    let mut median = 0usize;
    for (d, &c) in hist.iter().enumerate() {
        cum += c;
        if (cum as usize) * 2 >= n {
            median = d;
            break;
        }
    }

    println!("\n== {label} (N={n}, R={max_r}) ==");
    println!(
        "  mean degree {mean:.1}  median {median}  at-R {} ({:.1}%)  isolated {isolated}",
        at_max,
        at_max as f64 / n as f64 * 100.0,
    );

    // bucketed histogram, 8 buckets across 0..=R
    let bucket = (max_r + 1).div_ceil(8);
    println!("  {:>10}{:>10}{:>9}", "degree", "nodes", "share");
    for b in 0..8 {
        let lo = b * bucket;
        let hi = ((b + 1) * bucket - 1).min(max_r);
        if lo > max_r {
            break;
        }
        let c: u32 = hist[lo..=hi].iter().sum();
        if c == 0 {
            continue;
        }
        println!(
            "  {:>10}{:>10}{:>8.1}%",
            format!("{lo}-{hi}"),
            c,
            c as f64 / n as f64 * 100.0,
        );
    }

    // byte models
    let fixed = n * FIXED_NODE_BYTES;
    // CSR: offset array (n+1) u32, degree implicit from offset diff.
    let offsets = (n + 1) * 4;
    let csr32 = offsets + total_edges as usize * 4;
    let csr24 = offsets + total_edges as usize * 3;
    let mb = |b: usize| b as f64 / (1024.0 * 1024.0);
    println!("  layout bytes (graph only):");
    println!("    fixed-width 260B/node : {:>8.1} MB  (1.00x)", mb(fixed));
    println!(
        "    CSR + neighbor u32    : {:>8.1} MB  ({:.2}x)",
        mb(csr32),
        fixed as f64 / csr32 as f64,
    );
    println!(
        "    CSR + neighbor 24-bit : {:>8.1} MB  ({:.2}x)",
        mb(csr24),
        fixed as f64 / csr24 as f64,
    );
}

fn main() {
    eprintln!("Phase 0.a - degree histogram + packed-layout byte models\n");
    let cfg = VamanaConfig::default();

    // Real mxbai-embed-large 10K.
    if let Some((corpus, n, dim)) = load_npy(CORPUS_NPY) {
        let ids: Vec<u64> = (0..n as u64).collect();
        let index = VamanaIndex::build(corpus, ids, dim, &cfg);
        report("mxbai-embed-large real 10K", &index);
    } else {
        eprintln!("  (mxbai npy not found, skipping the real case)");
    }

    // Two synthetic datasets at 100K. uniform-sphere is the validated proxy
    // for real isotropic embeddings at scale; clustered is the contrast case
    // (shows the degree distribution is data-dependent, not fixed in N).
    let n = 100_000usize;
    let dim = 1024;
    let ids: Vec<u64> = (0..n as u64).collect();

    let index = VamanaIndex::build(uniform_sphere(n, dim, 7), ids.clone(), dim, &cfg);
    report(
        &format!("uniform-sphere synthetic {n} (real-scale proxy)"),
        &index,
    );

    let index = VamanaIndex::build(clustered(n, dim, 7), ids, dim, &cfg);
    report(&format!("clustered synthetic {n} (contrast)"), &index);

    // Decisive control: does the mean degree saturate toward R as N grows?
    // mxbai-10K (mean 43) vs uniform-100K (mean 64) could be scale, not data
    // character. uniform-sphere at 10K/30K isolates the variable.
    println!("\n== uniform-sphere: degree vs N (scale control) ==");
    println!(
        "  {:>10}{:>14}{:>10}{:>14}",
        "N", "mean degree", "at-R %", "CSR24 factor"
    );
    for &sn in &[10_000usize, 30_000, 100_000] {
        let sids: Vec<u64> = (0..sn as u64).collect();
        let idx = VamanaIndex::build(uniform_sphere(sn, dim, 7), sids, dim, &cfg);
        let hist = idx.degree_histogram();
        let r = hist.len() - 1;
        let edges: u64 = hist
            .iter()
            .enumerate()
            .map(|(d, &c)| d as u64 * u64::from(c))
            .sum();
        let mean = edges as f64 / sn as f64;
        let at_r = f64::from(hist[r]) / sn as f64 * 100.0;
        let fixed = sn * FIXED_NODE_BYTES;
        let csr24 = (sn + 1) * 4 + edges as usize * 3;
        println!(
            "  {:>10}{:>14.1}{:>9.1}%{:>13.2}x",
            sn,
            mean,
            at_r,
            fixed as f64 / csr24 as f64,
        );
    }
}
