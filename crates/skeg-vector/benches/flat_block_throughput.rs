//! Block kernel throughput bench: measures `tq4_block32_score_u8_neon`
//! against the existing row-major `tq4_adc_i8_neon` on identical
//! corpora. Gates against the pre-registered targets in
//! `skeg-internal/bench-compare/BLOCK-KERNEL-PLAN.md` (G-B1 0.5x
//! turbovec, G-B2 0.7x turbovec).
//!
//! Same env-tunable shape as `flat_throughput.rs`:
//!   SKEG_BLOCK_N         corpus size       (default 100000)
//!   SKEG_BLOCK_DIMS      comma list of dim (default 384,1024,1536)
//!   SKEG_BLOCK_QUERIES   query count       (default 200)
//!   SKEG_BLOCK_WARMUP    warmup queries    (default 8)

#![deny(unsafe_code)]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::hint::black_box;
use std::time::Instant;

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use skeg_simd::{
    BLOCK, build_tq4_lut_f32, interleave_tq4_codes, quantize_tq4_lut_u8, tq4_adc_i8,
    tq4_block32_score_u8_scalar,
};

#[cfg(target_arch = "aarch64")]
use skeg_simd::tq4_block32_score_u8_neon;

fn env_usize(k: &str, default: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_list_usize(k: &str, default: &[usize]) -> Vec<usize> {
    std::env::var(k)
        .ok()
        .map(|v| {
            v.split(',')
                .filter_map(|x| x.trim().parse().ok())
                .collect::<Vec<_>>()
        })
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| default.to_vec())
}

/// Generate random vectors in `[-1, 1]^dim` then unit-normalise (so the
/// rotation invariant matters for TurboQuant). Distribution shape is
/// distribution-agnostic for throughput.
fn generate_normalised(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n * dim);
    for _ in 0..n {
        let mut v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in &mut v {
            *x /= norm;
        }
        out.extend(v);
    }
    out
}

/// Centroids matched to the standard normal quantile set used by the
/// scalar reference path - same magnitude / sign distribution as the
/// production codepath uses, so the bench is representative.
fn synthetic_centroids() -> [f32; 16] {
    [
        -1.5, -1.05, -0.74, -0.49, -0.28, -0.13, -0.04, -0.01, 0.01, 0.04, 0.13, 0.28, 0.49, 0.74,
        1.05, 1.5,
    ]
}

fn quantise_corpus_tq4(
    corpus: &[f32],
    n: usize,
    dim: usize,
    centroids: &[f32; 16],
) -> Vec<Vec<u8>> {
    let mut rows = Vec::with_capacity(n);
    let n_groups = dim / 2;
    for v in 0..n {
        let row_f32 = &corpus[v * dim..(v + 1) * dim];
        let mut codes = vec![0u8; n_groups];
        for g in 0..n_groups {
            let lo_val = row_f32[2 * g];
            let hi_val = row_f32[2 * g + 1];
            let lo_idx = nearest_centroid(lo_val, centroids);
            let hi_idx = nearest_centroid(hi_val, centroids);
            codes[g] = (hi_idx << 4) | lo_idx;
        }
        rows.push(codes);
    }
    rows
}

fn nearest_centroid(v: f32, centroids: &[f32; 16]) -> u8 {
    let mut best = 0u8;
    let mut best_d = f32::INFINITY;
    for (i, &c) in centroids.iter().enumerate() {
        let d = (v - c).abs();
        if d < best_d {
            best_d = d;
            best = i as u8;
        }
    }
    best
}

fn quantise_centroids_i8(centroids: &[f32; 16]) -> ([i8; 16], f32) {
    let max_abs = centroids.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
    let inv = 1.0 / scale;
    let mut out = [0i8; 16];
    for (o, &c) in out.iter_mut().zip(centroids.iter()) {
        let q = (c * inv).round().clamp(-127.0, 127.0) as i8;
        *o = q;
    }
    (out, scale)
}

fn run_block(
    corpus_codes: &[Vec<u8>],
    queries_f32: &[Vec<f32>],
    centroids: &[f32; 16],
    dim: usize,
    use_neon: bool,
) -> (f64, f64) {
    let n = corpus_codes.len();
    let n_groups = dim / 2;
    let mut interleaved_blocks: Vec<Vec<u8>> = Vec::with_capacity(n / BLOCK);
    for block_idx in 0..(n / BLOCK) {
        let mut row_refs: Vec<&[u8]> = Vec::with_capacity(BLOCK);
        for v in 0..BLOCK {
            row_refs.push(&corpus_codes[block_idx * BLOCK + v]);
        }
        let mut buf = vec![0u8; n_groups * BLOCK];
        interleave_tq4_codes(&row_refs, dim, &mut buf);
        interleaved_blocks.push(buf);
    }
    let mut lut_f32 = vec![0.0f32; n_groups * 32];
    let mut lut_u8 = vec![0u8; n_groups * 32];
    let mut out = [0.0f32; BLOCK];

    let t0 = Instant::now();
    for q in queries_f32 {
        build_tq4_lut_f32(q, centroids, dim, &mut lut_f32);
        let (inv_scale, bias_per_group) = quantize_tq4_lut_u8(&lut_f32, &mut lut_u8);
        for block in &interleaved_blocks {
            if use_neon {
                #[cfg(target_arch = "aarch64")]
                {
                    tq4_block32_score_u8_neon(
                        block,
                        &lut_u8,
                        inv_scale,
                        bias_per_group,
                        dim,
                        &mut out,
                    );
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    tq4_block32_score_u8_scalar(
                        block,
                        &lut_u8,
                        inv_scale,
                        bias_per_group,
                        dim,
                        &mut out,
                    );
                }
            } else {
                tq4_block32_score_u8_scalar(
                    block,
                    &lut_u8,
                    inv_scale,
                    bias_per_group,
                    dim,
                    &mut out,
                );
            }
            // Defeat dead-code elimination - the bench compiler is
            // free to drop the kernel call otherwise because nothing
            // consumes the per-block scores.
            black_box(&out);
        }
    }
    let elapsed_s = t0.elapsed().as_secs_f64();
    let qps = queries_f32.len() as f64 / elapsed_s;
    let ms_per_q = elapsed_s / queries_f32.len() as f64 * 1000.0;
    (qps, ms_per_q)
}

fn run_row(
    corpus_codes: &[Vec<u8>],
    queries_f32: &[Vec<f32>],
    centroids_i8: &[i8; 16],
    i8_scale: f32,
    dim: usize,
) -> (f64, f64) {
    // Build f32 q_rot per query (no rotation here; bench is on
    // synthetic data so just pass the normalised query straight in).
    let t0 = Instant::now();
    for q in queries_f32 {
        for row in corpus_codes {
            black_box(tq4_adc_i8(row, centroids_i8, i8_scale, q, dim));
        }
    }
    let elapsed_s = t0.elapsed().as_secs_f64();
    let qps = queries_f32.len() as f64 / elapsed_s;
    let ms_per_q = elapsed_s / queries_f32.len() as f64 * 1000.0;
    (qps, ms_per_q)
}

fn main() {
    let n = env_usize("SKEG_BLOCK_N", 100_000);
    let dims = env_list_usize("SKEG_BLOCK_DIMS", &[384, 1024, 1536]);
    let n_queries = env_usize("SKEG_BLOCK_QUERIES", 200);
    let warmup = env_usize("SKEG_BLOCK_WARMUP", 8);

    println!("# block kernel throughput, synthetic normalised vectors");
    println!("# N={n} dims={dims:?} queries={n_queries} warmup={warmup}");
    println!(
        "# host: target_os={} target_arch={}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    println!("kernel,dim,n,qps,ms_per_q");

    let centroids = synthetic_centroids();
    let (centroids_i8, i8_scale) = quantise_centroids_i8(&centroids);

    for &dim in &dims {
        let corpus_f32 = generate_normalised(n, dim, 0xC0FFEE);
        let queries_f32: Vec<Vec<f32>> = (0..n_queries)
            .map(|_| generate_normalised(1, dim, 0xBEEF))
            .collect();
        let corpus_codes = quantise_corpus_tq4(&corpus_f32, n, dim, &centroids);

        // Warmup runs (discarded).
        let warmup_queries: Vec<Vec<f32>> = queries_f32.iter().take(warmup).cloned().collect();
        let _ = run_block(&corpus_codes, &warmup_queries, &centroids, dim, true);
        let _ = run_row(&corpus_codes, &warmup_queries, &centroids_i8, i8_scale, dim);

        let (qps_block, ms_block) = run_block(&corpus_codes, &queries_f32, &centroids, dim, true);
        println!("block_neon,{dim},{n},{qps_block:.1},{ms_block:.3}");
        let (qps_row, ms_row) = run_row(&corpus_codes, &queries_f32, &centroids_i8, i8_scale, dim);
        println!("row_neon,{dim},{n},{qps_row:.1},{ms_row:.3}");
        let speedup = qps_block / qps_row;
        eprintln!(
            "# dim={dim}: block {qps_block:.0} QPS / row {qps_row:.0} QPS = speedup {speedup:.2}x"
        );
    }
}
