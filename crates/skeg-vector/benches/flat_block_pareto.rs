//! Pareto comparison: skeg-flat-pq128 vs skeg-flat-tq4-row vs
//! skeg-flat-tq4-block on the same synthetic corpus. Reports QPS,
//! recall@10 vs f32 ground truth, and an RSS estimate per tier.
//!
//! Uses `FlatIndex` for the pq128 and tq4-row baselines (the
//! production path) and a hand-rolled top-K loop driven by the block
//! kernel for the new block-tier candidate.
//!
//! Tunable env:
//!   SKEG_PARETO_N         corpus size       (default 50000)
//!   SKEG_PARETO_DIM       dim               (default 1024)
//!   SKEG_PARETO_QUERIES   queries           (default 200)
//!   SKEG_PARETO_K         top-K             (default 10)

#![deny(unsafe_code)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop
)]

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs::File;
use std::hint::black_box;
use std::io::{self, Read};
use std::path::Path;
use std::time::Instant;

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use skeg_vector::{FlatIndex, QuantKind};

use skeg_simd::{BLOCK, build_tq4_lut_f32, interleave_tq4_codes, quantize_tq4_lut_u8};

#[cfg(target_arch = "aarch64")]
use skeg_simd::tq4_block32_score_u8_neon;

fn env_usize(k: &str, default: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Minimal NumPy `.npy` v1/v2 reader for f32 little-endian arrays.
/// The bench inputs are produced by the bench-compare embed scripts
/// which always emit `<f4` in C order; the reader rejects anything
/// else loudly so silent recall regressions never sneak through.
fn load_npy_f32(path: &Path) -> io::Result<(Vec<f32>, Vec<usize>)> {
    let mut f = File::open(path)?;
    let mut magic = [0u8; 6];
    f.read_exact(&mut magic)?;
    if &magic != b"\x93NUMPY" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad npy magic"));
    }
    let mut ver = [0u8; 2];
    f.read_exact(&mut ver)?;
    let header_len = if ver[0] == 1 {
        let mut buf = [0u8; 2];
        f.read_exact(&mut buf)?;
        u16::from_le_bytes(buf) as usize
    } else {
        let mut buf = [0u8; 4];
        f.read_exact(&mut buf)?;
        u32::from_le_bytes(buf) as usize
    };
    let mut header_buf = vec![0u8; header_len];
    f.read_exact(&mut header_buf)?;
    let header = String::from_utf8_lossy(&header_buf);
    if !header.contains("'descr': '<f4'") && !header.contains("\"descr\": \"<f4\"") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "npy descr is not <f4",
        ));
    }
    if header.contains("'fortran_order': True") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "fortran-order npy not supported",
        ));
    }
    let shape_start = header
        .find("'shape':")
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing shape in npy header"))?
        + 8;
    let after = &header[shape_start..];
    let lp = after
        .find('(')
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "bad shape paren"))?
        + 1;
    let rp = after
        .find(')')
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "bad shape paren"))?;
    let dims: Vec<usize> = after[lp..rp]
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let total: usize = dims.iter().product();
    let mut byte_buf = vec![0u8; total * 4];
    f.read_exact(&mut byte_buf)?;
    let mut data = Vec::with_capacity(total);
    for chunk in byte_buf.chunks_exact(4) {
        data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok((data, dims))
}

/// Read corpus + queries from .npy paths and return as
/// `Vec<Vec<f32>>` rows (so the rest of the bench keeps its existing
/// shape). Truncates to the first `limit` rows when set.
fn load_npy_rows(path: &Path, limit: Option<usize>) -> io::Result<Vec<Vec<f32>>> {
    let (data, shape) = load_npy_f32(path)?;
    if shape.len() != 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "npy must be 2-D (rows, dim)",
        ));
    }
    let rows = shape[0];
    let dim = shape[1];
    let n = limit.map(|l| l.min(rows)).unwrap_or(rows);
    let mut out = Vec::with_capacity(n);
    for r in 0..n {
        let start = r * dim;
        let row: Vec<f32> = data[start..start + dim].to_vec();
        out.push(row);
    }
    Ok(out)
}

fn synthetic_centroids() -> [f32; 16] {
    [
        -1.5, -1.05, -0.74, -0.49, -0.28, -0.13, -0.04, -0.01, 0.01, 0.04, 0.13, 0.28, 0.49, 0.74,
        1.05, 1.5,
    ]
}

fn generate_normalised(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in &mut v {
            *x /= norm;
        }
        out.push(v);
    }
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
    }
    dot
}

#[cfg(target_arch = "aarch64")]
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

#[cfg(target_arch = "aarch64")]
fn quantise_corpus_tq4(corpus: &[Vec<f32>], dim: usize, centroids: &[f32; 16]) -> Vec<Vec<u8>> {
    let n_groups = dim / 2;
    let mut rows = Vec::with_capacity(corpus.len());
    for v in corpus {
        let mut codes = vec![0u8; n_groups];
        for g in 0..n_groups {
            let lo = nearest_centroid(v[2 * g], centroids);
            let hi = nearest_centroid(v[2 * g + 1], centroids);
            codes[g] = (hi << 4) | lo;
        }
        rows.push(codes);
    }
    rows
}

/// Brute-force exact top-K ids by f32 cosine. Oracle for recall.
fn ground_truth(corpus: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> Vec<Vec<u64>> {
    queries
        .iter()
        .map(|q| {
            let mut scored: Vec<(f32, u64)> = corpus
                .iter()
                .enumerate()
                .map(|(i, v)| (cosine(q, v), i as u64))
                .collect();
            scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
            scored.truncate(k);
            scored.into_iter().map(|(_, id)| id).collect()
        })
        .collect()
}

fn recall_at(got: &[Vec<u64>], truth: &[Vec<u64>], k: usize) -> f32 {
    let mut total = 0.0f32;
    for (g, t) in got.iter().zip(truth.iter()) {
        let truth_set: std::collections::HashSet<u64> = t.iter().take(k).copied().collect();
        let hits = g.iter().take(k).filter(|id| truth_set.contains(id)).count();
        total += hits as f32 / k as f32;
    }
    total / got.len() as f32
}

/// Run FlatIndex (production search path with rerank) for a given tier.
fn run_flat_index(
    corpus: &[Vec<f32>],
    queries: &[Vec<f32>],
    dim: usize,
    kind: QuantKind,
    k: usize,
) -> (f64, Vec<Vec<u64>>) {
    let mut idx = FlatIndex::new(dim, kind);
    for (i, v) in corpus.iter().enumerate() {
        idx.insert(i as u64, v);
    }
    // Force quantization init.
    let _ = idx.search(&queries[0], k);

    let t0 = Instant::now();
    let mut got: Vec<Vec<u64>> = Vec::with_capacity(queries.len());
    for q in queries {
        let hits = idx.search(q, k);
        got.push(hits.into_iter().map(|(id, _score)| id).collect());
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let qps = queries.len() as f64 / elapsed;
    (qps, got)
}

/// Block kernel search emulation. Builds the interleaved layout
/// once, then for each query precomputes the u8 LUT, scores every
/// block, and reranks the top-`rerank_width` candidates with exact
/// f32 cosine. The pattern mirrors what `FlatIndex` does for the
/// row kernel, swapping in the block scoring path.
#[cfg(target_arch = "aarch64")]
fn run_block_search(
    corpus_f32: &[Vec<f32>],
    queries: &[Vec<f32>],
    centroids: &[f32; 16],
    dim: usize,
    k: usize,
) -> (f64, Vec<Vec<u64>>) {
    let n = corpus_f32.len();
    let n_groups = dim / 2;
    let row_codes = quantise_corpus_tq4(corpus_f32, dim, centroids);

    // Interleave into BLOCK-sized blocks; remainder (n % BLOCK) is
    // handled by a per-row scalar fallback so the bench covers the
    // whole corpus.
    let n_blocks = n / BLOCK;
    let remainder_start = n_blocks * BLOCK;
    let mut blocks: Vec<Vec<u8>> = Vec::with_capacity(n_blocks);
    for b in 0..n_blocks {
        let mut row_refs: Vec<&[u8]> = Vec::with_capacity(BLOCK);
        for v in 0..BLOCK {
            row_refs.push(&row_codes[b * BLOCK + v]);
        }
        let mut buf = vec![0u8; n_groups * BLOCK];
        interleave_tq4_codes(&row_refs, dim, &mut buf);
        blocks.push(buf);
    }

    let rerank_width = (k * 20).max(64);

    let mut lut_f32 = vec![0.0f32; n_groups * 32];
    let mut lut_u8 = vec![0u8; n_groups * 32];
    let mut block_out = [0.0f32; BLOCK];

    // Force first query to warm caches.
    {
        build_tq4_lut_f32(&queries[0], centroids, dim, &mut lut_f32);
        let (inv_scale, bias_per_group) = quantize_tq4_lut_u8(&lut_f32, &mut lut_u8);
        for block in &blocks {
            tq4_block32_score_u8_neon(
                block,
                &lut_u8,
                inv_scale,
                bias_per_group,
                dim,
                &mut block_out,
            );
            black_box(&block_out);
        }
    }

    let t0 = Instant::now();
    let mut got: Vec<Vec<u64>> = Vec::with_capacity(queries.len());
    for q in queries {
        build_tq4_lut_f32(q, centroids, dim, &mut lut_f32);
        let (inv_scale, bias_per_group) = quantize_tq4_lut_u8(&lut_f32, &mut lut_u8);

        // Candidates: keep a max-heap of (negated proxy, id) so smaller
        // proxies (worse matches) sit at the top of the heap and get
        // evicted first; we want to retain the highest scores.
        let mut cands: BinaryHeap<std::cmp::Reverse<(i32, u64)>> =
            BinaryHeap::with_capacity(rerank_width + 1);
        // Scale floats to a fixed-point i32 for ordering so the heap
        // stays Ord; precision matters only at the comparator level.
        let scale_to_i32 = 1_000_000.0f32;
        for (b, block) in blocks.iter().enumerate() {
            tq4_block32_score_u8_neon(
                block,
                &lut_u8,
                inv_scale,
                bias_per_group,
                dim,
                &mut block_out,
            );
            for lane in 0..BLOCK {
                let id = (b * BLOCK + lane) as u64;
                let score_i32 = (block_out[lane] * scale_to_i32) as i32;
                cands.push(std::cmp::Reverse((score_i32, id)));
                if cands.len() > rerank_width {
                    cands.pop();
                }
            }
        }
        // Tail (< BLOCK leftover vectors) scored via f32 cosine
        // directly since they are too few to justify another block.
        for i in remainder_start..n {
            let score = cosine(q, &corpus_f32[i]);
            let score_i32 = (score * scale_to_i32) as i32;
            cands.push(std::cmp::Reverse((score_i32, i as u64)));
            if cands.len() > rerank_width {
                cands.pop();
            }
        }

        // Rerank with exact f32 cosine over the top `rerank_width`.
        let mut top: Vec<(f32, u64)> = cands
            .into_iter()
            .map(|r| {
                let id = r.0.1;
                (cosine(q, &corpus_f32[id as usize]), id)
            })
            .collect();
        top.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        top.truncate(k);
        got.push(top.into_iter().map(|(_, id)| id).collect());
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let qps = queries.len() as f64 / elapsed;
    (qps, got)
}

#[cfg(not(target_arch = "aarch64"))]
fn run_block_search(
    _corpus_f32: &[Vec<f32>],
    queries: &[Vec<f32>],
    _centroids: &[f32; 16],
    _dim: usize,
    _k: usize,
) -> (f64, Vec<Vec<u64>>) {
    (0.0, vec![vec![]; queries.len()])
}

/// Static RSS estimate (MiB): codes + per-query LUT structures.
/// Excludes the f32 vectors kept by FlatIndex for re-rank because
/// that is identical across tiers.
fn rss_mib_codes_only(n: usize, dim: usize, kind: QuantKind, has_block_lut: bool) -> f32 {
    let codes_bytes = match kind {
        QuantKind::Pq { m, k } => n * m + m * k * 4, // n*m codes + LUT 4*m*k bytes
        QuantKind::TurboQuant { bits } => n * (dim * bits as usize / 8),
        QuantKind::Int8 => n * dim,
        QuantKind::F32 => n * dim * 4,
        QuantKind::Binary => n * (dim / 8),
    };
    let lut_extra = if has_block_lut {
        // Block kernel per-query u8 LUT (transient but resident).
        dim / 2 * 32
    } else {
        0
    };
    (codes_bytes + lut_extra) as f32 / (1024.0 * 1024.0)
}

fn main() {
    let n = env_usize("SKEG_PARETO_N", 50_000);
    let dim_synth = env_usize("SKEG_PARETO_DIM", 1024);
    let n_queries = env_usize("SKEG_PARETO_QUERIES", 200);
    let k = env_usize("SKEG_PARETO_K", 10);
    let corpus_path = std::env::var("SKEG_PARETO_CORPUS").ok();
    let queries_path = std::env::var("SKEG_PARETO_QUERIES_NPY").ok();

    let centroids = synthetic_centroids();

    // Source: real .npy when both paths are set, synthetic otherwise.
    let (corpus, queries, dim, source_label) =
        if let (Some(cp), Some(qp)) = (corpus_path.as_ref(), queries_path.as_ref()) {
            eprintln!("# loading real corpus {cp} (limit N={n})");
            let c = load_npy_rows(Path::new(cp), Some(n)).expect("corpus npy");
            eprintln!("# loading real queries {qp} (limit Q={n_queries})");
            let q = load_npy_rows(Path::new(qp), Some(n_queries)).expect("queries npy");
            let dim = c[0].len();
            let label = format!("real:{cp}");
            (c, q, dim, label)
        } else {
            let c = generate_normalised(n, dim_synth, 0xC0FFEE);
            let q = generate_normalised(n_queries, dim_synth, 0xBEEF);
            (c, q, dim_synth, "synthetic".to_owned())
        };

    println!("# block kernel pareto: skeg-pq128 / skeg-tq4-row / skeg-tq4-block");
    println!(
        "# source={source_label} N={} dim={dim} queries={} k={k}",
        corpus.len(),
        queries.len()
    );
    println!(
        "# host: target_os={} target_arch={}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );

    eprintln!("# computing ground truth (brute-force f32) ...");
    let truth = ground_truth(&corpus, &queries, k);

    println!("tier,qps,recall@10,recall@1,rss_codes_mib");

    // pq128
    let kind = QuantKind::Pq { m: 128, k: 256 };
    let (qps, got) = run_flat_index(&corpus, &queries, dim, kind, k);
    let recall10 = recall_at(&got, &truth, k);
    let recall1 = recall_at(&got, &truth, 1);
    let rss = rss_mib_codes_only(n, dim, kind, false);
    println!("skeg-flat-pq128,{qps:.1},{recall10:.4},{recall1:.4},{rss:.2}");

    // tq4 row
    let kind = QuantKind::TurboQuant { bits: 4 };
    let (qps, got) = run_flat_index(&corpus, &queries, dim, kind, k);
    let recall10 = recall_at(&got, &truth, k);
    let recall1 = recall_at(&got, &truth, 1);
    let rss = rss_mib_codes_only(n, dim, kind, false);
    println!("skeg-flat-tq4-row,{qps:.1},{recall10:.4},{recall1:.4},{rss:.2}");

    // tq4 block (this branch's new candidate)
    let (qps, got) = run_block_search(&corpus, &queries, &centroids, dim, k);
    let recall10 = recall_at(&got, &truth, k);
    let recall1 = recall_at(&got, &truth, 1);
    let rss = rss_mib_codes_only(n, dim, kind, true);
    println!("skeg-flat-tq4-block,{qps:.1},{recall10:.4},{recall1:.4},{rss:.2}");
}
