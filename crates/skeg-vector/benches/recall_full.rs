#![allow(clippy::cast_precision_loss)]
//! COMPLETE recall eval: recall@10 AND recall@100 (real k-searches, not the
//! k=10-vs-top-100 metric), tq1 vs tq2, at the default serving params (l=300)
//! and a wide walk (l=2000). One dataset via env; run it at every scale you
//! evaluate. Dims zero-padded to a multiple of 8.
//!   SKEG_BENCH_N=100000  SKEG_NQ=200  SKEG_CORPUS=<npy>  SKEG_QUERY=<npy>

use ahash::AHashSet;
use rayon::prelude::*;
use skeg_simd::cosine_f32;
use skeg_vector::{DiskVamanaIndex, QuantKind};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");

fn load_npy(path: &str) -> (Vec<f32>, usize, usize) {
    let bytes = std::fs::read(path).unwrap_or_else(|_| panic!("missing {path}"));
    let hl = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + hl]).unwrap();
    let sh = header.find("'shape':").unwrap();
    let lp = header[sh..].find('(').unwrap() + sh + 1;
    let rp = header[lp..].find(')').unwrap() + lp;
    let dims: Vec<usize> = header[lp..rp]
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let data: Vec<f32> = bytes[10 + hl..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (data, dims[0], dims[1])
}

fn load_prep(path: &str, n_cap: usize, pad: usize) -> (Vec<Vec<f32>>, usize) {
    let (data, rows, dim) = load_npy(path);
    let n = n_cap.min(rows);
    let out = (0..n)
        .map(|i| {
            let mut v = vec![0.0f32; pad];
            v[..dim].copy_from_slice(&data[i * dim..i * dim + dim]);
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
            v.iter_mut().for_each(|x| *x /= norm);
            v
        })
        .collect();
    (out, n)
}

fn truth(corpus: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> Vec<AHashSet<u64>> {
    queries
        .par_iter()
        .map(|q| {
            let mut t: Vec<(f32, u64)> = corpus
                .iter()
                .enumerate()
                .map(|(i, v)| (cosine_f32(q, v), i as u64))
                .collect();
            t.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            t.iter().take(k).map(|&(_, id)| id).collect()
        })
        .collect()
}

fn main() {
    let n_cap = std::env::var("SKEG_BENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);
    let nq = std::env::var("SKEG_NQ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let native: usize = std::env::var("SKEG_DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let cpath = std::env::var("SKEG_CORPUS").unwrap_or_else(|_| {
        format!("{ROOT}/skeg/bench-compare/embeddings_cache/corpus_mxbai-wiki.npy")
    });
    let qpath = std::env::var("SKEG_QUERY").unwrap_or_else(|_| {
        format!("{ROOT}/skeg/bench-compare/embeddings_cache/queries_mxbai-wiki_200.npy")
    });
    let pad = native.next_multiple_of(8);
    let (corpus, n) = load_prep(&cpath, n_cap, pad);
    let (queries, _) = load_prep(&qpath, nq, pad);
    let t10 = truth(&corpus, &queries, 10);
    let t100 = truth(&corpus, &queries, 100);
    println!("recall (real): {n} x {pad}, {} queries", queries.len());
    println!(
        "{:<5} {:<8}  {:>10}  {:>11}  {:>8}",
        "tier", "walk", "recall@10", "recall@100", "ms/q"
    );

    for bits in [1u8, 2] {
        let tier = QuantKind::TurboQuant { bits };
        let tmp = std::env::temp_dir().join(format!("skeg_rfull_{bits}"));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let mut idx = DiskVamanaIndex::create_empty_with_tier(&tmp, pad, 300, tier).unwrap();
        for (id, v) in corpus.iter().enumerate() {
            idx.insert(id as u64, v).unwrap();
        }
        idx.consolidate().unwrap();
        for &(label, ls, rr10, rr100) in &[
            ("default", 300usize, 80usize, 800usize),
            ("wide", 2000, 1280, 12800),
        ] {
            let mut h10 = 0usize;
            let mut h100 = 0usize;
            let t = std::time::Instant::now();
            for (q, tr) in queries.iter().zip(&t10) {
                h10 += idx
                    .search_with_params(q, 10, ls, rr10)
                    .unwrap()
                    .iter()
                    .filter(|(id, _)| tr.contains(id))
                    .count();
            }
            for (q, tr) in queries.iter().zip(&t100) {
                h100 += idx
                    .search_with_params(q, 100, ls, rr100)
                    .unwrap()
                    .iter()
                    .filter(|(id, _)| tr.contains(id))
                    .count();
            }
            let ms = t.elapsed().as_secs_f64() * 1e3 / (queries.len() * 2) as f64;
            println!(
                "tq{bits:<3} {label:<8}  {:>10.4}  {:>11.4}  {ms:>7.2}",
                h10 as f64 / (queries.len() * 10) as f64,
                h100 as f64 / (queries.len() * 100) as f64,
            );
        }
        drop(idx);
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
