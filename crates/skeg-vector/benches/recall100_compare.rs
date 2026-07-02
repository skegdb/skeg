#![allow(clippy::cast_precision_loss)]
//! CLEAN recall@100 (proper k=100 search, not the k=10-vs-top-100 metric the
//! over-wire harness reported). tq1 vs tq2 at the DEFAULT serving params
//! (l_search=300, rerank=k*8) and at a WIDE walk (l_search=2000), so the real
//! recall@100 gap and its latency cost are visible.
//!   SKEG_BENCH_N=100000  SKEG_NQ=200  SKEG_CORPUS=<npy>  SKEG_QUERY=<npy>

use ahash::AHashSet;
use rayon::prelude::*;
use skeg_simd::cosine_f32;
use skeg_vector::{DiskVamanaIndex, QuantKind};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");

fn load(path: &str, cap: usize) -> (Vec<Vec<f32>>, usize) {
    let bytes = std::fs::read(path).unwrap();
    let hl = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + hl]).unwrap();
    let sh = header.find("'shape':").unwrap();
    let lp = header[sh..].find('(').unwrap() + sh + 1;
    let rp = header[lp..].find(')').unwrap() + lp;
    let dims: Vec<usize> = header[lp..rp]
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let (rows, dim) = (dims[0], dims[1]);
    let data: Vec<f32> = bytes[10 + hl..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let n = cap.min(rows);
    let out = (0..n)
        .map(|i| {
            let mut v = data[i * dim..i * dim + dim].to_vec();
            let nrm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
            v.iter_mut().for_each(|x| *x /= nrm);
            v
        })
        .collect();
    (out, dim)
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
    let corpus_path = std::env::var("SKEG_CORPUS").unwrap_or_else(|_| {
        format!("{ROOT}/skeg/bench-compare/embeddings_cache/corpus_mxbai-wiki.npy")
    });
    let query_path = std::env::var("SKEG_QUERY").unwrap_or_else(|_| {
        format!("{ROOT}/skeg/bench-compare/embeddings_cache/queries_mxbai-wiki_200.npy")
    });
    let (corpus, dim) = load(&corpus_path, n_cap);
    let (queries, _) = load(&query_path, nq);
    let n = corpus.len();
    let truth: Vec<AHashSet<u64>> = queries
        .par_iter()
        .map(|q| {
            let mut t: Vec<(f32, u64)> = corpus
                .iter()
                .enumerate()
                .map(|(i, v)| (cosine_f32(q, v), i as u64))
                .collect();
            t.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            t.iter().take(100).map(|&(_, id)| id).collect()
        })
        .collect();
    println!(
        "recall@100 (real, k=100): {n} x {dim}, {} queries",
        queries.len()
    );

    for bits in [1u8, 2] {
        let tier = QuantKind::TurboQuant { bits };
        let tmp = std::env::temp_dir().join(format!("skeg_r100_tq{bits}"));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let mut idx = DiskVamanaIndex::create_empty_with_tier(&tmp, dim, 300, tier).unwrap();
        for (id, v) in corpus.iter().enumerate() {
            idx.insert(id as u64, v).unwrap();
        }
        idx.consolidate().unwrap();
        // (label, l_search, rerank).
        for &(label, ls, rr) in &[("default", 300usize, 800usize), ("wide", 2000, 12800)] {
            let mut hits = 0usize;
            let t = std::time::Instant::now();
            for (q, tr) in queries.iter().zip(&truth) {
                hits += idx
                    .search_with_params(q, 100, ls, rr)
                    .unwrap()
                    .iter()
                    .filter(|(id, _)| tr.contains(id))
                    .count();
            }
            let ms = t.elapsed().as_secs_f64() * 1e3 / queries.len() as f64;
            println!(
                "  tq{bits} {label:<8} (l={ls},rr={rr})  recall@100 {:.4}  {ms:.2} ms/q",
                hits as f64 / (queries.len() * 100) as f64
            );
        }
        drop(idx);
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
