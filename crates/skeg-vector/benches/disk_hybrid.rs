#![allow(clippy::cast_precision_loss, clippy::type_complexity)]
//! Hybrid filtered search end-to-end on the DISK path (proxy + rerank): the
//! IVF-routed shortlist (search_filtered_hybrid) vs the plain quantized scan
//! (score_ids_quantized = touch all |S|), across selectivity and filter
//! correlation, on the cached mxbai 500k tq2 index. Recall@10 + latency.
//!   SKEG_STUDY_DIR=<cache>  SKEG_BENCH_N=500000  SKEG_NQ=200

use ahash::AHashSet;
use rayon::prelude::*;
use skeg_simd::cosine_f32;
use skeg_vector::{DiskVamanaIndex, QuantKind};
use std::path::PathBuf;

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
const CORPUS: &str = "skeg/bench-compare/embeddings_cache/corpus_mxbai-wiki-chunked_500k.npy";
const QUERY: &str = "skeg/bench-compare/embeddings_cache/queries_mxbai-wiki-chunked_1000.npy";
const K: usize = 10;
const RR: usize = 80;

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
        .unwrap_or(500_000);
    let nq = std::env::var("SKEG_NQ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let (corpus, dim) = load(&format!("{ROOT}/{CORPUS}"), n_cap);
    let (queries, _) = load(&format!("{ROOT}/{QUERY}"), nq);
    let n = corpus.len();
    let dir: PathBuf = std::env::var("SKEG_STUDY_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("skeg_tq1_study"))
        .join(format!("rw_tq2_n{n}"));
    let mut idx = DiskVamanaIndex::open_with_tier(&dir, QuantKind::TurboQuant { bits: 2 })
        .expect("cached tq2 500k (run tq1ctl_vs_tq2 first)");
    println!(
        "disk hybrid: mxbai {n} x {dim}, {} queries, k={K}",
        queries.len()
    );
    let t = std::time::Instant::now();
    idx.build_ivf(0, 8).unwrap();
    println!("build_ivf: {:.0}s", t.elapsed().as_secs_f64());

    // correlated cluster ranking (by distance to a fixed center).
    let mut by_center: Vec<u64> = (0..n as u64).collect();
    let center = corpus[42].clone();
    by_center.sort_unstable_by(|&a, &b| {
        cosine_f32(&center, &corpus[b as usize])
            .total_cmp(&cosine_f32(&center, &corpus[a as usize]))
    });

    for &(fname, corr) in &[("uniform", false), ("correlated", true)] {
        for &sel in &[0.01f64, 0.05, 0.10, 0.50] {
            let msize = (n as f64 * sel) as usize;
            let mut s: Vec<u64> = if corr {
                by_center[..msize].to_vec()
            } else {
                let step = (1.0 / sel).round() as u64;
                (0..n as u64).filter(|id| id % step == 0).collect()
            };
            s.sort_unstable();
            let sset: AHashSet<u64> = s.iter().copied().collect();
            let truth: Vec<AHashSet<u64>> = queries
                .par_iter()
                .map(|q| {
                    let mut t: Vec<(f32, u64)> = s
                        .iter()
                        .map(|&id| (cosine_f32(q, &corpus[id as usize]), id))
                        .collect();
                    t.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                    t.iter().take(K).map(|&(_, id)| id).collect()
                })
                .collect();
            let _ = &sset;
            println!("-- {fname} {:.0}% (|S|={}) --", sel * 100.0, s.len());
            let row = |name: &str, f: &dyn Fn(&[f32]) -> Vec<(u64, f32)>| {
                let mut hits = 0usize;
                let t = std::time::Instant::now();
                for (q, tr) in queries.iter().zip(&truth) {
                    hits += f(q).iter().filter(|(id, _)| tr.contains(id)).count();
                }
                let ms = t.elapsed().as_secs_f64() * 1e3 / queries.len() as f64;
                println!(
                    "   {name:<10} recall {:.4}  {ms:.2} ms/q",
                    hits as f64 / (queries.len() * K) as f64
                );
            };
            row("hybrid", &|q| {
                idx.search_filtered_hybrid(q, &s, K, RR).unwrap()
            });
            row("qscan", &|q| idx.score_ids_quantized(q, &s, K, RR).unwrap());
        }
    }
}
