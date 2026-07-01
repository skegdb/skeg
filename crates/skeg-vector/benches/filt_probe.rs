#![allow(clippy::cast_precision_loss)]
//! Is skeg's sparse-filtered recall a fundamental loss or a budget artifact?
//! Opens the cached tq2 500k index and sweeps l_search at 1% selectivity. If
//! recall climbs with the walk budget, it's budget (like the plain rerank lever),
//! not the engine. Reuses SKEG_STUDY_DIR/rw_tq2_n500000.

use ahash::AHashSet;
use rayon::prelude::*;
use skeg_simd::cosine_f32;
use skeg_vector::{DiskVamanaIndex, QuantKind};
use std::path::PathBuf;

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
const CORPUS: &str = "skeg/bench-compare/embeddings_cache/corpus_mxbai-wiki-chunked_500k.npy";
const QUERY: &str = "skeg/bench-compare/embeddings_cache/queries_mxbai-wiki-chunked_1000.npy";
const K: usize = 10;

fn load_npy(path: &str) -> Option<(Vec<f32>, usize, usize)> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        return None;
    }
    let hl = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + hl]).ok()?;
    let sh = header.find("'shape':")?;
    let lp = header[sh..].find('(')? + sh + 1;
    let rp = header[lp..].find(')')? + lp;
    let dims: Vec<usize> = header[lp..rp]
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let data: Vec<f32> = bytes[10 + hl..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Some((data, dims[0], dims[1]))
}

fn prep(path: &str, cap: usize) -> (Vec<Vec<f32>>, usize) {
    let (data, rows, dim) = load_npy(path).expect("npy");
    let n = cap.min(rows);
    let out = (0..n)
        .map(|i| {
            let mut v = data[i * dim..i * dim + dim].to_vec();
            let nrm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if nrm > 1e-10 {
                v.iter_mut().for_each(|x| *x /= nrm);
            }
            v
        })
        .collect();
    (out, n)
}

fn main() {
    let n = std::env::var("SKEG_BENCH_N").ok().and_then(|s| s.parse().ok()).unwrap_or(500_000);
    let nq = std::env::var("SKEG_NQ").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    let (corpus, n) = prep(&format!("{ROOT}/{CORPUS}"), n);
    let (queries, _) = prep(&format!("{ROOT}/{QUERY}"), nq);
    let dir: PathBuf = std::env::var("SKEG_STUDY_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("skeg_tq1_study"))
        .join(format!("rw_tq2_n{n}"));
    let idx = DiskVamanaIndex::open_with_tier(&dir, QuantKind::TurboQuant { bits: 2 })
        .expect("cached tq2 index (run tq1ctl_vs_tq2 first)");

    for &sel in &[0.01_f64, 0.10] {
        let step = (1.0 / sel).round() as u64;
        let matches = move |id: u64| id % step == 0;
        let truth: Vec<AHashSet<u64>> = queries
            .par_iter()
            .map(|q| {
                let mut s: Vec<(f32, u64)> = corpus
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| (*i as u64) % step == 0)
                    .map(|(i, v)| (cosine_f32(q, v), i as u64))
                    .collect();
                s.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                s.iter().take(K).map(|&(_, id)| id).collect()
            })
            .collect();
        println!("-- selectivity {:.0}% --", sel * 100.0);
        for &l in &[300usize, 800, 1500, 4000] {
            let mut hits = 0usize;
            let t = std::time::Instant::now();
            for (q, tr) in queries.iter().zip(&truth) {
                let got = idx.search_filtered(q, K, l, &matches, &[], sel as f32).unwrap();
                hits += got.iter().filter(|(id, _)| tr.contains(id)).count();
            }
            let ms = t.elapsed().as_secs_f64() * 1e3 / queries.len() as f64;
            println!("   L{l:<5} recall {:.4}   {ms:.2} ms/q", hits as f64 / (queries.len() * K) as f64);
        }
    }
}
