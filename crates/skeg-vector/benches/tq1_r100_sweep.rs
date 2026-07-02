#![allow(clippy::cast_precision_loss)]
//! tq1 recall@100 lever: is it limited by the WALK (l_search - true top-100 not
//! surfaced by the graph) or by the RERANK budget (surfaced but not enough
//! re-scored from disk)? Build tq1 100k once, sweep (l_search, rerank) for
//! recall@100 + ms/q. Tells the adaptive controller which knob actually moves it.
//!   SKEG_BENCH_N=100000  SKEG_NQ=200

use ahash::AHashSet;
use rayon::prelude::*;
use skeg_simd::cosine_f32;
use skeg_vector::{DiskVamanaIndex, QuantKind};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
const CORPUS: &str = "skeg/bench-compare/embeddings_cache/corpus_mxbai-wiki.npy";
const QUERY: &str = "skeg/bench-compare/embeddings_cache/queries_mxbai-wiki_200.npy";

fn load(path: &str, cap: usize) -> (Vec<Vec<f32>>, usize) {
    let bytes = std::fs::read(path).unwrap();
    let hl = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + hl]).unwrap();
    let sh = header.find("'shape':").unwrap();
    let lp = header[sh..].find('(').unwrap() + sh + 1;
    let rp = header[lp..].find(')').unwrap() + lp;
    let dims: Vec<usize> = header[lp..rp].split(',').filter_map(|s| s.trim().parse().ok()).collect();
    let (rows, dim) = (dims[0], dims[1]);
    let data: Vec<f32> = bytes[10 + hl..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
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
    let n_cap = std::env::var("SKEG_BENCH_N").ok().and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let nq = std::env::var("SKEG_NQ").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    let (corpus, dim) = load(&format!("{ROOT}/{CORPUS}"), n_cap);
    let (queries, _) = load(&format!("{ROOT}/{QUERY}"), nq);
    let n = corpus.len();
    let t100: Vec<AHashSet<u64>> = queries
        .par_iter()
        .map(|q| {
            let mut t: Vec<(f32, u64)> = corpus.iter().enumerate().map(|(i, v)| (cosine_f32(q, v), i as u64)).collect();
            t.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            t.iter().take(100).map(|&(_, id)| id).collect()
        })
        .collect();
    let tier = QuantKind::TurboQuant { bits: 1 };
    let tmp = std::env::temp_dir().join("skeg_tq1_r100");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let mut idx = DiskVamanaIndex::create_empty_with_tier(&tmp, dim, 300, tier).unwrap();
    for (id, v) in corpus.iter().enumerate() {
        idx.insert(id as u64, v).unwrap();
    }
    idx.consolidate().unwrap();
    println!("tq1 recall@100 sweep: mxbai {n} x {dim}, {} queries", queries.len());

    // (walk beam l_search, rerank disk-read budget). If recall climbs with rerank
    // at fixed l_search -> rerank-limited; if it needs a deeper l_search too ->
    // walk-limited.
    for &(ls, rr) in &[
        (300usize, 800usize),
        (300, 3200),
        (300, 12800),
        (600, 3200),
        (1000, 6400),
        (2000, 12800),
    ] {
        let mut hits = 0usize;
        let t = std::time::Instant::now();
        for (q, tr) in queries.iter().zip(&t100) {
            hits += idx.search_with_params(q, 100, ls, rr).unwrap().iter().filter(|(id, _)| tr.contains(id)).count();
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / queries.len() as f64;
        println!("  l_search {ls:<4} rerank {rr:<6} recall@100 {:.4}  {ms:.2} ms/q", hits as f64 / (queries.len() * 100) as f64);
    }
    let _ = std::fs::remove_dir_all(&tmp);
}
