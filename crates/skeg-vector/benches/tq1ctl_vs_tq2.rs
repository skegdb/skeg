#![allow(clippy::cast_precision_loss)]

//! Head-to-head at scale, all RW (streaming insert): skeg tq1 with the LIVE
//! online controller (search_adaptive) vs skeg tq2 (current default), on mxbai
//! 500k. Plain + filtered (selectivity sweep), recall@10 + p50/p99 + QPS + RAM.
//! Indexes are built via insert()+consolidate() (the RW path) and cached under
//! SKEG_STUDY_DIR. This is the skeg side of the skeg-vs-qdrant comparison.
//!
//!   SKEG_BENCH_N (default 500000)  SKEG_NQ (default 200)  SKEG_PASSES (default 3)

use ahash::AHashSet;
use skeg_simd::cosine_f32;
use skeg_vector::{DiskVamanaIndex, QuantKind};
use std::path::{Path, PathBuf};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
const CORPUS: &str = "skeg/bench-compare/embeddings_cache/corpus_mxbai-wiki-chunked_500k.npy";
const QUERY: &str = "skeg/bench-compare/embeddings_cache/queries_mxbai-wiki-chunked_1000.npy";
const K: usize = 10;
const L_SEARCH: usize = 300;
const SELECTIVITIES: [f64; 3] = [0.01, 0.10, 0.50];

fn env(name: &str, d: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}

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
    if dims.len() != 2 {
        return None;
    }
    let data: Vec<f32> = bytes[10 + hl..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Some((data, dims[0], dims[1]))
}

fn load_prep(path: &str, n_cap: usize) -> Option<(Vec<Vec<f32>>, usize, usize)> {
    let (data, rows, dim) = load_npy(path)?;
    let n = n_cap.min(rows);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut v = data[i * dim..i * dim + dim].to_vec();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in &mut v {
                *x /= norm;
            }
        }
        out.push(v);
    }
    Some((out, n, dim))
}

fn brute_filtered(corpus: &[Vec<f32>], q: &[f32], step: u64) -> AHashSet<u64> {
    use rayon::prelude::*;
    let mut s: Vec<(f32, u64)> = corpus
        .par_iter()
        .enumerate()
        .filter(|(i, _)| (*i as u64) % step == 0)
        .map(|(i, v)| (cosine_f32(q, v), i as u64))
        .collect();
    s.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    s.iter().take(K).map(|&(_, id)| id).collect()
}

fn study_dir() -> PathBuf {
    std::env::var("SKEG_STUDY_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("skeg_tq1_study"))
}

fn build_rw(
    dir: &Path,
    corpus: &[Vec<f32>],
    dim: usize,
    n: usize,
    tier: QuantKind,
) -> (DiskVamanaIndex, f64) {
    if let Ok(i) = DiskVamanaIndex::open_with_tier(dir, tier)
        && i.len() == n
    {
        return (i, 0.0);
    }
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let t = std::time::Instant::now();
    let mut idx = DiskVamanaIndex::create_empty_with_tier(dir, dim, L_SEARCH, tier).unwrap();
    for (id, v) in corpus.iter().enumerate() {
        idx.insert(id as u64, v).unwrap(); // RW streaming insert
    }
    idx.consolidate().unwrap();
    let s = t.elapsed().as_secs_f64();
    drop(idx);
    (DiskVamanaIndex::open_with_tier(dir, tier).unwrap(), s)
}

fn pctl(mut lats: Vec<f64>) -> (f64, f64, f64) {
    lats.sort_unstable_by(f64::total_cmp);
    let mean = lats.iter().sum::<f64>() / lats.len() as f64;
    (
        lats[lats.len() / 2],
        lats[lats.len() * 99 / 100],
        1e6 / mean,
    )
}

fn main() {
    let n_cap = env("SKEG_BENCH_N", 500_000);
    let nq_cap = env("SKEG_NQ", 200);
    let passes = env("SKEG_PASSES", 3);
    let Some((corpus, n, dim)) = load_prep(&format!("{ROOT}/{CORPUS}"), n_cap) else {
        println!("corpus missing ({CORPUS})");
        return;
    };
    let (qc, _, _) = load_prep(&format!("{ROOT}/{QUERY}"), nq_cap).expect("queries");
    let queries: Vec<Vec<f32>> = qc;
    let nq = queries.len();
    println!(
        "skeg RW head-to-head: mxbai {n} x {dim}, {nq} queries, k={K}, L={L_SEARCH}, passes={passes}"
    );

    // Build both tiers via the RW path (cached).
    let (tq1, b1) = build_rw(
        &study_dir().join(format!("rw_tq1_n{n}")),
        &corpus,
        dim,
        n,
        QuantKind::TurboQuant { bits: 1 },
    );
    tq1.enable_tq1_controller();
    let (tq2, b2) = build_rw(
        &study_dir().join(format!("rw_tq2_n{n}")),
        &corpus,
        dim,
        n,
        QuantKind::TurboQuant { bits: 2 },
    );
    println!("build (RW insert+consolidate): tq1 {b1:.0}s  tq2 {b2:.0}s  (0 = cached)");

    // Ground truth (unfiltered = filter step 1).
    let truth: Vec<AHashSet<u64>> = queries
        .iter()
        .map(|q| brute_filtered(&corpus, q, 1))
        .collect();

    // Plain search: tq1 via the live controller (search_adaptive), tq2 default.
    println!("\n-- plain --   tier          recall@10  p50us  p99us   qps   ctl-mode");
    for (name, is_tq1) in [("tq1+ctl", true), ("tq2", false)] {
        let mut hits = 0usize;
        let mut lats = Vec::new();
        for _ in 0..passes {
            for (q, t) in queries.iter().zip(&truth) {
                let s = std::time::Instant::now();
                let got = if is_tq1 {
                    tq1.search_adaptive(q, K).unwrap()
                } else {
                    tq2.search(q, K).unwrap()
                };
                lats.push(s.elapsed().as_secs_f64() * 1e6);
                hits += got.iter().filter(|(id, _)| t.contains(id)).count();
            }
        }
        let (p50, p99, qps) = pctl(lats);
        let recall = hits as f64 / (nq * passes * K) as f64;
        let mode = if is_tq1 {
            format!("{:?}", tq1.tq1_controller_mode())
        } else {
            "-".into()
        };
        println!(
            "              {name:<12}  {recall:.4}    {p50:>5.0}  {p99:>5.0}  {qps:>5.0}   {mode}"
        );
    }

    // Filtered search (tq1 uses its dim-default proxy = hybrid at 1024).
    println!("\n-- filtered --  tier      sel   recall@10  p50us  p99us   qps");
    for &sel in &SELECTIVITIES {
        let step = (1.0 / sel).round() as u64;
        let matches = move |id: u64| id % step == 0;
        let ft: Vec<AHashSet<u64>> = queries
            .iter()
            .map(|q| brute_filtered(&corpus, q, step))
            .collect();
        for (name, is_tq1) in [("tq1", true), ("tq2", false)] {
            let mut hits = 0usize;
            let mut lats = Vec::new();
            for _ in 0..passes {
                for (q, t) in queries.iter().zip(&ft) {
                    let s = std::time::Instant::now();
                    let got = if is_tq1 { &tq1 } else { &tq2 }
                        .search_filtered(q, K, L_SEARCH, &matches, &[], sel as f32)
                        .unwrap();
                    lats.push(s.elapsed().as_secs_f64() * 1e6);
                    hits += got.iter().filter(|(id, _)| t.contains(id)).count();
                }
            }
            let (p50, p99, qps) = pctl(lats);
            let recall = hits as f64 / (nq * passes * K) as f64;
            println!(
                "              {name:<8}  {:>4.0}%  {recall:.4}    {p50:>5.0}  {p99:>5.0}  {qps:>5.0}",
                sel * 100.0
            );
        }
    }
}
