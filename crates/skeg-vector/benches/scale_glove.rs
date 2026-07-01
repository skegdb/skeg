#![allow(clippy::cast_precision_loss)]
//! Scale test at N = 1.18M (glove, dim 100). Builds a tq2 DiskVamana once
//! (cached) via RW insert, then measures the filtered planner's two tiers -
//! quantized scan vs walk - at FIXED absolute matching-set sizes. Confirms the
//! qscan crossover (~30k matches) is N-independent: same |s|, same qscan cost,
//! at 1.18M as at 500k. glove is low-dim so absolute recall is modest; the point
//! is the scaling + tier mechanism, not glove recall.

use ahash::AHashSet;
use rayon::prelude::*;
use skeg_simd::cosine_f32;
use skeg_vector::{DiskVamanaIndex, QuantKind};
use std::path::{Path, PathBuf};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
const CORPUS: &str = "skeg-bench/data/glove_corpus.npy";
const QUERY: &str = "skeg-bench/data/glove_queries.npy";
const K: usize = 10;
const PAD: usize = 104; // glove 100 -> multiple of 8
const TARGET_MATCHES: [usize; 4] = [5_000, 25_000, 50_000, 120_000];

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
            let mut v = vec![0.0f32; PAD];
            v[..dim].copy_from_slice(&data[i * dim..i * dim + dim]);
            let nrm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if nrm > 1e-10 {
                v.iter_mut().for_each(|x| *x /= nrm);
            }
            v
        })
        .collect();
    (out, n)
}

fn build(dir: &Path, corpus: &[Vec<f32>], n: usize) -> DiskVamanaIndex {
    let tier = QuantKind::TurboQuant { bits: 2 };
    if let Ok(i) = DiskVamanaIndex::open_with_tier(dir, tier)
        && i.len() == n
    {
        return i;
    }
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let t = std::time::Instant::now();
    let mut idx = DiskVamanaIndex::create_empty_with_tier(dir, PAD, 300, tier).unwrap();
    for (id, v) in corpus.iter().enumerate() {
        idx.insert(id as u64, v).unwrap();
    }
    idx.consolidate().unwrap();
    println!("build {n} vecs (RW): {:.0}s", t.elapsed().as_secs_f64());
    drop(idx);
    DiskVamanaIndex::open_with_tier(dir, tier).unwrap()
}

fn main() {
    let nq = std::env::var("SKEG_NQ").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    let (corpus, n) = prep(&format!("{ROOT}/{CORPUS}"), usize::MAX);
    let (queries, _) = prep(&format!("{ROOT}/{QUERY}"), nq);
    println!("scale test: glove {n} x {PAD}, {} queries, k={K}", queries.len());
    let dir: PathBuf = std::env::var("SKEG_STUDY_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("skeg_tq1_study"))
        .join(format!("glove_n{n}"));
    let idx = build(&dir, &corpus, n);

    for &target in &TARGET_MATCHES {
        let step = (n / target).max(1) as u64;
        let matchn = n / step as usize;
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
        println!("-- ~{matchn} matches (of {n}) --");
        let mut row = |name: &str, f: &dyn Fn(&[f32]) -> Vec<(u64, f32)>| {
            let mut hits = 0usize;
            let t = std::time::Instant::now();
            for (q, tr) in queries.iter().zip(&truth) {
                hits += f(q).iter().filter(|(id, _)| tr.contains(id)).count();
            }
            let ms = t.elapsed().as_secs_f64() * 1e3 / queries.len() as f64;
            println!("   {name:<12} recall {:.4}   {ms:.2} ms/q", hits as f64 / (queries.len() * K) as f64);
        };
        let sel = matchn as f32 / n as f32;
        row("walk L1500", &|q| idx.search_filtered(q, K, 1500, &matches, &[], sel).unwrap());
        let ids: Vec<u64> = (0..n as u64).filter(|id| id % step == 0).collect();
        row("qscan rr80", &|q| idx.score_ids_quantized(q, &ids, K, 80).unwrap());
    }
}
