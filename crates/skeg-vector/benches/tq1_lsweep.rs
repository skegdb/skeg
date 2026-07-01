#![allow(clippy::cast_precision_loss)]

//! Since the tq1 hybrid walk is cheap (popcount), can we spend a wider L_search
//! for more recall? This sweeps L on the REAL DiskVamana hybrid path (mxbai,
//! 1024d) and reports recall@10 + latency at each L. Wider L visits more nodes
//! and feeds more asym-rescored candidates into the FIXED disk-rerank budget, so
//! recall rises with mostly in-RAM cost (no extra disk reads).

use ahash::AHashSet;
use skeg_simd::cosine_f32;
use skeg_vector::{DiskVamanaIndex, QuantKind};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
const CORPUS: &str = "skeg/bench-compare/embeddings_cache/corpus_mxbai-wiki.npy";
const QUERY: &str = "skeg/bench-compare/embeddings_cache/queries_mxbai-wiki_200.npy";
const K: usize = 10;
const LS: [usize; 6] = [100, 200, 300, 500, 800, 1200];

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

fn unit(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let inv = if n > 1e-10 { 1.0 / n } else { 0.0 };
    v.iter().map(|&x| x * inv).collect()
}

fn brute(corpus: &[Vec<f32>], q: &[f32]) -> AHashSet<u64> {
    let mut s: Vec<(f32, u64)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (cosine_f32(q, v), i as u64))
        .collect();
    s.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    s.iter().take(K).map(|&(_, id)| id).collect()
}

fn main() {
    let Some((raw, rows, dim)) = load_npy(&format!("{ROOT}/{CORPUS}")) else {
        println!("corpus missing; skipping");
        return;
    };
    let (qraw, qrows, _) = load_npy(&format!("{ROOT}/{QUERY}")).expect("queries");
    let n = env("SKEG_BENCH_N", 20_000).min(rows);
    let nq = env("SKEG_NQ", 100).min(qrows);
    println!("tq1 hybrid L-sweep on DiskVamana (mxbai): dim {dim}, n {n}, queries {nq}");

    let corpus: Vec<Vec<f32>> = (0..n).map(|i| unit(&raw[i * dim..(i + 1) * dim])).collect();
    let queries: Vec<Vec<f32>> = (0..nq)
        .map(|i| unit(&qraw[i * dim..(i + 1) * dim]))
        .collect();
    let truth: Vec<AHashSet<u64>> = queries.iter().map(|q| brute(&corpus, q)).collect();

    let tmp = tempfile::TempDir::new().unwrap();
    let mut idx = DiskVamanaIndex::create_empty_with_tier(
        tmp.path(),
        dim,
        LS[LS.len() - 1], // build with the widest so all query-time L are valid
        QuantKind::TurboQuant { bits: 1 },
    )
    .unwrap();
    for (id, v) in corpus.iter().enumerate() {
        idx.insert(id as u64, v).unwrap();
    }
    idx.consolidate().unwrap();
    drop(idx);
    let idx =
        DiskVamanaIndex::open_with_tier(tmp.path(), QuantKind::TurboQuant { bits: 1 }).unwrap();

    println!("   L    recall@10   mean us/query");
    for &l in &LS {
        let mut hits = 0usize;
        let mut us = 0.0f64;
        for (q, t) in queries.iter().zip(&truth) {
            let start = std::time::Instant::now();
            let got = idx.search_with_l(q, K, l).unwrap();
            us += start.elapsed().as_secs_f64() * 1e6;
            hits += got.iter().filter(|(id, _)| t.contains(id)).count();
        }
        println!(
            "  {l:>4}    {:.4}      {:.0}",
            hits as f64 / (nq * K) as f64,
            us / nq as f64
        );
    }
}
