#![allow(clippy::cast_precision_loss)]

//! End-to-end verification of the tq1 hybrid proxy on the REAL DiskVamana path,
//! with REAL embeddings (mxbai-wiki, 1024d >= TQ1_HYBRID_MIN_DIM so the search
//! auto-selects the hybrid: popcount walk + asymmetric re-score before the
//! bounded disk rerank).
//!
//! Builds an on-disk tq1 index, inserts the corpus, consolidates, reopens, and
//! measures recall@10 vs brute force. Exercises the wired `DiskVamanaIndex::search`
//! (quantize_query -> hybrid code -> walk -> proxy_rescore -> disk rerank), so a
//! wiring regression shows up as a recall drop vs the ~0.99 the standalone
//! matrix predicts for mxbai-1024.

use ahash::AHashSet;
use skeg_simd::cosine_f32;
use skeg_vector::{DiskVamanaIndex, QuantKind, QuantizedVectors, Tq1ProxyMode};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
const CORPUS: &str = "skeg/bench-compare/embeddings_cache/corpus_mxbai-wiki.npy";
const QUERY: &str = "skeg/bench-compare/embeddings_cache/queries_mxbai-wiki_200.npy";
const K: usize = 10;

fn cap_n() -> usize {
    std::env::var("SKEG_BENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000)
}
fn n_queries() -> usize {
    std::env::var("SKEG_NQ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50)
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
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let inv = if norm > 1e-10 { 1.0 / norm } else { 0.0 };
    v.iter().map(|&x| x * inv).collect()
}

fn brute_top_k(corpus: &[Vec<f32>], q: &[f32]) -> Vec<u64> {
    let mut s: Vec<(f32, u64)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (cosine_f32(q, v), i as u64))
        .collect();
    s.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    s.iter().take(K).map(|&(_, id)| id).collect()
}

fn main() {
    let (raw, rows, dim) = match load_npy(&format!("{ROOT}/{CORPUS}")) {
        Some(x) => x,
        None => {
            println!("corpus missing ({CORPUS}); skipping");
            return;
        }
    };
    let (qraw, qrows, qdim) = load_npy(&format!("{ROOT}/{QUERY}")).expect("queries");
    assert_eq!(dim, qdim);
    let n = cap_n().min(rows);
    let nq = n_queries().min(qrows);
    println!("tq1 hybrid end-to-end on DiskVamana (mxbai): dim {dim}, n {n}, queries {nq}");

    let corpus: Vec<Vec<f32>> = (0..n).map(|i| unit(&raw[i * dim..(i + 1) * dim])).collect();
    let queries: Vec<Vec<f32>> = (0..nq)
        .map(|i| unit(&qraw[i * dim..(i + 1) * dim]))
        .collect();

    let probe = QuantizedVectors::build(&corpus[0], dim, QuantKind::TurboQuant { bits: 1 });
    assert_eq!(
        probe.tq1_proxy_mode(),
        Some(Tq1ProxyMode::Hybrid),
        "dim {dim} must select the hybrid proxy"
    );
    println!("  proxy mode at dim {dim}: Hybrid (confirmed)");

    let tmp = tempfile::TempDir::new().unwrap();
    let t_build = std::time::Instant::now();
    let mut idx = DiskVamanaIndex::create_empty_with_tier(
        tmp.path(),
        dim,
        100,
        QuantKind::TurboQuant { bits: 1 },
    )
    .unwrap();
    for (id, v) in corpus.iter().enumerate() {
        idx.insert(id as u64, v).unwrap();
    }
    idx.consolidate().unwrap();
    drop(idx);
    let build_s = t_build.elapsed().as_secs_f64();

    let idx =
        DiskVamanaIndex::open_with_tier(tmp.path(), QuantKind::TurboQuant { bits: 1 }).unwrap();

    let mut hits = 0usize;
    let mut lat_us = 0.0f64;
    for q in &queries {
        let truth: AHashSet<u64> = brute_top_k(&corpus, q).into_iter().collect();
        let t = std::time::Instant::now();
        let got = idx.search(q, K).unwrap();
        lat_us += t.elapsed().as_secs_f64() * 1e6;
        hits += got.iter().filter(|(id, _)| truth.contains(id)).count();
    }
    let recall = hits as f64 / (nq * K) as f64;
    println!(
        "  build {build_s:.1}s   recall@{K} {recall:.4}   mean query {:.0} us",
        lat_us / nq as f64
    );
    assert!(
        recall >= 0.90,
        "hybrid end-to-end recall {recall:.4} below 0.90 - the wired path regressed"
    );
    println!("  PASS: hybrid DiskVamana recall@{K} = {recall:.4} (>= 0.90)");
}
