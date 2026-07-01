#![allow(clippy::cast_precision_loss)]

//! Rerank-budget sweep on the REAL DiskVamana tq1 path, all embeddings.
//!
//! The rerank budget is the number of candidates read from disk and scored with
//! exact f32 - the query-time recall/disk-read knob (does not touch writes, so
//! it fits the RW/streaming identity). Baseline is the current default k*4 = 40
//! for k=10. This sweeps it and reports recall@10 + mean latency per dataset, so
//! we can see whether raising it breaks the ~0.98 ceiling that L_search and
//! l_build both saturate below, and at what latency (disk-read) cost.
//!
//! dim < TQ1_HYBRID_MIN_DIM (512) runs the asymmetric proxy, >= runs hybrid;
//! the rerank knob helps both. dim is zero-padded to a multiple of 8.

use ahash::AHashSet;
use skeg_simd::cosine_f32;
use skeg_vector::{DiskVamanaIndex, QuantKind};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
const K: usize = 10;
const L_SEARCH: usize = 300;
const RERANKS: [usize; 5] = [40, 80, 160, 320, 640];

const DATASETS: &[(&str, &str, &str, usize)] = &[
    (
        "glove",
        "skeg-bench/data/glove_corpus.npy",
        "skeg-bench/data/glove_queries.npy",
        100,
    ),
    (
        "minilm",
        "skeg/bench-compare/embeddings_cache/corpus_minilm-wiki.npy",
        "skeg/bench-compare/embeddings_cache/queries_minilm-wiki_200.npy",
        384,
    ),
    (
        "mnist",
        "skeg-bench/data/mnist_corpus_60k.npy",
        "skeg-bench/data/mnist_queries_200.npy",
        784,
    ),
    (
        "mxbai",
        "skeg/bench-compare/embeddings_cache/corpus_mxbai-wiki.npy",
        "skeg/bench-compare/embeddings_cache/queries_mxbai-wiki_200.npy",
        1024,
    ),
    (
        "qwen3-emb-4b",
        "skeg/bench-compare/embeddings_cache/corpus_qwen3emb4b_100k.npy",
        "skeg/bench-compare/embeddings_cache/queries_qwen3emb4b_1k.npy",
        2560,
    ),
];

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

/// Load, cap, zero-pad each row to `pad`, unit-normalise.
fn load_prep(path: &str, n_cap: usize, pad: usize) -> Option<(Vec<Vec<f32>>, usize)> {
    let (data, rows, dim) = load_npy(path)?;
    let n = n_cap.min(rows);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut v = vec![0.0f32; pad];
        v[..dim].copy_from_slice(&data[i * dim..i * dim + dim]);
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in &mut v {
                *x /= norm;
            }
        }
        out.push(v);
    }
    Some((out, n))
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

fn run(
    label: &str,
    corpus_rel: &str,
    query_rel: &str,
    native_dim: usize,
    n_cap: usize,
    nq_cap: usize,
) {
    let pad = native_dim.next_multiple_of(8);
    let Some((corpus, n)) = load_prep(&format!("{ROOT}/{corpus_rel}"), n_cap, pad) else {
        println!("  {label:<12} dataset missing");
        return;
    };
    let Some((queries, _)) = load_prep(&format!("{ROOT}/{query_rel}"), nq_cap, pad) else {
        println!("  {label:<12} queries missing");
        return;
    };
    let nq = queries.len();
    let truth: Vec<AHashSet<u64>> = queries.iter().map(|q| brute(&corpus, q)).collect();

    let tmp = tempfile::TempDir::new().unwrap();
    let mut idx = DiskVamanaIndex::create_empty_with_tier(
        tmp.path(),
        pad,
        L_SEARCH,
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

    let mode = if pad >= 512 { "hybrid" } else { "asym" };
    print!("  {label:<12} dim {pad:<4} n {n:<6} [{mode}]  ");
    for &rr in &RERANKS {
        let mut hits = 0usize;
        let mut us = 0.0f64;
        for (q, t) in queries.iter().zip(&truth) {
            let start = std::time::Instant::now();
            let got = idx.search_with_params(q, K, L_SEARCH, rr).unwrap();
            us += start.elapsed().as_secs_f64() * 1e6;
            hits += got.iter().filter(|(id, _)| t.contains(id)).count();
        }
        let recall = hits as f64 / (nq * K) as f64;
        print!("rr{rr}: {recall:.3}/{:.0}us  ", us / nq as f64);
    }
    println!();
}

fn main() {
    let n_cap = env("SKEG_BENCH_N", 10_000);
    let nq_cap = env("SKEG_NQ", 50);
    println!("=====================================================================");
    println!("tq1 rerank-budget sweep on real DiskVamana  (L_search={L_SEARCH}, k={K})");
    println!("baseline rr40 = current default (k*4); each cell = recall@10 / mean us");
    println!("N={n_cap} nq={nq_cap}");
    println!("=====================================================================");
    for &(label, c, q, dim) in DATASETS {
        run(label, c, q, dim, n_cap, nq_cap);
    }
}
