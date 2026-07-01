#![allow(clippy::cast_precision_loss)]

//! Exhaustive FILTERED-search study for tq1: recall@10 + p50/mean latency across
//! filter selectivity (sparse -> dense), embeddings up to qwen, hybrid vs asym.
//! Exercises the real DiskVamanaIndex::search_filtered path (two-walk for sparse
//! filters, navigate-all for dense) with the tq1 hybrid proxy + asym re-score.
//!
//! Reuses the mode-study index cache (SKEG_STUDY_DIR, same N) so it does not
//! rebuild. Force the proxy with SKEG_TQ1_MODE=hybrid|asym|pop:
//!   for m in hybrid asym; do SKEG_TQ1_MODE=$m cargo bench ... tq1_filtered_study; done

use ahash::AHashSet;
use skeg_simd::cosine_f32;
use skeg_vector::{DiskVamanaIndex, QuantKind};
use std::path::{Path, PathBuf};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
const K: usize = 10;
const L_SEARCH: usize = 300;
/// Filter selectivities: 1% (sparse two-walk), 10% (dense boundary), 50% (dense).
const SELECTIVITIES: [f64; 3] = [0.01, 0.10, 0.50];

const DATASETS: &[(&str, &str, &str, usize)] = &[
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

/// Brute top-k restricted to ids the filter accepts.
fn brute_filtered(corpus: &[Vec<f32>], q: &[f32], step: u64) -> AHashSet<u64> {
    let mut s: Vec<(f32, u64)> = corpus
        .iter()
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

fn open_or_build(dir: &Path, corpus: &[Vec<f32>], pad: usize, n: usize) -> DiskVamanaIndex {
    if let Ok(i) = DiskVamanaIndex::open_with_tier(dir, QuantKind::TurboQuant { bits: 1 })
        && i.len() == n
    {
        return i;
    }
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let mut idx = DiskVamanaIndex::create_empty_with_tier(
        dir,
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
    DiskVamanaIndex::open_with_tier(dir, QuantKind::TurboQuant { bits: 1 }).unwrap()
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
        println!("  {label:<10} dataset missing");
        return;
    };
    let Some((queries, _)) = load_prep(&format!("{ROOT}/{query_rel}"), nq_cap, pad) else {
        println!("  {label:<10} queries missing");
        return;
    };
    let nq = queries.len();
    let dir = study_dir().join(format!("{label}_n{n}_d{pad}"));
    let idx = open_or_build(&dir, &corpus, pad, n);
    let mode = std::env::var("SKEG_TQ1_MODE").unwrap_or_else(|_| "auto".into());

    for &sel in &SELECTIVITIES {
        let step = (1.0 / sel).round() as u64;
        let matches = move |id: u64| id % step == 0;
        let truth: Vec<AHashSet<u64>> = queries
            .iter()
            .map(|q| brute_filtered(&corpus, q, step))
            .collect();

        let mut hits = 0usize;
        let mut lats = Vec::with_capacity(nq);
        for (q, t) in queries.iter().zip(&truth) {
            let start = std::time::Instant::now();
            let got = idx
                .search_filtered(q, K, L_SEARCH, &matches, &[], sel as f32)
                .unwrap();
            lats.push(start.elapsed().as_secs_f64() * 1e6);
            hits += got.iter().filter(|(id, _)| t.contains(id)).count();
        }
        lats.sort_unstable_by(f64::total_cmp);
        let recall = hits as f64 / (nq * K) as f64;
        println!(
            "  {label:<10} dim {pad:<4} n {n:<7} mode {mode:<7} sel {:>4.0}%  recall {recall:.4}  p50 {:>5.0}us  mean {:>5.0}us",
            sel * 100.0,
            lats[nq / 2],
            lats.iter().sum::<f64>() / nq as f64,
        );
    }
}

fn main() {
    let n_cap = env("SKEG_BENCH_N", 50_000);
    let nq_cap = env("SKEG_NQ", 200);
    println!("=====================================================================");
    println!(
        "tq1 FILTERED study (SKEG_TQ1_MODE={})  L_search={L_SEARCH} k={K} N<={n_cap} nq<={nq_cap}",
        std::env::var("SKEG_TQ1_MODE").unwrap_or_else(|_| "auto".into())
    );
    println!("selectivity = fraction of the corpus the filter accepts");
    println!("=====================================================================");
    for &(label, c, q, dim) in DATASETS {
        run(label, c, q, dim, n_cap, nq_cap);
    }
}
