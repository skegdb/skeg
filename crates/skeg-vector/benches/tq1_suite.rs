#![allow(clippy::cast_precision_loss)]

//! tq1 benchmark suite - publication-grade, reproducible.
//!
//! For each real embedding set, on the real DiskVamana path, reports the full
//! picture for one configuration: recall@10, recall@100, p50/p99 query latency,
//! single-thread QPS, tq1 RAM/vector, and build time. Configuration is explicit
//! and reproducible:
//!   SKEG_TQ1_MODE = pop | hybrid | asym   (proxy; unset = auto by dim)
//!   SKEG_SUITE_RR = rerank budget (disk reads; default 80 = new k*8)
//!   SKEG_LSEARCH  = walk list size (default 300)
//!   SKEG_BENCH_N  = corpus cap (default 50000)   SKEG_NQ = queries (default 200)
//!   SKEG_PASSES   = latency passes for stable percentiles (default 5)
//!
//! Recall is deterministic (index + queries fixed, seed fixed). Latency is the
//! pooled per-query distribution over PASSES, single-thread sequential. Indexes
//! are built once and cached (SKEG_STUDY_DIR) so baseline vs shipped runs reuse
//! them. Drive both configs:
//!   SKEG_TQ1_MODE=asym   SKEG_SUITE_RR=40 cargo bench ... tq1_suite   # main baseline
//!   SKEG_TQ1_MODE=hybrid SKEG_SUITE_RR=80 cargo bench ... tq1_suite   # shipped

use ahash::AHashSet;
use skeg_simd::cosine_f32;
use skeg_vector::{DiskVamanaIndex, QuantKind};
use std::path::{Path, PathBuf};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
const L_BUILD: usize = 300;

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
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
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

fn brute(corpus: &[Vec<f32>], q: &[f32], k: usize) -> Vec<u64> {
    let mut s: Vec<(f32, u64)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (cosine_f32(q, v), i as u64))
        .collect();
    s.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    s.iter().take(k).map(|&(_, id)| id).collect()
}

fn study_dir() -> PathBuf {
    std::env::var("SKEG_STUDY_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("skeg_tq1_study"))
}

fn open_or_build(dir: &Path, corpus: &[Vec<f32>], pad: usize, n: usize) -> (DiskVamanaIndex, f64) {
    if let Ok(i) = DiskVamanaIndex::open_with_tier(dir, QuantKind::TurboQuant { bits: 1 })
        && i.len() == n
    {
        return (i, 0.0);
    }
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let t = std::time::Instant::now();
    let mut idx =
        DiskVamanaIndex::create_empty_with_tier(dir, pad, L_BUILD, QuantKind::TurboQuant { bits: 1 })
            .unwrap();
    for (id, v) in corpus.iter().enumerate() {
        idx.insert(id as u64, v).unwrap();
    }
    idx.consolidate().unwrap();
    let build_s = t.elapsed().as_secs_f64();
    drop(idx);
    (
        DiskVamanaIndex::open_with_tier(dir, QuantKind::TurboQuant { bits: 1 }).unwrap(),
        build_s,
    )
}

fn recall(got: &[(u64, f32)], truth: &[u64], k: usize) -> f64 {
    let t: AHashSet<u64> = truth.iter().take(k).copied().collect();
    got.iter().take(k).filter(|(id, _)| t.contains(id)).count() as f64 / k as f64
}

fn run(label: &str, corpus_rel: &str, query_rel: &str, native_dim: usize, cfg: (usize, usize, usize, usize)) {
    let (n_cap, nq_cap, l_search, rr) = cfg;
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
    let dir = study_dir().join(format!("{label}_n{n}_d{pad}"));
    let (idx, build_s) = open_or_build(&dir, &corpus, pad, n);

    let truth10: Vec<Vec<u64>> = queries.iter().map(|q| brute(&corpus, q, 10)).collect();
    let truth100: Vec<Vec<u64>> = queries.iter().map(|q| brute(&corpus, q, 100)).collect();

    // Recall (deterministic).
    let mut r10 = 0.0;
    let mut r100 = 0.0;
    for (qi, q) in queries.iter().enumerate() {
        let g10 = idx.search_with_params(q, 10, l_search, rr).unwrap();
        let g100 = idx.search_with_params(q, 100, l_search, rr.max(100)).unwrap();
        r10 += recall(&g10, &truth10[qi], 10);
        r100 += recall(&g100, &truth100[qi], 100);
    }
    r10 /= nq as f64;
    r100 /= nq as f64;

    // Latency: pooled per-query over PASSES passes (single-thread, k=10).
    let passes = env("SKEG_PASSES", 5);
    let mut lats = Vec::with_capacity(nq * passes);
    for _ in 0..passes {
        for q in &queries {
            let s = std::time::Instant::now();
            let _ = idx.search_with_params(q, 10, l_search, rr).unwrap();
            lats.push(s.elapsed().as_secs_f64() * 1e6);
        }
    }
    lats.sort_unstable_by(f64::total_cmp);
    let p50 = lats[lats.len() / 2];
    let p99 = lats[lats.len() * 99 / 100];
    let mean = lats.iter().sum::<f64>() / lats.len() as f64;
    let qps = 1e6 / mean;
    let ram_mb = (pad / 8) as f64 * n as f64 / 1e6;

    println!(
        "  {label:<12} {pad:<4} {n:<7} {r10:.4}   {r100:.4}   {p50:>5.0}   {p99:>5.0}   {qps:>6.0}   {ram_mb:>5.1}   {build_s:>4.1}"
    );
}

fn main() {
    let cfg = (
        env("SKEG_BENCH_N", 50_000),
        env("SKEG_NQ", 200),
        env("SKEG_LSEARCH", 300),
        env("SKEG_SUITE_RR", 80),
    );
    let mode = std::env::var("SKEG_TQ1_MODE").unwrap_or_else(|_| "auto".into());
    println!("=====================================================================================");
    println!(
        "skeg tq1 suite | proxy={mode} rerank={} l_search={} N<={} nq<={} passes={} | k=10",
        cfg.3, cfg.2, cfg.0, cfg.1, env("SKEG_PASSES", 5)
    );
    println!("recall deterministic; latency = pooled single-thread us over passes; RAM = tq1 codes");
    println!("=====================================================================================");
    println!("  dataset      dim  n       rec@10   rec@100  p50   p99   qps      RAM    bld");
    for &(label, c, q, dim) in DATASETS {
        run(label, c, q, dim, cfg);
    }
}
