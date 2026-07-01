#![allow(clippy::cast_precision_loss)]

//! Does pure popcount, given its low per-candidate cost, reach hybrid recall by
//! searching DEEPER (wider L_search) instead of paying the asymmetric re-score?
//! Sweeps L_search x rerank for the mode in SKEG_TQ1_MODE on mxbai + qwen (high
//! dim, where the hybrid's asym re-score is most expensive), reporting recall@10
//! + p50 latency. Compare pop's (L, rr) curve against hybrid at L300/rr80.
//!
//! Reuses the study index cache (SKEG_STUDY_DIR). Run:
//!   for m in pop hybrid asym; do SKEG_TQ1_MODE=$m cargo bench ... tq1_pop_deep; done

use ahash::AHashSet;
use skeg_simd::cosine_f32;
use skeg_vector::{DiskVamanaIndex, QuantKind};
use std::path::{Path, PathBuf};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
const K: usize = 10;
const LS: [usize; 4] = [300, 600, 1000, 1500];
const RERANKS: [usize; 2] = [80, 160];

const DATASETS: &[(&str, &str, &str, usize)] = &[
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

fn brute(corpus: &[Vec<f32>], q: &[f32]) -> AHashSet<u64> {
    let mut s: Vec<(f32, u64)> = corpus
        .iter()
        .enumerate()
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
    let mut idx =
        DiskVamanaIndex::create_empty_with_tier(dir, pad, 1500, QuantKind::TurboQuant { bits: 1 })
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
        println!("  {label} dataset missing");
        return;
    };
    let Some((queries, _)) = load_prep(&format!("{ROOT}/{query_rel}"), nq_cap, pad) else {
        println!("  {label} queries missing");
        return;
    };
    let nq = queries.len();
    let truth: Vec<AHashSet<u64>> = queries.iter().map(|q| brute(&corpus, q)).collect();
    // NOTE: cache key uses the deep-build l_search (1500) to avoid clashing with
    // the other studies' shallower-built indexes.
    let dir = study_dir().join(format!("deep_{label}_n{n}_d{pad}"));
    let idx = open_or_build(&dir, &corpus, pad, n);
    let mode = std::env::var("SKEG_TQ1_MODE").unwrap_or_else(|_| "auto".into());

    for &rr in &RERANKS {
        for &l in &LS {
            let mut hits = 0usize;
            let mut lats = Vec::with_capacity(nq);
            for (q, t) in queries.iter().zip(&truth) {
                let s = std::time::Instant::now();
                let got = idx.search_with_params(q, K, l, rr).unwrap();
                lats.push(s.elapsed().as_secs_f64() * 1e6);
                hits += got.iter().filter(|(id, _)| t.contains(id)).count();
            }
            lats.sort_unstable_by(f64::total_cmp);
            println!(
                "  {label:<8} {mode:<7} L{l:<4} rr{rr:<3}  recall {:.4}  p50 {:>5.0}us",
                hits as f64 / (nq * K) as f64,
                lats[nq / 2],
            );
        }
    }
}

fn main() {
    let n_cap = env("SKEG_BENCH_N", 50_000);
    let nq_cap = env("SKEG_NQ", 200);
    println!("=====================================================================");
    println!(
        "tq1 pop-deep: can popcount reach hybrid via a deeper walk?  mode={}",
        std::env::var("SKEG_TQ1_MODE").unwrap_or_else(|_| "auto".into())
    );
    println!("=====================================================================");
    for &(label, c, q, dim) in DATASETS {
        run(label, c, q, dim, n_cap, nq_cap);
    }
}
