//! skeg-bench: one unified benchmark tool so we stop rewriting ad-hoc benches
//! (and repeating the same measurement mistakes). Correct-by-construction:
//!   - recall@10 AND recall@100, both from REAL k-searches vs brute truth
//!   - RSS measured in a SUBPROCESS that opens the index but never loads the
//!     corpus (so RSS is the index's, not 2 GB of corpus + jemalloc retention)
//!   - build time, p50/p99, QPS, per (dataset, tier, config)
//!   - datasets from a registry, zero-padded to a multiple of 8
//!
//! Usage:
//!   skeg-bench --dataset mxbai500k --tier tq2 --tier tq1 [--n N] [--nq NQ]
//!              [--l-search 300] [--rerank 800] [--mmap]
//!   (internal) skeg-bench --measure DIR BITS MMAP QPATH DIM NQ LSEARCH RERANK

use ahash::AHashSet;
use rayon::prelude::*;
use skeg_simd::cosine_f32;
use skeg_vector::{DiskVamanaIndex, QuantKind};
use std::path::{Path, PathBuf};

/// Workspace root (skeg-bench lives under crates/skeg-vector).
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

/// (name, corpus rel-path, query rel-path, native dim).
const DATASETS: &[(&str, &str, &str, usize)] = &[
    ("glove", "skeg-bench/data/glove_corpus.npy", "skeg-bench/data/glove_queries.npy", 100),
    ("minilm", "skeg/bench-compare/embeddings_cache/corpus_minilm-wiki.npy", "skeg/bench-compare/embeddings_cache/queries_minilm-wiki_200.npy", 384),
    ("mnist", "skeg-bench/data/mnist_corpus_60k.npy", "skeg-bench/data/mnist_queries_200.npy", 784),
    ("mxbai", "skeg/bench-compare/embeddings_cache/corpus_mxbai-wiki.npy", "skeg/bench-compare/embeddings_cache/queries_mxbai-wiki_200.npy", 1024),
    ("mxbai500k", "skeg/bench-compare/embeddings_cache/corpus_mxbai-wiki-chunked_500k.npy", "skeg/bench-compare/embeddings_cache/queries_mxbai-wiki-chunked_1000.npy", 1024),
    ("mxbai1m", "skeg/bench-compare/embeddings_cache/corpus_mxbai-wiki-chunked_1m.npy", "skeg/bench-compare/embeddings_cache/queries_mxbai-wiki-chunked_1000.npy", 1024),
    ("qwen", "skeg/bench-compare/embeddings_cache/corpus_qwen3emb4b_100k.npy", "skeg/bench-compare/embeddings_cache/queries_qwen3emb4b_1k.npy", 2560),
];

fn tier_of(s: &str) -> (QuantKind, u8) {
    match s {
        "tq1" => (QuantKind::TurboQuant { bits: 1 }, 1),
        "tq2" => (QuantKind::TurboQuant { bits: 2 }, 2),
        "tq4" => (QuantKind::TurboQuant { bits: 4 }, 4),
        _ => panic!("unknown tier '{s}' (tq1|tq2|tq4)"),
    }
}

fn load_npy(path: &Path, cap: usize, pad: usize) -> (Vec<Vec<f32>>, usize) {
    let bytes = std::fs::read(path).unwrap_or_else(|_| panic!("missing dataset: {}", path.display()));
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
            let mut v = vec![0.0f32; pad];
            v[..dim].copy_from_slice(&data[i * dim..i * dim + dim]);
            let nrm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
            v.iter_mut().for_each(|x| *x /= nrm);
            v
        })
        .collect();
    (out, dim)
}

fn truth(corpus: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> Vec<AHashSet<u64>> {
    queries
        .par_iter()
        .map(|q| {
            let mut t: Vec<(f32, u64)> = corpus.iter().enumerate().map(|(i, v)| (cosine_f32(q, v), i as u64)).collect();
            t.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            t.iter().take(k).map(|&(_, id)| id).collect()
        })
        .collect()
}

fn rss_mib() -> f64 {
    std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .map(|kb| kb / 1024.0)
        .unwrap_or(0.0)
}

/// Subprocess entry: open the pre-built index (NO corpus) and report RSS + p50 +
/// p99 + qps for a k=10 search. Keeps RSS free of corpus/jemalloc pollution.
fn measure_mode(args: &[String]) {
    let dir = PathBuf::from(&args[0]);
    let bits: u8 = args[1].parse().unwrap();
    let mmap = args[2] == "1";
    let qpath = PathBuf::from(&args[3]);
    let dim: usize = args[4].parse().unwrap();
    let nq: usize = args[5].parse().unwrap();
    let ls: usize = args[6].parse().unwrap();
    let rr: usize = args[7].parse().unwrap();
    let (queries, _) = load_npy(&qpath, nq, dim);
    let idx = DiskVamanaIndex::open_with_tier_full(&dir, QuantKind::TurboQuant { bits }, mmap, false).unwrap();
    for q in queries.iter().take(32) {
        idx.search_with_params(q, 10, ls, rr).unwrap();
    }
    let mut lat: Vec<f64> = Vec::with_capacity(queries.len());
    for q in &queries {
        let s = std::time::Instant::now();
        idx.search_with_params(q, 10, ls, rr).unwrap();
        lat.push(s.elapsed().as_secs_f64() * 1e6);
    }
    lat.sort_by(|a, b| a.total_cmp(b));
    let p50 = lat[lat.len() / 2];
    let p99 = lat[(lat.len() as f64 * 0.99) as usize];
    let qps = 1e6 / (lat.iter().sum::<f64>() / lat.len() as f64);
    // Parseable line for the parent.
    println!("MEASURE rss={:.0} p50={p50:.0} p99={p99:.0} qps={qps:.0}", rss_mib());
}

struct Args {
    datasets: Vec<String>,
    tiers: Vec<String>,
    n: usize,
    nq: usize,
    l_search: usize,
    rerank: usize,
    mmap: bool,
}

fn parse_args(argv: &[String]) -> Args {
    let mut a = Args {
        datasets: vec![],
        tiers: vec![],
        n: usize::MAX,
        nq: 200,
        l_search: 300,
        rerank: 800,
        mmap: false,
    };
    let mut i = 0;
    while i < argv.len() {
        let next = || argv.get(i + 1).cloned().unwrap_or_default();
        match argv[i].as_str() {
            "--dataset" => a.datasets.push(next()),
            "--tier" => a.tiers.push(next()),
            "--n" => a.n = next().parse().unwrap(),
            "--nq" => a.nq = next().parse().unwrap(),
            "--l-search" => a.l_search = next().parse().unwrap(),
            "--rerank" => a.rerank = next().parse().unwrap(),
            "--mmap" => {
                a.mmap = true;
                i += 1;
                continue;
            }
            other => panic!("unknown arg '{other}'"),
        }
        i += 2;
    }
    if a.datasets.is_empty() {
        a.datasets.push("mxbai".into());
    }
    if a.tiers.is_empty() {
        a.tiers = vec!["tq1".into(), "tq2".into()];
    }
    a
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.first().map(String::as_str) == Some("--measure") {
        measure_mode(&argv[1..]);
        return;
    }
    let a = parse_args(&argv);
    let exe = std::env::current_exe().unwrap();
    println!("RSS in a subprocess (no corpus). recall = real k-searches. mmap={}", a.mmap);
    println!(
        "{:<11} {:<4} {:>4} {:>8} {:>7} {:>6} {:>9} {:>10} {:>7} {:>7} {:>5}",
        "dataset", "tier", "dim", "n", "build_s", "RSS", "recall@10", "recall@100", "p50us", "p99us", "qps"
    );
    for dname in &a.datasets {
        let &(_, cpath, qpath, native) = DATASETS.iter().find(|d| d.0 == dname).unwrap_or_else(|| panic!("unknown dataset '{dname}'"));
        let pad = native.next_multiple_of(8);
        let (corpus, dim) = load_npy(&root().join(cpath), a.n, pad);
        let (queries, _) = load_npy(&root().join(qpath), a.nq, pad);
        let n = corpus.len();
        let t10 = truth(&corpus, &queries, 10);
        let t100 = truth(&corpus, &queries, 100);
        for tname in &a.tiers {
            let (tier, bits) = tier_of(tname);
            let dir = std::env::temp_dir().join(format!("skeg_bench_{dname}_{bits}"));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            let t = std::time::Instant::now();
            let mut idx = DiskVamanaIndex::create_empty_with_tier(&dir, dim, 300, tier).unwrap();
            for (id, v) in corpus.iter().enumerate() {
                idx.insert(id as u64, v).unwrap();
            }
            idx.consolidate().unwrap();
            let build_s = t.elapsed().as_secs_f64();
            let r = |k: usize, tr: &[AHashSet<u64>], rr: usize| -> f64 {
                let mut h = 0usize;
                for (q, t) in queries.iter().zip(tr) {
                    h += idx.search_with_params(q, k, a.l_search, rr).unwrap().iter().filter(|(id, _)| t.contains(id)).count();
                }
                h as f64 / (queries.len() * k) as f64
            };
            let r10 = r(10, &t10, a.rerank.max(80));
            let r100 = r(100, &t100, a.rerank);
            drop(idx);
            // RSS + latency in a clean subprocess (no corpus in RAM).
            let out = std::process::Command::new(&exe)
                .args(["--measure", dir.to_str().unwrap(), &bits.to_string(), if a.mmap { "1" } else { "0" }, &root().join(qpath).to_string_lossy(), &pad.to_string(), &a.nq.to_string(), &a.l_search.to_string(), &a.rerank.max(80).to_string()])
                .output()
                .unwrap();
            let line = String::from_utf8_lossy(&out.stdout);
            let get = |k: &str| line.split_whitespace().find_map(|w| w.strip_prefix(k)).unwrap_or("?").to_string();
            println!(
                "{dname:<11} {tname:<4} {pad:>4} {n:>8} {build_s:>7.0} {:>6} {r10:>9.4} {r100:>10.4} {:>7} {:>7} {:>5}",
                get("rss="), get("p50="), get("p99="), get("qps=")
            );
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}
