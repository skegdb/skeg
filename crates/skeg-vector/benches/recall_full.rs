#![allow(clippy::cast_precision_loss)]
//! COMPLETE recall eval: recall@10 AND recall@100 (real k-searches, not the
//! k=10-vs-top-100 metric), tq1 vs tq2, at the default serving params (l=300)
//! and a wide walk (l=2000). One dataset via env; run it at every scale you
//! evaluate. Dims zero-padded to a multiple of 8.
//!   SKEG_BENCH_N=100000  SKEG_NQ=200  SKEG_CORPUS=<npy>  SKEG_QUERY=<npy>

use ahash::AHashSet;
use rayon::prelude::*;
use skeg_simd::cosine_f32;
use skeg_vector::{DiskVamanaIndex, QuantKind};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");

fn load_npy(path: &str) -> (Vec<f32>, usize, usize) {
    let bytes = std::fs::read(path).unwrap_or_else(|_| panic!("missing {path}"));
    let hl = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + hl]).unwrap();
    let sh = header.find("'shape':").unwrap();
    let lp = header[sh..].find('(').unwrap() + sh + 1;
    let rp = header[lp..].find(')').unwrap() + lp;
    let dims: Vec<usize> = header[lp..rp]
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let data: Vec<f32> = bytes[10 + hl..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (data, dims[0], dims[1])
}

fn load_prep(path: &str, n_cap: usize, pad: usize) -> (Vec<Vec<f32>>, usize) {
    let (data, rows, dim) = load_npy(path);
    let n = n_cap.min(rows);
    let out = (0..n)
        .map(|i| {
            let mut v = vec![0.0f32; pad];
            v[..dim].copy_from_slice(&data[i * dim..i * dim + dim]);
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
            v.iter_mut().for_each(|x| *x /= norm);
            v
        })
        .collect();
    (out, n)
}

fn truth(corpus: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> Vec<AHashSet<u64>> {
    queries
        .par_iter()
        .map(|q| {
            let mut t: Vec<(f32, u64)> = corpus
                .iter()
                .enumerate()
                .map(|(i, v)| (cosine_f32(q, v), i as u64))
                .collect();
            t.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            t.iter().take(k).map(|&(_, id)| id).collect()
        })
        .collect()
}

/// Current process RSS in MB (macOS `ps -o rss=` is KB).
fn rss_mb() -> f64 {
    let pid = std::process::id().to_string();
    std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .map(|kb| kb / 1024.0)
        .unwrap_or(0.0)
}

fn main() {
    let n_cap = std::env::var("SKEG_BENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);
    let nq = std::env::var("SKEG_NQ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let native: usize = std::env::var("SKEG_DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let cpath = std::env::var("SKEG_CORPUS").unwrap_or_else(|_| {
        format!("{ROOT}/skeg/bench-compare/embeddings_cache/corpus_mxbai-wiki.npy")
    });
    let qpath = std::env::var("SKEG_QUERY").unwrap_or_else(|_| {
        format!("{ROOT}/skeg/bench-compare/embeddings_cache/queries_mxbai-wiki_200.npy")
    });
    let pad = native.next_multiple_of(8);
    let (mut corpus, n) = load_prep(&cpath, n_cap, pad);
    let (queries, _) = load_prep(&qpath, nq, pad);
    let t10 = truth(&corpus, &queries, 10);
    let t100 = truth(&corpus, &queries, 100);
    // Guard: recall@k must compare a top-k search against the true top-k. The
    // old bug reported "top-10 search inside top-100 truth" as recall@100 - a
    // flatteringly high number that hid real degradation. Pin the truth sizes so
    // t10/t100 can never be silently swapped or truncated to the wrong k.
    assert!(
        t10.iter().all(|s| s.len() == 10.min(n)) && t100.iter().all(|s| s.len() == 100.min(n)),
        "recall ground truth must hold exactly k ids per query (10 / 100)"
    );
    println!("recall (real): {n} x {pad}, {} queries", queries.len());
    println!(
        "{:<5} {:<8}  {:>10}  {:>11}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>9}",
        "tier",
        "walk",
        "recall@10",
        "recall@100",
        "p50ms",
        "p99ms",
        "ramIdle",
        "rssIdle",
        "rssHot",
        "qps"
    );

    // SKEG_BITS=1 (or 2/4) restricts to one tier - skips the wasted tq2 build on
    // big-N runs. Default: both 1 and 2.
    let bits_list: Vec<u8> = match std::env::var("SKEG_BITS").ok().and_then(|s| s.parse().ok()) {
        Some(b) => vec![b],
        None => vec![1u8, 2],
    };
    // SKEG_TIER=int8 tests the int8 tier instead of TurboQuant (only int8 and
    // turboquant are RW-rebuildable). Default: TurboQuant per bits_list.
    let tiers: Vec<(String, QuantKind)> = match std::env::var("SKEG_TIER").ok().as_deref() {
        Some("int8") => vec![("int8".to_string(), QuantKind::Int8)],
        _ => bits_list
            .iter()
            .map(|&b| (format!("tq{b}"), QuantKind::TurboQuant { bits: b }))
            .collect(),
    };
    for (tier_label, tier) in tiers {
        let tmp = std::env::temp_dir().join(format!("skeg_rfull_{tier_label}"));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let build_t = std::time::Instant::now();
        let mut idx = DiskVamanaIndex::create_empty_with_tier(&tmp, pad, 300, tier).unwrap();
        for (id, v) in corpus.iter().enumerate() {
            idx.insert(id as u64, v).unwrap();
        }
        idx.consolidate().unwrap();
        let build_s = build_t.elapsed().as_secs_f64();
        // resident = logical index footprint (graph + tq1 codes).
        let ram_mb = idx.resident_bytes() as f64 / (1024.0 * 1024.0);
        // Free the bench's f32 corpus so RSS reflects index + runtime, not the
        // corpus (which lives on disk in production). Only safe with one tier.
        let single = matches!(
            std::env::var("SKEG_BITS").ok().as_deref(),
            Some("1" | "2" | "4")
        ) || std::env::var("SKEG_TIER").is_ok();
        if single {
            corpus = Vec::new();
        }
        // RSS at rest: no query yet, so vectors.bin pages are cold.
        let rss_idle = rss_mb();
        let mut rss_hot = rss_idle;
        // Custom operating point via env (SKEG_LS + SKEG_RR) for uniform
        // cross-dataset sweeps; falls back to the two fixed points.
        let custom: Option<(usize, usize)> = match (
            std::env::var("SKEG_LS").ok().and_then(|s| s.parse().ok()),
            std::env::var("SKEG_RR").ok().and_then(|s| s.parse().ok()),
        ) {
            (Some(ls), Some(rr)) => Some((ls, rr)),
            _ => None,
        };
        let points: Vec<(&str, usize, usize, usize)> = match custom {
            Some((ls, rr)) => vec![("custom", ls, rr / 10, rr)],
            None => vec![
                ("default", 300usize, 80usize, 800usize),
                ("wide", 2000, 1280, 12800),
            ],
        };
        for &(label, ls, rr10, rr100) in &points {
            // warmup (discard) to avoid first-query mmap/cache noise
            for q in queries.iter().take(5) {
                let _ = idx.search_with_params(q, 100, ls, rr100).unwrap();
            }
            let mut h10 = 0usize;
            let mut h100 = 0usize;
            for (q, tr) in queries.iter().zip(&t10) {
                h10 += idx
                    .search_with_params(q, 10, ls, rr10)
                    .unwrap()
                    .iter()
                    .filter(|(id, _)| tr.contains(id))
                    .count();
            }
            let mut lat = Vec::with_capacity(queries.len());
            for (q, tr) in queries.iter().zip(&t100) {
                let t = std::time::Instant::now();
                let res = idx.search_with_params(q, 100, ls, rr100).unwrap();
                lat.push(t.elapsed().as_secs_f64() * 1e3);
                h100 += res.iter().filter(|(id, _)| tr.contains(id)).count();
            }
            lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let pct = |p: f64| lat[(((lat.len() as f64) * p) as usize).min(lat.len() - 1)];
            // QPS: real multi-thread throughput - all queries x reps run
            // concurrently across the rayon pool (not 1/latency).
            let jobs: Vec<&Vec<f32>> = (0..10).flat_map(|_| queries.iter()).collect();
            let qt = std::time::Instant::now();
            jobs.par_iter().for_each(|q| {
                let _ = idx.search_with_params(q, 100, ls, rr100).unwrap();
            });
            let qps = jobs.len() as f64 / qt.elapsed().as_secs_f64();
            // After all queries, vectors.bin rerank pages are hot -> RSS peak.
            rss_hot = rss_hot.max(rss_mb());
            println!(
                "{tier_label:<5} {label:<8}  {:>10.4}  {:>11.4}  {:>8.2}  {:>8.2}  {:>8.1}  {:>8.1}  {:>8.1}  {:>9.0}  {:>8.2}",
                h10 as f64 / (queries.len() * 10) as f64,
                h100 as f64 / (queries.len() * 100) as f64,
                pct(0.50),
                pct(0.99),
                ram_mb,
                rss_idle,
                rss_hot,
                qps,
                build_s,
            );
        }
        drop(idx);
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
