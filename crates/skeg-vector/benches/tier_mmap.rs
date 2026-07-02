#![allow(clippy::cast_precision_loss)]
//! tq2 tier-mmap: does mmap'ing the TurboQuant codes lower RSS, and what does it
//! cost in latency (page faults during the walk)? In-process (no server, no serve
//! mode): build once, drop the corpus, then open the SAME index twice - codes
//! owned (RAM) vs mmap - measuring RSS (ps on self), recall@10/@100, p50/p99.
//!   SKEG_BENCH_N=500000  SKEG_NQ=200  SKEG_CORPUS=<npy>  SKEG_QUERY=<npy>

use ahash::AHashSet;
use rayon::prelude::*;
use skeg_simd::cosine_f32;
use skeg_vector::{DiskVamanaIndex, QuantKind};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");

fn load(path: &str, cap: usize) -> (Vec<Vec<f32>>, usize) {
    let bytes = std::fs::read(path).unwrap();
    let hl = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + hl]).unwrap();
    let sh = header.find("'shape':").unwrap();
    let lp = header[sh..].find('(').unwrap() + sh + 1;
    let rp = header[lp..].find(')').unwrap() + lp;
    let dims: Vec<usize> = header[lp..rp]
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let (rows, dim) = (dims[0], dims[1]);
    let data: Vec<f32> = bytes[10 + hl..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let n = cap.min(rows);
    let out = (0..n)
        .map(|i| {
            let mut v = data[i * dim..i * dim + dim].to_vec();
            let nrm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
            v.iter_mut().for_each(|x| *x /= nrm);
            v
        })
        .collect();
    (out, dim)
}

/// Resident set size of THIS process, in MiB (via `ps`; macOS/Linux).
fn rss_mib() -> f64 {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok();
    out.and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .map(|kb| kb / 1024.0)
        .unwrap_or(0.0)
}

fn pctl(mut v: Vec<f64>) -> (f64, f64) {
    v.sort_by(|a, b| a.total_cmp(b));
    (v[v.len() / 2], v[(v.len() as f64 * 0.99) as usize])
}

fn dir_for(bits: u8) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("skeg_tiermmap_{bits}"))
}

fn main() {
    let n_cap = std::env::var("SKEG_BENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(500_000);
    let nq = std::env::var("SKEG_NQ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let qpath = std::env::var("SKEG_QUERY").unwrap_or_else(|_| {
        format!("{ROOT}/skeg/bench-compare/embeddings_cache/queries_mxbai-wiki_200.npy")
    });

    // MEASURE mode: open a PRE-BUILT index (never load the corpus) so RSS reflects
    // the index only. `SKEG_MEASURE=<bits> SKEG_MMAP=<0|1>`.
    if let Ok(bits) = std::env::var("SKEG_MEASURE").map(|s| s.parse::<u8>().unwrap()) {
        let mmap = std::env::var("SKEG_MMAP").as_deref() == Ok("1");
        let (queries, _) = load(&qpath, nq);
        let tier = QuantKind::TurboQuant { bits };
        let idx = DiskVamanaIndex::open_with_tier_full(&dir_for(bits), tier, mmap, false).unwrap();
        for q in queries.iter().take(32) {
            idx.search_with_params(q, 10, 300, 80).unwrap();
        }
        let mut lat = Vec::with_capacity(queries.len());
        for q in &queries {
            let s = std::time::Instant::now();
            idx.search_with_params(q, 10, 300, 80).unwrap();
            lat.push(s.elapsed().as_secs_f64() * 1e6);
        }
        for _ in 0..queries.len() {
            idx.search_with_params(&queries[0], 100, 300, 800).unwrap();
        }
        let rss = rss_mib();
        let (p50, p99) = pctl(lat.clone());
        let qps = 1e6 / (lat.iter().sum::<f64>() / lat.len() as f64);
        println!(
            "  tq{bits} {:<6} RSS {rss:>6.0} MiB  p50 {p50:.0}us  p99 {p99:.0}us  qps {qps:.0}",
            if mmap { "mmap" } else { "owned" }
        );
        return;
    }

    // BUILD mode (default): build tq2 + tq1 to fixed dirs, print recall (needs the
    // corpus), then exit so the corpus memory is gone before any MEASURE run.
    let cpath = std::env::var("SKEG_CORPUS").unwrap_or_else(|_| {
        format!("{ROOT}/skeg/bench-compare/embeddings_cache/corpus_mxbai-wiki.npy")
    });
    let (corpus, dim) = load(&cpath, n_cap);
    let (queries, _) = load(&qpath, nq);
    let n = corpus.len();
    let mk = |k: usize| -> Vec<AHashSet<u64>> {
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
    };
    let t10 = mk(10);
    let t100 = mk(100);
    println!("build: {n} x {dim}, {} queries", queries.len());
    for bits in [2u8, 1] {
        let dir = dir_for(bits);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut idx =
            DiskVamanaIndex::create_empty_with_tier(&dir, dim, 300, QuantKind::TurboQuant { bits })
                .unwrap();
        for (id, v) in corpus.iter().enumerate() {
            idx.insert(id as u64, v).unwrap();
        }
        idx.consolidate().unwrap();
        // recall (valid; mmap does not change results) - measured here with the corpus.
        let mut h10 = 0usize;
        let mut h100 = 0usize;
        for (q, tr) in queries.iter().zip(&t10) {
            h10 += idx
                .search_with_params(q, 10, 300, 80)
                .unwrap()
                .iter()
                .filter(|(id, _)| tr.contains(id))
                .count();
        }
        for (q, tr) in queries.iter().zip(&t100) {
            h100 += idx
                .search_with_params(q, 100, 300, 800)
                .unwrap()
                .iter()
                .filter(|(id, _)| tr.contains(id))
                .count();
        }
        println!(
            "  tq{bits} built (codes {} MiB)  recall@10 {:.4}  recall@100 {:.4}",
            n * dim * bits as usize / 8 / 1_048_576,
            h10 as f64 / (queries.len() * 10) as f64,
            h100 as f64 / (queries.len() * 100) as f64,
        );
    }
}
