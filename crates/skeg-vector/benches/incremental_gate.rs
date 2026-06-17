//! Incremental-insert gate (branch `feat/incremental-insert`).
//!
//! Pre-registered gate for the LSM-of-segments design
//! (`docs/incremental-insert-design.md`). Streams real embeddings into a disk
//! VINDEX in batches, consolidating on the production geometric schedule, and
//! measures the two properties the design must hold:
//!
//!   1. recall@10 vs brute-force GT stays >= 0.98 throughout the stream
//!      (today: ~1.0, the delta is an exact f32 scan -- this catches a future
//!      merge bug, not quantization).
//!   2. p50 query latency at the pre-consolidation high-water mark is <= ~2x
//!      the latency just after a consolidation. TODAY THIS FAILS: the delta is
//!      brute-force scanned, so latency is O(delta) and the ratio blows past 2x.
//!      The LSM design (navigable runs instead of a flat delta scan) is what
//!      turns this gate green. This file is the red baseline.
//!
//! Run: `cargo bench -p skeg-vector --bench incremental_gate`.

use std::time::Instant;

use rayon::prelude::*;
use skeg_vector::{DiskVamanaIndex, QuantKind};

const MXBAI_CORPUS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../bench-compare/embeddings_cache/corpus_mxbai-wiki.npy"
);
const MXBAI_QUERY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../bench-compare/embeddings_cache/queries_mxbai-wiki_200.npy"
);

const K: usize = 10;
const N_QUERIES: usize = 100;
const L_SEARCH: usize = 200;
const N_STREAM: usize = 30_000; // cap for gate runtime
const DISK_CONSOLIDATE_MIN: usize = 4096; // mirror the server schedule

fn load_npy(path: &str) -> Option<(Vec<f32>, usize, usize)> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        return None;
    }
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + header_len]).ok()?;
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
    let (n, dim) = (dims[0], dims[1]);
    let data_off = 10 + header_len;
    let floats: Vec<f32> = bytes[data_off..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Some((floats, n, dim))
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-9)
}

fn brute_top_k(corpus: &[f32], n: usize, dim: usize, query: &[f32], k: usize) -> Vec<u64> {
    let mut scored: Vec<(f32, u64)> = (0..n)
        .into_par_iter()
        .map(|i| (cosine(query, &corpus[i * dim..(i + 1) * dim]), i as u64))
        .collect();
    scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    scored.iter().take(k).map(|&(_, id)| id).collect()
}

fn recall_p50(
    idx: &DiskVamanaIndex,
    corpus: &[f32],
    n: usize,
    dim: usize,
    queries: &[f32],
) -> (f32, f64) {
    let mut recalls = Vec::with_capacity(N_QUERIES);
    let mut lats = Vec::with_capacity(N_QUERIES);
    for q in 0..N_QUERIES {
        let query = &queries[q * dim..(q + 1) * dim];
        let truth: std::collections::HashSet<u64> =
            brute_top_k(corpus, n, dim, query, K).into_iter().collect();
        let t0 = Instant::now();
        let got = idx.search_with_l(query, K, L_SEARCH).expect("search");
        lats.push(t0.elapsed().as_secs_f64() * 1000.0);
        let hit = got.iter().take(K).filter(|(id, _)| truth.contains(id)).count();
        recalls.push(hit as f32 / K as f32);
    }
    lats.sort_unstable_by(|a, b| a.total_cmp(b));
    let recall = recalls.iter().sum::<f32>() / recalls.len() as f32;
    (recall, lats[lats.len() / 2])
}

fn main() {
    let Some((corpus, n_full, dim)) = load_npy(MXBAI_CORPUS) else {
        eprintln!("incremental_gate: corpus missing, skipping");
        return;
    };
    let (queries, _qn, qdim) = load_npy(MXBAI_QUERY).expect("queries");
    assert_eq!(dim, qdim);
    let n = n_full.min(N_STREAM);

    let tmp = std::env::temp_dir().join(format!("inc-gate-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let mut idx =
        DiskVamanaIndex::create_empty_with_tier(&tmp, dim, L_SEARCH, QuantKind::TurboQuant { bits: 2 })
            .expect("create");

    println!("=======================================================");
    println!("Incremental-insert gate - stream {n} vectors (mxbai {dim}d)");
    println!("Pass: recall@10 >= 0.98 throughout AND p50(high-water) <= 2x p50(post-consolidate)");
    println!("=======================================================");

    // Stream with the production geometric consolidation schedule.
    let mut min_recall = 1.0f32;
    for i in 0..n {
        idx.insert(i as u64, &corpus[i * dim..(i + 1) * dim]).unwrap();
        if idx.delta_len() >= idx.main_len().max(DISK_CONSOLIDATE_MIN) {
            idx.consolidate().unwrap();
            let (r, _) = recall_p50(&idx, &corpus[..n * dim], i + 1, dim, &queries);
            min_recall = min_recall.min(r);
        }
    }
    // Just after a consolidation: delta ~ 0.
    idx.consolidate().unwrap();
    let (recall_lo, p50_lo) = recall_p50(&idx, &corpus[..n * dim], n, dim, &queries);
    min_recall = min_recall.min(recall_lo);

    // Grow the delta to the pre-consolidation high-water mark WITHOUT folding it,
    // by re-inserting the most recent main_len vectors (overwrite = delta entry).
    let hw = idx.main_len().min(n);
    for i in (n - hw)..n {
        idx.insert(i as u64, &corpus[i * dim..(i + 1) * dim]).unwrap();
    }
    let (recall_hi, p50_hi) = recall_p50(&idx, &corpus[..n * dim], n, dim, &queries);
    min_recall = min_recall.min(recall_hi);

    let ratio = p50_hi / p50_lo.max(1e-6);
    let recall_pass = min_recall >= 0.98;
    let latency_pass = ratio <= 2.0;

    println!("\n=== Incremental gate verdict ===");
    println!("  min recall@10 over stream  {min_recall:.4}   [{}]",
             if recall_pass { "PASS" } else { "FAIL" });
    println!("  p50 post-consolidate       {p50_lo:.2} ms");
    println!("  p50 at high-water (delta~main_len)  {p50_hi:.2} ms");
    println!("  latency ratio              {ratio:.1}x        [{}]",
             if latency_pass { "PASS" } else { "FAIL (delta is brute-forced today; LSM runs fix this)" });
    println!("\n  recall -> {}", if recall_pass { "PASS" } else { "FAIL" });
    println!("  bounded latency -> {}", if latency_pass { "PASS" } else { "FAIL (expected red on this branch until the LSM lands)" });

    let _ = std::fs::remove_dir_all(&tmp);
}
