#![allow(clippy::cast_precision_loss)]
//! Final tq1-salvage test: at 500k, k=10, can more rerank (and a slightly wider
//! walk) recover tq1's recall@10 to ~tq2-default WITHOUT paying the full wide
//! walk? tq1 sweep over (l_search, rerank); tq2 default is the reference bar.
//! Real recall@10 + k=10-only ms/q. Decision: tq1 viable iff it reaches
//! ~tq2-default recall@10 at clearly lower latency.
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

fn build(corpus: &[Vec<f32>], dim: usize, bits: u8, tag: &str) -> DiskVamanaIndex {
    let tmp = std::env::temp_dir().join(format!("skeg_salvage_{tag}"));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let mut idx =
        DiskVamanaIndex::create_empty_with_tier(&tmp, dim, 300, QuantKind::TurboQuant { bits })
            .unwrap();
    for (id, v) in corpus.iter().enumerate() {
        idx.insert(id as u64, v).unwrap();
    }
    idx.consolidate().unwrap();
    idx
}

/// (recall@10, recall@100, k=10 ms/q) at a config. recall@100 uses the same
/// l_search but a non-starved rerank (>=800) so it reflects the walk, not the
/// rerank; the ms is the k=10 cost (the salvage target).
fn eval(
    idx: &DiskVamanaIndex,
    queries: &[Vec<f32>],
    t10: &[AHashSet<u64>],
    t100: &[AHashSet<u64>],
    ls: usize,
    rr: usize,
) -> (f64, f64, f64) {
    let mut h10 = 0usize;
    let t = std::time::Instant::now();
    for (q, tr) in queries.iter().zip(t10) {
        h10 += idx
            .search_with_params(q, 10, ls, rr)
            .unwrap()
            .iter()
            .filter(|(id, _)| tr.contains(id))
            .count();
    }
    let ms = t.elapsed().as_secs_f64() * 1e3 / queries.len() as f64;
    let mut h100 = 0usize;
    for (q, tr) in queries.iter().zip(t100) {
        h100 += idx
            .search_with_params(q, 100, ls, rr.max(800))
            .unwrap()
            .iter()
            .filter(|(id, _)| tr.contains(id))
            .count();
    }
    (
        h10 as f64 / (queries.len() * 10) as f64,
        h100 as f64 / (queries.len() * 100) as f64,
        ms,
    )
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
    let cpath = std::env::var("SKEG_CORPUS").unwrap_or_else(|_| {
        format!("{ROOT}/skeg/bench-compare/embeddings_cache/corpus_mxbai-wiki.npy")
    });
    let qpath = std::env::var("SKEG_QUERY").unwrap_or_else(|_| {
        format!("{ROOT}/skeg/bench-compare/embeddings_cache/queries_mxbai-wiki_200.npy")
    });
    let (corpus, dim) = load(&cpath, n_cap);
    let (queries, _) = load(&qpath, nq);
    let n = corpus.len();
    let t10: Vec<AHashSet<u64>> = queries
        .par_iter()
        .map(|q| {
            let mut t: Vec<(f32, u64)> = corpus
                .iter()
                .enumerate()
                .map(|(i, v)| (cosine_f32(q, v), i as u64))
                .collect();
            t.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            t.iter().take(10).map(|&(_, id)| id).collect()
        })
        .collect();
    let t100: Vec<AHashSet<u64>> = queries
        .par_iter()
        .map(|q| {
            let mut t: Vec<(f32, u64)> = corpus
                .iter()
                .enumerate()
                .map(|(i, v)| (cosine_f32(q, v), i as u64))
                .collect();
            t.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            t.iter().take(100).map(|&(_, id)| id).collect()
        })
        .collect();
    println!(
        "tq1 salvage (k=10, but reporting @100 too): {n} x {dim}, {} queries",
        queries.len()
    );

    // Reference bar: tq2 default.
    let tq2 = build(&corpus, dim, 2, "tq2");
    let (r2, r2c, m2) = eval(&tq2, &queries, &t10, &t100, 300, 80);
    println!(
        "  REF tq2 default (l=300,rr=80)   recall@10 {r2:.4}  recall@100 {r2c:.4}  {m2:.2} ms/q"
    );
    drop(tq2);

    let tq1 = build(&corpus, dim, 1, "tq1");
    // tq1 is "salvaged" for k=10 only if it reaches tq2's recall@10 at clearly
    // lower latency. recall@100 is shown to confirm it stays broken (k=100 is tq2).
    let mark = |r: f64, m: f64| {
        if r >= r2 && m < m2 {
            "  <= tq2 recall@10, lower latency"
        } else {
            ""
        }
    };
    println!("  -- tq1: rerank sweep at l_search=300 --");
    for &rr in &[80usize, 160, 320, 640, 1280] {
        let (r, rc, m) = eval(&tq1, &queries, &t10, &t100, 300, rr);
        println!(
            "     l=300 rr={rr:<5} recall@10 {r:.4}  recall@100 {rc:.4}  {m:.2} ms/q{}",
            mark(r, m)
        );
    }
    println!("  -- tq1: walk mini-sweep --");
    for &(ls, rr) in &[(400usize, 160usize), (400, 320), (600, 160), (600, 320)] {
        let (r, rc, m) = eval(&tq1, &queries, &t10, &t100, ls, rr);
        println!(
            "     l={ls} rr={rr:<5} recall@10 {r:.4}  recall@100 {rc:.4}  {m:.2} ms/q{}",
            mark(r, m)
        );
    }
}
