#![allow(clippy::cast_precision_loss)]
//! tq1 rescue gate: is there a CHEAP per-query signal, available from the default
//! tq1 result, that predicts when tq1 under-recalls (so a planner could promote
//! just those queries to a wider walk)? For each query we know the true per-query
//! recall@100 (oracle) and several cheap signals from the default top-100 f32
//! scores; we bucket queries by recall and show whether the signals separate the
//! "bad" ones. If they do NOT, tq1 can't be a safe invisible default - ship tq2.
//!   SKEG_BENCH_N=100000  SKEG_NQ=1000  SKEG_CORPUS=<npy>  SKEG_QUERY=<npy>

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

fn main() {
    let n_cap = std::env::var("SKEG_BENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);
    let nq = std::env::var("SKEG_NQ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);
    let corpus_path = std::env::var("SKEG_CORPUS").unwrap_or_else(|_| {
        format!("{ROOT}/skeg/bench-compare/embeddings_cache/corpus_mxbai-wiki.npy")
    });
    let query_path = std::env::var("SKEG_QUERY").unwrap_or_else(|_| {
        format!("{ROOT}/skeg/bench-compare/embeddings_cache/queries_mxbai-wiki_200.npy")
    });
    let (corpus, dim) = load(&corpus_path, n_cap);
    let (queries, _) = load(&query_path, nq);
    let n = corpus.len();
    let truth: Vec<AHashSet<u64>> = queries
        .par_iter()
        .map(|q| {
            let mut t: Vec<(f32, u64)> = corpus
                .iter()
                .enumerate()
                .map(|(i, v)| (cosine_f32(q, v), i as u64))
                .collect();
            t.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            (t.iter().take(100).map(|&(_, id)| id).collect(), t[99].0)
        })
        .map(|(s, _true_100th): (AHashSet<u64>, f32)| s)
        .collect();
    // also keep the TRUE 100th-neighbour cosine per query (the ideal s_min).
    let true_100th: Vec<f32> = queries
        .par_iter()
        .map(|q| {
            let mut s: Vec<f32> = corpus.iter().map(|v| cosine_f32(q, v)).collect();
            s.sort_unstable_by(|a, b| b.total_cmp(a));
            s[99]
        })
        .collect();

    let tier = QuantKind::TurboQuant { bits: 1 };
    let tmp = std::env::temp_dir().join("skeg_risk_oracle");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let mut idx = DiskVamanaIndex::create_empty_with_tier(&tmp, dim, 300, tier).unwrap();
    for (id, v) in corpus.iter().enumerate() {
        idx.insert(id as u64, v).unwrap();
    }
    idx.consolidate().unwrap();
    println!(
        "tq1 risk oracle: {n} x {dim}, {} queries, k=100 default(l=300,rr=800)",
        queries.len()
    );

    // Per query: recall@100 + cheap signals from the default top-100 f32 scores.
    // s_min = 100th returned score; deficit = true_100th - s_min (needs no truth
    // at serve time IF approximated, but here we use the oracle to test the ceiling).
    struct Row {
        recall: f64,
        s_min: f32,
        s_mean: f32,
        s10: f32,
        deficit: f32, // true_100th - s_min (oracle upper-bound signal)
    }
    let rows: Vec<Row> = queries
        .iter()
        .zip(&truth)
        .zip(&true_100th)
        .map(|((q, tr), &t100)| {
            let got = idx.search_with_params(q, 100, 300, 800).unwrap();
            let hits = got.iter().filter(|(id, _)| tr.contains(id)).count();
            let scores: Vec<f32> = got.iter().map(|&(_, s)| s).collect();
            let s_min = scores.last().copied().unwrap_or(0.0);
            let s10 = scores.get(9).copied().unwrap_or(0.0);
            let s_mean = scores.iter().sum::<f32>() / scores.len().max(1) as f32;
            Row {
                recall: hits as f64 / 100.0,
                s_min,
                s_mean,
                s10,
                deficit: t100 - s_min,
            }
        })
        .collect();

    // Split into bad (recall < 0.90) vs good (>= 0.90) and compare signal means.
    let (bad, good): (Vec<&Row>, Vec<&Row>) = rows.iter().partition(|r| r.recall < 0.90);
    let mean = |xs: &[&Row], f: fn(&Row) -> f32| {
        xs.iter().map(|r| f(r)).sum::<f32>() / xs.len().max(1) as f32
    };
    println!(
        "  bad (recall<0.90): {} queries   good: {}",
        bad.len(),
        good.len()
    );
    println!("  signal         bad-mean   good-mean   (separation if different)");
    println!(
        "  s_min          {:.4}     {:.4}",
        mean(&bad, |r| r.s_min),
        mean(&good, |r| r.s_min)
    );
    println!(
        "  s10            {:.4}     {:.4}",
        mean(&bad, |r| r.s10),
        mean(&good, |r| r.s10)
    );
    println!(
        "  s_mean         {:.4}     {:.4}",
        mean(&bad, |r| r.s_mean),
        mean(&good, |r| r.s_mean)
    );
    println!(
        "  deficit*       {:.4}     {:.4}   (*oracle: true_100th - s_min)",
        mean(&bad, |r| r.deficit),
        mean(&good, |r| r.deficit)
    );

    // Practical test: if we promote the worst X% by s_min, how much recall do we
    // recover and how many queries do we touch? (s_min needs no truth at serve.)
    let mut by_smin: Vec<&Row> = rows.iter().collect();
    by_smin.sort_unstable_by(|a, b| a.s_min.total_cmp(&b.s_min)); // lowest s_min first
    let overall = rows.iter().map(|r| r.recall).sum::<f64>() / rows.len() as f64;
    println!("  overall recall@100 (default) {overall:.4}");
    for pct in [10usize, 20, 30] {
        let cut = rows.len() * pct / 100;
        // Queries flagged = the `cut` lowest s_min. How many bad ones did we catch?
        let flagged_bad = by_smin[..cut].iter().filter(|r| r.recall < 0.90).count();
        let total_bad = bad.len().max(1);
        println!(
            "  promote worst {pct}% by s_min ({cut} q): catches {flagged_bad}/{} bad ({:.0}%)",
            bad.len(),
            100.0 * flagged_bad as f64 / total_bad as f64
        );
    }
    let _ = std::fs::remove_dir_all(&tmp);
}
