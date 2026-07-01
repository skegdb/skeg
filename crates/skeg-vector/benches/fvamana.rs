#![allow(clippy::cast_precision_loss)]
//! FilteredVamana (increment 1, in-memory, exact f32) - does baking labels into
//! the graph give a fast, navigable SINGLE filtered walk? Builds a labelled
//! Vamana over mxbai 100k (label = id%100), then at 1% / 10% selectivity
//! compares: (a) FilteredVamana single walk at small L, vs (b) plain walk +
//! post-filter (needs a big k to catch matches). Recall@10 + p50, vs the qdrant
//! numbers (10%: 1.0 @ ~1.7ms).

use ahash::AHashSet;
use skeg_simd::cosine_f32;
use skeg_vector::{VamanaConfig, VamanaIndex};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
const CORPUS: &str = "skeg/bench-compare/embeddings_cache/corpus_mxbai-wiki.npy";
const QUERY: &str = "skeg/bench-compare/embeddings_cache/queries_mxbai-wiki_200.npy";
const K: usize = 10;

fn load_npy(path: &str) -> Option<(Vec<f32>, usize, usize)> {
    let bytes = std::fs::read(path).ok()?;
    if &bytes[0..6] != b"\x93NUMPY" {
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
    let data = bytes[10 + hl..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Some((data, dims[0], dims[1]))
}

fn prep(path: &str, cap: usize) -> (Vec<f32>, usize, usize) {
    let (data, rows, dim) = load_npy(path).expect("npy");
    let n = cap.min(rows);
    let mut out = vec![0.0f32; n * dim];
    for i in 0..n {
        let v = &data[i * dim..i * dim + dim];
        let nrm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
        for j in 0..dim {
            out[i * dim + j] = v[j] / nrm;
        }
    }
    (out, n, dim)
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
    let (corpus, n, dim) = prep(&format!("{ROOT}/{CORPUS}"), n_cap);
    let (qflat, nqr, _) = prep(&format!("{ROOT}/{QUERY}"), nq);
    let queries: Vec<&[f32]> = (0..nqr).map(|i| &qflat[i * dim..i * dim + dim]).collect();
    println!("FilteredVamana: mxbai {n} x {dim}, {nqr} queries, k={K}");

    let ids: Vec<u64> = (0..n as u64).collect();
    let labels: Vec<u64> = (0..n as u64).map(|i| i % 100).collect(); // g = id%100
    let cfg = VamanaConfig::default();
    let t = std::time::Instant::now();
    let idx = VamanaIndex::build_labeled(corpus.clone(), ids.clone(), labels, dim, &cfg);
    println!("labeled build: {:.0}s", t.elapsed().as_secs_f64());
    let plain = VamanaIndex::build(corpus.clone(), ids, dim, &cfg);

    let row = |i: usize| &corpus[i * dim..i * dim + dim];

    // UNFILTERED recall: does baking labels into the graph hurt plain search?
    // Compare plain-search recall@10 on the LABELED graph vs the PLAIN graph.
    let utruth: Vec<AHashSet<u64>> = queries
        .iter()
        .map(|q| {
            let mut s: Vec<(f32, u64)> =
                (0..n).map(|i| (cosine_f32(q, row(i)), i as u64)).collect();
            s.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            s.iter().take(K).map(|&(_, id)| id).collect()
        })
        .collect();
    for (name, g) in [("plain graph", &plain), ("labeled graph", &idx)] {
        let mut hits = 0;
        for (q, tr) in queries.iter().zip(&utruth) {
            hits += g
                .search(q, K)
                .iter()
                .filter(|(id, _)| tr.contains(id))
                .count();
        }
        println!(
            "-- UNFILTERED search on {name}: recall@10 {:.4}",
            hits as f64 / (queries.len() * K) as f64
        );
    }

    for &thresh in &[1u64, 10] {
        let sel = thresh as f64 / 100.0;
        let ml: Vec<u64> = (0..thresh).collect();
        // brute truth over matching (id%100 < thresh)
        let truth: Vec<AHashSet<u64>> = queries
            .iter()
            .map(|q| {
                let mut s: Vec<(f32, u64)> = (0..n)
                    .filter(|i| (*i as u64) % 100 < thresh)
                    .map(|i| (cosine_f32(q, row(i)), i as u64))
                    .collect();
                s.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                s.iter().take(K).map(|&(_, id)| id).collect()
            })
            .collect();
        println!("-- selectivity {:.0}% --", sel * 100.0);

        // FilteredVamana single walk, small L sweep.
        for &l in &[50usize, 100, 300] {
            let mut hits = 0;
            let t = std::time::Instant::now();
            for (q, tr) in queries.iter().zip(&truth) {
                let got = idx.search_filtered_labeled(q, K, &ml, l);
                hits += got.iter().filter(|(id, _)| tr.contains(id)).count();
            }
            let ms = t.elapsed().as_secs_f64() * 1e3 / queries.len() as f64;
            println!(
                "   FVamana L{l:<4} recall {:.4}  {ms:.3} ms/q",
                hits as f64 / (queries.len() * K) as f64
            );
        }
        // Baseline: plain walk + post-filter (needs big k to catch matches).
        for &mult in &[4usize, 20] {
            let kk = (K as f64 / sel) as usize * mult / 4;
            let mut hits = 0;
            let t = std::time::Instant::now();
            for (q, tr) in queries.iter().zip(&truth) {
                let got = plain.search(q, kk);
                let filt: Vec<u64> = got
                    .iter()
                    .map(|(id, _)| *id)
                    .filter(|id| id % 100 < thresh)
                    .take(K)
                    .collect();
                hits += filt.iter().filter(|id| tr.contains(id)).count();
            }
            let ms = t.elapsed().as_secs_f64() * 1e3 / queries.len() as f64;
            println!(
                "   plain+filt k{kk:<5} recall {:.4}  {ms:.3} ms/q",
                hits as f64 / (queries.len() * K) as f64
            );
        }
    }
}
