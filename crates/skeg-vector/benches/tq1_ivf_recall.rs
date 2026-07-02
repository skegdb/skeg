#![allow(clippy::cast_precision_loss, clippy::doc_lazy_continuation)]
//! Does the IVF router (proxy-INDEPENDENT candidate generation via f32 centroids)
//! rescue tq1's recall@100? The graph walk navigates via the weak 1-bit proxy and
//! misses the true top-100 at a narrow beam; IVF gathers candidates from the
//! query-nearest f32 cells instead. We compare, on tq1 500k, recall@100 + ms:
//!   - graph walk (search_with_params) at l=300 and l=600
//!   - IVF search (search_filtered_hybrid with s = all ids) at a few budgets
//! plus tq2 default as the bar. If IVF hits ~tq2 recall@100 cheaply, tq1+IVF is
//! the rescue the wide walk couldn't be.
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

fn build(corpus: &[Vec<f32>], dim: usize, bits: u8) -> DiskVamanaIndex {
    let tmp = std::env::temp_dir().join(format!("skeg_ivfrec_{bits}"));
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
    let all_ids: Vec<u64> = (0..n as u64).collect();
    println!(
        "tq1 IVF-rescue recall@100: {n} x {dim}, {} queries",
        queries.len()
    );

    let walk = |idx: &DiskVamanaIndex, ls: usize, rr: usize| -> (f64, f64) {
        let mut h = 0usize;
        let t = std::time::Instant::now();
        for (q, tr) in queries.iter().zip(&t100) {
            h += idx
                .search_with_params(q, 100, ls, rr)
                .unwrap()
                .iter()
                .filter(|(id, _)| tr.contains(id))
                .count();
        }
        (
            h as f64 / (queries.len() * 100) as f64,
            t.elapsed().as_secs_f64() * 1e3 / queries.len() as f64,
        )
    };
    let ivf = |idx: &DiskVamanaIndex, rr: usize| -> (f64, f64) {
        let mut h = 0usize;
        let t = std::time::Instant::now();
        for (q, tr) in queries.iter().zip(&t100) {
            h += idx
                .search_filtered_hybrid(q, &all_ids, 100, rr)
                .unwrap()
                .iter()
                .filter(|(id, _)| tr.contains(id))
                .count();
        }
        (
            h as f64 / (queries.len() * 100) as f64,
            t.elapsed().as_secs_f64() * 1e3 / queries.len() as f64,
        )
    };

    let tq2 = build(&corpus, dim, 2);
    let (r, m) = walk(&tq2, 300, 800);
    println!("  REF tq2 walk l=300         recall@100 {r:.4}  {m:.2} ms/q");
    drop(tq2);

    let mut tq1 = build(&corpus, dim, 1);
    let (r, m) = walk(&tq1, 300, 800);
    println!("  tq1 walk  l=300            recall@100 {r:.4}  {m:.2} ms/q");
    let (r, m) = walk(&tq1, 600, 800);
    println!("  tq1 walk  l=600            recall@100 {r:.4}  {m:.2} ms/q");
    let tb = std::time::Instant::now();
    tq1.build_ivf(0, 8).unwrap();
    println!(
        "  (build_ivf: {:.0}s, {} cells)",
        tb.elapsed().as_secs_f64(),
        0
    );
    for &rr in &[800usize, 2000, 4000] {
        let (r, m) = ivf(&tq1, rr);
        println!("  tq1 IVF   rerank={rr:<5}     recall@100 {r:.4}  {m:.2} ms/q");
    }
}
