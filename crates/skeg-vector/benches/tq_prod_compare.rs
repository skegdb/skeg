#![allow(clippy::too_many_arguments, clippy::type_complexity)]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
#![allow(clippy::needless_range_loop)]

//! Production-kernel comparison: popcount-tq1 vs tq1-asym vs tq2, ALL on the
//! kernels that actually ship (no prototype structs).
//!
//!   - tq2        -> QuantizedVectors::proxy -> tq2_adc_i8 (NEON) + FastRotation
//!   - tq1-asym   -> QuantizedVectors::proxy -> tq1_adc_swar  + FastRotation
//!   - popcount   -> FastRotation sign-bit codes + hamming_binary
//!
//! All three use FastRotation (O(dim log dim)), so the rotation share is real,
//! not the dense O(dim^2) prototype. Reports recall@10 along the rerank dial,
//! RAM/vector, and end-to-end per-query latency (rotate + walk + rerank),
//! queries served sequentially, in-RAM rerank. Set SKEG_BENCH_N to cap corpus.

use ahash::AHashSet;
use rayon::prelude::*;
use skeg_simd::{cosine_f32, hamming_binary};
use skeg_vector::{
    FastRotation, QuantKind, QuantizedVectors, QueryCode, VamanaConfig, VamanaIndex,
};

const MXBAI_CORPUS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../bench-compare/embeddings_cache/corpus_mxbai-wiki.npy"
);
const MXBAI_QUERY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../bench-compare/embeddings_cache/queries_mxbai-wiki_200.npy"
);
const MINILM_CORPUS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../bench-compare/embeddings_cache/corpus_minilm-wiki.npy"
);
const MINILM_QUERY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../bench-compare/embeddings_cache/queries_minilm-wiki_200.npy"
);

const K: usize = 10;
const N_QUERIES: usize = 100;
/// Walk candidate-list size. Override via SKEG_LSEARCH (production default is
/// 300; this bench defaults to 100 unless set). Wider L = better recall, more
/// walk cost - the primary recall lever.
fn l_search() -> usize {
    std::env::var("SKEG_LSEARCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100)
}
const RERANK_DIAL: [usize; 5] = [1, 2, 4, 8, 16];
const POP_SEED: u64 = 0xC0DE_BEEF;

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
    let data: Vec<f32> = bytes[10 + header_len..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Some((data, dims[0], dims[1]))
}

fn brute_top_k(corpus: &[f32], n: usize, dim: usize, query: &[f32], k: usize) -> Vec<u32> {
    let mut scored: Vec<(f32, u32)> = (0..n)
        .into_par_iter()
        .map(|i| (cosine_f32(query, &corpus[i * dim..(i + 1) * dim]), i as u32))
        .collect();
    scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    scored.iter().take(k).map(|&(_, id)| id).collect()
}

fn recall_at_k(approx: &[u32], truth: &[u32], k: usize) -> f32 {
    let truth_set: AHashSet<u32> = truth.iter().take(k).copied().collect();
    approx
        .iter()
        .take(k)
        .filter(|id| truth_set.contains(id))
        .count() as f32
        / k as f32
}

fn greedy_walk(
    medoid: u32,
    neighbors: impl Fn(u32) -> Vec<u32>,
    proxy: impl Fn(u32) -> f32,
    list_size: usize,
) -> Vec<u32> {
    let mut seen: AHashSet<u32> = AHashSet::new();
    let mut visited: AHashSet<u32> = AHashSet::new();
    let mut list: Vec<(f32, u32)> = Vec::new();
    list.push((proxy(medoid), medoid));
    seen.insert(medoid);
    loop {
        let next = list.iter().copied().find(|&(_, id)| !visited.contains(&id));
        let Some((_, cur)) = next else { break };
        visited.insert(cur);
        for nbr in neighbors(cur) {
            if !seen.insert(nbr) {
                continue;
            }
            list.push((proxy(nbr), nbr));
        }
        list.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        list.truncate(list_size);
    }
    list.into_iter().map(|(_, id)| id).collect()
}

fn unit(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let inv = if norm > 1e-10 { 1.0 / norm } else { 0.0 };
    v.iter().map(|&x| x * inv).collect()
}

/// Rotated unit vector -> LSB-first sign bits (same packing as the production
/// tq1 code: bit i set when rotated coord i > 0).
fn rot_signs(rot: &FastRotation, v: &[f32]) -> Vec<u8> {
    let r = rot.apply_alloc(&unit(v));
    let mut bits = vec![0u8; r.len().div_ceil(8)];
    for (i, &x) in r.iter().enumerate() {
        if x > 0.0 {
            bits[i / 8] |= 1u8 << (i % 8);
        }
    }
    bits
}

/// recall@10 at each dial width, walking once per query with `rotate`+`score`.
fn recall_sweep<Q, R, S>(
    corpus: &[f32],
    queries: &[f32],
    n: usize,
    dim: usize,
    nq: usize,
    index: &VamanaIndex,
    rotate: R,
    score: S,
) -> [f32; RERANK_DIAL.len()]
where
    Q: Send,
    R: Fn(&[f32]) -> Q + Sync,
    S: Fn(usize, &Q) -> f32 + Sync,
{
    let medoid = index.medoid();
    let per_query: Vec<[f32; RERANK_DIAL.len()]> = (0..nq)
        .into_par_iter()
        .map(|q_idx| {
            let query = &queries[q_idx * dim..(q_idx + 1) * dim];
            let truth = brute_top_k(corpus, n, dim, query, K);
            let q_enc = rotate(query);
            let ordered = greedy_walk(
                medoid,
                |id| index.neighbors(id).to_vec(),
                |id| -score(id as usize, &q_enc),
                l_search().max(K),
            );
            let mut recalls = [0.0f32; RERANK_DIAL.len()];
            for (slot, &mult) in RERANK_DIAL.iter().enumerate() {
                let width = (mult * K).min(ordered.len());
                let mut rr: Vec<(f32, u32)> = ordered[..width]
                    .iter()
                    .map(|&id| {
                        (
                            cosine_f32(query, &corpus[id as usize * dim..(id as usize + 1) * dim]),
                            id,
                        )
                    })
                    .collect();
                rr.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                let approx: Vec<u32> = rr.iter().take(K).map(|&(_, id)| id).collect();
                recalls[slot] = recall_at_k(&approx, &truth, K);
            }
            recalls
        })
        .collect();
    let mut means = [0.0f32; RERANK_DIAL.len()];
    for recalls in &per_query {
        for (m, r) in means.iter_mut().zip(recalls) {
            *m += r;
        }
    }
    for m in &mut means {
        *m /= nq as f32;
    }
    means
}

/// End-to-end per-query latency: rotate once, walk, rerank top `width`.
/// Sequential (single query in flight). Returns (p50_ms, mean_ms, walk_ms,
/// rerank_ms); rotation share = total - walk - rerank.
fn latency<Q, R, S>(
    queries: &[f32],
    nq: usize,
    dim: usize,
    corpus: &[f32],
    index: &VamanaIndex,
    width: usize,
    rotate: R,
    score: S,
) -> (f64, f64, f64, f64)
where
    R: Fn(&[f32]) -> Q,
    S: Fn(usize, &Q) -> f32,
{
    use std::hint::black_box;
    let medoid = index.medoid();
    let mut totals = Vec::with_capacity(nq);
    let mut walk_us = 0.0;
    let mut rr_us = 0.0;
    for q_idx in 0..nq {
        let query = &queries[q_idx * dim..(q_idx + 1) * dim];
        let t0 = std::time::Instant::now();
        let q_enc = rotate(query);
        let tw = std::time::Instant::now();
        let ordered = greedy_walk(
            medoid,
            |id| index.neighbors(id).to_vec(),
            |id| -score(id as usize, &q_enc),
            l_search().max(K),
        );
        walk_us += tw.elapsed().as_secs_f64() * 1e6;
        let tr = std::time::Instant::now();
        let w = width.min(ordered.len());
        let mut rr: Vec<(f32, u32)> = ordered[..w]
            .iter()
            .map(|&id| {
                (
                    cosine_f32(query, &corpus[id as usize * dim..(id as usize + 1) * dim]),
                    id,
                )
            })
            .collect();
        rr.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        let top: Vec<u32> = rr.iter().take(K).map(|&(_, id)| id).collect();
        rr_us += tr.elapsed().as_secs_f64() * 1e6;
        black_box(&top);
        totals.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    totals.sort_unstable_by(f64::total_cmp);
    let mean = totals.iter().sum::<f64>() / nq as f64;
    (
        totals[nq / 2],
        mean,
        walk_us / nq as f64 / 1e3,
        rr_us / nq as f64 / 1e3,
    )
}

fn run(label: &str, corpus_npy: &str, query_npy: &str) {
    let Some((corpus, n_full, dim)) = load_npy(corpus_npy) else {
        println!("\n=== {label}: dataset missing ===");
        return;
    };
    let Some((queries, q_n, q_dim)) = load_npy(query_npy) else {
        println!("\n=== {label}: queries missing ===");
        return;
    };
    assert_eq!(dim, q_dim);
    let cap = std::env::var("SKEG_BENCH_N")
        .ok()
        .and_then(|s| s.parse().ok());
    let n: usize = cap.map_or(n_full, |c: usize| c.min(n_full));
    // Production proxy assumes unit-norm vectors (the ip clamp at 4.0 saturates
    // otherwise). skeg ingest feeds normalized embeddings; the raw npy is not
    // always unit (mxbai norms > 1), so normalize here to match production.
    let mut corpus = corpus[..n * dim].to_vec();
    for row in corpus.chunks_exact_mut(dim) {
        let nrm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if nrm > 1e-10 {
            for x in row.iter_mut() {
                *x /= nrm;
            }
        }
    }
    let corpus = corpus.as_slice();
    let nq = N_QUERIES.min(q_n);
    println!("\n=== {label}: corpus {n} x {dim}, queries {nq} ===");

    let ids: Vec<u64> = (0..n as u64).collect();
    let index = VamanaIndex::build(corpus.to_vec(), ids, dim, &VamanaConfig::default());

    // Production quantizers (ship kernels).
    let qv2 = QuantizedVectors::build(corpus, dim, QuantKind::TurboQuant { bits: 2 });
    let qv1 = QuantizedVectors::build(corpus, dim, QuantKind::TurboQuant { bits: 1 });
    // Popcount path: FastRotation sign-bit codes.
    let rot = FastRotation::new(dim, POP_SEED);
    let cb1 = dim / 8;
    let mut codes_pop = vec![0u8; n * cb1];
    codes_pop
        .par_chunks_exact_mut(cb1)
        .enumerate()
        .for_each(|(i, slot)| {
            slot.copy_from_slice(&rot_signs(&rot, &corpus[i * dim..(i + 1) * dim]))
        });

    let r_pop = recall_sweep(
        corpus,
        &queries,
        n,
        dim,
        nq,
        &index,
        |q| rot_signs(&rot, q),
        |id, qb: &Vec<u8>| -(hamming_binary(qb, &codes_pop[id * cb1..(id + 1) * cb1]) as f32),
    );
    let r_t1 = recall_sweep(
        corpus,
        &queries,
        n,
        dim,
        nq,
        &index,
        |q| qv1.quantize_query(q),
        |id, c: &QueryCode| qv1.proxy(id, c) as f32,
    );
    let r_t2 = recall_sweep(
        corpus,
        &queries,
        n,
        dim,
        nq,
        &index,
        |q| qv2.quantize_query(q),
        |id, c: &QueryCode| qv2.proxy(id, c) as f32,
    );

    let w = 10 * K;
    let l_pop = latency(
        &queries,
        nq,
        dim,
        corpus,
        &index,
        w,
        |q| rot_signs(&rot, q),
        |id, qb: &Vec<u8>| -(hamming_binary(qb, &codes_pop[id * cb1..(id + 1) * cb1]) as f32),
    );
    let l_t1 = latency(
        &queries,
        nq,
        dim,
        corpus,
        &index,
        w,
        |q| qv1.quantize_query(q),
        |id, c: &QueryCode| qv1.proxy(id, c) as f32,
    );
    let l_t2 = latency(
        &queries,
        nq,
        dim,
        corpus,
        &index,
        w,
        |q| qv2.quantize_query(q),
        |id, c: &QueryCode| qv2.proxy(id, c) as f32,
    );

    println!("  rerank dial (xK):         {}", header());
    println!("  popcount tq1 ({:>3}B):     {}", cb1, row(&r_pop));
    println!("  tq1-asym prod ({:>3}B):    {}", dim / 8, row(&r_t1));
    println!("  tq2 prod      ({:>3}B):    {}", dim / 4, row(&r_t2));
    println!(
        "  RAM @100K:  tq1 {:.1} MB   tq2 {:.1} MB",
        (dim / 8) as f64 * 0.1,
        (dim / 4) as f64 * 0.1
    );
    println!(
        "  END-TO-END latency/query (rotate+walk+rerank@{w}, FastRotation, in-RAM, sequential):"
    );
    lat_row("popcount tq1 ", l_pop);
    lat_row("tq1-asym prod", l_t1);
    lat_row("tq2 prod     ", l_t2);
}

fn lat_row(name: &str, l: (f64, f64, f64, f64)) {
    let (p50, mean, walk, rr) = l;
    let rot = (mean - walk - rr).max(0.0);
    println!(
        "    {name}:  p50 {p50:.3} ms  mean {mean:.3} ms   (rot {rot:.3} + walk {walk:.3} + rerank {rr:.3})"
    );
}

fn header() -> String {
    RERANK_DIAL
        .iter()
        .map(|m| format!("{:>8}", format!("{m}x")))
        .collect()
}

fn row(r: &[f32]) -> String {
    r.iter().map(|v| format!("{v:>8.4}")).collect()
}

fn main() {
    println!("=========================================================");
    println!("PRODUCTION-kernel compare: popcount-tq1 / tq1-asym / tq2");
    println!("tq2->tq2_adc_i8 NEON  tq1->tq1_adc_swar  all FastRotation");
    println!("=========================================================");
    // Override the mxbai corpus (e.g. the 500K/1M chunked files) via
    // SKEG_MXBAI_CORPUS; the mxbai query set works for any mxbai corpus.
    // When set, only the mxbai run executes (MiniLM stays on its 100K file).
    if let Ok(path) = std::env::var("SKEG_MXBAI_CORPUS") {
        run("mxbai 1024d (override)", &path, MXBAI_QUERY);
        return;
    }
    run("mxbai 1024d wiki", MXBAI_CORPUS, MXBAI_QUERY);
    run("MiniLM 384d wiki", MINILM_CORPUS, MINILM_QUERY);
}
