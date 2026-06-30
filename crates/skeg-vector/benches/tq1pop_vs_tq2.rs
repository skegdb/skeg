#![allow(clippy::too_many_arguments, clippy::type_complexity)]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
#![allow(clippy::needless_range_loop)]

//! Head-to-head: symmetric-popcount tq1 vs tq2, on mxbai + MiniLM, up to 100K.
//!
//! Decides the real tq1-popcount-vs-tq2 trade across the three axes that
//! actually pick a tier:
//!   - recall@10 along the rerank dial (1x..16x of K)
//!   - RAM: bytes per vector (tq1 = dim/8, tq2 = dim/4 -> tq1 is HALF)
//!   - proxy latency per candidate
//!
//! Recall and footprint are production-accurate (kernel speed does not change
//! ranking). Latency is directional: popcount uses the real skeg-simd kernel;
//! tq2 uses the prototype scalar `approx_inner` (production wires a NEON i8 ADC
//! that is faster than this but still O(dim) float work, never popcount-class).
//!
//! Corpus size: full 100K by default; set SKEG_BENCH_N to cap (e.g. 10000) for
//! a quick run.

use ahash::AHashSet;
use rayon::prelude::*;
use skeg_simd::{cosine_f32, hamming_binary};
use skeg_vector::{TurboQuant1, TurboQuant2, VamanaConfig, VamanaIndex};

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
const L_SEARCH: usize = 100;
const RERANK_DIAL: [usize; 5] = [1, 2, 4, 8, 16];

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
    let hit = approx
        .iter()
        .take(k)
        .filter(|id| truth_set.contains(id))
        .count();
    hit as f32 / k as f32
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

fn pack_query_signs(q_rot: &[f32]) -> Vec<u8> {
    let mut bits = vec![0u8; q_rot.len().div_ceil(8)];
    for (i, &r) in q_rot.iter().enumerate() {
        if r > 0.0 {
            bits[i / 8] |= 1u8 << (i % 8);
        }
    }
    bits
}

/// recall@10 at each dial width + f32 walk reference. Walks once per query,
/// reranks at every width from the same candidate list.
fn recall_sweep<P>(
    corpus: &[f32],
    queries: &[f32],
    n: usize,
    dim: usize,
    nq: usize,
    index: &VamanaIndex,
    proxy: P,
) -> ([f32; RERANK_DIAL.len()], f32)
where
    P: Fn(usize, &[f32]) -> f32 + Sync,
{
    let medoid = index.medoid();
    let per_query: Vec<([f32; RERANK_DIAL.len()], f32)> = (0..nq)
        .into_par_iter()
        .map(|q_idx| {
            let query = &queries[q_idx * dim..(q_idx + 1) * dim];
            let truth = brute_top_k(corpus, n, dim, query, K);
            let ordered = greedy_walk(
                medoid,
                |id| index.neighbors(id).to_vec(),
                |id| -proxy(id as usize, query),
                L_SEARCH.max(K),
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
            let ref_ids: Vec<u32> = index
                .search(query, K)
                .iter()
                .map(|&(id, _)| id as u32)
                .collect();
            (recalls, recall_at_k(&ref_ids, &truth, K))
        })
        .collect();
    let mut means = [0.0f32; RERANK_DIAL.len()];
    let mut ref_mean = 0.0f32;
    for (recalls, r_ref) in &per_query {
        for (m, r) in means.iter_mut().zip(recalls) {
            *m += r;
        }
        ref_mean += r_ref;
    }
    for m in &mut means {
        *m /= nq as f32;
    }
    (means, ref_mean / nq as f32)
}

/// End-to-end per-query latency: rotate query once, greedy walk scored by the
/// per-candidate `score` closure, then f32 rerank of the top `rerank_width`.
/// Queries served sequentially (single query in flight, server-style). Returns
/// (mean_ms, p50_ms, walk_share_ms, rerank_share_ms). In-RAM rerank (no SSD);
/// `rotate` here is the dense prototype rotation (production FastRotation is
/// O(dim log dim), faster, and shared by both tiers).
fn global_latency_ms<Q, R, S>(
    queries: &[f32],
    nq: usize,
    dim: usize,
    corpus: &[f32],
    index: &VamanaIndex,
    rerank_width: usize,
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
    let mut walk_us = 0.0f64;
    let mut rerank_us = 0.0f64;
    for q_idx in 0..nq {
        let query = &queries[q_idx * dim..(q_idx + 1) * dim];
        let t0 = std::time::Instant::now();
        let q_enc = rotate(query); // once per query
        let tw = std::time::Instant::now();
        let ordered = greedy_walk(
            medoid,
            |id| index.neighbors(id).to_vec(),
            |id| -score(id as usize, &q_enc),
            L_SEARCH.max(K),
        );
        walk_us += tw.elapsed().as_secs_f64() * 1e6;
        let tr = std::time::Instant::now();
        let width = rerank_width.min(ordered.len());
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
        let top: Vec<u32> = rr.iter().take(K).map(|&(_, id)| id).collect();
        rerank_us += tr.elapsed().as_secs_f64() * 1e6;
        black_box(&top);
        totals.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    totals.sort_unstable_by(f64::total_cmp);
    let mean = totals.iter().sum::<f64>() / nq as f64;
    let p50 = totals[nq / 2];
    (
        mean,
        p50,
        walk_us / nq as f64 / 1e3,
        rerank_us / nq as f64 / 1e3,
    )
}

fn kernel_ns<F: Fn(usize) -> f32>(samples: usize, reps: usize, score: F) -> f64 {
    use std::hint::black_box;
    let mut acc = 0.0f32;
    for i in 0..samples {
        acc += score(i);
    }
    black_box(acc);
    let t = std::time::Instant::now();
    let mut acc = 0.0f32;
    for _ in 0..reps {
        for i in 0..samples {
            acc += black_box(score(i));
        }
    }
    black_box(acc);
    t.elapsed().as_secs_f64() * 1e9 / (samples * reps) as f64
}

fn encode_all<E>(
    n: usize,
    dim: usize,
    corpus: &[f32],
    code_bytes: usize,
    enc: E,
) -> (Vec<u8>, Vec<f32>)
where
    E: Fn(&[f32]) -> (Vec<u8>, f32) + Sync,
{
    let mut codes = vec![0u8; n * code_bytes];
    let scales: Vec<f32> = codes
        .par_chunks_exact_mut(code_bytes)
        .enumerate()
        .map(|(i, slot)| {
            let (c, s) = enc(&corpus[i * dim..(i + 1) * dim]);
            slot.copy_from_slice(&c);
            s
        })
        .collect();
    (codes, scales)
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
    assert_eq!(dim, q_dim, "corpus/query dim mismatch");
    let cap = std::env::var("SKEG_BENCH_N")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());
    let n = cap.map_or(n_full, |c| c.min(n_full));
    let corpus = &corpus[..n * dim];
    let nq = N_QUERIES.min(q_n);
    println!("\n=== {label}: corpus {n} x {dim}, queries {nq} ===");

    let ids: Vec<u64> = (0..n as u64).collect();
    let index = VamanaIndex::build(corpus.to_vec(), ids, dim, &VamanaConfig::default());

    let tq1 = TurboQuant1::new(dim, 0xC0DE_BEEF);
    let tq2 = TurboQuant2::new(dim, 0xC0DE_BEEF);
    let cb1 = tq1.code_bytes();
    let cb2 = tq2.code_bytes();
    let (codes1, _scales1) = encode_all(n, dim, corpus, cb1, |v| tq1.encode(v));
    let (codes2, scales2) = encode_all(n, dim, corpus, cb2, |v| tq2.encode(v));

    let (r_pop, ref_recall) = recall_sweep(corpus, &queries, n, dim, nq, &index, |id, q| {
        let q_bits = pack_query_signs(&tq1.rotate_query(q));
        -(hamming_binary(&q_bits, &codes1[id * cb1..(id + 1) * cb1]) as f32)
    });
    let (r_tq2, _) = recall_sweep(corpus, &queries, n, dim, nq, &index, |id, q| {
        let q_rot = tq2.rotate_query(q);
        tq2.approx_inner(&codes2[id * cb2..(id + 1) * cb2], scales2[id], &q_rot)
    });

    // Latency, one fixed query, rotation outside the timed loop.
    let q0 = &queries[0..dim];
    let q_bits = pack_query_signs(&tq1.rotate_query(q0));
    let q_rot2 = tq2.rotate_query(q0);
    let reps = (5_000_000 / n.max(1)).max(1);
    let pop_ns = kernel_ns(n, reps, |i| {
        hamming_binary(&q_bits, &codes1[i * cb1..(i + 1) * cb1]) as f32
    });
    let tq2_ns = kernel_ns(n, reps, |i| {
        tq2.approx_inner(&codes2[i * cb2..(i + 1) * cb2], scales2[i], &q_rot2)
    });

    // End-to-end per-query latency at a fixed rerank width (10x K = 100),
    // queries served sequentially. Rotation once per query.
    let rr_width = 10 * K;
    let (pop_mean, pop_p50, pop_walk, pop_rr) = global_latency_ms(
        &queries,
        nq,
        dim,
        corpus,
        &index,
        rr_width,
        |q| pack_query_signs(&tq1.rotate_query(q)),
        |id, qb: &Vec<u8>| -(hamming_binary(qb, &codes1[id * cb1..(id + 1) * cb1]) as f32),
    );
    let (t2_mean, t2_p50, t2_walk, t2_rr) = global_latency_ms(
        &queries,
        nq,
        dim,
        corpus,
        &index,
        rr_width,
        |q| tq2.rotate_query(q),
        |id, qr: &Vec<f32>| tq2.approx_inner(&codes2[id * cb2..(id + 1) * cb2], scales2[id], qr),
    );

    println!("  f32 walk reference recall@10: {ref_recall:.4}");
    println!("  rerank dial (xK):       {}", header());
    println!("  popcount tq1 ({cb1:>3}B):  {}", row(&r_pop));
    println!("  tq2          ({cb2:>3}B):  {}", row(&r_tq2));
    println!(
        "  RAM @100K:  tq1 {:>5.1} MB   tq2 {:>5.1} MB   (tq1 = {:.0}% of tq2)",
        cb1 as f64 * 100_000.0 / 1e6,
        cb2 as f64 * 100_000.0 / 1e6,
        cb1 as f64 / cb2 as f64 * 100.0
    );
    println!("  proxy kernel only:  popcount {pop_ns:.1} ns   tq2-proto {tq2_ns:.1} ns/candidate");
    println!("  GLOBAL latency/query (rotate+walk+rerank@{rr_width}, in-RAM, sequential):");
    println!(
        "    popcount tq1:  p50 {pop_p50:.3} ms  mean {pop_mean:.3} ms   (walk {pop_walk:.3} + rerank {pop_rr:.3})"
    );
    println!(
        "    tq2:           p50 {t2_p50:.3} ms  mean {t2_mean:.3} ms   (walk {t2_walk:.3} + rerank {t2_rr:.3})"
    );
    println!(
        "    note: total = rotation + walk + rerank; rotation share (total - walk - rerank) is the dense prototype O(dim^2) rotation, which production FastRotation O(dim log dim) cuts for both tiers equally"
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
    println!("popcount-tq1  vs  tq2   (mxbai + MiniLM, up to 100K)");
    println!("recall@10 along rerank dial + RAM/vec + proxy latency");
    println!("=========================================================");
    run("mxbai 1024d wiki", MXBAI_CORPUS, MXBAI_QUERY);
    run("MiniLM 384d wiki", MINILM_CORPUS, MINILM_QUERY);
}
