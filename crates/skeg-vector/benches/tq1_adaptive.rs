#![allow(clippy::too_many_arguments, clippy::type_complexity)]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
#![allow(clippy::needless_range_loop)]

//! End-to-end demo of the online tq1 proxy controller on real embeddings.
//!
//! For each dataset: seed the controller with the dim-based prior, then stream
//! queries. On each shadow step run BOTH proxies' walks, rerank, measure each
//! proxy's recall against exact top-k, and feed the pair to the controller.
//! Print the mode trajectory so you can see it converge: popcount on high-dim
//! sets (qwen-2560), asymmetric on low-dim / awkward-distribution sets
//! (mnist-784, glove-104) - the per-index, data-driven decision the static dim
//! rule can only approximate.
//!
//! This shadows EVERY query (demo, so convergence shows within a few hundred
//! queries); production samples ~1/SHADOW_EVERY. Recall here uses exact
//! brute-force truth; production approximates it with a union rerank (no brute).

use ahash::AHashSet;
use rayon::prelude::*;
use skeg_simd::{cosine_f32, hamming_binary, tq1_masked_sum};
use skeg_vector::{
    FastRotation, Tq1ProxyController, Tq1ProxyMode, VamanaConfig, VamanaIndex, tq1_proxy_mode_for,
};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");

const DATASETS: &[(&str, &str, &str, usize)] = &[
    (
        "glove",
        "skeg-bench/data/glove_corpus.npy",
        "skeg-bench/data/glove_queries.npy",
        100,
    ),
    (
        "mnist",
        "skeg-bench/data/mnist_corpus_60k.npy",
        "skeg-bench/data/mnist_queries_200.npy",
        784,
    ),
    (
        "mxbai",
        "skeg/bench-compare/embeddings_cache/corpus_mxbai-wiki.npy",
        "skeg/bench-compare/embeddings_cache/queries_mxbai-wiki_200.npy",
        1024,
    ),
    (
        "qwen3-emb-4b",
        "skeg/bench-compare/embeddings_cache/corpus_qwen3emb4b_100k.npy",
        "skeg/bench-compare/embeddings_cache/queries_qwen3emb4b_1k.npy",
        2560,
    ),
];

const K: usize = 10;
const RERANK_W: usize = 100;
const SEED: u64 = 0xC0DE_BEEF;

fn cap_n() -> usize {
    std::env::var("SKEG_BENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000)
}
fn n_queries() -> usize {
    std::env::var("SKEG_NQ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(300)
}
fn l_search() -> usize {
    std::env::var("SKEG_LSEARCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(300)
}

fn load_npy(path: &str) -> Option<(Vec<f32>, usize, usize)> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
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
    if dims.len() != 2 {
        return None;
    }
    let data: Vec<f32> = bytes[10 + hl..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Some((data, dims[0], dims[1]))
}

fn load_prep(path: &str, n_cap: usize, pad: usize) -> Option<(Vec<f32>, usize)> {
    let (data, rows, dim) = load_npy(path)?;
    let n = n_cap.min(rows);
    let mut out = vec![0.0f32; n * pad];
    for i in 0..n {
        let dst = &mut out[i * pad..i * pad + pad];
        dst[..dim].copy_from_slice(&data[i * dim..i * dim + dim]);
        let norm = dst.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in dst.iter_mut() {
                *x /= norm;
            }
        }
    }
    Some((out, n))
}

fn brute_top_k(corpus: &[f32], n: usize, dim: usize, q: &[f32]) -> Vec<u32> {
    let mut s: Vec<(f32, u32)> = (0..n)
        .into_par_iter()
        .map(|i| (cosine_f32(q, &corpus[i * dim..(i + 1) * dim]), i as u32))
        .collect();
    s.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    s.iter().take(K).map(|&(_, id)| id).collect()
}

fn greedy_walk(
    medoid: u32,
    index: &VamanaIndex,
    proxy: impl Fn(u32) -> f32,
    list_size: usize,
) -> Vec<u32> {
    let mut seen: AHashSet<u32> = AHashSet::new();
    let mut visited: AHashSet<u32> = AHashSet::new();
    let mut list: Vec<(f32, u32)> = vec![(proxy(medoid), medoid)];
    seen.insert(medoid);
    loop {
        let next = list.iter().copied().find(|&(_, id)| !visited.contains(&id));
        let Some((_, cur)) = next else { break };
        visited.insert(cur);
        for &nbr in index.neighbors(cur) {
            if seen.insert(nbr) {
                list.push((proxy(nbr), nbr));
            }
        }
        list.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        list.truncate(list_size);
    }
    list.into_iter().map(|(_, id)| id).collect()
}

/// recall@K of a walk's reranked top-K against exact truth.
fn walk_recall(
    corpus: &[f32],
    dim: usize,
    q: &[f32],
    index: &VamanaIndex,
    medoid: u32,
    truth: &AHashSet<u32>,
    proxy: impl Fn(u32) -> f32,
) -> f32 {
    // `proxy` is "greater = closer"; greedy_walk wants "smaller = closer".
    let ordered = greedy_walk(medoid, index, |id| -proxy(id), l_search().max(K));
    let w = RERANK_W.min(ordered.len());
    let mut rr: Vec<(f32, u32)> = ordered[..w]
        .iter()
        .map(|&id| {
            (
                cosine_f32(q, &corpus[id as usize * dim..(id as usize + 1) * dim]),
                id,
            )
        })
        .collect();
    rr.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    rr.iter()
        .take(K)
        .filter(|(_, id)| truth.contains(id))
        .count() as f32
        / K as f32
}

fn run_dataset(label: &str, corpus_rel: &str, query_rel: &str, native_dim: usize) {
    let pad = native_dim.next_multiple_of(8);
    let Some((corpus, n)) = load_prep(&format!("{ROOT}/{corpus_rel}"), cap_n(), pad) else {
        println!("  {label:<12} dataset missing");
        return;
    };
    let Some((queries, _)) = load_prep(&format!("{ROOT}/{query_rel}"), n_queries(), pad) else {
        println!("  {label:<12} queries missing");
        return;
    };
    let dim = pad;
    let nq = (queries.len() / dim).min(n_queries());

    let ids: Vec<u64> = (0..n as u64).collect();
    let index = VamanaIndex::build(corpus.clone(), ids, dim, &VamanaConfig::default());
    let medoid = index.medoid();

    // Encode tq1 codes (rotated sign bits) + asymmetric per-vector scales.
    let rot = FastRotation::new(dim, SEED);
    let pos_c = 0.797_884_6f32 / (dim as f32).sqrt();
    let cb = dim / 8;
    let mut codes = vec![0u8; n * cb];
    let scales: Vec<f32> = codes
        .par_chunks_exact_mut(cb)
        .enumerate()
        .map(|(i, slot)| {
            let r = rot.apply_alloc(&corpus[i * dim..(i + 1) * dim]);
            let mut inner = 0.0f32;
            for (j, &rv) in r.iter().enumerate() {
                inner += rv * if rv > 0.0 { pos_c } else { -pos_c };
                if rv > 0.0 {
                    slot[j / 8] |= 1u8 << (j % 8);
                }
            }
            1.0 / inner.max(1e-10)
        })
        .collect();

    let popcount =
        |q_bits: &[u8], id: usize| -(hamming_binary(q_bits, &codes[id * cb..(id + 1) * cb]) as f32);
    let asym = |qr: &[f32], qs: f32, id: usize| {
        scales[id] * pos_c * (2.0 * tq1_masked_sum(&codes[id * cb..(id + 1) * cb], qr, dim) - qs)
    };

    // Seed with the dim prior, then learn from real shadow measurements.
    let prior = tq1_proxy_mode_for(dim, 1);
    let mut ctl = Tq1ProxyController::new(prior).with_policy(0.01, 15, 3);
    let checkpoints = [nq / 4, nq / 2, 3 * nq / 4, nq.saturating_sub(1)];
    let mut trace: Vec<Tq1ProxyMode> = Vec::new();

    for qi in 0..nq {
        let q = &queries[qi * dim..(qi + 1) * dim];
        let truth: AHashSet<u32> = brute_top_k(&corpus, n, dim, q).into_iter().collect();
        // Shadow every query (demo). Both proxies, real recall.
        let q_bits = {
            let r = rot.apply_alloc(q);
            let mut b = vec![0u8; cb];
            for (j, &v) in r.iter().enumerate() {
                if v > 0.0 {
                    b[j / 8] |= 1u8 << (j % 8);
                }
            }
            b
        };
        let qr = rot.apply_alloc(q);
        let qs: f32 = qr.iter().sum();
        let r_pop = walk_recall(&corpus, dim, q, &index, medoid, &truth, |id| {
            popcount(&q_bits, id as usize)
        });
        let r_asym = walk_recall(&corpus, dim, q, &index, medoid, &truth, |id| {
            asym(&qr, qs, id as usize)
        });
        ctl.record_shadow(r_pop, r_asym);
        if checkpoints.contains(&qi) {
            trace.push(ctl.mode());
        }
    }

    let (ep, ea) = ctl.estimates();
    let m = |o: Option<f32>| o.map_or("-".into(), |v| format!("{v:.3}"));
    let tr: Vec<&str> = trace.iter().map(|m| mode_str(*m)).collect();
    println!(
        "  {label:<12} dim {dim:<4}  prior {:<4} -> final {:<4}  [25/50/75/100%: {}]  est pop {} asym {}",
        mode_str(prior),
        mode_str(ctl.mode()),
        tr.join(" "),
        m(ep),
        m(ea),
    );
}

fn mode_str(m: Tq1ProxyMode) -> &'static str {
    match m {
        Tq1ProxyMode::Popcount => "POP",
        Tq1ProxyMode::Asymmetric => "ASYM",
    }
}

fn main() {
    println!("=====================================================================");
    println!("tq1 online controller - convergence on real embeddings");
    println!(
        "N={} nq={} L={}  (shadow every query for the demo)",
        cap_n(),
        n_queries(),
        l_search()
    );
    println!("=====================================================================");
    for &(label, c, q, dim) in DATASETS {
        run_dataset(label, c, q, dim);
    }
}
