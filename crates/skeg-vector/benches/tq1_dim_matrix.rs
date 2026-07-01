#![allow(clippy::too_many_arguments, clippy::type_complexity)]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
#![allow(clippy::needless_range_loop)]

//! tq1 proxy matrix: popcount vs asymmetric across every embedding set we have,
//! at a fixed corpus size, to see how the popcount/asymmetric recall gap moves
//! with dimension and where to set TQ1_POPCOUNT_MIN_DIM.
//!
//! Both proxies read the SAME stored codes (rotated sign bits). The asymmetric
//! proxy replicates production `tq1_adc_swar` exactly (masked-sum x per-vector
//! scale, centroid c = 0.7978846/sqrt(dim)); the popcount proxy is
//! `hamming_binary`. So this is an apples-to-apples proxy comparison independent
//! of the auto-selector. N is capped (SKEG_BENCH_N, default 20000) so dimension
//! is the only variable across rows. dim is zero-padded up to a multiple of 8
//! (tq1 packs 8 codes/byte); cosine is unaffected by zero padding.

use ahash::AHashSet;
use rayon::prelude::*;
use skeg_simd::{cosine_f32, hamming_binary, tq1_masked_sum};
use skeg_vector::{FastRotation, VamanaConfig, VamanaIndex};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");

/// (label, corpus path rel to workspace root, query path, native dim).
const DATASETS: &[(&str, &str, &str, usize)] = &[
    (
        "glove",
        "skeg-bench/data/glove_corpus.npy",
        "skeg-bench/data/glove_queries.npy",
        100,
    ),
    (
        "minilm",
        "skeg/bench-compare/embeddings_cache/corpus_minilm-wiki.npy",
        "skeg/bench-compare/embeddings_cache/queries_minilm-wiki_200.npy",
        384,
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
const N_QUERIES: usize = 100;
const RERANK_DIAL: [usize; 5] = [1, 2, 4, 8, 16];
const SEED: u64 = 0xC0DE_BEEF;

fn cap_n() -> usize {
    std::env::var("SKEG_BENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000)
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

/// Load, cap to `n` rows, zero-pad each row from `dim` to `pad` (>= dim, a
/// multiple of 8), and unit-normalise. Returns the padded/normalised flat buffer.
fn load_prep(path: &str, n_cap: usize, pad: usize) -> Option<(Vec<f32>, usize)> {
    let (data, rows, dim) = load_npy(path)?;
    let n = n_cap.min(rows);
    let mut out = vec![0.0f32; n * pad];
    for i in 0..n {
        let src = &data[i * dim..i * dim + dim];
        let dst = &mut out[i * pad..i * pad + pad];
        dst[..dim].copy_from_slice(src);
        let norm = dst.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in dst.iter_mut() {
                *x /= norm;
            }
        }
    }
    Some((out, n))
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
    let t: AHashSet<u32> = truth.iter().take(k).copied().collect();
    approx.iter().take(k).filter(|id| t.contains(id)).count() as f32 / k as f32
}

fn greedy_walk(
    medoid: u32,
    neighbors: impl Fn(u32) -> Vec<u32>,
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
        for nbr in neighbors(cur) {
            if seen.insert(nbr) {
                list.push((proxy(nbr), nbr));
            }
        }
        list.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        list.truncate(list_size);
    }
    list.into_iter().map(|(_, id)| id).collect()
}

fn rot_signs(rot: &FastRotation, unit: &[f32], dim: usize) -> Vec<u8> {
    let r = rot.apply_alloc(unit);
    let mut bits = vec![0u8; dim / 8];
    for (i, &x) in r.iter().enumerate() {
        if x > 0.0 {
            bits[i / 8] |= 1u8 << (i % 8);
        }
    }
    bits
}

/// recall@10 at each dial width for one proxy. `prepare` maps the raw query to
/// the per-query state Q; `score(row, &Q)` is the proxy (greater = closer).
fn sweep<Q, P, S>(
    corpus: &[f32],
    queries: &[f32],
    n: usize,
    dim: usize,
    nq: usize,
    index: &VamanaIndex,
    prepare: P,
    score: S,
) -> [f32; RERANK_DIAL.len()]
where
    Q: Send,
    P: Fn(&[f32]) -> Q + Sync,
    S: Fn(usize, &Q) -> f32 + Sync,
{
    let medoid = index.medoid();
    let per: Vec<[f32; RERANK_DIAL.len()]> = (0..nq)
        .into_par_iter()
        .map(|qi| {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let truth = brute_top_k(corpus, n, dim, q, K);
            let st = prepare(q);
            let ordered = greedy_walk(
                medoid,
                |id| index.neighbors(id).to_vec(),
                |id| -score(id as usize, &st),
                l_search().max(K),
            );
            let mut r = [0.0f32; RERANK_DIAL.len()];
            for (slot, &m) in RERANK_DIAL.iter().enumerate() {
                let w = (m * K).min(ordered.len());
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
                let approx: Vec<u32> = rr.iter().take(K).map(|&(_, id)| id).collect();
                r[slot] = recall_at_k(&approx, &truth, K);
            }
            r
        })
        .collect();
    let mut mean = [0.0f32; RERANK_DIAL.len()];
    for r in &per {
        for (a, b) in mean.iter_mut().zip(r) {
            *a += b;
        }
    }
    for a in &mut mean {
        *a /= nq as f32;
    }
    mean
}

/// Hybrid: popcount walks (cheap navigation), then the L_search survivors are
/// re-scored by the asymmetric proxy (in-RAM, no disk) and re-sorted before the
/// rerank window. Recovers the ranking-loss part of popcount's gap at ~L extra
/// asym evals instead of the ~6400 the full asym walk pays.
fn sweep_hybrid(
    corpus: &[f32],
    queries: &[f32],
    n: usize,
    dim: usize,
    nq: usize,
    index: &VamanaIndex,
    rot: &FastRotation,
    codes: &[u8],
    scales: &[f32],
    pos_c: f32,
    cb: usize,
) -> [f32; RERANK_DIAL.len()] {
    let medoid = index.medoid();
    let per: Vec<[f32; RERANK_DIAL.len()]> = (0..nq)
        .into_par_iter()
        .map(|qi| {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let truth = brute_top_k(corpus, n, dim, q, K);
            // Pass 1: popcount walk.
            let qb = rot_signs(rot, q, dim);
            let mut ordered = greedy_walk(
                medoid,
                |id| index.neighbors(id).to_vec(),
                |id| hamming_binary(&qb, &codes[id as usize * cb..(id as usize + 1) * cb]) as f32,
                l_search().max(K),
            );
            // Pass 2: asym re-score the survivors and re-sort (greater = closer).
            let qr = rot.apply_alloc(q);
            let qs: f32 = qr.iter().sum();
            ordered.sort_by(|&a, &b| {
                let sa = scales[a as usize]
                    * pos_c
                    * (2.0
                        * tq1_masked_sum(&codes[a as usize * cb..(a as usize + 1) * cb], &qr, dim)
                        - qs);
                let sb = scales[b as usize]
                    * pos_c
                    * (2.0
                        * tq1_masked_sum(&codes[b as usize * cb..(b as usize + 1) * cb], &qr, dim)
                        - qs);
                sb.total_cmp(&sa)
            });
            let mut r = [0.0f32; RERANK_DIAL.len()];
            for (slot, &m) in RERANK_DIAL.iter().enumerate() {
                let w = (m * K).min(ordered.len());
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
                let approx: Vec<u32> = rr.iter().take(K).map(|&(_, id)| id).collect();
                r[slot] = recall_at_k(&approx, &truth, K);
            }
            r
        })
        .collect();
    let mut mean = [0.0f32; RERANK_DIAL.len()];
    for r in &per {
        for (a, b) in mean.iter_mut().zip(r) {
            *a += b;
        }
    }
    for a in &mut mean {
        *a /= nq as f32;
    }
    mean
}

fn run_dataset(label: &str, corpus_rel: &str, query_rel: &str, native_dim: usize) {
    let pad = native_dim.next_multiple_of(8);
    let n_cap = cap_n();
    let cpath = format!("{ROOT}/{corpus_rel}");
    let qpath = format!("{ROOT}/{query_rel}");
    let Some((corpus, n)) = load_prep(&cpath, n_cap, pad) else {
        println!("  {label:<12} dim {native_dim:<4}  dataset missing ({corpus_rel})");
        return;
    };
    let Some((queries, _)) = load_prep(&qpath, N_QUERIES, pad) else {
        println!("  {label:<12} dim {native_dim:<4}  queries missing");
        return;
    };
    let nq = N_QUERIES.min(queries.len() / pad);
    let dim = pad;

    let ids: Vec<u64> = (0..n as u64).collect();
    let index = VamanaIndex::build(corpus.clone(), ids, dim, &VamanaConfig::default());

    // Encode tq1 codes (rotated sign bits) + asymmetric per-vector scales.
    let rot = FastRotation::new(dim, SEED);
    let pos_c = 0.797_884_6f32 / (dim as f32).sqrt();
    let cb = dim / 8;
    let mut codes = vec![0u8; n * cb];
    let scales: Vec<f32> = codes
        .par_chunks_exact_mut(cb)
        .enumerate()
        .map(|(i, slot)| {
            let unit = &corpus[i * dim..(i + 1) * dim]; // already unit-norm
            let r = rot.apply_alloc(unit);
            let mut inner = 0.0f32;
            for (j, &rv) in r.iter().enumerate() {
                let c = if rv > 0.0 { pos_c } else { -pos_c };
                inner += rv * c;
                if rv > 0.0 {
                    slot[j / 8] |= 1u8 << (j % 8);
                }
            }
            // Corpus is unit-norm so ||v|| = 1; scale = ||v|| / inner.
            1.0 / inner.max(1e-10)
        })
        .collect();

    // Popcount proxy: -hamming(query sign bits, code).
    let r_pop = sweep(
        &corpus,
        &queries,
        n,
        dim,
        nq,
        &index,
        |q| rot_signs(&rot, q, dim),
        |id, qb: &Vec<u8>| -(hamming_binary(qb, &codes[id * cb..(id + 1) * cb]) as f32),
    );
    // Asymmetric proxy: scale * pos_c * (2*masked - q_sum) - production tq1.
    let r_asym = sweep(
        &corpus,
        &queries,
        n,
        dim,
        nq,
        &index,
        |q| {
            let qr = rot.apply_alloc(q);
            let qs: f32 = qr.iter().sum();
            (qr, qs)
        },
        |id, (qr, qs): &(Vec<f32>, f32)| {
            let masked = tq1_masked_sum(&codes[id * cb..(id + 1) * cb], qr, dim);
            scales[id] * pos_c * (2.0 * masked - qs)
        },
    );

    // Hybrid: popcount walk + asym re-score of the shortlist.
    let r_hyb = sweep_hybrid(
        &corpus, &queries, n, dim, nq, &index, &rot, &codes, &scales, pos_c, cb,
    );

    // Per-candidate kernel latency (ns): popcount (walk) vs asym (rescore).
    let q0 = &queries[0..dim];
    let qb0 = rot_signs(&rot, q0, dim);
    let qr0 = rot.apply_alloc(q0);
    let qs0: f32 = qr0.iter().sum();
    let pop_ns = kernel_ns(n, |i| {
        hamming_binary(&qb0, &codes[i * cb..(i + 1) * cb]) as f32
    });
    let asym_ns = kernel_ns(n, |i| {
        scales[i] * pos_c * (2.0 * tq1_masked_sum(&codes[i * cb..(i + 1) * cb], &qr0, dim) - qs0)
    });
    let ram_mb = cb as f64 * n as f64 / 1e6;

    let row = |r: &[f32; RERANK_DIAL.len()]| -> String {
        r.iter().map(|v| format!("{v:>7.3}")).collect()
    };
    println!(
        "\n[{label}]  dim {dim} (nat {native_dim})  n={n}  tq1 RAM {ram_mb:.1} MB  ({cb} B/vec)"
    );
    println!("    recall@   {}", dial_hdr());
    println!("    popcount {}", row(&r_pop));
    println!("    hybrid   {}", row(&r_hyb));
    println!("    asym     {}", row(&r_asym));
    println!(
        "    kernel ns/candidate:  popcount {pop_ns:.1}   asym {asym_ns:.1}   (hybrid walk=popcount, rescore=asym x L_search)"
    );
}

/// Per-candidate kernel latency in ns: score all `n` rows against one fixed
/// pre-encoded query, timed after a warmup, `black_box`ed against DCE.
fn kernel_ns<F: Fn(usize) -> f32>(n: usize, score: F) -> f64 {
    use std::hint::black_box;
    let reps = (5_000_000 / n.max(1)).max(1);
    let mut acc = 0.0f32;
    for i in 0..n {
        acc += score(i);
    }
    black_box(acc);
    let t = std::time::Instant::now();
    let mut acc = 0.0f32;
    for _ in 0..reps {
        for i in 0..n {
            acc += black_box(score(i));
        }
    }
    black_box(acc);
    t.elapsed().as_secs_f64() * 1e9 / (n * reps) as f64
}

fn dial_hdr() -> String {
    RERANK_DIAL
        .iter()
        .map(|m| format!("{:>7}", format!("{m}x")))
        .collect()
}

fn main() {
    println!("=====================================================================");
    println!("tq1 proxy matrix: popcount vs hybrid vs asymmetric across dims");
    println!(
        "N={} (SKEG_BENCH_N)  L_search={} (SKEG_LSEARCH)  k={K}",
        cap_n(),
        l_search()
    );
    println!("recall along the rerank dial + per-candidate kernel latency + RAM");
    println!("=====================================================================");
    for &(label, c, q, dim) in DATASETS {
        run_dataset(label, c, q, dim);
    }
}
