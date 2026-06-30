#![allow(clippy::too_many_arguments, clippy::type_complexity)]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
#![allow(clippy::needless_range_loop)]

//! TurboQuant 1-bit symmetric-popcount gate.
//!
//! Question: does a Qdrant-style symmetric binary first pass (rotate query,
//! take sign bits, score by Hamming popcount) match the recall of the current
//! asymmetric f32-ADC tq1 proxy once you turn the rerank dial, while running
//! much faster per candidate?
//!
//! The stored tq1 code is ALREADY the rotated sign-bit string (see
//! `TurboQuant1::encode`), so the popcount path adds no storage and no rebuild:
//! it only changes how the query is encoded (f32 rotated -> rotated sign bits)
//! and how a candidate is scored (asymmetric float dot -> XOR + count_ones).
//! The per-vector scale correction is magnitude information that Hamming
//! discards by construction, so the symmetric path leans entirely on rerank to
//! recover precision - exactly the trade Qdrant reports (binary-only 0.61,
//! +rerank 4x -> 0.93).
//!
//! Pre-registered hypotheses (decide BEFORE reading the numbers):
//!   H1 latency : popcount proxy >= 5x faster per call than f32 ADC.
//!   H2 recall  : popcount + rerank 8x clears recall@10 >= 0.95 on BOTH
//!                distributions (mxbai 1024d isotropic, MiniLM 384d aniso).
//!   H3 cond.   : if MiniLM stays < 0.95 even at 16x rerank, popcount-tq1 is
//!                conditional (isotropic-leaning models only), documented the
//!                same way RESULTS.md documents asymmetric tq1.
//!
//! A fail on H1 kills the branch (no latency win, no reason to lose recall).
//! A fail on H2 but pass on H3 ships it as a conditional fast tier. The
//! asymmetric f32-ADC tq1 column is the baseline both hypotheses measure
//! against, run side by side on the same graph and queries.

use ahash::AHashSet;
use rayon::prelude::*;
use skeg_simd::{cosine_f32, hamming_binary};
use skeg_vector::{TurboQuant1, VamanaConfig, VamanaIndex};

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
/// Rerank dial measured as a multiple of K, mirroring the Qdrant article's
/// oversampling axis. 64 (~6.4x) is the value the existing tq_walk_gate fixes.
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

/// Greedy walk scored by `proxy` (smaller = closer), returning the final
/// candidate list ordered by proxy ascending (closest first).
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

/// Rotated query -> LSB-first sign bits, packed exactly as `TurboQuant1::encode`
/// packs the stored code (`code[i/8] |= 1 << (i % 8)` when the coord is > 0).
/// Matching the packing is what makes `hamming_binary(q_bits, code)` a valid
/// sign-agreement count.
fn pack_query_signs(q_rot: &[f32]) -> Vec<u8> {
    let mut bits = vec![0u8; q_rot.len().div_ceil(8)];
    for (i, &r) in q_rot.iter().enumerate() {
        if r > 0.0 {
            bits[i / 8] |= 1u8 << (i % 8);
        }
    }
    bits
}

/// Recall@10 at each rerank-dial width for one proxy, plus the f32 walk
/// reference. Walks ONCE per query (capturing the full L_SEARCH list), then
/// reranks at every dial width from the same list - so the dial sweep is free
/// and the proxy's candidate quality is held fixed across widths.
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

/// Per-call latency of a proxy kernel, ns. Scores `samples` rows against one
/// fixed (pre-rotated / pre-binarized) query in a tight loop; `black_box`
/// keeps the optimizer from hoisting the work out.
fn kernel_ns<F: Fn(usize) -> f32>(samples: usize, reps: usize, score: F) -> f64 {
    use std::hint::black_box;
    // Warm up.
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

fn run_gate(label: &str, corpus_npy: &str, query_npy: &str) {
    let Some((corpus, n, dim)) = load_npy(corpus_npy) else {
        println!("\n=== {label}: dataset missing ===");
        return;
    };
    let Some((queries, q_n, q_dim)) = load_npy(query_npy) else {
        println!("\n=== {label}: queries missing ===");
        return;
    };
    assert_eq!(dim, q_dim, "corpus/query dim mismatch");
    let nq = N_QUERIES.min(q_n);
    println!("\n=== {label}: corpus {n} x {dim}, queries {nq} ===");

    let ids: Vec<u64> = (0..n as u64).collect();
    let index = VamanaIndex::build(corpus.clone(), ids, dim, &VamanaConfig::default());

    let tq1 = TurboQuant1::new(dim, 0xC0DE_BEEF);
    let cb = tq1.code_bytes();
    let mut codes = vec![0u8; n * cb];
    let scales: Vec<f32> = (0..n)
        .map(|i| {
            let (c, s) = tq1.encode(&corpus[i * dim..(i + 1) * dim]);
            codes[i * cb..(i + 1) * cb].copy_from_slice(&c);
            s
        })
        .collect();

    // Asymmetric f32-ADC baseline (current production tq1 proxy).
    let (adc_recall, ref_recall) = recall_sweep(&corpus, &queries, n, dim, nq, &index, |id, q| {
        let q_rot = tq1.rotate_query(q);
        tq1.approx_inner(&codes[id * cb..(id + 1) * cb], scales[id], &q_rot)
    });

    // Symmetric popcount candidate (the innovation).
    let (pop_recall, _) = recall_sweep(&corpus, &queries, n, dim, nq, &index, |id, q| {
        let q_bits = pack_query_signs(&tq1.rotate_query(q));
        // Fewer differing sign bits = closer; negate so "greater = closer"
        // matches the asymmetric proxy's sign convention.
        -(hamming_binary(&q_bits, &codes[id * cb..(id + 1) * cb]) as f32)
    });

    // Kernel latency, isolated from rotation (rotation is shared by both
    // paths). One fixed query, rotated/binarized once outside the timed loop.
    let q0 = &queries[0..dim];
    let q_rot = tq1.rotate_query(q0);
    let q_bits = pack_query_signs(&q_rot);
    let reps = (5_000_000 / n.max(1)).max(1);
    let adc_ns = kernel_ns(n, reps, |i| {
        tq1.approx_inner(&codes[i * cb..(i + 1) * cb], scales[i], &q_rot)
    });
    let pop_ns = kernel_ns(n, reps, |i| {
        hamming_binary(&q_bits, &codes[i * cb..(i + 1) * cb]) as f32
    });

    println!("  f32 walk reference recall@10: {ref_recall:.4}");
    println!("  rerank dial (xK):     {}", dial_header());
    println!("  asym f32-ADC tq1:     {}", row(&adc_recall));
    println!("  sym  popcount tq1:    {}", row(&pop_recall));
    println!(
        "  kernel ns/candidate:  adc {adc_ns:.1}   popcount {pop_ns:.1}   speedup {:.1}x",
        adc_ns / pop_ns
    );

    // Verdicts against pre-registered hypotheses.
    let h1 = adc_ns / pop_ns >= 5.0;
    let dial8 = RERANK_DIAL.iter().position(|&m| m == 8).unwrap();
    let dial16 = RERANK_DIAL.iter().position(|&m| m == 16).unwrap();
    let h2 = pop_recall[dial8] >= 0.95;
    let h3 = pop_recall[dial16] >= 0.95;
    println!(
        "  H1 latency >=5x: {}   H2 recall@8x >=0.95: {}   H3 recall@16x >=0.95: {}",
        pass(h1),
        pass(h2),
        pass(h3),
    );
}

fn dial_header() -> String {
    RERANK_DIAL
        .iter()
        .map(|m| format!("{:>8}", format!("{m}x")))
        .collect()
}

fn row(r: &[f32]) -> String {
    r.iter().map(|v| format!("{v:>8.4}")).collect()
}

fn pass(b: bool) -> &'static str {
    if b { "PASS" } else { "FAIL" }
}

fn main() {
    println!("=========================================================");
    println!("TurboQuant 1-bit symmetric-popcount gate (vs asym f32-ADC)");
    println!("H1 latency>=5x  H2 recall@8x>=0.95  H3 recall@16x>=0.95");
    println!("=========================================================");
    run_gate("mxbai 1024d wiki-100K", MXBAI_CORPUS, MXBAI_QUERY);
    run_gate("MiniLM 384d wiki-100K", MINILM_CORPUS, MINILM_QUERY);
}
