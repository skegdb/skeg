#![allow(clippy::too_many_arguments, clippy::type_complexity)]

//! TurboQuant Gate 1 - flat scan recall on wiki-100K, dual-distribution.
//!
//! Pre-registered (`turboquant/PLAN.md` §4):
//!   - hypothesis: TurboQuant 4-bit flat recall@10 >= 0.99 on wiki-100K
//!   - protocol: brute-force ground truth, top-L candidates from TurboQuant
//!     scan, re-rank with exact f32 cosine, recall@10
//!   - dual-distribution: gate must hold on mxbai (isotropic) AND MiniLM
//!     (anisotropic with dim-deads). Single-distribution pass is insufficient
//!     (memory feedback-multi-distribution: 10 prior falsifications on mxbai
//!     alone, asymmetry positives=upper-bound).
//!
//! Reporting harness (`harness = false`): timing not measured, prints recall.

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
#![allow(clippy::needless_range_loop)]

use rayon::prelude::*;
use skeg_simd::cosine_f32;
use skeg_vector::{TurboQuant1, TurboQuant2, TurboQuant4};

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
const RERANK_L: usize = 300;

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
        .map(|i| {
            let row = &corpus[i * dim..(i + 1) * dim];
            (cosine_f32(query, row), i as u32)
        })
        .collect();
    scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    scored.iter().take(k).map(|&(_, id)| id).collect()
}

fn recall_at_k(approx: &[u32], truth: &[u32], k: usize) -> f32 {
    let take = approx.len().min(k);
    let truth_set: std::collections::HashSet<u32> = truth.iter().take(k).copied().collect();
    let hit = approx[..take]
        .iter()
        .filter(|id| truth_set.contains(id))
        .count();
    hit as f32 / k as f32
}

/// Run flat-scan gate with a generic quantizer (4-bit or 2-bit).
fn run_gate_with<E, Q>(
    label: &str,
    bits_label: &str,
    corpus: &[f32],
    queries: &[f32],
    n: usize,
    dim: usize,
    nq: usize,
    code_bytes: usize,
    encode: E,
    rotate_query: impl Fn(&[f32]) -> Vec<f32> + Sync,
    approx_inner: Q,
) -> f32
where
    E: Fn(&[f32]) -> (Vec<u8>, f32) + Sync,
    Q: Fn(&[u8], f32, &[f32]) -> f32 + Sync,
{
    let t_enc = std::time::Instant::now();
    let enc: Vec<(Vec<u8>, f32)> = (0..n)
        .into_par_iter()
        .map(|i| encode(&corpus[i * dim..(i + 1) * dim]))
        .collect();
    let mut codes = Vec::with_capacity(n * code_bytes);
    let mut scales = Vec::with_capacity(n);
    for (c, s) in enc {
        codes.extend_from_slice(&c);
        scales.push(s);
    }
    println!(
        "  [{}] encode {} vectors: {:.2}s",
        bits_label,
        n,
        t_enc.elapsed().as_secs_f32()
    );

    let t_search = std::time::Instant::now();
    let recalls: Vec<f32> = (0..nq)
        .into_par_iter()
        .map(|q_idx| {
            let query = &queries[q_idx * dim..(q_idx + 1) * dim];
            let truth = brute_top_k(corpus, n, dim, query, K);
            let q_rot = rotate_query(query);
            let mut scored: Vec<(f32, u32)> = (0..n)
                .map(|i| {
                    let code = &codes[i * code_bytes..(i + 1) * code_bytes];
                    (approx_inner(code, scales[i], &q_rot), i as u32)
                })
                .collect();
            let l = RERANK_L.min(n);
            scored.select_nth_unstable_by(l - 1, |a, b| b.0.total_cmp(&a.0));
            scored.truncate(l);
            let mut rerank: Vec<(f32, u32)> = scored
                .iter()
                .map(|&(_, id)| {
                    let row = &corpus[id as usize * dim..(id as usize + 1) * dim];
                    (cosine_f32(query, row), id)
                })
                .collect();
            rerank.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            let approx: Vec<u32> = rerank.iter().take(K).map(|&(_, id)| id).collect();
            recall_at_k(&approx, &truth, K)
        })
        .collect();
    let recall = recalls.iter().sum::<f32>() / nq as f32;
    println!(
        "  [{}] search {} queries: {:.2}s   recall@10 = {:.4}   bytes/vec = {}",
        bits_label,
        nq,
        t_search.elapsed().as_secs_f32(),
        recall,
        code_bytes,
    );
    let _ = label;
    recall
}

fn run_gate(label: &str, corpus_npy: &str, query_npy: &str) -> Option<(f32, f32, f32)> {
    let (corpus, n, dim) = load_npy(corpus_npy)?;
    let (queries, q_n, q_dim) = load_npy(query_npy)?;
    assert_eq!(dim, q_dim, "corpus/query dim mismatch");
    let nq = N_QUERIES.min(q_n);
    println!(
        "\n=== {label}: corpus {n} x {dim}, queries {q_n} (using first {nq}) ===",
        label = label,
        n = n,
        dim = dim,
        q_n = q_n,
        nq = nq
    );

    let t_build = std::time::Instant::now();
    let tq4 = TurboQuant4::new(dim, 0xC0DE_BEEF);
    let tq2 = TurboQuant2::new(dim, 0xC0DE_BEEF);
    let tq1 = TurboQuant1::new(dim, 0xC0DE_BEEF);
    println!(
        "  rotation build (shared seed, x3): {:.2}s",
        t_build.elapsed().as_secs_f32()
    );

    let recall_4 = run_gate_with(
        label,
        "4-bit",
        &corpus,
        &queries,
        n,
        dim,
        nq,
        tq4.code_bytes(),
        |v| tq4.encode(v),
        |q| tq4.rotate_query(q),
        |c, s, q| tq4.approx_inner(c, s, q),
    );
    let recall_2 = run_gate_with(
        label,
        "2-bit",
        &corpus,
        &queries,
        n,
        dim,
        nq,
        tq2.code_bytes(),
        |v| tq2.encode(v),
        |q| tq2.rotate_query(q),
        |c, s, q| tq2.approx_inner(c, s, q),
    );
    let recall_1 = run_gate_with(
        label,
        "1-bit",
        &corpus,
        &queries,
        n,
        dim,
        nq,
        tq1.code_bytes(),
        |v| tq1.encode(v),
        |q| tq1.rotate_query(q),
        |c, s, q| tq1.approx_inner(c, s, q),
    );
    Some((recall_4, recall_2, recall_1))
}

fn main() {
    println!("=======================================================");
    println!("TurboQuant Gate 1 - flat scan recall@10 dual-distribution");
    println!("Pass criterion: recall@10 >= 0.99 on BOTH distributions");
    println!("=======================================================");

    let mxbai = run_gate("mxbai 1024d wiki-100K", MXBAI_CORPUS, MXBAI_QUERY);
    let minilm = run_gate("MiniLM 384d wiki-100K", MINILM_CORPUS, MINILM_QUERY);

    println!("\n=== Gate 1 verdict ===");
    let summarize = |label: &str, res: Option<(f32, f32, f32)>| match res {
        Some((r4, r2, r1)) => {
            let p4 = r4 >= 0.99;
            let p2 = r2 >= 0.99;
            let p1 = r1 >= 0.99;
            println!(
                "  {label:<14}  4b {r4:.4} {l4}   2b {r2:.4} {l2}   1b {r1:.4} {l1}",
                label = label,
                r4 = r4,
                l4 = if p4 { "PASS" } else { "FAIL" },
                r2 = r2,
                l2 = if p2 { "PASS" } else { "FAIL" },
                r1 = r1,
                l1 = if p1 { "PASS" } else { "FAIL" }
            );
            (p4, p2, p1)
        }
        None => {
            println!("  {label:<14}  dataset missing");
            (false, false, false)
        }
    };
    let (mx4, mx2, mx1) = summarize("mxbai 1024d", mxbai);
    let (mn4, mn2, mn1) = summarize("MiniLM 384d", minilm);

    let verdict = |p_mx: bool, p_mn: bool| match (p_mx, p_mn) {
        (true, true) => "PASS dual-distribution",
        (true, false) => "PARTIAL (mxbai only)",
        (false, true) => "PARTIAL (MiniLM only)",
        (false, false) => "FAIL",
    };
    println!("\n  TurboQuant 4-bit -> {}", verdict(mx4, mn4));
    println!("  TurboQuant 2-bit -> {}", verdict(mx2, mn2));
    println!("  TurboQuant 1-bit -> {}", verdict(mx1, mn1));
}
