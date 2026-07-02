#![allow(clippy::cast_precision_loss)]
//! COMPLETE real recall@100 (k=100) across ALL datasets, tq1 vs tq2, at the
//! default serving params (l=300, rr=800) and a wide walk (l=2000). Dims are
//! zero-padded to a multiple of 8. Answers whether tq1's recall@100 gap is
//! dataset-specific or general.
//!   SKEG_BENCH_N=100000  SKEG_NQ=200

use ahash::AHashSet;
use rayon::prelude::*;
use skeg_simd::cosine_f32;
use skeg_vector::{DiskVamanaIndex, QuantKind};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");

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
        "qwen3-4b",
        "skeg/bench-compare/embeddings_cache/corpus_qwen3emb4b_100k.npy",
        "skeg/bench-compare/embeddings_cache/queries_qwen3emb4b_1k.npy",
        2560,
    ),
];

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

fn load_prep(path: &str, n_cap: usize, pad: usize) -> Option<(Vec<Vec<f32>>, usize)> {
    let (data, rows, dim) = load_npy(path)?;
    let n = n_cap.min(rows);
    let out = (0..n)
        .map(|i| {
            let mut v = vec![0.0f32; pad];
            v[..dim].copy_from_slice(&data[i * dim..i * dim + dim]);
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
            v.iter_mut().for_each(|x| *x /= norm);
            v
        })
        .collect();
    Some((out, n))
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
    println!("real recall@100 (k=100) - all datasets, N<={n_cap}");
    println!(
        "{:<10} {:>5} {:>7}  {:>18}  {:>18}",
        "dataset", "dim", "n", "tq1 def / wide", "tq2 def / wide"
    );
    for &(label, cpath, qpath, native) in DATASETS {
        let pad = native.next_multiple_of(8);
        let Some((corpus, n)) = load_prep(&format!("{ROOT}/{cpath}"), n_cap, pad) else {
            println!("{label:<10} (corpus missing)");
            continue;
        };
        let Some((queries, _)) = load_prep(&format!("{ROOT}/{qpath}"), nq, pad) else {
            println!("{label:<10} (queries missing)");
            continue;
        };
        let truth: Vec<AHashSet<u64>> = queries
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
        let mut cell = [String::new(), String::new()];
        for (bi, bits) in [1u8, 2].iter().enumerate() {
            let tier = QuantKind::TurboQuant { bits: *bits };
            let tmp = std::env::temp_dir().join(format!("skeg_r100all_{label}_{bits}"));
            let _ = std::fs::remove_dir_all(&tmp);
            std::fs::create_dir_all(&tmp).unwrap();
            let mut idx = DiskVamanaIndex::create_empty_with_tier(&tmp, pad, 300, tier).unwrap();
            for (id, v) in corpus.iter().enumerate() {
                idx.insert(id as u64, v).unwrap();
            }
            idx.consolidate().unwrap();
            let rec = |ls: usize, rr: usize| -> f64 {
                let mut hits = 0usize;
                for (q, tr) in queries.iter().zip(&truth) {
                    hits += idx
                        .search_with_params(q, 100, ls, rr)
                        .unwrap()
                        .iter()
                        .filter(|(id, _)| tr.contains(id))
                        .count();
                }
                hits as f64 / (queries.len() * 100) as f64
            };
            cell[bi] = format!("{:.4} / {:.4}", rec(300, 800), rec(2000, 12800));
            drop(idx);
            let _ = std::fs::remove_dir_all(&tmp);
        }
        println!(
            "{label:<10} {pad:>5} {n:>7}  {:>18}  {:>18}",
            cell[0], cell[1]
        );
    }
}
