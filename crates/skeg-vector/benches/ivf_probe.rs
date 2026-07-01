#![allow(clippy::cast_precision_loss, clippy::needless_range_loop)]
//! Branch A validation: does IVF coarse-routing ∩ filter recover the filtered
//! top-k CHEAPLY (touching << |S|) at low selectivity? Isolated micro-bench:
//! k-means cells + postings, probe nearest cells ∩ S, EXACT rerank of the pool
//! (isolates routing quality from any proxy). Reports recall@10 + candidates
//! touched, vs qscan (which touches all |S|). Uncorrelated + correlated filters.
//!   SKEG_BENCH_N=100000  SKEG_NQ=200  SKEG_CELLS=1024

use ahash::AHashSet;
use rayon::prelude::*;
use skeg_simd::cosine_f32;

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
const CORPUS: &str = "skeg/bench-compare/embeddings_cache/corpus_mxbai-wiki.npy";
const QUERY: &str = "skeg/bench-compare/embeddings_cache/queries_mxbai-wiki_200.npy";
const K: usize = 10;

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

/// Lloyd k-means, few iters, on a sample; returns centroids (row-major).
fn kmeans(corpus: &[Vec<f32>], n_cells: usize, dim: usize, iters: usize) -> Vec<Vec<f32>> {
    // init: evenly-spaced samples
    let step = (corpus.len() / n_cells).max(1);
    let mut cent: Vec<Vec<f32>> = (0..n_cells)
        .map(|c| corpus[(c * step) % corpus.len()].clone())
        .collect();
    for _ in 0..iters {
        // assign (parallel) + accumulate
        let sums: Vec<(Vec<f32>, u32)> = corpus
            .par_iter()
            .fold(
                || vec![(vec![0.0f32; dim], 0u32); n_cells],
                |mut acc, v| {
                    let c = nearest(&cent, v);
                    for j in 0..dim {
                        acc[c].0[j] += v[j];
                    }
                    acc[c].1 += 1;
                    acc
                },
            )
            .reduce(
                || vec![(vec![0.0f32; dim], 0u32); n_cells],
                |mut a, b| {
                    for c in 0..n_cells {
                        for j in 0..dim {
                            a[c].0[j] += b[c].0[j];
                        }
                        a[c].1 += b[c].1;
                    }
                    a
                },
            );
        for c in 0..n_cells {
            if sums[c].1 > 0 {
                for j in 0..dim {
                    cent[c][j] = sums[c].0[j] / sums[c].1 as f32;
                }
            }
        }
    }
    cent
}

fn nearest(cent: &[Vec<f32>], v: &[f32]) -> usize {
    let mut best = 0;
    let mut bd = f32::NEG_INFINITY;
    for (c, ce) in cent.iter().enumerate() {
        let d = cosine_f32(v, ce);
        if d > bd {
            bd = d;
            best = c;
        }
    }
    best
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
    let n_cells = std::env::var("SKEG_CELLS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let (corpus, dim) = load(&format!("{ROOT}/{CORPUS}"), n_cap);
    let (queries, _) = load(&format!("{ROOT}/{QUERY}"), nq);
    let n = corpus.len();
    println!(
        "IVF probe: mxbai {n} x {dim}, {} queries, {n_cells} cells",
        queries.len()
    );

    let t = std::time::Instant::now();
    let cent = kmeans(&corpus, n_cells, dim, 8);
    // postings: cell -> sorted vec ids
    let mut postings: Vec<Vec<u32>> = vec![Vec::new(); n_cells];
    let assign: Vec<usize> = corpus.par_iter().map(|v| nearest(&cent, v)).collect();
    for (id, &c) in assign.iter().enumerate() {
        postings[c].push(id as u32);
    }
    println!(
        "kmeans+assign: {:.0}s (cell size avg {})",
        t.elapsed().as_secs_f64(),
        n / n_cells
    );

    // correlated cluster (by distance to a fixed center), for the hard case.
    let mut by_center: Vec<u32> = (0..n as u32).collect();
    {
        let center = &corpus[42];
        by_center.sort_unstable_by(|&a, &b| {
            cosine_f32(center, &corpus[b as usize])
                .total_cmp(&cosine_f32(center, &corpus[a as usize]))
        });
    }

    for &(fname, correlated) in &[("uniform", false), ("correlated", true)] {
        for &sel in &[0.01f64, 0.05, 0.10] {
            let msize = (n as f64 * sel) as usize;
            let sset: AHashSet<u32> = if correlated {
                by_center[..msize].iter().copied().collect()
            } else {
                let step = (1.0 / sel).round() as u32;
                (0..n as u32).filter(|id| id % step == 0).collect()
            };
            // truth: brute top-k over S
            let truth: Vec<AHashSet<u32>> = queries
                .par_iter()
                .map(|q| {
                    let mut s: Vec<(f32, u32)> = sset
                        .iter()
                        .map(|&id| (cosine_f32(q, &corpus[id as usize]), id))
                        .collect();
                    s.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                    s.iter().take(K).map(|&(_, id)| id).collect()
                })
                .collect();
            println!("-- {fname} {:.0}% (|S|={}) --", sel * 100.0, sset.len());
            // predicate-aware: cells that actually contain S members (compute once).
            let s_cells: Vec<usize> = (0..n_cells)
                .filter(|&c| postings[c].iter().any(|id| sset.contains(id)))
                .collect();
            // "near" = rank all cells by query distance; "PA" = rank only S-cells.
            for &(strat, pool) in &[("near", None), ("PA", Some(&s_cells))] {
                for &nprobe in &[16usize, 64, 256] {
                    let mut hits = 0usize;
                    let mut touched = 0usize;
                    let t = std::time::Instant::now();
                    for (q, tr) in queries.iter().zip(&truth) {
                        let mut cd: Vec<(f32, usize)> = match pool {
                            Some(sc) => sc.iter().map(|&c| (cosine_f32(q, &cent[c]), c)).collect(),
                            None => cent
                                .iter()
                                .enumerate()
                                .map(|(c, ce)| (cosine_f32(q, ce), c))
                                .collect(),
                        };
                        cd.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                        let mut cand: Vec<(f32, u32)> = Vec::new();
                        for &(_, c) in cd.iter().take(nprobe) {
                            for &id in &postings[c] {
                                if sset.contains(&id) {
                                    cand.push((cosine_f32(q, &corpus[id as usize]), id));
                                    touched += 1;
                                }
                            }
                        }
                        cand.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                        hits += cand
                            .iter()
                            .take(K)
                            .filter(|(_, id)| tr.contains(id))
                            .count();
                    }
                    let ms = t.elapsed().as_secs_f64() * 1e3 / queries.len() as f64;
                    println!(
                        "   {strat:<4} nprobe {nprobe:<4} recall {:.4}  touched {:>6}/{} ({:.0}%)  {ms:.3} ms/q",
                        hits as f64 / (queries.len() * K) as f64,
                        touched / queries.len(),
                        sset.len(),
                        100.0 * (touched / queries.len()) as f64 / sset.len().max(1) as f64,
                    );
                }
            }
        }
    }
}
