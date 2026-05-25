//! Step 7 gate - graph-page cache locality of the Vamana greedy walk, and
//! whether a BFS-from-medoid node reordering creates the locality paging
//! needs. Run at N=100K so the cache can hold a query's working set (the
//! 10K run was confounded: there a 10%-of-graph cache was smaller than one
//! walk's working set - see OBSERVATIONS).
//!
//! Not a Criterion bench: a reporting harness (`harness = false`).
//!
//! Two synthetic datasets at N=100K, plus the real 10K corpus as reference:
//!   - uniform on the unit sphere: the worst case for locality, no geometric
//!     structure to exploit. A pass here is decisive.
//!   - clustered: natural structure, an optimistic proxy for real embeddings.
//!
//! Method: build the graph, record the greedy walk's node-access traces,
//! permute node ids by BFS from the medoid (graph-adjacent nodes -> adjacent
//! pages), re-simulate an LRU page cache on the same traces under both
//! layouts. Gate (pre-registered): BFS hit rate at a 10%-of-graph cache
//! >= 0.50.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use skeg_vector::{DiskVamanaIndex, VamanaConfig, VamanaIndex};

const NODES_PER_PAGE: u32 = 63; // 16 KB page / 260-byte node
const PAGE_KIB: f64 = 16.0;
const N_QUERIES: usize = 200;
const CORPUS_NPY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../bench-compare/embeddings_cache/corpus_mxbai-embed-large_10000.npy"
);
const QUERY_NPY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../bench-compare/embeddings_cache/queries_mxbai-embed-large_200.npy"
);

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

/// One standard-normal sample (Box-Muller).
fn gaussian(rng: &mut StdRng) -> f32 {
    let u1 = rng.random::<f32>().max(1e-9);
    let u2 = rng.random::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// `n` vectors uniformly distributed on the unit sphere - the worst case for
/// any locality structure.
fn uniform_sphere(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n * dim);
    for _ in 0..n {
        let mut v: Vec<f32> = (0..dim).map(|_| gaussian(&mut rng)).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for x in &mut v {
            *x /= norm;
        }
        out.extend(v);
    }
    out
}

/// `n` clustered vectors: ~n/100 centres, each point a centre plus noise.
fn clustered(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let nc = (n / 100).max(8);
    let centers: Vec<Vec<f32>> = (0..nc)
        .map(|_| (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect())
        .collect();
    let mut out = Vec::with_capacity(n * dim);
    for _ in 0..n {
        let c = &centers[rng.random_range(0..nc)];
        for &x in c {
            out.push(x + rng.random_range(-0.15..0.15));
        }
    }
    out
}

/// LRU hit rate over `trace` with a cache of `capacity` pages.
fn lru_hit_rate(trace: &[u32], capacity: usize) -> f64 {
    if capacity == 0 || trace.is_empty() {
        return 0.0;
    }
    let mut cache: Vec<u32> = Vec::with_capacity(capacity + 1);
    let mut hits = 0usize;
    for &p in trace {
        if let Some(pos) = cache.iter().position(|&x| x == p) {
            cache.remove(pos);
            cache.push(p);
            hits += 1;
        } else {
            cache.push(p);
            if cache.len() > capacity {
                cache.remove(0);
            }
        }
    }
    hits as f64 / trace.len() as f64
}

/// Build the graph, trace the walks, and report current vs BFS-reorder cache
/// hit rates. Returns the BFS hit rate at a 10%-of-graph cache (the gate).
fn analyze(
    label: &str,
    corpus: Vec<f32>,
    n: usize,
    dim: usize,
    queries: &[f32],
    n_q: usize,
) -> f64 {
    let ids: Vec<u64> = (0..n as u64).collect();
    let t = Instant::now();
    let index = VamanaIndex::build(corpus, ids, dim, &VamanaConfig::default());
    let build_s = t.elapsed().as_secs_f64();
    let tmp = tempfile::TempDir::new().expect("tempdir");
    index.save(tmp.path()).expect("save");
    drop(index);
    let disk = DiskVamanaIndex::open(tmp.path()).expect("open");

    let total_pages = (n as u32).div_ceil(NODES_PER_PAGE);
    let node_traces: Vec<Vec<u32>> = (0..n_q)
        .map(|qi| {
            disk.search_node_trace(&queries[qi * dim..(qi + 1) * dim])
                .expect("trace")
        })
        .collect();
    let total_accesses: usize = node_traces.iter().map(Vec::len).sum();
    let avg_nodes = total_accesses as f64 / n_q as f64;

    // BFS-from-medoid permutation: perm[old_id] = new position.
    let bfs = disk.bfs_order();
    let mut perm = vec![0u32; n];
    for (new_pos, &old_id) in bfs.iter().enumerate() {
        perm[old_id as usize] = new_pos as u32;
    }
    let mut trace_current: Vec<u32> = Vec::with_capacity(total_accesses);
    let mut trace_bfs: Vec<u32> = Vec::with_capacity(total_accesses);
    for nodes in &node_traces {
        for &node in nodes {
            trace_current.push(node / NODES_PER_PAGE);
            trace_bfs.push(perm[node as usize] / NODES_PER_PAGE);
        }
    }

    println!("\n== {label}: N={n}, dim={dim} ==");
    println!(
        "  build {build_s:.1}s | graph {total_pages} pages ({:.1} MiB) | walk avg {avg_nodes:.0} nodes",
        f64::from(total_pages) * PAGE_KIB / 1024.0
    );
    println!(
        "  {:>8}{:>10}{:>14}{:>14}",
        "cache", "MiB", "current", "BFS reorder"
    );
    let mut bfs_at_10 = 0.0;
    for &frac in &[0.05f64, 0.10, 0.25, 0.50, 1.00] {
        let cap = ((f64::from(total_pages) * frac).round() as usize).max(1);
        let hr_cur = lru_hit_rate(&trace_current, cap);
        let hr_bfs = lru_hit_rate(&trace_bfs, cap);
        if (frac - 0.10).abs() < 1e-9 {
            bfs_at_10 = hr_bfs;
        }
        println!(
            "  {:>7.0}%{:>10.2}{:>14.4}{:>14.4}",
            frac * 100.0,
            cap as f64 * PAGE_KIB / 1024.0,
            hr_cur,
            hr_bfs,
        );
    }
    bfs_at_10
}

fn main() {
    eprintln!("Vamana graph-page cache locality - BFS reorder gate at scale\n");

    // Reference: real mxbai-embed-large 10K (confounded scale, for context).
    if let (Some((corpus, n, dim)), Some((queries, n_q, q_dim))) =
        (load_npy(CORPUS_NPY), load_npy(QUERY_NPY))
    {
        if dim == q_dim {
            analyze("real mxbai 10K (reference)", corpus, n, dim, &queries, n_q);
        }
    } else {
        eprintln!("(real embeddings cache missing - skipping the 10K reference row)");
    }

    let dim = 1024;
    let n = 100_000;

    // The decisive gate: uniform on the sphere, the worst case for locality.
    let uni_q = uniform_sphere(N_QUERIES, dim, 0xBEEF);
    let uni_gate = analyze(
        "uniform sphere 100K (GATE)",
        uniform_sphere(n, dim, 1),
        n,
        dim,
        &uni_q,
        N_QUERIES,
    );

    // Optimistic proxy: clustered data has natural structure.
    let clu_q = clustered(N_QUERIES, dim, 0xCAFE);
    let clu_gate = analyze(
        "clustered 100K (optimistic)",
        clustered(n, dim, 2),
        n,
        dim,
        &clu_q,
        N_QUERIES,
    );

    println!("\n== gate (BFS reorder, cache 10%) ==");
    let verdict = |label: &str, hr: f64| {
        println!(
            "  {label}: BFS {hr:.4} {} 0.50",
            if hr >= 0.50 { ">= PASS" } else { "< MISS" }
        );
    };
    verdict("uniform sphere 100K (decisive)", uni_gate);
    verdict("clustered 100K    (optimistic)", clu_gate);
}
