//! M8 - memory + performance of in-memory vs on-disk Vamana, measured at
//! 10K, 100K and 1M. Not a Criterion bench: a reporting harness
//! (`harness = false`).
//!
//! The 10K row runs on the real Ollama embeddings cached by `bench-compare`,
//! where recall is meaningful. The 100K and 1M rows run on clustered
//! synthetic vectors: recall on synthetic data is not meaningful (see earlier
//! notes), but RAM, build time and search latency are data-structure
//! properties and are measured for real here, replacing the design-doc
//! extrapolations.
//!
//! Reported per row: build time, in-memory bytes (`heap_bytes`), on-disk
//! resident bytes (`resident_bytes`), process RSS, and search latency.

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::process::Command;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use skeg_vector::{DiskVamanaIndex, VamanaConfig, VamanaIndex};

const MIB: f64 = 1024.0 * 1024.0;
const K: usize = 10;
const DIM: usize = 1024;
const QUERIES: usize = 100;
const CORPUS_NPY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../bench-compare/embeddings_cache/corpus_mxbai-embed-large_10000.npy"
);
const QUERY_NPY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../bench-compare/embeddings_cache/queries_mxbai-embed-large_200.npy"
);

/// Minimal `.npy` reader (v1.0, little-endian f32, C order).
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

/// Resident set of this process in MiB, via `ps`.
fn self_rss_mib() -> f64 {
    let pid = std::process::id().to_string();
    Command::new("ps")
        .args(["-o", "rss=", "-p", &pid])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .map_or(0.0, |kb| kb / 1024.0)
}

/// Clustered synthetic corpus: ~n/100 centres, points are centre + noise.
fn clustered(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let n_clusters = (n / 100).max(8);
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect())
        .collect();
    let mut out = Vec::with_capacity(n * dim);
    for _ in 0..n {
        let c = &centers[rng.random_range(0..n_clusters)];
        for &x in c {
            out.push(x + rng.random_range(-0.15..0.15));
        }
    }
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na * nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

fn brute_force(vectors: &[f32], dim: usize, query: &[f32], k: usize) -> Vec<u64> {
    let n = vectors.len() / dim;
    let mut scored: Vec<(f32, u64)> = (0..n)
        .map(|i| (cosine(query, &vectors[i * dim..(i + 1) * dim]), i as u64))
        .collect();
    scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    scored.into_iter().take(k).map(|(_, id)| id).collect()
}

/// Build the index over `vectors`, save it, open the on-disk form, and report.
/// `oracle` enables a recall column (real data only).
fn measure(
    label: &str,
    n: usize,
    dim: usize,
    vectors: Vec<f32>,
    queries: &[f32],
    oracle: Option<&[Vec<u64>]>,
) {
    let ids: Vec<u64> = (0..n as u64).collect();

    let t0 = Instant::now();
    let index = VamanaIndex::build(vectors, ids, dim, &VamanaConfig::default());
    let build_s = t0.elapsed().as_secs_f64();
    let inmem_mib = index.heap_bytes() as f64 / MIB;
    let rss_inmem = self_rss_mib();

    let tmp = tempfile::TempDir::new().expect("tempdir");
    index.save(tmp.path()).expect("save");
    drop(index); // free the in-memory index before opening the on-disk one
    let disk = DiskVamanaIndex::open(tmp.path()).expect("open");
    let disk_mib = disk.resident_bytes() as f64 / MIB;
    let rss_disk = self_rss_mib();

    let n_q = queries.len() / dim;
    let mut t_us = 0u128;
    let mut hits = 0usize;
    let mut total = 0usize;
    for qi in 0..n_q {
        let q = &queries[qi * dim..(qi + 1) * dim];
        let s = Instant::now();
        let got = disk.search(q, K).expect("search");
        t_us += s.elapsed().as_micros();
        if let Some(orc) = oracle {
            hits += got.iter().filter(|(id, _)| orc[qi].contains(id)).count();
            total += K;
        }
    }
    let recall = if total > 0 {
        format!("{:.4}", hits as f64 / total as f64)
    } else {
        "synth".to_string()
    };

    println!(
        "{label:>6} {n:>9} {build_s:>9.2} {inmem_mib:>11.1} {disk_mib:>10.1} {:>7.2}x \
         {rss_inmem:>9.0} {rss_disk:>9.0} {:>9} {recall:>9}",
        inmem_mib / disk_mib,
        t_us / n_q.max(1) as u128,
    );
}

fn main() {
    eprintln!("M8 Vamana - in-memory vs on-disk, dim={DIM}, k={K}");
    eprintln!("inmem/disk MiB = heap_bytes / resident_bytes; rss = process RSS via ps\n");
    println!(
        "{:>6} {:>9} {:>9} {:>11} {:>10} {:>8} {:>9} {:>9} {:>9} {:>9}",
        "src",
        "N",
        "build_s",
        "inmem_MiB",
        "disk_MiB",
        "ratio",
        "rss_im",
        "rss_dk",
        "search_us",
        "recall",
    );

    // N=10K on real Ollama embeddings, with real query embeddings - recall is
    // meaningful here.
    match (load_npy(CORPUS_NPY), load_npy(QUERY_NPY)) {
        (Some((corpus, n_full, dim)), Some((queries, n_q, q_dim))) if dim == q_dim => {
            let n = 10_000.min(n_full);
            let vectors = corpus[..n * dim].to_vec();
            let oracle: Vec<Vec<u64>> = (0..n_q)
                .map(|qi| brute_force(&vectors, dim, &queries[qi * dim..(qi + 1) * dim], K))
                .collect();
            measure("real", n, dim, vectors, &queries, Some(&oracle));
        }
        _ => eprintln!("(real embeddings cache missing - skipping the 10K real-data row)"),
    }

    // N=100K and N=1M on clustered synthetic vectors - RAM/build/latency.
    for &n in &[100_000usize, 1_000_000] {
        let vectors = clustered(n, DIM, 1);
        let queries = clustered(QUERIES, DIM, 0xBEEF);
        measure("synth", n, DIM, vectors, &queries, None);
    }
}
