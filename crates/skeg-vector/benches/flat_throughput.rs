//! Slice D step 1: flat-scan throughput baseline on synthetic data.
//!
//! Custom main (`harness = false`): prints CSV rows to stdout, one per
//! (tier, dim, N, threading) cell. Comparison target: turbovec's
//! verified numbers in benchmarks/results/speed_d{1536,3072}_4bit_arm_*.json
//! (1.99 ms/q st, 0.185 ms/q mt on M3 Max at d=1536 N=100k 4-bit).
//!
//! Why synthetic: the Rust bench harness does not need the embedding
//! pipeline. Synthetic isotropic Gaussian vectors give us throughput
//! numbers per tier kind; real-data recall is covered by tq_flat_gate
//! and by the Python bench-compare harness.
//!
//! Tunable via env:
//!   SKEG_FLAT_N            corpus size       (default 100000)
//!   SKEG_FLAT_DIMS         comma list of dim (default 384,768,1024,1536)
//!   SKEG_FLAT_QUERIES      query count       (default 1000)
//!   SKEG_FLAT_K            top-K             (default 10)
//!   SKEG_FLAT_THREADS      comma list (default 1,4,8) (8 ~= M1 Pro perf)
//!   SKEG_FLAT_TIERS        comma list (default int8,pq128,tq2,tq4,f32)
//!   SKEG_FLAT_WARMUP       warmup queries    (default 32)

#![deny(unsafe_code)]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use skeg_vector::{FlatIndex, QuantKind};

fn env_usize(k: &str, default: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_list_usize(k: &str, default: &[usize]) -> Vec<usize> {
    std::env::var(k)
        .ok()
        .map(|v| {
            v.split(',')
                .filter_map(|x| x.trim().parse().ok())
                .collect::<Vec<_>>()
        })
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| default.to_vec())
}

fn env_list_string(k: &str, default: &[&str]) -> Vec<String> {
    std::env::var(k)
        .ok()
        .map(|v| {
            v.split(',')
                .map(|x| x.trim().to_owned())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
        })
        .filter(|v: &Vec<String>| !v.is_empty())
        .unwrap_or_else(|| default.iter().map(|&s| s.to_owned()).collect())
}

fn parse_tier(name: &str) -> Option<QuantKind> {
    match name {
        "int8" => Some(QuantKind::Int8),
        "pq128" => Some(QuantKind::Pq { m: 128, k: 256 }),
        "pq64" => Some(QuantKind::Pq { m: 64, k: 256 }),
        "pq32" => Some(QuantKind::Pq { m: 32, k: 256 }),
        "tq1" => Some(QuantKind::TurboQuant { bits: 1 }),
        "tq2" => Some(QuantKind::TurboQuant { bits: 2 }),
        "tq4" => Some(QuantKind::TurboQuant { bits: 4 }),
        "f32" => Some(QuantKind::F32),
        _ => None,
    }
}

/// Random vectors in `[-1, 1]^dim`. Seed-determined so runs of the
/// same config produce the same corpus. Distribution is uniform, not
/// gaussian: throughput is distribution-agnostic so this keeps the
/// dep surface to the workspace `rand` alone.
fn generate(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect())
        .collect()
}

/// Single-thread sequential search loop. Returns elapsed nanoseconds.
fn run_st(index: &Arc<Mutex<FlatIndex>>, queries: &[Vec<f32>], k: usize) -> u128 {
    let t0 = Instant::now();
    for q in queries {
        let _ = index.lock().expect("flat mutex").search(q, k);
    }
    t0.elapsed().as_nanos()
}

/// Multi-thread fan-out: spawn `threads` workers, distribute queries
/// round-robin. Each worker grabs the shared mutex per query. The
/// model mirrors a single-shard server under N concurrent clients.
fn run_mt(index: &Arc<Mutex<FlatIndex>>, queries: &[Vec<f32>], k: usize, threads: usize) -> u128 {
    let chunk = queries.len().div_ceil(threads);
    let t0 = Instant::now();
    std::thread::scope(|s| {
        for tid in 0..threads {
            let lo = tid * chunk;
            let hi = (lo + chunk).min(queries.len());
            if lo >= hi {
                continue;
            }
            let qs = &queries[lo..hi];
            let idx = Arc::clone(index);
            s.spawn(move || {
                for q in qs {
                    let _ = idx.lock().expect("flat mutex").search(q, k);
                }
            });
        }
    });
    t0.elapsed().as_nanos()
}

fn main() {
    let n = env_usize("SKEG_FLAT_N", 100_000);
    let dims = env_list_usize("SKEG_FLAT_DIMS", &[384, 768, 1024, 1536]);
    let n_queries = env_usize("SKEG_FLAT_QUERIES", 1000);
    let k = env_usize("SKEG_FLAT_K", 10);
    let thread_settings = env_list_usize("SKEG_FLAT_THREADS", &[1, 4, 8]);
    let tier_names = env_list_string("SKEG_FLAT_TIERS", &["int8", "pq128", "tq2", "tq4", "f32"]);
    let warmup = env_usize("SKEG_FLAT_WARMUP", 32);

    println!("# slice D step 1: flat-scan throughput, synthetic isotropic Gaussian");
    println!(
        "# config: N={n} dims={dims:?} k={k} threads={thread_settings:?} \
         tiers={tier_names:?} warmup={warmup} queries={n_queries}"
    );
    println!(
        "# host: target_os={} target_arch={} cores={}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0)
    );
    println!("tier,dim,n,threads,qps,p_avg_us,build_s,corpus_s");

    for &dim in &dims {
        let t_corpus = Instant::now();
        let corpus = generate(n, dim, 0xC0FFEE);
        let corpus_s = t_corpus.elapsed().as_secs_f64();
        let queries = generate(n_queries, dim, 0xBEEF);

        for name in &tier_names {
            let Some(kind) = parse_tier(name) else {
                eprintln!("# skipping unknown tier '{name}'");
                continue;
            };

            // Build the index from scratch; insert corpus.
            let t_build = Instant::now();
            let mut idx = FlatIndex::new(dim, kind);
            for (i, v) in corpus.iter().enumerate() {
                idx.insert(i as u64, v);
            }
            // Force quant init once outside the timed phase so the
            // first search does not bias the result.
            let _ = idx.search(&queries[0], k);
            // Warm up to land into hot cache state.
            for q in queries.iter().take(warmup) {
                let _ = idx.search(q, k);
            }
            let build_s = t_build.elapsed().as_secs_f64();
            let shared = Arc::new(Mutex::new(idx));

            for &threads in &thread_settings {
                let ns = if threads <= 1 {
                    run_st(&shared, &queries, k)
                } else {
                    run_mt(&shared, &queries, k, threads)
                };
                let elapsed_s = ns as f64 / 1e9;
                let qps = n_queries as f64 / elapsed_s;
                let p_avg_us = (ns as f64 / n_queries as f64) / 1e3;
                println!(
                    "{name},{dim},{n},{threads},{qps:.1},{p_avg_us:.2},{build_s:.2},{corpus_s:.2}"
                );
            }
        }
    }
}
