//! Build-optimization gate - where does `VamanaIndex::build` spend time, and
//! how well does it already parallelize?
//!
//! Not a Criterion bench: a reporting harness (`harness = false`).
//!
//! A planned batch + nav-graph build plus a parallel build promise a 20-30x
//! build speedup. The build is *already* parallel (rayon). This harness
//! measures, before committing weeks to that work:
//!   - thread scaling: build the same dataset at 1/2/4/8 threads -> is the
//!     parallelism already efficient, or is there contention to fix?
//!   - phase split: greedy walk vs robust-prune vs back-edge (the batch work
//!     targets the walk - shared beam, nav graph - so its leverage depends on
//!     the walk's share).
//!   - the real build time at 1M.

#![allow(clippy::cast_precision_loss)]

use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::ThreadPoolBuilder;
use skeg_vector::{VamanaConfig, VamanaIndex, build_phase_times_ns, reset_build_phase_times};

const DIM: usize = 1024;

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

fn build_secs(vectors: Vec<f32>, n: usize) -> f64 {
    let ids: Vec<u64> = (0..n as u64).collect();
    let t = Instant::now();
    let index = VamanaIndex::build(vectors, ids, DIM, &VamanaConfig::default());
    let s = t.elapsed().as_secs_f64();
    std::hint::black_box(&index);
    s
}

fn main() {
    eprintln!("VamanaIndex::build profile - thread scaling + phase split, dim={DIM}\n");

    // ── thread scaling ────────────────────────────────────────────────────
    let scale_n = 50_000;
    let scale_data = clustered(scale_n, DIM, 1);
    println!("== thread scaling (build N={scale_n}) ==");
    println!(
        "  {:>8}{:>12}{:>12}{:>14}",
        "threads", "build s", "speedup", "efficiency"
    );
    let mut base = 0.0;
    for (i, &t) in [1usize, 2, 4, 8].iter().enumerate() {
        let pool = ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .expect("pool");
        let data = scale_data.clone();
        let s = pool.install(|| build_secs(data, scale_n));
        if i == 0 {
            base = s;
        }
        let speedup = base / s;
        println!(
            "  {:>8}{:>12.1}{:>11.2}x{:>13.0}%",
            t,
            s,
            speedup,
            speedup / t as f64 * 100.0
        );
    }

    // ── phase split (full threads, N=100K) ────────────────────────────────
    let phase_n = 100_000;
    let data = clustered(phase_n, DIM, 2);
    reset_build_phase_times();
    let total = build_secs(data, phase_n);
    let (walk, prune, back) = build_phase_times_ns();
    // walk/prune/back are nanoseconds summed across worker threads.
    let sum = (walk + prune + back).max(1) as f64;
    println!("\n== build phase split (N={phase_n}, all threads, wall {total:.1}s) ==");
    println!("  {:>12}{:>16}{:>12}", "phase", "thread-CPU s", "share");
    let row = |label: &str, ns: u64| {
        println!(
            "  {:>12}{:>16.1}{:>11.1}%",
            label,
            ns as f64 / 1e9,
            ns as f64 / sum * 100.0,
        );
    };
    row("greedy walk", walk);
    row("robust prune", prune);
    row("back-edge", back);
    println!("  (medoid + patch_connectivity not separately timed; known small)");

    // ── real build time at 1M ─────────────────────────────────────────────
    let m = 1_000_000;
    println!("\n== build at N={m} (all threads) ==");
    let data = clustered(m, DIM, 3);
    let s = build_secs(data, m);
    println!("  build: {s:.1}s ({:.1} min)", s / 60.0);
}
