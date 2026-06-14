//! Microbench: parallel pread + cosine rerank vs sequential.
//!
//! Emulates the DiskVamanaIndex rerank step on a synthetic `vectors.bin`
//! file. The corpus is generated once and written to a tempfile; the
//! bench picks a fixed candidate list per query and measures the two
//! rerank paths on identical inputs.
//!
//! Env knobs:
//!   SKEG_PIPE_N        corpus size              (default 100_000)
//!   SKEG_PIPE_DIM      vector dim               (default 1024)
//!   SKEG_PIPE_RERANK   candidate list per query (default 40)
//!   SKEG_PIPE_QUERIES  measurement reps         (default 200)
//!   SKEG_PIPE_WARMUP   warmup reps              (default 8)
//!   SKEG_PIPE_NOCACHE  set to 1 to open vectors.bin with F_NOCACHE
//!                      on macOS so every pread bypasses the page
//!                      cache (cold-perpetual simulation).

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::fs::{File, OpenOptions};
use std::hint::black_box;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::time::Instant;

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;
use skeg_simd::cosine_f32;
use tempfile::TempDir;

const HEADER_LEN: usize = 64;

fn env_usize(k: &str, default: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Build a synthetic `vectors.bin`-shaped file: 64 byte header then
/// `n * dim` f32 LE values.
fn build_vectors_file(path: &std::path::Path, n: usize, dim: usize, seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut f = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)
        .expect("open vectors.bin");
    f.write_all(&[0u8; HEADER_LEN]).expect("hdr");
    let mut buf = Vec::with_capacity(dim * 4);
    for _ in 0..n {
        buf.clear();
        let mut v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in &mut v {
            *x /= norm;
        }
        for x in &v {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        f.write_all(&buf).expect("write");
    }
    f.flush().expect("flush");
}

fn synth_query(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in &mut v {
        *x /= norm;
    }
    v
}

fn synth_candidates(n: usize, r: usize, seed: u64) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..r).map(|_| rng.random_range(0..n) as u32).collect()
}

fn read_vector(file: &File, id: u32, dim: usize) -> std::io::Result<Vec<f32>> {
    let offset = HEADER_LEN as u64 + u64::from(id) * dim as u64 * 4;
    let mut buf = vec![0u8; dim * 4];
    file.read_exact_at(&mut buf, offset)?;
    Ok(buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn run_sequential(
    file: &File,
    queries: &[Vec<f32>],
    candidates_per_query: &[Vec<u32>],
    dim: usize,
) -> (f64, f64) {
    let mut total = 0u64;
    let t0 = Instant::now();
    for (q, cands) in queries.iter().zip(candidates_per_query.iter()) {
        for &id in cands {
            let v = read_vector(file, id, dim).expect("read");
            let s = cosine_f32(q, &v);
            total = total.wrapping_add(s.to_bits() as u64);
        }
    }
    let elapsed_ns = t0.elapsed().as_nanos() as f64;
    black_box(total);
    let per_query_ns = elapsed_ns / queries.len() as f64;
    let per_read_ns = elapsed_ns / (queries.len() * candidates_per_query[0].len()) as f64;
    (per_query_ns, per_read_ns)
}

fn run_parallel(
    file: &File,
    queries: &[Vec<f32>],
    candidates_per_query: &[Vec<u32>],
    dim: usize,
    inter_query_sleep_us: u64,
) -> (f64, f64) {
    let mut total = 0u64;
    let t0 = Instant::now();
    for (q, cands) in queries.iter().zip(candidates_per_query.iter()) {
        let scores: Vec<f32> = cands
            .par_iter()
            .map(|&id| {
                let v = read_vector(file, id, dim).expect("read");
                cosine_f32(q, &v)
            })
            .collect();
        for s in scores {
            total = total.wrapping_add(s.to_bits() as u64);
        }
        if inter_query_sleep_us > 0 {
            std::thread::sleep(std::time::Duration::from_micros(inter_query_sleep_us));
        }
    }
    let elapsed_ns = t0.elapsed().as_nanos() as f64
        - (inter_query_sleep_us as f64 * 1000.0 * queries.len() as f64);
    black_box(total);
    let per_query_ns = elapsed_ns / queries.len() as f64;
    let per_read_ns = elapsed_ns / (queries.len() * candidates_per_query[0].len()) as f64;
    (per_query_ns, per_read_ns)
}

fn main() {
    let n = env_usize("SKEG_PIPE_N", 100_000);
    let dim = env_usize("SKEG_PIPE_DIM", 1024);
    let rerank = env_usize("SKEG_PIPE_RERANK", 40);
    let queries = env_usize("SKEG_PIPE_QUERIES", 200);
    let warmup = env_usize("SKEG_PIPE_WARMUP", 8);

    println!("# vamana rerank PIPE microbench");
    println!("# N={n} dim={dim} rerank={rerank} queries={queries} warmup={warmup}");
    println!(
        "# host: target_os={} target_arch={}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );

    let tmp = TempDir::new().expect("tmpdir");
    let path = tmp.path().join("vectors.bin");
    eprintln!("# building synthetic vectors.bin at {}", path.display());
    let t_build = Instant::now();
    build_vectors_file(&path, n, dim, 0xC0FFEE_u64);
    eprintln!("# build took {:.2}s", t_build.elapsed().as_secs_f64());
    let file = OpenOptions::new().read(true).open(&path).expect("open ro");

    let nocache = std::env::var("SKEG_PIPE_NOCACHE")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(0)
        != 0;
    #[cfg(target_os = "macos")]
    if nocache {
        // F_NOCACHE = 48 on Darwin. Sets non-caching IO on the fd so
        // every pread goes to NVMe instead of the page cache.
        // SAFETY: `file` is a valid open fd; F_NOCACHE takes a single
        // int arg by value and never reads memory through `file`.
        use std::os::fd::AsRawFd;
        let fd = file.as_raw_fd();
        let ret = unsafe {
            libc::fcntl(fd, 48 /* F_NOCACHE */, 1)
        };
        assert_eq!(
            ret,
            0,
            "fcntl F_NOCACHE failed: {}",
            std::io::Error::last_os_error()
        );
        eprintln!("# F_NOCACHE active (cold-perpetual mode)");
    }
    #[cfg(not(target_os = "macos"))]
    if nocache {
        eprintln!("# SKEG_PIPE_NOCACHE=1 ignored (not macOS)");
    }

    let q_warm: Vec<Vec<f32>> = (0..warmup)
        .map(|q| synth_query(dim, 0xF00D_u64.wrapping_add(q as u64)))
        .collect();
    let cands_warm: Vec<Vec<u32>> = (0..warmup)
        .map(|q| synth_candidates(n, rerank, 0xBEEF_u64.wrapping_add(q as u64)))
        .collect();
    let q_meas: Vec<Vec<f32>> = (0..queries)
        .map(|q| synth_query(dim, 0xCAFE_u64.wrapping_add(q as u64)))
        .collect();
    let cands_meas: Vec<Vec<u32>> = (0..queries)
        .map(|q| synth_candidates(n, rerank, 0xACE_u64.wrapping_add(q as u64)))
        .collect();

    // Warmups (discard timing).
    let _ = run_sequential(&file, &q_warm, &cands_warm, dim);
    let _ = run_parallel(&file, &q_warm, &cands_warm, dim, 0);

    println!("kernel,n,dim,rerank,per_query_ns,per_read_ns,sleep_us");
    let (seq_q, seq_r) = run_sequential(&file, &q_meas, &cands_meas, dim);
    println!("sequential,{n},{dim},{rerank},{seq_q:.1},{seq_r:.2},0");
    // Tight loop (status quo microbench: keeps rayon pool warm).
    let (par_q, par_r) = run_parallel(&file, &q_meas, &cands_meas, dim, 0);
    println!("parallel,{n},{dim},{rerank},{par_q:.1},{par_r:.2},0");
    // Sleep windows that mimic the server's inter-query gap.
    for sleep_us in [500u64, 1500, 5000] {
        let (par_q, par_r) = run_parallel(&file, &q_meas, &cands_meas, dim, sleep_us);
        println!("parallel,{n},{dim},{rerank},{par_q:.1},{par_r:.2},{sleep_us}");
    }
    let speedup = seq_q / par_q;
    eprintln!(
        "# rerank speedup parallel/sequential tight loop = {speedup:.2}x (seq {seq_q:.0} ns/query, par {par_q:.0} ns/query)"
    );
}
