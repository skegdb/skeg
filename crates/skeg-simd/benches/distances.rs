//! Distance-kernel benchmarks.
//!
//! Two layers:
//!   `kernel/*`     single-call latency, scalar vs hand-rolled NEON kernel
//!   `flat_scan/*`  stressful throughput: scan a large vector set vs a query,
//!                  the inner loop of a flat-scan VSEARCH, through the
//!                  dispatched public API (GB/s)

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use skeg_simd::{
    cosine_f32, cosine_f32_scalar, dot_int8, dot_int8_scalar, hamming_binary,
    hamming_binary_scalar, simd_backend,
};
#[cfg(target_arch = "aarch64")]
use skeg_simd::{cosine_f32_neon, dot_int8_neon, hamming_binary_neon};

const DIM: usize = 1536; // typical embedding dimension
const BIN_BYTES: usize = DIM / 8; // 1536-bit binary code = 192 bytes

/// Cheap deterministic pseudo-random fill (xorshift), reproducible.
fn fill(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed | 1;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        v.push((s & 0xFF) as u8);
    }
    v
}

fn fill_f32(n: usize, seed: u64) -> Vec<f32> {
    fill(n, seed)
        .into_iter()
        .map(|b| (f32::from(b) - 128.0) / 128.0) // roughly [-1, 1)
        .collect()
}

fn fill_i8(n: usize, seed: u64) -> Vec<i8> {
    fill(n, seed).into_iter().map(u8::cast_signed).collect()
}

// ── kernel/* : single-call, scalar vs NEON ───────────────────────────────────

fn bench_kernels(c: &mut Criterion) {
    eprintln!("simd backend: {}", simd_backend());

    let fa = fill_f32(DIM, 1);
    let fb = fill_f32(DIM, 2);
    let ba = fill(BIN_BYTES, 3);
    let bb = fill(BIN_BYTES, 4);
    let ia = fill_i8(DIM, 5);
    let ib = fill_i8(DIM, 6);

    let mut g = c.benchmark_group("kernel");
    g.throughput(Throughput::Elements(DIM as u64));

    g.bench_function("cosine_f32_scalar", |x| {
        x.iter(|| cosine_f32_scalar(black_box(&fa), black_box(&fb)));
    });
    g.bench_function("hamming_scalar", |x| {
        x.iter(|| hamming_binary_scalar(black_box(&ba), black_box(&bb)));
    });
    g.bench_function("dot_int8_scalar", |x| {
        x.iter(|| dot_int8_scalar(black_box(&ia), black_box(&ib)));
    });

    // NEON kernels measured directly (the public dispatch may route around
    // a kernel that loses to the auto-vectorized scalar, e.g. dot_int8).
    #[cfg(target_arch = "aarch64")]
    {
        g.bench_function("cosine_f32_neon", |x| {
            x.iter(|| cosine_f32_neon(black_box(&fa), black_box(&fb)));
        });
        g.bench_function("hamming_neon", |x| {
            x.iter(|| hamming_binary_neon(black_box(&ba), black_box(&bb)));
        });
        g.bench_function("dot_int8_neon", |x| {
            x.iter(|| dot_int8_neon(black_box(&ia), black_box(&ib)));
        });
    }
    g.finish();
}

// ── flat_scan/* : stressful throughput over large vector sets ────────────────
//
// These run through the dispatched public API (cosine_f32 / hamming_binary /
// dot_int8) - the path the vector tier actually takes.

fn bench_flat_scan(c: &mut Criterion) {
    // f32 cosine flat scan: 50K x 1536-dim = ~307 MB scanned per iteration.
    {
        const N: usize = 50_000;
        let query = fill_f32(DIM, 10);
        let db = fill_f32(N * DIM, 11);
        let mut g = c.benchmark_group("flat_scan");
        g.sample_size(20);
        g.throughput(Throughput::Bytes((N * DIM * 4) as u64));
        g.bench_function("cosine_f32_50k", |x| {
            x.iter(|| {
                let mut best = f32::MIN;
                for i in 0..N {
                    let v = &db[i * DIM..(i + 1) * DIM];
                    best = best.max(cosine_f32(black_box(&query), v));
                }
                black_box(best)
            });
        });
        g.finish();
    }

    // binary Hamming flat scan: 1M x 192-byte codes = ~192 MB scanned.
    {
        const N: usize = 1_000_000;
        let query = fill(BIN_BYTES, 20);
        let db = fill(N * BIN_BYTES, 21);
        let mut g = c.benchmark_group("flat_scan");
        g.sample_size(20);
        g.throughput(Throughput::Bytes((N * BIN_BYTES) as u64));
        g.bench_function("hamming_1m_binary", |x| {
            x.iter(|| {
                let mut best = u32::MAX;
                for i in 0..N {
                    let v = &db[i * BIN_BYTES..(i + 1) * BIN_BYTES];
                    best = best.min(hamming_binary(black_box(&query), v));
                }
                black_box(best)
            });
        });
        g.finish();
    }

    // int8 dot flat scan: 100K x 1536-dim = ~153 MB scanned.
    {
        const N: usize = 100_000;
        let query = fill_i8(DIM, 30);
        let db = fill_i8(N * DIM, 31);
        let mut g = c.benchmark_group("flat_scan");
        g.sample_size(20);
        g.throughput(Throughput::Bytes((N * DIM) as u64));
        g.bench_function("dot_int8_100k", |x| {
            x.iter(|| {
                let mut best = i32::MIN;
                for i in 0..N {
                    let v = &db[i * DIM..(i + 1) * DIM];
                    best = best.max(dot_int8(black_box(&query), v));
                }
                black_box(best)
            });
        });
        g.finish();
    }
}

// ── walk/* : scattered access of a greedy graph walk ─────────────────────────
//
// A Vamana greedy walk touches ~1280 nodes scattered across the dataset - not
// the sequential stream of a flat scan. This measures the per-distance cost
// under that scattered access. It is the gate for the NEON-int8 /
// RaBitQ workaround: M1's int8 prefilter lost because the int8 kernel was no
// faster than f32 cosine; before committing to RaBitQ this confirms the
// Hamming kernel keeps its single-call speed in the walk's random-access
// pattern (not only in a sequential flat scan).

fn bench_walk_pattern(c: &mut Criterion) {
    const DIM_1024: usize = 1024; // mxbai-embed-large dimension
    const BIN_1024: usize = DIM_1024 / 8; // 128-byte binary / RaBitQ code
    const POOL: usize = 50_000;
    const WALK: usize = 1280; // distance computations in one greedy walk

    // Scattered access indices, deterministic (xorshift).
    let idx: Vec<usize> = {
        let mut s: u64 = 0x9E37_79B9;
        (0..WALK)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s as usize) % POOL
            })
            .collect()
    };

    let bin_query = fill(BIN_1024, 40);
    let bin_pool = fill(POOL * BIN_1024, 41);
    let f_query = fill_f32(DIM_1024, 42);
    let f_pool = fill_f32(POOL * DIM_1024, 43);

    let mut g = c.benchmark_group("walk");
    g.throughput(Throughput::Elements(WALK as u64));

    g.bench_function("hamming_1024_x1280", |x| {
        x.iter(|| {
            let mut sum = 0u32;
            for &i in &idx {
                let v = &bin_pool[i * BIN_1024..(i + 1) * BIN_1024];
                sum = sum.wrapping_add(hamming_binary(black_box(&bin_query), v));
            }
            black_box(sum)
        });
    });
    g.bench_function("cosine_f32_1024_x1280", |x| {
        x.iter(|| {
            let mut sum = 0.0f32;
            for &i in &idx {
                let v = &f_pool[i * DIM_1024..(i + 1) * DIM_1024];
                sum += cosine_f32(black_box(&f_query), v);
            }
            black_box(sum)
        });
    });
    g.finish();
}

criterion_group!(benches, bench_kernels, bench_flat_scan, bench_walk_pattern);
criterion_main!(benches);
