//! Distance-kernel benchmarks.
//!
//! Two layers:
//!   `kernel/*`     single-call latency, scalar vs hand-rolled NEON kernel
//!   `flat_scan/*`  stressful throughput: scan a large vector set vs a query,
//!                  the inner loop of a flat-scan VSEARCH, through the
//!                  dispatched public API (GB/s)

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use skeg_simd::{
    BLOCK, bucketize_x8, bucketize_x8_scalar, cosine_f32, cosine_f32_scalar, dot_f32_scalar,
    dot_int8, dot_int8_scalar, flip_signs, flip_signs_scalar, fwht_f32, fwht_f32_scalar,
    hamming_binary, hamming_binary_scalar, simd_backend, tq1_bitplane_score,
    tq1_bitplane_score_scalar, tq1_masked_sum, tq1_masked_sum_scalar, tq2_adc_i8, tq2_adc_qi8, tq4_adc_qi8,
    tq2_adc_i8_scalar, tq4_adc_i8, tq4_adc_i8_scalar, tq4_block32_score_u8,
    tq4_block32_score_u8_scalar,
};
#[cfg(target_arch = "x86_64")]
use skeg_simd::{
    bucketize_x8_avx2, cosine_f32_avx2, dot_f32_avx2, dot_int8_avx2, flip_signs_avx2,
    fwht_f32_avx2, hamming_binary_avx2, tq1_masked_sum_avx2, tq2_adc_i8_avx2, tq4_adc_i8_avx2,
    tq4_block32_score_u8_avx2,
};
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
use skeg_simd::{
    bucketize_x8_avx512, dot_int8_avx512, flip_signs_avx512, fwht_f32_avx512,
    hamming_binary_avx512, tq2_adc_i8_avx512, tq4_adc_i8_avx512, tq4_block32_score_u8_avx512,
};
#[cfg(target_arch = "aarch64")]
use skeg_simd::{
    bucketize_x8_neon, cosine_f32_neon, dot_int8_neon, flip_signs_neon, fwht_f32_neon,
    hamming_binary_neon, tq1_masked_sum_neon, tq2_adc_i8_neon, tq4_adc_i8_neon,
};

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
    let tq1_bytes = DIM / 8;
    let tq1_planes = fill(tq1_bytes * 3, 7);
    let tq1_code = fill(tq1_bytes, 8);

    let mut g = c.benchmark_group("kernel");
    g.throughput(Throughput::Elements(DIM as u64));

    g.bench_function("cosine_f32_scalar", |x| {
        x.iter(|| cosine_f32_scalar(black_box(&fa), black_box(&fb)));
    });
    g.bench_function("dot_f32_scalar", |x| {
        x.iter(|| dot_f32_scalar(black_box(&fa), black_box(&fb)));
    });
    g.bench_function("hamming_scalar", |x| {
        x.iter(|| hamming_binary_scalar(black_box(&ba), black_box(&bb)));
    });
    g.bench_function("dot_int8_scalar", |x| {
        x.iter(|| dot_int8_scalar(black_box(&ia), black_box(&ib)));
    });
    g.bench_function("tq1_bitplane_scalar", |x| {
        x.iter(|| {
            tq1_bitplane_score_scalar(black_box(&tq1_planes), 3, tq1_bytes, black_box(&tq1_code))
        });
    });
    g.bench_function("tq1_bitplane_dispatched", |x| {
        x.iter(|| tq1_bitplane_score(black_box(&tq1_planes), 3, tq1_bytes, black_box(&tq1_code)));
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
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            g.bench_function("hamming_avx2", |x| {
                x.iter(|| unsafe { hamming_binary_avx2(black_box(&ba), black_box(&bb)) });
            });
            g.bench_function("dot_int8_avx2", |x| {
                x.iter(|| unsafe { dot_int8_avx2(black_box(&ia), black_box(&ib)) });
            });
        }
        // Same pair, wider ISA: VPOPCNTDQ for hamming, VNNI for the int8 dot.
        #[cfg(feature = "avx512")]
        {
            if std::is_x86_feature_detected!("avx512f")
                && std::is_x86_feature_detected!("avx512bw")
                && std::is_x86_feature_detected!("avx512vpopcntdq")
            {
                g.bench_function("hamming_avx512", |x| {
                    x.iter(|| unsafe { hamming_binary_avx512(black_box(&ba), black_box(&bb)) });
                });
            }
            if std::is_x86_feature_detected!("avx512f")
                && std::is_x86_feature_detected!("avx512bw")
                && std::is_x86_feature_detected!("avx512vnni")
            {
                g.bench_function("dot_int8_avx512", |x| {
                    x.iter(|| unsafe { dot_int8_avx512(black_box(&ia), black_box(&ib)) });
                });
            }
        }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            g.bench_function("cosine_f32_avx2", |x| {
                x.iter(|| unsafe { cosine_f32_avx2(black_box(&fa), black_box(&fb)) });
            });
            g.bench_function("dot_f32_avx2", |x| {
                x.iter(|| unsafe { dot_f32_avx2(black_box(&fa), black_box(&fb)) });
            });
        }
    }
    g.finish();
}

// ── adc/* : per-query ADC single-call, scalar vs dispatched vs bare kernel ───
//
// The Vamana walk calls the ADC ~6400 times per
// query, so this is the primary per-query cost on the search path (unlike
// `tq4_block` below, which scores a whole 32-lane block against a
// precomputed LUT). `dispatched` goes through the public API the walk
// actually calls; the bare NEON/AVX2 entries isolate the kernel from
// dispatch overhead.

fn bench_adc(c: &mut Criterion) {
    let centroids: [i8; 16] = {
        let v = fill_i8(16, 60);
        let mut a = [0i8; 16];
        a.copy_from_slice(&v);
        a
    };
    let i8_scale = 0.01;
    let q_rot = fill_f32(DIM, 61);

    let mut g = c.benchmark_group("adc");
    g.throughput(Throughput::Elements(DIM as u64));

    let tq4_code = fill(DIM / 2, 62);
    g.bench_function("tq4_scalar", |x| {
        x.iter(|| {
            tq4_adc_i8_scalar(
                black_box(&tq4_code),
                black_box(&centroids),
                i8_scale,
                black_box(&q_rot),
                DIM,
            )
        });
    });
    g.bench_function("tq4_dispatched", |x| {
        x.iter(|| {
            tq4_adc_i8(
                black_box(&tq4_code),
                black_box(&centroids),
                i8_scale,
                black_box(&q_rot),
                DIM,
            )
        });
    });
    #[cfg(target_arch = "aarch64")]
    {
        g.bench_function("tq4_neon", |x| {
            x.iter(|| {
                tq4_adc_i8_neon(
                    black_box(&tq4_code),
                    black_box(&centroids),
                    i8_scale,
                    black_box(&q_rot),
                    DIM,
                )
            });
        });
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            g.bench_function("tq4_avx2", |x| {
                x.iter(|| unsafe {
                    tq4_adc_i8_avx2(
                        black_box(&tq4_code),
                        black_box(&centroids),
                        i8_scale,
                        black_box(&q_rot),
                        DIM,
                    )
                });
            });
        }
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        if std::is_x86_feature_detected!("avx512f") {
            g.bench_function("tq4_avx512", |x| {
                x.iter(|| unsafe {
                    tq4_adc_i8_avx512(
                        black_box(&tq4_code),
                        black_box(&centroids),
                        i8_scale,
                        black_box(&q_rot),
                        DIM,
                    )
                });
            });
        }
    }

    let tq2_code = fill(DIM / 4, 63);
    g.bench_function("tq2_scalar", |x| {
        x.iter(|| {
            tq2_adc_i8_scalar(
                black_box(&tq2_code),
                black_box(&centroids),
                i8_scale,
                black_box(&q_rot),
                DIM,
            )
        });
    });
    g.bench_function("tq2_dispatched", |x| {
        x.iter(|| {
            tq2_adc_i8(
                black_box(&tq2_code),
                black_box(&centroids),
                i8_scale,
                black_box(&q_rot),
                DIM,
            )
        });
    });
    #[cfg(target_arch = "aarch64")]
    {
        g.bench_function("tq2_neon", |x| {
            x.iter(|| {
                tq2_adc_i8_neon(
                    black_box(&tq2_code),
                    black_box(&centroids),
                    i8_scale,
                    black_box(&q_rot),
                    DIM,
                )
            });
        });
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            let q_i8: Vec<i8> = (0..DIM).map(|i| ((i * 13 % 255) as i16 - 127) as i8).collect();
            g.bench_function("tq2_qi8_sdot", |x| {
                x.iter(|| {
                    tq2_adc_qi8(
                        black_box(&tq2_code),
                        black_box(&centroids),
                        black_box(&q_i8),
                        DIM,
                    )
                });
            });
        }
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            let q_i8: Vec<i8> = (0..DIM).map(|i| ((i * 13 % 255) as i16 - 127) as i8).collect();
            g.bench_function("tq4_qi8_sdot", |x| {
                x.iter(|| {
                    tq4_adc_qi8(
                        black_box(&tq4_code),
                        black_box(&centroids),
                        black_box(&q_i8),
                        DIM,
                    )
                });
            });
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            g.bench_function("tq2_avx2", |x| {
                x.iter(|| unsafe {
                    tq2_adc_i8_avx2(
                        black_box(&tq2_code),
                        black_box(&centroids),
                        i8_scale,
                        black_box(&q_rot),
                        DIM,
                    )
                });
            });
        }
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        if std::is_x86_feature_detected!("avx512f") {
            g.bench_function("tq2_avx512", |x| {
                x.iter(|| unsafe {
                    tq2_adc_i8_avx512(
                        black_box(&tq2_code),
                        black_box(&centroids),
                        i8_scale,
                        black_box(&q_rot),
                        DIM,
                    )
                });
            });
        }
    }

    let tq1_code = fill(DIM / 8, 64);
    g.bench_function("tq1_scalar", |x| {
        x.iter(|| tq1_masked_sum_scalar(black_box(&tq1_code), black_box(&q_rot), DIM));
    });
    g.bench_function("tq1_dispatched", |x| {
        x.iter(|| tq1_masked_sum(black_box(&tq1_code), black_box(&q_rot), DIM));
    });
    #[cfg(target_arch = "aarch64")]
    {
        g.bench_function("tq1_neon", |x| {
            x.iter(|| tq1_masked_sum_neon(black_box(&tq1_code), black_box(&q_rot), DIM));
        });
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            g.bench_function("tq1_avx2", |x| {
                x.iter(|| unsafe {
                    tq1_masked_sum_avx2(black_box(&tq1_code), black_box(&q_rot), DIM)
                });
            });
        }
    }

    // Low dimension: at 104 coordinates the per-call setup (centroid widen,
    // table load) is a larger share of the work than the loop, and 104 is a
    // multiple of 8 but not of 16, so AVX2 runs tailless while AVX-512 does
    // not. Both effects are invisible at DIM.
    // Dimension sweep: AVX-512 carries a per-call fixed cost (four
    // accumulators plus a 16-lane horizontal reduction) that AVX2 does not.
    // It is invisible at 1536 and dominant at 104, so the crossover is a
    // number to measure rather than guess.
    for dim in [104usize, 128, 192, 256, 384] {
        let q = fill_f32(dim, 63);
        let code = fill(dim / 2, 64);
        let code2 = fill(dim / 4, 66);
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                g.bench_function(format!("tq4_d{dim}_avx2"), |x| {
                    x.iter(|| unsafe {
                        tq4_adc_i8_avx2(
                            black_box(&code),
                            black_box(&centroids),
                            i8_scale,
                            black_box(&q),
                            dim,
                        )
                    });
                });
            }
            g.bench_function(format!("tq2_d{dim}_avx2"), |x| {
                x.iter(|| unsafe {
                    tq2_adc_i8_avx2(
                        black_box(&code2),
                        black_box(&centroids),
                        i8_scale,
                        black_box(&q),
                        dim,
                    )
                });
            });
            #[cfg(feature = "avx512")]
            {
                if std::is_x86_feature_detected!("avx512f") {
                    g.bench_function(format!("tq4_d{dim}_avx512"), |x| {
                        x.iter(|| unsafe {
                            tq4_adc_i8_avx512(
                                black_box(&code),
                                black_box(&centroids),
                                i8_scale,
                                black_box(&q),
                                dim,
                            )
                        });
                    });
                    g.bench_function(format!("tq2_d{dim}_avx512"), |x| {
                        x.iter(|| unsafe {
                            tq2_adc_i8_avx512(
                                black_box(&code2),
                                black_box(&centroids),
                                i8_scale,
                                black_box(&q),
                                dim,
                            )
                        });
                    });
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = (&q, &code, &code2, dim);
    }

    g.finish();
}

// ── rotation/* : FastRotation's per-insert AND per-query kernels ─────────────
//
// fwht/flip_signs run in `FastRotation::apply` (rotate_query
// AND encode both call it), bucketize_x8 runs once per `encode`.

fn bench_rotation(c: &mut Criterion) {
    let mut g = c.benchmark_group("rotation");

    // fwht: a single 512-coord block (the largest power-of-two block
    // `FastRotation` ever splits DIM=1536 into: 1536 = 3 * 512).
    const BLOCK_DIM: usize = 512;
    g.throughput(Throughput::Elements(BLOCK_DIM as u64));
    let fwht_x = fill_f32(BLOCK_DIM, 70);
    g.bench_function("fwht_scalar", |x| {
        x.iter(|| {
            let mut v = fwht_x.clone();
            fwht_f32_scalar(black_box(&mut v));
            black_box(v)
        });
    });
    g.bench_function("fwht_dispatched", |x| {
        x.iter(|| {
            let mut v = fwht_x.clone();
            fwht_f32(black_box(&mut v));
            black_box(v)
        });
    });
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            g.bench_function("fwht_avx2", |x| {
                x.iter(|| {
                    let mut v = fwht_x.clone();
                    unsafe { fwht_f32_avx2(black_box(&mut v)) };
                    black_box(v)
                });
            });
        }
        #[cfg(feature = "avx512")]
        {
            if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx2") {
                g.bench_function("fwht_avx512", |x| {
                    x.iter(|| {
                        let mut v = fwht_x.clone();
                        unsafe { fwht_f32_avx512(black_box(&mut v)) };
                        black_box(v)
                    });
                });
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        g.bench_function("fwht_neon", |x| {
            x.iter(|| {
                let mut v = fwht_x.clone();
                fwht_f32_neon(black_box(&mut v));
                black_box(v)
            });
        });
    }

    // flip_signs and bucketize_x8 run once per DIM-length rotation/encode.
    g.throughput(Throughput::Elements(DIM as u64));
    let mut fs_x = fill_f32(DIM, 71);
    let fs_mask = fill(DIM.div_ceil(8), 72);
    g.bench_function("flip_signs_scalar", |x| {
        x.iter(|| flip_signs_scalar(black_box(&mut fs_x), black_box(&fs_mask)));
    });
    g.bench_function("flip_signs_dispatched", |x| {
        x.iter(|| flip_signs(black_box(&mut fs_x), black_box(&fs_mask)));
    });
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            g.bench_function("flip_signs_avx2", |x| {
                x.iter(|| unsafe { flip_signs_avx2(black_box(&mut fs_x), black_box(&fs_mask)) });
            });
        }
        #[cfg(feature = "avx512")]
        {
            if std::is_x86_feature_detected!("avx512f") {
                g.bench_function("flip_signs_avx512", |x| {
                    x.iter(|| unsafe {
                        flip_signs_avx512(black_box(&mut fs_x), black_box(&fs_mask));
                    });
                });
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        g.bench_function("flip_signs_neon", |x| {
            x.iter(|| flip_signs_neon(black_box(&mut fs_x), black_box(&fs_mask)));
        });
    }

    // bucketize_x8: 15 boundaries, matching the tq4 (4-bit) shape.
    let boundaries: Vec<f32> = (0i32..15).map(|i| (i as f32 - 7.0) / 3.0).collect();
    let bx = fill_f32(DIM, 73);
    let mut buckets = vec![0u8; DIM];
    g.bench_function("bucketize_x8_scalar", |x| {
        x.iter(|| {
            bucketize_x8_scalar(
                black_box(&bx),
                black_box(&boundaries),
                black_box(&mut buckets),
            );
        });
    });
    g.bench_function("bucketize_x8_dispatched", |x| {
        x.iter(|| {
            bucketize_x8(
                black_box(&bx),
                black_box(&boundaries),
                black_box(&mut buckets),
            );
        });
    });
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            g.bench_function("bucketize_x8_avx2", |x| {
                x.iter(|| unsafe {
                    bucketize_x8_avx2(
                        black_box(&bx),
                        black_box(&boundaries),
                        black_box(&mut buckets),
                    );
                });
            });
        }
        #[cfg(feature = "avx512")]
        {
            if std::is_x86_feature_detected!("avx512f") {
                g.bench_function("bucketize_x8_avx512", |x| {
                    x.iter(|| unsafe {
                        bucketize_x8_avx512(
                            black_box(&bx),
                            black_box(&boundaries),
                            black_box(&mut buckets),
                        );
                    });
                });
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        g.bench_function("bucketize_x8_neon", |x| {
            x.iter(|| {
                bucketize_x8_neon(
                    black_box(&bx),
                    black_box(&boundaries),
                    black_box(&mut buckets),
                );
            });
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

fn bench_tq4_block(c: &mut Criterion) {
    let n_groups = DIM / 2;
    let codes = fill(n_groups * BLOCK, 50);
    let lut: Vec<u8> = fill(n_groups * 32, 51)
        .into_iter()
        .map(|value| value & 0x7f)
        .collect();
    let mut scalar_out = [0.0; BLOCK];
    let mut dispatched_out = [0.0; BLOCK];
    let mut g = c.benchmark_group("tq4_block");
    g.throughput(Throughput::Elements((DIM * BLOCK) as u64));
    g.bench_function("scalar", |x| {
        x.iter(|| {
            tq4_block32_score_u8_scalar(
                black_box(&codes),
                black_box(&lut),
                0.013,
                -1.7,
                DIM,
                black_box(&mut scalar_out),
            );
            black_box(scalar_out)
        });
    });
    g.bench_function("dispatched", |x| {
        x.iter(|| {
            tq4_block32_score_u8(
                black_box(&codes),
                black_box(&lut),
                0.013,
                -1.7,
                DIM,
                black_box(&mut dispatched_out),
            );
            black_box(dispatched_out)
        });
    });
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            let mut avx2_out = [0.0; BLOCK];
            g.bench_function("avx2", |x| {
                x.iter(|| {
                    unsafe {
                        tq4_block32_score_u8_avx2(
                            black_box(&codes),
                            black_box(&lut),
                            0.013,
                            -1.7,
                            DIM,
                            black_box(&mut avx2_out),
                        );
                    }
                    black_box(avx2_out)
                });
            });
        }
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx2")
        {
            let mut avx512_out = [0.0; BLOCK];
            g.bench_function("avx512", |x| {
                x.iter(|| {
                    unsafe {
                        tq4_block32_score_u8_avx512(
                            black_box(&codes),
                            black_box(&lut),
                            0.013,
                            -1.7,
                            DIM,
                            black_box(&mut avx512_out),
                        );
                    }
                    black_box(avx512_out)
                });
            });
        }
    }
    g.finish();
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

criterion_group!(
    benches,
    bench_kernels,
    bench_adc,
    bench_rotation,
    bench_flat_scan,
    bench_tq4_block,
    bench_walk_pattern
);
criterion_main!(benches);
