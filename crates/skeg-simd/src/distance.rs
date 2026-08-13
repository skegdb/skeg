//! Distance kernels: cosine and dot over f32, dot over i8, and Hamming over
//! packed binary codes. Every kernel keeps its scalar reference next to its
//! NEON, AVX2 and AVX-512 counterparts, because a technique fix lands on all
//! of them at once.

/// Cosine similarity of two equal-length f32 slices. Result in `[-1.0, 1.0]`.
#[must_use]
pub fn cosine_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 { 0.0 } else { dot / denom }
}

/// Hamming distance (popcount of XOR) of two equal-length byte slices.
#[must_use]
pub fn hamming_binary_scalar(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Dot product of two equal-length i8 slices, accumulated in i32.
#[must_use]
pub fn dot_int8_scalar(a: &[i8], b: &[i8]) -> i32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| i32::from(*x) * i32::from(*y))
        .sum()
}

/// Cosine similarity, NEON. 16 f32 per iteration over 4 independent
/// accumulator groups to hide FMA latency (the f32 reduction is not
/// auto-vectorized because float addition is non-associative).
#[cfg(target_arch = "aarch64")]
#[must_use]
pub fn cosine_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    // The loop bound comes from `a`, the loads come from both: a shorter `b`
    // would be read past its end. The dispatcher checks this; a caller reaching
    // the kernel directly does not go through the dispatcher.
    assert_eq!(a.len(), b.len(), "dimension mismatch");
    use std::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};
    let n = a.len();
    let block = n - (n % 16);
    let mut i = 0;
    // SAFETY: each `vld1q_f32` reads 4 f32 at offset `i + k*4 < block <= n`, in
    // bounds for both slices (caller guarantees `a.len() == b.len()`). Reads
    // only, no aliasing. NEON is baseline on aarch64.
    let (mut sdot, mut sna, mut snb) = unsafe {
        let mut dot = [vdupq_n_f32(0.0); 4];
        let mut na = [vdupq_n_f32(0.0); 4];
        let mut nb = [vdupq_n_f32(0.0); 4];
        while i < block {
            for k in 0..4 {
                let va = vld1q_f32(a.as_ptr().add(i + k * 4));
                let vb = vld1q_f32(b.as_ptr().add(i + k * 4));
                dot[k] = vfmaq_f32(dot[k], va, vb);
                na[k] = vfmaq_f32(na[k], va, va);
                nb[k] = vfmaq_f32(nb[k], vb, vb);
            }
            i += 16;
        }
        let dot = vaddq_f32(vaddq_f32(dot[0], dot[1]), vaddq_f32(dot[2], dot[3]));
        let na = vaddq_f32(vaddq_f32(na[0], na[1]), vaddq_f32(na[2], na[3]));
        let nb = vaddq_f32(vaddq_f32(nb[0], nb[1]), vaddq_f32(nb[2], nb[3]));
        (vaddvq_f32(dot), vaddvq_f32(na), vaddvq_f32(nb))
    };
    for i in i..n {
        sdot += a[i] * b[i];
        sna += a[i] * a[i];
        snb += b[i] * b[i];
    }
    let denom = sna.sqrt() * snb.sqrt();
    if denom == 0.0 { 0.0 } else { sdot / denom }
}

/// Cosine similarity, AVX2 plus FMA. Four independent accumulator groups
/// consume 32 f32 values per iteration, avoiding a dependency chain through
/// the FMA latency. The caller must check CPU support before calling it.
///
/// # Safety
///
/// The current CPU must support AVX2 and FMA, and both slices must have the
/// same length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,sse3")]
#[must_use]
pub unsafe fn cosine_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    // The loop bound comes from `a`, the loads come from both: a shorter `b`
    // would be read past its end. The dispatcher checks this; a caller reaching
    // the kernel directly does not go through the dispatcher.
    assert_eq!(a.len(), b.len(), "dimension mismatch");
    use std::arch::x86_64::{
        _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_movehdup_ps, _mm_movehl_ps, _mm256_add_ps,
        _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_loadu_ps,
        _mm256_setzero_ps,
    };

    let n = a.len();
    let block = n - (n % 32);
    let mut i = 0;
    // SAFETY: the caller selected AVX2 and FMA at runtime. Each load reads 8
    // f32 values at `i + k*8 < block <= a.len() == b.len()`. The slices are
    // read-only and unaligned loads accept every valid slice alignment.
    let (mut sdot, mut sna, mut snb) = unsafe {
        let mut dot = [_mm256_setzero_ps(); 4];
        let mut na = [_mm256_setzero_ps(); 4];
        let mut nb = [_mm256_setzero_ps(); 4];
        while i < block {
            for k in 0..4 {
                let va = _mm256_loadu_ps(a.as_ptr().add(i + k * 8));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i + k * 8));
                dot[k] = _mm256_fmadd_ps(va, vb, dot[k]);
                na[k] = _mm256_fmadd_ps(va, va, na[k]);
                nb[k] = _mm256_fmadd_ps(vb, vb, nb[k]);
            }
            i += 32;
        }

        let reduce = |acc: [std::arch::x86_64::__m256; 4]| {
            let sum = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
            let sum = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
            let sum = _mm_add_ps(sum, _mm_movehdup_ps(sum));
            _mm_cvtss_f32(_mm_add_ss(sum, _mm_movehl_ps(sum, sum)))
        };
        (reduce(dot), reduce(na), reduce(nb))
    };
    for i in i..n {
        sdot += a[i] * b[i];
        sna += a[i] * a[i];
        snb += b[i] * b[i];
    }
    let denom = sna.sqrt() * snb.sqrt();
    if denom == 0.0 { 0.0 } else { sdot / denom }
}

/// Hamming distance, NEON. 64 bytes per iteration: `vcntq_u8` per-byte
/// popcount, widening-accumulated into u16 lanes so the horizontal reduction
/// runs once at the end rather than once per iteration.
#[cfg(target_arch = "aarch64")]
#[must_use]
pub fn hamming_binary_neon(a: &[u8], b: &[u8]) -> u32 {
    // The loop bound comes from `a`, the loads come from both: a shorter `b`
    // would be read past its end. The dispatcher checks this; a caller reaching
    // the kernel directly does not go through the dispatcher.
    assert_eq!(a.len(), b.len(), "dimension mismatch");
    use std::arch::aarch64::{
        vaddlvq_u16, vaddq_u16, vcntq_u8, vdupq_n_u16, veorq_u8, vld1q_u8, vpadalq_u8,
    };
    let n = a.len();
    let block = n - (n % 64);
    let mut i = 0;
    // SAFETY: each `vld1q_u8` reads 16 bytes at offset `i + k*16 < block <= n`,
    // in bounds for both slices (`a.len() == b.len()`). Reads only. Each u16
    // lane gains at most 32 per 64-byte iteration, so it cannot overflow for
    // inputs below ~256 KiB (binary codes are far smaller).
    let mut sum: u32 = unsafe {
        let mut acc = [vdupq_n_u16(0); 4];
        while i < block {
            for (k, acc_k) in acc.iter_mut().enumerate() {
                let va = vld1q_u8(a.as_ptr().add(i + k * 16));
                let vb = vld1q_u8(b.as_ptr().add(i + k * 16));
                *acc_k = vpadalq_u8(*acc_k, vcntq_u8(veorq_u8(va, vb)));
            }
            i += 64;
        }
        let acc = vaddq_u16(vaddq_u16(acc[0], acc[1]), vaddq_u16(acc[2], acc[3]));
        vaddlvq_u16(acc)
    };
    for i in i..n {
        sum += (a[i] ^ b[i]).count_ones();
    }
    sum
}

/// Count set bits after either XOR or AND, 64 bytes per iteration with AVX2.
/// `vpshufb` performs a nibble lookup in each 128-bit lane and `vpsadbw`
/// reduces byte counts into four independent u64 lanes.
///
/// # Safety
///
/// The current CPU must support AVX2, and both slices must have the same
/// length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn byte_popcount_avx2(a: &[u8], b: &[u8], and: bool) -> u32 {
    use std::arch::x86_64::{
        _mm256_add_epi8, _mm256_add_epi64, _mm256_and_si256, _mm256_loadu_si256, _mm256_sad_epu8,
        _mm256_set1_epi8, _mm256_setr_epi8, _mm256_setzero_si256, _mm256_shuffle_epi8,
        _mm256_srli_epi16, _mm256_storeu_si256, _mm256_xor_si256,
    };

    let block = a.len() - (a.len() % 64);
    let mut i = 0;
    // SAFETY: the caller selected AVX2. The two loads at `i + k * 32` are
    // bounded by `block <= a.len() == b.len()`, and all stores target the
    // local four-lane reduction buffer.
    let total = unsafe {
        let nibble_counts = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2,
            3, 3, 4,
        );
        let mask = _mm256_set1_epi8(0x0f);
        let zero = _mm256_setzero_si256();
        let mut acc = _mm256_setzero_si256();
        while i < block {
            for k in 0..2 {
                let va = _mm256_loadu_si256(a.as_ptr().add(i + k * 32).cast());
                let vb = _mm256_loadu_si256(b.as_ptr().add(i + k * 32).cast());
                let combined = if and {
                    _mm256_and_si256(va, vb)
                } else {
                    _mm256_xor_si256(va, vb)
                };
                let low = _mm256_and_si256(combined, mask);
                let high = _mm256_and_si256(_mm256_srli_epi16(combined, 4), mask);
                let counts = _mm256_add_epi8(
                    _mm256_shuffle_epi8(nibble_counts, low),
                    _mm256_shuffle_epi8(nibble_counts, high),
                );
                acc = _mm256_add_epi64(acc, _mm256_sad_epu8(counts, zero));
            }
            i += 64;
        }
        let mut lanes = [0u64; 4];
        _mm256_storeu_si256(lanes.as_mut_ptr().cast(), acc);
        lanes.into_iter().sum::<u64>()
    };
    let tail: u64 = a[i..]
        .iter()
        .zip(&b[i..])
        .map(|(&x, &y)| {
            u64::from(if and {
                (x & y).count_ones()
            } else {
                (x ^ y).count_ones()
            })
        })
        .sum();
    (total + tail) as u32
}

/// Hamming distance (popcount of XOR), AVX2.
///
/// # Safety
///
/// The current CPU must support AVX2, and both slices must have the same
/// length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[must_use]
pub unsafe fn hamming_binary_avx2(a: &[u8], b: &[u8]) -> u32 {
    // `byte_popcount_*` bounds itself by `a.len()` and loads from both.
    assert_eq!(a.len(), b.len(), "dimension mismatch");
    // SAFETY: forwarded from this function's contract.
    unsafe { byte_popcount_avx2(a, b, false) }
}

/// `popcount(a AND b)`, AVX2. Used by the TQ1 bit-plane proxy.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[must_use]
pub(crate) unsafe fn and_popcnt_avx2(a: &[u8], b: &[u8]) -> u32 {
    // SAFETY: forwarded from this function's contract.
    unsafe { byte_popcount_avx2(a, b, true) }
}

/// Count set bits after either XOR or AND, 64 bytes per iteration with
/// AVX-512. `VPOPCNTDQ` popcounts each 64-bit lane directly - no nibble-LUT
/// shuffle trick needed, since this ISA adds the instruction AVX2 lacks
/// instead of just wider registers for the same trick.
///
/// # Safety
///
/// The current CPU must support AVX-512F/BW/VPOPCNTDQ, and both slices must
/// have the same length.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vpopcntdq")]
unsafe fn byte_popcount_avx512(a: &[u8], b: &[u8], and: bool) -> u32 {
    use std::arch::x86_64::{
        _mm512_add_epi64, _mm512_and_si512, _mm512_loadu_si512, _mm512_popcnt_epi64,
        _mm512_setzero_si512, _mm512_storeu_si512, _mm512_xor_si512,
    };

    let block = a.len() - (a.len() % 64);
    let mut i = 0;
    // SAFETY: the caller selected AVX-512F/BW/VPOPCNTDQ. Each load reads 64
    // bytes at `i < block <= a.len() == b.len()`. `lanes` is a
    // fully-initialised stack array read back immediately.
    let total = unsafe {
        let mut acc = _mm512_setzero_si512();
        while i < block {
            let va = _mm512_loadu_si512(a.as_ptr().add(i).cast());
            let vb = _mm512_loadu_si512(b.as_ptr().add(i).cast());
            let combined = if and {
                _mm512_and_si512(va, vb)
            } else {
                _mm512_xor_si512(va, vb)
            };
            // Popcount of each 64-bit (8-byte) lane; summing all 8 lanes
            // gives the popcount of the whole 64-byte register.
            acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(combined));
            i += 64;
        }
        let mut lanes = [0u64; 8];
        _mm512_storeu_si512(lanes.as_mut_ptr().cast(), acc);
        lanes.into_iter().sum::<u64>()
    };
    let tail: u64 = a[i..]
        .iter()
        .zip(&b[i..])
        .map(|(&x, &y)| {
            u64::from(if and {
                (x & y).count_ones()
            } else {
                (x ^ y).count_ones()
            })
        })
        .sum();
    (total + tail) as u32
}

/// Hamming distance (popcount of XOR), AVX-512.
///
/// # Safety
///
/// The current CPU must support AVX-512F/BW/VPOPCNTDQ, and both slices must
/// have the same length.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vpopcntdq")]
#[must_use]
pub unsafe fn hamming_binary_avx512(a: &[u8], b: &[u8]) -> u32 {
    // `byte_popcount_*` bounds itself by `a.len()` and loads from both.
    assert_eq!(a.len(), b.len(), "dimension mismatch");
    // SAFETY: forwarded from this function's contract.
    unsafe { byte_popcount_avx512(a, b, false) }
}

/// `popcount(a AND b)`, AVX-512. Used by the TQ1 bit-plane proxy.
///
/// # Safety
///
/// The current CPU must support AVX-512F/BW/VPOPCNTDQ, and both slices must
/// have the same length.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vpopcntdq")]
pub(crate) unsafe fn and_popcnt_avx512(a: &[u8], b: &[u8]) -> u32 {
    // SAFETY: forwarded from this function's contract.
    unsafe { byte_popcount_avx512(a, b, true) }
}

/// `popcount(a AND b)` over equal-length byte slices, NEON. Same shape as
/// [`hamming_binary_neon`] with `vandq_u8` instead of `veorq_u8`.
#[cfg(target_arch = "aarch64")]
#[must_use]
pub(crate) fn and_popcnt_neon(a: &[u8], b: &[u8]) -> u32 {
    use std::arch::aarch64::{
        vaddlvq_u16, vaddq_u16, vandq_u8, vcntq_u8, vdupq_n_u16, vld1q_u8, vpadalq_u8,
    };
    let n = a.len();
    let block = n - (n % 64);
    let mut i = 0;
    // SAFETY: each `vld1q_u8` reads 16 bytes at `i + k*16 < block <= n`, in
    // bounds for both slices (`a.len() == b.len()`, asserted by the caller).
    // Read-only. Each u16 lane gains <= 32 per 64-byte iter, no overflow for
    // code sizes far below 256 KiB.
    let mut sum: u32 = unsafe {
        let mut acc = [vdupq_n_u16(0); 4];
        while i < block {
            for (k, acc_k) in acc.iter_mut().enumerate() {
                let va = vld1q_u8(a.as_ptr().add(i + k * 16));
                let vb = vld1q_u8(b.as_ptr().add(i + k * 16));
                *acc_k = vpadalq_u8(*acc_k, vcntq_u8(vandq_u8(va, vb)));
            }
            i += 64;
        }
        let acc = vaddq_u16(vaddq_u16(acc[0], acc[1]), vaddq_u16(acc[2], acc[3]));
        vaddlvq_u16(acc)
    };
    for i in i..n {
        sum += (a[i] & b[i]).count_ones();
    }
    sum
}

/// Dot product of i8 slices, NEON. 64 i8 per iteration over 4 independent
/// accumulators to hide `vpadalq` latency.
///
/// Uses baseline NEON (`vmull_s8` widening multiply + `vpadalq_s16` pairwise
/// accumulate). The `dotprod` extension (`vdotq_s32`, one instruction instead
/// of three) is still unstable in `std::arch`; switch to it once stabilized.
#[cfg(target_arch = "aarch64")]
#[must_use]
pub fn dot_int8_neon(a: &[i8], b: &[i8]) -> i32 {
    // The loop bound comes from `a`, the loads come from both: a shorter `b`
    // would be read past its end. The dispatcher checks this; a caller reaching
    // the kernel directly does not go through the dispatcher.
    assert_eq!(a.len(), b.len(), "dimension mismatch");
    use std::arch::aarch64::{
        vaddq_s32, vaddvq_s32, vdupq_n_s32, vget_high_s8, vget_low_s8, vld1q_s8, vmull_s8,
        vpadalq_s16,
    };
    let n = a.len();
    let block = n - (n % 64);
    let mut i = 0;
    // SAFETY: each `vld1q_s8` reads 16 i8 at offset `i + k*16 < block <= n`, in
    // bounds for both slices (`a.len() == b.len()`). Reads only. NEON is
    // baseline on aarch64. `vmull_s8` cannot overflow i16 (127*127 = 16129).
    let mut sum = unsafe {
        let mut acc = [vdupq_n_s32(0); 4];
        while i < block {
            for (k, acc_k) in acc.iter_mut().enumerate() {
                let va = vld1q_s8(a.as_ptr().add(i + k * 16));
                let vb = vld1q_s8(b.as_ptr().add(i + k * 16));
                *acc_k = vpadalq_s16(*acc_k, vmull_s8(vget_low_s8(va), vget_low_s8(vb)));
                *acc_k = vpadalq_s16(*acc_k, vmull_s8(vget_high_s8(va), vget_high_s8(vb)));
            }
            i += 64;
        }
        let acc = vaddq_s32(vaddq_s32(acc[0], acc[1]), vaddq_s32(acc[2], acc[3]));
        vaddvq_s32(acc)
    };
    for i in i..n {
        sum += i32::from(a[i]) * i32::from(b[i]);
    }
    sum
}

/// Dot product of i8 slices, AVX2. Widened i16 products are pairwise summed
/// into i32 lanes before reduction, so every intermediate is exact.
///
/// # Safety
///
/// The current CPU must support AVX2, and both slices must have the same
/// length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[must_use]
pub unsafe fn dot_int8_avx2(a: &[i8], b: &[i8]) -> i32 {
    // The loop bound comes from `a`, the loads come from both: a shorter `b`
    // would be read past its end. The dispatcher checks this; a caller reaching
    // the kernel directly does not go through the dispatcher.
    assert_eq!(a.len(), b.len(), "dimension mismatch");
    use std::arch::x86_64::{
        _mm256_add_epi32, _mm256_castsi256_si128, _mm256_cvtepi8_epi16, _mm256_extracti128_si256,
        _mm256_loadu_si256, _mm256_madd_epi16, _mm256_mullo_epi16, _mm256_set1_epi16,
        _mm256_setzero_si256, _mm256_storeu_si256,
    };

    let block = a.len() - (a.len() % 64);
    let mut i = 0;
    // SAFETY: the caller selected AVX2. Each 256-bit load reads 32 i8 values
    // at `i + k*32 < block <= a.len() == b.len()`. Products fit in i16 and
    // `_mm256_madd_epi16` widens their pair sums to i32.
    let reduced = unsafe {
        let ones = _mm256_set1_epi16(1);
        let mut acc = [_mm256_setzero_si256(); 2];
        while i < block {
            for (k, slot) in acc.iter_mut().enumerate() {
                let va = _mm256_loadu_si256(a.as_ptr().add(i + k * 32).cast());
                let vb = _mm256_loadu_si256(b.as_ptr().add(i + k * 32).cast());
                let products = |va, vb| _mm256_madd_epi16(_mm256_mullo_epi16(va, vb), ones);
                let low = products(
                    _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va)),
                    _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb)),
                );
                let high = products(
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1)),
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1)),
                );
                *slot = _mm256_add_epi32(*slot, _mm256_add_epi32(low, high));
            }
            i += 64;
        }
        let sum = _mm256_add_epi32(acc[0], acc[1]);
        let mut lanes = [0i32; 8];
        _mm256_storeu_si256(lanes.as_mut_ptr().cast(), sum);
        lanes.into_iter().sum()
    };
    let mut sum = reduced;
    for i in i..a.len() {
        sum += i32::from(a[i]) * i32::from(b[i]);
    }
    sum
}

/// Dot product of two i8 slices, AVX-512 VNNI. `_mm512_dpbusd_epi32` computes
/// four-byte-group `unsigned * signed -> i32` dot products in one
/// instruction - the ISA gives us an actual multiply-accumulate instruction
/// here, not just wider registers for AVX2's widen-then-`vpmaddwd` trick.
///
/// `dpbusd` requires its first byte operand unsigned and its second signed,
/// but `dot_int8` has two *signed* operands. Standard fixup: flipping the
/// sign bit of every byte of `a` (`XOR 0x80`) reinterprets it as the unsigned
/// value `u8 = i8 + 128` (mod 256) without changing any bits `dpbusd` reads
/// differently; then `i8 = u8 - 128`, so:
/// `dot(a, b) = sum(a_i8 * b_i8) = sum((u8_a - 128) * b_i8)
///            = dpbusd(u8_a, b) - 128 * sum(b_i8)`
/// `sum(b_i8)` is itself obtained via the same instruction, multiplying `b`
/// by an all-ones unsigned operand.
///
/// # Safety
///
/// The current CPU must support AVX-512F/BW/VNNI, and both slices must have
/// the same length.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
#[must_use]
pub unsafe fn dot_int8_avx512(a: &[i8], b: &[i8]) -> i32 {
    // The loop bound comes from `a`, the loads come from both: a shorter `b`
    // would be read past its end. The dispatcher checks this; a caller reaching
    // the kernel directly does not go through the dispatcher.
    assert_eq!(a.len(), b.len(), "dimension mismatch");
    use std::arch::x86_64::{
        _mm512_dpbusd_epi32, _mm512_loadu_si512, _mm512_set1_epi8, _mm512_setzero_si512,
        _mm512_storeu_si512, _mm512_xor_si512,
    };

    let block = a.len() - (a.len() % 64);
    let mut i = 0;
    // SAFETY: the caller selected AVX-512F/BW/VNNI. Each 512-bit load reads
    // 64 i8 values at `i < block <= a.len() == b.len()`. `dpbusd` widens its
    // four-byte-group products to i32 with no overflow risk at this input
    // size (i8 range keeps each product well within i32).
    let (dot_reduced, bsum_reduced) = unsafe {
        let sign_flip = _mm512_set1_epi8(-128); // 0x80 in every byte lane
        let ones = _mm512_set1_epi8(1);
        let mut dot_acc = _mm512_setzero_si512();
        let mut bsum_acc = _mm512_setzero_si512();
        while i < block {
            let va = _mm512_loadu_si512(a.as_ptr().add(i).cast());
            let vb = _mm512_loadu_si512(b.as_ptr().add(i).cast());
            let va_unsigned = _mm512_xor_si512(va, sign_flip);
            dot_acc = _mm512_dpbusd_epi32(dot_acc, va_unsigned, vb);
            bsum_acc = _mm512_dpbusd_epi32(bsum_acc, ones, vb);
            i += 64;
        }
        let mut dot_lanes = [0i32; 16];
        let mut bsum_lanes = [0i32; 16];
        _mm512_storeu_si512(dot_lanes.as_mut_ptr().cast(), dot_acc);
        _mm512_storeu_si512(bsum_lanes.as_mut_ptr().cast(), bsum_acc);
        (
            dot_lanes.into_iter().sum::<i32>(),
            bsum_lanes.into_iter().sum::<i32>(),
        )
    };
    let mut sum = dot_reduced - 128 * bsum_reduced;
    for i in i..a.len() {
        sum += i32::from(a[i]) * i32::from(b[i]);
    }
    sum
}

/// Cosine similarity of two f32 slices. Result in `[-1.0, 1.0]`.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
#[must_use]
pub fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dimension mismatch");
    #[cfg(target_arch = "aarch64")]
    {
        cosine_f32_neon(a, b)
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: both target features were checked immediately above.
            return unsafe { cosine_f32_avx2(a, b) };
        }
        cosine_f32_scalar(a, b)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        cosine_f32_scalar(a, b)
    }
}

/// Dot product of two f32 slices. Scalar reference; `.sum()` does not
/// auto-vectorize (f32 addition is non-associative).
#[must_use]
pub fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Dot product, NEON. 16 f32 per iteration over 4 independent accumulator groups
/// to hide FMA latency - same shape as [`cosine_f32_neon`] without the two norm
/// reductions, so ~3x fewer FMAs when the norms are already known.
#[cfg(target_arch = "aarch64")]
#[must_use]
pub fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    // The loop bound comes from `a`, the loads come from both: a shorter `b`
    // would be read past its end. The dispatcher checks this; a caller reaching
    // the kernel directly does not go through the dispatcher.
    assert_eq!(a.len(), b.len(), "dimension mismatch");
    use std::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};
    let n = a.len();
    let block = n - (n % 16);
    let mut i = 0;
    // SAFETY: each `vld1q_f32` reads 4 f32 at offset `i + k*4 < block <= n`, in
    // bounds for both slices (caller guarantees equal length). Reads only, no
    // aliasing. NEON is baseline on aarch64.
    let mut sdot = unsafe {
        let mut dot = [vdupq_n_f32(0.0); 4];
        while i < block {
            for (k, d) in dot.iter_mut().enumerate() {
                let va = vld1q_f32(a.as_ptr().add(i + k * 4));
                let vb = vld1q_f32(b.as_ptr().add(i + k * 4));
                *d = vfmaq_f32(*d, va, vb);
            }
            i += 16;
        }
        let dot = vaddq_f32(vaddq_f32(dot[0], dot[1]), vaddq_f32(dot[2], dot[3]));
        vaddvq_f32(dot)
    };
    for i in i..n {
        sdot += a[i] * b[i];
    }
    sdot
}

/// Dot product, AVX2 plus FMA. Four accumulators hide FMA latency while each
/// iteration consumes 32 f32 values. The caller must check CPU support before
/// calling this function.
///
/// # Safety
///
/// The current CPU must support AVX2 and FMA, and both slices must have the
/// same length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,sse3")]
#[must_use]
pub unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    // The loop bound comes from `a`, the loads come from both: a shorter `b`
    // would be read past its end. The dispatcher checks this; a caller reaching
    // the kernel directly does not go through the dispatcher.
    assert_eq!(a.len(), b.len(), "dimension mismatch");
    use std::arch::x86_64::{
        _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_movehdup_ps, _mm_movehl_ps, _mm256_add_ps,
        _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_loadu_ps,
        _mm256_setzero_ps,
    };

    let block = a.len() - (a.len() % 32);
    let mut i = 0;
    // SAFETY: the caller selected AVX2 and FMA at runtime. Each load reads 8
    // f32 values at `i + k*8 < block <= a.len() == b.len()`. The slices are
    // read-only and unaligned loads accept every valid slice alignment.
    let reduced = unsafe {
        let mut acc = [_mm256_setzero_ps(); 4];
        while i < block {
            for (k, slot) in acc.iter_mut().enumerate() {
                let va = _mm256_loadu_ps(a.as_ptr().add(i + k * 8));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i + k * 8));
                *slot = _mm256_fmadd_ps(va, vb, *slot);
            }
            i += 32;
        }
        let sum = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
        let sum = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
        let sum = _mm_add_ps(sum, _mm_movehdup_ps(sum));
        _mm_cvtss_f32(_mm_add_ss(sum, _mm_movehl_ps(sum, sum)))
    };
    let mut sum = reduced;
    for i in i..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// Dot product of two f32 slices.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
#[must_use]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dimension mismatch");
    #[cfg(target_arch = "aarch64")]
    {
        dot_f32_neon(a, b)
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: both target features were checked immediately above.
            return unsafe { dot_f32_avx2(a, b) };
        }
        dot_f32_scalar(a, b)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        dot_f32_scalar(a, b)
    }
}

/// Hamming distance (number of differing bits) of two byte slices.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
#[must_use]
pub fn hamming_binary(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len(), "dimension mismatch");
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vpopcntdq")
        {
            // SAFETY: all three target features were checked immediately above.
            return unsafe { hamming_binary_avx512(a, b) };
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 was checked immediately above.
            return unsafe { hamming_binary_avx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        hamming_binary_neon(a, b)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        hamming_binary_scalar(a, b)
    }
}

/// Dot product of two i8 slices, accumulated in i32.
///
/// Dispatches to AVX2 on supporting x86 CPUs. AArch64 keeps the scalar path:
/// LLVM auto-vectorizes its widening multiply-accumulate better than baseline
/// NEON can without the `dotprod` extension.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
#[must_use]
pub fn dot_int8(a: &[i8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len(), "dimension mismatch");
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vnni")
        {
            // SAFETY: all three target features were checked immediately above.
            return unsafe { dot_int8_avx512(a, b) };
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 was checked immediately above.
            return unsafe { dot_int8_avx2(a, b) };
        }
    }
    dot_int8_scalar(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_vectors() {
        let v = &[1.0f32, 2.0, 3.0, 4.0, 5.0];
        assert!((cosine_f32(v, v) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = &[1.0f32, 0.0, 0.0, 0.0];
        let b = &[0.0f32, 1.0, 0.0, 0.0];
        assert!(cosine_f32(a, b).abs() < 1e-6);
    }

    #[test]
    #[allow(clippy::float_cmp)] // returns the exact literal 0.0 on a zero norm
    fn cosine_zero_vector_returns_zero() {
        let z = &[0.0f32; 5];
        let v = &[1.0f32, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(cosine_f32(z, v), 0.0);
    }

    #[test]
    fn cosine_non_multiple_of_4_dim() {
        // Exercises the scalar remainder tail (dim 7 = 4 lanes + 3).
        let a = &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        assert!((cosine_f32(a, a) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn hamming_identical() {
        let a = &[0xFFu8, 0x00, 0xAB, 0xCD];
        assert_eq!(hamming_binary(a, a), 0);
    }

    #[test]
    fn hamming_all_different() {
        let a = &[0xFFu8; 20];
        let b = &[0x00u8; 20];
        assert_eq!(hamming_binary(a, b), 160); // 20 bytes * 8 bits
    }

    #[test]
    fn dot_int8_basic() {
        let a = &[1i8, 2, 3];
        let b = &[4i8, 5, 6];
        assert_eq!(dot_int8(a, b), 4 + 10 + 18);
    }

    #[test]
    fn dot_int8_long_with_negatives() {
        let a: Vec<i8> = (0i8..40).map(|i| i % 7 - 3).collect();
        let b: Vec<i8> = (0i8..40).map(|i| i % 5 - 2).collect();
        let expect = dot_int8_scalar(&a, &b);
        assert_eq!(dot_int8(&a, &b), expect);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_f32_kernels_match_scalar_for_unaligned_tail() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) {
            return;
        }

        let a: Vec<f32> = (0..259).map(|i| (i as f32 - 97.0) / 31.0).collect();
        let b: Vec<f32> = (0..259).map(|i| (73.0 - i as f32) / 19.0).collect();
        let a = &a[1..258];
        let b = &b[1..258];

        let dot = unsafe { dot_f32_avx2(a, b) };
        let cosine = unsafe { cosine_f32_avx2(a, b) };
        assert!((dot - dot_f32_scalar(a, b)).abs() < 1e-3);
        assert!((cosine - cosine_f32_scalar(a, b)).abs() < 1e-5);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_int8_kernel_matches_scalar_for_unaligned_tail() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let a: Vec<i8> = (0..259).map(|i| (i as i8).wrapping_mul(19)).collect();
        let b: Vec<i8> = (0..259)
            .map(|i| (71i8).wrapping_sub((i as i8).wrapping_mul(13)))
            .collect();
        let a = &a[1..258];
        let b = &b[1..258];
        let dot = unsafe { dot_int8_avx2(a, b) };
        assert_eq!(dot, dot_int8_scalar(a, b));
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[test]
    fn avx512_int8_kernel_matches_scalar_for_unaligned_tail() {
        if !(std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vnni"))
        {
            return;
        }
        let a: Vec<i8> = (0..259).map(|i| (i as i8).wrapping_mul(19)).collect();
        let b: Vec<i8> = (0..259)
            .map(|i| (71i8).wrapping_sub((i as i8).wrapping_mul(13)))
            .collect();
        let a = &a[1..258];
        let b = &b[1..258];
        let dot = unsafe { dot_int8_avx512(a, b) };
        assert_eq!(dot, dot_int8_scalar(a, b));
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[test]
    fn avx512_int8_extremes_i8_min_max() {
        // i8::MIN * i8::MIN is the largest-magnitude product this kernel
        // ever sees (16384); confirms the sign-flip fixup doesn't overflow
        // or misfire at the boundary values the trick is riskiest for.
        if !(std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vnni"))
        {
            return;
        }
        let a = vec![i8::MIN; 128];
        let b = vec![i8::MIN; 128];
        assert_eq!(unsafe { dot_int8_avx512(&a, &b) }, dot_int8_scalar(&a, &b));
        let a = vec![i8::MAX; 128];
        let b = vec![i8::MIN; 128];
        assert_eq!(unsafe { dot_int8_avx512(&a, &b) }, dot_int8_scalar(&a, &b));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_hamming_kernel_matches_scalar_for_unaligned_tail() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let a: Vec<u8> = (0..259).map(|i| (i as u8).wrapping_mul(19)).collect();
        let b: Vec<u8> = (0..259)
            .map(|i| (71u8).wrapping_sub((i as u8).wrapping_mul(13)))
            .collect();
        let a = &a[1..258];
        let b = &b[1..258];
        let distance = unsafe { hamming_binary_avx2(a, b) };
        assert_eq!(distance, hamming_binary_scalar(a, b));
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[test]
    fn avx512_hamming_kernel_matches_scalar_for_unaligned_tail() {
        if !(std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vpopcntdq"))
        {
            return;
        }
        let a: Vec<u8> = (0..259).map(|i| (i as u8).wrapping_mul(19)).collect();
        let b: Vec<u8> = (0..259)
            .map(|i| (71u8).wrapping_sub((i as u8).wrapping_mul(13)))
            .collect();
        let a = &a[1..258];
        let b = &b[1..258];
        let distance = unsafe { hamming_binary_avx512(a, b) };
        assert_eq!(distance, hamming_binary_scalar(a, b));
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[test]
    fn avx512_and_popcnt_matches_avx2_and_scalar() {
        if !(std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vpopcntdq"))
        {
            return;
        }
        let a: Vec<u8> = (0..259).map(|i| (i as u8).wrapping_mul(37)).collect();
        let b: Vec<u8> = (0..259).map(|i| (i as u8).wrapping_mul(53)).collect();
        let a = &a[1..258];
        let b = &b[1..258];
        let avx512 = unsafe { and_popcnt_avx512(a, b) };
        let scalar: u32 = a.iter().zip(b).map(|(&x, &y)| (x & y).count_ones()).sum();
        assert_eq!(avx512, scalar);
        if std::is_x86_feature_detected!("avx2") {
            assert_eq!(avx512, unsafe { and_popcnt_avx2(a, b) });
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[test]
    fn avx512_hamming_matches_dispatched_hamming_binary() {
        // The dispatcher should prefer AVX-512 when the feature is compiled
        // in and the CPU supports it; confirm it agrees with calling the
        // kernel directly (regression guard for the dispatch order itself).
        if !(std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vpopcntdq"))
        {
            return;
        }
        let a: Vec<u8> = (0..300).map(|i| (i as u8).wrapping_mul(7)).collect();
        let b: Vec<u8> = (0..300).map(|i| (i as u8).wrapping_mul(11)).collect();
        assert_eq!(hamming_binary(&a, &b), unsafe {
            hamming_binary_avx512(&a, &b)
        });
    }

    // ── proptest: NEON kernels must match their scalar reference ─────────────

    #[cfg(target_arch = "aarch64")]
    proptest::proptest! {
        #[test]
        fn prop_cosine_neon_matches_scalar(
            pairs in proptest::collection::vec(
                (-1.0f32..1.0, -1.0f32..1.0), 1..600,
            ),
        ) {
            let a: Vec<f32> = pairs.iter().map(|&(x, _)| x).collect();
            let b: Vec<f32> = pairs.iter().map(|&(_, y)| y).collect();
            let neon = cosine_f32_neon(&a, &b);
            let scalar = cosine_f32_scalar(&a, &b);
            // FMA vs sequential summation differ only in rounding.
            proptest::prop_assert!(
                (neon - scalar).abs() < 1e-3,
                "neon={neon} scalar={scalar}",
            );
        }

        #[test]
        fn prop_hamming_neon_matches_scalar(
            pairs in proptest::collection::vec(
                (proptest::num::u8::ANY, proptest::num::u8::ANY), 0..600,
            ),
        ) {
            let a: Vec<u8> = pairs.iter().map(|&(x, _)| x).collect();
            let b: Vec<u8> = pairs.iter().map(|&(_, y)| y).collect();
            // Integer kernel: must match exactly.
            proptest::prop_assert_eq!(
                hamming_binary_neon(&a, &b),
                hamming_binary_scalar(&a, &b),
            );
        }

        #[test]
        fn prop_dot_int8_neon_matches_scalar(
            pairs in proptest::collection::vec(
                (proptest::num::i8::ANY, proptest::num::i8::ANY), 0..600,
            ),
        ) {
            let a: Vec<i8> = pairs.iter().map(|&(x, _)| x).collect();
            let b: Vec<i8> = pairs.iter().map(|&(_, y)| y).collect();
            // Integer kernel: must match exactly.
            proptest::prop_assert_eq!(dot_int8_neon(&a, &b), dot_int8_scalar(&a, &b));
        }
    }

    #[cfg(target_arch = "x86_64")]
    proptest::proptest! {
        #[test]
        fn prop_dot_int8_avx2_matches_scalar(
            pairs in proptest::collection::vec(
                (proptest::num::i8::ANY, proptest::num::i8::ANY), 0..600,
            ),
        ) {
            if std::is_x86_feature_detected!("avx2") {
                let a: Vec<i8> = pairs.iter().map(|&(x, _)| x).collect();
                let b: Vec<i8> = pairs.iter().map(|&(_, y)| y).collect();
                proptest::prop_assert_eq!(
                    unsafe { dot_int8_avx2(&a, &b) },
                    dot_int8_scalar(&a, &b),
                );
            }
        }

        #[cfg(feature = "avx512")]
        #[test]
        fn prop_dot_int8_avx512_matches_scalar(
            pairs in proptest::collection::vec(
                (proptest::num::i8::ANY, proptest::num::i8::ANY), 0..600,
            ),
        ) {
            if std::is_x86_feature_detected!("avx512f")
                && std::is_x86_feature_detected!("avx512bw")
                && std::is_x86_feature_detected!("avx512vnni")
            {
                let a: Vec<i8> = pairs.iter().map(|&(x, _)| x).collect();
                let b: Vec<i8> = pairs.iter().map(|&(_, y)| y).collect();
                proptest::prop_assert_eq!(
                    unsafe { dot_int8_avx512(&a, &b) },
                    dot_int8_scalar(&a, &b),
                );
            }
        }

        #[test]
        fn prop_hamming_avx2_matches_scalar(
            pairs in proptest::collection::vec(
                (proptest::num::u8::ANY, proptest::num::u8::ANY), 0..600,
            ),
        ) {
            if std::is_x86_feature_detected!("avx2") {
                let a: Vec<u8> = pairs.iter().map(|&(x, _)| x).collect();
                let b: Vec<u8> = pairs.iter().map(|&(_, y)| y).collect();
                proptest::prop_assert_eq!(
                    unsafe { hamming_binary_avx2(&a, &b) },
                    hamming_binary_scalar(&a, &b),
                );
            }
        }

        #[cfg(feature = "avx512")]
        #[test]
        fn prop_hamming_avx512_matches_scalar(
            pairs in proptest::collection::vec(
                (proptest::num::u8::ANY, proptest::num::u8::ANY), 0..600,
            ),
        ) {
            if std::is_x86_feature_detected!("avx512f")
                && std::is_x86_feature_detected!("avx512bw")
                && std::is_x86_feature_detected!("avx512vpopcntdq")
            {
                let a: Vec<u8> = pairs.iter().map(|&(x, _)| x).collect();
                let b: Vec<u8> = pairs.iter().map(|&(_, y)| y).collect();
                proptest::prop_assert_eq!(
                    unsafe { hamming_binary_avx512(&a, &b) },
                    hamming_binary_scalar(&a, &b),
                );
            }
        }
    }
}
