//! `skeg-simd` - distance kernels, SIMD-accelerated on aarch64 (NEON).
//!
//! The public functions ([`cosine_f32`], [`hamming_binary`], [`dot_int8`])
//! dispatch to whichever kernel benchmarks fastest on the target: a hand-rolled
//! NEON kernel for `cosine_f32` and `hamming_binary` on aarch64, and the
//! portable scalar kernel for `dot_int8` (LLVM auto-vectorizes its
//! multiply-accumulate better than baseline NEON without `dotprod`, as
//! benchmarks confirmed). Each scalar kernel doubles as the reference oracle for
//! its NEON counterpart.
//!
//! unsafe is allowed in this crate (NEON intrinsics + raw-pointer loads);
//! every unsafe block documents the bounds invariant that makes it sound.

pub mod block;
pub use block::{
    BLOCK, FLUSH_EVERY, build_tq4_lut_f32, interleave_tq4_codes, quantize_tq4_lut_u8,
    tq4_block32_score_scalar, tq4_block32_score_u8_scalar,
};

#[cfg(target_arch = "aarch64")]
pub use block::tq4_block32_score_u8_neon;

// ── Scalar kernels (portable; also the reference for NEON equivalence) ────────

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

// ── NEON kernels (aarch64) ────────────────────────────────────────────────────

/// Cosine similarity, NEON. 16 f32 per iteration over 4 independent
/// accumulator groups to hide FMA latency (the f32 reduction is not
/// auto-vectorized because float addition is non-associative).
#[cfg(target_arch = "aarch64")]
#[must_use]
pub fn cosine_f32_neon(a: &[f32], b: &[f32]) -> f32 {
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

/// Hamming distance, NEON. 64 bytes per iteration: `vcntq_u8` per-byte
/// popcount, widening-accumulated into u16 lanes so the horizontal reduction
/// runs once at the end rather than once per iteration.
#[cfg(target_arch = "aarch64")]
#[must_use]
pub fn hamming_binary_neon(a: &[u8], b: &[u8]) -> u32 {
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

/// Dot product of i8 slices, NEON. 64 i8 per iteration over 4 independent
/// accumulators to hide `vpadalq` latency.
///
/// Uses baseline NEON (`vmull_s8` widening multiply + `vpadalq_s16` pairwise
/// accumulate). The `dotprod` extension (`vdotq_s32`, one instruction instead
/// of three) is still unstable in `std::arch`; switch to it once stabilized.
#[cfg(target_arch = "aarch64")]
#[must_use]
pub fn dot_int8_neon(a: &[i8], b: &[i8]) -> i32 {
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

// ── Public dispatching API ────────────────────────────────────────────────────

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
    #[cfg(not(target_arch = "aarch64"))]
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
    #[cfg(not(target_arch = "aarch64"))]
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
/// Dispatches to the scalar kernel even on aarch64: LLVM auto-vectorizes the
/// scalar widening multiply-accumulate better than baseline NEON can without
/// the `dotprod` extension. [`dot_int8_neon`] is kept as an equivalence oracle
/// and becomes the dispatch target once `vdotq_s32` stabilizes.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
#[must_use]
pub fn dot_int8(a: &[i8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len(), "dimension mismatch");
    dot_int8_scalar(a, b)
}

/// Hardware prefetch hint for the sparse access pattern of the Vamana greedy
/// walk. Suggests the CPU pull the cache line containing `ptr`
/// into L1 while it keeps working on something else - hides L2/RAM latency
/// behind the SIMD compute of the current cosine.
///
/// On aarch64 emits `prfm pldl1keep, [ptr]` via inline asm: prefetch for
/// load into L1 cache, keep (high locality - the data will be touched
/// soon). The `_prefetch` intrinsic in `core::arch::aarch64` is not yet
/// stable in Rust (tracking issue #117217), so inline asm (stable since
/// 1.59) is used instead.
///
/// On other architectures this is a no-op: a portable prefetch fallback
/// in stable Rust does not exist (nightly has `core::intrinsics::prefetch_*`).
///
/// Safety: prefetch is a *hardware hint* - if `ptr` points to unmapped
/// memory the CPU silently ignores it. It cannot segfault and does not
/// alter observable program behaviour; the only effect is the hardware
/// cache state. The safe `pub fn` signature is sound because no caller can
/// cause unsoundness by passing an arbitrary pointer.
pub fn prefetch_read(ptr: *const u8) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: PRFM is a hint instruction. The CPU does not raise a fault
        // on unmapped pointers (ARM spec C5.6.114). `nostack`/`preserves_flags`
        // guarantee no interaction with the stack or global flag state.
        unsafe {
            core::arch::asm!(
                "prfm pldl1keep, [{p}]",
                p = in(reg) ptr,
                options(nostack, preserves_flags, readonly),
            );
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = ptr;
    }
}

/// `TurboQuant` 4-bit asymmetric inner product (ADC) kernel with i8-quantised
/// centroids. Computes `sum_i q_rot[i] * centroid[code[i]]` where `code[i]`
/// is a 4-bit index (two coords per byte, low nibble first) and the 16
/// `Lloyd-Max` centroids are stored as i8 in `centroids_i8`. The returned
/// value is the inner product *before* the per-vector scale correction the
/// caller layers on; `i8_scale` dequantises the centroid lookups inside
/// the kernel (one f32 multiply per dim).
///
/// On aarch64 the hot loop processes 16 coords per iteration: SWAR unpacks
/// 8 bytes into 16 lane indices, `vqtbl1q_s8` looks up 16 centroid bytes
/// from a single Q register in one cycle, the i8 values widen to f32 and
/// pair with `vld1q_f32` loads of `q_rot` for four `vfmaq_f32` FMAs into
/// a vector accumulator. On other targets a portable scalar fallback
/// runs the same algorithm.
///
/// # Recall implication
///
/// Centroid quantisation to 8 bits collapses the 16 Lloyd-Max levels onto
/// an i8 grid. For the typical scaled-Gaussian variance `1/dim` this loses
/// ~1.5% MSE relative to f32 centroids; recall is a function of the
/// distance ordering, not the magnitudes, so the practical recall hit at
/// the walk is small. The caller is expected to gate.
///
/// # Panics
///
/// Panics if `code.len() != dim / 2` or `q_rot.len() != dim`.
#[must_use]
pub fn tq4_adc_i8(
    code: &[u8],
    centroids_i8: &[i8; 16],
    i8_scale: f32,
    q_rot: &[f32],
    dim: usize,
) -> f32 {
    assert_eq!(code.len(), dim / 2, "tq4 code length / dim mismatch");
    assert_eq!(q_rot.len(), dim, "tq4 q_rot length / dim mismatch");
    #[cfg(target_arch = "aarch64")]
    {
        tq4_adc_i8_neon(code, centroids_i8, i8_scale, q_rot, dim)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        tq4_adc_i8_scalar(code, centroids_i8, i8_scale, q_rot, dim)
    }
}

/// Portable scalar reference for [`tq4_adc_i8`]. Used directly on non-aarch64
/// targets and as the oracle in the equivalence test.
#[must_use]
pub fn tq4_adc_i8_scalar(
    code: &[u8],
    centroids_i8: &[i8; 16],
    i8_scale: f32,
    q_rot: &[f32],
    dim: usize,
) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..dim {
        let byte = code[i / 2];
        let bucket = (if i % 2 == 0 { byte & 0x0F } else { byte >> 4 }) as usize;
        acc += q_rot[i] * (f32::from(centroids_i8[bucket]) * i8_scale);
    }
    acc
}

/// NEON kernel for [`tq4_adc_i8`]. Hot loop processes 16 coords per
/// iteration via `vqtbl1q_s8` lookup + 4x `vfmaq_f32`.
#[cfg(target_arch = "aarch64")]
#[must_use]
pub fn tq4_adc_i8_neon(
    code: &[u8],
    centroids_i8: &[i8; 16],
    i8_scale: f32,
    q_rot: &[f32],
    dim: usize,
) -> f32 {
    use std::arch::aarch64::{
        vaddq_f32, vaddvq_f32, vand_u8, vcombine_u8, vcvtq_f32_s32, vdup_n_u8, vdupq_n_f32,
        vfmaq_f32, vget_high_s8, vget_high_s16, vget_low_s8, vget_low_s16, vld1_u8, vld1q_f32,
        vld1q_s8, vmovl_s8, vmovl_s16, vqtbl1q_s8, vshr_n_u8, vzip1_u8, vzip2_u8,
    };
    let chunks = dim / 16;
    let mut tail_acc = 0.0f32;
    // SAFETY: NEON is baseline on aarch64 so all `vld1q_*` / `vfmaq_*` /
    // `vqtbl1q_s8` intrinsics are available without `target_feature`.
    // - `vld1q_s8(centroids_i8.as_ptr())` reads exactly 16 i8 from a 16-byte
    //   array - in bounds.
    // - `vld1q_u8(indices.as_ptr())` reads exactly 16 u8 from a stack
    //   `[u8; 16]` we just filled - in bounds.
    // - `vld1q_f32(q_rot.as_ptr().add(q_base + k*4))` reads 4 f32 at offset
    //   `q_base + k*4` for `k in 0..4`. The largest offset across the loop
    //   is `(chunks - 1) * 16 + 12`, and `chunks * 16 <= dim`, so the read
    //   ends at `dim - 1` - within `q_rot.len() == dim` (asserted above).
    // - `code` is read 8 bytes per chunk at offset `chunk * 8`; the largest
    //   end offset is `(chunks - 1) * 8 + 8 = chunks * 8 <= dim / 2 ==
    //   code.len()`.
    // - `vqtbl1q_s8` is total: out-of-range indices yield zero per ARM
    //   spec, so the 4-bit nibbles (0..15) need no validation.
    // - No aliasing: `code`, `q_rot`, `centroids_i8`, and the local
    //   `indices` buffer are distinct allocations.
    // 2x unrolled hot loop: process 32 coords per outer iteration via
    // two chunks. Reuses the LUT once, runs 8 independent f32x4
    // accumulators so the ARM core can keep multiple FMAs and
    // widen chains in flight simultaneously.
    let pair_count = chunks / 2;
    let tail_chunk = chunks % 2;
    let acc_total = unsafe {
        let lut = vld1q_s8(centroids_i8.as_ptr());
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let mut acc4 = vdupq_n_f32(0.0);
        let mut acc5 = vdupq_n_f32(0.0);
        let mut acc6 = vdupq_n_f32(0.0);
        let mut acc7 = vdupq_n_f32(0.0);
        let nibble_mask = vdup_n_u8(0x0F);
        for pair in 0..pair_count {
            let off = pair * 16;
            let q_base = pair * 32;
            // First half (16 coords) - NEON-native nibble unpack:
            //   load 8 packed bytes once,
            //   low nibbles = bytes & 0x0F,
            //   high nibbles = bytes >> 4,
            //   interleave low/high to land 16 lane indices in coord order.
            // Replaces a SWAR + stack roundtrip; saves ~6 cycles per
            // chunk over the previous implementation.
            let bytes_a = vld1_u8(code.as_ptr().add(off));
            let low_a = vand_u8(bytes_a, nibble_mask);
            let high_a = vshr_n_u8(bytes_a, 4);
            let idx_lo_a = vzip1_u8(low_a, high_a);
            let idx_hi_a = vzip2_u8(low_a, high_a);
            let idx_a = vcombine_u8(idx_lo_a, idx_hi_a);
            let centroids_a = vqtbl1q_s8(lut, idx_a);
            let low_i16_a = vmovl_s8(vget_low_s8(centroids_a));
            let high_i16_a = vmovl_s8(vget_high_s8(centroids_a));
            let c0a = vcvtq_f32_s32(vmovl_s16(vget_low_s16(low_i16_a)));
            let c1a = vcvtq_f32_s32(vmovl_s16(vget_high_s16(low_i16_a)));
            let c2a = vcvtq_f32_s32(vmovl_s16(vget_low_s16(high_i16_a)));
            let c3a = vcvtq_f32_s32(vmovl_s16(vget_high_s16(high_i16_a)));
            let q0a = vld1q_f32(q_rot.as_ptr().add(q_base));
            let q1a = vld1q_f32(q_rot.as_ptr().add(q_base + 4));
            let q2a = vld1q_f32(q_rot.as_ptr().add(q_base + 8));
            let q3a = vld1q_f32(q_rot.as_ptr().add(q_base + 12));

            // Second half (next 16 coords) - same nibble-unpack trick.
            let bytes_b = vld1_u8(code.as_ptr().add(off + 8));
            let low_b = vand_u8(bytes_b, nibble_mask);
            let high_b = vshr_n_u8(bytes_b, 4);
            let idx_lo_b = vzip1_u8(low_b, high_b);
            let idx_hi_b = vzip2_u8(low_b, high_b);
            let idx_b = vcombine_u8(idx_lo_b, idx_hi_b);
            let centroids_b = vqtbl1q_s8(lut, idx_b);
            let low_i16_b = vmovl_s8(vget_low_s8(centroids_b));
            let high_i16_b = vmovl_s8(vget_high_s8(centroids_b));
            let c0b = vcvtq_f32_s32(vmovl_s16(vget_low_s16(low_i16_b)));
            let c1b = vcvtq_f32_s32(vmovl_s16(vget_high_s16(low_i16_b)));
            let c2b = vcvtq_f32_s32(vmovl_s16(vget_low_s16(high_i16_b)));
            let c3b = vcvtq_f32_s32(vmovl_s16(vget_high_s16(high_i16_b)));
            let q0b = vld1q_f32(q_rot.as_ptr().add(q_base + 16));
            let q1b = vld1q_f32(q_rot.as_ptr().add(q_base + 20));
            let q2b = vld1q_f32(q_rot.as_ptr().add(q_base + 24));
            let q3b = vld1q_f32(q_rot.as_ptr().add(q_base + 28));

            acc0 = vfmaq_f32(acc0, q0a, c0a);
            acc1 = vfmaq_f32(acc1, q1a, c1a);
            acc2 = vfmaq_f32(acc2, q2a, c2a);
            acc3 = vfmaq_f32(acc3, q3a, c3a);
            acc4 = vfmaq_f32(acc4, q0b, c0b);
            acc5 = vfmaq_f32(acc5, q1b, c1b);
            acc6 = vfmaq_f32(acc6, q2b, c2b);
            acc7 = vfmaq_f32(acc7, q3b, c3b);
        }
        // Optional trailing single chunk when `chunks` is odd.
        if tail_chunk == 1 {
            let chunk = pair_count * 2;
            let off = chunk * 8;
            let q_base = chunk * 16;
            let bytes = vld1_u8(code.as_ptr().add(off));
            let low = vand_u8(bytes, nibble_mask);
            let high = vshr_n_u8(bytes, 4);
            let idx = vcombine_u8(vzip1_u8(low, high), vzip2_u8(low, high));
            let centroids = vqtbl1q_s8(lut, idx);
            let low_i16 = vmovl_s8(vget_low_s8(centroids));
            let high_i16 = vmovl_s8(vget_high_s8(centroids));
            let c0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(low_i16)));
            let c1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(low_i16)));
            let c2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(high_i16)));
            let c3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(high_i16)));
            let q0 = vld1q_f32(q_rot.as_ptr().add(q_base));
            let q1 = vld1q_f32(q_rot.as_ptr().add(q_base + 4));
            let q2 = vld1q_f32(q_rot.as_ptr().add(q_base + 8));
            let q3 = vld1q_f32(q_rot.as_ptr().add(q_base + 12));
            acc0 = vfmaq_f32(acc0, q0, c0);
            acc1 = vfmaq_f32(acc1, q1, c1);
            acc2 = vfmaq_f32(acc2, q2, c2);
            acc3 = vfmaq_f32(acc3, q3, c3);
        }
        // Tree reduce + horizontal sum + deferred scale.
        let s01 = vaddq_f32(acc0, acc1);
        let s23 = vaddq_f32(acc2, acc3);
        let s45 = vaddq_f32(acc4, acc5);
        let s67 = vaddq_f32(acc6, acc7);
        let acc = vaddq_f32(vaddq_f32(s01, s23), vaddq_f32(s45, s67));
        vaddvq_f32(acc) * i8_scale
    };
    // Tail: dim not a multiple of 16. mxbai 1024 and MiniLM 384 both fit.
    let tail_start = chunks * 16;
    for i in tail_start..dim {
        let byte = code[i / 2];
        let bucket = (if i % 2 == 0 { byte & 0x0F } else { byte >> 4 }) as usize;
        tail_acc += q_rot[i] * (f32::from(centroids_i8[bucket]) * i8_scale);
    }
    acc_total + tail_acc
}

/// TurboQuant 2-bit asymmetric inner product (ADC) kernel with i8-quantised
/// centroids. Same shape as [`tq4_adc_i8`] but processes 32 coords per
/// 8-byte chunk (4 codes per byte at 2 bits each) via two `vqtbl1q_s8`
/// lookups against a 16-byte LUT where only the first 4 entries are real
/// centroids (5..15 are zero-padded; the 2-bit codes only ever hit indices
/// 0..3).
///
/// # Panics
///
/// Panics if `code.len() != dim / 4` or `q_rot.len() != dim`.
#[must_use]
pub fn tq2_adc_i8(
    code: &[u8],
    centroids_i8: &[i8; 16],
    i8_scale: f32,
    q_rot: &[f32],
    dim: usize,
) -> f32 {
    assert_eq!(code.len(), dim / 4, "tq2 code length / dim mismatch");
    assert_eq!(q_rot.len(), dim, "tq2 q_rot length / dim mismatch");
    #[cfg(target_arch = "aarch64")]
    {
        tq2_adc_i8_neon(code, centroids_i8, i8_scale, q_rot, dim)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        tq2_adc_i8_scalar(code, centroids_i8, i8_scale, q_rot, dim)
    }
}

/// Portable scalar reference for [`tq2_adc_i8`]. Oracle in proptest.
#[must_use]
pub fn tq2_adc_i8_scalar(
    code: &[u8],
    centroids_i8: &[i8; 16],
    i8_scale: f32,
    q_rot: &[f32],
    dim: usize,
) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..dim {
        let byte = code[i / 4];
        let shift = (i % 4) * 2;
        let bucket = ((byte >> shift) & 0x03) as usize;
        acc += q_rot[i] * (f32::from(centroids_i8[bucket]) * i8_scale);
    }
    acc
}

/// NEON kernel for [`tq2_adc_i8`]. 32 coords per 8-byte chunk via two
/// `vqtbl1q_s8` calls.
#[cfg(target_arch = "aarch64")]
#[must_use]
pub fn tq2_adc_i8_neon(
    code: &[u8],
    centroids_i8: &[i8; 16],
    i8_scale: f32,
    q_rot: &[f32],
    dim: usize,
) -> f32 {
    use std::arch::aarch64::{
        vaddq_f32, vaddvq_f32, vandq_u8, vcombine_u8, vcvtq_f32_s32, vdupq_n_f32, vdupq_n_u8,
        vfmaq_f32, vget_high_s8, vget_high_s16, vget_low_s8, vget_low_s16, vld1_u8, vld1q_f32,
        vld1q_s8, vld1q_u8, vmovl_s8, vmovl_s16, vqtbl1q_s8, vqtbl1q_u8, vshlq_u8,
    };
    let chunks = dim / 32;
    let mut tail_acc = 0.0f32;
    // SAFETY: same envelope as `tq4_adc_i8_neon`. Bounds reasoning:
    // - `code` read 8 bytes per chunk, max end offset `(chunks-1)*8 + 8 =
    //   chunks * 8 = dim/4 = code.len()`.
    // - `q_rot` read 32 f32 per chunk in 8 lanes of 4, max end at
    //   `(chunks-1)*32 + 28 + 4 = chunks * 32 = dim`, within `q_rot.len()`.
    // - `centroids_i8` is a 16-byte array, `vld1q_s8` reads exactly 16 i8.
    // - The two `indices_*` arrays are stack `[u8; 16]` we fill before
    //   each `vld1q_u8`. All 2-bit indices in 0..3, well below 16.
    // - `vqtbl1q_s8` is total per ARM spec (out-of-range -> zero); not
    //   exercised here but documented for correctness.
    let acc_total = unsafe {
        let lut = vld1q_s8(centroids_i8.as_ptr());
        // Eight independent accumulators (4 per half-chunk). Same pattern
        // as tq4: break the serial FMA dependency chain so the issue
        // rate, not the FMA latency, bounds the loop. i8_scale is
        // deferred to one multiply on the horizontal sum.
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let mut acc4 = vdupq_n_f32(0.0);
        let mut acc5 = vdupq_n_f32(0.0);
        let mut acc6 = vdupq_n_f32(0.0);
        let mut acc7 = vdupq_n_f32(0.0);
        // NEON-native unpack constants. The shift table holds per-lane
        // right-shift amounts as negative i8 (vshlq_u8 with negative
        // shifts performs right shifts). The byte-replicate tables
        // (`rep_lo`, `rep_hi`) route each input byte to four
        // consecutive output lanes so the per-lane shifts pull out
        // the four 2-bit codes from each source byte in sequential
        // coord order.
        let mask03 = vdupq_n_u8(0x03);
        let shifts =
            vld1q_s8([0i8, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6].as_ptr());
        let rep_lo_tbl = vld1q_u8([0u8, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3].as_ptr());
        let rep_hi_tbl = vld1q_u8([4u8, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7].as_ptr());
        for chunk in 0..chunks {
            let off = chunk * 8;
            let q_base = chunk * 32;

            // Load 8 packed bytes once, duplicate to 16 lanes so the
            // table lookup can fan each byte out to four output lanes.
            let bytes8 = vld1_u8(code.as_ptr().add(off));
            let bytes16 = vcombine_u8(bytes8, bytes8);
            let rep_a = vqtbl1q_u8(bytes16, rep_lo_tbl);
            let rep_b = vqtbl1q_u8(bytes16, rep_hi_tbl);
            let indices_a_v = vandq_u8(vshlq_u8(rep_a, shifts), mask03);
            let indices_b_v = vandq_u8(vshlq_u8(rep_b, shifts), mask03);

            // First half: 16 lookups + 4 f32x4 FMAs.
            let c_i8_a = vqtbl1q_s8(lut, indices_a_v);
            let low_i16_a = vmovl_s8(vget_low_s8(c_i8_a));
            let high_i16_a = vmovl_s8(vget_high_s8(c_i8_a));
            let c0a = vcvtq_f32_s32(vmovl_s16(vget_low_s16(low_i16_a)));
            let c1a = vcvtq_f32_s32(vmovl_s16(vget_high_s16(low_i16_a)));
            let c2a = vcvtq_f32_s32(vmovl_s16(vget_low_s16(high_i16_a)));
            let c3a = vcvtq_f32_s32(vmovl_s16(vget_high_s16(high_i16_a)));
            let q0a = vld1q_f32(q_rot.as_ptr().add(q_base));
            let q1a = vld1q_f32(q_rot.as_ptr().add(q_base + 4));
            let q2a = vld1q_f32(q_rot.as_ptr().add(q_base + 8));
            let q3a = vld1q_f32(q_rot.as_ptr().add(q_base + 12));
            acc0 = vfmaq_f32(acc0, q0a, c0a);
            acc1 = vfmaq_f32(acc1, q1a, c1a);
            acc2 = vfmaq_f32(acc2, q2a, c2a);
            acc3 = vfmaq_f32(acc3, q3a, c3a);

            // Second half: another 16 lookups + 4 FMAs.
            let c_i8_b = vqtbl1q_s8(lut, indices_b_v);
            let low_i16_b = vmovl_s8(vget_low_s8(c_i8_b));
            let high_i16_b = vmovl_s8(vget_high_s8(c_i8_b));
            let c0b = vcvtq_f32_s32(vmovl_s16(vget_low_s16(low_i16_b)));
            let c1b = vcvtq_f32_s32(vmovl_s16(vget_high_s16(low_i16_b)));
            let c2b = vcvtq_f32_s32(vmovl_s16(vget_low_s16(high_i16_b)));
            let c3b = vcvtq_f32_s32(vmovl_s16(vget_high_s16(high_i16_b)));
            let q0b = vld1q_f32(q_rot.as_ptr().add(q_base + 16));
            let q1b = vld1q_f32(q_rot.as_ptr().add(q_base + 20));
            let q2b = vld1q_f32(q_rot.as_ptr().add(q_base + 24));
            let q3b = vld1q_f32(q_rot.as_ptr().add(q_base + 28));
            acc4 = vfmaq_f32(acc4, q0b, c0b);
            acc5 = vfmaq_f32(acc5, q1b, c1b);
            acc6 = vfmaq_f32(acc6, q2b, c2b);
            acc7 = vfmaq_f32(acc7, q3b, c3b);
        }
        // Tree reduce then horizontal sum then deferred scale.
        let s01 = vaddq_f32(acc0, acc1);
        let s23 = vaddq_f32(acc2, acc3);
        let s45 = vaddq_f32(acc4, acc5);
        let s67 = vaddq_f32(acc6, acc7);
        let acc = vaddq_f32(vaddq_f32(s01, s23), vaddq_f32(s45, s67));
        vaddvq_f32(acc) * i8_scale
    };
    // Tail: dim not a multiple of 32. mxbai 1024 and MiniLM 384 both fit.
    let tail_start = chunks * 32;
    for i in tail_start..dim {
        let byte = code[i / 4];
        let shift = (i % 4) * 2;
        let bucket = ((byte >> shift) & 0x03) as usize;
        tail_acc += q_rot[i] * (f32::from(centroids_i8[bucket]) * i8_scale);
    }
    acc_total + tail_acc
}

// ── TurboQuant 1-bit masked sum ───────────────────────────────────────────────
//
// The 1-bit asymmetric inner product reduces to `pos_c * (2 * q_masked -
// q_sum)`, where `q_masked = sum of q_rot[i] over coords whose stored sign bit
// is 1` (see `skeg-vector::turboquant::tq1_adc_swar`). The per-vector wrap is
// cheap scalar; the hot part is this masked sum, which these kernels compute.
// `q_sum` is precomputed once per query, so the walk only pays the masked sum.

/// Sum of `q_rot[i]` over coords whose code bit is set. `code.len() == dim/8`,
/// `q_rot.len() == dim`, bits packed LSB-first (bit `i%8` of byte `i/8`).
#[must_use]
pub fn tq1_masked_sum_scalar(code: &[u8], q_rot: &[f32], dim: usize) -> f32 {
    let mut acc = 0.0f32;
    for (byte_idx, &byte) in code.iter().take(dim / 8).enumerate() {
        let base = byte_idx * 8;
        for b in 0..8 {
            let bit = ((byte >> b) & 1) as f32;
            acc += q_rot[base + b] * bit;
        }
    }
    acc
}

/// NEON kernel for [`tq1_masked_sum`]. Builds the per-lane `{0, !0}` selection
/// mask in-register with `vtstq_u32` (broadcast the code byte, test against
/// per-lane bit selectors) instead of gathering it from a memory table - the
/// table loads were the bottleneck once the graph stopped fitting in cache.
/// Processes 32 coords per iteration (four code bytes) across eight independent
/// accumulators - same shape as `tq2`/`tq4` - so the loop is bound by add
/// throughput rather than the ~3-cycle add latency. A scalar tail covers the
/// trailing `bytes % 4` bytes. `dim % 8 == 0` is required (tq1 packs whole
/// bytes); the dispatcher [`tq1_masked_sum`] asserts the lengths.
#[cfg(target_arch = "aarch64")]
#[must_use]
pub fn tq1_masked_sum_neon(code: &[u8], q_rot: &[f32], dim: usize) -> f32 {
    use std::arch::aarch64::{
        vaddq_f32, vaddvq_f32, vandq_u32, vdupq_n_f32, vdupq_n_u32, vld1q_f32, vld1q_u32,
        vreinterpretq_f32_u32, vreinterpretq_u32_f32, vtstq_u32,
    };
    let bytes = dim / 8;
    let quads = bytes / 4;
    // SAFETY: `bytes = dim/8`, `dim % 8 == 0`. The quad loop reads bytes
    // `4q..4q+4 < bytes <= code.len()` and eight f32x4 spanning `[q*32, q*32+32)`
    // with `q*32+32 <= quads*32 <= dim = q_rot.len()`. Bit selectors are 16-byte
    // stack arrays read in full by `vld1q_u32`, loaded once.
    let acc = unsafe {
        // Lane j tests bit j of the nibble: lo = bits 0..3, hi = bits 4..7.
        let sel_lo = vld1q_u32([1u32, 2, 4, 8].as_ptr());
        let sel_hi = vld1q_u32([16u32, 32, 64, 128].as_ptr());
        let mut acc = [vdupq_n_f32(0.0); 8];
        for q in 0..quads {
            let base = q * 32;
            for (i, a) in acc.chunks_exact_mut(2).enumerate() {
                let byte = vdupq_n_u32(u32::from(*code.get_unchecked(4 * q + i)));
                let qlo = vld1q_f32(q_rot.as_ptr().add(base + i * 8));
                let qhi = vld1q_f32(q_rot.as_ptr().add(base + i * 8 + 4));
                a[0] = vaddq_f32(
                    a[0],
                    vreinterpretq_f32_u32(vandq_u32(
                        vreinterpretq_u32_f32(qlo),
                        vtstq_u32(byte, sel_lo),
                    )),
                );
                a[1] = vaddq_f32(
                    a[1],
                    vreinterpretq_f32_u32(vandq_u32(
                        vreinterpretq_u32_f32(qhi),
                        vtstq_u32(byte, sel_hi),
                    )),
                );
            }
        }
        let s0 = vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3]));
        let s1 = vaddq_f32(vaddq_f32(acc[4], acc[5]), vaddq_f32(acc[6], acc[7]));
        vaddvq_f32(vaddq_f32(s0, s1))
    };
    // Tail: the trailing `bytes % 4` bytes (8 coords each) the quad loop skipped.
    let mut tail = 0.0f32;
    for (b, &byte) in code.iter().enumerate().skip(quads * 4) {
        let base = b * 8;
        for bit_pos in 0..8 {
            let bit = ((byte >> bit_pos) & 1) as f32;
            tail += q_rot[base + bit_pos] * bit;
        }
    }
    acc + tail
}

/// Masked sum for the tq1 asymmetric inner product, NEON on aarch64.
///
/// # Panics
///
/// Panics if `dim % 8 != 0`, `code.len() != dim / 8`, or `q_rot.len() != dim`.
/// These are runtime asserts (not `debug_assert`) because the NEON kernel does
/// unchecked pointer loads that rely on them - matching the `tq2_adc_i8` /
/// `tq4_adc_i8` contract so a length mismatch can never reach the kernel in a
/// release build.
#[must_use]
pub fn tq1_masked_sum(code: &[u8], q_rot: &[f32], dim: usize) -> f32 {
    assert_eq!(dim % 8, 0, "tq1 dim must be a multiple of 8");
    assert_eq!(code.len(), dim / 8, "tq1 code length / dim mismatch");
    assert_eq!(q_rot.len(), dim, "tq1 q_rot length / dim mismatch");
    #[cfg(target_arch = "aarch64")]
    {
        tq1_masked_sum_neon(code, q_rot, dim)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        tq1_masked_sum_scalar(code, q_rot, dim)
    }
}

/// Quantise a slice of f32 Lloyd-Max centroids to i8 with a shared scale.
/// Returns `(i8_centroids, i8_scale)` such that
/// `i8_centroids[k] * i8_scale ≈ centroids[k]`. The scale uses the max
/// absolute value (symmetric quantisation, max maps to ±127), so signs
/// are preserved exactly.
#[must_use]
pub fn quantise_centroids_i8<const N: usize>(centroids: &[f32; N]) -> ([i8; N], f32) {
    let max_abs = centroids.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
    let inv = 1.0 / scale;
    let mut out = [0i8; N];
    for (o, &c) in out.iter_mut().zip(centroids.iter()) {
        #[allow(clippy::cast_possible_truncation)] // clamped into i8 range first
        let q = (c * inv).round().clamp(-127.0, 127.0) as i8;
        *o = q;
    }
    (out, scale)
}

/// Name of the active SIMD backend, for observability and tests.
#[must_use]
pub fn simd_backend() -> &'static str {
    #[cfg(target_arch = "aarch64")]
    {
        "neon"
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        "scalar"
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

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

    #[test]
    fn simd_backend_is_reported() {
        let backend = simd_backend();
        assert!(["neon", "scalar"].contains(&backend));
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

        #[test]
        fn prop_tq2_adc_i8_neon_matches_scalar(
            chunks in 1usize..16,
            codes_data in proptest::collection::vec(proptest::num::u8::ANY, 0..128),
            // tq2 uses only the first 4 entries; rest must be present in
            // the slice but won't be hit (codes guaranteed 0..3).
            centroids in proptest::collection::vec(-127i8..=127i8, 16..=16),
            q_data in proptest::collection::vec(-1.0f32..1.0, 0..600),
            i8_scale in 1e-6f32..1e-2,
        ) {
            let dim = chunks * 32;
            let mut code = codes_data;
            code.resize(dim / 4, 0);
            let mut q = q_data;
            q.resize(dim, 0.0);
            let mut centroids_arr = [0i8; 16];
            for (i, &c) in centroids.iter().enumerate() { centroids_arr[i] = c; }
            // Zero out entries 4..15: the scalar reference uses
            // `centroids_i8[bucket]` with bucket in 0..3, so zeros in the
            // tail must not change either output (both kernels see them).
            for slot in &mut centroids_arr[4..16] { *slot = 0; }

            let neon = tq2_adc_i8_neon(&code, &centroids_arr, i8_scale, &q, dim);
            let scalar = tq2_adc_i8_scalar(&code, &centroids_arr, i8_scale, &q, dim);
            let denom = scalar.abs().max(1.0);
            proptest::prop_assert!(
                (neon - scalar).abs() / denom < 1e-3,
                "tq2_adc_i8 neon {} scalar {} dim {}",
                neon, scalar, dim,
            );
        }

        #[test]
        fn prop_tq1_masked_sum_neon_matches_scalar(
            bytes in 1usize..64,
            codes_data in proptest::collection::vec(proptest::num::u8::ANY, 0..64),
            q_data in proptest::collection::vec(-1.0f32..1.0, 0..512),
        ) {
            let dim = bytes * 8;
            let mut code = codes_data;
            code.resize(dim / 8, 0);
            let mut q = q_data;
            q.resize(dim, 0.0);
            let neon = tq1_masked_sum_neon(&code, &q, dim);
            let scalar = tq1_masked_sum_scalar(&code, &q, dim);
            let denom = scalar.abs().max(1.0);
            proptest::prop_assert!(
                (neon - scalar).abs() / denom < 1e-3,
                "tq1_masked_sum neon {} scalar {} dim {}",
                neon, scalar, dim,
            );
        }

        #[test]
        fn prop_tq4_adc_i8_neon_matches_scalar(
            // 16-coord chunks: dims 16..512 cover the SWAR loop body and
            // trigger the tail handling at non-multiples of 16.
            chunks in 1usize..32,
            codes_data in proptest::collection::vec(proptest::num::u8::ANY, 0..256),
            centroids in proptest::collection::vec(-127i8..=127i8, 16..=16),
            q_data in proptest::collection::vec(-1.0f32..1.0, 0..600),
            i8_scale in 1e-6f32..1e-2,
        ) {
            let dim = chunks * 16;
            // Trim/pad inputs to required sizes.
            let mut code = codes_data;
            code.resize(dim / 2, 0);
            let mut q = q_data;
            q.resize(dim, 0.0);
            let mut centroids_arr = [0i8; 16];
            for (i, &c) in centroids.iter().enumerate() { centroids_arr[i] = c; }

            let neon = tq4_adc_i8_neon(&code, &centroids_arr, i8_scale, &q, dim);
            let scalar = tq4_adc_i8_scalar(&code, &centroids_arr, i8_scale, &q, dim);
            // FMA vs sequential summation: |neon - scalar| / (|scalar| + 1) < 1e-3
            // covers both relative and absolute drift at the f32 precision.
            let denom = scalar.abs().max(1.0);
            proptest::prop_assert!(
                (neon - scalar).abs() / denom < 1e-3,
                "tq4_adc_i8 neon {} scalar {} dim {}",
                neon, scalar, dim,
            );
        }
    }
}
