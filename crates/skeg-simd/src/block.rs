//! Block-SIMD scoring for 4-bit TurboQuant codes.
//!
//! First stage of the block-kernel work: scalar reference and codes-layout
//! helpers, gated by an equivalence test against [`tq4_adc_i8_scalar`]. NEON
//! implementation lands in a follow-up commit; this file documents
//! the interleaved layout and the reference arithmetic so the NEON
//! pass can be diffed against a working oracle.
//!
//! Layout invariants the block kernel assumes:
//! - `codes` is a contiguous byte array of length `dim/2 * BLOCK` for
//!   a single 32-vector block. Byte `codes[g * BLOCK + v]` holds the
//!   4-bit codes for coords `(2*g, 2*g+1)` of vector `v` in the
//!   block; the low nibble is coord `2*g`, the high nibble is coord
//!   `2*g+1`. This is the same byte-packing as
//!   `tq4_adc_i8_scalar` uses per-row; the difference is the outer
//!   iteration order (byte-group major, then vector inside the block).
//! - `lut` is a flat `f32` table of length `dim/2 * 32`. Each per-byte-
//!   group sub-table holds 32 f32 values: indices 0..16 are the
//!   pre-computed scores for the low-nibble coord (i.e.
//!   `q_rot[2*g] * centroid_f32[c]` for c in 0..16), indices 16..32
//!   are the same for the high-nibble coord at `2*g+1`. This pre-
//!   compute is done once per query.
//!
//! Why the f32 LUT instead of u8 like turbovec: the scalar reference
//! only needs to be a clear oracle. The NEON path will swap in a
//! u8 LUT with periodic widen flush; that is a separate landing.

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

/// Vectors scored in parallel per block. Matches turbovec's BLOCK
/// constant and keeps the NEON layout aligned with 16-lane u8x16
/// loads (two halves per block).
pub const BLOCK: usize = 32;

/// Pre-compute the f32 LUT for a single query.
///
/// `q_rot`: length `dim`, the rotated and normalised query.
/// `centroids`: length 16, the Lloyd-Max centroid values for the
/// current 4-bit codebook.
///
/// Output buffer must be length `dim / 2 * 32`. After the call,
/// `lut[g * 32 + c]` for `c in 0..16` is the per-byte-group score
/// for the low-nibble code `c`, and `lut[g * 32 + 16 + c]` is the
/// same for the high-nibble code `c`.
///
/// # Panics
///
/// Panics if any slice length is wrong.
pub fn build_tq4_lut_f32(q_rot: &[f32], centroids: &[f32; 16], dim: usize, lut: &mut [f32]) {
    assert_eq!(q_rot.len(), dim, "q_rot length mismatch");
    assert_eq!(dim % 2, 0, "dim must be even for 4-bit codes");
    let n_groups = dim / 2;
    assert_eq!(lut.len(), n_groups * 32, "lut length mismatch");
    for g in 0..n_groups {
        let q_lo = q_rot[2 * g];
        let q_hi = q_rot[2 * g + 1];
        for c in 0..16 {
            lut[g * 32 + c] = q_lo * centroids[c];
            lut[g * 32 + 16 + c] = q_hi * centroids[c];
        }
    }
}

/// Lay out the 4-bit codes of 32 vectors in block-interleaved order.
///
/// `rows_codes` is a slice of `BLOCK` slices, each `dim / 2` bytes
/// (the row-major layout the existing kernel reads). Output is
/// `dim / 2 * BLOCK` bytes where consecutive bytes hold the same
/// byte-group from consecutive vectors. This is the layout the
/// block kernel reads.
///
/// # Panics
///
/// Panics if any row has the wrong length or there are not exactly
/// `BLOCK` rows.
pub fn interleave_tq4_codes(rows_codes: &[&[u8]], dim: usize, out: &mut [u8]) {
    assert_eq!(rows_codes.len(), BLOCK, "expected exactly BLOCK rows");
    let n_groups = dim / 2;
    assert_eq!(out.len(), n_groups * BLOCK, "out length mismatch");
    for (v, row) in rows_codes.iter().enumerate() {
        assert_eq!(row.len(), n_groups, "row length mismatch");
        for g in 0..n_groups {
            out[g * BLOCK + v] = row[g];
        }
    }
}

/// Flush window for the u8 accumulator. [`quantize_tq4_lut_u8`] caps
/// every LUT entry at 127, so one low + one high nibble lookup sums to
/// at most `254` and the byte add (`vaddq_u8` / `_mm256_add_epi8`)
/// never wraps. With `FLUSH_EVERY = 32` the worst-case u16 accumulator
/// before flush is `32 * 254 = 8'128`, well below the
/// `u16::MAX = 65'535` ceiling, so the widen-then-flush step is safe.
/// The same constant doubles as the unroll factor in the NEON and AVX2
/// kernels.
pub const FLUSH_EVERY: usize = 32;

/// Pre-compute the u8 LUT for a single query plus the scale+bias
/// the block kernel needs to reconstruct f32 scores.
///
/// The u8 LUT lives next to the float LUT it derives from: the same
/// `[g * 32, g * 32 + 16)` low-nibble window and
/// `[g * 32 + 16, g * 32 + 32)` high-nibble window. The whole table
/// is quantized with a single shared (min, range) so we can recover
/// `f32_score = sum_u8 * inv_scale + bias_per_group * n_groups` after
/// the dot-product loop.
///
/// `scale_out` returns `range / 127` (named `inv_scale`: it is the
/// factor the reconstruction *multiplies* the u8 sum by) such that
/// `u8_v = ((f32_v - min) / inv_scale).round().clamp(0, 127)`.
/// `bias_per_group_out` returns the per-byte-group constant `2 * min`
/// (one `min` contribution per nibble looked up in a group), so the
/// reconstruction adds `n_groups * bias_per_group_out` total -
/// encapsulated in the scoring routines below to keep callers from
/// getting the scaling wrong.
///
/// # Panics
///
/// Panics if any slice length is wrong.
pub fn quantize_tq4_lut_u8(lut_f32: &[f32], lut_u8: &mut [u8]) -> (f32, f32) {
    assert_eq!(
        lut_f32.len(),
        lut_u8.len(),
        "lut_f32 and lut_u8 must be the same length"
    );
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &v in lut_f32 {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    // Range = 0 means the query is degenerate (all coords zero).
    // The block kernel still has to return finite scores, so we
    // collapse the LUT to its single value and let the scaler turn
    // that into a flat output.
    let range = (hi - lo).max(f32::MIN_POSITIVE);
    // Quantise to [0, 127] (not [0, 255]): each byte-group of the
    // block kernel sums one low-nibble lookup and one high-nibble
    // lookup with `vaddq_u8`, which truncates at `u8::MAX`. Capping
    // each entry at 127 keeps the per-group sum safe in u8 before
    // the widening accumulate.
    let scale = 127.0 / range;
    let inv_scale = range / 127.0;
    for (dst, &v) in lut_u8.iter_mut().zip(lut_f32.iter()) {
        let qf = ((v - lo) * scale).round().clamp(0.0, 127.0);
        *dst = qf as u8;
    }
    // Bias per byte-group: a low + a high nibble lookup, each
    // contributes `lo` worth of offset that needs subtracting.
    let bias_per_group = 2.0 * lo;
    (inv_scale, bias_per_group)
}

/// Scalar block scoring against a u8 LUT, mirroring what the NEON
/// kernel will do. Accumulates per-vector contributions as `u32`
/// in `FLUSH_EVERY` windows, flushing to an `f32` accumulator at
/// each boundary; the final reconstruction applies the saved
/// `inv_scale` and `bias_per_group` from
/// [`quantize_tq4_lut_u8`].
///
/// This is the oracle the NEON path is tested against. It is
/// intentionally close to the NEON layout (per-vector u32 accs +
/// periodic flush) so the equivalence test catches drift.
///
/// # Panics
///
/// Panics if any input is the wrong length.
pub fn tq4_block32_score_u8_scalar(
    codes: &[u8], // n_groups * BLOCK bytes, interleaved
    lut_u8: &[u8],
    inv_scale: f32,
    bias_per_group: f32,
    dim: usize,
    out: &mut [f32; BLOCK],
) {
    assert_eq!(dim % 2, 0, "dim must be even for 4-bit codes");
    let n_groups = dim / 2;
    assert_eq!(codes.len(), n_groups * BLOCK, "codes length mismatch");
    assert_eq!(lut_u8.len(), n_groups * 32, "lut length mismatch");

    let mut acc_f32 = [0.0f32; BLOCK];
    let mut acc_u32 = [0u32; BLOCK];
    let mut window: usize = 0;
    for g in 0..n_groups {
        let lut_lo = &lut_u8[g * 32..g * 32 + 16];
        let lut_hi = &lut_u8[g * 32 + 16..g * 32 + 32];
        let codes_g = &codes[g * BLOCK..(g + 1) * BLOCK];
        for v in 0..BLOCK {
            let code = codes_g[v];
            let low = (code & 0x0F) as usize;
            let high = (code >> 4) as usize;
            acc_u32[v] += u32::from(lut_lo[low]) + u32::from(lut_hi[high]);
        }
        window += 1;
        if window == FLUSH_EVERY {
            for v in 0..BLOCK {
                acc_f32[v] += acc_u32[v] as f32;
                acc_u32[v] = 0;
            }
            window = 0;
        }
    }
    if window != 0 {
        for v in 0..BLOCK {
            acc_f32[v] += acc_u32[v] as f32;
        }
    }
    // Reconstruct f32 scores: undo the LUT quantisation and add back
    // the per-group bias the quantizer subtracted when building the u8 LUT.
    let bias_total = bias_per_group * n_groups as f32;
    for v in 0..BLOCK {
        out[v] = acc_f32[v] * inv_scale + bias_total;
    }
}

/// Scalar block-SIMD scoring reference. Computes inner-product
/// scores for `BLOCK` (32) vectors against a single query whose LUT
/// has already been built via [`build_tq4_lut_f32`]. The result is
/// the unscaled per-vector sum-of-LUT-contributions; the caller
/// multiplies by each vector's `scales[v]` factor as usual for
/// TurboQuant.
///
/// # Panics
///
/// Panics if any input is the wrong length.
pub fn tq4_block32_score_scalar(
    codes: &[u8], // n_groups * BLOCK bytes, interleaved
    lut: &[f32],  // n_groups * 32 floats, built by build_tq4_lut_f32
    dim: usize,
    out: &mut [f32; BLOCK],
) {
    assert_eq!(dim % 2, 0, "dim must be even for 4-bit codes");
    let n_groups = dim / 2;
    assert_eq!(codes.len(), n_groups * BLOCK, "codes length mismatch");
    assert_eq!(lut.len(), n_groups * 32, "lut length mismatch");

    let mut acc = [0.0f32; BLOCK];
    for g in 0..n_groups {
        let lut_lo = &lut[g * 32..g * 32 + 16];
        let lut_hi = &lut[g * 32 + 16..g * 32 + 32];
        let codes_g = &codes[g * BLOCK..(g + 1) * BLOCK];
        for v in 0..BLOCK {
            let code = codes_g[v];
            let low = (code & 0x0F) as usize;
            let high = (code >> 4) as usize;
            acc[v] += lut_lo[low] + lut_hi[high];
        }
    }
    *out = acc;
}

/// NEON block scoring against a u8 LUT.
///
/// Scores 32 vectors in parallel per byte-group via two
/// `vqtbl1q_u8` lookups (low + high nibble) per half-block, mirroring
/// the structure of turbovec's `score_4bit_block_neon`. Accumulates
/// into u16 with a flush to f32 every `FLUSH_EVERY` byte-groups so
/// the u16 lanes never overflow. Final reconstruction applies the
/// `inv_scale` / `bias_per_group` returned by
/// [`quantize_tq4_lut_u8`].
///
/// # Panics
///
/// Panics if any input is the wrong length.
#[cfg(target_arch = "aarch64")]
pub fn tq4_block32_score_u8_neon(
    codes: &[u8],
    lut_u8: &[u8],
    inv_scale: f32,
    bias_per_group: f32,
    dim: usize,
    out: &mut [f32; BLOCK],
) {
    use std::arch::aarch64::{
        vaddq_f32, vaddq_u8, vaddw_u8, vandq_u8, vcvtq_f32_u32, vdupq_n_f32, vdupq_n_u8,
        vdupq_n_u16, vfmaq_f32, vget_high_u8, vget_high_u16, vget_low_u8, vget_low_u16, vld1q_u8,
        vmovl_u16, vqtbl1q_u8, vshrq_n_u8, vst1q_f32,
    };
    assert_eq!(dim % 2, 0, "dim must be even for 4-bit codes");
    let n_groups = dim / 2;
    assert_eq!(codes.len(), n_groups * BLOCK, "codes length mismatch");
    assert_eq!(lut_u8.len(), n_groups * 32, "lut length mismatch");

    // SAFETY: `codes`, `lut_u8`, `out` are all bounds-checked above.
    // - `codes.as_ptr().add(g * 32)` and the same + 16 read 16 bytes
    //   each for `g in 0..n_groups`; the maximum end offset is
    //   `(n_groups - 1) * 32 + 32 = n_groups * 32 = codes.len()`.
    // - `lut_u8.as_ptr().add(g * 32)` and the same + 16 read 16
    //   bytes each within `lut_u8.len() = n_groups * 32`.
    // - `out.as_mut_ptr().add(i * 4)` for `i in 0..8` writes 4 f32
    //   at offset `i * 4`; final end at offset 32 = BLOCK.
    // - NEON intrinsics are baseline on aarch64; vqtbl1q_u8 is total
    //   (out-of-range indices return zero per ARM spec).
    unsafe {
        let mask = vdupq_n_u8(0x0F);
        let mut fa = [vdupq_n_f32(0.0); 8];
        let mut accum_u16 = [vdupq_n_u16(0); 4];
        let mut window: usize = 0;

        for g in 0..n_groups {
            let lut_base = lut_u8.as_ptr().add(g * 32);
            let lut_lo_v = vld1q_u8(lut_base);
            let lut_hi_v = vld1q_u8(lut_base.add(16));

            let codes_base = codes.as_ptr().add(g * 32);
            let c0 = vld1q_u8(codes_base);
            let c1 = vld1q_u8(codes_base.add(16));

            let s0 = vaddq_u8(
                vqtbl1q_u8(lut_lo_v, vandq_u8(c0, mask)),
                vqtbl1q_u8(lut_hi_v, vshrq_n_u8::<4>(c0)),
            );
            let s1 = vaddq_u8(
                vqtbl1q_u8(lut_lo_v, vandq_u8(c1, mask)),
                vqtbl1q_u8(lut_hi_v, vshrq_n_u8::<4>(c1)),
            );

            accum_u16[0] = vaddw_u8(accum_u16[0], vget_low_u8(s0));
            accum_u16[1] = vaddw_u8(accum_u16[1], vget_high_u8(s0));
            accum_u16[2] = vaddw_u8(accum_u16[2], vget_low_u8(s1));
            accum_u16[3] = vaddw_u8(accum_u16[3], vget_high_u8(s1));

            window += 1;
            if window == FLUSH_EVERY {
                for i in 0..4 {
                    let lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(accum_u16[i])));
                    let hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(accum_u16[i])));
                    fa[i * 2] = vaddq_f32(fa[i * 2], lo);
                    fa[i * 2 + 1] = vaddq_f32(fa[i * 2 + 1], hi);
                    accum_u16[i] = vdupq_n_u16(0);
                }
                window = 0;
            }
        }
        if window != 0 {
            for i in 0..4 {
                let lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(accum_u16[i])));
                let hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(accum_u16[i])));
                fa[i * 2] = vaddq_f32(fa[i * 2], lo);
                fa[i * 2 + 1] = vaddq_f32(fa[i * 2 + 1], hi);
            }
        }

        let inv_scale_v = vdupq_n_f32(inv_scale);
        let bias_v = vdupq_n_f32(bias_per_group * n_groups as f32);
        for (i, fa_i) in fa.iter().enumerate() {
            let scaled = vfmaq_f32(bias_v, *fa_i, inv_scale_v);
            vst1q_f32(out.as_mut_ptr().add(i * 4), scaled);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
/// TQ4 block scorer for AVX2.
///
/// # Safety
///
/// The current CPU must support AVX2.
pub unsafe fn tq4_block32_score_u8_avx2(
    codes: &[u8],
    lut_u8: &[u8],
    inv_scale: f32,
    bias_per_group: f32,
    dim: usize,
    out: &mut [f32; BLOCK],
) {
    use std::arch::x86_64::{
        _mm_loadu_si128, _mm256_add_epi8, _mm256_add_epi16, _mm256_and_si256,
        _mm256_broadcastsi128_si256, _mm256_castsi256_si128, _mm256_cvtepu8_epi16,
        _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_set1_epi8, _mm256_setzero_si256,
        _mm256_shuffle_epi8, _mm256_srli_epi16, _mm256_storeu_si256,
    };

    assert_eq!(dim % 2, 0, "dim must be even for 4-bit codes");
    let n_groups = dim / 2;
    assert_eq!(codes.len(), n_groups * BLOCK, "codes length mismatch");
    assert_eq!(lut_u8.len(), n_groups * 32, "lut length mismatch");

    // SAFETY: the caller selected AVX2. Bounds are validated above: each loop
    // reads one 32-byte block from `codes` and two 16-byte LUTs. The u16
    // accumulators flush every 32 groups, below their overflow limit.
    unsafe {
        let mask = _mm256_set1_epi8(0x0f);
        let zero = _mm256_setzero_si256();
        let mut accumulated = [zero; 2];
        let mut scores = [0.0f32; BLOCK];
        let mut window = 0;

        for g in 0..n_groups {
            let lut_base = lut_u8.as_ptr().add(g * 32);
            let lut_lo = _mm256_broadcastsi128_si256(_mm_loadu_si128(lut_base.cast()));
            let lut_hi = _mm256_broadcastsi128_si256(_mm_loadu_si128(lut_base.add(16).cast()));
            let code = _mm256_loadu_si256(codes.as_ptr().add(g * BLOCK).cast());
            let low = _mm256_and_si256(code, mask);
            let high = _mm256_and_si256(_mm256_srli_epi16(code, 4), mask);
            let sum = _mm256_add_epi8(
                _mm256_shuffle_epi8(lut_lo, low),
                _mm256_shuffle_epi8(lut_hi, high),
            );
            accumulated[0] = _mm256_add_epi16(
                accumulated[0],
                _mm256_cvtepu8_epi16(_mm256_castsi256_si128(sum)),
            );
            accumulated[1] = _mm256_add_epi16(
                accumulated[1],
                _mm256_cvtepu8_epi16(_mm256_extracti128_si256(sum, 1)),
            );

            window += 1;
            if window == FLUSH_EVERY {
                for (block, acc) in accumulated.iter_mut().enumerate() {
                    let mut lanes = [0u16; 16];
                    _mm256_storeu_si256(lanes.as_mut_ptr().cast(), *acc);
                    for (lane, value) in lanes.into_iter().enumerate() {
                        scores[block * 16 + lane] += f32::from(value);
                    }
                    *acc = zero;
                }
                window = 0;
            }
        }
        if window != 0 {
            for (block, acc) in accumulated.iter().enumerate() {
                let mut lanes = [0u16; 16];
                _mm256_storeu_si256(lanes.as_mut_ptr().cast(), *acc);
                for (lane, value) in lanes.into_iter().enumerate() {
                    scores[block * 16 + lane] += f32::from(value);
                }
            }
        }
        let bias = bias_per_group * n_groups as f32;
        for (out, score) in out.iter_mut().zip(scores) {
            *out = score * inv_scale + bias;
        }
    }
}

/// AVX-512 TQ4 block scorer. Same `vpshufb` lookup as the AVX2 kernel, but a
/// zmm holds two groups' worth of codes, so each iteration consumes two
/// groups and two LUTs (broadcast into the matching 128-bit lanes, since
/// `vpshufb` is per-lane). Both halves accumulate into the same 32 rows.
///
/// # Safety
///
/// The current CPU must support AVX-512F, AVX-512BW and AVX2.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[target_feature(enable = "avx512f,avx512bw,avx2")]
pub unsafe fn tq4_block32_score_u8_avx512(
    codes: &[u8],
    lut_u8: &[u8],
    inv_scale: f32,
    bias_per_group: f32,
    dim: usize,
    out: &mut [f32; BLOCK],
) {
    use std::arch::x86_64::{
        __m512i, _mm_loadu_si128, _mm256_broadcastsi128_si256, _mm512_add_epi8, _mm512_add_epi16,
        _mm512_and_si512, _mm512_castsi256_si512, _mm512_castsi512_si256, _mm512_cvtepu8_epi16,
        _mm512_extracti64x4_epi64, _mm512_inserti64x4, _mm512_loadu_si512, _mm512_set1_epi8,
        _mm512_setzero_si512, _mm512_shuffle_epi8, _mm512_srli_epi16, _mm512_storeu_si512,
    };

    assert_eq!(dim % 2, 0, "dim must be even for 4-bit codes");
    let n_groups = dim / 2;
    assert_eq!(codes.len(), n_groups * BLOCK, "codes length mismatch");
    assert_eq!(lut_u8.len(), n_groups * 32, "lut length mismatch");

    let pairs = n_groups / 2;
    let mut scores = [0.0f32; BLOCK];
    // SAFETY: the caller selected AVX-512F/BW and AVX2. Bounds are validated
    // above: each iteration reads two 32-byte code blocks at `g * BLOCK` with
    // `g + 1 < n_groups`, and four 16-byte LUTs inside `n_groups * 32`. The
    // u16 accumulators flush every `FLUSH_EVERY` groups, below overflow.
    unsafe {
        let mask = _mm512_set1_epi8(0x0f);
        let zero = _mm512_setzero_si512();
        // One accumulator per zmm half: both index rows 0..32, they are summed
        // together at flush time.
        let mut accumulated = [zero; 2];
        let mut window = 0;

        let widen_pair = |lo_ptr: *const u8, hi_ptr: *const u8| -> __m512i {
            let lo = _mm256_broadcastsi128_si256(_mm_loadu_si128(lo_ptr.cast()));
            let hi = _mm256_broadcastsi128_si256(_mm_loadu_si128(hi_ptr.cast()));
            _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1)
        };

        let flush = |accumulated: &mut [__m512i; 2], scores: &mut [f32; BLOCK]| {
            for acc in accumulated.iter_mut() {
                let mut lanes = [0u16; 32];
                _mm512_storeu_si512(lanes.as_mut_ptr().cast(), *acc);
                for (row, value) in lanes.into_iter().enumerate() {
                    scores[row] += f32::from(value);
                }
                *acc = zero;
            }
        };

        for pair in 0..pairs {
            let g = pair * 2;
            let lut_base = lut_u8.as_ptr().add(g * 32);
            let lut_lo = widen_pair(lut_base, lut_base.add(32));
            let lut_hi = widen_pair(lut_base.add(16), lut_base.add(48));
            let code = _mm512_loadu_si512(codes.as_ptr().add(g * BLOCK).cast());
            let low = _mm512_and_si512(code, mask);
            let high = _mm512_and_si512(_mm512_srli_epi16::<4>(code), mask);
            let sum = _mm512_add_epi8(
                _mm512_shuffle_epi8(lut_lo, low),
                _mm512_shuffle_epi8(lut_hi, high),
            );
            accumulated[0] = _mm512_add_epi16(
                accumulated[0],
                _mm512_cvtepu8_epi16(_mm512_castsi512_si256(sum)),
            );
            accumulated[1] = _mm512_add_epi16(
                accumulated[1],
                _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(sum)),
            );

            window += 2;
            if window >= FLUSH_EVERY {
                flush(&mut accumulated, &mut scores);
                window = 0;
            }
        }
        if window != 0 {
            flush(&mut accumulated, &mut scores);
        }
        // Odd group count: the leftover group has no partner to fill the
        // upper half, so it goes through the scalar reference.
        if n_groups % 2 == 1 {
            let g = n_groups - 1;
            for (row, score) in scores.iter_mut().enumerate() {
                let code = codes[g * BLOCK + row];
                let low = usize::from(code & 0x0f);
                let high = usize::from(code >> 4);
                *score += f32::from(lut_u8[g * 32 + low]) + f32::from(lut_u8[g * 32 + 16 + high]);
            }
        }
    }
    let bias = bias_per_group * n_groups as f32;
    for (out, score) in out.iter_mut().zip(scores) {
        *out = score * inv_scale + bias;
    }
}

/// Score one TQ4 block through the best available kernel for this CPU.
pub fn tq4_block32_score_u8(
    codes: &[u8],
    lut_u8: &[u8],
    inv_scale: f32,
    bias_per_group: f32,
    dim: usize,
    out: &mut [f32; BLOCK],
) {
    // No AVX-512 arm. The rule this crate follows is that AVX-512 earns its
    // place only where the ISA gives an instruction AVX2 lacks: VNNI for the
    // int8 dot, VPOPCNTDQ for hamming, a mask register for tq1 and
    // flip_signs, `vpermps` over a 16-entry table for the ADC. This kernel is
    // `vpshufb` either way, so 512-bit registers only add pressure: measured
    // on a Threadripper 7960X (Zen 4, 2026-08-09) at 681 ns against AVX2's
    // 600 ns in the quietest of three runs, and never faster in any of them.
    // `tq4_block32_score_u8_avx512` stays public, tested and benched.
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 was checked immediately above.
            unsafe {
                tq4_block32_score_u8_avx2(codes, lut_u8, inv_scale, bias_per_group, dim, out);
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        tq4_block32_score_u8_neon(codes, lut_u8, inv_scale, bias_per_group, dim, out);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        tq4_block32_score_u8_scalar(codes, lut_u8, inv_scale, bias_per_group, dim, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Equivalence: the scalar block kernel must produce, for each
    /// vector in a 32-vector block, the same inner-product score the
    /// row-major path would produce (modulo float associativity).
    ///
    /// The row-major reference here is the explicit dot product
    /// between `q_rot` and the f32 centroid values referenced by the
    /// codes; the production kernel `tq4_adc_i8_neon` quantises the
    /// centroids to i8 first, which loses some precision and is
    /// covered by a separate equivalence test in `lib.rs`. The
    /// scalar block kernel works in f32 throughout, so it must match
    /// the f32 reference to ~1e-5.
    #[test]
    fn block32_matches_row_dot_product() {
        // Deterministic small synthetic case.
        let dim = 32;
        let n_groups = dim / 2; // 16 byte-groups
        // Centroids: spread out, deterministic.
        let centroids: [f32; 16] = [
            -1.0, -0.85, -0.65, -0.48, -0.32, -0.18, -0.08, -0.02, 0.02, 0.08, 0.18, 0.32, 0.48,
            0.65, 0.85, 1.0,
        ];

        // 32 row-major code sequences, each of length n_groups.
        let mut rows: Vec<Vec<u8>> = (0..BLOCK)
            .map(|v| {
                (0..n_groups)
                    .map(|g| {
                        let low = ((v + g) % 16) as u8;
                        let high = ((v * 3 + g) % 16) as u8;
                        (high << 4) | low
                    })
                    .collect()
            })
            .collect();

        // A deterministic non-trivial query.
        let q_rot: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.137).sin()).collect();

        // Build the LUT once.
        let mut lut = vec![0.0f32; n_groups * 32];
        build_tq4_lut_f32(&q_rot, &centroids, dim, &mut lut);

        // Interleave the codes.
        let mut codes_block = vec![0u8; n_groups * BLOCK];
        let row_refs: Vec<&[u8]> = rows.iter_mut().map(|r| r.as_slice()).collect();
        interleave_tq4_codes(&row_refs, dim, &mut codes_block);

        // Score via the block kernel.
        let mut block_scores = [0.0f32; BLOCK];
        tq4_block32_score_scalar(&codes_block, &lut, dim, &mut block_scores);

        // Reference: per-vector explicit dot product against the
        // centroid values keyed by the codes.
        for v in 0..BLOCK {
            let mut expected = 0.0f32;
            for g in 0..n_groups {
                let code = rows[v][g];
                let low = (code & 0x0F) as usize;
                let high = (code >> 4) as usize;
                expected += q_rot[2 * g] * centroids[low];
                expected += q_rot[2 * g + 1] * centroids[high];
            }
            let got = block_scores[v];
            let delta = (got - expected).abs();
            assert!(
                delta < 1e-4,
                "vector {v}: expected {expected}, got {got} (delta {delta})"
            );
        }
    }

    /// The u8 LUT path must produce scores close to the f32 LUT path.
    /// The quantisation step adds at most `range / 255 * n_groups *
    /// 2` of cumulative error, which for the synthetic case below
    /// is well below 1% of the score magnitude.
    #[test]
    fn block32_u8_path_matches_f32_path() {
        let dim = 128;
        let n_groups = dim / 2;
        let centroids: [f32; 16] = [
            -0.9, -0.75, -0.6, -0.45, -0.3, -0.15, -0.05, -0.01, 0.01, 0.05, 0.15, 0.3, 0.45, 0.6,
            0.75, 0.9,
        ];
        let rows: Vec<Vec<u8>> = (0..BLOCK)
            .map(|v| {
                (0..n_groups)
                    .map(|g| {
                        let low = ((v * 5 + g) % 16) as u8;
                        let high = ((v + g * 7) % 16) as u8;
                        (high << 4) | low
                    })
                    .collect()
            })
            .collect();
        let q_rot: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.07).cos()).collect();

        // Build both LUTs and the quantised companion.
        let mut lut_f32 = vec![0.0f32; n_groups * 32];
        build_tq4_lut_f32(&q_rot, &centroids, dim, &mut lut_f32);
        let mut lut_u8 = vec![0u8; n_groups * 32];
        let (inv_scale, bias_per_group) = quantize_tq4_lut_u8(&lut_f32, &mut lut_u8);

        // Interleave the codes.
        let mut codes_block = vec![0u8; n_groups * BLOCK];
        let row_refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
        interleave_tq4_codes(&row_refs, dim, &mut codes_block);

        // Score via both paths.
        let mut scores_f32 = [0.0f32; BLOCK];
        let mut scores_u8 = [0.0f32; BLOCK];
        tq4_block32_score_scalar(&codes_block, &lut_f32, dim, &mut scores_f32);
        tq4_block32_score_u8_scalar(
            &codes_block,
            &lut_u8,
            inv_scale,
            bias_per_group,
            dim,
            &mut scores_u8,
        );

        // Equivalence: the u8 path is a quantised approximation of
        // the f32 path. Per-vector absolute error must stay under
        // a budget proportional to `n_groups * inv_scale`, which is
        // the worst-case accumulated rounding from the LUT
        // quantisation.
        let budget = n_groups as f32 * inv_scale * 2.0; // 2 per byte-group (low + high nibble)
        for v in 0..BLOCK {
            let delta = (scores_f32[v] - scores_u8[v]).abs();
            assert!(
                delta < budget,
                "vector {v}: f32={} u8={} delta={} budget={}",
                scores_f32[v],
                scores_u8[v],
                delta,
                budget
            );
        }
    }

    /// NEON kernel equivalence: must match the scalar u8 path bit
    /// for bit when running over the same `(codes, lut_u8,
    /// inv_scale, bias_per_group)` input.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn block32_neon_matches_u8_scalar() {
        let dim = 256;
        let n_groups = dim / 2;
        let centroids: [f32; 16] = [
            -0.95, -0.8, -0.65, -0.5, -0.35, -0.2, -0.1, -0.03, 0.03, 0.1, 0.2, 0.35, 0.5, 0.65,
            0.8, 0.95,
        ];
        let rows: Vec<Vec<u8>> = (0..BLOCK)
            .map(|v| {
                (0..n_groups)
                    .map(|g| {
                        let low = ((v + g * 5) % 16) as u8;
                        let high = ((v * 11 + g * 3) % 16) as u8;
                        (high << 4) | low
                    })
                    .collect()
            })
            .collect();
        let q_rot: Vec<f32> = (0..dim)
            .map(|i| ((i as f32) * 0.123).sin() + ((i as f32) * 0.077).cos() * 0.5)
            .collect();

        let mut lut_f32 = vec![0.0f32; n_groups * 32];
        build_tq4_lut_f32(&q_rot, &centroids, dim, &mut lut_f32);
        let mut lut_u8 = vec![0u8; n_groups * 32];
        let (inv_scale, bias_per_group) = quantize_tq4_lut_u8(&lut_f32, &mut lut_u8);

        let mut codes_block = vec![0u8; n_groups * BLOCK];
        let row_refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
        interleave_tq4_codes(&row_refs, dim, &mut codes_block);

        let mut scores_scalar = [0.0f32; BLOCK];
        let mut scores_neon = [0.0f32; BLOCK];
        tq4_block32_score_u8_scalar(
            &codes_block,
            &lut_u8,
            inv_scale,
            bias_per_group,
            dim,
            &mut scores_scalar,
        );
        tq4_block32_score_u8_neon(
            &codes_block,
            &lut_u8,
            inv_scale,
            bias_per_group,
            dim,
            &mut scores_neon,
        );

        for v in 0..BLOCK {
            let delta = (scores_scalar[v] - scores_neon[v]).abs();
            assert!(
                delta < 1e-3,
                "vector {v}: scalar={} neon={} delta={}",
                scores_scalar[v],
                scores_neon[v],
                delta
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn block32_avx2_matches_u8_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let dim = 258;
        let n_groups = dim / 2;
        let codes: Vec<u8> = (0..n_groups * BLOCK)
            .map(|i| (i as u8).wrapping_mul(37))
            .collect();
        let lut: Vec<u8> = (0..n_groups * 32)
            .map(|i| ((i * 13 + 7) % 128) as u8)
            .collect();
        let inv_scale = 0.013;
        let bias_per_group = -1.7;
        let mut scalar = [0.0; BLOCK];
        let mut avx2 = [0.0; BLOCK];
        tq4_block32_score_u8_scalar(&codes, &lut, inv_scale, bias_per_group, dim, &mut scalar);
        unsafe {
            tq4_block32_score_u8_avx2(&codes, &lut, inv_scale, bias_per_group, dim, &mut avx2);
        }
        assert_eq!(avx2, scalar);
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[test]
    fn block32_avx512_matches_u8_scalar() {
        if !(std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx2"))
        {
            return;
        }
        // 258 gives 129 groups (odd, so the scalar leftover group runs) and
        // 256 gives 128 (even, and a multiple of FLUSH_EVERY).
        for dim in [258usize, 256] {
            let n_groups = dim / 2;
            let codes: Vec<u8> = (0..n_groups * BLOCK)
                .map(|i| (i as u8).wrapping_mul(37))
                .collect();
            let lut: Vec<u8> = (0..n_groups * 32)
                .map(|i| ((i * 13 + 7) % 128) as u8)
                .collect();
            let inv_scale = 0.013;
            let bias_per_group = -1.7;
            let mut scalar = [0.0; BLOCK];
            let mut avx512 = [0.0; BLOCK];
            tq4_block32_score_u8_scalar(&codes, &lut, inv_scale, bias_per_group, dim, &mut scalar);
            unsafe {
                tq4_block32_score_u8_avx512(
                    &codes,
                    &lut,
                    inv_scale,
                    bias_per_group,
                    dim,
                    &mut avx512,
                );
            }
            assert_eq!(avx512, scalar, "dim {dim}");
        }
    }

    /// Layout invariant: interleaving then de-interleaving should
    /// round-trip the codes.
    #[test]
    fn interleave_round_trip() {
        let dim = 16;
        let n_groups = dim / 2;
        let rows: Vec<Vec<u8>> = (0..BLOCK)
            .map(|v| {
                (0..n_groups)
                    .map(|g| ((v * 7 + g * 3) % 256) as u8)
                    .collect()
            })
            .collect();

        let mut blocked = vec![0u8; n_groups * BLOCK];
        let row_refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
        interleave_tq4_codes(&row_refs, dim, &mut blocked);

        // De-interleave: codes[g * BLOCK + v] must equal rows[v][g].
        for v in 0..BLOCK {
            for g in 0..n_groups {
                let expected = rows[v][g];
                let got = blocked[g * BLOCK + v];
                assert_eq!(
                    got, expected,
                    "round-trip failed at v={v} g={g}: expected {expected}, got {got}"
                );
            }
        }
    }
}
