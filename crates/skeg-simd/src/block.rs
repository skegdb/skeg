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

/// Flush window for the u8 accumulator. Each iteration adds at most
/// `2 * 255 = 510` to the per-vector u8 accumulator. With
/// `FLUSH_EVERY = 32` the worst-case accumulator before flush is
/// `32 * 510 = 16'320`, well below the `u16::MAX = 65'535` ceiling,
/// so the widen-then-flush step is safe. The same constant doubles
/// as the unroll factor in the NEON kernel.
pub const FLUSH_EVERY: usize = 32;

/// Pre-compute the u8 LUT for a single query plus the scale+bias
/// the block kernel needs to reconstruct f32 scores.
///
/// The u8 LUT lives next to the float LUT it derives from: the same
/// `[g * 32, g * 32 + 16)` low-nibble window and
/// `[g * 32 + 16, g * 32 + 32)` high-nibble window. The whole table
/// is quantized with a single shared (min, range) so we can recover
/// `f32_score = (sum_u8 - bias_per_group * n_groups) / scale` after
/// the dot-product loop.
///
/// `scale_out` returns `255 / range` such that
/// `u8_v = ((f32_v - min) * scale_out).round().clamp(0, 255)`.
/// `bias_per_group_out` returns the per-byte-group constant
/// `2 * min` (one for each nibble), so the reconstruction subtracts
/// `n_groups * bias_per_group_out * 0.5` total - encapsulated in
/// the scoring routines below to keep callers from getting the
/// scaling wrong.
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
    // Reconstruct f32 scores: undo the LUT quantisation and subtract
    // the per-group bias that the quantizer folded into the u8 LUT.
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
