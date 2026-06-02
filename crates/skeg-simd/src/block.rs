//! Block-SIMD scoring for 4-bit TurboQuant codes.
//!
//! Step 1 of the block-kernel plan (`skeg-internal/bench-compare/
//! BLOCK-KERNEL-PLAN.md`): scalar reference and codes-layout helpers,
//! gated by an equivalence test against [`tq4_adc_i8_scalar`]. NEON
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
