//! Asymmetric distance computation for TurboQuant codes: tq1 (1-bit,
//! mask-sum), tq2 (2-bit) and tq4 (4-bit, both a centroid-table lookup).
//! One module so the four ISA implementations of a kernel sit together.

/// tq1 bit-plane proxy: `(sum_p 2^p * popcount(plane_p AND code), popcount(code))`.
/// `b+1` AND-popcount passes over `bytes`-byte masks (the tq1 default nav kernel).
#[must_use]
pub fn tq1_bitplane_score_scalar(planes: &[u8], b: u8, bytes: usize, code: &[u8]) -> (u64, u32) {
    assert_tq1_bitplane_layout(planes, b, bytes, code);
    let code_pc: u32 = code.iter().map(|c| c.count_ones()).sum();
    let mut weighted = 0u64;
    for p in 0..b as usize {
        let plane = &planes[p * bytes..(p + 1) * bytes];
        let pc: u64 = plane
            .iter()
            .zip(code)
            .map(|(&pl, &cd)| u64::from((pl & cd).count_ones()))
            .sum();
        weighted += pc << p;
    }
    (weighted, code_pc)
}

/// tq1 bit-plane proxy, NEON. One AND-popcount kernel serves both terms:
/// `popcount(code) == popcount(code AND code)`.
#[cfg(target_arch = "aarch64")]
#[must_use]
pub fn tq1_bitplane_score_neon(planes: &[u8], b: u8, bytes: usize, code: &[u8]) -> (u64, u32) {
    assert_tq1_bitplane_layout(planes, b, bytes, code);
    let code_pc = crate::distance::and_popcnt_neon(code, code);
    let mut weighted = 0u64;
    for p in 0..b as usize {
        let plane = &planes[p * bytes..(p + 1) * bytes];
        weighted += u64::from(crate::distance::and_popcnt_neon(plane, code)) << p;
    }
    (weighted, code_pc)
}

/// tq1 bit-plane proxy: dispatches to AVX2 or NEON when available.
///
/// # Panics
///
/// Panics if `planes.len() != b as usize * bytes` or `code.len() != bytes`.
#[must_use]
pub fn tq1_bitplane_score(planes: &[u8], b: u8, bytes: usize, code: &[u8]) -> (u64, u32) {
    assert_tq1_bitplane_layout(planes, b, bytes, code);
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vpopcntdq")
        {
            // SAFETY: all three target features were checked immediately
            // above. Slices are derived from the caller-validated TQ1 layout.
            let code_pc = unsafe { crate::distance::and_popcnt_avx512(code, code) };
            let mut weighted = 0u64;
            for p in 0..b as usize {
                let plane = &planes[p * bytes..(p + 1) * bytes];
                weighted +=
                    u64::from(unsafe { crate::distance::and_popcnt_avx512(plane, code) }) << p;
            }
            return (weighted, code_pc);
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 was checked immediately above. Slices are derived
            // from the caller-validated TQ1 layout.
            let code_pc = unsafe { crate::distance::and_popcnt_avx2(code, code) };
            let mut weighted = 0u64;
            for p in 0..b as usize {
                let plane = &planes[p * bytes..(p + 1) * bytes];
                weighted +=
                    u64::from(unsafe { crate::distance::and_popcnt_avx2(plane, code) }) << p;
            }
            return (weighted, code_pc);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        tq1_bitplane_score_neon(planes, b, bytes, code)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        tq1_bitplane_score_scalar(planes, b, bytes, code)
    }
}

/// Validate the bit-plane layout shared by the scalar, NEON and dispatched
/// TQ1 navigation scores before any SIMD kernel uses one slice as another's
/// loop bound.
#[inline]
fn assert_tq1_bitplane_layout(planes: &[u8], b: u8, bytes: usize, code: &[u8]) {
    let planes_len = usize::from(b)
        .checked_mul(bytes)
        .expect("tq1_bitplane_score plane layout overflows usize");
    assert_eq!(
        planes.len(),
        planes_len,
        "tq1_bitplane_score planes length / (b, bytes) mismatch"
    );
    assert_eq!(
        code.len(),
        bytes,
        "tq1_bitplane_score code length / bytes mismatch"
    );
}

/// The tq1 counterpart of [`assert_adc_layout`]: 1-bit codes pack 8 coordinates
/// per byte. Every tq1 kernel calls this, not only the dispatcher: the NEON one
/// is a safe public function that indexes with `get_unchecked` and raw pointer
/// loads, so a caller reaching it directly with a short slice would be
/// undefined behaviour rather than a panic.
#[inline]
fn assert_tq1_layout(code: &[u8], q_rot: &[f32], dim: usize) {
    assert_eq!(dim % 8, 0, "tq1 dim must be a multiple of 8");
    assert_eq!(code.len(), dim / 8, "tq1 code length / dim mismatch");
    assert_eq!(q_rot.len(), dim, "tq1 q_rot length / dim mismatch");
}

/// Reject an input whose packing does not line up before any kernel touches it.
///
/// Codes are bit-packed `BITS` per coordinate, so a dimension that is not a
/// whole number of bytes has no representation. A length check alone does not
/// catch it: `dim * BITS / 8` truncates, so `dim = 1` with 4-bit codes expects
/// zero bytes and an empty slice passes. The AVX-512 tail would then mask in a
/// live byte and read past the slice. `tq1_masked_sum` has always checked this;
/// tq2 and tq4 did not.
#[inline]
fn assert_adc_layout<const BITS: usize>(code: &[u8], q_rot: &[f32], dim: usize) {
    const { assert!(BITS == 2 || BITS == 4, "only 2-bit and 4-bit codes") }
    let per_byte = 8 / BITS;
    assert_eq!(q_rot.len(), dim, "q_rot length / dim mismatch");
    assert_eq!(
        dim % per_byte,
        0,
        "dim must be a multiple of {per_byte} for {BITS}-bit codes"
    );
    assert_eq!(code.len(), dim * BITS / 8, "code length / dim mismatch");
}

/// Bucket index of coordinate `i` in a `BITS`-wide packed code.
#[inline(always)]
fn code_index<const BITS: usize>(code: &[u8], i: usize) -> usize {
    const { assert!(BITS == 2 || BITS == 4, "only 2-bit and 4-bit codes") }
    let per_byte = 8 / BITS;
    let byte = code[i / per_byte];
    usize::from((byte >> ((i % per_byte) * BITS)) & ((1u8 << BITS) - 1))
}

/// Scalar ADC over a `BITS`-wide code. The tier-specific
/// [`tq2_adc_i8_scalar`] and [`tq4_adc_i8_scalar`] wrap this.
///
/// # Panics
///
/// Panics unless the input is exactly representable: `q_rot.len() == dim`,
/// `dim` a multiple of `8 / BITS`, and `code.len() == dim * BITS / 8`. See
/// [`assert_adc_layout`].
#[must_use]
pub fn adc_i8_scalar<const BITS: usize>(
    code: &[u8],
    centroids_i8: &[i8; 16],
    i8_scale: f32,
    q_rot: &[f32],
    dim: usize,
) -> f32 {
    assert_adc_layout::<BITS>(code, q_rot, dim);
    let centroids_f32: [f32; 16] = std::array::from_fn(|i| f32::from(centroids_i8[i]));
    let mut acc = 0.0f32;
    for i in 0..dim {
        acc += q_rot[i] * (centroids_f32[code_index::<BITS>(code, i)] * i8_scale);
    }
    acc
}

/// Expand 16 packed `BITS`-wide codes starting at coordinate `base` into
/// 16 table indices, one per byte lane.
///
/// # Safety
///
/// `code` must hold at least `(base + 16) * BITS / 8` bytes.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn unpack_indices_neon<const BITS: usize>(
    code: &[u8],
    base: usize,
) -> std::arch::aarch64::uint8x16_t {
    use std::arch::aarch64::{
        vand_u8, vandq_u8, vcombine_u8, vcreate_u8, vdup_n_u8, vdupq_n_u8, vld1_u8, vld1q_s8,
        vld1q_u8, vqtbl1q_u8, vshlq_u8, vshr_n_u8, vzip1_u8, vzip2_u8,
    };
    const { assert!(BITS == 2 || BITS == 4, "only 2-bit and 4-bit codes") }
    // SAFETY: the caller guarantees the byte range; every load below reads
    // `16 * BITS / 8` bytes starting at `base * BITS / 8`.
    unsafe {
        if BITS == 4 {
            // 8 bytes -> 16 nibbles: low nibbles on even lanes, high on odd.
            let packed = vld1_u8(code.as_ptr().add(base / 2));
            let low = vand_u8(packed, vdup_n_u8(0x0F));
            let high = vshr_n_u8::<4>(packed);
            vcombine_u8(vzip1_u8(low, high), vzip2_u8(low, high))
        } else {
            // 4 bytes -> 16 pairs: each byte feeds four consecutive lanes.
            // `vld1_u8` reads a full 8-byte D register, which would overread
            // the `(base + 16) * BITS / 8 == base / 4 + 4` bytes the caller
            // actually guarantees; `vcreate_u8` builds the register from a
            // u64 with no memory access, so only the bounds-checked 4-byte
            // slice read below can fault.
            let word = u32::from_le_bytes(
                code[base / 4..base / 4 + 4]
                    .try_into()
                    .expect("slice is exactly 4 bytes"),
            );
            let packed = vcreate_u8(u64::from(word));
            let combined = vcombine_u8(packed, packed);
            let spread = vqtbl1q_u8(
                combined,
                vld1q_u8([0u8, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3].as_ptr()),
            );
            let shifts =
                vld1q_s8([0i8, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6].as_ptr());
            vandq_u8(vshlq_u8(spread, shifts), vdupq_n_u8(0x03))
        }
    }
}

/// NEON ADC over a `BITS`-wide code, table held in a register.
///
/// # Panics
///
/// Panics unless the input is exactly representable: `q_rot.len() == dim`,
/// `dim` a multiple of `8 / BITS`, and `code.len() == dim * BITS / 8`. See
/// [`assert_adc_layout`].
#[cfg(target_arch = "aarch64")]
#[must_use]
pub fn adc_i8_neon<const BITS: usize>(
    code: &[u8],
    centroids_i8: &[i8; 16],
    i8_scale: f32,
    q_rot: &[f32],
    dim: usize,
) -> f32 {
    use std::arch::aarch64::{
        vaddq_f32, vaddvq_f32, vcvtq_f32_s32, vdupq_n_f32, vfmaq_f32, vget_high_s8, vget_high_s16,
        vget_low_s8, vget_low_s16, vld1q_f32, vld1q_s8, vmovl_s8, vmovl_s16, vqtbl1q_s8,
    };
    assert_adc_layout::<BITS>(code, q_rot, dim);
    let centroids_f32: [f32; 16] = std::array::from_fn(|i| f32::from(centroids_i8[i]));
    let block = dim - (dim % 16);
    // SAFETY: NEON is baseline on aarch64. Every `q_rot` load covers 4 lanes
    // inside `[base, base + 16)` with `base + 16 <= block <= dim`, the code
    // reads are bounded by the length assert above, and `vqtbl1q_s8` is total
    // (out-of-range indices yield zero, and 2- and 4-bit codes cannot exceed
    // 15 anyway).
    let mut acc = unsafe {
        let lut = vld1q_s8(centroids_i8.as_ptr());
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let mut base = 0;
        while base < block {
            let idx = unpack_indices_neon::<BITS>(code, base);
            let picked = vqtbl1q_s8(lut, idx);
            let lo = vmovl_s8(vget_low_s8(picked));
            let hi = vmovl_s8(vget_high_s8(picked));
            let c0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo)));
            let c1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo)));
            let c2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi)));
            let c3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi)));
            acc0 = vfmaq_f32(acc0, vld1q_f32(q_rot.as_ptr().add(base)), c0);
            acc1 = vfmaq_f32(acc1, vld1q_f32(q_rot.as_ptr().add(base + 4)), c1);
            acc2 = vfmaq_f32(acc2, vld1q_f32(q_rot.as_ptr().add(base + 8)), c2);
            acc3 = vfmaq_f32(acc3, vld1q_f32(q_rot.as_ptr().add(base + 12)), c3);
            base += 16;
        }
        vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)))
    };
    for i in block..dim {
        acc += q_rot[i] * centroids_f32[code_index::<BITS>(code, i)];
    }
    acc * i8_scale
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
    assert_adc_layout::<4>(code, q_rot, dim);
    #[cfg(target_arch = "aarch64")]
    {
        tq4_adc_i8_neon(code, centroids_i8, i8_scale, q_rot, dim)
    }
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "avx512")]
        {
            if avx512_adc_supported() {
                // SAFETY: every feature the kernel is compiled with was checked
                // immediately above, through the one shared predicate.
                return unsafe { tq4_adc_i8_avx512(code, centroids_i8, i8_scale, q_rot, dim) };
            }
        }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: both target features were checked immediately above.
            return unsafe { tq4_adc_i8_avx2(code, centroids_i8, i8_scale, q_rot, dim) };
        }
        tq4_adc_i8_scalar(code, centroids_i8, i8_scale, q_rot, dim)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
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

/// NEON kernel for [`tq4_adc_i8`].
#[cfg(target_arch = "aarch64")]
#[must_use]
pub fn tq4_adc_i8_neon(
    code: &[u8],
    centroids_i8: &[i8; 16],
    i8_scale: f32,
    q_rot: &[f32],
    dim: usize,
) -> f32 {
    adc_i8_neon::<4>(code, centroids_i8, i8_scale, q_rot, dim)
}

/// Expand 8 packed `BITS`-wide codes starting at `base` into 8 lane indices.
///
/// # Safety
///
/// The caller must have selected AVX2, and `code` must hold at least
/// `(base + 8) * BITS / 8` bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn unpack_indices_avx2<const BITS: usize>(
    code: &[u8],
    base: usize,
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::{
        _mm_loadu_si16, _mm_loadu_si32, _mm_set_epi8, _mm_shuffle_epi8, _mm_unpacklo_epi8,
        _mm256_and_si256, _mm256_blend_epi32, _mm256_cvtepu8_epi32, _mm256_set1_epi32,
        _mm256_setr_epi32, _mm256_srli_epi32, _mm256_srlv_epi32,
    };
    const { assert!(BITS == 2 || BITS == 4, "only 2-bit and 4-bit codes") }
    // SAFETY: the caller guarantees the feature and the byte range.
    unsafe {
        if BITS == 4 {
            let bytes = _mm_loadu_si32(code.as_ptr().add(base / 2));
            let doubled = _mm256_cvtepu8_epi32(_mm_unpacklo_epi8(bytes, bytes));
            let nibble = _mm256_set1_epi32(0x0F);
            let low = _mm256_and_si256(doubled, nibble);
            let high = _mm256_and_si256(_mm256_srli_epi32::<4>(doubled), nibble);
            _mm256_blend_epi32::<0b1010_1010>(low, high)
        } else {
            let bytes = _mm_loadu_si16(code.as_ptr().add(base / 4));
            let spread = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0);
            let doubled = _mm256_cvtepu8_epi32(_mm_shuffle_epi8(bytes, spread));
            let shifts = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
            _mm256_and_si256(_mm256_srlv_epi32(doubled, shifts), _mm256_set1_epi32(0x03))
        }
    }
}

/// AVX2 ADC over a `BITS`-wide code. A ymm holds 8 f32, half the table, so
/// tq4 permutes both halves and blends on bit 3 of the index; tq2 only ever
/// reaches indices 0..4 and needs one permute.
///
/// # Safety
///
/// The current CPU must support AVX2 and FMA.
///
/// # Panics
///
/// Panics unless the input is exactly representable: `q_rot.len() == dim`,
/// `dim` a multiple of `8 / BITS`, and `code.len() == dim * BITS / 8`. See
/// [`assert_adc_layout`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[must_use]
pub unsafe fn adc_i8_avx2<const BITS: usize>(
    code: &[u8],
    centroids_i8: &[i8; 16],
    i8_scale: f32,
    q_rot: &[f32],
    dim: usize,
) -> f32 {
    use std::arch::x86_64::{
        _mm256_and_si256, _mm256_blendv_ps, _mm256_castsi256_ps, _mm256_fmadd_ps, _mm256_loadu_ps,
        _mm256_permutevar8x32_ps, _mm256_set1_epi32, _mm256_setzero_ps, _mm256_slli_epi32,
        _mm256_storeu_ps,
    };

    assert_adc_layout::<BITS>(code, q_rot, dim);
    let centroids_f32: [f32; 16] = std::array::from_fn(|i| f32::from(centroids_i8[i]));
    let block = dim - (dim % 8);
    // SAFETY: the caller selected AVX2+FMA. Each `q_rot` load reads 8 f32 at
    // `base < block <= dim`, and the code reads are bounded by the assert.
    let mut acc = unsafe {
        let lut_lo = _mm256_loadu_ps(centroids_f32.as_ptr());
        let lut_hi = _mm256_loadu_ps(centroids_f32.as_ptr().add(8));
        let low_three = _mm256_set1_epi32(0x07);
        let mut acc = _mm256_setzero_ps();
        let mut base = 0;
        while base < block {
            let idx = unpack_indices_avx2::<BITS>(code, base);
            let centroids_v = if BITS == 4 {
                let masked = _mm256_and_si256(idx, low_three);
                let pick = _mm256_permutevar8x32_ps(lut_lo, masked);
                let pick_hi = _mm256_permutevar8x32_ps(lut_hi, masked);
                let selector = _mm256_slli_epi32::<28>(idx);
                _mm256_blendv_ps(pick, pick_hi, _mm256_castsi256_ps(selector))
            } else {
                _mm256_permutevar8x32_ps(lut_lo, idx)
            };
            let q_v = _mm256_loadu_ps(q_rot.as_ptr().add(base));
            acc = _mm256_fmadd_ps(q_v, centroids_v, acc);
            base += 8;
        }
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
        lanes.into_iter().sum::<f32>()
    };
    for i in block..dim {
        acc += q_rot[i] * centroids_f32[code_index::<BITS>(code, i)];
    }
    acc * i8_scale
}

/// AVX2 kernel for [`tq4_adc_i8`].
///
/// # Safety
///
/// The current CPU must support AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[must_use]
pub unsafe fn tq4_adc_i8_avx2(
    code: &[u8],
    centroids_i8: &[i8; 16],
    i8_scale: f32,
    q_rot: &[f32],
    dim: usize,
) -> f32 {
    // SAFETY: the caller selected AVX2+FMA, forwarded to `adc_i8_avx2`.
    unsafe { adc_i8_avx2::<4>(code, centroids_i8, i8_scale, q_rot, dim) }
}

/// Expand 16 packed `BITS`-wide codes starting at `base` into 16 lane indices.
///
/// # Safety
///
/// The caller must have selected AVX-512F, AVX-512BW and SSSE3, and `code`
/// must hold at least `(base + 16) * BITS / 8` bytes.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[target_feature(enable = "avx512f,avx512bw,ssse3")]
unsafe fn unpack_indices_avx512<const BITS: usize>(
    packed: *const u8,
) -> std::arch::x86_64::__m512i {
    use std::arch::x86_64::{
        __m128i, _mm_loadl_epi64, _mm_loadu_si32, _mm_set_epi8, _mm_shuffle_epi8,
        _mm_unpacklo_epi8, _mm512_and_si512, _mm512_cvtepu8_epi32, _mm512_set1_epi32,
        _mm512_setr_epi32, _mm512_srli_epi32, _mm512_srlv_epi32,
    };
    const { assert!(BITS == 2 || BITS == 4, "only 2-bit and 4-bit codes") }
    // SAFETY: the caller guarantees the features and the byte range.
    unsafe {
        if BITS == 4 {
            let bytes = _mm_loadl_epi64(packed.cast::<__m128i>());
            let doubled = _mm512_cvtepu8_epi32(_mm_unpacklo_epi8(bytes, bytes));
            let nibble = _mm512_set1_epi32(0x0F);
            let low = _mm512_and_si512(doubled, nibble);
            let high = _mm512_and_si512(_mm512_srli_epi32::<4>(doubled), nibble);
            std::arch::x86_64::_mm512_mask_blend_epi32(0xAAAA, low, high)
        } else {
            let bytes = _mm_loadu_si32(packed);
            let spread = _mm_set_epi8(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
            let doubled = _mm512_cvtepu8_epi32(_mm_shuffle_epi8(bytes, spread));
            let shifts = _mm512_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6);
            _mm512_and_si512(_mm512_srlv_epi32(doubled, shifts), _mm512_set1_epi32(0x03))
        }
    }
}

/// Tail variant of [`unpack_indices_avx512`]: the packed bytes come from a
/// masked load, so the read stops exactly at the end of `code` without a
/// staging copy. A staging buffer costs a store-to-load forwarding stall,
/// measured at 17 ns per call at dim 104, which is worse than the scalar loop
/// this replaced.
///
/// # Safety
///
/// The caller must have selected AVX-512F, AVX-512BW, AVX-512VL and SSSE3.
/// `byte_mask` must cover only bytes that exist in the code slice.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,ssse3")]
unsafe fn unpack_tail_indices_avx512<const BITS: usize>(
    packed: *const u8,
    byte_mask: u16,
) -> std::arch::x86_64::__m512i {
    use std::arch::x86_64::{
        _mm_maskz_loadu_epi8, _mm_set_epi8, _mm_shuffle_epi8, _mm_unpacklo_epi8, _mm512_and_si512,
        _mm512_cvtepu8_epi32, _mm512_mask_blend_epi32, _mm512_set1_epi32, _mm512_setr_epi32,
        _mm512_srli_epi32, _mm512_srlv_epi32,
    };
    const { assert!(BITS == 2 || BITS == 4, "only 2-bit and 4-bit codes") }
    // SAFETY: the caller guarantees the features, and the mask stops the load
    // at the end of the slice; masked-out bytes read as zero and land in lanes
    // whose query coordinate the caller has already zeroed.
    unsafe {
        let bytes = _mm_maskz_loadu_epi8(byte_mask, packed.cast());
        if BITS == 4 {
            let doubled = _mm512_cvtepu8_epi32(_mm_unpacklo_epi8(bytes, bytes));
            let nibble = _mm512_set1_epi32(0x0F);
            let low = _mm512_and_si512(doubled, nibble);
            let high = _mm512_and_si512(_mm512_srli_epi32::<4>(doubled), nibble);
            _mm512_mask_blend_epi32(0xAAAA, low, high)
        } else {
            let spread = _mm_set_epi8(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
            let doubled = _mm512_cvtepu8_epi32(_mm_shuffle_epi8(bytes, spread));
            let shifts = _mm512_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6);
            _mm512_and_si512(_mm512_srlv_epi32(doubled, shifts), _mm512_set1_epi32(0x03))
        }
    }
}

/// AVX-512 ADC over a `BITS`-wide code. The 16 centroids are exactly one zmm
/// of f32, so the table stays in a register and each block of 16 coordinates
/// costs one `vpermps`.
///
/// # Safety
///
/// The current CPU must satisfy [`avx512_adc_supported`]: AVX-512F,
/// AVX-512BW, AVX-512VL and SSSE3.
///
/// # Panics
///
/// Panics unless the input is exactly representable: `q_rot.len() == dim`,
/// `dim` a multiple of `8 / BITS`, and `code.len() == dim * BITS / 8`. See
/// [`assert_adc_layout`].
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,ssse3")]
#[must_use]
pub unsafe fn adc_i8_avx512<const BITS: usize>(
    code: &[u8],
    centroids_i8: &[i8; 16],
    i8_scale: f32,
    q_rot: &[f32],
    dim: usize,
) -> f32 {
    use std::arch::x86_64::{
        _mm512_add_ps, _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_maskz_loadu_ps,
        _mm512_permutexvar_ps, _mm512_reduce_add_ps, _mm512_setzero_ps,
    };

    assert_adc_layout::<BITS>(code, q_rot, dim);
    let centroids_f32: [f32; 16] = std::array::from_fn(|i| f32::from(centroids_i8[i]));
    let block = dim - (dim % 16);
    // SAFETY: the caller selected the features. Each `q_rot` load spans
    // `[base, base + 16)` with `base + 16 <= block <= dim`, the code reads are
    // bounded by the assert, and `vpermps` cannot index outside the table.
    let acc = unsafe {
        let lut = _mm512_loadu_ps(centroids_f32.as_ptr());
        let mut accs = [_mm512_setzero_ps(); 4];
        let mut base = 0;
        while base + 64 <= block {
            for (slot, acc) in accs.iter_mut().enumerate() {
                let at = base + slot * 16;
                let idx = unpack_indices_avx512::<BITS>(code.as_ptr().add(at * BITS / 8));
                let centroids_v = _mm512_permutexvar_ps(idx, lut);
                let q_v = _mm512_loadu_ps(q_rot.as_ptr().add(at));
                *acc = _mm512_fmadd_ps(q_v, centroids_v, *acc);
            }
            base += 64;
        }
        while base < block {
            let idx = unpack_indices_avx512::<BITS>(code.as_ptr().add(base * BITS / 8));
            let centroids_v = _mm512_permutexvar_ps(idx, lut);
            let q_v = _mm512_loadu_ps(q_rot.as_ptr().add(base));
            accs[0] = _mm512_fmadd_ps(q_v, centroids_v, accs[0]);
            base += 16;
        }
        // Tail, in SIMD rather than one coordinate at a time. A scalar tail
        // costs more than the extra width saves whenever `dim % 16` is large:
        // at dim 104 it made this kernel 8-11% slower end to end than the AVX2
        // one, which has no tail there because 104 is a multiple of 8.
        //
        // The masked load zeroes the query lanes past the tail, so the lanes
        // whose indices come from staged zero bytes multiply a finite centroid
        // by zero and contribute nothing.
        let rem = dim - block;
        if rem > 0 {
            #[allow(clippy::cast_possible_truncation)]
            let mask = ((1u32 << rem) - 1) as u16;
            #[allow(clippy::cast_possible_truncation)]
            let byte_mask = ((1u32 << (rem * BITS).div_ceil(8)) - 1) as u16;
            let idx =
                unpack_tail_indices_avx512::<BITS>(code.as_ptr().add(block * BITS / 8), byte_mask);
            let centroids_v = _mm512_permutexvar_ps(idx, lut);
            let q_v = _mm512_maskz_loadu_ps(mask, q_rot.as_ptr().add(block));
            accs[0] = _mm512_fmadd_ps(q_v, centroids_v, accs[0]);
        }
        let pair0 = _mm512_add_ps(accs[0], accs[1]);
        let pair1 = _mm512_add_ps(accs[2], accs[3]);
        _mm512_reduce_add_ps(_mm512_add_ps(pair0, pair1))
    };
    acc * i8_scale
}

/// Every target feature [`adc_i8_avx512`] is compiled with, hence every one a
/// caller must have. The kernel, both tier wrappers and both dispatchers read
/// this single predicate: declaring a narrower set anywhere is how a caller
/// ends up executing an instruction its CPU does not have.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[must_use]
pub fn avx512_adc_supported() -> bool {
    std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512bw")
        && std::is_x86_feature_detected!("avx512vl")
        && std::is_x86_feature_detected!("ssse3")
}

/// AVX-512F kernel for [`tq4_adc_i8`].
///
/// # Safety
///
/// The current CPU must satisfy [`avx512_adc_supported`]: AVX-512F, AVX-512BW,
/// AVX-512VL and SSSE3. This is the kernel's full requirement, not the subset
/// this tier happens to reach today.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,ssse3")]
#[must_use]
pub unsafe fn tq4_adc_i8_avx512(
    code: &[u8],
    centroids_i8: &[i8; 16],
    i8_scale: f32,
    q_rot: &[f32],
    dim: usize,
) -> f32 {
    // SAFETY: the caller satisfied `avx512_adc_supported`, which is exactly
    // the feature set this wrapper and `adc_i8_avx512` are compiled with.
    unsafe { adc_i8_avx512::<4>(code, centroids_i8, i8_scale, q_rot, dim) }
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
    assert_adc_layout::<2>(code, q_rot, dim);
    #[cfg(target_arch = "aarch64")]
    {
        tq2_adc_i8_neon(code, centroids_i8, i8_scale, q_rot, dim)
    }
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "avx512")]
        {
            if avx512_adc_supported() {
                // SAFETY: every feature the kernel is compiled with was checked
                // immediately above, through the one shared predicate.
                return unsafe { tq2_adc_i8_avx512(code, centroids_i8, i8_scale, q_rot, dim) };
            }
        }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: both target features were checked immediately above.
            return unsafe { tq2_adc_i8_avx2(code, centroids_i8, i8_scale, q_rot, dim) };
        }
        tq2_adc_i8_scalar(code, centroids_i8, i8_scale, q_rot, dim)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
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

/// NEON kernel for [`tq2_adc_i8`].
#[cfg(target_arch = "aarch64")]
#[must_use]
pub fn tq2_adc_i8_neon(
    code: &[u8],
    centroids_i8: &[i8; 16],
    i8_scale: f32,
    q_rot: &[f32],
    dim: usize,
) -> f32 {
    adc_i8_neon::<2>(code, centroids_i8, i8_scale, q_rot, dim)
}

/// AVX2 kernel for [`tq2_adc_i8`].
///
/// # Safety
///
/// The current CPU must support AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[must_use]
pub unsafe fn tq2_adc_i8_avx2(
    code: &[u8],
    centroids_i8: &[i8; 16],
    i8_scale: f32,
    q_rot: &[f32],
    dim: usize,
) -> f32 {
    // SAFETY: the caller selected AVX2+FMA, forwarded to `adc_i8_avx2`.
    unsafe { adc_i8_avx2::<2>(code, centroids_i8, i8_scale, q_rot, dim) }
}

/// AVX-512F kernel for [`tq2_adc_i8`].
///
/// # Safety
///
/// The current CPU must satisfy [`avx512_adc_supported`]: AVX-512F,
/// AVX-512BW, AVX-512VL and SSSE3.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,ssse3")]
#[must_use]
pub unsafe fn tq2_adc_i8_avx512(
    code: &[u8],
    centroids_i8: &[i8; 16],
    i8_scale: f32,
    q_rot: &[f32],
    dim: usize,
) -> f32 {
    // SAFETY: the caller selected the features, forwarded to `adc_i8_avx512`.
    unsafe { adc_i8_avx512::<2>(code, centroids_i8, i8_scale, q_rot, dim) }
}

/// [`tq2_adc_i8`] with the QUERY quantised to i8 too: `sum_i q_i8[i] *
/// centroid_i8[code[i]]`, exact in i32. The f32 kernel widens every picked
/// centroid through i16/i32 to f32 and pays four FMAs per 16 dims; here the
/// TBL-decoded levels feed `sdot` directly - sixteen multiply-accumulates
/// per instruction - and the caller applies `i8_scale * q_scale` once.
///
/// Quantising the query costs accuracy the way quantising the centroids
/// does; the caller gates recall, as with the i8 centroids themselves.
///
/// # Panics
///
/// Panics if `code.len() != dim / 4` or `q_i8.len() != dim`.
#[must_use]
pub fn tq2_adc_qi8(code: &[u8], centroids_i8: &[i8; 16], q_i8: &[i8], dim: usize) -> i32 {
    adc_qi8::<2>(code, centroids_i8, q_i8, dim)
}

/// [`tq4_adc_i8`] with the query quantised to i8: same permute-dot shape as
/// [`tq2_adc_qi8`], one code nibble per dim.
///
/// # Panics
///
/// Panics if `code.len() != dim / 2` or `q_i8.len() != dim`.
#[must_use]
pub fn tq4_adc_qi8(code: &[u8], centroids_i8: &[i8; 16], q_i8: &[i8], dim: usize) -> i32 {
    adc_qi8::<4>(code, centroids_i8, q_i8, dim)
}

fn adc_qi8<const BITS: usize>(
    code: &[u8],
    centroids_i8: &[i8; 16],
    q_i8: &[i8],
    dim: usize,
) -> i32 {
    assert_eq!(code.len(), dim * BITS / 8, "code length");
    assert_eq!(q_i8.len(), dim, "query length");
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            // SAFETY: dotprod was checked immediately above.
            return unsafe { adc_qi8_sdot::<BITS>(code, centroids_i8, q_i8, dim) };
        }
    }
    adc_qi8_scalar::<BITS>(code, centroids_i8, q_i8, dim)
}

/// Portable scalar reference for [`tq2_adc_qi8`]. Oracle in proptest.
#[must_use]
pub fn tq2_adc_qi8_scalar(code: &[u8], centroids_i8: &[i8; 16], q_i8: &[i8], dim: usize) -> i32 {
    adc_qi8_scalar::<2>(code, centroids_i8, q_i8, dim)
}

fn adc_qi8_scalar<const BITS: usize>(
    code: &[u8],
    centroids_i8: &[i8; 16],
    q_i8: &[i8],
    dim: usize,
) -> i32 {
    let mut acc = 0i32;
    for i in 0..dim {
        acc += i32::from(q_i8[i]) * i32::from(centroids_i8[code_index::<BITS>(code, i)]);
    }
    acc
}

/// `sdot` kernel for [`tq2_adc_qi8`]: TBL-decode 16 code indices to i8
/// levels, one `sdot` against 16 query bytes - the reference permute-dot
/// shape, single-vector form. Four independent accumulators hide the
/// 3-cycle `sdot` latency. `vdotq_s32` is still unstable in `std::arch`,
/// so the instruction is emitted with inline asm.
///
/// # Safety
///
/// The current CPU must support the `dotprod` extension.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
#[must_use]
unsafe fn adc_qi8_sdot<const BITS: usize>(
    code: &[u8],
    centroids_i8: &[i8; 16],
    q_i8: &[i8],
    dim: usize,
) -> i32 {
    use std::arch::aarch64::{
        int32x4_t, int8x16_t, vaddq_s32, vaddvq_s32, vdupq_n_s32, vld1q_s8, vqtbl1q_s8,
    };
    use std::arch::asm;
    assert_eq!(code.len(), dim * BITS / 8, "code length");
    assert_eq!(q_i8.len(), dim, "query length");
    let block = dim - (dim % 64);
    // SAFETY: every load covers lanes inside the asserted lengths; `vqtbl1q_s8`
    // is total (2-bit indices cannot exceed 3); the asm is a register-only
    // `sdot`, no memory, no stack.
    let mut sum = unsafe {
        let lut = vld1q_s8(centroids_i8.as_ptr());
        let mut acc = [vdupq_n_s32(0); 4];
        let mut base = 0;
        while base < block {
            for (k, acc_k) in acc.iter_mut().enumerate() {
                let at = base + k * 16;
                let idx = unpack_indices_neon::<BITS>(code, at);
                let picked: int8x16_t = vqtbl1q_s8(lut, idx);
                let q: int8x16_t = vld1q_s8(q_i8.as_ptr().add(at));
                let mut a: int32x4_t = *acc_k;
                asm!(
                    "sdot {a:v}.4s, {p:v}.16b, {q:v}.16b",
                    a = inout(vreg) a,
                    p = in(vreg) picked,
                    q = in(vreg) q,
                    options(pure, nomem, nostack)
                );
                *acc_k = a;
            }
            base += 64;
        }
        vaddvq_s32(vaddq_s32(vaddq_s32(acc[0], acc[1]), vaddq_s32(acc[2], acc[3])))
    };
    for i in block..dim {
        sum += i32::from(q_i8[i]) * i32::from(centroids_i8[code_index::<BITS>(code, i)]);
    }
    sum
}

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

/// [`tq1_masked_sum`] with the QUERY quantised to i8: `sum q_i8[i]` over
/// coords whose code bit is set, exact in i32. The f32 kernel builds a
/// full-width mask and pays f32 adds per lane; here the bit-test produces a
/// 0/1 i8 vector that feeds `sdot` - sixteen coords per instruction. The
/// caller applies the query scale once, like the 2-/4-bit qi8 kernels.
///
/// # Panics
///
/// Panics if `code.len() != dim / 8` or `q_i8.len() != dim`.
#[must_use]
pub fn tq1_masked_dot_qi8(code: &[u8], q_i8: &[i8], dim: usize) -> i32 {
    assert_eq!(code.len(), dim / 8, "code length");
    assert_eq!(q_i8.len(), dim, "query length");
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            // SAFETY: dotprod was checked immediately above.
            return unsafe { tq1_masked_dot_qi8_sdot(code, q_i8, dim) };
        }
    }
    tq1_masked_dot_qi8_scalar(code, q_i8, dim)
}

/// Portable scalar reference for [`tq1_masked_dot_qi8`]. Oracle in tests.
#[must_use]
pub fn tq1_masked_dot_qi8_scalar(code: &[u8], q_i8: &[i8], dim: usize) -> i32 {
    let mut acc = 0i32;
    for i in 0..dim {
        if (code[i / 8] >> (i % 8)) & 1 == 1 {
            acc += i32::from(q_i8[i]);
        }
    }
    acc
}

/// `sdot` kernel for [`tq1_masked_dot_qi8`]: broadcast each code byte over 8
/// i8 lanes (two bytes per 16-lane register via a zip of the selector
/// pattern), bit-test into a 0xFF/0 mask, mask down to 0/1, one `sdot`
/// against 16 query bytes.
///
/// # Safety
///
/// The current CPU must support the `dotprod` extension.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
#[must_use]
unsafe fn tq1_masked_dot_qi8_sdot(code: &[u8], q_i8: &[i8], dim: usize) -> i32 {
    use std::arch::aarch64::{
        int32x4_t, int8x16_t, vaddq_s32, vaddvq_s32, vandq_s8, vdupq_n_s32, vdupq_n_s8, vld1q_s8,
        vld1q_u8, vqtbl1q_u8, vreinterpretq_s8_u8, vreinterpretq_u8_s8, vtstq_s8,
    };
    use std::arch::asm;
    assert_eq!(code.len(), dim / 8, "code length");
    assert_eq!(q_i8.len(), dim, "query length");
    let block = dim - (dim % 64);
    // SAFETY: the byte-pair table indexes select code bytes 2b and 2b+1 which
    // exist for every 16-dim group inside `block <= dim`; q loads cover
    // `[at, at+16) <= dim`. Reads only; the asm is a register-only sdot.
    let mut sum = unsafe {
        // For 16 dims we need code bytes [2g, 2g+1] broadcast 8 lanes each.
        // Load 8 code bytes (64 dims) once, then TBL-broadcast per group.
        let sel: int8x16_t = vld1q_s8(
            [1i8, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128].as_ptr(),
        );
        let one = vdupq_n_s8(1);
        let mut acc = [vdupq_n_s32(0); 4];
        let mut base = 0;
        while base < block {
            for (k, acc_k) in acc.iter_mut().enumerate() {
                let at = base + k * 16;
                let b0 = i8::from_ne_bytes([code[at / 8]]);
                let b1 = i8::from_ne_bytes([code[at / 8 + 1]]);
                // Broadcast the two bytes over lanes 0..7 and 8..15.
                let idx: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
                let pair: int8x16_t = {
                    let two = [b0, b1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
                    let v = vld1q_s8(two.as_ptr());
                    vreinterpretq_s8_u8(vqtbl1q_u8(
                        vreinterpretq_u8_s8(v),
                        vld1q_u8(idx.as_ptr()),
                    ))
                };
                let bits01 = vandq_s8(
                    vreinterpretq_s8_u8(vtstq_s8(pair, sel)),
                    one,
                );
                let q: int8x16_t = vld1q_s8(q_i8.as_ptr().add(at));
                let mut a: int32x4_t = *acc_k;
                asm!(
                    "sdot {a:v}.4s, {m:v}.16b, {q:v}.16b",
                    a = inout(vreg) a,
                    m = in(vreg) bits01,
                    q = in(vreg) q,
                    options(pure, nomem, nostack)
                );
                *acc_k = a;
            }
            base += 64;
        }
        vaddvq_s32(vaddq_s32(vaddq_s32(acc[0], acc[1]), vaddq_s32(acc[2], acc[3])))
    };
    for i in block..dim {
        if (code[i / 8] >> (i % 8)) & 1 == 1 {
            sum += i32::from(q_i8[i]);
        }
    }
    sum
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
    assert_tq1_layout(code, q_rot, dim);
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

/// AVX2 kernel for [`tq1_masked_sum`]. No gather needed here (no centroid
/// table): builds an 8-lane masked-`q_rot`-or-zero array with a scalar bit
/// test, then a single vector add per group. `dim % 8 == 0` is asserted by
/// the dispatcher, so the scalar tail loop below never runs in practice.
///
/// # Safety
///
/// The current CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[must_use]
pub unsafe fn tq1_masked_sum_avx2(code: &[u8], q_rot: &[f32], dim: usize) -> f32 {
    assert_tq1_layout(code, q_rot, dim);
    use std::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_setzero_ps, _mm256_storeu_ps};

    let block = dim - (dim % 8);
    let mut masked = [0.0f32; 8];
    // SAFETY: the caller selected AVX2. `masked` is a fully-initialised
    // stack array read back immediately by `_mm256_loadu_ps`; each `q_rot`
    // read at `global = base + k < block <= dim == q_rot.len()`.
    let mut acc = unsafe {
        let mut acc = _mm256_setzero_ps();
        let mut base = 0;
        while base < block {
            for (k, slot) in masked.iter_mut().enumerate() {
                let global = base + k;
                let byte = code[global / 8];
                let bit = (byte >> (global % 8)) & 1;
                *slot = if bit == 1 { q_rot[global] } else { 0.0 };
            }
            let m = _mm256_loadu_ps(masked.as_ptr());
            acc = _mm256_add_ps(acc, m);
            base += 8;
        }
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
        lanes.into_iter().sum::<f32>()
    };
    for i in block..dim {
        let byte = code[i / 8];
        let bit = (byte >> (i % 8)) & 1;
        if bit == 1 {
            acc += q_rot[i];
        }
    }
    acc
}

/// AVX-512 kernel for [`tq1_masked_sum`]. `code`'s bit-plane packing (bit
/// `i % 8` of byte `i / 8` selects coordinate `i`) already *is* a 16-lane
/// mask in the exact layout `_mm512_maskz_loadu_ps` wants, so two adjacent
/// code bytes read directly as a `u16` need no shuffle or bit-test loop at
/// all - the hardware mask register does the selection AVX2 had to build by
/// hand. This is the one place where AVX-512 removes work the AVX2 kernel
/// needed, rather than only running it wider.
///
/// # Safety
///
/// The current CPU must support AVX-512F.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[target_feature(enable = "avx512f")]
#[must_use]
pub unsafe fn tq1_masked_sum_avx512(code: &[u8], q_rot: &[f32], dim: usize) -> f32 {
    assert_tq1_layout(code, q_rot, dim);
    use std::arch::x86_64::{
        _mm512_add_ps, _mm512_maskz_loadu_ps, _mm512_setzero_ps, _mm512_storeu_ps,
    };

    let block = dim - (dim % 16);
    // SAFETY: the caller selected AVX-512F. Each `q_rot` load covers 16
    // lanes at `base + 16 <= block <= dim == q_rot.len()`; masked-out lanes
    // are not read by the hardware, only the up-to-16 addressed ones that
    // are in bounds. `code[base/8]`/`code[base/8 + 1]` are in bounds because
    // `base + 16 <= dim` implies `base/8 + 1 < dim/8 == code.len()`.
    let mut acc = unsafe {
        let mut acc = _mm512_setzero_ps();
        let mut base = 0;
        while base < block {
            let byte_idx = base / 8;
            let mask = u16::from(code[byte_idx]) | (u16::from(code[byte_idx + 1]) << 8);
            let v = _mm512_maskz_loadu_ps(mask, q_rot.as_ptr().add(base).cast());
            acc = _mm512_add_ps(acc, v);
            base += 16;
        }
        let mut lanes = [0.0f32; 16];
        _mm512_storeu_ps(lanes.as_mut_ptr(), acc);
        lanes.into_iter().sum::<f32>()
    };
    for i in block..dim {
        let byte = code[i / 8];
        let bit = (byte >> (i % 8)) & 1;
        if bit == 1 {
            acc += q_rot[i];
        }
    }
    acc
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
    assert_tq1_layout(code, q_rot, dim);
    #[cfg(target_arch = "aarch64")]
    {
        tq1_masked_sum_neon(code, q_rot, dim)
    }
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "avx512")]
        {
            if std::is_x86_feature_detected!("avx512f") {
                // SAFETY: the target feature was checked immediately above.
                return unsafe { tq1_masked_sum_avx512(code, q_rot, dim) };
            }
        }
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: the target feature was checked immediately above.
            return unsafe { tq1_masked_sum_avx2(code, q_rot, dim) };
        }
        tq1_masked_sum_scalar(code, q_rot, dim)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
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

#[cfg(test)]
mod qi8_tests {
    use super::*;

    #[test]
    fn tq2_qi8_sdot_matches_scalar_on_random_and_ragged_dims() {
        let mut state = 0x243f6a8885a308d3u64;
        let mut next = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u8
        };
        let centroids: [i8; 16] =
            std::array::from_fn(|i| if i < 4 { [-100i8, -33, 33, 100][i] } else { 0 });
        // Ragged dims cover the 64-wide block, its tail, and small inputs.
        for dim in [4usize, 16, 64, 68, 100, 512, 1024, 1536] {
            let code: Vec<u8> = (0..dim / 4).map(|_| next()).collect();
            let q: Vec<i8> = (0..dim).map(|_| next() as i8).collect();
            assert_eq!(
                tq2_adc_qi8(&code, &centroids, &q, dim),
                tq2_adc_qi8_scalar(&code, &centroids, &q, dim),
                "dim {dim}"
            );
        }
    }

    #[test]
    fn tq1_masked_dot_qi8_matches_scalar_on_random_and_ragged_dims() {
        let mut state = 0x9216d5d98979fb1bu64;
        let mut next = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u8
        };
        for dim in [8usize, 64, 72, 128, 512, 1024, 1536] {
            let code: Vec<u8> = (0..dim / 8).map(|_| next()).collect();
            let q: Vec<i8> = (0..dim).map(|_| next() as i8).collect();
            assert_eq!(
                tq1_masked_dot_qi8(&code, &q, dim),
                tq1_masked_dot_qi8_scalar(&code, &q, dim),
                "dim {dim}"
            );
        }
    }

    #[test]
    fn tq4_qi8_sdot_matches_scalar_on_random_and_ragged_dims() {
        let mut state = 0x452821e638d01377u64;
        let mut next = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u8
        };
        let centroids: [i8; 16] = std::array::from_fn(|i| (i as i8 - 8) * 15);
        for dim in [2usize, 16, 64, 68, 100, 512, 1024, 1536] {
            let code: Vec<u8> = (0..dim / 2).map(|_| next()).collect();
            let q: Vec<i8> = (0..dim).map(|_| next() as i8).collect();
            assert_eq!(
                tq4_adc_qi8(&code, &centroids, &q, dim),
                {
                    let mut acc = 0i32;
                    for i in 0..dim {
                        acc += i32::from(q[i]) * i32::from(centroids[code_index::<4>(&code, i)]);
                    }
                    acc
                },
                "dim {dim}"
            );
        }
    }

    /// The i32 path with an exactly-representable query must agree with the
    /// f32 kernel: same picks, same products, only the accumulation differs.
    #[test]
    fn tq2_qi8_agrees_with_the_f32_kernel_on_integer_queries() {
        let centroids: [i8; 16] =
            std::array::from_fn(|i| if i < 4 { [-90i8, -30, 30, 90][i] } else { 0 });
        let dim = 1024;
        let code: Vec<u8> = (0..dim / 4).map(|i| (i * 37 % 256) as u8).collect();
        let q_i8: Vec<i8> = (0..dim).map(|i| ((i * 13 % 255) as i16 - 127) as i8).collect();
        let q_f32: Vec<f32> = q_i8.iter().map(|&x| f32::from(x)).collect();
        let exact = tq2_adc_qi8(&code, &centroids, &q_i8, dim) as f32;
        let viaf32 = tq2_adc_i8(&code, &centroids, 1.0, &q_f32, dim);
        assert!(
            (exact - viaf32).abs() <= viaf32.abs() * 1e-4 + 1.0,
            "i32 {exact} vs f32 {viaf32}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_adc_matches_the_tier_specific_scalar_kernels() {
        let dim = 260;
        let centroids: [i8; 16] = std::array::from_fn(|i| (i as i8) * 8 - 64);
        let q_rot: Vec<f32> = (0..dim).map(|i| (i as f32 - 130.0) / 41.0).collect();
        let scale = 0.003;

        let tq4_code: Vec<u8> = (0..dim / 2).map(|i| (i as u8).wrapping_mul(37)).collect();
        assert_eq!(
            adc_i8_scalar::<4>(&tq4_code, &centroids, scale, &q_rot, dim),
            tq4_adc_i8_scalar(&tq4_code, &centroids, scale, &q_rot, dim),
        );

        let tq2_code: Vec<u8> = (0..dim / 4).map(|i| (i as u8).wrapping_mul(53)).collect();
        assert_eq!(
            adc_i8_scalar::<2>(&tq2_code, &centroids, scale, &q_rot, dim),
            tq2_adc_i8_scalar(&tq2_code, &centroids, scale, &q_rot, dim),
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn generic_adc_neon_matches_the_tier_specific_kernels() {
        let dim = 260;
        let centroids: [i8; 16] = std::array::from_fn(|i| (i as i8) * 8 - 64);
        let q_rot: Vec<f32> = (0..dim).map(|i| (i as f32 - 130.0) / 41.0).collect();
        let scale = 0.003;

        let tq4_code: Vec<u8> = (0..dim / 2).map(|i| (i as u8).wrapping_mul(37)).collect();
        let generic = adc_i8_neon::<4>(&tq4_code, &centroids, scale, &q_rot, dim);
        let tier = tq4_adc_i8_scalar(&tq4_code, &centroids, scale, &q_rot, dim);
        assert!(
            (generic - tier).abs() / tier.abs().max(1.0) < 1e-3,
            "{generic} vs {tier}"
        );

        let tq2_code: Vec<u8> = (0..dim / 4).map(|i| (i as u8).wrapping_mul(53)).collect();
        let generic = adc_i8_neon::<2>(&tq2_code, &centroids, scale, &q_rot, dim);
        let tier = tq2_adc_i8_scalar(&tq2_code, &centroids, scale, &q_rot, dim);
        assert!(
            (generic - tier).abs() / tier.abs().max(1.0) < 1e-3,
            "{generic} vs {tier}"
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_tq4_adc_i8_matches_scalar_for_unaligned_tail() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) {
            return;
        }
        // tq4 packs 2 coords/byte, so dim must be even (`code.len() == dim/2`
        // has no valid value otherwise) - 258 = 32*8 + 2, still exercises the
        // AVX2 8-wide block plus a non-multiple-of-8 tail.
        let dim = 258;
        let code: Vec<u8> = (0..dim / 2 + 1)
            .map(|i| (i as u8).wrapping_mul(37))
            .collect();
        let q_rot: Vec<f32> = (0..dim).map(|i| (i as f32 - 130.0) / 41.0).collect();
        let mut centroids = [0i8; 16];
        for (i, c) in centroids.iter_mut().enumerate() {
            *c = (i as i8) * 8 - 64;
        }
        let i8_scale = 0.003;
        let avx2 = unsafe { tq4_adc_i8_avx2(&code[..dim / 2], &centroids, i8_scale, &q_rot, dim) };
        let scalar = tq4_adc_i8_scalar(&code[..dim / 2], &centroids, i8_scale, &q_rot, dim);
        assert!((avx2 - scalar).abs() / scalar.abs().max(1.0) < 1e-3);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_tq2_adc_i8_matches_scalar_for_unaligned_tail() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) {
            return;
        }
        // tq2 packs 4 coords/byte, so dim must be a multiple of 4
        // (`code.len() == dim/4` has no valid value otherwise) - 268 =
        // 33*8 + 4, still exercises the AVX2 8-wide block plus a
        // non-multiple-of-8 tail.
        let dim = 268;
        let code: Vec<u8> = (0..dim / 4 + 1)
            .map(|i| (i as u8).wrapping_mul(53))
            .collect();
        let q_rot: Vec<f32> = (0..dim).map(|i| (i as f32 - 133.0) / 29.0).collect();
        let mut centroids = [0i8; 16];
        for c in &mut centroids[4..] {
            *c = 0;
        }
        centroids[0] = -100;
        centroids[1] = -30;
        centroids[2] = 30;
        centroids[3] = 100;
        let i8_scale = 0.004;
        let avx2 = unsafe { tq2_adc_i8_avx2(&code[..dim / 4], &centroids, i8_scale, &q_rot, dim) };
        let scalar = tq2_adc_i8_scalar(&code[..dim / 4], &centroids, i8_scale, &q_rot, dim);
        assert!((avx2 - scalar).abs() / scalar.abs().max(1.0) < 1e-3);
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[test]
    fn avx512_tq4_adc_i8_matches_scalar_for_unaligned_tail() {
        if !avx512_adc_supported() {
            return;
        }
        let dim = 258;
        let code: Vec<u8> = (0..dim / 2 + 1)
            .map(|i| (i as u8).wrapping_mul(37))
            .collect();
        let q_rot: Vec<f32> = (0..dim).map(|i| (i as f32 - 130.0) / 41.0).collect();
        let centroids = std::array::from_fn(|i| (i as i8) * 8 - 64);
        let scale = 0.003;
        let avx512 = unsafe { tq4_adc_i8_avx512(&code[..dim / 2], &centroids, scale, &q_rot, dim) };
        let scalar = tq4_adc_i8_scalar(&code[..dim / 2], &centroids, scale, &q_rot, dim);
        assert!((avx512 - scalar).abs() / scalar.abs().max(1.0) < 1e-3);
        let dispatched = tq4_adc_i8(&code[..dim / 2], &centroids, scale, &q_rot, dim);
        assert!((dispatched - avx512).abs() / avx512.abs().max(1.0) < 1e-6);
    }

    /// The tier kernels are public. The NEON one is safe and indexes with
    /// `get_unchecked` and raw pointer loads, so reaching it with a short slice
    /// is undefined behaviour, not a panic. Every kernel validates for itself
    /// rather than trusting the dispatcher that usually calls it.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn tq1_neon_rejects_slices_the_dispatcher_would_have_caught() {
        let cases: [(&[u8], usize, &str); 3] = [
            (&[], 32, "code length"),        // code far too short for dim
            (&[0u8; 4], 33, "multiple of"),  // dim not a whole number of bytes
            (&[0u8; 4], 32, "q_rot length"), // q_rot short, checked below
        ];
        for (code, dim, want) in cases {
            let q_rot = if want == "q_rot length" {
                vec![0.5f32; dim - 1]
            } else {
                vec![0.5f32; dim]
            };
            let payload = std::panic::catch_unwind(|| {
                let _ = tq1_masked_sum_neon(code, &q_rot, dim);
            })
            .expect_err("the kernel accepted an input it cannot index safely");
            let msg = panic_message(&payload);
            assert!(
                msg.contains(want),
                "dim {dim} panicked for the wrong reason: {msg}",
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn tq1_bitplane_neon_rejects_a_short_code_before_it_reaches_the_kernel() {
        let planes = vec![0u8; 1];
        let payload = std::panic::catch_unwind(|| {
            let _ = tq1_bitplane_score_neon(&planes, 1, 1, &[]);
        })
        .expect_err("the bit-plane kernel accepted a code shorter than bytes");
        let msg = panic_message(&payload);
        assert!(
            msg.contains("code length"),
            "bit-plane kernel panicked for the wrong reason: {msg}",
        );
    }

    /// The text of a caught panic, whichever payload type it carries.
    fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
        payload.downcast_ref::<&str>().map_or_else(
            || {
                payload
                    .downcast_ref::<String>()
                    .cloned()
                    .unwrap_or_else(|| "<non-string panic>".to_owned())
            },
            |s| (*s).to_owned(),
        )
    }

    /// A dimension that is not a whole number of packed bytes has no
    /// representation, and the length check alone lets it through: `dim * BITS
    /// / 8` truncates, so tq4 with `dim = 1` expects zero bytes and an empty
    /// slice satisfies it. These are safe public functions, so the input has to
    /// be rejected before it reaches the AVX-512 tail, which would mask in a
    /// live byte and read past the slice.
    #[test]
    fn tq4_rejects_a_dim_that_is_not_whole_bytes() {
        let centroids = [0i8; 16];
        for dim in [1usize, 3, 5] {
            let code = vec![0u8; dim / 2];
            let q_rot = vec![0.5f32; dim];
            // The message matters: without the layout check this input still
            // panics, but later and by accident, on an out-of-bounds index in
            // the scalar tail. Only the layout rejection proves the fix.
            let payload = std::panic::catch_unwind(|| {
                let _ = tq4_adc_i8(&code, &centroids, 0.01, &q_rot, dim);
            })
            .expect_err("tq4 accepted dim {dim}, which packs to no whole byte");
            let msg = panic_message(&payload);
            assert!(
                msg.contains("multiple of"),
                "tq4 dim {dim} panicked for the wrong reason: {msg}",
            );
        }
    }

    #[test]
    fn tq2_rejects_a_dim_that_is_not_whole_bytes() {
        let centroids = [0i8; 16];
        for dim in [1usize, 2, 3, 5, 6, 7] {
            let code = vec![0u8; dim / 4];
            let q_rot = vec![0.5f32; dim];
            let payload = std::panic::catch_unwind(|| {
                let _ = tq2_adc_i8(&code, &centroids, 0.01, &q_rot, dim);
            })
            .expect_err("tq2 accepted dim {dim}, which packs to no whole byte");
            let msg = panic_message(&payload);
            assert!(
                msg.contains("multiple of"),
                "tq2 dim {dim} panicked for the wrong reason: {msg}",
            );
        }
    }

    /// The aligned cases must keep working: this check rejects a layout, not a
    /// dimension that merely looks unusual.
    #[test]
    fn aligned_dims_are_still_accepted() {
        let centroids = [0i8; 16];
        for dim in [2usize, 4, 8, 104, 258] {
            let q_rot = vec![0.5f32; dim];
            let _ = tq4_adc_i8(&vec![0u8; dim / 2], &centroids, 0.01, &q_rot, dim);
        }
        for dim in [4usize, 8, 104, 256] {
            let q_rot = vec![0.5f32; dim];
            let _ = tq2_adc_i8(&vec![0u8; dim / 4], &centroids, 0.01, &q_rot, dim);
        }
    }

    /// Dimensions that are a multiple of 8 but not of 16 are the worst case
    /// for this kernel: AVX2 has no tail there, AVX-512 has eight coordinates
    /// of it. 104 and 200 are real embedding dimensions (GloVe), and the
    /// dimensions where a scalar tail cost 8-11% end to end before the tail
    /// moved into SIMD.
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[test]
    fn avx512_adc_matches_scalar_when_dim_is_eight_mod_sixteen() {
        if !avx512_adc_supported() {
            return;
        }
        let centroids: [i8; 16] = std::array::from_fn(|i| (i as i8) * 8 - 64);
        let scale = 0.003;
        for dim in [104usize, 200, 8, 24] {
            let q_rot: Vec<f32> = (0..dim).map(|i| (i as f32 - 50.0) / 17.0).collect();

            let tq4: Vec<u8> = (0..dim / 2).map(|i| (i as u8).wrapping_mul(37)).collect();
            let simd = unsafe { tq4_adc_i8_avx512(&tq4, &centroids, scale, &q_rot, dim) };
            let scalar = tq4_adc_i8_scalar(&tq4, &centroids, scale, &q_rot, dim);
            assert!(
                (simd - scalar).abs() / scalar.abs().max(1.0) < 1e-3,
                "tq4 dim {dim}: {simd} vs {scalar}",
            );

            let tq2: Vec<u8> = (0..dim / 4).map(|i| (i as u8).wrapping_mul(53)).collect();
            let simd = unsafe { tq2_adc_i8_avx512(&tq2, &centroids, scale, &q_rot, dim) };
            let scalar = tq2_adc_i8_scalar(&tq2, &centroids, scale, &q_rot, dim);
            assert!(
                (simd - scalar).abs() / scalar.abs().max(1.0) < 1e-3,
                "tq2 dim {dim}: {simd} vs {scalar}",
            );
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[test]
    fn avx512_tq2_adc_i8_matches_scalar_for_unaligned_tail() {
        if !avx512_adc_supported() {
            return;
        }
        let dim = 268;
        let code: Vec<u8> = (0..dim / 4 + 1)
            .map(|i| (i as u8).wrapping_mul(53))
            .collect();
        let q_rot: Vec<f32> = (0..dim).map(|i| (i as f32 - 133.0) / 29.0).collect();
        let mut centroids = [0i8; 16];
        centroids[..4].copy_from_slice(&[-100, -30, 30, 100]);
        let scale = 0.004;
        let avx512 = unsafe { tq2_adc_i8_avx512(&code[..dim / 4], &centroids, scale, &q_rot, dim) };
        let scalar = tq2_adc_i8_scalar(&code[..dim / 4], &centroids, scale, &q_rot, dim);
        assert!((avx512 - scalar).abs() / scalar.abs().max(1.0) < 1e-3);
        let dispatched = tq2_adc_i8(&code[..dim / 4], &centroids, scale, &q_rot, dim);
        assert!((dispatched - avx512).abs() / avx512.abs().max(1.0) < 1e-6);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_tq1_masked_sum_matches_scalar_for_unaligned_tail() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let dim = 264; // multiple of 8 (tq1 requires it), not a multiple of 64
        let code: Vec<u8> = (0..dim / 8).map(|i| (i as u8).wrapping_mul(29)).collect();
        let q_rot: Vec<f32> = (0..dim).map(|i| (i as f32 - 132.0) / 19.0).collect();
        let avx2 = unsafe { tq1_masked_sum_avx2(&code, &q_rot, dim) };
        let scalar = tq1_masked_sum_scalar(&code, &q_rot, dim);
        assert!((avx2 - scalar).abs() / scalar.abs().max(1.0) < 1e-3);
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[test]
    fn avx512_tq1_masked_sum_matches_scalar_for_unaligned_tail() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        // dim % 16 != 0 (264 / 16 = 16.5) exercises the AVX-512 16-wide
        // block plus the scalar tail; dim % 8 == 0 as tq1 requires.
        let dim = 264;
        let code: Vec<u8> = (0..dim / 8).map(|i| (i as u8).wrapping_mul(29)).collect();
        let q_rot: Vec<f32> = (0..dim).map(|i| (i as f32 - 132.0) / 19.0).collect();
        let avx512 = unsafe { tq1_masked_sum_avx512(&code, &q_rot, dim) };
        let scalar = tq1_masked_sum_scalar(&code, &q_rot, dim);
        assert!((avx512 - scalar).abs() / scalar.abs().max(1.0) < 1e-3);
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[test]
    fn avx512_tq1_masked_sum_all_bits_set_and_all_clear() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        let dim = 32;
        let q_rot: Vec<f32> = (0..dim).map(|i| (i as f32 - 16.0) / 5.0).collect();
        let all_set = vec![0xFFu8; dim / 8];
        let all_clear = vec![0x00u8; dim / 8];
        // The kernel sums 16 lanes in tree order, the scalar reference left to
        // right, so the two differ in the last ulp - compare with a tolerance.
        let set = unsafe { tq1_masked_sum_avx512(&all_set, &q_rot, dim) };
        let set_scalar = tq1_masked_sum_scalar(&all_set, &q_rot, dim);
        assert!(
            (set - set_scalar).abs() / set_scalar.abs().max(1.0) < 1e-6,
            "tq1_masked_sum avx512 {set} scalar {set_scalar}",
        );
        assert_eq!(
            unsafe { tq1_masked_sum_avx512(&all_clear, &q_rot, dim) },
            0.0,
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_tq1_bitplane_dispatch_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let bytes = 257;
        let b = 3;
        let planes: Vec<u8> = (0..bytes * b as usize)
            .map(|i| (i as u8).wrapping_mul(31))
            .collect();
        let code: Vec<u8> = (0..bytes).map(|i| (i as u8).wrapping_mul(17)).collect();
        assert_eq!(
            tq1_bitplane_score(&planes, b, bytes, &code),
            tq1_bitplane_score_scalar(&planes, b, bytes, &code)
        );
    }

    // ── proptest: NEON kernels must match their scalar reference ─────────────

    #[cfg(target_arch = "aarch64")]
    proptest::proptest! {
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

        #[test]
        #[cfg(target_arch = "aarch64")]
        fn prop_tq1_bitplane_neon_matches_scalar(
            // bytes = 8..64 covers the 64-byte SWAR block + tail; b = 1..=8 bit-planes.
            bytes in 8usize..=64,
            b in 1u8..=8,
            seed in proptest::collection::vec(proptest::num::u8::ANY, 0..640),
        ) {
            let mut code = seed.clone();
            code.resize(bytes, 0);
            let mut planes = seed;
            planes.resize(bytes * b as usize, 0);
            // Integer kernel: must match exactly.
            proptest::prop_assert_eq!(
                tq1_bitplane_score_neon(&planes, b, bytes, &code),
                tq1_bitplane_score_scalar(&planes, b, bytes, &code),
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    proptest::proptest! {
        #[test]
        fn prop_tq4_adc_i8_avx2_matches_scalar(
            chunks in 1usize..32,
            codes_data in proptest::collection::vec(proptest::num::u8::ANY, 0..256),
            centroids in proptest::collection::vec(-127i8..=127i8, 16..=16),
            q_data in proptest::collection::vec(-1.0f32..1.0, 0..600),
            i8_scale in 1e-6f32..1e-2,
        ) {
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                let dim = chunks * 16;
                let mut code = codes_data;
                code.resize(dim / 2, 0);
                let mut q = q_data;
                q.resize(dim, 0.0);
                let mut centroids_arr = [0i8; 16];
                for (i, &c) in centroids.iter().enumerate() { centroids_arr[i] = c; }

                let avx2 = unsafe { tq4_adc_i8_avx2(&code, &centroids_arr, i8_scale, &q, dim) };
                let scalar = tq4_adc_i8_scalar(&code, &centroids_arr, i8_scale, &q, dim);
                let denom = scalar.abs().max(1.0);
                proptest::prop_assert!(
                    (avx2 - scalar).abs() / denom < 1e-3,
                    "tq4_adc_i8 avx2 {} scalar {} dim {}",
                    avx2, scalar, dim,
                );
            }
        }

        #[cfg(feature = "avx512")]
        #[test]
        fn prop_tq4_adc_i8_avx512_matches_scalar(
            chunks in 1usize..32,
            codes_data in proptest::collection::vec(proptest::num::u8::ANY, 0..256),
            centroids in proptest::collection::vec(-127i8..=127i8, 16..=16),
            q_data in proptest::collection::vec(-1.0f32..1.0, 0..600),
            i8_scale in 1e-6f32..1e-2,
        ) {
            if avx512_adc_supported() {
                let dim = chunks * 16;
                let mut code = codes_data;
                code.resize(dim / 2, 0);
                let mut q = q_data;
                q.resize(dim, 0.0);
                let mut centroids_arr = [0i8; 16];
                for (i, &c) in centroids.iter().enumerate() { centroids_arr[i] = c; }
                let avx512 = unsafe {
                    tq4_adc_i8_avx512(&code, &centroids_arr, i8_scale, &q, dim)
                };
                let scalar = tq4_adc_i8_scalar(&code, &centroids_arr, i8_scale, &q, dim);
                let denom = scalar.abs().max(1.0);
                proptest::prop_assert!(
                    (avx512 - scalar).abs() / denom < 1e-3,
                    "tq4_adc_i8 avx512 {} scalar {} dim {}",
                    avx512, scalar, dim,
                );
            }
        }

        #[test]
        fn prop_tq2_adc_i8_avx2_matches_scalar(
            chunks in 1usize..16,
            codes_data in proptest::collection::vec(proptest::num::u8::ANY, 0..128),
            centroids in proptest::collection::vec(-127i8..=127i8, 16..=16),
            q_data in proptest::collection::vec(-1.0f32..1.0, 0..600),
            i8_scale in 1e-6f32..1e-2,
        ) {
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                let dim = chunks * 32;
                let mut code = codes_data;
                code.resize(dim / 4, 0);
                let mut q = q_data;
                q.resize(dim, 0.0);
                let mut centroids_arr = [0i8; 16];
                for (i, &c) in centroids.iter().enumerate() { centroids_arr[i] = c; }
                for slot in &mut centroids_arr[4..16] { *slot = 0; }

                let avx2 = unsafe { tq2_adc_i8_avx2(&code, &centroids_arr, i8_scale, &q, dim) };
                let scalar = tq2_adc_i8_scalar(&code, &centroids_arr, i8_scale, &q, dim);
                let denom = scalar.abs().max(1.0);
                proptest::prop_assert!(
                    (avx2 - scalar).abs() / denom < 1e-3,
                    "tq2_adc_i8 avx2 {} scalar {} dim {}",
                    avx2, scalar, dim,
                );
            }
        }

        #[cfg(feature = "avx512")]
        #[test]
        fn prop_tq2_adc_i8_avx512_matches_scalar(
            chunks in 1usize..16,
            codes_data in proptest::collection::vec(proptest::num::u8::ANY, 0..128),
            centroids in proptest::collection::vec(-127i8..=127i8, 16..=16),
            q_data in proptest::collection::vec(-1.0f32..1.0, 0..600),
            i8_scale in 1e-6f32..1e-2,
        ) {
            if avx512_adc_supported() {
                let dim = chunks * 32;
                let mut code = codes_data;
                code.resize(dim / 4, 0);
                let mut q = q_data;
                q.resize(dim, 0.0);
                let mut centroids_arr = [0i8; 16];
                for (i, &c) in centroids.iter().enumerate() { centroids_arr[i] = c; }
                for slot in &mut centroids_arr[4..] { *slot = 0; }
                let avx512 = unsafe {
                    tq2_adc_i8_avx512(&code, &centroids_arr, i8_scale, &q, dim)
                };
                let scalar = tq2_adc_i8_scalar(&code, &centroids_arr, i8_scale, &q, dim);
                let denom = scalar.abs().max(1.0);
                proptest::prop_assert!(
                    (avx512 - scalar).abs() / denom < 1e-3,
                    "tq2_adc_i8 avx512 {} scalar {} dim {}",
                    avx512, scalar, dim,
                );
            }
        }

        #[test]
        fn prop_tq1_masked_sum_avx2_matches_scalar(
            bytes in 1usize..64,
            codes_data in proptest::collection::vec(proptest::num::u8::ANY, 0..64),
            q_data in proptest::collection::vec(-1.0f32..1.0, 0..512),
        ) {
            if std::is_x86_feature_detected!("avx2") {
                let dim = bytes * 8;
                let mut code = codes_data;
                code.resize(dim / 8, 0);
                let mut q = q_data;
                q.resize(dim, 0.0);
                let avx2 = unsafe { tq1_masked_sum_avx2(&code, &q, dim) };
                let scalar = tq1_masked_sum_scalar(&code, &q, dim);
                let denom = scalar.abs().max(1.0);
                proptest::prop_assert!(
                    (avx2 - scalar).abs() / denom < 1e-3,
                    "tq1_masked_sum avx2 {} scalar {} dim {}",
                    avx2, scalar, dim,
                );
            }
        }

        #[cfg(feature = "avx512")]
        #[test]
        fn prop_tq1_masked_sum_avx512_matches_scalar(
            bytes in 1usize..64,
            codes_data in proptest::collection::vec(proptest::num::u8::ANY, 0..64),
            q_data in proptest::collection::vec(-1.0f32..1.0, 0..512),
        ) {
            if std::is_x86_feature_detected!("avx512f") {
                let dim = bytes * 8;
                let mut code = codes_data;
                code.resize(dim / 8, 0);
                let mut q = q_data;
                q.resize(dim, 0.0);
                let avx512 = unsafe { tq1_masked_sum_avx512(&code, &q, dim) };
                let scalar = tq1_masked_sum_scalar(&code, &q, dim);
                let denom = scalar.abs().max(1.0);
                proptest::prop_assert!(
                    (avx512 - scalar).abs() / denom < 1e-3,
                    "tq1_masked_sum avx512 {} scalar {} dim {}",
                    avx512, scalar, dim,
                );
            }
        }
    }
}
