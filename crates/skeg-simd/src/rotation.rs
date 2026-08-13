//! `TurboQuant` rotation-path kernels: the Walsh-Hadamard transform, its
//! sign-diagonal flips, and the Lloyd-Max bucket lookup - the per-insert
//! *and* per-query cost of `FastRotation::apply` / `encode`.
//!
//! ## Why `fwht_f32_avx2` does not vectorise every stage
//!
//! The plan for this slice called for the FWHT's small-`h` stages (`h < 8`,
//! where the butterfly pair `(x[j], x[j+h])` sits *inside* one 8-lane
//! register) to be vectorised too, via an in-register shuffle/permute
//! network. That code was deliberately kept out of the initial implementation:
//! cross-compilation only proves it *compiles*, not that a lane-shuffle network
//! computes the right answer. Later Zen 4/Zen 5 validation covers the shipped
//! scalar-small-stage/vector-wide-stage implementation. A subtly wrong permute
//! is exactly the kind of bug that survives a clean compile and passes nothing
//! until it is run.
//!
//! `h >= 8` stages need no shuffle at all: `x[j]` and `x[j+h]` are `h`
//! elements apart with `h` a multiple of 8, so they land in two whole,
//! disjoint 8-lane loads - a plain vectorised add/sub, trivially checkable
//! by inspection and safe to ship without ever having run it. The `h < 8`
//! stages (at most 3, however large the block) are left as the exact
//! existing scalar code. This is a deliberate, documented scope reduction
//! from the plan, not an oversight: the risk of an unverifiable in-register
//! shuffle bug outweighs the win of vectorising three fixed-cost stages.

#[inline]
fn assert_fwht_layout(x: &[f32]) {
    assert!(x.len().is_power_of_two(), "FWHT requires a power of two");
}

/// In-place Walsh-Hadamard transform, unnormalised. `x.len()` must be a
/// power of two. After this call `||x'|| = sqrt(n) * ||x||`; the caller
/// scales as appropriate.
#[must_use]
/// # Panics
///
/// Panics if `x.len()` is not a power of two. An empty slice panics too:
/// zero is not a power of two, and a rotation of nothing is a caller bug
/// (`RotationTransform` derives its block from `largest_pow2_factor(dim)`,
/// asserted `>= 2`).
///
pub fn fwht_f32_scalar_ref(x: &[f32]) -> Vec<f32> {
    let mut out = x.to_vec();
    fwht_f32_scalar(&mut out);
    out
}

/// In-place Walsh-Hadamard transform, unnormalised, scalar reference.
///
/// # Panics
///
/// Panics if `x.len()` is not a power of two. An empty slice panics too:
/// zero is not a power of two, and a rotation of nothing is a caller bug
/// (`RotationTransform` derives its block from `largest_pow2_factor(dim)`,
/// asserted `>= 2`).
///
pub fn fwht_f32_scalar(x: &mut [f32]) {
    assert_fwht_layout(x);
    let n = x.len();
    let mut h = 1;
    while h < n {
        fwht_stage_scalar(x, h);
        h *= 2;
    }
}

/// One FWHT butterfly stage at distance `h`: `(x[j], x[j+h]) -> (x[j]+x[j+h],
/// x[j]-x[j+h])` for every `j` in every block of `2h`.
fn fwht_stage_scalar(x: &mut [f32], h: usize) {
    let n = x.len();
    let mut i = 0;
    while i < n {
        for j in i..i + h {
            let a = x[j];
            let b = x[j + h];
            x[j] = a + b;
            x[j + h] = a - b;
        }
        i += h * 2;
    }
}

/// AVX2 kernel for [`fwht_f32`]. See the module docs for why only the
/// `h >= 8` stages vectorise.
///
/// # Safety
///
/// The current CPU must support AVX2.
/// # Panics
///
/// Panics if `x.len()` is not a power of two. An empty slice panics too:
/// zero is not a power of two, and a rotation of nothing is a caller bug
/// (`RotationTransform` derives its block from `largest_pow2_factor(dim)`,
/// asserted `>= 2`).
///
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn fwht_f32_avx2(x: &mut [f32]) {
    assert_fwht_layout(x);
    use std::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_storeu_ps, _mm256_sub_ps};

    let n = x.len();
    let mut h = 1;
    while h < 8 && h < n {
        fwht_stage_scalar(x, h);
        h *= 2;
    }
    // SAFETY: the caller selected AVX2. `h` is a multiple of 8 for the rest
    // of this loop, so `j` (stepping by 8 within `i..i+h`) always leaves a
    // full 8-lane load in bounds: every visited `i` satisfies `i + 2h <= n`
    // (the same block-bound invariant the scalar stage relies on), and the
    // largest `j` reached within that block is `i + h - 8`, so the highest
    // index either load touches is `(i + h - 8) + h + 7 = i + 2h - 1 <= n - 1`.
    unsafe {
        while h < n {
            let mut i = 0;
            while i < n {
                let mut j = i;
                while j < i + h {
                    let a = _mm256_loadu_ps(x.as_ptr().add(j));
                    let b = _mm256_loadu_ps(x.as_ptr().add(j + h));
                    _mm256_storeu_ps(x.as_mut_ptr().add(j), _mm256_add_ps(a, b));
                    _mm256_storeu_ps(x.as_mut_ptr().add(j + h), _mm256_sub_ps(a, b));
                    j += 8;
                }
                i += h * 2;
            }
            h *= 2;
        }
    }
}

/// AVX-512 kernel for [`fwht_f32`]. Identical shape to the AVX2 one with
/// 16-lane butterflies; the `h == 8` stage still runs 8-wide because a
/// 16-lane load would cross the block boundary there.
///
/// # Safety
///
/// The current CPU must support AVX-512F and AVX2.
/// # Panics
///
/// Panics if `x.len()` is not a power of two. An empty slice panics too:
/// zero is not a power of two, and a rotation of nothing is a caller bug
/// (`RotationTransform` derives its block from `largest_pow2_factor(dim)`,
/// asserted `>= 2`).
///
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[target_feature(enable = "avx512f,avx2")]
pub unsafe fn fwht_f32_avx512(x: &mut [f32]) {
    assert_fwht_layout(x);
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_loadu_ps, _mm256_storeu_ps, _mm256_sub_ps, _mm512_add_ps,
        _mm512_loadu_ps, _mm512_storeu_ps, _mm512_sub_ps,
    };

    let n = x.len();
    let mut h = 1;
    while h < 8 && h < n {
        fwht_stage_scalar(x, h);
        h *= 2;
    }
    // SAFETY: the caller selected AVX-512F and AVX2. The bound argument is the
    // AVX2 kernel's, unchanged: within a block starting at `i` the largest
    // index either load touches is `i + 2h - 1 <= n - 1`, and `h` is a
    // multiple of the lane count of whichever width runs the stage.
    unsafe {
        if h == 8 && h < n {
            let mut i = 0;
            while i < n {
                let mut j = i;
                while j < i + h {
                    let a = _mm256_loadu_ps(x.as_ptr().add(j));
                    let b = _mm256_loadu_ps(x.as_ptr().add(j + h));
                    _mm256_storeu_ps(x.as_mut_ptr().add(j), _mm256_add_ps(a, b));
                    _mm256_storeu_ps(x.as_mut_ptr().add(j + h), _mm256_sub_ps(a, b));
                    j += 8;
                }
                i += h * 2;
            }
            h *= 2;
        }
        while h < n {
            let mut i = 0;
            while i < n {
                let mut j = i;
                while j < i + h {
                    let a = _mm512_loadu_ps(x.as_ptr().add(j));
                    let b = _mm512_loadu_ps(x.as_ptr().add(j + h));
                    _mm512_storeu_ps(x.as_mut_ptr().add(j), _mm512_add_ps(a, b));
                    _mm512_storeu_ps(x.as_mut_ptr().add(j + h), _mm512_sub_ps(a, b));
                    j += 16;
                }
                i += h * 2;
            }
            h *= 2;
        }
    }
}

/// NEON kernel for [`fwht_f32`]. Stages with `h >= 4` are two disjoint 4-lane
/// loads and a plain add/sub, no shuffle; the `h < 4` stages stay scalar for
/// the same reason the AVX2 kernel leaves `h < 8` scalar.
/// # Panics
///
/// Panics if `x.len()` is not a power of two. An empty slice panics too:
/// zero is not a power of two, and a rotation of nothing is a caller bug
/// (`RotationTransform` derives its block from `largest_pow2_factor(dim)`,
/// asserted `>= 2`).
///
#[cfg(target_arch = "aarch64")]
pub fn fwht_f32_neon(x: &mut [f32]) {
    assert_fwht_layout(x);
    use std::arch::aarch64::{vaddq_f32, vld1q_f32, vst1q_f32, vsubq_f32};

    let n = x.len();
    let mut h = 1;
    while h < 4 && h < n {
        fwht_stage_scalar(x, h);
        h *= 2;
    }
    // SAFETY: NEON is baseline on aarch64. `h` is a multiple of 4 from here
    // on, so within a block starting at `i` the highest index either load
    // touches is `(i + h - 4) + h + 3 = i + 2h - 1 <= n - 1`.
    unsafe {
        while h < n {
            let mut i = 0;
            while i < n {
                let mut j = i;
                while j < i + h {
                    let a = vld1q_f32(x.as_ptr().add(j));
                    let b = vld1q_f32(x.as_ptr().add(j + h));
                    vst1q_f32(x.as_mut_ptr().add(j), vaddq_f32(a, b));
                    vst1q_f32(x.as_mut_ptr().add(j + h), vsubq_f32(a, b));
                    j += 4;
                }
                i += h * 2;
            }
            h *= 2;
        }
    }
}

/// In-place Walsh-Hadamard transform, unnormalised. Dispatches to AVX2 on
/// x86_64 when available (see the module docs for its scope), to the
/// hand-written NEON kernel on aarch64 (see [`fwht_f32_neon`]; LLVM does not
/// auto-vectorise the scalar loop there, it emits scalar `s0`/`s1`/`s2`
/// registers, so the hand-written kernel is measured faster: 1.39 us against
/// 1.94 us at dim 1536 on an M1 Pro), the plain scalar loop elsewhere.
///
/// # Panics
///
/// Panics if `x.len()` is not a power of two.
pub fn fwht_f32(x: &mut [f32]) {
    assert_fwht_layout(x);
    // No AVX-512 arm: measured 1439 ns against AVX2's 1437 ns on a
    // Threadripper 7960X, a wash. The butterfly stages are memory-bound, not
    // width-bound. `fwht_f32_avx512` stays public and benched.
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 was checked immediately above.
            unsafe { fwht_f32_avx2(x) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        fwht_f32_neon(x)
    }
    #[cfg(not(target_arch = "aarch64"))]
    fwht_f32_scalar(x);
}

/// In-place sign flip on the indices marked by `mask` (LSB-first packing:
/// bit `i % 8` of byte `i / 8` selects coordinate `i`).
pub fn flip_signs_scalar(x: &mut [f32], mask: &[u8]) {
    for i in 0..x.len() {
        if (mask[i / 8] >> (i % 8)) & 1 == 1 {
            x[i] = -x[i];
        }
    }
}

/// AVX2 kernel for [`flip_signs`]. 8 lanes per iteration: the mask bits for
/// the current 8 coordinates are unpacked with a scalar loop (cheap, and
/// avoids needing any in-register bit-shuffle to line them up with the
/// float lanes - see the module docs on why unverifiable shuffles are
/// avoided here) into a `{0, 0x8000_0000}` sign-bit mask, then XORed into
/// the loaded floats in one vector op.
///
/// # Safety
///
/// The current CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn flip_signs_avx2(x: &mut [f32], mask: &[u8]) {
    assert!(
        mask.len() >= x.len().div_ceil(8),
        "mask must cover every coordinate: {} bytes for {} coords",
        mask.len(),
        x.len()
    );
    use std::arch::x86_64::{
        _mm256_castps_si256, _mm256_castsi256_ps, _mm256_loadu_ps, _mm256_loadu_si256,
        _mm256_storeu_ps, _mm256_xor_si256,
    };

    let n = x.len();
    let block = n - (n % 8);
    let mut i = 0;
    let mut bits = [0i32; 8];
    // SAFETY: the caller selected AVX2. `bits` is a fully-initialised stack
    // array read back immediately. Each `x` load/store covers 8 lanes at
    // `i + k < block <= n`.
    unsafe {
        while i < block {
            for (k, slot) in bits.iter_mut().enumerate() {
                let idx = i + k;
                *slot = if (mask[idx / 8] >> (idx % 8)) & 1 == 1 {
                    i32::MIN // 0x8000_0000: the f32 sign bit alone
                } else {
                    0
                };
            }
            let sign = _mm256_loadu_si256(bits.as_ptr().cast());
            let v = _mm256_castps_si256(_mm256_loadu_ps(x.as_ptr().add(i)));
            let flipped = _mm256_castsi256_ps(_mm256_xor_si256(v, sign));
            _mm256_storeu_ps(x.as_mut_ptr().add(i), flipped);
            i += 8;
        }
    }
    for idx in block..n {
        if (mask[idx / 8] >> (idx % 8)) & 1 == 1 {
            x[idx] = -x[idx];
        }
    }
}

/// AVX-512 kernel for [`flip_signs`]. The mask's LSB-first bit packing is
/// already the layout of an AVX-512 mask register, so two mask bytes read as
/// a `u16` drive `vsubps` under mask directly - no per-lane unpack loop, the
/// same trick that makes `tq1_masked_sum_avx512` cheap.
///
/// # Safety
///
/// The current CPU must support AVX-512F.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn flip_signs_avx512(x: &mut [f32], mask: &[u8]) {
    assert!(
        mask.len() >= x.len().div_ceil(8),
        "mask must cover every coordinate: {} bytes for {} coords",
        mask.len(),
        x.len()
    );
    use std::arch::x86_64::{
        _mm512_loadu_ps, _mm512_mask_sub_ps, _mm512_setzero_ps, _mm512_storeu_ps,
    };

    let n = x.len();
    let block = n - (n % 16);
    let mut i = 0;
    // SAFETY: the caller selected AVX-512F. Each `x` load/store covers 16
    // lanes at `i + k < block <= n`, and the two mask bytes read per
    // iteration are at `i / 8` and `i / 8 + 1`, both below `mask.len()`
    // because `i + 16 <= block <= n <= mask.len() * 8`.
    unsafe {
        let zero = _mm512_setzero_ps();
        while i < block {
            let bits = u16::from(mask[i / 8]) | (u16::from(mask[i / 8 + 1]) << 8);
            let v = _mm512_loadu_ps(x.as_ptr().add(i));
            // Negate exactly the selected lanes, leave the rest untouched.
            _mm512_storeu_ps(x.as_mut_ptr().add(i), _mm512_mask_sub_ps(v, bits, zero, v));
            i += 16;
        }
    }
    for idx in block..n {
        if (mask[idx / 8] >> (idx % 8)) & 1 == 1 {
            x[idx] = -x[idx];
        }
    }
}

/// NEON kernel for [`flip_signs`]. The mask byte for 8 coordinates is
/// broadcast and shifted per lane, so the sign-bit mask is built in-register
/// instead of by the scalar unpack loop the AVX2 kernel uses.
#[cfg(target_arch = "aarch64")]
pub fn flip_signs_neon(x: &mut [f32], mask: &[u8]) {
    assert!(
        mask.len() >= x.len().div_ceil(8),
        "mask must cover every coordinate: {} bytes for {} coords",
        mask.len(),
        x.len()
    );
    use std::arch::aarch64::{
        vandq_u32, vdupq_n_u32, veorq_u32, vld1q_f32, vld1q_s32, vreinterpretq_f32_u32,
        vreinterpretq_u32_f32, vshlq_n_u32, vshlq_u32, vst1q_f32,
    };

    let n = x.len();
    let block = n - (n % 8);
    let mut i = 0;
    // SAFETY: NEON is baseline on aarch64. Each iteration covers 8 lanes at
    // `i + 8 <= block <= n`, and reads mask byte `i / 8`, in bounds because
    // `mask.len() >= n.div_ceil(8)`.
    unsafe {
        let low_shifts = vld1q_s32([0i32, -1, -2, -3].as_ptr());
        let high_shifts = vld1q_s32([-4i32, -5, -6, -7].as_ptr());
        while i < block {
            let byte = vdupq_n_u32(u32::from(mask[i / 8]));
            let one = vdupq_n_u32(1);
            for (half, shifts) in [low_shifts, high_shifts].into_iter().enumerate() {
                let at = i + half * 4;
                // bit k of the byte -> lane k, moved into the f32 sign bit.
                let bits = vandq_u32(vshlq_u32(byte, shifts), one);
                let sign = vshlq_n_u32::<31>(bits);
                let v = vreinterpretq_u32_f32(vld1q_f32(x.as_ptr().add(at)));
                vst1q_f32(
                    x.as_mut_ptr().add(at),
                    vreinterpretq_f32_u32(veorq_u32(v, sign)),
                );
            }
            i += 8;
        }
    }
    for idx in block..n {
        if (mask[idx / 8] >> (idx % 8)) & 1 == 1 {
            x[idx] = -x[idx];
        }
    }
}

/// In-place sign flip on the indices marked by `mask`. Dispatches to AVX2
/// on x86_64 when available, NEON on aarch64, scalar elsewhere.
pub fn flip_signs(x: &mut [f32], mask: &[u8]) {
    assert!(
        mask.len() >= x.len().div_ceil(8),
        "mask must cover every coordinate: {} bytes for {} coords",
        mask.len(),
        x.len()
    );
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        if std::is_x86_feature_detected!("avx512f") {
            // SAFETY: AVX-512F was checked immediately above.
            unsafe { flip_signs_avx512(x, mask) };
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 was checked immediately above.
            unsafe { flip_signs_avx2(x, mask) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        flip_signs_neon(x, mask)
    }
    #[cfg(not(target_arch = "aarch64"))]
    flip_signs_scalar(x, mask);
}

/// Map one coordinate to its bucket index by counting how many sorted
/// `boundaries` it exceeds. Linear scan: for the 15-boundary (4-bit) case
/// this beats a branchless binary search on short inputs because the
/// boundary array stays in L1 and the compare-increment compiles tight.
#[must_use]
pub fn bucketize_scalar(x: f32, boundaries: &[f32]) -> usize {
    let mut bucket = 0usize;
    for &b in boundaries {
        if x > b {
            bucket += 1;
        }
    }
    bucket
}

/// Scalar reference for [`bucketize_x8`]: batches [`bucketize_scalar`] over
/// every coordinate in `x`.
pub fn bucketize_x8_scalar(x: &[f32], boundaries: &[f32], out: &mut [u8]) {
    assert_eq!(x.len(), out.len(), "bucketize_x8 length mismatch");
    for (o, &v) in out.iter_mut().zip(x.iter()) {
        #[allow(clippy::cast_possible_truncation)]
        {
            *o = bucketize_scalar(v, boundaries) as u8;
        }
    }
}

/// AVX2 kernel for [`bucketize_x8`]: 8 coordinates per iteration, one
/// `boundaries.len()`-deep pass of compare + conditional increment per
/// group (matches the scalar linear-scan shape, just 8-wide). Generalises
/// over any boundary count - both the 15-boundary (tq4) and 3-boundary
/// (tq2) callers share this kernel.
///
/// # Safety
///
/// The current CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn bucketize_x8_avx2(x: &[f32], boundaries: &[f32], out: &mut [u8]) {
    use std::arch::x86_64::{
        _CMP_GT_OQ, _mm256_add_epi32, _mm256_and_si256, _mm256_castps_si256, _mm256_cmp_ps,
        _mm256_loadu_ps, _mm256_set1_epi32, _mm256_set1_ps, _mm256_setzero_si256,
        _mm256_storeu_si256,
    };

    assert_eq!(x.len(), out.len(), "bucketize_x8 length mismatch");
    let n = x.len();
    let block = n - (n % 8);
    let ones = _mm256_set1_epi32(1);
    let mut i = 0;
    // SAFETY: the caller selected AVX2. Each `x` load covers 8 lanes at
    // `i + k < block <= n`; `lanes` is a fully-initialised stack array read
    // back immediately by `_mm256_storeu_si256`.
    unsafe {
        while i < block {
            let v = _mm256_loadu_ps(x.as_ptr().add(i));
            let mut acc = _mm256_setzero_si256();
            for &b in boundaries {
                let bv = _mm256_set1_ps(b);
                // OQ ("ordered, quiet"): NaN compares false, matching the
                // scalar `x > b` (false for NaN) instead of trapping.
                let mask = _mm256_castps_si256(_mm256_cmp_ps(v, bv, _CMP_GT_OQ));
                acc = _mm256_add_epi32(acc, _mm256_and_si256(mask, ones));
            }
            let mut lanes = [0i32; 8];
            _mm256_storeu_si256(lanes.as_mut_ptr().cast(), acc);
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            for (k, &lane) in lanes.iter().enumerate() {
                out[i + k] = lane as u8;
            }
            i += 8;
        }
    }
    for idx in block..n {
        #[allow(clippy::cast_possible_truncation)]
        {
            out[idx] = bucketize_scalar(x[idx], boundaries) as u8;
        }
    }
}

/// AVX-512 kernel for [`bucketize_x8`]: 16 coordinates per iteration. The
/// compare lands straight in a mask register, so the count is a masked
/// increment instead of the AVX2 and-with-ones, and `vpmovdb` writes the 16
/// bucket bytes in one store instead of a scalar write-back loop.
///
/// # Safety
///
/// The current CPU must support AVX-512F. `x.len()` must equal `out.len()`.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn bucketize_x8_avx512(x: &[f32], boundaries: &[f32], out: &mut [u8]) {
    use std::arch::x86_64::{
        __m128i, _CMP_GT_OQ, _mm_storeu_si128, _mm512_cmp_ps_mask, _mm512_cvtepi32_epi8,
        _mm512_loadu_ps, _mm512_mask_add_epi32, _mm512_set1_epi32, _mm512_set1_ps,
        _mm512_setzero_si512,
    };

    assert_eq!(x.len(), out.len(), "bucketize_x8 length mismatch");
    let n = x.len();
    let block = n - (n % 16);
    let ones = _mm512_set1_epi32(1);
    let mut i = 0;
    // SAFETY: the caller selected AVX-512F. Each `x` load covers 16 lanes at
    // `i + k < block <= n`, and the matching 16-byte store lands at the same
    // offset in `out`, whose length equals `x`'s (asserted above).
    unsafe {
        while i < block {
            let v = _mm512_loadu_ps(x.as_ptr().add(i));
            let mut acc = _mm512_setzero_si512();
            for &b in boundaries {
                // GT_OQ ("greater than, ordered, quiet"): NaN compares false,
                // matching the scalar `x > b` and the AVX2 kernel. NLE is the
                // unordered negation and counts NaN as above every boundary,
                // which is how this kernel first shipped and what the NaN test
                // caught on real hardware.
                let mask = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(v, _mm512_set1_ps(b));
                acc = _mm512_mask_add_epi32(acc, mask, acc, ones);
            }
            _mm_storeu_si128(
                out.as_mut_ptr().add(i).cast::<__m128i>(),
                _mm512_cvtepi32_epi8(acc),
            );
            i += 16;
        }
    }
    for idx in block..n {
        #[allow(clippy::cast_possible_truncation)]
        {
            out[idx] = bucketize_scalar(x[idx], boundaries) as u8;
        }
    }
}

/// NEON kernel for [`bucketize_x8`]. `vcgtq_f32` yields an all-ones lane for
/// `x > b` and all-zero otherwise (NaN compares false, matching the scalar
/// reference), so subtracting the compare result is the per-lane increment.
///
/// # Panics
///
/// Panics if `x.len() != out.len()`.
#[cfg(target_arch = "aarch64")]
pub fn bucketize_x8_neon(x: &[f32], boundaries: &[f32], out: &mut [u8]) {
    use std::arch::aarch64::{
        vcgtq_f32, vcombine_u16, vdupq_n_f32, vdupq_n_u32, vld1q_f32, vmovn_u16, vmovn_u32,
        vst1_u8, vsubq_u32,
    };

    assert_eq!(x.len(), out.len(), "bucketize_x8 length mismatch");
    let n = x.len();
    let block = n - (n % 8);
    let mut i = 0;
    // SAFETY: NEON is baseline on aarch64. Each iteration reads 8 f32 and
    // writes 8 u8 at `i + 8 <= block <= n`, and `out.len() == x.len()`.
    unsafe {
        while i < block {
            let v0 = vld1q_f32(x.as_ptr().add(i));
            let v1 = vld1q_f32(x.as_ptr().add(i + 4));
            let mut a0 = vdupq_n_u32(0);
            let mut a1 = vdupq_n_u32(0);
            for &b in boundaries {
                let bv = vdupq_n_f32(b);
                a0 = vsubq_u32(a0, vcgtq_f32(v0, bv));
                a1 = vsubq_u32(a1, vcgtq_f32(v1, bv));
            }
            let narrowed = vcombine_u16(vmovn_u32(a0), vmovn_u32(a1));
            vst1_u8(out.as_mut_ptr().add(i), vmovn_u16(narrowed));
            i += 8;
        }
    }
    for idx in block..n {
        #[allow(clippy::cast_possible_truncation)]
        {
            out[idx] = bucketize_scalar(x[idx], boundaries) as u8;
        }
    }
}

/// Map every coordinate in `x` to its bucket index (how many sorted
/// `boundaries` it exceeds). Dispatches to AVX2 on x86_64 when available,
/// NEON on aarch64, scalar elsewhere.
///
/// # Panics
///
/// Panics if `x.len() != out.len()`.
pub fn bucketize_x8(x: &[f32], boundaries: &[f32], out: &mut [u8]) {
    assert_eq!(x.len(), out.len(), "bucketize_x8 length mismatch");
    // No AVX-512 arm: measured on a Threadripper 7960X (Zen 4, 2026-08-09)
    // the 512-bit kernel runs 2070 ns against AVX2's 836 ns, 2.5x slower.
    // `bucketize_x8_avx512` stays public, tested and benched so the call can
    // be remeasured on a different microarchitecture.
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 was checked immediately above.
            unsafe { bucketize_x8_avx2(x, boundaries, out) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        bucketize_x8_neon(x, boundaries, out)
    }
    #[cfg(not(target_arch = "aarch64"))]
    bucketize_x8_scalar(x, boundaries, out);
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn norm(x: &[f32]) -> f32 {
        x.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    #[test]
    fn fwht_scalar_norm_invariant() {
        // ||FWHT(x)|| = sqrt(n) * ||x|| for any power-of-two n.
        let x = vec![1.0f32, 2.0, -3.0, 0.5, 4.0, -1.5, 2.5, -0.5];
        let n0 = norm(&x);
        let y = fwht_f32_scalar_ref(&x);
        let n1 = norm(&y);
        assert!((n1 - n0 * 8f32.sqrt()).abs() / n0 < 1e-4);
    }

    #[test]
    fn fwht_known_4point() {
        // Textbook 4-point WHT: [1,1,1,1] is a fixed ray (up to the sqrt(n)
        // scale) since it is the DC-only signal.
        let mut x = vec![1.0f32, 1.0, 1.0, 1.0];
        fwht_f32_scalar(&mut x);
        assert_eq!(x, vec![4.0, 0.0, 0.0, 0.0]);
    }

    /// Pinned, not incidental: before the guard was promoted from
    /// `debug_assert`, a release build treated an empty slice as a silent
    /// no-op. Rejecting it is the deliberate choice.
    #[test]
    fn fwht_rejects_an_empty_slice() {
        let mut x: Vec<f32> = Vec::new();
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| fwht_f32(&mut x))).is_err(),
            "FWHT accepted an empty slice",
        );
    }

    #[test]
    fn fwht_rejects_a_non_power_of_two_before_dispatch() {
        let mut x = vec![1.0f32; 3];
        let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| fwht_f32(&mut x)))
            .expect_err("FWHT accepted a non-power-of-two length");
        let msg = if let Some(s) = payload.downcast_ref::<&str>() {
            (*s).to_owned()
        } else if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        } else {
            "<non-string panic>".to_owned()
        };
        assert!(
            msg.contains("FWHT requires a power of two"),
            "FWHT panicked for the wrong reason: {msg}",
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn fwht_avx2_matches_scalar_various_sizes() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        for &n in &[8usize, 16, 32, 128, 1024] {
            let x: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 2.0) / 7.0).collect();
            let mut avx2 = x.clone();
            unsafe { fwht_f32_avx2(&mut avx2) };
            let mut scalar = x.clone();
            fwht_f32_scalar(&mut scalar);
            for (a, s) in avx2.iter().zip(scalar.iter()) {
                let denom = s.abs().max(1.0);
                assert!(
                    (a - s).abs() / denom < 1e-3,
                    "fwht avx2 {a} scalar {s} n {n}"
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn fwht_avx2_matches_scalar_unaligned_block_count() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        // n = 16 has a single h=8 AVX2 stage of exactly one 8-lane group -
        // exercises the boundary where the AVX2 loop body runs exactly once.
        let x: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) / 3.0).collect();
        let mut avx2 = x.clone();
        unsafe { fwht_f32_avx2(&mut avx2) };
        let mut scalar = x.clone();
        fwht_f32_scalar(&mut scalar);
        for (a, s) in avx2.iter().zip(scalar.iter()) {
            assert!((a - s).abs() / s.abs().max(1.0) < 1e-3);
        }
    }

    #[test]
    fn flip_signs_scalar_flips_marked_only() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        flip_signs_scalar(&mut x, &[0b0000_0101]); // flip coords 0 and 2
        assert_eq!(x, vec![-1.0, 2.0, -3.0, 4.0]);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn flip_signs_avx2_matches_scalar_for_unaligned_tail() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let n: usize = 259;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 - 100.0) / 17.0).collect();
        let mask: Vec<u8> = (0..n.div_ceil(8))
            .map(|i| (i as u8).wrapping_mul(53))
            .collect();
        let mut avx2 = x.clone();
        unsafe { flip_signs_avx2(&mut avx2, &mask) };
        let mut scalar = x.clone();
        flip_signs_scalar(&mut scalar, &mask);
        assert_eq!(
            avx2, scalar,
            "flip_signs is an exact sign-bit op, no tolerance needed"
        );
    }

    #[test]
    fn bucketize_scalar_matches_expected_buckets() {
        let boundaries = [-1.0f32, 0.0, 1.0];
        assert_eq!(bucketize_scalar(-2.0, &boundaries), 0);
        assert_eq!(bucketize_scalar(-0.5, &boundaries), 1);
        assert_eq!(bucketize_scalar(0.5, &boundaries), 2);
        assert_eq!(bucketize_scalar(2.0, &boundaries), 3);
    }

    #[test]
    fn bucketize_x8_scalar_matches_per_coord_scalar() {
        let boundaries = [-1.0f32, 0.0, 1.0];
        let x: Vec<f32> = (-10i16..10).map(|i| f32::from(i) / 4.0).collect();
        let mut out = vec![0u8; x.len()];
        bucketize_x8_scalar(&x, &boundaries, &mut out);
        for (i, &v) in x.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            {
                assert_eq!(out[i], bucketize_scalar(v, &boundaries) as u8);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn bucketize_x8_avx2_matches_scalar_15_boundaries() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        // 15 sorted boundaries, matching the tq4 (4-bit) case.
        let boundaries: Vec<f32> = (0..15).map(|i| (i as f32 - 7.0) / 3.0).collect();
        let x: Vec<f32> = (0..259).map(|i| (i as f32 - 130.0) / 11.0).collect();
        let mut avx2 = vec![0u8; x.len()];
        unsafe { bucketize_x8_avx2(&x, &boundaries, &mut avx2) };
        let mut scalar = vec![0u8; x.len()];
        bucketize_x8_scalar(&x, &boundaries, &mut scalar);
        assert_eq!(avx2, scalar);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn bucketize_x8_avx2_matches_scalar_3_boundaries() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        // 3 boundaries, matching the tq2 (2-bit) case.
        let boundaries = [-0.9816f32, 0.0, 0.9816];
        let x: Vec<f32> = (0..137).map(|i| (i as f32 - 68.0) / 13.0).collect();
        let mut avx2 = vec![0u8; x.len()];
        unsafe { bucketize_x8_avx2(&x, &boundaries, &mut avx2) };
        let mut scalar = vec![0u8; x.len()];
        bucketize_x8_scalar(&x, &boundaries, &mut scalar);
        assert_eq!(avx2, scalar);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn bucketize_x8_avx2_handles_nan_like_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let boundaries = [-1.0f32, 0.0, 1.0];
        let x = vec![f32::NAN; 8];
        let mut avx2 = vec![0u8; 8];
        unsafe { bucketize_x8_avx2(&x, &boundaries, &mut avx2) };
        let mut scalar = vec![0u8; 8];
        bucketize_x8_scalar(&x, &boundaries, &mut scalar);
        assert_eq!(avx2, scalar);
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[test]
    fn fwht_avx512_matches_scalar() {
        if !(std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx2")) {
            return;
        }
        // 1024 exercises every stage width; 8 and 16 pin the boundary
        // between the scalar, 8-wide and 16-wide stages.
        for n in [8usize, 16, 64, 1024] {
            let source: Vec<f32> = (0..n).map(|i| (i as f32 - 17.0) / 7.0).collect();
            let mut avx512 = source.clone();
            unsafe { fwht_f32_avx512(&mut avx512) };
            let mut scalar = source;
            fwht_f32_scalar(&mut scalar);
            for (a, s) in avx512.iter().zip(scalar.iter()) {
                assert!((a - s).abs() / s.abs().max(1.0) < 1e-6, "n {n}: {a} vs {s}");
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[test]
    fn flip_signs_avx512_matches_scalar_for_unaligned_tail() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        // 133 is not a multiple of 16: the scalar tail runs too.
        let n: usize = 133;
        let source: Vec<f32> = (0..n).map(|i| (i as f32 - 66.0) / 5.0).collect();
        let mask: Vec<u8> = (0..n.div_ceil(8))
            .map(|i| (i as u8).wrapping_mul(89))
            .collect();
        let mut avx512 = source.clone();
        unsafe { flip_signs_avx512(&mut avx512, &mask) };
        let mut scalar = source;
        flip_signs_scalar(&mut scalar, &mask);
        assert_eq!(avx512, scalar);
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[test]
    fn flip_signs_avx512_all_set_and_all_clear() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        let source: Vec<f32> = (0..32).map(|i| (i as f32) - 16.0).collect();
        let mut all_set = source.clone();
        unsafe { flip_signs_avx512(&mut all_set, &[0xFF; 4]) };
        for (flipped, original) in all_set.iter().zip(source.iter()) {
            assert_eq!(*flipped, -*original);
        }
        let mut all_clear = source.clone();
        unsafe { flip_signs_avx512(&mut all_clear, &[0x00; 4]) };
        assert_eq!(all_clear, source);
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[test]
    fn bucketize_x8_avx512_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        // 15 boundaries (tq4) and 3 (tq2), both with a non-multiple-of-16 len.
        let tq4: Vec<f32> = (0..15).map(|i| (i as f32 - 7.0) / 3.0).collect();
        let tq2 = vec![-0.9816f32, 0.0, 0.9816];
        for boundaries in [tq4, tq2] {
            let x: Vec<f32> = (0..259).map(|i| (i as f32 - 130.0) / 11.0).collect();
            let mut avx512 = vec![0u8; x.len()];
            unsafe { bucketize_x8_avx512(&x, &boundaries, &mut avx512) };
            let mut scalar = vec![0u8; x.len()];
            bucketize_x8_scalar(&x, &boundaries, &mut scalar);
            assert_eq!(avx512, scalar);
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[test]
    fn bucketize_x8_avx512_handles_nan_like_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        let boundaries = [-1.0f32, 0.0, 1.0];
        let x = vec![f32::NAN; 16];
        let mut avx512 = vec![0u8; 16];
        unsafe { bucketize_x8_avx512(&x, &boundaries, &mut avx512) };
        let mut scalar = vec![0u8; 16];
        bucketize_x8_scalar(&x, &boundaries, &mut scalar);
        assert_eq!(avx512, scalar);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn fwht_neon_matches_scalar() {
        for n in [4usize, 8, 64, 1024] {
            let source: Vec<f32> = (0..n).map(|i| (i as f32 - 17.0) / 7.0).collect();
            let mut neon = source.clone();
            fwht_f32_neon(&mut neon);
            let mut scalar = source;
            fwht_f32_scalar(&mut scalar);
            for (a, s) in neon.iter().zip(scalar.iter()) {
                assert!((a - s).abs() / s.abs().max(1.0) < 1e-6, "n {n}: {a} vs {s}");
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn flip_signs_neon_matches_scalar_for_unaligned_tail() {
        let n: usize = 133;
        let source: Vec<f32> = (0..n).map(|i| (i as f32 - 66.0) / 5.0).collect();
        let mask: Vec<u8> = (0..n.div_ceil(8))
            .map(|i| (i as u8).wrapping_mul(89))
            .collect();
        let mut neon = source.clone();
        flip_signs_neon(&mut neon, &mask);
        let mut scalar = source;
        flip_signs_scalar(&mut scalar, &mask);
        assert_eq!(neon, scalar);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn bucketize_x8_neon_matches_scalar() {
        let tq4: Vec<f32> = (0..15).map(|i| (i as f32 - 7.0) / 3.0).collect();
        let tq2 = vec![-0.9816f32, 0.0, 0.9816];
        for boundaries in [tq4, tq2] {
            let x: Vec<f32> = (0..259).map(|i| (i as f32 - 130.0) / 11.0).collect();
            let mut neon = vec![0u8; x.len()];
            bucketize_x8_neon(&x, &boundaries, &mut neon);
            let mut scalar = vec![0u8; x.len()];
            bucketize_x8_scalar(&x, &boundaries, &mut scalar);
            assert_eq!(neon, scalar);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn bucketize_x8_neon_handles_nan_like_scalar() {
        let boundaries = [-1.0f32, 0.0, 1.0];
        let x = vec![f32::NAN; 16];
        let mut neon = vec![0u8; 16];
        bucketize_x8_neon(&x, &boundaries, &mut neon);
        let mut scalar = vec![0u8; 16];
        bucketize_x8_scalar(&x, &boundaries, &mut scalar);
        assert_eq!(neon, scalar);
    }
}
