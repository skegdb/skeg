//! `skeg-simd` - distance kernels, SIMD-accelerated on aarch64 (NEON) and
//! x86_64 (AVX2/FMA when available).
//!
//! The public functions ([`cosine_f32`], [`hamming_binary`], [`dot_int8`])
//! dispatch to whichever kernel benchmarks fastest on the target: a hand-rolled
//! NEON and AVX2/FMA kernels for `cosine_f32` on their respective targets, and the
//! portable scalar fallback for `dot_int8` (AArch64 keeps it because LLVM
//! auto-vectorizes its multiply-accumulate better than baseline NEON without
//! `dotprod`, as benchmarks confirmed). Each scalar kernel doubles as the
//! reference oracle for its SIMD counterpart.
//!
//! unsafe is allowed in this crate (NEON intrinsics + raw-pointer loads);
//! every unsafe block documents the bounds invariant that makes it sound.
//!
//! The optional `avx512` feature (off by default) uses
//! intrinsics stabilised in Rust 1.89, one point release above this crate's
//! baseline MSRV (1.88) - enabling the feature already requires that newer
//! toolchain, so `clippy::incompatible_msrv` is silenced only for the code
//! it gates, not the crate as a whole.
//!
//! Measured on an EPYC 4564P (Zen 4, 2026-08-08): AVX-512 wins on hamming
//! (2.74 ns vs 4.05 ns), the int8 dot (14.2 ns vs 32.7 ns) and tq1_masked_sum
//! (49 ns vs 559 ns). The tq2/tq4 ADC kernels keep their table in a register
//! and permute it (`vpermps`) instead of gathering it from memory.
#![cfg_attr(feature = "avx512", allow(clippy::incompatible_msrv))]

pub mod block;
pub use block::{
    BLOCK, FLUSH_EVERY, build_tq4_lut_f32, interleave_tq4_codes, quantize_tq4_lut_u8,
    tq4_block32_score_scalar, tq4_block32_score_u8, tq4_block32_score_u8_scalar,
};

#[cfg(target_arch = "x86_64")]
pub use block::tq4_block32_score_u8_avx2;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub use block::tq4_block32_score_u8_avx512;
#[cfg(target_arch = "aarch64")]
pub use block::tq4_block32_score_u8_neon;

pub mod rotation;
pub use rotation::{
    bucketize_scalar, bucketize_x8, bucketize_x8_scalar, flip_signs, flip_signs_scalar, fwht_f32,
    fwht_f32_scalar,
};
#[cfg(target_arch = "x86_64")]
pub use rotation::{bucketize_x8_avx2, flip_signs_avx2, fwht_f32_avx2};
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub use rotation::{bucketize_x8_avx512, flip_signs_avx512, fwht_f32_avx512};
#[cfg(target_arch = "aarch64")]
pub use rotation::{bucketize_x8_neon, flip_signs_neon, fwht_f32_neon};

pub mod distance;
pub use distance::{
    cosine_f32, cosine_f32_scalar, dot_f32, dot_f32_scalar, dot_int8, dot_int8_scalar,
    hamming_binary, hamming_binary_scalar,
};
#[cfg(target_arch = "x86_64")]
pub use distance::{cosine_f32_avx2, dot_f32_avx2, dot_int8_avx2, hamming_binary_avx2};
#[cfg(target_arch = "aarch64")]
pub use distance::{cosine_f32_neon, dot_f32_neon, dot_int8_neon, dot_int8_sdot, hamming_binary_neon};
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub use distance::{dot_int8_avx512, hamming_binary_avx512};

pub mod adc;
pub use adc::{
    quantise_centroids_i8, tq1_bitplane_score, tq1_bitplane_score_scalar, tq1_masked_sum,
    tq1_masked_sum_scalar, tq1_masked_dot_qi8, tq1_masked_dot_qi8_scalar, tq2_adc_i8, tq2_adc_qi8, tq2_adc_qi8_scalar,
    tq4_adc_qi8, tq2_adc_i8_scalar, tq4_adc_i8, tq4_adc_i8_scalar,
};
#[cfg(target_arch = "aarch64")]
pub use adc::{tq1_bitplane_score_neon, tq1_masked_sum_neon, tq2_adc_i8_neon, tq4_adc_i8_neon};
#[cfg(target_arch = "x86_64")]
pub use adc::{tq1_masked_sum_avx2, tq2_adc_i8_avx2, tq4_adc_i8_avx2};
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub use adc::{tq1_masked_sum_avx512, tq2_adc_i8_avx512, tq4_adc_i8_avx512};

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
/// On x86_64 emits `_mm_prefetch` with a temporal L1 locality hint. Other
/// architectures use a no-op because stable Rust has no portable fallback.
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
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        // SAFETY: prefetch is a hardware hint. `_mm_prefetch` accepts an
        // arbitrary address and cannot read or write Rust memory directly.
        unsafe { _mm_prefetch(ptr.cast(), _MM_HINT_T0) };
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = ptr;
    }
}

/// Name of the active SIMD backend, for observability and tests.
#[must_use]
pub fn simd_backend() -> &'static str {
    #[cfg(target_arch = "aarch64")]
    {
        "neon"
    }
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "avx512")]
        {
            // This label is coarse on purpose: the kernels do not share one
            // requirement. tq1 needs avx512f alone; the ADC needs bw, vl and
            // ssse3 too (see `adc::avx512_adc_supported`); hamming and the
            // int8 dot need vpopcntdq and vnni. "avx512f" here means "some
            // AVX-512 kernel is reachable", not that every one of them is.
            if std::is_x86_feature_detected!("avx512f") {
                if std::is_x86_feature_detected!("avx512bw")
                    && std::is_x86_feature_detected!("avx512vpopcntdq")
                    && std::is_x86_feature_detected!("avx512vnni")
                {
                    return "avx512";
                }
                return "avx512f";
            }
        }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            "avx2"
        } else {
            "scalar"
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        "scalar"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_backend_is_reported() {
        let backend = simd_backend();
        assert!(["neon", "avx512", "avx512f", "avx2", "scalar"].contains(&backend));
    }
}
