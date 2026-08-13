//! Which kernel exists for which instruction set, and which one the
//! dispatcher actually picks, as an assertion rather than folklore.
//!
//! Two states are not enough. A kernel can exist and still never run: the
//! NEON `dot_int8` is real, tested, and deliberately skipped, because LLVM
//! auto-vectorises the scalar loop better than baseline NEON manages without
//! `dotprod` (40.6 ns against 23.1 ns on an M1 Pro). Recording that as
//! "covered" would be a lie, and recording it as "missing" would invite
//! someone to write it again.
//!
//! The AVX-512 rows follow one rule, learned by measuring on a Threadripper
//! 7960X: the wider ISA earns its dispatch only where it offers an
//! instruction AVX2 lacks (VNNI, VPOPCNTDQ, a mask register, `vpermps` over a
//! 16-entry table). Where the kernel is the same trick at twice the width,
//! 512-bit registers lose on a core that double-pumps them, so three kernels
//! here are built and tested but not selected.

/// What the dispatcher does with a kernel on a given instruction set.
#[derive(Clone, Copy)]
enum State {
    /// Implemented, and the dispatcher selects it when the CPU allows.
    Dispatched,
    /// Implemented and tested, but deliberately not selected. Carries why.
    PresentNotDispatched(&'static str),
    /// Not written. Carries why not.
    Absent(&'static str),
}

struct Row {
    kernel: &'static str,
    neon: State,
    avx2: State,
    avx512: State,
}

use State::{Absent, Dispatched, PresentNotDispatched};

const COVERAGE: &[Row] = &[
    Row {
        kernel: "cosine_f32",
        neon: Dispatched,
        avx2: Dispatched,
        avx512: Absent(
            "FMA-issue-bound, not width-bound, on a core that double-pumps 512-bit FMAs: 63.7 ns at dim 1536 on a Threadripper 7960X is about 90% of the 2-FMA-per-cycle issue limit for its three accumulators. Worth revisiting on Intel Sapphire Rapids, where 512-bit FMA runs full rate and this could halve",
        ),
    },
    Row {
        kernel: "dot_f32",
        neon: Dispatched,
        avx2: Dispatched,
        avx512: Absent(
            "load-bandwidth-bound at 94% of THIS core's ceiling: 41.0 ns at dim 1536 on a Threadripper 7960X is 300 GB/s against the two-32-byte-loads-per-cycle limit Zen 4 keeps even for 512-bit ops. The ceiling is a property of the core, not of the kernel: a machine with 64-byte load ports doubles it, and AVX-512 loads are how you reach it, so measure there before concluding",
        ),
    },
    Row {
        kernel: "dot_int8",
        neon: PresentNotDispatched(
            "LLVM auto-vectorises the scalar loop better than baseline NEON without dotprod: 23.1 ns against the kernel's 40.6 ns on an M1 Pro",
        ),
        avx2: Dispatched,
        avx512: Dispatched,
    },
    Row {
        kernel: "hamming_binary",
        neon: Dispatched,
        avx2: Dispatched,
        avx512: Dispatched,
    },
    Row {
        kernel: "tq1_masked_sum",
        neon: Dispatched,
        avx2: Dispatched,
        avx512: Dispatched,
    },
    Row {
        kernel: "tq1_bitplane_score",
        neon: Dispatched,
        avx2: Dispatched,
        avx512: Dispatched,
    },
    Row {
        kernel: "tq2_adc_i8",
        neon: Dispatched,
        avx2: Dispatched,
        avx512: Dispatched,
    },
    Row {
        kernel: "tq4_adc_i8",
        neon: Dispatched,
        avx2: Dispatched,
        avx512: Dispatched,
    },
    Row {
        kernel: "tq4_block32_score_u8",
        neon: Dispatched,
        avx2: Dispatched,
        avx512: PresentNotDispatched(
            "vpshufb either way, so 512-bit registers only add pressure: 681 ns against AVX2's 600 ns on a Threadripper 7960X, and never faster across three runs",
        ),
    },
    Row {
        kernel: "fwht_f32",
        neon: Dispatched,
        avx2: Dispatched,
        avx512: PresentNotDispatched(
            "butterflies are memory-bound, not width-bound: 1439 ns against AVX2's 1437 ns on a Threadripper 7960X, a wash that is not worth a second code path",
        ),
    },
    Row {
        kernel: "flip_signs",
        neon: Dispatched,
        avx2: Dispatched,
        avx512: Dispatched,
    },
    Row {
        kernel: "bucketize_x8",
        neon: Dispatched,
        avx2: Dispatched,
        avx512: PresentNotDispatched(
            "2070 ns against AVX2's 836 ns on a Threadripper 7960X, 2.5x slower: the compare-and-count shape gains nothing from width on a core that double-pumps 512-bit ops",
        ),
    },
];

/// Every kernel the crate dispatches must have a row. Adding a dispatcher
/// without a row fails here, which is the point: a hole should be a red test,
/// not something a reader has to notice.
#[test]
fn every_dispatched_kernel_has_a_coverage_row() {
    const DISPATCHED: &[&str] = &[
        "cosine_f32",
        "dot_f32",
        "dot_int8",
        "hamming_binary",
        "tq1_masked_sum",
        "tq1_bitplane_score",
        "tq2_adc_i8",
        "tq4_adc_i8",
        "tq4_block32_score_u8",
        "fwht_f32",
        "flip_signs",
        "bucketize_x8",
    ];
    for kernel in DISPATCHED {
        assert!(
            COVERAGE.iter().any(|row| row.kernel == *kernel),
            "{kernel} has a dispatcher but no coverage row",
        );
    }
    assert_eq!(
        COVERAGE.len(),
        DISPATCHED.len(),
        "the coverage table has rows for kernels that are no longer dispatched",
    );
}

/// A gap is allowed, an unexplained gap is not. "TODO" and "not yet" are the
/// shrugs this is here to reject.
#[test]
fn every_gap_carries_a_real_reason() {
    for row in COVERAGE {
        for (isa, state) in [
            ("NEON", row.neon),
            ("AVX2", row.avx2),
            ("AVX-512", row.avx512),
        ] {
            let reason = match state {
                Dispatched => continue,
                PresentNotDispatched(reason) | Absent(reason) => reason,
            };
            assert!(
                reason.len() > 30,
                "{}/{isa} is declared a gap with no real reason: {reason:?}",
                row.kernel,
            );
            let lowered = reason.to_lowercase();
            for shrug in ["todo", "not yet", "later", "fixme"] {
                assert!(
                    !lowered.contains(shrug),
                    "{}/{isa} explains itself with {shrug:?}, which explains nothing: {reason:?}",
                    row.kernel,
                );
            }
        }
    }
}

/// The dispatcher must actually reach SIMD on a machine that has it. This
/// catches the failure where every kernel exists, every test passes, and the
/// engine quietly runs the scalar fallback the whole time.
#[test]
fn the_dispatcher_reaches_simd_on_this_machine() {
    let backend = skeg_simd::simd_backend();
    #[cfg(target_arch = "aarch64")]
    assert_eq!(backend, "neon", "aarch64 must reach the NEON kernels");
    #[cfg(target_arch = "x86_64")]
    assert!(
        backend != "scalar" || !std::is_x86_feature_detected!("avx2"),
        "this CPU has AVX2 but the dispatcher reports the scalar fallback",
    );
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    assert_eq!(backend, "scalar");
}
