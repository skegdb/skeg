//! Quantized vector representations for the flat-scan tier.
//!
//! A [`QuantizedVectors`] holds a whole vector set in a compact form that the
//! flat scan walks fast: 8-bit integers (1/4 the bytes of f32) or 1-bit signs
//! (1/32). The quantized distance is only a *proxy* for ranking; the caller
//! re-ranks the surviving candidates with exact f32 cosine.

use rayon::prelude::*;
use skeg_platform::MappedFile;
use skeg_simd::{dot_int8, hamming_binary, quantise_centroids_i8, tq2_adc_i8, tq4_adc_i8};

/// Storage backing for a TurboQuant code buffer. The default is `Owned` (a
/// heap `Vec<u8>`, matching the path the rest of the tier uses today).
/// `Mapped` holds the same byte sequence as a memory-mapped file - the OS
/// page cache decides which pages stay resident, so under memory pressure
/// the tier pages can be reclaimed and re-read from disk, instead of being
/// pushed to anonymous swap. Opt-in via `--tier-mmap`.
#[derive(Debug)]
pub(crate) enum CodeBacking {
    Owned(Vec<u8>),
    Mapped(MappedFile),
}

impl CodeBacking {
    pub(crate) fn as_slice(&self) -> &[u8] {
        match self {
            CodeBacking::Owned(v) => v.as_slice(),
            CodeBacking::Mapped(m) => m.as_bytes(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            CodeBacking::Owned(v) => v.len(),
            CodeBacking::Mapped(m) => m.len(),
        }
    }
}

use crate::turboquant::{FastRotation, lm_boundaries_n01, lm_centroids_n01, tq1_adc_swar};

/// PQ codebook training sample cap: k-means trains on at most this many rows.
const PQ_TRAIN_SAMPLE: usize = 50_000;
/// Maps an ADC squared-L2 distance to the i32 "greater is closer" proxy
/// contract. Unit-vector squared-L2 is in `[0, 4]`; `1e7` keeps the ordering
/// with ~1e-7 resolution and stays well inside `i32`.
const PQ_PROXY_SCALE: f32 = 1e7;
/// Maps a TurboQuant asymmetric inner product (in roughly `[-1, 1]` for unit
/// vectors) to the i32 "greater is closer" proxy contract. Resolution
/// ~1e-7, well inside `i32` even with adversarial scales.
const TQ_PROXY_SCALE: f32 = 1e7;
/// Default rotation seed for production TurboQuant tier. Fixed: a future
/// algorithm change would require a version bump; for v0.2 the rotation is
/// deterministic from `(dim, bits, TQ_ROTATION_SEED)`.
const TQ_ROTATION_SEED: u64 = 0xC0DE_BEEF;

/// Low-dim tq1 auto dim-expansion: a 1-bit code at native dim `d` gives only `d`
/// bits of discrimination, too few below ~256 (glove-100: recall@100 0.79). The
/// rotation is data-oblivious, so zero-padding the unit vector up to
/// [`TQ1_EXPAND_TO`] before rotating projects the `d` real coords onto more
/// random sign bits (a JL-style richer signature) - glove-100 -> 0.99 @ 1M, +~50
/// bits RAM/vector, f32 stays native on disk. Only the 1-bit codes grow.
const TQ1_EXPAND_BELOW: usize = 256;
const TQ1_EXPAND_TO: usize = 512;

/// Working (rotation/code) dimension for a TurboQuant tier: native `dim`, except
/// low-dim 1-bit which expands to [`TQ1_EXPAND_TO`] for more code bits.
#[must_use]
fn tq_code_dim(dim: usize, bits: u8) -> usize {
    if bits == 1 && dim < TQ1_EXPAND_BELOW {
        TQ1_EXPAND_TO
    } else {
        dim
    }
}

/// How a vector set is quantized for the flat scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantKind {
    /// Full precision; the flat scan runs f32 cosine directly, no re-rank.
    F32,
    /// Symmetric 8-bit integers; scan with an int8 dot product proxy.
    Int8,
    /// 1-bit sign quantization; scan with a Hamming-distance proxy.
    Binary,
    /// Product Quantization: `m` subvectors, each mapped to one of `k`
    /// k-means centroids. `m` bytes per vector. Scan with an asymmetric
    /// distance (ADC) proxy over a per-query lookup table.
    Pq { m: usize, k: usize },
    /// TurboQuant (Zandieh et al. 2025): random orthogonal rotation +
    /// Lloyd-Max scalar quantizer + per-vector scale correction. Data-
    /// oblivious (no training pass, deterministic from seed). `bits` in
    /// {1, 2, 4} chooses the precision/byte tradeoff:
    /// - 1-bit: `dim/8` bytes/vec, walk recall ~0.97 (tight on dim-deads
    ///   anisotropic models like MiniLM)
    /// - 2-bit: `dim/4` bytes/vec, walk recall ~0.98
    /// - 4-bit: `dim/2` bytes/vec, walk recall ~0.99
    TurboQuant { bits: u8 },
}

impl QuantKind {
    /// Check that this kind can quantize vectors of `dim`, so the create path
    /// can reject a bad combination cleanly instead of panicking deep in the
    /// builder. TurboQuant bit-packs `8 / bits` codes per byte (and its block
    /// rotation needs an even `dim`, which divisibility by `8 / bits` already
    /// guarantees); PQ splits `dim` into `m` subvectors. Other kinds take any
    /// positive `dim`.
    ///
    /// # Errors
    ///
    /// Returns a human-readable reason if `dim` is incompatible with this kind.
    pub fn validate_dim(self, dim: usize) -> Result<(), String> {
        match self {
            QuantKind::TurboQuant { bits } => {
                let codes_per_byte = 8 / usize::from(bits);
                if dim % codes_per_byte != 0 {
                    return Err(format!(
                        "tq{bits} requires a dimension divisible by {codes_per_byte} (got {dim})"
                    ));
                }
                Ok(())
            }
            QuantKind::Pq { m, .. } => {
                if m == 0 || dim % m != 0 {
                    return Err(format!(
                        "PQ requires the dimension ({dim}) to be divisible by m ({m})"
                    ));
                }
                Ok(())
            }
            QuantKind::F32 | QuantKind::Int8 | QuantKind::Binary => Ok(()),
        }
    }
}

/// A query vector quantized to match a [`QuantizedVectors`] set.
#[derive(Debug, Clone)]
pub enum QueryCode {
    /// 8-bit integer query, `dim` elements.
    Int8(Vec<i8>),
    /// 1-bit sign query, `dim.div_ceil(8)` bytes.
    Binary(Vec<u8>),
    /// PQ asymmetric-distance lookup table, `m * k` f32: `lut[s * k + c]` is
    /// the squared L2 of query subvector `s` to centroid `c`.
    Pq(Vec<f32>),
    /// TurboQuant rotated unit query. `q_rot` is the rotation applied to
    /// the unit-normalised query (`dim` f32); `q_sum = sum(q_rot)` is the
    /// query-level scalar needed by the 1-bit algebraic ADC reduction
    /// `c * (2 * masked_sum - q_sum)`, precomputed here so the walk does
    /// not recompute it per ADC call (6400 ADC/query x dim adds saved on
    /// the tq1 path). The 4/2-bit kernels ignore `q_sum`.
    ///
    /// With 1-bit anisotropy compensation, `q_rot` holds the *compensated*
    /// query `q_rot[d]*inv_scale[d]`, `q_sum` its sum, and `qm = sum(q_raw*shift)`
    /// the scalar shift correction (`E_hat = c*(2*masked - q_sum) - qm`). `qm = 0`
    /// and `q_rot` unchanged when compensation is off (identity), so the kernel
    /// contract is unchanged.
    TurboQuant {
        q_rot: Vec<f32>,
        q_sum: f32,
        qm: f32,
    },
    /// TurboQuant 1-bit symmetric query: the rotated unit query reduced to its
    /// sign bits (`dim.div_ceil(8)` bytes, same LSB-first packing as the stored
    /// codes). Scored by Hamming popcount against a stored code. Produced only
    /// when the index's tq1 proxy mode is [`Tq1ProxyMode::Popcount`].
    TurboQuant1Popcount { q_bits: Vec<u8> },
    /// TurboQuant 1-bit hybrid query: carries BOTH the sign bits (for the cheap
    /// popcount walk) and the rotated f32 query + `q_sum` (for the asymmetric
    /// re-score of the walk's survivors). The walk uses [`proxy`](QuantizedVectors::proxy)
    /// (popcount, fast navigation); the candidate list is then reordered by
    /// [`proxy_rescore`](QuantizedVectors::proxy_rescore) (asymmetric, in-RAM)
    /// before the exact rerank, so the limited f32/disk rerank budget is spent
    /// on the asymmetrically-best candidates. Produced for [`Tq1ProxyMode::Hybrid`].
    TurboQuant1Hybrid {
        q_bits: Vec<u8>,
        q_rot: Vec<f32>,
        q_sum: f32,
        qm: f32,
    },
    /// TurboQuant 1-bit bit-plane query: the rotated query scalar-quantized to
    /// `b` bits and transposed into `b` bit-planes (`planes[p]` is one
    /// `dim.div_ceil(8)`-byte mask). Scored against a stored sign code by
    /// `b + 1` integer popcounts. `m` (min), `sq` (quant step), `sum_q`
    /// (sum of quantized levels) are the query scalars for the exact
    /// reconstruction `q_i ~= m + sq*Q_i`. Produced for [`Tq1ProxyMode::BitPlane`].
    TurboQuant1BitPlane {
        planes: Vec<u8>,
        b: u8,
        bytes: usize,
        m: f32,
        sq: f32,
        sum_q: f32,
        qm: f32,
    },
}

impl QueryCode {
    /// True if this is a tq1 hybrid query, i.e. the post-walk candidate list
    /// should be re-scored via [`QuantizedVectors::proxy_rescore`] before the
    /// exact rerank. Lets the search gate the re-score without importing the
    /// variant or paying it for other modes.
    #[must_use]
    pub fn is_tq1_hybrid(&self) -> bool {
        matches!(self, QueryCode::TurboQuant1Hybrid { .. })
    }

    /// True when the proxy score is on the cosine scale (`ip * TQ_PROXY_SCALE`,
    /// i.e. `-pscore/1e7 ~= cosine`): the asymmetric and bit-plane tq1 arms. The
    /// adaptive rerank bound relies on this - popcount (Hamming), int8 (raw
    /// dot), and PQ scores are NOT cosine-scaled, so the bound must not prune
    /// them (it would compare a ~1e-3 or negative estimate to a ~0.8 threshold
    /// and skip every remaining candidate).
    #[must_use]
    pub fn is_cosine_scale_proxy(&self) -> bool {
        matches!(
            self,
            QueryCode::TurboQuant1BitPlane { .. } | QueryCode::TurboQuant { .. }
        )
    }
}

/// Which proxy the 1-bit TurboQuant tier uses during the graph walk. Both modes
/// read the *same* stored codes (rotated sign bits); only the query encoding and
/// kernel differ, so this is a pure query-time choice with no storage cost.
///
/// It is a deterministic function of `(dim, bits)` (see [`tq1_proxy_mode_for`]),
/// so it is recomputed on load rather than persisted - nothing to migrate, and
/// it never drifts under streaming inserts (the rotation is data-oblivious).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tq1ProxyMode {
    /// Query stays f32; masked-sum ADC times the per-vector scale. Robust recall
    /// at any dimension. The safe default (still half the RAM of tq2).
    Asymmetric,
    /// Query binarized to sign bits; Hamming popcount, scale ignored. ~2x faster
    /// per candidate, but sign-only recall is only competitive at high dim.
    Popcount,
    /// Popcount for the walk (cheap navigation), then the survivors are
    /// re-scored by the asymmetric proxy in-RAM before the exact rerank. Recovers
    /// ~90% of popcount's recall gap (it is mostly ranking, not navigation) for
    /// a small fraction of the asymmetric walk cost - and spends the disk rerank
    /// budget on the better candidates. The preferred fast tier: it dominates
    /// pure popcount on recall at a tiny extra cost.
    Hybrid,
    /// Asymmetric-but-integer: the query is scalar-quantized to `B` bits and
    /// transposed into `B` bit-planes; the inner product against a stored sign
    /// code is `B+1` integer popcounts (no f32 per candidate). Recovers most of
    /// the asymmetric recall at a fraction of the f32-ADC latency (TurboQuant
    /// "multi-bit query x 1-bit code"). `B` from `SKEG_TQ1_BITPLANE_B` (default 4).
    BitPlane,
}

/// Bit-width for the [`Tq1ProxyMode::BitPlane`] query. `SKEG_TQ1_BITPLANE_B`, 1..=8, default 4.
fn tq1_bitplane_bits() -> u8 {
    static B: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *B.get_or_init(|| {
        std::env::var("SKEG_TQ1_BITPLANE_B")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|b| (1..=8).contains(b))
            .unwrap_or(4)
    })
}

/// Minimum dimension at which the fast tq1 path (hybrid) is selected. Below it,
/// even the asym re-score can't recover popcount's navigation loss (glove-104
/// recovers only 63% vs 88-96% at dim >= 384), so the safe asymmetric proxy is
/// used. The hybrid recovers ~90% of the popcount gap from dim 384 up, so the
/// fast path is viable far below the pure-popcount threshold; 512 is a
/// conservative floor. A design constant, not a per-index calibration.
pub const TQ1_HYBRID_MIN_DIM: usize = 512;

/// Pick the tq1 proxy mode from the static index parameters alone - decided at
/// `VINDEX.CREATE` from `dim`, before any insert, with no reads. Only 1-bit has
/// a choice; 2/4-bit always report `Asymmetric`. The 1-bit default is BitPlane
/// (B-bit integer asymmetric ADC): it matches asymmetric recall at ~40% lower
/// p50 and dominates the old dim-gated popcount/hybrid switch at every dim, so
/// it is a single mode with no dimensional special-casing.
#[must_use]
pub fn tq1_proxy_mode_for(dim: usize, bits: u8) -> Tq1ProxyMode {
    let _ = dim; // no longer dimension-gated; kept for API/signature stability
    if bits != 1 {
        return Tq1ProxyMode::Asymmetric;
    }
    // Benchmark/tuning hook: `SKEG_TQ1_MODE=pop|hybrid|asym|bitplane` forces the
    // 1-bit proxy so each can be measured on one built index. Read once
    // (OnceLock) - zero per-query cost, no effect when unset.
    if let Some(m) = tq1_mode_override() {
        return m;
    }
    Tq1ProxyMode::BitPlane
}

fn tq1_mode_override() -> Option<Tq1ProxyMode> {
    static OVERRIDE: std::sync::OnceLock<Option<Tq1ProxyMode>> = std::sync::OnceLock::new();
    *OVERRIDE.get_or_init(|| match std::env::var("SKEG_TQ1_MODE").ok().as_deref() {
        Some("pop" | "popcount") => Some(Tq1ProxyMode::Popcount),
        Some("hybrid") => Some(Tq1ProxyMode::Hybrid),
        Some("asym" | "asymmetric") => Some(Tq1ProxyMode::Asymmetric),
        Some("bitplane" | "bp") => Some(Tq1ProxyMode::BitPlane),
        _ => None,
    })
}

#[derive(Debug)]
enum QuantRepr {
    /// Row-major i8, `n * dim` elements. `scale` maps i8 units back to f32.
    Int8 { data: Vec<i8>, scale: f32 },
    /// Row-major packed sign bits, `n * bytes` elements.
    Binary { data: Vec<u8>, bytes: usize },
    /// PQ codes (`n * m` bytes, one centroid id per subvector) plus the
    /// trained codebook (`m * k * sub_dim` f32, centroids of subvector `s`
    /// at `codebook[s]`). Trained on unit-normalised vectors so the ADC
    /// squared-L2 ranks identically to cosine.
    Pq {
        codes: Vec<u8>,
        codebook: Vec<Vec<f32>>,
        m: usize,
        k: usize,
        sub_dim: usize,
    },
    /// TurboQuant codes (`n * code_bytes` bytes, bit-packed Lloyd-Max
    /// indices) plus per-vector scale correction. The rotation is a fast
    /// block Walsh-Hadamard with random sign masks: `O(d log block)` per
    /// apply, ~30-100x faster than the dense Gram-Schmidt at dim 1024,
    /// with a 3-pass structured-random construction that converges
    /// statistically to a uniform orthogonal transform. Data-oblivious:
    /// rotation and Lloyd-Max levels are deterministic functions of
    /// `(dim, bits, seed)`, no training pass.
    TurboQuant {
        codes: CodeBacking,
        scales: Vec<f32>,
        rotation: Box<FastRotation>,
        /// Lloyd-Max centroids scaled to the post-rotation Beta variance
        /// `1/dim`. `2^bits` entries, ascending.
        centroids: Vec<f32>,
        /// Lloyd-Max bucket boundaries scaled to `1/dim`. `2^bits - 1`
        /// entries (empty when `bits == 1`, in which case the threshold is
        /// exactly zero).
        boundaries: Vec<f32>,
        /// `bits == 4` only: i8-quantised centroids and their shared scale,
        /// used by the NEON `vqtbl1q_s8` SIMD ADC kernel. The 16 i8 values
        /// fit in one Q register; `centroids_i8[k] * i8_scale` recovers
        /// the f32 centroid to ~1.5% MSE on a typical Gaussian variance.
        /// Empty + scale 0.0 when `bits != 4`.
        centroids_i8: [i8; 16],
        i8_scale: f32,
        bits: u8,
        /// Bytes per stored code = `dim * bits / 8`.
        code_bytes: usize,
        /// 1-bit anisotropy compensation `(shift, inv_scale)`; identity (empty)
        /// unless `SKEG_TQ1_ANISO` is set and `bits == 1`.
        aniso: Tq1Aniso,
    },
}

/// A vector set stored in a quantized form for fast flat scanning.
#[derive(Debug)]
pub struct QuantizedVectors {
    dim: usize,
    n: usize,
    repr: QuantRepr,
}

/// Calibrate a symmetric int8 scale: the largest magnitude maps to 127.
fn calibrate_int8(data: &[f32]) -> f32 {
    let max_abs = data.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 }
}

/// Quantize one f32 component to i8 with the given scale.
#[allow(clippy::cast_possible_truncation)] // value is clamped into i8 range first
fn quantize_i8(x: f32, scale: f32) -> i8 {
    (x / scale).round().clamp(-127.0, 127.0) as i8
}

/// Pack sign bits of `vec` into `out`: bit `i` set iff component `i` is > 0.
fn pack_signs(vec: &[f32], out: &mut [u8]) {
    for byte in out.iter_mut() {
        *byte = 0;
    }
    for (i, &x) in vec.iter().enumerate() {
        if x > 0.0 {
            out[i / 8] |= 1 << (i % 8);
        }
    }
}

/// Squared L2 distance between two equal-length slices.
fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| (x - y) * (x - y)).sum()
}

/// Unit-normalise `v` into `out` (same length). A zero vector is copied as-is.
fn normalize_into(v: &[f32], out: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        out.copy_from_slice(v);
    } else {
        for (o, &x) in out.iter_mut().zip(v) {
            *o = x / norm;
        }
    }
}

/// QuIVer: rotated-sign metric vectors for graph construction. Each row is the
/// tq1 sign pattern (`+/-1`) of `FastRotation(unit(v))` - the *same* signs the
/// 1-bit code stores. Squared-L2 on `{+/-1}` vectors is `4*Hamming`, so f32
/// cosine/L2 distance on these ranks monotone in the popcount distance the
/// 1-bit walk navigates with. Building the Vamana graph on this metric (instead
/// of exact f32) co-designs the topology with the quantizer, so the cheap
/// popcount walk can actually follow its edges (QuIVer, arXiv 2605.02171).
/// Data-oblivious rotation (seed [`TQ_ROTATION_SEED`]) - no dependence on the
/// quant tier, computed straight from the f32 rows at build time.
#[must_use]
pub fn tq1_sign_metric_vectors(vectors: &[f32], n: usize, dim: usize) -> Vec<f32> {
    let rotation = FastRotation::new(dim, TQ_ROTATION_SEED);
    let mut out = vec![0.0f32; n * dim];
    out.par_chunks_mut(dim)
        .zip(vectors.par_chunks(dim))
        .for_each(|(o, row)| {
            let mut unit = vec![0.0f32; dim];
            normalize_into(row, &mut unit);
            let rot = rotation.apply_alloc(&unit);
            for (oi, &r) in o.iter_mut().zip(&rot) {
                *oi = if r >= 0.0 { 1.0 } else { -1.0 };
            }
        });
    out
}

/// Lloyd's k-means over `n` rows of `dim` (flat `points`), `k` centroids,
/// `iters` iterations, deterministic xorshift init. Empty clusters reseed to
/// a random point. Returns the `k * dim` centroids.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn kmeans(points: &[f32], n: usize, dim: usize, k: usize, iters: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut centroids = vec![0.0f32; k * dim];
    for c in 0..k {
        let r = (next() as usize) % n;
        centroids[c * dim..(c + 1) * dim].copy_from_slice(&points[r * dim..(r + 1) * dim]);
    }
    let mut assign = vec![0u32; n];
    for _ in 0..iters {
        for (i, a) in assign.iter_mut().enumerate() {
            let p = &points[i * dim..(i + 1) * dim];
            let mut best = 0u32;
            let mut best_d = f32::MAX;
            for c in 0..k {
                let d = sq_l2(p, &centroids[c * dim..(c + 1) * dim]);
                if d < best_d {
                    best_d = d;
                    best = c as u32;
                }
            }
            *a = best;
        }
        let mut sums = vec![0.0f32; k * dim];
        let mut counts = vec![0u32; k];
        for (i, &a) in assign.iter().enumerate() {
            let c = a as usize;
            counts[c] += 1;
            for (s, &x) in sums[c * dim..(c + 1) * dim]
                .iter_mut()
                .zip(&points[i * dim..(i + 1) * dim])
            {
                *s += x;
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                let r = (next() as usize) % n;
                centroids[c * dim..(c + 1) * dim].copy_from_slice(&points[r * dim..(r + 1) * dim]);
            } else {
                let inv = 1.0 / counts[c] as f32;
                for s in 0..dim {
                    centroids[c * dim + s] = sums[c * dim + s] * inv;
                }
            }
        }
    }
    centroids
}

/// Train one k-means codebook per subvector. `unit_sample` is `sample_n`
/// unit-normalised rows of `dim`; each codebook holds `k` centroids of
/// `dim / m` dims. Subvectors train in parallel.
#[allow(clippy::cast_possible_truncation)]
fn train_pq_codebook(
    unit_sample: &[f32],
    sample_n: usize,
    dim: usize,
    m: usize,
    k: usize,
) -> Vec<Vec<f32>> {
    let sub_dim = dim / m;
    (0..m)
        .into_par_iter()
        .map(|s| {
            let mut sub = vec![0.0f32; sample_n * sub_dim];
            for i in 0..sample_n {
                sub[i * sub_dim..(i + 1) * sub_dim].copy_from_slice(
                    &unit_sample[i * dim + s * sub_dim..i * dim + (s + 1) * sub_dim],
                );
            }
            kmeans(&sub, sample_n, sub_dim, k, 12, 0x9E37_79B9 ^ s as u64)
        })
        .collect()
}

/// PQ-encode one unit-normalised row into `out` (`out.len()` centroid ids).
#[allow(clippy::cast_possible_truncation)]
fn pq_encode_row(
    codebook: &[Vec<f32>],
    unit_row: &[f32],
    k: usize,
    sub_dim: usize,
    out: &mut [u8],
) {
    for (s, slot) in out.iter_mut().enumerate() {
        let sub = &unit_row[s * sub_dim..(s + 1) * sub_dim];
        let cb = &codebook[s];
        let mut best = 0u8;
        let mut best_d = f32::MAX;
        for c in 0..k {
            let d = sq_l2(sub, &cb[c * sub_dim..(c + 1) * sub_dim]);
            if d < best_d {
                best_d = d;
                best = c as u8;
            }
        }
        *slot = best;
    }
}

/// PQ-encode a chunk of `raw` rows (row-major, `dim` each, any norm) into
/// `out` (`m` bytes per row). Rows are unit-normalised here so the ADC
/// squared-L2 ranks like cosine. Parallel over rows.
fn pq_encode_chunk(
    codebook: &[Vec<f32>],
    raw: &[f32],
    dim: usize,
    m: usize,
    k: usize,
    sub_dim: usize,
    out: &mut [u8],
) {
    out.par_chunks_mut(m)
        .zip(raw.par_chunks(dim))
        .for_each(|(o, r)| {
            let mut unit = vec![0.0f32; dim];
            normalize_into(r, &mut unit);
            pq_encode_row(codebook, &unit, k, sub_dim, o);
        });
}

/// Train a PQ codebook and quantise every row of an in-memory `f32_data`.
/// The codebook is k-means-trained on up to [`PQ_TRAIN_SAMPLE`] strided rows.
/// Compute the per-dim Lloyd-Max levels scaled to the Beta((d-1)/2,(d-1)/2)
/// variance `1/d` (Gaussian approximation for d >= 200, MSE penalty ~0.5%
/// vs exact Beta - see `turboquant.rs` for the derivation).
fn turboquant_levels(dim: usize, bits: u8) -> (Vec<f32>, Vec<f32>) {
    let scale = 1.0 / (dim as f32).sqrt();
    let centroids: Vec<f32> = lm_centroids_n01(bits).iter().map(|x| x * scale).collect();
    let boundaries: Vec<f32> = lm_boundaries_n01(bits).iter().map(|x| x * scale).collect();
    (centroids, boundaries)
}

/// Anisotropy compensation for 1-bit TurboQuant: per-coordinate `(shift,
/// inv_scale)` learned from a bounded sample of rotated coords. Empty vecs =
/// identity (the legacy path).
///
/// MEASURED NULL RESULT (2026-07-04, `SKEG_TQ1_MODE=asym`, mxbai-100k /
/// qwen3-20k): full shift+scale collapses recall (qwen3 0.966->0.022 - the
/// shift term drives the per-vector renorm `inner` toward zero on mean-heavy
/// data). Scale-only is stable but flat/negative (mxbai 0.970->0.970, qwen3
/// 0.966->0.953). The article's +6-9pp assume an 8-bit *quantized* query;
/// skeg's asym path already uses an f32 query + `norm/inner` renorm, so the
/// headroom compensation would fill is already filled. Kept flag-gated
/// (`SKEG_TQ1_ANISO`, `SKEG_TQ1_SHIFT`) for reproducibility; do not enable
/// expecting a win on the f32-asym path. `inv_scale[d] = 1/scale[d] = sqrt(dim)*sigma_d`;
/// stored as the reciprocal so the query hot-path multiplies. Applied ONLY on
/// the encode self-inner and the asymmetric query (never the popcount walk -
/// per-coord weights break the uniform-weight Hamming). See the article's
/// "Trick 3": on N(0,1/dim) data it collapses to identity, adding no noise.
#[derive(Debug, Default, Clone)]
pub struct Tq1Aniso {
    shift: Vec<f32>,
    inv_scale: Vec<f32>,
    /// Pre-rotation mean (unit-vector space). Non-empty => subtract from the
    /// unit vector BEFORE rotation on both encode and query (Pleshkov-style
    /// centering on the RAW distribution, where the mean is meaningful - unlike
    /// the post-rotation shift). Decoupled from shift/inv_scale.
    center: Vec<f32>,
}

impl Tq1Aniso {
    #[must_use]
    fn is_identity(&self) -> bool {
        self.inv_scale.is_empty()
    }
    #[must_use]
    fn has_center(&self) -> bool {
        !self.center.is_empty()
    }
    fn bytes(&self) -> usize {
        (self.shift.len() + self.inv_scale.len() + self.center.len()) * 4
    }
}

/// `sqrt(2/pi)` - the standardized 1-bit Lloyd-Max level; `Phi(+/-C_STD)` gives
/// the codebook-edge quantile probabilities used for calibration.
const TQ1_C_STD: f32 = 0.797_884_6;
const TQ1_P_LO: f32 = 0.212_47; // Phi(-C_STD)
const TQ1_P_HI: f32 = 0.787_53; // Phi(+C_STD)
const TQ1_CALIB_SAMPLE: usize = 8192;
const TQ1_CALIB_MIN: usize = 256; // below -> identity (too few to calibrate)
const TQ1_INV_MIN: f32 = 0.125;
const TQ1_INV_MAX: f32 = 8.0;

/// Interpolated order statistic of a pre-sorted slice.
fn quantile_sorted(sorted: &[f32], p: f32) -> f32 {
    let n = sorted.len();
    let h = (n as f32 - 1.0) * p;
    let i = h.floor() as usize;
    let frac = h - h.floor();
    if i + 1 >= n {
        sorted[n - 1]
    } else {
        sorted[i] + (sorted[i + 1] - sorted[i]) * frac
    }
}

/// Calibrate `(shift, inv_scale)` from `sample` rows of ROTATED coords (each
/// `dim`). Codebook-edge quantiles per coordinate; collapses to identity on
/// isotropic N(0,1/dim). Returns identity `Tq1Aniso` if the sample is too small.
fn calibrate_tq1(sample: &[f32], n_rows: usize, dim: usize) -> Tq1Aniso {
    if n_rows < TQ1_CALIB_MIN {
        return Tq1Aniso::default();
    }
    let c_outer = TQ1_C_STD / (dim as f32).sqrt();
    let mut shift = vec![0.0f32; dim];
    let mut inv_scale = vec![1.0f32; dim];
    let mut col = vec![0.0f32; n_rows];
    for d in 0..dim {
        for (r, c) in col.iter_mut().enumerate() {
            *c = sample[r * dim + d];
        }
        col.sort_unstable_by(f32::total_cmp);
        let q_lo = quantile_sorted(&col, TQ1_P_LO);
        let q_hi = quantile_sorted(&col, TQ1_P_HI);
        let range = q_hi - q_lo;
        if range < 1e-6 {
            inv_scale[d] = 0.0; // dead coord: drop from E_hat and qm
            continue;
        }
        inv_scale[d] = (range / (2.0 * c_outer)).clamp(TQ1_INV_MIN, TQ1_INV_MAX);
        // Shift (mean removal) is opt-in: folding it into the per-vector renorm
        // `scale = norm/inner` can drive `inner` toward zero on mean-heavy /
        // extreme-anisotropy data (qwen3), which blows up the scale and wrecks
        // recall. Scale-only compensation keeps `inner` strictly positive and is
        // the stable default; SKEG_TQ1_SHIFT=on re-enables the shift term.
        if tq1_shift_enabled() {
            shift[d] = -0.5 * (q_lo + q_hi);
        }
    }
    Tq1Aniso {
        shift,
        inv_scale,
        center: Vec::new(),
    }
}

/// Opt-in for the anisotropy *shift* term (mean removal). Off by default -
/// scale-only compensation is numerically stable; shift can destabilize the
/// per-vector renorm on mean-heavy data.
fn tq1_shift_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("SKEG_TQ1_SHIFT").ok().as_deref(),
            Some("on" | "1" | "true")
        )
    })
}

/// Opt-in for pre-rotation mean centering (subtract corpus mean in unit space
/// before rotation, both sides). Off by default. First-pass ranking experiment.
fn tq1_center_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("SKEG_TQ1_CENTER").ok().as_deref(),
            Some("on" | "1" | "true")
        )
    })
}

/// Read the SKEG_TQ1_ANISO opt-in once. Off by default.
fn tq1_aniso_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("SKEG_TQ1_ANISO").ok().as_deref(),
            Some("on" | "1" | "true")
        )
    })
}

/// Bit-plane inner-product primitive: returns `(sum_p 2^p*popcount(plane_p AND
/// code), popcount(code))`. `b+1` popcount passes over `bytes`-byte masks.
fn tq1_bitplane_score(planes: &[u8], b: u8, bytes: usize, code: &[u8]) -> (u64, u32) {
    // NEON vcntq_u8/vandq_u8 kernel on aarch64, scalar elsewhere.
    skeg_simd::tq1_bitplane_score(planes, b, bytes, code)
}

/// Map a coordinate to its `2^bits` bucket index using sorted boundaries.
/// For 1-bit (`boundaries.is_empty()`) the threshold is exactly zero.
fn turboquant_bucket(x: f32, boundaries: &[f32]) -> usize {
    if boundaries.is_empty() {
        return usize::from(x > 0.0);
    }
    let mut bucket = 0usize;
    for &b in boundaries {
        if x > b {
            bucket += 1;
        }
    }
    bucket
}

/// Pack a single bucket index into the bit-packed code buffer at coord `i`.
/// Bit packing density: `bits` codes per 8/bits bytes, with `i mod (8/bits)`
/// choosing the slot within a byte.
fn turboquant_pack(code: &mut [u8], i: usize, bucket: usize, bits: u8) {
    let bits = bits as usize;
    let codes_per_byte = 8 / bits;
    let byte = i / codes_per_byte;
    let shift = (i % codes_per_byte) * bits;
    code[byte] |= (bucket as u8) << shift;
}

/// Encode one f32 vector into a TurboQuant `(code, scale)` pair.
fn turboquant_encode_vec(
    v: &[f32],
    dim: usize,
    bits: u8,
    code_bytes: usize,
    rotation: &FastRotation,
    centroids: &[f32],
    boundaries: &[f32],
    aniso: &Tq1Aniso,
) -> (Vec<u8>, f32) {
    // Raw-space centering (Pleshkov): subtract the corpus mean mu from the RAW
    // vector, THEN normalize - removes the dominant embedding-mean direction
    // before it hits the sphere. Bits then land on the discriminative residual.
    // Without centering this is the plain unit-normalize.
    let mut unit = vec![0.0f32; dim];
    if aniso.has_center() {
        for ((u, &x), &m) in unit.iter_mut().zip(v.iter()).zip(aniso.center.iter()) {
            *u = x - m;
        }
    } else {
        unit.copy_from_slice(v);
    }
    let norm = unit.iter().map(|x| x * x).sum::<f32>().sqrt();
    let inv = if norm > 1e-10 { 1.0 / norm } else { 0.0 };
    for u in &mut unit {
        *u *= inv;
    }
    // Low-dim expansion: zero-pad the unit vector up to the rotation's working
    // dim (zeros don't change the norm) before rotating. No-op when equal.
    let rotated = if rotation.dim() == dim {
        rotation.apply_alloc(&unit)
    } else {
        let mut padded = vec![0.0f32; rotation.dim()];
        padded[..dim].copy_from_slice(&unit);
        rotation.apply_alloc(&padded)
    };
    let mut code = vec![0u8; code_bytes];
    let mut inner = 0.0f32;
    // 1-bit + compensation: the stored bit stays sign(r). The per-vector scale
    // uses the SCALE-only reconstruction r_est = c_outer*sign(r)*inv_scale, i.e.
    // inner = c_outer*sum(|r|*inv_scale). The shift (mean removal) is a
    // QUERY-side scalar (qm) only, decoupled from the renorm - folding it into
    // `inner` drives it toward zero on mean-heavy data (Pleshkov's method keeps
    // the stored scale as the plain RaBitQ renorm).
    let comp = bits == 1 && !aniso.is_identity();
    for (i, &r) in rotated.iter().enumerate() {
        let bucket = turboquant_bucket(r, boundaries);
        if comp {
            inner += r.abs() * centroids[1] * aniso.inv_scale[i];
        } else {
            inner += r * centroids[bucket];
        }
        turboquant_pack(&mut code, i, bucket, bits);
    }
    let inner = inner.max(1e-10);
    let scale = norm / inner;
    (code, scale)
}

fn build_turboquant(f32_data: &[f32], dim: usize, bits: u8) -> QuantRepr {
    assert!(
        matches!(bits, 1 | 2 | 4),
        "TurboQuant bits must be in {{1, 2, 4}}"
    );
    let codes_per_byte: usize = 8 / (bits as usize);
    assert_eq!(
        dim % codes_per_byte,
        0,
        "dim must be divisible by {} for {}-bit TurboQuant packing",
        codes_per_byte,
        bits
    );
    let n = f32_data.len() / dim;
    // Low-dim 1-bit expands the working dim: rotation, codes, and Lloyd-Max
    // levels all operate at `rot_dim`; only the input rows stay native `dim`
    // (padded per-vector in encode). rot_dim == dim for the normal path.
    let rot_dim = tq_code_dim(dim, bits);
    let code_bytes = rot_dim * (bits as usize) / 8;
    let rotation = Box::new(FastRotation::new(rot_dim, TQ_ROTATION_SEED));
    let (centroids, boundaries) = turboquant_levels(rot_dim, bits);
    let (centroids_i8, i8_scale) = quantise_tq_centroids(&centroids, bits);
    let mut aniso = if bits == 1 && tq1_aniso_enabled() && tq_code_dim(dim, bits) == dim {
        let s = n.min(TQ1_CALIB_SAMPLE);
        let step = (n / s.max(1)).max(1);
        let mut sample = vec![0.0f32; s * dim];
        let mut unit = vec![0.0f32; dim];
        for j in 0..s {
            normalize_into(
                &f32_data[(j * step) * dim..(j * step) * dim + dim],
                &mut unit,
            );
            let rot = rotation.apply_alloc(&unit);
            sample[j * dim..(j + 1) * dim].copy_from_slice(&rot);
        }
        calibrate_tq1(&sample, s, dim)
    } else {
        Tq1Aniso::default()
    };
    // Raw-space centering: mu = mean of the corpus RAW vectors (sampled).
    if bits == 1 && n >= TQ1_CALIB_MIN && tq1_center_enabled() {
        let s = n.min(TQ1_CALIB_SAMPLE);
        let step = (n / s.max(1)).max(1);
        let mut mean = vec![0.0f32; dim];
        for j in 0..s {
            let row = &f32_data[(j * step) * dim..(j * step) * dim + dim];
            for (m, &x) in mean.iter_mut().zip(row.iter()) {
                *m += x;
            }
        }
        for m in &mut mean {
            *m /= s as f32;
        }
        aniso.center = mean;
    }

    let encoded: Vec<(Vec<u8>, f32)> = (0..n)
        .into_par_iter()
        .map(|i| {
            turboquant_encode_vec(
                &f32_data[i * dim..(i + 1) * dim],
                dim,
                bits,
                code_bytes,
                &rotation,
                &centroids,
                &boundaries,
                &aniso,
            )
        })
        .collect();
    let mut codes_buf = Vec::with_capacity(n * code_bytes);
    let mut scales = Vec::with_capacity(n);
    for (c, s) in encoded {
        codes_buf.extend_from_slice(&c);
        scales.push(s);
    }
    QuantRepr::TurboQuant {
        codes: CodeBacking::Owned(codes_buf),
        scales,
        rotation,
        centroids,
        boundaries,
        centroids_i8,
        i8_scale,
        bits,
        code_bytes,
        aniso,
    }
}

/// Quantise the `2^bits` Lloyd-Max centroids into the 16-entry i8 LUT layout
/// the NEON `vqtbl1q_s8` kernel expects. For `bits == 4` all 16 slots hold
/// real centroids; for `bits == 2` slots 0..3 hold centroids and 4..15 are
/// zero-padded (codes are 2-bit so only indices 0..3 are exercised);
/// `bits == 1` returns the all-zero LUT since the 1-bit ADC uses the
/// algebraic `c * (2*masked - q_sum)` reduction, not vtbl.
fn quantise_tq_centroids(centroids: &[f32], bits: u8) -> ([i8; 16], f32) {
    match bits {
        4 => {
            let mut arr = [0.0f32; 16];
            for (a, &c) in arr.iter_mut().zip(centroids.iter()) {
                *a = c;
            }
            quantise_centroids_i8(&arr)
        }
        2 => {
            // Quantise the 4 real centroids together so the i8 scale
            // matches the centroid magnitudes (avoid wasted resolution
            // from including 12 zero entries in the max-abs calibration).
            let mut arr = [0.0f32; 4];
            for (a, &c) in arr.iter_mut().zip(centroids.iter()) {
                *a = c;
            }
            let (q4, scale) = quantise_centroids_i8(&arr);
            let mut out = [0i8; 16];
            out[..4].copy_from_slice(&q4);
            (out, scale)
        }
        _ => ([0i8; 16], 0.0),
    }
}

fn build_pq(f32_data: &[f32], dim: usize, m: usize, k: usize) -> QuantRepr {
    assert!(
        m > 0 && k > 0 && k <= 256,
        "PQ needs 0 < m and 0 < k <= 256"
    );
    assert_eq!(dim % m, 0, "dim must be divisible by m");
    let sub_dim = dim / m;
    let n = f32_data.len() / dim;
    let sample_n = n.clamp(1, PQ_TRAIN_SAMPLE);
    let step = (n / sample_n).max(1);
    let mut sample = vec![0.0f32; sample_n * dim];
    for (t, row) in (0..sample_n).zip((0..n).step_by(step)) {
        normalize_into(
            &f32_data[row * dim..(row + 1) * dim],
            &mut sample[t * dim..(t + 1) * dim],
        );
    }
    let codebook = train_pq_codebook(&sample, sample_n, dim, m, k);
    let mut codes = vec![0u8; n * m];
    pq_encode_chunk(&codebook, f32_data, dim, m, k, sub_dim, &mut codes);
    QuantRepr::Pq {
        codes,
        codebook,
        m,
        k,
        sub_dim,
    }
}

impl QuantizedVectors {
    /// Build a quantized set from `n` row-major f32 vectors of `dim` each.
    ///
    /// # Panics
    ///
    /// Panics if `dim == 0`, if `f32_data.len()` is not a multiple of `dim`,
    /// or if `kind` is [`QuantKind::F32`] (which needs no quantized form).
    #[must_use]
    pub fn build(f32_data: &[f32], dim: usize, kind: QuantKind) -> QuantizedVectors {
        assert!(dim > 0, "dim must be positive");
        assert_eq!(f32_data.len() % dim, 0, "f32_data is not a multiple of dim");
        let n = f32_data.len() / dim;
        let repr = match kind {
            QuantKind::Int8 => {
                let scale = calibrate_int8(f32_data);
                let data = f32_data.iter().map(|&x| quantize_i8(x, scale)).collect();
                QuantRepr::Int8 { data, scale }
            }
            QuantKind::Binary => {
                let bytes = dim.div_ceil(8);
                let mut data = vec![0u8; n * bytes];
                for (row, chunk) in data.chunks_exact_mut(bytes).enumerate() {
                    pack_signs(&f32_data[row * dim..(row + 1) * dim], chunk);
                }
                QuantRepr::Binary { data, bytes }
            }
            QuantKind::Pq { m, k } => build_pq(f32_data, dim, m, k),
            QuantKind::TurboQuant { bits } => build_turboquant(f32_data, dim, bits),
            QuantKind::F32 => panic!("QuantizedVectors::build does not accept QuantKind::F32"),
        };
        QuantizedVectors { dim, n, repr }
    }

    /// Build an int8 set from a streamed f32 source, without ever holding the
    /// whole f32 set in RAM at once.
    ///
    /// `for_each_row` walks the `n` rows (`dim` f32 each), invoking the given
    /// callback once per row. It is called **twice**: first to calibrate the
    /// int8 scale, then to quantize. Both walks must yield identical rows in
    /// the same order. The disk index `open` path uses this so a large
    /// `vectors.bin` becomes the int8 tier with peak RAM of one read chunk
    /// plus the tier, never a transient the size of the f32 set.
    ///
    /// # Panics
    ///
    /// Panics if `dim == 0`.
    ///
    /// # Errors
    ///
    /// Propagates any error returned by `for_each_row`.
    pub fn build_int8_streaming(
        n: usize,
        dim: usize,
        mut for_each_row: impl FnMut(&mut dyn FnMut(&[f32])) -> std::io::Result<()>,
    ) -> std::io::Result<QuantizedVectors> {
        assert!(dim > 0, "dim must be positive");
        // Pass 1: calibrate the symmetric scale from the running max magnitude.
        let mut max_abs = 0.0f32;
        for_each_row(&mut |row| {
            for &x in row {
                max_abs = max_abs.max(x.abs());
            }
        })?;
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        // Pass 2: quantize each row into the packed i8 buffer.
        let mut data: Vec<i8> = Vec::with_capacity(n * dim);
        for_each_row(&mut |row| {
            data.extend(row.iter().map(|&x| quantize_i8(x, scale)));
        })?;
        debug_assert_eq!(data.len(), n * dim, "streamed row count disagrees with n");
        Ok(QuantizedVectors {
            dim,
            n,
            repr: QuantRepr::Int8 { data, scale },
        })
    }

    /// Build a PQ set from a streamed f32 source, without holding the whole
    /// f32 set in RAM. `for_each_row` walks the `n` rows; it is called
    /// **twice**: pass 1 collects a strided training sample and trains the
    /// codebook, pass 2 quantises every row (chunked, parallel). Peak RAM is
    /// the training sample plus one quantisation chunk plus the codes - never
    /// a transient the size of the f32 set. The disk index `open` path uses
    /// this so a large `vectors.bin` becomes a PQ tier.
    ///
    /// # Panics
    ///
    /// Panics if `dim == 0`, `dim % m != 0`, or `k` is not in `1..=256`.
    ///
    /// # Errors
    ///
    /// Propagates any error returned by `for_each_row`.
    pub fn build_pq_streaming(
        n: usize,
        dim: usize,
        m: usize,
        k: usize,
        mut for_each_row: impl FnMut(&mut dyn FnMut(&[f32])) -> std::io::Result<()>,
    ) -> std::io::Result<QuantizedVectors> {
        assert!(dim > 0 && m > 0 && (1..=256).contains(&k), "bad PQ params");
        assert_eq!(dim % m, 0, "dim must be divisible by m");
        let sub_dim = dim / m;
        if n == 0 {
            return Ok(QuantizedVectors {
                dim,
                n: 0,
                repr: QuantRepr::Pq {
                    codes: Vec::new(),
                    codebook: Vec::new(),
                    m,
                    k,
                    sub_dim,
                },
            });
        }
        // Pass 1: strided, unit-normalised training sample.
        let sample_n = n.min(PQ_TRAIN_SAMPLE);
        let step = (n / sample_n).max(1);
        let mut sample: Vec<f32> = Vec::with_capacity(sample_n * dim);
        let mut seen = 0usize;
        for_each_row(&mut |row| {
            if seen % step == 0 && sample.len() < sample_n * dim {
                let mut unit = vec![0.0f32; dim];
                normalize_into(row, &mut unit);
                sample.extend_from_slice(&unit);
            }
            seen += 1;
        })?;
        let got = sample.len() / dim;
        let codebook = train_pq_codebook(&sample, got, dim, m, k);
        // Pass 2: quantise every row, chunked so peak RAM stays bounded.
        let mut codes = vec![0u8; n * m];
        let chunk_rows = 4096.min(n);
        let mut buf: Vec<f32> = Vec::with_capacity(chunk_rows * dim);
        let mut written = 0usize;
        for_each_row(&mut |row| {
            buf.extend_from_slice(row);
            if buf.len() >= chunk_rows * dim {
                let rows = buf.len() / dim;
                pq_encode_chunk(
                    &codebook,
                    &buf,
                    dim,
                    m,
                    k,
                    sub_dim,
                    &mut codes[written * m..(written + rows) * m],
                );
                written += rows;
                buf.clear();
            }
        })?;
        if !buf.is_empty() {
            let rows = buf.len() / dim;
            pq_encode_chunk(
                &codebook,
                &buf,
                dim,
                m,
                k,
                sub_dim,
                &mut codes[written * m..(written + rows) * m],
            );
            written += rows;
        }
        debug_assert_eq!(written, n, "streamed row count disagrees with n");
        Ok(QuantizedVectors {
            dim,
            n,
            repr: QuantRepr::Pq {
                codes,
                codebook,
                m,
                k,
                sub_dim,
            },
        })
    }

    /// Build a TurboQuant set from a streamed f32 source. Same streaming
    /// contract as [`build_pq_streaming`](Self::build_pq_streaming): the
    /// callback is invoked once per row, the whole f32 set never lives in
    /// RAM at once. `bits` in `{1, 2, 4}` selects the precision.
    ///
    /// Unlike PQ, TurboQuant is *data-oblivious*: there is no training pass.
    /// The rotation matrix and Lloyd-Max levels are computed once from
    /// `(dim, bits, seed)` (single pass over rows for encoding).
    ///
    /// # Panics
    ///
    /// Panics if `dim == 0`, `bits` is not in `{1, 2, 4}`, or `dim` is not
    /// divisible by `8 / bits`.
    ///
    /// # Errors
    ///
    /// Propagates any error returned by `for_each_row`.
    pub fn build_turboquant_streaming(
        n: usize,
        dim: usize,
        bits: u8,
        mut for_each_row: impl FnMut(&mut dyn FnMut(&[f32])) -> std::io::Result<()>,
    ) -> std::io::Result<QuantizedVectors> {
        assert!(dim > 0, "dim must be positive");
        assert!(
            matches!(bits, 1 | 2 | 4),
            "TurboQuant bits must be in {{1, 2, 4}}"
        );
        let codes_per_byte: usize = 8 / (bits as usize);
        assert_eq!(
            dim % codes_per_byte,
            0,
            "dim must be divisible by {} for {}-bit TurboQuant packing",
            codes_per_byte,
            bits
        );
        // Low-dim 1-bit expands the working dim (see `build_turboquant`).
        let rot_dim = tq_code_dim(dim, bits);
        let code_bytes = rot_dim * (bits as usize) / 8;
        let rotation = Box::new(FastRotation::new(rot_dim, TQ_ROTATION_SEED));
        let (centroids, boundaries) = turboquant_levels(rot_dim, bits);
        let (centroids_i8, i8_scale) = quantise_tq_centroids(&centroids, bits);
        if n == 0 {
            return Ok(QuantizedVectors {
                dim,
                n: 0,
                repr: QuantRepr::TurboQuant {
                    codes: CodeBacking::Owned(Vec::new()),
                    scales: Vec::new(),
                    rotation,
                    centroids,
                    boundaries,
                    centroids_i8,
                    i8_scale,
                    bits,
                    code_bytes,
                    aniso: Tq1Aniso::default(),
                },
            });
        }
        let mut codes_buf: Vec<u8> = Vec::with_capacity(n * code_bytes);
        let mut scales: Vec<f32> = Vec::with_capacity(n);
        let enc = |row: &[f32], aniso: &Tq1Aniso, codes: &mut Vec<u8>, scales: &mut Vec<f32>| {
            let (c, s) = turboquant_encode_vec(
                row,
                dim,
                bits,
                code_bytes,
                &rotation,
                &centroids,
                &boundaries,
                aniso,
            );
            codes.extend_from_slice(&c);
            scales.push(s);
        };
        let aniso = if bits == 1 && tq1_aniso_enabled() && tq_code_dim(dim, bits) == dim {
            // Buffered single pass: collect rows until the calibration sample is
            // full, calibrate, encode the buffered rows, then encode the rest on
            // the fly. Bounded extra RAM = TQ1_CALIB_SAMPLE * dim f32.
            let sample_rows = n.min(TQ1_CALIB_SAMPLE);
            let mut buf: Vec<f32> = Vec::with_capacity(sample_rows * dim);
            let mut sample: Vec<f32> = Vec::with_capacity(sample_rows * dim);
            let mut unit = vec![0.0f32; dim];
            let mut calib: Option<Tq1Aniso> = None;
            for_each_row(&mut |row| {
                if let Some(a) = &calib {
                    enc(row, a, &mut codes_buf, &mut scales);
                } else {
                    buf.extend_from_slice(row);
                    normalize_into(row, &mut unit);
                    sample.extend_from_slice(&rotation.apply_alloc(&unit));
                    if buf.len() / dim >= sample_rows {
                        let a = calibrate_tq1(&sample, buf.len() / dim, dim);
                        for j in 0..buf.len() / dim {
                            enc(
                                &buf[j * dim..(j + 1) * dim],
                                &a,
                                &mut codes_buf,
                                &mut scales,
                            );
                        }
                        buf.clear();
                        calib = Some(a);
                    }
                }
            })?;
            // Stream ended before the sample filled (n <= sample never triggers
            // the mid-loop flush): calibrate and encode whatever was buffered.
            if calib.is_none() {
                let rows = buf.len() / dim;
                let a = calibrate_tq1(&sample, rows, dim);
                for j in 0..rows {
                    enc(
                        &buf[j * dim..(j + 1) * dim],
                        &a,
                        &mut codes_buf,
                        &mut scales,
                    );
                }
                calib = Some(a);
            }
            calib.unwrap_or_default()
        } else {
            // No cross-row dependency (tq2/tq4, or tq1 on an expanded dim): each
            // row's rotate+quantise is independent, and the rotation (FHT) is the
            // dominant cost. Buffer rows into a chunk and encode the chunk across
            // all cores; output stays in row order. Bounded extra RAM = one chunk
            // of f32 rows. This is the serve-open hot path at 500k+.
            const TQ_PAR_CHUNK_ROWS: usize = 8192;
            let identity = Tq1Aniso::default();
            let mut buf: Vec<f32> = Vec::with_capacity(TQ_PAR_CHUNK_ROWS * dim);
            let flush = |buf: &[f32], codes_buf: &mut Vec<u8>, scales: &mut Vec<f32>| {
                let rows = buf.len() / dim;
                let out: Vec<(Vec<u8>, f32)> = (0..rows)
                    .into_par_iter()
                    .map(|j| {
                        turboquant_encode_vec(
                            &buf[j * dim..(j + 1) * dim],
                            dim,
                            bits,
                            code_bytes,
                            &rotation,
                            &centroids,
                            &boundaries,
                            &identity,
                        )
                    })
                    .collect();
                for (c, s) in out {
                    codes_buf.extend_from_slice(&c);
                    scales.push(s);
                }
            };
            for_each_row(&mut |row| {
                buf.extend_from_slice(row);
                if buf.len() >= TQ_PAR_CHUNK_ROWS * dim {
                    flush(&buf, &mut codes_buf, &mut scales);
                    buf.clear();
                }
            })?;
            if !buf.is_empty() {
                flush(&buf, &mut codes_buf, &mut scales);
            }
            identity
        };
        debug_assert_eq!(scales.len(), n, "streamed row count disagrees with n");
        Ok(QuantizedVectors {
            dim,
            n,
            repr: QuantRepr::TurboQuant {
                codes: CodeBacking::Owned(codes_buf),
                scales,
                rotation,
                centroids,
                boundaries,
                centroids_i8,
                i8_scale,
                bits,
                code_bytes,
                aniso,
            },
        })
    }

    /// Number of vectors in the set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// True if the set holds no vectors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Bytes of quantized vector data held in RAM.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        match &self.repr {
            QuantRepr::Int8 { data, .. } => data.len(),
            QuantRepr::Binary { data, .. } => data.len(),
            QuantRepr::Pq {
                codes, codebook, ..
            } => codes.len() + codebook.iter().map(|c| c.len() * 4).sum::<usize>(),
            QuantRepr::TurboQuant {
                codes,
                scales,
                centroids,
                boundaries,
                aniso,
                ..
            } => {
                // FastRotation stores only three dim-bit sign masks and a
                // scale, so the rotation footprint is ~3 * dim / 8 bytes
                // (negligible). Codes + scales + per-bits LUTs dominate.
                let rot_bytes = 3 * self.dim.div_ceil(8);
                codes.len()
                    + scales.len() * 4
                    + centroids.len() * 4
                    + boundaries.len() * 4
                    + rot_bytes
                    + aniso.bytes()
            }
        }
    }

    /// Persist the TurboQuant `codes` buffer to `path` and swap the in-RAM
    /// `Vec<u8>` for a memory-mapped view of the file. The OS page cache
    /// then decides which pages stay resident under memory pressure
    /// instead of swapping anonymous memory.
    ///
    /// No-op (returns `Ok(())`) for non-TurboQuant tiers and for an empty
    /// TurboQuant set. Skips the write if the on-disk file is already
    /// present at the expected size: the rotation seed is fixed so the
    /// codes are deterministic, the file from a previous open is reusable.
    ///
    /// # Errors
    ///
    /// Propagates any I/O error from writing the file or mapping it.
    pub fn swap_turboquant_codes_to_mmap(&mut self, path: &std::path::Path) -> std::io::Result<()> {
        let QuantRepr::TurboQuant { codes, .. } = &mut self.repr else {
            return Ok(());
        };
        let len = codes.len();
        if len == 0 {
            return Ok(());
        }
        // Write the codes to disk (or trust an existing file of the right
        // size - the rotation seed is fixed, so the byte sequence is
        // deterministic from the parent index alone).
        let needs_write = match std::fs::metadata(path) {
            Ok(meta) => meta.len() as usize != len,
            Err(_) => true,
        };
        if needs_write {
            // Drop the buffer slice after the write so peak RAM does not
            // double during the swap.
            let bytes = codes.as_slice();
            let tmp = path.with_extension("cache.bin.tmp");
            std::fs::write(&tmp, bytes)?;
            std::fs::rename(&tmp, path)?;
        }
        let mapped = MappedFile::open(path)?;
        // Guard against a truncated or substituted cache.bin: if the file
        // changed size between the metadata check and the mmap open (rare
        // - same process holds both ends), the proxy would slice past the
        // buffer on a high-row lookup. Surface a clean error here rather
        // than panic from a bounds-check downstream.
        if mapped.len() != len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "tier cache size mismatch: file={} bytes, expected={} bytes",
                    mapped.len(),
                    len
                ),
            ));
        }
        // Greedy walk hops random nodes; tell the kernel to skip readahead
        // so it doesn't speculate on pages we'll never touch. Log if the
        // call fails (sandbox, unusual fs) - it's a hint, not load-bearing.
        if let Err(e) = mapped.advise_random() {
            tracing::debug!("tier mmap MADV_RANDOM failed: {e}");
        }
        *codes = CodeBacking::Mapped(mapped);
        Ok(())
    }

    /// True when the underlying representation is `TurboQuant { bits = 4 }`
    /// and therefore eligible for the block-32 SIMD scoring path.
    #[must_use]
    pub fn supports_tq4_block(&self) -> bool {
        matches!(self.repr, QuantRepr::TurboQuant { bits: 4, .. })
    }

    /// f32 Lloyd-Max centroids for the TurboQuant 4-bit codebook, if
    /// the representation matches. Used by the block kernel's LUT
    /// pre-compute step.
    #[must_use]
    pub fn tq4_centroids(&self) -> Option<&[f32]> {
        match &self.repr {
            QuantRepr::TurboQuant {
                bits: 4, centroids, ..
            } => Some(centroids.as_slice()),
            _ => None,
        }
    }

    /// Per-vector scales for the TurboQuant codebook. Used by the
    /// block kernel after the inner-product proxy is reconstructed.
    #[must_use]
    pub fn tq4_scales(&self) -> Option<&[f32]> {
        match &self.repr {
            QuantRepr::TurboQuant {
                bits: 4, scales, ..
            } => Some(scales.as_slice()),
            _ => None,
        }
    }

    /// 1-bit reconstruction quality g = inner/v = 1/scale for a row (RaBitQ
    /// per-vector confidence; high g = code aligns well, tight error bound).
    #[must_use]
    pub fn tq1_recon_g(&self, row: usize) -> Option<f32> {
        match &self.repr {
            QuantRepr::TurboQuant {
                bits: 1, scales, ..
            } => scales
                .get(row)
                .map(|s| if *s > 0.0 { 1.0 / *s } else { 0.0 }),
            _ => None,
        }
    }

    /// Row-major 4-bit codes (dim/2 bytes per vector). Caller is
    /// responsible for interleaving them into the block layout via
    /// `skeg_simd::interleave_tq4_codes` before scoring.
    #[must_use]
    pub fn tq4_codes(&self) -> Option<&[u8]> {
        match &self.repr {
            QuantRepr::TurboQuant { bits: 4, codes, .. } => Some(codes.as_slice()),
            _ => None,
        }
    }

    /// Apply the TurboQuant rotation to a unit-normalised query and
    /// return the rotated f32 vector. Required by the block kernel
    /// to build the per-query LUT.
    #[must_use]
    pub fn tq4_rotate_query(&self, unit_query: &[f32]) -> Option<Vec<f32>> {
        match &self.repr {
            QuantRepr::TurboQuant {
                bits: 4, rotation, ..
            } => Some(rotation.apply_alloc(unit_query)),
            _ => None,
        }
    }

    pub fn int8_scale(&self) -> Option<f32> {
        match self.repr {
            QuantRepr::Int8 { scale, .. } => Some(scale),
            QuantRepr::Binary { .. } | QuantRepr::Pq { .. } | QuantRepr::TurboQuant { .. } => None,
        }
    }

    /// Quantize `query` into the same representation as this set.
    ///
    /// # Panics
    ///
    /// Panics if `query.len()` does not equal the set dimension.
    #[must_use]
    pub fn quantize_query(&self, query: &[f32]) -> QueryCode {
        self.quantize_query_with_mode(query, None)
    }

    /// Like [`quantize_query`](Self::quantize_query) but with an explicit tq1
    /// proxy mode (used by the online controller to force hybrid/asym/pop per
    /// query). `None` = the default `tq1_proxy_mode_for(dim, bits)`. Ignored for
    /// non-tq1 tiers.
    #[must_use]
    pub fn quantize_query_with_mode(
        &self,
        query: &[f32],
        mode_override: Option<Tq1ProxyMode>,
    ) -> QueryCode {
        assert_eq!(query.len(), self.dim, "query dim mismatch");
        match &self.repr {
            QuantRepr::Int8 { scale, .. } => {
                QueryCode::Int8(query.iter().map(|&x| quantize_i8(x, *scale)).collect())
            }
            QuantRepr::Binary { bytes, .. } => {
                let mut code = vec![0u8; *bytes];
                pack_signs(query, &mut code);
                QueryCode::Binary(code)
            }
            QuantRepr::Pq {
                codebook,
                m,
                k,
                sub_dim,
                ..
            } => {
                let mut unit = vec![0.0f32; self.dim];
                normalize_into(query, &mut unit);
                let mut lut = vec![0.0f32; m * k];
                for s in 0..*m {
                    let qsub = &unit[s * sub_dim..(s + 1) * sub_dim];
                    for c in 0..*k {
                        lut[s * k + c] = sq_l2(qsub, &codebook[s][c * sub_dim..(c + 1) * sub_dim]);
                    }
                }
                QueryCode::Pq(lut)
            }
            QuantRepr::TurboQuant {
                rotation,
                bits,
                aniso,
                ..
            } => {
                let mut unit = vec![0.0f32; self.dim];
                if aniso.has_center() {
                    // Raw-space centering: subtract mu from the raw query, THEN
                    // normalize (mirrors encode).
                    for ((u, &x), &m) in unit.iter_mut().zip(query.iter()).zip(aniso.center.iter())
                    {
                        *u = x - m;
                    }
                    let n = unit.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let inv = if n > 1e-10 { 1.0 / n } else { 0.0 };
                    for u in &mut unit {
                        *u *= inv;
                    }
                } else {
                    normalize_into(query, &mut unit);
                }
                // Low-dim expansion: zero-pad the unit query to the rotation's
                // working dim before rotating (mirrors encode). No-op when equal.
                let q_rot = if rotation.dim() == self.dim {
                    rotation.apply_alloc(&unit)
                } else {
                    let mut padded = vec![0.0f32; rotation.dim()];
                    padded[..self.dim].copy_from_slice(&unit);
                    rotation.apply_alloc(&padded)
                };
                // Anisotropy compensation folds onto the ASYMMETRIC query only:
                // q_plus[d] = q_rot[d]*inv_scale[d], qm = sum(q_rot*shift). The
                // popcount sign bits use the RAW q_rot (walk stays uncompensated).
                // Identity when compensation is off -> byte-identical legacy query.
                let compensate = |q_rot: &[f32]| -> (Vec<f32>, f32, f32) {
                    if aniso.is_identity() {
                        return (q_rot.to_vec(), q_rot.iter().sum(), 0.0);
                    }
                    let mut qp = vec![0.0f32; q_rot.len()];
                    let (mut s, mut qm) = (0.0f32, 0.0f32);
                    for d in 0..q_rot.len() {
                        qp[d] = q_rot[d] * aniso.inv_scale[d];
                        s += qp[d];
                        qm += q_rot[d] * aniso.shift[d];
                    }
                    (qp, s, qm)
                };
                let mode = mode_override.unwrap_or_else(|| tq1_proxy_mode_for(self.dim, *bits));
                match mode {
                    Tq1ProxyMode::Popcount => {
                        // Same sign-bit packing as the stored codes, so a
                        // Hamming popcount counts sign disagreements.
                        let mut q_bits = vec![0u8; rotation.dim().div_ceil(8)];
                        pack_signs(&q_rot, &mut q_bits);
                        QueryCode::TurboQuant1Popcount { q_bits }
                    }
                    Tq1ProxyMode::Hybrid => {
                        // Sign bits (raw) for the popcount walk; compensated
                        // q_plus/q_sum/qm for the asym re-score of survivors.
                        let mut q_bits = vec![0u8; rotation.dim().div_ceil(8)];
                        pack_signs(&q_rot, &mut q_bits);
                        let (q_plus, q_sum, qm) = compensate(&q_rot);
                        QueryCode::TurboQuant1Hybrid {
                            q_bits,
                            q_rot: q_plus,
                            q_sum,
                            qm,
                        }
                    }
                    Tq1ProxyMode::Asymmetric => {
                        let (q_plus, q_sum, qm) = compensate(&q_rot);
                        QueryCode::TurboQuant {
                            q_rot: q_plus,
                            q_sum,
                            qm,
                        }
                    }
                    Tq1ProxyMode::BitPlane => {
                        // Scalar-quantize the (optionally compensated) query to
                        // b bits and transpose into b sign-aligned bit-planes.
                        let (q_plus, _, qm) = compensate(&q_rot);
                        let b = tq1_bitplane_bits();
                        let bytes = rotation.dim().div_ceil(8);
                        let levels = ((1u32 << b) - 1) as f32;
                        let m = q_plus.iter().copied().fold(f32::INFINITY, f32::min);
                        let mx = q_plus.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let sq = (mx - m).max(1e-20) / levels;
                        let mut planes = vec![0u8; b as usize * bytes];
                        let mut sum_q = 0.0f32;
                        for (i, &q) in q_plus.iter().enumerate() {
                            let qi = (((q - m) / sq).round() as i32).clamp(0, levels as i32) as u32;
                            sum_q += qi as f32;
                            let (byte, bit) = (i / 8, i % 8);
                            for p in 0..b as usize {
                                if (qi >> p) & 1 == 1 {
                                    planes[p * bytes + byte] |= 1u8 << bit;
                                }
                            }
                        }
                        QueryCode::TurboQuant1BitPlane {
                            planes,
                            b,
                            bytes,
                            m,
                            sq,
                            sum_q,
                            qm,
                        }
                    }
                }
            }
        }
    }

    /// Proxy closeness of `row` to a quantized query: greater is closer.
    ///
    /// This ordering is only an approximation of f32 cosine; the caller is
    /// expected to re-rank survivors exactly.
    ///
    /// # Panics
    ///
    /// Panics if `row` is out of range or if `code` does not match the set's
    /// quantization.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn proxy(&self, row: usize, code: &QueryCode) -> i32 {
        assert!(row < self.n, "row out of range");
        match (&self.repr, code) {
            (QuantRepr::Int8 { data, .. }, QueryCode::Int8(q)) => {
                dot_int8(q, &data[row * self.dim..(row + 1) * self.dim])
            }
            (QuantRepr::Binary { data, bytes }, QueryCode::Binary(q)) => {
                let h = hamming_binary(q, &data[row * bytes..(row + 1) * bytes]);
                // Fewer differing sign bits = closer, so negate for "greater is
                // closer". Hamming <= dim, always within i32.
                -i32::try_from(h).expect("hamming distance fits i32")
            }
            (QuantRepr::Pq { codes, m, k, .. }, QueryCode::Pq(lut)) => {
                // ADC: sum the per-subvector squared-L2 from the query LUT.
                let row_code = &codes[row * m..(row + 1) * m];
                let mut adc = 0.0f32;
                for (s, &c) in row_code.iter().enumerate() {
                    adc += lut[s * k + c as usize];
                }
                // Smaller ADC = closer; negate and scale into the i32 contract.
                -(adc.min(40.0) * PQ_PROXY_SCALE) as i32
            }
            (
                QuantRepr::TurboQuant {
                    codes,
                    scales,
                    centroids,
                    centroids_i8,
                    i8_scale,
                    bits,
                    code_bytes,
                    ..
                },
                QueryCode::TurboQuant { q_rot, q_sum, qm },
            ) => {
                // Asymmetric inner product: for each coord, multiply the
                // rotated query coord by the Lloyd-Max centroid keyed by the
                // stored bucket id. Per-vector scale corrects the quantizer
                // shrinkage so the result tracks the true cosine.
                let code = &codes.as_slice()[row * code_bytes..(row + 1) * code_bytes];
                let acc = match *bits {
                    // 4-bit and 2-bit: NEON `vqtbl1q_s8` kernel - 16
                    // parallel centroid lookups + f32x4 FMA per chunk.
                    // Centroid i8 quantization introduces ~1.5% MSE, gated.
                    // q_rot.len() = working (rotation) dim: == self.dim except
                    // low-dim 1-bit expansion where the code is wider.
                    4 => tq4_adc_i8(code, centroids_i8, *i8_scale, q_rot, q_rot.len()),
                    2 => tq2_adc_i8(code, centroids_i8, *i8_scale, q_rot, q_rot.len()),
                    // 1-bit: algebraic reduction `c * (2*masked - q_sum)`.
                    // q_sum precomputed at query time; SWAR scalar inner.
                    1 => tq1_adc_swar(code, centroids, q_rot, q_rot.len(), *q_sum),
                    // `build_turboquant` asserts bits in {1, 2, 4}.
                    _ => unreachable!("TurboQuant bits must be in {{1, 2, 4}}"),
                };
                // `qm` is the anisotropy shift correction (0 without it, so the
                // legacy path is unchanged); E_hat = acc - qm.
                let ip = scales[row] * (acc - qm);
                // Greater inner product = closer. Clamp wide enough for
                // compensated queries (inv_scale up to 8 lifts |ip| past 1);
                // 32*1e7 << i32::MAX so no overflow.
                (ip.clamp(-32.0, 32.0) * TQ_PROXY_SCALE) as i32
            }
            (
                QuantRepr::TurboQuant {
                    codes, code_bytes, ..
                },
                QueryCode::TurboQuant1Popcount { q_bits },
            ) => {
                // Symmetric 1-bit proxy: the stored code is the vector's rotated
                // sign bits, q_bits the query's; Hamming counts sign
                // disagreements. Fewer = closer, so negate for "greater =
                // closer". Hamming <= dim, always fits i32. Scale is unused
                // (magnitude is exactly what the symmetric path drops).
                let code = &codes.as_slice()[row * code_bytes..(row + 1) * code_bytes];
                let h = hamming_binary(q_bits, code);
                -i32::try_from(h).expect("hamming distance fits i32")
            }
            (
                QuantRepr::TurboQuant {
                    codes,
                    scales,
                    centroids,
                    code_bytes,
                    bits: 1,
                    ..
                },
                QueryCode::TurboQuant1BitPlane {
                    planes,
                    b,
                    bytes,
                    m,
                    sq,
                    sum_q,
                    qm,
                },
            ) => {
                // Asymmetric integer ADC: <q, s> reconstructed from b bit-planes.
                // <q,s> = m*(2*pc(code)-dim) + sq*(2*weighted - sum_q); decoded
                // level is c_outer = centroids[1]. Matches tq1_adc_swar as b->inf.
                // qm = anisotropy shift correction (0 without it); E_hat = c*dot - qm.
                let code = &codes.as_slice()[row * code_bytes..(row + 1) * code_bytes];
                let (weighted, code_pc) = tq1_bitplane_score(planes, *b, *bytes, code);
                // code_bytes*8 = working (rotation) dim (wider than self.dim
                // under low-dim 1-bit expansion); code_pc counts over it.
                let dot = m * (2.0 * code_pc as f32 - (code_bytes * 8) as f32)
                    + sq * (2.0 * weighted as f32 - sum_q);
                let ip = scales[row] * (centroids[1] * dot - qm);
                (ip.clamp(-32.0, 32.0) * TQ_PROXY_SCALE) as i32
            }
            (
                QuantRepr::TurboQuant {
                    codes, code_bytes, ..
                },
                QueryCode::TurboQuant1Hybrid { q_bits, .. },
            ) => {
                // Hybrid walk: cheap popcount navigation. The asymmetric re-score
                // of the survivors happens in `proxy_rescore`, not here.
                let code = &codes.as_slice()[row * code_bytes..(row + 1) * code_bytes];
                let h = hamming_binary(q_bits, code);
                -i32::try_from(h).expect("hamming distance fits i32")
            }
            _ => panic!("query code does not match index quantization"),
        }
    }

    /// Re-score `row` for the final ordering that gates the exact rerank. Same as
    /// [`proxy`](Self::proxy) for every mode EXCEPT tq1 hybrid, where the walk
    /// used the cheap popcount proxy but the candidate list should be ranked by
    /// the more accurate asymmetric inner product (in-RAM, no disk) so the
    /// limited rerank budget lands on the best candidates. Callers can always
    /// route the post-walk candidate ordering through this uniformly.
    ///
    /// # Panics
    ///
    /// Panics if `row` is out of range or `code` does not match the set.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn proxy_rescore(&self, row: usize, code: &QueryCode) -> i32 {
        match (&self.repr, code) {
            (
                QuantRepr::TurboQuant {
                    codes,
                    scales,
                    centroids,
                    code_bytes,
                    bits: 1,
                    ..
                },
                QueryCode::TurboQuant1Hybrid {
                    q_rot, q_sum, qm, ..
                },
            ) => {
                assert!(row < self.n, "row out of range");
                let code = &codes.as_slice()[row * code_bytes..(row + 1) * code_bytes];
                let acc = tq1_adc_swar(code, centroids, q_rot, q_rot.len(), *q_sum);
                let ip = scales[row] * (acc - qm);
                (ip.clamp(-32.0, 32.0) * TQ_PROXY_SCALE) as i32
            }
            // Every non-hybrid mode ranks by the same proxy it walked with.
            _ => self.proxy(row, code),
        }
    }

    /// The tq1 proxy mode this set will use, or `None` if it is not a 1-bit
    /// TurboQuant set. Derived from `(dim, bits)`; observability for the
    /// auto-selection (e.g. logging which proxy a vindex picked at create).
    #[must_use]
    pub fn tq1_proxy_mode(&self) -> Option<Tq1ProxyMode> {
        match &self.repr {
            QuantRepr::TurboQuant { bits: 1, .. } => Some(tq1_proxy_mode_for(self.dim, 1)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Anisotropy calibration must collapse to identity on N(0,1/dim) and
    /// up-weight (not down-weight) a high-variance coordinate.
    #[test]
    fn calibrate_collapses_to_identity_and_scales_variance() {
        let dim = 256;
        let rows = 8192;
        // Box-Muller N(0, 1/dim) samples (post-rotation target variance).
        let sd = 1.0 / (dim as f32).sqrt();
        let mut s: u64 = 0xDEAD_BEEF_1234_5678;
        let mut nrm = || {
            let mut u = |b: u32| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (((s >> b) & 0xFF_FFFF) as f32) / (0x100_0000 as f32)
            };
            let (u1, u2) = (u(8).max(1e-9), u(32));
            (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
        };
        let mut sample = vec![0.0f32; rows * dim];
        for v in &mut sample {
            *v = sd * nrm();
        }
        let a = calibrate_tq1(&sample, rows, dim);
        // isotropic -> identity (shift~0, inv_scale~1)
        for d in 0..dim {
            assert!(
                a.shift[d].abs() < 0.05 * sd,
                "shift[{d}]={} too large",
                a.shift[d]
            );
            assert!(
                (a.inv_scale[d] - 1.0).abs() < 0.15,
                "inv_scale[{d}]={}",
                a.inv_scale[d]
            );
        }
        // Double one coord's std -> inv_scale ~ 2.0 (UP-weight; 0.5 = inverted bug).
        for r in 0..rows {
            sample[r * dim + 7] *= 2.0;
        }
        let a2 = calibrate_tq1(&sample, rows, dim);
        assert!(
            (a2.inv_scale[7] - 2.0).abs() < 0.3,
            "2x-std inv_scale={} (want ~2.0)",
            a2.inv_scale[7]
        );
        // too-small sample -> identity
        assert!(calibrate_tq1(&sample, TQ1_CALIB_MIN - 1, dim).is_identity());
    }

    /// The bit-plane primitive must reconstruct `sum_{code_i=1} Q_i` exactly:
    /// that is what makes `<q,s> = m*(2*pc-dim) + sq*(2*weighted - sum_q)` equal
    /// the asymmetric ADC. A wrong plane/AND alignment would corrupt ranking.
    #[test]
    fn bitplane_score_reconstructs_masked_sum() {
        let b = 3u8;
        let bytes = 1;
        // Stored sign code: coords 0, 2, 5 are "positive" (bit set).
        let code = [0b0010_0101u8];
        // Per-coord 3-bit query levels.
        let q = [5u32, 1, 3, 4, 6, 7, 2, 0];
        let mut planes = vec![0u8; b as usize * bytes];
        for (i, &qi) in q.iter().enumerate() {
            for p in 0..b as usize {
                if (qi >> p) & 1 == 1 {
                    planes[p * bytes] |= 1u8 << i;
                }
            }
        }
        let (weighted, code_pc) = tq1_bitplane_score(&planes, b, bytes, &code);
        // sum over set coords {0,2,5}: 5 + 3 + 7 = 15; popcount(code) = 3.
        assert_eq!(weighted, 15, "weighted masked sum");
        assert_eq!(code_pc, 3, "code popcount");
    }

    /// Deterministic pseudo-random unit vectors, `n` of dimension `dim`.
    fn unit_rows(n: usize, dim: usize) -> Vec<f32> {
        let mut data = vec![0.0f32; n * dim];
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
        for row in data.chunks_exact_mut(dim) {
            for x in row.iter_mut() {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                *x = (s >> 11) as f32 / (1u64 << 53) as f32 - 0.5;
            }
            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            for x in row.iter_mut() {
                *x /= norm;
            }
        }
        data
    }

    #[test]
    fn tq1_auto_proxy_is_bitplane_at_every_dim() {
        // 1-bit default is BitPlane at every dim (no dimensional switch), with
        // the integer bit-plane query code. Each dim's proxy ranks a vector
        // highest against itself.
        for dim in [384usize, 1024, 2560] {
            let data = unit_rows(8, dim);
            let qv = QuantizedVectors::build(&data, dim, QuantKind::TurboQuant { bits: 1 });
            assert_eq!(
                qv.tq1_proxy_mode(),
                Some(Tq1ProxyMode::BitPlane),
                "dim {dim} should select bit-plane"
            );
            let code = qv.quantize_query(&data[3 * dim..4 * dim]);
            assert!(matches!(code, QueryCode::TurboQuant1BitPlane { .. }));
            let self_best = (0..8)
                .map(|r| qv.proxy(r, &code))
                .enumerate()
                .max_by_key(|&(_, s)| s)
                .unwrap()
                .0;
            assert_eq!(self_best, 3, "bit-plane proxy self-rank at dim {dim}");
        }

        // 2-bit/4-bit never report a tq1 mode.
        let lo = unit_rows(4, 384);
        let q2 = QuantizedVectors::build(&lo, 384, QuantKind::TurboQuant { bits: 2 });
        assert_eq!(q2.tq1_proxy_mode(), None);
    }

    #[test]
    fn tq1_low_dim_expands_and_stays_consistent() {
        // dim < TQ1_EXPAND_BELOW: the 1-bit code expands to TQ1_EXPAND_TO bits.
        // Encode and query must pad identically so a vector still ranks highest
        // against itself (the padding-mismatch guard: a wrong rot_dim anywhere
        // in the proxy/query breaks self-rank).
        let dim = 128;
        assert!(dim < TQ1_EXPAND_BELOW);
        let data = unit_rows(8, dim);
        let qv = QuantizedVectors::build(&data, dim, QuantKind::TurboQuant { bits: 1 });
        assert_eq!(qv.tq1_proxy_mode(), Some(Tq1ProxyMode::BitPlane));
        for probe in 0..8 {
            let code = qv.quantize_query(&data[probe * dim..(probe + 1) * dim]);
            let best = (0..8)
                .map(|r| qv.proxy(r, &code))
                .enumerate()
                .max_by_key(|&(_, s)| s)
                .unwrap()
                .0;
            assert_eq!(best, probe, "expanded self-rank probe {probe}");
        }
    }

    #[test]
    fn validate_dim_turboquant_packing() {
        // tq1 packs 8 codes/byte -> dim must be % 8; 100 % 8 == 4 -> reject.
        assert!(QuantKind::TurboQuant { bits: 1 }.validate_dim(128).is_ok());
        assert!(QuantKind::TurboQuant { bits: 1 }.validate_dim(100).is_err());
        // tq2 needs % 4, tq4 needs % 2 -> both fine at 100.
        assert!(QuantKind::TurboQuant { bits: 2 }.validate_dim(100).is_ok());
        assert!(QuantKind::TurboQuant { bits: 4 }.validate_dim(100).is_ok());
        // Non-packed kinds accept any positive dim.
        assert!(QuantKind::Int8.validate_dim(100).is_ok());
        assert!(QuantKind::F32.validate_dim(100).is_ok());
    }

    #[test]
    fn int8_calibration_maps_max_to_127() {
        // Largest magnitude is 4.0 -> scale 4/127 -> that component quantizes to 127.
        let data = [1.0f32, -2.0, 4.0, 0.5];
        let q = QuantizedVectors::build(&data, 4, QuantKind::Int8);
        let scale = q.int8_scale().unwrap();
        assert!((scale - 4.0 / 127.0).abs() < 1e-9);
    }

    #[test]
    fn int8_quantize_query_is_consistent_with_rows() {
        let data = [1.0f32, -2.0, 4.0, 0.5];
        let q = QuantizedVectors::build(&data, 4, QuantKind::Int8);
        // Quantizing the same vector as a query reproduces the stored row, so
        // the proxy of a vector against itself is its squared int8 norm.
        let code = q.quantize_query(&data);
        let QueryCode::Int8(qd) = &code else {
            panic!("expected int8")
        };
        let self_dot: i32 = qd.iter().map(|&x| i32::from(x) * i32::from(x)).sum();
        assert_eq!(q.proxy(0, &code), self_dot);
    }

    #[test]
    fn binary_pack_known_pattern() {
        // signs: + - + - + + + +  -> bits 0,2,4,5,6,7 set -> 0b1111_0101 = 0xF5
        let v = [1.0f32, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0];
        let q = QuantizedVectors::build(&v, 8, QuantKind::Binary);
        // A query equal to v has Hamming distance 0 -> proxy 0 (max closeness).
        let code = q.quantize_query(&v);
        assert_eq!(q.proxy(0, &code), 0);
    }

    #[test]
    fn binary_hamming_distance_is_exact() {
        // Two rows differing in 3 sign positions.
        let row0 = [1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let row1 = [-1.0f32, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mut data = Vec::new();
        data.extend_from_slice(&row0);
        data.extend_from_slice(&row1);
        let q = QuantizedVectors::build(&data, 8, QuantKind::Binary);
        let code = q.quantize_query(&row0);
        assert_eq!(q.proxy(0, &code), 0); // identical
        assert_eq!(q.proxy(1, &code), -3); // 3 differing sign bits
    }

    #[test]
    fn binary_dim_not_multiple_of_8() {
        // dim 11 -> 2 bytes per vector; trailing 5 bits are padding.
        let v: Vec<f32> = (0..11)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let q = QuantizedVectors::build(&v, 11, QuantKind::Binary);
        let code = q.quantize_query(&v);
        assert_eq!(q.proxy(0, &code), 0);
    }

    #[test]
    fn pq_self_query_ranks_top() {
        // 4 vectors, dim 8, m=2 (sub_dim 4), k=4. Each subvector half is one
        // of 4 distinct one-hot patterns: k=4 k-means recovers them exactly,
        // so a vector queried against itself has ADC 0 (maximal proxy).
        let data: [f32; 32] = [
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let q = QuantizedVectors::build(&data, 8, QuantKind::Pq { m: 2, k: 4 });
        for i in 0..4 {
            let code = q.quantize_query(&data[i * 8..(i + 1) * 8]);
            let self_p = q.proxy(i, &code);
            for j in 0..4 {
                if j != i {
                    assert!(
                        q.proxy(j, &code) <= self_p,
                        "row {i} should rank at least as close as row {j}"
                    );
                }
            }
        }
    }

    #[test]
    fn pq_memory_bytes_counts_codes_and_codebook() {
        // 4 vectors dim 8, m=2 k=4: codes = 4*2 = 8 bytes; codebook = m*k*sub_dim
        // f32 = 2*4*4*4 = 128 bytes.
        let data = vec![0.0f32; 32];
        let q = QuantizedVectors::build(&data, 8, QuantKind::Pq { m: 2, k: 4 });
        assert_eq!(q.memory_bytes(), 8 + 128);
        assert_eq!(q.int8_scale(), None);
    }

    #[test]
    #[should_panic(expected = "dim must be divisible by m")]
    fn pq_rejects_dim_not_divisible_by_m() {
        let data = vec![0.0f32; 30]; // dim 10, m 3 -> not divisible
        let _ = QuantizedVectors::build(&data, 10, QuantKind::Pq { m: 3, k: 2 });
    }
}
