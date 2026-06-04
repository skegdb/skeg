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
/// pushed to anonymous swap. Opt-in via `--tier-mmap` (Position 2 of the
/// VeloANN paging discussion, OBSERVATIONS 2026-05-21).
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
    TurboQuant { q_rot: Vec<f32>, q_sum: f32 },
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

/// Inverse of `turboquant_pack`: read the `bits`-wide bucket id at coord `i`.
fn turboquant_unpack(code: &[u8], i: usize, bits: u8) -> usize {
    let bits = bits as usize;
    let codes_per_byte = 8 / bits;
    let byte = i / codes_per_byte;
    let shift = (i % codes_per_byte) * bits;
    let mask = (1u8 << bits) - 1;
    ((code[byte] >> shift) & mask) as usize
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
) -> (Vec<u8>, f32) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let inv = if norm > 1e-10 { 1.0 / norm } else { 0.0 };
    let mut unit = vec![0.0f32; dim];
    for (u, &x) in unit.iter_mut().zip(v.iter()) {
        *u = x * inv;
    }
    let rotated = rotation.apply_alloc(&unit);
    let mut code = vec![0u8; code_bytes];
    let mut inner = 0.0f32;
    for (i, &r) in rotated.iter().enumerate() {
        let bucket = turboquant_bucket(r, boundaries);
        inner += r * centroids[bucket];
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
    let code_bytes = dim * (bits as usize) / 8;
    let rotation = Box::new(FastRotation::new(dim, TQ_ROTATION_SEED));
    let (centroids, boundaries) = turboquant_levels(dim, bits);
    let (centroids_i8, i8_scale) = quantise_tq_centroids(&centroids, bits);

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
        let code_bytes = dim * (bits as usize) / 8;
        let rotation = Box::new(FastRotation::new(dim, TQ_ROTATION_SEED));
        let (centroids, boundaries) = turboquant_levels(dim, bits);
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
                },
            });
        }
        let mut codes_buf: Vec<u8> = Vec::with_capacity(n * code_bytes);
        let mut scales: Vec<f32> = Vec::with_capacity(n);
        // Single pass: encode each row as it arrives. No training pass needed
        // (data-oblivious). Sequential because the input stream is sequential;
        // parallelism would require buffering all rows.
        for_each_row(&mut |row| {
            let (c, s) = turboquant_encode_vec(
                row,
                dim,
                bits,
                code_bytes,
                &rotation,
                &centroids,
                &boundaries,
            );
            codes_buf.extend_from_slice(&c);
            scales.push(s);
        })?;
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
            }
        }
    }

    /// Persist the TurboQuant `codes` buffer to `path` and swap the in-RAM
    /// `Vec<u8>` for a memory-mapped view of the file. The OS page cache
    /// then decides which pages stay resident under memory pressure
    /// instead of swapping anonymous memory (Position 2 of the VeloANN
    /// paging discussion, OBSERVATIONS 2026-05-21).
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
            QuantRepr::TurboQuant { rotation, .. } => {
                let mut unit = vec![0.0f32; self.dim];
                normalize_into(query, &mut unit);
                let q_rot = rotation.apply_alloc(&unit);
                let q_sum = q_rot.iter().sum();
                QueryCode::TurboQuant { q_rot, q_sum }
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
                QueryCode::TurboQuant { q_rot, q_sum },
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
                    4 => tq4_adc_i8(code, centroids_i8, *i8_scale, q_rot, self.dim),
                    2 => tq2_adc_i8(code, centroids_i8, *i8_scale, q_rot, self.dim),
                    // 1-bit: algebraic reduction `c * (2*masked - q_sum)`.
                    // q_sum precomputed at query time; SWAR scalar inner.
                    1 => tq1_adc_swar(code, centroids, q_rot, self.dim, *q_sum),
                    _ => {
                        // Fallback general scalar path. The supported set is
                        // {1, 2, 4} - this arm is for future-proofing.
                        let mut a = 0.0f32;
                        for i in 0..self.dim {
                            let bucket = turboquant_unpack(code, i, *bits);
                            a += q_rot[i] * centroids[bucket];
                        }
                        a
                    }
                };
                let ip = scales[row] * acc;
                // Greater inner product = closer. Clamp into a safe range so
                // adversarial scales never overflow i32 - unit vectors give
                // |ip| <= 1, the clamp at 4.0 leaves ample headroom.
                (ip.clamp(-4.0, 4.0) * TQ_PROXY_SCALE) as i32
            }
            _ => panic!("query code does not match index quantization"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
