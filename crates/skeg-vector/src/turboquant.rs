//! TurboQuant 4-bit quantizer (prototype, gate-driven).
//!
//! Pipeline:
//!   - Random orthogonal rotation `Q` (deterministic from seed)
//!   - Normalize: u = v / ||v||
//!   - Rotate: u' = Q * u (each coordinate of u' follows
//!     Beta((d-1)/2, (d-1)/2) on [-1, 1], approximately N(0, 1/d) for d >= 200)
//!   - Quantize each coord of u' to a 4-bit nibble via pre-computed Lloyd-Max
//!     levels scaled by 1/sqrt(d)
//!   - Store: nibble code (dim/2 bytes) + per-vector scale (4 bytes)
//!
//! Distance to query `q`:
//!   - Rotate query once: q' = Q * q
//!   - Estimated <v, q> ~= scale * sum_i q'[i] * centroid[code[i]]
//!
//! Approximations vs the turbovec reference:
//!   - Lloyd-Max levels: Gaussian N(0,1) (Joel Max 1960) scaled 1/sqrt(d-1).
//!     For d >= 200 the Beta((d-1)/2, (d-1)/2) is very close to a Gaussian and
//!     the relative MSE penalty is ~0.5%. Acceptable for the gate.
//!   - No bit-plane packing: code is one nibble per coord stored as one
//!     packed byte per two coords. Costs an unpack in the hot loop; SIMD is
//!     out of scope for the prototype.
//!
//! References:
//!   - Zandieh, Daliri, Hadian, Mirrokni (2025). TurboQuant. arXiv:2504.19874.
//!   - Codrai (2025). turbovec. github.com/RyanCodrai/turbovec (oracle for
//!     pipeline shape, scale-correction formula).
//!   - Max (1960). Quantizing for Minimum Distortion. IRE Trans. Information
//!     Theory (source of the Gaussian Lloyd-Max levels).

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

const TQ_BITS: usize = 4;
const TQ_LEVELS: usize = 1 << TQ_BITS;

/// Joel Max (1960) optimal Lloyd-Max centroids for N(0, 1) with 16 levels,
/// ordered ascending. Symmetric around zero.
const LM_CENTROIDS_N01: [f32; TQ_LEVELS] = [
    -2.7326, -2.0691, -1.6180, -1.2562, -0.9424, -0.6568, -0.3881, -0.1284, 0.1284, 0.3881, 0.6568,
    0.9424, 1.2562, 1.6180, 2.0691, 2.7326,
];

/// Midpoints between consecutive centroids: 15 thresholds that partition the
/// real line into 16 buckets.
const LM_BOUNDARIES_N01: [f32; TQ_LEVELS - 1] = [
    -2.4009, -1.8436, -1.4371, -1.0993, -0.7996, -0.5225, -0.2583, 0.0, 0.2583, 0.5225, 0.7996,
    1.0993, 1.4371, 1.8436, 2.4009,
];

/// N(0, 1) Lloyd-Max centroids by bit width. `bits` in {1, 2, 4} returns a
/// slice of `2^bits` ascending values. Other widths panic - the supported
/// set is fixed for the v0.2 production tier.
pub(crate) fn lm_centroids_n01(bits: u8) -> &'static [f32] {
    match bits {
        1 => &LM1_CENTROIDS_N01,
        2 => &LM2_CENTROIDS_N01,
        4 => &LM_CENTROIDS_N01,
        _ => panic!("TurboQuant supports bits in {{1, 2, 4}}, got {}", bits),
    }
}

/// N(0, 1) Lloyd-Max bucket boundaries by bit width. 1-bit returns an empty
/// slice (the threshold is exactly zero, handled inline by the bucketiser).
pub(crate) fn lm_boundaries_n01(bits: u8) -> &'static [f32] {
    match bits {
        1 => &[],
        2 => &LM2_BOUNDARIES_N01,
        4 => &LM_BOUNDARIES_N01,
        _ => panic!("TurboQuant supports bits in {{1, 2, 4}}, got {}", bits),
    }
}

/// A dim x dim orthogonal matrix obtained by Gram-Schmidt on a Gaussian
/// matrix seeded with `seed`. Row-major. `O(dim^2)` storage,
/// `O(dim^3)` build time, applied as `O(dim^2)` matvec per call.
#[derive(Debug)]
pub struct Rotation {
    dim: usize,
    q: Vec<f32>,
}

impl Rotation {
    /// Build a deterministic random orthogonal rotation.
    ///
    /// # Panics
    ///
    /// Panics if `dim == 0`.
    #[must_use]
    pub fn new(dim: usize, seed: u64) -> Self {
        assert!(dim > 0, "dim must be positive");
        let mut rng = StdRng::seed_from_u64(seed);
        let mut q = vec![0.0f32; dim * dim];
        // Box-Muller: each pair of uniforms yields two independent N(0,1)
        // samples. Avoids pulling in rand_distr.
        let mut i = 0;
        while i + 1 < q.len() {
            let (z0, z1) = box_muller(&mut rng);
            q[i] = z0;
            q[i + 1] = z1;
            i += 2;
        }
        if i < q.len() {
            let (z0, _) = box_muller(&mut rng);
            q[i] = z0;
        }
        gram_schmidt(&mut q, dim);
        Rotation { dim, q }
    }

    /// Dimension of vectors this rotation operates on.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Apply the rotation to `x`, writing into `out`. `O(dim^2)`.
    ///
    /// # Panics
    ///
    /// Panics if either slice does not have length `dim`.
    pub fn apply(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.dim);
        assert_eq!(out.len(), self.dim);
        for i in 0..self.dim {
            let row = &self.q[i * self.dim..(i + 1) * self.dim];
            let mut acc = 0.0f32;
            for j in 0..self.dim {
                acc += row[j] * x[j];
            }
            out[i] = acc;
        }
    }

    /// Convenience wrapper that allocates the output vector.
    #[must_use]
    pub fn apply_alloc(&self, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.dim];
        self.apply(x, &mut out);
        out
    }
}

/// Box-Muller transform: two independent N(0,1) samples per pair of uniforms.
fn box_muller(rng: &mut StdRng) -> (f32, f32) {
    let mut u1: f32 = rng.random();
    if u1 < 1e-10 {
        u1 = 1e-10;
    }
    let u2: f32 = rng.random();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

/// Modified Gram-Schmidt in place on a row-major `dim x dim` matrix. Each row
/// is normalised against the rows above it, producing an orthonormal basis.
fn gram_schmidt(q: &mut [f32], dim: usize) {
    for i in 0..dim {
        for k in 0..i {
            let mut dot = 0.0f32;
            for j in 0..dim {
                dot += q[i * dim + j] * q[k * dim + j];
            }
            for j in 0..dim {
                q[i * dim + j] -= dot * q[k * dim + j];
            }
        }
        let mut norm_sq = 0.0f32;
        for j in 0..dim {
            norm_sq += q[i * dim + j] * q[i * dim + j];
        }
        let inv = 1.0 / norm_sq.sqrt();
        for j in 0..dim {
            q[i * dim + j] *= inv;
        }
    }
}

/// TurboQuant 4-bit quantizer parameterised by dimension. Holds the
/// orthogonal rotation and the dimension-scaled Lloyd-Max boundaries
/// and centroids.
pub struct TurboQuant4 {
    rotation: Rotation,
    /// Lloyd-Max boundaries scaled to the post-rotation distribution
    /// (variance ~1/d). 15 thresholds, ascending.
    boundaries: [f32; TQ_LEVELS - 1],
    /// Lloyd-Max centroids scaled to the post-rotation distribution.
    centroids: [f32; TQ_LEVELS],
}

impl TurboQuant4 {
    /// Build a quantizer for `dim`-dim vectors with a seeded rotation.
    ///
    /// # Panics
    ///
    /// Panics if `dim == 0` or `dim` is odd (nibble packing requires even
    /// dim).
    #[must_use]
    pub fn new(dim: usize, seed: u64) -> Self {
        assert!(dim > 0, "dim must be positive");
        assert_eq!(dim % 2, 0, "dim must be even for nibble packing");
        let scale = 1.0 / (dim as f32).sqrt();
        let mut boundaries = [0.0f32; TQ_LEVELS - 1];
        for (b, &n) in boundaries.iter_mut().zip(LM_BOUNDARIES_N01.iter()) {
            *b = n * scale;
        }
        let mut centroids = [0.0f32; TQ_LEVELS];
        for (c, &n) in centroids.iter_mut().zip(LM_CENTROIDS_N01.iter()) {
            *c = n * scale;
        }
        TurboQuant4 {
            rotation: Rotation::new(dim, seed),
            boundaries,
            centroids,
        }
    }

    /// Dimension this quantizer is parameterised on.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.rotation.dim()
    }

    /// Bytes stored per encoded vector (one nibble per coord, packed two per
    /// byte).
    #[must_use]
    pub fn code_bytes(&self) -> usize {
        self.rotation.dim() / 2
    }

    /// Rotate a query into the encoding domain. Cosine of unit `q` against a
    /// stored unit `v` equals `<Q q, Q v>`; the rotated query is then dotted
    /// against dequantised centroids in [`approx_inner`](Self::approx_inner).
    #[must_use]
    pub fn rotate_query(&self, q: &[f32]) -> Vec<f32> {
        self.rotation.apply_alloc(q)
    }

    /// Encode one vector. Returns the packed nibble code (`code_bytes`
    /// bytes) and the per-vector scale correction `||v|| / <u, x_hat>` -
    /// applying it to the asymmetric dot in [`approx_inner`](Self::approx_inner)
    /// removes the systematic shrinkage from quantisation. For a unit input
    /// the scale equals `1 / <u, x_hat>`.
    ///
    /// # Panics
    ///
    /// Panics if `v.len() != dim`.
    pub fn encode(&self, v: &[f32]) -> (Vec<u8>, f32) {
        let dim = self.dim();
        assert_eq!(v.len(), dim);
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let inv = if norm > 1e-10 { 1.0 / norm } else { 0.0 };
        let mut unit = vec![0.0f32; dim];
        for (u, &x) in unit.iter_mut().zip(v.iter()) {
            *u = x * inv;
        }
        let rotated = self.rotation.apply_alloc(&unit);
        let mut code = vec![0u8; dim / 2];
        let mut inner = 0.0f32;
        for (i, &r) in rotated.iter().enumerate() {
            let bucket = bucketize(r, &self.boundaries);
            inner += r * self.centroids[bucket];
            // Pack two nibbles per byte: even coord -> low nibble, odd -> high.
            let byte = i / 2;
            let shift = (i % 2) * 4;
            code[byte] |= (bucket as u8) << shift;
        }
        let inner = inner.max(1e-10);
        let scale = norm / inner;
        (code, scale)
    }

    /// Approximate `<v, q>` for a stored `(code, scale)` pair and a rotated
    /// query `q_rot` (output of [`rotate_query`](Self::rotate_query)).
    /// O(dim).
    ///
    /// # Panics
    ///
    /// Panics if lengths do not match.
    #[must_use]
    pub fn approx_inner(&self, code: &[u8], scale: f32, q_rot: &[f32]) -> f32 {
        let dim = self.dim();
        assert_eq!(code.len(), dim / 2);
        assert_eq!(q_rot.len(), dim);
        let mut acc = 0.0f32;
        for byte_idx in 0..dim / 2 {
            let byte = code[byte_idx];
            let lo = (byte & 0x0F) as usize;
            let hi = ((byte >> 4) & 0x0F) as usize;
            acc += q_rot[2 * byte_idx] * self.centroids[lo];
            acc += q_rot[2 * byte_idx + 1] * self.centroids[hi];
        }
        scale * acc
    }
}

/// Map a coordinate to its 4-bit bucket index using the sorted boundaries.
/// Linear scan: with 15 boundaries this beats a branchless binary search on
/// short inputs because the boundary array stays in L1 and the compare-
/// increment compiles tight.
fn bucketize(x: f32, boundaries: &[f32]) -> usize {
    let mut bucket = 0usize;
    for &b in boundaries {
        if x > b {
            bucket += 1;
        }
    }
    bucket
}

// -- Fast rotation: block Walsh-Hadamard + signed diagonals -------------------

/// Random orthogonal transform via block Walsh-Hadamard with random sign
/// diagonal masks. `O(d log b)` per apply (where `b` is the block size,
/// the largest power of 2 dividing `d`) versus `O(d^2)` of the explicit
/// Gram-Schmidt matrix. Three rounds of `D_k . FWHT_b` are composed; this
/// is the standard fast-Johnson-Lindenstrauss construction (Lance and
/// turbovec ship the same shape) and converges to a near-uniform
/// orthogonal transform in three rounds for the dimensions we care about
/// (256-1024).
///
/// `dim` does not need to be a power of two: it is broken into
/// `dim / largest_pow2(dim)` blocks of `largest_pow2(dim)` floats. mxbai
/// 1024d -> one block of 1024; MiniLM 384d -> three blocks of 128. The
/// rotation is still orthogonal because each block is independent and
/// each block transform is orthogonal.
#[derive(Debug)]
pub struct FastRotation {
    dim: usize,
    block: usize,
    /// Three rounds of `dim`-bit sign masks, packed `dim.div_ceil(8)` bytes
    /// each. Bit `i` set means coord `i` flips sign in that round.
    sign_masks: [Vec<u8>; 3],
    /// Per-FWHT-pass scaling factor `1/sqrt(block)`. Composing three passes
    /// gives the orthogonal normaliser `1/block^(3/2)`.
    pass_scale: f32,
}

impl FastRotation {
    /// Build a deterministic fast random orthogonal transform from `seed`.
    ///
    /// # Panics
    ///
    /// Panics if `dim == 0` or if `dim` has no power-of-2 factor `>= 2`
    /// (i.e. `dim` is odd).
    #[must_use]
    pub fn new(dim: usize, seed: u64) -> Self {
        assert!(dim > 0, "dim must be positive");
        let block = largest_pow2_factor(dim);
        assert!(block >= 2, "dim must have a power-of-2 factor >= 2");
        let mut rng = StdRng::seed_from_u64(seed);
        let mask_bytes = dim.div_ceil(8);
        let make_mask = |rng: &mut StdRng| {
            let mut m = vec![0u8; mask_bytes];
            for i in 0..dim {
                let r: u32 = rng.random();
                if r & 1 == 1 {
                    m[i / 8] |= 1 << (i % 8);
                }
            }
            m
        };
        let sign_masks = [
            make_mask(&mut rng),
            make_mask(&mut rng),
            make_mask(&mut rng),
        ];
        let pass_scale = 1.0 / (block as f32).sqrt();
        FastRotation {
            dim,
            block,
            sign_masks,
            pass_scale,
        }
    }

    /// The transform's working dimension (may exceed the index's native dim
    /// when low-dim tq1 is zero-padded up for more code bits).
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Apply the transform to `x`, writing into `out`.
    ///
    /// # Panics
    ///
    /// Panics if either slice does not have length `dim`.
    pub fn apply(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.dim);
        assert_eq!(out.len(), self.dim);
        out.copy_from_slice(x);
        for round in 0..3 {
            flip_signs(out, &self.sign_masks[round]);
            for block in out.chunks_exact_mut(self.block) {
                fwht_inplace(block);
            }
        }
        // Three FWHT passes (each scales norm by sqrt(block)) need the
        // normaliser applied once with cubic power. Folding it into one
        // multiplication keeps the inner loops branch-free.
        let total = self.pass_scale.powi(3);
        for v in out.iter_mut() {
            *v *= total;
        }
    }

    /// Allocating wrapper around [`apply`](Self::apply).
    #[must_use]
    pub fn apply_alloc(&self, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.dim];
        self.apply(x, &mut out);
        out
    }
}

/// Largest power of two that divides `n`. Returns 1 for an odd `n`.
fn largest_pow2_factor(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let mut p = 1;
    let mut m = n;
    while m % 2 == 0 {
        p *= 2;
        m /= 2;
    }
    p
}

/// In-place sign flip on the indices marked by `mask` (LSB-first packing
/// matching [`FastRotation::sign_masks`]).
fn flip_signs(x: &mut [f32], mask: &[u8]) {
    for i in 0..x.len() {
        if (mask[i / 8] >> (i % 8)) & 1 == 1 {
            x[i] = -x[i];
        }
    }
}

/// In-place Walsh-Hadamard transform, unnormalised. `x.len()` must be a
/// power of two. After this call `||x'|| = sqrt(n) * ||x||`; the caller
/// scales as appropriate.
fn fwht_inplace(x: &mut [f32]) {
    let n = x.len();
    debug_assert!(n.is_power_of_two(), "FWHT requires a power-of-two length");
    let mut h = 1;
    while h < n {
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
        h *= 2;
    }
}

// -- SWAR ADC inner accumulators (stable Rust, no `unsafe`) -------------------
//
// The walk calls `proxy(row, query)` ~6400 times per query. Each call boils
// down to `sum_i q_rot[i] * centroid[code[i]]` where `code[i]` is `bits` wide
// and bit-packed. SWAR (SIMD Within A Register) processes 64 bits of code at
// a time via `u64` masking, replacing the per-coord `shift + mask + lookup`
// scalar dance with a few register-wide bitops. The inner FMA loop is then
// short, contiguous, and auto-vectorises on aarch64 without intrinsics.
//
// These helpers are pub(crate) so `quant.rs` (the production proxy path)
// dispatches to them per-bits. They live here next to the constants they
// reference.

/// Inner accumulator for `bits=4`: 16 coords per 8-byte SWAR chunk.
/// Returns the asymmetric inner product **before** the per-vector scale
/// correction. `code.len() == dim / 2`, `centroids.len() == 16`.
///
/// Superseded in production by `skeg_simd::tq4_adc_i8` (NEON `vqtbl1q_s8`,
/// 4x `vfmaq_f32`). Kept only as the oracle for the SWAR-vs-scalar
/// equivalence proptest, gated to test builds to keep the release binary lean.
#[cfg(test)]
pub(crate) fn tq4_adc_swar(code: &[u8], centroids: &[f32], q_rot: &[f32], dim: usize) -> f32 {
    debug_assert_eq!(code.len(), dim / 2);
    debug_assert_eq!(q_rot.len(), dim);
    debug_assert_eq!(centroids.len(), 16);
    let mut acc = 0.0f32;
    let chunks = dim / 16;
    for chunk in 0..chunks {
        let off = chunk * 8;
        let packed = u64::from_le_bytes(code[off..off + 8].try_into().unwrap());
        let low = (packed & 0x0F0F_0F0F_0F0F_0F0F).to_le_bytes();
        let high = ((packed >> 4) & 0x0F0F_0F0F_0F0F_0F0F).to_le_bytes();
        let q_base = chunk * 16;
        // 16 FMAs: contiguous, the auto-vectoriser pairs them into NEON FMA.
        for j in 0..8 {
            acc += q_rot[q_base + 2 * j] * centroids[low[j] as usize];
            acc += q_rot[q_base + 2 * j + 1] * centroids[high[j] as usize];
        }
    }
    // Tail: dims not a multiple of 16. mxbai 1024 and MiniLM 384 both fit.
    let mut i = chunks * 16;
    while i < dim {
        let byte = code[i / 2];
        let bucket = (if i % 2 == 0 { byte & 0x0F } else { byte >> 4 }) as usize;
        acc += q_rot[i] * centroids[bucket];
        i += 1;
    }
    acc
}

/// Inner accumulator for `bits=2`: 32 coords per 8-byte SWAR chunk.
/// `code.len() == dim / 4`, `centroids.len() == 4`.
///
/// Superseded in production by `skeg_simd::tq2_adc_i8` (NEON vtbl, 32
/// coords per chunk via two `vqtbl1q_s8` lookups). Test-only oracle for
/// the equivalence proptest, gated so it stays out of the release build.
#[cfg(test)]
pub(crate) fn tq2_adc_swar(code: &[u8], centroids: &[f32], q_rot: &[f32], dim: usize) -> f32 {
    debug_assert_eq!(code.len(), dim / 4);
    debug_assert_eq!(q_rot.len(), dim);
    debug_assert_eq!(centroids.len(), 4);
    let mut acc = 0.0f32;
    let chunks = dim / 32;
    for chunk in 0..chunks {
        let off = chunk * 8;
        let packed = u64::from_le_bytes(code[off..off + 8].try_into().unwrap());
        // Each byte holds four 2-bit codes at shifts 0, 2, 4, 6.
        let b0 = (packed & 0x0303_0303_0303_0303).to_le_bytes();
        let b1 = ((packed >> 2) & 0x0303_0303_0303_0303).to_le_bytes();
        let b2 = ((packed >> 4) & 0x0303_0303_0303_0303).to_le_bytes();
        let b3 = ((packed >> 6) & 0x0303_0303_0303_0303).to_le_bytes();
        let q_base = chunk * 32;
        for j in 0..8 {
            acc += q_rot[q_base + 4 * j] * centroids[b0[j] as usize];
            acc += q_rot[q_base + 4 * j + 1] * centroids[b1[j] as usize];
            acc += q_rot[q_base + 4 * j + 2] * centroids[b2[j] as usize];
            acc += q_rot[q_base + 4 * j + 3] * centroids[b3[j] as usize];
        }
    }
    let mut i = chunks * 32;
    while i < dim {
        let byte = code[i / 4];
        let shift = (i % 4) * 2;
        let bucket = ((byte >> shift) & 0x03) as usize;
        acc += q_rot[i] * centroids[bucket];
        i += 1;
    }
    acc
}

/// Inner accumulator for `bits=1`: 64 coords per 8-byte SWAR chunk.
/// Uses the symmetry of the 2-level Lloyd-Max levels (`centroids = +/-c`) to
/// reduce the kernel to `c * (2 * sum_{bit=1} q[i] - sum_i q[i])` - a single
/// masked partial sum plus a query-independent term. Branchless multiply by
/// the bit-as-f32 keeps the inner loop FMA-friendly.
///
/// `q_sum` is the query-level `sum(q_rot)`, precomputed once at query time
/// (in `quantize_query`) so the walk does not pay `dim` adds per ADC call.
/// `code.len() == dim / 8`, `centroids.len() == 2`.
pub(crate) fn tq1_adc_swar(
    code: &[u8],
    centroids: &[f32],
    q_rot: &[f32],
    dim: usize,
    q_sum: f32,
) -> f32 {
    debug_assert_eq!(code.len(), dim / 8);
    debug_assert_eq!(q_rot.len(), dim);
    debug_assert_eq!(centroids.len(), 2);
    // centroids[0] = -c, centroids[1] = +c. The asymmetric inner product
    // reduces to `c * (2 * q_masked - q_sum)`. The masked sum is the hot loop;
    // skeg-simd dispatches it to a NEON kernel on aarch64.
    let pos_c = centroids[1];
    let q_masked = skeg_simd::tq1_masked_sum(code, q_rot, dim);
    pos_c * (2.0 * q_masked - q_sum)
}

// -- TurboQuant 1-bit ---------------------------------------------------------

const TQ1_LEVELS: usize = 2;

/// Joel Max optimal 1-bit Lloyd-Max centroids for N(0, 1): the conditional
/// means of a half-Gaussian, +/- sqrt(2/PI) ~= +/- 0.7979. The boundary is
/// exactly zero (the midpoint of the two centroids). With per-vector scale
/// correction the asymmetric inner product matches the cosine ranking of
/// the underlying f32 vectors much better than a plain sign-only proxy.
const LM1_CENTROIDS_N01: [f32; TQ1_LEVELS] = [-0.7978846, 0.7978846];

/// TurboQuant 1-bit quantizer. Eight codes per byte: at dim 1024 that is
/// 128 bytes per vector, matching the PQ-128 footprint. The walk-navigation
/// barrier failed at this resolution for `binary` (0.748) and `RaBitQ` (0.881);
/// the unknown is whether the per-vector scale correction (which RaBitQ in
/// our 2026 implementation lacked) carries the proxy past 0.95 - the same
/// jump the scale correction delivered from `4-bit naive` (0.689) to
/// `TurboQuant 2-bit` (0.984).
pub struct TurboQuant1 {
    rotation: Rotation,
    centroids: [f32; TQ1_LEVELS],
}

impl TurboQuant1 {
    /// Build a 1-bit quantizer for `dim`-dim vectors.
    ///
    /// # Panics
    ///
    /// Panics if `dim == 0` or `dim % 8 != 0`.
    #[must_use]
    pub fn new(dim: usize, seed: u64) -> Self {
        assert!(dim > 0, "dim must be positive");
        assert_eq!(dim % 8, 0, "dim must be divisible by 8 for 1-bit packing");
        let scale = 1.0 / (dim as f32).sqrt();
        let mut centroids = [0.0f32; TQ1_LEVELS];
        for (c, &n) in centroids.iter_mut().zip(LM1_CENTROIDS_N01.iter()) {
            *c = n * scale;
        }
        TurboQuant1 {
            rotation: Rotation::new(dim, seed),
            centroids,
        }
    }

    #[must_use]
    pub fn dim(&self) -> usize {
        self.rotation.dim()
    }

    /// Bytes per encoded vector (eight 1-bit codes per byte).
    #[must_use]
    pub fn code_bytes(&self) -> usize {
        self.rotation.dim() / 8
    }

    #[must_use]
    pub fn rotate_query(&self, q: &[f32]) -> Vec<f32> {
        self.rotation.apply_alloc(q)
    }

    /// Encode one vector with sign-bit packing and the per-vector scale
    /// `||v|| / <u, x_hat>` correction.
    ///
    /// # Panics
    ///
    /// Panics if `v.len() != dim`.
    pub fn encode(&self, v: &[f32]) -> (Vec<u8>, f32) {
        let dim = self.dim();
        assert_eq!(v.len(), dim);
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let inv = if norm > 1e-10 { 1.0 / norm } else { 0.0 };
        let mut unit = vec![0.0f32; dim];
        for (u, &x) in unit.iter_mut().zip(v.iter()) {
            *u = x * inv;
        }
        let rotated = self.rotation.apply_alloc(&unit);
        let mut code = vec![0u8; dim / 8];
        let mut inner = 0.0f32;
        for (i, &r) in rotated.iter().enumerate() {
            let bit = usize::from(r > 0.0);
            inner += r * self.centroids[bit];
            // Eight bits per byte: bit i mod 8 of byte i / 8.
            if bit == 1 {
                code[i / 8] |= 1u8 << (i % 8);
            }
        }
        let inner = inner.max(1e-10);
        let scale = norm / inner;
        (code, scale)
    }

    /// Approximate `<v, q>` for a stored `(code, scale)` pair and rotated
    /// query.
    ///
    /// # Panics
    ///
    /// Panics if lengths do not match.
    #[must_use]
    pub fn approx_inner(&self, code: &[u8], scale: f32, q_rot: &[f32]) -> f32 {
        let dim = self.dim();
        assert_eq!(code.len(), dim / 8);
        assert_eq!(q_rot.len(), dim);
        let mut acc = 0.0f32;
        for byte_idx in 0..dim / 8 {
            let byte = code[byte_idx];
            for bit_pos in 0..8 {
                let bit = ((byte >> bit_pos) & 1) as usize;
                acc += q_rot[byte_idx * 8 + bit_pos] * self.centroids[bit];
            }
        }
        scale * acc
    }
}

// -- TurboQuant 2-bit ---------------------------------------------------------

const TQ2_BITS: usize = 2;
const TQ2_LEVELS: usize = 1 << TQ2_BITS;

/// Joel Max (1960) optimal Lloyd-Max centroids for N(0, 1) with 4 levels,
/// ordered ascending.
const LM2_CENTROIDS_N01: [f32; TQ2_LEVELS] = [-1.5104, -0.4528, 0.4528, 1.5104];

/// Midpoints between consecutive centroids: 3 thresholds.
const LM2_BOUNDARIES_N01: [f32; TQ2_LEVELS - 1] = [-0.9816, 0.0, 0.9816];

/// TurboQuant 2-bit quantizer. Four codes per byte instead of two, half the
/// bytes per vector at the cost of resolution. Walk-navigation gate is the
/// barrier: 1-bit (binary, RaBitQ) failed at 0.75-0.88; naive 4-bit failed at
/// 0.69; 4-bit rotated + scale passed at 0.99. 2-bit rotated + scale sits in
/// the middle and is the unknown to measure.
pub struct TurboQuant2 {
    rotation: Rotation,
    boundaries: [f32; TQ2_LEVELS - 1],
    centroids: [f32; TQ2_LEVELS],
}

impl TurboQuant2 {
    /// Build a 2-bit quantizer for `dim`-dim vectors.
    ///
    /// # Panics
    ///
    /// Panics if `dim == 0` or `dim % 4 != 0`.
    #[must_use]
    pub fn new(dim: usize, seed: u64) -> Self {
        assert!(dim > 0, "dim must be positive");
        assert_eq!(dim % 4, 0, "dim must be divisible by 4 for 2-bit packing");
        let scale = 1.0 / (dim as f32).sqrt();
        let mut boundaries = [0.0f32; TQ2_LEVELS - 1];
        for (b, &n) in boundaries.iter_mut().zip(LM2_BOUNDARIES_N01.iter()) {
            *b = n * scale;
        }
        let mut centroids = [0.0f32; TQ2_LEVELS];
        for (c, &n) in centroids.iter_mut().zip(LM2_CENTROIDS_N01.iter()) {
            *c = n * scale;
        }
        TurboQuant2 {
            rotation: Rotation::new(dim, seed),
            boundaries,
            centroids,
        }
    }

    /// Dimension this quantizer is parameterised on.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.rotation.dim()
    }

    /// Bytes stored per encoded vector (four 2-bit codes per byte).
    #[must_use]
    pub fn code_bytes(&self) -> usize {
        self.rotation.dim() / 4
    }

    /// Rotate a query into the encoding domain.
    #[must_use]
    pub fn rotate_query(&self, q: &[f32]) -> Vec<f32> {
        self.rotation.apply_alloc(q)
    }

    /// Encode one vector. Same scale-correction contract as
    /// [`TurboQuant4::encode`](TurboQuant4::encode).
    ///
    /// # Panics
    ///
    /// Panics if `v.len() != dim`.
    pub fn encode(&self, v: &[f32]) -> (Vec<u8>, f32) {
        let dim = self.dim();
        assert_eq!(v.len(), dim);
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let inv = if norm > 1e-10 { 1.0 / norm } else { 0.0 };
        let mut unit = vec![0.0f32; dim];
        for (u, &x) in unit.iter_mut().zip(v.iter()) {
            *u = x * inv;
        }
        let rotated = self.rotation.apply_alloc(&unit);
        let mut code = vec![0u8; dim / 4];
        let mut inner = 0.0f32;
        for (i, &r) in rotated.iter().enumerate() {
            let bucket = bucketize(r, &self.boundaries);
            inner += r * self.centroids[bucket];
            // Pack four 2-bit codes per byte: i%4 chooses the 2-bit slot.
            let byte = i / 4;
            let shift = (i % 4) * 2;
            code[byte] |= (bucket as u8) << shift;
        }
        let inner = inner.max(1e-10);
        let scale = norm / inner;
        (code, scale)
    }

    /// Approximate `<v, q>` for a stored `(code, scale)` pair and rotated
    /// query. Mirrors [`TurboQuant4::approx_inner`].
    ///
    /// # Panics
    ///
    /// Panics if lengths do not match.
    #[must_use]
    pub fn approx_inner(&self, code: &[u8], scale: f32, q_rot: &[f32]) -> f32 {
        let dim = self.dim();
        assert_eq!(code.len(), dim / 4);
        assert_eq!(q_rot.len(), dim);
        let mut acc = 0.0f32;
        for byte_idx in 0..dim / 4 {
            let byte = code[byte_idx];
            let c0 = (byte & 0x03) as usize;
            let c1 = ((byte >> 2) & 0x03) as usize;
            let c2 = ((byte >> 4) & 0x03) as usize;
            let c3 = ((byte >> 6) & 0x03) as usize;
            acc += q_rot[4 * byte_idx] * self.centroids[c0];
            acc += q_rot[4 * byte_idx + 1] * self.centroids[c1];
            acc += q_rot[4 * byte_idx + 2] * self.centroids[c2];
            acc += q_rot[4 * byte_idx + 3] * self.centroids[c3];
        }
        scale * acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIM: usize = 256;

    fn rand_vec(rng: &mut StdRng, dim: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(dim);
        let mut i = 0;
        while i + 1 < dim {
            let (z0, z1) = box_muller(rng);
            out.push(z0);
            out.push(z1);
            i += 2;
        }
        if i < dim {
            let (z0, _) = box_muller(rng);
            out.push(z0);
        }
        out
    }

    fn normalize(v: &[f32]) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let inv = 1.0 / n;
        v.iter().map(|x| x * inv).collect()
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn rotation_preserves_norm() {
        let rot = Rotation::new(DIM, 42);
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..16 {
            let x = rand_vec(&mut rng, DIM);
            let y = rot.apply_alloc(&x);
            let nx = x.iter().map(|x| x * x).sum::<f32>().sqrt();
            let ny = y.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (nx - ny).abs() / nx < 1e-3,
                "norm changed: {} -> {}",
                nx,
                ny
            );
        }
    }

    #[test]
    fn rotation_preserves_inner_product() {
        let rot = Rotation::new(DIM, 11);
        let mut rng = StdRng::seed_from_u64(99);
        for _ in 0..16 {
            let x = rand_vec(&mut rng, DIM);
            let y = rand_vec(&mut rng, DIM);
            let qx = rot.apply_alloc(&x);
            let qy = rot.apply_alloc(&y);
            let ip_before = dot(&x, &y);
            let ip_after = dot(&qx, &qy);
            let denom = ip_before.abs().max(1.0);
            assert!(
                (ip_before - ip_after).abs() / denom < 1e-3,
                "IP changed: {} -> {}",
                ip_before,
                ip_after
            );
        }
    }

    #[test]
    fn fast_rotation_preserves_norm() {
        // Same harness as `rotation_preserves_norm`; FWHT-block must be
        // orthogonal to the same tolerance as the QR rotation.
        let rot = FastRotation::new(DIM, 42);
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..16 {
            let x = rand_vec(&mut rng, DIM);
            let y = rot.apply_alloc(&x);
            let nx = x.iter().map(|x| x * x).sum::<f32>().sqrt();
            let ny = y.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (nx - ny).abs() / nx < 1e-3,
                "FastRotation norm changed: {} -> {}",
                nx,
                ny
            );
        }
    }

    #[test]
    fn fast_rotation_preserves_inner_product() {
        let rot = FastRotation::new(DIM, 11);
        let mut rng = StdRng::seed_from_u64(99);
        for _ in 0..16 {
            let x = rand_vec(&mut rng, DIM);
            let y = rand_vec(&mut rng, DIM);
            let qx = rot.apply_alloc(&x);
            let qy = rot.apply_alloc(&y);
            let ip_before: f32 = x.iter().zip(&y).map(|(a, b)| a * b).sum();
            let ip_after: f32 = qx.iter().zip(&qy).map(|(a, b)| a * b).sum();
            let denom = ip_before.abs().max(1.0);
            assert!(
                (ip_before - ip_after).abs() / denom < 1e-3,
                "FastRotation IP changed: {} -> {}",
                ip_before,
                ip_after
            );
        }
    }

    #[test]
    fn fast_rotation_seed_deterministic() {
        let a = FastRotation::new(DIM, 12345);
        let b = FastRotation::new(DIM, 12345);
        let mut rng = StdRng::seed_from_u64(1);
        let x = rand_vec(&mut rng, DIM);
        assert_eq!(a.apply_alloc(&x), b.apply_alloc(&x));
    }

    #[test]
    fn fast_rotation_works_on_non_power_of_two_dim() {
        // MiniLM dim 384 = 3 * 128. Three independent FWHT blocks of size
        // 128 each.
        let dim = 384;
        let rot = FastRotation::new(dim, 7);
        let mut rng = StdRng::seed_from_u64(3);
        let x = rand_vec(&mut rng, dim);
        let y = rot.apply_alloc(&x);
        let nx = x.iter().map(|x| x * x).sum::<f32>().sqrt();
        let ny = y.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((nx - ny).abs() / nx < 1e-3);
    }

    /// Reference scalar ADC implementation to compare SWAR helpers against.
    /// Mirrors the original per-coord loop the SWAR helpers replace.
    fn scalar_adc(code: &[u8], centroids: &[f32], q_rot: &[f32], dim: usize, bits: u8) -> f32 {
        let codes_per_byte = (8 / bits) as usize;
        let mask = (1u8 << bits) - 1;
        let mut acc = 0.0f32;
        for i in 0..dim {
            let byte = code[i / codes_per_byte];
            let shift = ((i % codes_per_byte) * bits as usize) as u8;
            let bucket = ((byte >> shift) & mask) as usize;
            acc += q_rot[i] * centroids[bucket];
        }
        acc
    }

    fn random_unit_vec(rng: &mut StdRng, dim: usize) -> Vec<f32> {
        let v = (0..dim).map(|_| box_muller(rng).0).collect::<Vec<f32>>();
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    }

    #[test]
    fn tq4_swar_matches_scalar() {
        // For every (dim, query) pair, both ADC paths must agree to f32
        // precision.
        let dim = 256;
        let tq = TurboQuant4::new(dim, 7);
        let mut rng = StdRng::seed_from_u64(11);
        for _ in 0..8 {
            let v = random_unit_vec(&mut rng, dim);
            let (code, _) = tq.encode(&v);
            let q = random_unit_vec(&mut rng, dim);
            let q_rot = tq.rotate_query(&q);
            // Read the scaled centroids from a fresh TurboQuant4 build
            // (avoid touching private state).
            let mut centroids = [0.0f32; 16];
            for (c, &n) in centroids.iter_mut().zip(LM_CENTROIDS_N01.iter()) {
                *c = n / (dim as f32).sqrt();
            }
            let swar = tq4_adc_swar(&code, &centroids, &q_rot, dim);
            let scalar = scalar_adc(&code, &centroids, &q_rot, dim, 4);
            assert!(
                (swar - scalar).abs() < 1e-4,
                "tq4 swar {} != scalar {}",
                swar,
                scalar
            );
        }
    }

    #[test]
    fn tq2_swar_matches_scalar() {
        let dim = 256;
        let tq = TurboQuant2::new(dim, 7);
        let mut rng = StdRng::seed_from_u64(13);
        for _ in 0..8 {
            let v = random_unit_vec(&mut rng, dim);
            let (code, _) = tq.encode(&v);
            let q_rot = tq.rotate_query(&random_unit_vec(&mut rng, dim));
            let mut centroids = [0.0f32; 4];
            for (c, &n) in centroids.iter_mut().zip(LM2_CENTROIDS_N01.iter()) {
                *c = n / (dim as f32).sqrt();
            }
            let swar = tq2_adc_swar(&code, &centroids, &q_rot, dim);
            let scalar = scalar_adc(&code, &centroids, &q_rot, dim, 2);
            assert!(
                (swar - scalar).abs() < 1e-4,
                "tq2 swar {} != scalar {}",
                swar,
                scalar
            );
        }
    }

    #[test]
    fn tq1_swar_matches_scalar() {
        let dim = 256;
        let tq = TurboQuant1::new(dim, 7);
        let mut rng = StdRng::seed_from_u64(17);
        for _ in 0..8 {
            let v = random_unit_vec(&mut rng, dim);
            let (code, _) = tq.encode(&v);
            let q_rot = tq.rotate_query(&random_unit_vec(&mut rng, dim));
            let mut centroids = [0.0f32; 2];
            for (c, &n) in centroids.iter_mut().zip(LM1_CENTROIDS_N01.iter()) {
                *c = n / (dim as f32).sqrt();
            }
            let q_sum: f32 = q_rot.iter().sum();
            let swar = tq1_adc_swar(&code, &centroids, &q_rot, dim, q_sum);
            let scalar = scalar_adc(&code, &centroids, &q_rot, dim, 1);
            // 1-bit path uses the algebraic reduction (q_masked + q_sum
            // factoring); tolerance slightly looser due to FMA reordering.
            assert!(
                (swar - scalar).abs() < 1e-3,
                "tq1 swar {} != scalar {}",
                swar,
                scalar
            );
        }
    }

    #[test]
    fn rotation_seed_deterministic() {
        let a = Rotation::new(DIM, 12345);
        let b = Rotation::new(DIM, 12345);
        let mut rng = StdRng::seed_from_u64(1);
        let x = rand_vec(&mut rng, DIM);
        let qa = a.apply_alloc(&x);
        let qb = b.apply_alloc(&x);
        for (u, v) in qa.iter().zip(qb.iter()) {
            assert_eq!(u, v);
        }
    }

    #[test]
    fn encode_decode_round_trip_close() {
        let tq = TurboQuant4::new(DIM, 7);
        let mut rng = StdRng::seed_from_u64(2);
        let mut total_err = 0.0f32;
        let n = 64;
        for _ in 0..n {
            let v = normalize(&rand_vec(&mut rng, DIM));
            let q_rot = tq.rotate_query(&v);
            let (code, scale) = tq.encode(&v);
            let approx = tq.approx_inner(&code, scale, &q_rot);
            // approx ~= <v, v> = 1 for a unit vector
            total_err += (1.0 - approx).abs();
        }
        let mean_err = total_err / n as f32;
        assert!(
            mean_err < 0.05,
            "self-IP estimate off by {} on average",
            mean_err
        );
    }

    #[test]
    fn tq1_encode_decode_round_trip_close() {
        let tq = TurboQuant1::new(DIM, 7);
        let mut rng = StdRng::seed_from_u64(2);
        let mut total_err = 0.0f32;
        let n = 64;
        for _ in 0..n {
            let v = normalize(&rand_vec(&mut rng, DIM));
            let q_rot = tq.rotate_query(&v);
            let (code, scale) = tq.encode(&v);
            let approx = tq.approx_inner(&code, scale, &q_rot);
            total_err += (1.0 - approx).abs();
        }
        let mean_err = total_err / n as f32;
        // 1-bit is the coarsest; per-vector scale should still bring the
        // self-IP close to 1.
        assert!(
            mean_err < 0.3,
            "1-bit self-IP estimate off by {} on average",
            mean_err
        );
    }

    #[test]
    fn tq2_encode_decode_round_trip_close() {
        let tq = TurboQuant2::new(DIM, 7);
        let mut rng = StdRng::seed_from_u64(2);
        let mut total_err = 0.0f32;
        let n = 64;
        for _ in 0..n {
            let v = normalize(&rand_vec(&mut rng, DIM));
            let q_rot = tq.rotate_query(&v);
            let (code, scale) = tq.encode(&v);
            let approx = tq.approx_inner(&code, scale, &q_rot);
            total_err += (1.0 - approx).abs();
        }
        let mean_err = total_err / n as f32;
        // 2-bit is coarser than 4-bit; loosen the threshold accordingly.
        assert!(
            mean_err < 0.15,
            "2-bit self-IP estimate off by {} on average",
            mean_err
        );
    }

    #[test]
    fn flat_scan_top1_recovery() {
        // 256 random unit vectors, 32 queries: assert top-1 is the true match
        // most of the time. Sanity check that the pipeline lines up the
        // approximate score with the true cosine.
        let tq = TurboQuant4::new(DIM, 99);
        let mut rng = StdRng::seed_from_u64(33);
        let n = 256;
        let q = 32;
        let corpus: Vec<Vec<f32>> = (0..n)
            .map(|_| normalize(&rand_vec(&mut rng, DIM)))
            .collect();
        let encoded: Vec<(Vec<u8>, f32)> = corpus.iter().map(|v| tq.encode(v)).collect();
        let queries: Vec<Vec<f32>> = (0..q)
            .map(|_| normalize(&rand_vec(&mut rng, DIM)))
            .collect();

        let mut hits = 0;
        for query in &queries {
            // True top-1
            let true_idx = corpus
                .iter()
                .enumerate()
                .max_by(|a, b| dot(query, a.1).total_cmp(&dot(query, b.1)))
                .unwrap()
                .0;
            // Approx top-1
            let q_rot = tq.rotate_query(query);
            let approx_idx = encoded
                .iter()
                .enumerate()
                .max_by(|a, b| {
                    tq.approx_inner(&a.1.0, a.1.1, &q_rot)
                        .total_cmp(&tq.approx_inner(&b.1.0, b.1.1, &q_rot))
                })
                .unwrap()
                .0;
            if true_idx == approx_idx {
                hits += 1;
            }
        }
        // Random vectors are nearly orthogonal in high-dim, top-1 is hard;
        // assert we recover at least half of the true tops.
        assert!(hits >= q / 2, "top-1 recovery {} / {} too low", hits, q);
    }
}
