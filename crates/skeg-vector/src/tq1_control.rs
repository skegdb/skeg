//! Online self-tuning controller for the 1-bit TurboQuant proxy.
//!
//! The two tq1 proxies (symmetric popcount, asymmetric masked-sum) read the
//! *same* stored codes, so switching between them is free at query time. This
//! controller decides which one to use, learning from observed recall instead of
//! trusting the static dimension heuristic alone.
//!
//! Signal source: on a small sampled fraction of queries the caller runs a
//! *shadow A/B* - both proxies' walks, reranked against the union with exact
//! cosine - and reports each proxy's recall. That reuses the f32 rerank the
//! query already does, so the only extra cost is one shadow walk on ~1/[`SHADOW_EVERY`]
//! queries. No build pass, no separate reads: RW/streaming-safe, and it adapts
//! to per-index distribution and drift (which the dim rule cannot).
//!
//! Policy: keep the faster arm (popcount) while its recall stays within
//! `tolerance` of the asymmetric arm; otherwise fall back to asymmetric. The
//! dim-based [`crate::Tq1ProxyMode`] seeds the controller as the cold-start
//! prior; data only overrides it after `warmup` samples, and a flip needs
//! `hysteresis` consecutive agreeing decisions to avoid flapping.

use crate::Tq1ProxyMode;

/// Suggested shadow-sampling cadence: run the A/B probe on 1 in this many
/// queries (~2%). The controller does not sample itself - the caller uses this
/// to decide when to call [`Tq1ProxyController::record_shadow`].
pub const SHADOW_EVERY: u64 = 50;

/// Online controller choosing the tq1 proxy from observed recall.
#[derive(Debug, Clone)]
pub struct Tq1ProxyController {
    /// Currently active arm - what [`mode`](Self::mode) returns for the walk.
    mode: Tq1ProxyMode,
    /// Allowed recall deficit for popcount vs asymmetric before falling back.
    tolerance: f32,
    /// EMA weight for new shadow samples.
    alpha: f32,
    /// Exponential moving averages of each arm's measured recall.
    ema_fast: Option<f32>,
    ema_asym: Option<f32>,
    /// Shadow samples seen; data-driven flips are gated until `>= warmup`.
    samples: u32,
    warmup: u32,
    /// Consecutive samples whose decision disagrees with the active mode; a flip
    /// happens only once this reaches `hysteresis`.
    pending: u8,
    hysteresis: u8,
}

impl Tq1ProxyController {
    /// Create a controller seeded with the dim-based `prior` (cold start).
    #[must_use]
    pub fn new(prior: Tq1ProxyMode) -> Self {
        Self {
            mode: prior,
            tolerance: 0.01,
            alpha: 0.2,
            ema_fast: None,
            ema_asym: None,
            samples: 0,
            warmup: 20,
            pending: 0,
            hysteresis: 3,
        }
    }

    /// Override the default policy knobs. `tolerance` is the recall deficit
    /// popcount may have vs asymmetric; `warmup` shadow samples must accrue
    /// before data overrides the prior; `hysteresis` consecutive agreeing
    /// decisions are required to flip.
    #[must_use]
    pub fn with_policy(mut self, tolerance: f32, warmup: u32, hysteresis: u8) -> Self {
        self.tolerance = tolerance;
        self.warmup = warmup;
        self.hysteresis = hysteresis.max(1);
        self
    }

    /// The proxy the walk should use right now.
    #[must_use]
    pub fn mode(&self) -> Tq1ProxyMode {
        self.mode
    }

    /// Whether query number `query_index` should run the shadow A/B probe.
    /// Deterministic (no RNG): every [`SHADOW_EVERY`]-th query.
    #[must_use]
    pub fn should_shadow(&self, query_index: u64) -> bool {
        SHADOW_EVERY != 0 && query_index % SHADOW_EVERY == 0
    }

    /// Feed one shadow A/B measurement: the recall each proxy achieved on the
    /// same query (from the caller's union rerank). Updates the estimates and
    /// may flip the active mode.
    pub fn record_shadow(&mut self, recall_fast: f32, recall_asym: f32) {
        self.samples = self.samples.saturating_add(1);
        self.ema_fast = Some(ema(self.ema_fast, recall_fast, self.alpha));
        self.ema_asym = Some(ema(self.ema_asym, recall_asym, self.alpha));

        // Not enough evidence yet: stay on the prior.
        if self.samples < self.warmup {
            return;
        }
        let (Some(p), Some(a)) = (self.ema_fast, self.ema_asym) else {
            return;
        };
        // Prefer the fast arm (hybrid) while it stays within tolerance of asym.
        let target = if p >= a - self.tolerance {
            Tq1ProxyMode::Hybrid
        } else {
            Tq1ProxyMode::Asymmetric
        };
        if target == self.mode {
            self.pending = 0;
            return;
        }
        self.pending += 1;
        if self.pending >= self.hysteresis {
            self.mode = target;
            self.pending = 0;
        }
    }

    /// Current recall estimates `(popcount, asymmetric)`, if sampled yet.
    #[must_use]
    pub fn estimates(&self) -> (Option<f32>, Option<f32>) {
        (self.ema_fast, self.ema_asym)
    }

    /// Serialize the learned state to a fixed 30-byte record so it survives an
    /// index reopen (persist alongside the vindex metadata). The controller
    /// keeps converging from where it left off instead of cold-starting.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; Self::BYTES] {
        let mut b = [0u8; Self::BYTES];
        b[0] = 1; // format version
        b[1] = mode_tag(self.mode);
        b[2..6].copy_from_slice(&self.tolerance.to_le_bytes());
        b[6..10].copy_from_slice(&self.alpha.to_le_bytes());
        b[10] = u8::from(self.ema_fast.is_some());
        b[11..15].copy_from_slice(&self.ema_fast.unwrap_or(0.0).to_le_bytes());
        b[15] = u8::from(self.ema_asym.is_some());
        b[16..20].copy_from_slice(&self.ema_asym.unwrap_or(0.0).to_le_bytes());
        b[20..24].copy_from_slice(&self.samples.to_le_bytes());
        b[24..28].copy_from_slice(&self.warmup.to_le_bytes());
        b[28] = self.pending;
        b[29] = self.hysteresis;
        b
    }

    /// Reconstruct from [`to_bytes`](Self::to_bytes). Returns `None` on a bad
    /// length or unknown version, so the caller can fall back to a fresh
    /// controller seeded from the dim prior.
    #[must_use]
    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        if b.len() != Self::BYTES || b[0] != 1 {
            return None;
        }
        let f32_at = |i: usize| f32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
        let u32_at = |i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
        Some(Self {
            mode: mode_from_tag(b[1])?,
            tolerance: f32_at(2),
            alpha: f32_at(6),
            ema_fast: (b[10] == 1).then(|| f32_at(11)),
            ema_asym: (b[15] == 1).then(|| f32_at(16)),
            samples: u32_at(20),
            warmup: u32_at(24),
            pending: b[28],
            hysteresis: b[29].max(1),
        })
    }

    /// Serialized size of the controller state.
    pub const BYTES: usize = 30;
}

fn mode_tag(m: Tq1ProxyMode) -> u8 {
    match m {
        Tq1ProxyMode::Asymmetric => 0,
        Tq1ProxyMode::Popcount => 1,
        Tq1ProxyMode::Hybrid => 2,
    }
}

fn mode_from_tag(t: u8) -> Option<Tq1ProxyMode> {
    match t {
        0 => Some(Tq1ProxyMode::Asymmetric),
        1 => Some(Tq1ProxyMode::Popcount),
        2 => Some(Tq1ProxyMode::Hybrid),
        _ => None,
    }
}

fn ema(prev: Option<f32>, x: f32, alpha: f32) -> f32 {
    match prev {
        None => x,
        Some(p) => alpha * x + (1.0 - alpha) * p,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feed `n` samples of a fixed (pop, asym) recall pair.
    fn feed(c: &mut Tq1ProxyController, n: usize, pop: f32, asym: f32) {
        for _ in 0..n {
            c.record_shadow(pop, asym);
        }
    }

    #[test]
    fn cold_start_returns_prior() {
        assert_eq!(
            Tq1ProxyController::new(Tq1ProxyMode::Asymmetric).mode(),
            Tq1ProxyMode::Asymmetric
        );
        assert_eq!(
            Tq1ProxyController::new(Tq1ProxyMode::Popcount).mode(),
            Tq1ProxyMode::Popcount
        );
    }

    #[test]
    fn converges_to_hybrid_when_recall_matches() {
        // Prior says asymmetric, but data shows the fast arm (hybrid) is
        // basically as good -> switch to it.
        let mut c = Tq1ProxyController::new(Tq1ProxyMode::Asymmetric);
        feed(&mut c, 40, 0.995, 0.999); // deficit 0.004 < tolerance 0.01
        assert_eq!(c.mode(), Tq1ProxyMode::Hybrid);
    }

    #[test]
    fn stays_asymmetric_when_popcount_deficit_large() {
        // mnist-like: popcount clearly worse -> keep asymmetric.
        let mut c = Tq1ProxyController::new(Tq1ProxyMode::Asymmetric);
        feed(&mut c, 40, 0.946, 1.000); // deficit 0.054 > tolerance
        assert_eq!(c.mode(), Tq1ProxyMode::Asymmetric);
    }

    #[test]
    fn falls_back_from_bad_popcount_prior() {
        // Prior (dim) said popcount, but this index's distribution makes it bad.
        let mut c = Tq1ProxyController::new(Tq1ProxyMode::Popcount);
        feed(&mut c, 40, 0.90, 0.99);
        assert_eq!(c.mode(), Tq1ProxyMode::Asymmetric);
    }

    #[test]
    fn warmup_holds_prior() {
        // Before warmup samples accrue, the prior stands even against contrary data.
        let mut c = Tq1ProxyController::new(Tq1ProxyMode::Asymmetric);
        feed(&mut c, 5, 0.999, 0.999); // would prefer popcount, but < warmup
        assert_eq!(c.mode(), Tq1ProxyMode::Asymmetric);
    }

    #[test]
    fn hysteresis_ignores_single_outlier() {
        // Settle on popcount, then one bad sample must not flip immediately.
        let mut c = Tq1ProxyController::new(Tq1ProxyMode::Asymmetric);
        feed(&mut c, 40, 0.999, 0.999);
        assert_eq!(c.mode(), Tq1ProxyMode::Hybrid);
        c.record_shadow(0.5, 0.99); // one outlier
        assert_eq!(
            c.mode(),
            Tq1ProxyMode::Hybrid,
            "single outlier must not flip"
        );
    }

    #[test]
    fn persist_round_trips_learned_state() {
        let mut c = Tq1ProxyController::new(Tq1ProxyMode::Asymmetric).with_policy(0.02, 10, 2);
        feed(&mut c, 40, 0.995, 0.999); // converged to popcount, EMAs populated
        let restored = Tq1ProxyController::from_bytes(&c.to_bytes()).unwrap();
        assert_eq!(restored.mode(), c.mode());
        assert_eq!(restored.estimates(), c.estimates());
        // A fresh controller with the same policy keeps converging identically.
        assert_eq!(restored.mode(), Tq1ProxyMode::Hybrid);
        // Bad input -> None (caller cold-starts from the dim prior).
        assert!(Tq1ProxyController::from_bytes(&[0u8; 4]).is_none());
        assert!(Tq1ProxyController::from_bytes(&[9u8; Tq1ProxyController::BYTES]).is_none());
    }

    #[test]
    fn shadow_cadence_is_periodic() {
        let c = Tq1ProxyController::new(Tq1ProxyMode::Popcount);
        assert!(c.should_shadow(0));
        assert!(c.should_shadow(SHADOW_EVERY));
        assert!(!c.should_shadow(1));
        assert!(!c.should_shadow(SHADOW_EVERY - 1));
    }
}
