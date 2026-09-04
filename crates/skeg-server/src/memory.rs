//! A memory budget the process enforces, instead of one the kernel enforces
//! for it.
//!
//! Today skeg learns its limit by being killed. The delta grows unbounded
//! with the server's inline auto-flush off, payload caches and routers and
//! quantised tiers are all charged to the same process, and a fold allocates
//! proportionally to the live set on top of everything already resident. The
//! first symptom of crossing the line is SIGKILL: no error to the client, no
//! entry in the log, no clean shutdown, and a WAL to replay on the way back
//! up.
//!
//! What replaces it is ordinary admission control. Before an allocation that
//! is worth counting - a write, a fold, a run merge - the caller reserves the
//! bytes it is about to need. If the reservation would carry the process past
//! its usable limit, the caller is told NO and the client sees an error. A
//! refused write is a bad afternoon; a killed process mid-fold is a bad week.
//!
//! Three deliberate shapes here:
//!
//! The question is HEADROOM, not a limit. Admission asks what is left, and
//! `limit - current` on one cgroup does not answer it: every cgroup from the
//! process's own up to the root applies at once, so the one that runs out
//! first is the one with the least headroom - which is not the one with the
//! smallest limit. A leaf capped at 256 MiB holding 10 MiB looks like 246 MiB
//! of room, but inside a parent capped at 512 MiB that siblings have already
//! filled to 500 MiB, twelve more megabytes end the process.
//!
//! That figure comes from the CGROUP, not from RSS, because the cgroup number
//! is the one the kernel kills on. RSS is what this process thinks it is
//! using; the cgroup also charges page cache for files it mapped. Off Linux
//! there is nothing to read, so the governor runs unlimited and says so,
//! which is honest, and is one more reason the Linux gate is a release
//! blocker rather than a nicety.
//!
//! And it arrives through an injectable [`MemorySource`] rather than being
//! read directly. A budget test that works by really allocating gigabytes is
//! a test nobody runs, which is a guard that guards nothing - the same lesson
//! the `#[ignore]` tests taught. Behind a trait the rejection path runs
//! deterministically, in milliseconds, at any limit.

pub use skeg_platform::Headroom;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

/// 64 MiB: the floor for headroom, and the smallest reserve worth having.
const MIN_RESERVE: u64 = 64 * 1024 * 1024;
/// A tenth of the limit, when that is larger than the floor.
const RESERVE_DIVISOR: u64 = 10;

/// Where the governor learns what is being used, and what the ceiling is.
///
/// Behind a trait so the rejection path can be driven deterministically.
pub trait MemorySource: std::fmt::Debug + Send + Sync + 'static {
    /// How much room is left, and when there is no number, why.
    ///
    /// Headroom, not a limit. Admission asks what is LEFT, and
    /// `limit - current` on one cgroup does not answer it: every cgroup from
    /// the process's own to the root applies at once, and the one that runs
    /// out first is the one with the least room - not the one with the
    /// smallest limit. A leaf at 256 MiB holding 10 MiB looks like 246 MiB of
    /// room, but inside a parent at 512 MiB that siblings have filled to
    /// 500 MiB, twelve more megabytes end the process.
    ///
    /// Three states, not two. `Unlimited` and `Unknown` are opposites, and an
    /// `Option` that collapses them lets an unreadable `memory.current`
    /// silently switch the governor off - the placebo it exists to prevent,
    /// coming back through a different door.
    fn headroom(&self) -> Headroom;
}

/// The real one: whatever the platform can actually observe.
#[derive(Debug)]
pub struct PlatformMemory;

impl MemorySource for PlatformMemory {
    fn headroom(&self) -> Headroom {
        skeg_platform::memory_status().available
    }
}

/// A source read at most once per `ttl`, serving the last answer in between.
///
/// The real source walks the cgroup chain, reading `/proc/self/cgroup` and two
/// files per level, on EVERY call. Admission happens per write, so consulting
/// it directly would put several file reads on the hot path - which the plan
/// forbids in the same breath as it asks for admission control, and rightly:
/// a governor that costs more than the work it admits is not a governor.
///
/// The staleness this buys is bounded and paid for. Within one window the
/// budget can be over-admitted by whatever arrives in it, which is why the
/// governor holds a reserve back: the window is small, the reserve is not.
/// Outstanding reservations are NOT stale - they are atomic - so a burst
/// inside one window is still counted against itself.
#[derive(Debug)]
pub struct CachedMemory {
    inner: Arc<dyn MemorySource>,
    ttl: Duration,
    /// `Mutex` and not a lock-free cell: the value is two words and the
    /// contention window is a clock read. A racing pair may both refresh,
    /// which costs a duplicate read and never a wrong answer.
    last: Mutex<(Instant, Headroom)>,
}

impl CachedMemory {
    #[must_use]
    pub fn new(inner: Arc<dyn MemorySource>, ttl: Duration) -> Self {
        let now = inner.headroom();
        Self {
            inner,
            ttl,
            last: Mutex::new((Instant::now(), now)),
        }
    }
}

impl MemorySource for CachedMemory {
    fn headroom(&self) -> Headroom {
        let mut guard = self.last.lock();
        if guard.0.elapsed() >= self.ttl {
            *guard = (Instant::now(), self.inner.headroom());
        }
        guard.1
    }
}

/// Why a reservation was refused. Both variants are reported to the client
/// with their numbers, because "out of memory" without them is untriageable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryRejected {
    /// The reservation would consume more than the headroom left.
    NoHeadroom {
        /// Already promised to other callers and not yet allocated.
        reserved: u64,
        requested: u64,
        /// Headroom on the tightest cgroup, less the reserve held back.
        usable: u64,
    },
    /// The budget cannot be established: a cgroup limit applies but the room
    /// left could not be computed, and no operator override was configured.
    ///
    /// Fail-CLOSED, and deliberately. The alternative is to admit everything
    /// whenever accounting is momentarily unreadable, which turns the governor
    /// off exactly when it cannot see - and an operator who prefers to run
    /// without one can say so with `SKEG_MEMORY_LIMIT_BYTES`.
    Unknown,
    /// The request is so large that adding it overflows. A caller computing a
    /// size from wire input can produce this, so it is refused rather than
    /// wrapped - a wrapped total is a reservation that always succeeds.
    ArithmeticOverflow { requested: u64 },
}

impl std::fmt::Display for MemoryRejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoHeadroom {
                reserved,
                requested,
                usable,
            } => write!(
                f,
                "memory budget: reserved={reserved} requested={requested} \
                 usable={usable}"
            ),
            Self::Unknown => write!(
                f,
                "memory budget: this process is under a cgroup limit whose \
                 headroom cannot be read; set SKEG_MEMORY_LIMIT_BYTES to run \
                 without one"
            ),
            Self::ArithmeticOverflow { requested } => {
                write!(f, "memory budget: requested={requested} overflows")
            }
        }
    }
}

impl std::error::Error for MemoryRejected {}

/// The operator's own ceiling on headroom, when they set one.
///
/// Named for what it IS. It was `explicit_limit` and was compared against
/// headroom, which is not the same quantity: a process already holding
/// 800 MiB under a configured "limit" of 1 GiB would be granted almost
/// another gigabyte. A total is not a remainder.
///
/// The cgroup stays a ceiling regardless. Configuration can only ever make
/// the budget SMALLER: the kernel does not consult it.
fn effective_headroom(explicit: Option<u64>, cgroup: Headroom) -> Result<Option<u64>, ()> {
    match (explicit, cgroup) {
        // Configured and measured: the tighter of the two.
        (Some(e), Headroom::Known(k)) => Ok(Some(e.min(k))),
        // Configured, and nothing measured to be tightened against. This is
        // also how an operator deliberately runs without cgroup accounting.
        (Some(e), _) => Ok(Some(e)),
        (None, Headroom::Known(k)) => Ok(Some(k)),
        // Nothing caps this process anywhere: no budget to enforce.
        (None, Headroom::Unlimited) => Ok(None),
        // A limit applies but its headroom is unreadable, and nobody said what
        // to do about it. Refusing is the only answer that does not quietly
        // disable the governor at the moment it stops being able to see.
        (None, Headroom::Unknown) => Err(()),
    }
}

/// Headroom held back from the budget: the allocations too small or too hot
/// to be worth reserving individually, plus what the allocator has not
/// returned to the OS yet.
fn default_reserve(limit: u64) -> u64 {
    MIN_RESERVE.max(limit / RESERVE_DIVISOR)
}

/// How long a cgroup reading is reused.
///
/// Sized against the reserve it can overshoot, not picked round. At the
/// fastest ingest measured - about 8,500 rows/s - a hundred milliseconds
/// admits some 850 rows, which at 1024 dimensions is 3.4 MiB. The smallest
/// reserve is 64 MiB, so the worst stale window spends about five per cent of
/// the margin held back for exactly this.
pub const MEMORY_CACHE_TTL: Duration = Duration::from_millis(100);

/// Parse a byte count an operator set. Empty and unparseable are both "not
/// set": a typo must not silently become a budget of zero, which would refuse
/// every write while looking configured.
fn parse_bytes(v: Option<&str>) -> Option<u64> {
    v.map(str::trim)
        .filter(|s| !s.is_empty())
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|n| *n > 0)
}

/// What the governor has to offer, in the same three states the platform
/// reports and for the same reason: `Unlimited` and `Unreadable` are opposites,
/// and an `Option` that collapses them lets an unreadable ceiling look like no
/// ceiling. `Result<Option<u64>, ()>` said it too, in a shape nobody reads
/// twice the same way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Budget {
    /// A ceiling applies and this much is on offer, the reserve already taken
    /// out.
    Room(u64),
    /// No ceiling applies: nothing to be killed for.
    Unlimited,
    /// A ceiling applies and the room left could not be read. A refusal, not a
    /// licence.
    Unreadable,
}

/// Admission control over a single process-wide budget.
#[derive(Debug)]
pub struct MemoryGovernor {
    source: Arc<dyn MemorySource>,
    /// An operator's ceiling on headroom, if any. NOT a cached cgroup reading:
    /// headroom moves with every allocation this process and its cgroup
    /// siblings make, and a governor holding the number it saw at startup
    /// keeps admitting against memory a sibling has since taken.
    explicit: Option<u64>,
    reserve: u64,
    /// Bytes promised to callers but not yet charged to the cgroup, so not yet
    /// reflected in the headroom the source reports. Without this, N
    /// concurrent reservations all read the same figure and all succeed - the
    /// classic overbooking race.
    outstanding: AtomicU64,
}

impl MemoryGovernor {
    #[cfg(test)]
    pub(crate) fn unlimited_for_tests() -> Arc<Self> {
        #[derive(Debug)]
        struct Unlimited;
        impl MemorySource for Unlimited {
            fn headroom(&self) -> Headroom {
                Headroom::Unlimited
            }
        }
        Arc::new(Self::new(Arc::new(Unlimited), None, Some(0)).expect("unlimited governor"))
    }

    /// Build from an operator's headroom ceiling and a source.
    ///
    /// # Errors
    /// Refuses when the reserve is not smaller than the budget: one whose
    /// headroom exceeds it has no usable space at all and would reject every
    /// request while looking configured.
    pub fn new(
        source: Arc<dyn MemorySource>,
        explicit_headroom: Option<u64>,
        explicit_reserve: Option<u64>,
    ) -> Result<Self, String> {
        let explicit = explicit_headroom.filter(|v| *v > 0);
        // Sized once, from whatever is known now: the reserve is a
        // configuration choice, not a live measurement.
        let known_now = effective_headroom(explicit, source.headroom()).unwrap_or(None);
        let reserve = match (explicit_reserve, known_now) {
            (Some(r), _) => r,
            (None, Some(l)) => default_reserve(l),
            (None, None) => MIN_RESERVE,
        };
        if let Some(l) = known_now
            && reserve >= l
        {
            return Err(format!(
                "memory reserve {reserve} is not smaller than the budget {l}: \
                 it would reject every request"
            ));
        }
        Ok(Self {
            source,
            explicit,
            reserve,
            outstanding: AtomicU64::new(0),
        })
    }

    /// The one an operator gets: the platform, cached, with the overrides the
    /// refusal messages promise.
    ///
    /// `SKEG_MEMORY_LIMIT_BYTES` is that promise. It was named in two error
    /// strings and read by nobody, so the escape hatch the governor offered
    /// when it could not see did not exist.
    ///
    /// # Errors
    /// Propagates [`MemoryGovernor::new`].
    pub fn from_env() -> Result<Self, String> {
        Self::from_settings(
            std::env::var("SKEG_MEMORY_LIMIT_BYTES").ok().as_deref(),
            std::env::var("SKEG_MEMORY_RESERVE_BYTES").ok().as_deref(),
        )
    }

    /// [`MemoryGovernor::from_env`] with the values supplied, so the parsing
    /// is testable without touching process-wide state.
    ///
    /// # Errors
    /// Propagates [`MemoryGovernor::new`].
    pub fn from_settings(limit: Option<&str>, reserve: Option<&str>) -> Result<Self, String> {
        Self::new(
            Arc::new(CachedMemory::new(
                Arc::new(PlatformMemory),
                MEMORY_CACHE_TTL,
            )),
            parse_bytes(limit),
            parse_bytes(reserve),
        )
    }

    /// Headroom the governor will hand out: what is left, less the reserve.
    ///
    /// `Ok(None)` means no budget applies. `Err(())` means one does and its
    /// size is unreadable, which is a refusal, not a licence.
    fn usable(&self) -> Result<Option<u64>, ()> {
        Ok(effective_headroom(self.explicit, self.source.headroom())?
            .map(|l| l.saturating_sub(self.reserve)))
    }

    /// The margin held back, so an operator can see how much of the ceiling is
    /// deliberately not for sale.
    pub fn reserve_bytes(&self) -> u64 {
        self.reserve
    }

    /// What the governor believes right now.
    ///
    /// Reported, not just enforced. A budget nobody can read is the same
    /// problem as a governor nobody asks: the operator finds out from the
    /// refusals, or from the OOM.
    pub fn budget(&self) -> Budget {
        match self.usable() {
            Ok(Some(room)) => Budget::Room(room),
            Ok(None) => Budget::Unlimited,
            Err(()) => Budget::Unreadable,
        }
    }

    pub fn reserved_bytes(&self) -> u64 {
        self.outstanding.load(Ordering::Acquire)
    }

    /// Reserve `bytes`, or say why not.
    ///
    /// The compare-exchange loop is the whole point: two callers that both
    /// read the headroom before either allocates would both be admitted, and
    /// the SUM of their allocations is what kills the process. Only one wins
    /// the exchange; the other retries against the new total.
    pub fn try_reserve(self: &Arc<Self>, bytes: u64) -> Result<MemoryReservation, MemoryRejected> {
        let usable = match self.usable() {
            Ok(Some(u)) => u,
            // No budget applies: still tracked, so the counters report
            // something true and a limit can be applied later.
            Ok(None) => {
                self.outstanding.fetch_add(bytes, Ordering::AcqRel);
                return Ok(MemoryReservation {
                    governor: Arc::clone(self),
                    bytes,
                });
            }
            Err(()) => return Err(MemoryRejected::Unknown),
        };
        // `usable` is HEADROOM: current usage is already subtracted, on the
        // tightest cgroup of the chain. What must fit inside it is what is
        // promised but not yet allocated, plus this request.
        let mut held = self.outstanding.load(Ordering::Acquire);
        loop {
            let wanted = held
                .checked_add(bytes)
                .ok_or(MemoryRejected::ArithmeticOverflow { requested: bytes })?;
            if wanted > usable {
                return Err(MemoryRejected::NoHeadroom {
                    reserved: held,
                    requested: bytes,
                    usable,
                });
            }
            match self.outstanding.compare_exchange_weak(
                held,
                wanted,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Ok(MemoryReservation {
                        governor: Arc::clone(self),
                        bytes,
                    });
                }
                Err(actual) => held = actual,
            }
        }
    }
}

/// A reservation, released when dropped.
///
/// Dropping releases only the PROMISE. Whether the memory itself came back is
/// a separate question, answered by the cgroup's own headroom on the next
/// reservation - which is why a fold that really did grow the heap does not
/// get its budget back just because its job object went away.
/// The governor reports its own gauges, pulled at dump time.
///
/// The alternative is what `SKEG.STATS` used to do: read the governor by hand
/// in one handler and format the lines there, which is why `/metrics` - the
/// surface an operator actually scrapes - could not see the budget at all.
/// Nothing on the reserve path changes; this is a reader.
impl skeg_telemetry::GaugeSource for MemoryGovernor {
    fn sample(&self, out: &mut Vec<skeg_telemetry::GaugeSample>) {
        use skeg_telemetry::GaugeSample as S;
        // A STATE SET: all three series, every time, exactly one at 1.
        //
        // This used to emit only the state that was true. Two things go wrong
        // with that in production: an alert on "the budget went unreadable"
        // can only be written with `absent()`, and after a transition the
        // series that WAS true keeps its last value until it goes stale, so a
        // dashboard shows two states at once.
        let budget = self.budget();
        for (labels, matches) in [
            ("state=\"known\"", matches!(budget, Budget::Room(_))),
            ("state=\"unlimited\"", matches!(budget, Budget::Unlimited)),
            ("state=\"unknown\"", matches!(budget, Budget::Unreadable)),
        ] {
            out.push(S::labelled(
                "skeg_memory_budget_state",
                labels,
                u64::from(matches),
            ));
        }
        // ABSENT when nobody could read it. Publishing 0 for a headroom that
        // could not be read says "no room left", which is a different fact
        // and the one an operator would page on.
        if let Budget::Room(usable) = budget {
            out.push(S::new("skeg_memory_headroom_bytes", usable));
        }
        out.push(S::new("skeg_memory_reserved_bytes", self.reserved_bytes()));
        out.push(S::new("skeg_memory_reserve_bytes", self.reserve_bytes()));
    }
}

impl MemoryGovernor {
    /// Publish this governor's gauges on both telemetry surfaces.
    ///
    /// Called once where the `Arc` is made. Keyed, so a process that opens
    /// several shard sets - every integration test does - ends with one
    /// governor reporting rather than one series per shard set ever opened.
    pub fn register_metrics(self: &Arc<Self>) {
        skeg_telemetry::register_gauge_source(
            "skeg-server::memory",
            Arc::downgrade(self) as std::sync::Weak<dyn skeg_telemetry::GaugeSource>,
        );
    }
}

#[derive(Debug)]
pub struct MemoryReservation {
    governor: Arc<MemoryGovernor>,
    bytes: u64,
}

impl MemoryReservation {
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        self.governor
            .outstanding
            .fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    // A note paid for twice: a `MemoryReservation` released at the end of the
    // statement that produced it holds nothing. `assert!(g.try_reserve(n)
    // .is_ok())` therefore leaves the budget empty, and any assertion after it
    // about the budget being full passes while testing nothing. Bind it.
    use super::*;

    #[derive(Debug)]
    struct Fake(Headroom);

    impl MemorySource for Fake {
        fn headroom(&self) -> Headroom {
            self.0
        }
    }

    /// Counts how often the source underneath was actually asked.
    #[derive(Debug)]
    struct Counting(AtomicU64, Headroom);

    impl MemorySource for Counting {
        fn headroom(&self) -> Headroom {
            self.0.fetch_add(1, Ordering::AcqRel);
            self.1
        }
    }

    #[test]
    fn an_operator_override_that_is_not_a_number_is_not_a_budget_of_zero() {
        // A typo reaching the governor as `Some(0)` would refuse every write
        // while looking configured - the placebo in reverse.
        assert_eq!(parse_bytes(Some("1048576")), Some(1_048_576));
        assert_eq!(parse_bytes(Some("  4096  ")), Some(4096));
        for bad in ["", "   ", "banana", "-1", "1MiB", "0"] {
            assert_eq!(parse_bytes(Some(bad)), None, "for {bad:?}");
        }
        assert_eq!(parse_bytes(None), None);
    }

    #[test]
    fn the_cache_asks_the_source_once_per_window() {
        // The real source reads several files per call and admission runs per
        // write. Without this the governor would cost more than the work.
        let src = Arc::new(Counting(AtomicU64::new(0), Headroom::Known(1_000)));
        let cache = CachedMemory::new(src.clone(), Duration::from_secs(3600));
        // One read at construction, so the first answer is never a guess.
        assert_eq!(src.0.load(Ordering::Acquire), 1);
        for _ in 0..1_000 {
            assert_eq!(cache.headroom(), Headroom::Known(1_000));
        }
        assert_eq!(
            src.0.load(Ordering::Acquire),
            1,
            "a thousand admissions asked the cgroup more than once"
        );
    }

    #[test]
    fn the_cache_refreshes_after_its_window() {
        let src = Arc::new(Counting(AtomicU64::new(0), Headroom::Known(1_000)));
        let cache = CachedMemory::new(src.clone(), Duration::ZERO);
        cache.headroom();
        cache.headroom();
        assert!(
            src.0.load(Ordering::Acquire) >= 3,
            "a zero window must not freeze the answer forever"
        );
    }

    #[test]
    fn the_cache_does_not_turn_unknown_into_a_number() {
        // Unknown is a refusal. A cache that smoothed it into the last known
        // value would switch the governor off exactly when it cannot see.
        let src = Arc::new(Counting(AtomicU64::new(0), Headroom::Unknown));
        let cache = CachedMemory::new(src, Duration::from_secs(3600));
        assert_eq!(cache.headroom(), Headroom::Unknown);
    }

    fn gov(h: Headroom, reserve: u64) -> Arc<MemoryGovernor> {
        Arc::new(MemoryGovernor::new(Arc::new(Fake(h)), None, Some(reserve)).unwrap())
    }

    // ---- what the three states mean ----

    #[test]
    fn unknown_headroom_refuses_rather_than_admitting_everything() {
        // The fail-open this exists to close: a cgroup limit applies, its
        // usage is momentarily unreadable, and admitting everything would
        // switch the governor off precisely when it cannot see. `Unlimited`
        // and `Unknown` are opposites, and collapsing them into one `None`
        // let the second behave like the first.
        let g = gov(Headroom::Unknown, 0);
        assert_eq!(g.try_reserve(1).unwrap_err(), MemoryRejected::Unknown);
    }

    #[test]
    fn unlimited_admits_everything_and_still_counts_it() {
        // No cgroup caps this process, so there is no budget to enforce - a
        // different situation from not being able to read one.
        let g = gov(Headroom::Unlimited, 0);
        let _r = g.try_reserve(u64::MAX / 2).unwrap();
        assert_eq!(g.reserved_bytes(), u64::MAX / 2);
    }

    #[test]
    fn an_operator_override_makes_unknown_workable_again() {
        // Fail-closed must not mean unusable: someone who wants to run without
        // cgroup accounting says so, and gets a budget rather than a wall.
        let src = Arc::new(Fake(Headroom::Unknown));
        let g = Arc::new(MemoryGovernor::new(src, Some(1000), Some(100)).unwrap());
        // BOUND. `assert!(g.try_reserve(900).is_ok())` drops the reservation
        // at the end of the statement, so nothing stays outstanding and the
        // next line admits happily - the assertion passes while testing
        // nothing. Caught twice writing this module.
        let _held = g.try_reserve(900).unwrap();
        assert!(g.try_reserve(1).is_err());
    }

    // ---- what the override means ----

    #[test]
    fn the_override_is_a_ceiling_on_headroom_never_a_licence_above_the_cgroup() {
        // Configuration can only ever make the budget SMALLER: the kernel does
        // not consult it. An override of 10 GiB inside a cgroup with 100 bytes
        // left does not create 10 GiB.
        assert_eq!(
            effective_headroom(Some(10 << 30), Headroom::Known(100)),
            Ok(Some(100))
        );
        // And the other way round, the tighter one still wins.
        assert_eq!(
            effective_headroom(Some(100), Headroom::Known(10 << 30)),
            Ok(Some(100))
        );
    }

    #[test]
    fn without_an_override_the_cgroup_decides() {
        assert_eq!(
            effective_headroom(None, Headroom::Known(999)),
            Ok(Some(999))
        );
        assert_eq!(effective_headroom(None, Headroom::Unlimited), Ok(None));
        assert_eq!(effective_headroom(None, Headroom::Unknown), Err(()));
    }

    #[test]
    fn a_zero_override_is_not_an_override() {
        // An empty environment variable parses to nothing, not to "no memory".
        let g = gov(Headroom::Known(500), 0);
        assert!(g.try_reserve(500).is_ok());
        let src = Arc::new(Fake(Headroom::Known(500)));
        let g2 = Arc::new(MemoryGovernor::new(src, Some(0), Some(0)).unwrap());
        assert!(g2.try_reserve(500).is_ok(), "zero must not cap it at zero");
    }

    // ---- admission ----

    #[test]
    fn the_reserve_floor_applies_to_small_budgets() {
        assert_eq!(default_reserve(1024), MIN_RESERVE);
        assert_eq!(default_reserve(10 * 1024 * 1024 * 1024), 1024 * 1024 * 1024);
    }

    #[test]
    fn a_reserve_that_swallows_the_budget_refuses_to_start() {
        let src = Arc::new(Fake(Headroom::Known(1000)));
        let err = MemoryGovernor::new(src, None, Some(1000)).unwrap_err();
        assert!(err.contains("1000"), "the refusal must show the numbers");
    }

    #[test]
    fn a_reservation_within_the_headroom_is_admitted() {
        let g = gov(Headroom::Known(1000), 100);
        let held = g.try_reserve(500).unwrap();
        assert_eq!(held.bytes(), 500);
        assert_eq!(g.reserved_bytes(), 500);
    }

    #[test]
    fn a_reservation_past_the_headroom_is_refused() {
        let g = gov(Headroom::Known(1000), 100); // usable 900
        match g.try_reserve(901).unwrap_err() {
            MemoryRejected::NoHeadroom {
                requested, usable, ..
            } => assert_eq!((requested, usable), (901, 900)),
            other => panic!("wrong rejection: {other:?}"),
        }
    }

    #[test]
    fn shrinking_headroom_tightens_admission_without_a_restart() {
        // The figure moves with every allocation this process AND its cgroup
        // siblings make, so it is re-read on each attempt. A governor that
        // latched it at startup would keep admitting against memory a sibling
        // has since taken - which is what the constructor used to do, by
        // caching the cgroup reading in the same field as the override.
        #[derive(Debug)]
        struct Shrinking(AtomicU64);
        impl MemorySource for Shrinking {
            fn headroom(&self) -> Headroom {
                Headroom::Known(self.0.load(Ordering::Acquire))
            }
        }
        let src = Arc::new(Shrinking(AtomicU64::new(1000)));
        let g = Arc::new(MemoryGovernor::new(src.clone(), None, Some(0)).unwrap());
        // HELD, not dropped: an earlier version let this fall out of scope, so
        // nothing was outstanding and admitting one more byte was correct.
        let _held = g.try_reserve(800).unwrap();
        src.0.store(100, Ordering::Release);
        assert!(
            g.try_reserve(1).is_err(),
            "800 promised against 100 of headroom must refuse"
        );
    }

    #[test]
    fn dropping_a_reservation_restores_headroom() {
        let g = gov(Headroom::Known(1000), 100);
        {
            let _r = g.try_reserve(900).unwrap();
            assert!(g.try_reserve(1).is_err(), "budget full while held");
        }
        assert_eq!(g.reserved_bytes(), 0);
        assert!(g.try_reserve(900).is_ok(), "headroom returns on drop");
    }

    #[test]
    fn outstanding_reservations_count_even_before_they_allocate() {
        // The race this exists to stop: the cgroup has not been charged yet,
        // but the bytes are already promised.
        let g = gov(Headroom::Known(1000), 0);
        let _a = g.try_reserve(600).unwrap();
        assert!(
            g.try_reserve(600).is_err(),
            "a second 600 must not be admitted against unchanged headroom"
        );
    }

    #[test]
    fn concurrent_reservations_never_overbook() {
        // Ten threads racing for headroom that fits six. Exactly six may win;
        // a lost compare-exchange must retry against the NEW total.
        //
        // Repeated on purpose: a read-check-write governor overbooks most
        // rounds but not every round, so a single-round version passes about
        // one time in five against code that is plainly wrong. Measured: naive
        // store fails 4 of 5 single rounds, 25 of 25 here.
        for round in 0..25 {
            let g = gov(Headroom::Known(700), 100); // usable 600
            let admitted = Arc::new(AtomicU64::new(0));
            std::thread::scope(|s| {
                for _ in 0..10 {
                    let g = Arc::clone(&g);
                    let admitted = Arc::clone(&admitted);
                    s.spawn(move || {
                        if let Ok(r) = g.try_reserve(100) {
                            admitted.fetch_add(1, Ordering::AcqRel);
                            std::thread::yield_now();
                            std::mem::forget(r); // held for the whole round
                        }
                    });
                }
            });
            assert_eq!(admitted.load(Ordering::Acquire), 6, "round {round}");
            assert_eq!(g.reserved_bytes(), 600, "round {round}");
        }
    }

    #[test]
    fn an_overflowing_request_is_refused_not_wrapped() {
        // A wrapped total is a reservation that always succeeds, which is
        // worse than no governor at all.
        let g = gov(Headroom::Known(u64::MAX), 1);
        let _held = g.try_reserve(1).unwrap();
        assert_eq!(
            g.try_reserve(u64::MAX).unwrap_err(),
            MemoryRejected::ArithmeticOverflow {
                requested: u64::MAX
            }
        );
    }

    // ── P0.5: the governor reports its own gauges ───────────────────────────

    /// The samples one governor reports, isolated from every other source in
    /// the process by filtering on the names this object owns.
    fn gauges_of(g: &Arc<MemoryGovernor>) -> Vec<skeg_telemetry::GaugeSample> {
        let mut out = Vec::new();
        skeg_telemetry::GaugeSource::sample(g.as_ref(), &mut out);
        out
    }

    fn value_of(samples: &[skeg_telemetry::GaugeSample], name: &str, labels: &str) -> Option<u64> {
        samples
            .iter()
            .find(|s| s.name == name && s.labels == labels)
            .map(|s| s.value)
    }

    #[test]
    fn the_budget_state_is_a_set_of_three_series_with_exactly_one_at_one() {
        for (headroom, live) in [
            (Headroom::Known(1000), "known"),
            (Headroom::Unlimited, "unlimited"),
            (Headroom::Unknown, "unknown"),
        ] {
            let g = gov(headroom, 0);
            let samples = gauges_of(&g);
            let states: Vec<(&str, u64)> = ["known", "unlimited", "unknown"]
                .iter()
                .map(|s| {
                    (
                        *s,
                        value_of(
                            &samples,
                            "skeg_memory_budget_state",
                            match *s {
                                "known" => "state=\"known\"",
                                "unlimited" => "state=\"unlimited\"",
                                _ => "state=\"unknown\"",
                            },
                        )
                        .unwrap_or_else(|| panic!("{s} is absent for {headroom:?}")),
                    )
                })
                .collect();
            assert_eq!(
                states.iter().filter(|(_, v)| *v == 1).count(),
                1,
                "{headroom:?}: exactly one state is true, got {states:?}"
            );
            assert_eq!(
                states.iter().find(|(_, v)| *v == 1).map(|(s, _)| *s),
                Some(live),
                "{headroom:?} is state {live}"
            );
        }
    }

    #[test]
    fn headroom_is_absent_when_nobody_could_read_it() {
        let known = gauges_of(&gov(Headroom::Known(1000), 0));
        assert_eq!(
            value_of(&known, "skeg_memory_headroom_bytes", ""),
            Some(1000),
            "a readable headroom is a number"
        );
        for headroom in [Headroom::Unlimited, Headroom::Unknown] {
            let samples = gauges_of(&gov(headroom, 0));
            assert!(
                value_of(&samples, "skeg_memory_headroom_bytes", "").is_none(),
                "{headroom:?}: publishing 0 for a headroom nobody could read \
                 reads as 'no room left', which is a different fact"
            );
        }
    }
}
