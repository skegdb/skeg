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

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// 64 MiB: the floor for headroom, and the smallest reserve worth having.
const MIN_RESERVE: u64 = 64 * 1024 * 1024;
/// A tenth of the limit, when that is larger than the floor.
const RESERVE_DIVISOR: u64 = 10;

/// Where the governor learns what is being used, and what the ceiling is.
///
/// Behind a trait so the rejection path can be driven deterministically.
pub trait MemorySource: std::fmt::Debug + Send + Sync + 'static {
    /// Bytes this process may still allocate before the FIRST cgroup on its
    /// chain is saturated, or `None` when that is not known.
    ///
    /// Headroom, not a limit. Admission is a question about what is left, and
    /// `limit - current` on one cgroup does not answer it: every cgroup from
    /// the process's own to the root applies at once, and the one that runs
    /// out first is the one with the least headroom - which is not the one
    /// with the smallest limit. A leaf at 256 MiB holding 10 MiB has 246 MiB
    /// free, but inside a parent at 512 MiB already holding 500 MiB of
    /// siblings' memory, twelve more megabytes end the process.
    ///
    /// `None` means unknown, and unknown is not zero: no limit anywhere, or
    /// one whose usage could not be read. Substituting either extreme is how
    /// a governor becomes unsafe in one direction or useless in the other.
    fn available_bytes(&self) -> Option<u64>;
}

/// The real one: whatever the platform can actually observe.
#[derive(Debug)]
pub struct PlatformMemory;

impl MemorySource for PlatformMemory {
    fn available_bytes(&self) -> Option<u64> {
        skeg_platform::memory_status().available_bytes
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
            Self::ArithmeticOverflow { requested } => {
                write!(f, "memory budget: requested={requested} overflows")
            }
        }
    }
}

impl std::error::Error for MemoryRejected {}

/// Decide the hard limit from the two possible sources.
///
/// An explicit setting wins over the cgroup deliberately: an operator running
/// several processes in one container needs to divide that container's memory
/// between them, and the cgroup number is the whole container's.
///
/// Kept pure so its cases are tested without touching process environment,
/// which tests running in parallel share.
fn resolve_limit(explicit: Option<u64>, cgroup: Option<u64>) -> Option<u64> {
    match explicit {
        Some(v) if v > 0 => Some(v),
        _ => cgroup.filter(|v| *v > 0),
    }
}

/// Headroom held back from the budget: the allocations too small or too hot
/// to be worth reserving individually, plus what the allocator has not
/// returned to the OS yet.
fn default_reserve(limit: u64) -> u64 {
    MIN_RESERVE.max(limit / RESERVE_DIVISOR)
}

/// Admission control over a single process-wide budget.
#[derive(Debug)]
pub struct MemoryGovernor {
    source: Arc<dyn MemorySource>,
    /// An operator's override, if any. NOT a cached cgroup reading.
    explicit: Option<u64>,
    reserve: u64,
    /// Bytes promised to callers but not yet charged to the cgroup, so not
    /// yet reflected in the headroom the source reports.
    /// Without this, N concurrent reservations all read the same pre-
    /// allocation usage and all succeed - the classic overbooking race.
    outstanding: AtomicU64,
}

impl MemoryGovernor {
    /// Build from an explicit limit and a source.
    ///
    /// # Errors
    /// Refuses when the reserve is not smaller than the limit: a budget whose
    /// headroom exceeds it has no usable space at all, and would reject every
    /// request while looking configured.
    pub fn new(
        source: Arc<dyn MemorySource>,
        explicit_limit: Option<u64>,
        explicit_reserve: Option<u64>,
    ) -> Result<Self, String> {
        // Only an EXPLICIT override is stored. The cgroup's figure must not
        // be cached here: headroom moves with every allocation this process
        // and its cgroup siblings make, and a governor holding the number it
        // saw at startup keeps admitting against memory a sibling has since
        // taken. It is read afresh on every attempt instead.
        let explicit = explicit_limit.filter(|v| *v > 0);
        // The reserve is sized once, from whatever is known now - it is a
        // configuration choice, not a live measurement.
        let known_now = resolve_limit(explicit_limit, source.available_bytes());
        let reserve = match (explicit_reserve, known_now) {
            (Some(r), _) => r,
            (None, Some(l)) => default_reserve(l),
            (None, None) => MIN_RESERVE,
        };
        if let Some(l) = known_now
            && reserve >= l
        {
            return Err(format!(
                "memory reserve {reserve} is not smaller than the limit {l}: \
                 the budget would reject every request"
            ));
        }
        Ok(Self {
            source,
            explicit,
            reserve,
            outstanding: AtomicU64::new(0),
        })
    }

    /// The ceiling admissions are actually measured against.
    /// Headroom the governor will actually hand out: what the tightest cgroup
    /// has left, less the reserve. Re-read each time, because the number moves
    /// with every allocation this process and its cgroup siblings make.
    pub fn usable(&self) -> Option<u64> {
        self.effective_limit()
            .map(|l| l.saturating_sub(self.reserve))
    }

    /// The headroom figure in force: an explicit override, else the live one.
    fn effective_limit(&self) -> Option<u64> {
        self.explicit.or_else(|| self.source.available_bytes())
    }

    pub fn reserved_bytes(&self) -> u64 {
        self.outstanding.load(Ordering::Acquire)
    }

    /// Reserve `bytes`, or say why not.
    ///
    /// The compare-exchange loop is the whole point: two callers that both
    /// read usage before either allocates would both be admitted, and the sum
    /// of their allocations is what kills the process. Only one of them wins
    /// the exchange; the other retries against the new total.
    pub fn try_reserve(self: &Arc<Self>, bytes: u64) -> Result<MemoryReservation, MemoryRejected> {
        let Some(usable) = self.usable() else {
            // No headroom figure to enforce against. Still tracked, so the
            // counters report something true and a limit can apply later.
            self.outstanding.fetch_add(bytes, Ordering::AcqRel);
            return Ok(MemoryReservation {
                governor: Arc::clone(self),
                bytes,
            });
        };
        // `usable` is HEADROOM: it already has current usage subtracted, on
        // the tightest cgroup of the chain. What has to fit inside it is what
        // is promised but not yet allocated, plus this request.
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
    use super::*;

    /// A source the test drives directly. It reports HEADROOM, which is what
    /// admission is a question about.
    #[derive(Debug)]
    struct Fake {
        available: Option<u64>,
    }

    impl MemorySource for Fake {
        fn available_bytes(&self) -> Option<u64> {
            self.available
        }
    }

    fn gov(available: Option<u64>, reserve: u64) -> Arc<MemoryGovernor> {
        let src = Arc::new(Fake { available });
        Arc::new(MemoryGovernor::new(src, None, Some(reserve)).unwrap())
    }

    #[test]
    fn an_explicit_limit_beats_the_cgroup() {
        assert_eq!(resolve_limit(Some(100), Some(999)), Some(100));
    }

    #[test]
    fn the_cgroup_headroom_is_used_when_nothing_is_explicit() {
        assert_eq!(resolve_limit(None, Some(999)), Some(999));
    }

    #[test]
    fn a_zero_limit_is_not_a_limit() {
        // Zero means "unset" from an empty env var, never "no memory".
        assert_eq!(resolve_limit(Some(0), Some(999)), Some(999));
        assert_eq!(resolve_limit(Some(0), None), None);
    }

    #[test]
    fn no_limit_anywhere_is_unlimited() {
        assert_eq!(resolve_limit(None, None), None);
    }

    #[test]
    fn the_reserve_floor_applies_to_small_limits() {
        assert_eq!(default_reserve(1024), MIN_RESERVE);
        assert_eq!(default_reserve(10 * 1024 * 1024 * 1024), 1024 * 1024 * 1024);
    }

    #[test]
    fn a_reserve_that_swallows_the_limit_refuses_to_start() {
        let src = Arc::new(Fake {
            available: Some(1000),
        });
        let err = MemoryGovernor::new(src, None, Some(1000)).unwrap_err();
        assert!(err.contains("1000"), "the refusal must show the numbers");
    }

    #[test]
    fn a_reservation_within_the_headroom_is_admitted() {
        let g = gov(Some(1000), 100);
        let r = g.try_reserve(500).unwrap();
        assert_eq!(r.bytes(), 500);
        assert_eq!(g.reserved_bytes(), 500);
    }

    #[test]
    fn a_reservation_past_the_headroom_is_refused() {
        // usable = 1000 - 100 = 900.
        let g = gov(Some(1000), 100);
        let err = g.try_reserve(901).unwrap_err();
        match err {
            MemoryRejected::NoHeadroom {
                requested, usable, ..
            } => assert_eq!((requested, usable), (901, 900)),
            other => panic!("wrong rejection: {other:?}"),
        }
    }

    #[test]
    fn shrinking_headroom_tightens_admission_without_a_restart() {
        // The number moves with every allocation this process AND its cgroup
        // siblings make, so it is re-read on each attempt rather than cached.
        // A governor that latched the figure at startup would keep admitting
        // against memory a sibling has since taken.
        #[derive(Debug)]
        struct Shrinking(std::sync::atomic::AtomicU64);
        impl MemorySource for Shrinking {
            fn available_bytes(&self) -> Option<u64> {
                Some(self.0.load(Ordering::Acquire))
            }
        }
        let src = Arc::new(Shrinking(AtomicU64::new(1000)));
        let g = Arc::new(MemoryGovernor::new(src.clone(), None, Some(0)).unwrap());
        // HELD, not dropped: the first version of this test let the
        // reservation fall out of scope immediately, so nothing was
        // outstanding and admitting one more byte was correct. The test was
        // wrong, not the governor.
        let _held = g.try_reserve(800).unwrap();
        src.0.store(100, Ordering::Release);
        assert!(
            g.try_reserve(1).is_err(),
            "800 promised against 100 of headroom must refuse"
        );
    }

    #[test]
    fn the_reserve_is_held_back_from_the_headroom() {
        let g = gov(Some(1000), 100);
        assert_eq!(g.usable(), Some(900));
        assert!(g.try_reserve(1000).is_err());
        assert!(g.try_reserve(900).is_ok());
    }

    #[test]
    fn dropping_a_reservation_restores_headroom() {
        let g = gov(Some(1000), 100);
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
        let g = gov(Some(1000), 0);
        let _a = g.try_reserve(600).unwrap();
        assert!(
            g.try_reserve(600).is_err(),
            "a second 600 must not be admitted against unchanged headroom"
        );
    }

    #[test]
    fn concurrent_reservations_never_overbook() {
        // Ten threads racing for headroom that fits six. Exactly six may win;
        // a lost compare-exchange must retry against the NEW total, not the
        // value it first read.
        //
        // Repeated, on purpose. A read-check-write governor overbooks most
        // rounds but not every round, so a single-round version of this test
        // passes roughly one time in five against code that is plainly wrong.
        // Measured: naive store fails 4 of 5 single rounds, 25 of 25 here.
        for round in 0..25 {
            let g = gov(Some(700), 100); // usable 600
            let admitted = Arc::new(AtomicU64::new(0));
            std::thread::scope(|s| {
                for _ in 0..10 {
                    let g = Arc::clone(&g);
                    let admitted = Arc::clone(&admitted);
                    s.spawn(move || {
                        if let Ok(r) = g.try_reserve(100) {
                            admitted.fetch_add(1, Ordering::AcqRel);
                            std::thread::yield_now();
                            // Held for the whole round: releasing early would
                            // let a seventh in legitimately and hide a real
                            // overbooking behind a plausible count.
                            std::mem::forget(r);
                        }
                    });
                }
            });
            assert_eq!(
                admitted.load(Ordering::Acquire),
                6,
                "round {round}: exactly six 100-byte reservations fit in 600"
            );
            assert_eq!(g.reserved_bytes(), 600, "round {round}");
        }
    }

    #[test]
    fn an_overflowing_request_is_refused_not_wrapped() {
        // A wrapped total is a reservation that always succeeds, which is
        // worse than no governor at all.
        let g = gov(Some(u64::MAX), 1);
        let _held = g.try_reserve(1).unwrap();
        assert_eq!(
            g.try_reserve(u64::MAX).unwrap_err(),
            MemoryRejected::ArithmeticOverflow {
                requested: u64::MAX
            }
        );
    }

    #[test]
    fn unknown_headroom_admits_but_still_counts() {
        // `None` is "not known", not "no memory". Rejecting everything on an
        // unreadable cgroup - or on macOS, where there is nothing to read -
        // would make the server unusable rather than safe.
        let g = gov(None, 0);
        let _r = g.try_reserve(u64::MAX / 2).unwrap();
        assert_eq!(g.usable(), None);
        assert_eq!(g.reserved_bytes(), u64::MAX / 2);
    }
}
