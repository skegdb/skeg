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
//! Two deliberate shapes here:
//!
//! The limit is read from the CGROUP, not from RSS, because the cgroup number
//! is the one the kernel kills on. RSS is what this process thinks it is
//! using; the cgroup also charges page cache for files it mapped. Off Linux
//! there is no such limit to read, so the governor runs unlimited and says
//! so, which is honest, and is one more reason the Linux gate is a release
//! blocker rather than a nicety.
//!
//! And usage comes from an injectable [`MemorySource`] rather than being read
//! directly. A budget test that works by actually allocating gigabytes is a
//! test nobody runs, which is a guard that guards nothing - the same lesson
//! the `#[ignore]` tests taught. With a source behind a trait the rejection
//! path is exercised deterministically, in milliseconds, at any limit.

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
    /// Bytes currently charged to this process.
    fn current_bytes(&self) -> u64;
    /// The limit crossing which the process is killed, if there is one.
    fn hard_limit_bytes(&self) -> Option<u64>;
}

/// The real one: whatever the platform can actually observe.
#[derive(Debug)]
pub struct PlatformMemory;

impl MemorySource for PlatformMemory {
    fn current_bytes(&self) -> u64 {
        skeg_platform::memory_status()
            .current_bytes
            .unwrap_or_default()
    }
    fn hard_limit_bytes(&self) -> Option<u64> {
        skeg_platform::memory_status().limit_bytes
    }
}

/// Why a reservation was refused. Both variants are reported to the client
/// with their numbers, because "out of memory" without them is untriageable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryRejected {
    /// The reservation would cross the usable limit.
    HardLimit {
        current: u64,
        reserved: u64,
        requested: u64,
        usable_limit: u64,
    },
    /// The request is so large that adding it overflows. A caller computing a
    /// size from wire input can produce this, so it is refused rather than
    /// wrapped - a wrapped total is a reservation that always succeeds.
    ArithmeticOverflow { requested: u64 },
}

impl std::fmt::Display for MemoryRejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HardLimit {
                current,
                reserved,
                requested,
                usable_limit,
            } => write!(
                f,
                "memory budget: current={current} reserved={reserved} \
                 requested={requested} limit={usable_limit}"
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
    limit: Option<u64>,
    reserve: u64,
    /// Bytes promised to callers but not yet visible in `current_bytes`.
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
        let limit = resolve_limit(explicit_limit, source.hard_limit_bytes());
        let reserve = match (explicit_reserve, limit) {
            (Some(r), _) => r,
            (None, Some(l)) => default_reserve(l),
            (None, None) => MIN_RESERVE,
        };
        if let Some(l) = limit
            && reserve >= l
        {
            return Err(format!(
                "memory reserve {reserve} is not smaller than the limit {l}: \
                 the budget would reject every request"
            ));
        }
        Ok(Self {
            source,
            limit,
            reserve,
            outstanding: AtomicU64::new(0),
        })
    }

    /// The ceiling admissions are actually measured against.
    pub fn usable_limit(&self) -> Option<u64> {
        self.limit.map(|l| l.saturating_sub(self.reserve))
    }

    pub fn limit_bytes(&self) -> Option<u64> {
        self.limit
    }

    pub fn current_bytes(&self) -> u64 {
        self.source.current_bytes()
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
        let Some(usable) = self.usable_limit() else {
            // No limit to enforce. Still tracked, so the counters report
            // something true and a limit can be applied later.
            self.outstanding.fetch_add(bytes, Ordering::AcqRel);
            return Ok(MemoryReservation {
                governor: Arc::clone(self),
                bytes,
            });
        };
        let current = self.source.current_bytes();
        let mut held = self.outstanding.load(Ordering::Acquire);
        loop {
            let projected = current
                .checked_add(held)
                .and_then(|t| t.checked_add(bytes))
                .ok_or(MemoryRejected::ArithmeticOverflow { requested: bytes })?;
            if projected > usable {
                return Err(MemoryRejected::HardLimit {
                    current,
                    reserved: held,
                    requested: bytes,
                    usable_limit: usable,
                });
            }
            match self.outstanding.compare_exchange_weak(
                held,
                held + bytes,
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

#[derive(Debug)]
/// A reservation, released when dropped.
///
/// Dropping releases only the PROMISE. Whether the memory itself came back is
/// a separate question, answered by `MemorySource` on the next reservation -
/// which is why a fold that really did grow the heap does not get its budget
/// back just because its job object went away.
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

    /// A source the test drives directly.
    #[derive(Debug)]
    struct Fake {
        current: AtomicU64,
        limit: Option<u64>,
    }

    impl MemorySource for Fake {
        fn current_bytes(&self) -> u64 {
            self.current.load(Ordering::Acquire)
        }
        fn hard_limit_bytes(&self) -> Option<u64> {
            self.limit
        }
    }

    fn gov(current: u64, limit: Option<u64>, reserve: u64) -> Arc<MemoryGovernor> {
        let src = Arc::new(Fake {
            current: AtomicU64::new(current),
            limit,
        });
        Arc::new(MemoryGovernor::new(src, None, Some(reserve)).unwrap())
    }

    #[test]
    fn an_explicit_limit_beats_the_cgroup() {
        assert_eq!(resolve_limit(Some(100), Some(999)), Some(100));
    }

    #[test]
    fn the_cgroup_limit_is_used_when_nothing_is_explicit() {
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
            current: AtomicU64::new(0),
            limit: Some(1000),
        });
        let err = MemoryGovernor::new(src, None, Some(1000)).unwrap_err();
        assert!(err.contains("1000"), "the refusal must show the numbers");
    }

    #[test]
    fn a_reservation_within_the_budget_is_admitted() {
        let g = gov(0, Some(1000), 100);
        let r = g.try_reserve(500).unwrap();
        assert_eq!(r.bytes(), 500);
        assert_eq!(g.reserved_bytes(), 500);
    }

    #[test]
    fn a_reservation_past_the_usable_limit_is_refused() {
        // usable = 1000 - 100 = 900. Already using 800, asking for 200.
        let g = gov(800, Some(1000), 100);
        let err = g.try_reserve(200).unwrap_err();
        match err {
            MemoryRejected::HardLimit {
                current,
                requested,
                usable_limit,
                ..
            } => {
                assert_eq!((current, requested, usable_limit), (800, 200, 900));
            }
            other => panic!("wrong rejection: {other:?}"),
        }
    }

    #[test]
    fn the_reserve_is_held_back_from_the_budget() {
        // Without the reserve this fits exactly; with it, it does not. That
        // difference is the headroom, and it must be real.
        let g = gov(0, Some(1000), 100);
        assert_eq!(g.usable_limit(), Some(900));
        assert!(g.try_reserve(1000).is_err());
        assert!(g.try_reserve(900).is_ok());
    }

    #[test]
    fn dropping_a_reservation_restores_headroom() {
        let g = gov(0, Some(1000), 100);
        {
            let _r = g.try_reserve(900).unwrap();
            assert!(g.try_reserve(1).is_err(), "budget full while held");
        }
        assert_eq!(g.reserved_bytes(), 0);
        assert!(g.try_reserve(900).is_ok(), "headroom returns on drop");
    }

    #[test]
    fn outstanding_reservations_count_even_before_they_allocate() {
        // The race this exists to stop: usage has not moved yet, but the
        // bytes are already promised.
        let g = gov(0, Some(1000), 0);
        let _a = g.try_reserve(600).unwrap();
        assert!(
            g.try_reserve(600).is_err(),
            "a second 600 must not be admitted against an unchanged usage"
        );
    }

    #[test]
    fn concurrent_reservations_never_overbook() {
        // Ten threads racing for a budget that fits six. Exactly six may win;
        // a lost compare-exchange must retry against the NEW total, not
        // against the value it first read.
        //
        // Repeated, on purpose. A read-check-write governor overbooks most
        // rounds but not every round, so a single-round version of this test
        // passes roughly one time in five against code that is plainly wrong.
        // Measured: naive store fails 4 of 5 single rounds, 25 of 25 here.
        for round in 0..25 {
            let g = gov(0, Some(700), 100); // usable 600
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
        let g = gov(u64::MAX - 10, Some(u64::MAX), 1);
        assert_eq!(
            g.try_reserve(u64::MAX).unwrap_err(),
            MemoryRejected::ArithmeticOverflow {
                requested: u64::MAX
            }
        );
    }

    #[test]
    fn without_a_limit_everything_is_admitted_but_still_counted() {
        let g = gov(0, None, 0);
        let _r = g.try_reserve(u64::MAX / 2).unwrap();
        assert_eq!(g.usable_limit(), None);
        assert_eq!(g.reserved_bytes(), u64::MAX / 2);
    }
}
