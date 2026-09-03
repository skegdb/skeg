//! What the network is allowed to pin, counted against the same budget as
//! everything else.
//!
//! The memory governor admits the delta - the rows a write puts on the heap -
//! and nothing else. Everything the network holds arrives BEFORE any command
//! exists to be admitted: a connection's read buffer grows to whatever the
//! peer dribbles into it, the parser copies it again, and only then does a
//! `VSET` reach a governor that has already lost the argument. One RESP3
//! connection could pin a frame's worth of buffer plus its parse copy, and the
//! connection semaphore multiplied that by a thousand.
//!
//! So ingress gets a budget, and it is the SAME budget. `IngressBudget`
//! reserves through [`crate::memory::MemoryGovernor`], so a byte a socket
//! holds and a byte a delta holds compete for one headroom figure and appear
//! in one `outstanding` total. What is new here is a class CAP - a fraction of
//! that headroom past which ingress will not go, however much room the delta
//! has left - and a per-connection allowance inside it, so one greedy peer
//! cannot spend the class on itself.
//!
//! # The three charges, and why they do not overlap
//!
//! 1. **Ingress** (here): the bytes a connection's read buffer holds, charged
//!    as CAPACITY rather than length - the allocation is what the process
//!    pays for, and `BytesMut::reserve` rounds up - times [`PARSE_FACTOR`],
//!    which covers the one copy the parser makes out of that buffer. Held for
//!    as long as the buffer is that big, released when it shrinks or the
//!    connection closes.
//! 2. **Delta** (`Vindex::reserve_memory`): the rows a committed write puts in
//!    the in-memory delta, charged per megabyte as the delta grows. A row is
//!    charged there only after it has stopped being wire bytes here: the
//!    buffer it arrived in is drained before the command runs.
//! 3. **`MAX_VMSET_BYTES`** (`resp3_handler`): a constant early reject on one
//!    command's vector bytes, with NO reservation. It is a ceiling on what a
//!    single frame may ask for, applied upstream of both budgets, and
//!    reserving for it would charge the same bytes a second time.
//!
//! # One budget, one counter, and the class share
//!
//! Every byte charged here is reserved through the governor's single
//! `outstanding`. The [`IngressBudget::held_bytes`] figure is not a second
//! budget: it is this class's share of that one total, which has to be
//! tracked separately only because the cap is per class. Both move together -
//! a `ConnectionBudget` releases the governor reservation and the class share
//! in the same drop.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::memory::{MemoryGovernor, MemoryRejected};

/// Bytes charged per byte of buffer capacity: one for the buffer, one for the
/// copy the parser makes out of it. The parse copy is real and simultaneous -
/// `parse_bulk` copies the bulk out of the decoder buffer while the buffer
/// still holds it - so charging capacity alone would under-count by half at
/// exactly the moment the peak happens.
pub const PARSE_FACTOR: u64 = 2;

/// The idle buffer a connection is given for existing: 4 KiB of capacity, so
/// [`PARSE_FACTOR`] times that in charge.
pub const FLOOR_BYTES: u64 = 4096 * PARSE_FACTOR;

/// The granularity growth is charged in: the 256 KiB chunk the RESP3 read loop
/// reserves once a frame is mid-flight, times [`PARSE_FACTOR`].
pub const CHUNK_BYTES: u64 = 256 * 1024 * PARSE_FACTOR;

/// Share of the governor's usable headroom that ingress may hold, as a
/// percentage.
pub const DEFAULT_FRACTION: u64 = 25;

/// The cap applied when no ceiling exists anywhere.
pub const UNLIMITED_DEFAULT_CAP: u64 = 1 << 30;

/// How long a connection whose growth was refused waits before the frame is
/// refused outright.
pub const DEFAULT_STALL: Duration = Duration::from_millis(500);

/// What ingress may hold in total, and how that figure was arrived at.
///
/// Three states, like the governor's own `Budget`, and for the same reason:
/// "no ceiling" and "a ceiling nobody can read" are opposites, and one number
/// cannot report both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IngressCap {
    /// A ceiling applies and this many bytes are the class's share of it.
    Room(u64),
    /// No ceiling applies anywhere, so [`UNLIMITED_DEFAULT_CAP`] stands in.
    Default(u64),
    /// A ceiling applies and its headroom could not be read.
    FloorOnly,
}

impl IngressCap {
    /// The number of bytes on offer, floor-only counting as none.
    #[must_use]
    pub fn bytes(self) -> u64 {
        match self {
            IngressCap::Room(b) | IngressCap::Default(b) => b,
            IngressCap::FloorOnly => 0,
        }
    }

    /// The label this state carries in `SKEG.STATS`.
    #[must_use]
    pub fn state_name(self) -> &'static str {
        match self {
            IngressCap::Room(_) => "known",
            IngressCap::Default(_) => "default",
            IngressCap::FloorOnly => "unreadable",
        }
    }
}

/// Why ingress refused, with the numbers, because a refusal without them is
/// untriageable and a client cannot tell a retry from a mistake.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IngressRejected {
    /// The class is full: other connections hold the budget. Retryable.
    ClassFull { held: u64, requested: u64, cap: u64 },
    /// This connection asked for more than one connection may ever hold. Not
    /// retryable: the same frame will be refused again on the next attempt.
    OverConnectionAllowance { requested: u64, allowance: u64 },
    /// The budget cannot be established, so nothing may grow past the floor.
    Unreadable { requested: u64 },
    /// The process-wide governor refused: ingress is not the only claimant on
    /// the headroom, and the delta had already taken it.
    Governor(MemoryRejected),
}

impl IngressRejected {
    /// Should the client try the same frame again?
    #[must_use]
    pub fn is_retryable(self) -> bool {
        match self {
            IngressRejected::ClassFull { .. } | IngressRejected::Unreadable { .. } => true,
            IngressRejected::OverConnectionAllowance { .. } => false,
            IngressRejected::Governor(e) => match e {
                MemoryRejected::NoHeadroom { .. } => true,
                MemoryRejected::Unknown | MemoryRejected::ArithmeticOverflow { .. } => false,
            },
        }
    }

    /// The error line a client sees, code first.
    #[must_use]
    pub fn wire_message(self) -> String {
        let code = if self.is_retryable() {
            "BACKPRESSURE"
        } else {
            "ERR"
        };
        format!("{code} {self}")
    }
}

impl std::fmt::Display for IngressRejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ClassFull {
                held,
                requested,
                cap,
            } => write!(
                f,
                "ingress budget: held={held} requested={requested} cap={cap}"
            ),
            Self::OverConnectionAllowance {
                requested,
                allowance,
            } => write!(
                f,
                "ingress budget: this connection may hold at most {allowance} \
                 bytes and the frame needs {requested}"
            ),
            Self::Unreadable { requested } => write!(
                f,
                "ingress budget: this process is under a limit whose headroom \
                 cannot be read, so a connection holds only its floor; \
                 requested={requested}"
            ),
            Self::Governor(e) => write!(f, "ingress budget: {e}"),
        }
    }
}

impl std::error::Error for IngressRejected {}

/// The aggregate ingress budget: a class cap over the process-wide governor.
///
/// Scaffolding. The cap is not computed and nothing is charged yet - the
/// enforcement lands with the tests below.
#[derive(Debug)]
pub struct IngressBudget {
    governor: Arc<MemoryGovernor>,
    cap: IngressCap,
    per_conn_max: u64,
    stall: Duration,
    held: AtomicU64,
}

impl IngressBudget {
    /// Build the budget from the settings an operator supplied.
    #[must_use]
    pub fn new(
        governor: Arc<MemoryGovernor>,
        _fraction_percent: Option<u64>,
        _explicit_bytes: Option<u64>,
        stall: Option<Duration>,
        _per_connection_ceiling: u64,
    ) -> Self {
        Self {
            governor,
            cap: IngressCap::Default(0),
            per_conn_max: 0,
            stall: stall.unwrap_or(DEFAULT_STALL),
            held: AtomicU64::new(0),
        }
    }

    /// [`IngressBudget::new`] reading the operator's environment.
    #[must_use]
    pub fn from_env(governor: Arc<MemoryGovernor>, per_connection_ceiling: u64) -> Self {
        let num = |k: &str| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok())
        };
        Self::new(
            governor,
            num("SKEG_INGRESS_FRACTION"),
            num("SKEG_INGRESS_BUDGET_BYTES"),
            num("SKEG_INGRESS_STALL_MS").map(Duration::from_millis),
            per_connection_ceiling,
        )
    }

    /// The governor this budget reserves through. One process, one total.
    #[must_use]
    pub fn governor(&self) -> &Arc<MemoryGovernor> {
        &self.governor
    }

    /// What ingress may hold in total, and how that was decided.
    #[must_use]
    pub fn cap(&self) -> IngressCap {
        self.cap
    }

    /// The most one connection may hold.
    #[must_use]
    pub fn per_connection_max(&self) -> u64 {
        self.per_conn_max
    }

    /// How long a connection whose growth was refused waits.
    #[must_use]
    pub fn stall(&self) -> Duration {
        self.stall
    }

    /// Ingress's current share of the governor's outstanding total.
    #[must_use]
    pub fn held_bytes(&self) -> u64 {
        self.held.load(Ordering::Acquire)
    }

    /// Take the floor for a newly accepted connection, or say why not.
    ///
    /// # Errors
    /// Propagates the class, allowance and governor refusals.
    pub fn try_accept(self: &Arc<Self>) -> Result<ConnectionBudget, IngressRejected> {
        Ok(ConnectionBudget {
            budget: Arc::clone(self),
            held: 0,
        })
    }
}

/// One connection's charge, released when the connection task ends.
#[derive(Debug)]
pub struct ConnectionBudget {
    budget: Arc<IngressBudget>,
    held: u64,
}

impl ConnectionBudget {
    /// Bytes this connection currently holds.
    #[must_use]
    pub fn held_bytes(&self) -> u64 {
        self.held
    }

    /// The most this connection may hold.
    #[must_use]
    pub fn allowance(&self) -> u64 {
        self.budget.per_conn_max
    }

    /// The budget this connection draws on.
    #[must_use]
    pub fn budget(&self) -> &Arc<IngressBudget> {
        &self.budget
    }

    /// Charge for a buffer about to hold `buffer_capacity` bytes.
    ///
    /// # Errors
    /// Propagates the class, allowance and governor refusals.
    pub fn grow_to(&mut self, _buffer_capacity: usize) -> Result<(), IngressRejected> {
        Ok(())
    }

    /// Give back what a drained buffer no longer needs, down to the floor.
    pub fn shrink_to(&mut self, _buffer_capacity: usize) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{Headroom, MemorySource};

    #[derive(Debug)]
    struct Fixed(Headroom);

    impl MemorySource for Fixed {
        fn headroom(&self) -> Headroom {
            self.0
        }
    }

    fn governor(h: Headroom) -> Arc<MemoryGovernor> {
        Arc::new(MemoryGovernor::new(Arc::new(Fixed(h)), None, Some(0)).expect("a governor"))
    }

    /// A budget whose cap is exactly `cap` bytes, over headroom four times
    /// that so the governor is never the thing refusing.
    fn budget_with_cap(cap: u64) -> Arc<IngressBudget> {
        Arc::new(IngressBudget::new(
            governor(Headroom::Known(cap * 4)),
            None,
            Some(cap),
            None,
            u64::from(u32::MAX),
        ))
    }

    #[test]
    #[ignore = "opens in commit 4 (server: IngressBudget)"]
    fn a_drained_buffer_gives_its_growth_back() {
        // `BytesMut` keeps its allocation, so a charge that only grew would
        // make every connection's peak permanent and the class cap a
        // high-water mark of the whole uptime.
        let budget = budget_with_cap(64 * CHUNK_BYTES);
        let mut conn = budget.try_accept().expect("the floor");
        assert_eq!(conn.held_bytes(), FLOOR_BYTES);
        conn.grow_to(1024 * 1024).expect("growth");
        let grown = conn.held_bytes();
        assert!(grown > FLOOR_BYTES, "the growth was not charged: {grown}");
        assert_eq!(budget.held_bytes(), grown);
        assert_eq!(budget.governor().reserved_bytes(), grown);

        conn.shrink_to(4096);
        assert_eq!(
            conn.held_bytes(),
            FLOOR_BYTES,
            "the growth was not returned"
        );
        assert_eq!(budget.held_bytes(), FLOOR_BYTES);
        assert_eq!(budget.governor().reserved_bytes(), FLOOR_BYTES);
    }

    #[test]
    #[ignore = "opens in commit 4 (server: IngressBudget)"]
    fn one_greedy_connection_cannot_starve_the_others() {
        // The per-connection allowance is the whole reason the class cap is
        // not just a bigger place to be starved in: without it the first
        // connection to ask takes everything and the rest get their floor and
        // nothing more.
        let cap = 64 * CHUNK_BYTES;
        let budget = budget_with_cap(cap);
        assert_eq!(budget.per_connection_max(), cap / 4);

        let mut greedy = budget.try_accept().expect("the floor");
        // Ask for the whole class in one go: refused by the allowance, not by
        // the class, and the connection keeps what it had.
        let err = greedy
            .grow_to(usize::try_from(cap).expect("cap fits"))
            .expect_err("one connection must not take the class");
        assert!(
            matches!(err, IngressRejected::OverConnectionAllowance { .. }),
            "wrong refusal: {err:?}"
        );
        assert_eq!(greedy.held_bytes(), FLOOR_BYTES);
        // What it MAY take, it takes.
        greedy
            .grow_to(usize::try_from(cap / 4 / PARSE_FACTOR).expect("quarter fits"))
            .expect("its own quarter");
        assert_eq!(greedy.held_bytes(), cap / 4);

        // And three more connections still get theirs.
        let mut others = Vec::new();
        for i in 0..3 {
            let mut c = budget.try_accept().expect("the floor");
            c.grow_to(usize::try_from(cap / 4 / PARSE_FACTOR).expect("quarter fits"))
                .unwrap_or_else(|e| panic!("connection {i} starved: {e}"));
            others.push(c);
        }
        assert!(budget.held_bytes() <= cap, "the class cap was overspent");
    }

    #[test]
    #[ignore = "opens in commit 4 (server: IngressBudget)"]
    fn an_accepted_connection_always_gets_its_floor_or_is_refused_by_name() {
        // Accept is the one place the budget must never await: a listener that
        // parks on memory is an outage. Every connection either has its floor
        // or is told, with the numbers, that it does not.
        let budget = budget_with_cap(4 * FLOOR_BYTES);
        let mut held = Vec::new();
        for i in 0..4 {
            let c = budget
                .try_accept()
                .unwrap_or_else(|e| panic!("connection {i} refused its floor: {e}"));
            assert_eq!(c.held_bytes(), FLOOR_BYTES);
            held.push(c);
        }
        let err = budget
            .try_accept()
            .expect_err("a full class must refuse, not overspend");
        let message = err.wire_message();
        assert!(
            message.starts_with("BACKPRESSURE "),
            "a full class is retryable, so the code must say so: {message}"
        );
        assert!(
            message.contains(&(4 * FLOOR_BYTES).to_string()),
            "the refusal must carry the numbers: {message}"
        );
        drop(held);
        assert_eq!(budget.held_bytes(), 0);
        budget.try_accept().expect("room again once they close");
    }

    #[test]
    #[ignore = "opens in commit 4 (server: IngressBudget)"]
    fn the_ingress_and_delta_reservations_sum_under_one_budget() {
        // The point of the whole module: ingress is not a second budget beside
        // the governor's, it is a class inside it. A byte a socket holds and a
        // byte the delta holds have to be visible in ONE outstanding total, or
        // the sum of the two can still take the process out.
        //
        // `governor.try_reserve` is exactly what `Vindex::reserve_memory`
        // calls, so this is the delta's own path, not a stand-in for it.
        let usable = 16 * CHUNK_BYTES;
        let gov = governor(Headroom::Known(usable));
        let budget = Arc::new(IngressBudget::new(
            Arc::clone(&gov),
            None,
            Some(4 * CHUNK_BYTES),
            None,
            u64::from(u32::MAX),
        ));
        let mut conn = budget.try_accept().expect("the floor");
        conn.grow_to(usize::try_from(CHUNK_BYTES / PARSE_FACTOR).expect("chunk fits"))
            .expect("growth");
        let ingress = conn.held_bytes();
        assert_eq!(gov.reserved_bytes(), ingress, "ingress is not in the total");

        let delta = gov
            .try_reserve(1 << 20)
            .expect("the delta's own reservation");
        assert_eq!(
            gov.reserved_bytes(),
            ingress + delta.bytes(),
            "the two classes must sum under one total"
        );

        // And the sum is what refuses: the delta takes the rest, and the
        // socket's next chunk is refused by the governor rather than by the
        // class, which still has room.
        let rest = gov
            .try_reserve(usable - gov.reserved_bytes())
            .expect("the rest of the headroom");
        let err = conn
            .grow_to(usize::try_from(2 * CHUNK_BYTES / PARSE_FACTOR).expect("two chunks fit"))
            .expect_err("no headroom left for ingress to grow into");
        assert!(
            matches!(err, IngressRejected::Governor(_)),
            "the governor is what ran out, not the class: {err:?}"
        );
        drop((delta, rest));
    }

    #[test]
    #[ignore = "opens in commit 4 (server: IngressBudget)"]
    fn an_unreadable_budget_serves_small_frames_and_refuses_growth() {
        // A ceiling applies and its headroom cannot be read. Refusing every
        // connection would turn an accounting fault into an outage; admitting
        // growth would switch the budget off exactly when it cannot see. The
        // answer is the floor, and only the floor.
        let budget = Arc::new(IngressBudget::new(
            governor(Headroom::Unknown),
            None,
            None,
            None,
            u64::from(u32::MAX),
        ));
        assert_eq!(budget.cap(), IngressCap::FloorOnly);
        assert_eq!(budget.cap().state_name(), "unreadable");

        let mut conn = budget.try_accept().expect("a small frame is still served");
        assert_eq!(conn.held_bytes(), FLOOR_BYTES);
        let err = conn
            .grow_to(1024 * 1024)
            .expect_err("growth under an unreadable budget must be refused");
        assert!(
            matches!(err, IngressRejected::Unreadable { .. }),
            "wrong refusal: {err:?}"
        );
        assert!(
            err.wire_message().starts_with("BACKPRESSURE "),
            "an unreadable budget may become readable: {}",
            err.wire_message()
        );
        assert_eq!(conn.held_bytes(), FLOOR_BYTES, "the floor stays");
    }
}
