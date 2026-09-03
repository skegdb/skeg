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
use std::time::{Duration, Instant};

use crate::memory::{Budget, MemoryGovernor, MemoryRejected, MemoryReservation};

/// Bytes charged per byte of buffer capacity: one for the buffer, one for the
/// copy the parser makes out of it. The parse copy is real and simultaneous -
/// `parse_bulk` copies the bulk out of the decoder buffer while the buffer
/// still holds it - so charging capacity alone would under-count by half at
/// exactly the moment the peak happens.
pub const PARSE_FACTOR: u64 = 2;

/// The idle buffer a connection is given for existing: 4 KiB of capacity, so
/// [`PARSE_FACTOR`] times that in charge. Small enough that a thousand idle
/// connections cost megabytes rather than gigabytes, large enough for any
/// session command to arrive in one read.
pub const FLOOR_BYTES: u64 = 4096 * PARSE_FACTOR;

/// The granularity growth is charged in: the 256 KiB chunk the RESP3 read loop
/// reserves once a frame is mid-flight, times [`PARSE_FACTOR`]. Charging in
/// chunks rather than per byte keeps the reservation count per connection in
/// the low hundreds at the ceiling instead of one per read.
pub const CHUNK_BYTES: u64 = 256 * 1024 * PARSE_FACTOR;

/// Share of the governor's usable headroom that ingress may hold, as a
/// percentage. A quarter: the delta, the folds and the caches share the rest,
/// and a class that could take all of it would just move the OOM.
pub const DEFAULT_FRACTION: u64 = 25;

/// The cap applied when no ceiling exists anywhere.
///
/// A static figure on purpose. `Budget::Unlimited` is the normal state off
/// Linux, where there is no cgroup to read, and a budget that switched itself
/// off there would mean the enforcement path never runs on a developer's
/// machine and every test of it would have to fake a limit. One gigabyte is
/// far above any legitimate ingress and far below what a thousand connections
/// could pin unbudgeted, so the code path is the same one production takes.
///
/// Raised to four maximum frames where that is larger (it is, by 3%: the
/// protocol's own frame ceiling is 129 MiB and a connection's quarter share
/// has to hold one of those plus its parse copy). A default that narrowed a
/// ceiling the protocol already enforces would refuse a legitimate VMSET on a
/// machine with no memory limit at all, which is not a budget, it is a
/// regression.
pub const UNLIMITED_DEFAULT_CAP: u64 = 1 << 30;

/// How long a connection whose growth was refused waits before the frame is
/// refused outright. While it waits it does not read, which is free TCP
/// backpressure; after it, the peer is told to retry.
pub const DEFAULT_STALL: Duration = Duration::from_millis(500);

/// The smallest class cap worth having: below this a single legitimate
/// pipelined burst cannot be buffered, and every connection would stall on its
/// first frame. A ceiling this tight is reported, not silently rounded up -
/// see [`IngressBudget::cap`].
const MIN_CAP: u64 = 4 * CHUNK_BYTES;

/// What ingress may hold in total, and how that figure was arrived at.
///
/// Three states, like the governor's own [`Budget`], and for the same reason:
/// "no ceiling" and "a ceiling nobody can read" are opposites, and one number
/// cannot report both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IngressCap {
    /// A ceiling applies and this many bytes are the class's share of it.
    Room(u64),
    /// No ceiling applies anywhere, so [`UNLIMITED_DEFAULT_CAP`] stands in.
    Default(u64),
    /// A ceiling applies and its headroom could not be read. Every connection
    /// still gets its floor - refusing to accept at all would take the server
    /// down for an accounting fault - and nothing may grow past it.
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
    ///
    /// The whole difference between backpressure and an error. A class that is
    /// momentarily full, or a governor whose headroom the delta has taken,
    /// clears on its own; a frame larger than one connection may hold does
    /// not, and telling a client to retry it is telling it to loop.
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
    ///
    /// The first word IS the code, so a retryable refusal must not be dressed
    /// as `ERR`: that turns backpressure into a failure the caller gives up
    /// on.
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
#[derive(Debug)]
pub struct IngressBudget {
    governor: Arc<MemoryGovernor>,
    cap: IngressCap,
    per_conn_max: u64,
    stall: Duration,
    /// This class's share of the governor's `outstanding`. See the module
    /// note: not a second budget, the same bytes seen by class.
    held: AtomicU64,
}

impl IngressBudget {
    /// Build the budget from the settings an operator supplied.
    ///
    /// The cap is decided ONCE, here. Headroom moves with every allocation the
    /// process and its cgroup siblings make, and a class cap that moved with
    /// it would let a connection be refused for a growth it was granted a
    /// millisecond earlier, on a store doing nothing different. What stays
    /// live is the governor underneath: a reservation still has to fit the
    /// headroom of the moment.
    #[must_use]
    pub fn new(
        governor: Arc<MemoryGovernor>,
        fraction_percent: Option<u64>,
        explicit_bytes: Option<u64>,
        stall: Option<Duration>,
        per_connection_ceiling: u64,
    ) -> Self {
        let fraction = fraction_percent
            .filter(|f| *f > 0 && *f <= 100)
            .unwrap_or(DEFAULT_FRACTION);
        let cap = match (explicit_bytes.filter(|b| *b > 0), governor.budget()) {
            // An operator's figure replaces the fraction outright - the two
            // are alternatives, not a pair to be intersected - but it is
            // never a licence above the governor: the reservation behind it
            // still has to fit the headroom of the moment.
            (Some(b), Budget::Room(usable)) => IngressCap::Room(b.min(usable)),
            (Some(b), Budget::Unlimited) => IngressCap::Default(b),
            // A figure was given for exactly this case: an unreadable ceiling
            // is a refusal only while nobody has said what to do about it.
            (Some(b), Budget::Unreadable) => IngressCap::Room(b),
            // The derived figure gets a floor, because a share so small that
            // no legitimate pipelined burst fits would stall every connection
            // on its first frame - and never above what the governor has.
            (None, Budget::Room(usable)) => IngressCap::Room(
                (usable.saturating_mul(fraction) / 100)
                    .max(MIN_CAP)
                    .min(usable),
            ),
            (None, Budget::Unlimited) => IngressCap::Default(
                UNLIMITED_DEFAULT_CAP.max(4 * per_connection_ceiling.saturating_mul(PARSE_FACTOR)),
            ),
            (None, Budget::Unreadable) => IngressCap::FloorOnly,
        };
        if matches!(cap, IngressCap::FloorOnly) {
            // Said out loud, once, at startup. A server quietly serving only
            // small frames looks to a client like refusals it cannot explain,
            // and the counter below turns "quietly" into a number.
            tracing::warn!(
                "ingress budget: a memory limit applies to this process and its \
                 headroom cannot be read; connections will be given their {} \
                 byte floor and no growth. Set SKEG_MEMORY_LIMIT_BYTES or \
                 SKEG_INGRESS_BUDGET_BYTES to run with a budget.",
                FLOOR_BYTES
            );
        }
        // Fairness: a quarter each, so four greedy connections is the worst a
        // single peer can arrange for the fifth. Never above the frame ceiling
        // the protocol handler already enforces, and never below the floor - a
        // cap so small that the quarter rounds under the floor would refuse
        // every connection its first read.
        let per_conn_max = (cap.bytes() / 4)
            .min(per_connection_ceiling.saturating_mul(PARSE_FACTOR))
            .max(FLOOR_BYTES);
        Self {
            governor,
            cap,
            per_conn_max,
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

    /// How long a refused connection waits before its frame is refused.
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
    /// NON-BLOCKING, and called before the connection semaphore's permit: a
    /// budget awaited at accept is a listener that stops accepting, which is
    /// how a memory limit becomes an availability outage. Either the floor is
    /// there and the connection is served, or the peer is told by name.
    ///
    /// # Errors
    /// Propagates the class, allowance and governor refusals.
    pub fn try_accept(self: &Arc<Self>) -> Result<ConnectionBudget, IngressRejected> {
        // The floor under an unreadable budget is granted WITHOUT a governor
        // reservation, because the governor cannot answer at all in that
        // state and refusing every connection would turn an accounting fault
        // into an outage. What bounds it is the connection semaphore: at its
        // default of 1024 the whole floor is 8 MiB, which is knowable without
        // reading anything.
        if matches!(self.cap, IngressCap::FloorOnly) {
            skeg_telemetry::tick_counter(skeg_telemetry::Counter::IngressBudgetUnreadable);
            self.held.fetch_add(FLOOR_BYTES, Ordering::AcqRel);
            return Ok(ConnectionBudget {
                budget: Arc::clone(self),
                chunks: vec![Chunk {
                    bytes: FLOOR_BYTES,
                    reservation: None,
                }],
                held: FLOOR_BYTES,
            });
        }
        let reservation = self.reserve(FLOOR_BYTES)?;
        Ok(ConnectionBudget {
            budget: Arc::clone(self),
            chunks: vec![Chunk {
                bytes: FLOOR_BYTES,
                reservation: Some(reservation),
            }],
            held: FLOOR_BYTES,
        })
    }

    /// Charge `bytes` to the class AND to the governor, or refuse.
    ///
    /// The class share moves first and is rolled back if the governor refuses:
    /// the other order would let a connection see a class total that no
    /// reservation stands behind.
    fn reserve(&self, bytes: u64) -> Result<MemoryReservation, IngressRejected> {
        let cap = match self.cap {
            IngressCap::FloorOnly => return Err(IngressRejected::Unreadable { requested: bytes }),
            IngressCap::Room(c) | IngressCap::Default(c) => c,
        };
        let mut held = self.held.load(Ordering::Acquire);
        loop {
            let wanted = held.checked_add(bytes).ok_or(IngressRejected::ClassFull {
                held,
                requested: bytes,
                cap,
            })?;
            if wanted > cap {
                return Err(IngressRejected::ClassFull {
                    held,
                    requested: bytes,
                    cap,
                });
            }
            match self
                .held
                .compare_exchange_weak(held, wanted, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => break,
                Err(actual) => held = actual,
            }
        }
        match self.governor.try_reserve(bytes) {
            Ok(r) => Ok(r),
            Err(e) => {
                self.held.fetch_sub(bytes, Ordering::AcqRel);
                Err(IngressRejected::Governor(e))
            }
        }
    }

    /// Give `bytes` of the class share back. The governor reservation behind
    /// them is released by dropping it, in the same step.
    fn release(&self, bytes: u64) {
        self.held.fetch_sub(bytes, Ordering::AcqRel);
    }
}

/// One granted step of a connection's charge, and the governor reservation
/// standing behind it.
///
/// `reservation` is `None` only for a floor granted under an unreadable
/// budget, where the governor has no answer to give.
#[derive(Debug)]
struct Chunk {
    bytes: u64,
    reservation: Option<MemoryReservation>,
}

/// One connection's charge, released when the connection task ends.
///
/// The connection asks for a BUFFER CAPACITY and the budget converts it: the
/// charge is capacity times [`PARSE_FACTOR`], rounded up to [`CHUNK_BYTES`].
/// Capacity and not length, because `BytesMut::reserve` rounds up and the
/// process pays for the allocation, not for the bytes that arrived in it.
#[derive(Debug)]
pub struct ConnectionBudget {
    budget: Arc<IngressBudget>,
    chunks: Vec<Chunk>,
    held: u64,
}

/// The charge a buffer of `buffer_capacity` bytes carries.
#[must_use]
pub fn charge_for(buffer_capacity: usize) -> u64 {
    let want = (buffer_capacity as u64).saturating_mul(PARSE_FACTOR);
    if want <= FLOOR_BYTES {
        FLOOR_BYTES
    } else {
        want.div_ceil(CHUNK_BYTES).saturating_mul(CHUNK_BYTES)
    }
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

    /// How many of this connection's charges are backed by a governor
    /// reservation.
    ///
    /// All of them, except a floor granted under an unreadable budget - the
    /// one case where the governor has no answer to give and the floor is
    /// bounded by the connection semaphore instead. Reported so a test can
    /// tell the two apart, because from the outside they hold the same bytes.
    #[must_use]
    pub fn reserved_chunks(&self) -> usize {
        self.chunks
            .iter()
            .filter(|c| c.reservation.is_some())
            .count()
    }

    /// Charge for a buffer about to hold `buffer_capacity` bytes.
    ///
    /// RESERVE BEFORE GROW: the caller must not let the buffer reach a size
    /// this has not granted. A refusal is the signal to stop reading, which is
    /// TCP backpressure the peer feels for free.
    ///
    /// # Errors
    /// Propagates the class, allowance and governor refusals.
    pub fn grow_to(&mut self, buffer_capacity: usize) -> Result<(), IngressRejected> {
        let want = charge_for(buffer_capacity);
        if want <= self.held {
            return Ok(());
        }
        // Asked before the allowance, because under an unreadable budget the
        // allowance IS the floor, and answering "your frame is too big" would
        // name the wrong cause and the wrong remedy.
        if matches!(self.budget.cap, IngressCap::FloorOnly) {
            return Err(IngressRejected::Unreadable { requested: want });
        }
        if want > self.budget.per_conn_max {
            return Err(IngressRejected::OverConnectionAllowance {
                requested: want,
                allowance: self.budget.per_conn_max,
            });
        }
        let delta = want - self.held;
        let reservation = self.budget.reserve(delta)?;
        self.chunks.push(Chunk {
            bytes: delta,
            reservation: Some(reservation),
        });
        self.held = want;
        Ok(())
    }

    /// Give back what a drained buffer no longer needs, down to the floor.
    ///
    /// Without this a connection that bursted once would hold its peak for the
    /// rest of its life: `BytesMut` keeps its allocation across `split_to`,
    /// and a charge that only ever grew would make the class cap a high-water
    /// mark of every connection that ever existed.
    pub fn shrink_to(&mut self, buffer_capacity: usize) {
        let want = charge_for(buffer_capacity);
        while self.chunks.len() > 1 {
            let top = self.chunks[self.chunks.len() - 1].bytes;
            if self.held - top < want {
                break;
            }
            self.chunks.pop();
            self.held -= top;
            self.budget.release(top);
        }
    }
}

impl Drop for ConnectionBudget {
    fn drop(&mut self) {
        self.budget.release(self.held);
    }
}

/// How often a stalled connection asks whether the room has come back.
///
/// Short against the stall it lives inside: the point of waiting is to let a
/// burst on another connection finish, and those finish in milliseconds. A
/// stalled connection is not reading, so the cost of asking often is a timer,
/// not a syscall.
const STALL_POLL: Duration = Duration::from_millis(10);

/// Charge for the buffer the connection is about to need, waiting out the
/// stall if the answer is a refusal that could change.
///
/// RESERVE BEFORE GROW. While this waits the connection does not read, which
/// is TCP backpressure the peer feels without anything being sent - the
/// cheapest form of "slow down" there is. Only when the room has not come back
/// within the stall is the frame refused, and then with a retryable code,
/// because what refused it was other traffic and not this client's request.
///
/// A refusal that waiting cannot change - a frame larger than the connection
/// will ever be allowed - is returned at once: stalling on it would just make
/// the client wait for the same answer.
pub async fn grow_or_stall(
    budget: &mut ConnectionBudget,
    want: usize,
    fp_key: &str,
) -> Result<(), IngressRejected> {
    fn attempt(
        budget: &mut ConnectionBudget,
        want: usize,
        fp_key: &str,
    ) -> Result<(), IngressRejected> {
        if crate::fp_ingress!(
            crate::failpoint::IngressFailpoint::GrowRefusedMidFrame,
            fp_key
        ) {
            return Err(IngressRejected::ClassFull {
                held: budget.budget().held_bytes(),
                requested: charge_for(want),
                cap: budget.budget().cap().bytes(),
            });
        }
        budget.grow_to(want)
    }

    let mut last = match attempt(budget, want, fp_key) {
        Ok(()) => return Ok(()),
        Err(e) => e,
    };
    if !last.is_retryable() {
        return Err(last);
    }
    skeg_telemetry::tick_counter(skeg_telemetry::Counter::IngressStalls);
    let deadline = Instant::now() + budget.budget().stall();
    while Instant::now() < deadline {
        tokio::time::sleep(STALL_POLL).await;
        match attempt(budget, want, fp_key) {
            Ok(()) => return Ok(()),
            Err(e) => last = e,
        }
    }
    Err(last)
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
    fn a_drained_buffer_gives_its_growth_back() {
        // `BytesMut` keeps its allocation, so a charge that only grew would
        // make every connection's peak permanent and the class cap a
        // high-water mark of the whole uptime.
        let budget = budget_with_cap(64 * CHUNK_BYTES);
        let mut conn = budget.try_accept().expect("the floor");
        assert_eq!(conn.held_bytes(), FLOOR_BYTES);
        assert_eq!(
            conn.reserved_chunks(),
            1,
            "the floor stands on a reservation"
        );
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
    fn the_ingress_and_delta_reservations_sum_under_one_budget() {
        // The point of the whole module: ingress is not a second budget beside
        // the governor's, it is a class inside it. A byte a socket holds and a
        // byte the delta holds have to be visible in ONE outstanding total, or
        // the sum of the two can still take the process out.
        //
        // `governor.try_reserve` is exactly what `Vindex::reserve_memory`
        // calls, so this is the delta's own path, not a stand-in for it.
        // The class cap is deliberately generous against the headroom: what
        // must refuse the second growth is the GOVERNOR, with room left in
        // the class, or the test would be proving the class cap again.
        let usable = 20 * CHUNK_BYTES;
        let gov = governor(Headroom::Known(usable));
        let budget = Arc::new(IngressBudget::new(
            Arc::clone(&gov),
            None,
            Some(16 * CHUNK_BYTES),
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
        assert!(
            budget.held_bytes() + 2 * CHUNK_BYTES <= budget.cap().bytes(),
            "the class still had room, so the refusal came from the shared total"
        );
        drop((delta, rest));
    }

    /// A megabyte, so the arithmetic below reads as the numbers an operator
    /// would compute by hand.
    const MIB: u64 = 1 << 20;

    /// The worked example, pinned. Everything documented about what a
    /// container-sized budget offers is derived here, and derived numbers that
    /// only live in prose stop being true the first time a constant moves.
    #[test]
    fn the_arithmetic_of_a_256_mib_container_is_what_the_docs_say() {
        // A 256 MiB cgroup with about 100 MiB resident leaves ~150 MiB.
        let gov = Arc::new(
            MemoryGovernor::new(Arc::new(Fixed(Headroom::Known(150 * MIB))), None, None)
                .expect("a governor"),
        );
        // The governor holds a tenth back, floored at 64 MiB - the floor wins
        // at this size.
        assert_eq!(gov.reserve_bytes(), 64 * MIB);
        assert_eq!(gov.budget(), Budget::Room(86 * MIB), "150 less the reserve");

        let budget = Arc::new(IngressBudget::new(
            Arc::clone(&gov),
            None,
            None,
            None,
            crate::resp3_handler::MAX_CONN_BUFFER as u64,
        ));
        // A quarter of the usable headroom for the whole ingress class...
        assert_eq!(budget.cap(), IngressCap::Room(22_544_384), "21.5 MiB");
        // ...and a quarter of that for any one connection, in CHARGE, which is
        // twice the buffer it may hold.
        assert_eq!(budget.per_connection_max(), 5_636_096, "5.375 MiB charged");
        assert_eq!(
            budget.per_connection_max() / PARSE_FACTOR,
            2_818_048,
            "2.6875 MiB of buffer"
        );

        // So a VMSET at the protocol's frame ceiling does not fit, and this is
        // the consequence worth declaring: in a 256 MiB container that request
        // is refused BY NAME, where before it was admitted and the process was
        // killed for it. The refusal is not retryable, because the same frame
        // will not fit next time either.
        let mut conn = budget.try_accept().expect("the floor");
        let err = conn
            .grow_to(crate::resp3_handler::MAX_CONN_BUFFER)
            .expect_err("a 129 MiB frame cannot fit a 21.5 MiB class");
        assert!(
            matches!(err, IngressRejected::OverConnectionAllowance { .. }),
            "wrong refusal: {err:?}"
        );
        assert!(!err.is_retryable(), "retrying will not make it fit");
        assert!(
            err.wire_message().starts_with("ERR "),
            "a permanent refusal must not wear a retryable code: {}",
            err.wire_message()
        );
    }

    /// Off Linux there is no cgroup to read, so the default cap is what every
    /// developer machine and most of the test suite actually runs. It must not
    /// narrow the frame ceiling the protocol already enforces - a budget that
    /// refuses a legitimate VMSET on a machine with no memory limit at all is
    /// not a budget, it is a regression.
    #[test]
    fn the_default_cap_still_admits_one_maximum_frame_per_connection() {
        let budget = Arc::new(IngressBudget::new(
            governor(Headroom::Unlimited),
            None,
            None,
            None,
            crate::resp3_handler::MAX_CONN_BUFFER as u64,
        ));
        assert_eq!(budget.cap().state_name(), "default");
        assert!(
            budget.cap().bytes() >= UNLIMITED_DEFAULT_CAP,
            "the documented 1 GiB is the floor of this figure, not a ceiling"
        );
        assert_eq!(
            budget.per_connection_max() / PARSE_FACTOR,
            crate::resp3_handler::MAX_CONN_BUFFER as u64,
            "the per-connection buffer allowance is exactly the frame ceiling"
        );
        let mut conn = budget.try_accept().expect("the floor");
        conn.grow_to(crate::resp3_handler::MAX_CONN_BUFFER)
            .expect("a maximum frame must still fit under the default cap");
    }

    #[test]
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
        assert_eq!(
            conn.reserved_chunks(),
            0,
            "the governor cannot answer at all here, so the floor is bounded \
             by the connection semaphore and not by a reservation"
        );
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
