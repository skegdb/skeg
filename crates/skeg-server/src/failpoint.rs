//! Typed write-path failpoints.
//!
//! A correctness test for a crash window needs to make ONE step of a
//! multi-step write fail, on purpose, at a named point. The convention this
//! repo grew up with is environmental - a directory where a file is expected,
//! `chmod 000`, a quota set below the current count - and it still covers most
//! of what the suite needs. It stops covering the interesting windows the
//! moment two steps of the same operation touch the same files: "the
//! destination write fails but the source delete does not" cannot be expressed
//! by taking permissions off a directory both of them use.
//!
//! So: a failpoint registry, and a deliberately small one.
//!
//! - **Typed, not string-keyed.** A `fail::cfg("shard::reshard_dst", ...)`
//!   registry answers "no such point" by doing nothing, so a renamed site
//!   turns a passing test into a passing test that verifies nothing. Here the
//!   name is an enum variant: rename the site and the test stops compiling.
//! - **No external crate.** One `AtomicU64` bitmask and a macro.
//! - **Absent from a normal build.** Without `cfg(any(test, feature =
//!   "failpoints"))` the arming state does not exist and [`crate::fp!`] expands to a
//!   type annotation, so the write path carries no branch at all - and the
//!   variant name is still checked by the compiler.
//!
//! There are two ways to arm a point, and which one a site can use is decided
//! by the thread it runs on.
//!
//! - [`arm`] is PER THREAD, so a test that arms a point cannot fail a test
//!   running beside it. It only reaches sites that run on the caller's own
//!   thread: the coordinator half of `ShardSet` does, a shard worker does not
//!   (`run_shard` spawns the request onto its own runtime thread).
//! - [`arm_at`] is process-wide but SCOPED TO A KEY - the vindex name - so a
//!   site inside a shard worker is reachable while two tests naming different
//!   indexes still cannot see each other. Every test in this repo builds its
//!   own index under its own name, so that is the whole isolation rule - and
//!   it is a rule about NAMES, which nothing but convention keeps unique.
//!   [`arm_at`] therefore refuses to arm a point that is already armed for the
//!   same key: two tests in one binary sharing a name would otherwise fail
//!   each other at random, which is the failure mode a per-thread mask exists
//!   to avoid and this one has to catch instead.
//!
//! Both record whether the point actually FIRED, and every test asserts it: an
//! armed point that never fires makes its test pass for the wrong reason.

/// A point on the server's vector write path that a test can make fail.
///
/// Always compiled, feature or not, so `fp!(...)` type-checks the variant name
/// in every build.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WriteFailpoint {
    /// The `Vset` that writes the moved row to its new owner during a reshard.
    ReshardDestinationWrite,
    /// The `Vdel` that removes the source copy once the move committed.
    ReshardSourceDelete,
    /// The `Vset` that writes a boundary replica during an overlap.
    OverlapReplicaWrite,
    /// The `Vdel` cleaning up the old copy after a routed overwrite committed.
    OverwriteOldCopyDelete,
    /// Writing a VSET's payload blob to the vLog, BEFORE the commit point.
    /// Nothing is published yet, so this one is a plain failure.
    PayloadPrepare,
    /// The vector write that IS the commit point: the WAL append the engine
    /// performs under the vindex write lock.
    VectorCommit,
    /// Indexing the payload's fields, immediately AFTER the commit point.
    PayloadApply,
    /// Reclaiming the blob a committed VSET superseded. Post-commit.
    PayloadPostCommitCleanup,
    /// Reclaiming a deleted row's blob, after the tombstone is durable.
    VdelBlobDelete,
    /// The blob sweep a `VINDEX.DROP` runs once the catalogue no longer names
    /// the index.
    DropBlobSweep,
}

impl WriteFailpoint {
    /// This point's bit in the armed mask. Exhaustive on purpose: a new
    /// variant does not compile until it is given a bit.
    #[cfg(any(test, feature = "failpoints"))]
    const fn bit(self) -> u64 {
        match self {
            WriteFailpoint::ReshardDestinationWrite => 1 << 0,
            WriteFailpoint::ReshardSourceDelete => 1 << 1,
            WriteFailpoint::OverlapReplicaWrite => 1 << 2,
            WriteFailpoint::OverwriteOldCopyDelete => 1 << 3,
            WriteFailpoint::PayloadPrepare => 1 << 4,
            WriteFailpoint::VectorCommit => 1 << 5,
            WriteFailpoint::PayloadApply => 1 << 6,
            WriteFailpoint::PayloadPostCommitCleanup => 1 << 7,
            WriteFailpoint::VdelBlobDelete => 1 << 8,
            WriteFailpoint::DropBlobSweep => 1 << 9,
        }
    }
}

#[cfg(any(test, feature = "failpoints"))]
mod armed_state {
    use super::WriteFailpoint;
    use std::cell::Cell;

    thread_local! {
        /// PER THREAD, not process-wide.
        ///
        /// The harness runs a crate's tests in parallel threads, so a global
        /// mask means a test that arms a point fails whichever unrelated test
        /// happens to be running beside it. That is how this was first
        /// written, and it took exactly one run to show: a fold in another
        /// test came back "failpoint: versions.bin write refused".
        ///
        /// The cost is a real constraint rather than a free win - the site has
        /// to fire on the thread that armed it. Every point here does: they
        /// all sit on the caller's own thread, not on a shard worker and not
        /// inside a rayon pool. `fired` is what turns that from a note into
        /// something a test can check.
        static ARMED: Cell<u64> = const { Cell::new(0) };
        /// Points that have actually FIRED since they were armed.
        ///
        /// An armed point that never fires makes its test pass for the wrong
        /// reason: the operation succeeds, the assertion about the aftermath
        /// holds trivially, and nothing says the window was never entered. A
        /// site moved onto another thread, or behind a branch that no longer
        /// runs, fails silently that way.
        static FIRED: Cell<u64> = const { Cell::new(0) };
    }

    /// Make `fp` fail on THIS THREAD until it is disarmed. Clears its fired
    /// flag, so [`fired`] answers about this arming and not an earlier one.
    pub fn arm(fp: WriteFailpoint) {
        ARMED.with(|a| a.set(a.get() | fp.bit()));
        FIRED.with(|f| f.set(f.get() & !fp.bit()));
    }

    /// Stop `fp` failing. The fired flag is left alone: a test disarms before
    /// it asserts.
    pub fn disarm(fp: WriteFailpoint) {
        ARMED.with(|a| a.set(a.get() & !fp.bit()));
    }

    /// Disarm every point. Cheap insurance at the end of a test, and it too
    /// leaves the fired flags for the assertions that follow.
    pub fn disarm_all() {
        ARMED.with(|a| a.set(0));
    }

    /// Is `fp` armed? Called by [`crate::fp`], not usually by hand. Records
    /// the hit.
    #[must_use]
    pub fn armed(fp: WriteFailpoint) -> bool {
        let hit = ARMED.with(Cell::get) & fp.bit() != 0;
        if hit {
            FIRED.with(|f| f.set(f.get() | fp.bit()));
        }
        hit
    }

    /// Did `fp` fire since it was armed? Assert it: an armed point that never
    /// fired means the test proved nothing.
    #[must_use]
    pub fn fired(fp: WriteFailpoint) -> bool {
        FIRED.with(Cell::get) & fp.bit() != 0
    }
}

#[cfg(any(test, feature = "failpoints"))]
pub use armed_state::{arm, armed, disarm, disarm_all, fired};

#[cfg(any(test, feature = "failpoints"))]
mod keyed_state {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// One armed (point, key) pair.
    struct Entry {
        bit: u64,
        key: String,
        armed: bool,
        fired: bool,
    }

    /// A process-wide, key-scoped arming table for one FAMILY of failpoints.
    ///
    /// One instance per enum, so a bit means the same thing everywhere it is
    /// read: two enums sharing a table would have `1 << 0` name two different
    /// sites, and arming one would fire the other.
    pub struct KeyedRegistry {
        /// Bits with at least one key armed anywhere in the process.
        ///
        /// The guarded sites sit on hot paths, and under `cargo test
        /// --workspace` the feature is on for every consumer of this crate -
        /// so the cost of a DISARMED point has to be nothing worth measuring.
        /// One relaxed load, and the mutex below is touched only once a test
        /// has armed that exact point.
        any: AtomicU64,
        state: Mutex<Vec<Entry>>,
    }

    impl KeyedRegistry {
        pub const fn new() -> Self {
            Self {
                any: AtomicU64::new(0),
                state: Mutex::new(Vec::new()),
            }
        }

        fn with<R>(&self, f: impl FnOnce(&mut Vec<Entry>) -> R) -> R {
            // A poisoned lock here means a test panicked mid-assertion; the
            // state is a few booleans and recovering it is strictly better
            // than turning every later test in the binary into a panic about
            // the first one.
            let mut g = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            f(&mut g)
        }

        fn refresh_any(&self, entries: &[Entry]) {
            let mask = entries.iter().filter(|e| e.armed).fold(0, |m, e| m | e.bit);
            self.any.store(mask, Ordering::Relaxed);
        }

        /// Make `bit` fail wherever it is reached FOR `key`, until disarmed.
        /// Clears its fired flag, so [`KeyedRegistry::fired`] answers about
        /// this arming.
        ///
        /// # Panics
        ///
        /// If `bit` is ALREADY armed for `key`. The mask is process-wide and
        /// the key is a plain string, so isolation between two tests in one
        /// binary rests entirely on their choosing different keys. Nothing
        /// enforces that, and when it is broken the symptom is a test failing
        /// because of what another test armed - the exact kind of failure that
        /// gets rerun until it passes. A test that arms in a loop disarms each
        /// time round, so this is never a legitimate state.
        pub fn arm(&self, bit: u64, key: &str, what: &dyn std::fmt::Debug) {
            self.with(|entries| {
                match entries.iter_mut().find(|e| e.bit == bit && e.key == key) {
                    Some(e) => {
                        assert!(
                            !e.armed,
                            "{what:?} is already armed for '{key}': two tests in this \
                             binary are sharing a failpoint key, so each can make \
                             the other fail. Give them different keys."
                        );
                        e.armed = true;
                        e.fired = false;
                    }
                    None => entries.push(Entry {
                        bit,
                        key: key.to_owned(),
                        armed: true,
                        fired: false,
                    }),
                }
                self.refresh_any(entries);
            });
        }

        /// Stop `bit` failing for `key`. The fired flag is left alone: a test
        /// disarms before it asserts.
        pub fn disarm(&self, bit: u64, key: &str) {
            self.with(|entries| {
                if let Some(e) = entries.iter_mut().find(|e| e.bit == bit && e.key == key) {
                    e.armed = false;
                }
                self.refresh_any(entries);
            });
        }

        /// Is `bit` armed for `key`? Records the hit.
        pub fn armed(&self, bit: u64, key: &str) -> bool {
            if self.any.load(Ordering::Relaxed) & bit == 0 {
                return false;
            }
            self.with(|entries| {
                match entries
                    .iter_mut()
                    .find(|e| e.bit == bit && e.key == key && e.armed)
                {
                    Some(e) => {
                        e.fired = true;
                        true
                    }
                    None => false,
                }
            })
        }

        /// Did `bit` fire for `key` since it was armed? Assert it: an armed
        /// point that never fired means the test proved nothing.
        pub fn fired(&self, bit: u64, key: &str) -> bool {
            self.with(|entries| {
                entries
                    .iter()
                    .any(|e| e.bit == bit && e.key == key && e.fired)
            })
        }
    }
}

#[cfg(any(test, feature = "failpoints"))]
mod armed_at_state {
    use super::WriteFailpoint;
    use super::keyed_state::KeyedRegistry;

    static WRITE: KeyedRegistry = KeyedRegistry::new();

    /// Make `fp` fail wherever it is reached FOR `key`, until it is disarmed.
    /// Clears its fired flag, so [`fired_at`] answers about this arming.
    ///
    /// # Panics
    ///
    /// If `fp` is ALREADY armed for `key`; see [`KeyedRegistry::arm`].
    pub fn arm_at(fp: WriteFailpoint, key: &str) {
        WRITE.arm(fp.bit(), key, &fp);
    }

    /// Stop `fp` failing for `key`. The fired flag is left alone: a test
    /// disarms before it asserts.
    pub fn disarm_at(fp: WriteFailpoint, key: &str) {
        WRITE.disarm(fp.bit(), key);
    }

    /// Is `fp` armed for `key`? Called by [`crate::fp_at`], not usually by
    /// hand. Records the hit.
    #[must_use]
    pub fn armed_at(fp: WriteFailpoint, key: &str) -> bool {
        WRITE.armed(fp.bit(), key)
    }

    /// Did `fp` fire for `key` since it was armed? Assert it: an armed point
    /// that never fired means the test proved nothing.
    #[must_use]
    pub fn fired_at(fp: WriteFailpoint, key: &str) -> bool {
        WRITE.fired(fp.bit(), key)
    }
}

#[cfg(any(test, feature = "failpoints"))]
pub use armed_at_state::{arm_at, armed_at, disarm_at, fired_at};

/// A point on the INGRESS path - accept, buffer growth, connection close -
/// that a test can make fail.
///
/// Its own enum, not more variants on [`WriteFailpoint`]. The two families
/// have nothing to do with each other: one guards a durable write, the other
/// guards an allocation, and a single enum would let an ingress test arm a
/// bit a shard worker reads. Its own registry too, so the bits cannot collide.
///
/// KEYED, always. A connection runs on a task the accept loop spawned, so it
/// is never on the thread that armed anything, and the per-thread mask cannot
/// reach it. The key is the LISTENER PORT as a string: every test in this
/// binary binds port 0 and gets its own, so isolation does not depend on
/// anyone remembering to choose a unique name.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IngressFailpoint {
    /// The mid-frame `grow_to` in the RESP3 read loop: refuse the growth as if
    /// the class cap were full, so the stall and the typed refusal can be
    /// driven without really exhausting a budget.
    GrowRefusedMidFrame,
    /// The point where a closing connection is about to release its budget.
    /// Marks that the close path was actually reached, so "the budget came
    /// back" cannot pass on a connection that never got that far.
    ReleaseDeferredOnClose,
}

impl IngressFailpoint {
    /// This point's bit in its own registry. Exhaustive on purpose: a new
    /// variant does not compile until it is given a bit.
    #[cfg(any(test, feature = "failpoints"))]
    const fn bit(self) -> u64 {
        match self {
            IngressFailpoint::GrowRefusedMidFrame => 1 << 0,
            IngressFailpoint::ReleaseDeferredOnClose => 1 << 1,
        }
    }
}

#[cfg(any(test, feature = "failpoints"))]
mod ingress_at_state {
    use super::IngressFailpoint;
    use super::keyed_state::KeyedRegistry;

    static INGRESS: KeyedRegistry = KeyedRegistry::new();

    /// Make `fp` fail for the listener on `key`, until it is disarmed.
    ///
    /// # Panics
    ///
    /// If `fp` is ALREADY armed for `key`; see [`KeyedRegistry::arm`].
    pub fn arm_ingress_at(fp: IngressFailpoint, key: &str) {
        INGRESS.arm(fp.bit(), key, &fp);
    }

    /// Stop `fp` failing for `key`. The fired flag is left for the assertion.
    pub fn disarm_ingress_at(fp: IngressFailpoint, key: &str) {
        INGRESS.disarm(fp.bit(), key);
    }

    /// Is `fp` armed for `key`? Records the hit.
    #[must_use]
    pub fn armed_ingress_at(fp: IngressFailpoint, key: &str) -> bool {
        INGRESS.armed(fp.bit(), key)
    }

    /// Did `fp` fire for `key` since it was armed? Every ingress failpoint
    /// test asserts this.
    #[must_use]
    pub fn fired_ingress_at(fp: IngressFailpoint, key: &str) -> bool {
        INGRESS.fired(fp.bit(), key)
    }
}

#[cfg(any(test, feature = "failpoints"))]
pub use ingress_at_state::{arm_ingress_at, armed_ingress_at, disarm_ingress_at, fired_ingress_at};

/// A point where a write is REFUSED before any of it happens.
///
/// Its own family, next to [`WriteFailpoint`] and [`IngressFailpoint`], and
/// for the same reason as those two: the bits must not collide and an
/// admission test must not be able to arm a point a durable write reads.
///
/// The conditions here are real and reachable, but not on demand. A memory
/// refusal needs a governor with no headroom while a write is in flight; a
/// quota refusal needs a tenant already at its ceiling, and the native
/// listener - which is always tenant `0` - has no limits to be at. Arming the
/// point is how the SAME condition is put on both wires, which is the whole
/// question P0.5 asks.
///
/// KEYED on the vindex name, like [`WriteFailpoint`]: the site runs inside a
/// shard worker, never on the thread that armed it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdmissionFailpoint {
    /// The memory governor's answer at the VSET admission step: refuse as if
    /// the headroom were gone.
    MemoryRefusedAtVset,
    /// The tenant vector quota's answer at the VSET admission step: refuse as
    /// if the tenant were at its limit.
    QuotaRefusedAtVset,
    /// The bounded VSEARCH pool's answer: refuse as if every worker permit
    /// were taken. Saturating a real pool from a socket test means racing a
    /// semaphore, which is not something a deterministic test may do.
    VsearchQueueFullAtSearch,
}

impl AdmissionFailpoint {
    /// This point's bit in its own registry. Exhaustive on purpose: a new
    /// variant does not compile until it is given a bit.
    #[cfg(any(test, feature = "failpoints"))]
    const fn bit(self) -> u64 {
        match self {
            AdmissionFailpoint::MemoryRefusedAtVset => 1 << 0,
            AdmissionFailpoint::QuotaRefusedAtVset => 1 << 1,
            AdmissionFailpoint::VsearchQueueFullAtSearch => 1 << 2,
        }
    }
}

#[cfg(any(test, feature = "failpoints"))]
mod admission_at_state {
    use super::AdmissionFailpoint;
    use super::keyed_state::KeyedRegistry;

    static ADMISSION: KeyedRegistry = KeyedRegistry::new();

    /// Make `fp` refuse for the vindex named `key`, until it is disarmed.
    ///
    /// # Panics
    ///
    /// If `fp` is ALREADY armed for `key`; see [`KeyedRegistry::arm`].
    pub fn arm_admission_at(fp: AdmissionFailpoint, key: &str) {
        ADMISSION.arm(fp.bit(), key, &fp);
    }

    /// Stop `fp` refusing for `key`. The fired flag is left for the assertion.
    pub fn disarm_admission_at(fp: AdmissionFailpoint, key: &str) {
        ADMISSION.disarm(fp.bit(), key);
    }

    /// Is `fp` armed for `key`? Records the hit.
    #[must_use]
    pub fn armed_admission_at(fp: AdmissionFailpoint, key: &str) -> bool {
        ADMISSION.armed(fp.bit(), key)
    }

    /// Did `fp` fire for `key` since it was armed? Every admission failpoint
    /// test asserts this: a refusal that never happened proves nothing about
    /// how a refusal is reported.
    #[must_use]
    pub fn fired_admission_at(fp: AdmissionFailpoint, key: &str) -> bool {
        ADMISSION.fired(fp.bit(), key)
    }
}

#[cfg(any(test, feature = "failpoints"))]
pub use admission_at_state::{
    arm_admission_at, armed_admission_at, disarm_admission_at, fired_admission_at,
};

/// A point on the PLACEMENT path - the moment an owner map is published -
/// that a test can PARK rather than fail.
///
/// Its own family, its own registry, for the same reason as the three above:
/// the bits must not collide.
///
/// A gate, not a failure. The other families answer "make this step fail";
/// the question B0 asks is "what does a concurrent VSET/VDEL observe WHILE the
/// placement of an index is being replaced", and a failure cannot express it -
/// there is nothing wrong with the publish, the whole point is that it takes
/// time and something else runs during it. So the site parks until the test
/// releases it, which turns a microsecond-wide window that reproduced in
/// 30-70% of runs (audit 18) into a window the test opens and closes by hand.
///
/// KEYED on the vindex name, and process-wide: the publish runs on whatever
/// task called it, which is never the thread that armed the gate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlacementFailpoint {
    /// Inside the exclusive placement section of `rebuild_owner_maps`,
    /// immediately before the wholesale `owners.insert(name, map)`. Every
    /// shard has been scanned; nothing is published yet.
    OwnerMapPublish,
}

impl PlacementFailpoint {
    /// This point's bit in its own registry. Exhaustive on purpose: a new
    /// variant does not compile until it is given a bit.
    #[cfg(any(test, feature = "failpoints"))]
    const fn bit(self) -> u64 {
        match self {
            PlacementFailpoint::OwnerMapPublish => 1 << 0,
        }
    }
}

#[cfg(any(test, feature = "failpoints"))]
mod gate_state {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    /// How long a parked site waits before it gives up and panics. A gate a
    /// test forgets to release must not turn into a suite that hangs until CI
    /// kills it with no name attached to the hang.
    const MAX_PARK: Duration = Duration::from_secs(60);
    /// How often the park loop re-checks. Only reached when the watch channel
    /// says nothing, which is every 250 ms of a legitimate park.
    const POLL: Duration = Duration::from_millis(250);

    /// One armed (point, key) pair, plus the two signals a gate needs that a
    /// failure point does not: "the site was reached" and "the site may go on".
    struct Entry {
        bit: u64,
        key: String,
        armed: bool,
        /// The site was actually REACHED. Same role as `fired` in
        /// [`super::keyed_state::KeyedRegistry`]: a gate nothing ever entered
        /// makes its test pass for the wrong reason.
        fired: bool,
        /// Flipped true when the site is entered, so [`KeyedGate::wait_reached`]
        /// can await it without polling a boolean.
        reached: tokio::sync::watch::Sender<bool>,
        /// Flipped true by [`KeyedGate::release`], which is what unparks the
        /// site.
        release: tokio::sync::watch::Sender<bool>,
    }

    /// A process-wide, key-scoped registry of GATES for one family.
    ///
    /// Deliberately shaped like `KeyedRegistry` - same `any` fast path, same
    /// "arming an armed key is a bug" rule - because the isolation story is
    /// identical: the mask is process-wide and what keeps two tests apart is
    /// their choosing different vindex names.
    pub struct KeyedGate {
        /// Bits with at least one key armed anywhere in the process. The
        /// guarded site sits on the placement path, so a DISARMED gate has to
        /// cost one relaxed load and nothing else.
        any: AtomicU64,
        state: Mutex<Vec<Entry>>,
    }

    impl KeyedGate {
        pub const fn new() -> Self {
            Self {
                any: AtomicU64::new(0),
                state: Mutex::new(Vec::new()),
            }
        }

        fn with<R>(&self, f: impl FnOnce(&mut Vec<Entry>) -> R) -> R {
            let mut g = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            f(&mut g)
        }

        fn refresh_any(&self, entries: &[Entry]) {
            let mask = entries.iter().filter(|e| e.armed).fold(0, |m, e| m | e.bit);
            self.any.store(mask, Ordering::Relaxed);
        }

        /// Arm the gate at `bit` for `key`: the next task to reach that site
        /// parks there until [`KeyedGate::release`].
        ///
        /// # Panics
        ///
        /// If `bit` is ALREADY armed for `key` - two tests in one binary
        /// sharing a key can each park the other, which is exactly the
        /// failure mode that gets rerun until it passes.
        pub fn arm(&self, bit: u64, key: &str, what: &dyn std::fmt::Debug) {
            self.with(|entries| {
                match entries.iter_mut().find(|e| e.bit == bit && e.key == key) {
                    Some(e) => {
                        assert!(
                            !e.armed,
                            "{what:?} is already armed for '{key}': two tests in this \
                             binary are sharing a gate key, so each can park the \
                             other. Give them different keys."
                        );
                        e.armed = true;
                        e.fired = false;
                        e.reached.send_replace(false);
                        e.release.send_replace(false);
                    }
                    None => entries.push(Entry {
                        bit,
                        key: key.to_owned(),
                        armed: true,
                        fired: false,
                        reached: tokio::sync::watch::Sender::new(false),
                        release: tokio::sync::watch::Sender::new(false),
                    }),
                }
                self.refresh_any(entries);
            });
        }

        /// Enter the gate. Records the hit, wakes anyone in
        /// [`KeyedGate::wait_reached`], and parks until released.
        ///
        /// # Panics
        ///
        /// If the gate is never released within [`MAX_PARK`]: a forgotten
        /// release must name itself instead of hanging the suite.
        pub async fn enter(&self, bit: u64, key: &str, what: &(dyn std::fmt::Debug + Sync)) {
            if self.any.load(Ordering::Relaxed) & bit == 0 {
                return;
            }
            let waiter = self.with(|entries| {
                entries
                    .iter_mut()
                    .find(|e| e.bit == bit && e.key == key && e.armed)
                    .map(|e| {
                        e.fired = true;
                        e.reached.send_replace(true);
                        e.release.subscribe()
                    })
            });
            let Some(mut rx) = waiter else { return };
            let since = Instant::now();
            loop {
                if *rx.borrow_and_update() {
                    return;
                }
                match tokio::time::timeout(POLL, rx.changed()).await {
                    // Released, or the registry entry went away (it never
                    // does - the Vec only grows - but a dead sender is not a
                    // reason to park forever).
                    Ok(Ok(()) | Err(_)) => {}
                    Err(_) => assert!(
                        since.elapsed() < MAX_PARK,
                        "{what:?} parked at '{key}' for {:?} and was never released",
                        since.elapsed()
                    ),
                }
            }
        }

        /// Wait until the site guarded by `bit`/`key` has been ENTERED.
        ///
        /// # Panics
        ///
        /// If the site is not reached within [`MAX_PARK`].
        pub async fn wait_reached(&self, bit: u64, key: &str, what: &(dyn std::fmt::Debug + Sync)) {
            let waiter = self.with(|entries| {
                entries
                    .iter()
                    .find(|e| e.bit == bit && e.key == key)
                    .map(|e| e.reached.subscribe())
            });
            let Some(mut rx) = waiter else {
                panic!("{what:?} was never armed for '{key}'")
            };
            let since = Instant::now();
            loop {
                if *rx.borrow_and_update() {
                    return;
                }
                match tokio::time::timeout(POLL, rx.changed()).await {
                    Ok(Ok(()) | Err(_)) => {}
                    Err(_) => assert!(
                        since.elapsed() < MAX_PARK,
                        "{what:?} at '{key}' was not reached in {:?}",
                        since.elapsed()
                    ),
                }
            }
        }

        /// Let the parked site go on, and disarm: a second pass through the
        /// same site in the same test must not park again. The fired flag is
        /// left alone for the assertion that follows.
        pub fn release(&self, bit: u64, key: &str) {
            self.with(|entries| {
                if let Some(e) = entries.iter_mut().find(|e| e.bit == bit && e.key == key) {
                    e.armed = false;
                    e.release.send_replace(true);
                }
                self.refresh_any(entries);
            });
        }

        /// Was the gate at `bit`/`key` ever entered since it was armed?
        pub fn fired(&self, bit: u64, key: &str) -> bool {
            self.with(|entries| {
                entries
                    .iter()
                    .any(|e| e.bit == bit && e.key == key && e.fired)
            })
        }
    }
}

#[cfg(any(test, feature = "failpoints"))]
mod placement_gate_state {
    use super::PlacementFailpoint;
    use super::gate_state::KeyedGate;

    static PLACEMENT: KeyedGate = KeyedGate::new();

    /// Park the next task that reaches `fp` for the vindex named `key`, until
    /// [`release_gate_at`].
    ///
    /// # Panics
    ///
    /// If `fp` is ALREADY armed for `key`; see [`KeyedGate::arm`].
    pub fn arm_gate_at(fp: PlacementFailpoint, key: &str) {
        PLACEMENT.arm(fp.bit(), key, &fp);
    }

    /// Enter `fp` for `key`: park if armed, return immediately if not. Called
    /// by [`crate::gate_at`], not usually by hand.
    ///
    /// # Panics
    ///
    /// If the gate is armed and never released; see [`KeyedGate::enter`].
    pub async fn gate_at(fp: PlacementFailpoint, key: &str) {
        PLACEMENT.enter(fp.bit(), key, &fp).await;
    }

    /// Wait until the site guarded by `fp` for `key` is parked in the gate.
    ///
    /// # Panics
    ///
    /// If `fp` was never armed for `key`, or the site is not reached in time.
    pub async fn wait_reached(fp: PlacementFailpoint, key: &str) {
        PLACEMENT.wait_reached(fp.bit(), key, &fp).await;
    }

    /// Let the parked site go on, and disarm it.
    pub fn release_gate_at(fp: PlacementFailpoint, key: &str) {
        PLACEMENT.release(fp.bit(), key);
    }

    /// Did `fp` fire for `key` since it was armed? Every gate test asserts
    /// this: a gate nothing entered proves nothing about the window.
    #[must_use]
    pub fn fired_gate_at(fp: PlacementFailpoint, key: &str) -> bool {
        PLACEMENT.fired(fp.bit(), key)
    }
}

#[cfg(any(test, feature = "failpoints"))]
pub use placement_gate_state::{
    arm_gate_at, fired_gate_at, gate_at, release_gate_at, wait_reached,
};

/// PARK at `$fp` for `$key` while the gate is armed.
///
/// Expands to an `.await`, so the site must be in an async fn. In a build
/// without the failpoints it expands to a type annotation and the placement
/// path carries no branch and no await at all.
#[macro_export]
#[cfg(any(test, feature = "failpoints"))]
macro_rules! gate_at {
    ($fp:expr, $key:expr) => {
        $crate::failpoint::gate_at($fp, $key).await
    };
}

/// The disabled expansion: still names the variant, so a renamed point fails
/// to compile instead of quietly never parking.
#[macro_export]
#[cfg(not(any(test, feature = "failpoints")))]
macro_rules! gate_at {
    ($fp:expr, $key:expr) => {{
        let _: $crate::failpoint::PlacementFailpoint = $fp;
        let _ = &$key;
    }};
}

/// Is `$fp` armed for `$key`, as a boolean?
///
/// The admission family's form, shaped like [`fp_ingress!`] rather than
/// [`fp_at!`]: an admission site turns the hit into the SAME typed refusal
/// the real condition produces, so the value - not a `return` - is what the
/// caller needs.
#[macro_export]
#[cfg(any(test, feature = "failpoints"))]
macro_rules! fp_admission {
    ($fp:expr, $key:expr) => {
        $crate::failpoint::armed_admission_at($fp, $key)
    };
}

/// The disabled expansion: still names the variant, so a renamed point fails
/// to compile instead of quietly never firing.
#[macro_export]
#[cfg(not(any(test, feature = "failpoints")))]
macro_rules! fp_admission {
    ($fp:expr, $key:expr) => {{
        let _: $crate::failpoint::AdmissionFailpoint = $fp;
        let _ = &$key;
        false
    }};
}

/// Is `$fp` armed for `$key`, as a boolean?
///
/// The ingress family's form. Unlike [`fp_at!`] it does not `return`: an
/// ingress site has to turn the hit into its own typed refusal, which is the
/// value the caller then handles.
#[macro_export]
#[cfg(any(test, feature = "failpoints"))]
macro_rules! fp_ingress {
    ($fp:expr, $key:expr) => {
        $crate::failpoint::armed_ingress_at($fp, $key)
    };
}

/// The disabled expansion: still names the variant, so a renamed point fails
/// to compile instead of quietly never firing.
#[macro_export]
#[cfg(not(any(test, feature = "failpoints")))]
macro_rules! fp_ingress {
    ($fp:expr, $key:expr) => {{
        let _: $crate::failpoint::IngressFailpoint = $fp;
        let _ = &$key;
        false
    }};
}

/// Fail at `$fp` with `$err` when the point is armed.
///
/// Expands to nothing but a type annotation in a build without the failpoints,
/// so the write path is byte-identical to one that never heard of them.
#[macro_export]
#[cfg(any(test, feature = "failpoints"))]
macro_rules! fp {
    ($fp:expr, $err:expr) => {
        if $crate::failpoint::armed($fp) {
            return $err;
        }
    };
}

/// The disabled expansion: still names the variant, so a renamed point fails
/// to compile instead of quietly never firing.
#[macro_export]
#[cfg(not(any(test, feature = "failpoints")))]
macro_rules! fp {
    ($fp:expr, $err:expr) => {
        let _: $crate::failpoint::WriteFailpoint = $fp;
    };
}

/// Fail at `$fp` with `$err` when the point is armed FOR `$key`.
///
/// The keyed form, for a site that does not run on the thread that armed it -
/// anything inside a shard worker. `$key` is the vindex name.
#[macro_export]
#[cfg(any(test, feature = "failpoints"))]
macro_rules! fp_at {
    ($fp:expr, $key:expr, $err:expr) => {
        if $crate::failpoint::armed_at($fp, $key) {
            return $err;
        }
    };
}

/// The disabled expansion: still names the variant and evaluates nothing.
#[macro_export]
#[cfg(not(any(test, feature = "failpoints")))]
macro_rules! fp_at {
    ($fp:expr, $key:expr, $err:expr) => {{
        let _: $crate::failpoint::WriteFailpoint = $fp;
        let _ = &$key;
    }};
}

/// [`fp_at!`] as a VALUE rather than a `return`, for a post-commit site that
/// has to carry on rather than leave.
#[macro_export]
#[cfg(any(test, feature = "failpoints"))]
macro_rules! fp_check_at {
    ($fp:expr, $key:expr, $err:expr) => {
        if $crate::failpoint::armed_at($fp, $key) {
            $err
        } else {
            Ok(())
        }
    };
}

/// The disabled expansion: still names the variant.
#[macro_export]
#[cfg(not(any(test, feature = "failpoints")))]
macro_rules! fp_check_at {
    ($fp:expr, $key:expr, $err:expr) => {{
        let _: $crate::failpoint::WriteFailpoint = $fp;
        let _ = &$key;
        Ok(())
    }};
}

/// Like [`fp!`], but yields the failure as a VALUE rather than returning it.
///
/// For a site that cannot simply leave: the write it guards has already
/// happened, and something has to be undone before the error goes back to the
/// caller. `fp!` expands to a `return`, which is exactly the shape that leaves
/// the mess behind.
#[macro_export]
#[cfg(any(test, feature = "failpoints"))]
macro_rules! fp_check {
    ($fp:expr, $err:expr) => {
        if $crate::failpoint::armed($fp) {
            $err
        } else {
            Ok(())
        }
    };
}

/// The disabled expansion: still names the variant.
#[macro_export]
#[cfg(not(any(test, feature = "failpoints")))]
macro_rules! fp_check {
    ($fp:expr, $err:expr) => {{
        let _: $crate::failpoint::WriteFailpoint = $fp;
        Ok(())
    }};
}

#[cfg(test)]
mod tests {
    use super::{WriteFailpoint, arm, armed, disarm, disarm_all};

    fn guarded(fp: WriteFailpoint) -> Result<&'static str, &'static str> {
        crate::fp!(fp, Err("failed at the failpoint"));
        Ok("wrote")
    }

    /// One test, not two: the mask is per thread, so two tests that arm
    /// points would be checking different masks and neither would say
    /// anything about the other.
    #[test]
    fn points_fire_only_while_armed_and_only_the_one_armed() {
        disarm_all();
        assert_eq!(
            guarded(WriteFailpoint::ReshardDestinationWrite),
            Ok("wrote")
        );

        arm(WriteFailpoint::ReshardDestinationWrite);
        assert_eq!(
            guarded(WriteFailpoint::ReshardDestinationWrite),
            Err("failed at the failpoint")
        );
        // One bit each. A shared flag would pass every test that arms a single
        // point and break the first one that arms two.
        assert!(!armed(WriteFailpoint::ReshardSourceDelete));
        assert!(!armed(WriteFailpoint::OverlapReplicaWrite));
        assert!(!armed(WriteFailpoint::OverwriteOldCopyDelete));

        disarm(WriteFailpoint::ReshardDestinationWrite);
        assert_eq!(
            guarded(WriteFailpoint::ReshardDestinationWrite),
            Ok("wrote")
        );

        arm(WriteFailpoint::OverlapReplicaWrite);
        arm(WriteFailpoint::OverwriteOldCopyDelete);
        assert!(armed(WriteFailpoint::OverlapReplicaWrite));
        assert!(armed(WriteFailpoint::OverwriteOldCopyDelete));
        disarm_all();
        assert!(!armed(WriteFailpoint::OverlapReplicaWrite));
        assert!(!armed(WriteFailpoint::OverwriteOldCopyDelete));
    }

    /// The keyed mask is process-wide, so what keeps two tests apart is the
    /// key. This checks exactly that, plus the fired flag the assertions in
    /// every failpoint test lean on.
    #[test]
    fn a_keyed_point_fires_only_for_the_index_it_was_armed_for() {
        use super::{arm_at, armed_at, disarm_at, fired_at};
        let mine = "failpoint-unit-mine";
        let yours = "failpoint-unit-yours";
        assert!(!armed_at(WriteFailpoint::PayloadPrepare, mine));
        arm_at(WriteFailpoint::PayloadPrepare, mine);
        assert!(!fired_at(WriteFailpoint::PayloadPrepare, mine), "not yet");
        assert!(armed_at(WriteFailpoint::PayloadPrepare, mine));
        assert!(
            fired_at(WriteFailpoint::PayloadPrepare, mine),
            "and now it has: an armed point that never fires proves nothing"
        );
        assert!(
            !armed_at(WriteFailpoint::PayloadPrepare, yours),
            "another index must not be caught by this arming"
        );
        assert!(
            !armed_at(WriteFailpoint::VectorCommit, mine),
            "one bit each"
        );
        disarm_at(WriteFailpoint::PayloadPrepare, mine);
        assert!(!armed_at(WriteFailpoint::PayloadPrepare, mine));
        assert!(
            fired_at(WriteFailpoint::PayloadPrepare, mine),
            "disarming leaves the record of the hit for the assertion that follows"
        );
        // And arming it again, after the disarm, is the ordinary loop.
        arm_at(WriteFailpoint::PayloadPrepare, mine);
        disarm_at(WriteFailpoint::PayloadPrepare, mine);
    }

    /// The isolation rule is "two tests, two names", and nothing but
    /// convention keeps it. This is what makes breaking it loud instead of
    /// making one of the two tests fail for the other one's reason.
    #[test]
    #[should_panic(expected = "sharing a failpoint key")]
    fn arming_a_key_that_is_already_armed_is_refused() {
        use super::arm_at;
        let shared = "failpoint-unit-shared";
        arm_at(WriteFailpoint::VdelBlobDelete, shared);
        arm_at(WriteFailpoint::VdelBlobDelete, shared);
    }

    /// A gate is not a failure: the site must STOP there and go on only when
    /// the test says so. Both halves matter - a gate that never parks makes
    /// the concurrency test it exists for prove nothing, and a gate that
    /// never releases hangs the suite.
    #[tokio::test]
    async fn a_gate_parks_and_releases() {
        use super::{
            PlacementFailpoint, arm_gate_at, fired_gate_at, release_gate_at, wait_reached,
        };
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        const FP: PlacementFailpoint = PlacementFailpoint::OwnerMapPublish;
        let mine = "failpoint-unit-gate-mine";
        let yours = "failpoint-unit-gate-yours";

        // A DISARMED gate is a no-op, and says so.
        assert!(!fired_gate_at(FP, mine));
        crate::gate_at!(FP, mine);
        assert!(!fired_gate_at(FP, mine), "a disarmed gate is not entered");

        arm_gate_at(FP, mine);
        let through = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&through);
        let parked = tokio::spawn(async move {
            crate::gate_at!(FP, mine);
            flag.store(true, Ordering::SeqCst);
        });
        wait_reached(FP, mine).await;
        assert!(
            fired_gate_at(FP, mine),
            "the site was reached, so the gate must record the hit"
        );
        // The whole point: it is still in there.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(
            !through.load(Ordering::SeqCst),
            "an armed gate must PARK the site, not merely count it"
        );
        // And another index is not caught by this arming.
        crate::gate_at!(FP, yours);
        assert!(!fired_gate_at(FP, yours), "one key each");

        release_gate_at(FP, mine);
        parked.await.unwrap();
        assert!(through.load(Ordering::SeqCst), "released means released");
        assert!(
            fired_gate_at(FP, mine),
            "releasing leaves the record of the hit for the assertion that follows"
        );
        // Released is also disarmed: a second pass does not park again.
        crate::gate_at!(FP, mine);
    }

    /// Two families, two registries. Bit 0 of the ingress enum and bit 0 of
    /// the write enum are the same number, so a shared table would make
    /// arming an ingress point fail a shard worker - under the same key, with
    /// no test naming the site that broke.
    #[test]
    fn the_two_failpoint_families_do_not_share_a_table() {
        use super::{
            IngressFailpoint, arm_ingress_at, armed_at, armed_ingress_at, disarm_ingress_at,
            fired_ingress_at,
        };
        let key = "failpoint-unit-families";
        arm_ingress_at(IngressFailpoint::GrowRefusedMidFrame, key);
        assert!(
            !armed_at(WriteFailpoint::ReshardDestinationWrite, key),
            "an ingress arming reached the write family"
        );
        assert!(armed_ingress_at(IngressFailpoint::GrowRefusedMidFrame, key));
        assert!(
            !armed_ingress_at(IngressFailpoint::ReleaseDeferredOnClose, key),
            "one bit each"
        );
        disarm_ingress_at(IngressFailpoint::GrowRefusedMidFrame, key);
        assert!(fired_ingress_at(IngressFailpoint::GrowRefusedMidFrame, key));
    }
}
