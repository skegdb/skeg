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
//!   own index under its own name, so that is the whole isolation rule.
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
mod armed_at_state {
    use super::WriteFailpoint;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// One armed (point, key) pair.
    struct Entry {
        bit: u64,
        key: String,
        armed: bool,
        fired: bool,
    }

    /// Bits with at least one key armed anywhere in the process.
    ///
    /// The guarded sites sit on the vector write path, and under
    /// `cargo test --workspace` the feature is on for every consumer of this
    /// crate - so the cost of a DISARMED point has to be nothing worth
    /// measuring. One relaxed load, and the mutex below is touched only once
    /// a test has armed that exact point.
    static ANY: AtomicU64 = AtomicU64::new(0);
    static STATE: Mutex<Vec<Entry>> = Mutex::new(Vec::new());

    fn with<R>(f: impl FnOnce(&mut Vec<Entry>) -> R) -> R {
        // A poisoned lock here means a test panicked mid-assertion; the state
        // is a few booleans and recovering it is strictly better than turning
        // every later test in the binary into a panic about the first one.
        let mut g = STATE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        f(&mut g)
    }

    fn refresh_any(entries: &[Entry]) {
        let mask = entries.iter().filter(|e| e.armed).fold(0, |m, e| m | e.bit);
        ANY.store(mask, Ordering::Relaxed);
    }

    /// Make `fp` fail wherever it is reached FOR `key`, until it is disarmed.
    /// Clears its fired flag, so [`fired_at`] answers about this arming.
    pub fn arm_at(fp: WriteFailpoint, key: &str) {
        with(|entries| {
            match entries
                .iter_mut()
                .find(|e| e.bit == fp.bit() && e.key == key)
            {
                Some(e) => {
                    e.armed = true;
                    e.fired = false;
                }
                None => entries.push(Entry {
                    bit: fp.bit(),
                    key: key.to_owned(),
                    armed: true,
                    fired: false,
                }),
            }
            refresh_any(entries);
        });
    }

    /// Stop `fp` failing for `key`. The fired flag is left alone: a test
    /// disarms before it asserts.
    pub fn disarm_at(fp: WriteFailpoint, key: &str) {
        with(|entries| {
            if let Some(e) = entries
                .iter_mut()
                .find(|e| e.bit == fp.bit() && e.key == key)
            {
                e.armed = false;
            }
            refresh_any(entries);
        });
    }

    /// Is `fp` armed for `key`? Called by [`crate::fp_at`], not usually by
    /// hand. Records the hit.
    #[must_use]
    pub fn armed_at(fp: WriteFailpoint, key: &str) -> bool {
        if ANY.load(Ordering::Relaxed) & fp.bit() == 0 {
            return false;
        }
        with(|entries| {
            match entries
                .iter_mut()
                .find(|e| e.bit == fp.bit() && e.key == key && e.armed)
            {
                Some(e) => {
                    e.fired = true;
                    true
                }
                None => false,
            }
        })
    }

    /// Did `fp` fire for `key` since it was armed? Assert it: an armed point
    /// that never fired means the test proved nothing.
    #[must_use]
    pub fn fired_at(fp: WriteFailpoint, key: &str) -> bool {
        with(|entries| {
            entries
                .iter()
                .any(|e| e.bit == fp.bit() && e.key == key && e.fired)
        })
    }
}

#[cfg(any(test, feature = "failpoints"))]
pub use armed_at_state::{arm_at, armed_at, disarm_at, fired_at};

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
    }
}
