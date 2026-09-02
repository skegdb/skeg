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
//!   "failpoints"))` the arming state does not exist and [`fp!`] expands to a
//!   type annotation, so the write path carries no branch at all - and the
//!   variant name is still checked by the compiler.
//!
//! Points are armed process-wide, so a test that arms one must not run
//! concurrently with another that cares. Tests here are `#[tokio::test]` on
//! their own runtime but share the process: arm, exercise, disarm.

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
        }
    }
}

#[cfg(any(test, feature = "failpoints"))]
mod armed_state {
    use super::WriteFailpoint;
    use std::sync::atomic::{AtomicU64, Ordering};

    static ARMED: AtomicU64 = AtomicU64::new(0);

    /// Make `fp` fail until it is disarmed.
    pub fn arm(fp: WriteFailpoint) {
        ARMED.fetch_or(fp.bit(), Ordering::SeqCst);
    }

    /// Stop `fp` failing.
    pub fn disarm(fp: WriteFailpoint) {
        ARMED.fetch_and(!fp.bit(), Ordering::SeqCst);
    }

    /// Disarm every point. Cheap insurance at the end of a test.
    pub fn disarm_all() {
        ARMED.store(0, Ordering::SeqCst);
    }

    /// Is `fp` armed? Called by [`crate::fp`], not usually by hand.
    #[must_use]
    pub fn armed(fp: WriteFailpoint) -> bool {
        ARMED.load(Ordering::SeqCst) & fp.bit() != 0
    }
}

#[cfg(any(test, feature = "failpoints"))]
pub use armed_state::{arm, armed, disarm, disarm_all};

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

#[cfg(test)]
mod tests {
    use super::{WriteFailpoint, arm, armed, disarm, disarm_all};

    fn guarded(fp: WriteFailpoint) -> Result<&'static str, &'static str> {
        crate::fp!(fp, Err("failed at the failpoint"));
        Ok("wrote")
    }

    /// One test, not two: the armed mask is process-wide and the test harness
    /// runs a crate's tests in parallel threads, so two tests arming points
    /// would race each other rather than test anything.
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
}
