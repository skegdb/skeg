//! Typed failpoints for the durability window of a segment build.
//!
//! Same shape and the same reasoning as `skeg-server`'s: an enum variant
//! rather than a string key, one `AtomicU64` bitmask, no external crate, and
//! nothing at all in a build without the feature. See that module for why.
//!
//! The points here are the ones environmental injection cannot reach: a fold
//! creates its own staging directory, so "this ONE file inside it fails to
//! write" has no filesystem expression.

/// A point in a durable write this crate performs that a test can make fail.
///
/// Always compiled, feature or not, so [`fp!`] type-checks the variant name in
/// every build.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WriteFailpoint {
    /// Writing a segment's `versions.bin` while a fold or flush builds it.
    VersionsSidecarWrite,
    /// Appending a record to the delta WAL.
    DeltaWalAppend,
}

impl WriteFailpoint {
    /// This point's bit in the armed mask. Exhaustive on purpose: a new
    /// variant does not compile until it is given a bit.
    #[cfg(any(test, feature = "failpoints"))]
    const fn bit(self) -> u64 {
        match self {
            WriteFailpoint::VersionsSidecarWrite => 1 << 0,
            WriteFailpoint::DeltaWalAppend => 1 << 1,
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
        /// inside a rayon pool.
        static ARMED: Cell<u64> = const { Cell::new(0) };
    }

    /// Make `fp` fail on THIS THREAD until it is disarmed.
    pub fn arm(fp: WriteFailpoint) {
        ARMED.with(|a| a.set(a.get() | fp.bit()));
    }

    /// Stop `fp` failing.
    pub fn disarm(fp: WriteFailpoint) {
        ARMED.with(|a| a.set(a.get() & !fp.bit()));
    }

    /// Disarm every point. Cheap insurance at the end of a test.
    pub fn disarm_all() {
        ARMED.with(|a| a.set(0));
    }

    /// Is `fp` armed? Called by [`crate::fp`], not usually by hand.
    #[must_use]
    pub fn armed(fp: WriteFailpoint) -> bool {
        ARMED.with(Cell::get) & fp.bit() != 0
    }
}

#[cfg(any(test, feature = "failpoints"))]
pub use armed_state::{arm, armed, disarm, disarm_all};

/// Fail at `$fp` with `$err` when the point is armed.
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
    use super::{WriteFailpoint, arm, armed, disarm_all};

    fn guarded(fp: WriteFailpoint) -> std::io::Result<()> {
        crate::fp!(
            fp,
            Err(std::io::Error::other("failpoint: sidecar write refused"))
        );
        Ok(())
    }

    /// One test, not two: the mask is per thread, so splitting this would
    /// only produce two tests that cannot see each other.
    #[test]
    fn points_fire_only_while_armed_and_only_the_one_armed() {
        disarm_all();
        assert!(guarded(WriteFailpoint::VersionsSidecarWrite).is_ok());
        arm(WriteFailpoint::VersionsSidecarWrite);
        assert!(guarded(WriteFailpoint::VersionsSidecarWrite).is_err());
        assert!(!armed(WriteFailpoint::DeltaWalAppend));
        disarm_all();
        assert!(guarded(WriteFailpoint::VersionsSidecarWrite).is_ok());
    }
}
