#![deny(unsafe_code)]

//! Typed commit-path failpoints.
//!
//! The committers turn a batch of appends into one write and one sync. A test
//! that wants to prove the *failure* half of that contract - that a flush
//! which could not write, or could not sync, says so instead of answering
//! `Ok(())` - needs to make exactly one of those two syscalls fail on demand.
//! Environmental injection (a read-only directory, `chmod 000`, a full disk)
//! cannot express it: both steps use the same descriptor, and the interesting
//! case is the one where the write lands and the sync does not.
//!
//! So: a registry, shaped like `skeg-server`'s `failpoint.rs`, with the same
//! three properties and one difference.
//!
//! - **Typed, not string-keyed.** The name is an enum variant, so a renamed
//!   site stops compiling instead of quietly never firing.
//! - **Absent from a normal build.** Without `cfg(any(test, feature =
//!   "failpoints"))` the arming state does not exist and [`hit`] is a `const
//!   false`, so the commit path carries no branch - and the variant name is
//!   still type-checked in every build.
//! - **`fired` is recorded and asserted.** An armed point that never fires
//!   makes its test pass for the wrong reason: the flush succeeds, the
//!   assertion about the aftermath holds trivially, and nothing says the
//!   window was entered.
//!
//! The difference is the key. `skeg-server` keys on the vindex name, a string
//! that nothing but convention keeps unique between two tests in one binary.
//! Here the key is the FILE the committer is about to write: its
//! `Arc<PlatformFile>` address, taken with [`file_key`]. Both the site and the
//! test hold the same `Arc`, so the key is unique by construction, needs no
//! plumbing through the committer, and cannot be accidentally shared by two
//! tests. It is meaningless as a name and never printed as one - it exists
//! only to answer "is this the file the test armed?".
//!
//! Keying by file is also what the sites need: a per-file committer owns one
//! file, and the shared committer's batch touches several, so "fail the write
//! of THIS file, but not that one" is exactly the aggregation case.

use skeg_platform::PlatformFile;

/// A step of a batch commit that a test can make fail.
///
/// Always compiled, feature or not, so [`hit`] type-checks the variant name in
/// every build.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommitFailpoint {
    /// The `write_vectored_at` that puts a per-file committer's whole batch on
    /// disk. Nothing is acked yet, so this is a plain failure: every waiter in
    /// the batch is owed an error and the write offset must not advance.
    PerFileBatchWrite,
    /// The `sync_data`/`sync_durable` that makes a per-file committer's batch
    /// durable. The bytes ARE on disk; what failed is the proof that they
    /// survive. Waiters are owed an error all the same.
    PerFileBatchSync,
    /// One file's combined `write_at` inside the shared (device-global)
    /// committer. Only that file's waiters are affected - the other files in
    /// the same batch still commit, which is the case the flush result has to
    /// aggregate.
    SharedBatchWrite,
    /// The single device-wide sync that covers every file a shared-committer
    /// batch wrote.
    SharedBatchSync,
}

impl CommitFailpoint {
    /// This point's bit in the registry. Exhaustive on purpose: a new variant
    /// does not compile until it is given a bit.
    #[cfg(any(test, feature = "failpoints"))]
    const fn bit(self) -> u64 {
        match self {
            CommitFailpoint::PerFileBatchWrite => 1 << 0,
            CommitFailpoint::PerFileBatchSync => 1 << 1,
            CommitFailpoint::SharedBatchWrite => 1 << 2,
            CommitFailpoint::SharedBatchSync => 1 << 3,
        }
    }
}

/// The registry key for `file`: the address of the `PlatformFile` the
/// committer is about to write.
///
/// Stable for as long as the `Arc` lives, which is longer than any batch that
/// names it, and unique across live files by construction. Only ever compared,
/// never displayed.
#[cfg(any(test, feature = "failpoints"))]
#[must_use]
pub fn file_key(file: &PlatformFile) -> u64 {
    std::ptr::from_ref(file) as u64
}

/// Serialises the tests that assert on the process-wide flush-failure counter.
///
/// The counter is one atomic for the whole process, so two tests in the same
/// binary that both make a flush fail see each other's ticks. Every test that
/// makes a flush fail holds this first, whether or not it reads the counter -
/// a test that only injects a failure still moves the number a concurrent
/// test is measuring.
///
/// Async, because a test holds it across the awaits that drive the committer;
/// a `std::sync::Mutex` there is the deadlock `await_holding_lock` denies.
#[cfg(any(test, feature = "failpoints"))]
pub static FLUSH_FAILURE_COUNTER_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[cfg(any(test, feature = "failpoints"))]
mod state {
    use super::CommitFailpoint;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// One armed (point, file) pair.
    struct Entry {
        bit: u64,
        key: u64,
        armed: bool,
        fired: bool,
    }

    /// Bits with at least one file armed anywhere in the process.
    ///
    /// One relaxed load on a disarmed point; the mutex below is touched only
    /// once a test has armed that exact point.
    static ANY: AtomicU64 = AtomicU64::new(0);
    static ENTRIES: Mutex<Vec<Entry>> = Mutex::new(Vec::new());

    fn with<R>(f: impl FnOnce(&mut Vec<Entry>) -> R) -> R {
        // A poisoned lock here means a test panicked mid-assertion; the state
        // is a few booleans and recovering it is strictly better than turning
        // every later test in the binary into a panic about the first one.
        let mut g = ENTRIES
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        f(&mut g)
    }

    fn refresh_any(entries: &[Entry]) {
        let mask = entries.iter().filter(|e| e.armed).fold(0, |m, e| m | e.bit);
        ANY.store(mask, Ordering::Relaxed);
    }

    /// Make `fp` fail for the file `key` names, until it is disarmed. Clears
    /// its fired flag, so [`fired_at`] answers about this arming.
    ///
    /// # Panics
    ///
    /// If `fp` is ALREADY armed for `key`. Two live `Arc<PlatformFile>`s never
    /// share an address, so this can only mean the same test armed twice
    /// without disarming - and then "did it fire?" is a question about which
    /// arming.
    pub fn arm_at(fp: CommitFailpoint, key: u64) {
        with(|entries| {
            match entries
                .iter_mut()
                .find(|e| e.bit == fp.bit() && e.key == key)
            {
                Some(e) => {
                    assert!(!e.armed, "{fp:?} is already armed for this file");
                    e.armed = true;
                    e.fired = false;
                }
                None => entries.push(Entry {
                    bit: fp.bit(),
                    key,
                    armed: true,
                    fired: false,
                }),
            }
            refresh_any(entries);
        });
    }

    /// Stop `fp` failing for `key`. The fired flag is left alone: a test
    /// disarms before it asserts.
    pub fn disarm_at(fp: CommitFailpoint, key: u64) {
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

    /// Is `fp` armed for `key`? Records the hit.
    pub fn armed_at(fp: CommitFailpoint, key: u64) -> bool {
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
    pub fn fired_at(fp: CommitFailpoint, key: u64) -> bool {
        with(|entries| {
            entries
                .iter()
                .any(|e| e.bit == fp.bit() && e.key == key && e.fired)
        })
    }
}

#[cfg(any(test, feature = "failpoints"))]
pub use state::{arm_at, disarm_at, fired_at};

/// Is `fp` armed for the file `key` names?
///
/// A value, not a `return`: every commit site has waiters to ack and an offset
/// to leave alone, so the hit has to be turned into the same `io::Error` the
/// real syscall would have produced, not jumped over.
///
/// The enabled form. Records the hit.
#[cfg(any(test, feature = "failpoints"))]
#[must_use]
#[inline]
pub fn hit(fp: CommitFailpoint, key: u64) -> bool {
    state::armed_at(fp, key)
}

/// The disabled form: still names the variant, so a renamed point fails to
/// compile instead of quietly never firing, and folds to `false` at the call
/// site so the commit path carries no branch.
#[cfg(not(any(test, feature = "failpoints")))]
#[must_use]
#[inline(always)]
pub fn hit(fp: CommitFailpoint, key: u64) -> bool {
    let _ = (fp, key);
    false
}

/// The key of the file a disabled build would have looked up.
///
/// Present so the call sites read the same either way; the value is never
/// used, because [`hit`] ignores it.
#[cfg(not(any(test, feature = "failpoints")))]
#[must_use]
#[inline(always)]
pub fn file_key(file: &PlatformFile) -> u64 {
    let _ = file;
    0
}

/// The `io::Error` a fired commit failpoint stands in for.
///
/// One shape for every point, and deliberately `ErrorKind::Other`: a caller
/// that special-cases `StorageFull` to surface ENOSPC to its own user must not
/// be reachable from an armed failpoint, or a test could pass down a branch no
/// injected failure is allowed to take.
#[must_use]
pub(crate) fn injected(what: &'static str) -> std::io::Error {
    std::io::Error::other(format!("commit failpoint: {what}"))
}
