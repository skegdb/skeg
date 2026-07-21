//! Advisory whole-directory lock, so two processes cannot open the same store
//! at once. Concurrent opens would let both append to the same value log and
//! diverge their in-memory key indexes, corrupting the segment files.
//!
//! Backed by `flock(2)`. The lock is bound to the open file description, and the
//! kernel drops it when the process exits, a crash included. So a crashed
//! process leaves no stale lock and the next open acquires cleanly. A
//! lock-by-existence marker file would survive the crash and need manual
//! cleanup or a PID-liveness check.
//!
//! The lock is advisory: it only excludes other processes that also take it.
//! skeg takes it on every store open, so two skeg instances coordinate; a
//! foreign process writing the files raw is not stopped. That matches the threat
//! model, a second skeg on the same directory, not arbitrary writers.
//!
//! Semantics are reliable on local filesystems (APFS, ext4, xfs, btrfs). On
//! networked filesystems (NFS, SMB) `flock` behaviour has historically varied,
//! so those are out of the supported set; a caller can check the filesystem
//! before relying on the lock.

use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::io::AsRawFd;
use std::path::Path;

/// Filename of the lock inside a store directory. Not a segment (`.seg`), so
/// `list_segments` ignores it.
pub const LOCK_FILE: &str = ".skeg.lock";

/// Holds a store's advisory lock for as long as the store is open. Dropping it,
/// or the process exiting, releases the lock.
#[derive(Debug)]
pub struct DirLock {
    // Keeping the `File` alive keeps its fd open, which keeps the flock held.
    _file: File,
}

impl DirLock {
    /// Take an exclusive advisory lock on `LOCK_FILE` inside `dir` (created if
    /// absent). `dir` must already exist.
    ///
    /// # Errors
    ///
    /// Returns [`io::ErrorKind::WouldBlock`] if another process already holds
    /// the lock (the store is open elsewhere), or an I/O error if the lock file
    /// cannot be opened.
    pub fn acquire_exclusive(dir: &Path) -> io::Result<Self> {
        let path = dir.join(LOCK_FILE);
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false) // lock file holds no content; never clobber
            .open(&path)?;
        // SAFETY: `flock` is a syscall on a valid fd owned by `file`. LOCK_NB
        // makes it non-blocking: it returns 0 on success, or -1 with errno
        // EWOULDBLOCK when another open file description already holds the lock.
        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc != 0 {
            let err = io::Error::last_os_error();
            if err.raw_os_error() == Some(libc::EWOULDBLOCK) {
                return Err(io::Error::new(
                    io::ErrorKind::WouldBlock,
                    format!("store already open by another process: {}", dir.display()),
                ));
            }
            return Err(err);
        }
        Ok(Self { _file: file })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn second_acquire_on_same_dir_is_refused() {
        let dir = tempfile::TempDir::new().unwrap();
        let _held = DirLock::acquire_exclusive(dir.path()).unwrap();
        // A second live handle on the same directory (this process stands in for
        // a second one; flock is per open file description, so it contends).
        let err = DirLock::acquire_exclusive(dir.path()).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::WouldBlock);
    }

    #[test]
    fn releases_on_drop() {
        let dir = tempfile::TempDir::new().unwrap();
        {
            let _held = DirLock::acquire_exclusive(dir.path()).unwrap();
        }
        // Prior handle dropped, so the lock is free again.
        DirLock::acquire_exclusive(dir.path()).expect("lock free after drop");
    }
}
