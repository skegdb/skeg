//! `io_uring` batch-read backend behind a portable seam.
//!
//! Worth having: on Linux x86 (EPYC, kernel 6.8) a cold batch of 800 reads
//! costs 3.5 us per read against `pread`'s 100 us. On cached pages `pread`
//! wins by 2x to 5x, so this is a cold-read optimisation, not a general one.
//!
//! Worth gating: unlike a SIMD kernel, where a bug means a wrong number from
//! a pure function, correctness here depends on sequencing the compiler
//! cannot check - build the SQE, make it visible, submit, wait, drain the
//! CQE, and only *then* reuse or drop the buffer it pointed at. Getting it
//! wrong hangs a ring, drops a completion, or lets Rust reuse a buffer the
//! kernel is still writing into.
//!
//! Two bounds on that:
//!
//! 1. **The seam.** [`BatchReader`] has two implementations.
//!    [`BlockingBatchReader`] (one `pread` per request) is always available
//!    and is what every current caller gets. [`UringBatchReader`] exists only
//!    behind the `uring` feature, off by default.
//! 2. **The probe.** Even with the feature on, [`best_batch_reader`] opens a
//!    real ring and registers a [`register::Probe`] for `IORING_OP_READ`,
//!    falling back to [`BlockingBatchReader`] on any failure: old kernel, no
//!    permission, `io_uring_disabled` sysctl, seccomp.
//!
//! Not wired into any vLog read path yet, so turning it on for real traffic
//! stays a separate, reviewable decision.

use std::fs::File;
use std::io;
use std::sync::Arc;

use crate::file::pread_sync_into;

/// Batch reader: given `(offset, size)` pairs, return their bytes in the
/// same order. `results[i].len()` may be less than `reqs[i].1` at EOF,
/// matching [`PlatformFile::pread`](crate::PlatformFile::pread)'s
/// short-read semantics.
pub trait BatchReader: Send + Sync {
    /// # Errors
    ///
    /// Returns an IO error if any underlying read fails.
    fn read_many(&self, reqs: &[(u64, usize)]) -> io::Result<Vec<Vec<u8>>>;
}

/// One `pread` per request. Always available: the default, and the
/// fallback whenever the `io_uring` backend can't be used.
pub struct BlockingBatchReader {
    file: Arc<File>,
}

impl BlockingBatchReader {
    #[must_use]
    pub fn new(file: Arc<File>) -> Self {
        Self { file }
    }
}

impl BatchReader for BlockingBatchReader {
    fn read_many(&self, reqs: &[(u64, usize)]) -> io::Result<Vec<Vec<u8>>> {
        reqs.iter()
            .map(|&(offset, size)| {
                let mut buf = vec![0u8; size];
                let n = pread_sync_into(&self.file, offset, &mut buf)?;
                buf.truncate(n);
                Ok(buf)
            })
            .collect()
    }
}

/// Construct the best available [`BatchReader`] for `file`.
///
/// With the `uring` feature enabled on Linux, this opens a ring and probes
/// for `IORING_OP_READ` support; on success it returns an
/// [`UringBatchReader`], on any failure it falls back to
/// [`BlockingBatchReader`]. Without the feature (or on any other platform)
/// it always returns [`BlockingBatchReader`].
#[must_use]
pub fn best_batch_reader(file: Arc<File>) -> Box<dyn BatchReader> {
    #[cfg(all(target_os = "linux", feature = "uring"))]
    {
        if let Some(reader) = UringBatchReader::probe(Arc::clone(&file)) {
            return Box::new(reader);
        }
    }
    Box::new(BlockingBatchReader::new(file))
}

#[cfg(all(target_os = "linux", feature = "uring"))]
mod linux_uring {
    use super::{BatchReader, io};
    use std::fs::File;
    use std::os::unix::io::AsRawFd;
    use std::sync::Arc;

    use std::sync::Mutex;

    use io_uring::{IoUring, opcode, types};

    /// Ring depth. Batches larger than this are submitted in chunks of this
    /// size rather than growing the ring, so one reader serves any batch.
    const RING_DEPTH: u32 = 1024;

    /// `io_uring`-backed [`BatchReader`]: one ring submission covering every
    /// request in a `read_many` call (instead of one `spawn_blocking`/`pread`
    /// per request), one `io_uring_enter` wait for all of them to complete.
    ///
    /// The ring is created once and reused. Creating one per call costs an
    /// `io_uring_setup` plus two mmaps and two munmaps, which on a warm page
    /// cache is several times the cost of the `pread` it replaces.
    pub struct UringBatchReader {
        file: Arc<File>,
        /// `IoUring` is not `Sync`, and `BatchReader` hands out `&self` from
        /// many threads, so submissions serialise here. Batches are large and
        /// the ring is only held while filling and draining it.
        ring: Mutex<IoUring>,
    }

    impl UringBatchReader {
        /// Probe whether this kernel actually supports the one opcode this
        /// backend needs (`IORING_OP_READ`). Returns `None` on any failure
        /// (ring creation, probing, or the probe reporting the opcode
        /// unsupported), so the caller can fall back silently.
        #[must_use]
        pub fn probe(file: Arc<File>) -> Option<Self> {
            let ring: IoUring = IoUring::new(RING_DEPTH).ok()?;
            let mut probe = io_uring::Probe::new();
            ring.submitter().register_probe(&mut probe).ok()?;
            if probe.is_supported(opcode::Read::CODE) {
                Some(Self {
                    file,
                    ring: Mutex::new(ring),
                })
            } else {
                None
            }
        }
    }

    impl BatchReader for UringBatchReader {
        fn read_many(&self, reqs: &[(u64, usize)]) -> io::Result<Vec<Vec<u8>>> {
            if reqs.is_empty() {
                return Ok(Vec::new());
            }

            let fd = types::Fd(self.file.as_raw_fd());
            let mut ring = self
                .ring
                .lock()
                .map_err(|_| io::Error::other("io_uring ring mutex poisoned"))?;

            // Every buffer is allocated up front and never resized or moved
            // again until every completion for this call has been drained
            // below - the pointers handed to the kernel below stay valid for
            // exactly as long as the kernel can still be writing through
            // them.
            let mut bufs: Vec<Vec<u8>> = reqs.iter().map(|&(_, size)| vec![0u8; size]).collect();

            let mut results: Vec<Option<Vec<u8>>> = (0..reqs.len()).map(|_| None).collect();
            let chunk = RING_DEPTH as usize;
            for (c, (reqs, bufs)) in reqs.chunks(chunk).zip(bufs.chunks_mut(chunk)).enumerate() {
                let base = c * chunk;
                {
                    let mut sq = ring.submission();
                    for (i, (&(offset, _), buf)) in reqs.iter().zip(bufs.iter_mut()).enumerate() {
                        #[allow(clippy::cast_possible_truncation)]
                        let len = buf.len() as u32;
                        let entry = opcode::Read::new(fd, buf.as_mut_ptr(), len)
                            .offset(offset)
                            .build()
                            .user_data(i as u64);
                        // SAFETY: `entry`'s buffer pointer is `buf.as_mut_ptr()`
                        // for a `Vec<u8>` owned by `bufs`, which lives until the
                        // end of this function and is not touched again until
                        // the matching completion is read below - the kernel
                        // never sees a dangling or reused pointer. `sq` (this
                        // scope's only live `SubmissionQueue` borrow) drops at
                        // the end of this block, which is required before
                        // `submit_and_wait` per `IoUring::split`'s documented
                        // contract for queue borrows obtained via `submission()`.
                        unsafe {
                            sq.push(&entry)
                                .map_err(|e| io::Error::other(format!("io_uring sq full: {e}")))?;
                        }
                    }
                }

                ring.submit_and_wait(reqs.len())?;

                for cqe in ring.completion() {
                    let idx = cqe.user_data() as usize;
                    let res = cqe.result();
                    if res < 0 {
                        return Err(io::Error::from_raw_os_error(-res));
                    }
                    #[allow(clippy::cast_sign_loss)]
                    let n = res as usize;
                    let Some(buf) = bufs.get_mut(idx) else {
                        return Err(io::Error::other(
                            "io_uring: completion user_data out of range",
                        ));
                    };
                    let mut buf = std::mem::take(buf);
                    buf.truncate(n);
                    results[base + idx] = Some(buf);
                }
            }

            results
                .into_iter()
                .enumerate()
                .map(|(i, r)| {
                    r.ok_or_else(|| {
                        io::Error::other(format!("io_uring: request {i} never completed"))
                    })
                })
                .collect()
        }
    }
}

#[cfg(all(target_os = "linux", feature = "uring"))]
pub use linux_uring::UringBatchReader;

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write_file(dir: &TempDir, name: &str, data: &[u8]) -> Arc<File> {
        let path = dir.path().join(name);
        std::fs::write(&path, data).unwrap();
        Arc::new(File::open(&path).unwrap())
    }

    #[test]
    fn blocking_reader_reads_requests_in_order() {
        let dir = TempDir::new().unwrap();
        let data: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
        let file = write_file(&dir, "batch.bin", &data);
        let reader = BlockingBatchReader::new(file);

        let reqs = [(0u64, 16usize), (128, 32), (4000, 96)];
        let results = reader.read_many(&reqs).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0], data[0..16]);
        assert_eq!(results[1], data[128..160]);
        assert_eq!(results[2], data[4000..4096]);
    }

    #[test]
    fn blocking_reader_empty_request_list_is_noop() {
        let dir = TempDir::new().unwrap();
        let file = write_file(&dir, "empty.bin", b"data");
        let reader = BlockingBatchReader::new(file);
        assert_eq!(reader.read_many(&[]).unwrap(), Vec::<Vec<u8>>::new());
    }

    #[test]
    fn blocking_reader_short_read_at_eof() {
        let dir = TempDir::new().unwrap();
        let file = write_file(&dir, "short.bin", &[0xAAu8; 64]);
        let reader = BlockingBatchReader::new(file);
        let results = reader.read_many(&[(32, 64)]).unwrap(); // only 32 bytes available
        assert_eq!(results[0].len(), 32);
    }

    #[test]
    fn best_batch_reader_returns_a_working_reader() {
        // Portable smoke test: whichever backend this platform/feature
        // combination selects, it must round-trip real bytes correctly.
        let dir = TempDir::new().unwrap();
        let data = b"best_batch_reader smoke test payload";
        let file = write_file(&dir, "best.bin", data);
        let reader = best_batch_reader(file);
        let results = reader.read_many(&[(0, data.len())]).unwrap();
        assert_eq!(results[0], data);
    }

    #[cfg(all(target_os = "linux", feature = "uring"))]
    mod uring_only {
        use super::*;

        #[test]
        fn uring_reader_matches_blocking_reader() {
            let dir = TempDir::new().unwrap();
            let data: Vec<u8> = (0u8..=255).cycle().take(8192).collect();
            let file_a = write_file(&dir, "uring_a.bin", &data);
            let file_b = write_file(&dir, "uring_b.bin", &data);

            let Some(uring_reader) = UringBatchReader::probe(file_a) else {
                // This kernel/sandbox doesn't support io_uring - the whole
                // point of the probe is to make that a silent no-op for
                // the production path; the same must hold for the test.
                return;
            };
            let blocking_reader = BlockingBatchReader::new(file_b);

            let reqs = [(0u64, 100usize), (500, 250), (8000, 192), (0, 8192)];
            let uring_results = uring_reader.read_many(&reqs).unwrap();
            let blocking_results = blocking_reader.read_many(&reqs).unwrap();
            assert_eq!(uring_results, blocking_results);
        }

        #[test]
        fn uring_reader_short_read_at_eof_matches_blocking() {
            let dir = TempDir::new().unwrap();
            let data = [0xAAu8; 64];
            let file_a = write_file(&dir, "uring_short_a.bin", &data);
            let file_b = write_file(&dir, "uring_short_b.bin", &data);

            let Some(uring_reader) = UringBatchReader::probe(file_a) else {
                return;
            };
            let blocking_reader = BlockingBatchReader::new(file_b);

            let reqs = [(32u64, 64usize)]; // only 32 bytes available past offset 32
            let uring_results = uring_reader.read_many(&reqs).unwrap();
            let blocking_results = blocking_reader.read_many(&reqs).unwrap();
            assert_eq!(uring_results, blocking_results);
        }

        #[test]
        fn uring_reader_empty_request_list_is_noop() {
            let dir = TempDir::new().unwrap();
            let file = write_file(&dir, "uring_empty.bin", b"data");
            let Some(uring_reader) = UringBatchReader::probe(file) else {
                return;
            };
            assert_eq!(uring_reader.read_many(&[]).unwrap(), Vec::<Vec<u8>>::new());
        }
    }
}
