//! Platform-optimised file I/O.
//!
//! On macOS: `F_NOCACHE` disables the OS page cache; `F_FULLFSYNC` ensures data
//! reaches hardware storage (stronger than `fsync`) - used unconditionally,
//! since it is a device-wide write barrier regardless of whether the file's
//! size is stable.
//! On Linux (and other non-macOS Unix): `sync_durable` defaults to `fsync`
//! (via `sync_all`, which also commits inode metadata such as file length).
//! A handle whose size has been fixed with [`PlatformFile::preallocate`]
//! never changes length again, so there is no metadata to re-commit on
//! later flushes; for such a handle `sync_durable` downgrades to
//! `fdatasync` (`sync_data`) on Linux - as durable for the bytes actually
//! written, without the journal commit.
//!
//! Async methods offload blocking syscalls to `tokio::task::spawn_blocking`.
//! Sync methods are thin wrappers for use inside `spawn_blocking` closures or
//! recovery code that already runs on a blocking thread.
//!
//! unsafe here is intentional for fcntl/fallocate syscalls - see SAFETY comments.

use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

#[cfg(target_os = "linux")]
use std::os::fd::AsRawFd;
#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(target_os = "macos")]
use std::os::unix::io::AsRawFd;

/// Buffer alignment recommended for this platform's DMA requirements.
#[cfg(target_os = "macos")]
pub const BUFFER_ALIGNMENT: usize = 16_384; // 16 KB Apple Silicon DMA
#[cfg(not(target_os = "macos"))]
pub const BUFFER_ALIGNMENT: usize = 4_096;

/// Platform-optimised file handle.
///
/// Wraps `Arc<File>` so the handle can be cloned cheaply and passed into
/// `spawn_blocking` closures without lifetime constraints.
#[derive(Clone)]
pub struct PlatformFile {
    inner: Arc<File>,
    /// Number of `sync_durable` calls - useful for testing batch behaviour.
    sync_count: Arc<AtomicU64>,
    /// Number of `pread` calls - useful for testing cache behaviour.
    read_count: Arc<AtomicU64>,
    /// Set once [`preallocate`](Self::preallocate) has fixed this file's
    /// length. `sync_durable` reads this to pick `fdatasync` over `fsync`
    /// on Linux.
    size_fixed: Arc<AtomicBool>,
}

impl PlatformFile {
    /// Create a new file (fails if it already exists) and apply `F_NOCACHE`.
    ///
    /// # Errors
    ///
    /// Returns an IO error if the file cannot be created or `F_NOCACHE` fails.
    pub fn create(path: &Path) -> io::Result<Self> {
        let mut opts = OpenOptions::new();
        opts.create_new(true).read(true).write(true);
        // Data/WAL segments hold every tenant's KV and vector values. Without an
        // explicit mode they inherit the umask (typically 0644), leaving them
        // world-readable on a shared host. Owner-only, matching the auth store.
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let file = opts.open(path)?;
        let pf = Self {
            inner: Arc::new(file),
            sync_count: Arc::new(AtomicU64::new(0)),
            read_count: Arc::new(AtomicU64::new(0)),
            size_fixed: Arc::new(AtomicBool::new(false)),
        };
        pf.apply_nocache()?;
        Ok(pf)
    }

    /// Open an existing file for read+write and apply `F_NOCACHE`.
    ///
    /// # Errors
    ///
    /// Returns an IO error if the file does not exist or `F_NOCACHE` fails.
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut opts = OpenOptions::new();
        opts.read(true).write(true);
        // Segment/WAL files in the data dir are never symlinks in normal
        // operation. Refusing to follow one closes a local symlink-swap attack
        // where an attacker with write access to the data dir points a segment
        // at a victim file we then truncate/extend with the process's rights.
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.custom_flags(libc::O_NOFOLLOW);
        }
        let file = opts.open(path)?;
        let pf = Self {
            inner: Arc::new(file),
            sync_count: Arc::new(AtomicU64::new(0)),
            read_count: Arc::new(AtomicU64::new(0)),
            size_fixed: Arc::new(AtomicBool::new(false)),
        };
        pf.apply_nocache()?;
        Ok(pf)
    }

    /// Current file size in bytes.
    ///
    /// # Errors
    ///
    /// Returns an IO error if the metadata cannot be read.
    pub fn size(&self) -> io::Result<u64> {
        self.inner.metadata().map(|m| m.len())
    }

    /// Number of times `sync_durable` has been called on this handle.
    #[must_use]
    pub fn sync_count(&self) -> u64 {
        self.sync_count.load(Ordering::Relaxed)
    }

    /// Number of `pread` calls issued on this handle.
    #[must_use]
    pub fn read_count(&self) -> u64 {
        self.read_count.load(Ordering::Relaxed)
    }

    /// True once [`preallocate`](Self::preallocate) has fixed this handle's
    /// file length.
    #[must_use]
    pub fn is_size_fixed(&self) -> bool {
        self.size_fixed.load(Ordering::Relaxed)
    }

    // ── Async API ─────────────────────────────────────────────────────────────

    /// Read up to `size` bytes at `offset`. Returns fewer bytes at EOF.
    ///
    /// # Errors
    ///
    /// Returns an IO error on read failure or if `spawn_blocking` is not available.
    pub async fn pread(&self, offset: u64, size: usize) -> io::Result<Vec<u8>> {
        self.read_count.fetch_add(1, Ordering::Relaxed);
        let file = self.inner.clone();
        tokio::task::spawn_blocking(move || pread_sync(&file, offset, size)).await?
    }

    /// Write all of `data` at `offset` using `pwrite` (seekless).
    ///
    /// # Errors
    ///
    /// Returns an IO error on write failure.
    pub async fn write_at(&self, offset: u64, data: Vec<u8>) -> io::Result<()> {
        let file = self.inner.clone();
        tokio::task::spawn_blocking(move || write_at_sync(&file, offset, &data)).await?
    }

    /// Write every buffer in `chunks` back-to-back starting at `offset`,
    /// without first copying them into one contiguous buffer.
    ///
    /// Linux: a single `pwritev` call per retry (retried only on `EINTR` or
    /// a short write, both rare for a regular file). Other platforms:
    /// sequential `pwrite` calls, one per buffer - still no combined-buffer
    /// copy, just more syscalls than the Linux path.
    ///
    /// # Errors
    ///
    /// Returns an IO error on write failure.
    pub async fn write_vectored_at(&self, offset: u64, chunks: Vec<Vec<u8>>) -> io::Result<()> {
        let file = self.inner.clone();
        tokio::task::spawn_blocking(move || write_vectored_at_sync(&file, offset, &chunks)).await?
    }

    /// Ask the kernel to start writeback for `[offset, offset + len)`
    /// without waiting for it to finish; `len == 0` means "through the
    /// current end of file". Linux only (`sync_file_range` with
    /// `SYNC_FILE_RANGE_WRITE`); a no-op elsewhere.
    ///
    /// This is *not* a durability primitive - it gives no guarantee the
    /// data has reached storage, only that the kernel has started pushing
    /// the dirty pages toward it. Use it to pace writeback ahead of a real
    /// durability call ([`sync_durable`](Self::sync_durable) /
    /// [`sync_data`](Self::sync_data)) on a large run of low-durability-tier
    /// writes (e.g. compaction relocating many records at
    /// `Durability::Relaxed`), so dirty pages do not pile up unbounded
    /// until the eventual flush is forced to push them all out at once.
    ///
    /// # Errors
    ///
    /// Returns an IO error if the underlying syscall fails. Callers that
    /// only want the pacing effect (not correctness) may choose to ignore
    /// the result - the real durability call that should follow still
    /// gives the actual guarantee either way.
    pub async fn hint_writeback(&self, offset: u64, len: u64) -> io::Result<()> {
        let file = self.inner.clone();
        tokio::task::spawn_blocking(move || hint_writeback_sync(&file, offset, len)).await?
    }

    /// Flush to hardware storage - power-loss durable.
    ///
    /// Uses `F_FULLFSYNC` on macOS. Elsewhere: `sync_all` (`fsync`), or
    /// `sync_data` (`fdatasync`) on Linux once [`preallocate`](Self::preallocate)
    /// has fixed this handle's length - see the module docs.
    ///
    /// # Errors
    ///
    /// Returns an IO error if the flush fails.
    pub async fn sync_durable(&self) -> io::Result<()> {
        let file = self.inner.clone();
        let counter = self.sync_count.clone();
        let size_fixed = self.is_size_fixed();
        tokio::task::spawn_blocking(move || {
            let result = sync_durable_sync_inner(&file, size_fixed);
            if result.is_ok() {
                counter.fetch_add(1, Ordering::Relaxed);
            }
            result
        })
        .await?
    }

    /// Preallocate the file to `len` bytes so later writes within that range
    /// never change the file's length - the durability call for a file whose
    /// length is fixed only needs to flush data, not inode metadata.
    ///
    /// Linux: `fallocate` (mode 0), which reserves the blocks *and* extends
    /// the reported length to `len` immediately (unlike `FALLOC_FL_KEEP_SIZE`).
    /// macOS: `F_PREALLOCATE` (contiguous, falling back to any extent) reserves
    /// the blocks; `set_len` then bumps the reported length, since
    /// `F_PREALLOCATE` alone does not move EOF. Other platforms: `set_len`
    /// alone (a sparse file, no block reservation).
    ///
    /// After this call succeeds, [`sync_durable`](Self::sync_durable) uses
    /// `fdatasync` instead of `fsync` on Linux. The caller must never write
    /// past `len` on this handle for that to stay sound - skeg-core's vLog
    /// enforces this by rotating to a fresh segment before a write would
    /// exceed the preallocated size.
    ///
    /// # Errors
    ///
    /// Returns an IO error if the underlying syscall fails.
    pub async fn preallocate(&self, len: u64) -> io::Result<()> {
        let file = self.inner.clone();
        let flag = self.size_fixed.clone();
        tokio::task::spawn_blocking(move || {
            preallocate_file(&file, len)?;
            flag.store(true, Ordering::Relaxed);
            Ok(())
        })
        .await?
    }

    /// Flush file data - kernel-crash durable, *not* power-loss durable.
    ///
    /// Uses `fsync` on macOS / `fdatasync` on Linux. Cheaper than
    /// [`sync_durable`](Self::sync_durable): it does not force the drive's
    /// write cache out to the storage media.
    ///
    /// # Errors
    ///
    /// Returns an IO error if the flush fails.
    pub async fn sync_data(&self) -> io::Result<()> {
        let file = self.inner.clone();
        let counter = self.sync_count.clone();
        tokio::task::spawn_blocking(move || {
            let result = file.sync_data();
            if result.is_ok() {
                counter.fetch_add(1, Ordering::Relaxed);
            }
            result
        })
        .await?
    }

    /// Truncate file to `len` bytes.
    ///
    /// # Errors
    ///
    /// Returns an IO error on failure.
    pub async fn truncate(&self, len: u64) -> io::Result<()> {
        let file = self.inner.clone();
        tokio::task::spawn_blocking(move || file.set_len(len)).await?
    }

    // ── Sync API (for use inside spawn_blocking or recovery) ──────────────────

    /// Sync read at `offset` into `buf`. Returns bytes actually read.
    ///
    /// # Errors
    ///
    /// Returns an IO error on read failure.
    pub fn pread_sync(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        pread_sync_into(&self.inner, offset, buf)
    }

    /// Sync pwrite at `offset`.
    ///
    /// # Errors
    ///
    /// Returns an IO error on write failure.
    pub fn write_at_sync(&self, offset: u64, data: &[u8]) -> io::Result<()> {
        write_at_sync(&self.inner, offset, data)
    }

    /// Sync vectored write. See [`write_vectored_at`](Self::write_vectored_at).
    ///
    /// # Errors
    ///
    /// Returns an IO error on write failure.
    pub fn write_vectored_at_sync(&self, offset: u64, chunks: &[Vec<u8>]) -> io::Result<()> {
        write_vectored_at_sync(&self.inner, offset, chunks)
    }

    /// Sync flush to hardware.
    ///
    /// # Errors
    ///
    /// Returns an IO error if the flush fails.
    pub fn sync_durable_sync(&self) -> io::Result<()> {
        let result = sync_durable_sync_inner(&self.inner, self.is_size_fixed());
        if result.is_ok() {
            self.sync_count.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    /// Sync truncate.
    ///
    /// # Errors
    ///
    /// Returns an IO error on failure.
    pub fn truncate_sync(&self, len: u64) -> io::Result<()> {
        self.inner.set_len(len)
    }

    /// Sync preallocate, for use inside `spawn_blocking` or single-threaded
    /// recovery code. See [`preallocate`](Self::preallocate).
    ///
    /// # Errors
    ///
    /// Returns an IO error if the underlying syscall fails.
    pub fn preallocate_sync(&self, len: u64) -> io::Result<()> {
        preallocate_file(&self.inner, len)?;
        self.size_fixed.store(true, Ordering::Relaxed);
        Ok(())
    }

    fn apply_nocache(&self) -> io::Result<()> {
        #[cfg(target_os = "macos")]
        {
            // SAFETY: `F_NOCACHE` is a valid macOS fcntl command.
            // `self.inner` is open and the fd is valid. Return value is checked.
            let ret = unsafe { libc::fcntl(self.inner.as_raw_fd(), libc::F_NOCACHE, 1) };
            if ret < 0 {
                return Err(io::Error::last_os_error());
            }
        }
        Ok(())
    }
}

// ── Read-only memory map ──────────────────────────────────────────────────────

/// A read-only memory map of an existing file. Dereferences to the file's
/// bytes. The offline index build maps a large vector dataset through this so
/// it never copies the whole dataset into the heap.
pub struct MappedFile {
    mmap: memmap2::Mmap,
}

impl std::fmt::Debug for MappedFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MappedFile")
            .field("len", &self.mmap.len())
            .finish()
    }
}

impl MappedFile {
    /// Memory-map `path` read-only.
    ///
    /// # Errors
    ///
    /// Returns an IO error if the file cannot be opened or mapped.
    pub fn open(path: &Path) -> io::Result<MappedFile> {
        let file = File::open(path)?;
        // SAFETY: `Mmap::map` is unsafe because a concurrent writer or
        // truncation of the file would change the mapped bytes under us. skeg
        // maps a stable, already-written input dataset that is not modified
        // while a build runs, so the mapping stays valid for its lifetime.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Ok(MappedFile { mmap })
    }

    /// Memory-map `path` read-only, prefaulting page tables up front
    /// (`MAP_POPULATE` on Linux; a no-op elsewhere, same mapping as
    /// [`open`](Self::open)).
    ///
    /// Use at cold start for a mapping the very first query will touch: the
    /// read-ahead this triggers happens once, during startup, instead of as
    /// scattered page faults during the first requests served.
    ///
    /// # Errors
    ///
    /// Returns an IO error if the file cannot be opened or mapped.
    pub fn open_populated(path: &Path) -> io::Result<MappedFile> {
        let file = File::open(path)?;
        // SAFETY: same envelope as `open` - a stable, already-written input
        // dataset not modified while mapped. `populate()` only changes when
        // the kernel prefaults pages, not the safety contract of the mapping.
        let mmap = unsafe { memmap2::MmapOptions::new().populate().map(&file)? };
        Ok(MappedFile { mmap })
    }

    /// The mapped bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// Length of the mapped region in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    /// True if the mapped file is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }

    /// Hint the kernel that the mapping will be accessed in random order
    /// (`MADV_RANDOM`). Disables read-ahead so a greedy walk does not waste
    /// L1/L2 bandwidth on speculative neighbour pages it will not visit.
    ///
    /// No-op on platforms where `madvise` is unavailable.
    ///
    /// # Errors
    ///
    /// Propagates the underlying `madvise` failure (rare; mainly EINVAL on
    /// platforms that do not implement the hint).
    pub fn advise_random(&self) -> io::Result<()> {
        self.mmap.advise(memmap2::Advice::Random)
    }

    /// Hint the kernel that the mapping will be accessed sequentially
    /// (`MADV_SEQUENTIAL`). Enables aggressive read-ahead; useful at build
    /// time when the whole file is read once.
    ///
    /// # Errors
    ///
    /// Propagates the underlying `madvise` failure.
    pub fn advise_sequential(&self) -> io::Result<()> {
        self.mmap.advise(memmap2::Advice::Sequential)
    }

    /// Hint the kernel to back this mapping with transparent huge pages
    /// (`MADV_HUGEPAGE`) where the OS supports opting in. The Vamana walk's
    /// random access over quantized codes and graph adjacency is TLB-bound;
    /// 2 MiB pages cut TLB misses by roughly the ratio to the 4 KiB base
    /// page size for the same footprint.
    ///
    /// Linux only - `Advice::HugePage` does not exist on other platforms,
    /// so this is a no-op there (`Ok(())`).
    ///
    /// # Errors
    ///
    /// Propagates the underlying `madvise` failure (e.g. the kernel was
    /// built without `CONFIG_TRANSPARENT_HUGEPAGE`).
    pub fn advise_huge(&self) -> io::Result<()> {
        #[cfg(target_os = "linux")]
        {
            self.mmap.advise(memmap2::Advice::HugePage)
        }
        #[cfg(not(target_os = "linux"))]
        {
            Ok(())
        }
    }
}

impl std::ops::Deref for MappedFile {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.mmap
    }
}

/// Hint that a bounded range of `file` will be read sequentially.
///
/// Linux uses `POSIX_FADV_SEQUENTIAL` so the kernel may increase readahead for
/// maintenance scans. macOS already applies its file-cache policy at open and
/// continues to use mmap advice where a mapping is available, so this is a
/// no-op there and on other platforms.
///
/// # Errors
///
/// Returns an error when Linux rejects the range or the byte offsets cannot fit
/// its signed file-offset type.
pub fn advise_sequential_file(file: &File, offset: u64, len: u64) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        let offset = libc::off_t::try_from(offset).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "file advice offset is too large",
            )
        })?;
        let len = libc::off_t::try_from(len).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "file advice range is too large",
            )
        })?;
        // SAFETY: `file` owns a valid file descriptor for the duration of the
        // call. `posix_fadvise` only records a kernel hint and does not access
        // user memory. The converted offset and length are non-negative.
        let result = unsafe {
            libc::posix_fadvise(file.as_raw_fd(), offset, len, libc::POSIX_FADV_SEQUENTIAL)
        };
        if result != 0 {
            return Err(io::Error::from_raw_os_error(result));
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (file, offset, len);
    }
    Ok(())
}

// ── Free functions used in spawn_blocking closures ────────────────────────────

fn pread_sync(file: &File, offset: u64, size: usize) -> io::Result<Vec<u8>> {
    let mut buf = vec![0u8; size];
    let n = pread_sync_into(file, offset, &mut buf)?;
    buf.truncate(n);
    Ok(buf)
}

pub(crate) fn pread_sync_into(file: &File, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
    let mut pos = 0usize;
    while pos < buf.len() {
        #[cfg(unix)]
        let n = file.read_at(&mut buf[pos..], offset + pos as u64)?;
        #[cfg(not(unix))]
        let n = {
            use std::io::{Read, Seek, SeekFrom};
            let mut f = file.try_clone()?;
            f.seek(SeekFrom::Start(offset + pos as u64))?;
            f.read(&mut buf[pos..])?
        };
        if n == 0 {
            break;
        }
        pos += n;
    }
    Ok(pos)
}

fn write_at_sync(file: &File, offset: u64, data: &[u8]) -> io::Result<()> {
    let mut pos = 0usize;
    while pos < data.len() {
        #[cfg(unix)]
        let n = file.write_at(&data[pos..], offset + pos as u64)?;
        #[cfg(not(unix))]
        let n = {
            use std::io::{Seek, SeekFrom, Write};
            let mut f = file.try_clone()?;
            f.seek(SeekFrom::Start(offset + pos as u64))?;
            f.write(&data[pos..])?
        };
        if n == 0 {
            // A short write for a non-empty remaining buffer would spin
            // this loop forever without this guard, matching the same
            // defensive check `write_pwritev_linux` already has.
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "write_at wrote 0 bytes",
            ));
        }
        pos += n;
    }
    Ok(())
}

/// Write every buffer in `chunks` back-to-back starting at `offset`. See
/// [`PlatformFile::write_vectored_at`].
fn write_vectored_at_sync(file: &File, offset: u64, chunks: &[Vec<u8>]) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        write_pwritev_linux(file, offset, chunks)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let mut pos = offset;
        for chunk in chunks {
            write_at_sync(file, pos, chunk)?;
            pos += chunk.len() as u64;
        }
        Ok(())
    }
}

/// `pwritev`-based vectored write with the standard partial-write retry
/// loop: `pwritev` on a regular file almost always writes everything in one
/// call, but must still be retried on `EINTR` or a short write.
#[cfg(target_os = "linux")]
fn write_pwritev_linux(file: &File, mut offset: u64, chunks: &[Vec<u8>]) -> io::Result<()> {
    let mut chunk_idx = 0usize;
    let mut byte_in_chunk = 0usize;
    while chunk_idx < chunks.len() {
        let iovecs: Vec<libc::iovec> = chunks[chunk_idx..]
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let bytes = if i == 0 { &c[byte_in_chunk..] } else { &c[..] };
                libc::iovec {
                    // `iovec.iov_base` is `*mut c_void` because the type is
                    // shared with `readv`/`preadv`; `pwritev` only ever
                    // reads through it. `cast_mut` here does not grant real
                    // mutable access, it satisfies the C signature.
                    iov_base: bytes.as_ptr().cast_mut().cast(),
                    iov_len: bytes.len(),
                }
            })
            .collect();
        let off = libc::off_t::try_from(offset)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "pwritev offset too large"))?;
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let iovcnt = iovecs.len() as libc::c_int;
        // SAFETY: `file` owns a valid fd for the call's duration. Every
        // `iovec` points into a live slice of `chunks`, which outlives this
        // call (it is a `&[Vec<u8>]` borrowed from the caller); `iovcnt`
        // matches `iovecs.len()` exactly.
        let n = unsafe { libc::pwritev(file.as_raw_fd(), iovecs.as_ptr(), iovcnt, off) };
        if n < 0 {
            let err = io::Error::last_os_error();
            if err.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(err);
        }
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "pwritev wrote 0 bytes",
            ));
        }
        #[allow(clippy::cast_sign_loss)]
        let mut written = n as usize;
        offset += written as u64;
        while written > 0 {
            let remaining_in_chunk = chunks[chunk_idx].len() - byte_in_chunk;
            if written < remaining_in_chunk {
                byte_in_chunk += written;
                written = 0;
            } else {
                written -= remaining_in_chunk;
                chunk_idx += 1;
                byte_in_chunk = 0;
            }
        }
    }
    Ok(())
}

/// Nudge writeback for a byte range. See [`PlatformFile::hint_writeback`].
fn hint_writeback_sync(file: &File, offset: u64, len: u64) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        let off = libc::off64_t::try_from(offset).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "writeback offset too large")
        })?;
        let nbytes = libc::off64_t::try_from(len).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "writeback length too large")
        })?;
        // SAFETY: `file` owns a valid fd for the call's duration.
        // `SYNC_FILE_RANGE_WRITE` only requests that the kernel start
        // writeback for the range; it does not touch user memory and
        // gives no durability guarantee (unlike `fsync`/`fdatasync`).
        let ret = unsafe {
            libc::sync_file_range(file.as_raw_fd(), off, nbytes, libc::SYNC_FILE_RANGE_WRITE)
        };
        if ret != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (file, offset, len);
        Ok(())
    }
}

fn sync_durable_sync_inner(file: &File, size_fixed: bool) -> io::Result<()> {
    #[cfg(target_os = "macos")]
    {
        let _ = size_fixed; // F_FULLFSYNC is a device-wide barrier either way.
        use std::os::unix::io::AsRawFd;
        // SAFETY: `F_FULLFSYNC` is a valid macOS fcntl command that flushes
        // the write buffer all the way to the storage hardware.
        // The file is open and the fd remains valid throughout this call.
        let ret = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_FULLFSYNC) };
        if ret < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
    #[cfg(target_os = "linux")]
    {
        // A size-fixed file (preallocated, never grows again) never needs
        // its inode metadata re-committed on a later flush - `fdatasync`
        // covers the data with the same power-loss guarantee at lower cost.
        if size_fixed {
            file.sync_data()
        } else {
            file.sync_all()
        }
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = size_fixed;
        file.sync_all()
    }
}

/// The most a small fixed-format sidecar is ever allowed to be.
///
/// `CURRENT` is one byte, `LAYOUT` is 44, the tier marker is a short word. A
/// page is generous for all of them and small enough that a corrupt file
/// cannot matter.
pub const SMALL_FILE_MAX: u64 = 4096;

/// Read a small sidecar file, reading at most [`SMALL_FILE_MAX`] bytes.
///
/// `std::fs::read` and `read_to_string` allocate the WHOLE file before the
/// caller can look at its size, which makes the size of an allocation a
/// property of a file on disk. That is fine for a graph or a WAL, whose size
/// is the point; it is not fine for a pointer, a marker or a manifest, which
/// have a fixed shape and whose callers all run at startup.
///
/// A file longer than the bound comes back truncated rather than as an error,
/// on purpose: "this is not a valid record" is the parser's judgement to make
/// and its message to give, not this function's.
///
/// # Errors
///
/// Returns an IO error if the file cannot be opened or read. `NotFound` is
/// passed through, since an absent sidecar is usually a legitimate state.
pub fn read_small_file(path: &Path) -> io::Result<String> {
    use std::io::Read;
    let mut buf = String::new();
    File::open(path)?
        .take(SMALL_FILE_MAX)
        .read_to_string(&mut buf)?;
    Ok(buf)
}

/// [`read_small_file`] for a binary record: the same bound, no UTF-8.
///
/// # Errors
///
/// Returns an IO error if the file cannot be opened or read.
pub fn read_small_bytes(path: &Path) -> io::Result<Vec<u8>> {
    use std::io::Read;
    let mut buf = Vec::with_capacity(64);
    File::open(path)?
        .take(SMALL_FILE_MAX)
        .read_to_end(&mut buf)?;
    Ok(buf)
}

/// fsync a directory so a newly created or renamed entry within it survives
/// power loss. Syncing a *file* does not persist its directory entry on ext4
/// (data=ordered) or APFS, so a segment created during rotation - or a
/// snapshot just renamed into place - can vanish on reboot even after the file
/// itself was fsynced, silently dropping a write acked as `Durability::Power`.
/// Call this once after creating/renaming, before the write is acked durable.
///
/// # Errors
///
/// Returns an IO error if the directory cannot be opened or synced.
pub fn sync_dir(dir: &Path) -> io::Result<()> {
    let d = File::open(dir)?;
    #[cfg(target_os = "macos")]
    {
        use std::os::unix::io::AsRawFd;
        // SAFETY: `d` owns a valid fd for the call; F_FULLFSYNC on a directory
        // fd flushes its metadata (the new dirent) to hardware.
        let ret = unsafe { libc::fcntl(d.as_raw_fd(), libc::F_FULLFSYNC) };
        if ret < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
    #[cfg(not(target_os = "macos"))]
    {
        d.sync_all()
    }
}

/// Preallocate `file` to `len` bytes. See [`PlatformFile::preallocate`].
fn preallocate_file(file: &File, len: u64) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        let len = libc::off_t::try_from(len).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "preallocate length too large")
        })?;
        // SAFETY: `file` owns a valid fd for the call's duration. Mode 0
        // (no `FALLOC_FL_KEEP_SIZE`) both reserves the blocks and extends
        // the file's apparent size to `len` - the property that makes later
        // in-range writes size-stable.
        let ret = unsafe { libc::fallocate(file.as_raw_fd(), 0, 0, len) };
        if ret != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
    #[cfg(target_os = "macos")]
    {
        #[allow(clippy::cast_possible_wrap)]
        let mut fstore = libc::fstore_t {
            fst_flags: libc::F_ALLOCATECONTIG,
            fst_posmode: libc::F_PEOFPOSMODE,
            fst_offset: 0,
            fst_length: len as i64,
            fst_bytesalloc: 0,
        };
        // SAFETY: `file` owns a valid fd; `fstore` is a fully-initialised,
        // stack-local value read by the kernel only for the call's duration.
        let mut ret = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_PREALLOCATE, &fstore) };
        if ret < 0 {
            // Contiguous allocation can fail on a fragmented volume; retry
            // allowing any extents before giving up.
            fstore.fst_flags = libc::F_ALLOCATEALL;
            // SAFETY: same envelope as above.
            ret = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_PREALLOCATE, &fstore) };
        }
        if ret < 0 {
            return Err(io::Error::last_os_error());
        }
        // F_PREALLOCATE reserves blocks but does not move EOF.
        file.set_len(len)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        file.set_len(len)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn sequential_file_advice_accepts_a_regular_file_range() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("advice.bin");
        std::fs::write(&path, vec![0u8; 8192]).unwrap();
        let file = File::open(path).unwrap();

        advise_sequential_file(&file, 0, 8192).unwrap();
    }

    #[tokio::test]
    async fn test_f_nocache_set() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("nocache.bin");
        let file = PlatformFile::create(&path).unwrap();

        let data = b"hello, nocache!";
        file.write_at(0, data.to_vec()).await.unwrap();
        let read = file.pread(0, data.len()).await.unwrap();
        assert_eq!(read, data);
    }

    #[tokio::test]
    async fn test_f_fullfsync() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("fullfsync.bin");
        let file = PlatformFile::create(&path).unwrap();

        let data = b"durable data";
        file.write_at(0, data.to_vec()).await.unwrap();
        file.sync_durable().await.unwrap();
        assert_eq!(file.sync_count(), 1);

        let read = file.pread(0, data.len()).await.unwrap();
        assert_eq!(read, data);
    }

    #[tokio::test]
    async fn test_async_file_pread_correctness() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("pread.bin");
        let file = PlatformFile::create(&path).unwrap();

        // Write a known pattern
        let pattern: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
        file.write_at(0, pattern.clone()).await.unwrap();

        // Read a sub-range and verify byte-for-byte
        let slice = file.pread(128, 256).await.unwrap();
        assert_eq!(slice, &pattern[128..384]);
    }

    #[tokio::test]
    async fn test_write_read_large() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("large.bin");
        let file = PlatformFile::create(&path).unwrap();

        let data: Vec<u8> = (0u8..=255).cycle().take(128 * 1024).collect();
        file.write_at(0, data.clone()).await.unwrap();
        let read = file.pread(0, data.len()).await.unwrap();
        assert_eq!(read, data);
    }

    #[tokio::test]
    async fn test_truncate() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("trunc.bin");
        let file = PlatformFile::create(&path).unwrap();

        file.write_at(0, vec![0xFFu8; 1024]).await.unwrap();
        assert_eq!(file.size().unwrap(), 1024);
        file.truncate(512).await.unwrap();
        assert_eq!(file.size().unwrap(), 512);
    }

    #[tokio::test]
    async fn test_pread_at_eof_returns_short() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("short.bin");
        let file = PlatformFile::create(&path).unwrap();

        file.write_at(0, vec![0u8; 64]).await.unwrap();
        let read = file.pread(32, 64).await.unwrap(); // only 32 bytes available
        assert_eq!(read.len(), 32);
    }

    #[tokio::test]
    async fn test_preallocate_extends_reported_size() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("prealloc.bin");
        let file = PlatformFile::create(&path).unwrap();

        assert_eq!(file.size().unwrap(), 0);
        file.preallocate(64 * 1024).await.unwrap();
        assert_eq!(file.size().unwrap(), 64 * 1024);
    }

    #[test]
    fn test_preallocate_sync_extends_reported_size() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("prealloc_sync.bin");
        let file = PlatformFile::create(&path).unwrap();

        file.preallocate_sync(32 * 1024).unwrap();
        assert_eq!(file.size().unwrap(), 32 * 1024);
    }

    #[tokio::test]
    async fn test_preallocate_marks_size_fixed() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("prealloc_flag.bin");
        let file = PlatformFile::create(&path).unwrap();

        assert!(!file.is_size_fixed());
        file.preallocate(4096).await.unwrap();
        assert!(file.is_size_fixed());
    }

    #[tokio::test]
    async fn test_preallocate_is_idempotent_and_extendable() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("prealloc_repeat.bin");
        let file = PlatformFile::create(&path).unwrap();

        file.preallocate(4096).await.unwrap();
        file.preallocate(4096).await.unwrap(); // repeat at the same length
        assert_eq!(file.size().unwrap(), 4096);
        file.preallocate(8192).await.unwrap(); // grow further
        assert_eq!(file.size().unwrap(), 8192);
    }

    #[tokio::test]
    async fn test_write_vectored_at_matches_combined_write() {
        let dir = TempDir::new().unwrap();
        let path_a = dir.path().join("vectored.bin");
        let path_b = dir.path().join("combined.bin");
        let file_a = PlatformFile::create(&path_a).unwrap();
        let file_b = PlatformFile::create(&path_b).unwrap();

        let chunks: Vec<Vec<u8>> = vec![
            vec![0xAAu8; 37],
            vec![], // an empty entry must be a true no-op, not an error
            vec![0xBBu8; 4096],
            vec![0xCCu8; 1],
            (0u8..=255).collect(),
        ];
        let combined: Vec<u8> = chunks.iter().flatten().copied().collect();

        file_a.write_vectored_at(0, chunks).await.unwrap();
        file_b.write_at(0, combined.clone()).await.unwrap();

        let read_a = file_a.pread(0, combined.len()).await.unwrap();
        let read_b = file_b.pread(0, combined.len()).await.unwrap();
        assert_eq!(read_a, combined, "vectored write must match a plain pwrite");
        assert_eq!(read_a, read_b);
    }

    #[tokio::test]
    async fn test_write_vectored_at_empty_chunk_list_is_noop() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("empty_vectored.bin");
        let file = PlatformFile::create(&path).unwrap();

        file.write_vectored_at(0, Vec::new()).await.unwrap();
        assert_eq!(file.size().unwrap(), 0);
    }

    #[test]
    fn test_write_vectored_at_sync_matches_manual_combine() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("vectored_sync.bin");
        let file = PlatformFile::create(&path).unwrap();
        let chunks: Vec<Vec<u8>> = vec![vec![1u8; 10], vec![2u8; 20], vec![3u8; 3]];
        let combined: Vec<u8> = chunks.iter().flatten().copied().collect();

        file.write_vectored_at_sync(0, &chunks).unwrap();
        let mut buf = vec![0u8; combined.len()];
        let n = file.pread_sync(0, &mut buf).unwrap();
        assert_eq!(n, combined.len());
        assert_eq!(buf, combined);
    }

    #[tokio::test]
    async fn test_write_vectored_at_nonzero_offset() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("vectored_offset.bin");
        let file = PlatformFile::create(&path).unwrap();

        file.write_at(0, vec![0u8; 16]).await.unwrap();
        let chunks = vec![vec![9u8; 8], vec![7u8; 8]];
        file.write_vectored_at(16, chunks).await.unwrap();

        let read = file.pread(16, 16).await.unwrap();
        assert_eq!(read, [vec![9u8; 8], vec![7u8; 8]].concat());
    }

    #[test]
    fn test_mapped_file_open_populated_matches_plain_open() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("populated.bin");
        let data: Vec<u8> = (0u8..=255).cycle().take(8192).collect();
        std::fs::write(&path, &data).unwrap();

        let plain = MappedFile::open(&path).unwrap();
        let populated = MappedFile::open_populated(&path).unwrap();
        assert_eq!(plain.as_bytes(), &data[..]);
        assert_eq!(populated.as_bytes(), &data[..]);
        assert_eq!(populated.len(), data.len());
    }

    #[test]
    fn test_mapped_file_advise_huge_does_not_disturb_data() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("huge_advice.bin");
        let data: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
        std::fs::write(&path, &data).unwrap();

        let mapped = MappedFile::open(&path).unwrap();
        mapped.advise_huge().unwrap();
        assert_eq!(mapped.as_bytes(), &data[..]);
    }

    #[tokio::test]
    async fn test_hint_writeback_does_not_disturb_data() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("writeback.bin");
        let file = PlatformFile::create(&path).unwrap();

        let data = b"data survives a writeback hint";
        file.write_at(0, data.to_vec()).await.unwrap();
        file.hint_writeback(0, 0).await.unwrap(); // 0 length = "through EOF"
        file.hint_writeback(0, data.len() as u64).await.unwrap();

        let read = file.pread(0, data.len()).await.unwrap();
        assert_eq!(read, data);
    }

    #[tokio::test]
    async fn test_sync_durable_correct_after_preallocate() {
        // Regression guard for the fdatasync fast path on Linux: data
        // written into a preallocated (size-fixed) file must still read
        // back correctly after `sync_durable`.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("prealloc_durable.bin");
        let file = PlatformFile::create(&path).unwrap();

        file.preallocate(64 * 1024).await.unwrap();
        let data = b"durable data in a preallocated file";
        file.write_at(0, data.to_vec()).await.unwrap();
        file.sync_durable().await.unwrap();
        assert_eq!(file.sync_count(), 1);

        let read = file.pread(0, data.len()).await.unwrap();
        assert_eq!(read, data);
        // The preallocated tail is still there - length must not have
        // shrunk back down to the written range.
        assert_eq!(file.size().unwrap(), 64 * 1024);
    }
}
