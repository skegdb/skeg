//! Platform-optimised file I/O.
//!
//! On macOS: `F_NOCACHE` disables the OS page cache; `F_FULLFSYNC` ensures data
//! reaches hardware storage (stronger than `fsync`).
//! On other Unix: uses `fdatasync` for durability.
//!
//! Async methods offload blocking syscalls to `tokio::task::spawn_blocking`.
//! Sync methods are thin wrappers for use inside `spawn_blocking` closures or
//! recovery code that already runs on a blocking thread.
//!
//! unsafe here is intentional for fcntl syscalls - see SAFETY comments.

use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

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
}

impl PlatformFile {
    /// Create a new file (fails if it already exists) and apply `F_NOCACHE`.
    ///
    /// # Errors
    ///
    /// Returns an IO error if the file cannot be created or `F_NOCACHE` fails.
    pub fn create(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(path)?;
        let pf = Self {
            inner: Arc::new(file),
            sync_count: Arc::new(AtomicU64::new(0)),
            read_count: Arc::new(AtomicU64::new(0)),
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
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let pf = Self {
            inner: Arc::new(file),
            sync_count: Arc::new(AtomicU64::new(0)),
            read_count: Arc::new(AtomicU64::new(0)),
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

    /// Flush to hardware storage - power-loss durable.
    ///
    /// Uses `F_FULLFSYNC` on macOS, `sync_all` elsewhere.
    ///
    /// # Errors
    ///
    /// Returns an IO error if the flush fails.
    pub async fn sync_durable(&self) -> io::Result<()> {
        let file = self.inner.clone();
        let counter = self.sync_count.clone();
        tokio::task::spawn_blocking(move || {
            let result = sync_durable_sync_inner(&file);
            if result.is_ok() {
                counter.fetch_add(1, Ordering::Relaxed);
            }
            result
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

    /// Sync flush to hardware.
    ///
    /// # Errors
    ///
    /// Returns an IO error if the flush fails.
    pub fn sync_durable_sync(&self) -> io::Result<()> {
        let result = sync_durable_sync_inner(&self.inner);
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
}

impl std::ops::Deref for MappedFile {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.mmap
    }
}

// ── Free functions used in spawn_blocking closures ────────────────────────────

fn pread_sync(file: &File, offset: u64, size: usize) -> io::Result<Vec<u8>> {
    let mut buf = vec![0u8; size];
    let n = pread_sync_into(file, offset, &mut buf)?;
    buf.truncate(n);
    Ok(buf)
}

fn pread_sync_into(file: &File, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
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
        pos += n;
    }
    Ok(())
}

fn sync_durable_sync_inner(file: &File) -> io::Result<()> {
    #[cfg(target_os = "macos")]
    {
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
    #[cfg(not(target_os = "macos"))]
    {
        file.sync_all()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

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
}
