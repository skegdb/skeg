//! Heap-allocated buffers with explicit alignment.
//!
//! Required for `O_DIRECT` on Linux; useful for Apple Silicon DMA alignment.
//! unsafe here is intentional - see SAFETY comments.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ptr::NonNull;

/// Owned byte buffer allocated at a specific memory alignment.
///
/// The pointer is always aligned to `align` bytes.
/// Dropping frees the memory via `dealloc`.
pub struct AlignedBytes {
    ptr: NonNull<u8>,
    len: usize,
    align: usize,
}

// SAFETY: AlignedBytes owns its allocation uniquely; no aliased mutability.
unsafe impl Send for AlignedBytes {}
unsafe impl Sync for AlignedBytes {}

impl AlignedBytes {
    /// Allocate a zero-filled buffer of `size` bytes with `align` alignment.
    ///
    /// # Panics
    ///
    /// Panics if `align` is not a power of two, `size == 0`, or OOM.
    #[must_use]
    pub fn alloc(size: usize, align: usize) -> Self {
        assert!(align.is_power_of_two(), "align must be power of two");
        assert!(size > 0, "size must be > 0");
        let layout = Layout::from_size_align(size, align).expect("valid layout");
        // SAFETY: layout is non-zero and properly aligned.
        // alloc_zeroed returns null on OOM; NonNull::new detects and panics.
        let ptr = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).expect("allocation failed: OOM");
        Self {
            ptr,
            len: size,
            align,
        }
    }

    /// Return the number of bytes in the buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if the buffer has zero bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Raw pointer to the start of the buffer.
    #[must_use]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// View as a byte slice.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is valid for `len` bytes, no mutable alias exists.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// View as a mutable byte slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: ptr is valid for `len` bytes, we hold exclusive ownership (`&mut self`).
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Copy `src` into this buffer.
    ///
    /// # Panics
    ///
    /// Panics if `src.len() != self.len()`.
    pub fn copy_from_slice(&mut self, src: &[u8]) {
        self.as_mut_slice().copy_from_slice(src);
    }
}

impl std::ops::Deref for AlignedBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl std::ops::DerefMut for AlignedBytes {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

impl Drop for AlignedBytes {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.len, self.align).expect("valid layout");
        // SAFETY: ptr was allocated with this exact layout; not yet freed.
        unsafe { dealloc(self.ptr.as_ptr(), layout) };
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_buf_alignment_16k() {
        let buf = AlignedBytes::alloc(4096, 16_384);
        assert_eq!(
            buf.as_ptr() as usize % 16_384,
            0,
            "pointer must be 16K-aligned"
        );
    }

    #[test]
    fn test_aligned_buf_alignment_4k() {
        let buf = AlignedBytes::alloc(512, 4096);
        assert_eq!(buf.as_ptr() as usize % 4096, 0);
    }

    #[test]
    fn test_aligned_buf_zero_filled() {
        let buf = AlignedBytes::alloc(256, 64);
        assert!(
            buf.iter().all(|&b| b == 0),
            "freshly allocated buffer must be zero"
        );
    }

    #[test]
    fn test_aligned_buf_read_write() {
        let mut buf = AlignedBytes::alloc(128, 64);
        buf.copy_from_slice(&[0xABu8; 128]);
        assert!(buf.iter().all(|&b| b == 0xAB));
    }

    #[test]
    fn test_aligned_buf_len() {
        let buf = AlignedBytes::alloc(1024, 128);
        assert_eq!(buf.len(), 1024);
        assert!(!buf.is_empty());
    }

    #[test]
    #[should_panic(expected = "align must be power of two")]
    fn test_aligned_buf_bad_align_panics() {
        let _ = AlignedBytes::alloc(64, 3);
    }

    #[test]
    #[should_panic(expected = "size must be > 0")]
    fn test_aligned_buf_zero_size_panics() {
        let _ = AlignedBytes::alloc(0, 64);
    }
}
