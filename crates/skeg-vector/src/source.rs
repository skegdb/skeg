//! Abstract f32 vector storage for the index build.
//!
//! The Vamana build reads every vector many times - the greedy walk, robust
//! pruning, the medoid scan. [`VectorSource`] lets the build draw those rows
//! from either an owned `Vec<f32>` ([`InMemoryVectorSource`]) or a
//! memory-mapped file ([`MmapVectorSource`]); the latter lets `skeg-tool`
//! build an index without copying the whole dataset into the heap.

use std::io;
use std::path::Path;

use skeg_platform::MappedFile;

/// Row-major f32 vectors addressed by row id, shared read-only across the
/// parallel build.
pub trait VectorSource: Send + Sync {
    /// Vector dimension.
    fn dim(&self) -> usize;
    /// Number of vectors.
    fn len(&self) -> usize;
    /// True if the source holds no vectors.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Row `id` as a contiguous `dim`-length slice.
    ///
    /// # Panics
    ///
    /// Panics if `id` is out of range.
    fn row(&self, id: u32) -> &[f32];
    /// Heap bytes the source occupies. A memory-mapped source reports 0: its
    /// pages are file-backed and reclaimable, not heap.
    fn heap_bytes(&self) -> usize;
}

/// A [`VectorSource`] backed by an owned `Vec<f32>` in the heap.
pub struct InMemoryVectorSource {
    data: Vec<f32>,
    dim: usize,
}

impl InMemoryVectorSource {
    /// Wrap `data` (row-major, `dim` values per row).
    ///
    /// # Panics
    ///
    /// Panics if `dim == 0` or `data.len()` is not a multiple of `dim`.
    #[must_use]
    pub fn new(data: Vec<f32>, dim: usize) -> InMemoryVectorSource {
        assert!(dim > 0, "dim must be positive");
        assert_eq!(data.len() % dim, 0, "data length is not a multiple of dim");
        InMemoryVectorSource { data, dim }
    }
}

impl VectorSource for InMemoryVectorSource {
    fn dim(&self) -> usize {
        self.dim
    }
    fn len(&self) -> usize {
        self.data.len() / self.dim
    }
    fn row(&self, id: u32) -> &[f32] {
        let start = id as usize * self.dim;
        &self.data[start..start + self.dim]
    }
    fn heap_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }
}

/// A [`VectorSource`] backed by a memory-mapped file. The f32 payload begins
/// at `byte_offset` and holds `n * dim` little-endian values; the caller,
/// which understands the file format, supplies those parameters.
pub struct MmapVectorSource {
    mapped: MappedFile,
    byte_offset: usize,
    n: usize,
    dim: usize,
}

impl MmapVectorSource {
    /// Map `path` and expose the f32 payload at `byte_offset` (`n * dim`
    /// little-endian f32 values).
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be mapped, the mapping is shorter
    /// than the declared payload, or the payload is not 4-byte aligned within
    /// the mapping (required to view the bytes as `f32`).
    ///
    /// # Panics
    ///
    /// Panics if `dim == 0`.
    pub fn open(
        path: &Path,
        byte_offset: usize,
        n: usize,
        dim: usize,
    ) -> io::Result<MmapVectorSource> {
        assert!(dim > 0, "dim must be positive");
        let overflow =
            || io::Error::new(io::ErrorKind::InvalidInput, "vector payload size overflow");
        let payload = n
            .checked_mul(dim)
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(overflow)?;
        let need = byte_offset.checked_add(payload).ok_or_else(overflow)?;

        let mapped = MappedFile::open(path)?;
        if mapped.len() < need {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "mapped file shorter than the declared vector payload",
            ));
        }
        // The payload bytes are viewed as f32, so the start must be 4-byte
        // aligned. A memory map begins on a page boundary, so in practice this
        // reduces to `byte_offset` being a multiple of 4.
        if (mapped.as_bytes().as_ptr().addr() + byte_offset) % 4 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "vector payload is not 4-byte aligned for f32 access",
            ));
        }
        Ok(MmapVectorSource {
            mapped,
            byte_offset,
            n,
            dim,
        })
    }
}

impl VectorSource for MmapVectorSource {
    fn dim(&self) -> usize {
        self.dim
    }
    fn len(&self) -> usize {
        self.n
    }
    fn row(&self, id: u32) -> &[f32] {
        let start = self.byte_offset + id as usize * self.dim * 4;
        // Alignment and length were verified in `open`, so this never panics.
        bytemuck::cast_slice(&self.mapped.as_bytes()[start..start + self.dim * 4])
    }
    fn heap_bytes(&self) -> usize {
        0
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)] // test sizes are tiny
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn in_memory_rows() {
        let src = InMemoryVectorSource::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3);
        assert_eq!(src.dim(), 3);
        assert_eq!(src.len(), 2);
        assert_eq!(src.row(0), &[1.0, 2.0, 3.0]);
        assert_eq!(src.row(1), &[4.0, 5.0, 6.0]);
        assert!(src.heap_bytes() >= 24);
    }

    #[test]
    fn mmap_rows_match_in_memory() {
        // An .fbin-style file: [u32 n][u32 dim][f32 payload], payload at byte 8.
        let n = 5usize;
        let dim = 4usize;
        let data: Vec<f32> = (0..n * dim).map(|i| i as f32 * 0.25).collect();
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("vec.fbin");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&u32::try_from(n).unwrap().to_le_bytes())
                .unwrap();
            f.write_all(&u32::try_from(dim).unwrap().to_le_bytes())
                .unwrap();
            for &x in &data {
                f.write_all(&x.to_le_bytes()).unwrap();
            }
        }
        let src = MmapVectorSource::open(&path, 8, n, dim).unwrap();
        assert_eq!((src.dim(), src.len()), (dim, n));
        assert_eq!(src.heap_bytes(), 0);
        let mem = InMemoryVectorSource::new(data, dim);
        for id in 0..n as u32 {
            assert_eq!(src.row(id), mem.row(id));
        }
    }

    #[test]
    fn mmap_rejects_short_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("short.fbin");
        std::fs::write(&path, [0u8; 16]).unwrap();
        // Declares 100 vectors but the file holds only 8 bytes of payload.
        assert!(MmapVectorSource::open(&path, 8, 100, 4).is_err());
    }
}
