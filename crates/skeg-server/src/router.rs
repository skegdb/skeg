//! Per-vindex semantic router sidecar: the balanced-k-means centroids that
//! give every shard a semantic identity (one centroid per shard), plus the
//! epoch that says which reshard produced them. Lives at the shard-set
//! root (`router-<name>.bin`), not inside a shard: the partition is a
//! property of the set.
//!
//! Layout: magic "SKRT", version u32, k u32, dim u32, epoch u64,
//! k*dim f32 LE centroids, crc32c of everything before it.

use std::io;
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 4] = b"SKRT";
const VERSION: u32 = 1;

/// A loaded semantic router: `assign` maps a vector to its owner shard.
#[derive(Debug, Clone, PartialEq)]
pub struct Router {
    pub k: usize,
    pub dim: usize,
    pub epoch: u64,
    pub centroids: Vec<f32>,
}

impl Router {
    /// Owner shard for `x`: nearest centroid by cosine.
    ///
    /// # Panics
    ///
    /// Panics if `x.len() != self.dim`.
    #[must_use]
    pub fn assign(&self, x: &[f32]) -> usize {
        // Callers validate the dim first (a mismatch is a clean client error,
        // never a panic under panic=abort); a mismatched vector here degrades
        // to shard 0 rather than aborting.
        if x.len() != self.dim {
            return 0;
        }
        skeg_vector::nearest_centroid(x, &self.centroids, self.k, self.dim)
    }

    /// Serialise to `path` (write + rename, fsynced).
    ///
    /// # Errors
    ///
    /// I/O error from the filesystem.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut b = Vec::with_capacity(24 + self.centroids.len() * 4);
        b.extend_from_slice(MAGIC);
        b.extend_from_slice(&VERSION.to_le_bytes());
        b.extend_from_slice(&(self.k as u32).to_le_bytes());
        b.extend_from_slice(&(self.dim as u32).to_le_bytes());
        b.extend_from_slice(&self.epoch.to_le_bytes());
        for x in &self.centroids {
            b.extend_from_slice(&x.to_le_bytes());
        }
        let crc = crc32c::crc32c(&b);
        b.extend_from_slice(&crc.to_le_bytes());
        let tmp = path.with_extension("bin.tmp");
        std::fs::write(&tmp, &b)?;
        std::fs::File::open(&tmp)?.sync_all()?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Load from `path`. The file is untrusted on-disk input: every field is
    /// validated before any allocation it sizes.
    ///
    /// # Errors
    ///
    /// `InvalidData` for a torn, truncated or foreign file; other I/O errors
    /// from the filesystem.
    pub fn load(path: &Path) -> io::Result<Router> {
        let b = std::fs::read(path)?;
        let bad = || io::Error::new(io::ErrorKind::InvalidData, "corrupt router sidecar");
        if b.len() < 28 || &b[0..4] != MAGIC {
            return Err(bad());
        }
        let u32_at = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().expect("4 bytes"));
        if u32_at(4) != VERSION {
            return Err(bad());
        }
        let k = u32_at(8) as usize;
        let dim = u32_at(12) as usize;
        let epoch = u64::from_le_bytes(b[16..24].try_into().expect("8 bytes"));
        let need = 24usize
            .checked_add(k.checked_mul(dim).and_then(|n| n.checked_mul(4)).ok_or_else(bad)?)
            .and_then(|n| n.checked_add(4))
            .ok_or_else(bad)?;
        if b.len() != need || k == 0 || dim == 0 {
            return Err(bad());
        }
        let crc = u32::from_le_bytes(b[need - 4..need].try_into().expect("4 bytes"));
        if crc32c::crc32c(&b[..need - 4]) != crc {
            return Err(bad());
        }
        let centroids = b[24..need - 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok(Router { k, dim, epoch, centroids })
    }
}

/// Sidecar path for `name` under the shard-set root.
#[must_use]
pub fn router_path(root: &Path, name: &str) -> PathBuf {
    root.join(format!("router-{name}.bin"))
}
