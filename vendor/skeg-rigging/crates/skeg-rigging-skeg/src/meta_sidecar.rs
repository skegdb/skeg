//! JSON metadata sidecar.
//!
//! The vector data itself lives in skeg-vector's `DiskVamanaIndex`
//! (under `<tenant_dir>/vectors.bin`, `graph.vmn`, etc.). This sidecar
//! holds the *metadata* hansa needs to filter - the `shareable` flag,
//! tags, and the raw payload - keyed by record id. No embedding bytes
//! are duplicated here.
//!
//! v0.1 uses a single JSON file overwritten on `flush`. Concurrent
//! writers within one process are serialised by a `parking_lot::Mutex`
//! upstream. Cross-process concurrent writes are out of scope
//! (one writer process per tenant).

use std::path::Path;

use serde::{Deserialize, Serialize};
use skeg_rigging::{RecordId, TenantId};

use crate::tenant::TenantError;

/// Per-record metadata. The vector lives in DiskVamana, not here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordEntry {
    /// Record identifier (matches the vamana id).
    pub record_id: u64,
    /// Whether this record is visible to peers in a hansa membrane.
    pub shareable: bool,
    /// Tag strings.
    pub tags: Vec<String>,
    /// Raw payload bytes. Held as a Vec<u8>; serde encodes it as a JSON
    /// array of numbers - fine for medium-size payloads, switch to
    /// base64 if/when bandwidth matters.
    pub payload: Vec<u8>,
}

/// The serialised tenant snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataSidecar {
    /// Stable tenant id as raw bytes.
    pub tenant_id: [u8; 16],
    /// Embedding dimension. All records share it (validated against
    /// DiskVamana at open).
    pub embedding_dim: u32,
    /// Records, ordered by insertion.
    pub records: Vec<RecordEntry>,
}

impl MetadataSidecar {
    /// Fresh empty sidecar for a tenant with the given id and dim.
    pub fn empty(tenant_id: TenantId, embedding_dim: u32) -> Self {
        Self {
            tenant_id: tenant_id.0,
            embedding_dim,
            records: Vec::new(),
        }
    }

    /// Read from JSON at `path`.
    pub fn read_from_path(path: &Path) -> Result<Self, TenantError> {
        let bytes = std::fs::read(path)?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    /// Write to JSON at `path` (overwrites; caller is responsible for
    /// atomic replacement when needed).
    pub fn write_to_path(&self, path: &Path) -> Result<(), TenantError> {
        let bytes = serde_json::to_vec(self)?;
        std::fs::write(path, &bytes)?;
        Ok(())
    }

    /// Find a record's metadata by id.
    pub fn get(&self, id: RecordId) -> Option<&RecordEntry> {
        self.records.iter().find(|r| r.record_id == id.0)
    }

    /// True when no records have been inserted.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}
