//! `Tenant`: skeg-rigging-skeg's concrete tenant implementation.
//!
//! `Tenant` aggregates a `skeg_vector::DiskVamanaIndex` (authoritative
//! storage for embeddings; on-disk graph + delta log + tier
//! quantisation) with a JSON metadata sidecar (per-record `shareable`
//! flag, tags, and payload bytes). Together they satisfy the rigging
//! capability surface without modifying skeg.
//!
//! The layout under a tenant directory:
//!
//! ```text
//! <tenant_dir>/
//!   meta.json            ← MetadataSidecar (this crate)
//!   graph.vmn            ← DiskVamanaIndex (skeg-vector)
//!   vectors.bin          ← DiskVamanaIndex
//!   delta.log            ← DiskVamanaIndex streaming inserts
//!   tier.cache.bin?      ← DiskVamanaIndex optional mmap tier
//! ```
//!
//! Replacing the previous `FlatIndex` backend means the adapter scales
//! to millions of records with the latency profile skeg's README
//! advertises (~419 MiB RSS, p99 3.6 ms at 1M vectors, dim 1024).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc;

use bytes::Bytes;
use parking_lot::Mutex;
use skeg_rigging::{
    CAP_VECTOR_KV, CapabilityId, Event, EventFilter, EventStream, Filter, Hit, IterVectors,
    OpenError, QueryError, QueryFiltered, ReadOnlyView, RecordId, RecordMeta, TenantEvents,
    TenantId, TenantInfo, TenantLifecycle, TenantStats, TenantWrite,
};
use skeg_vector::DiskVamanaIndex;

use crate::meta_sidecar::{MetadataSidecar, RecordEntry};

/// Default L_search for vamana queries. Higher means better recall at
/// the cost of latency; 100 matches skeg's documented default.
const DEFAULT_L_SEARCH: usize = 100;

/// Errors that arise during tenant construction or persistence.
#[derive(Debug, thiserror::Error)]
pub enum TenantError {
    /// I/O failure.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// JSON (de)serialisation failure on the sidecar.
    #[error("metadata sidecar JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// Caller asked for a dim that does not match the tenant on disk.
    #[error("embedding dim mismatch: tenant {on_disk}, requested {requested}")]
    DimMismatch {
        /// Dim recorded in the sidecar (or returned by DiskVamana).
        on_disk: u32,
        /// Dim the caller asked for.
        requested: u32,
    },
    /// Tenant directory missing or empty.
    #[error("tenant not found at {0}")]
    NotFound(PathBuf),
}

impl From<TenantError> for OpenError {
    fn from(e: TenantError) -> Self {
        match e {
            TenantError::Io(io) => OpenError::Io(io),
            TenantError::NotFound(_) => OpenError::NotFound,
            TenantError::Json(j) => OpenError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                j.to_string(),
            )),
            TenantError::DimMismatch { .. } => OpenError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                e.to_string(),
            )),
        }
    }
}

/// Skeg-flavoured rigging tenant backed by `DiskVamanaIndex`.
///
/// Capability identifier: `vector.kv` (see `skeg-rigging` §11.2).
pub struct Tenant {
    inner: Arc<Mutex<Inner>>,
    tenant_id: TenantId,
    embedding_dim: u32,
    dir: PathBuf,
    read_only: bool,
    /// Live event subscribers. Senders whose receivers have dropped
    /// are pruned on the next `emit` call. Stored in a separate mutex
    /// from `inner` so an emit never has to hold the index lock.
    subscribers: Arc<Mutex<Vec<mpsc::Sender<Event>>>>,
}

struct Inner {
    index: DiskVamanaIndex,
    sidecar: MetadataSidecar,
}

impl Tenant {
    /// Path of the metadata sidecar inside a tenant directory.
    pub fn meta_path(dir: &Path) -> PathBuf {
        dir.join("meta.json")
    }

    /// Whether `dir` already holds a `DiskVamanaIndex`. `tier.kind` is the
    /// first file a create writes and `CURRENT` the last; either present
    /// means an index (or the remains of one) that must not be created over.
    fn index_exists(dir: &Path) -> bool {
        dir.join("tier.kind").exists() || dir.join("CURRENT").exists()
    }

    /// Open or create a tenant at `dir` for read-write access.
    ///
    /// - If `meta.json` exists, reopen both sidecar and `DiskVamanaIndex`.
    ///   The requested `embedding_dim` must match the on-disk value.
    /// - If `meta.json` is absent but the directory already holds an
    ///   index (a tenant that inserted and never flushed - the sidecar is
    ///   only written by `flush`), reopen the index and start from an
    ///   empty sidecar: the vectors are in the engine's WAL, the record
    ///   metadata that was never flushed is gone.
    /// - Otherwise create both: fresh sidecar + empty `DiskVamanaIndex`,
    ///   and write the sidecar at once so a later open never takes the
    ///   create path over live data.
    pub fn open(
        dir: impl AsRef<Path>,
        tenant_id: TenantId,
        embedding_dim: u32,
    ) -> Result<Self, TenantError> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        let meta_path = Self::meta_path(&dir);
        let (sidecar, index) = if meta_path.exists() {
            let loaded = MetadataSidecar::read_from_path(&meta_path)?;
            if loaded.embedding_dim != embedding_dim {
                return Err(TenantError::DimMismatch {
                    on_disk: loaded.embedding_dim,
                    requested: embedding_dim,
                });
            }
            let idx = DiskVamanaIndex::open(&dir)?;
            if idx.dim() as u32 != embedding_dim {
                return Err(TenantError::DimMismatch {
                    on_disk: idx.dim() as u32,
                    requested: embedding_dim,
                });
            }
            (loaded, idx)
        } else if Self::index_exists(&dir) {
            let idx = DiskVamanaIndex::open(&dir)?;
            if idx.dim() as u32 != embedding_dim {
                return Err(TenantError::DimMismatch {
                    on_disk: idx.dim() as u32,
                    requested: embedding_dim,
                });
            }
            let sidecar = MetadataSidecar::empty(tenant_id, embedding_dim);
            sidecar.write_to_path(&meta_path)?;
            (sidecar, idx)
        } else {
            let idx =
                DiskVamanaIndex::create_empty(&dir, embedding_dim as usize, DEFAULT_L_SEARCH)?;
            let sidecar = MetadataSidecar::empty(tenant_id, embedding_dim);
            sidecar.write_to_path(&meta_path)?;
            (sidecar, idx)
        };

        Ok(Self {
            inner: Arc::new(Mutex::new(Inner { index, sidecar })),
            tenant_id,
            embedding_dim,
            dir,
            subscribers: Arc::new(Mutex::new(Vec::new())),
            read_only: false,
        })
    }

    /// Open an existing tenant in read-only mode. The caller cannot
    /// insert; the underlying `DiskVamanaIndex` is still opened with
    /// its normal `open()` (skeg-vector does not have a strict
    /// read-only mode for now - writes are gated upstream).
    pub fn open_readonly_at(dir: impl AsRef<Path>) -> Result<Self, TenantError> {
        let dir = dir.as_ref().to_path_buf();
        let meta_path = Self::meta_path(&dir);
        if !meta_path.exists() {
            return Err(TenantError::NotFound(dir));
        }
        let sidecar = MetadataSidecar::read_from_path(&meta_path)?;
        let tenant_id = TenantId::from_bytes(sidecar.tenant_id);
        let embedding_dim = sidecar.embedding_dim;
        let index = DiskVamanaIndex::open(&dir)?;
        if index.dim() as u32 != embedding_dim {
            return Err(TenantError::DimMismatch {
                on_disk: index.dim() as u32,
                requested: embedding_dim,
            });
        }
        Ok(Self {
            inner: Arc::new(Mutex::new(Inner { index, sidecar })),
            tenant_id,
            embedding_dim,
            dir,
            subscribers: Arc::new(Mutex::new(Vec::new())),
            read_only: true,
        })
    }

    /// Stable id of this tenant.
    pub fn tenant_id(&self) -> TenantId {
        self.tenant_id
    }

    /// Embedding dimension.
    pub fn dim(&self) -> u32 {
        self.embedding_dim
    }

    /// Insert (or overwrite) a record. The vector goes into the
    /// DiskVamana streaming delta; metadata into the in-memory
    /// sidecar (persisted on [`Self::flush`]).
    pub fn insert(
        &self,
        record_id: RecordId,
        embedding: Vec<f32>,
        shareable: bool,
        tags: Vec<String>,
        payload: Vec<u8>,
    ) -> Result<(), TenantError> {
        if self.read_only {
            return Err(TenantError::Io(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "tenant opened read-only",
            )));
        }
        if embedding.len() as u32 != self.embedding_dim {
            return Err(TenantError::DimMismatch {
                on_disk: self.embedding_dim,
                requested: embedding.len() as u32,
            });
        }
        let mut inner = self.inner.lock();
        inner.index.insert(record_id.0, &embedding)?;
        let entry = RecordEntry {
            record_id: record_id.0,
            shareable,
            tags,
            payload,
        };
        if let Some(pos) = inner
            .sidecar
            .records
            .iter()
            .position(|r| r.record_id == record_id.0)
        {
            inner.sidecar.records[pos] = entry;
        } else {
            inner.sidecar.records.push(entry);
        }
        drop(inner);
        self.emit(Event::RecordInserted {
            record_id,
            shareable,
        });
        Ok(())
    }

    /// Remove a record from the tenant. Returns `true` when the row
    /// was present (and emits [`Event::RecordDeleted`] to subscribers);
    /// missing ids are a silent no-op.
    pub fn delete(&self, record_id: RecordId) -> Result<bool, TenantError> {
        if self.read_only {
            return Err(TenantError::Io(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "tenant opened read-only",
            )));
        }
        let mut inner = self.inner.lock();
        let before = inner.sidecar.records.len();
        inner.sidecar.records.retain(|r| r.record_id != record_id.0);
        let removed = inner.sidecar.records.len() != before;
        drop(inner);
        if removed {
            self.emit(Event::RecordDeleted { record_id });
        }
        Ok(removed)
    }

    /// Broadcast `event` to every live subscriber. Senders whose
    /// receivers have been dropped are pruned. Called by `insert` /
    /// `delete` after the underlying state has been mutated.
    fn emit(&self, event: Event) {
        let mut subs = self.subscribers.lock();
        subs.retain(|s| s.send(event.clone()).is_ok());
    }

    /// Persist the sidecar JSON. DiskVamana writes incrementally to
    /// its own log; this method only flushes the metadata side.
    pub fn flush(&self) -> Result<(), TenantError> {
        if self.read_only {
            return Ok(());
        }
        let inner = self.inner.lock();
        let path = Self::meta_path(&self.dir);
        inner.sidecar.write_to_path(&path)?;
        Ok(())
    }

    /// Trigger a DiskVamana consolidation - merges the streaming
    /// delta into the main graph. Optional for correctness (search
    /// works on delta + main), but recommended after large insert
    /// batches to keep query latency bounded.
    pub fn consolidate(&self) -> Result<(), TenantError> {
        if self.read_only {
            return Ok(());
        }
        let mut inner = self.inner.lock();
        match inner.index.consolidate()? {
            skeg_vector::FinishOutcome::Committed => Ok(()),
            skeg_vector::FinishOutcome::CommittedCleanupFailed(error) => Err(error.into()),
        }
    }

    /// Create a brand-new tenant at `path`. Fails with
    /// `AlreadyExists` if `path` already holds a sidecar (use
    /// [`Self::open`] for the "open-or-create" semantics).
    pub fn create_new(
        path: &Path,
        tenant_id: TenantId,
        embedding_dim: u32,
    ) -> Result<Self, TenantError> {
        if Self::meta_path(path).exists() {
            return Err(TenantError::Io(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!("tenant already exists at {}", path.display()),
            )));
        }
        Self::open(path, tenant_id, embedding_dim)
    }

    /// Restore a tenant from snapshot directory `src` into a fresh
    /// directory `dest`. Recursive copy then [`Self::open`] using
    /// the sidecar's recorded id + dim.
    pub fn restore_from(src: &Path, dest: &Path) -> Result<Self, TenantError> {
        if Self::meta_path(dest).exists() {
            return Err(TenantError::Io(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!("restore destination already populated: {}", dest.display()),
            )));
        }
        if !Self::meta_path(src).exists() {
            return Err(TenantError::NotFound(src.to_path_buf()));
        }
        std::fs::create_dir_all(dest)?;
        copy_dir_contents(src, dest)?;
        let sidecar = MetadataSidecar::read_from_path(&Self::meta_path(dest))?;
        Self::open(
            dest,
            TenantId::from_bytes(sidecar.tenant_id),
            sidecar.embedding_dim,
        )
    }
}

/// Recursively copy the immediate contents of `src` into `dest`.
fn copy_dir_contents(src: &Path, dest: &Path) -> std::io::Result<()> {
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dest_path = dest.join(entry.file_name());
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            std::fs::create_dir_all(&dest_path)?;
            copy_dir_contents(&src_path, &dest_path)?;
        } else if file_type.is_file() {
            std::fs::copy(&src_path, &dest_path)?;
        }
    }
    Ok(())
}

impl TenantInfo for Tenant {
    fn tenant_id(&self) -> TenantId {
        self.tenant_id
    }
    fn embedding_dim(&self) -> u32 {
        self.embedding_dim
    }
    fn record_count(&self) -> u64 {
        let inner = self.inner.lock();
        inner.sidecar.records.len() as u64
    }
    fn capabilities(&self) -> Vec<CapabilityId> {
        vec![CAP_VECTOR_KV]
    }
}

impl TenantEvents for Tenant {
    fn subscribe(&self, filter: EventFilter) -> EventStream {
        let (tx, rx) = mpsc::channel();
        self.subscribers.lock().push(tx);
        EventStream::new(rx, filter)
    }
}

impl TenantStats for Tenant {
    fn bytes_on_disk(&self) -> u64 {
        sum_dir_bytes(&self.dir).unwrap_or(0)
    }
    fn record_count(&self) -> u64 {
        <Self as TenantInfo>::record_count(self)
    }
    fn memory_resident(&self) -> u64 {
        // DiskVamana mmaps from the OS page cache; tracking RSS at
        // per-tenant granularity is not free and not portable. v0.1
        // returns 0 - orchestrators that need RSS sample at the
        // process level instead. Future skeg-vector revisions may
        // expose a residency counter; this stub will route to it.
        0
    }
}

/// Recursive sum of every regular file's `len()` under `dir`.
/// Best effort: a missing or unreadable entry contributes 0 rather
/// than aborting the walk, so `bytes_on_disk` stays cheap to poll
/// even when a concurrent writer is rotating files.
fn sum_dir_bytes(dir: &Path) -> std::io::Result<u64> {
    let mut total = 0u64;
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return Ok(0),
    };
    for entry in entries.flatten() {
        let Ok(ft) = entry.file_type() else { continue };
        if ft.is_file() {
            if let Ok(md) = entry.metadata() {
                total += md.len();
            }
        } else if ft.is_dir() {
            total += sum_dir_bytes(&entry.path()).unwrap_or(0);
        }
    }
    Ok(total)
}

impl TenantLifecycle for Tenant {
    fn snapshot(&self, dest: &Path) -> Result<(), OpenError> {
        if !dest.exists() {
            std::fs::create_dir_all(dest)?;
        }
        self.flush().map_err(OpenError::from)?;
        copy_dir_contents(&self.dir, dest).map_err(OpenError::Io)?;
        Ok(())
    }

    fn destroy(self: Box<Self>) -> Result<(), OpenError> {
        let dir = self.dir.clone();
        drop(self);
        std::fs::remove_dir_all(&dir).map_err(OpenError::Io)?;
        Ok(())
    }
}

impl TenantWrite for Tenant {
    type Error = TenantError;

    /// Delegates to the inherent [`Tenant::insert`]. The trait takes
    /// `&mut self` for object-safety across writers that need it; this
    /// engine uses interior mutability, so a shared borrow suffices.
    fn insert(
        &mut self,
        record_id: RecordId,
        embedding: &[f32],
        shareable: bool,
        tags: Vec<String>,
        payload: Vec<u8>,
    ) -> Result<(), TenantError> {
        Tenant::insert(
            self,
            record_id,
            embedding.to_vec(),
            shareable,
            tags,
            payload,
        )
    }

    fn flush(&mut self) -> Result<(), TenantError> {
        Tenant::flush(self)
    }

    fn embedding_dim(&self) -> u32 {
        self.embedding_dim
    }
}

impl IterVectors for Tenant {
    fn iter_vectors(&self) -> Box<dyn Iterator<Item = (RecordId, Vec<f32>)> + '_> {
        // Walk every record in the sidecar; pull the vector from
        // DiskVamana for each. O(N) reads - fine for the saga-build
        // cadence (rare, owner-only).
        let inner = self.inner.lock();
        let mut out: Vec<(RecordId, Vec<f32>)> = Vec::with_capacity(inner.sidecar.records.len());
        for rec in &inner.sidecar.records {
            match inner.index.get(rec.record_id) {
                Ok(Some(v)) => out.push((RecordId(rec.record_id), v)),
                Ok(None) => {} // tombstoned / missing
                Err(_) => {}   // skip on I/O error; caller can detect via count
            }
        }
        Box::new(out.into_iter())
    }

    fn record_count(&self) -> u64 {
        let inner = self.inner.lock();
        inner.sidecar.records.len() as u64
    }

    fn embedding_dim(&self) -> u32 {
        self.embedding_dim
    }
}

impl QueryFiltered for Tenant {
    fn query_filtered(
        &self,
        embedding: &[f32],
        top_k: u32,
        filter: &dyn Filter,
    ) -> Result<Vec<Hit>, QueryError> {
        if embedding.len() as u32 != self.embedding_dim {
            return Err(QueryError::EmbeddingDimMismatch {
                expected: self.embedding_dim,
                got: embedding.len() as u32,
            });
        }
        let inner = self.inner.lock();
        // v0.1 oversample-and-filter: filter-in-traversal would need a
        // skeg-vector API extension we cannot add. Oversample 4× is
        // enough when shareable selectivity is 25%+.
        let oversample = ((top_k as usize) * 4).max(64);
        let raw = inner
            .index
            .search(embedding, oversample)
            .map_err(QueryError::Io)?;

        let mut out = Vec::with_capacity(top_k as usize);
        for (id, sim) in raw {
            let Some(rec) = inner.sidecar.records.iter().find(|r| r.record_id == id) else {
                continue;
            };
            let tags: Vec<&str> = rec.tags.iter().map(String::as_str).collect();
            let meta = RecordMeta {
                record_id: RecordId(id),
                shareable: rec.shareable,
                tags: &tags,
            };
            if !filter.accept(&meta) {
                continue;
            }
            // Pull the stored embedding so downstream context dedup can
            // run cosine over hits. Best effort: if the vector cannot be
            // recovered, the hit is still surfaced with `embedding = None`.
            let embedding = inner.index.get(id).ok().flatten();
            out.push(Hit {
                record_id: RecordId(id),
                similarity: sim,
                payload: Bytes::from(rec.payload.clone()),
                embedding,
            });
            if out.len() >= top_k as usize {
                break;
            }
        }
        Ok(out)
    }
}

impl ReadOnlyView for Tenant {
    fn tenant_id(&self) -> TenantId {
        self.tenant_id
    }
    fn close(self: Box<Self>) -> Result<(), OpenError> {
        Ok(())
    }
}

/// Open a tenant at `path` in read-only mode and return it as a
/// trait-object. Mirrors [`skeg_rigging::open_readonly`] but with a
/// concrete adapter behind it.
pub fn open_readonly(path: &Path) -> Result<Box<dyn ReadOnlyView>, OpenError> {
    let v = Tenant::open_readonly_at(path)?;
    Ok(Box::new(v))
}
