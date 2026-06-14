#![deny(unsafe_code)]

//! Value-log: append-only durable KV log with an in-RAM index.
//!
//! Writes go through a [`GroupCommitter`] per active segment; the durability
//! tier (see [`Durability`]) chosen per write decides how hard the batch is
//! flushed. Reads use seekless `pread` fronted by an S3-FIFO cache.
//!
//! A `VLog` is single-shard, single-thread: it is `Rc`-backed and accessed only
//! from its owning shard's runtime. Interior mutability via `RefCell` is sound
//! because no borrow is ever held across an `.await`. The one exception is the
//! per-tenant disk counter, an `Arc<Mutex>` shared across shards so the disk
//! quota is global per tenant; it is locked only for a single map op.
//!
//! ## Recovery ordering
//!
//! On open, every segment is scanned and the record with the highest timestamp
//! wins per key. Timestamp order (not segment-scan order) is authoritative
//! because compaction relocates old records into newer segments while
//! preserving their original timestamp.

use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use ahash::AHashMap;
use bytes::Bytes;
use parking_lot::Mutex;
use skeg_platform::PlatformFile;

use crate::cache::S3Fifo;
use crate::group_commit::{Durability, GroupCommitter};
use crate::index::{Index, IndexEntry, fingerprint};
use crate::record::{Record, RecordKind, decode_record, encode_record, padded_record_size};
use crate::segment::{MAX_SEGMENT_SIZE, list_segments, scan_file, segment_path};
use crate::snapshot;
use crate::{Error, Result};

/// Per-shard hot-key cache byte budget. `F_NOCACHE` disables the OS page
/// cache, so this RAM cache is what keeps repeated reads off the SSD. A byte
/// budget (not an entry count) keeps RAM bounded regardless of value size;
/// 32 MiB/shard is a frugal default that holds ~8K typical embeddings.
const CACHE_BUDGET_BYTES: usize = 32 * 1024 * 1024;

/// A segment whose live-bytes ratio falls below this gets compacted.
const COMPACTION_LIVE_RATIO: f64 = 0.5;

/// A read handle on a segment file, plus its tracked live-bytes counter.
struct ReadSegment {
    id: u16,
    file: Arc<PlatformFile>,
    /// Bytes of records the index currently points into this segment.
    live: Cell<u64>,
}

/// The currently-appended segment and its committer.
struct ActiveState {
    id: u16,
    size: u64,
    committer: GroupCommitter,
}

struct VLogInner {
    dir: PathBuf,
    max_seg_size: u64,
    read_segments: RefCell<Vec<ReadSegment>>,
    index: RefCell<Index>,
    cache: RefCell<S3Fifo<Bytes>>,
    /// Live on-disk bytes (padded record size) per tenant, for the disk quota.
    /// The tenant is the key's 16-byte prefix; entries drop to absent at zero.
    ///
    /// Shared across a shard set via `Arc<Mutex>` so the disk quota is GLOBAL per
    /// tenant (a tenant's keys spread over shards by hash); the lock is held only
    /// for a single map op, negligible against the append I/O around it. A
    /// standalone `VLog` gets its own counter (single shard = already global).
    tenant_disk: Arc<Mutex<AHashMap<u128, u64>>>,
    active: RefCell<ActiveState>,
    clock: Cell<u64>,
}

/// A per-tenant live-disk-byte counter shared across a shard set, so the disk
/// quota is global per tenant. Create with [`new_shared_disk`].
pub type SharedTenantDisk = Arc<Mutex<AHashMap<u128, u64>>>;

/// A fresh, empty shared disk counter for a shard set.
#[must_use]
pub fn new_shared_disk() -> SharedTenantDisk {
    Arc::new(Mutex::new(AHashMap::new()))
}

/// Value-log handle. Cheap to clone; clones share the same storage.
#[derive(Clone)]
pub struct VLog {
    inner: Rc<VLogInner>,
}

/// The tenant a key is charged to for the disk quota: its 16-byte prefix as a
/// `u128` (little-endian). Scoped keys carry the tenant id as this prefix, so it
/// matches the id used at write time. Keys shorter than 16 bytes (and unscoped
/// tenant-0 keys) fall back to `0`; tenant 0 is unlimited, so the approximation
/// is invisible to enforcement.
fn tenant_from_key(key: &[u8]) -> u128 {
    if key.len() >= 16 {
        u128::from_le_bytes(key[..16].try_into().expect("checked >= 16"))
    } else {
        0
    }
}

/// Add a recovered index's live disk bytes into the (possibly shared) per-tenant
/// counter. Called once per shard at open, so a shared counter accumulates the
/// whole shard set's totals.
fn recover_tenant_disk(index: &Index, disk: &Mutex<AHashMap<u128, u64>>) {
    let mut g = disk.lock();
    for (key, entry) in index.iter() {
        *g.entry(tenant_from_key(key)).or_insert(0) += u64::from(entry.size);
    }
}

impl VLog {
    /// Open (or create) a `VLog` in `dir`, using the default 512 MiB segment size.
    ///
    /// # Errors
    ///
    /// Returns an error on IO failure.
    pub async fn open(dir: &Path) -> Result<Self> {
        Self::open_with_max_segment(dir, MAX_SEGMENT_SIZE).await
    }

    /// Open with the default segment size, sharing `tenant_disk` across a shard
    /// set so the disk quota is global per tenant.
    ///
    /// # Errors
    ///
    /// Returns an error on IO failure.
    pub async fn open_with_shared_disk(
        dir: &Path,
        tenant_disk: Arc<Mutex<AHashMap<u128, u64>>>,
    ) -> Result<Self> {
        Self::open_shared(dir, MAX_SEGMENT_SIZE, tenant_disk).await
    }

    /// Open with an explicit max segment size, using a fresh (per-`VLog`) disk
    /// counter. A standalone `VLog` is a single shard, so its counter is already
    /// the global per-tenant total.
    ///
    /// # Errors
    ///
    /// Returns an error on IO failure.
    pub async fn open_with_max_segment(dir: &Path, max_seg_size: u64) -> Result<Self> {
        Self::open_shared(dir, max_seg_size, Arc::new(Mutex::new(AHashMap::new()))).await
    }

    /// Open sharing `tenant_disk` across a shard set, so the disk quota is global
    /// per tenant. Each shard adds its recovered live bytes into the shared map.
    ///
    /// `async` even though it does not await: it starts a [`GroupCommitter`],
    /// whose internal `tokio::spawn` requires an active runtime context.
    ///
    /// # Errors
    ///
    /// Returns an error on IO failure.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::unused_async,
        clippy::too_many_lines
    )]
    async fn open_shared(
        dir: &Path,
        max_seg_size: u64,
        tenant_disk: Arc<Mutex<AHashMap<u128, u64>>>,
    ) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let seg_ids = list_segments(dir)?;
        let last_id = seg_ids.last().copied();

        // Open every segment file.
        let mut read_segments: Vec<ReadSegment> = seg_ids
            .iter()
            .map(|&id| {
                PlatformFile::open(&segment_path(dir, id)).map(|pf| ReadSegment {
                    id,
                    file: Arc::new(pf),
                    live: Cell::new(0),
                })
            })
            .collect::<std::io::Result<Vec<_>>>()?;

        // Fast path: a snapshot is usable only if every segment it references
        // still exists (a compaction since the snapshot would have removed one).
        let snap = snapshot::read(dir).filter(|s| {
            s.entries
                .iter()
                .all(|(_, e)| seg_ids.binary_search(&e.segment_id).is_ok())
        });

        let mut index = Index::new();
        let mut max_ts = 0u64;

        if let Some(snap) = snap {
            // Seed the index from the snapshot, then rescan only the segments
            // written since (id >= hwm), applying records last-wins.
            max_ts = snap.max_ts;
            for (key, entry) in snap.entries {
                index.set(key, entry);
            }
            for seg in &read_segments {
                if seg.id < snap.hwm {
                    continue;
                }
                let id = seg.id;
                let last_valid = scan_file(&seg.file, |offset, rec| {
                    max_ts = max_ts.max(rec.ts);
                    if rec.kind == RecordKind::Tombstone {
                        index.remove(&rec.key);
                    } else {
                        let entry = IndexEntry {
                            fingerprint: fingerprint(&rec.key),
                            segment_id: id,
                            _pad: 0,
                            offset: offset as u32,
                            size: padded_record_size(rec.key.len(), rec.value.len()) as u32,
                        };
                        index.set(rec.key, entry);
                    }
                })?;
                if Some(id) == last_id {
                    seg.file.truncate_sync(last_valid)?;
                }
            }
        } else {
            // Full scan: keep the highest-timestamp record per key.
            let mut winners: AHashMap<Vec<u8>, (u64, RecordKind, IndexEntry)> = AHashMap::new();
            for seg in &read_segments {
                let id = seg.id;
                let last_valid = scan_file(&seg.file, |offset, rec| {
                    if rec.ts > max_ts {
                        max_ts = rec.ts;
                    }
                    let newer = winners
                        .get(&rec.key)
                        .is_none_or(|&(wts, _, _)| rec.ts > wts);
                    if newer {
                        let entry = IndexEntry {
                            fingerprint: fingerprint(&rec.key),
                            segment_id: id,
                            _pad: 0,
                            offset: offset as u32,
                            size: padded_record_size(rec.key.len(), rec.value.len()) as u32,
                        };
                        winners.insert(rec.key, (rec.ts, rec.kind, entry));
                    }
                })?;
                if Some(id) == last_id {
                    seg.file.truncate_sync(last_valid)?;
                }
            }
            for (key, (_ts, kind, entry)) in winners {
                if kind != RecordKind::Tombstone {
                    index.set(key, entry);
                }
            }
        }

        // Seed the per-segment live-bytes counters from the final index.
        for (_key, entry) in index.iter() {
            if let Some(seg) = read_segments.iter().find(|s| s.id == entry.segment_id) {
                seg.live.set(seg.live.get() + u64::from(entry.size));
            }
        }

        let (active_id, active_size, active_file) = if let Some(seg) = read_segments.last() {
            (seg.id, seg.file.size()?, seg.file.clone())
        } else {
            let pf = Arc::new(PlatformFile::create(&segment_path(dir, 0))?);
            read_segments.push(ReadSegment {
                id: 0,
                file: pf.clone(),
                live: Cell::new(0),
            });
            (0u16, 0u64, pf)
        };

        let committer = GroupCommitter::start(active_file, active_size).await;

        // Add this shard's recovered live bytes into the (shared) disk counter.
        recover_tenant_disk(&index, &tenant_disk);

        Ok(Self {
            inner: Rc::new(VLogInner {
                dir: dir.to_owned(),
                max_seg_size,
                read_segments: RefCell::new(read_segments),
                index: RefCell::new(index),
                cache: RefCell::new(S3Fifo::new(CACHE_BUDGET_BYTES)),
                tenant_disk,
                active: RefCell::new(ActiveState {
                    id: active_id,
                    size: active_size,
                    committer,
                }),
                clock: Cell::new(max_ts + 1),
            }),
        })
    }

    /// GET a key. Returns `None` if not found.
    ///
    /// Checks the hot-key cache first; on a miss reads the vLog and populates it.
    ///
    /// # Errors
    ///
    /// Returns an error on IO failure, CRC mismatch, or corrupt record.
    pub async fn get(&self, key: &[u8]) -> Result<Option<Bytes>> {
        self.get_scoped(key, 0).await
    }

    /// GET a key, charging any read-path cache insert to `tenant`. Tenant `0` is
    /// the unscoped default; reach this via [`VLog::tenant`] for a scoped view.
    async fn get_scoped(&self, key: &[u8], tenant: u128) -> Result<Option<Bytes>> {
        if let Some(value) = self.inner.cache.borrow_mut().get(key) {
            return Ok(Some(value));
        }
        let entry = { self.inner.index.borrow().get(key).copied() };
        let Some(entry) = entry else {
            return Ok(None);
        };
        let file = {
            let segs = self.inner.read_segments.borrow();
            segs.iter()
                .find(|s| s.id == entry.segment_id)
                .map(|s| s.file.clone())
        };
        let file = file.ok_or(Error::InvalidRecord {
            msg: "index references missing segment",
        })?;
        let buf = file
            .pread(u64::from(entry.offset), entry.size as usize)
            .await?;
        let rec = decode_record(&buf)?;
        let value = Bytes::from(rec.value);
        self.inner
            .cache
            .borrow_mut()
            .insert_for(key, value.clone(), value.len(), tenant);
        Ok(Some(value))
    }

    /// SET key = value at the given durability.
    ///
    /// # Errors
    ///
    /// Returns an error on IO failure.
    pub async fn set(&self, key: &[u8], value: &[u8], durability: Durability) -> Result<()> {
        self.set_scoped(key, value, durability, 0, None).await
    }

    /// SET key = value, charging the write-through cache entry to `tenant` and
    /// enforcing `disk_limit` (if any) against the key's tenant disk quota.
    /// Tenant `0` is the unscoped default; reach this via [`VLog::tenant`].
    ///
    /// An over-limit set is rejected with [`Error::DiskQuota`] BEFORE anything
    /// is written, so storage never crosses the limit.
    async fn set_scoped(
        &self,
        key: &[u8],
        value: &[u8],
        durability: Durability,
        tenant: u128,
        disk_limit: Option<u64>,
    ) -> Result<()> {
        // Prior on-disk charge for this key, and the prospective new one. Both
        // known without writing: enforce the disk quota first.
        let prev = { self.inner.index.borrow().get(key).copied() };
        let old_disk = prev.map_or(0, |p| u64::from(p.size));
        let dtenant = tenant_from_key(key);
        let new_disk = padded_record_size(key.len(), value.len()) as u64;
        if let Some(limit) = disk_limit {
            let projected = self
                .tenant_disk_bytes(dtenant)
                .saturating_sub(old_disk)
                .saturating_add(new_disk);
            if projected > limit {
                return Err(Error::DiskQuota);
            }
        }

        let ts = self.next_ts();
        let (seg_id, offset, padded) = self
            .append_raw(key, value, RecordKind::Scalar, ts, durability)
            .await?;

        if let Some(prev) = prev {
            self.dec_live(prev.segment_id, prev.size);
        }
        self.inc_live(seg_id, padded);

        let entry = IndexEntry {
            fingerprint: fingerprint(key),
            segment_id: seg_id,
            _pad: 0,
            offset,
            size: padded,
        };
        self.inner.index.borrow_mut().set(key.to_vec(), entry);
        // Write-through: a just-written key is likely to be read back hot.
        self.inner.cache.borrow_mut().insert_for(
            key,
            Bytes::copy_from_slice(value),
            value.len(),
            tenant,
        );
        // Disk accounting: replace this key's old charge with the new one. The
        // tenant is key-derived so write-time and recovery agree for scoped keys.
        {
            let mut disk = self.inner.tenant_disk.lock();
            let e = disk.entry(dtenant).or_insert(0);
            *e = e.saturating_sub(old_disk) + u64::from(padded);
        }
        Ok(())
    }

    /// DEL a key at the given durability. Returns `true` if the key existed.
    ///
    /// # Errors
    ///
    /// Returns an error on IO failure.
    pub async fn del(&self, key: &[u8], durability: Durability) -> Result<bool> {
        let prev = { self.inner.index.borrow().get(key).copied() };
        let Some(prev) = prev else {
            return Ok(false);
        };

        let ts = self.next_ts();
        self.append_raw(key, b"", RecordKind::Tombstone, ts, durability)
            .await?;
        self.dec_live(prev.segment_id, prev.size);

        self.inner.index.borrow_mut().remove(key);
        self.inner.cache.borrow_mut().remove(key);
        // Disk quota: release the deleted record's bytes, key-derived tenant.
        {
            let dtenant = tenant_from_key(key);
            let mut disk = self.inner.tenant_disk.lock();
            if let Some(e) = disk.get_mut(&dtenant) {
                *e = e.saturating_sub(u64::from(prev.size));
                if *e == 0 {
                    disk.remove(&dtenant);
                }
            }
        }
        Ok(true)
    }

    /// MGET multiple keys. Returns a `Vec` parallel to `keys`.
    ///
    /// # Errors
    ///
    /// Returns an error on IO failure, CRC mismatch, or corrupt record.
    pub async fn mget(&self, keys: &[&[u8]]) -> Result<Vec<Option<Bytes>>> {
        let mut out = Vec::with_capacity(keys.len());
        for k in keys {
            out.push(self.get(k).await?);
        }
        Ok(out)
    }

    /// Flush the active segment's pending writes to durable storage.
    ///
    /// # Errors
    ///
    /// Returns an error if the flush fails.
    pub async fn flush(&self) -> Result<()> {
        let committer = { self.inner.active.borrow().committer.clone() };
        committer.flush().await?;
        Ok(())
    }

    /// Write an index snapshot for fast recovery. A later `open` loads the
    /// index directly and rescans only segments written after this point,
    /// turning recovery from O(dataset) into O(recent writes).
    ///
    /// # Errors
    ///
    /// Returns an error if the snapshot file cannot be written.
    pub async fn write_snapshot(&self) -> Result<()> {
        let hwm = self.inner.active.borrow().id;
        let max_ts = self.inner.clock.get();
        let entries: Vec<(Vec<u8>, IndexEntry)> = {
            self.inner
                .index
                .borrow()
                .iter()
                .map(|(k, e)| (k.to_vec(), *e))
                .collect()
        };
        let dir = self.inner.dir.clone();
        tokio::task::spawn_blocking(move || snapshot::write(&dir, hwm, max_ts, &entries))
            .await
            .map_err(|e| Error::Io(std::io::Error::other(e.to_string())))??;
        Ok(())
    }

    /// Number of live keys in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.index.borrow().len()
    }

    /// True if no live keys.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.index.borrow().is_empty()
    }

    /// Number of segment files currently open.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.inner.read_segments.borrow().len()
    }

    /// Approximate total bytes the vlog owns on disk, including dead
    /// records pending compaction. Sealed segments are sized at their
    /// rotation cap (`max_seg_size`); the trailing active segment uses
    /// its current write offset. Cheap to call (no `stat()`), suitable
    /// for the `STATS` / `/metrics` refresh path.
    #[must_use]
    pub fn disk_bytes_total(&self) -> u64 {
        let segs = self.inner.read_segments.borrow();
        let active_id = self.inner.active.borrow().id;
        let active_size = self.inner.active.borrow().size;
        let sealed_count = segs.iter().filter(|s| s.id != active_id).count() as u64;
        sealed_count.saturating_mul(self.inner.max_seg_size) + active_size
    }

    /// Scope storage operations to `tenant` for per-tenant cache accounting.
    ///
    /// Returns a zero-cost [`TenantView`] (a `&VLog` plus the tenant id). Every
    /// write/read through the view charges its cache residency to `tenant`,
    /// instead of the unscoped default `0` used by the bare [`VLog`] methods.
    /// This is the first-class scope on which per-tenant quota, eviction
    /// fairness, and snapshots hang.
    #[must_use]
    pub fn tenant(&self, tenant: u128) -> TenantView<'_> {
        TenantView {
            vlog: self,
            tenant,
            disk_limit: None,
        }
    }

    /// Bytes of hot-key cache currently charged to `tenant` (0 if none).
    #[must_use]
    pub fn tenant_cache_bytes(&self, tenant: u128) -> usize {
        self.inner.cache.borrow().charged_bytes(tenant)
    }

    /// Live on-disk bytes (padded KV records) charged to `tenant` (0 if none).
    #[must_use]
    pub fn tenant_disk_bytes(&self, tenant: u128) -> u64 {
        self.inner
            .tenant_disk
            .lock()
            .get(&tenant)
            .copied()
            .unwrap_or(0)
    }

    /// Hot-key cache statistics: `(bytes_used, evictions, n_keys, byte_budget)`.
    #[must_use]
    pub fn cache_stats(&self) -> (u64, u64, u64, u64) {
        let cache = self.inner.cache.borrow();
        (
            cache.current_bytes() as u64,
            cache.evictions(),
            self.inner.index.borrow().len() as u64,
            cache.budget() as u64,
        )
    }

    /// Total `pread` calls across all segments; a cache-effectiveness stat.
    /// The recovery scan uses synchronous reads and is not counted.
    #[must_use]
    pub fn disk_reads(&self) -> u64 {
        self.inner
            .read_segments
            .borrow()
            .iter()
            .map(|s| s.file.read_count())
            .sum()
    }

    // ── Compaction ────────────────────────────────────────────────────────────

    /// Pick a non-active segment whose live-bytes ratio is below `threshold`.
    #[must_use]
    pub fn pick_compaction_candidate(&self, threshold: f64) -> Option<u16> {
        let active_id = self.inner.active.borrow().id;
        let segs = self.inner.read_segments.borrow();
        for s in segs.iter() {
            if s.id == active_id {
                continue;
            }
            let total = s.file.size().unwrap_or(0);
            if total == 0 {
                continue;
            }
            #[allow(clippy::cast_precision_loss)]
            let ratio = s.live.get() as f64 / total as f64;
            if ratio < threshold {
                return Some(s.id);
            }
        }
        None
    }

    /// Compact one segment if any is below the live-ratio threshold.
    ///
    /// Returns the id of the segment that was compacted, if any.
    ///
    /// # Errors
    ///
    /// Returns an error on IO failure.
    pub async fn maybe_compact(&self) -> Result<Option<u16>> {
        match self.pick_compaction_candidate(COMPACTION_LIVE_RATIO) {
            Some(id) => {
                self.compact_segment(id).await?;
                Ok(Some(id))
            }
            None => Ok(None),
        }
    }

    /// Compact `seg_id`: relocate its still-live records into the active
    /// segment, then delete the source file. The active segment is never
    /// compacted.
    ///
    /// Relocations are written at `Relaxed` durability so they batch through
    /// the committer; before the source is unlinked, every destination segment
    /// gets one `F_FULLFSYNC`. This turns compaction from one fsync per record
    /// into one per destination segment (usually one, two if the active
    /// segment rotated mid-compaction).
    ///
    /// Returns the number of live records relocated.
    ///
    /// # Errors
    ///
    /// Returns an error on IO failure.
    pub async fn compact_segment(&self, seg_id: u16) -> Result<usize> {
        if self.inner.active.borrow().id == seg_id {
            return Ok(0); // never compact the segment being appended to
        }
        let file = {
            let segs = self.inner.read_segments.borrow();
            segs.iter().find(|s| s.id == seg_id).map(|s| s.file.clone())
        };
        let Some(file) = file else {
            return Ok(0); // already gone
        };
        // Telemetry: track that this compaction is in flight from now
        // until any return path. The RAII guard handles the `--` side
        // for all early returns and the normal end.
        let _gauge = InFlightCompaction::start();

        // Scan the source segment fully (synchronous reads).
        let mut records: Vec<(u64, Record)> = Vec::new();
        scan_file(&file, |off, rec| records.push((off, rec)))?;

        let mut moved = 0usize;
        // Track padded bytes relocated during this compaction run so we can
        // tick the `CompactionBytesTotal` counter once at the end. Padded
        // sizes are what we actually wrote to the destination segment.
        let mut moved_bytes: u64 = 0;
        let mut dest_segments: Vec<u16> = Vec::new();
        let note_dest = |seg: u16, dests: &mut Vec<u16>| {
            if !dests.contains(&seg) {
                dests.push(seg);
            }
        };

        for (offset, rec) in records {
            if rec.kind == RecordKind::Tombstone {
                // Carry the tombstone forward only while the key stays deleted.
                // A concurrent re-SET (higher timestamp) wins on recovery, so a
                // stale carried tombstone can never resurrect a key.
                let still_deleted = self.inner.index.borrow().get(&rec.key).is_none();
                if still_deleted {
                    let (nseg, _, _) = self
                        .append_raw(
                            &rec.key,
                            b"",
                            RecordKind::Tombstone,
                            rec.ts,
                            Durability::Relaxed,
                        )
                        .await?;
                    note_dest(nseg, &mut dest_segments);
                }
                continue;
            }

            // A data record is live iff the index still points at this exact
            // location.
            if !self.index_points_at(&rec.key, seg_id, offset) {
                continue;
            }
            let (nseg, noff, npad) = self
                .append_raw(&rec.key, &rec.value, rec.kind, rec.ts, Durability::Relaxed)
                .await?;
            moved += 1;
            moved_bytes += u64::from(npad);
            note_dest(nseg, &mut dest_segments);

            // Recheck-CAS: a concurrent SET may have moved the key during the
            // await above. Only adopt the relocated copy if nothing changed.
            if self.index_points_at(&rec.key, seg_id, offset) {
                let entry = IndexEntry {
                    fingerprint: fingerprint(&rec.key),
                    segment_id: nseg,
                    _pad: 0,
                    offset: noff,
                    size: npad,
                };
                self.inner.index.borrow_mut().set(rec.key.clone(), entry);
                self.inc_live(nseg, npad);
            }
            // else: the relocated copy is dead bytes in the active segment; it
            // will be reclaimed when that segment is itself compacted.
        }

        // Each relocation's `append_raw` resolved only after its `write_at`, so
        // every relocated byte is in the page cache. One `F_FULLFSYNC` per
        // destination segment makes them power-durable before the source goes.
        for dest in dest_segments {
            let dest_file = {
                let segs = self.inner.read_segments.borrow();
                segs.iter().find(|s| s.id == dest).map(|s| s.file.clone())
            };
            if let Some(dest_file) = dest_file {
                dest_file.sync_durable().await?;
            }
        }

        // Telemetry: amount of live data the compaction moved across to
        // new segments. Ticked once per successful run.
        skeg_telemetry::add_counter(skeg_telemetry::Counter::CompactionBytesTotal, moved_bytes);

        self.inner
            .read_segments
            .borrow_mut()
            .retain(|s| s.id != seg_id);
        std::fs::remove_file(segment_path(&self.inner.dir, seg_id))?;
        // Any existing snapshot may reference the now-removed segment.
        let _ = snapshot::remove(&self.inner.dir);
        Ok(moved)
    }

    fn index_points_at(&self, key: &[u8], seg_id: u16, offset: u64) -> bool {
        self.inner
            .index
            .borrow()
            .get(key)
            .is_some_and(|e| e.segment_id == seg_id && u64::from(e.offset) == offset)
    }

    // ── Internals ─────────────────────────────────────────────────────────────

    /// Encode and append a record at `ts`, returning `(segment_id, offset, padded_size)`.
    #[allow(clippy::cast_possible_truncation)]
    async fn append_raw(
        &self,
        key: &[u8],
        value: &[u8],
        kind: RecordKind,
        ts: u64,
        durability: Durability,
    ) -> Result<(u16, u32, u32)> {
        self.maybe_rotate(padded_record_size(key.len(), value.len()) as u64)
            .await?;
        let encoded = encode_record(key, value, kind, ts);
        let (committer, seg_id) = {
            let a = self.inner.active.borrow();
            (a.committer.clone(), a.id)
        };
        let (offset, padded) = committer.append(encoded, durability).await?;
        self.bump_active_size(seg_id, offset + u64::from(padded));
        Ok((seg_id, offset as u32, padded))
    }

    fn next_ts(&self) -> u64 {
        let t = self.inner.clock.get();
        self.inner.clock.set(t + 1);
        t
    }

    fn inc_live(&self, seg_id: u16, bytes: u32) {
        if let Some(s) = self
            .inner
            .read_segments
            .borrow()
            .iter()
            .find(|s| s.id == seg_id)
        {
            s.live.set(s.live.get() + u64::from(bytes));
        }
    }

    fn dec_live(&self, seg_id: u16, bytes: u32) {
        if let Some(s) = self
            .inner
            .read_segments
            .borrow()
            .iter()
            .find(|s| s.id == seg_id)
        {
            s.live.set(s.live.get().saturating_sub(u64::from(bytes)));
        }
    }

    /// Update the tracked active-segment size, but only if `seg_id` is still
    /// the active segment (a rotation may have happened during the `.await`).
    fn bump_active_size(&self, seg_id: u16, end_offset: u64) {
        let mut a = self.inner.active.borrow_mut();
        if a.id == seg_id {
            a.size = a.size.max(end_offset);
        }
    }

    /// Rotate to a fresh segment if the next record would overflow the active
    /// one. Awaits when the new committer is `DeviceGlobal`-backed (the
    /// `SharedCommitter` attach round-trips through its bg task); the
    /// `PerFile` path resolves synchronously.
    async fn maybe_rotate(&self, incoming: u64) -> Result<()> {
        let needs = {
            let a = self.inner.active.borrow();
            a.size + incoming > self.inner.max_seg_size
        };
        if !needs {
            return Ok(());
        }

        let old_id = self.inner.active.borrow().id;
        let new_id = old_id.checked_add(1).ok_or(Error::InvalidRecord {
            msg: "segment id overflow",
        })?;
        let pf = Arc::new(PlatformFile::create(&segment_path(
            &self.inner.dir,
            new_id,
        ))?);
        let committer = GroupCommitter::start(pf.clone(), 0).await;
        self.inner.read_segments.borrow_mut().push(ReadSegment {
            id: new_id,
            file: pf,
            live: Cell::new(0),
        });
        // VlogSegmentsLive + VlogTotalBytes are refreshed from the
        // server's STATS handler, which has cheap access to both via
        // `segment_count()` and `disk_bytes_total()`; no per-rotation
        // gauge writes here.
        *self.inner.active.borrow_mut() = ActiveState {
            id: new_id,
            size: 0,
            committer,
        };
        Ok(())
    }
}

/// A [`VLog`] scoped to one tenant for per-tenant cache accounting.
///
/// Created by [`VLog::tenant`]. Zero-cost: a borrow of the `VLog` plus the
/// tenant id. Operations delegate to the same private core the bare `VLog`
/// methods use, so there is one implementation, not a parallel API. This is the
/// extension point for per-tenant quota, eviction fairness, and snapshots.
#[derive(Clone, Copy)]
pub struct TenantView<'a> {
    vlog: &'a VLog,
    tenant: u128,
    /// Disk-quota limit applied on `set`. `None` skips enforcement.
    disk_limit: Option<u64>,
}

impl TenantView<'_> {
    /// Apply a disk-quota limit on this view's `set`. An over-limit set is
    /// rejected with [`Error::DiskQuota`] before anything is written.
    #[must_use]
    pub fn with_disk_limit(mut self, limit: Option<u64>) -> Self {
        self.disk_limit = limit;
        self
    }

    /// GET a key, charging any read-path cache insert to this tenant.
    ///
    /// # Errors
    ///
    /// Returns an error on IO failure, CRC mismatch, or corrupt record.
    pub async fn get(&self, key: &[u8]) -> Result<Option<Bytes>> {
        self.vlog.get_scoped(key, self.tenant).await
    }

    /// SET key = value, charging the write-through cache entry to this tenant.
    ///
    /// # Errors
    ///
    /// Returns an error on IO failure.
    pub async fn set(&self, key: &[u8], value: &[u8], durability: Durability) -> Result<()> {
        self.vlog
            .set_scoped(key, value, durability, self.tenant, self.disk_limit)
            .await
    }

    /// DEL a key. Returns `true` if it existed. Cache release is self-attributed
    /// from the evicted entry, so this needs no tenant argument.
    ///
    /// # Errors
    ///
    /// Returns an error on IO failure.
    pub async fn del(&self, key: &[u8], durability: Durability) -> Result<bool> {
        self.vlog.del(key, durability).await
    }

    /// MGET keys, charging read-path cache inserts to this tenant.
    ///
    /// # Errors
    ///
    /// Returns an error on IO failure, CRC mismatch, or corrupt record.
    pub async fn mget(&self, keys: &[&[u8]]) -> Result<Vec<Option<Bytes>>> {
        let mut out = Vec::with_capacity(keys.len());
        for k in keys {
            out.push(self.vlog.get_scoped(k, self.tenant).await?);
        }
        Ok(out)
    }
}

/// RAII telemetry guard for [`VLog::compact_segment`]. Increments the
/// `CompactionInProgress` + `VlogSegmentsCompacting` gauges on
/// construction and decrements both on drop, so every return path
/// (early return, `?`, normal end) leaves the gauges balanced.
struct InFlightCompaction;

impl InFlightCompaction {
    fn start() -> Self {
        skeg_telemetry::incr_gauge(skeg_telemetry::Gauge::CompactionInProgress);
        skeg_telemetry::incr_gauge(skeg_telemetry::Gauge::VlogSegmentsCompacting);
        Self
    }
}

impl Drop for InFlightCompaction {
    fn drop(&mut self) {
        skeg_telemetry::decr_gauge(skeg_telemetry::Gauge::CompactionInProgress);
        skeg_telemetry::decr_gauge(skeg_telemetry::Gauge::VlogSegmentsCompacting);
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_get_not_found() {
        let dir = TempDir::new().unwrap();
        let v = VLog::open(dir.path()).await.unwrap();
        assert!(v.get(b"missing").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_set_get_roundtrip() {
        let dir = TempDir::new().unwrap();
        let v = VLog::open(dir.path()).await.unwrap();
        v.set(b"hello", b"world", Durability::Kernel).await.unwrap();
        assert_eq!(
            v.get(b"hello").await.unwrap().as_deref(),
            Some(b"world".as_slice())
        );
    }

    #[tokio::test]
    async fn test_overwrite_returns_latest() {
        let dir = TempDir::new().unwrap();
        let v = VLog::open(dir.path()).await.unwrap();
        v.set(b"k", b"v1", Durability::Kernel).await.unwrap();
        v.set(b"k", b"v2", Durability::Kernel).await.unwrap();
        assert_eq!(
            v.get(b"k").await.unwrap().as_deref(),
            Some(b"v2".as_slice())
        );
    }

    #[tokio::test]
    async fn test_del_not_found_returns_false() {
        let dir = TempDir::new().unwrap();
        let v = VLog::open(dir.path()).await.unwrap();
        assert!(!v.del(b"ghost", Durability::Kernel).await.unwrap());
    }

    #[tokio::test]
    async fn test_mget_mixed() {
        let dir = TempDir::new().unwrap();
        let v = VLog::open(dir.path()).await.unwrap();
        v.set(b"a", b"1", Durability::Kernel).await.unwrap();
        v.set(b"b", b"2", Durability::Kernel).await.unwrap();
        let res = v
            .mget(&[b"a".as_slice(), b"missing", b"b".as_slice()])
            .await
            .unwrap();
        assert_eq!(res[0].as_deref(), Some(b"1".as_slice()));
        assert!(res[1].is_none());
        assert_eq!(res[2].as_deref(), Some(b"2".as_slice()));
    }

    // ── TenantView per-tenant cache attribution ───────────────────────────────

    #[tokio::test]
    async fn test_tenant_view_set_charges_tenant() {
        let dir = TempDir::new().unwrap();
        let v = VLog::open(dir.path()).await.unwrap();
        v.tenant(7)
            .set(b"k", b"val", Durability::Kernel)
            .await
            .unwrap();
        assert!(
            v.tenant_cache_bytes(7) > 0,
            "write-through must charge tenant 7"
        );
        assert_eq!(v.tenant_cache_bytes(0), 0, "tenant 0 must hold nothing");
    }

    #[tokio::test]
    async fn test_bare_set_charges_tenant_zero() {
        let dir = TempDir::new().unwrap();
        let v = VLog::open(dir.path()).await.unwrap();
        v.set(b"k", b"v", Durability::Kernel).await.unwrap();
        assert!(v.tenant_cache_bytes(0) > 0, "bare set charges tenant 0");
        assert_eq!(v.tenant_cache_bytes(7), 0);
    }

    #[tokio::test]
    async fn test_tenant_view_two_tenants_isolated_accounting() {
        let dir = TempDir::new().unwrap();
        let v = VLog::open(dir.path()).await.unwrap();
        v.tenant(7)
            .set(b"a", b"x", Durability::Kernel)
            .await
            .unwrap();
        v.tenant(9)
            .set(b"b", b"yy", Durability::Kernel)
            .await
            .unwrap();
        let b7 = v.tenant_cache_bytes(7);
        let b9 = v.tenant_cache_bytes(9);
        assert!(b7 > 0 && b9 > 0);
        assert!(b9 > b7, "tenant 9 wrote a larger value, charged more");
        assert_eq!(v.tenant_cache_bytes(0), 0);
    }

    #[tokio::test]
    async fn test_tenant_view_del_releases_tenant() {
        let dir = TempDir::new().unwrap();
        let v = VLog::open(dir.path()).await.unwrap();
        v.tenant(7)
            .set(b"k", b"v", Durability::Kernel)
            .await
            .unwrap();
        assert!(v.tenant_cache_bytes(7) > 0);
        assert!(v.tenant(7).del(b"k", Durability::Kernel).await.unwrap());
        assert_eq!(
            v.tenant_cache_bytes(7),
            0,
            "del releases the tenant's bytes"
        );
    }

    #[tokio::test]
    async fn test_tenant_view_read_path_charges_tenant() {
        // After reopen the cache is cold; a get must charge the read value to the
        // querying tenant, not tenant 0.
        let dir = TempDir::new().unwrap();
        {
            let v = VLog::open(dir.path()).await.unwrap();
            v.tenant(7)
                .set(b"k", b"val", Durability::Kernel)
                .await
                .unwrap();
            v.write_snapshot().await.unwrap();
            v.flush().await.unwrap();
        }
        let v = VLog::open(dir.path()).await.unwrap();
        assert_eq!(v.tenant_cache_bytes(7), 0, "cold cache after reopen");
        assert_eq!(
            v.tenant(7).get(b"k").await.unwrap().as_deref(),
            Some(b"val".as_slice())
        );
        assert!(
            v.tenant_cache_bytes(7) > 0,
            "read-path cache insert must charge the querying tenant"
        );
        assert_eq!(v.tenant_cache_bytes(0), 0);
    }

    // ── Per-tenant disk_bytes tracking ────────────────────────────────────────

    /// A scoped key: tenant id (16B LE) prefix + raw key, matching how the
    /// RESP3 handler scopes keys. `tenant_from_key` recovers the tenant.
    fn scoped(tenant: u128, raw: &[u8]) -> Vec<u8> {
        let mut k = tenant.to_le_bytes().to_vec();
        k.extend_from_slice(raw);
        k
    }

    #[tokio::test]
    async fn test_disk_set_tracks_per_tenant() {
        let dir = TempDir::new().unwrap();
        let v = VLog::open(dir.path()).await.unwrap();
        let k = scoped(7, b"a");
        v.set(&k, b"val", Durability::Kernel).await.unwrap();
        assert_eq!(
            v.tenant_disk_bytes(7),
            padded_record_size(k.len(), 3) as u64
        );
        assert_eq!(v.tenant_disk_bytes(9), 0, "other tenant uncharged");
    }

    #[tokio::test]
    async fn test_disk_overwrite_replaces_not_adds() {
        let dir = TempDir::new().unwrap();
        let v = VLog::open(dir.path()).await.unwrap();
        let k = scoped(7, b"a");
        v.set(&k, b"x", Durability::Kernel).await.unwrap();
        let small = v.tenant_disk_bytes(7);
        let big = vec![0u8; 200];
        v.set(&k, &big, Durability::Kernel).await.unwrap();
        assert_eq!(
            v.tenant_disk_bytes(7),
            padded_record_size(k.len(), 200) as u64,
            "overwrite replaces the charge, not adds"
        );
        assert!(v.tenant_disk_bytes(7) > small);
    }

    #[tokio::test]
    async fn test_disk_del_releases() {
        let dir = TempDir::new().unwrap();
        let v = VLog::open(dir.path()).await.unwrap();
        let k = scoped(7, b"a");
        v.set(&k, b"v", Durability::Kernel).await.unwrap();
        assert!(v.tenant_disk_bytes(7) > 0);
        assert!(v.del(&k, Durability::Kernel).await.unwrap());
        assert_eq!(v.tenant_disk_bytes(7), 0, "del releases disk bytes");
    }

    #[tokio::test]
    async fn test_disk_recovered_on_reopen() {
        let dir = TempDir::new().unwrap();
        let k1 = scoped(7, b"a");
        let k2 = scoped(7, b"b");
        let total;
        {
            let v = VLog::open(dir.path()).await.unwrap();
            v.set(&k1, b"xx", Durability::Kernel).await.unwrap();
            v.set(&k2, b"yyyy", Durability::Kernel).await.unwrap();
            total = v.tenant_disk_bytes(7);
            v.write_snapshot().await.unwrap();
            v.flush().await.unwrap();
        }
        let v = VLog::open(dir.path()).await.unwrap();
        assert!(total > 0);
        assert_eq!(
            v.tenant_disk_bytes(7),
            total,
            "per-tenant disk rebuilt from the recovered index"
        );
    }

    #[tokio::test]
    async fn test_len_and_is_empty() {
        let dir = TempDir::new().unwrap();
        let v = VLog::open(dir.path()).await.unwrap();
        assert!(v.is_empty());
        v.set(b"k1", b"v1", Durability::Kernel).await.unwrap();
        v.set(b"k2", b"v2", Durability::Kernel).await.unwrap();
        assert_eq!(v.len(), 2);
        v.del(b"k1", Durability::Kernel).await.unwrap();
        assert_eq!(v.len(), 1);
    }

    #[tokio::test]
    async fn test_crash_recovery_full_scan() {
        let dir = TempDir::new().unwrap();
        {
            let v = VLog::open(dir.path()).await.unwrap();
            for i in 0u64..10 {
                v.set(
                    format!("k{i}").as_bytes(),
                    format!("v{i}").as_bytes(),
                    Durability::Kernel,
                )
                .await
                .unwrap();
            }
        }
        let v = VLog::open(dir.path()).await.unwrap();
        for i in 0u64..10 {
            let val = v.get(format!("k{i}").as_bytes()).await.unwrap();
            assert_eq!(val.as_deref(), Some(format!("v{i}").as_bytes()));
        }
    }

    #[tokio::test]
    async fn test_tombstone_hides_key() {
        let dir = TempDir::new().unwrap();
        {
            let v = VLog::open(dir.path()).await.unwrap();
            v.set(b"k", b"v", Durability::Kernel).await.unwrap();
            v.del(b"k", Durability::Kernel).await.unwrap();
        }
        let v = VLog::open(dir.path()).await.unwrap();
        assert_eq!(v.get(b"k").await.unwrap(), None);
    }

    #[tokio::test]
    async fn test_crash_mid_record() {
        use std::io::Write as _;
        let dir = TempDir::new().unwrap();
        {
            let v = VLog::open(dir.path()).await.unwrap();
            v.set(b"good", b"data", Durability::Kernel).await.unwrap();
            v.flush().await.unwrap();
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(crate::segment::segment_path(dir.path(), 0))
                .unwrap();
            f.write_all(&[0xDE, 0xAD]).unwrap();
        }
        let v = VLog::open(dir.path()).await.unwrap();
        assert_eq!(
            v.get(b"good").await.unwrap().as_deref(),
            Some(b"data".as_slice())
        );
        assert_eq!(v.len(), 1);
    }

    #[tokio::test]
    async fn test_segment_rotation() {
        let dir = TempDir::new().unwrap();
        let v = VLog::open_with_max_segment(dir.path(), 512).await.unwrap();
        for i in 0u64..20 {
            v.set(format!("rk{i}").as_bytes(), b"value", Durability::Kernel)
                .await
                .unwrap();
        }
        v.flush().await.unwrap();
        assert!(v.segment_count() >= 2, "expected rotation");
        for i in 0u64..20 {
            let got = v.get(format!("rk{i}").as_bytes()).await.unwrap();
            assert_eq!(got.as_deref(), Some(b"value".as_slice()));
        }
    }

    #[tokio::test]
    async fn test_rotation_survives_recovery() {
        let dir = TempDir::new().unwrap();
        {
            let v = VLog::open_with_max_segment(dir.path(), 512).await.unwrap();
            for i in 0u64..20 {
                v.set(format!("rk{i}").as_bytes(), b"value", Durability::Kernel)
                    .await
                    .unwrap();
            }
            v.flush().await.unwrap();
        }
        let v = VLog::open(dir.path()).await.unwrap();
        for i in 0u64..20 {
            let got = v.get(format!("rk{i}").as_bytes()).await.unwrap();
            assert_eq!(got.as_deref(), Some(b"value".as_slice()));
        }
    }

    #[tokio::test]
    async fn test_cache_hit_after_set() {
        let dir = TempDir::new().unwrap();
        let v = VLog::open(dir.path()).await.unwrap();
        v.set(b"k", b"v", Durability::Kernel).await.unwrap();
        let reads_before = v.disk_reads();
        let got = v.get(b"k").await.unwrap();
        assert_eq!(got.as_deref(), Some(b"v".as_slice()));
        assert_eq!(
            v.disk_reads(),
            reads_before,
            "cache hit must not touch disk"
        );
    }

    #[tokio::test]
    async fn test_cache_miss_triggers_disk_read() {
        let dir = TempDir::new().unwrap();
        {
            let v = VLog::open(dir.path()).await.unwrap();
            v.set(b"k", b"v", Durability::Kernel).await.unwrap();
        }
        let v = VLog::open(dir.path()).await.unwrap();
        assert_eq!(v.disk_reads(), 0);
        let got = v.get(b"k").await.unwrap();
        assert_eq!(got.as_deref(), Some(b"v".as_slice()));
        assert_eq!(v.disk_reads(), 1, "cold miss must read the disk once");
        let got2 = v.get(b"k").await.unwrap();
        assert_eq!(got2.as_deref(), Some(b"v".as_slice()));
        assert_eq!(v.disk_reads(), 1, "warm hit must not touch disk");
    }

    // ── compaction ───────────────────────────────────────────────────────────

    /// Fill segment 0 with `n` keys, then rotate so segment 0 is sealed.
    async fn fill_and_rotate(v: &VLog, n: u64, prefix: &str) {
        for i in 0..n {
            v.set(format!("{prefix}{i}").as_bytes(), b"x", Durability::Kernel)
                .await
                .unwrap();
        }
        v.flush().await.unwrap();
    }

    #[tokio::test]
    async fn test_live_ratio_estimator() {
        let dir = TempDir::new().unwrap();
        // Small segments so a handful of keys seal segment 0.
        let v = VLog::open_with_max_segment(dir.path(), 1024).await.unwrap();
        fill_and_rotate(&v, 30, "k").await;
        // No candidate yet: segment 0 is all live.
        assert_eq!(v.pick_compaction_candidate(COMPACTION_LIVE_RATIO), None);
        // Overwrite every key; the new records land in later segments, so
        // segment 0's live bytes collapse.
        for i in 0u64..30 {
            v.set(format!("k{i}").as_bytes(), b"y", Durability::Kernel)
                .await
                .unwrap();
        }
        v.flush().await.unwrap();
        assert_eq!(
            v.pick_compaction_candidate(COMPACTION_LIVE_RATIO),
            Some(0),
            "segment 0 is now mostly dead and must be a candidate",
        );
    }

    #[tokio::test]
    async fn test_compaction_removes_dead() {
        let dir = TempDir::new().unwrap();
        let v = VLog::open_with_max_segment(dir.path(), 1024).await.unwrap();
        fill_and_rotate(&v, 30, "k").await;
        for i in 0u64..30 {
            v.set(format!("k{i}").as_bytes(), b"y", Durability::Kernel)
                .await
                .unwrap();
        }
        v.flush().await.unwrap();

        let segs_before = v.segment_count();
        let moved = v.compact_segment(0).await.unwrap();
        // Every record in segment 0 was superseded: nothing live to relocate.
        assert_eq!(moved, 0, "all of segment 0 was dead");
        assert_eq!(v.segment_count(), segs_before - 1, "source segment removed");
        // Data is still intact.
        for i in 0u64..30 {
            let got = v.get(format!("k{i}").as_bytes()).await.unwrap();
            assert_eq!(got.as_deref(), Some(b"y".as_slice()));
        }
    }

    #[tokio::test]
    async fn test_compaction_relocates_live() {
        let dir = TempDir::new().unwrap();
        // 1024-byte segments hold 8 fixed-128-byte records each: segment 0 ends
        // up holding exactly k0..k7.
        let v = VLog::open_with_max_segment(dir.path(), 1024).await.unwrap();
        for i in 0u64..16 {
            v.set(format!("k{i}").as_bytes(), b"x", Durability::Kernel)
                .await
                .unwrap();
        }
        v.flush().await.unwrap();
        // Overwrite k0..k3: those four records in segment 0 become dead, k4..k7
        // stay live there.
        for i in 0u64..4 {
            v.set(format!("k{i}").as_bytes(), b"new", Durability::Kernel)
                .await
                .unwrap();
        }
        v.flush().await.unwrap();

        let moved = v.compact_segment(0).await.unwrap();
        assert_eq!(
            moved, 4,
            "the 4 still-live keys of segment 0 must be relocated"
        );
        // All keys readable, with the right values.
        for i in 0u64..16 {
            let got = v.get(format!("k{i}").as_bytes()).await.unwrap();
            let expect: &[u8] = if i < 4 { b"new" } else { b"x" };
            assert_eq!(got.as_deref(), Some(expect), "key k{i}");
        }
    }

    #[tokio::test]
    async fn test_compaction_index_consistency_after_recovery() {
        let dir = TempDir::new().unwrap();
        {
            let v = VLog::open_with_max_segment(dir.path(), 1024).await.unwrap();
            fill_and_rotate(&v, 20, "k").await;
            for i in 0u64..10 {
                v.set(format!("k{i}").as_bytes(), b"new", Durability::Kernel)
                    .await
                    .unwrap();
            }
            v.flush().await.unwrap();
            v.compact_segment(0).await.unwrap();
            v.flush().await.unwrap();
        }
        // Re-open: the relocated records must win over the (now deleted) originals.
        let v = VLog::open(dir.path()).await.unwrap();
        for i in 0u64..20 {
            let got = v.get(format!("k{i}").as_bytes()).await.unwrap();
            let expect: &[u8] = if i < 10 { b"new" } else { b"x" };
            assert_eq!(got.as_deref(), Some(expect), "key k{i} after recovery");
        }
    }

    #[tokio::test]
    async fn test_compaction_concurrent_set() {
        let dir = TempDir::new().unwrap();
        let v = VLog::open_with_max_segment(dir.path(), 1024).await.unwrap();
        fill_and_rotate(&v, 20, "k").await;

        // Compact segment 0 while concurrently overwriting one of its live keys.
        // The recheck-CAS must ensure the concurrent SET wins, never the
        // relocated stale copy.
        let (compact_res, set_res) = tokio::join!(
            v.compact_segment(0),
            v.set(b"k5", b"WINNER", Durability::Kernel),
        );
        compact_res.unwrap();
        set_res.unwrap();

        assert_eq!(
            v.get(b"k5").await.unwrap().as_deref(),
            Some(b"WINNER".as_slice())
        );

        // And it survives recovery.
        v.flush().await.unwrap();
        drop(v);
        let v = VLog::open(dir.path()).await.unwrap();
        assert_eq!(
            v.get(b"k5").await.unwrap().as_deref(),
            Some(b"WINNER".as_slice())
        );
    }

    #[tokio::test]
    async fn test_compaction_preserves_tombstone() {
        let dir = TempDir::new().unwrap();
        {
            let v = VLog::open_with_max_segment(dir.path(), 1024).await.unwrap();
            v.set(b"doomed", b"v", Durability::Kernel).await.unwrap();
            fill_and_rotate(&v, 20, "k").await; // seal segment 0 with `doomed` in it
            v.del(b"doomed", Durability::Kernel).await.unwrap(); // tombstone in a later segment
            v.flush().await.unwrap();
            // Compact segment 0: it still holds the original `doomed` record.
            v.compact_segment(0).await.unwrap();
            v.flush().await.unwrap();
        }
        // After recovery the key must stay deleted: the tombstone outranks the
        // relocated original by timestamp.
        let v = VLog::open(dir.path()).await.unwrap();
        assert_eq!(
            v.get(b"doomed").await.unwrap(),
            None,
            "deleted key must stay deleted"
        );
    }

    #[tokio::test]
    async fn test_maybe_compact_picks_and_runs() {
        let dir = TempDir::new().unwrap();
        let v = VLog::open_with_max_segment(dir.path(), 1024).await.unwrap();
        fill_and_rotate(&v, 30, "k").await;
        // Overwrite every key, so all of the early sealed segments go dead.
        for i in 0u64..30 {
            v.set(format!("k{i}").as_bytes(), b"y", Durability::Kernel)
                .await
                .unwrap();
        }
        v.flush().await.unwrap();

        // The first call must pick the oldest dead segment.
        assert_eq!(v.maybe_compact().await.unwrap(), Some(0));

        // Draining all candidates eventually leaves nothing to compact.
        let mut guard = 0;
        while v.maybe_compact().await.unwrap().is_some() {
            guard += 1;
            assert!(guard < 100, "compaction did not converge");
        }
        // All data is intact afterwards.
        for i in 0u64..30 {
            let got = v.get(format!("k{i}").as_bytes()).await.unwrap();
            assert_eq!(got.as_deref(), Some(b"y".as_slice()), "key k{i}");
        }
    }

    // ── index snapshot ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_snapshot_fast_recovery() {
        let dir = TempDir::new().unwrap();
        {
            let v = VLog::open(dir.path()).await.unwrap();
            for i in 0u64..50 {
                v.set(
                    format!("k{i}").as_bytes(),
                    format!("v{i}").as_bytes(),
                    Durability::Kernel,
                )
                .await
                .unwrap();
            }
            v.flush().await.unwrap();
            v.write_snapshot().await.unwrap();
        }
        assert!(
            crate::snapshot::read(dir.path()).is_some(),
            "snapshot file must exist"
        );
        let v = VLog::open(dir.path()).await.unwrap();
        assert_eq!(v.len(), 50);
        for i in 0u64..50 {
            let got = v.get(format!("k{i}").as_bytes()).await.unwrap();
            assert_eq!(got.as_deref(), Some(format!("v{i}").as_bytes()));
        }
    }

    #[tokio::test]
    async fn test_snapshot_post_writes_recovered() {
        let dir = TempDir::new().unwrap();
        {
            let v = VLog::open(dir.path()).await.unwrap();
            for i in 0u64..20 {
                v.set(format!("k{i}").as_bytes(), b"old", Durability::Kernel)
                    .await
                    .unwrap();
            }
            v.flush().await.unwrap();
            v.write_snapshot().await.unwrap();
            // Writes after the snapshot: new keys, an overwrite, a delete.
            for i in 20u64..30 {
                v.set(format!("k{i}").as_bytes(), b"new", Durability::Kernel)
                    .await
                    .unwrap();
            }
            v.set(b"k5", b"changed", Durability::Kernel).await.unwrap();
            v.del(b"k7", Durability::Kernel).await.unwrap();
            v.flush().await.unwrap();
        }
        let v = VLog::open(dir.path()).await.unwrap();
        assert_eq!(
            v.get(b"k0").await.unwrap().as_deref(),
            Some(b"old".as_slice())
        );
        assert_eq!(
            v.get(b"k5").await.unwrap().as_deref(),
            Some(b"changed".as_slice())
        );
        assert_eq!(
            v.get(b"k7").await.unwrap(),
            None,
            "post-snapshot delete must persist"
        );
        assert_eq!(
            v.get(b"k25").await.unwrap().as_deref(),
            Some(b"new".as_slice())
        );
    }

    #[tokio::test]
    async fn test_snapshot_invalidated_by_compaction() {
        let dir = TempDir::new().unwrap();
        {
            let v = VLog::open_with_max_segment(dir.path(), 1024).await.unwrap();
            for i in 0u64..30 {
                v.set(format!("k{i}").as_bytes(), b"x", Durability::Kernel)
                    .await
                    .unwrap();
            }
            v.flush().await.unwrap();
            v.write_snapshot().await.unwrap();
            assert!(crate::snapshot::read(dir.path()).is_some());
            for i in 0u64..30 {
                v.set(format!("k{i}").as_bytes(), b"y", Durability::Kernel)
                    .await
                    .unwrap();
            }
            v.flush().await.unwrap();
            v.compact_segment(0).await.unwrap();
            assert!(
                crate::snapshot::read(dir.path()).is_none(),
                "compaction must drop the snapshot"
            );
        }
        // Recovery falls back to a full scan and is still correct.
        let v = VLog::open(dir.path()).await.unwrap();
        for i in 0u64..30 {
            assert_eq!(
                v.get(format!("k{i}").as_bytes()).await.unwrap().as_deref(),
                Some(b"y".as_slice()),
            );
        }
    }

    #[tokio::test]
    async fn test_snapshot_corrupt_falls_back_to_scan() {
        let dir = TempDir::new().unwrap();
        {
            let v = VLog::open(dir.path()).await.unwrap();
            for i in 0u64..20 {
                v.set(format!("k{i}").as_bytes(), b"v", Durability::Kernel)
                    .await
                    .unwrap();
            }
            v.flush().await.unwrap();
        }
        std::fs::write(
            crate::snapshot::snapshot_path(dir.path()),
            b"not a real snapshot",
        )
        .unwrap();
        let v = VLog::open(dir.path()).await.unwrap();
        assert_eq!(v.len(), 20);
        assert_eq!(
            v.get(b"k3").await.unwrap().as_deref(),
            Some(b"v".as_slice())
        );
    }
}
