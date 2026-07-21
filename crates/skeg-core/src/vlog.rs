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
use futures_util::stream::{self, StreamExt};

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

/// Survivor relocations kept in flight while compacting one segment, so they
/// share group-commit flushes instead of paying one each. Matches the shard's
/// erase-sweep bound; segment rotation stays inside the concurrency ordinary
/// writes already drive.
const RELOCATE_CONCURRENCY: usize = 256;

/// One relocation's contribution: `(dest_segment, padded_bytes, is_data)`.
/// `None` when the record was skipped (a re-SET tombstone or a stale data
/// record the index no longer points at).
type RelocOutcome = Result<Option<(u16, u32, bool)>>;

/// Recovery-time gate for atomic batches ([`VLog::set_many`]). A `BatchBegin(N)`
/// header opens a run of `N` records; the gate withholds them until all `N`
/// have been scanned, then releases them for application. A run torn by a crash
/// (fewer than `N` durable before the scan hits the corrupt tail) is never
/// released, which is what makes `set_many` all-or-nothing on reopen. A batch
/// never spans segments (its blob is written to one segment), so a fresh gate
/// per segment is enough; anything still pending when a segment ends is dropped.
#[derive(Default)]
struct BatchGate {
    expect: Option<usize>,
    pending: Vec<(u64, Record)>,
}

impl BatchGate {
    /// Feed one scanned record, invoking `apply` on each record that is ready.
    /// A record outside a batch is applied straight through (zero allocation -
    /// the hot path for every normal recovery); a batch's members are held in
    /// `pending` and applied together once the batch completes.
    fn feed(&mut self, offset: u64, rec: Record, mut apply: impl FnMut(u64, Record)) {
        if rec.kind == RecordKind::BatchBegin {
            let n = rec
                .value
                .get(..4)
                .and_then(|b| b.try_into().ok())
                .map_or(0, u32::from_le_bytes) as usize;
            self.pending.clear();
            self.expect = (n > 0).then_some(n);
            return;
        }
        match self.expect {
            Some(n) => {
                self.pending.push((offset, rec));
                if self.pending.len() == n {
                    self.expect = None;
                    for (o, r) in self.pending.drain(..) {
                        apply(o, r);
                    }
                }
            }
            None => apply(offset, rec),
        }
    }
}

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
    /// Serialises segment rotation. Appends run concurrently (the shard spawns a
    /// task per request, plus the compaction/reclaim drivers), so two of them
    /// can both observe the active segment full and both try to create the same
    /// `old_id + 1` file, colliding with `EEXIST`. Rotation is rare (once per
    /// full segment), so the fast path skips the lock entirely and only a
    /// rotation pays for it; the body re-checks the fill under the lock.
    rotate_lock: tokio::sync::Mutex<()>,
    /// Serialises `append`'s read-modify-write. Requests run concurrently on the
    /// shard, so two appends to the same key would otherwise both read the old
    /// value and both write old+delta, silently dropping one delta. Held across
    /// the whole read+write, so appends on this store are serial; ordinary
    /// set/get/del are untouched.
    append_lock: tokio::sync::Mutex<()>,
    clock: Cell<u64>,
    /// Exclusive advisory lock on the store directory, held for the lifetime of
    /// the open store so a second process cannot open it concurrently and race
    /// appends into the same segments. Released when the last `VLog` clone drops
    /// (or the process exits). Declared last so it is dropped after the rest of
    /// the store state.
    _lock: skeg_platform::DirLock,
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
        // Take the store lock before touching any segment: recovery can truncate
        // a torn tail, so even opening must be single-process.
        let store_lock = skeg_platform::DirLock::acquire_exclusive(dir)?;
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
                let mut gate = BatchGate::default();
                let last_valid = scan_file(&seg.file, |offset, rec| {
                    max_ts = max_ts.max(rec.ts);
                    gate.feed(offset, rec, |off, r| {
                        if r.kind == RecordKind::Tombstone {
                            index.remove(&r.key);
                        } else {
                            let entry = IndexEntry {
                                fingerprint: fingerprint(&r.key),
                                segment_id: id,
                                _pad: 0,
                                offset: off as u32,
                                size: padded_record_size(r.key.len(), r.value.len()) as u32,
                            };
                            index.set(r.key, entry);
                        }
                    });
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
                let mut gate = BatchGate::default();
                let last_valid = scan_file(&seg.file, |offset, rec| {
                    if rec.ts > max_ts {
                        max_ts = rec.ts;
                    }
                    gate.feed(offset, rec, |off, r| {
                        let newer = winners.get(&r.key).is_none_or(|&(wts, _, _)| r.ts > wts);
                        if newer {
                            let entry = IndexEntry {
                                fingerprint: fingerprint(&r.key),
                                segment_id: id,
                                _pad: 0,
                                offset: off as u32,
                                size: padded_record_size(r.key.len(), r.value.len()) as u32,
                            };
                            winners.insert(r.key, (r.ts, r.kind, entry));
                        }
                    });
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
                rotate_lock: tokio::sync::Mutex::new(()),
                append_lock: tokio::sync::Mutex::new(()),
                clock: Cell::new(max_ts + 1),
                _lock: store_lock,
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

    /// Append `value` to `key`'s current value (creating the key if absent) and
    /// return the new value's length, mirroring Redis `APPEND`.
    ///
    /// Read-modify-write: it reads the whole current value, concatenates, and
    /// writes the whole result as one new record - so a single append is
    /// O(current value length). A key whose value grows over many appends costs
    /// O(n^2) in total; that is the naive log-store cost, fine for bounded
    /// values (e.g. a capped adjacency list) but not for an unbounded one.
    ///
    /// Atomic against concurrent appends: the read and the write are serialised
    /// by the store's `append_lock`, so two appends to the same key never both
    /// read the old value and drop a delta. Appends on one store are therefore
    /// serial; `set`/`get`/`del` are unaffected.
    ///
    /// # Errors
    ///
    /// Returns an error on IO failure or if the disk quota would be exceeded.
    pub async fn append(&self, key: &[u8], value: &[u8], durability: Durability) -> Result<u64> {
        self.append_scoped(key, value, durability, 0, None).await
    }

    async fn append_scoped(
        &self,
        key: &[u8],
        value: &[u8],
        durability: Durability,
        tenant: u128,
        disk_limit: Option<u64>,
    ) -> Result<u64> {
        let _serialise = self.inner.append_lock.lock().await;
        let combined = match self.get_scoped(key, tenant).await? {
            Some(current) => {
                let mut buf = Vec::with_capacity(current.len() + value.len());
                buf.extend_from_slice(&current);
                buf.extend_from_slice(value);
                buf
            }
            None => value.to_vec(),
        };
        let new_len = combined.len() as u64;
        self.set_scoped(key, &combined, durability, tenant, disk_limit)
            .await?;
        Ok(new_len)
    }

    /// Atomically write every `(key, value)` pair, or none. The pairs are
    /// encoded behind a single `BatchBegin(N)` header and submitted as one
    /// group-commit append - one contiguous write, one flush. If a crash tears
    /// the write, recovery finds fewer than `N` members after the header and
    /// drops the whole batch, so a reopened store never shows a partial MSET.
    ///
    /// All-or-nothing holds for THIS store only. A caller that shards keys
    /// across several `VLog`s gets per-store atomicity, not a global
    /// transaction - there is no cross-store coordination here.
    ///
    /// Later pairs win over earlier ones for a duplicate key (each carries a
    /// higher timestamp), matching last-write-wins.
    ///
    /// # Errors
    ///
    /// Returns an error on IO failure, or if the encoded batch exceeds one
    /// segment (`max_seg_size`) - split into smaller batches.
    pub async fn set_many(&self, pairs: &[(&[u8], &[u8])], durability: Durability) -> Result<()> {
        if pairs.is_empty() {
            return Ok(());
        }

        // Build the contiguous blob: BatchBegin(N) then one Scalar per pair,
        // each with its own increasing timestamp so a repeated key resolves
        // last-wins on recovery. Record the padded offset of each member
        // relative to the blob start so we can index them after the append.
        let n = u32::try_from(pairs.len()).map_err(|_| Error::InvalidRecord {
            msg: "batch too large",
        })?;
        let header = encode_record(
            b"",
            &n.to_le_bytes(),
            RecordKind::BatchBegin,
            self.next_ts(),
        );
        let mut blob = header;
        let mut member_rel: Vec<(u32, u32)> = Vec::with_capacity(pairs.len()); // (rel_offset, padded)
        for (key, value) in pairs {
            let rel = u32::try_from(blob.len()).map_err(|_| Error::InvalidRecord {
                msg: "batch too large",
            })?;
            let rec = encode_record(key, value, RecordKind::Scalar, self.next_ts());
            let padded = u32::try_from(rec.len()).expect("padded record fits u32");
            member_rel.push((rel, padded));
            blob.extend_from_slice(&rec);
        }

        let blob_len = blob.len() as u64;
        if blob_len > self.inner.max_seg_size {
            return Err(Error::InvalidRecord {
                msg: "batch exceeds one segment",
            });
        }

        // One rotation decision for the whole blob, then one append: the members
        // cannot be split across segments or interleaved with another writer's
        // records, which is what makes the header/member run contiguous for
        // recovery.
        self.maybe_rotate(blob_len).await?;
        let (committer, seg_id) = {
            let a = self.inner.active.borrow();
            (a.committer.clone(), a.id)
        };
        let (start, padded_total) = committer.append(blob, durability).await?;
        self.bump_active_size(seg_id, start + u64::from(padded_total));

        // Now durable: apply each member to the index + accounting. Same steps
        // as `set_scoped`, minus the per-key durability wait (already paid once).
        let mut index = self.inner.index.borrow_mut();
        let mut cache = self.inner.cache.borrow_mut();
        let mut disk = self.inner.tenant_disk.lock();
        for ((key, value), (rel, padded)) in pairs.iter().zip(&member_rel) {
            let offset = start + u64::from(*rel);
            if let Some(prev) = index.get(key).copied() {
                self.dec_live(prev.segment_id, prev.size);
                let dtenant = tenant_from_key(key);
                if let Some(e) = disk.get_mut(&dtenant) {
                    *e = e.saturating_sub(u64::from(prev.size));
                }
            }
            self.inc_live(seg_id, *padded);
            index.set(
                key.to_vec(),
                IndexEntry {
                    fingerprint: fingerprint(key),
                    segment_id: seg_id,
                    _pad: 0,
                    offset: u32::try_from(offset).expect("offset fits u32"),
                    size: *padded,
                },
            );
            cache.insert_for(
                key,
                Bytes::copy_from_slice(value),
                value.len(),
                tenant_from_key(key),
            );
            *disk.entry(tenant_from_key(key)).or_insert(0) += u64::from(*padded);
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

    /// Monotonic write sequence for this store: it advances by one on every
    /// durable SET and DEL (an N-pair MSET advances it by N) and never moves on
    /// a read. Restored across restart from the snapshot, so it does not reset.
    /// O(1), no lock.
    ///
    /// A cheap store version stamp: read it once, and an unchanged value means
    /// nothing was written since. Useful for change-detected STATS refresh,
    /// snapshot/backup consistency checks, and invalidating derived state.
    #[must_use]
    pub fn write_seq(&self) -> u64 {
        self.inner.clock.get()
    }

    /// Every live key, in unspecified order. One pass over the in-memory index;
    /// dead (deleted/tombstoned) keys are already absent from it, so they never
    /// appear. The returned `Vec` owns its bytes and outlives the internal
    /// borrow.
    ///
    /// For GC / compaction / backup / export / key-space audits that must visit
    /// every key once. Not a resumable cursor: the order is not stable across
    /// mutations (the index is a hash map). Materialises the whole key set; at
    /// large key counts prefer [`for_each_key`](Self::for_each_key), which
    /// streams without allocating.
    #[must_use]
    pub fn keys(&self) -> Vec<Vec<u8>> {
        let mut out = Vec::with_capacity(self.inner.index.borrow().len());
        self.for_each_key(|k| out.push(k.to_vec()));
        out
    }

    /// Visit every live key once, in unspecified order, without allocating: `f`
    /// is called with a borrow of each key. The zero-copy form of
    /// [`keys`](Self::keys) for GC / erasure / backup sweeps that do not need to
    /// keep the keys around.
    ///
    /// The index is borrowed for the whole walk, so `f` must not call back into
    /// this store in a way that mutates it (`set` / `del` would panic on the
    /// borrow). Collect what you need and act after the walk returns.
    pub fn for_each_key(&self, mut f: impl FnMut(&[u8])) {
        for (k, _) in self.inner.index.borrow().iter() {
            f(k);
        }
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

    /// Physically reclaim every dead byte in the store: seal the active
    /// segment, then compact every sealed segment that holds any dead record.
    /// Returns the number of dead bytes reclaimed.
    ///
    /// `del`/overwrite only tombstone: the old value stays in its segment until
    /// compaction rewrites the segment without it. The background loop only
    /// compacts a segment once it is past `COMPACTION_LIVE_RATIO` dead, so a
    /// freshly deleted value can linger indefinitely. This forces the issue for
    /// erasure-with-guarantee (GDPR): after it returns, no tombstoned value
    /// remains on disk.
    ///
    /// Store-wide, not scoped: a segment interleaves every tenant's records in
    /// write order, so "reclaim only tenant T" is not expressible at this
    /// layer. Reclaiming other tenants' already-dead bytes too is harmless.
    /// O(bytes in segments that have any dead record) - heavy; call it after a
    /// batch of deletes, not per key.
    ///
    /// # Errors
    ///
    /// Returns an error on IO failure.
    pub async fn reclaim_all_dead(&self) -> Result<u64> {
        // Seal the active segment so its own dead bytes become compactable
        // (the active segment is never a compaction candidate). A huge
        // `incoming` forces the rotation unless the active is empty, in which
        // case it holds nothing to reclaim anyway.
        self.maybe_rotate(self.inner.max_seg_size).await?;

        // Snapshot the targets up front - sealed segments with any dead byte -
        // so freshly written relocation segments are never re-picked and the
        // loop cannot diverge.
        let targets: Vec<(u16, u64)> = {
            let active_id = self.inner.active.borrow().id;
            let segs = self.inner.read_segments.borrow();
            segs.iter()
                .filter(|s| s.id != active_id)
                .filter_map(|s| {
                    let total = s.file.size().unwrap_or(0);
                    let dead = total.saturating_sub(s.live.get());
                    (total > 0 && dead > 0).then_some((s.id, dead))
                })
                .collect()
        };

        let mut freed = 0u64;
        for (id, dead) in targets {
            self.compact_segment(id).await?;
            freed += dead;
        }
        Ok(freed)
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

        // Relocate concurrently. Each record awaits its own `append_raw`; doing
        // them one at a time puts a single record in every group-commit batch
        // and pays a flush per record (~3.6 ms each, ~90 s to relocate 25k
        // survivors). `buffer_unordered` keeps many in flight so they share
        // flushes. Safe: `append_raw` never holds a RefCell borrow across its
        // await (the shard already runs concurrent set/del on this same path),
        // and each record touches only its own key's index entry, so distinct
        // records never collide. Bound so a single compaction cannot swamp the
        // committer past what ordinary write traffic drives.
        let outcomes: Vec<RelocOutcome> = stream::iter(records)
            .map(|(offset, rec)| async move {
                if rec.kind == RecordKind::Tombstone {
                    // Carry the tombstone forward only while the key stays
                    // deleted. A concurrent re-SET (higher timestamp) wins on
                    // recovery, so a stale carried tombstone cannot resurrect.
                    if self.inner.index.borrow().get(&rec.key).is_none() {
                        let (nseg, _, _) = self
                            .append_raw(
                                &rec.key,
                                b"",
                                RecordKind::Tombstone,
                                rec.ts,
                                Durability::Relaxed,
                            )
                            .await?;
                        return Ok(Some((nseg, 0u32, false)));
                    }
                    return Ok(None);
                }
                // A data record is live iff the index still points at this
                // exact location.
                if !self.index_points_at(&rec.key, seg_id, offset) {
                    return Ok(None);
                }
                let (nseg, noff, npad) = self
                    .append_raw(&rec.key, &rec.value, rec.kind, rec.ts, Durability::Relaxed)
                    .await?;
                // Recheck-CAS: a concurrent SET may have moved the key during
                // the await above. Only adopt the relocated copy if nothing
                // changed; otherwise it is dead bytes in the active segment,
                // reclaimed when that segment is itself compacted.
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
                Ok(Some((nseg, npad, true)))
            })
            .buffer_unordered(RELOCATE_CONCURRENCY)
            .collect()
            .await;

        for outcome in outcomes {
            if let Some((nseg, npad, is_data)) = outcome? {
                note_dest(nseg, &mut dest_segments);
                moved_bytes += u64::from(npad);
                if is_data {
                    moved += 1;
                }
            }
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
        // Tolerate an already-removed file: `compact_segment` now has two
        // callers on the same store (the background maintenance loop and
        // `reclaim_all_dead`). They interleave at the `await`s above, so both
        // can pass the `find` guard on the same seg_id with a cloned file
        // handle, relocate its records (harmless - the recheck-CAS dedups), and
        // then both reach here. The second unlink would be `ENOENT`; that is the
        // work already done, not an error.
        match std::fs::remove_file(segment_path(&self.inner.dir, seg_id)) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
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

        // Serialise the rotation itself. Without this, two concurrent appends
        // both see the segment full and both create `old_id + 1` (EEXIST on the
        // loser). Re-check the fill under the lock: if a peer rotated while we
        // waited, the active is fresh and our record now fits, so there is
        // nothing to do.
        let _rotating = self.inner.rotate_lock.lock().await;
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

    /// APPEND to a key, charged and quota-checked against this view's tenant.
    /// Returns the new value length. See [`VLog::append`].
    ///
    /// # Errors
    ///
    /// Returns an error on IO failure or if the disk quota would be exceeded.
    pub async fn append(&self, key: &[u8], value: &[u8], durability: Durability) -> Result<u64> {
        self.vlog
            .append_scoped(key, value, durability, self.tenant, self.disk_limit)
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
    async fn write_seq_advances_on_writes_only_and_survives_reopen() {
        let dir = TempDir::new().unwrap();
        let after_del;
        {
            let v = VLog::open(dir.path()).await.unwrap();
            let s0 = v.write_seq();
            v.set(b"k1", b"v1", Durability::Kernel).await.unwrap();
            let s1 = v.write_seq();
            assert!(s1 > s0, "SET must advance write_seq");
            // A read leaves it untouched.
            v.get(b"k1").await.unwrap();
            assert_eq!(v.write_seq(), s1, "a read must not advance write_seq");
            v.del(b"k1", Durability::Kernel).await.unwrap();
            after_del = v.write_seq();
            assert!(after_del > s1, "DEL must advance write_seq");
            v.write_snapshot().await.unwrap();
            v.flush().await.unwrap();
        }
        // Reopen: the sequence resumes past the last write, never resets to 0.
        let v = VLog::open(dir.path()).await.unwrap();
        assert!(
            v.write_seq() >= after_del,
            "write_seq must not reset across reopen"
        );
    }

    #[tokio::test]
    async fn write_seq_counts_effective_writes_exactly() {
        let dir = TempDir::new().unwrap();
        let v = VLog::open(dir.path()).await.unwrap();
        let base = v.write_seq();
        // N distinct SETs advance by exactly N.
        for i in 0u32..5 {
            v.set(format!("k{i}").as_bytes(), b"v", Durability::Kernel)
                .await
                .unwrap();
        }
        assert_eq!(v.write_seq(), base + 5, "5 SETs advance by exactly 5");
        // Overwriting an existing key is still a write.
        v.set(b"k0", b"v2", Durability::Kernel).await.unwrap();
        assert_eq!(v.write_seq(), base + 6, "overwrite advances");
        // DEL of a missing key is a no-op: it must not advance.
        let before = v.write_seq();
        assert!(!v.del(b"absent", Durability::Kernel).await.unwrap());
        assert_eq!(
            v.write_seq(),
            before,
            "DEL of a missing key must not advance"
        );
        // DEL of a present key advances by exactly one.
        assert!(v.del(b"k0", Durability::Kernel).await.unwrap());
        assert_eq!(
            v.write_seq(),
            before + 1,
            "DEL of a present key advances by one"
        );
    }

    #[tokio::test]
    async fn keys_lists_every_live_key_and_no_dead_ones() {
        use std::collections::BTreeSet;
        let dir = TempDir::new().unwrap();
        let after_reopen;
        {
            let v = VLog::open(dir.path()).await.unwrap();
            assert!(v.keys().is_empty(), "empty store has no keys");
            v.set(b"a", b"1", Durability::Kernel).await.unwrap();
            v.set(b"b", b"2", Durability::Kernel).await.unwrap();
            v.set(b"c", b"3", Durability::Kernel).await.unwrap();
            // Overwrite must not duplicate the key.
            v.set(b"a", b"1b", Durability::Kernel).await.unwrap();
            // Delete must drop the key from the listing.
            v.del(b"b", Durability::Kernel).await.unwrap();

            let got: BTreeSet<Vec<u8>> = v.keys().into_iter().collect();
            let want: BTreeSet<Vec<u8>> = [b"a".to_vec(), b"c".to_vec()].into_iter().collect();
            assert_eq!(got, want, "keys = live set, order-agnostic");
            assert_eq!(v.keys().len(), v.len(), "keys count matches len()");
            assert_eq!(v.keys().len(), 2);

            v.write_snapshot().await.unwrap();
            v.flush().await.unwrap();
            after_reopen = got;
        }
        // Keys survive a reopen (recovered from the index).
        let v = VLog::open(dir.path()).await.unwrap();
        let got: std::collections::BTreeSet<Vec<u8>> = v.keys().into_iter().collect();
        assert_eq!(got, after_reopen, "keys recovered on reopen");
    }

    #[tokio::test]
    async fn for_each_key_streams_the_same_live_set_as_keys() {
        use std::collections::BTreeSet;
        let dir = TempDir::new().unwrap();
        let v = VLog::open(dir.path()).await.unwrap();
        // Empty store: f is never called.
        let mut calls = 0usize;
        v.for_each_key(|_| calls += 1);
        assert_eq!(calls, 0, "no keys, no calls");

        for i in 0u32..20 {
            v.set(format!("k{i}").as_bytes(), b"v", Durability::Kernel)
                .await
                .unwrap();
        }
        v.del(b"k0", Durability::Kernel).await.unwrap();

        // for_each_key visits exactly the live set, once each.
        let mut streamed: Vec<Vec<u8>> = Vec::new();
        v.for_each_key(|k| streamed.push(k.to_vec()));
        assert_eq!(streamed.len(), v.len(), "one call per live key");
        let streamed: BTreeSet<Vec<u8>> = streamed.into_iter().collect();
        let via_keys: BTreeSet<Vec<u8>> = v.keys().into_iter().collect();
        assert_eq!(
            streamed, via_keys,
            "for_each_key sees the same set as keys()"
        );
        assert!(
            !streamed.contains(b"k0".as_slice()),
            "deleted key not streamed"
        );
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

    /// Concatenate every segment `.seg` file's raw bytes - what a verbatim
    /// backup of the data dir would capture.
    fn raw_disk_bytes(dir: &std::path::Path) -> Vec<u8> {
        let mut all = Vec::new();
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|e| e == "seg") {
                all.extend(std::fs::read(&path).unwrap());
            }
        }
        all
    }

    #[tokio::test]
    async fn append_creates_accumulates_and_survives_reopen() {
        let dir = TempDir::new().unwrap();
        {
            let v = VLog::open(dir.path()).await.unwrap();
            // Absent key: append creates it, returns its length.
            assert_eq!(v.append(b"k", b"ab", Durability::Kernel).await.unwrap(), 2);
            assert_eq!(
                v.get(b"k").await.unwrap().as_deref(),
                Some(b"ab".as_slice())
            );
            // Subsequent appends accumulate; the return is the new length.
            assert_eq!(v.append(b"k", b"cd", Durability::Kernel).await.unwrap(), 4);
            assert_eq!(v.append(b"k", b"e", Durability::Kernel).await.unwrap(), 5);
            assert_eq!(
                v.get(b"k").await.unwrap().as_deref(),
                Some(b"abcde".as_slice())
            );
            v.flush().await.unwrap();
        }
        let v = VLog::open(dir.path()).await.unwrap();
        assert_eq!(
            v.get(b"k").await.unwrap().as_deref(),
            Some(b"abcde".as_slice())
        );
    }

    #[tokio::test]
    async fn concurrent_same_key_appends_lose_nothing() {
        // Without the append_lock, interleaved read-modify-write would drop
        // deltas. Fire many single-byte appends at one key concurrently and
        // assert every byte lands (order-agnostic).
        let dir = TempDir::new().unwrap();
        let v = std::rc::Rc::new(VLog::open(dir.path()).await.unwrap());
        let n = 200u32;
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let mut handles = Vec::new();
                for _ in 0..n {
                    let v = v.clone();
                    handles.push(tokio::task::spawn_local(async move {
                        v.append(b"log", b"x", Durability::Relaxed).await.unwrap();
                    }));
                }
                for h in handles {
                    h.await.unwrap();
                }
            })
            .await;
        let got = v.get(b"log").await.unwrap().unwrap();
        assert_eq!(got.len(), n as usize, "every concurrent append survived");
        assert!(got.iter().all(|&b| b == b'x'));
    }

    #[tokio::test]
    async fn set_many_writes_all_pairs_and_survives_reopen() {
        let dir = TempDir::new().unwrap();
        {
            let v = VLog::open(dir.path()).await.unwrap();
            v.set(b"pre", b"0", Durability::Kernel).await.unwrap();
            v.set_many(
                &[
                    (b"a".as_slice(), b"1".as_slice()),
                    (b"b", b"2"),
                    (b"c", b"3"),
                    // Duplicate key in the batch: last write wins.
                    (b"a", b"1b"),
                ],
                Durability::Kernel,
            )
            .await
            .unwrap();
            assert_eq!(v.len(), 4, "pre + a,b,c (a deduped)");
            assert_eq!(
                v.get(b"a").await.unwrap().as_deref(),
                Some(b"1b".as_slice())
            );
            assert_eq!(v.get(b"b").await.unwrap().as_deref(), Some(b"2".as_slice()));
            v.flush().await.unwrap();
        }
        // A complete batch is fully present after a reopen (recovery applies it).
        let v = VLog::open(dir.path()).await.unwrap();
        assert_eq!(v.len(), 4);
        assert_eq!(
            v.get(b"pre").await.unwrap().as_deref(),
            Some(b"0".as_slice())
        );
        assert_eq!(
            v.get(b"a").await.unwrap().as_deref(),
            Some(b"1b".as_slice())
        );
        assert_eq!(v.get(b"b").await.unwrap().as_deref(), Some(b"2".as_slice()));
        assert_eq!(v.get(b"c").await.unwrap().as_deref(), Some(b"3".as_slice()));
    }

    #[tokio::test]
    async fn torn_batch_is_dropped_whole_on_recovery() {
        use crate::record::encode_record;
        use crate::segment::segment_path;

        let dir = TempDir::new().unwrap();
        {
            let v = VLog::open(dir.path()).await.unwrap();
            v.set(b"survivor", b"ok", Durability::Kernel).await.unwrap();
            v.flush().await.unwrap();
        }
        // Simulate a crash mid-`set_many`: a `BatchBegin(3)` header but only two
        // members reach disk before the tear. Append them raw to the segment.
        {
            let mut blob = encode_record(b"", &3u32.to_le_bytes(), RecordKind::BatchBegin, 100);
            blob.extend_from_slice(&encode_record(b"x", b"1", RecordKind::Scalar, 101));
            blob.extend_from_slice(&encode_record(b"y", b"2", RecordKind::Scalar, 102));
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(segment_path(dir.path(), 0))
                .unwrap();
            f.write_all(&blob).unwrap();
            f.sync_all().unwrap();
        }
        // Recovery must drop the incomplete batch WHOLE: neither member appears,
        // and the pre-batch survivor is untouched.
        let v = VLog::open(dir.path()).await.unwrap();
        assert_eq!(
            v.get(b"survivor").await.unwrap().as_deref(),
            Some(b"ok".as_slice())
        );
        assert_eq!(
            v.get(b"x").await.unwrap(),
            None,
            "torn batch member x must not survive"
        );
        assert_eq!(
            v.get(b"y").await.unwrap(),
            None,
            "torn batch member y must not survive"
        );
        assert_eq!(v.len(), 1, "only the survivor");
    }

    #[tokio::test]
    async fn reclaim_all_dead_spans_many_segments() {
        let dir = TempDir::new().unwrap();
        // Small segments so a few hundred keys spread over many sealed segments;
        // deleting half leaves dead bytes in most of them, so reclaim must
        // compact many segments in one pass (the multi-segment path).
        let v = VLog::open_with_max_segment(dir.path(), 512).await.unwrap();
        for i in 0u32..400 {
            v.set(
                format!("k{i:04}").as_bytes(),
                b"some-value-bytes",
                Durability::Kernel,
            )
            .await
            .unwrap();
        }
        assert!(v.segment_count() > 5, "expected many segments");
        // Delete every even key: dead bytes scattered across most segments.
        for i in (0u32..400).step_by(2) {
            v.del(format!("k{i:04}").as_bytes(), Durability::Kernel)
                .await
                .unwrap();
        }
        v.flush().await.unwrap();

        let freed = v.reclaim_all_dead().await.unwrap();
        assert!(freed > 0, "reclaimed bytes across segments");
        v.flush().await.unwrap();

        // Survivors (odd keys) intact, deleted (even) gone.
        for i in 0u32..400 {
            let got = v.get(format!("k{i:04}").as_bytes()).await.unwrap();
            if i % 2 == 0 {
                assert_eq!(got, None, "even key {i} should be gone");
            } else {
                assert_eq!(
                    got.as_deref(),
                    Some(b"some-value-bytes".as_slice()),
                    "odd key {i}"
                );
            }
        }
        // Reopen: the compacted layout is consistent on disk.
        drop(v);
        let v = VLog::open(dir.path()).await.unwrap();
        assert_eq!(
            v.get(b"k0001").await.unwrap().as_deref(),
            Some(b"some-value-bytes".as_slice())
        );
        assert_eq!(v.get(b"k0000").await.unwrap(), None);
    }

    #[tokio::test]
    async fn reclaim_races_the_background_compactor_without_erroring() {
        let dir = TempDir::new().unwrap();
        let v = VLog::open_with_max_segment(dir.path(), 512).await.unwrap();
        for i in 0u32..400 {
            v.set(
                format!("k{i:04}").as_bytes(),
                b"some-value-bytes",
                Durability::Kernel,
            )
            .await
            .unwrap();
        }
        // Delete 3 of every 4 so segments are well past the 50% dead the
        // background `maybe_compact` needs, and it targets the same segments
        // `reclaim_all_dead` does.
        for i in 0u32..400 {
            if i % 4 != 0 {
                v.del(format!("k{i:04}").as_bytes(), Durability::Kernel)
                    .await
                    .unwrap();
            }
        }
        v.flush().await.unwrap();

        // join! polls both on this one task, interleaving them at every await -
        // the exact contention the shard's background loop creates against a
        // reclaim. Neither must error on a segment the other already unlinked.
        let (reclaimed, compacted) = tokio::join!(v.reclaim_all_dead(), v.maybe_compact());
        reclaimed.expect("reclaim tolerates a concurrent compaction");
        compacted.expect("compaction tolerates a concurrent reclaim");
        v.flush().await.unwrap();

        for i in 0u32..400 {
            let got = v.get(format!("k{i:04}").as_bytes()).await.unwrap();
            if i % 4 == 0 {
                assert_eq!(
                    got.as_deref(),
                    Some(b"some-value-bytes".as_slice()),
                    "survivor {i}"
                );
            } else {
                assert_eq!(got, None, "deleted {i}");
            }
        }
    }

    #[tokio::test]
    async fn reclaim_all_dead_physically_removes_deleted_values() {
        let needle = b"SECRET-PII-Rossi-Mario";
        let dir = TempDir::new().unwrap();
        // Small segments so the deleted value lands in a sealed segment that
        // later rotations leave behind, i.e. the realistic case.
        let v = VLog::open_with_max_segment(dir.path(), 256).await.unwrap();
        v.set(b"subject", needle, Durability::Kernel).await.unwrap();
        // Write past a rotation so the secret's segment is sealed, not active.
        for i in 0u64..20 {
            v.set(format!("filler{i}").as_bytes(), b"x", Durability::Kernel)
                .await
                .unwrap();
        }
        v.del(b"subject", Durability::Kernel).await.unwrap();
        v.flush().await.unwrap();

        // Gone logically at once...
        assert_eq!(v.get(b"subject").await.unwrap(), None);
        // ...but the raw value bytes are still on disk (tombstone only).
        let before = raw_disk_bytes(dir.path());
        assert!(
            before.windows(needle.len()).any(|w| w == needle),
            "precondition: deleted value should still be on disk before reclaim"
        );

        let freed = v.reclaim_all_dead().await.unwrap();
        v.flush().await.unwrap();
        assert!(freed > 0, "reclaim reported bytes freed");

        // Now the bytes are physically gone, and survivors are intact.
        let after = raw_disk_bytes(dir.path());
        assert!(
            !after.windows(needle.len()).any(|w| w == needle),
            "deleted value must not survive reclaim on disk"
        );
        assert_eq!(v.get(b"subject").await.unwrap(), None, "still deleted");
        for i in 0u64..20 {
            assert_eq!(
                v.get(format!("filler{i}").as_bytes())
                    .await
                    .unwrap()
                    .as_deref(),
                Some(b"x".as_slice()),
                "reclaim kept live survivors"
            );
        }
        // Survives a reopen: reclaim rewrote the segments, it did not just hide.
        drop(v);
        let v = VLog::open(dir.path()).await.unwrap();
        assert!(
            !raw_disk_bytes(dir.path())
                .windows(needle.len())
                .any(|w| w == needle),
            "value stays gone after reopen"
        );
        assert_eq!(v.get(b"subject").await.unwrap(), None);
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
