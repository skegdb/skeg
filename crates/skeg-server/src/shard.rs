//! Thread-per-core sharding.
//!
//! Each shard owns a `VLog` and runs on a dedicated thread pinned to a
//! performance core. Shards are shared-nothing: the `VLog` is touched only by
//! its own worker thread, so no locking is needed on the storage fast path.
//!
//! Requests reach a shard over a `crossbeam-channel`; the worker replies on a
//! per-request `tokio::oneshot`. Keys route deterministically by
//! `xxh3_64(key) % n_shards`.

use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

/// How often each shard checks whether a segment needs compacting.
const COMPACTION_INTERVAL: Duration = Duration::from_secs(60);

/// How often each shard writes an index snapshot for fast recovery.
const SNAPSHOT_INTERVAL: Duration = Duration::from_secs(300);

use bytes::Bytes;
use skeg_core::{Durability, VLog};
use skeg_vector::{DiskVamanaIndex, FlatIndex, QuantKind};
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::sync::oneshot;
use tracing::error;
use xxhash_rust::xxh3::xxh3_64;

/// Per-shard vector indexes, keyed by VINDEX name. Each shard holds the
/// fragment of every index whose `vec_id` hashes to that shard; a VSEARCH
/// scatters across all shards and the results are merged.
///
/// Each entry is wrapped in its own `Arc<RwLock<…>>` so vector ops on
/// **different** vindexes can run in parallel (the outer map is locked
/// only for the duration of the lookup, then released). Ops on the
/// **same** vindex still serialize, which is required for correctness:
/// `VectorBackend::search` takes `&mut self` because the disk path
/// mutates the working-set cache and the streaming-insert buffer.
type VectorEntry = Arc<RwLock<VectorBackend>>;
type VindexSet = HashMap<String, VectorEntry>;

/// Query-time search list size for a disk Vamana index.
const VAMANA_L_SEARCH: usize = 100;

/// A disk Vamana index folds its delta into the graph once it reaches
/// `max(main / 20, MIN)` pending inserts: a flat floor at small sizes, then
/// 5%-of-main so the consolidation count stays roughly logarithmic in N
/// rather than linear (a fixed threshold would make bulk growth O(N^2)).
const DISK_CONSOLIDATE_MIN: usize = 4096;

/// A VINDEX is backed either by an in-RAM `FlatIndex` (M7) or by an on-disk
/// Vamana graph (`DiskVamanaIndex`, M8) - f32 vectors on disk, graph + int8
/// tier in RAM. The choice is made at `VINDEX CREATE`.
enum VectorBackend {
    Flat(FlatIndex),
    Disk(DiskVamanaIndex),
}

impl VectorBackend {
    fn dim(&self) -> usize {
        match self {
            VectorBackend::Flat(i) => i.dim(),
            VectorBackend::Disk(i) => i.dim(),
        }
    }

    fn len(&self) -> usize {
        match self {
            VectorBackend::Flat(i) => i.len(),
            VectorBackend::Disk(i) => i.len(),
        }
    }

    /// Approximate RAM footprint of this vindex in bytes. Used to
    /// refresh the `VindexSizeBytes` gauge from STATS. Cheap enough
    /// (no allocation, just arithmetic on len/dim) to be polled.
    ///
    /// Flat indexes carry the full f32 row buffer in RAM. Disk indexes
    /// only keep tier-1 codes resident (int8 today = 1 byte per
    /// coordinate); the graph and full f32 vectors live on disk and
    /// are paged in by the OS, so they are not counted here.
    fn approx_ram_bytes(&self) -> u64 {
        let n = self.len() as u64;
        let d = self.dim() as u64;
        match self {
            VectorBackend::Flat(_) => n * d * 4,
            VectorBackend::Disk(_) => n * d, // int8 tier
        }
    }

    /// Wire kind byte for this VINDEX (mirrors `QuantKind::wire()` on the
    /// disk side; flat indexes always carry full f32 in RAM).
    fn kind_byte(&self) -> u8 {
        match self {
            // Flat indexes store full f32 vectors in RAM.
            VectorBackend::Flat(_) => 0,
            // Disk indexes carry the tier-1 quantisation choice; expose it
            // via the wire byte that VINDEX CREATE uses.
            VectorBackend::Disk(_) => 1, // int8 is the only on-disk tier today
        }
    }

    /// Wire backend byte: 0 = flat, 1 = disk Vamana.
    fn backend_byte(&self) -> u8 {
        match self {
            VectorBackend::Flat(_) => 0,
            VectorBackend::Disk(_) => 1,
        }
    }

    /// Insert a vector; a disk backend consolidates once its delta is full.
    fn insert(&mut self, id: u64, vector: &[f32]) -> std::io::Result<()> {
        match self {
            VectorBackend::Flat(i) => {
                i.insert(id, vector);
                Ok(())
            }
            VectorBackend::Disk(i) => {
                i.insert(id, vector)?;
                let threshold = (i.main_len() / 20).max(DISK_CONSOLIDATE_MIN);
                if i.delta_len() >= threshold {
                    i.consolidate()?;
                }
                Ok(())
            }
        }
    }

    fn delete(&mut self, id: u64) -> std::io::Result<bool> {
        match self {
            VectorBackend::Flat(i) => Ok(i.delete(id)),
            VectorBackend::Disk(i) => i.delete(id),
        }
    }

    fn get(&self, id: u64) -> std::io::Result<Option<Vec<f32>>> {
        match self {
            VectorBackend::Flat(i) => Ok(i.get(id)),
            VectorBackend::Disk(i) => i.get(id),
        }
    }

    fn search(
        &mut self,
        query: &[f32],
        k: usize,
        l_search: u32,
    ) -> std::io::Result<Vec<(u64, f32)>> {
        match self {
            // Flat is brute-force: no search-list, l_search does not apply.
            VectorBackend::Flat(i) => Ok(i.search(query, k)),
            VectorBackend::Disk(i) => i.search_with_l(query, k, l_search as usize),
        }
    }
}

/// Bounded inbox capacity per shard. A full inbox makes `send` await, which
/// propagates backpressure up to the connection handler (it stops reading new
/// frames) instead of letting queues grow without bound (OOM-safety, not
/// latency - see OBSERVATIONS Q2/Q6).
const SHARD_INBOX_CAPACITY: usize = 4096;

/// Maximum requests a shard processes concurrently. The inbox bounds the
/// *queue*; this bounds the *in-flight* request tasks. When both are full,
/// `send` blocks and backpressure reaches the client.
const MAX_INFLIGHT_PER_SHARD: usize = 1024;

/// Route a key to a shard index.
#[must_use]
pub fn shard_for(key: &[u8], n_shards: usize) -> usize {
    debug_assert!(n_shards >= 1);
    #[allow(clippy::cast_possible_truncation)]
    let idx = (xxh3_64(key) % n_shards as u64) as usize;
    idx
}

/// Error returned by `ShardSet` operations.
#[derive(Debug, thiserror::Error)]
pub enum ShardError {
    #[error("shard unavailable")]
    Unavailable,
    #[error("storage error: {0}")]
    Storage(String),
}

// ── Channel protocol ──────────────────────────────────────────────────────────

enum ShardReq {
    /// `(key, tenant)`: tenant `0` is the unscoped default.
    Get(Bytes, u128),
    /// `(key, value, durability, tenant)`.
    Set(Bytes, Bytes, Durability, u128),
    Del(Bytes, Durability),
    /// `(original_index, key)` pairs for a multi-get fragment, plus the tenant.
    MgetBatch(Vec<(usize, Bytes)>, u128),
    /// Bytes of hot-key cache charged to a tenant on this shard.
    TenantCacheBytes(u128),
    Stats,
    VindexCreate {
        name: String,
        dim: usize,
        kind: QuantKind,
        disk: bool,
    },
    VindexDrop {
        name: String,
    },
    /// Enumerate VINDEXes known to this shard. Replicated across all
    /// shards so callers can ask any one shard.
    VindexList,
    Vset {
        name: String,
        id: u64,
        vector: Vec<f32>,
    },
    Vget {
        name: String,
        id: u64,
    },
    Vdel {
        name: String,
        id: u64,
    },
    Vsearch {
        name: String,
        query: Vec<f32>,
        k: usize,
        l_search: u32,
    },
}

enum ShardResp {
    Value(Option<Bytes>),
    Done,
    Existed(bool),
    /// Bytes of hot-key cache charged to a tenant on the answering shard.
    CacheBytes(usize),
    MgetBatch(Vec<(usize, Option<Bytes>)>),
    /// `(cache_bytes, cache_evictions, n_keys, cache_budget)`.
    Stats(u64, u64, u64, u64),
    /// `(name, dim, kind_wire_byte, backend_wire_byte, n_vectors)` per VINDEX.
    VindexList(Vec<(String, u32, u8, u8, u64)>),
    /// VGET result: the stored f32 vector, or `None` if absent.
    Vector(Option<Vec<f32>>),
    /// VSEARCH result for this shard's fragment: `(vec_id, cosine)` hits.
    Vsearch(Vec<(u64, f32)>),
    Err(String),
}

struct ShardMsg {
    req: ShardReq,
    reply: oneshot::Sender<ShardResp>,
}

// ── VINDEX registry (Q10: disk-backed indexes survive a restart) ──────────────
//
// Each shard records its disk-backed VINDEXes in `vindexes.registry`; on
// startup it reopens them from their `vindex-<name>/` directories. Flat
// indexes are in-RAM and ephemeral by design - they are not registered.

const VINDEX_REGISTRY: &str = "vindexes.registry";

/// Rewrite the registry file: `[u32 count]` then `[u16 nlen][name][u32 dim]`.
#[allow(clippy::cast_possible_truncation)] // index names are short, dims fit u32
fn write_registry(dir: &Path, entries: &[(&str, usize)]) -> std::io::Result<()> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for (name, dim) in entries {
        buf.extend_from_slice(&(name.len() as u16).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(&(*dim as u32).to_le_bytes());
    }
    let tmp = dir.join(format!("{VINDEX_REGISTRY}.tmp"));
    std::fs::write(&tmp, &buf)?;
    std::fs::rename(&tmp, dir.join(VINDEX_REGISTRY))
}

/// Read the registry: `(name, dim)` per disk-backed VINDEX. Missing or
/// truncated registry yields whatever parsed cleanly.
fn read_registry(dir: &Path) -> Vec<(String, usize)> {
    let Ok(bytes) = std::fs::read(dir.join(VINDEX_REGISTRY)) else {
        return Vec::new();
    };
    if bytes.len() < 4 {
        return Vec::new();
    }
    let count = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let mut out = Vec::with_capacity(count);
    let mut pos = 4;
    for _ in 0..count {
        if pos + 2 > bytes.len() {
            break;
        }
        let nlen = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]) as usize;
        pos += 2;
        if pos + nlen + 4 > bytes.len() {
            break;
        }
        let name = String::from_utf8_lossy(&bytes[pos..pos + nlen]).into_owned();
        pos += nlen;
        let dim = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        out.push((name, dim));
    }
    out
}

/// Rewrite the registry from the current disk-backed VINDEXes (best-effort).
fn persist_registry(dir: &Path, vindexes: &RwLock<VindexSet>) {
    let vs = vindexes.read();
    let entries: Vec<(String, usize)> = vs
        .iter()
        .filter_map(|(name, entry)| {
            let backend = entry.read();
            match &*backend {
                VectorBackend::Disk(i) => Some((name.clone(), i.dim())),
                VectorBackend::Flat(_) => None,
            }
        })
        .collect();
    let entries_ref: Vec<(&str, usize)> = entries.iter().map(|(n, d)| (n.as_str(), *d)).collect();
    if let Err(e) = write_registry(dir, &entries_ref) {
        error!("vindex registry write failed: {e}");
    }
}

/// Reopen the disk-backed VINDEXes recorded in the registry, with the given
/// tier-1 quantisation (`Int8` for the read-write path, configurable in serve
/// mode). `mmap_tier` swaps the TurboQuant codes for a memory-mapped view
/// (`--tier-mmap`); `mmap_graph` swaps the graph Node array for a mmap'd
/// view of `graph.vmn` (`--graph-mmap`). Other tiers are unaffected.
fn recover_vindexes(
    shard_id: usize,
    dir: &Path,
    tier: QuantKind,
    mmap_tier: bool,
    mmap_graph: bool,
) -> VindexSet {
    let mut set = VindexSet::new();
    for (name, _dim) in read_registry(dir) {
        let vdir = dir.join(format!("vindex-{name}"));
        match DiskVamanaIndex::open_with_tier_full(&vdir, tier, mmap_tier, mmap_graph) {
            Ok(idx) => {
                set.insert(name, Arc::new(RwLock::new(VectorBackend::Disk(idx))));
            }
            Err(e) => error!("shard {shard_id}: recovering vindex '{name}' failed: {e}"),
        }
    }
    set
}

// ── Worker ────────────────────────────────────────────────────────────────────

// The worker is a thread entry point: it must own `dir` and `rx` for the
// thread's `'static` lifetime, so by-value arguments are required here.
//
// With `read_only` set the shard rejects every mutation and skips background
// compaction and snapshots: the `--mode serve` path over an offline-built
// index (PLAN-POST-Q10 Step 1.5).
#[allow(clippy::needless_pass_by_value)]
fn run_shard(
    shard_id: usize,
    dir: PathBuf,
    mut rx: Receiver<ShardMsg>,
    read_only: bool,
    tier: QuantKind,
    workers: usize,
    mmap_tier: bool,
    mmap_graph: bool,
) {
    skeg_platform::pin_current_thread_to_performance_core();

    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            error!("shard {shard_id}: runtime build failed: {e}");
            return;
        }
    };

    rt.block_on(async move {
        let vlog = match VLog::open(&dir).await {
            Ok(v) => v,
            Err(e) => {
                error!("shard {shard_id}: VLog::open failed: {e}");
                while let Some(msg) = rx.recv().await {
                    let _ = msg
                        .reply
                        .send(ShardResp::Err("shard storage unavailable".to_owned()));
                }
                return;
            }
        };

        // Each request runs as its own task on the LocalSet, so concurrent
        // writes to this shard are batched into one group commit. `VLog` is
        // `!Send` (Rc-backed), so `spawn_local` is required.
        // Caps the number of request tasks running at once on this shard.
        let inflight = std::sync::Arc::new(tokio::sync::Semaphore::new(MAX_INFLIGHT_PER_SHARD));
        // Vector indexes: disk-backed ones are recovered from the registry
        // (Q10); flat ones are in-RAM and start empty. Shared across the
        // per-request tasks on this single-threaded LocalSet via Rc/RefCell.
        // `Arc<RwLock>` (instead of the previous `Rc<RefCell>`) so the
        // VindexSet is `Send + Sync`: an optional worker pool can dispatch
        // VSEARCH to a blocking thread (`--workers N`, default 0 = inline).
        // The read/write locks are uncontended in inline mode (~10ns acquire
        // on M1), so there is no measurable cost for the default path.
        let vindexes: Arc<RwLock<VindexSet>> = Arc::new(RwLock::new(recover_vindexes(
            shard_id, &dir, tier, mmap_tier, mmap_graph,
        )));
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                // Background compaction and snapshots only earn their keep when
                // the shard accepts writes; a serve-mode shard skips both.
                if !read_only {
                    // Background compaction: reclaim dead space on this shard.
                    // Telemetry tick each time a compaction run starts.
                    let cvlog = vlog.clone();
                    tokio::task::spawn_local(async move {
                        loop {
                            tokio::time::sleep(COMPACTION_INTERVAL).await;
                            match cvlog.maybe_compact().await {
                                Ok(Some(_seg_id)) => {
                                    skeg_telemetry::tick_counter(
                                        skeg_telemetry::Counter::CompactionRunsTotal,
                                    );
                                }
                                Ok(None) => { /* nothing to compact this tick */ }
                                Err(e) => {
                                    error!("shard {shard_id}: compaction failed: {e}");
                                }
                            }
                        }
                    });

                    // Background snapshot: keep restart recovery fast.
                    let svlog = vlog.clone();
                    tokio::task::spawn_local(async move {
                        loop {
                            tokio::time::sleep(SNAPSHOT_INTERVAL).await;
                            if let Err(e) = svlog.write_snapshot().await {
                                error!("shard {shard_id}: snapshot failed: {e}");
                            }
                        }
                    });
                }

                while let Some(msg) = rx.recv().await {
                    // Block here once MAX_INFLIGHT_PER_SHARD tasks are running;
                    // this stops draining the inbox and propagates backpressure.
                    let permit = inflight
                        .clone()
                        .acquire_owned()
                        .await
                        .expect("inflight semaphore is never closed");
                    let vlog = vlog.clone();
                    let vindexes = vindexes.clone();
                    let dir = dir.clone();
                    let shard_id_u16 = shard_id as u16;
                    tokio::task::spawn_local(async move {
                        // Telemetry: classify the op, time the work, record.
                        // `op_kind` is borrowed-only · no Send cost on the
                        // hot path (the enum is `Copy`).
                        let op_kind = telemetry_op(&msg.req);
                        let t0 = std::time::Instant::now();
                        let resp =
                            process(&vlog, &vindexes, &dir, msg.req, read_only, workers).await;
                        if let Some(op) = op_kind {
                            skeg_telemetry::record_op(op, shard_id_u16, t0.elapsed());
                        }
                        let _ = msg.reply.send(resp);
                        drop(permit); // release on completion
                    });
                }
                // Channel closed: flush the active committer for durability.
                let _ = vlog.flush().await;
            })
            .await;
    });
}

/// True for requests that change durable state. A serve-mode shard rejects
/// these so an offline-built index is served strictly read-only.
fn is_mutation(req: &ShardReq) -> bool {
    matches!(
        req,
        ShardReq::Set(..)
            | ShardReq::Del(..)
            | ShardReq::VindexCreate { .. }
            | ShardReq::VindexDrop { .. }
            | ShardReq::Vset { .. }
            | ShardReq::Vdel { .. }
    )
}

/// Map a `ShardReq` to the corresponding telemetry op classifier.
///
/// Returns `None` for ops that should not appear in the operation
/// counters (e.g. internal `Stats` ping). Kept as a free function so
/// the hot-path call site stays trivially inlineable.
#[inline]
fn telemetry_op(req: &ShardReq) -> Option<skeg_telemetry::Op> {
    use skeg_telemetry::Op;
    match req {
        ShardReq::Get(..) | ShardReq::MgetBatch(..) => Some(Op::Get),
        ShardReq::Set(..) => Some(Op::Set),
        ShardReq::Del(..) => Some(Op::Del),
        ShardReq::Vset { .. } => Some(Op::VSet),
        ShardReq::Vsearch { .. } => Some(Op::VSearch),
        ShardReq::Vdel { .. } => Some(Op::VDel),
        ShardReq::Vget { .. }
        | ShardReq::VindexCreate { .. }
        | ShardReq::VindexList
        | ShardReq::VindexDrop { .. }
        | ShardReq::TenantCacheBytes(_)
        | ShardReq::Stats => None,
    }
}

async fn process(
    vlog: &VLog,
    vindexes: &Arc<RwLock<VindexSet>>,
    dir: &Path,
    req: ShardReq,
    read_only: bool,
    workers: usize,
) -> ShardResp {
    if read_only && is_mutation(&req) {
        return ShardResp::Err("server is in serve mode (read-only)".to_owned());
    }
    // Optional worker-pool path for VSEARCH (Q11, Tier 2 of
    // `optimizations/PLAN.md`): when `workers > 0` the search runs on a
    // blocking thread so it does not stall queued KV ops on the shard
    // thread. KV ops always stay inline since they finish in microseconds.
    if workers > 0
        && let ShardReq::Vsearch {
            name,
            query,
            k,
            l_search,
        } = req
    {
        // Look up + clone the per-vindex Arc on the shard thread; the
        // blocking task only holds the inner lock. This lets two
        // concurrent VSEARCH calls on different vindexes run in
        // parallel on the blocking pool (the previous design serialised
        // them on the outer RwLock).
        let entry = vindexes.read().get(&name).cloned();
        let join = tokio::task::spawn_blocking(move || -> ShardResp {
            let Some(arc) = entry else {
                return ShardResp::Err(format!("vindex '{name}' not found"));
            };
            let mut idx = arc.write();
            if idx.dim() != query.len() {
                return ShardResp::Err(format!(
                    "vindex '{name}' dim {} but query has {}",
                    idx.dim(),
                    query.len()
                ));
            }
            match idx.search(&query, k, l_search) {
                Ok(hits) => ShardResp::Vsearch(hits),
                Err(e) => ShardResp::Err(format!("vsearch failed: {e}")),
            }
        });
        return match join.await {
            Ok(resp) => resp,
            Err(_) => ShardResp::Err("vsearch worker task failed".to_owned()),
        };
    }
    let vindexes: &RwLock<VindexSet> = vindexes;
    match req {
        // ── vector ops (synchronous; no await while the RefCell is borrowed) ──
        ShardReq::VindexCreate {
            name,
            dim,
            kind,
            disk,
        } => {
            use std::collections::hash_map::Entry;
            let result = match vindexes.write().entry(name) {
                Entry::Occupied(e) => Err(format!("vindex '{}' already exists", e.key())),
                Entry::Vacant(e) => {
                    if disk {
                        let vdir = dir.join(format!("vindex-{}", e.key()));
                        match DiskVamanaIndex::create_empty(&vdir, dim, VAMANA_L_SEARCH) {
                            Ok(idx) => {
                                e.insert(Arc::new(RwLock::new(VectorBackend::Disk(idx))));
                                Ok(true)
                            }
                            Err(err) => Err(format!("vindex disk create failed: {err}")),
                        }
                    } else {
                        e.insert(Arc::new(RwLock::new(VectorBackend::Flat(FlatIndex::new(
                            dim, kind,
                        )))));
                        Ok(false)
                    }
                }
            };
            match result {
                Ok(created_disk) => {
                    if created_disk {
                        persist_registry(dir, vindexes);
                    }
                    ShardResp::Done
                }
                Err(e) => ShardResp::Err(e),
            }
        }
        ShardReq::VindexList => {
            let vs = vindexes.read();
            let mut rows: Vec<(String, u32, u8, u8, u64)> = vs
                .iter()
                .map(|(name, entry)| {
                    let backend = entry.read();
                    (
                        name.clone(),
                        backend.dim() as u32,
                        backend.kind_byte(),
                        backend.backend_byte(),
                        backend.len() as u64,
                    )
                })
                .collect();
            // Stable order so the TUI doesn't flicker between polls.
            rows.sort_by(|a, b| a.0.cmp(&b.0));
            ShardResp::VindexList(rows)
        }
        ShardReq::VindexDrop { name } => {
            // Pop the entry from the outer map first; this prevents new ops
            // from observing it. In-flight ops on this vindex keep their
            // cloned `Arc` alive and finish their inner lock window before
            // dropping it. `remove_dir_all` on POSIX deletes the path
            // immediately even if open file handles persist on the still-
            // alive Arc clones; the handles close when the last clone drops.
            let removed = vindexes.write().remove(&name);
            match removed {
                Some(arc) => {
                    let was_disk = matches!(*arc.read(), VectorBackend::Disk(_));
                    drop(arc);
                    if was_disk {
                        let _ = std::fs::remove_dir_all(dir.join(format!("vindex-{name}")));
                        persist_registry(dir, vindexes);
                    }
                    ShardResp::Done
                }
                None => ShardResp::Err(format!("vindex '{name}' not found")),
            }
        }
        ShardReq::Vset { name, id, vector } => {
            // Outer read to look up the entry; clone the Arc and drop the
            // outer lock before taking the per-vindex write. This lets
            // another vindex's ops run in parallel with this one.
            let entry = vindexes.read().get(&name).cloned();
            match entry {
                None => ShardResp::Err(format!("vindex '{name}' not found")),
                Some(arc) => {
                    let mut idx = arc.write();
                    if idx.dim() != vector.len() {
                        ShardResp::Err(format!(
                            "vindex '{name}' dim {} but vector has {}",
                            idx.dim(),
                            vector.len()
                        ))
                    } else {
                        match idx.insert(id, &vector) {
                            Ok(()) => ShardResp::Done,
                            Err(e) => ShardResp::Err(format!("vset failed: {e}")),
                        }
                    }
                }
            }
        }
        ShardReq::Vget { name, id } => {
            let entry = vindexes.read().get(&name).cloned();
            match entry {
                None => ShardResp::Err(format!("vindex '{name}' not found")),
                Some(arc) => {
                    let idx = arc.read();
                    match idx.get(id) {
                        Ok(v) => ShardResp::Vector(v),
                        Err(e) => ShardResp::Err(format!("vget failed: {e}")),
                    }
                }
            }
        }
        ShardReq::Vdel { name, id } => {
            let entry = vindexes.read().get(&name).cloned();
            match entry {
                None => ShardResp::Err(format!("vindex '{name}' not found")),
                Some(arc) => {
                    let mut idx = arc.write();
                    match idx.delete(id) {
                        Ok(existed) => ShardResp::Existed(existed),
                        Err(e) => ShardResp::Err(format!("vdel failed: {e}")),
                    }
                }
            }
        }
        ShardReq::Vsearch {
            name,
            query,
            k,
            l_search,
        } => {
            let entry = vindexes.read().get(&name).cloned();
            match entry {
                None => ShardResp::Err(format!("vindex '{name}' not found")),
                Some(arc) => {
                    let mut idx = arc.write();
                    if idx.dim() != query.len() {
                        ShardResp::Err(format!(
                            "vindex '{name}' dim {} but query has {}",
                            idx.dim(),
                            query.len()
                        ))
                    } else {
                        match idx.search(&query, k, l_search) {
                            Ok(hits) => ShardResp::Vsearch(hits),
                            Err(e) => ShardResp::Err(format!("vsearch failed: {e}")),
                        }
                    }
                }
            }
        }

        ShardReq::Get(key, tenant) => match vlog.tenant(tenant).get(&key).await {
            Ok(v) => ShardResp::Value(v),
            Err(e) => ShardResp::Err(e.to_string()),
        },
        ShardReq::Set(key, val, dur, tenant) => {
            match vlog.tenant(tenant).set(&key, &val, dur).await {
                Ok(()) => ShardResp::Done,
                Err(e) => ShardResp::Err(e.to_string()),
            }
        }
        ShardReq::Del(key, dur) => match vlog.del(&key, dur).await {
            Ok(b) => ShardResp::Existed(b),
            Err(e) => ShardResp::Err(e.to_string()),
        },
        ShardReq::TenantCacheBytes(tenant) => {
            ShardResp::CacheBytes(vlog.tenant_cache_bytes(tenant))
        }
        ShardReq::MgetBatch(items, tenant) => {
            let view = vlog.tenant(tenant);
            let mut out = Vec::with_capacity(items.len());
            for (idx, key) in items {
                match view.get(&key).await {
                    Ok(v) => out.push((idx, v)),
                    Err(e) => return ShardResp::Err(e.to_string()),
                }
            }
            ShardResp::MgetBatch(out)
        }
        ShardReq::Stats => {
            let (bytes, evictions, n_keys, budget) = vlog.cache_stats();
            // Refresh telemetry gauges from live vlog + vindex state.
            // Gauges use `store` semantics, so polling them via STATS
            // does not double-count.
            use skeg_telemetry::Gauge;
            skeg_telemetry::set_gauge(Gauge::VlogLiveBytes, bytes);
            skeg_telemetry::set_gauge(Gauge::VlogSegmentsLive, vlog.segment_count() as u64);
            skeg_telemetry::set_gauge(Gauge::VlogTotalBytes, vlog.disk_bytes_total());
            let (n_vec, sz_bytes) = {
                let v = vindexes.read();
                v.values().fold((0u64, 0u64), |(acc_n, acc_b), entry| {
                    let b = entry.read();
                    (acc_n + b.len() as u64, acc_b + b.approx_ram_bytes())
                })
            };
            skeg_telemetry::set_gauge(Gauge::VindexVectors, n_vec);
            skeg_telemetry::set_gauge(Gauge::VindexSizeBytes, sz_bytes);
            ShardResp::Stats(bytes, evictions, n_keys, budget)
        }
    }
}

// ── ShardSet ──────────────────────────────────────────────────────────────────

struct ShardSetInner {
    senders: Vec<Sender<ShardMsg>>,
    handles: Vec<JoinHandle<()>>,
    n: usize,
}

impl Drop for ShardSetInner {
    fn drop(&mut self) {
        // Drop senders first: workers see the channel disconnect and exit.
        self.senders.clear();
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

/// Set of shards. Cheap to clone - clones share the same worker threads.
#[derive(Clone)]
pub struct ShardSet {
    inner: Arc<ShardSetInner>,
}

impl ShardSet {
    /// Open `n_shards` read-write shards, each storing into
    /// `base_dir/shard-{id}/`.
    ///
    /// # Errors
    ///
    /// Returns an IO error if a worker thread cannot be spawned.
    ///
    /// # Panics
    ///
    /// Panics if `n_shards` is zero.
    pub fn open(base_dir: &Path, n_shards: usize) -> std::io::Result<Self> {
        Self::open_mode(base_dir, n_shards, false, QuantKind::Int8)
    }

    /// Open `n_shards` shards. With `read_only`, every shard rejects mutations
    /// (KV and vector) and skips background compaction/snapshots: this serves
    /// an offline-built index at its clean resident footprint. `tier` is the
    /// tier-1 quantisation rebuilt for each disk VINDEX at open (`Int8` for the
    /// read-write path; serve mode may pick `Pq` for a smaller footprint).
    ///
    /// # Errors
    ///
    /// Returns an IO error if a worker thread cannot be spawned.
    ///
    /// # Panics
    ///
    /// Panics if `n_shards` is zero.
    pub fn open_mode(
        base_dir: &Path,
        n_shards: usize,
        read_only: bool,
        tier: QuantKind,
    ) -> std::io::Result<Self> {
        Self::open_mode_with_workers(base_dir, n_shards, read_only, tier, 0)
    }

    /// Like [`open_mode`](Self::open_mode), with an opt-in worker pool that
    /// dispatches `VSEARCH` requests off the shard thread via
    /// `tokio::task::spawn_blocking`. `workers == 0` (default) keeps the
    /// inline path used by the public bench. `workers > 0` enables the pool;
    /// the value is informational today (Tokio's blocking pool sizes itself
    /// at runtime), but is plumbed through so a future dedicated pool can
    /// honour it. KV ops always stay on the shard thread.
    ///
    /// Tradeoff: with the pool enabled, KV latency under mixed VSEARCH+KV
    /// load no longer queues behind multi-ms vector searches; in exchange
    /// the VindexSet is touched under a `RwLock` (uncontended ~10ns on M1).
    ///
    /// # Errors
    ///
    /// Returns an IO error if a worker thread cannot be spawned.
    ///
    /// # Panics
    ///
    /// Panics if `n_shards` is zero.
    pub fn open_mode_with_workers(
        base_dir: &Path,
        n_shards: usize,
        read_only: bool,
        tier: QuantKind,
        workers: usize,
    ) -> std::io::Result<Self> {
        Self::open_mode_full(base_dir, n_shards, read_only, tier, workers, false)
    }

    /// Like [`open_mode_with_workers`](Self::open_mode_with_workers), plus
    /// the opt-in `mmap_tier` and `mmap_graph` flags that swap, respectively,
    /// the TurboQuant codes buffer and the graph Node array for memory-mapped
    /// views at open time. Other tiers (`int8`, `pq`) are unaffected by
    /// `mmap_tier`; `mmap_graph` applies to any disk VINDEX regardless of
    /// tier (the graph file format is the same).
    ///
    /// # Errors
    ///
    /// Returns an IO error if a worker thread cannot be spawned.
    ///
    /// # Panics
    ///
    /// Panics if `n_shards` is zero.
    pub fn open_mode_full(
        base_dir: &Path,
        n_shards: usize,
        read_only: bool,
        tier: QuantKind,
        workers: usize,
        mmap_tier: bool,
    ) -> std::io::Result<Self> {
        Self::open_mode_full_mmap(
            base_dir, n_shards, read_only, tier, workers, mmap_tier, false,
        )
    }

    /// All-knobs constructor. Adds `mmap_graph` to
    /// [`open_mode_full`](Self::open_mode_full).
    ///
    /// # Errors
    ///
    /// Returns an IO error if a worker thread cannot be spawned.
    ///
    /// # Panics
    ///
    /// Panics if `n_shards` is zero.
    pub fn open_mode_full_mmap(
        base_dir: &Path,
        n_shards: usize,
        read_only: bool,
        tier: QuantKind,
        workers: usize,
        mmap_tier: bool,
        mmap_graph: bool,
    ) -> std::io::Result<Self> {
        assert!(n_shards >= 1, "n_shards must be >= 1");
        let mut senders = Vec::with_capacity(n_shards);
        let mut handles = Vec::with_capacity(n_shards);
        for id in 0..n_shards {
            let dir = base_dir.join(format!("shard-{id}"));
            let (tx, rx) = tokio::sync::mpsc::channel::<ShardMsg>(SHARD_INBOX_CAPACITY);
            let handle = std::thread::Builder::new()
                .name(format!("skeg-shard-{id}"))
                .spawn(move || {
                    run_shard(id, dir, rx, read_only, tier, workers, mmap_tier, mmap_graph)
                })?;
            senders.push(tx);
            handles.push(handle);
        }
        Ok(Self {
            inner: Arc::new(ShardSetInner {
                senders,
                handles,
                n: n_shards,
            }),
        })
    }

    /// Number of shards.
    #[must_use]
    pub fn n_shards(&self) -> usize {
        self.inner.n
    }

    async fn call(&self, shard: usize, req: ShardReq) -> Result<ShardResp, ShardError> {
        let (tx, rx) = oneshot::channel();
        // A bounded `send` awaits if the shard inbox is full: backpressure.
        self.inner.senders[shard]
            .send(ShardMsg { req, reply: tx })
            .await
            .map_err(|_| ShardError::Unavailable)?;
        rx.await.map_err(|_| ShardError::Unavailable)
    }

    /// GET a key.
    ///
    /// # Errors
    ///
    /// Returns an error if the shard is unavailable or storage fails.
    pub async fn get(&self, key: &[u8]) -> Result<Option<Bytes>, ShardError> {
        self.get_scoped(key, 0).await
    }

    async fn get_scoped(&self, key: &[u8], tenant: u128) -> Result<Option<Bytes>, ShardError> {
        let shard = shard_for(key, self.inner.n);
        match self
            .call(shard, ShardReq::Get(Bytes::copy_from_slice(key), tenant))
            .await?
        {
            ShardResp::Value(v) => Ok(v),
            ShardResp::Err(e) => Err(ShardError::Storage(e)),
            _ => Err(ShardError::Unavailable),
        }
    }

    /// SET a key-value pair at the given durability.
    ///
    /// # Errors
    ///
    /// Returns an error if the shard is unavailable or storage fails.
    pub async fn set(
        &self,
        key: &[u8],
        value: &[u8],
        durability: Durability,
    ) -> Result<(), ShardError> {
        self.set_scoped(key, value, durability, 0).await
    }

    async fn set_scoped(
        &self,
        key: &[u8],
        value: &[u8],
        durability: Durability,
        tenant: u128,
    ) -> Result<(), ShardError> {
        let shard = shard_for(key, self.inner.n);
        let req = ShardReq::Set(
            Bytes::copy_from_slice(key),
            Bytes::copy_from_slice(value),
            durability,
            tenant,
        );
        match self.call(shard, req).await? {
            ShardResp::Done => Ok(()),
            ShardResp::Err(e) => Err(ShardError::Storage(e)),
            _ => Err(ShardError::Unavailable),
        }
    }

    /// DEL a key at the given durability. Returns `true` if it existed.
    ///
    /// # Errors
    ///
    /// Returns an error if the shard is unavailable or storage fails.
    pub async fn del(&self, key: &[u8], durability: Durability) -> Result<bool, ShardError> {
        let shard = shard_for(key, self.inner.n);
        let req = ShardReq::Del(Bytes::copy_from_slice(key), durability);
        match self.call(shard, req).await? {
            ShardResp::Existed(b) => Ok(b),
            ShardResp::Err(e) => Err(ShardError::Storage(e)),
            _ => Err(ShardError::Unavailable),
        }
    }

    /// MGET multiple keys. Returns a `Vec` parallel to `keys`.
    ///
    /// Keys are bucketed by shard, dispatched in parallel, then reassembled.
    ///
    /// # Errors
    ///
    /// Returns an error if any shard is unavailable or storage fails.
    pub async fn mget(&self, keys: &[Bytes]) -> Result<Vec<Option<Bytes>>, ShardError> {
        self.mget_scoped(keys, 0).await
    }

    async fn mget_scoped(
        &self,
        keys: &[Bytes],
        tenant: u128,
    ) -> Result<Vec<Option<Bytes>>, ShardError> {
        let n = self.inner.n;
        let mut buckets: Vec<Vec<(usize, Bytes)>> = vec![Vec::new(); n];
        for (i, key) in keys.iter().enumerate() {
            buckets[shard_for(key, n)].push((i, key.clone()));
        }

        // Dispatch every non-empty bucket, then await all replies.
        let mut pending = Vec::new();
        for (shard, bucket) in buckets.into_iter().enumerate() {
            if bucket.is_empty() {
                continue;
            }
            let (tx, rx) = oneshot::channel();
            self.inner.senders[shard]
                .send(ShardMsg {
                    req: ShardReq::MgetBatch(bucket, tenant),
                    reply: tx,
                })
                .await
                .map_err(|_| ShardError::Unavailable)?;
            pending.push(rx);
        }

        let mut result: Vec<Option<Bytes>> = vec![None; keys.len()];
        for rx in pending {
            match rx.await.map_err(|_| ShardError::Unavailable)? {
                ShardResp::MgetBatch(items) => {
                    for (idx, val) in items {
                        result[idx] = val;
                    }
                }
                ShardResp::Err(e) => return Err(ShardError::Storage(e)),
                _ => return Err(ShardError::Unavailable),
            }
        }
        Ok(result)
    }

    /// Scope KV operations to `tenant` for per-tenant cache accounting.
    ///
    /// Mirrors [`skeg_core::VLog::tenant`] at the shard-set level: a zero-cost
    /// view that routes every `get/set/mget` with the tenant id, so the shard's
    /// `VLog` charges cache residency to `tenant` instead of the unscoped `0`.
    #[must_use]
    pub fn tenant(&self, tenant: u128) -> ShardTenantView<'_> {
        ShardTenantView {
            shards: self,
            tenant,
        }
    }

    /// Bytes of hot-key cache charged to `tenant`, summed across every shard.
    ///
    /// # Errors
    ///
    /// Returns an error if a shard is unavailable.
    pub async fn tenant_cache_bytes(&self, tenant: u128) -> Result<usize, ShardError> {
        let mut total = 0usize;
        for shard in 0..self.inner.n {
            match self.call(shard, ShardReq::TenantCacheBytes(tenant)).await? {
                ShardResp::CacheBytes(b) => total += b,
                ShardResp::Err(e) => return Err(ShardError::Storage(e)),
                _ => return Err(ShardError::Unavailable),
            }
        }
        Ok(total)
    }

    /// Aggregate cache statistics summed across every shard.
    ///
    /// # Errors
    ///
    /// Returns an error if a shard is unavailable.
    pub async fn stats(&self) -> Result<skeg_proto::ServerStats, ShardError> {
        let mut acc = skeg_proto::ServerStats::default();
        for shard in 0..self.inner.n {
            match self.call(shard, ShardReq::Stats).await? {
                ShardResp::Stats(bytes, evictions, n_keys, budget) => {
                    acc.cache_bytes += bytes;
                    acc.cache_evictions += evictions;
                    acc.n_keys += n_keys;
                    acc.cache_budget += budget;
                }
                ShardResp::Err(e) => return Err(ShardError::Storage(e)),
                _ => return Err(ShardError::Unavailable),
            }
        }
        Ok(acc)
    }

    /// Per-shard stats breakdown. The aggregate `stats()` is the sum of
    /// these rows. Used by `SKEG.SHARDS` / `Op::Shards` so observability
    /// tools (skeg-top TUI, ops dashboards) can render hot-shard skew.
    ///
    /// # Errors
    ///
    /// Returns `ShardError::Unavailable` if any shard mailbox is closed,
    /// or the first storage error encountered.
    pub async fn stats_per_shard(&self) -> Result<Vec<skeg_proto::ShardStats>, ShardError> {
        let mut rows = Vec::with_capacity(self.inner.n);
        for shard in 0..self.inner.n {
            match self.call(shard, ShardReq::Stats).await? {
                ShardResp::Stats(bytes, evictions, n_keys, budget) => {
                    rows.push(skeg_proto::ShardStats {
                        shard_id: shard as u32,
                        cache_bytes: bytes,
                        cache_evictions: evictions,
                        n_keys,
                        cache_budget: budget,
                    });
                }
                ShardResp::Err(e) => return Err(ShardError::Storage(e)),
                _ => return Err(ShardError::Unavailable),
            }
        }
        Ok(rows)
    }

    /// Send a request to every shard and require each to return `Done`.
    /// Used for VINDEX CREATE/DROP, which every shard must apply.
    async fn broadcast(&self, mut make_req: impl FnMut() -> ShardReq) -> Result<(), ShardError> {
        let mut pending = Vec::with_capacity(self.inner.n);
        for sender in &self.inner.senders {
            let (tx, rx) = oneshot::channel();
            sender
                .send(ShardMsg {
                    req: make_req(),
                    reply: tx,
                })
                .await
                .map_err(|_| ShardError::Unavailable)?;
            pending.push(rx);
        }
        let mut first_err = None;
        for rx in pending {
            match rx.await.map_err(|_| ShardError::Unavailable)? {
                ShardResp::Done => {}
                ShardResp::Err(e) => {
                    first_err.get_or_insert(e);
                }
                _ => return Err(ShardError::Unavailable),
            }
        }
        match first_err {
            Some(e) => Err(ShardError::Storage(e)),
            None => Ok(()),
        }
    }

    /// Create a vector index across all shards.
    ///
    /// `kind` is the raw wire byte: 0 = f32, 1 = int8, 2 = binary. `backend`
    /// is 0 = flat (in-RAM) or 1 = disk Vamana graph.
    ///
    /// # Errors
    ///
    /// Returns an error for a bad `dim`/`kind`/`backend`, a duplicate name, or
    /// an unavailable shard.
    pub async fn vindex_create(
        &self,
        name: &str,
        dim: u32,
        kind: u8,
        backend: u8,
    ) -> Result<(), ShardError> {
        if dim == 0 {
            return Err(ShardError::Storage(
                "vindex dim must be positive".to_owned(),
            ));
        }
        let kind = match kind {
            0 => QuantKind::F32,
            1 => QuantKind::Int8,
            2 => QuantKind::Binary,
            other => return Err(ShardError::Storage(format!("unknown vindex kind {other}"))),
        };
        let disk = match backend {
            0 => false,
            1 => true,
            other => {
                return Err(ShardError::Storage(format!(
                    "unknown vindex backend {other}"
                )));
            }
        };
        let dim = dim as usize;
        let name = name.to_owned();
        self.broadcast(|| ShardReq::VindexCreate {
            name: name.clone(),
            dim,
            kind,
            disk,
        })
        .await
    }

    /// Drop a vector index across all shards.
    ///
    /// # Errors
    ///
    /// Returns an error if the index does not exist or a shard is unavailable.
    pub async fn vindex_drop(&self, name: &str) -> Result<(), ShardError> {
        let name = name.to_owned();
        self.broadcast(|| ShardReq::VindexDrop { name: name.clone() })
            .await
    }

    /// List every VINDEX. `(name, dim, kind_byte, backend_byte, n_vectors)`
    /// per index, with `n_vectors` summed across shards (VINDEX is
    /// replicated per shard, but VSET routes by vec_id so each shard
    /// only stores its own fragment).
    ///
    /// # Errors
    ///
    /// Returns an error if every shard is unavailable.
    pub async fn vindex_list(&self) -> Result<Vec<(String, u32, u8, u8, u64)>, ShardError> {
        use std::collections::BTreeMap;
        let mut agg: BTreeMap<String, (u32, u8, u8, u64)> = BTreeMap::new();
        for shard in 0..self.inner.n {
            match self.call(shard, ShardReq::VindexList).await? {
                ShardResp::VindexList(rows) => {
                    for (name, dim, kind, backend, n_vectors) in rows {
                        let entry = agg.entry(name).or_insert((dim, kind, backend, 0));
                        entry.3 = entry.3.saturating_add(n_vectors);
                    }
                }
                ShardResp::Err(e) => return Err(ShardError::Storage(e)),
                _ => return Err(ShardError::Unavailable),
            }
        }
        Ok(agg
            .into_iter()
            .map(|(name, (dim, kind, backend, n))| (name, dim, kind, backend, n))
            .collect())
    }

    /// Insert a vector under `id` into `name`. Routes by `id`.
    ///
    /// # Errors
    ///
    /// Returns an error if the index is missing, the dim mismatches, or the
    /// shard is unavailable.
    pub async fn vset(&self, name: &str, id: u64, vector: Vec<f32>) -> Result<(), ShardError> {
        let shard = shard_for(&id.to_le_bytes(), self.inner.n);
        let req = ShardReq::Vset {
            name: name.to_owned(),
            id,
            vector,
        };
        match self.call(shard, req).await? {
            ShardResp::Done => Ok(()),
            ShardResp::Err(e) => Err(ShardError::Storage(e)),
            _ => Err(ShardError::Unavailable),
        }
    }

    /// Fetch the stored f32 vector for `id` in `name`. Routes by `id`.
    ///
    /// # Errors
    ///
    /// Returns an error if the index is missing or the shard is unavailable.
    pub async fn vget(&self, name: &str, id: u64) -> Result<Option<Vec<f32>>, ShardError> {
        let shard = shard_for(&id.to_le_bytes(), self.inner.n);
        let req = ShardReq::Vget {
            name: name.to_owned(),
            id,
        };
        match self.call(shard, req).await? {
            ShardResp::Vector(v) => Ok(v),
            ShardResp::Err(e) => Err(ShardError::Storage(e)),
            _ => Err(ShardError::Unavailable),
        }
    }

    /// Tombstone the vector for `id` in `name`. Routes by `id`.
    ///
    /// # Errors
    ///
    /// Returns an error if the index is missing or the shard is unavailable.
    pub async fn vdel(&self, name: &str, id: u64) -> Result<bool, ShardError> {
        let shard = shard_for(&id.to_le_bytes(), self.inner.n);
        let req = ShardReq::Vdel {
            name: name.to_owned(),
            id,
        };
        match self.call(shard, req).await? {
            ShardResp::Existed(b) => Ok(b),
            ShardResp::Err(e) => Err(ShardError::Storage(e)),
            _ => Err(ShardError::Unavailable),
        }
    }

    /// Search `name` for the `k` nearest vectors to `query`.
    ///
    /// Scatters to every shard, then merges each fragment's local top-k into a
    /// global top-k ranked by cosine.
    ///
    /// # Errors
    ///
    /// Returns an error if the index is missing on every shard, the dim
    /// mismatches, or a shard is unavailable.
    pub async fn vsearch(
        &self,
        name: &str,
        query: Vec<f32>,
        k: usize,
        l_search: u32,
    ) -> Result<Vec<(u64, f32)>, ShardError> {
        let mut pending = Vec::with_capacity(self.inner.n);
        for sender in &self.inner.senders {
            let (tx, rx) = oneshot::channel();
            let req = ShardReq::Vsearch {
                name: name.to_owned(),
                query: query.clone(),
                k,
                l_search,
            };
            sender
                .send(ShardMsg { req, reply: tx })
                .await
                .map_err(|_| ShardError::Unavailable)?;
            pending.push(rx);
        }
        let mut merged: Vec<(u64, f32)> = Vec::new();
        let mut first_err = None;
        for rx in pending {
            match rx.await.map_err(|_| ShardError::Unavailable)? {
                ShardResp::Vsearch(hits) => merged.extend(hits),
                ShardResp::Err(e) => {
                    first_err.get_or_insert(e);
                }
                _ => return Err(ShardError::Unavailable),
            }
        }
        // Every shard erroring with no hits means the index is missing or the
        // query dim is wrong - surface that rather than an empty result.
        if merged.is_empty()
            && let Some(e) = first_err
        {
            return Err(ShardError::Storage(e));
        }
        merged.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        merged.truncate(k);
        Ok(merged)
    }
}

/// A [`ShardSet`] scoped to one tenant for per-tenant cache accounting.
///
/// Created by [`ShardSet::tenant`]. Zero-cost: a borrow of the `ShardSet` plus
/// the tenant id. KV operations route with the tenant so each shard's `VLog`
/// charges cache residency correctly. Mirrors `skeg_core::TenantView`.
#[derive(Clone, Copy)]
pub struct ShardTenantView<'a> {
    shards: &'a ShardSet,
    tenant: u128,
}

impl ShardTenantView<'_> {
    /// GET a key, charging any read-path cache insert to this tenant.
    ///
    /// # Errors
    ///
    /// Returns an error if the shard is unavailable or storage fails.
    pub async fn get(&self, key: &[u8]) -> Result<Option<Bytes>, ShardError> {
        self.shards.get_scoped(key, self.tenant).await
    }

    /// SET a key-value pair, charging the write-through entry to this tenant.
    ///
    /// # Errors
    ///
    /// Returns an error if the shard is unavailable or storage fails.
    pub async fn set(
        &self,
        key: &[u8],
        value: &[u8],
        durability: Durability,
    ) -> Result<(), ShardError> {
        self.shards
            .set_scoped(key, value, durability, self.tenant)
            .await
    }

    /// MGET multiple keys, charging read-path cache inserts to this tenant.
    ///
    /// # Errors
    ///
    /// Returns an error if any shard is unavailable or storage fails.
    pub async fn mget(&self, keys: &[Bytes]) -> Result<Vec<Option<Bytes>>, ShardError> {
        self.shards.mget_scoped(keys, self.tenant).await
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_shard_routing_deterministic() {
        for n in [1usize, 2, 4, 7, 16] {
            for key in [b"alpha".as_slice(), b"beta", b"", b"\x00\xFF\x01"] {
                let a = shard_for(key, n);
                let b = shard_for(key, n);
                assert_eq!(a, b, "same key must route to same shard");
                assert!(a < n, "shard index in range");
            }
        }
    }

    #[test]
    fn test_shard_routing_distribution() {
        let n = 4usize;
        let mut counts = vec![0usize; n];
        let total = 1_000_000usize;
        for i in 0..total {
            let key = format!("key_{i}");
            counts[shard_for(key.as_bytes(), n)] += 1;
        }
        let expected = total / n;
        for (s, &c) in counts.iter().enumerate() {
            let lo = expected * 9 / 10;
            let hi = expected * 11 / 10;
            assert!(
                c >= lo && c <= hi,
                "shard {s} got {c}, expected ~{expected} (±10%)"
            );
        }
    }

    #[tokio::test]
    async fn test_cross_shard_set_get() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 4).unwrap();

        // Keys deliberately spread across shards.
        for i in 0u32..50 {
            let key = format!("ck{i}");
            shards
                .set(
                    key.as_bytes(),
                    format!("v{i}").as_bytes(),
                    Durability::Kernel,
                )
                .await
                .unwrap();
        }
        for i in 0u32..50 {
            let key = format!("ck{i}");
            let got = shards.get(key.as_bytes()).await.unwrap();
            assert_eq!(got.as_deref(), Some(format!("v{i}").as_bytes()));
        }
    }

    #[tokio::test]
    async fn test_shard_isolation() {
        let dir = TempDir::new().unwrap();
        let base = dir.path().to_owned();
        {
            let shards = ShardSet::open(&base, 2).unwrap();
            for i in 0u32..40 {
                let key = format!("iso{i}");
                shards
                    .set(key.as_bytes(), b"x", Durability::Kernel)
                    .await
                    .unwrap();
            }
            // Dropping ShardSet joins all worker threads (files closed).
        }

        // Re-open each shard's VLog directly: a key must live only in the
        // shard that `shard_for` selects, never the other.
        let s0 = VLog::open(&base.join("shard-0")).await.unwrap();
        let s1 = VLog::open(&base.join("shard-1")).await.unwrap();
        for i in 0u32..40 {
            let key = format!("iso{i}");
            let in0 = s0.get(key.as_bytes()).await.unwrap().is_some();
            let in1 = s1.get(key.as_bytes()).await.unwrap().is_some();
            let expect = shard_for(key.as_bytes(), 2);
            assert_eq!(in0, expect == 0, "key {key} shard-0 membership");
            assert_eq!(in1, expect == 1, "key {key} shard-1 membership");
            assert!(in0 ^ in1, "key {key} must live in exactly one shard");
        }
    }

    #[tokio::test]
    async fn test_shardset_tenant_isolated_accounting() {
        // G-S2b-1: two tenants writing through the ShardSet have separate cache
        // accounting; neither charges the anonymous tenant 0.
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 4).unwrap();
        shards
            .tenant(7)
            .set(b"ak", b"v", Durability::Kernel)
            .await
            .unwrap();
        shards
            .tenant(9)
            .set(b"bk", b"vv", Durability::Kernel)
            .await
            .unwrap();
        assert!(
            shards.tenant_cache_bytes(7).await.unwrap() > 0,
            "tenant 7 charged"
        );
        assert!(
            shards.tenant_cache_bytes(9).await.unwrap() > 0,
            "tenant 9 charged"
        );
        assert_eq!(
            shards.tenant_cache_bytes(0).await.unwrap(),
            0,
            "anon tenant uncharged"
        );
        assert_eq!(
            shards.tenant(7).get(b"ak").await.unwrap().as_deref(),
            Some(b"v".as_slice())
        );
    }

    #[tokio::test]
    async fn test_shardset_bare_set_is_tenant_zero() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 2).unwrap();
        shards.set(b"k", b"v", Durability::Kernel).await.unwrap();
        assert!(
            shards.tenant_cache_bytes(0).await.unwrap() > 0,
            "bare set charges tenant 0"
        );
        assert_eq!(shards.tenant_cache_bytes(7).await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_mget_cross_shard() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 4).unwrap();

        shards.set(b"a", b"va", Durability::Kernel).await.unwrap();
        shards.set(b"b", b"vb", Durability::Kernel).await.unwrap();
        shards.set(b"c", b"vc", Durability::Kernel).await.unwrap();

        let keys = [
            Bytes::from_static(b"a"),
            Bytes::from_static(b"missing"),
            Bytes::from_static(b"c"),
            Bytes::from_static(b"b"),
        ];
        let res = shards.mget(&keys).await.unwrap();
        assert_eq!(res[0].as_deref(), Some(b"va".as_slice()));
        assert!(res[1].is_none());
        assert_eq!(res[2].as_deref(), Some(b"vc".as_slice()));
        assert_eq!(res[3].as_deref(), Some(b"vb".as_slice()));
    }

    #[tokio::test]
    async fn test_n_clients_concurrent() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 4).unwrap();

        let mut handles = Vec::new();
        for i in 0u32..100 {
            let shards = shards.clone();
            handles.push(tokio::spawn(async move {
                let key = format!("cc{i}");
                let val = format!("vv{i}");
                shards
                    .set(key.as_bytes(), val.as_bytes(), Durability::Kernel)
                    .await
                    .unwrap();
                let got = shards.get(key.as_bytes()).await.unwrap();
                assert_eq!(got.as_deref(), Some(val.as_bytes()));
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
    }

    #[tokio::test]
    async fn test_get_latency_flat_across_payload_size() {
        // Regression guard for the zero-copy invariant: `VLog::get` returns
        // `Bytes` (a refcount bump, not a memcpy) and `ShardSet` moves it
        // through a oneshot, so a large value must not cost meaningfully more
        // than a small one. Measured baseline: ~4.4-4.9 us flat for 4 KiB,
        // 64 KiB, 1 MiB. If this ratio blows up, a memcpy crept onto the read
        // path.
        use std::time::Instant;

        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 1).unwrap();
        shards
            .set(b"small", &vec![1u8; 4096], Durability::Kernel)
            .await
            .unwrap();
        shards
            .set(b"large", &vec![1u8; 1024 * 1024], Durability::Kernel)
            .await
            .unwrap();

        for _ in 0..64 {
            let _ = shards.get(b"small").await.unwrap();
            let _ = shards.get(b"large").await.unwrap();
        }

        let n = 2000;
        let t0 = Instant::now();
        for _ in 0..n {
            let _ = shards.get(b"small").await.unwrap();
        }
        let small = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..n {
            let _ = shards.get(b"large").await.unwrap();
        }
        let large = t1.elapsed();

        // `large` carries 256x the bytes of `small`. A zero-copy path keeps the
        // time within a small band; a linear (memcpy) regression would make it
        // ~256x. The 8x ceiling tolerates measurement noise while still
        // catching a real copy.
        assert!(
            large < small * 8,
            "GET latency scales with payload size: small={small:?} large={large:?} \
             - the zero-copy read path regressed (a memcpy was introduced)"
        );
    }

    #[tokio::test]
    async fn test_del_routes_correctly() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 4).unwrap();

        shards.set(b"dk", b"v", Durability::Kernel).await.unwrap();
        assert!(shards.del(b"dk", Durability::Kernel).await.unwrap());
        assert!(!shards.del(b"dk", Durability::Kernel).await.unwrap());
        assert!(shards.get(b"dk").await.unwrap().is_none());
    }

    /// Deterministic 64-dim test vector.
    #[allow(clippy::cast_precision_loss)]
    fn tvec(seed: u64) -> Vec<f32> {
        let mut s = (seed << 1) | 1;
        (0..64)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s & 0xFFFF) as f32 / 32768.0) - 1.0
            })
            .collect()
    }

    // Q10: a disk-backed VINDEX must survive a server restart - its files plus
    // the registry plus the WAL recover the full live set.
    #[tokio::test]
    async fn test_vindex_disk_survives_restart() {
        let dir = TempDir::new().unwrap();
        let base = dir.path().to_owned();
        {
            let shards = ShardSet::open(&base, 4).unwrap();
            // backend=1 -> disk Vamana.
            shards.vindex_create("persist", 64, 0, 1).await.unwrap();
            for id in 0u64..150 {
                shards.vset("persist", id, tvec(id + 1)).await.unwrap();
            }
            // `shards` dropped here: worker threads flush and exit.
        }

        // Restart: a fresh ShardSet on the same dir recovers the disk VINDEX
        // from the registry + WAL.
        let shards = ShardSet::open(&base, 4).unwrap();
        let hits = shards.vsearch("persist", tvec(89), 5, 0).await.unwrap();
        assert_eq!(hits[0].0, 88, "disk VINDEX must be recovered after restart");
        // A flat VINDEX, by contrast, is in-RAM and would not survive - so the
        // recovered set contains exactly the disk-backed index.
        assert!(shards.vget("persist", 42).await.unwrap().is_some());
    }

    /// Two VSEARCH callers hitting **different** vindexes on the same
    /// shard must not serialize against each other.
    ///
    /// We measure two regimes back-to-back on one shard with a 2-worker
    /// blocking pool:
    /// - **baseline**  : both tasks search the same vindex (serialized
    ///   by the per-vindex write lock, intentionally).
    /// - **concurrent**: each task searches its own vindex (per-vindex
    ///   write locks are disjoint, so both can hold their lock at the
    ///   same time on the blocking pool).
    ///
    /// SoL gate: `baseline / concurrent >= 1.2×`. The theoretical
    /// ceiling is 2.0× (perfect parallelism on two cores); a floor of
    /// 1.2× is enough to distinguish "the lock refactor parallelised
    /// the work" (always above the floor in practice) from "the
    /// searches still serialise" (a 1.0× or sub-1.0× ratio, which
    /// would have been the result on the old single-`RwLock`
    /// `VindexSet`). The gap below 2.0 absorbs the shared Tokio
    /// blocking pool, allocator noise from interleaved tests, and CI
    /// runners with fewer real cores than the developer M1.
    ///
    /// Measured locally on M1: 1.5×–2.0× depending on warm-up and
    /// concurrent system load; never below 1.4×.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_per_vindex_locks_concurrency_gate() {
        let dir = TempDir::new().unwrap();
        let shards =
            ShardSet::open_mode_with_workers(dir.path(), 1, false, skeg_vector::QuantKind::Int8, 2)
                .unwrap();

        // Two flat (in-RAM, no disk I/O contention) vindexes, 256-dim.
        // Flat search is brute-force cosine over the row buffer, so the
        // wall time scales linearly with `n` and is large enough to
        // dominate the lock-acquire overhead by ~3 orders of magnitude.
        shards.vindex_create("a", 256, 0, 0).await.unwrap();
        shards.vindex_create("b", 256, 0, 0).await.unwrap();

        let dim = 256;
        let n: u64 = 2_000;
        let make_vec = |seed: u64| -> Vec<f32> {
            (0..dim)
                .map(|d| (((seed.wrapping_mul(2654435761)) ^ d as u64) as f32) * 1e-9)
                .collect()
        };
        for id in 0..n {
            let v = make_vec(id);
            shards.vset("a", id, v.clone()).await.unwrap();
            shards.vset("b", id, v).await.unwrap();
        }

        let query = make_vec(99_999);
        let iters = 60u64;

        // Baseline: two tasks racing on the same vindex (write lock).
        let s1 = shards.clone();
        let q1 = query.clone();
        let s2 = shards.clone();
        let q2 = query.clone();
        let t = std::time::Instant::now();
        let h1 = tokio::spawn(async move {
            for _ in 0..iters {
                let _ = s1.vsearch("a", q1.clone(), 10, 0).await.unwrap();
            }
        });
        let h2 = tokio::spawn(async move {
            for _ in 0..iters {
                let _ = s2.vsearch("a", q2.clone(), 10, 0).await.unwrap();
            }
        });
        h1.await.unwrap();
        h2.await.unwrap();
        let baseline = t.elapsed();

        // Concurrent: one task per vindex.
        let s1 = shards.clone();
        let q1 = query.clone();
        let s2 = shards.clone();
        let q2 = query.clone();
        let t = std::time::Instant::now();
        let h1 = tokio::spawn(async move {
            for _ in 0..iters {
                let _ = s1.vsearch("a", q1.clone(), 10, 0).await.unwrap();
            }
        });
        let h2 = tokio::spawn(async move {
            for _ in 0..iters {
                let _ = s2.vsearch("b", q2.clone(), 10, 0).await.unwrap();
            }
        });
        h1.await.unwrap();
        h2.await.unwrap();
        let concurrent = t.elapsed();

        let ratio = baseline.as_secs_f64() / concurrent.as_secs_f64();
        eprintln!(
            "per-vindex lock gate · baseline {baseline:?} concurrent {concurrent:?} ratio {ratio:.2}x"
        );
        assert!(
            ratio >= 1.2,
            "per-vindex locks did not parallelise (baseline {baseline:?}, concurrent {concurrent:?}, ratio {ratio:.2}x; expected >= 1.2x)"
        );
    }
}
