//! Thread-per-core sharding.
//!
//! Each shard owns a `VLog` and runs on a dedicated thread pinned to a
//! performance core. Shards are shared-nothing: the `VLog` is touched only by
//! its own worker thread, so no locking is needed on the storage fast path.
//!
//! Requests reach a shard over an `mpsc` channel; the worker replies on a
//! per-request `tokio::oneshot`. Keys route deterministically by
//! `xxh3_64(key) % n_shards`.

use parking_lot::{Mutex, RwLock};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, mpsc as std_mpsc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Monotonic milliseconds since process start. Cheap; used to stamp a vindex's
/// `last_access` for the tiering controller's LRU ordering.
fn now_ms() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_millis() as u64
}

/// Upper bounds on the attacker-controlled VSEARCH knobs. `l_search` sizes the
/// disk-graph beam (and thus a `SmallVec::with_capacity`); `k` sizes the result
/// set and rerank pool. Generous enough for any real query, low enough that a
/// single request cannot drive a multi-GiB allocation into `panic=abort`.
const MAX_VSEARCH_K: usize = 4_096;
const MAX_VSEARCH_L_SEARCH: u32 = 8_192;

/// How often each shard checks whether a segment needs compacting.
const COMPACTION_INTERVAL: Duration = Duration::from_secs(60);

/// How often each shard writes an index snapshot for fast recovery.
const SNAPSHOT_INTERVAL: Duration = Duration::from_secs(300);

/// How often a shard checks its disk vindexes for maintenance (runs-merge /
/// delete-patch / idle consolidate). Default 10 s; `SKEG_IDLE_MAINT_MS`
/// overrides (tuning knob, and lets a stress test crank it low).
fn idle_maint_interval() -> Duration {
    std::env::var("SKEG_IDLE_MAINT_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .map_or(Duration::from_secs(10), Duration::from_millis)
}

/// Only fold an idle delta once it is at least this large; below this the flat
/// scan is cheap and a rebuild would churn the graph for little gain.
const IDLE_CONSOLIDATE_MIN: usize = 4096;

/// Flush the delta (L0) into a run off-thread once it reaches this many rows.
/// The server disables the engine's inline auto-flush and drives it here, so
/// ingest never blocks on a run build.
const FLUSH_ROWS: usize = 4096;

/// Fold the LSM runs into one (cheap, O(runs), base untouched) once the run
/// count reaches this. Keeps per-query walk cost bounded under churn without
/// waiting for a full consolidate. Fires regardless of idle - it is cheap and
/// takes only two short locks around an off-thread build.
const RUNS_MERGE_TRIGGER: usize = 4;

/// Reclaim dead base rows in place (delete-patch, O(deleted)) once tombstones
/// reach base/this (~6%): frequent enough to stay in delete-patch's cheap
/// regime, so the base stays clean without a full O(live) rebuild.
const DELETE_PATCH_DEAD_DIVISOR: usize = 16;

/// Below this base size a full consolidate is already cheap; don't bother with
/// an in-place delete-patch.
const DELETE_PATCH_MIN_BASE: usize = 4096;

use bytes::Bytes;
use futures_util::stream::{self, StreamExt};
use skeg_core::{Durability, VLog};

use crate::payload::{Filter, PayloadIndex, parse_fields};
use skeg_vector::{
    ConsolidateBuilt, ConsolidateJob, DeletePatchBuilt, DeletePatchJob, DiskVamanaIndex, FlatIndex,
    FlushBuilt, FlushJob, IvfBuilt, IvfJob, PayloadRef, QuantKind, RunMergeBuilt, RunMergeJob,
    VectorVersion,
};
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::sync::{Semaphore, oneshot};
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
/// A vindex: the vector backend plus its payload index, behind one lock so a
/// VSET updates both atomically and a filtered VSEARCH reads a consistent view.
struct Vindex {
    backend: VectorBackend,
    /// What this index has promised the process-wide governor for the heap its
    /// delta is holding. Resized as the delta grows, and released when the
    /// `Vindex` is dropped.
    ///
    /// Held for as long as the memory is, which double-counts against a LIVE
    /// source: the cgroup already subtracts the delta's allocation from its own
    /// headroom, and this promises it again. The effect is conservative - the
    /// delta is admitted up to about half the usable budget instead of all of
    /// it - and it is the price of a bound that does not depend on how quickly
    /// `memory.current` catches up. The transient alternative (reserve, insert,
    /// release) bounds nothing on its own: it leans entirely on the cgroup
    /// shrinking in time, which under fast ingest is exactly what it does not
    /// do. Upgrade path, if the halving ever costs: subtract what is
    /// outstanding from the observed headroom inside the source.
    memory: Option<crate::memory::MemoryReservation>,
    /// Effective quantization kind exposed to clients. Disk f32/binary requests
    /// use the int8 disk fallback, so this records the tier actually in use.
    kind: u8,
    /// Which incarnation of this name the open index is. Minted at
    /// `VINDEX.CREATE`, persisted in the registry, and part of every payload
    /// blob key this index writes.
    generation: IndexGeneration,
    payload: PayloadIndex,
    /// True once the payload index reflects all stored blobs. A freshly created
    /// vindex starts loaded (VSETs fill the index directly); a recovered one
    /// starts unloaded and is rebuilt from the blobs on the first filtered
    /// search (see `ensure_payload_loaded`).
    payload_loaded: bool,
    /// `now_ms()` of the last access (lookup via `get_or_reopen`). Drives the
    /// tiering controller's LRU eviction ordering. Atomic so a stamp needs only
    /// a shared borrow (no write lock on the read path).
    last_access: AtomicU64,
    /// One HEAVY maintenance job at a time for THIS vindex (consolidate,
    /// runs-merge, delete-patch, ivf). The engine's comments assumed a single
    /// outstanding job; nothing enforced it, so an explicit command could
    /// overlap the automatic loop and two runs-merges could snapshot the same
    /// runs. The flush is deliberately NOT gated: it is designed to coexist
    /// (it preserves delta/flushing) and must never queue behind a heavy job
    /// waiting on the global budget.
    ///
    /// An `Arc` so a caller can clone it under a brief read lock and then
    /// acquire it holding NO lock at all.
    heavy: Arc<tokio::sync::Semaphore>,
    /// Consecutive maintenance ticks the flush has won while a merge was
    /// also due. Fixed priority plus an early return means a permanently-hot
    /// rung starves the ones below it forever; this is the aging that breaks
    /// the tie.
    flush_streak: AtomicU64,
}

/// Bytes reserved in one step. The delta grows a row at a time and the
/// governor is an atomic compare-exchange, so asking per row would be correct
/// and wasteful; a megabyte is about a thousand rows at 256 dimensions.
const MEMORY_CHUNK: u64 = 1024 * 1024;

impl Vindex {
    /// Hold a promise covering `want` bytes of delta, rounded up to whole
    /// chunks. Grows and SHRINKS: a flush that gave the heap back is picked up
    /// here on the next write, which is the only moment the answer matters.
    /// An idle index keeps its last promise, which is memory it is not using -
    /// harmless, because an idle store is not the one under pressure.
    ///
    /// Growing takes the new promise BEFORE dropping the old, so a refusal
    /// leaves the existing one intact. It also means the two overlap for an
    /// instant; chunking makes that instant rare.
    fn reserve_memory(
        &mut self,
        governor: &Arc<crate::memory::MemoryGovernor>,
        want: u64,
    ) -> Result<(), crate::memory::MemoryRejected> {
        let chunks = want.div_ceil(MEMORY_CHUNK);
        let need = chunks.saturating_mul(MEMORY_CHUNK);
        if self.memory.as_ref().is_some_and(|r| r.bytes() == need) {
            return Ok(()); // the common case: same chunk count as last time
        }
        let held = self
            .memory
            .as_ref()
            .map_or(0, crate::memory::MemoryReservation::bytes);
        if need <= held {
            // SHRINKING: release first. Taking the smaller promise while still
            // holding the larger one puts both outstanding for an instant, and
            // against a tight budget that instant can be refused - turning a
            // reduction into a rejected write.
            self.memory = None;
            if need == 0 {
                return Ok(());
            }
        }
        // Growing: take the new promise before dropping the old, so a refusal
        // leaves the existing one intact.
        let next = governor.try_reserve(need)?;
        self.memory = Some(next);
        Ok(())
    }

    /// An index whose incarnation is not known: the pre-generation one. Used
    /// by the tests that build a `Vindex` by hand, where there is no
    /// catalogue to have recorded anything else.
    #[cfg_attr(not(test), allow(dead_code))]
    fn new(backend: VectorBackend, kind: u8) -> Self {
        Self::new_at(backend, kind, IndexGeneration::LEGACY)
    }

    fn new_at(backend: VectorBackend, kind: u8, generation: IndexGeneration) -> Self {
        Self {
            backend,
            memory: None,
            kind,
            generation,
            payload: PayloadIndex::default(),
            payload_loaded: true,
            last_access: AtomicU64::new(now_ms()),
            heavy: Arc::new(tokio::sync::Semaphore::new(1)),
            flush_streak: AtomicU64::new(0),
        }
    }

    /// A vindex reopened from disk: its payload index is empty and must be
    /// rebuilt from the stored blobs before a filtered search can use it.
    fn recovered(backend: VectorBackend, kind: u8, generation: IndexGeneration) -> Self {
        Self {
            payload_loaded: false,
            ..Self::new_at(backend, kind, generation)
        }
    }

    /// Stamp this vindex as accessed now.
    fn touch(&self) {
        self.last_access.store(now_ms(), Ordering::Relaxed);
    }

    /// Milliseconds (process clock) of the last recorded access.
    fn last_access_ms(&self) -> u64 {
        self.last_access.load(Ordering::Relaxed)
    }
}

type VectorEntry = Arc<RwLock<Vindex>>;
type VindexSet = HashMap<String, VectorEntry>;

/// Query-time search list size for a disk Vamana index.
const VAMANA_L_SEARCH: usize = 100;

/// A VINDEX is backed either by an in-RAM `FlatIndex` or by an on-disk
/// Vamana graph (`DiskVamanaIndex`) - f32 vectors on disk, graph + int8
/// tier in RAM. The choice is made at `VINDEX CREATE`.
// Both variants boxed: DiskVamanaIndex and FlatIndex are each hundreds of bytes,
// so an unboxed variant would size every VectorBackend to the larger one (clippy
// large_enum_variant). One heap indirection per vindex, off the hot path.
/// What a `Vset` decided under the write lock, before anything was staged.
///
/// Carried out of the lock rather than re-read afterwards: the values below
/// describe the write's own decisions, and re-reading them would be reading a
/// state a concurrent write could have moved - which is exactly the class of
/// bug the row stripe and the version exist to close.
struct Admitted {
    /// The version this write publishes the row at. Its blob key is built
    /// from it, so a copy at any other version has its own.
    version: u64,
    /// The version this shard held for the row before, when it held one at
    /// all. `None` for a row arriving here for the first time, which is also
    /// the answer to "is there a previous blob to carry forward or reclaim".
    previous: Option<u64>,
    /// Whether a quota slot was actually reserved, so a later failure gives
    /// back exactly what was taken and nothing else.
    charged: bool,
    /// The incarnation of the index, for the blob key.
    generation: IndexGeneration,
}

/// Write a payload blob to the key of the version it belongs to. `Ok(true)`
/// once the blob is there.
///
/// Split out because both the supplied-payload and the carry-forward paths
/// write to the same place under the same failpoint, and a second copy of that
/// is a second place for the key to be built differently.
async fn stage_payload_blob(
    vlog: &VLog,
    scope: BlobScope<'_>,
    id: u64,
    version: u64,
    blob: &[u8],
) -> Result<bool, String> {
    crate::fp_at!(
        crate::failpoint::WriteFailpoint::PayloadPrepare,
        scope.name,
        Err("vset payload failed: failpoint: payload staging refused".to_owned())
    );
    vlog.tenant(scope.tenant)
        .set(&scope.key(id, version), blob, PAYLOAD_DURABILITY)
        .await
        .map(|()| {
            skeg_telemetry::tick_counter(skeg_telemetry::Counter::PayloadBlobsStaged);
            true
        })
        .map_err(|e| format!("vset payload failed: {e}"))
}

enum VectorBackend {
    Flat(Box<FlatIndex>),
    Disk(Box<DiskVamanaIndex>),
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

    /// RAM footprint of this vindex in bytes. Used to refresh the
    /// `VindexSizeBytes` gauge from STATS. Cheap (arithmetic / a sum
    /// over resident structures), safe to poll.
    ///
    /// Flat indexes carry the full f32 row buffer in RAM. Disk indexes
    /// keep graph + quantized tier + delta resident; the full f32
    /// vectors live on disk and are paged in by the OS, so they are not
    /// counted here. The disk number is the index's own deterministic
    /// `resident_bytes()` (tier-accurate: int8 vs tq1/tq2/tq4), not a
    /// process RSS sample.
    fn approx_ram_bytes(&self) -> u64 {
        match self {
            // The index's OWN number, not live rows times dim. A flat index
            // never gives a row back: a deleted row keeps its f32, and the
            // versions of ids it was told to delete and never held are held
            // too. Multiplying the LIVE count by `dim` reported none of that,
            // so a workload of deletes grew the process while `SKEG.STATS`,
            // the tiering controller and memory admission all read a number
            // that was going down.
            VectorBackend::Flat(i) => i.resident_bytes() as u64,
            VectorBackend::Disk(i) => i.resident_bytes() as u64,
        }
    }

    /// Wire backend byte: 0 = flat, 1 = disk Vamana.
    fn backend_byte(&self) -> u8 {
        match self {
            VectorBackend::Flat(_) => 0,
            VectorBackend::Disk(_) => 1,
        }
    }

    /// Insert a vector. A disk backend APPENDS it to the delta (cheap: buffer +
    /// WAL, no graph work) and rebuilds the graph on a GEOMETRIC schedule - once
    /// the un-built delta matches the built size (a doubling). That is O(log N)
    /// bulk builds over a load = O(N) total build work, and every build is a
    /// clean two-pass Vamana graph.
    ///
    /// We do NOT insert incrementally into the graph: that produced a graph that
    /// stays connected but is poorly NAVIGABLE by the greedy walk (back-edge
    /// re-pruning erodes the long-range edges), so plain recall@10 fell to ~0.31
    /// vs 1.00 for a bulk build at 100k. Bulk-building beats incremental on both
    /// recall and bulk-load speed; incremental insert is kept for an explicit
    /// streaming path only.
    ///
    /// `version` says WHICH copy of the row this is. A write older than the
    /// copy the backend already holds is dropped by the backend itself, which
    /// is what stops a relocation republishing a value a user write replaced.
    ///
    /// `payload_ref` says what this copy's payload blob is, and the record
    /// that carries it is the commit point of the pair. A FLAT index has no
    /// WAL and therefore nothing to record it in, which is the whole of what
    /// "flat is ephemeral by design" costs here: an in-RAM index recovers
    /// nothing, so there is no recovery to describe.
    fn insert(
        &mut self,
        id: u64,
        vector: &[f32],
        version: VectorVersion,
        payload_ref: PayloadRef,
    ) -> std::io::Result<()> {
        match self {
            VectorBackend::Flat(i) => {
                // `FlatIndex::insert` panics on a dim mismatch and its
                // signature is shared with callers that rely on that. Check
                // here so a bad dimension from one client cannot take down a
                // shard thread that is serving every other vindex.
                if i.dim() != vector.len() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!("vector has {} dims, index has {}", vector.len(), i.dim()),
                    ));
                }
                let _ = payload_ref;
                i.insert_versioned(id, vector, version);
                Ok(())
            }
            VectorBackend::Disk(i) => {
                // Append only. The geometric fold (delta >= built size) runs
                // OFF-THREAD in the background maintenance loop (with the cheap
                // consolidate_begin), so ingest never blocks the shard on a fold.
                i.insert_with_payload(id, vector, version, payload_ref)
            }
        }
    }

    fn delete(&mut self, id: u64, version: VectorVersion) -> std::io::Result<bool> {
        match self {
            VectorBackend::Flat(i) => Ok(i.delete_versioned(id, version)),
            VectorBackend::Disk(i) => i.delete_versioned(id, version),
        }
    }

    /// The newest version this shard holds for `id`, live or tombstoned.
    fn version_of(&self, id: u64) -> VectorVersion {
        match self {
            VectorBackend::Flat(i) => i.version_of(id),
            VectorBackend::Disk(i) => i.version_of(id),
        }
    }

    /// Every live id with the version of the copy this shard holds. What an
    /// owner-map rebuild needs: the ids say which shards hold a row, the
    /// versions say which of them holds the one to serve.
    fn live_ids_with_versions(&self) -> Vec<(u64, VectorVersion)> {
        match self {
            VectorBackend::Flat(i) => i.live_ids_with_versions(),
            VectorBackend::Disk(i) => i.live_ids_with_versions(),
        }
    }

    /// The highest version this shard has recorded for the index, tombstones
    /// included. Separate from the live ids because a DELETED row can be the
    /// high-water mark and is in no list of live ones.
    fn max_version(&self) -> VectorVersion {
        match self {
            VectorBackend::Flat(i) => i.max_version(),
            VectorBackend::Disk(i) => i.max_version(),
        }
    }

    /// True if `id` is currently stored (live). Cheap, in-memory; lets the
    /// quota tell a new insert from an overwrite.
    fn contains(&self, id: u64) -> bool {
        match self {
            VectorBackend::Flat(i) => i.contains(id),
            VectorBackend::Disk(i) => i.contains(id),
        }
    }

    fn get(&self, id: u64) -> std::io::Result<Option<Vec<f32>>> {
        match self {
            VectorBackend::Flat(i) => Ok(i.get(id)),
            VectorBackend::Disk(i) => i.get(id),
        }
    }

    /// Every live vector id. Used to reclaim per-id payload blobs on
    /// `VINDEX.DROP`, where the blobs live in the KV vLog under a reserved key.
    fn live_ids(&self) -> Vec<u64> {
        match self {
            VectorBackend::Flat(i) => i.live_ids(),
            VectorBackend::Disk(i) => i.live_ids(),
        }
    }

    /// Fold a disk index's streaming delta into the graph (one full rebuild),
    /// leaving it in its fast fully-indexed state. Flat has no delta: no-op.
    /// Consolidated base-graph size (flat: its full length). Sizes the
    /// delete-patch trigger against the tombstone count.
    fn main_len(&self) -> usize {
        match self {
            VectorBackend::Flat(i) => i.len(),
            VectorBackend::Disk(i) => i.main_len(),
        }
    }

    /// LSM run count (0 for flat). Drives the runs-merge trigger.
    fn run_count(&self) -> usize {
        match self {
            VectorBackend::Flat(_) => 0,
            VectorBackend::Disk(i) => i.run_count(),
        }
    }

    /// Total rows in flushed runs (0 for flat). Drives the consolidate trigger.
    fn run_rows(&self) -> usize {
        match self {
            VectorBackend::Flat(_) => 0,
            VectorBackend::Disk(i) => i.run_rows(),
        }
    }

    /// Rows in the LARGEST single run, not their sum: a merge can cut both
    /// the count and the sum while leaving one enormous segment behind.
    fn max_run_rows(&self) -> usize {
        match self {
            VectorBackend::Flat(_) => 0,
            VectorBackend::Disk(i) => i.max_run_rows(),
        }
    }

    /// `(physical, live, garbage)` rows in the runs.
    fn run_contents(&self) -> (usize, usize, usize) {
        match self {
            VectorBackend::Flat(_) => (0, 0, 0),
            VectorBackend::Disk(i) => i.run_contents(),
        }
    }

    /// Run debt over live rows, from the engine that owns the definition.
    fn run_debt_ratio(&self) -> f32 {
        match self {
            VectorBackend::Flat(_) => 0.0,
            VectorBackend::Disk(i) => i.run_debt_ratio(),
        }
    }

    /// Flush (L0): snapshot the delta into a run off-thread. `None` if empty /
    /// already flushing / flat.
    fn flush_begin(&mut self) -> std::io::Result<Option<FlushJob>> {
        match self {
            VectorBackend::Flat(_) => Ok(None),
            VectorBackend::Disk(i) => i.flush_begin(),
        }
    }

    fn flush_finish(&mut self, built: FlushBuilt) -> skeg_vector::FinishResult {
        match self {
            VectorBackend::Flat(_) => Ok(skeg_vector::FinishOutcome::Committed),
            VectorBackend::Disk(i) => i.flush_finish(built),
        }
    }

    fn flush_abort(&mut self) {
        if let VectorBackend::Disk(i) = self {
            i.flush_abort();
        }
    }

    /// IVF rebuild, off-thread: dup the base fd. `None` if flat / empty.
    fn ivf_begin(&self) -> std::io::Result<Option<IvfJob>> {
        match self {
            VectorBackend::Flat(_) => Ok(None),
            VectorBackend::Disk(i) => i.ivf_begin(0, 8),
        }
    }

    fn ivf_finish(&mut self, built: IvfBuilt) -> skeg_vector::FinishResult {
        match self {
            VectorBackend::Flat(_) => Ok(skeg_vector::FinishOutcome::Committed),
            VectorBackend::Disk(i) => i.ivf_finish(built),
        }
    }

    /// Live tombstones (0 for flat). Cheap gate for the delete-patch trigger.
    fn tombstone_count(&self) -> usize {
        match self {
            VectorBackend::Flat(_) => 0,
            VectorBackend::Disk(i) => i.tombstone_count(),
        }
    }

    // ── Off-thread maintenance: begin (short lock) → build (off-lock, on the
    // blocking pool) → finish (short lock). Flat has no maintenance, so `begin`
    // returns `None` and the caller skips straight past. ──────────────────────

    /// L1: snapshot for a full background consolidate (fold delta + runs into a
    /// fresh base). `None` if nothing to fold or flat.
    fn consolidate_begin(&mut self) -> std::io::Result<Option<ConsolidateJob>> {
        match self {
            VectorBackend::Flat(_) => Ok(None),
            VectorBackend::Disk(i) => i.consolidate_begin(),
        }
    }

    fn consolidate_finish(&mut self, built: ConsolidateBuilt) -> skeg_vector::FinishResult {
        match self {
            VectorBackend::Flat(_) => Ok(skeg_vector::FinishOutcome::Committed),
            VectorBackend::Disk(i) => i.consolidate_finish(built),
        }
    }

    /// L2: snapshot for a runs-only merge (fold the LSM runs into one, base
    /// untouched). `None` if fewer than two runs or flat.
    fn merge_runs_begin(&mut self) -> std::io::Result<Option<RunMergeJob>> {
        match self {
            VectorBackend::Flat(_) => Ok(None),
            VectorBackend::Disk(i) => i.merge_runs_begin(),
        }
    }

    fn merge_runs_finish(&mut self, built: RunMergeBuilt) -> skeg_vector::FinishResult {
        match self {
            VectorBackend::Flat(_) => Ok(skeg_vector::FinishOutcome::Committed),
            VectorBackend::Disk(i) => i.merge_runs_finish(built),
        }
    }

    /// L3: snapshot for a delete-patch (reclaim dead base rows in place, runs
    /// untouched). `None` if there is nothing to reclaim or flat.
    fn delete_patch_begin(&mut self) -> std::io::Result<Option<DeletePatchJob>> {
        match self {
            VectorBackend::Flat(_) => Ok(None),
            VectorBackend::Disk(i) => i.delete_patch_begin(),
        }
    }

    fn delete_patch_finish(&mut self, built: DeletePatchBuilt) -> skeg_vector::FinishResult {
        match self {
            VectorBackend::Flat(_) => Ok(skeg_vector::FinishOutcome::Committed),
            VectorBackend::Disk(i) => i.delete_patch_finish(built),
        }
    }

    /// True if the disk index would benefit from (and lacks) an IVF router.
    /// The background idle-consolidate builds it off the ingest path.
    fn wants_ivf(&self) -> bool {
        match self {
            VectorBackend::Flat(_) => false,
            VectorBackend::Disk(i) => i.wants_ivf(),
        }
    }

    /// Un-consolidated streaming inserts (the in-RAM delta that a bulk load
    /// leaves behind). 0 for flat. Drives the idle-consolidate trigger.
    fn delta_len(&self) -> usize {
        match self {
            VectorBackend::Flat(_) => 0,
            VectorBackend::Disk(i) => i.delta_len(),
        }
    }

    /// Heap this index holds: for the disk backend the delta plus any flush
    /// staging, which a flush gives back; for the flat backend everything,
    /// which nothing gives back.
    fn resident_bytes(&self) -> u64 {
        match self {
            // A flat index is ALL of it: it keeps every vector in RAM and
            // nothing flushes it, so it is the case admission most needs to
            // cover rather than the one it can skip. Reporting zero here
            // exempted the only backend that never gives anything back - and
            // did not even exempt it kindly, since a cost that never grows
            // rounds every insert up to one whole chunk.
            VectorBackend::Flat(i) => i.resident_bytes() as u64,
            VectorBackend::Disk(i) => i.resident_bytes() as u64,
        }
    }

    /// Top-`k` for a payload filter whose matching id set is `s` (SORTED, from
    /// `Filter::evaluate`, already per-shard). Flat is brute-force. On disk, the
    /// hybrid planner ([`search_filtered_hybrid`]): a tiny filter is an exact
    /// quantized scan of `s`; a larger one is IVF-routed to the query-nearest
    /// matching cells (sub-linear, scales to 1M/10M), then f32-reranked. Both
    /// bound the disk reads to `k*8`, unlike a raw scan that goes O(|s|).
    fn filtered_search(
        &self,
        query: &[f32],
        k: usize,
        _l_search: u32,
        s: &[u64],
    ) -> std::io::Result<Vec<(u64, f32)>> {
        const RERANK: usize = 8;
        let k = k.min(MAX_VSEARCH_K);
        let rerank = (k * RERANK).max(64);
        match self {
            VectorBackend::Flat(i) => {
                if i.dim() != query.len() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!("query has {} dims, index has {}", query.len(), i.dim()),
                    ));
                }
                Ok(i.score_ids(query, s, k))
            }
            VectorBackend::Disk(i) => i.search_filtered_hybrid(query, s, k, rerank),
        }
    }

    fn search(
        &mut self,
        query: &[f32],
        k: usize,
        l_search: u32,
    ) -> std::io::Result<Vec<(u64, f32)>> {
        // `k` and `l_search` are attacker-controlled wire fields. On the Disk
        // backend `l_search` becomes the beam width, sized straight into a
        // `SmallVec::with_capacity`, so an unclamped `u32::MAX` requests tens of
        // GiB and (panic=abort) takes down the whole server on one packet.
        let k = k.min(MAX_VSEARCH_K);
        let l_search = l_search.min(MAX_VSEARCH_L_SEARCH);
        match self {
            // Flat is brute-force: no search-list, l_search does not apply.
            VectorBackend::Flat(i) => {
                if i.dim() != query.len() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!("query has {} dims, index has {}", query.len(), i.dim()),
                    ));
                }
                Ok(i.search(query, k))
            }
            VectorBackend::Disk(i) => i.search_with_l(query, k, l_search as usize),
        }
    }
}

/// Bounded inbox capacity per shard. A full inbox makes `send` await, which
/// propagates backpressure up to the connection handler (it stops reading new
/// frames) instead of letting queues grow without bound (OOM-safety, not
/// latency).
const SHARD_INBOX_CAPACITY: usize = 4096;

/// Maximum requests a shard processes concurrently. The inbox bounds the
/// *queue*; this bounds the *in-flight* request tasks. When both are full,
/// `send` blocks and backpressure reaches the client.
const MAX_INFLIGHT_PER_SHARD: usize = 1024;

/// Deletes kept in flight during a tenant erasure. High enough that the group
/// committer batches them into shared flushes instead of one flush per key; no
/// higher than the concurrency the write path already handles from ordinary
/// clients (`MAX_INFLIGHT_PER_SHARD`), so segment rotation stays inside a
/// regime it is already exercised at.
const ERASE_CONCURRENCY: usize = 256;

/// Payload blob reads in flight while rebuilding one vindex's payload index.
///
/// Modest on purpose: every shard rebuilds at once, so the device already sees
/// one stream per shard and this multiplies that. Past the point where the
/// queue is full, more requests only add latency to each.
const PAYLOAD_READ_CONCURRENCY: usize = 32;

/// Route a key to a shard index.
#[must_use]
pub fn shard_for(key: &[u8], n_shards: usize) -> usize {
    debug_assert!(n_shards >= 1);
    #[allow(clippy::cast_possible_truncation)]
    let idx = (xxh3_64(key) % n_shards as u64) as usize;
    idx
}

/// Reject a vindex name that would escape the data dir. The name flows into
/// `dir.join(format!("vindex-{name}"))` for create / `File::create` /
/// `remove_dir_all`. The RESP3 layer already applies a strict charset, but the
/// native binary protocol does not - this is the choke point both protocols
/// cross, so it must hold on its own. Permits `:` for the `{tenant}::{name}`
/// scope prefix the RESP3 layer prepends; rejects anything that could traverse.
pub(crate) fn validate_vindex_name(name: &str) -> Result<(), ShardError> {
    let ok = !name.is_empty()
        && name.len() <= 255
        && name != "."
        && name != ".."
        && !name.contains("..")
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b':'));
    if ok {
        Ok(())
    } else {
        Err(ShardError::Storage(
            "vindex name: 1-255 chars, [A-Za-z0-9._:-] only, no '..'".to_owned(),
        ))
    }
}

/// The separator that carries the tenant scope inside a vindex map key: a key
/// is either a bare name (tenant 0) or `<32 hex>::<name>`. Written by
/// [`scope_key`], read back by [`unscope_key`].
pub(crate) const SCOPE_SEP: &str = "::";

/// Refuse the scope separator in a RAW, client-supplied vindex name.
///
/// A tenant is not something a client gets to spell. `unscope_key` derives the
/// owner of an index FROM its map key, and `EraseTenant`, the payload warm-up,
/// the blob sweeps and `IndexStat` all act on what it says. That is sound only
/// while the `<32 hex>::` prefix can only have been put there by the server -
/// and it could not be: `scope_key(0, name)` returns the raw name unchanged,
/// and the native protocol always calls with tenant 0 and the client's own
/// name. A client could therefore create `<32 hex of B>::x`, which every one
/// of those sites then attributed to tenant B.
///
/// So the separator is refused at the create door, on every entry path. That
/// is the whole guarantee: no key with a scope prefix exists unless the server
/// wrote the prefix itself, from an authenticated tenant id.
pub(crate) fn reject_scope_separator(name: &str) -> Result<(), ShardError> {
    if name.contains(SCOPE_SEP) {
        return Err(ShardError::Storage(format!(
            "vindex name must not contain '{SCOPE_SEP}' (reserved for tenant scoping)"
        )));
    }
    Ok(())
}

/// Error returned by `ShardSet` operations.
#[derive(Debug, thiserror::Error)]
pub enum ShardError {
    #[error("shard unavailable")]
    Unavailable,
    #[error("vsearch queue is full")]
    Busy,
    #[error("storage error: {0}")]
    Storage(String),
}

// ── Channel protocol ──────────────────────────────────────────────────────────

enum ShardReq {
    /// `(key, tenant)`: tenant `0` is the unscoped default.
    Get(Bytes, u128),
    /// `(key, value, durability, tenant, disk_limit)`.
    Set(Bytes, Bytes, Durability, u128, Option<u64>),
    /// Atomic multi-key write for the keys of one MSET that route to this shard.
    /// Keys are already tenant-scoped; the batch is all-or-nothing on this shard.
    SetMany(Vec<(Bytes, Bytes)>, Durability),
    /// `(key, value, durability, tenant, disk_limit)`: append to a scoped key,
    /// reply with the new value length.
    Append(Bytes, Bytes, Durability, u128, Option<u64>),
    Del(Bytes, Durability),
    /// Erase everything owned by `tenant` on this shard: its vindexes (with
    /// their payload blobs and vector quota) and every KV key carrying its
    /// 16-byte prefix. Refused for tenant `0`, whose keys are unscoped.
    EraseTenant {
        tenant: u128,
        durability: Durability,
    },
    /// Erase a subject within a tenant: every KV key under `tenant`'s prefix
    /// followed by `subject`. No vindex drop. Refused for tenant `0`.
    ErasePrefix {
        tenant: u128,
        subject: Vec<u8>,
        durability: Durability,
    },
    /// Physically reclaim every dead byte on this shard's store (compact all
    /// sealed segments with dead records). The durable other half of an erase.
    Reclaim,
    /// Count this shard's live KV keys carrying `tenant`'s 16-byte prefix.
    CountTenantKeys(u128),
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
        /// Which incarnation of this name is being created. Minted ONCE, by
        /// the coordinator, so every shard records the same one: a
        /// per-shard mint would give the same logical index a different
        /// blob namespace on each shard.
        generation: IndexGeneration,
    },
    VindexDrop {
        name: String,
        /// Owning tenant, so its vector quota is credited for the dropped
        /// fragment. `0` for the unscoped default.
        tenant: u128,
        /// Who gives the tenant its slots back. Decided by the coordinator,
        /// which is the only place that knows whether the index is routed.
        credit: DropCredit,
        /// Whether a shard that does not have this index is an error.
        ///
        /// True for a client's DROP: it asked for a named thing and deserves to
        /// hear that it was not there. False when the drop is CONVERGENCE - the
        /// rollback of a create that reached only some shards - where the
        /// shards that never got it are the normal case and reporting them as
        /// failures would hide whether the shards that DID get it were cleaned.
        require_present: bool,
    },
    /// Enumerate VINDEXes known to this shard. Replicated across all
    /// shards so callers can ask any one shard.
    VindexList,
    /// Integrity report for one vindex (the operator's fsck).
    VindexCheck {
        name: String,
    },
    /// Does this vindex still want an IVF router? Contract probe: the only
    /// caller is the test that pins "an explicit consolidate leaves the
    /// index in the same state the maintenance ladder would".
    #[cfg_attr(not(test), allow(dead_code))]
    WantsIvf {
        name: String,
    },
    /// Fold a disk vindex's streaming delta into its graph on this shard.
    /// Write the vlog snapshot and a payload index per vindex, both stamped
    /// with the same log position. Normally the background task's job; exposed
    /// so a caller (and the tests that check the cache cannot go stale) can
    /// force one at a known point.
    SnapshotAndPayloadIndexes,
    VindexConsolidate {
        name: String,
    },
    /// Non-destructive evict: drop the in-RAM entry for `name` without touching
    /// its files, so the next access reopens it lazily. `name` is the scoped
    /// map key. Allowed in read-only (serve) mode: it frees RAM, not data.
    Evict {
        name: String,
    },
    /// Per-index RAM / access stats for this shard's open vindexes. Used by the
    /// tiering controller to decide what to evict.
    IndexStats,
    Vset {
        name: String,
        id: u64,
        vector: Vec<f32>,
        /// Owning tenant for vector-quota accounting (`0` = unscoped).
        tenant: u128,
        /// Tenant's max vectors, if any. `None` skips quota enforcement.
        limit: Option<u64>,
        /// What this write means for the tenant's logical cardinality. The
        /// quota counts distinct rows and a shard can only see its own, so
        /// the coordinator - which knows whether the row is arriving, being
        /// overwritten, moved or replicated - says which it is and
        /// [`QuotaEffect`](crate::quota::QuotaEffect) decides what it costs.
        effect: crate::quota::QuotaEffect,
        /// Which copy of the row this is.
        ///
        /// `Some` when the coordinator knows: a user write ALLOCATES one under
        /// the id's stripe, and a write that only relocates a row - a reshard
        /// move, a boundary replica - CARRIES the version it read, so it
        /// cannot land on top of the user write that replaced the row while it
        /// was in flight.
        ///
        /// `None` on the hash-placed path, where a row never moves between
        /// shards and the answering shard's own counter is the whole truth: it
        /// allocates one past whatever it already holds for the id.
        version: Option<u64>,
        /// Optional opaque payload blob stored alongside the vector. `None`
        /// leaves the write path byte-identical to a payload-less VSET.
        payload: Option<Bytes>,
    },
    Vget {
        name: String,
        id: u64,
    },
    /// Uniform sample of up to `count` live f32 vectors from `name` on this
    /// shard (router training input).
    SampleVectors {
        name: String,
        count: usize,
    },
    /// One reshard batch: live ids after `after` whose semantic owner (by the
    /// carried centroids) is NOT `own` - returned with vector and payload so
    /// the set can move them. `limit` bounds the batch.
    CollectMoves {
        name: String,
        centroids: Arc<crate::router::Router>,
        own: u8,
        after: u64,
        limit: usize,
        tenant: u128,
    },
    /// Every live id of `name` on this shard (owner-map rebuild at open).
    LiveIds {
        name: String,
    },
    /// Payload blobs this shard still holds for `name`, across every
    /// generation of the name and including the pre-generation key.
    ///
    /// The blobs are KV keys under a reserved marker: nothing on the vector
    /// side can see them, so without this the reclamation paths have no
    /// observable behaviour to test.
    PayloadBlobs {
        tenant: u128,
        name: String,
    },
    /// Is `id` still the exact row a relocation read - live here, and at
    /// `version`? The one question the owner map cannot answer: a successful
    /// delete REMOVES the map entry, and an absent entry is also the normal
    /// state of a row that has never moved. The tombstone is here.
    StillCurrent {
        name: String,
        id: u64,
        version: u64,
    },
    /// Boundary rows for the targeted overlap: live rows whose margin
    /// between the two nearest centroids is below `tau` - returned with
    /// vector, payload and the SECOND-nearest shard they replicate to.
    CollectBoundary {
        name: String,
        centroids: Arc<crate::router::Router>,
        after: u64,
        limit: usize,
        tau: f32,
        tenant: u128,
    },
    /// Sample of the shard's base graph for visual exploration.
    GraphSample {
        name: String,
        count: usize,
    },
    Vdel {
        name: String,
        id: u64,
        /// Owning tenant, so its vector quota is credited on a real delete.
        tenant: u128,
        /// What this delete means for the tenant's logical cardinality. Only
        /// [`Delete`](crate::quota::QuotaEffect::Delete) credits: the far
        /// side of a move and a replica whose primary was already credited
        /// remove a copy of a row the tenant still has.
        effect: crate::quota::QuotaEffect,
        /// Which copy of the row this removes. Same rule as `Vset`: allocated
        /// for a user delete, carried for the far side of a move or a replica,
        /// `None` on the hash-placed path.
        version: Option<u64>,
    },
    Vsearch {
        name: String,
        query: Vec<f32>,
        k: usize,
        l_search: u32,
        /// Owning tenant, needed to locate each hit's payload blob.
        tenant: u128,
        /// When true, attach each hit's stored payload to the response.
        want_payload: bool,
        /// Optional payload filter. When set, the search is the exact
        /// brute-force over the matching id set instead of the ANN walk.
        filter: Option<Filter>,
    },
}

/// Who gives a tenant its vector-quota slots back when an index is dropped.
///
/// The quota counts LOGICAL rows and a shard holds PHYSICAL ones, and for a
/// routed index those are not the same number: an overlap keeps a second copy
/// of every boundary row, and a crash between a move's write and its source
/// delete leaves another. Letting each shard credit what it held therefore
/// gave back more slots than the tenant had ever spent - the subtraction
/// saturated at zero and the tenant could write a whole `max_vectors` on top
/// of what it already held, until the next restart repaired the count.
///
/// So the decision is made where the logical cardinality is known, and this
/// enum is how the coordinator tells the shard which of them is crediting.
/// Exhaustive at the one site that acts on it: a third kind of drop has to
/// say what it costs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DropCredit {
    /// The shard credits what it physically holds. Correct, and cheapest,
    /// for a hash-placed index: an id maps to exactly one shard for ever, so
    /// its fragment IS its logical share.
    Fragment,
    /// The shard credits nothing; the coordinator does it once, from the
    /// owner map. For a routed index, and for an erasure - which ends with
    /// the tenant holding nothing, so the only number that can be right
    /// afterwards is zero.
    Coordinator,
}

/// One VINDEX row as the shards report it: identity plus the LSM debt the
/// maintenance ladder acts on. `n_vectors`/`delta`/`run_rows`/`tombs`/`base`
/// sum across shards; `runs` sums too (total run segments held).
#[derive(Debug, Clone)]
pub struct VindexRow {
    pub name: String,
    /// Shards on which this index is OPEN, out of the shards in the STORE.
    ///
    /// Every other number in this row is summed over the resident shards only,
    /// so `shards_resident < shards_total` means the row is a partial reading
    /// and must be reported as one.
    ///
    /// `shards_total` is deliberately the store's shard count and not the
    /// number of shards that answered with this index. Counting answers made an
    /// index committed on three shards of four - a create that failed partway
    /// and rolled nothing back - read as `3/3`, complete: the same "partial
    /// answer that looks complete" the field exists to prevent, one level up.
    /// The per-shard value is a placeholder; the coordinator fills it in.
    pub shards_resident: u32,
    pub shards_total: u32,
    pub dim: u32,
    pub kind: u8,
    pub backend: u8,
    pub n_vectors: u64,
    pub delta: u64,
    pub runs: u64,
    pub run_rows: u64,
    /// Rows in the LARGEST single run. Distinct from `run_rows`, which is
    /// their sum: a merge can cut the count and the sum while leaving one
    /// enormous segment behind, and only this number shows it.
    pub max_run_rows: u64,
    /// Run rows over live rows, computed ONCE by the engine that owns the
    /// definition. Every P0 found today had the same shape - one rule
    /// implemented in two places, one of them wrong - so this number is not
    /// recomputed by its readers.
    ///
    /// AMPLIFICATION, not garbage: a run holding exactly the live set scores
    /// 1.0, and so does one of the same size holding nothing but corpses.
    /// Use `run_live`/`run_dead` to tell those apart.
    pub run_debt_ratio: f32,
    /// Run rows that are still the newest live copy of their id.
    pub run_live: u64,
    /// Run rows a vacuum could reclaim: dead, tombstoned or superseded.
    pub run_dead: u64,
    pub tombs: u64,
    pub base: u64,
}

/// Default probe width for routed searches: `SKEG_PROBE` (0 = full
/// fan-out). Ships at 0 until the probe gate clears on the live corpus.
fn probe_default() -> usize {
    static P: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *P.get_or_init(|| {
        std::env::var("SKEG_PROBE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

/// Load every `router-<name>.bin` under the shard-set root.
///
/// FAIL CLOSED on a sidecar that is there and will not read. It used to warn
/// and skip, which is not a smaller failure - it is a silent change of how the
/// store behaves. The index comes back UNROUTED, so point ops fall through to
/// hash placement on rows that have been physically re-partitioned, and the
/// vector-quota rebuild adds the shards' counts instead of deduplicating their
/// ids: measured at 119 against 60 logical rows, after which the tenant is
/// refused at half its limit. An open that cannot know how the store is routed
/// has no serviceable state to offer, and the same stance is already taken by
/// the registry that will not round-trip and by the owner-map rebuild that
/// cannot read a shard.
///
/// A MISSING sidecar is a different thing and stays legal: an index that was
/// never resharded has none, and `read_dir` simply does not yield one. Only a
/// file that exists under the name and does not parse refuses, and the error
/// names it, because renaming or removing that file is the whole repair.
///
/// # Known gap: a sidecar that was DELETED reads as one that never existed
///
/// The two cases are the same absence on disk, and this function cannot tell
/// them apart: an index that WAS resharded and whose sidecar was deleted - or
/// not restored from a backup - opens as unrouted and gets exactly the damage
/// the corrupt branch above refuses. Measured at 79 counted against 40 logical
/// rows, plus point ops placed by hash over rows the reshard moved.
///
/// Nothing here can close it, because the question is "was this index ever
/// routed", and only the vindex registry could answer it - it does not record
/// that today. Recording it there turns a missing sidecar into the same
/// refusal a corrupt one now gets, and that is the fix. Until then it is an
/// operator check, stated in the CHANGELOG: after restoring or hand-editing a
/// store root, every resharded vindex must still have its `router-<name>.bin`,
/// and the repair is to put the file back or to reshard the index again.
fn load_routers(root: &Path) -> std::io::Result<HashMap<String, Arc<crate::router::Router>>> {
    let mut out = HashMap::new();
    // The root is created by the layout manifest before this runs, so a
    // directory that will not list is itself a store this open cannot
    // describe - and "no sidecars" is exactly the wrong thing to conclude
    // from it.
    let entries = std::fs::read_dir(root).map_err(|e| {
        std::io::Error::other(format!(
            "vindex router sidecars in {} cannot be listed: {e}",
            root.display()
        ))
    })?;
    for entry in entries.flatten() {
        let file = entry.file_name().to_string_lossy().into_owned();
        if let Some(name) = file
            .strip_prefix("router-")
            .and_then(|s| s.strip_suffix(".bin"))
        {
            match crate::router::Router::load(&entry.path()) {
                Ok(r) => {
                    out.insert(name.to_owned(), Arc::new(r));
                }
                Err(e) => {
                    return Err(std::io::Error::other(format!(
                        "vindex router sidecar {file} will not read ({e}); refusing to \
                         open '{name}' as an unrouted index, which would place its point \
                         ops by hash over rows the reshard moved and count its rows twice. \
                         Restore the file from a backup, or remove it and reshard again."
                    )));
                }
            }
        }
    }
    Ok(out)
}

enum ShardResp {
    Value(Option<Bytes>),
    Done,
    Existed(bool),
    /// What an `EraseTenant` removed on the answering shard.
    Erased {
        vindexes: u64,
        keys: u64,
    },
    /// Live key count for a tenant on the answering shard.
    Count(u64),
    /// Integrity problems found; empty = healthy.
    Problems(Vec<String>),
    /// New value length after an APPEND.
    Len(u64),
    /// Dead bytes physically reclaimed on the answering shard.
    Reclaimed(u64),
    /// Bytes of hot-key cache charged to a tenant on the answering shard.
    CacheBytes(usize),
    MgetBatch(Vec<(usize, Option<Bytes>)>),
    /// `(cache_bytes, cache_evictions, n_keys, cache_budget)`.
    Stats(u64, u64, u64, u64),
    /// `(name, dim, kind_wire_byte, backend_wire_byte, n_vectors)` per VINDEX.
    VindexList(Vec<VindexRow>),
    /// Flattened row-major sample rows plus their dim.
    Sample(Vec<f32>, u32),
    /// A reshard batch: (id, vector, payload, owner, version) plus the resume
    /// cursor (`None` when the shard is exhausted). The version travels with
    /// the row: it is what the mover writes at the destination, and what tells
    /// it the row has moved on since the batch was collected.
    Moves(Vec<MoveRow>, Option<u64>),
    /// Answer to `LiveIds`. See [`LiveIdsAnswer`]: "this shard does not have
    /// the index" and "this shard cannot read the index" are separate states.
    LiveIds(LiveIdsAnswer),
    /// Graph sample: (id, degree) nodes and (from, to) edges.
    Graph(Vec<(u64, u32)>, Vec<(u64, u64)>),
    /// VGET result: the stored f32 vector, or `None` if absent.
    Vector(Option<Vec<f32>>),
    /// VSEARCH result for this shard's fragment: `(vec_id, cosine, payload,
    /// version)` hits. `payload` is `Some` only when the request set
    /// `want_payload`; otherwise always `None`, so the non-payload path
    /// encodes identically.
    Vsearch(Vec<VsearchHit>),
    /// Evict result: `true` if an entry was present and removed, `false` if it
    /// was already absent (evicted or never open) on this shard.
    Evicted(bool),
    /// Per-index stats for this shard: `(scoped_name, resident_bytes,
    /// last_access_ms, n_vectors, evictable)`.
    IndexStats(Vec<(String, usize, u64, usize, bool)>),
    Err(String),
}

struct ShardMsg {
    req: ShardReq,
    reply: oneshot::Sender<ShardResp>,
}

type VsearchResult = Result<Vec<(u64, f32)>, String>;
type VsearchTask = Box<dyn FnOnce() -> VsearchResult + Send>;

struct VsearchJob {
    task: VsearchTask,
    reply: oneshot::Sender<VsearchResult>,
}

struct VsearchPool {
    sender: Option<std_mpsc::SyncSender<VsearchJob>>,
    handles: Mutex<Vec<JoinHandle<()>>>,
}

impl VsearchPool {
    fn new(workers: usize) -> std::io::Result<Arc<Self>> {
        assert!(workers > 0, "vsearch workers must be positive");
        let (sender, receiver) = std_mpsc::sync_channel::<VsearchJob>(workers);
        let receiver = Arc::new(Mutex::new(receiver));
        let mut handles = Vec::with_capacity(workers);
        for id in 0..workers {
            let receiver = receiver.clone();
            handles.push(
                std::thread::Builder::new()
                    .name(format!("skeg-vsearch-{id}"))
                    .spawn(move || {
                        loop {
                            let job = { receiver.lock().recv() };
                            match job {
                                Ok(job) => {
                                    let _ = job.reply.send((job.task)());
                                }
                                Err(_) => return,
                            }
                        }
                    })?,
            );
        }
        Ok(Arc::new(Self {
            sender: Some(sender),
            handles: Mutex::new(handles),
        }))
    }

    fn submit<F>(&self, task: F) -> Result<oneshot::Receiver<VsearchResult>, &'static str>
    where
        F: FnOnce() -> VsearchResult + Send + 'static,
    {
        let (reply, receive) = oneshot::channel();
        let job = VsearchJob {
            task: Box::new(task),
            reply,
        };
        match self
            .sender
            .as_ref()
            .ok_or("vsearch pool unavailable")?
            .try_send(job)
        {
            Ok(()) => Ok(receive),
            Err(std_mpsc::TrySendError::Full(_)) => Err("vsearch queue is full"),
            Err(std_mpsc::TrySendError::Disconnected(_)) => Err("vsearch pool unavailable"),
        }
    }
}

impl Drop for VsearchPool {
    fn drop(&mut self) {
        self.sender.take();
        for handle in self.handles.get_mut().drain(..) {
            let _ = handle.join();
        }
    }
}

// ── Vector payload sidecar ────────────────────────────────────────────────────
//
// An optional opaque payload blob per vector id, stored in the shard's KV vLog
// (not the index, so the quantized walk stays dense) for free crash-safety and
// recovery. The key is `tenant(16B LE) ++ marker(3B) ++ name ++ id(8B)`: the
// tenant prefix scopes the blob (A's is unreadable by B), the marker keeps it
// clear of user KV keys, and id-last makes the layout injective per (name, id).
/// Pre-generation marker. Read only: a store written before generations
/// existed keeps its blobs here, and the legacy generation is what makes them
/// findable.
const PAYLOAD_MARKER: &[u8; 3] = b"\x00vp";
/// Marker of a blob keyed by incarnation and row version.
const PAYLOAD_MARKER_V2: &[u8; 3] = b"\x00vq";
/// Payload durability is matched to the VECTOR it annotates. The vector lands in
/// the in-RAM delta + a raw (un-fsync'd) WAL append - i.e. `Relaxed` (survives a
/// process crash via the OS buffer; the durable checkpoint is `consolidate`). So
/// the payload uses `Relaxed` too: `Kernel` here would fsync per blob (a
/// device-wide barrier on macOS, ~7 ms) making the payload STRONGER than its own
/// vector and turning a 100k bulk load into ~13 min (31 s without it). Group
/// commit can't amortise it because VSETs serialise on the per-vindex lock.
///
/// # What that buys, exactly
///
/// A vector and its payload are ONE write, and the WAL record is its commit
/// point. The blob is staged first, at the key of the version the write is
/// about to take, which no live row carries - so it is unreachable until the
/// record lands, and reachable the instant it does.
///
/// **Process death** - a kill, a panic, an OOM - is therefore atomic on the
/// pair WITH NO FSYNC. Both halves are in the kernel; the record either
/// reached the file or it did not, and the blob cannot be read until it does.
/// A restart sees the old row with the old payload, or the new row with the
/// new payload, and never a mixture.
///
/// **Power loss** can reach exactly ONE skew, and it is the survivable one.
/// The device can commit neither half, the blob only, or both - "record only"
/// is not an ordering the write path can produce. So the worst a power cut
/// leaves is a vector whose payload blob did not survive: never a vector
/// wearing the payload of the value it replaced, and never a payload with no
/// vector. That row reads back with no payload, and the open-time reclamation
/// has nothing to collect for it.
///
/// See `docs/adr-payload-transaction.md`.
const PAYLOAD_DURABILITY: Durability = Durability::Relaxed;

/// Which INCARNATION of a vindex name a payload blob belongs to.
///
/// A vindex name is reusable: dropping `notes` and creating `notes` again is a
/// perfectly ordinary thing to do, and the blob keys of the two are otherwise
/// identical - same tenant, same name, same ids. The drop sweeps the old
/// blobs, but the sweep is best-effort and runs AFTER the catalogue has
/// already stopped naming the index, so a crash (or a failure) in between
/// leaves blobs that the next incarnation then serves as its own.
///
/// This is the fact that makes the two incarnations different things: minted
/// once by the coordinator at `VINDEX.CREATE`, broadcast to every shard,
/// persisted in the registry, and carried in every blob key.
///
/// [`LEGACY`](IndexGeneration::LEGACY) - zero - is what an index recorded by
/// the older `SVI2` registry reads as. Those keep the pre-generation blob key,
/// so a store written before this existed opens and answers exactly as it did.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct IndexGeneration(u128);

impl IndexGeneration {
    /// An index recorded before generations existed.
    pub const LEGACY: IndexGeneration = IndexGeneration(0);

    /// A generation from its raw value.
    #[must_use]
    pub const fn new(v: u128) -> Self {
        Self(v)
    }

    /// The raw value, for encoding.
    #[must_use]
    pub const fn get(self) -> u128 {
        self.0
    }

    /// True for an index that predates generations.
    #[must_use]
    pub const fn is_legacy(self) -> bool {
        self.0 == 0
    }
}

impl std::fmt::Display for IndexGeneration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "g{:032x}", self.0)
    }
}

/// A generation no earlier incarnation of any name can have used.
///
/// It has one job: differ from the generation of the index this name held
/// before, on this store. The wall clock in nanoseconds does that on its own
/// unless two creates land in the same nanosecond, and the hasher below covers
/// that - `RandomState` is seeded per process and stepped per instance, so two
/// creates in one nanosecond, in one process or in two, do not collide.
///
/// Not a UUID crate and not a counter. A counter would have to be persisted to
/// survive a restart, and a persisted counter that gets rolled back by a crash
/// hands the next incarnation the previous one's namespace - which is the
/// whole failure this exists to prevent.
fn mint_generation() -> IndexGeneration {
    use std::hash::{BuildHasher, Hasher};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0u128, |d| d.as_nanos());
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u64(SEQ.fetch_add(1, Ordering::Relaxed));
    h.write_u128(nanos);
    let v = ((nanos as u64) as u128) << 64 | u128::from(h.finish());
    // Zero is the legacy generation and means "recorded before generations
    // existed". A minted one must never be mistaken for it.
    IndexGeneration::new(if v == 0 { 1 } else { v })
}

/// The vLog key holding the payload blob of one COPY of one row:
/// `tenant(16) | marker(3) | generation(16) | name | id(8) | version(8)`.
///
/// Everything but the name is fixed-width, and the name sits between two
/// fixed-width runs, so the layout is injective - no pair of (generation,
/// name, id, version) can spell another pair's key.
///
/// Two facts are in here that the pre-generation key did not carry, and each
/// closes a hole:
///
/// - the GENERATION, so a recreated name does not inherit the blobs of the
///   index it replaced when the drop's sweep did not finish;
/// - the row VERSION, so staging a blob for a write that has not committed
///   yet cannot overwrite the blob of the value it is replacing. That is what
///   makes prepare-before-commit possible at all: the staged blob sits at a
///   key no live row names, and becomes the row's payload at the instant the
///   WAL record for that version lands.
///
/// A legacy-generation index writes here too. What it also does is READ the
/// pre-generation key when this one misses, which is how a store written
/// before any of this keeps answering (see [`read_payload_blob`]).
fn payload_blob_key(
    tenant: u128,
    generation: IndexGeneration,
    name: &str,
    id: u64,
    version: u64,
) -> Vec<u8> {
    let mut k = Vec::with_capacity(16 + PAYLOAD_MARKER_V2.len() + 16 + name.len() + 8 + 8);
    k.extend_from_slice(&tenant.to_le_bytes());
    k.extend_from_slice(PAYLOAD_MARKER_V2);
    k.extend_from_slice(&generation.get().to_le_bytes());
    k.extend_from_slice(name.as_bytes());
    k.extend_from_slice(&id.to_le_bytes());
    k.extend_from_slice(&version.to_le_bytes());
    k
}

/// Everything the blob key of one vindex is made of except the row.
///
/// Passed around rather than three loose arguments because getting one of them
/// wrong produces a key that reads back empty rather than an error, and an
/// empty payload is exactly the failure these commits exist to remove.
#[derive(Clone, Copy)]
struct BlobScope<'a> {
    tenant: u128,
    generation: IndexGeneration,
    name: &'a str,
}

impl BlobScope<'_> {
    fn key(self, id: u64, version: u64) -> Vec<u8> {
        payload_blob_key(self.tenant, self.generation, self.name, id, version)
    }

    /// The pre-generation key, for an index that predates generations. `None`
    /// for every index created since, which therefore never pays the second
    /// lookup.
    fn legacy_key(self, id: u64) -> Option<Vec<u8>> {
        self.generation
            .is_legacy()
            .then(|| payload_key(self.tenant, self.name, id))
    }
}

/// Read the payload blob of one copy of one row.
///
/// The fallback is the whole compatibility story: an index recorded by the
/// older registry reads as the legacy generation, and its blobs are wherever
/// the previous version left them. An index created since never takes the
/// second lookup, so the common path is one read.
async fn read_payload_blob(
    vlog: &VLog,
    scope: BlobScope<'_>,
    id: u64,
    version: u64,
) -> Result<Option<Bytes>, skeg_core::Error> {
    let store = vlog.tenant(scope.tenant);
    if let Some(blob) = store.get(&scope.key(id, version)).await? {
        return Ok(Some(blob));
    }
    match scope.legacy_key(id) {
        Some(key) => store.get(&key).await,
        None => Ok(None),
    }
}

fn payload_key(tenant: u128, name: &str, id: u64) -> Vec<u8> {
    let mut k = Vec::with_capacity(16 + 3 + name.len() + 8);
    k.extend_from_slice(&tenant.to_le_bytes());
    k.extend_from_slice(PAYLOAD_MARKER);
    k.extend_from_slice(name.as_bytes());
    k.extend_from_slice(&id.to_le_bytes());
    k
}

/// Rebuild a recovered vindex's payload index from its stored blobs, once, the
/// first time a filtered search needs it. Idempotent and tenant-scoped: the
/// query's tenant locates the blobs (a real vindex is single-tenant). Runs the
/// blob reads outside the lock, then populates under it.
async fn ensure_payload_loaded(
    vlog: &VLog,
    vdir: &Path,
    arc: &VectorEntry,
    tenant: u128,
    name: &str,
    allow_cache: bool,
) -> Result<bool, String> {
    let (scope_generation, ids) = {
        let g = arc.read();
        if g.payload_loaded {
            return Ok(false);
        }
        (
            g.generation,
            g.backend
                .live_ids_with_versions()
                .into_iter()
                .map(|(id, v)| (id, v.get()))
                .collect::<Vec<_>>(),
        )
    };
    let scope = BlobScope {
        tenant,
        generation: scope_generation,
        name,
    };

    // The fast path: the index itself, read back from `payload.idx`. Only the
    // ids the log tail touched since the file was stamped are re-read, because
    // only those can have changed; everything else is answered straight from
    // the file, which is why this does not put the corpus back on the heap.
    if allow_cache
        && let Some(rec) = vlog.recovered_from()
        && let Some(disk) =
            crate::payload_disk::DiskPostings::open(vdir, rec.stamp, scope.generation.get())
    {
        let covered = disk.len();
        let mut payload = PayloadIndex::from_disk(disk);
        let mut refreshed = 0usize;
        for &(id, version) in &ids {
            // The keys the row could be under: its own, and - for an index
            // that predates generations - the pre-generation one. A tail that
            // touched EITHER is a tail that changed this row.
            let keys: Vec<Vec<u8>> = std::iter::once(scope.key(id, version))
                .chain(scope.legacy_key(id))
                .collect();
            if !keys.iter().any(|k| rec.tail_keys.contains(k)) {
                continue;
            }
            refreshed += 1;
            let mut found = None;
            for key in &keys {
                match vlog.get_uncached(key).await {
                    Ok(Some(blob)) => {
                        found = Some(blob);
                        break;
                    }
                    Ok(None) => {}
                    Err(e) => return Err(format!("payload index rebuild failed: {e}")),
                }
            }
            match found {
                Some(blob) => payload.upsert(id, parse_fields(&blob)),
                // Written and then deleted after the stamp: the file may still
                // list it, so say so explicitly rather than leaving it there.
                None => payload.remove(id),
            }
        }
        let mut g = arc.write();
        if !g.payload_loaded {
            g.payload = payload;
            g.payload_loaded = true;
            skeg_telemetry::add_counter(
                skeg_telemetry::Counter::PayloadIndexFromDisk,
                (covered.saturating_sub(refreshed)) as u64,
            );
            skeg_telemetry::add_counter(
                skeg_telemetry::Counter::PayloadIndexRefreshed,
                refreshed as u64,
            );
            tracing::info!(
                "payload index for '{name}' read from disk: {covered} ids, {refreshed} re-read from the log"
            );
        }
        return Ok(false);
    }

    // The persisted cache, when the store recovered from the snapshot it was
    // stamped with. `cached` holds only ids the replayed tail did not touch:
    // anything the tail wrote changed after the stamp, so its cached blob is a
    // dead value and has to come from the vlog. Both conditions must hold; a
    // filtered search served from a stale payload index drops results with no
    // error anywhere, which is far worse than a slow open.
    // Refusing the file is always safe, so this is not an error, but reading
    // the whole log back while a payload.idx sits there unused is the kind of
    // thing that goes unnoticed for months. Absent is normal and says nothing.
    if allow_cache && vdir.join(crate::payload_disk::FILE).exists() {
        tracing::warn!(
            "payload index for '{name}' refused (stale stamp, or damaged); \
             rebuilding from the log"
        );
    }

    // Reads in flight, because this path is latency-bound whenever the store is
    // cold.
    //
    // An earlier version read them one at a time, on a measurement that said
    // serial was faster. That measurement was taken in the wrong regime: with
    // the payloads already in page cache a read costs ~1 us, the device is
    // never waited on, and concurrency only adds scheduling. On a cold store
    // the same read costs ~280 us, 212 times more, and a serial loop leaves the
    // disk idle between every one of them. The reads are independent either
    // way, so serialising them buys nothing but that idle time.
    //
    // `get_uncached`: this reads every blob exactly once and keeps the parsed
    // fields in the payload index, so caching the blobs would store a second
    // copy of data we already hold, and past the cache's byte budget it would
    // evict whatever was genuinely hot to do it.
    let mut stream = futures_util::stream::iter(ids.into_iter().map(|(id, version)| {
        let key = scope.key(id, version);
        let legacy = scope.legacy_key(id);
        async move {
            match vlog.get_uncached(&key).await {
                Ok(Some(blob)) => Ok((id, Some(blob))),
                Ok(None) => match legacy {
                    Some(k) => vlog.get_uncached(&k).await.map(|blob| (id, blob)),
                    None => Ok((id, None)),
                },
                Err(e) => Err(e),
            }
        }
    }))
    .buffer_unordered(PAYLOAD_READ_CONCURRENCY);
    let mut parsed = Vec::new();
    while let Some(next) = futures_util::StreamExt::next(&mut stream).await {
        match next {
            Ok((id, Some(blob))) => parsed.push((id, parse_fields(&blob))),
            Ok((_, None)) => {}
            Err(e) => return Err(format!("payload index rebuild failed: {e}")),
        }
    }

    let mut g = arc.write();
    // Re-check under the write lock: another task may have loaded it meanwhile.
    if !g.payload_loaded {
        for (id, fields) in parsed {
            g.payload.upsert(id, fields);
        }
        g.payload_loaded = true;
        skeg_telemetry::tick_counter(skeg_telemetry::Counter::PayloadIndexRebuilds);
    }
    Ok(true)
}

/// Snapshot the vindex map as owned handles.
///
/// Both callers below await while working on each vindex, and the map lives
/// behind a lock that must not be held across an await. The clone is a name
/// and an `Arc` per vindex, taken once.
fn vindex_handles(vindexes: &RwLock<VindexSet>) -> Vec<(String, VectorEntry)> {
    vindexes
        .read()
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

/// Write the vlog snapshot, then a payload index per vindex stamped with the
/// position that snapshot covers.
///
/// The two must carry the same stamp, which is why they are written together
/// here rather than on their own schedules: the stamp is the only thing that
/// tells the next open whether the cached payloads still describe the log it
/// recovered from.
///
/// Writing the cache is best effort. It is an optimisation for the next open,
/// so a failure is logged and the store carries on; the worst outcome is a slow
/// warm, and the snapshot itself has already succeeded by then.
async fn snapshot_and_payload_indexes(
    vlog: &VLog,
    vindexes: &RwLock<VindexSet>,
    dir: &Path,
    shard_id: usize,
    written: &mut HashMap<String, (u64, u64)>,
) {
    let stamp = match vlog.write_snapshot().await {
        Ok(s) => s,
        Err(e) => {
            error!("shard {shard_id}: snapshot failed: {e}");
            return;
        }
    };
    for (scoped, arc) in vindex_handles(vindexes) {
        // Built from the in-memory index, not by reading the blobs back. A
        // rebuild-from-log here would repeat the whole open-time read storm on
        // a server that is serving traffic, every snapshot interval, which is
        // a worse problem than the one this file solves. The index already
        // holds everything, and `field_blob` returns it in the form the parser
        // accepts.
        //
        // An index that was never loaded has nothing to persist, and writing
        // an empty file for it would look like "this vindex has no payloads"
        // at the next open. Skip it instead.
        // Nothing was appended since the last write, so the stamp is the same
        // and the file would come out byte for byte identical. On a store that
        // is only being read, snapshots keep firing and rewriting hundreds of
        // megabytes for no reason; the stamp is exactly the signal that says
        // so, because it only moves when the log does.
        if written.get(&scoped) == Some(&stamp) {
            continue;
        }
        let vdir = dir.join(format!("vindex-{scoped}"));
        let result = {
            let g = arc.read();
            if !g.payload_loaded {
                continue;
            }
            g.payload.persist(&vdir, stamp, g.generation.get())
        };
        match result {
            Ok(()) => {
                written.insert(scoped.clone(), stamp);
            }
            Err(e) => {
                tracing::warn!(
                    "shard {shard_id}: writing payload index for '{scoped}' failed: {e}"
                );
            }
        }
    }
}

/// Load every recovered vindex's payload index before the shard reports ready.
///
/// The rebuild reads each live id's payload blob, and deferring it to the first
/// filtered search put the whole cost on one user's query: measured 5.2s
/// against 6ms for every search after it, on 223k vectors. The readiness
/// barrier already waits for recovery, so this belongs there.
///
/// Best effort by design: a vindex that cannot be warmed is logged and left
/// alone, and it still loads lazily on its first filtered search. Readiness
/// must not hinge on an optimisation.
/// Returns whether any vindex had to be rebuilt from the log rather than read
/// from its file, which is what makes persisting it straight afterwards worth
/// the write.
async fn warm_payload_indexes(vlog: &VLog, vindexes: &RwLock<VindexSet>, dir: &Path) -> bool {
    let mut rebuilt_any = false;
    for (scoped, arc) in vindex_handles(vindexes) {
        // The scoped name, not the bare index name: the query path builds its
        // payload keys from what `get_or_reopen` was given, which is scoped.
        // Warming with the bare name would build different keys, find nothing,
        // and still mark the index loaded, leaving every filtered search for
        // that tenant silently empty. Only tenant 0 makes the two the same,
        // which is exactly why this hid.
        let (tenant, _) = unscope_key(&scoped);
        let n = arc.read().backend.live_ids().len();
        let t0 = std::time::Instant::now();
        let vdir = dir.join(format!("vindex-{scoped}"));
        // `true` only here. This runs inside the readiness barrier, before the
        // shard has served anything, so the store is still exactly where
        // recovery left it and `tail_keys` describes every change since the
        // stamp. A vindex reopened later, after an eviction, is a different
        // situation: hours of writes may have landed that no tail records, so
        // it rebuilds from the log and the cache is not consulted.
        match ensure_payload_loaded(vlog, &vdir, &arc, tenant, &scoped, true).await {
            Ok(rebuilt) => rebuilt_any |= rebuilt,
            Err(e) => {
                tracing::warn!("warming payload index for vindex '{scoped}' failed: {e}");
                continue;
            }
        }
        // Logged because this is now a visible share of open time, and an
        // operator staring at a slow start should not have to guess which
        // part of it is this.
        tracing::info!(
            "warmed payload index for vindex '{scoped}': {n} vectors in {:?}",
            t0.elapsed()
        );
    }
    rebuilt_any
}

/// Run a VSEARCH against one vindex: exact brute-force over a filter's matching
/// ids, or the ANN walk when there is no filter. Shared by the inline and
/// worker-pool paths; the caller holds the vindex write lock.
fn search_vindex(
    idx: &mut Vindex,
    name: &str,
    query: &[f32],
    k: usize,
    l_search: u32,
    filter: &Option<Filter>,
) -> Result<Vec<(u64, f32)>, String> {
    if idx.backend.dim() != query.len() {
        return Err(format!(
            "vindex '{name}' dim {} but query has {}",
            idx.backend.dim(),
            query.len()
        ));
    }
    if let Some(f) = filter {
        let s = f.evaluate(&idx.payload);
        idx.backend
            .filtered_search(query, k, l_search, &s)
            .map_err(|e| format!("vsearch failed: {e}"))
    } else {
        idx.backend
            .search(query, k, l_search)
            .map_err(|e| format!("vsearch failed: {e}"))
    }
}

/// Attach each hit's stored payload when `want_payload`, else leave it `None`.
/// Shared by the inline and worker-pool VSEARCH paths. Payloads ride as `Bytes`
/// (refcounted vLog reads) so no blob is copied on the way to the response.
async fn attach_payloads(
    vlog: &VLog,
    scope: BlobScope<'_>,
    hits: Vec<(u64, f32, u64)>,
    want_payload: bool,
) -> Result<Vec<(u64, f32, Option<Bytes>)>, String> {
    let mut out = Vec::with_capacity(hits.len());
    for (id, score, version) in hits {
        let blob = if want_payload {
            read_payload_blob(vlog, scope, id, version)
                .await
                .map_err(|e| format!("vsearch payload failed: {e}"))?
        } else {
            None
        };
        out.push((id, score, blob));
    }
    Ok(out)
}

// ── VINDEX registry (disk-backed indexes survive a restart) ───────────────────
//
// Each shard records its disk-backed VINDEXes in `vindexes.registry`; on
// startup it reopens them from their `vindex-<name>/` directories. Flat
// indexes are in-RAM and ephemeral by design - they are not registered.

const VINDEX_REGISTRY: &str = "vindexes.registry";
const VINDEX_REGISTRY_V2_MAGIC: [u8; 4] = *b"SVI2";
/// V3 adds the 16-byte [`IndexGeneration`] to every record.
const VINDEX_REGISTRY_V3_MAGIC: [u8; 4] = *b"SVI3";

/// The most vindexes one shard will hold.
///
/// A bound on the catalogue, not a capacity plan. It exists so the registry
/// has a size the reader can state up front, and so a shard cannot be talked
/// into an unbounded one.
pub(crate) const MAX_VINDEXES_PER_SHARD: usize = 1024;

/// The widest a single registry record can be: a 2-byte name length, a name
/// at the 255-byte cap `validate_vindex_name` enforces, a 4-byte dim, a
/// 1-byte tier and a 16-byte generation.
const MAX_REGISTRY_RECORD: usize = 2 + 255 + 4 + 1 + 16;

/// What the registry may weigh: magic, count, and every record at its widest.
///
/// Derived from the two numbers above rather than picked, because the reader
/// and the writer MUST agree. They did not: the reader used the 4 KiB sidecar
/// bound while the writer had none, so sixteen indexes with maximum-length
/// names (8 + 16 x 262 = 4,200 bytes) wrote successfully and then refused to
/// be read back on the next open. A limit only one side knows is not a limit.
const MAX_REGISTRY_BYTES: u64 = (8 + MAX_VINDEXES_PER_SHARD * MAX_REGISTRY_RECORD) as u64;

#[derive(Clone, Debug)]
struct RegistryEntry {
    name: String,
    dim: usize,
    /// Effective VINDEX wire kind. `None` denotes the legacy registry format,
    /// which did not persist a per-index tier.
    kind: Option<u8>,
    /// Which incarnation of this name the entry records.
    /// [`IndexGeneration::LEGACY`] for a `SVI2` (or older) file, which
    /// recorded no such thing.
    generation: IndexGeneration,
}

/// Rewrite the versioned registry: `[SVI3][u32 count]` then
/// `[u16 nlen][name][u32 dim][u8 kind][u128 generation]` per disk-backed
/// VINDEX.
///
/// The generation is the LAST field of the record on purpose: an older reader
/// would still walk the fields before it correctly, so the only thing that
/// stops it opening this file is the magic - a refusal, not a misparse.
#[allow(clippy::cast_possible_truncation)] // index names are short, dims fit u32
fn write_registry(
    dir: &Path,
    entries: &[(&str, usize, u8, IndexGeneration)],
) -> std::io::Result<()> {
    let bad = |msg: String| std::io::Error::new(std::io::ErrorKind::InvalidInput, msg);
    // Refuse BEFORE publishing. The reader enforces the same bound, and a file
    // the writer is willing to produce but the reader will not accept is the
    // worst of both worlds: the write succeeds and the next open fails.
    if entries.len() > MAX_VINDEXES_PER_SHARD {
        return Err(bad(format!(
            "{} vindexes exceeds the {MAX_VINDEXES_PER_SHARD} a shard holds",
            entries.len()
        )));
    }
    let mut buf = Vec::new();
    buf.extend_from_slice(&VINDEX_REGISTRY_V3_MAGIC);
    let count =
        u32::try_from(entries.len()).map_err(|_| bad("entry count overflows u32".to_owned()))?;
    buf.extend_from_slice(&count.to_le_bytes());
    for (name, dim, kind, generation) in entries {
        // Checked, not `as`: a silent truncation writes a length that does not
        // match the bytes beside it, and the reader then walks off into the
        // next record.
        let nlen = u16::try_from(name.len())
            .map_err(|_| bad(format!("vindex name is {} bytes, too long", name.len())))?;
        let dim32 =
            u32::try_from(*dim).map_err(|_| bad(format!("dim {dim} overflows its field")))?;
        buf.extend_from_slice(&nlen.to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(&dim32.to_le_bytes());
        buf.push(*kind);
        buf.extend_from_slice(&generation.get().to_le_bytes());
    }
    if buf.len() as u64 > MAX_REGISTRY_BYTES {
        return Err(bad(format!(
            "registry would be {} bytes, over the {MAX_REGISTRY_BYTES} the \
             reader accepts",
            buf.len()
        )));
    }
    // Durable publish, the same discipline as the layout manifest and the
    // generation pointer: write, fsync the file, rename, fsync the directory.
    // A rename whose directory entry has not reached the disk can vanish on
    // reboot, taking the registry - and therefore every index it lists - with
    // it, while the vindex directories sit there unreferenced.
    let tmp = dir.join(format!("{VINDEX_REGISTRY}.tmp"));
    {
        use std::io::Write;
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&buf)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, dir.join(VINDEX_REGISTRY))?;
    skeg_platform::sync_dir(dir)
}

/// Read the registry of on-disk vindexes.
///
/// An ABSENT registry is an empty store - the ordinary state of a fresh data
/// directory. Everything else that does not parse exactly is an error.
///
/// It used to return "whatever parsed cleanly", which means indexes silently
/// DISAPPEAR: three of seven entries read back as three indexes, with full
/// confidence and no warning. That is the serve-mode failure in a different
/// costume - partial data accepted as complete - and the fix is the same one:
/// refuse rather than guess.
///
/// `kind: None` still means the old `[u32 count]` format, which recorded no
/// tier and legitimately takes the caller's default. A V2 entry naming a wire
/// kind this build does not know is NOT that: it opens the index with the
/// process default quantiser, computing every distance against codes it
/// cannot interpret. Absent and unreadable are different states and only one
/// of them has a safe default.
fn read_registry(dir: &Path) -> std::io::Result<Vec<RegistryEntry>> {
    let path = dir.join(VINDEX_REGISTRY);
    let bytes = match skeg_platform::read_bounded(&path, MAX_REGISTRY_BYTES) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    let bad = |msg: String| std::io::Error::new(std::io::ErrorKind::InvalidData, msg);

    // Three shapes, oldest last: V3 (tier + generation), V2 (tier), and the
    // original headerless count. All three are READ; only V3 is written.
    let (mut pos, versioned, generational) = if bytes.starts_with(&VINDEX_REGISTRY_V3_MAGIC) {
        (4usize, true, true)
    } else if bytes.starts_with(&VINDEX_REGISTRY_V2_MAGIC) {
        (4usize, true, false)
    } else {
        (0usize, false, false)
    };
    if bytes.len() < pos + 4 {
        return Err(bad(format!(
            "{} is {} bytes: too short to hold an entry count",
            path.display(),
            bytes.len()
        )));
    }
    let count = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
    // A corrupt count must not drive a huge reservation: every entry is at
    // least a 2-byte length prefix, so it cannot exceed bytes.len()/2. The
    // loop bounds-checks each entry regardless.
    let mut out = Vec::with_capacity(count.min(bytes.len() / 2));
    pos += 4;
    for i in 0..count {
        let short = || {
            bad(format!(
                "{} claims {count} entries but runs out during entry {i}: \
                 refusing to report a truncated registry as a complete one",
                path.display()
            ))
        };
        if pos + 2 > bytes.len() {
            return Err(short());
        }
        let nlen = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]) as usize;
        pos += 2;
        let tail = 4 + usize::from(versioned) + if generational { 16 } else { 0 };
        if pos + nlen + tail > bytes.len() {
            return Err(short());
        }
        // NOT `from_utf8_lossy`: replacing corrupt bytes with U+FFFD produces a
        // name that no longer matches the directory on disk, so the index
        // quietly cannot be reopened.
        let name = std::str::from_utf8(&bytes[pos..pos + nlen])
            .map_err(|e| {
                bad(format!(
                    "{} entry {i} has a name that is not UTF-8: {e}",
                    path.display()
                ))
            })?
            .to_owned();
        // Valid UTF-8 is not a valid NAME. This string is joined onto the data
        // directory to build a path, and it arrives from a file: `x/../../elsewhere`
        // is perfectly good UTF-8. The protocol already refuses such names on
        // the way in, so the registry refuses them on the way out - one rule,
        // both directions.
        validate_vindex_name(&name).map_err(|e| {
            bad(format!(
                "{} entry {i} has a name this build will not use as a path: {e:?}",
                path.display()
            ))
        })?;
        // And a valid name is not a valid KEY. Everything downstream asks
        // `unscope_key` who owns this index - `EraseTenant` to decide what to
        // destroy, the warm-up to decide whose blobs to read - so a key whose
        // owner cannot be read back unambiguously must not be served under a
        // guess. `scope_key(unscope_key(k))` is the server's own round trip:
        // whatever it does not reproduce, the server did not write. Fail
        // closed and name the key, because the alternative is a silent
        // misattribution to whichever tenant the string happens to spell.
        let (tenant, index) = unscope_key(&name);
        if scope_key(tenant, &index) != name {
            return Err(bad(format!(
                "{} entry {i} is keyed '{name}', which this build cannot \
                 attribute to a tenant: it does not survive the scope round \
                 trip (read back as tenant {tenant}, index '{index}'). \
                 Refusing to open it under a guessed owner.",
                path.display()
            )));
        }
        pos += nlen;
        let dim = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let kind = if versioned {
            let k = bytes[pos];
            pos += 1;
            if QuantKind::from_wire(k).is_none() {
                return Err(bad(format!(
                    "{} entry {i} ({name}) names tier byte {k}, which this \
                     build does not know: refusing to open it with a different \
                     quantiser",
                    path.display()
                )));
            }
            Some(k)
        } else {
            None
        };
        // An index recorded before generations existed HAS no incarnation to
        // name, and inventing one here would move its blobs out from under it.
        // Legacy is not a default standing in for a missing value: it IS the
        // value, and it is what makes the pre-generation blob key readable.
        let generation = if generational {
            let g = IndexGeneration::new(u128::from_le_bytes(
                bytes[pos..pos + 16].try_into().unwrap(),
            ));
            pos += 16;
            g
        } else {
            IndexGeneration::LEGACY
        };
        out.push(RegistryEntry {
            name,
            dim,
            kind,
            generation,
        });
    }
    if pos != bytes.len() {
        return Err(bad(format!(
            "{} has {} bytes after its {count} entries: either the count or \
             the file is wrong",
            path.display(),
            bytes.len() - pos
        )));
    }
    Ok(out)
}

/// Rewrite the registry (best-effort).
///
/// The registry tracks every on-disk vindex (the `vindex-<name>/` dirs), NOT
/// only the resident ones. An evicted vindex is gone from the map but its dir
/// stays; rebuilding the registry purely from the map would drop it and break
/// lazy reopen + restart recovery. So: start from the existing registry, fold
/// in the resident disk vindexes, then keep only entries whose dir still
/// exists: create adds (dir + map), evict keeps (dir, not in map).
///
/// A DROP is NOT inferred from the filesystem. It used to be - "drop prunes
/// (no dir)" - and that only worked while the directory was deleted BEFORE
/// the publish. Committing first inverted it: at publish time the directory
/// is still there, so the entry survived, the commit republished the index it
/// was meant to remove, and the following delete left the registry naming
/// something gone. The next start then refused to open, which is precisely
/// the failure commit-first exists to prevent.
///
/// So a removal is stated, not deduced.
fn persist_registry(dir: &Path, vindexes: &RwLock<VindexSet>) -> std::io::Result<()> {
    persist_registry_removing(dir, vindexes, None)
}

/// [`persist_registry`], with `removing` naming an entry that must be dropped
/// whether or not its directory is still on disk.
fn persist_registry_removing(
    dir: &Path,
    vindexes: &RwLock<VindexSet>,
    removing: Option<&str>,
) -> std::io::Result<()> {
    use std::collections::BTreeMap;
    // A registry that will not parse must NOT be rewritten from the resident
    // map alone. The map holds only what is currently open, so the rewrite
    // would drop every evicted vindex - turning an unreadable file into a
    // permanently and silently smaller store. Leave the file for repair and
    // say so; the next successful read picks the work back up.
    let existing = match read_registry(dir) {
        Ok(entries) => entries,
        Err(e) => {
            tracing::error!(
                dir = %dir.display(),
                error = %e,
                "refusing to rewrite an unreadable vindex registry: rebuilding \
                 it from the resident set would drop every evicted index"
            );
            return Err(e);
        }
    };
    let mut by_name: BTreeMap<String, (usize, u8, IndexGeneration)> = existing
        .into_iter()
        .map(|entry| {
            (
                entry.name,
                (entry.dim, entry.kind.unwrap_or(1), entry.generation),
            )
        })
        .collect();
    {
        let vs = vindexes.read();
        for (name, entry) in vs.iter() {
            let vindex = entry.read();
            if let VectorBackend::Disk(i) = &vindex.backend {
                by_name.insert(name.clone(), (i.dim(), vindex.kind, vindex.generation));
            }
        }
    }
    // The explicit removal first: the resident-map fold above cannot express
    // it (the index is already out of the map, so it simply is not re-added),
    // and the directory test below cannot either (the directory is still
    // there, deleted only after this commit lands).
    if let Some(gone) = removing {
        by_name.remove(gone);
    }
    by_name.retain(|name, _| dir.join(format!("vindex-{name}")).exists());
    let entries: Vec<(&str, usize, u8, IndexGeneration)> = by_name
        .iter()
        .map(|(name, (dim, kind, generation))| (name.as_str(), *dim, *kind, *generation))
        .collect();
    write_registry(dir, &entries)
}

/// Reopen the disk-backed VINDEXes recorded in the registry, with the given
/// tier-1 quantisation (`Int8` for the read-write path, configurable in serve
/// mode). `mmap_tier` swaps the TurboQuant codes for a memory-mapped view
/// (`--tier-mmap`); `mmap_graph` swaps the graph Node array for a mmap'd
/// view of `graph.vmn` (`--graph-mmap`). Other tiers are unaffected.
/// Run one off-thread maintenance op on a vindex: `begin` under a short write
/// lock, `build` on the blocking pool with NO lock held (so queries to this
/// vindex do not stall for the whole rebuild), `finish` under a short write
/// lock. `begin` returning `None` means nothing to do. Best-effort: errors are
/// logged, not propagated. Returns true if an op actually ran.
///
/// No lock is held across the `.await` (the guards are confined to the sync
/// blocks), so the shard thread is free during the build.
/// One maintenance decision for one vindex, off the request path.
///
/// Priority, cheapest first, and the order is the whole point:
///   1. flush (L0), delta past `FLUSH_ROWS`. The common case under ingest; it
///      keeps the flat delta scan small and builds no graph on the shard thread.
///   2. delete-patch (L3), dead base rows past the tombstone threshold. Reuses
///      the surviving edges; measured 10,6x the fold at 1% dead, 4,5x at 3%.
///   3. runs-merge (L2), runs piling up while the base is fine. O(runs), the
///      base is never touched.
///   4. consolidate, only when the runs have really grown to base size. Folds
///      everything into a fresh base and truncates the WAL. This is the one
///      that costs O(live), so it goes last.
///
/// The fold used to be checked second, which meant that whenever it was due the
/// two cheap paths never got a turn. It also had an idle clause, so a store
/// going quiet with 4096 pending rows rebuilt itself entirely; a quiet store
/// does not need that, it needs its runs not to pile up, which is (3).
///
/// The fold stays for the regime the verdicts measured it winning in: above
/// roughly a quarter of the base dead, delete-patch loses (0,3x at 40%), and
/// once the runs have grown to base size there is nothing left to reuse.
///
/// One operation per vindex per tick. Returns whether a consolidate ran, which
/// is what tells the caller to reset this vindex's idle tracking.
///
/// This is a named function rather than a block inside the idle loop because
/// it decides flush against consolidate against merge against patch, which
/// makes it the code most likely to be read during a memory or latency
/// incident. Inlined, it appeared in traces as `run_shard::{{closure}}::{{closure}}`.
async fn maintenance_tick(arc: &VectorEntry, vdir: &Path, shard_id: usize, idle: bool) -> bool {
    maintenance_tick_at(arc, vdir, shard_id, idle, FLUSH_ROWS).await
}

/// The LSM state one tick decides from, read once under one lock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LsmState {
    delta: usize,
    runs: usize,
    run_rows: usize,
    tombs: usize,
    base: usize,
    /// Consecutive ticks a due merge has lost to the flush.
    flush_streak: u64,
}

/// One rung of the maintenance ladder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rung {
    RunsMerge,
    Flush,
    DeletePatch,
    Consolidate,
}

impl LsmState {
    /// Count OR mass. The count alone left a single fat dirty run untouched
    /// forever - one run is not "four runs" however much garbage it holds. A
    /// lone run is only OFFERED; whether it is dirty enough to rewrite is
    /// decided in `merge_runs_begin`, from the survivor set, where the number
    /// is exact.
    fn merge_due(&self) -> bool {
        self.runs >= RUNS_MERGE_TRIGGER || (self.runs >= 1 && self.tombs > 0)
    }

    /// Past roughly a quarter of the base dead, delete-patch is in its
    /// measured losing regime (0.3x at 40%), so a heavy-dead base goes
    /// straight to the full fold. Without this arm nothing would ever reclaim
    /// a heavily-tombstoned base: the geometric trigger only watches runs.
    fn heavy_dead(&self) -> bool {
        self.base >= DELETE_PATCH_MIN_BASE && self.tombs * 4 > self.base
    }

    fn consolidate_due(&self) -> bool {
        self.run_rows >= self.base.max(IDLE_CONSOLIDATE_MIN) || self.heavy_dead()
    }

    fn delete_patch_due(&self) -> bool {
        self.base >= DELETE_PATCH_MIN_BASE
            && self.tombs >= self.base / DELETE_PATCH_DEAD_DIVISOR
            && self.tombs * 4 <= self.base
    }
}

/// The rungs due this tick, in the order to attempt them. PURE: no I/O, no
/// lock, no clock.
///
/// Separated from the doing for two reasons, both paid for.
///
/// The decision is what the tests care about, and testing it through real
/// maintenance walks ONE trajectory through the state space at the cost of a
/// graph build per step. Against a function the space itself can be swept.
///
/// And a list cannot repeat itself by accident. The ladder used to reach
/// `runs-merge` from two places: once as the anti-starvation attempt, and
/// again further down where `runs >= RUNS_MERGE_TRIGGER` was still true. The
/// second was reachable ONLY as a duplicate - `runs >= RUNS_MERGE_TRIGGER`
/// implies `merge_due`, and getting past the flush rung requires `!flush_due`,
/// which is exactly when the first attempt already fired. So it re-ran an
/// operation that had just declined, and that refusal costs a full survivor
/// set - one hash insert per row of every run - under the write lock.
///
/// Cheap first, expensive last. The fold rebuilds the whole base and costs
/// O(live); the other three are proportional to what changed. Checking the
/// fold first meant that whenever it was due the cheap paths never got a turn,
/// which is why the project's own note says the delete-patch plus runs-merge
/// cycle was never closed as a replacement for it.
fn ladder_plan(s: &LsmState, flush_rows: usize) -> Vec<Rung> {
    let mut plan = Vec::with_capacity(2);
    let flush_due = s.delta >= flush_rows;
    let merge_due = s.merge_due();

    // Flush and merge ALTERNATE when both are due.
    //
    // Every rung ends the tick, so a permanently hot rung starves every rung
    // below it. Under sustained write churn the delta is ALWAYS over the flush
    // threshold, so the tick took the flush branch forever - and each flush
    // adds a run. Measured on the churn gate: runs climbed 0 -> 33 over ten
    // turnovers and never merged, and recall fell 0.9925 -> 0.7180 alongside.
    //
    // The MECHANISM of that fall was never isolated, and this comment used to
    // assert one: that run graphs are walked with a short beam. Two defects found
    // later explain it at least as well - unfiltered search scored candidates
    // against SUPERSEDED vectors, so a shortlist drawn from many runs was
    // contaminated by stale copies. Run debt certainly exposed the defect;
    // calling it the cause is a claim nobody measured. What the gate does show
    // is the correlation and that the fix holds, and the store stayed correct
    // throughout (0/120 stale) - starvation, not corruption.
    //
    // A ceiling alone was NOT enough, and the measurement said so: run counts
    // an operator reads are SUMMED across shards, so the "33 runs" were four
    // per shard - under any per-shard ceiling worth having, and already at the
    // ordinary merge trigger. The rung was due every tick and lost every tick.
    //
    // So the flush still wins normally, but it cannot win TWICE in a row while
    // a merge is due. Under sustained churn that gives the merge every other
    // tick, which keeps the count flat, and needs no threshold guessed right.
    let starved = s.flush_streak >= 1;
    if merge_due && (!flush_due || starved) {
        plan.push(Rung::RunsMerge);
    }
    if flush_due {
        plan.push(Rung::Flush);
    }
    if s.delete_patch_due() {
        plan.push(Rung::DeletePatch);
    }
    if s.consolidate_due() {
        plan.push(Rung::Consolidate);
    }
    plan
}

/// The tick with the flush threshold as a parameter.
///
/// The invariant worth protecting here is the LADDER'S CHOICE, not the work
/// it schedules - and driving real maintenance past a 4096-row threshold
/// twenty times takes half an hour, which is how the starvation test ended up
/// too slow to run and therefore protecting nothing. With the threshold as an
/// argument the same decision sequence is exercised in seconds.
async fn maintenance_tick_at(
    arc: &VectorEntry,
    vdir: &Path,
    shard_id: usize,
    idle: bool,
    flush_rows: usize,
) -> bool {
    let state = {
        let g = arc.read();
        LsmState {
            delta: g.backend.delta_len(),
            runs: g.backend.run_count(),
            run_rows: g.backend.run_rows(),
            tombs: g.backend.tombstone_count(),
            base: g.backend.main_len(),
            flush_streak: g.flush_streak.load(Ordering::Relaxed),
        }
    };
    let merge_due = state.merge_due();

    for rung in ladder_plan(&state, flush_rows) {
        let d = vdir.to_path_buf();
        match rung {
            Rung::RunsMerge => {
                let outcome = off_thread_maintenance(
                    arc,
                    "runs-merge",
                    shard_id,
                    |b| b.merge_runs_begin(),
                    move |job| job.build(&d),
                    |b, built| b.merge_runs_finish(built),
                )
                .await;
                // ONLY a merge that actually ran consumes the tick.
                //
                // `NotNeeded` means the exact check said there is nothing worth
                // rewriting - a clean run offered by a coarse trigger - and
                // treating that as work done left a run past the consolidate
                // threshold sitting there tick after tick. `BudgetBusy` and
                // `Failed` must not stop the flush either: that would trade the
                // starvation this removes for the one below it.
                if outcome == MaintenanceOutcome::Ran {
                    arc.read().flush_streak.store(0, Ordering::Relaxed);
                    return false;
                }
            }
            Rung::Flush => {
                // WITH the abort: a flush that fails before its commit has
                // already moved the delta into the `flushing` staging buffer,
                // and without giving it back the rows are searched but never
                // re-flushed. The other rungs have nothing to hand back.
                off_thread_maintenance_with_abort(
                    arc,
                    "flush",
                    shard_id,
                    |b| b.flush_begin(),
                    move |job| job.build(&d),
                    |b, built| b.flush_finish(built),
                    VectorBackend::flush_abort,
                )
                .await;
                if merge_due {
                    arc.read().flush_streak.fetch_add(1, Ordering::Relaxed);
                }
                return false;
            }
            Rung::DeletePatch => {
                off_thread_maintenance(
                    arc,
                    "delete-patch",
                    shard_id,
                    |b| b.delete_patch_begin(),
                    move |job| job.build(&d),
                    |b, built| b.delete_patch_finish(built),
                )
                .await;
                return false;
            }
            Rung::Consolidate => {
                // The pace depends on WHY we are folding, and the engine knows
                // which. Triggered by quiet there is no traffic to protect and
                // finishing sooner is better; triggered by churn there are live
                // queries and every core taken is one they do not get.
                let pace = if idle {
                    skeg_vector::ConsolidatePace::Idle
                } else {
                    skeg_vector::ConsolidatePace::Serving
                };
                let ran = off_thread_maintenance(
                    arc,
                    "consolidate",
                    shard_id,
                    |b| b.consolidate_begin(),
                    move |job| job.build_with_threads(&d, pace),
                    |b, built| b.consolidate_finish(built),
                )
                .await;
                if ran == MaintenanceOutcome::Ran && arc.read().backend.wants_ivf() {
                    // Base changed: rebuild the IVF router off the request path
                    // AND off the write lock. `begin` dups the fd, `build` runs
                    // k-means off-thread, `finish` swaps under a short lock.
                    off_thread_maintenance(
                        arc,
                        "ivf",
                        shard_id,
                        |b| b.ivf_begin(),
                        |job| job.build(),
                        |b, built| b.ivf_finish(built),
                    )
                    .await;
                }
                // `Ran`, not `true`: with the fold budget a due consolidate can
                // skip its tick, and reporting it as done would reset the
                // caller's idle tracking over work that never happened. The
                // retry is the next tick.
                return ran == MaintenanceOutcome::Ran;
            }
        }
    }
    false
}

/// Process-wide budget on concurrent HEAVY maintenance builds.
///
/// The per-fold thread cap does not compose: an explicit consolidate
/// broadcasts to every shard, and eight folds at two capped threads each
/// still took the whole machine (measured 947-977% CPU on 10 cores, searches
/// at p50 158 / p99 534 ms for the duration). The cap protects against one
/// fold; this protects against the broadcast, by letting at most
/// `SKEG_FOLD_CONCURRENCY` heavy builds run at once and queueing the rest.
///
/// Flushes are exempt: they are frequent, cheap, and gating them behind a
/// parked fold would stall ingest. The trade is explicit: a gated broadcast
/// takes longer wall-clock (shards serialise), and in exchange the queries
/// keep most of the machine. Waiting is safe here because `begin` has already
/// snapshotted and released the vindex lock, and writes that land while a job
/// is parked are replayed from the WAL suffix at finish.
fn fold_budget() -> &'static tokio::sync::Semaphore {
    static BUDGET: std::sync::OnceLock<tokio::sync::Semaphore> = std::sync::OnceLock::new();
    BUDGET.get_or_init(|| {
        let n = std::env::var("SKEG_FOLD_CONCURRENCY")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n >= 1)
            // In the test binary dozens of tests fold in parallel and two
            // process-wide permits would be contended for the whole run,
            // starving every tick-outcome assertion. The parking behaviour is
            // still proven by the test that drains all permits explicitly.
            .unwrap_or(if cfg!(test) { 64 } else { 2 });
        tokio::sync::Semaphore::new(n)
    })
}

/// Which maintenance kinds count against the fold budget.
///
/// Everything that rebuilds or rewrites a graph does; the flush only stacks
/// the delta into a run and must never wait behind a fold.
fn is_budgeted(label: &str) -> bool {
    matches!(label, "consolidate" | "runs-merge" | "delete-patch" | "ivf")
}

/// The three phases (begin under a short lock, build off the lock, finish under
/// a short lock), with the error propagated rather than only logged.
///
/// Automatic maintenance can afford to log and retry on the next tick; an
/// explicit command cannot: it has to tell the client what went wrong. One
/// implementation for both: it is the same dance.
///
/// `Ok(false)` means there was nothing to do (begin returned None).
/// What a maintenance attempt actually did. "Nothing to do" and "the budget
/// was busy" used to be the same `false`, which is how a priority rung could
/// consume a tick without doing anything AND without letting the rung below
/// it run: starvation traded for starvation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaintenanceOutcome {
    /// The job ran to completion.
    Ran,
    /// The backend had nothing to do.
    NotNeeded,
    /// The process-wide fold budget, or this vindex's heavy gate, was taken;
    /// retry next tick. The caller MUST consider running a cheaper rung
    /// instead of returning.
    BudgetBusy,
    /// The job ran and FAILED. Distinct from `NotNeeded`, which used to
    /// swallow it: a failed merge reported as "nothing to do" hides a real
    /// problem from every signal above it, and made the anti-starvation
    /// comment ("only a merge that actually ran consumes the tick") untrue.
    Failed,
}

async fn try_off_thread_maintenance<T, B>(
    arc: &VectorEntry,
    label: &str,
    wait_for_budget: bool,
    begin: impl FnOnce(&mut VectorBackend) -> std::io::Result<Option<T>>,
    build: impl FnOnce(T) -> std::io::Result<B> + Send + 'static,
    finish: impl FnOnce(&mut VectorBackend, B) -> skeg_vector::FinishResult,
) -> Result<MaintenanceOutcome, String>
where
    T: Send + 'static,
    B: Send + 'static,
{
    try_off_thread_maintenance_with_abort(arc, label, wait_for_budget, begin, build, finish, |_| {})
        .await
}

async fn try_off_thread_maintenance_with_abort<T, B>(
    arc: &VectorEntry,
    label: &str,
    wait_for_budget: bool,
    begin: impl FnOnce(&mut VectorBackend) -> std::io::Result<Option<T>>,
    build: impl FnOnce(T) -> std::io::Result<B> + Send + 'static,
    finish: impl FnOnce(&mut VectorBackend, B) -> skeg_vector::FinishResult,
    abort: impl FnOnce(&mut VectorBackend),
) -> Result<MaintenanceOutcome, String>
where
    T: Send + 'static,
    B: Send + 'static,
{
    let mut abort = Some(abort);
    // The PER-VINDEX heavy gate comes FIRST, the global budget second.
    //
    // The other order wastes the scarcer resource: a second job on the same
    // index would take a global permit and then park on the local gate,
    // holding a machine-wide permit away from every OTHER index while doing
    // nothing. Taking the local gate first means a permit is only ever held
    // by a job that can proceed. Safe now that the flush is explicitly
    // exempt from both.
    //
    // Cloned under a brief read lock and acquired holding nothing.
    let heavy_sem = if is_budgeted(label) {
        Some(Arc::clone(&arc.read().heavy))
    } else {
        None
    };
    let _heavy = match heavy_sem {
        Some(s) if wait_for_budget => Some(
            s.acquire_owned()
                .await
                .expect("per-vindex heavy semaphore is never closed"),
        ),
        Some(s) => match s.try_acquire_owned() {
            Ok(p) => Some(p),
            Err(_) => {
                // Another heavy job owns this vindex. Same contract as a busy
                // global budget: do not consume the tick, let the flush run.
                skeg_telemetry::tick_counter(skeg_telemetry::Counter::MaintenanceBudgetSkips);
                return Ok(MaintenanceOutcome::BudgetBusy);
            }
        },
        None => None,
    };
    // Then the global budget - still before the snapshot. `begin` is not free -
    // a runs-merge dups file descriptors, builds the survivor map and bumps
    // run_seq - and taking the permit afterwards meant throwing all of that
    // away, descriptors and sequence gap included, whenever the budget
    // happened to be busy.
    //
    // Waiting here holds NO lock, so reads and writes proceed normally.
    let _permit = if is_budgeted(label) {
        if wait_for_budget {
            skeg_telemetry::incr_gauge(skeg_telemetry::Gauge::FoldsWaiting);
            let p = fold_budget().acquire().await;
            skeg_telemetry::decr_gauge(skeg_telemetry::Gauge::FoldsWaiting);
            Some(p.expect("fold budget semaphore is never closed"))
        } else {
            match fold_budget().try_acquire() {
                Ok(p) => Some(p),
                Err(_) => {
                    skeg_telemetry::tick_counter(skeg_telemetry::Counter::MaintenanceBudgetSkips);
                    return Ok(MaintenanceOutcome::BudgetBusy);
                }
            }
        }
    } else {
        None
    };
    // Short lock: snapshot only, no O(live) reads and no graph build.
    let job = {
        let mut g = arc.write();
        match begin(&mut g.backend) {
            Ok(Some(j)) => j,
            Ok(None) => return Ok(MaintenanceOutcome::NotNeeded),
            Err(e) => return Err(format!("{label} begin failed: {e}")),
        }
    };
    // NO lock is held here: this is what lets reads proceed while the graph
    // is rebuilt. Heavy kinds also respect the process-wide budget, and HOW
    // they respect it depends on who asked. An explicit client request parks
    // and waits its turn. The maintenance loop must NEVER park: it runs one
    // op per vindex per tick, sequentially, so a parked runs-merge froze the
    // whole shard's maintenance behind an explicit broadcast - no flushes for
    // eight minutes, the delta grew unbounded, and every search paid a flat
    // scan over it. That was measured, not imagined: flush_total stayed 0
    // across 233k writes while folds_waiting read 6. If there is no permit
    // now, maintenance skips and retries next tick; the flush always gets its
    // turn.
    let built = match tokio::task::spawn_blocking(move || build(job)).await {
        Ok(Ok(b)) => b,
        result => {
            {
                let mut g = arc.write();
                abort
                    .take()
                    .expect("abort callback is consumed only on failure")(
                    &mut g.backend
                );
            }
            return match result {
                Ok(Err(e)) => Err(format!("{label} build failed: {e}")),
                Err(e) => Err(format!("{label} build task panicked: {e}")),
                Ok(Ok(_)) => unreachable!("the successful build arm returned above"),
            };
        }
    };
    let cleanup = {
        let mut g = arc.write();
        match finish(&mut g.backend, built) {
            Ok(outcome) => outcome,
            // BEFORE the commit point: nothing took effect, so undoing is
            // both safe and required.
            Err(e) => {
                abort
                    .take()
                    .expect("abort callback is consumed only on failure")(
                    &mut g.backend
                );
                return Err(format!("{label} finish failed: {e}"));
            }
        }
    };
    // AFTER it: the job is done and visible. Rolling back here would undo a
    // change the store has already published, and reporting failure would
    // have the ladder retry work that happened - so this is `Ran`, with the
    // leftovers reported as leftovers.
    if let Some(e) = cleanup.cleanup_error() {
        skeg_telemetry::tick_counter(skeg_telemetry::Counter::MaintenanceCleanupFailures);
        tracing::error!(
            job = label,
            error = %e,
            "{label} committed but could not reclaim what it replaced: the \
             change stands, something was left on disk"
        );
    }
    Ok(MaintenanceOutcome::Ran)
}

/// Which counter a maintenance label belongs to.
///
/// Counted here, at the single point every kind passes through, rather than at
/// four call sites that could drift apart.
fn maintenance_counter(label: &str) -> Option<skeg_telemetry::Counter> {
    use skeg_telemetry::Counter;
    match label {
        "flush" => Some(Counter::MaintenanceFlush),
        "consolidate" => Some(Counter::MaintenanceConsolidate),
        "runs-merge" => Some(Counter::MaintenanceRunsMerge),
        "delete-patch" => Some(Counter::MaintenanceDeletePatch),
        _ => None,
    }
}

async fn off_thread_maintenance<T, B>(
    arc: &VectorEntry,
    label: &str,
    shard_id: usize,
    begin: impl FnOnce(&mut VectorBackend) -> std::io::Result<Option<T>>,
    build: impl FnOnce(T) -> std::io::Result<B> + Send + 'static,
    finish: impl FnOnce(&mut VectorBackend, B) -> skeg_vector::FinishResult,
) -> MaintenanceOutcome
where
    T: Send + 'static,
    B: Send + 'static,
{
    off_thread_maintenance_with_abort(arc, label, shard_id, begin, build, finish, |_| {}).await
}

async fn off_thread_maintenance_with_abort<T, B>(
    arc: &VectorEntry,
    label: &str,
    shard_id: usize,
    begin: impl FnOnce(&mut VectorBackend) -> std::io::Result<Option<T>>,
    build: impl FnOnce(T) -> std::io::Result<B> + Send + 'static,
    finish: impl FnOnce(&mut VectorBackend, B) -> skeg_vector::FinishResult,
    abort: impl FnOnce(&mut VectorBackend),
) -> MaintenanceOutcome
where
    T: Send + 'static,
    B: Send + 'static,
{
    match try_off_thread_maintenance_with_abort(arc, label, false, begin, build, finish, abort)
        .await
    {
        Ok(outcome) => {
            if outcome == MaintenanceOutcome::Ran
                && let Some(c) = maintenance_counter(label)
            {
                skeg_telemetry::tick_counter(c);
            }
            outcome
        }
        Err(e) => {
            // Automatic maintenance retries on the next tick - but the caller
            // is told it FAILED, not that there was nothing to do.
            error!("shard {shard_id}: {e}");
            skeg_telemetry::tick_counter(skeg_telemetry::Counter::MaintenanceFailures);
            MaintenanceOutcome::Failed
        }
    }
}

fn recover_vindexes(
    shard_id: usize,
    dir: &Path,
    tier: QuantKind,
    mmap_tier: bool,
    mmap_graph: bool,
    read_only: bool,
    in_flight: &[String],
) -> std::io::Result<(VindexSet, Vec<String>)> {
    let mut set = VindexSet::new();
    // Names this shard resolved. Their payload blobs are KV keys and can only
    // be reclaimed once the VLog is usable, which is the caller's scope.
    let mut resolved = Vec::new();
    // Names the coordinator decided to REMOVE, before any shard started: the
    // undo of a create that did not reach everywhere, or the completion of a
    // drop that did not. Deciding needs every shard's registry at once, which
    // is why it is not decided here; see `ShardSet::open_mode_full_mmap`. A
    // read-only open never arrives here with a non-empty list - it refuses to
    // open at all rather than serve around a half-state.
    debug_assert!(!read_only || in_flight.is_empty());
    for name in in_flight {
        if read_registry(dir)?.iter().any(|e| &e.name == name) {
            tracing::warn!(
                shard = shard_id,
                index = name,
                "resolving an unfinished catalogue operation by removing the index"
            );
            persist_registry_removing(dir, &RwLock::new(VindexSet::new()), Some(name))?;
            remove_vindex_dir(dir, name);
        }
        // Recorded whether or not the entry was still there. A crash between
        // that removal and the caller's blob sweep leaves exactly the state
        // where it is not, and a sweep conditional on it would then find
        // nothing to do and orphan those blobs for good.
        resolved.push(name.clone());
    }
    // Fail-closed: a registry that will not parse refuses the shard open. The
    // alternative is opening a store with an unknown number of its indexes
    // missing, which is precisely how serve mode served an eighth of one.
    for entry in read_registry(dir)? {
        let open_tier = entry.kind.and_then(QuantKind::from_wire).unwrap_or(tier);
        let kind = entry
            .kind
            .unwrap_or_else(|| open_tier.to_wire().unwrap_or(1));
        let vdir = dir.join(format!("vindex-{}", entry.name));
        let mut idx = DiskVamanaIndex::open_with_tier_full(&vdir, open_tier, mmap_tier, mmap_graph)
            .map_err(|e| {
                std::io::Error::new(
                    e.kind(),
                    format!(
                        "shard {shard_id}: recovering vindex '{}' failed: {e}",
                        entry.name
                    ),
                )
            })?;
        idx.set_auto_flush(false); // flushed off-thread by the maintenance loop
        let generation = entry.generation;
        set.insert(
            entry.name,
            Arc::new(RwLock::new(Vindex::recovered(
                VectorBackend::Disk(Box::new(idx)),
                kind,
                generation,
            ))),
        );
    }
    Ok((set, resolved))
}

/// Look up a vindex by its (already tenant-scoped) name, reopening it lazily if
/// it was evicted. Stamps `last_access` on every hit.
///
/// - Hit: clone the `Arc`. No await, no blocking - the hot path.
/// - Miss but the name is in the on-disk registry (a disk index that was
///   evicted, not dropped): reopen it. Reopened via `Vindex::recovered`, so its
///   payload index rebuilds on the first filtered search, exactly like a
///   restart.
/// - Miss and not in the registry: `None` (a genuine "not found").
///
/// The shard runs on a single-threaded (`current_thread`) runtime, so a
/// synchronous `open_with_tier_full` here would block the executor for the whole
/// reopen - starving every other request queued on this shard, other tenants
/// included. So the open runs on the blocking pool via `spawn_blocking`;
/// awaiting the join yields the shard thread back to other tasks. That is the
/// bulkhead around the cold-start cost: only the triggering request pays the
/// reopen latency (~the index's resident size in reads), not the whole shard.
///
/// No lock is held across the `.await` (a parking_lot guard is `!Send` and must
/// not anyway). The triggering request still waits for the reopen - unavoidable,
/// it needs the data. The tiering policy is responsible for not thrashing
/// (hysteresis); this is just the mechanism.
///
/// No in-flight dedup. Two requests racing on the same just-evicted
/// index both open it; the write-lock double-check below keeps the first
/// published and drops the loser. Wasteful (a second open) but correct. Upgrade
/// path if a reopen storm spikes RAM: an in-flight set + `Notify`.
async fn get_or_reopen(
    vindexes: &RwLock<VindexSet>,
    dir: &Path,
    tier: QuantKind,
    mmap_tier: bool,
    mmap_graph: bool,
    name: &str,
) -> Option<VectorEntry> {
    // Fast path: resident. Clone + stamp under a short read lock, no await.
    if let Some(entry) = vindexes.read().get(name).cloned() {
        entry.read().touch();
        return Some(entry);
    }
    // Miss. Only disk-backed indexes survive in the registry and can be
    // reopened; a flat (in-RAM) index that is gone is gone.
    // The registry parsed at open, so a failure HERE means the file changed
    // under a live shard. Log it: returning None would report the index as
    // absent, which is indistinguishable from a name that never existed.
    let registry = match read_registry(dir) {
        Ok(entries) => entries.into_iter().find(|entry| entry.name == name)?,
        Err(e) => {
            tracing::error!(
                dir = %dir.display(),
                index = name,
                error = %e,
                "vindex registry became unreadable while the shard was open: \
                 cannot reopen this index"
            );
            return None;
        }
    };
    let open_tier = registry.kind.and_then(QuantKind::from_wire).unwrap_or(tier);
    let kind = registry
        .kind
        .unwrap_or_else(|| open_tier.to_wire().unwrap_or(1));
    // Reopen OFF the shard thread (see the doc comment). No guard is held here.
    let vdir = dir.join(format!("vindex-{name}"));
    let opened = tokio::task::spawn_blocking(move || {
        DiskVamanaIndex::open_with_tier_full(&vdir, open_tier, mmap_tier, mmap_graph)
    })
    .await;
    let mut idx = match opened {
        Ok(Ok(idx)) => idx,
        Ok(Err(e)) => {
            error!("reopening evicted vindex '{name}' failed: {e}");
            return None;
        }
        Err(e) => {
            error!("reopen task for vindex '{name}' panicked: {e}");
            return None;
        }
    };
    // Publish under a brief write lock. A racing request may have reopened the
    // same index while we awaited; if so, drop the one we just opened (loser of
    // the race) and return the resident one.
    let mut w = vindexes.write();
    if let Some(entry) = w.get(name).cloned() {
        drop(idx);
        entry.read().touch();
        return Some(entry);
    }
    idx.set_auto_flush(false); // flushed off-thread by the maintenance loop
    let entry: VectorEntry = Arc::new(RwLock::new(Vindex::recovered(
        VectorBackend::Disk(Box::new(idx)),
        kind,
        registry.generation,
    )));
    entry.read().touch();
    w.insert(name.to_owned(), entry.clone());
    Some(entry)
}

/// Build the scoped map key for `index` under `tenant`, mirroring the RESP3
/// handler's `scoped_vindex_name`: tenant `0` (single-tenant / anonymous) uses
/// the raw name; otherwise the key is `<32 hex>::<index>`, the hex being the
/// tenant id's bytes in `to_le_bytes` order (matching `tenant_u128`).
fn scope_key(tenant: u128, index: &str) -> String {
    if tenant == 0 {
        return index.to_owned();
    }
    use std::fmt::Write;
    let mut s = String::with_capacity(32 + 2 + index.len());
    for b in tenant.to_le_bytes() {
        let _ = write!(s, "{b:02x}");
    }
    s.push_str("::");
    s.push_str(index);
    s
}

/// Whose vector quota the rows of the index at this REGISTRY KEY count
/// against.
///
/// A tenant is never taken from a string a client chose, and this is not one:
/// the scoped key is written by the server. `scope_key` builds it from a
/// tenant id the server authenticated; `vindex_create` - the door every
/// binary protocol and every library caller reaches - refuses `::` in a raw
/// name; `vindex_create_scoped` is the single pre-scoped entry and its one
/// in-tree caller is the RESP3 layer, which refuses the separator before
/// prepending its own prefix; and `read_registry` fails the open outright on
/// a key that does not round-trip through `scope_key(unscope_key(k))`. By the
/// time this runs, the key is a value this server wrote, and reading the
/// owner back out of it is a lookup rather than a guess.
///
/// It is also the SAME direction the write path travels: a request from
/// tenant `T` reaches an index by `scope_key(T, raw)`, so the tenant a write
/// is charged to is by construction the one this returns. Getting the answer
/// any other way needs the tenant stored beside the index, which the registry
/// does not carry - the structural fix, recorded and deliberately not taken
/// here.
///
/// Two states it cannot see. Both are recorded rather than guessed at:
///
/// - a `<32 lowercase hex>::name` key an OLDER build let a tenant-0 client
///   create round-trips through `scope_key`, so it is attributed to the
///   tenant its name spells. The CHANGELOG already says such a key exists
///   only in a store written before the door was closed and has to be found
///   by hand.
/// - an EMBEDDER calling `ShardSet::vset` with a tenant unrelated to the name
///   it passes. Both are parameters, and nothing pairs them; the RESP3 and
///   native handlers always scope the name with the tenant they charge.
fn tenant_of_scoped_index(key: &str) -> u128 {
    unscope_key(key).0
}

/// Inverse of [`scope_key`]: split a scoped map key into `(tenant, index)`. A
/// key with no `::` prefix (or a malformed one) is the tenant-`0` namespace.
fn unscope_key(key: &str) -> (u128, String) {
    if let Some((hex, index)) = key.split_once("::")
        && hex.len() == 32
    {
        let mut bytes = [0u8; 16];
        if (0..16).all(|i| {
            u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)
                .map(|b| bytes[i] = b)
                .is_ok()
        }) {
            return (u128::from_le_bytes(bytes), index.to_owned());
        }
    }
    (0, key.to_owned())
}

// ── Worker ────────────────────────────────────────────────────────────────────

// The worker is a thread entry point: it must own `dir` and `rx` for the
// thread's `'static` lifetime, so by-value arguments are required here.
//
/// What a shard hands the coordinator when it becomes queryable, so the
/// vector quota can be put back before the first write is admitted.
///
/// The counter is per TENANT and the rows are per SHARD, so neither side can
/// finish the sum alone: a shard knows what it holds and not who else holds a
/// copy of it, and the coordinator knows the routing and not the rows. This
/// carries the halves across the readiness barrier.
///
/// Two shapes, because a logical row is not always one physical row:
///
/// - an index with no semantic router is HASH-PLACED. An id maps to exactly
///   one shard, for ever, so the per-shard counts simply add up and only the
///   count has to travel.
/// - a ROUTED index can hold the same logical row twice - a boundary replica,
///   or a crash between a move's write and its source delete - so a count
///   would double it. Its ids travel instead, and the coordinator applies the
///   owner map's rule: one entry per logical id, however many copies exist.
#[derive(Debug, Default)]
struct ShardReady {
    /// `(scoped name, live rows here)` for the hash-placed indexes.
    counts: Vec<(String, u64)>,
    /// `(scoped name, live ids here)` for the routed ones. Ids rather than a
    /// count, because only the union across shards is the answer.
    routed_ids: Vec<(String, Vec<u64>)>,
}

// With `read_only` set the shard rejects every mutation and skips background
// compaction and snapshots: the `--mode serve` path over an offline-built
// index.
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
    quota: Arc<crate::quota::TenantVectorQuota>,
    memory: Arc<crate::memory::MemoryGovernor>,
    disk_counter: skeg_core::SharedTenantDisk,
    // Names whose catalogue fan-out never finished, decided by the coordinator
    // because deciding needs every shard's registry at once. Writable: remove
    // them. Read-only: decline to serve them.
    in_flight: Vec<String>,
    // Scoped names with a semantic router, decided by the coordinator because
    // the sidecars live beside the shards rather than inside them. A routed
    // index can hold the same logical row on two shards, so its rows are
    // reported as IDS to be deduplicated; everything else is reported as a
    // count. See `ShardReady`.
    routed: Arc<HashSet<String>>,
    // Reports the shard's startup outcome to `ShardSet::open`: `Ok` once recovery
    // is done and the request loop is about to run, `Err` if the store cannot be
    // opened (e.g. already locked by another process). `open` blocks on this and
    // aborts startup if any shard reports `Err`. Dropping the sender without
    // signalling (a panic) unblocks the waiter with a recv error.
    ready: std::sync::mpsc::Sender<Result<ShardReady, String>>,
) {
    skeg_platform::pin_current_thread_to_performance_core();

    let vsearch_pool = match (workers > 0).then(|| VsearchPool::new(workers)) {
        Some(Ok(pool)) => Some(pool),
        Some(Err(e)) => {
            let _ = ready.send(Err(format!("vsearch pool startup failed: {e}")));
            return;
        }
        None => None,
    };

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
        let opened = if read_only {
            VLog::open_read_only_with_shared_disk(&dir, disk_counter).await
        } else {
            VLog::open_with_shared_disk(&dir, disk_counter).await
        };
        let vlog = match opened {
            Ok(v) => v,
            Err(e) => {
                // Report the failure so `ShardSet::open` aborts startup instead
                // of binding a server whose storage never came up.
                let _ = ready.send(Err(format!("shard {shard_id}: VLog::open: {e}")));
                return;
            }
        };

        // Each request runs as its own task on the LocalSet, so concurrent
        // writes to this shard are batched into one group commit. `VLog` is
        // `!Send` (Rc-backed), so `spawn_local` is required.
        // Caps the number of request tasks running at once on this shard.
        let inflight = std::sync::Arc::new(tokio::sync::Semaphore::new(MAX_INFLIGHT_PER_SHARD));
        // Vector indexes: disk-backed ones are recovered from the registry;
        // flat ones are in-RAM and start empty. Shared across the
        // per-request tasks on this single-threaded LocalSet via Rc/RefCell.
        // `Arc<RwLock>` (instead of the previous `Rc<RefCell>`) so the
        // VindexSet is `Send + Sync`: dedicated VSEARCH workers can own an
        // index lock while the shard runtime continues serving KV work.
        // The read/write locks are uncontended in inline mode (~10ns acquire
        // on M1), so there is no measurable cost for the default path.
        let (vindexes, resolved) = match recover_vindexes(
            shard_id, &dir, tier, mmap_tier, mmap_graph, read_only, &in_flight,
        ) {
            Ok((vindexes, resolved)) => (Arc::new(RwLock::new(vindexes)), resolved),
            Err(e) => {
                let _ = ready.send(Err(e.to_string()));
                return;
            }
        };
        let vindexes: Arc<RwLock<VindexSet>> = vindexes;
        // The registry entry and the directory went during recovery, which is
        // what stops the index being served. Its payload blobs are KV keys and
        // need the VLog, so they go here - otherwise finishing a drop would
        // leave behind exactly the blobs a completed drop reclaims, and a later
        // index reusing the name and id would serve one of them.
        for name in resolved {
            if let Err(e) = sweep_payload_blobs(&vlog, unscope_key(&name).0, &name).await {
                error!("shard {shard_id}: reclaiming blobs of resolved '{name}': {e}");
            }
        }
        // What this shard now holds, read once and used twice: to reclaim the
        // blobs nothing names any more, and to report the rows the vector
        // quota has to be rebuilt from. Inside the readiness barrier for both
        // reasons: the store is quiescent, the registry has just been read,
        // no request has had a chance to stage a blob whose commit has not
        // landed YET - the one state the reclamation must not mistake for
        // garbage - and no write can be admitted against a count that is not
        // in place, because `open` has not returned.
        //
        // Skipped entirely in read-only: it admits no writes, so there is no
        // quota to enforce and nothing to reclaim, and this is the one O(rows)
        // pass in a serve-mode open.
        let mut report = ShardReady::default();
        if !read_only {
            let live = collect_live_rows(&vindexes);
            for (name, (_, rows)) in &live {
                if routed.contains(name.as_str()) {
                    report
                        .routed_ids
                        .push((name.clone(), rows.iter().map(|&(id, _)| id).collect()));
                } else {
                    report.counts.push((name.clone(), rows.len() as u64));
                }
            }
            let reclaimed = reclaim_orphan_blobs(&vlog, &live).await;
            if reclaimed > 0 {
                skeg_telemetry::add_counter(
                    skeg_telemetry::Counter::PayloadBlobsReclaimedAtOpen,
                    reclaimed,
                );
                tracing::info!(
                    shard = shard_id,
                    reclaimed,
                    "reclaimed payload blobs no live row names"
                );
            }
        }
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                // Recovery (incl. the multi-second quant-tier build at 500k+)
                // is done. Warm the payload indexes too, then signal ready so
                // `open` returns and the listener can bind: the barrier is
                // there so the port opening means queryable, and a search that
                // still has to rebuild an index is not being served, it is
                // finishing the startup someone else skipped.
                let rebuilt = warm_payload_indexes(&vlog, &vindexes, &dir).await;
                let _ = ready.send(Ok(report));
                // Ready first, then persist: the file is an optimisation for the
                // next open, never a precondition for serving this one.
                //
                // A warm that had to rebuild from the log just paid for the
                // expensive path, and until now it threw the result away unless
                // the process happened to survive to the next snapshot five
                // minutes later. Worse, the snapshot is written on its own
                // schedule whether or not the payload index can be persisted, so
                // a store that kept missing that window was guaranteed a slow
                // open every single time. Pay once.
                // Always, not only after a rebuild. Recovery replays whatever
                // the log holds past the last snapshot, and a store that
                // restarts more often than the snapshot interval never gets a
                // fresh one: this open replayed 3.667.425 records and took 21
                // minutes for exactly that reason. Snapshotting here caps the
                // next replay at whatever is written from now on, and the open
                // that just finished has already paid far more than this costs.
                let _ = rebuilt;
                if !read_only {
                    snapshot_and_payload_indexes(
                        &vlog,
                        &vindexes,
                        &dir,
                        shard_id,
                        &mut HashMap::new(),
                    )
                    .await;
                }
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
                    let svindexes = vindexes.clone();
                    let sdir = dir.clone();
                    tokio::task::spawn_local(async move {
                        // Stamp of the last cache written per vindex, so an
                        // idle store stops rewriting the same bytes.
                        let mut written: HashMap<String, (u64, u64)> = HashMap::new();
                        loop {
                            tokio::time::sleep(SNAPSHOT_INTERVAL).await;
                            snapshot_and_payload_indexes(
                                &svlog,
                                &svindexes,
                                &sdir,
                                shard_id,
                                &mut written,
                            )
                            .await;
                        }
                    });

                    // Idle consolidation: a bulk load leaves an un-consolidated
                    // delta (RAM + a per-query flat scan). Once a vindex's writes
                    // go quiet - its delta is unchanged across a tick - fold it,
                    // so the served index is lean BY DEFAULT, not only after an
                    // explicit VINDEX.CONSOLIDATE. The fold runs on a blocking
                    // thread so the build does not stall the shard runtime.
                    // The rebuild runs off the write lock (begin/build/finish via
                    // off_thread_maintenance), so a query to that vindex does not
                    // wait for the fold. The lock is taken only for the initial
                    // snapshot and the final swap, both short.
                    let kvindexes = vindexes.clone();
                    let kdir = dir.clone();
                    tokio::task::spawn_local(async move {
                        // Phase-shift shards so their idle folds (each a graph
                        // rebuild) do not all run at once and stack their build
                        // buffers into one RSS spike. A process-wide
                        // permit would serialise them exactly; the stagger is the
                        // cheap version with no cross-shard plumbing.
                        tokio::time::sleep(Duration::from_secs(shard_id as u64 * 2)).await;
                        let mut prev: HashMap<String, usize> = HashMap::new();
                        loop {
                            tokio::time::sleep(idle_maint_interval()).await;
                            let snap: Vec<(String, VectorEntry)> = {
                                let g = kvindexes.read();
                                g.iter().map(|(n, a)| (n.clone(), a.clone())).collect()
                            };
                            for (name, arc) in snap {
                                let delta = arc.read().backend.delta_len();
                                // Idle == delta unchanged since the previous tick.
                                let idle = prev.insert(name.clone(), delta) == Some(delta);
                                let vdir = kdir.join(format!("vindex-{name}"));
                                if maintenance_tick(&arc, &vdir, shard_id, idle).await {
                                    prev.insert(name, 0); // folded
                                }
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
                    let quota = quota.clone();
                    let memory_task = memory.clone();
                    let vsearch_pool = vsearch_pool.clone();
                    let shard_id_u16 = shard_id as u16;
                    tokio::task::spawn_local(async move {
                        // Telemetry: classify the op, time the work, record.
                        // `op_kind` is borrowed-only · no Send cost on the
                        // hot path (the enum is `Copy`).
                        let op_kind = telemetry_op(&msg.req);
                        let t0 = std::time::Instant::now();
                        let resp = process(
                            &vlog,
                            &vindexes,
                            &dir,
                            msg.req,
                            read_only,
                            vsearch_pool.as_deref(),
                            &quota,
                            &memory_task,
                            tier,
                            mmap_tier,
                            mmap_graph,
                        )
                        .await;
                        // VSearch is recorded once by the scatter in `vsearch`,
                        // not once per shard: it is the only op that fans out to
                        // every worker, so a per-shard tick multiplies the count
                        // by the shard number and times a fragment of the search
                        // instead of the search. Per-shard counters carry no
                        // information for it either, since every shard receives
                        // exactly one message per search.
                        if let Some(op) = op_kind
                            && op != skeg_telemetry::Op::VSearch
                        {
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
            | ShardReq::SetMany(..)
            | ShardReq::Append(..)
            | ShardReq::Del(..)
            | ShardReq::EraseTenant { .. }
            | ShardReq::ErasePrefix { .. }
            | ShardReq::Reclaim
            | ShardReq::VindexCreate { .. }
            | ShardReq::VindexDrop { .. }
            | ShardReq::VindexConsolidate { .. }
            | ShardReq::SnapshotAndPayloadIndexes
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
        ShardReq::Set(..) | ShardReq::SetMany(..) | ShardReq::Append(..) => Some(Op::Set),
        ShardReq::Del(..) => Some(Op::Del),
        ShardReq::Vset { .. } => Some(Op::VSet),
        ShardReq::Vsearch { .. } => Some(Op::VSearch),
        ShardReq::Vdel { .. } => Some(Op::VDel),
        // An erase is one request but N deletes; counting it as a single Del
        // would understate it and counting it as N would hide its latency, so
        // it stays out of the op counters until it has its own classifier.
        ShardReq::EraseTenant { .. }
        | ShardReq::ErasePrefix { .. }
        | ShardReq::Reclaim
        | ShardReq::CountTenantKeys(_)
        | ShardReq::Vget { .. }
        | ShardReq::SampleVectors { .. }
        | ShardReq::CollectMoves { .. }
        | ShardReq::LiveIds { .. }
        | ShardReq::PayloadBlobs { .. }
        | ShardReq::StillCurrent { .. }
        | ShardReq::CollectBoundary { .. }
        | ShardReq::GraphSample { .. }
        | ShardReq::VindexCreate { .. }
        | ShardReq::VindexList
        | ShardReq::VindexCheck { .. }
        | ShardReq::WantsIvf { .. }
        | ShardReq::VindexDrop { .. }
        | ShardReq::VindexConsolidate { .. }
        | ShardReq::SnapshotAndPayloadIndexes
        | ShardReq::Evict { .. }
        | ShardReq::IndexStats
        | ShardReq::TenantCacheBytes(_)
        | ShardReq::Stats => None,
    }
}

/// Drop one vindex by its scoped map key, reclaiming its vector quota and its
/// payload blobs. `Ok(false)` means it was not there to begin with; the caller
/// decides whether that is an error (an explicit DROP) or a no-op (an erasure
/// sweep racing a concurrent drop).
///
/// "Not there" is decided by the CATALOGUE, not by the resident map. The map
/// holds what is open right now; an evicted disk index is committed, on disk,
/// and comes back on the next access. Reading absence off the map told a DROP
/// its target did not exist and let a tenant erasure walk past the tenant's own
/// vectors while reporting success - the KV sweep took the payload blobs and
/// left the index behind. So a miss reopens through `get_or_reopen`, which is
/// already the one place that turns a catalogue entry back into a live index,
/// and only a miss THERE is a genuine absence.
///
/// Reopening an index in order to delete it is not wasted work: its quota
/// fragment and the ids of its payload blobs are only knowable from the open
/// index, and both must be reclaimed here.
///
/// Pops the entry from the outer map first; this prevents new ops from
/// observing it. In-flight ops on this vindex keep their cloned `Arc` alive and
/// finish their inner lock window before dropping it. `remove_dir_all` on POSIX
/// deletes the path immediately even if open file handles persist on the still-
/// alive Arc clones; the handles close when the last clone drops.
async fn drop_vindex(
    vlog: &VLog,
    vindexes: &RwLock<VindexSet>,
    dir: &Path,
    quota: &Arc<crate::quota::TenantVectorQuota>,
    name: &str,
    tenant: u128,
    credit: DropCredit,
    tier: QuantKind,
    mmap_tier: bool,
    mmap_graph: bool,
) -> Result<bool, String> {
    // Bound to a local FIRST: a guard built in a `match` scrutinee lives for
    // the whole `match`, so the miss arm below would hold the write lock across
    // its `.await` and `get_or_reopen`'s own `read()` would never be granted.
    // Measured as a hang, not reasoned about. `clippy::await_holding_lock` is
    // denied workspace-wide so the compiler holds this rule now.
    let resident = vindexes.write().remove(name);
    let target = match resident {
        Some(arc) => DropTarget::Live(arc),
        None => {
            // Not resident. The catalogue decides existence, and it must be
            // READ, not guessed: an unreadable registry cannot be reported as
            // "no such index" when the index may well be there.
            let listed = read_registry(dir)
                .map_err(|e| {
                    format!("vindex registry unreadable, refusing to decide whether '{name}' exists: {e}")
                })?
                .into_iter()
                .any(|entry| entry.name == name);
            if !listed {
                return Ok(false);
            }
            // Listed. Reopening is worth attempting because only the open index
            // knows its quota fragment; if it will not open, the entry is still
            // committed and still has to go.
            match get_or_reopen(vindexes, dir, tier, mmap_tier, mmap_graph, name).await {
                // `None` from the map means a concurrent drop won the race.
                Some(_) => match vindexes.write().remove(name) {
                    Some(arc) => DropTarget::Live(arc),
                    None => return Ok(false),
                },
                None => DropTarget::Orphan,
            }
        }
    };
    let arc = match target {
        DropTarget::Live(arc) => arc,
        DropTarget::Orphan => {
            // Committed but unopenable. Same commit order as below - registry
            // first, files after - but nothing to roll back, because nothing
            // was taken out of the resident map. A failed commit leaves the
            // store exactly as it was.
            persist_registry_removing(dir, vindexes, Some(name))
                .map_err(|e| format!("vindex registry not updated: {e}"))?;
            remove_vindex_dir(dir, name);
            // The quota is deliberately NOT credited: only the open index knows
            // how many vectors it held, and subtracting a guess corrupts a
            // counter that other tenants share the meaning of. The tenant's
            // count stays high until it is rebuilt from the data.
            tracing::warn!(
                index = name,
                tenant,
                "dropped a committed vindex that would not open: its vector \
                 quota cannot be credited and must be rebuilt"
            );
            sweep_payload_blobs(vlog, tenant, name).await?;
            return Ok(true);
        }
    };
    // Read everything off the index in a tight block so the guard is gone
    // before the payload sweep below.
    let (was_disk, fragment) = {
        let guard = arc.read();
        (
            matches!(guard.backend, VectorBackend::Disk(_)),
            guard.backend.len() as u64,
        )
    };
    if was_disk {
        // COMMIT FIRST, then delete, and change NOTHING the client can observe
        // until the commit lands. The registry is the commit record, so the
        // order decides both what a crash means and what a failure means.
        //
        // It used to delete the directory and then rewrite the registry. A
        // crash between the two left the registry naming a directory that no
        // longer exists, and recovery opens every entry it lists - so the next
        // start failed outright, on a shard whose data was intact. This way a
        // crash after the commit leaves a directory nobody references, which
        // is reclaimable garbage rather than a store that will not open.
        //
        // The `arc` is still held and the quota untouched, so a failed commit
        // can be undone. Doing that work first made a refused DROP incoherent:
        // the index was gone from this process, still listed in the registry -
        // so it came back on the next open or lazy reopen - and its quota had
        // already been returned to the tenant.
        //
        // The rewrite drops the entry because the index is out of the resident
        // map; putting it back is what makes the rollback complete.
        if let Err(e) = persist_registry_removing(dir, vindexes, Some(name)) {
            vindexes.write().insert(name.to_owned(), arc);
            return Err(format!("vindex registry not updated: {e}"));
        }
    }
    // Past this point the drop is COMMITTED: the catalogue no longer lists it.
    drop(arc);
    // `fragment` is what THIS shard physically held, and the quota counts
    // LOGICAL rows. They are the same number only when an id lives on exactly
    // one shard, which is why the coordinator decides who credits.
    match credit {
        DropCredit::Fragment => quota.sub(tenant, fragment),
        DropCredit::Coordinator => {}
    }
    if was_disk {
        // Cleanup, after the fact. A failure here is NOT a failed drop - the
        // commit record is already published, so the index will not come back
        // at the next open, and telling the client otherwise invites a retry
        // of an operation that has already happened. What is left is an
        // orphan directory: reclaimable, protected from being overwritten by
        // a same-named create, and owed an ORPHAN line in HEALTH.
        remove_vindex_dir(dir, name);
    }
    // Same rule as the directory removal above, and for the same reason: the
    // commit record is published, so this is cleanup after the fact. Returning
    // an error here told the caller a drop had failed when the catalogue had
    // already forgotten the index, inviting a retry of something that has
    // happened. A blob left behind is reclaimable garbage, like an orphan
    // directory - and unlike a drop the caller now believes did not occur.
    if let Err(e) = sweep_payload_blobs(vlog, tenant, name).await {
        tracing::error!(
            index = name,
            error = %e,
            "vindex dropped but its payload blobs were not reclaimed"
        );
    }
    Ok(true)
}

/// What a DROP found when it looked for its target.
enum DropTarget {
    /// Open, in hand, and out of the resident map.
    Live(VectorEntry),
    /// In the catalogue, on disk, and it will not open. Still has to go.
    Orphan,
}

/// Remove a vindex's directory after its catalogue entry is already gone.
///
/// A failure here is NOT a failed drop: the commit record is published, so the
/// index will not come back at the next open, and telling the client otherwise
/// invites a retry of an operation that has already happened. What is left is
/// an orphan directory - reclaimable, protected from being overwritten by a
/// same-named create, and owed an ORPHAN line in HEALTH.
fn remove_vindex_dir(dir: &Path, name: &str) {
    let vdir = dir.join(format!("vindex-{name}"));
    match std::fs::remove_dir_all(&vdir) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => {
            tracing::error!(
                dir = %vdir.display(),
                error = %e,
                "vindex dropped from the registry but its directory could not \
                 be removed: it is now an orphan and must be reclaimed by hand"
            );
        }
    }
}

/// Delete every payload blob belonging to exactly this vindex.
///
/// Without this, a recreated index reusing the same name and id resurfaces a
/// stale blob under the same reserved key.
///
/// Selected by EXACT name, not by prefix. Nothing separates the name from the
/// fixed-width tail that follows it, so `starts_with` on the name would also
/// take the blobs of every index whose name this one prefixes - dropping `ev`
/// would eat `ev2`. Requiring the key to be the head plus exactly the tail is
/// what makes it exact, and it is checked for both key shapes: the
/// pre-generation one and the generation-and-version one.
///
/// EVERY generation of the name, not only the live one. A blob left by an
/// earlier incarnation is precisely what this is for - the generation stops it
/// being SERVED, and the sweep is what stops it being stored for ever.
///
/// Enumerating keys rather than the index's own `live_ids` is deliberate: it
/// needs no open index, so the orphan path reclaims blobs too, and it catches
/// blobs an earlier partial failure left behind. Deletes run
/// `buffer_unordered` so they share the group committer's flushes - awaiting
/// each in turn put one record per batch and paid a flush per key (~1.4 ms
/// each, ~14 s per 10k).
/// A payload blob key, taken apart.
///
/// One parser, so "what does this key say" has a single answer. The sweep, the
/// count and the open-time collection all ask it; three hand-rolled length
/// arithmetics would be three chances to reclaim a key that is not garbage.
struct BlobKey<'a> {
    tenant: u128,
    /// `None` for the pre-generation key, which names no incarnation.
    generation: Option<IndexGeneration>,
    name: &'a [u8],
    id: u64,
    /// `None` for the pre-generation key, which names no copy of the row.
    version: Option<u64>,
}

fn parse_payload_blob_key(key: &[u8]) -> Option<BlobKey<'_>> {
    if key.len() <= 16 + PAYLOAD_MARKER.len() {
        return None;
    }
    let tenant = u128::from_le_bytes(key[..16].try_into().ok()?);
    let take_u64 = |b: &[u8]| u64::from_le_bytes(b.try_into().expect("8-byte window"));
    let (marker, rest) = key[16..].split_at(PAYLOAD_MARKER.len());
    if marker == PAYLOAD_MARKER {
        // tenant | marker | name | id, and a name is never empty.
        if rest.len() <= 8 {
            return None;
        }
        let (name, id) = rest.split_at(rest.len() - 8);
        return Some(BlobKey {
            tenant,
            generation: None,
            name,
            id: take_u64(id),
            version: None,
        });
    }
    if marker == PAYLOAD_MARKER_V2 {
        // tenant | marker | generation | name | id | version.
        if rest.len() <= 16 + 16 {
            return None;
        }
        let generation = IndexGeneration::new(u128::from_le_bytes(rest[..16].try_into().ok()?));
        let body = &rest[16..];
        let (name, tail) = body.split_at(body.len() - 16);
        return Some(BlobKey {
            tenant,
            generation: Some(generation),
            name,
            id: take_u64(&tail[..8]),
            version: Some(take_u64(&tail[8..])),
        });
    }
    None
}

/// Is this a payload blob key of `(tenant, name)`, in ANY generation?
///
/// Every incarnation, deliberately: a blob left by an earlier one is precisely
/// what a sweep of the name must not leave behind.
fn is_payload_blob_key(key: &[u8], tenant: u128, name: &str) -> bool {
    parse_payload_blob_key(key).is_some_and(|k| k.tenant == tenant && k.name == name.as_bytes())
}

/// What one resident index would answer for: its incarnation, and every
/// `(id, version)` pair it holds live.
type LiveRows = (IndexGeneration, BTreeSet<(u64, u64)>);

/// Every resident index's [`LiveRows`], keyed by its scoped name.
///
/// Read ONCE per open, because two things need it and walking every vindex
/// twice would double the only O(rows) work inside the readiness barrier: the
/// orphan blob reclamation below, and the vector-quota rebuild that turns
/// these rows into a per-tenant count.
///
/// Keyed by the NAME, and the incarnation is checked against the GENERATION.
/// Not by the tenant, and above all not by a tenant recovered from the name: a
/// vindex name is a client-chosen string, `:` is a legal character in one, and
/// `scope_key` leaves a tenant-0 name untouched - so a client on tenant 0
/// could once call its index `<32 hex>::x` and have `unscope_key` read it back
/// as some other tenant's. That index is still tenant 0's: created by tenant
/// 0, blobs written under tenant 0, every read of them using tenant 0. Only
/// the recovery disagreed, and it disagreed by MISSING - the lookup found no
/// index of that name under that tenant, so every one of its blobs fell into
/// the "nothing here names this" branch and was deleted, leaving live rows
/// with no payload and nothing said about it.
///
/// The name alone identifies the index: a shard holds at most one per scoped
/// name, because that is the key of the map being read here. The generation
/// then pins WHICH incarnation of it, and a generation is minted by the
/// server, never spelled by a client. The tenant adds nothing those two do not
/// already decide, and it is the only part of the key a name can lie about.
fn collect_live_rows(vindexes: &RwLock<VindexSet>) -> HashMap<String, LiveRows> {
    let vs = vindexes.read();
    vs.iter()
        .map(|(scoped, arc)| {
            let g = arc.read();
            let rows = g
                .backend
                .live_ids_with_versions()
                .into_iter()
                .map(|(id, v)| (id, v.get()))
                .collect();
            (scoped.clone(), (g.generation, rows))
        })
        .collect()
}

/// Delete every payload blob no live row on this shard names.
///
/// Three kinds of garbage end up here, and they are the same kind of garbage:
///
/// - a blob STAGED for a write whose commit never landed. Prepare-before-
///   commit is what makes the pair atomic, and its price is exactly this: a
///   blob at a version no row ever took.
/// - a blob a committed overwrite SUPERSEDED, whose post-commit reclamation
///   did not run - because the process died, or because reporting the failure
///   would have lied to the client.
/// - a blob of an earlier INCARNATION of a name, or of an index this shard no
///   longer has, left by a drop whose sweep did not finish.
///
/// None of them can ever be read: a payload lookup is keyed by the index's
/// generation and the row's live version. Which is why they are collected
/// HERE, at open, once, and not by any request path - the store is quiescent,
/// the registry has just been read, and every live row's version is in hand.
///
/// # Cost
///
/// One pass over the WHOLE keyspace of the shard, unconditional and with no
/// ceiling - the same shape `count_tenant_keys` and the DROP sweep already
/// declare, but this one runs at every open rather than when an operator asks
/// for it. It is proportional to the number of KV keys the shard holds, not to
/// the number of blobs or of orphans, so a store with a large keyspace and no
/// vindexes pays for it too.
///
/// Measured 2026-09-03, release, macOS arm64, by an auditor isolating the pass
/// behind an env var: at 20k blobs the difference is UNDER THE NOISE (best of
/// three, 2,81 s with against 3,02 s without) and at 80k a single pair gives
/// +220 ms on a 14,1 s open. Neither of those opens is anywhere near the
/// 14 ms / 2,1 s cold-start budget to begin with - the cost is dominated by
/// pre-existing vLog recovery - which is the reason the pass does not show up,
/// not a reason to think it is free. Do not extrapolate from these two points:
/// they were taken on a store whose open is already an order of magnitude over
/// budget, and the slope has not been measured.
///
/// If it ever does hurt, the answer is the same one the DROP sweep's bench
/// records: an index on blob keys, not a return to walking `live_ids` - that
/// path cannot see an index which will not open.
async fn reclaim_orphan_blobs(vlog: &VLog, live: &HashMap<String, LiveRows>) -> u64 {
    let victims: Vec<Vec<u8>> = {
        let mut v = Vec::new();
        vlog.for_each_key(|k| {
            let Some(key) = parse_payload_blob_key(k) else {
                return;
            };
            // A vindex name is a Rust `String`, so a blob key whose name
            // bytes are not UTF-8 names no resident index - the same answer
            // the byte-for-byte lookup this replaced would give, reached one
            // step earlier.
            let named = std::str::from_utf8(key.name).ok().and_then(|n| live.get(n));
            let alive = match (named, key.generation, key.version) {
                // No such index here. Nothing on this shard can serve it, in
                // any generation.
                (None, _, _) => false,
                // The pre-generation key, and the index that predates
                // generations still reads it - for a row that is still live.
                (Some((generation, rows)), None, _) => {
                    generation.is_legacy() && rows.iter().any(|&(id, _)| id == key.id)
                }
                (Some((generation, rows)), Some(g), Some(version)) => {
                    *generation == g && rows.contains(&(key.id, version))
                }
                (Some(_), Some(_), None) => false,
            };
            if !alive {
                v.push(k.to_vec());
            }
        });
        v
    };
    let results: Vec<_> = stream::iter(victims.iter())
        .map(|key| vlog.del(key, PAYLOAD_DURABILITY))
        .buffer_unordered(ERASE_CONCURRENCY)
        .collect()
        .await;
    let mut reclaimed = 0u64;
    for r in results {
        match r {
            Ok(true) => reclaimed += 1,
            Ok(false) => {}
            // Best effort by design: this runs inside the readiness barrier
            // and it is a reclamation, not a repair. A blob that will not go
            // is disk, not a wrong answer, and refusing to open over it would
            // trade a leak for an outage.
            Err(e) => tracing::error!(error = %e, "reclaiming an orphaned payload blob failed"),
        }
    }
    reclaimed
}

/// Payload blobs the shard holds for `(tenant, name)`, every generation
/// included.
fn count_payload_blobs(vlog: &VLog, tenant: u128, name: &str) -> u64 {
    let mut n = 0u64;
    vlog.for_each_key(|k| {
        if is_payload_blob_key(k, tenant, name) {
            n += 1;
        }
    });
    n
}

async fn sweep_payload_blobs(vlog: &VLog, tenant: u128, name: &str) -> Result<u64, String> {
    // The sweep runs AFTER the catalogue has stopped naming the index, so its
    // failure is the state the generation exists to survive: blobs on disk
    // with nothing left to remove them. Modelled here rather than by taking
    // permissions off something, because the sweep is a KV walk and there is
    // no file to take them off.
    crate::fp_at!(crate::failpoint::WriteFailpoint::DropBlobSweep, name, Ok(0));
    let victims: Vec<Vec<u8>> = {
        let mut v = Vec::new();
        vlog.for_each_key(|k| {
            if is_payload_blob_key(k, tenant, name) {
                v.push(k.to_vec());
            }
        });
        v
    };
    let results: Vec<_> = stream::iter(victims.iter())
        .map(|key| vlog.del(key, PAYLOAD_DURABILITY))
        .buffer_unordered(ERASE_CONCURRENCY)
        .collect()
        .await;
    let mut deleted = 0u64;
    for r in results {
        match r {
            Ok(true) => deleted += 1,
            Ok(false) => {}
            Err(e) => return Err(format!("vindex drop payload failed: {e}")),
        }
    }
    Ok(deleted)
}

/// Delete every live KV key that starts with `prefix`, concurrently. Returns
/// the count deleted.
///
/// Collect under the index borrow, filtering on the prefix so only the matching
/// keys are materialised, then release the borrow before deleting: the
/// `for_each_key` callback is sync and `del` is async (and takes the index
/// mutably anyway). Deletes run `buffer_unordered` so they share the group
/// committer's flushes - awaiting each in turn put one record per batch and
/// paid a flush per key (~1.4 ms each, ~14 s per 10k). The bound stays inside
/// the concurrency ordinary client writes already drive the shard at.
///
/// Logical only: `del` tombstones, it does not reclaim the value bytes. Follow
/// with a `Reclaim` when the bytes must physically leave the disk.
async fn sweep_prefix(vlog: &VLog, prefix: &[u8], durability: Durability) -> Result<u64, String> {
    let victims: Vec<Vec<u8>> = {
        let mut v = Vec::new();
        vlog.for_each_key(|k| {
            if k.starts_with(prefix) {
                v.push(k.to_vec());
            }
        });
        v
    };
    let results: Vec<_> = stream::iter(victims.iter())
        .map(|key| vlog.del(key, durability))
        .buffer_unordered(ERASE_CONCURRENCY)
        .collect()
        .await;
    let mut erased = 0u64;
    for r in results {
        match r {
            Ok(true) => erased += 1,
            Ok(false) => {}
            Err(e) => return Err(format!("erase failed: {e}")),
        }
    }
    Ok(erased)
}

#[allow(clippy::too_many_arguments)]
async fn process(
    vlog: &VLog,
    vindexes: &Arc<RwLock<VindexSet>>,
    dir: &Path,
    req: ShardReq,
    read_only: bool,
    vsearch_pool: Option<&VsearchPool>,
    quota: &Arc<crate::quota::TenantVectorQuota>,
    memory: &Arc<crate::memory::MemoryGovernor>,
    tier: QuantKind,
    mmap_tier: bool,
    mmap_graph: bool,
) -> ShardResp {
    if read_only && is_mutation(&req) {
        return ShardResp::Err("server is in serve mode (read-only)".to_owned());
    }
    let vindexes: &RwLock<VindexSet> = vindexes;
    match req {
        // ── vector ops (synchronous; no await while the RefCell is borrowed) ──
        ShardReq::VindexCreate {
            name,
            dim,
            kind,
            disk,
            generation,
        } => {
            use std::collections::hash_map::Entry;
            // Kept for the rollback below: `name` is moved into `entry`.
            let created_name = name.clone();
            let result = match vindexes.write().entry(name) {
                Entry::Occupied(e) => Err(format!("vindex '{}' already exists", e.key())),
                Entry::Vacant(e) => {
                    if disk {
                        let vdir = dir.join(format!("vindex-{}", e.key()));
                        // An UNCOMMITTED directory is not free space.
                        //
                        // `create_empty_with_tier` calls `create_dir_all`, which
                        // succeeds on an existing directory, and then writes
                        // graph, vectors, CURRENT and the WAL over whatever is
                        // there. So a create naming an orphan silently destroys
                        // it - and an orphan can be a fully built index whose
                        // registry entry was lost, not just a dead create.
                        //
                        // "Not in the registry" means "not committed". It does
                        // not mean "reusable". Refuse and name the path: the
                        // operator decides whether that data is worth keeping.
                        if vdir.exists() {
                            Err(format!(
                                "vindex '{}' is not in the registry but {} \
                                 exists: refusing to overwrite an unregistered \
                                 index. Remove that directory to reuse the name.",
                                e.key(),
                                vdir.display()
                            ))
                        } else {
                            // The disk tier is int8 by default; TurboQuant gives
                            // sub-int8 RAM on the live write path (it needs no trained
                            // codebook). f32/binary are flat-only -> fall back to int8.
                            let tier = match kind {
                                QuantKind::TurboQuant { .. } => kind,
                                _ => QuantKind::Int8,
                            };
                            let kind = tier.to_wire().expect("disk tier has a VINDEX wire kind");
                            match DiskVamanaIndex::create_empty_with_tier(
                                &vdir,
                                dim,
                                VAMANA_L_SEARCH,
                                tier,
                            ) {
                                Ok(mut idx) => {
                                    idx.set_auto_flush(false); // flushed off-thread by the loop
                                    e.insert(Arc::new(RwLock::new(Vindex::new_at(
                                        VectorBackend::Disk(Box::new(idx)),
                                        kind,
                                        generation,
                                    ))));
                                    Ok(true)
                                }
                                Err(err) => Err(format!("vindex disk create failed: {err}")),
                            }
                        }
                    } else {
                        e.insert(Arc::new(RwLock::new(Vindex::new_at(
                            VectorBackend::Flat(Box::new(FlatIndex::new(dim, kind))),
                            kind.to_wire().unwrap_or(0),
                            generation,
                        ))));
                        Ok(false)
                    }
                }
            };
            match result {
                // The registry is the COMMIT RECORD. A create that cannot
                // publish one has not happened, and must not be acknowledged:
                // otherwise a directory on disk represents a confirmed
                // operation that no catalogue knows about, and "not in the
                // registry" stops meaning "not committed".
                //
                // The in-memory entry is rolled back too. Leaving it would
                // serve an index this open can see and the next one cannot.
                Ok(created_disk) => {
                    if created_disk && let Err(e) = persist_registry(dir, vindexes) {
                        vindexes.write().remove(&created_name);
                        ShardResp::Err(format!("vindex registry not updated: {e}"))
                    } else {
                        ShardResp::Done
                    }
                }
                Err(e) => ShardResp::Err(e),
            }
        }
        ShardReq::WantsIvf { name } => {
            let vs = vindexes.read();
            match vs.get(&name) {
                Some(entry) => ShardResp::Count(u64::from(entry.read().backend.wants_ivf())),
                None => ShardResp::Count(0),
            }
        }
        ShardReq::VindexCheck { name } => {
            // Cloned out from under a short read lock: the catalogue read below
            // is blocking I/O and must not happen with the map held.
            let resident = vindexes.read().get(&name).cloned();
            let Some(entry) = resident else {
                // Not open. "No problems" about an index nobody looked at is
                // the false green this command exists to prevent, so separate
                // the two reasons it can be missing: absent from the catalogue
                // (genuinely not here, and the coordinator already reports
                // that) versus committed but evicted (unchecked, and the report
                // has to say so instead of staying quiet).
                return match read_registry(dir) {
                    Ok(entries) if entries.iter().any(|e| e.name == name) => {
                        ShardResp::Problems(vec![
                            "not resident: not checked (its files were not opened)".to_owned(),
                        ])
                    }
                    Ok(_) => ShardResp::Problems(Vec::new()),
                    // A finding, not an error. The fail-closed rule forbids a
                    // short answer that looks complete; a report that names the
                    // damage IS complete, and this is the single most useful
                    // thing CHECK could tell an operator.
                    Err(e) => ShardResp::Problems(vec![format!(
                        "vindex registry unreadable, cannot tell an absent index \
                         from an unchecked one: {e}"
                    )]),
                };
            };
            let vindex = entry.read();
            match &vindex.backend {
                VectorBackend::Disk(idx) => match idx.check() {
                    Ok(p) => ShardResp::Problems(p),
                    Err(e) => ShardResp::Err(format!("check failed: {e}")),
                },
                // A flat index holds no graph, runs or generation slots:
                // there is nothing on disk for a check to disagree with.
                VectorBackend::Flat(_) => ShardResp::Problems(Vec::new()),
            }
        }
        ShardReq::VindexList => {
            let vs = vindexes.read();
            let mut rows: Vec<VindexRow> = vs
                .iter()
                .map(|(name, entry)| {
                    let vindex = entry.read();
                    let backend = &vindex.backend;
                    let (_, run_live, run_dead) = backend.run_contents();
                    VindexRow {
                        name: name.clone(),
                        shards_resident: 1,
                        shards_total: 1, // placeholder; the coordinator sets it
                        dim: backend.dim() as u32,
                        kind: vindex.kind,
                        backend: backend.backend_byte(),
                        n_vectors: backend.len() as u64,
                        delta: backend.delta_len() as u64,
                        runs: backend.run_count() as u64,
                        run_rows: backend.run_rows() as u64,
                        max_run_rows: backend.max_run_rows() as u64,
                        run_debt_ratio: backend.run_debt_ratio(),
                        run_live: run_live as u64,
                        run_dead: run_dead as u64,
                        tombs: backend.tombstone_count() as u64,
                        base: backend.main_len() as u64,
                    }
                })
                .collect();
            drop(vs);
            // Then the catalogue entries that are NOT resident. Only disk
            // indexes are ever in the registry, so backend is known; dim and
            // kind are recorded there too. Everything else is a property of the
            // open index and is left at zero - `shards_resident: 0` is what
            // tells the reader those zeros were not measured.
            //
            // Fail-closed, like the shard open and like VSEARCH: a registry
            // that will not parse must not become a silently shorter list,
            // which is indistinguishable from a store that lost indexes.
            let resident: std::collections::HashSet<&str> =
                rows.iter().map(|r| r.name.as_str()).collect();
            let extra: Vec<VindexRow> = match read_registry(dir) {
                Ok(entries) => entries
                    .into_iter()
                    .filter(|e| !resident.contains(e.name.as_str()))
                    .map(|e| VindexRow {
                        name: e.name,
                        shards_resident: 0,
                        shards_total: 1, // placeholder; the coordinator sets it
                        dim: e.dim as u32,
                        // Resolved the way the open resolves it. A hardcoded
                        // fallback made the same index report one kind while
                        // evicted and another while resident, on any registry
                        // entry old enough to predate the kind byte.
                        kind: e.kind.unwrap_or_else(|| tier.to_wire().unwrap_or(1)),
                        backend: 1,
                        n_vectors: 0,
                        delta: 0,
                        runs: 0,
                        run_rows: 0,
                        max_run_rows: 0,
                        run_debt_ratio: 0.0,
                        run_live: 0,
                        run_dead: 0,
                        tombs: 0,
                        base: 0,
                    })
                    .collect(),
                Err(e) => {
                    return ShardResp::Err(format!(
                        "vindex registry unreadable, refusing to return a \
                         possibly short list: {e}"
                    ));
                }
            };
            rows.extend(extra);
            // Stable order so the TUI doesn't flicker between polls.
            rows.sort_by(|a, b| a.name.cmp(&b.name));
            ShardResp::VindexList(rows)
        }
        ShardReq::SnapshotAndPayloadIndexes => {
            // Forced: no memo, so it always writes.
            snapshot_and_payload_indexes(vlog, vindexes, dir, 0, &mut HashMap::new()).await;
            ShardResp::Done
        }
        ShardReq::VindexConsolidate { name } => {
            let entry = get_or_reopen(vindexes, dir, tier, mmap_tier, mmap_graph, &name).await;
            match entry {
                None => ShardResp::Err(format!("vindex '{name}' not found")),
                Some(arc) => {
                    // This used to be `arc.write().backend.consolidate()`, which
                    // held the per-vindex write lock for the WHOLE rebuild: every
                    // query to that vindex queued behind it. Measured on 4,000
                    // vectors, a read waited 3.81s on a 3.81s consolidate. The
                    // automatic maintenance path already did this right; the
                    // explicit command did not.
                    let vdir = dir.join(format!("vindex-{name}"));
                    match try_off_thread_maintenance(
                        &arc,
                        "consolidate",
                        true,
                        |b| b.consolidate_begin(),
                        move |job| job.build(&vdir),
                        |b, built| b.consolidate_finish(built),
                    )
                    .await
                    {
                        Ok(_) => {
                            // Same lesson as the comment above, in a different
                            // field: the maintenance path rebuilds the IVF
                            // router after a consolidate and the explicit
                            // command did not. An index consolidated by hand
                            // (an operator, a bulk load) was then left with NO
                            // router, so every filtered search fell back to
                            // scanning the whole match set - measured: 19,976
                            // rows scored per shard on a 160k match set, i.e.
                            // all of it. Rebuild it here too.
                            if arc.read().backend.wants_ivf() {
                                let _ = try_off_thread_maintenance(
                                    &arc,
                                    "ivf",
                                    true,
                                    |b| b.ivf_begin(),
                                    |job| job.build(),
                                    |b, built| b.ivf_finish(built),
                                )
                                .await;
                            }
                            ShardResp::Done
                        }
                        Err(e) => ShardResp::Err(e),
                    }
                }
            }
        }
        ShardReq::Evict { name } => {
            // Pop the entry without touching its files. In-flight ops keep their
            // cloned `Arc` alive and finish; the graph + tier + delta free when
            // the last clone drops. A later access reopens it via
            // `get_or_reopen`. Files stay, so this is allowed in serve mode.
            let removed = vindexes.write().remove(&name).is_some();
            ShardResp::Evicted(removed)
        }
        ShardReq::IndexStats => {
            let vs = vindexes.read();
            let rows: Vec<(String, usize, u64, usize, bool)> = vs
                .iter()
                .map(|(name, entry)| {
                    let g = entry.read();
                    // Only disk-backed indexes can be reopened after an evict;
                    // a flat index lives only in RAM.
                    let evictable = matches!(g.backend, VectorBackend::Disk(_));
                    (
                        name.clone(),
                        g.backend.approx_ram_bytes() as usize,
                        g.last_access_ms(),
                        g.backend.len(),
                        evictable,
                    )
                })
                .collect();
            ShardResp::IndexStats(rows)
        }
        ShardReq::VindexDrop {
            name,
            tenant,
            credit,
            require_present,
        } => {
            match drop_vindex(
                vlog, vindexes, dir, quota, &name, tenant, credit, tier, mmap_tier, mmap_graph,
            )
            .await
            {
                Ok(true) => ShardResp::Done,
                Ok(false) if require_present => {
                    ShardResp::Err(format!("vindex '{name}' not found"))
                }
                Ok(false) => ShardResp::Done,
                Err(e) => ShardResp::Err(e),
            }
        }
        ShardReq::CountTenantKeys(tenant) => {
            // Read-only, so the streaming form pays off exactly as intended:
            // the count never materialises a key.
            let prefix = tenant.to_le_bytes();
            let mut n = 0u64;
            vlog.for_each_key(|k| {
                if k.len() >= 16 && k[..16] == prefix {
                    n += 1;
                }
            });
            ShardResp::Count(n)
        }
        ShardReq::EraseTenant { tenant, durability } => {
            // Erasing tenant 0 is refused: its keys are stored unscoped (no
            // 16-byte prefix), so there is no way to tell them from another
            // tenant's, and the sweep below would be a whole-store wipe.
            if tenant == 0 {
                return ShardResp::Err("cannot erase the anonymous tenant (0)".to_owned());
            }
            // Vindexes first. A vector's payload blob is itself a KV key under
            // this tenant's prefix, so the KV sweep below would delete the blob
            // and leave the index pointing at a hole. Dropping the index first
            // reclaims its blobs and its vector quota through the same path a
            // VINDEX.DROP takes, and leaves nothing dangling.
            // The tenant's indexes come from the CATALOGUE, unioned with the
            // resident map. Walking the map alone skipped every evicted index:
            // the sweep below then deleted its payload blobs - they are KV keys
            // under this tenant's prefix - and left the vectors on disk while
            // the call reported success. For an erasure that is the whole point
            // of the operation, so the registry read is fail-closed: an
            // unreadable catalogue must not become a successful erasure.
            let mut mine: BTreeSet<String> = match read_registry(dir) {
                Ok(entries) => entries
                    .into_iter()
                    .map(|entry| entry.name)
                    .filter(|k| unscope_key(k).0 == tenant)
                    .collect(),
                Err(e) => {
                    return ShardResp::Err(format!(
                        "vindex registry unreadable, refusing to report a \
                         partial erasure: {e}"
                    ));
                }
            };
            // Flat (in-RAM) indexes are never in the registry, so the resident
            // map is still authoritative for them.
            mine.extend(
                vindexes
                    .read()
                    .keys()
                    .filter(|k| unscope_key(k).0 == tenant)
                    .cloned(),
            );
            let mut dropped = 0u64;
            for name in mine {
                match drop_vindex(
                    vlog,
                    vindexes,
                    dir,
                    quota,
                    &name,
                    tenant,
                    // An erasure ends with this tenant holding nothing, and
                    // that is the only number that can be right afterwards -
                    // certainly not the sum of the physical fragments its
                    // indexes happened to be laid out in. The coordinator
                    // that drives the fan-out states it once, when every
                    // shard has answered.
                    DropCredit::Coordinator,
                    tier,
                    mmap_tier,
                    mmap_graph,
                )
                .await
                {
                    Ok(true) => dropped += 1,
                    // Lost a race with a concurrent drop: already gone, fine.
                    Ok(false) => {}
                    Err(e) => return ShardResp::Err(e),
                }
            }

            // Now the KV sweep over the whole tenant prefix.
            match sweep_prefix(vlog, &tenant.to_le_bytes(), durability).await {
                Ok(erased) => ShardResp::Erased {
                    vindexes: dropped,
                    keys: erased,
                },
                Err(e) => ShardResp::Err(e),
            }
        }
        ShardReq::ErasePrefix {
            tenant,
            subject,
            durability,
        } => {
            if tenant == 0 {
                return ShardResp::Err("cannot erase under the anonymous tenant (0)".to_owned());
            }
            // Subject-scoped erase: the tenant's 16 bytes followed by the
            // caller's subject bytes. No vindex drop - a subject is a slice of a
            // tenant, not the tenant; the caller reclaims a subject's vectors
            // with vdel + vindex_consolidate. This sweeps only the app's own KV
            // keys namespaced under the subject.
            let mut prefix = tenant.to_le_bytes().to_vec();
            prefix.extend_from_slice(&subject);
            match sweep_prefix(vlog, &prefix, durability).await {
                Ok(erased) => ShardResp::Erased {
                    vindexes: 0,
                    keys: erased,
                },
                Err(e) => ShardResp::Err(e),
            }
        }
        ShardReq::Reclaim => match vlog.reclaim_all_dead().await {
            Ok(freed) => ShardResp::Reclaimed(freed),
            Err(e) => ShardResp::Err(format!("reclaim failed: {e}")),
        },
        ShardReq::Vset {
            name,
            id,
            vector,
            tenant,
            limit,
            effect,
            version,
            payload,
        } => {
            // Outer read to look up the entry; clone the Arc and drop the
            // outer lock before taking the per-vindex write. This lets
            // another vindex's ops run in parallel with this one.
            let entry = get_or_reopen(vindexes, dir, tier, mmap_tier, mmap_graph, &name).await;
            let Some(arc) = entry else {
                return ShardResp::Err(format!("vindex '{name}' not found"));
            };

            // ── Admission, under the write lock ──────────────────────────
            //
            // Everything that can REFUSE the write happens here, before a
            // single byte is staged: a refusal has to leave nothing behind,
            // and the cheapest way to guarantee that is to have written
            // nothing yet. The lock is dropped before the staging below, as
            // it must be - a parking_lot guard cannot be held across an
            // await - and the coordinator holds this row's stripe for the
            // whole span, so nothing else can write this id in between.
            let admitted = {
                let mut idx = arc.write();
                if idx.backend.dim() != vector.len() {
                    Err(format!(
                        "vindex '{name}' dim {} but vector has {}",
                        idx.backend.dim(),
                        vector.len()
                    ))
                } else if version.is_some_and(|v| v < idx.backend.version_of(id).get()) {
                    // A relocation carrying a copy this shard has already
                    // moved past. The engine would refuse the vector by
                    // itself; what it cannot refuse is the rest of the write.
                    // Refusing it HERE is what keeps the quota unspent, the
                    // payload postings untouched and the blob unwritten - the
                    // row that stands must not end up described by the
                    // payload of the value it replaced.
                    Ok(None)
                } else {
                    // Quota: only a NEW id consumes a slot. Reserve before the
                    // insert (race-free under this write lock) so an
                    // over-limit insert is rejected without storing; an
                    // overwrite never touches the quota. The effect decides
                    // first: a moved or replicated row is new to THIS shard
                    // and not to the tenant.
                    //
                    // `None` means nobody upstream is tracking this row's
                    // versions - the hash-placed path, where the row never
                    // moves - so this shard allocates one past whatever it
                    // holds. Never zero: a legacy version would tie with the
                    // copy already here and hand the decision back to write
                    // order.
                    let previous = idx.backend.version_of(id).get();
                    let existed_before = idx.backend.contains(id);
                    // `next()`, not `previous + 1`: it saturates. An index
                    // that has issued 2^64 versions for one id has other
                    // problems, but wrapping to zero would turn every later
                    // row into a legacy one, and legacy loses to nothing -
                    // so the write after the wrap would be silently dropped.
                    let version =
                        version.unwrap_or_else(|| VectorVersion::new(previous).next().get());
                    let was_new = effect.charges(!existed_before);
                    if was_new
                        && let Some(max) = limit
                        && quota.try_add(tenant, 1, max).is_err()
                    {
                        return ShardResp::Err("tenant vector quota exceeded".to_owned());
                    }
                    // Memory admission, same shape as the quota above and for
                    // the same reason: refuse BEFORE storing. The cost is what
                    // the delta will hold - f32 per dimension - and it is
                    // charged against a promise covering the whole delta, not
                    // this row, so the bound is on the buffer rather than on
                    // the request.
                    let want = idx.backend.resident_bytes()
                        + (vector.len() * std::mem::size_of::<f32>()) as u64;
                    if let Err(rejected) = idx.reserve_memory(memory, want) {
                        if was_new && limit.is_some() {
                            quota.sub(tenant, 1);
                        }
                        skeg_telemetry::tick_counter(skeg_telemetry::Counter::MemoryRefused);
                        return ShardResp::Err(format!(
                            "BACKPRESSURE out of memory budget: {rejected}"
                        ));
                    }
                    Ok(Some(Admitted {
                        version,
                        previous: existed_before.then_some(previous),
                        charged: was_new && limit.is_some(),
                        generation: idx.generation,
                    }))
                }
            };
            let admitted = match admitted {
                Err(e) => return ShardResp::Err(e),
                // Not an error. The caller is a relocation and what it wants
                // is for the newest copy of the row to be the one that
                // stands, which it is.
                Ok(None) => return ShardResp::Done,
                Ok(Some(a)) => a,
            };
            let scope = BlobScope {
                tenant,
                generation: admitted.generation,
                name: &name,
            };
            let refund = |a: &Admitted| {
                if a.charged {
                    quota.sub(tenant, 1);
                }
            };

            // ── W1: stage the blob, before anything is published ─────────
            //
            // The blob goes to the key of the version this write is ABOUT to
            // take, which no live row carries yet: a search walks live ids and
            // a payload read is keyed by the row's version, so nothing can
            // reach it. That is what makes it safe to write first - and
            // writing first is what makes the pair atomic without an fsync,
            // because the record that publishes the row is then the only step
            // that has to survive.
            let staged = match &payload {
                Some(blob) => stage_payload_blob(vlog, scope, id, admitted.version, blob).await,
                // A payload-less overwrite keeps the payload the row already
                // had, which means CARRYING it to the new version's key.
                // Skipped for a row this shard did not already hold - every
                // insert, and every arriving relocation - so those pay no
                // lookup for a blob that cannot exist.
                None => match admitted.previous.filter(|&v| v != admitted.version) {
                    Some(old) => match read_payload_blob(vlog, scope, id, old).await {
                        Ok(Some(blob)) => {
                            skeg_telemetry::tick_counter(
                                skeg_telemetry::Counter::PayloadBlobsCarriedForward,
                            );
                            stage_payload_blob(vlog, scope, id, admitted.version, &blob).await
                        }
                        Ok(None) => Ok(false),
                        Err(e) => Err(format!("vset payload failed: {e}")),
                    },
                    None => Ok(false),
                },
            };
            let staged = match staged {
                Ok(staged) => staged,
                Err(e) => {
                    refund(&admitted);
                    return ShardResp::Err(e);
                }
            };
            let payload_ref = if staged {
                PayloadRef::Blob(admitted.version)
            } else {
                PayloadRef::Cleared
            };

            // ── W2: the commit point ─────────────────────────────────────
            let committed = {
                let mut idx = arc.write();
                crate::fp_at!(crate::failpoint::WriteFailpoint::VectorCommit, &name, {
                    refund(&admitted);
                    ShardResp::Err("failpoint: vector commit refused".to_owned())
                });
                let result = idx.backend.insert(
                    id,
                    &vector,
                    VectorVersion::new(admitted.version),
                    payload_ref,
                );
                match result {
                    Ok(()) => {
                        // ── W3: post-commit ──────────────────────────────
                        //
                        // The record is durable. Indexing the payload's fields
                        // is what makes it FILTERABLE, and a filter that
                        // cannot see a row is a wrong answer - but it is a
                        // recoverable one: the postings rebuild from the blobs
                        // at the next open. Reporting the write as failed
                        // would not be recoverable, because the client would
                        // believe a durable row is not there.
                        let applied: Result<(), &'static str> = crate::fp_check_at!(
                            crate::failpoint::WriteFailpoint::PayloadApply,
                            &name,
                            Err("failpoint: payload apply refused")
                        );
                        match applied {
                            Ok(()) => {
                                if let Some(blob) = &payload {
                                    idx.payload.upsert(id, parse_fields(blob));
                                }
                            }
                            Err(e) => {
                                skeg_telemetry::tick_counter(
                                    skeg_telemetry::Counter::PayloadPostCommitFailures,
                                );
                                tracing::error!(
                                    index = %name,
                                    id,
                                    error = e,
                                    "vector committed but its payload fields were not indexed: \
                                     filtered searches will miss this row until the postings \
                                     are rebuilt"
                                );
                            }
                        }
                        Ok(())
                    }
                    Err(e) => {
                        refund(&admitted);
                        Err(format!("vset failed: {e}"))
                    }
                }
            };
            if let Err(e) = committed {
                return ShardResp::Err(e);
            }

            // ── W4: reclaim what the commit superseded ───────────────────
            //
            // Past the commit point. Nothing here may fail the call: the row
            // is durable and readable, and telling the client otherwise
            // invites a retry of a write that has already happened. What is
            // left behind is a blob no live row names - reclaimable garbage,
            // collected at the next open.
            if let Some(old) = admitted.previous.filter(|&v| v != admitted.version) {
                let refused = crate::fp_check_at!(
                    crate::failpoint::WriteFailpoint::PayloadPostCommitCleanup,
                    &name,
                    Err("failpoint: post-commit blob cleanup refused".to_owned())
                );
                let outcome = match refused {
                    Err(e) => Err(e),
                    Ok(()) => {
                        let mut out = Ok(());
                        for key in std::iter::once(scope.key(id, old)).chain(scope.legacy_key(id)) {
                            if let Err(e) = vlog.del(&key, PAYLOAD_DURABILITY).await {
                                out = Err(format!("{e}"));
                                break;
                            }
                        }
                        out
                    }
                };
                if let Err(e) = outcome {
                    skeg_telemetry::tick_counter(
                        skeg_telemetry::Counter::PayloadPostCommitFailures,
                    );
                    tracing::error!(
                        index = %name,
                        id,
                        error = %e,
                        "vector committed but the payload blob it superseded was not \
                         reclaimed; it will be collected at the next open"
                    );
                }
            }
            ShardResp::Done
        }
        ShardReq::Vget { name, id } => {
            let entry = get_or_reopen(vindexes, dir, tier, mmap_tier, mmap_graph, &name).await;
            match entry {
                None => ShardResp::Err(format!("vindex '{name}' not found")),
                Some(arc) => {
                    let idx = arc.read();
                    match idx.backend.get(id) {
                        Ok(v) => ShardResp::Vector(v),
                        Err(e) => ShardResp::Err(format!("vget failed: {e}")),
                    }
                }
            }
        }
        ShardReq::SampleVectors { name, count } => {
            let entry = get_or_reopen(vindexes, dir, tier, mmap_tier, mmap_graph, &name).await;
            match entry {
                None => ShardResp::Err(format!("vindex '{name}' not found")),
                Some(arc) => {
                    let idx = arc.read();
                    let dim = idx.backend.dim() as u32;
                    let ids = idx.backend.live_ids();
                    let stride = (ids.len() / count.max(1)).max(1);
                    let mut out = Vec::with_capacity(count.min(ids.len()) * dim as usize);
                    let mut err = None;
                    for id in ids.into_iter().step_by(stride).take(count) {
                        match idx.backend.get(id) {
                            Ok(Some(v)) => out.extend_from_slice(&v),
                            Ok(None) => {}
                            Err(e) => {
                                err = Some(format!("sample read failed: {e}"));
                                break;
                            }
                        }
                    }
                    match err {
                        Some(e) => ShardResp::Err(e),
                        None => ShardResp::Sample(out, dim),
                    }
                }
            }
        }
        ShardReq::CollectMoves {
            name,
            centroids,
            own,
            after,
            limit,
            tenant,
        } => {
            let entry = get_or_reopen(vindexes, dir, tier, mmap_tier, mmap_graph, &name).await;
            match entry {
                None => ShardResp::Err(format!("vindex '{name}' not found")),
                Some(arc) => {
                    // Read phase under the read lock; payload blobs from the
                    // vlog after, so the lock never spans an await.
                    let (mut batch, cursor, generation) = {
                        let idx = arc.read();
                        let mut ids = idx.backend.live_ids();
                        ids.sort_unstable();
                        let mut out: Vec<MoveRow> = Vec::new();
                        let mut cursor = None;
                        for &id in ids.iter().filter(|&&i| i > after) {
                            match idx.backend.get(id) {
                                Ok(Some(v)) => {
                                    let owner = centroids.assign(&v) as u8;
                                    if owner != own {
                                        let version = idx.backend.version_of(id).get();
                                        out.push((id, v, None, owner, version));
                                    }
                                }
                                Ok(None) => {}
                                Err(e) => {
                                    return ShardResp::Err(format!("reshard read failed: {e}"));
                                }
                            }
                            if out.len() >= limit {
                                cursor = Some(id);
                                break;
                            }
                        }
                        (out, cursor, idx.generation)
                    };
                    let scope = BlobScope {
                        tenant,
                        generation,
                        name: &name,
                    };
                    for (id, _, payload, _, version) in &mut batch {
                        match read_payload_blob(vlog, scope, *id, *version).await {
                            Ok(b) => *payload = b,
                            Err(e) => {
                                return ShardResp::Err(format!("reshard payload read failed: {e}"));
                            }
                        }
                    }
                    ShardResp::Moves(batch, cursor)
                }
            }
        }
        ShardReq::CollectBoundary {
            name,
            centroids,
            after,
            limit,
            tau,
            tenant,
        } => {
            let entry = get_or_reopen(vindexes, dir, tier, mmap_tier, mmap_graph, &name).await;
            match entry {
                None => ShardResp::Err(format!("vindex '{name}' not found")),
                Some(arc) => {
                    let (mut batch, cursor, generation) = {
                        let idx = arc.read();
                        let mut ids = idx.backend.live_ids();
                        ids.sort_unstable();
                        let mut out: Vec<MoveRow> = Vec::new();
                        let mut cursor = None;
                        for &id in ids.iter().filter(|&&i| i > after) {
                            match idx.backend.get(id) {
                                Ok(Some(v)) => {
                                    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                                    let qn: Vec<f32> = v.iter().map(|x| x / n).collect();
                                    let mut best = (f32::NEG_INFINITY, 0usize);
                                    let mut second = (f32::NEG_INFINITY, 0usize);
                                    for j in 0..centroids.k {
                                        let c = &centroids.centroids
                                            [j * centroids.dim..(j + 1) * centroids.dim];
                                        let s: f32 = qn.iter().zip(c).map(|(a, b)| a * b).sum();
                                        if s > best.0 {
                                            second = best;
                                            best = (s, j);
                                        } else if s > second.0 {
                                            second = (s, j);
                                        }
                                    }
                                    if best.0 - second.0 < tau {
                                        let version = idx.backend.version_of(id).get();
                                        out.push((id, v, None, second.1 as u8, version));
                                    }
                                }
                                Ok(None) => {}
                                Err(e) => {
                                    return ShardResp::Err(format!("overlap read failed: {e}"));
                                }
                            }
                            if out.len() >= limit {
                                cursor = Some(id);
                                break;
                            }
                        }
                        (out, cursor, idx.generation)
                    };
                    let scope = BlobScope {
                        tenant,
                        generation,
                        name: &name,
                    };
                    for (id, _, payload, _, version) in &mut batch {
                        match read_payload_blob(vlog, scope, *id, *version).await {
                            Ok(b) => *payload = b,
                            Err(e) => {
                                return ShardResp::Err(format!("overlap payload read failed: {e}"));
                            }
                        }
                    }
                    ShardResp::Moves(batch, cursor)
                }
            }
        }
        ShardReq::LiveIds { name } => {
            let entry = get_or_reopen(vindexes, dir, tier, mmap_tier, mmap_graph, &name).await;
            match entry {
                Some(arc) => {
                    let idx = arc.read();
                    ShardResp::LiveIds(LiveIdsAnswer::Held {
                        ids: idx
                            .backend
                            .live_ids_with_versions()
                            .into_iter()
                            .map(|(id, v)| (id, v.get()))
                            .collect(),
                        high_water: idx.backend.max_version().get(),
                    })
                }
                // Absent, or present and unopenable? The CATALOGUE decides,
                // the same way a drop decides it. `get_or_reopen` gives the
                // same `None` for both.
                None => match read_registry(dir) {
                    Ok(entries) if entries.iter().any(|e| e.name == name) => {
                        ShardResp::LiveIds(LiveIdsAnswer::Unreadable(format!(
                            "vindex '{name}' is registered on this shard and did not open"
                        )))
                    }
                    Ok(_) => ShardResp::LiveIds(LiveIdsAnswer::Absent),
                    Err(e) => ShardResp::LiveIds(LiveIdsAnswer::Unreadable(format!(
                        "vindex registry unreadable, so whether '{name}' is here is unknown: {e}"
                    ))),
                },
            }
        }
        ShardReq::PayloadBlobs { tenant, name } => {
            ShardResp::Count(count_payload_blobs(vlog, tenant, &name))
        }
        ShardReq::StillCurrent { name, id, version } => {
            let entry = get_or_reopen(vindexes, dir, tier, mmap_tier, mmap_graph, &name).await;
            match entry {
                None => ShardResp::Err(format!("vindex '{name}' not found")),
                Some(arc) => {
                    let idx = arc.read();
                    // Live AND unchanged. Not "version has not advanced": a
                    // fold drops a tombstone along with the rows it masked, so
                    // a deleted row can read back as version 0 - lower than
                    // what the mover carries, not higher. Liveness is the
                    // question, and the version answers the other half.
                    ShardResp::Existed(
                        idx.backend.contains(id) && idx.backend.version_of(id).get() == version,
                    )
                }
            }
        }
        ShardReq::GraphSample { name, count } => {
            let entry = get_or_reopen(vindexes, dir, tier, mmap_tier, mmap_graph, &name).await;
            match entry {
                None => ShardResp::Err(format!("vindex '{name}' not found")),
                Some(arc) => {
                    let idx = arc.read();
                    match &idx.backend {
                        VectorBackend::Disk(i) => {
                            let (nodes, edges) = i.graph_sample(count);
                            ShardResp::Graph(nodes, edges)
                        }
                        VectorBackend::Flat(_) => {
                            ShardResp::Err("flat backend has no graph".into())
                        }
                    }
                }
            }
        }
        ShardReq::Vdel {
            name,
            id,
            tenant,
            effect,
            version,
        } => {
            let entry = get_or_reopen(vindexes, dir, tier, mmap_tier, mmap_graph, &name).await;
            match entry {
                None => ShardResp::Err(format!("vindex '{name}' not found")),
                Some(arc) => {
                    // Delete the vector and drop its payload postings under one
                    // lock; the blob in the vLog is reclaimed by the await below.
                    let (result, held, generation) = {
                        let mut g = arc.write();
                        // The version of the copy being removed, read BEFORE
                        // the tombstone lands on top of it: after the delete
                        // `version_of` answers about the tombstone, and the
                        // blob is filed under the row.
                        let held = g.backend.version_of(id).get();
                        let generation = g.generation;
                        // Saturating, same reason as the write path.
                        let version = version
                            .map_or_else(|| VectorVersion::new(held).next(), VectorVersion::new);
                        let r = g.backend.delete(id, version);
                        if matches!(r, Ok(true)) {
                            g.payload.remove(id);
                        }
                        (r, held, generation)
                    };
                    match result {
                        Ok(existed) => {
                            if existed {
                                if effect.credits() {
                                    quota.sub(tenant, 1);
                                }
                                let scope = BlobScope {
                                    tenant,
                                    generation,
                                    name: &name,
                                };
                                // Past the commit point. The tombstone is
                                // durable and the row is gone from every
                                // reader; reclaiming its blob is cleanup, and
                                // reporting a failure here told the client a
                                // delete had not happened when it had. What is
                                // left is a blob no live row names -
                                // reclaimable garbage, collected at the next
                                // open. Harmless when the id never had one
                                // (del returns false).
                                let outcome: Result<(), String> = match crate::fp_check_at!(
                                    crate::failpoint::WriteFailpoint::VdelBlobDelete,
                                    &name,
                                    Err("failpoint: blob reclamation refused".to_owned())
                                ) {
                                    Err(e) => Err(e),
                                    Ok(()) => {
                                        let mut out = Ok(());
                                        for key in std::iter::once(scope.key(id, held))
                                            .chain(scope.legacy_key(id))
                                        {
                                            if let Err(e) = vlog.del(&key, PAYLOAD_DURABILITY).await
                                            {
                                                out = Err(format!("{e}"));
                                                break;
                                            }
                                        }
                                        out
                                    }
                                };
                                if let Err(e) = outcome {
                                    skeg_telemetry::tick_counter(
                                        skeg_telemetry::Counter::PayloadPostCommitFailures,
                                    );
                                    tracing::error!(
                                        index = %name,
                                        id,
                                        error = %e,
                                        "row deleted but its payload blob was not reclaimed; \
                                         it will be collected at the next open"
                                    );
                                }
                            }
                            ShardResp::Existed(existed)
                        }
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
            tenant,
            want_payload,
            filter,
        } => {
            // One path whether or not there is a worker pool. The pooled and
            // inline versions used to be written out separately, which meant a
            // fix to one could miss the other and only show up in production,
            // where `workers > 0`, and never in a dev run, where it is 0.
            let Some(arc) = get_or_reopen(vindexes, dir, tier, mmap_tier, mmap_graph, &name).await
            else {
                return ShardResp::Err(format!("vindex '{name}' not found"));
            };
            // A filter reads the payload index; rebuild it from blobs first
            // (async, before the blocking walk) if this vindex was just
            // recovered from disk.
            if filter.is_some()
                && let Err(e) = ensure_payload_loaded(
                    vlog,
                    &dir.join(format!("vindex-{name}")),
                    &arc,
                    tenant,
                    &name,
                    false,
                )
                .await
            {
                return ShardResp::Err(e);
            }
            // The walk runs under the write lock; the guard is dropped before
            // any payload `await`, on both routes.
            //
            // Cloned because the pooled route MOVES `arc` into the worker
            // closure, and the versions have to be read from the same entry
            // after the walk.
            let versioned = arc.clone();
            let hits = match vsearch_pool {
                Some(pool) => {
                    let walk_name = name.clone();
                    let reply = match pool.submit(move || {
                        search_vindex(&mut arc.write(), &walk_name, &query, k, l_search, &filter)
                    }) {
                        Ok(reply) => reply,
                        Err(e) => return ShardResp::Err(e.to_owned()),
                    };
                    match reply.await {
                        Ok(inner) => inner,
                        Err(_) => return ShardResp::Err("vsearch worker task failed".to_owned()),
                    }
                }
                None => search_vindex(&mut arc.write(), &name, &query, k, l_search, &filter),
            };
            match hits {
                Err(e) => ShardResp::Err(e),
                // Fetch a payload per local hit; some get trimmed by the global
                // top-k merge, but k is small. Route the final-k by id if it
                // ever bites.
                Ok(hits) => {
                    // One lookup per local hit, k of them, off the walk and
                    // before any await: the merge upstream cannot rank two
                    // copies of an id without them.
                    let (generation, versions): (IndexGeneration, Vec<u64>) = {
                        let idx = versioned.read();
                        (
                            idx.generation,
                            hits.iter()
                                .map(|&(id, _)| idx.backend.version_of(id).get())
                                .collect(),
                        )
                    };
                    let scope = BlobScope {
                        tenant,
                        generation,
                        name: &name,
                    };
                    let hits: Vec<(u64, f32, u64)> = hits
                        .into_iter()
                        .zip(versions.iter().copied())
                        .map(|((id, score), version)| (id, score, version))
                        .collect();
                    match attach_payloads(vlog, scope, hits, want_payload).await {
                        Ok(out) => ShardResp::Vsearch(
                            out.into_iter()
                                .zip(versions)
                                .map(|((id, score, blob), version)| (id, score, blob, version))
                                .collect(),
                        ),
                        Err(e) => ShardResp::Err(e),
                    }
                }
            }
        }

        ShardReq::Get(key, tenant) => match vlog.tenant(tenant).get(&key).await {
            Ok(v) => ShardResp::Value(v),
            Err(e) => ShardResp::Err(e.to_string()),
        },
        ShardReq::Set(key, val, dur, tenant, disk_limit) => {
            match vlog
                .tenant(tenant)
                .with_disk_limit(disk_limit)
                .set(&key, &val, dur)
                .await
            {
                Ok(()) => ShardResp::Done,
                Err(e) => ShardResp::Err(e.to_string()),
            }
        }
        ShardReq::SetMany(pairs, dur) => {
            let refs: Vec<(&[u8], &[u8])> = pairs
                .iter()
                .map(|(k, v)| (k.as_ref(), v.as_ref()))
                .collect();
            match vlog.set_many(&refs, dur).await {
                Ok(()) => ShardResp::Done,
                Err(e) => ShardResp::Err(e.to_string()),
            }
        }
        ShardReq::Append(key, val, dur, tenant, disk_limit) => {
            match vlog
                .tenant(tenant)
                .with_disk_limit(disk_limit)
                .append(&key, &val, dur)
                .await
            {
                Ok(len) => ShardResp::Len(len),
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
                    let b = &entry.read().backend;
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

/// One row a reshard or an overlap is relocating: `(id, vector, payload,
/// destination shard, version)`. The version is the copy the SOURCE held when
/// the batch was collected, which is what the destination stores and what
/// tells the mover the row has been written again since.
type MoveRow = (u64, Vec<f32>, Option<Bytes>, u8, u64);

/// What a shard has to say when asked to enumerate an index.
///
/// `get_or_reopen` answers `None` for two completely different states - a name
/// this shard never had, and a name it has whose files will not open (it logs
/// the I/O error and swallows it) - and a plain error string flattened them
/// back together at the caller. Only one of the two has a safe answer, so they
/// are different variants.
enum LiveIdsAnswer {
    /// Not in this shard's registry. Normal: a routed index need not exist on
    /// every shard, and a create can still be in flight.
    Absent,
    /// Registered here and it did not open. The shard HAS rows and cannot say
    /// which, so nothing derived from this answer can be trusted.
    Unreadable(String),
    /// Live ids with the version of the copy held here, plus this shard's
    /// high-water version for the index - which the live rows do not give,
    /// since a tombstone can hold it.
    Held {
        ids: Vec<(u64, u64)>,
        high_water: u64,
    },
}

/// One hit from one shard's fragment of a search: `(id, cosine, payload,
/// version)`. The version is read from the shard that produced the hit, and it
/// is what the global merge uses to tell two copies of one id apart - the
/// owner map cannot, when it is the thing that is out of date.
type VsearchHit = (u64, f32, Option<Bytes>, u64);

/// id -> (primary shard, optional replica shard, version) for one routed
/// vindex.
///
/// The version is the version of the PRIMARY copy: what the map is asserting
/// is "shard p holds copy v of this row, and it is the live one". It is
/// derived, never persisted - a rebuild reads it back off the shards - and it
/// is what lets a relocation tell "this row is where I left it" from "this row
/// has been written again since I read it".
type OwnerMap = ahash::AHashMap<u64, (u8, Option<u8>, u64)>;

struct ShardSetInner {
    senders: Vec<Sender<ShardMsg>>,
    handles: Vec<JoinHandle<()>>,
    vsearch_admission: Option<Arc<Semaphore>>,
    n: usize,
    /// Per-tenant vector counter, shared across shards and consulted on
    /// VSET/VDEL/VINDEX.DROP to enforce `max_vectors`.
    quota: Arc<crate::quota::TenantVectorQuota>,
    /// Per-tenant live disk-byte counter, shared with every shard's `VLog` so
    /// the `max_disk_bytes` quota is global per tenant.
    disk_counter: skeg_core::SharedTenantDisk,
    /// Shard-set root: sidecars that describe the SET (semantic routers)
    /// live here, not inside a shard.
    root: std::path::PathBuf,
    /// Loaded semantic routers per vindex name (scoped names included).
    routers: parking_lot::RwLock<HashMap<String, Arc<crate::router::Router>>>,
    /// id -> owner shard, per semantically-resharded vindex. Hash placement
    /// makes an id's shard computable; semantic placement does not, so the
    /// set keeps the map (8B+overhead per id, rebuilt at open from the
    /// shards' live id sets) and point ops stay O(1) instead of broadcast.
    owners: parking_lot::RwLock<HashMap<String, OwnerMap>>,
    /// Per-vindex version allocator, so a user write gets a version that beats
    /// every copy of the row anywhere in the set.
    ///
    /// It has to be here rather than on a shard: a shard's own counter only
    /// knows the copies IT holds, and a row that moves to a shard which has
    /// never seen it would be handed a version below the one it already
    /// carried. Seeded past the highest version observed whenever the owner
    /// map is rebuilt, and floored by the version already recorded for the row
    /// on every allocation, so a value read off disk can never be reissued.
    ///
    /// Derived state, like `owners`: dropped with the index and rebuilt on
    /// demand.
    versions: parking_lot::RwLock<HashMap<String, Arc<std::sync::atomic::AtomicU64>>>,
    /// Process-wide memory admission. Kept here so `SKEG.STATS` can report the
    /// budget: a ceiling that is enforced and not readable leaves an operator
    /// to discover it from the refusals.
    memory: Arc<crate::memory::MemoryGovernor>,
    /// Serialises catalogue fan-outs (CREATE/DROP). They record an intent,
    /// touch every shard and clear it; two of them interleaving would read and
    /// write that one file underneath each other. It also removes the race
    /// where two CREATEs of the same name both find it absent.
    catalog: tokio::sync::Mutex<()>,
    /// Per-id serialisation for routed point ops (review finding): a routed
    /// vset/vdel is read-await-write on `owners` with shard calls between, so
    /// concurrent ops on the SAME id could interleave into an untracked
    /// duplicate or a wrong-shard entry. Ops on one id take its stripe; ops on
    /// different ids run free; hash-routed vindexes never take a stripe.
    owner_locks: Vec<tokio::sync::Mutex<()>>,
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

    /// Like [`open_mode`](Self::open_mode), with optional dedicated VSEARCH
    /// workers per shard. `workers == 0` keeps searches inline. `workers > 0`
    /// creates that many workers per shard and rejects overload before scatter.
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
        let memory =
            Arc::new(crate::memory::MemoryGovernor::from_env().map_err(std::io::Error::other)?);
        Self::open_full_with_memory(
            base_dir, n_shards, read_only, tier, workers, mmap_tier, mmap_graph, memory,
        )
    }

    /// [`ShardSet::open_mode_full_mmap`] with the memory governor supplied, so a
    /// test can hand it a budget it controls instead of the machine's.
    ///
    /// # Errors
    ///
    /// Returns an IO error if the layout refuses, a shard fails to open, or a
    /// worker thread cannot be spawned.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn open_full_with_memory(
        base_dir: &Path,
        n_shards: usize,
        read_only: bool,
        tier: QuantKind,
        workers: usize,
        mmap_tier: bool,
        mmap_graph: bool,
        memory: Arc<crate::memory::MemoryGovernor>,
    ) -> std::io::Result<Self> {
        // The store declares its own shape, and this is the ONE place that
        // asks. A caller's `n_shards` is a request, not an authority: if the
        // manifest disagrees, or the directories disagree with the manifest,
        // the open refuses instead of picking a winner. Read-only opens never
        // write - serve mode runs over copies an operator may have mounted
        // read-only, and over stores older than the manifest itself.
        //
        // The request goes through the same bound as the file does. It reaches
        // the same loop below - one OS thread and one channel per shard - so a
        // caller asking for ten million is refused for the same reason a file
        // declaring ten million is, and by the same code.
        let mode = if read_only {
            crate::layout_manifest::OpenMode::ReadOnly
        } else {
            crate::layout_manifest::OpenMode::ReadWrite {
                requested_shards: crate::layout_manifest::ShardCount::checked(
                    n_shards,
                    "this open",
                )?,
            }
        };
        let n_shards = crate::layout_manifest::LayoutManifest::open_or_migrate(base_dir, mode)?
            .shard_count()
            .get();
        let mut senders = Vec::with_capacity(n_shards);
        let mut handles = Vec::with_capacity(n_shards);
        // Readiness barrier: each shard reports its startup outcome once recovery
        // is done. We block below until all have, so `open` (and the caller's
        // bind-after-open) means "queryable": no phantom stall on the first
        // query. A shard that fails to open its store reports `Err` and aborts
        // the whole open.
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<ShardReady, String>>();
        // One vector quota shared across all shards: a tenant's vectors are
        // spread over shards by id, so the counter must aggregate cross-shard.
        let quota = Arc::new(crate::quota::TenantVectorQuota::new());
        // The semantic routers, loaded BEFORE the shards start rather than
        // after they report ready: which indexes are routed decides how each
        // shard has to report its rows for the quota rebuild, and a shard
        // cannot know - the sidecars sit beside the shard directories, not
        // inside them.
        let routers = load_routers(base_dir)?;
        let routed: Arc<HashSet<String>> = Arc::new(routers.keys().cloned().collect());
        let vsearch_admission = (workers > 0).then(|| Arc::new(Semaphore::new(workers)));
        // One disk counter shared across all shards, so the disk quota is global
        // per tenant (a tenant's keys spread over shards by hash).
        let disk_counter = skeg_core::new_shared_disk();
        // Catalogue fan-outs that never finished, decided ONCE and here, because
        // the decision needs to see every shard's registry and no shard can.
        //
        // `k` is the number of shards still listing the name. `k < n` means the
        // fan-out reached some and not others, and undoing a create is the same
        // action as finishing a drop: remove it. `k == n` means nothing was
        // removed anywhere - a create that succeeded unacknowledged, which is
        // undone, or a DROP THE STORE REFUSED, whose index is whole and whose
        // caller was told it failed. Removing that one would turn a refused
        // drop into a silent deletion.
        //
        // In a read-only open nothing is decided or written; the list is every
        // in-flight name, and the shards decline to serve them.
        let mut blocked: Vec<String> = Vec::new();
        let in_flight: Vec<String> = {
            let mut remove = Vec::new();
            for (op, name) in crate::catalog_intent::pending(base_dir)? {
                let mut listed = 0usize;
                for id in 0..n_shards {
                    // Fail-closed: a registry that will not parse cannot be
                    // counted as "does not have it", which would turn a refused
                    // drop into a deletion. The shard open refuses on it too.
                    if read_registry(&base_dir.join(format!("shard-{id}")))?
                        .iter()
                        .any(|e| e.name == name)
                    {
                        listed += 1;
                    }
                }
                // A drop the store refused on EVERY shard removed nothing and
                // left its index whole. Everything else needs finishing.
                let settled = op == crate::catalog_intent::Op::Drop && listed == n_shards;
                if !settled {
                    if listed > 0 {
                        tracing::warn!(
                            index = %name,
                            listed_on = listed,
                            of = n_shards,
                            "resolving an unfinished catalogue operation by removing the index"
                        );
                        // An index still catalogued somewhere is a half-state a
                        // reader cannot fix, which is what a read-only open
                        // refuses over. At `listed == 0` there is nothing left
                        // to serve, so nothing to refuse.
                        blocked.push(name.clone());
                    }
                    // Kept even at `listed == 0`, where no catalogue entry is
                    // left to remove: a crash between that removal and the blob
                    // sweep leaves exactly this state, and dropping it here
                    // orphans those blobs for good.
                    remove.push(name);
                }
            }
            remove
        };
        // A read-only open can DECIDE - reading registries writes nothing - but
        // it cannot act, so it refuses rather than serve around a half-state.
        //
        // Skipping those names at open was tried and is NOT a guard: the first
        // query reopens the index lazily from the registry, so the store served
        // it anyway. Measured, `resident=0` right after the open and fifteen
        // rows out of the next search. A guard that reads as protection and is
        // not one is worse than no guard.
        if read_only && !blocked.is_empty() {
            return Err(std::io::Error::other(format!(
                "read-only open refused: {} has unfinished catalogue operations on {}; open it writable once to resolve them",
                base_dir.display(),
                blocked.join(", ")
            )));
        }

        for id in 0..n_shards {
            let dir = base_dir.join(format!("shard-{id}"));
            let in_flight = in_flight.clone();
            let (tx, rx) = tokio::sync::mpsc::channel::<ShardMsg>(SHARD_INBOX_CAPACITY);
            let quota = quota.clone();
            let memory = memory.clone();
            let disk_counter = disk_counter.clone();
            let routed = routed.clone();
            let ready_tx = ready_tx.clone();
            let handle = std::thread::Builder::new()
                .name(format!("skeg-shard-{id}"))
                .spawn(move || {
                    run_shard(
                        id,
                        dir,
                        rx,
                        read_only,
                        tier,
                        workers,
                        mmap_tier,
                        mmap_graph,
                        quota,
                        memory,
                        disk_counter,
                        in_flight,
                        routed,
                        ready_tx,
                    )
                })?;
            senders.push(tx);
            handles.push(handle);
        }
        drop(ready_tx); // only shard threads hold senders now
        let mut fatal: Option<String> = None;
        let mut reports: Vec<ShardReady> = Vec::with_capacity(n_shards);
        for _ in 0..n_shards {
            // Each shard signals exactly once: `Ok` when queryable, `Err` when
            // its store could not be opened. recover_vindexes never panics (bad
            // indexes are logged and skipped), so this always gets its n signals.
            // A hard panic before signalling (e.g. OOM) drops the sender and
            // yields a recv error, which we treat as a failed shard.
            match ready_rx.recv() {
                Ok(Ok(report)) => reports.push(report),
                Ok(Err(msg)) => {
                    fatal.get_or_insert(msg);
                }
                Err(_) => {
                    fatal.get_or_insert_with(|| "a shard died before signalling".to_owned());
                }
            }
        }
        if let Some(msg) = fatal {
            // Abort startup. Dropping the senders closes each shard's inbox, so
            // the shard threads exit and release their store locks; joining makes
            // that deterministic before we return (and free the lock for a retry).
            drop(senders);
            for h in handles {
                let _ = h.join();
            }
            return Err(std::io::Error::other(format!(
                "shard startup failed: {msg}"
            )));
        }
        // ── The vector quota, put back before anything can spend it ──────
        //
        // `TenantVectorQuota` is process state and used to start every open
        // empty, so a tenant sitting at its limit got its whole budget back
        // by restarting the server: the limit was a limit per uptime. The
        // shards have just counted what they hold; this is where the halves
        // are added up.
        //
        // Still inside the barrier. Every shard has reported and none has
        // served a request, `open` has not returned, and the listener binds
        // after it does - so no write is admitted against a count that is not
        // yet in place. Read-only opens report nothing and rebuild nothing:
        // they admit no writes.
        let mut per_tenant: HashMap<u128, u64> = HashMap::new();
        for report in &reports {
            for (name, rows) in &report.counts {
                *per_tenant.entry(tenant_of_scoped_index(name)).or_default() += rows;
            }
        }
        // A routed index is the case a per-shard count cannot answer: a
        // boundary replica is a second PHYSICAL copy of one LOGICAL row, and
        // a crash between a move's write to the destination and the delete of
        // its source leaves another. Adding the shards up would charge the
        // tenant twice for a row it has once - and a restart is exactly when
        // both of those states are on disk.
        //
        // So: one entry per logical id, across every shard that holds a copy,
        // which is the same rule `rebuild_owner_maps` applies when it decides
        // who is primary. The dedup is done here rather than by calling that
        // function because `open` is synchronous - it cannot await a request
        // round trip - and because the answer needed here is a cardinality,
        // not a placement.
        let mut ids: HashMap<&str, HashSet<u64>> = HashMap::new();
        for report in &reports {
            for (name, shard_ids) in &report.routed_ids {
                ids.entry(name.as_str())
                    .or_default()
                    .extend(shard_ids.iter().copied());
            }
        }
        for (name, ids) in ids {
            *per_tenant.entry(tenant_of_scoped_index(name)).or_default() += ids.len() as u64;
        }
        for (tenant, count) in per_tenant {
            quota.rebuild(tenant, count);
        }
        let set = Self {
            inner: Arc::new(ShardSetInner {
                senders,
                handles,
                vsearch_admission,
                n: n_shards,
                quota,
                disk_counter,
                root: base_dir.to_path_buf(),
                routers: parking_lot::RwLock::new(routers),
                owners: parking_lot::RwLock::new(HashMap::new()),
                versions: parking_lot::RwLock::new(HashMap::new()),
                memory: memory.clone(),
                catalog: tokio::sync::Mutex::new(()),
                owner_locks: (0..256).map(|_| tokio::sync::Mutex::new(())).collect(),
            }),
        };
        // Every shard has now applied the decision against its own registry (it
        // happens before it reports ready), so what is left is the
        // coordinator's own derived state and the record itself.
        //
        // The router sidecar goes with the index: it is what a recreated name
        // would inherit stale centroids and a stale owner map from, and this is
        // the one path that removes an index without going through
        // `vindex_drop`. A failure is logged, not fatal - the catalogue is
        // already consistent, and a stale sidecar is a lesser fault than a
        // store that refuses to open.
        if !read_only {
            for name in &in_flight {
                if let Err(e) = set.drop_router_state(name) {
                    error!("router state of resolved vindex '{name}' not removed: {e}");
                }
            }
            for (_, name) in crate::catalog_intent::pending(base_dir)? {
                crate::catalog_intent::clear(base_dir, &name)?;
            }
        }
        Ok(set)
    }

    /// The process-wide memory governor, for reporting.
    #[must_use]
    pub fn memory(&self) -> &Arc<crate::memory::MemoryGovernor> {
        &self.inner.memory
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
        self.set_scoped(key, value, durability, 0, None).await
    }

    async fn set_scoped(
        &self,
        key: &[u8],
        value: &[u8],
        durability: Durability,
        tenant: u128,
        disk_limit: Option<u64>,
    ) -> Result<(), ShardError> {
        let shard = shard_for(key, self.inner.n);
        let req = ShardReq::Set(
            Bytes::copy_from_slice(key),
            Bytes::copy_from_slice(value),
            durability,
            tenant,
            disk_limit,
        );
        match self.call(shard, req).await? {
            ShardResp::Done => Ok(()),
            ShardResp::Err(e) => Err(ShardError::Storage(e)),
            _ => Err(ShardError::Unavailable),
        }
    }

    /// Multi-key set. Keys are grouped by shard and each shard's group is
    /// written as one atomic [`VLog::set_many`] batch, so a shard's portion is
    /// all-or-nothing across a crash.
    ///
    /// It is NOT globally atomic: keys route to shards by hash, so a batch that
    /// spans shards is several independent per-shard commits with no cross-shard
    /// coordination. A single-shard deployment (or an MSET whose keys all hash
    /// to one shard) is fully atomic. Callers wanting a global transaction must
    /// keep the keys on one shard.
    ///
    /// # Errors
    ///
    /// Returns an error if a shard is unavailable or a write fails.
    pub async fn mset(
        &self,
        pairs: &[(&[u8], &[u8])],
        durability: Durability,
    ) -> Result<(), ShardError> {
        let mut by_shard: Vec<Vec<(Bytes, Bytes)>> = vec![Vec::new(); self.inner.n];
        for (key, value) in pairs {
            let s = shard_for(key, self.inner.n);
            by_shard[s].push((Bytes::copy_from_slice(key), Bytes::copy_from_slice(value)));
        }
        for (shard, batch) in by_shard.into_iter().enumerate() {
            if batch.is_empty() {
                continue;
            }
            match self
                .call(shard, ShardReq::SetMany(batch, durability))
                .await?
            {
                ShardResp::Done => {}
                ShardResp::Err(e) => return Err(ShardError::Storage(e)),
                _ => return Err(ShardError::Unavailable),
            }
        }
        Ok(())
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

    async fn append_scoped(
        &self,
        key: &[u8],
        value: &[u8],
        durability: Durability,
        tenant: u128,
        disk_limit: Option<u64>,
    ) -> Result<u64, ShardError> {
        let shard = shard_for(key, self.inner.n);
        let req = ShardReq::Append(
            Bytes::copy_from_slice(key),
            Bytes::copy_from_slice(value),
            durability,
            tenant,
            disk_limit,
        );
        match self.call(shard, req).await? {
            ShardResp::Len(n) => Ok(n),
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
            disk_limit: None,
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

    /// Create a vector index across all shards from a RAW, client-supplied
    /// name.
    ///
    /// `kind` is the raw wire byte: 0 = f32, 1 = int8, 2 = binary. `backend`
    /// is 0 = flat (in-RAM) or 1 = disk Vamana graph.
    ///
    /// This is the door the native binary protocol and every admin helper
    /// reach, and the name arrives exactly as the client wrote it, so the
    /// tenant-scope separator is refused here (see
    /// [`reject_scope_separator`]). A caller that has ALREADY scoped the name
    /// against an authenticated tenant uses [`vindex_create_scoped`] instead -
    /// the only way a `::` prefix can enter a map key.
    ///
    /// # Errors
    ///
    /// Returns an error for a name carrying `::`, a bad `dim`/`kind`/`backend`,
    /// a duplicate name, or an unavailable shard.
    ///
    /// [`vindex_create_scoped`]: Self::vindex_create_scoped
    pub async fn vindex_create(
        &self,
        name: &str,
        dim: u32,
        kind: u8,
        backend: u8,
    ) -> Result<(), ShardError> {
        reject_scope_separator(name)?;
        self.vindex_create_scoped(name, dim, kind, backend).await
    }

    /// [`vindex_create`](Self::vindex_create) for a name the caller has
    /// ALREADY tenant-scoped.
    ///
    /// The caller is promising that the `<32 hex>::` prefix in `name` was
    /// written by this server from a tenant id it authenticated - not copied
    /// out of anything a client sent. In-tree that is the RESP3 layer, which
    /// refuses the separator in the raw name before prepending its own. Pass a
    /// client's string here and every site that reads the owner back out of
    /// the key believes it.
    ///
    /// # Errors
    ///
    /// Returns an error for a bad `dim`/`kind`/`backend`, a duplicate name, or
    /// an unavailable shard.
    pub async fn vindex_create_scoped(
        &self,
        name: &str,
        dim: u32,
        kind: u8,
        backend: u8,
    ) -> Result<(), ShardError> {
        validate_vindex_name(name)?;
        if dim == 0 {
            return Err(ShardError::Storage(
                "vindex dim must be positive".to_owned(),
            ));
        }
        let kind = QuantKind::from_wire(kind)
            .ok_or_else(|| ShardError::Storage(format!("unknown vindex kind {kind}")))?;
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
        // Reject a dim the tier cannot pack here, before any shard touches the
        // builder: a bad (kind, dim) must be a clean error, not a panic that
        // kills the shard thread.
        if let Err(reason) = kind.validate_dim(dim) {
            return Err(ShardError::Storage(reason));
        }
        let name = name.to_owned();
        let _catalog = self.inner.catalog.lock().await;
        let root = self.inner.root.clone();
        let intent = |e: std::io::Error| ShardError::Storage(format!("catalogue intent: {e}"));

        // A name left in flight by an earlier fan-out is not a name to build
        // on: recreating it is exactly how one name ended up with two dims,
        // because the shard that failed the first time is the one free to
        // accept the second. It is resolved at the next open, not here.
        if crate::catalog_intent::pending(&root)
            .map_err(intent)?
            .iter()
            .any(|(_, n)| n == &name)
        {
            return Err(ShardError::Storage(format!(
                "vindex '{name}' has an unfinished catalogue operation; reopen                  the store to resolve it"
            )));
        }
        crate::catalog_intent::record(&root, crate::catalog_intent::Op::Create, &name)
            .map_err(intent)?;

        // ONE generation for the whole fan-out. Every shard records the same
        // incarnation, so a blob key means the same thing wherever the row
        // ends up - including after a reshard moves it.
        let generation = mint_generation();
        match self
            .broadcast(|| ShardReq::VindexCreate {
                name: name.clone(),
                dim,
                kind,
                disk,
                generation,
            })
            .await
        {
            Ok(()) => {
                // Cleared BEFORE the success is returned: while the record
                // stands, the operation is not acknowledged, and the next open
                // will undo it. Failing to clear therefore has to be reported
                // as a failure - which the next open then makes true.
                crate::catalog_intent::clear(&root, &name).map_err(intent)?;
                Ok(())
            }
            Err(e) => {
                // Undo on whichever shards committed. `require_present: false`:
                // the shard that refused the create is expected not to have it,
                // and calling that a failure would tell us nothing about the
                // ones being cleaned up.
                let tenant = unscope_key(&name).0;
                let undone = self
                    .broadcast(|| ShardReq::VindexDrop {
                        name: name.clone(),
                        tenant,
                        // The create never returned, so nothing was written
                        // to this name and there is nothing to credit either
                        // way. `Fragment` of an empty index is zero.
                        credit: DropCredit::Fragment,
                        require_present: false,
                    })
                    .await;
                match undone {
                    // Fully undone, so the record has nothing left to describe.
                    Ok(()) => {
                        crate::catalog_intent::clear(&root, &name).map_err(intent)?;
                    }
                    // Left dirty. The record stays and the next open finishes
                    // it; the caller still hears the original failure, which is
                    // the one it can act on.
                    Err(undo) => tracing::error!(
                        index = %name,
                        error = %undo,
                        "could not undo a partly created vindex; it is recorded                          and will be removed at the next open"
                    ),
                }
                Err(e)
            }
        }
    }

    /// Drop a vector index across all shards.
    ///
    /// # Errors
    ///
    /// Returns an error if the index does not exist or a shard is unavailable.
    pub async fn vindex_drop(&self, name: &str, tenant: u128) -> Result<(), ShardError> {
        validate_vindex_name(name)?;
        let name = name.to_owned();
        let _catalog = self.inner.catalog.lock().await;
        let root = self.inner.root.clone();
        let intent = |e: std::io::Error| ShardError::Storage(format!("catalogue intent: {e}"));

        // How many LOGICAL rows this index holds, read before anything is
        // removed - afterwards there is nothing left to count.
        //
        // For a routed index the shards must not credit their own fragments:
        // an overlap keeps a second physical copy of every boundary row, and
        // a crash between a move's write and its source delete leaves
        // another, so the fragments add up to more than the tenant ever
        // spent. The owner map is where the logical cardinality lives - one
        // entry per id however many copies exist, the same rule the open-time
        // rebuild applies - and `ensure_owner_map` builds it from the shards
        // if this uptime has not needed it yet.
        //
        // BEST EFFORT, deliberately: a map that will not build credits
        // nothing, and the drop still happens. A drop is destructive and sits
        // on the erasure path; refusing one because the accounting is
        // unavailable would trade a leak in a counter for a store that cannot
        // delete. The tenant then stays charged for rows it no longer has
        // until the next open counts again - the same bargain the orphan
        // branch of `drop_vindex` already strikes, logged the same way.
        let routed = self.inner.routers.read().contains_key(&name);
        let logical = if routed {
            match self.ensure_owner_map(&name).await {
                Ok(()) => self.inner.owners.read().get(&name).map(|m| m.len() as u64),
                Err(e) => {
                    error!(
                        "vector quota of '{name}' cannot be credited on drop: its owner \
                         map would not rebuild ({e}); the count is repaired at the next open"
                    );
                    None
                }
            }
        } else {
            None
        };
        // A drop cannot be undone - by the time one shard refuses, the others
        // have already deleted their data - so it is recorded to be FINISHED.
        crate::catalog_intent::record(&root, crate::catalog_intent::Op::Drop, &name)
            .map_err(intent)?;
        let outcome = self
            .broadcast(|| ShardReq::VindexDrop {
                name: name.clone(),
                tenant,
                credit: if routed {
                    DropCredit::Coordinator
                } else {
                    DropCredit::Fragment
                },
                require_present: true,
            })
            .await;
        match outcome {
            Ok(()) => {
                // Only once every shard has committed. A partial failure
                // returns below having credited nothing, which leaves the
                // tenant charged for rows that are partly gone - the safe
                // direction, and the next open (which also finishes the drop)
                // settles it.
                if let Some(rows) = logical {
                    self.inner.quota.sub(tenant, rows);
                }
                crate::catalog_intent::clear(&root, &name).map_err(intent)?;
            }
            Err(e) => {
                // The sidecar describes an index that is at least partly gone,
                // and the record above guarantees the rest follows. Leaving it
                // would let a recreated name inherit centroids and an owner map
                // for data nobody has.
                if let Err(sidecar) = self.drop_router_state(&name) {
                    error!("router state of partly dropped '{name}' not removed: {sidecar}");
                }
                return Err(ShardError::Storage(format!(
                    "{e}; the drop is recorded and the remaining shards are cleared at the next open"
                )));
            }
        }
        // The semantic router is derived state that outlives the shards it
        // describes: without this, recreating the same name inherits stale
        // centroids and an owner map for gone data (review P0). Drop the
        // sidecar and the in-RAM state, propagating a sidecar-removal error.
        self.drop_router_state(&name)?;
        Ok(())
    }

    /// Remove a vindex's semantic-router sidecar and its in-RAM router, owner
    /// map and version allocator. Idempotent (absent sidecar is fine). A
    /// failed removal of a present sidecar is an error - a dropped index must
    /// not leave routing behind.
    fn drop_router_state(&self, name: &str) -> Result<(), ShardError> {
        self.inner.routers.write().remove(name);
        self.inner.owners.write().remove(name);
        // The allocator goes with them: a recreated name starts from nothing
        // on disk, and a counter left over from the old index would hand its
        // first row a version far above anything the new one holds. Harmless
        // in itself, but it is derived state describing data nobody has, which
        // is exactly what the owner map is dropped here for.
        self.inner.versions.write().remove(name);
        let path = crate::router::router_path(&self.inner.root, name);
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(ShardError::Storage(format!(
                "router sidecar {} not removed: {e}",
                path.display()
            ))),
        }
    }

    /// Consolidate a disk vindex on every shard: fold each shard's streaming
    /// delta into its graph. A no-op for flat indices. Useful after a bulk load.
    ///
    /// # Errors
    ///
    /// Returns an error if the index is missing or a shard is unavailable.
    /// Force a snapshot plus the payload caches that go with it, on every
    /// shard. Best effort per shard: failures are logged, not returned, since
    /// the result is only ever an optimisation for the next open.
    pub async fn write_snapshot_and_payload_indexes(&self) {
        let _ = self.broadcast(|| ShardReq::SnapshotAndPayloadIndexes).await;
    }

    /// Per-SHARD LSM state for one vindex, one line per shard.
    ///
    /// `VINDEX.LIST` sums across shards, and that sum has already cost real
    /// time: "33 runs" read as a crisis when it was four per shard - under
    /// every per-shard threshold in the engine - and the fix aimed at it
    /// could not fire. Aggregates hide exactly the quantity the thresholds
    /// are written against.
    ///
    /// # Errors
    ///
    /// Returns an error if the name is invalid or a shard is unavailable.
    pub async fn vindex_per_shard(&self, name: &str) -> Result<Vec<String>, ShardError> {
        validate_vindex_name(name)?;
        let mut out = Vec::new();
        for shard in 0..self.inner.n {
            let ShardResp::VindexList(rows) = self.call(shard, ShardReq::VindexList).await? else {
                return Err(ShardError::Unavailable);
            };
            let Some(r) = rows.iter().find(|r| r.name == name) else {
                out.push(format!("shard={shard} absent=1"));
                continue;
            };
            out.push(format!(
                "shard={shard} live={} base={} delta={} runs={} run_rows={} \
                 run_live={} run_dead={} max_run_rows={} tombs={} \
                 run_debt_ratio={:.3} run_garbage_ratio={:.3}",
                r.n_vectors,
                r.base,
                r.delta,
                r.runs,
                r.run_rows,
                r.run_live,
                r.run_dead,
                r.max_run_rows,
                r.tombs,
                r.run_debt_ratio,
                if r.run_rows == 0 {
                    0.0
                } else {
                    r.run_dead as f32 / r.run_rows as f32
                },
            ));
        }
        Ok(out)
    }

    /// Operational health, per vindex: is maintenance keeping up?
    ///
    /// Deliberately separate from [`check`](Self::check), which certifies
    /// INTEGRITY. The churn gate produced an index whose every structure was
    /// intact - CHECK said OK on all ten rounds - while its recall fell from
    /// 0.9925 to 0.7180 as the live set migrated into run segments. WHY it
    /// fell was never isolated - see `ladder_plan` - but an operator needs to
    /// see run debt climbing either way, and no integrity check ever will.
    ///
    /// Graded on run DEBT, not on run count: one merged run holding most of
    /// the index is a single run - healthy by any count - with the majority
    /// of the corpus behind a short beam.
    ///
    /// `run_debt_ratio` is run rows over live rows. Run rows include stale,
    /// shadowed and tombstoned copies, so it is physical debt and can exceed
    /// 1.0; it is a conservative signal, not the live fraction.
    ///
    /// # Errors
    ///
    /// Returns an error if the name is invalid or a shard is unavailable.
    pub async fn health(&self, name: &str) -> Result<Vec<String>, ShardError> {
        validate_vindex_name(name)?;
        let mut out = Vec::new();
        let (mut worst_runs, mut worst_shard) = (0usize, 0usize);
        let (mut worst_debt, mut worst_debt_shard) = (0.0f32, 0usize);
        let (mut tot_runs, mut tot_delta, mut tot_run_rows, mut tot_live) =
            (0u64, 0u64, 0u64, 0u64);
        let (mut tot_live_rows, mut tot_dead) = (0u64, 0u64);
        let mut max_run_rows = 0u64;
        let mut present: Vec<usize> = Vec::new();
        let mut assessed = 0usize;
        // Shards that could not answer. CHECK names them; this used to collapse
        // them into `Unavailable`, so the one command an operator runs to find
        // out what is wrong went dark on the shard that was wrong.
        let mut unlisted: Vec<String> = Vec::new();
        for shard in 0..self.inner.n {
            let rows = match self.call(shard, ShardReq::VindexList).await? {
                ShardResp::VindexList(rows) => rows,
                ShardResp::Err(e) => {
                    unlisted.push(format!("shard {shard} cannot be listed: {e}"));
                    continue;
                }
                _ => return Err(ShardError::Unavailable),
            };
            let Some(row) = rows.iter().find(|r| r.name == name) else {
                continue;
            };
            present.push(shard);
            // In the catalogue is not the same as readable. An evicted index's
            // row carries zeros nobody measured; folding them into the totals
            // below would report a quiet, healthy index.
            if row.shards_resident == 0 {
                continue;
            }
            assessed += 1;
            let runs = usize::try_from(row.runs).unwrap_or(usize::MAX);
            if runs > worst_runs {
                worst_runs = runs;
                worst_shard = shard;
            }
            let debt = row.run_debt_ratio;
            if debt > worst_debt {
                worst_debt = debt;
                worst_debt_shard = shard;
            }
            max_run_rows = max_run_rows.max(row.max_run_rows);
            tot_runs += row.runs;
            tot_delta += row.delta;
            tot_run_rows += row.run_rows;
            tot_live += row.n_vectors;
            tot_live_rows += row.run_live;
            tot_dead += row.run_dead;
        }
        // An index absent everywhere is not healthy - it is missing. Saying
        // OK over nothing is exactly the class of lie this command exists to
        // stop telling.
        if present.is_empty() && unlisted.is_empty() {
            return Err(ShardError::Storage(format!("no such vindex '{name}'")));
        }
        // ONE ordered state, worst-wins: MISSING > UNASSESSED > PARTIAL >
        // CRITICAL > DEGRADED > OK. Reporting OK and then adding a PARTIAL line
        // below it is the false green this command exists to prevent - a
        // monitor reads `state` and nothing else. UNASSESSED sits directly
        // under MISSING because it carries the same amount of information about
        // the index's health as MISSING does: none.
        let partial = present.len() != self.inner.n;
        // Committed somewhere, readable nowhere: every number below is a zero
        // nobody measured. Calling that OK is the same false green as calling
        // an absent index OK, so it gets its own state instead of being folded
        // into PARTIAL, which means "some shards".
        let unassessed = present.len() - assessed;
        let state = if assessed == 0 {
            "UNASSESSED"
        } else if partial || unassessed > 0 || !unlisted.is_empty() {
            "PARTIAL"
        } else if worst_debt >= 0.25 {
            "CRITICAL"
        } else if worst_debt >= 0.10 || worst_runs >= 4 {
            "DEGRADED"
        } else {
            "OK"
        };
        out.push(format!("state {state}"));
        out.extend(unlisted.iter().cloned());
        if unassessed > 0 {
            out.push(format!(
                "not_assessed {unassessed} of {} shards holding it (evicted; any \
                 count below covers only the rest)",
                present.len()
            ));
        }
        if partial {
            out.push(format!(
                "present_on {} of {} shards",
                present.len(),
                self.inner.n
            ));
        }
        out.push(format!(
            "run_debt_ratio {worst_debt:.3} shard {worst_debt_shard}"
        ));
        // The number that says whether the debt is WASTE or simply the live
        // set living in a run: amplification cannot tell those apart, and
        // reading it as if it could shipped an infinite rewrite loop today.
        out.push(format!(
            "run_garbage_ratio {:.3}",
            if tot_run_rows == 0 {
                0.0
            } else {
                tot_dead as f32 / tot_run_rows as f32
            }
        ));
        out.push(format!("run_live_total {tot_live_rows}"));
        out.push(format!("run_dead_total {tot_dead}"));
        out.push(format!("worst_runs {worst_runs} shard {worst_shard}"));
        out.push(format!("max_run_rows {max_run_rows}"));
        out.push(format!("runs_total {tot_runs}"));
        out.push(format!("run_rows_total {tot_run_rows}"));
        out.push(format!(
            "delta_mass {:.3}",
            tot_delta as f32 / tot_live.max(1) as f32
        ));
        out.push(format!("delta_rows_total {tot_delta}"));
        out.push(format!("live_rows_total {tot_live}"));
        // Process-wide, NOT per index: named so nobody reads them as this
        // vindex's own.
        out.push(format!(
            "process_budget_skips_total {}",
            skeg_telemetry::counter_value(skeg_telemetry::Counter::MaintenanceBudgetSkips)
        ));
        out.push(format!(
            "process_run_scan_fallback_total {}",
            skeg_telemetry::counter_value(skeg_telemetry::Counter::RunScanFallback)
        ));
        out.push(format!(
            "process_maintenance_failures_total {}",
            skeg_telemetry::counter_value(skeg_telemetry::Counter::MaintenanceFailures)
        ));
        out.push(format!(
            "reason {}",
            match state {
                "PARTIAL" =>
                    "the index is missing on at least one shard: part of the corpus cannot be searched at all",
                "CRITICAL" =>
                    "run debt is a quarter of the live count or more: search scans run segments instead of walking them, and maintenance is behind",
                "DEGRADED" => "runs are accumulating: a merge is due",
                _ => "maintenance is keeping up",
            }
        ));
        Ok(out)
    }

    /// Which shard each id lives on: `(primary, replica)` per id, in the
    /// order given. Diagnostic, and the enabling piece for a benchmark that
    /// must SHOW its placement instead of claiming to be representative: a
    /// uniformly-placed corpus and a semantically-resharded one need
    /// completely different per-shard budgets, and only the second is what
    /// production looks like.
    ///
    /// # Errors
    ///
    /// Returns an error if the name is invalid or a shard is unavailable.
    pub async fn owners_of(
        &self,
        name: &str,
        ids: &[u64],
    ) -> Result<Vec<(u8, Option<u8>)>, ShardError> {
        validate_vindex_name(name)?;
        self.ensure_owner_map(name).await?;
        let map = self.inner.owners.read();
        let owned = map.get(name);
        Ok(ids
            .iter()
            .map(|&id| {
                let placement = owned.and_then(|m| m.get(&id).map(|&(p, r, _)| (p, r)));
                placement.unwrap_or_else(|| {
                    // Unrouted vindex: placement is computable from the id.
                    #[allow(clippy::cast_possible_truncation)] // n <= 255 shards
                    (shard_for(&id.to_le_bytes(), self.inner.n) as u8, None)
                })
            })
            .collect())
    }

    /// Integrity report for `name` across every shard, plus the coordinator's
    /// own cross-checks (owner map vs the shards that actually hold the rows).
    /// Empty = healthy. Read-only; safe on a serving index.
    ///
    /// # Errors
    ///
    /// Returns an error if the name is invalid or a shard is unavailable.
    pub async fn check(&self, name: &str) -> Result<Vec<String>, ShardError> {
        validate_vindex_name(name)?;
        // An fsck that answers "healthy" for an index that does not exist is
        // a trap. Ask EVERY shard, not shard 0 as an oracle: an index present
        // only on some shards is itself a defect the report must name, and
        // shard 0 alone would hide it in either direction.
        let mut present: Vec<usize> = Vec::new();
        let mut dim = 0usize;
        // Shards that could not answer at all. Counting one of those as "not
        // present" reports a MISSING index, which is a different fault from an
        // unreadable shard and sends the operator the wrong way.
        let mut unlisted: Vec<(usize, String)> = Vec::new();
        for shard in 0..self.inner.n {
            match self.call(shard, ShardReq::VindexList).await? {
                ShardResp::VindexList(rows) => {
                    if let Some(row) = rows.iter().find(|r| r.name == name) {
                        present.push(shard);
                        dim = row.dim as usize;
                    }
                }
                ShardResp::Err(e) => {
                    unlisted.push((shard, format!("shard {shard}: cannot be listed: {e}")));
                }
                _ => return Err(ShardError::Unavailable),
            }
        }
        if present.is_empty() {
            return Err(ShardError::Storage(format!("no such vindex '{name}'")));
        }
        let mut out: Vec<String> = unlisted.iter().map(|(_, line)| line.clone()).collect();
        // MISSING is "the shard answered and does not have it". A shard that
        // could not answer is UNKNOWN and is reported on its own line above;
        // putting it in this list too would name one fault twice, as the wrong
        // one of the two.
        if present.len() + unlisted.len() != self.inner.n {
            let missing: Vec<String> = (0..self.inner.n)
                .filter(|s| !present.contains(s) && !unlisted.iter().any(|(u, _)| u == s))
                .map(|s| s.to_string())
                .collect();
            out.push(format!(
                "index missing on shard(s) {} of {}",
                missing.join(","),
                self.inner.n
            ));
        }
        for shard in 0..self.inner.n {
            let req = ShardReq::VindexCheck {
                name: name.to_owned(),
            };
            match self.call(shard, req).await? {
                ShardResp::Problems(p) => {
                    out.extend(p.into_iter().map(|line| format!("shard {shard}: {line}")));
                }
                ShardResp::Err(e) => return Err(ShardError::Storage(e)),
                _ => return Err(ShardError::Unavailable),
            }
        }
        // Coordinator-side: a routed vindex's owner map must name shards that
        // are in range, and its router's dim must match the index. Both have
        // bitten before, so both are checked, not assumed.
        if let Some(router) = self.router(name) {
            if dim != 0 && router.dim != dim {
                out.push(format!(
                    "router dim {} does not match index dim {dim}",
                    router.dim
                ));
            }
            // The invariant is one centroid PER shard: fewer leaves shards
            // unaddressable by the router, more routes rows to shards that do
            // not exist. Either way it is a defect, not just an excess.
            if router.k != self.inner.n {
                out.push(format!(
                    "router has {} centroids for {} shards (want one per shard)",
                    router.k, self.inner.n
                ));
            }
            if let Some(map) = self.inner.owners.read().get(name) {
                let bad = map
                    .values()
                    .filter(|(p, r, _)| {
                        usize::from(*p) >= self.inner.n
                            || r.is_some_and(|s| usize::from(s) >= self.inner.n)
                    })
                    .count();
                if bad > 0 {
                    out.push(format!(
                        "owner map: {bad} entries name a shard out of range"
                    ));
                }
            }
        }
        Ok(out)
    }

    /// Consolidate a VINDEX on every shard.
    pub async fn vindex_consolidate(&self, name: &str) -> Result<(), ShardError> {
        validate_vindex_name(name)?;
        let name = name.to_owned();
        self.broadcast(|| ShardReq::VindexConsolidate { name: name.clone() })
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
    /// Count the tenant's live KV keys across the shards. Streams the index
    /// with `for_each_key`, so nothing is materialised. Vector payload blobs
    /// are KV keys under the tenant's prefix, so they are counted too.
    ///
    /// O(whole keyspace): every shard walks its entire index. Use it for audits
    /// and post-erasure leak checks, not on a hot path.
    ///
    /// # Errors
    ///
    /// Returns an error if a shard is unavailable.
    pub async fn count_tenant_keys(&self, tenant: u128) -> Result<u64, ShardError> {
        let mut total = 0u64;
        for shard in 0..self.inner.n {
            match self.call(shard, ShardReq::CountTenantKeys(tenant)).await? {
                ShardResp::Count(n) => total += n,
                ShardResp::Err(e) => return Err(ShardError::Storage(e)),
                _ => return Err(ShardError::Unavailable),
            }
        }
        Ok(total)
    }

    /// Erase every trace of `tenant`: its vindexes (payload blobs and vector
    /// quota included) and every KV key under its prefix. Returns
    /// `(vindexes_dropped, keys_deleted)` summed over the shards.
    ///
    /// A tenant's keys hash across every shard, so this fans out to all of
    /// them. It is not atomic: shards are swept in turn, and a concurrent write
    /// from the tenant being erased can land behind the sweep and survive it.
    /// Bar the tenant's traffic first if the erasure has to be final.
    ///
    /// # Errors
    ///
    /// Returns an error for tenant `0` (its keys are unscoped, so a sweep would
    /// wipe the store), or if a shard is unavailable or a delete fails.
    pub async fn erase_tenant(
        &self,
        tenant: u128,
        durability: Durability,
    ) -> Result<(u64, u64), ShardError> {
        if tenant == 0 {
            return Err(ShardError::Storage(
                "cannot erase the anonymous tenant (0)".to_owned(),
            ));
        }
        let (mut vindexes, mut keys) = (0u64, 0u64);
        for shard in 0..self.inner.n {
            let req = ShardReq::EraseTenant { tenant, durability };
            match self.call(shard, req).await? {
                ShardResp::Erased {
                    vindexes: v,
                    keys: k,
                } => {
                    vindexes += v;
                    keys += k;
                }
                ShardResp::Err(e) => return Err(ShardError::Storage(e)),
                _ => return Err(ShardError::Unavailable),
            }
        }
        // Erasure must remove derived routers too, or centroids trained on the
        // erased vectors survive on disk (review P0). Scoped names are
        // `{tenant}::name`; drop every router under this tenant's prefix.
        //
        // The prefix comes from `scope_key`, the same helper that WRITES the
        // keys, and not from a hand-rolled `format!`. It was hand-rolled, and
        // it rendered the `u128` in decimal - `42::` - while every key carries
        // 32 hex digits of `to_le_bytes`. It matched nothing, so no non-zero
        // tenant ever lost a router and this whole block was dead. One helper
        // writes the prefix and one reads it, or they drift again.
        //
        // `scope_key(0, "")` is the empty string, which every key starts with:
        // tenant 0 returned at the top of this function, before anything here.
        debug_assert_ne!(tenant, 0, "tenant 0 is refused above; its prefix is empty");
        let prefix = scope_key(tenant, "");
        let scoped: Vec<String> = self
            .inner
            .routers
            .read()
            .keys()
            .filter(|k| k.starts_with(&prefix))
            .cloned()
            .collect();
        for name in scoped {
            self.drop_router_state(&name)?;
        }
        // Every index this tenant had is gone, so the only number that can be
        // right is zero - and it is reached by STATING it, not by adding up
        // what each shard happened to hold. The shards credit nothing on this
        // path (`DropCredit::Coordinator`): a routed index keeps a second
        // physical copy of every boundary row, so their fragments sum to more
        // than the tenant ever spent, and the subtraction saturating at zero
        // was the right answer only by accident.
        //
        // Reached only after every shard has answered `Erased`; a failure
        // returns above and leaves the count for the next open to settle.
        self.inner.quota.rebuild(tenant, 0);
        Ok((vindexes, keys))
    }

    /// Erase a subject within a tenant: every KV key whose bytes are
    /// `tenant` (16B LE) followed by `subject`. Returns the count deleted,
    /// summed over the shards. Logical only - follow with [`reclaim`] when the
    /// value bytes must physically leave the disk (GDPR).
    ///
    /// KV keys only. A subject's *vectors* live in the tenant's shared vindex;
    /// erase them with `vdel` by id, then `vindex_consolidate` to reclaim the
    /// f32 bytes. This call does not touch vindexes.
    ///
    /// [`reclaim`]: Self::reclaim
    ///
    /// # Errors
    ///
    /// Returns an error for tenant `0`, or if a shard is unavailable or a
    /// delete fails.
    pub async fn erase_prefix(
        &self,
        tenant: u128,
        subject: &[u8],
        durability: Durability,
    ) -> Result<u64, ShardError> {
        if tenant == 0 {
            return Err(ShardError::Storage(
                "cannot erase under the anonymous tenant (0)".to_owned(),
            ));
        }
        let mut keys = 0u64;
        for shard in 0..self.inner.n {
            let req = ShardReq::ErasePrefix {
                tenant,
                subject: subject.to_vec(),
                durability,
            };
            match self.call(shard, req).await? {
                ShardResp::Erased { keys: k, .. } => keys += k,
                ShardResp::Err(e) => return Err(ShardError::Storage(e)),
                _ => return Err(ShardError::Unavailable),
            }
        }
        Ok(keys)
    }

    /// Physically reclaim dead bytes across every shard: compact all sealed
    /// segments holding deleted/overwritten records so the old values leave the
    /// disk. Returns the total bytes reclaimed.
    ///
    /// The durable half of erasure. `erase_*`/`del` only tombstone; the value
    /// bytes sit in their segments until compaction rewrites them, and the
    /// background loop only compacts a segment once it is mostly dead. For a
    /// GDPR guarantee, call this after the deletes. Store-wide and heavy
    /// (O(bytes in segments with any dead record)) - batch your deletes, then
    /// reclaim once.
    ///
    /// Does not touch backups already taken, or vector f32 bytes (use
    /// `vindex_consolidate` for those).
    ///
    /// # Errors
    ///
    /// Returns an error if a shard is unavailable or a compaction fails.
    pub async fn reclaim(&self) -> Result<u64, ShardError> {
        let mut freed = 0u64;
        for shard in 0..self.inner.n {
            match self.call(shard, ShardReq::Reclaim).await? {
                ShardResp::Reclaimed(n) => freed += n,
                ShardResp::Err(e) => return Err(ShardError::Storage(e)),
                _ => return Err(ShardError::Unavailable),
            }
        }
        Ok(freed)
    }

    pub async fn vindex_list(&self) -> Result<Vec<VindexRow>, ShardError> {
        use std::collections::BTreeMap;
        let mut agg: BTreeMap<String, VindexRow> = BTreeMap::new();
        for shard in 0..self.inner.n {
            match self.call(shard, ShardReq::VindexList).await? {
                ShardResp::VindexList(rows) => {
                    for row in rows {
                        match agg.entry(row.name.clone()) {
                            std::collections::btree_map::Entry::Vacant(e) => {
                                e.insert(row);
                            }
                            std::collections::btree_map::Entry::Occupied(mut e) => {
                                let a = e.get_mut();
                                a.shards_resident += row.shards_resident;
                                a.n_vectors = a.n_vectors.saturating_add(row.n_vectors);
                                a.delta = a.delta.saturating_add(row.delta);
                                a.runs = a.runs.saturating_add(row.runs);
                                // The largest run is a MAX across shards, never a sum.
                                a.max_run_rows = a.max_run_rows.max(row.max_run_rows);
                                // Worst shard, never an average: a threshold is per shard.
                                a.run_debt_ratio = a.run_debt_ratio.max(row.run_debt_ratio);
                                a.run_live = a.run_live.saturating_add(row.run_live);
                                a.run_dead = a.run_dead.saturating_add(row.run_dead);
                                a.run_rows = a.run_rows.saturating_add(row.run_rows);
                                a.tombs = a.tombs.saturating_add(row.tombs);
                                a.base = a.base.saturating_add(row.base);
                            }
                        }
                    }
                }
                ShardResp::Err(e) => return Err(ShardError::Storage(e)),
                _ => return Err(ShardError::Unavailable),
            }
        }
        // The denominator is the STORE, not the shards that answered - see the
        // field's own note. Set once here, where the count is known.
        let total = u32::try_from(self.inner.n).unwrap_or(u32::MAX);
        Ok(agg
            .into_values()
            .map(|mut row| {
                row.shards_total = total;
                row
            })
            .collect())
    }

    /// Insert a vector under `id` into `name`. Routes by `id`.
    ///
    /// # Errors
    ///
    /// Returns an error if the index is missing, the dim mismatches, or the
    /// shard is unavailable.
    pub async fn vset(
        &self,
        name: &str,
        id: u64,
        vector: Vec<f32>,
        tenant: u128,
        limit: Option<u64>,
        payload: Option<Bytes>,
    ) -> Result<(), ShardError> {
        // Semantic placement when a router exists: the vector picks its owner
        // shard, and an overwrite whose old copy lives elsewhere has to remove
        // it.
        //
        // WRITE THE NEW COPY FIRST, then move the reader, then delete the old.
        //
        // It used to delete first, on the reasoning that set-then-delete would
        // duplicate under a same-id race. That race is now impossible: the
        // stripe lock below serialises the whole span for this id. What
        // delete-first cost instead was durability - a VSET that failed for
        // ANY reason (quota, WAL I/O, a missing index, an unavailable shard)
        // returned an error with the previously acknowledged version already
        // deleted. A refused write that destroys the value it was overwriting
        // is worse than either outcome a client can plan for. Reproduced
        // deterministically in tests/semantic_overwrite.rs.
        //
        // `reshard` already writes then deletes, for exactly this reason: a
        // duplicate is recoverable (the search path dedups, and the owner map
        // says which copy is live) and a hole is not.
        if let Some(router) = self.router(name) {
            // Serialise the whole read-delete-set-record span against a
            // concurrent routed vset/vdel on the SAME id (review finding).
            let _stripe = self.owner_stripe(name, id).lock().await;
            // A wrong-length vector reaches the router BEFORE any shard worker
            // validates the dim; Router::assign would assert and, under
            // panic=abort, kill the whole process for every tenant. Reject it
            // as a clean error here (found by the security review, C1).
            if vector.len() != router.dim {
                return Err(ShardError::Storage(format!(
                    "vector has {} dims, index has {}",
                    vector.len(),
                    router.dim
                )));
            }
            self.ensure_owner_map(name).await?;
            let owner = router.assign(&vector);
            let old = self
                .inner
                .owners
                .read()
                .get(name)
                .and_then(|m| m.get(&id).copied());
            // Allocated UNDER THE STRIPE, and floored by the version the map
            // already records for this row: strictly greater than every copy
            // that exists, so the write cannot lose to one of them.
            let version = self.next_version(name, old.map_or(0, |(_, _, v)| v));
            let req = ShardReq::Vset {
                name: name.to_owned(),
                id,
                vector,
                tenant,
                limit,
                // An overwrite of a row the tenant already owns changes no
                // cardinality, wherever the old copy happened to live.
                effect: if old.is_some() {
                    crate::quota::QuotaEffect::Overwrite
                } else {
                    crate::quota::QuotaEffect::Insert
                },
                version: Some(version),
                payload,
            };
            match self.call(owner, req).await? {
                // Nothing has been removed yet, so a refusal here leaves the
                // committed version exactly where it was.
                ShardResp::Err(e) => return Err(ShardError::Storage(e)),
                ShardResp::Done => {}
                _ => return Err(ShardError::Unavailable),
            }
            // THE COMMIT POINT: reads route through this map, so the new copy
            // becomes the live one here, before the old one is touched.
            self.inner
                .owners
                .write()
                .entry(name.to_owned())
                .or_default()
                .insert(id, (owner as u8, None, version));
            // Cleanup, after the fact. A failure here leaves a stale duplicate
            // on a shard the map no longer points at: search dedups it and the
            // next overwrite of this id removes it. Reporting an error would
            // tell the client a write failed that is committed and readable -
            // the same mistake as failing a DROP whose registry entry is
            // already published.
            //
            // Everything below is post-commit, so the failpoint returns the
            // SUCCESS the client is owed: it models the process dying here,
            // not the write failing.
            crate::fp!(
                crate::failpoint::WriteFailpoint::OverwriteOldCopyDelete,
                Ok(())
            );
            if let Some((old_primary, old_replica, _)) = old {
                for s in std::iter::once(old_primary).chain(old_replica) {
                    if usize::from(s) != owner {
                        match self
                            .call(
                                usize::from(s),
                                ShardReq::Vdel {
                                    name: name.to_owned(),
                                    id,
                                    tenant,
                                    // The row moved; it did not leave.
                                    effect: crate::quota::QuotaEffect::Move,
                                    // At the version of the copy that
                                    // replaced it, so the tombstone stands
                                    // against anything older that is still in
                                    // flight towards this shard.
                                    version: Some(version),
                                },
                            )
                            .await
                        {
                            Ok(ShardResp::Existed(_)) => {}
                            Ok(ShardResp::Err(e)) => {
                                tracing::error!(
                                    index = name,
                                    id,
                                    shard = usize::from(s),
                                    error = %e,
                                    "overwrite committed on shard {owner} but the \
                                     old copy was not removed"
                                );
                            }
                            Ok(_) | Err(_) => {
                                tracing::error!(
                                    index = name,
                                    id,
                                    shard = usize::from(s),
                                    "overwrite committed on shard {owner} but the \
                                     old copy could not be removed: shard \
                                     unavailable"
                                );
                            }
                        }
                    }
                }
            }
            return Ok(());
        }
        // The same row stripe the routed path takes, for a different reason.
        // Placement never moves here, so there is nothing to serialise about
        // WHERE the row lives - but a VSET is no longer one shard message: it
        // stages a blob, commits, then reclaims. The worker awaits between
        // those, and the shard runs its requests concurrently, so two writes
        // to one id would interleave their halves and leave the row described
        // by the other one's payload.
        let _stripe = self.owner_stripe(name, id).lock().await;
        let shard = shard_for(&id.to_le_bytes(), self.inner.n);
        let req = ShardReq::Vset {
            name: name.to_owned(),
            id,
            vector,
            tenant,
            limit,
            // Hash placement never moves a row, so the shard's own answer to
            // "did I hold this id already" is the whole truth here - which is
            // exactly what `Insert` defers to.
            effect: crate::quota::QuotaEffect::Insert,
            // And so is the shard's own version counter: an id maps to exactly
            // one shard, for ever, so nothing else can hold a copy of it.
            version: None,
            payload,
        };
        match self.call(shard, req).await? {
            ShardResp::Done => Ok(()),
            ShardResp::Err(e) => Err(ShardError::Storage(e)),
            _ => Err(ShardError::Unavailable),
        }
    }

    /// Bulk insert: each item runs the normal [`vset`](Self::vset) path, but
    /// CONCURRENTLY, so the per-vector durable payload-blob writes accumulate in
    /// the group committer and flush in batches instead of one fsync-barrier per
    /// vector. This is the whole bulk-ingest win (100k: ~770s serial -> ~34s).
    ///
    /// One result PER ITEM, in REQUEST order, and every item is attempted.
    ///
    /// A bulk write of n items is n writes. Reporting one outcome for all of
    /// them can only say "some prefix worked", and the client cannot tell
    /// which prefix - so it either re-sends rows that are already durable or
    /// drops rows that are not. Nothing here is atomic across items and
    /// nothing pretends to be: quota, admission and the dimension check are
    /// already decided per item, and making the batch atomic across shards
    /// would take a two-phase commit for a command whose whole reason to
    /// exist is throughput.
    ///
    /// The failure that made this urgent is worse than a vague reply. The
    /// previous body awaited a `JoinSet` with `??`, and a `JoinSet` ABORTS its
    /// outstanding tasks when it is dropped - including a task that had
    /// already committed its write and had not yet published the row in the
    /// owner map. That row is durable, acknowledged by the engine, and
    /// unreachable: an acknowledged write lost because a SIBLING was
    /// malformed.
    pub async fn vmset(
        &self,
        name: &str,
        items: Vec<(u64, Vec<f32>, Option<Bytes>)>,
        tenant: u128,
        limit: Option<u64>,
    ) -> Vec<Result<(), ShardError>> {
        let n = items.len();
        // Still concurrent, which is the whole point of the command: the
        // per-vector blob writes accumulate in the group committer and flush
        // in batches instead of one barrier per vector (100k: ~770s serial ->
        // ~34s). Only the ORDER is restored, by carrying the index with the
        // task rather than by waiting for each in turn.
        //
        // But bounded at [`VMSET_INFLIGHT`], which it was not: a maximum batch
        // is 4096 items, so one connection opened 4096 tasks and the default
        // connection limit made that four million. That memory is per REQUEST
        // and the ingress budget does not charge it - the budget covers the
        // socket buffers, not the tree a request expands into - so the only
        // ceiling it had was MAX_VMSET_ITEMS.
        //
        // Still spawned tasks rather than a buffered stream. A panic inside a
        // task is that task's; a panic inside a stream's future unwinds into
        // this call and takes every sibling with it, which is exactly the
        // property Task 2 added the per-item answers for.
        let mut set = tokio::task::JoinSet::new();
        let mut pending = items.into_iter().enumerate();
        let mut out: Vec<Option<Result<(), ShardError>>> = (0..n).map(|_| None).collect();
        let mut spawn_next = |set: &mut tokio::task::JoinSet<_>| {
            let Some((i, (id, vector, payload))) = pending.next() else {
                return false;
            };
            let this = self.clone();
            let name = name.to_owned();
            set.spawn(async move {
                let _inflight = vmset_inflight::Guard::enter();
                (
                    i,
                    this.vset(&name, id, vector, tenant, limit, payload).await,
                )
            });
            true
        };
        for _ in 0..VMSET_INFLIGHT {
            if !spawn_next(&mut set) {
                break;
            }
        }
        while let Some(joined) = set.join_next().await {
            match joined {
                Ok((i, r)) => out[i] = Some(r),
                // A task that panicked names no item, so nothing can be said
                // about a specific one. The `None`s below become
                // `Unavailable`, which is the honest answer for an item whose
                // outcome nobody observed - and its siblings, which are their
                // own tasks, are untouched.
                Err(e) => tracing::error!(index = name, error = %e, "a VMSET item task failed"),
            }
            // One out, one in: the window stays full until the batch runs out,
            // so the bound costs a scheduling round trip per item and not a
            // barrier per window.
            spawn_next(&mut set);
        }
        out.into_iter()
            .map(|r| r.unwrap_or(Err(ShardError::Unavailable)))
            .collect()
    }

    /// How many payload blobs the store still holds for `name`, summed over
    /// every shard and every generation of the name.
    ///
    /// The blobs are KV keys under a reserved marker, so nothing on the vector
    /// side can see them: a blob whose row is gone is invisible to `VINDEX
    /// LIST`, to `SKEG.STATS` and to the index's own length. This is the one
    /// number that says whether the reclamation paths - the drop sweep, the
    /// post-commit cleanup, the open-time collection of what a failed commit
    /// staged - are actually doing their job.
    ///
    /// O(whole keyspace) per shard, like [`count_tenant_keys`](Self::count_tenant_keys):
    /// an audit, not a hot path.
    ///
    /// # Errors
    ///
    /// Returns an error if a shard is unavailable.
    pub async fn payload_blobs_held(&self, tenant: u128, name: &str) -> Result<u64, ShardError> {
        let mut total = 0u64;
        for shard in 0..self.inner.n {
            let req = ShardReq::PayloadBlobs {
                tenant,
                name: name.to_owned(),
            };
            match self.call(shard, req).await? {
                ShardResp::Count(n) => total += n,
                ShardResp::Err(e) => return Err(ShardError::Storage(e)),
                _ => return Err(ShardError::Unavailable),
            }
        }
        Ok(total)
    }

    /// Vectors currently reserved by `tenant` against its quota (0 if
    /// untracked / unlimited). Read directly from the shared counter.
    #[must_use]
    pub fn tenant_vector_count(&self, tenant: u128) -> u64 {
        self.inner.quota.count(tenant)
    }

    /// Live on-disk KV bytes charged to `tenant`, aggregated across shards.
    #[must_use]
    pub fn tenant_disk_bytes(&self, tenant: u128) -> u64 {
        self.inner
            .disk_counter
            .lock()
            .get(&tenant)
            .copied()
            .unwrap_or(0)
    }

    /// Fetch the stored f32 vector for `id` in `name`. Routes by `id`.
    ///
    /// # Errors
    ///
    /// Returns an error if the index is missing or the shard is unavailable.
    pub async fn vget(&self, name: &str, id: u64) -> Result<Option<Vec<f32>>, ShardError> {
        self.ensure_owner_map(name).await?;
        let shard = self.point_shard(name, id);
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

    /// Train (or re-train) the semantic router for `name`: sample every
    /// shard, run balanced k-means with one centroid per shard, bump the
    /// epoch, persist the sidecar, and serve the new router immediately.
    /// Returns the new epoch.
    ///
    /// # Errors
    ///
    /// Returns an error if the index is missing, a shard is unavailable, or
    /// the sidecar cannot be written.
    pub async fn train_router(
        &self,
        name: &str,
        lambda: f32,
        iters: usize,
    ) -> Result<u64, ShardError> {
        const SAMPLE_PER_SHARD: usize = 4096;
        let mut data: Vec<f32> = Vec::new();
        let mut dim: Option<u32> = None;
        for shard in 0..self.inner.n {
            let req = ShardReq::SampleVectors {
                name: name.to_owned(),
                count: SAMPLE_PER_SHARD,
            };
            match self.call(shard, req).await? {
                ShardResp::Sample(rows, d) => {
                    if let Some(prev) = dim
                        && prev != d
                    {
                        return Err(ShardError::Storage(format!(
                            "shard dim mismatch: {prev} vs {d}"
                        )));
                    }
                    dim = Some(d);
                    data.extend_from_slice(&rows);
                }
                ShardResp::Err(e) => return Err(ShardError::Storage(e)),
                _ => return Err(ShardError::Unavailable),
            }
        }
        let dim = dim.unwrap_or(0) as usize;
        let n = data.len().checked_div(dim).unwrap_or(0);
        if n < self.inner.n {
            return Err(ShardError::Storage(format!(
                "not enough live vectors to train a router: {n}"
            )));
        }
        let k = self.inner.n;
        let centroids = skeg_vector::balanced_kmeans(&data, n, dim, k, lambda, iters, 0x5eed_5eed);
        let epoch = self.inner.routers.read().get(name).map_or(0, |r| r.epoch) + 1;
        let router = crate::router::Router {
            k,
            dim,
            epoch,
            centroids,
        };
        router
            .save(&crate::router::router_path(&self.inner.root, name))
            .map_err(|e| ShardError::Storage(format!("router save failed: {e}")))?;
        let arc = Arc::new(router);
        self.inner.routers.write().insert(name.to_owned(), arc);
        Ok(epoch)
    }

    /// Physically re-partition `name` by its semantic router (training one
    /// first): every live vector whose owner is another shard moves there,
    /// vset-then-vdel so a crash duplicates and never loses (the search
    /// merge dedups by id). Batched and cursor-resumable per shard; a move
    /// whose id the owner map already places at its destination only deletes
    /// the stale source copy (a concurrent post-router write won).
    /// Returns the number of rows moved.
    ///
    /// # Errors
    ///
    /// Index missing, a shard unavailable, or a batch failing.
    pub async fn reshard(
        &self,
        name: &str,
        lambda: f32,
        iters: usize,
        tenant: u128,
    ) -> Result<u64, ShardError> {
        const BATCH: usize = 512;
        self.train_router(name, lambda, iters).await?;
        let router = self.router(name).expect("router just trained");
        // From here every write routes semantically; the map records them.
        let mut moved = 0u64;
        for source in 0..self.inner.n {
            let mut after = 0u64;
            loop {
                let req = ShardReq::CollectMoves {
                    name: name.to_owned(),
                    centroids: router.clone(),
                    own: source as u8,
                    after,
                    limit: BATCH,
                    tenant,
                };
                let (batch, cursor) = match self.call(source, req).await? {
                    ShardResp::Moves(b, c) => (b, c),
                    ShardResp::Err(e) => return Err(ShardError::Storage(e)),
                    _ => return Err(ShardError::Unavailable),
                };
                for (id, vector, payload, owner, version) in batch {
                    // The version the source held when the batch was
                    // collected. It travels with the row and is what the
                    // destination stores, so a copy of this row written since
                    // then - by a user write that raced this move - stays the
                    // newer one everywhere.
                    self.observe_version(name, version);
                    // ONE ROW, under the same stripe a routed point op takes.
                    //
                    // Not the batch: a batch spans shard calls for other ids,
                    // and holding one id's stripe across them would serialise
                    // the whole reshard behind it. Per row is enough, because
                    // the thing being made atomic is exactly what a vset on
                    // this id also does - decide where the live copy is, and
                    // publish that decision.
                    let _stripe = self.owner_stripe(name, id).lock().await;
                    // Re-read UNDER the stripe. Everything above was decided
                    // from a batch collected before this lock existed.
                    let current = self
                        .inner
                        .owners
                        .read()
                        .get(name)
                        .and_then(|m| m.get(&id).copied());
                    if current.is_some_and(|(_, _, live)| live > version) {
                        // A user write replaced this row while the batch was
                        // in flight, and was acknowledged. What this loop
                        // holds is the value that write REPLACED: writing it
                        // to the new owner and pointing the map at it is how
                        // an acknowledged write used to disappear. Its own
                        // cleanup removes the source copy.
                        continue;
                    }
                    // And ask the SOURCE whether the row is still the row that
                    // was read. The map cannot answer that: a successful
                    // `vdel` REMOVES the entry, so a deleted row looks exactly
                    // like a row that has never moved - which is most rows on
                    // a first reshard - and the guard above does not fire.
                    // Writing it to its new owner then republishes a delete
                    // the client was told had happened, and the destination
                    // has no tombstone to lose against because the tombstone
                    // is here.
                    //
                    // One extra round trip per moved row, under the stripe the
                    // delete also takes, so the answer cannot go stale between
                    // the question and the write.
                    let req = ShardReq::StillCurrent {
                        name: name.to_owned(),
                        id,
                        version,
                    };
                    match self.call(source, req).await? {
                        ShardResp::Existed(true) => {}
                        ShardResp::Existed(false) => continue,
                        ShardResp::Err(e) => return Err(ShardError::Storage(e)),
                        _ => return Err(ShardError::Unavailable),
                    }
                    let already_there = current.is_some_and(|(p, _, _)| p == owner);
                    if !already_there {
                        crate::fp!(
                            crate::failpoint::WriteFailpoint::ReshardDestinationWrite,
                            Err(ShardError::Storage(
                                "failpoint: reshard destination write refused".to_owned()
                            ))
                        );
                        let req = ShardReq::Vset {
                            name: name.to_owned(),
                            id,
                            vector,
                            tenant,
                            limit: None,
                            // A reshard move: the row arrives, it is not new.
                            effect: crate::quota::QuotaEffect::Move,
                            version: Some(version),
                            payload,
                        };
                        match self.call(usize::from(owner), req).await? {
                            ShardResp::Done => {}
                            ShardResp::Err(e) => return Err(ShardError::Storage(e)),
                            _ => return Err(ShardError::Unavailable),
                        }
                    }
                    // Swallowing this Vdel's response would count the row moved
                    // while its stale source copy survives (review finding).
                    crate::fp!(
                        crate::failpoint::WriteFailpoint::ReshardSourceDelete,
                        Err(ShardError::Storage(
                            "failpoint: reshard source delete refused".to_owned()
                        ))
                    );
                    match self
                        .call(
                            source,
                            ShardReq::Vdel {
                                name: name.to_owned(),
                                id,
                                tenant,
                                // Same move, far side. Crediting here while the
                                // destination does not charge is what leaked
                                // the counter downward once per moved row.
                                effect: crate::quota::QuotaEffect::Move,
                                // The version this move read. A source copy
                                // that has been written again since carries a
                                // higher one, and the delete then does not
                                // apply - which is what stops a move removing
                                // a row it never actually copied.
                                version: Some(version),
                            },
                        )
                        .await?
                    {
                        ShardResp::Existed(_) => {}
                        ShardResp::Err(e) => return Err(ShardError::Storage(e)),
                        _ => return Err(ShardError::Unavailable),
                    }
                    self.inner
                        .owners
                        .write()
                        .entry(name.to_owned())
                        .or_default()
                        .insert(id, (owner, None, version));
                    moved += 1;
                }
                match cursor {
                    Some(c) => after = c,
                    None => break,
                }
            }
        }
        // The loop recorded only MOVED ids; rows already on their owner never
        // entered the map, and every consumer that trusts the map (overlap's
        // primary check, point ops after the hash fallback stops being
        // coincidentally right) needs the full picture.
        self.rebuild_owner_maps().await?;
        Ok(moved)
    }

    /// Targeted 2-way overlap (C5): every live row whose margin between its
    /// two nearest centroids is below `tau` gains a replica on the
    /// second-nearest shard, so a probe search finds boundary rows from
    /// either side. Idempotent (a re-run overwrites the same replicas);
    /// deletes and overwrites remove replicas through the owner map.
    /// Returns the number of rows replicated.
    ///
    /// # Errors
    ///
    /// Index or router missing, a shard unavailable, or a batch failing.
    pub async fn overlap(&self, name: &str, tau: f32, tenant: u128) -> Result<u64, ShardError> {
        const BATCH: usize = 512;
        let Some(router) = self.router(name) else {
            return Err(ShardError::Storage(format!(
                "vindex '{name}' has no semantic router; reshard first"
            )));
        };
        self.ensure_owner_map(name).await?;
        let mut replicated = 0u64;
        for source in 0..self.inner.n {
            let mut after = 0u64;
            loop {
                let req = ShardReq::CollectBoundary {
                    name: name.to_owned(),
                    centroids: router.clone(),
                    after,
                    limit: BATCH,
                    tau,
                    tenant,
                };
                let (batch, cursor) = match self.call(source, req).await? {
                    ShardResp::Moves(b, c) => (b, c),
                    ShardResp::Err(e) => return Err(ShardError::Storage(e)),
                    _ => return Err(ShardError::Unavailable),
                };
                for (id, vector, payload, second, version) in batch {
                    self.observe_version(name, version);
                    // Per row, same stripe, same reason as the reshard: the
                    // window this closes is one await wide and it is where a
                    // deleted row came back. The delete took both copies and
                    // the map entry while the replica was in flight, and the
                    // replica then landed on a shard nothing was left to clean
                    // up - reachable by search, invisible to the map.
                    let _stripe = self.owner_stripe(name, id).lock().await;
                    let current = self
                        .inner
                        .owners
                        .read()
                        .get(name)
                        .and_then(|m| m.get(&id).copied());
                    // No entry means the row was deleted while the batch was
                    // in flight; a higher version means it was rewritten, and
                    // this copy is the value that write replaced. Only rows
                    // whose PRIMARY still lives here replicate from here (the
                    // same row seen via its replica must not re-replicate).
                    let Some((primary, _, live)) = current else {
                        continue;
                    };
                    if live > version
                        || usize::from(primary) != source
                        || usize::from(second) == source
                    {
                        continue;
                    }
                    let req = ShardReq::Vset {
                        name: name.to_owned(),
                        id,
                        vector,
                        tenant,
                        limit: None,
                        // A boundary replica: a second physical copy of one
                        // logical row.
                        effect: crate::quota::QuotaEffect::Replica,
                        // Carried, not allocated: a replica is the SAME copy
                        // of the row, in a second place. Allocating here would
                        // make the replica outrank its own primary.
                        version: Some(version),
                        payload,
                    };
                    let outcome = match self.call(usize::from(second), req).await {
                        Ok(ShardResp::Done) => Ok(()),
                        Ok(ShardResp::Err(e)) => Err(ShardError::Storage(e)),
                        Ok(_) => Err(ShardError::Unavailable),
                        Err(e) => Err(e),
                    };
                    // The failpoint sits AFTER the write, because that is the
                    // half worth testing: a replica that landed and was then
                    // reported as a failure. `fp_check!`, not `fp!`, for the
                    // same reason - this site has to undo something before it
                    // can leave, and `fp!` expands to a bare `return`.
                    let outcome = outcome.and_then(|()| {
                        crate::fp_check!(
                            crate::failpoint::WriteFailpoint::OverlapReplicaWrite,
                            Err(ShardError::Storage(
                                "failpoint: overlap replica write refused".to_owned()
                            ))
                        )
                    });
                    if let Err(e) = outcome {
                        // The write may have LANDED and still reported a
                        // failure: the destination stores the vector and then
                        // the payload blob, and a blob error comes back after
                        // the row is in; a lost reply looks the same. A
                        // replica the map does not name is a ghost - `vdel`
                        // finds a second copy by reading the replica slot, so
                        // nothing would ever go looking for this one, and the
                        // deleted row stays searchable.
                        //
                        // Take it back out. Best effort, and LOGGED rather
                        // than returned: the error worth reporting is the one
                        // that got us here, and an overlap replica is an
                        // optimisation a re-run recreates.
                        let undo = ShardReq::Vdel {
                            name: name.to_owned(),
                            id,
                            tenant,
                            // Taking back the replica this overlap wrote: it
                            // never had a slot, so removing it gives none.
                            effect: crate::quota::QuotaEffect::Replica,
                            version: Some(version),
                        };
                        match self.call(usize::from(second), undo).await {
                            Ok(ShardResp::Existed(_)) => {}
                            Ok(ShardResp::Err(err)) => tracing::error!(
                                index = name,
                                id,
                                shard = usize::from(second),
                                error = %err,
                                "an overlap replica was written and could not be taken \
                                 back: it is a copy the owner map does not name"
                            ),
                            Ok(_) | Err(_) => tracing::error!(
                                index = name,
                                id,
                                shard = usize::from(second),
                                "an overlap replica was written and could not be taken \
                                 back: shard unavailable"
                            ),
                        }
                        return Err(e);
                    }
                    // Still under the stripe: the replica has to be IN the map
                    // before a delete can look for it, or the delete finds
                    // nothing to remove and the copy outlives the row.
                    if let Some(m) = self.inner.owners.write().get_mut(name)
                        && let Some(e) = m.get_mut(&id)
                    {
                        e.1 = Some(second);
                    }
                    replicated += 1;
                }
                match cursor {
                    Some(c) => after = c,
                    None => break,
                }
            }
        }
        Ok(replicated)
    }

    /// Rebuild the id -> owner-shard map for every routed vindex by asking
    /// each shard for its live ids. Called once after open.
    ///
    /// The primary is the copy with the HIGHEST VERSION, not the first one
    /// seen. It used to be the first, which means the lowest shard number,
    /// which means a restart promoted whichever copy of a duplicated row
    /// happened to sit lower - and a duplicate is a normal state here, left by
    /// a crash mid-move or a cleanup that failed after an overwrite committed
    /// elsewhere. Half the time the copy it promoted was the one the overwrite
    /// had replaced, and every read after the restart returned it. Equal
    /// versions still go to the lowest shard, so a set where nothing is
    /// versioned reopens exactly as it did.
    ///
    /// It also seeds this index's version allocator past everything on disk,
    /// which is what makes a user write after a restart beat every copy that
    /// already exists.
    ///
    /// # Errors
    ///
    /// A shard being unavailable.
    pub async fn rebuild_owner_maps(&self) -> Result<(), ShardError> {
        let names: Vec<String> = self.inner.routers.read().keys().cloned().collect();
        for name in names {
            let mut map: OwnerMap = ahash::AHashMap::new();
            let mut highest = 0u64;
            for shard in 0..self.inner.n {
                match self
                    .call(shard, ShardReq::LiveIds { name: name.clone() })
                    .await?
                {
                    ShardResp::LiveIds(LiveIdsAnswer::Held { ids, high_water }) => {
                        // The shard's own high-water, not the maximum over the
                        // live ids: a row deleted right after it was written
                        // leaves its version only in a tombstone, and seeding
                        // the allocator below that hands the next write to
                        // that id a version the tombstone beats - a write
                        // acknowledged and dropped.
                        highest = highest.max(high_water);
                        for (id, version) in ids {
                            match map.entry(id) {
                                std::collections::hash_map::Entry::Occupied(mut e) => {
                                    let (primary, _, live) = *e.get();
                                    if version > live {
                                        // The NEWER copy takes the primary
                                        // slot and demotes the one that was
                                        // there.
                                        e.insert((shard as u8, Some(primary), version));
                                    } else {
                                        e.get_mut().1 = Some(shard as u8);
                                    }
                                }
                                std::collections::hash_map::Entry::Vacant(e) => {
                                    e.insert((shard as u8, None, version));
                                }
                            }
                        }
                    }
                    // A shard that genuinely does not have the index yet.
                    ShardResp::LiveIds(LiveIdsAnswer::Absent) => {}
                    // One that has it and cannot read it. Building the map
                    // without its rows leaves the allocator seeded BELOW the
                    // versions that shard holds, and the next user write to
                    // one of them is refused as superseded at the
                    // destination - a `+OK` for a write that never happened.
                    // There is no answer here that is better than refusing.
                    ShardResp::LiveIds(LiveIdsAnswer::Unreadable(e)) => {
                        return Err(ShardError::Storage(format!(
                            "owner map for '{name}' cannot be rebuilt: {e}"
                        )));
                    }
                    ShardResp::Err(e) => return Err(ShardError::Storage(e)),
                    _ => return Err(ShardError::Unavailable),
                }
            }
            // Past everything on disk, so the next user write to this index
            // beats every copy the rebuild just saw.
            self.observe_version(&name, highest);
            self.inner.owners.write().insert(name, map);
        }
        Ok(())
    }

    /// Sample the base graph of one shard of `name` for visual exploration.
    ///
    /// # Errors
    ///
    /// Index missing, flat backend, shard out of range or unavailable.
    pub async fn graph_sample(
        &self,
        name: &str,
        shard: usize,
        count: usize,
    ) -> Result<(Vec<(u64, u32)>, Vec<(u64, u64)>), ShardError> {
        if shard >= self.inner.n {
            return Err(ShardError::Storage(format!(
                "shard {shard} out of range (0..{})",
                self.inner.n
            )));
        }
        let req = ShardReq::GraphSample {
            name: name.to_owned(),
            count,
        };
        match self.call(shard, req).await? {
            ShardResp::Graph(n, e) => Ok((n, e)),
            ShardResp::Err(e) => Err(ShardError::Storage(e)),
            _ => Err(ShardError::Unavailable),
        }
    }

    /// The loaded semantic router for `name`, if one has been trained.
    #[must_use]
    pub fn router(&self, name: &str) -> Option<Arc<crate::router::Router>> {
        self.inner.routers.read().get(name).cloned()
    }

    /// Tombstone the vector for `id` in `name`. Routes by `id`.
    ///
    /// # Errors
    ///
    /// Returns an error if the index is missing or the shard is unavailable.
    /// The stripe serialising routed point ops on `(name, id)`. Hashing the
    /// index name in too keeps equal ids across different indexes - and
    /// adversarial id distributions - from serialising on the same stripe
    /// (review P2).
    /// This vindex's version allocator, created empty on first use.
    fn version_counter(&self, name: &str) -> Arc<std::sync::atomic::AtomicU64> {
        if let Some(c) = self.inner.versions.read().get(name) {
            return c.clone();
        }
        self.inner
            .versions
            .write()
            .entry(name.to_owned())
            .or_insert_with(|| Arc::new(std::sync::atomic::AtomicU64::new(0)))
            .clone()
    }

    /// Allocate the version of a user write to `name`, never below `at_least`.
    ///
    /// The floor is what makes this safe without a durable counter: `at_least`
    /// is the version the caller already knows for the row, so a value read
    /// off disk can never be reissued even if the allocator was seeded from a
    /// map that had not seen it. The result is strictly greater, so a user
    /// write always beats every copy of the row that already exists.
    fn next_version(&self, name: &str, at_least: u64) -> u64 {
        use std::sync::atomic::Ordering;
        let c = self.version_counter(name);
        c.fetch_max(at_least, Ordering::SeqCst);
        c.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Raise the allocator's floor to a version seen on disk.
    fn observe_version(&self, name: &str, version: u64) {
        self.version_counter(name)
            .fetch_max(version, std::sync::atomic::Ordering::SeqCst);
    }

    fn owner_stripe(&self, name: &str, id: u64) -> &tokio::sync::Mutex<()> {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        name.hash(&mut h);
        id.hash(&mut h);
        &self.inner.owner_locks[(h.finish() % self.inner.owner_locks.len() as u64) as usize]
    }

    /// The shard a point op for `id` addresses: the owner map for a
    /// semantically-resharded vindex, hash otherwise (an id absent from the
    /// map is unknown; any shard answers None, hash picks one).
    fn point_shard(&self, name: &str, id: u64) -> usize {
        if let Some(m) = self.inner.owners.read().get(name)
            && let Some(&(s, _, _)) = m.get(&id)
        {
            return usize::from(s);
        }
        shard_for(&id.to_le_bytes(), self.inner.n)
    }

    /// A routed vindex whose owner map is missing (fresh open) rebuilds it
    /// before the first point op resolves: correctness cannot depend on a
    /// caller remembering an init step.
    async fn ensure_owner_map(&self, name: &str) -> Result<(), ShardError> {
        if self.inner.routers.read().contains_key(name)
            && !self.inner.owners.read().contains_key(name)
        {
            self.rebuild_owner_maps().await?;
        }
        Ok(())
    }

    pub async fn vdel(&self, name: &str, id: u64, tenant: u128) -> Result<bool, ShardError> {
        self.ensure_owner_map(name).await?;
        // Serialise against a concurrent vset on the same id. Taken whether or
        // not the index is routed: the delete is now a commit followed by a
        // blob reclamation with an await between them, so it has the same
        // interleaving to lose as the write does.
        let _guard = self.owner_stripe(name, id).lock().await;
        let shard = self.point_shard(name, id);
        // A user delete allocates, exactly like a user write: the tombstone
        // has to beat every copy of the row that exists, including one a
        // concurrent relocation is still carrying.
        let known = self
            .inner
            .owners
            .read()
            .get(name)
            .and_then(|m| m.get(&id).map(|&(_, _, v)| v))
            .unwrap_or(0);
        let version = self.next_version(name, known);
        let req = ShardReq::Vdel {
            name: name.to_owned(),
            id,
            tenant,
            // The row really is leaving the tenant: this is the one delete
            // that credits the quota.
            effect: crate::quota::QuotaEffect::Delete,
            version: Some(version),
        };
        let existed = match self.call(shard, req).await? {
            ShardResp::Existed(b) => b,
            ShardResp::Err(e) => return Err(ShardError::Storage(e)),
            _ => return Err(ShardError::Unavailable),
        };
        // A replicated id dies everywhere: the replica must not survive as a
        // ghost the search could resurrect.
        let replica = self
            .inner
            .owners
            .read()
            .get(name)
            .and_then(|m| m.get(&id).and_then(|&(_, r, _)| r));
        if let Some(rep) = replica {
            // A failed replica delete must not report success and drop the map
            // entry: that would leave a resurrectable ghost (review finding).
            match self
                .call(
                    usize::from(rep),
                    ShardReq::Vdel {
                        name: name.to_owned(),
                        id,
                        tenant,
                        // The primary's delete above already credited this row.
                        // Crediting again would drop the counter by two for one
                        // logical row.
                        effect: crate::quota::QuotaEffect::Replica,
                        // The same delete, in a second place: one version.
                        version: Some(version),
                    },
                )
                .await?
            {
                ShardResp::Existed(_) => {}
                ShardResp::Err(e) => return Err(ShardError::Storage(e)),
                _ => return Err(ShardError::Unavailable),
            }
        }
        if existed && let Some(m) = self.inner.owners.write().get_mut(name) {
            m.remove(&id);
        }
        Ok(existed)
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
        tenant: u128,
        want_payload: bool,
        filter: Option<Filter>,
    ) -> Result<Vec<(u64, f32, Option<Bytes>)>, ShardError> {
        self.vsearch_with_probe(
            name,
            query,
            k,
            l_search,
            tenant,
            want_payload,
            filter,
            probe_default(),
        )
        .await
    }

    /// [`vsearch`](Self::vsearch) with an explicit probe width. With a
    /// semantic router and `probe > 0`, an UNFILTERED search asks only the
    /// `probe` shards whose centroids are nearest the query - the measured
    /// coverage ceiling at probe 2 on the real corpus is 0,9965. A
    /// filtered search always fans out to every shard: the filter's matches
    /// are orthogonal to semantics and can live anywhere. `probe = 0` (or no
    /// router) keeps the full fan-out.
    #[allow(clippy::too_many_arguments)] // vsearch's params plus the probe
    pub async fn vsearch_with_probe(
        &self,
        name: &str,
        query: Vec<f32>,
        k: usize,
        l_search: u32,
        tenant: u128,
        want_payload: bool,
        filter: Option<Filter>,
        probe: usize,
    ) -> Result<Vec<(u64, f32, Option<Bytes>)>, ShardError> {
        let _permit = self
            .inner
            .vsearch_admission
            .as_ref()
            .map(|admission| {
                admission
                    .clone()
                    .try_acquire_owned()
                    .map_err(|_| ShardError::Busy)
            })
            .transpose()?;
        // The client waits for the scatter, every reply, and the merge: that
        // whole span is the search, and it is what the histogram must observe.
        let started = std::time::Instant::now();
        let targets: Vec<usize> = match (&filter, probe, self.router(name)) {
            (None, p, Some(router)) if p > 0 && p < self.inner.n => {
                let cn = |v: &[f32]| {
                    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                    v.iter().map(|x| x / n).collect::<Vec<f32>>()
                };
                let qn = cn(&query);
                let mut sims: Vec<(f32, usize)> = (0..router.k)
                    .map(|j| {
                        let c = &router.centroids[j * router.dim..(j + 1) * router.dim];
                        let s: f32 = qn.iter().zip(c).map(|(a, b)| a * b).sum();
                        (s, j)
                    })
                    .collect();
                sims.sort_by(|a, b| b.0.total_cmp(&a.0));
                sims.into_iter().take(p).map(|(_, j)| j).collect()
            }
            _ => (0..self.inner.n).collect(),
        };
        let mut pending = Vec::with_capacity(targets.len());
        for &shard in &targets {
            let sender = &self.inner.senders[shard];
            let (tx, rx) = oneshot::channel();
            let req = ShardReq::Vsearch {
                name: name.to_owned(),
                query: query.clone(),
                k,
                l_search,
                tenant,
                want_payload,
                filter: filter.clone(),
            };
            sender
                .send(ShardMsg { req, reply: tx })
                .await
                .map_err(|_| ShardError::Unavailable)?;
            pending.push((shard, rx));
        }
        // Hits carry the shard that produced them, so the dedup below can
        // prefer the COMMITTED copy rather than the best-scoring one.
        // (id, cosine, payload, answering shard, version)
        let mut merged: Vec<(u64, f32, Option<Bytes>, usize, u64)> = Vec::new();
        let mut first_err = None;
        for (shard, rx) in pending {
            match rx.await.map_err(|_| ShardError::Unavailable)? {
                ShardResp::Vsearch(hits) => {
                    merged.extend(hits.into_iter().map(|(id, s, p, v)| (id, s, p, shard, v)));
                }
                ShardResp::Err(e) => {
                    first_err.get_or_insert(e);
                }
                _ => return Err(ShardError::Unavailable),
            }
        }
        // FAIL-CLOSED: any shard failing fails the search.
        //
        // It used to surface an error only when the merged set came back
        // EMPTY, so a shard that failed while others returned hits produced a
        // shorter top-k that looks exactly like a complete one. The client
        // cannot tell the difference - not from the shape of the answer, not
        // from the scores, not from the count, since k is a maximum and a
        // genuine query can legitimately return fewer. An incomplete top-k is
        // not a slightly worse answer; it is a wrong answer that cannot be
        // recognised as one, and it silently corrupts anything built on top:
        // a RAG context missing its best passage, a dedup that misses the
        // duplicate, a recommendation that omits the obvious.
        //
        // Consistency over availability, deliberately. A best-effort mode is
        // a reasonable thing to want, but it has to be an explicit contract
        // that says so in the response - at minimum a partial flag and which
        // shards failed - not a silent default.
        if let Some(e) = first_err {
            return Err(ShardError::Storage(e));
        }
        // Dedup by id, preferring the copy the OWNER MAP calls live.
        //
        // Best-score-wins was not enough. A duplicate exists on purpose
        // (boundary overlap) and by accident (a crash mid-move, or a cleanup
        // that failed after an overwrite committed on the new shard), and in
        // the accidental case the two copies hold DIFFERENT vectors: the old
        // one and the new one. A query resembling the old vector then scores
        // the stale copy higher, and best-score-wins hands back the value the
        // write replaced - with a confident score. That is the same shape as
        // the stale-vector defects: a wrong answer that looks certain.
        //
        // The owner map is what VGET routes by, so preferring it also makes
        // search and point reads agree, which they otherwise would not.
        let owner_of: Option<std::collections::HashMap<u64, u8>> = {
            let owners = self.inner.owners.read();
            owners.get(name).map(|m| {
                merged
                    .iter()
                    .filter_map(|&(id, _, _, _, _)| m.get(&id).map(|&(p, _, _)| (id, p)))
                    .collect()
            })
        };
        merged.sort_unstable_by(|a, b| {
            let live = |h: &(u64, f32, Option<Bytes>, usize, u64)| {
                owner_of
                    .as_ref()
                    .and_then(|m| m.get(&h.0))
                    .is_some_and(|&p| usize::from(p) == h.3)
            };
            // NEWEST COPY FIRST, by the version each shard reported for the
            // row it answered with. The owner map was the only tie-break here
            // and it is derived state: when it is the thing that is stale -
            // a restart mid-reshard, a rebuild that has not run yet - it
            // points at the copy an overwrite replaced, and search then agrees
            // with the point read on the wrong value.
            //
            // The version is read on the shard that holds the row, but AFTER
            // the walk and under a separate lock, so it is the shard's current
            // version and not necessarily the version of the copy that
            // produced the hit: a write landing in between reports the newer
            // one. That decides which SHARD wins the merge for an id, never
            // which id is returned, and it errs towards the shard that has
            // just been written - which is the one a point read would pick.
            // What it is not is a snapshot.
            //
            // The map still decides what the version cannot: a boundary
            // replica is the SAME copy in a second place, same version, and
            // the primary is the one to return.
            b.4.cmp(&a.4)
                .then_with(|| live(b).cmp(&live(a)))
                // Then by score, as before.
                .then_with(|| b.1.total_cmp(&a.1))
        });
        let mut seen = ahash::AHashSet::new();
        merged.retain(|&(id, _, _, _, _)| seen.insert(id));
        // Scores decide the ranking, so restore that order once one hit per
        // id survives: the pass above only chose WHICH copy of an id to keep.
        merged.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        merged.truncate(k);
        let merged: Vec<(u64, f32, Option<Bytes>)> = merged
            .into_iter()
            .map(|(id, s, p, _, _)| (id, s, p))
            .collect();
        // Shard 0 by convention: a scattered op belongs to no single shard.
        skeg_telemetry::record_op(skeg_telemetry::Op::VSearch, 0, started.elapsed());
        Ok(merged)
    }

    /// A control-plane handle over these shards (enumerate / report RAM /
    /// evict vindexes). The eviction policy lives outside the engine.
    #[must_use]
    pub fn control_handle(&self) -> ControlHandle {
        ControlHandle {
            shards: self.clone(),
        }
    }
}

/// One open vindex on one shard, as seen by the tiering controller. A vindex is
/// sharded, so a logical index produces up to `n_shards` of these.
///
/// `#[non_exhaustive]`: this will grow (resident tier, pinned flag, ...); a new
/// field must not be a breaking change for the policy crate that reads it.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct IndexStat {
    /// Owning tenant id (`0` = single-tenant / anonymous namespace).
    pub tenant: u128,
    /// Index name without the tenant scope prefix.
    pub index: String,
    /// Shard this fragment lives on.
    pub shard: usize,
    /// RAM held by this fragment (graph + tier + delta for disk; f32 rows for
    /// flat).
    pub resident_bytes: usize,
    /// `now_ms()` of the last access on this shard.
    pub last_access_ms: u64,
    /// Live vectors in this fragment.
    pub vectors: usize,
    /// True if the fragment can be evicted and lazily reopened (disk-backed).
    pub evictable: bool,
}

impl IndexStat {
    /// Construct one stat row. Needed because the struct is `#[non_exhaustive]`,
    /// so downstream crates (the policy crate's tests, an alternative backend)
    /// cannot use a struct literal. Takes today's fields; any field added later
    /// defaults here and gets a `with_*` setter, so this signature stays stable
    /// as the struct grows - the whole point of `#[non_exhaustive]`.
    #[must_use]
    pub fn new(
        tenant: u128,
        index: String,
        shard: usize,
        resident_bytes: usize,
        last_access_ms: u64,
        vectors: usize,
        evictable: bool,
    ) -> Self {
        Self {
            tenant,
            index,
            shard,
            resident_bytes,
            last_access_ms,
            vectors,
            evictable,
        }
    }
}

/// Control-plane handle for vindex lifecycle: enumerate open indexes, report
/// their RAM, and evict (non-destructive, lazy-reopen) ones the policy chooses.
///
/// This is the *mechanism* half of the tiering seam. The *policy* - global RAM
/// budget, LRU vs working-set, anti-thrash hysteresis, hot-tenant pinning -
/// lives in a separate crate that drives this handle from a background task.
///
/// Cheap to clone; clones share the same shard worker threads.
#[derive(Clone)]
pub struct ControlHandle {
    shards: ShardSet,
}

impl ControlHandle {
    /// Snapshot of every open vindex across all shards, one row per
    /// (shard, index). The caller aggregates per logical index as it needs.
    pub async fn open_indices(&self) -> Vec<IndexStat> {
        let mut out = Vec::new();
        for shard in 0..self.shards.inner.n {
            if let Ok(ShardResp::IndexStats(rows)) =
                self.shards.call(shard, ShardReq::IndexStats).await
            {
                for (key, resident_bytes, last_access_ms, vectors, evictable) in rows {
                    let (tenant, index) = unscope_key(&key);
                    out.push(IndexStat::new(
                        tenant,
                        index,
                        shard,
                        resident_bytes,
                        last_access_ms,
                        vectors,
                        evictable,
                    ));
                }
            }
        }
        out
    }

    /// Evict `index` for `tenant` from RAM on every shard. Non-destructive: the
    /// files stay and the index reopens lazily on its next access. `Ok(true)`
    /// if any shard had it resident, `Ok(false)` if it was already absent
    /// everywhere.
    ///
    /// # Errors
    ///
    /// Returns an error if a shard is unavailable or reports a storage error.
    pub async fn evict(&self, tenant: u128, index: &str) -> Result<bool, ShardError> {
        let name = scope_key(tenant, index);
        let mut any = false;
        for shard in 0..self.shards.inner.n {
            match self
                .shards
                .call(shard, ShardReq::Evict { name: name.clone() })
                .await?
            {
                ShardResp::Evicted(b) => any |= b,
                ShardResp::Err(e) => return Err(ShardError::Storage(e)),
                _ => return Err(ShardError::Unavailable),
            }
        }
        Ok(any)
    }

    /// Total resident bytes of all open vindexes across all shards. Convenience
    /// for a global RAM budget check.
    pub async fn total_resident_bytes(&self) -> usize {
        self.open_indices()
            .await
            .iter()
            .map(|s| s.resident_bytes)
            .sum()
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
    /// Disk-quota limit applied on `set`. `None` skips enforcement.
    disk_limit: Option<u64>,
}

impl ShardTenantView<'_> {
    /// Apply a disk-quota limit on this view's `set`. An over-limit set is
    /// rejected before anything is written.
    #[must_use]
    pub fn with_disk_limit(mut self, limit: Option<u64>) -> Self {
        self.disk_limit = limit;
        self
    }

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
            .set_scoped(key, value, durability, self.tenant, self.disk_limit)
            .await
    }

    /// APPEND to a key, charged and quota-checked against this tenant. Returns
    /// the new value length.
    ///
    /// # Errors
    ///
    /// Returns an error if the shard is unavailable, storage fails, or the disk
    /// quota would be exceeded.
    pub async fn append(
        &self,
        key: &[u8],
        value: &[u8],
        durability: Durability,
    ) -> Result<u64, ShardError> {
        self.shards
            .append_scoped(key, value, durability, self.tenant, self.disk_limit)
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
    use std::collections::BTreeSet;
    use std::fs;
    use std::sync::{Barrier, mpsc};
    use std::time::Duration;

    use super::*;
    use tempfile::TempDir;

    /// A relocation carries the version it READ, and a shard can already hold
    /// a newer copy of that row: the owner map is derived state and is allowed
    /// to be behind, so a move can arrive at a destination that has moved on.
    ///
    /// The engine refuses the stale vector on its own. What it cannot refuse
    /// is everything hanging off it - the payload was indexed and written to
    /// the vLog whether or not the vector was stored, so the row that stood
    /// ended up described by the payload of the value it had replaced. A wrong
    /// answer with the right vector attached to it.
    #[tokio::test]
    async fn a_stale_internal_write_stores_neither_its_vector_nor_its_payload() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 1).unwrap();
        shards.vindex_create("p", 4, 0, 1).await.unwrap();
        let newer = vec![1.0f32, 0.0, 0.0, 0.0];
        let older = vec![0.0f32, 1.0, 0.0, 0.0];
        let put = |vector: Vec<f32>, version: u64, blob: &'static str| ShardReq::Vset {
            name: "p".to_owned(),
            id: 7,
            vector,
            tenant: 0,
            limit: None,
            effect: crate::quota::QuotaEffect::Move,
            version: Some(version),
            payload: Some(Bytes::from_static(blob.as_bytes())),
        };
        assert!(matches!(
            shards
                .call(0, put(newer.clone(), 9, "{\"tag\":\"new\"}"))
                .await
                .unwrap(),
            ShardResp::Done
        ));
        // The straggler: the same row as some earlier read saw it.
        assert!(matches!(
            shards
                .call(0, put(older, 4, "{\"tag\":\"old\"}"))
                .await
                .unwrap(),
            ShardResp::Done,
        ));

        assert_eq!(
            shards.vget("p", 7).await.unwrap().unwrap(),
            newer,
            "the newer vector must stand"
        );
        let hits = shards
            .vsearch("p", newer, 1, 0, 0, true, None)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        let blob = hits[0].2.clone().expect("the payload comes back");
        assert_eq!(
            &blob[..],
            b"{\"tag\":\"new\"}",
            "the stale copy's payload replaced the live row's"
        );
    }

    /// The client-reachable half of the same defect: `SKEG.VDEL` on a flat
    /// vindex with an id nobody ever inserted. Every routed delete carries a
    /// real version, so this used to allocate a dead row per call - unbounded
    /// growth from one command, and the reported RAM counted live rows only,
    /// so nothing anywhere showed it.
    #[tokio::test]
    async fn deleting_unknown_ids_on_a_flat_vindex_does_not_grow_the_reported_ram() {
        const DIM: u32 = 256;
        const N: u64 = 5_000;
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 1).unwrap();
        // backend byte 0 = flat.
        shards.vindex_create("f", DIM, 0, 0).await.unwrap();
        shards
            .vset("f", 1, vec![1.0; DIM as usize], 0, None, None)
            .await
            .unwrap();
        let reported = |stats: &[IndexStat]| -> usize {
            stats
                .iter()
                .filter(|s| s.index == "f")
                .map(|s| s.resident_bytes)
                .sum()
        };
        let before = reported(&shards.control_handle().open_indices().await);

        for id in 0..N {
            assert!(!shards.vdel("f", 1_000_000 + id, 0).await.unwrap());
        }
        let after = reported(&shards.control_handle().open_indices().await);
        assert!(
            after - before < N as usize * 32,
            "{N} deletes of ids the index never held grew it by {} bytes \
             (a dead row each would be {})",
            after - before,
            N as usize * DIM as usize * 4
        );
    }

    #[test]
    fn validate_vindex_name_blocks_path_traversal() {
        // The traversal vectors the native protocol could previously reach.
        for bad in [
            "..",
            ".",
            "../x",
            "../../../../tmp/pwned",
            "a/b",
            "a\\b",
            "with space",
            "nul\0byte",
            "",
        ] {
            assert!(
                validate_vindex_name(bad).is_err(),
                "expected reject: {bad:?}"
            );
        }
        // Legit names, including the `{tenant}::{name}` scoped form RESP3 builds.
        for ok in ["idx", "my-index_1", "v.2", "deadbeefcafe::user_idx"] {
            assert!(validate_vindex_name(ok).is_ok(), "expected accept: {ok:?}");
        }
        // Over-long is rejected.
        assert!(validate_vindex_name(&"a".repeat(256)).is_err());
    }

    #[tokio::test]
    async fn vsearch_pool_rejects_work_when_its_queue_is_full() {
        let pool = VsearchPool::new(1).unwrap();
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();

        let first = pool
            .submit(move || {
                started_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                Ok(Vec::new())
            })
            .unwrap();
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();

        let second = pool.submit(|| Ok(Vec::new())).unwrap();
        let err = pool.submit(|| Ok(Vec::new())).unwrap_err();
        assert_eq!(err, "vsearch queue is full");

        release_tx.send(()).unwrap();
        first.await.unwrap().unwrap();
        second.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn vsearch_pool_runs_jobs_in_parallel() {
        let pool = VsearchPool::new(2).unwrap();
        let barrier = Arc::new(Barrier::new(3));
        let (entered_tx, entered_rx) = mpsc::channel();
        let mut replies = Vec::new();

        for _ in 0..2 {
            let barrier = barrier.clone();
            let entered_tx = entered_tx.clone();
            replies.push(
                pool.submit(move || {
                    entered_tx.send(()).unwrap();
                    barrier.wait();
                    Ok(Vec::new())
                })
                .unwrap(),
            );
        }
        entered_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        entered_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        barrier.wait();
        for reply in replies {
            reply.await.unwrap().unwrap();
        }
    }

    /// One client search must move the vsearch counter by one, not by the
    /// shard count.
    ///
    /// vsearch is the only op that scatters to every shard, and the counter was
    /// ticked inside each shard worker: a single search on an 8-shard server
    /// reported 8 operations, and the latency histogram measured per-shard work
    /// instead of what the client waited for. Both numbers are read as query
    /// traffic and query latency, so both were wrong by the shard count.
    #[tokio::test]
    async fn one_search_counts_as_one_operation_not_one_per_shard() {
        const SHARDS: usize = 4;
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open_mode_with_workers(
            dir.path(),
            SHARDS,
            false,
            skeg_vector::QuantKind::Int8,
            1,
        )
        .unwrap();
        shards.vindex_create("idx", 4, 0, 0).await.unwrap();
        for id in 0..8u64 {
            shards
                .vset("idx", id, vec![id as f32 + 1.0; 4], 0, None, None)
                .await
                .unwrap();
        }

        let before = skeg_telemetry::op_total(skeg_telemetry::Op::VSearch);
        for _ in 0..3 {
            shards
                .vsearch("idx", vec![1.0; 4], 2, 16, 0, false, None)
                .await
                .unwrap();
        }
        assert_eq!(
            skeg_telemetry::op_total(skeg_telemetry::Op::VSearch) - before,
            3,
            "3 searches on {SHARDS} shards must count as 3 operations"
        );
    }

    /// A reopened index must serve a filtered search without rebuilding its
    /// payload index first.
    ///
    /// The rebuild reads every live id's payload from the vlog, one await per
    /// record, and it was deferred to the first filtered search. On a 223k
    /// corpus that first search took 5.2 seconds against 6 ms for every one
    /// after it: the whole cost of reopening landed on whichever user arrived
    /// first. It belongs at open, where the readiness barrier already waits.
    #[tokio::test]
    async fn reopen_loads_the_payload_index_before_serving() {
        let dir = TempDir::new().unwrap();
        {
            let shards = ShardSet::open_mode_with_workers(
                dir.path(),
                2,
                false,
                skeg_vector::QuantKind::Int8,
                1,
            )
            .unwrap();
            shards.vindex_create("idx", 4, 0, 1).await.unwrap();
            for id in 0..16u64 {
                shards
                    .vset(
                        "idx",
                        id,
                        vec![id as f32 + 1.0; 4],
                        0,
                        None,
                        Some(Bytes::from(format!("n={id}"))),
                    )
                    .await
                    .unwrap();
            }
        }

        let shards =
            ShardSet::open_mode_with_workers(dir.path(), 2, false, skeg_vector::QuantKind::Int8, 1)
                .unwrap();
        let before = skeg_telemetry::counter_value(skeg_telemetry::Counter::PayloadIndexRebuilds);
        let hits = shards
            .vsearch(
                "idx",
                vec![1.0; 4],
                4,
                32,
                0,
                false,
                Some(crate::payload::parse_filter("n EXISTS").unwrap()),
            )
            .await
            .unwrap();
        assert!(
            !hits.is_empty(),
            "the filter must find the reloaded payloads"
        );
        assert_eq!(
            skeg_telemetry::counter_value(skeg_telemetry::Counter::PayloadIndexRebuilds) - before,
            0,
            "the first filtered search rebuilt the payload index: that cost belongs \
             at open, not on whoever queries first"
        );
    }

    /// A payload overwritten after the cache was stamped must never be served
    /// from the cache.
    ///
    /// This is the failure the whole design is built around. The payload blobs
    /// live in the vlog and keep changing after the cache file is written, so a
    /// cache trusted on age alone would answer filtered searches from values
    /// that no longer exist: results silently missing, no error anywhere. The
    /// cache is therefore stamped with the vlog snapshot position and an id
    /// whose key appears in the replayed tail is refused.
    #[tokio::test]
    async fn a_payload_changed_after_the_snapshot_is_not_served_from_the_cache() {
        let dir = TempDir::new().unwrap();
        {
            let shards = ShardSet::open_mode_with_workers(
                dir.path(),
                1,
                false,
                skeg_vector::QuantKind::Int8,
                1,
            )
            .unwrap();
            shards.vindex_create("idx", 4, 0, 1).await.unwrap();
            for id in 0..8u64 {
                shards
                    .vset(
                        "idx",
                        id,
                        vec![id as f32 + 1.0; 4],
                        0,
                        None,
                        Some(Bytes::from("t=old")),
                    )
                    .await
                    .unwrap();
            }
            shards.write_snapshot_and_payload_indexes().await;
            // After the stamp: this id's payload changes, the cache still says
            // "t=old" for it.
            shards
                .vset("idx", 3, vec![4.0; 4], 0, None, Some(Bytes::from("t=new")))
                .await
                .unwrap();
        }

        let cached_before =
            skeg_telemetry::counter_value(skeg_telemetry::Counter::PayloadIndexFromDisk);
        let shards =
            ShardSet::open_mode_with_workers(dir.path(), 1, false, skeg_vector::QuantKind::Int8, 1)
                .unwrap();
        let old = shards
            .vsearch(
                "idx",
                vec![1.0; 4],
                8,
                32,
                0,
                false,
                Some(crate::payload::parse_filter("t = old").unwrap()),
            )
            .await
            .unwrap();
        let new = shards
            .vsearch(
                "idx",
                vec![1.0; 4],
                8,
                32,
                0,
                false,
                Some(crate::payload::parse_filter("t = new").unwrap()),
            )
            .await
            .unwrap();
        assert!(
            new.iter().any(|(id, _, _)| *id == 3),
            "id 3 was rewritten to t=new and the filter misses it: the cache served a dead value"
        );
        assert!(
            !old.iter().any(|(id, _, _)| *id == 3),
            "id 3 still reads t=old: the cache overrode what the log says"
        );
        assert_eq!(old.len(), 7, "the other seven must stay t=old");
        // Without this the test would also pass with the cache never read: green,
        // and proving nothing.
        assert!(
            skeg_telemetry::counter_value(skeg_telemetry::Counter::PayloadIndexFromDisk)
                - cached_before
                >= 7,
            "the cache was never read: this test is not proving what it claims"
        );
    }

    /// A tenant's filtered search must still work after a restart.
    ///
    /// Payload keys are built from the tenant-scoped vindex name. Warming with
    /// the bare name builds different keys, finds nothing, and still marks the
    /// index loaded, so every filtered search for that tenant comes back empty
    /// with no error anywhere. Tenant 0 makes scoped and bare identical, so a
    /// single-tenant test cannot see it.
    #[tokio::test]
    async fn a_tenant_filtered_search_survives_a_restart() {
        const T: u128 = 0x2b;
        let dir = TempDir::new().unwrap();
        let scoped = scope_key(T, "idx");
        {
            let shards = ShardSet::open_mode_with_workers(
                dir.path(),
                1,
                false,
                skeg_vector::QuantKind::Int8,
                1,
            )
            .unwrap();
            shards.vindex_create_scoped(&scoped, 4, 0, 1).await.unwrap();
            for id in 0..6u64 {
                shards
                    .vset(
                        &scoped,
                        id,
                        vec![id as f32 + 1.0; 4],
                        T,
                        None,
                        Some(Bytes::from("lic=mit")),
                    )
                    .await
                    .unwrap();
            }
        }

        let shards =
            ShardSet::open_mode_with_workers(dir.path(), 1, false, skeg_vector::QuantKind::Int8, 1)
                .unwrap();
        let hits = shards
            .vsearch(
                &scoped,
                vec![1.0; 4],
                6,
                32,
                T,
                false,
                Some(crate::payload::parse_filter("lic = mit").unwrap()),
            )
            .await
            .unwrap();
        assert_eq!(
            hits.len(),
            6,
            "the tenant's filtered search came back with {} of 6 payloads after a restart",
            hits.len()
        );
    }

    /// Helper: a one-shard set over `dir` with a disk-backed vindex.
    async fn open_one(dir: &std::path::Path) -> ShardSet {
        ShardSet::open_mode_with_workers(dir, 1, false, skeg_vector::QuantKind::Int8, 1).unwrap()
    }

    fn filt(s: &str) -> Option<Filter> {
        Some(crate::payload::parse_filter(s).unwrap())
    }

    /// A vindex evicted and reopened while the server runs must reflect writes
    /// that landed after the store opened.
    ///
    /// The persisted payload cache is only safe during the open-time warm,
    /// where the log tail describes every change since the stamp. A vindex
    /// reopened hours later has no such record: writes since then are invisible
    /// to that tail, so trusting the cache would serve values that were
    /// overwritten. This is the case that says the cache must stay out of the
    /// lazy reopen path.
    #[tokio::test]
    async fn an_evicted_vindex_reopens_with_writes_that_came_after_open() {
        let dir = TempDir::new().unwrap();
        {
            let shards = open_one(dir.path()).await;
            shards.vindex_create("idx", 4, 0, 1).await.unwrap();
            for id in 0..6u64 {
                shards
                    .vset(
                        "idx",
                        id,
                        vec![id as f32 + 1.0; 4],
                        0,
                        None,
                        Some(Bytes::from("t=old")),
                    )
                    .await
                    .unwrap();
            }
            shards.write_snapshot_and_payload_indexes().await;
        }

        let shards = open_one(dir.path()).await;
        // Warm has run and the cache says every id is t=old. Now change one and
        // force the vindex through the evict/reopen path.
        shards
            .vset("idx", 2, vec![3.0; 4], 0, None, Some(Bytes::from("t=new")))
            .await
            .unwrap();
        assert!(
            shards.control_handle().evict(0, "idx").await.unwrap(),
            "the vindex should have been evicted"
        );

        let new = shards
            .vsearch("idx", vec![1.0; 4], 6, 32, 0, false, filt("t = new"))
            .await
            .unwrap();
        assert!(
            new.iter().any(|(id, _, _)| *id == 2),
            "id 2 was rewritten before the evict and the reopen lost it"
        );
        let old = shards
            .vsearch("idx", vec![1.0; 4], 6, 32, 0, false, filt("t = old"))
            .await
            .unwrap();
        assert!(
            !old.iter().any(|(id, _, _)| *id == 2),
            "id 2 came back as t=old"
        );
        assert_eq!(old.len(), 5);
    }

    /// Dropping a vindex and creating another with the same name must not
    /// resurrect the old payloads through the cache file.
    #[tokio::test]
    async fn a_dropped_name_reused_does_not_resurrect_old_payloads() {
        let dir = TempDir::new().unwrap();
        {
            let shards = open_one(dir.path()).await;
            shards.vindex_create("idx", 4, 0, 1).await.unwrap();
            for id in 0..4u64 {
                shards
                    .vset(
                        "idx",
                        id,
                        vec![id as f32 + 1.0; 4],
                        0,
                        None,
                        Some(Bytes::from("gen=one")),
                    )
                    .await
                    .unwrap();
            }
            shards.write_snapshot_and_payload_indexes().await;
            shards.vindex_drop("idx", 0).await.unwrap();
            shards.vindex_create("idx", 4, 0, 1).await.unwrap();
            for id in 0..4u64 {
                shards
                    .vset(
                        "idx",
                        id,
                        vec![id as f32 + 1.0; 4],
                        0,
                        None,
                        Some(Bytes::from("gen=two")),
                    )
                    .await
                    .unwrap();
            }
        }

        let shards = open_one(dir.path()).await;
        let one = shards
            .vsearch("idx", vec![1.0; 4], 8, 32, 0, false, filt("gen = one"))
            .await
            .unwrap_or_default();
        let two = shards
            .vsearch("idx", vec![1.0; 4], 8, 32, 0, false, filt("gen = two"))
            .await
            .unwrap();
        assert!(
            one.is_empty(),
            "the dropped generation came back: {} hits",
            one.len()
        );
        assert_eq!(two.len(), 4, "the live generation is incomplete");
    }

    /// A deleted vector must not come back through the payload cache.
    #[tokio::test]
    async fn a_deleted_vector_stays_deleted_across_a_restart() {
        let dir = TempDir::new().unwrap();
        {
            let shards = open_one(dir.path()).await;
            shards.vindex_create("idx", 4, 0, 1).await.unwrap();
            for id in 0..5u64 {
                shards
                    .vset(
                        "idx",
                        id,
                        vec![id as f32 + 1.0; 4],
                        0,
                        None,
                        Some(Bytes::from("k=v")),
                    )
                    .await
                    .unwrap();
            }
            shards.write_snapshot_and_payload_indexes().await;
            assert!(shards.vdel("idx", 3, 0).await.unwrap());
        }

        let shards = open_one(dir.path()).await;
        let hits = shards
            .vsearch("idx", vec![1.0; 4], 8, 32, 0, false, filt("k = v"))
            .await
            .unwrap();
        assert!(
            !hits.iter().any(|(id, _, _)| *id == 3),
            "the deleted id came back"
        );
        assert_eq!(hits.len(), 4);
    }

    /// Plain KV must behave the same as before the uncached read path existed:
    /// a normal get still caches, a delete still hides the key, and neither is
    /// confused by a scan having read the same key.
    #[tokio::test]
    async fn kv_semantics_are_unchanged_by_the_uncached_read_path() {
        let dir = TempDir::new().unwrap();
        let shards = open_one(dir.path()).await;
        shards.set(b"a", b"1", Durability::Kernel).await.unwrap();
        shards.set(b"b", b"2", Durability::Kernel).await.unwrap();
        assert_eq!(
            shards.get(b"a").await.unwrap().as_deref(),
            Some(b"1".as_slice())
        );
        shards.set(b"a", b"3", Durability::Kernel).await.unwrap();
        assert_eq!(
            shards.get(b"a").await.unwrap().as_deref(),
            Some(b"3".as_slice()),
            "an overwrite must be visible through the cache"
        );
        assert!(shards.del(b"a", Durability::Kernel).await.unwrap());
        assert_eq!(
            shards.get(b"a").await.unwrap(),
            None,
            "a deleted key must stay deleted"
        );
        assert_eq!(
            shards.get(b"b").await.unwrap().as_deref(),
            Some(b"2".as_slice())
        );
        assert_eq!(shards.get(b"missing").await.unwrap(), None);
    }

    /// Consolidating rebuilds the graph; the payload index must survive it and
    /// keep answering filters over the same ids.
    #[tokio::test]
    async fn consolidate_keeps_the_payload_index_answering() {
        let dir = TempDir::new().unwrap();
        let shards = open_one(dir.path()).await;
        shards.vindex_create("idx", 4, 0, 1).await.unwrap();
        for id in 0..40u64 {
            let tag = if id % 2 == 0 { "p=even" } else { "p=odd" };
            shards
                .vset(
                    "idx",
                    id,
                    vec![id as f32 + 1.0; 4],
                    0,
                    None,
                    Some(Bytes::from(tag)),
                )
                .await
                .unwrap();
        }
        let before = shards
            .vsearch("idx", vec![1.0; 4], 40, 64, 0, false, filt("p = even"))
            .await
            .unwrap();
        shards.vindex_consolidate("idx").await.unwrap();
        let after = shards
            .vsearch("idx", vec![1.0; 4], 40, 64, 0, false, filt("p = even"))
            .await
            .unwrap();
        assert_eq!(before.len(), 20, "half the vectors are even");
        assert_eq!(
            after.len(),
            before.len(),
            "consolidate changed what the filter matches"
        );
    }

    /// The fold budget gates heavy builds and never gates a flush.
    ///
    /// This is the guard against the measured incident: an explicit
    /// consolidate broadcasts to every shard, and eight capped folds still
    /// took 947% of a 10-core machine. With the budget exhausted a
    /// consolidate must park; a flush must not, because gating the ingest
    /// path behind a parked fold would stall writes.
    #[tokio::test]
    async fn fold_budget_parks_heavy_builds_and_exempts_the_flush() {
        assert!(is_budgeted("consolidate"));
        assert!(is_budgeted("runs-merge"));
        assert!(is_budgeted("delete-patch"));
        assert!(!is_budgeted("flush"));

        let dir = TempDir::new().unwrap();
        let vdir = dir.path().join("vindex-b");
        let mut idx = DiskVamanaIndex::create_empty_with_tier(
            &vdir,
            64,
            64,
            QuantKind::TurboQuant { bits: 2 },
        )
        .unwrap();
        idx.set_auto_flush(false);
        for id in 0u64..(FLUSH_ROWS as u64 + 64) {
            idx.insert(id, &tvec(id + 1)).unwrap();
        }
        let arc: VectorEntry = Arc::new(RwLock::new(Vindex::new(
            VectorBackend::Disk(Box::new(idx)),
            4,
        )));

        // Exhaust the budget.
        let held = fold_budget()
            .acquire_many(fold_budget().available_permits() as u32)
            .await
            .unwrap();

        // A flush must complete regardless.
        let d = vdir.clone();
        let ran = try_off_thread_maintenance(
            &arc,
            "flush",
            true,
            |b| b.flush_begin(),
            move |job| job.build(&d),
            |b, built| b.flush_finish(built),
        )
        .await
        .unwrap();
        assert_eq!(
            ran,
            MaintenanceOutcome::Ran,
            "the flush must run with the budget exhausted"
        );

        // A consolidate must park until a permit frees.
        let arc2 = arc.clone();
        let d2 = vdir.clone();
        let fold = tokio::spawn(async move {
            try_off_thread_maintenance(
                &arc2,
                "consolidate",
                true,
                |b| b.consolidate_begin(),
                move |job| job.build(&d2),
                |b, built| b.consolidate_finish(built),
            )
            .await
        });
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        assert!(
            !fold.is_finished(),
            "the consolidate ran past an exhausted fold budget"
        );
        drop(held);
        let ran = fold.await.unwrap().unwrap();
        assert_eq!(
            ran,
            MaintenanceOutcome::Ran,
            "the parked consolidate must complete once a permit frees"
        );
    }

    #[tokio::test]
    async fn a_failed_flush_build_does_not_block_every_later_flush() {
        let dir = TempDir::new().unwrap();
        let vdir = dir.path().join("vindex-failed-flush");
        let mut idx = DiskVamanaIndex::create_empty_with_tier(
            &vdir,
            16,
            64,
            QuantKind::TurboQuant { bits: 2 },
        )
        .unwrap();
        idx.set_auto_flush(false);
        for id in 0u64..64 {
            idx.insert(id, &[id as f32; 16]).unwrap();
        }
        let arc: VectorEntry = Arc::new(RwLock::new(Vindex::new(
            VectorBackend::Disk(Box::new(idx)),
            4,
        )));

        let failed = try_off_thread_maintenance_with_abort(
            &arc,
            "flush",
            true,
            |b| b.flush_begin(),
            |_job| -> std::io::Result<FlushBuilt> {
                Err(std::io::Error::other("injected build failure"))
            },
            |b, built| b.flush_finish(built),
            VectorBackend::flush_abort,
        )
        .await;
        assert!(
            failed.unwrap_err().contains("injected build failure"),
            "the injected failure must reach the caller"
        );

        let retry = arc.write().backend.flush_begin().unwrap();
        assert!(
            retry.is_some(),
            "a build failure must return the staged delta so the next tick can retry"
        );
    }

    #[tokio::test]
    async fn a_failed_flush_finish_is_precommit_and_retryable() {
        let dir = TempDir::new().unwrap();
        let vdir = dir.path().join("vindex-failed-finish");
        let mut idx = DiskVamanaIndex::create_empty_with_tier(
            &vdir,
            16,
            64,
            QuantKind::TurboQuant { bits: 2 },
        )
        .unwrap();
        idx.set_auto_flush(false);
        for id in 0u64..64 {
            idx.insert(id, &[id as f32; 16]).unwrap();
        }
        let arc: VectorEntry = Arc::new(RwLock::new(Vindex::new(
            VectorBackend::Disk(Box::new(idx)),
            4,
        )));

        // `compact_wal` creates this path as a file. A directory is an
        // inexpensive deterministic I/O failure after the run build, inside
        // finish, with no environment-specific failpoint machinery.
        std::fs::create_dir(vdir.join("delta.log.compact")).unwrap();
        let build_dir = vdir.clone();
        let failed = try_off_thread_maintenance_with_abort(
            &arc,
            "flush",
            true,
            |b| b.flush_begin(),
            move |job| job.build(&build_dir),
            |b, built| b.flush_finish(built),
            VectorBackend::flush_abort,
        )
        .await;
        assert!(
            failed.unwrap_err().contains("finish failed"),
            "the injected finish failure must reach the caller"
        );

        let retry = arc.write().backend.flush_begin().unwrap();
        assert!(
            retry.is_some(),
            "a pre-commit finish failure must restore staging for a retry"
        );
    }

    /// A merge that COMMITS and then cannot unlink what it replaced is a
    /// merge that happened.
    ///
    /// `merge_runs_finish` marks the merged run durable and splices it in -
    /// that is the commit - and only then removes the old run directories.
    /// Those removals used to use `?`, so a failure returned an error, the
    /// caller ran the abort callback, and the ladder counted a failure: all
    /// of it over a merge that was already serving queries.
    ///
    /// The failpoint is targeted at the post-commit step alone: an old run
    /// directory with no write permission cannot have its contents unlinked,
    /// while everything before the commit touches other paths entirely.
    #[tokio::test]
    async fn a_merge_that_cannot_unlink_its_old_runs_still_counts_as_done() {
        use std::os::unix::fs::PermissionsExt;
        let dir = TempDir::new().unwrap();
        let vdir = dir.path().join("vindex-m");
        let mut idx = DiskVamanaIndex::create_empty_with_tier(
            &vdir,
            16,
            64,
            QuantKind::TurboQuant { bits: 2 },
        )
        .unwrap();
        idx.set_auto_flush(false);
        // Two runs, so there is a merge to do and an old directory to remove.
        for round in 0..2u64 {
            for id in 0..64u64 {
                idx.insert(round * 1000 + id, &[id as f32; 16]).unwrap();
            }
            let built = idx.flush_begin().unwrap().unwrap().build(&vdir).unwrap();
            idx.flush_finish(built).unwrap().expect_clean();
        }
        assert_eq!(
            idx.run_count(),
            2,
            "the fixture must give the merge something"
        );
        let live_before = idx.len();

        let arc: VectorEntry = Arc::new(RwLock::new(Vindex::new(
            VectorBackend::Disk(Box::new(idx)),
            4,
        )));

        let old_run = vdir.join("run-0");
        assert!(
            old_run.exists(),
            "the fixture must have an old run to unlink"
        );
        let saved = std::fs::metadata(&old_run).unwrap().permissions();
        std::fs::set_permissions(&old_run, PermissionsExt::from_mode(0o555)).unwrap();

        let build_dir = vdir.clone();
        let outcome = off_thread_maintenance(
            &arc,
            "runs-merge",
            0,
            |b| b.merge_runs_begin(),
            move |job| job.build(&build_dir),
            |b, built| b.merge_runs_finish(built),
        )
        .await;
        std::fs::set_permissions(&old_run, saved).unwrap();

        assert_eq!(
            outcome,
            MaintenanceOutcome::Ran,
            "a committed merge whose cleanup failed is not a failed merge"
        );
        let g = arc.read();
        assert_eq!(
            g.backend.run_count(),
            1,
            "and the merge really did take effect"
        );
        assert_eq!(g.backend.len(), live_before, "with every row still live");
    }

    /// The same rule for the fold, whose commit is the atomic CURRENT flip.
    ///
    /// Everything after `install_base_generation` used `?`, so a run directory
    /// that would not unlink turned a published generation into a reported
    /// failure - and `discard_runs` returned early, leaving those runs still
    /// listed in memory beside a base that had just folded them in.
    #[tokio::test]
    async fn a_fold_that_cannot_unlink_its_runs_still_counts_as_done() {
        use std::os::unix::fs::PermissionsExt;
        let dir = TempDir::new().unwrap();
        let vdir = dir.path().join("vindex-c");
        let mut idx = DiskVamanaIndex::create_empty_with_tier(
            &vdir,
            16,
            64,
            QuantKind::TurboQuant { bits: 2 },
        )
        .unwrap();
        idx.set_auto_flush(false);
        for id in 0..128u64 {
            idx.insert(id, &[id as f32; 16]).unwrap();
        }
        let built = idx.flush_begin().unwrap().unwrap().build(&vdir).unwrap();
        idx.flush_finish(built).unwrap().expect_clean();
        assert_eq!(idx.run_count(), 1);
        let live_before = idx.len();

        let arc: VectorEntry = Arc::new(RwLock::new(Vindex::new(
            VectorBackend::Disk(Box::new(idx)),
            4,
        )));

        let run = vdir.join("run-0");
        let saved = std::fs::metadata(&run).unwrap().permissions();
        std::fs::set_permissions(&run, PermissionsExt::from_mode(0o555)).unwrap();

        let build_dir = vdir.clone();
        let outcome = off_thread_maintenance(
            &arc,
            "consolidate",
            0,
            |b| b.consolidate_begin(),
            move |job| job.build(&build_dir),
            |b, built| b.consolidate_finish(built),
        )
        .await;
        std::fs::set_permissions(&run, saved).unwrap();

        assert_eq!(
            outcome,
            MaintenanceOutcome::Ran,
            "a published generation whose cleanup failed is not a failed fold"
        );
        let g = arc.read();
        assert_eq!(
            g.backend.run_count(),
            0,
            "the run must be gone from the layer set even though its directory \
             would not unlink: leaving it listed puts the pre-fold rows on top \
             of the base that just folded them in"
        );
        assert_eq!(g.backend.len(), live_before, "with every row still live");
    }

    #[tokio::test]
    async fn vsearch_pool_rejects_before_scattering_when_saturated() {
        let dir = TempDir::new().unwrap();
        let shards =
            ShardSet::open_mode_with_workers(dir.path(), 2, false, skeg_vector::QuantKind::Int8, 1)
                .unwrap();
        let permit = shards
            .inner
            .vsearch_admission
            .as_ref()
            .unwrap()
            .clone()
            .try_acquire_owned()
            .unwrap();

        let err = shards
            .vsearch("idx", vec![0.0; 4], 1, 0, 0, false, None)
            .await
            .unwrap_err();
        assert!(matches!(err, ShardError::Busy));
        drop(permit);
    }

    #[tokio::test]
    async fn worker_pool_returns_complete_results_for_concurrent_scatter() {
        let dir = TempDir::new().unwrap();
        let shards =
            ShardSet::open_mode_with_workers(dir.path(), 2, false, skeg_vector::QuantKind::Int8, 2)
                .unwrap();
        shards.vindex_create("idx", 4, 0, 0).await.unwrap();
        for id in 0..8 {
            shards
                .vset("idx", id, vec![id as f32 + 1.0; 4], 0, None, None)
                .await
                .unwrap();
        }

        let left = shards.vsearch("idx", vec![1.0; 4], 8, 0, 0, false, None);
        let right = shards.vsearch("idx", vec![1.0; 4], 8, 0, 0, false, None);
        let (left, right) = tokio::join!(left, right);
        assert_eq!(left.unwrap().len(), 8);
        assert_eq!(right.unwrap().len(), 8);
    }

    #[tokio::test]
    async fn worker_pool_matches_inline_vsearch_results() {
        let inline_dir = TempDir::new().unwrap();
        let pool_dir = TempDir::new().unwrap();
        let inline = ShardSet::open_mode_with_workers(
            inline_dir.path(),
            2,
            false,
            skeg_vector::QuantKind::Int8,
            0,
        )
        .unwrap();
        let pooled = ShardSet::open_mode_with_workers(
            pool_dir.path(),
            2,
            false,
            skeg_vector::QuantKind::Int8,
            2,
        )
        .unwrap();

        for shards in [&inline, &pooled] {
            shards.vindex_create("idx", 4, 0, 0).await.unwrap();
            for id in 0..8 {
                shards
                    .vset(
                        "idx",
                        id,
                        vec![id as f32 + 1.0, 1.0, 0.0, 0.0],
                        0,
                        None,
                        None,
                    )
                    .await
                    .unwrap();
            }
        }

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let inline_ids: Vec<_> = inline
            .vsearch("idx", query.clone(), 8, 0, 0, false, None)
            .await
            .unwrap()
            .into_iter()
            .map(|hit| hit.0)
            .collect();
        let pooled_ids: Vec<_> = pooled
            .vsearch("idx", query, 8, 0, 0, false, None)
            .await
            .unwrap()
            .into_iter()
            .map(|hit| hit.0)
            .collect();

        assert_eq!(pooled_ids, inline_ids);
    }

    /// The plain-query equivalence above misses the two branches where the
    /// pooled and inline routes used to be written separately: the payload
    /// index rebuild that a filter triggers, and the payload fetch. Those are
    /// exactly where a fix applied to one route and not the other would hide,
    /// because a dev run has `workers == 0` and production does not.
    #[tokio::test]
    async fn worker_pool_matches_inline_vsearch_with_filter_and_payload() {
        let inline_dir = TempDir::new().unwrap();
        let pool_dir = TempDir::new().unwrap();
        let inline = ShardSet::open_mode_with_workers(
            inline_dir.path(),
            2,
            false,
            skeg_vector::QuantKind::Int8,
            0,
        )
        .unwrap();
        let pooled = ShardSet::open_mode_with_workers(
            pool_dir.path(),
            2,
            false,
            skeg_vector::QuantKind::Int8,
            2,
        )
        .unwrap();

        for shards in [&inline, &pooled] {
            shards.vindex_create("idx", 4, 0, 0).await.unwrap();
            for id in 0..8u64 {
                let colour = if id % 2 == 0 { "red" } else { "blue" };
                // `parse_fields` reads whitespace-separated `key=value`, not JSON.
                let payload = Bytes::from(format!("colour={colour} id={id}"));
                shards
                    .vset(
                        "idx",
                        id,
                        vec![id as f32 + 1.0, 1.0, 0.0, 0.0],
                        0,
                        None,
                        Some(payload),
                    )
                    .await
                    .unwrap();
            }
        }

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let filter = Filter::Eq(
            "colour".to_owned(),
            crate::payload::Value::Keyword("red".to_owned()),
        );
        let inline_hits = inline
            .vsearch("idx", query.clone(), 8, 0, 0, true, Some(filter.clone()))
            .await
            .unwrap();
        let pooled_hits = pooled
            .vsearch("idx", query, 8, 0, 0, true, Some(filter))
            .await
            .unwrap();

        assert!(!inline_hits.is_empty(), "the filter matched nothing");
        assert_eq!(
            pooled_hits, inline_hits,
            "pooled and inline VSEARCH disagreed under a filter with payloads",
        );
        for hit in &inline_hits {
            assert_eq!(hit.0 % 2, 0, "filter let a non-red id through");
        }
    }

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

    /// A V2 entry naming a wire kind this build does not know is an ERROR.
    ///
    /// This test used to assert the opposite - "unknown kind falls back to
    /// legacy tier selection" - and pinned the dangerous behaviour as the
    /// desired one. The fallback runs through `kind.and_then(from_wire)
    /// .unwrap_or(tier)`, so an unreadable byte opens the index with the
    /// PROCESS DEFAULT quantiser: no crash, no warning, every distance
    /// computed against codes it cannot interpret. It is the same defect
    /// `read_tier` had for the on-disk tier sidecar, in its second home.
    ///
    /// `None` still means "V1 format, no tier recorded", which is a real state
    /// and legitimately takes the default. Absent and unreadable are different
    /// things and only one of them has a safe default.
    #[test]
    fn registry_v2_refuses_an_unknown_kind() {
        let dir = TempDir::new().unwrap();
        let mut bytes = VINDEX_REGISTRY_V2_MAGIC.to_vec();
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&3u16.to_le_bytes());
        bytes.extend_from_slice(b"idx");
        bytes.extend_from_slice(&64u32.to_le_bytes());
        bytes.push(99); // not a supported VINDEX wire kind
        fs::write(dir.path().join(VINDEX_REGISTRY), bytes).unwrap();

        let err = read_registry(dir.path()).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("99"), "must name the byte: {err}");
    }

    #[test]
    fn a_v1_registry_still_has_no_kind_and_that_is_fine() {
        // The old `[u32 count]` format records no tier. That is absence, not
        // corruption, and it keeps taking the caller's default.
        let dir = TempDir::new().unwrap();
        let mut bytes = 1u32.to_le_bytes().to_vec();
        bytes.extend_from_slice(&3u16.to_le_bytes());
        bytes.extend_from_slice(b"idx");
        bytes.extend_from_slice(&64u32.to_le_bytes());
        fs::write(dir.path().join(VINDEX_REGISTRY), bytes).unwrap();

        let entries = read_registry(dir.path()).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].kind, None);
    }

    #[test]
    fn an_absent_registry_is_an_empty_store() {
        let dir = TempDir::new().unwrap();
        assert!(read_registry(dir.path()).unwrap().is_empty());
    }

    #[test]
    fn a_truncated_registry_refuses_instead_of_returning_what_it_got() {
        // "Whatever parsed cleanly" means indexes silently DISAPPEAR: three of
        // seven entries read back as three indexes, with full confidence and
        // no warning. That is the serve-mode failure in a different costume -
        // partial data accepted as complete.
        let dir = TempDir::new().unwrap();
        let mut bytes = VINDEX_REGISTRY_V2_MAGIC.to_vec();
        bytes.extend_from_slice(&2u32.to_le_bytes()); // claims two
        bytes.extend_from_slice(&3u16.to_le_bytes());
        bytes.extend_from_slice(b"one");
        bytes.extend_from_slice(&64u32.to_le_bytes());
        bytes.push(1);
        bytes.extend_from_slice(&3u16.to_le_bytes()); // second entry, cut off
        fs::write(dir.path().join(VINDEX_REGISTRY), bytes).unwrap();

        let err = read_registry(dir.path()).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains('2'), "must name the count: {err}");
    }

    #[test]
    fn trailing_bytes_after_the_last_entry_refuse() {
        // A count that under-reports leaves data nobody reads. Either the
        // writer or the file is wrong; both are worth stopping for.
        let dir = TempDir::new().unwrap();
        let mut bytes = VINDEX_REGISTRY_V2_MAGIC.to_vec();
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&3u16.to_le_bytes());
        bytes.extend_from_slice(b"one");
        bytes.extend_from_slice(&64u32.to_le_bytes());
        bytes.push(1);
        bytes.extend_from_slice(b"leftover");
        fs::write(dir.path().join(VINDEX_REGISTRY), bytes).unwrap();
        assert_eq!(
            read_registry(dir.path()).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn an_invalid_name_refuses_instead_of_being_mangled() {
        // `from_utf8_lossy` turns corrupt bytes into U+FFFD and carries on, so
        // the name in the registry stops matching the directory on disk - and
        // the index quietly cannot be reopened.
        let dir = TempDir::new().unwrap();
        let mut bytes = VINDEX_REGISTRY_V2_MAGIC.to_vec();
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&3u16.to_le_bytes());
        bytes.extend_from_slice(&[0xFF, 0xFE, 0xFD]);
        bytes.extend_from_slice(&64u32.to_le_bytes());
        bytes.push(1);
        fs::write(dir.path().join(VINDEX_REGISTRY), bytes).unwrap();
        assert_eq!(
            read_registry(dir.path()).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn an_enormous_registry_refuses_without_reading_it_all() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join(VINDEX_REGISTRY), vec![0u8; 8 * 1024 * 1024]).unwrap();
        assert_eq!(
            read_registry(dir.path()).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn a_registry_name_that_would_escape_the_data_directory_refuses() {
        // Valid UTF-8 is not a valid name. This string is joined onto the data
        // directory to build a path, and it comes from a file.
        let dir = TempDir::new().unwrap();
        for evil in ["x/../../target", "..", "a/b", "", "n\u{0}m"] {
            let mut bytes = VINDEX_REGISTRY_V2_MAGIC.to_vec();
            bytes.extend_from_slice(&1u32.to_le_bytes());
            bytes.extend_from_slice(&(evil.len() as u16).to_le_bytes());
            bytes.extend_from_slice(evil.as_bytes());
            bytes.extend_from_slice(&64u32.to_le_bytes());
            bytes.push(2);
            fs::write(dir.path().join(VINDEX_REGISTRY), bytes).unwrap();
            assert_eq!(
                read_registry(dir.path()).unwrap_err().kind(),
                std::io::ErrorKind::InvalidData,
                "must refuse {evil:?}"
            );
        }
    }

    #[test]
    fn sixteen_maximum_length_names_survive_a_round_trip() {
        // The exact shape that broke: the reader used the 4 KiB SIDECAR bound
        // on a file that grows with the number of indexes, while the writer had
        // no bound at all. `validate_vindex_name` allows 255 bytes, so
        // 8 + 16 x (2 + 255 + 4 + 1) = 4,200 - the sixteenth index wrote
        // successfully and the next open refused to read the catalogue back.
        let dir = TempDir::new().unwrap();
        let names: Vec<String> = (0..16)
            .map(|i| format!("{i:02}{}", "n".repeat(253)))
            .collect();
        assert!(names.iter().all(|n| n.len() == 255));
        let entries: Vec<(&str, usize, u8, IndexGeneration)> = names
            .iter()
            .map(|n| (n.as_str(), 1024usize, 2u8, IndexGeneration::LEGACY))
            .collect();

        write_registry(dir.path(), &entries).unwrap();
        let size = fs::metadata(dir.path().join(VINDEX_REGISTRY))
            .unwrap()
            .len();
        assert!(size > 4096, "the case only bites over 4 KiB, got {size}");

        let back = read_registry(dir.path()).unwrap();
        assert_eq!(back.len(), 16);
        assert_eq!(back[15].name, names[15]);
    }

    #[test]
    fn a_registry_over_the_shard_limit_refuses_to_be_written() {
        // Refused at write time, not discovered at read time. What the writer
        // is willing to publish and what the reader will accept have to be one
        // number.
        let dir = TempDir::new().unwrap();
        let names: Vec<String> = (0..=MAX_VINDEXES_PER_SHARD)
            .map(|i| format!("n{i}"))
            .collect();
        let entries: Vec<(&str, usize, u8, IndexGeneration)> = names
            .iter()
            .map(|n| (n.as_str(), 8usize, 2u8, IndexGeneration::LEGACY))
            .collect();
        let err = write_registry(dir.path(), &entries).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        assert!(
            !dir.path().join(VINDEX_REGISTRY).exists(),
            "a refused write must not leave a file behind"
        );
    }

    #[test]
    fn the_writer_never_produces_what_the_reader_rejects() {
        // The two bounds are one number, checked here rather than trusted.
        let dir = TempDir::new().unwrap();
        let names: Vec<String> = (0..MAX_VINDEXES_PER_SHARD)
            .map(|i| format!("{i:04}{}", "x".repeat(251)))
            .collect();
        let entries: Vec<(&str, usize, u8, IndexGeneration)> = names
            .iter()
            .map(|n| (n.as_str(), 1024usize, 2u8, IndexGeneration::LEGACY))
            .collect();
        write_registry(dir.path(), &entries).expect("the largest legal catalogue must write");
        assert_eq!(
            read_registry(dir.path()).expect("and must read back").len(),
            MAX_VINDEXES_PER_SHARD
        );
    }

    /// A blob key names one COPY of one row of one incarnation of one name.
    /// Two of those differing must give two different keys, or a blob is
    /// served for a row that never wrote it.
    #[test]
    fn payload_key_is_injective_across_generations() {
        let g1 = IndexGeneration::new(0x1111_2222_3333_4444_5555_6666_7777_8888);
        let g2 = IndexGeneration::new(0x1111_2222_3333_4444_5555_6666_7777_8889);
        let keys = [
            ("generation", payload_blob_key(0, g1, "n", 7, 3)),
            ("other generation", payload_blob_key(0, g2, "n", 7, 3)),
            (
                "legacy generation",
                payload_blob_key(0, IndexGeneration::LEGACY, "n", 7, 3),
            ),
            ("row version", payload_blob_key(0, g1, "n", 7, 4)),
            ("id", payload_blob_key(0, g1, "n", 8, 3)),
            ("name", payload_blob_key(0, g1, "nn", 7, 3)),
            // The name is variable-length and everything after it is not, so
            // the two halves of a longer name must not be readable as a
            // shorter name plus a different id.
            ("name prefix", payload_blob_key(0, g1, "n\u{0}", 7, 3)),
            ("tenant", payload_blob_key(1, g1, "n", 7, 3)),
            ("pre-generation key", payload_key(0, "n", 7)),
        ];
        for (i, (what, a)) in keys.iter().enumerate() {
            for (other, b) in keys.iter().skip(i + 1) {
                assert_ne!(a, b, "{what} and {other} share a blob key");
            }
        }
    }

    /// The registry carries the generation from now on, and a file written
    /// before it did reads as the legacy one - not as an error, and not as a
    /// generation some other index could also mint.
    #[test]
    fn registry_v2_reads_as_the_legacy_generation() {
        let dir = TempDir::new().unwrap();
        let g = IndexGeneration::new(0x0102_0304_0506_0708_090a_0b0c_0d0e_0f10);
        write_registry(dir.path(), &[("alpha", 64usize, 2u8, g)]).unwrap();
        let back = read_registry(dir.path()).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(
            back[0].generation, g,
            "the registry must carry the generation it was given"
        );

        // A V2 file, byte for byte as the previous version wrote it.
        let mut buf = b"SVI2".to_vec();
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&(5u16).to_le_bytes());
        buf.extend_from_slice(b"alpha");
        buf.extend_from_slice(&64u32.to_le_bytes());
        buf.push(2);
        fs::write(dir.path().join(VINDEX_REGISTRY), &buf).unwrap();
        let back = read_registry(dir.path()).unwrap();
        assert_eq!(back.len(), 1, "a V2 registry still reads");
        assert_eq!(back[0].name, "alpha");
        assert_eq!(back[0].dim, 64);
        assert_eq!(back[0].kind, Some(2));
        assert_eq!(
            back[0].generation,
            IndexGeneration::LEGACY,
            "an index recorded before generations existed has none"
        );
    }

    /// A store written before the generation key existed keeps its blobs where
    /// it left them. The index reads as the legacy generation, and the legacy
    /// generation reads the legacy key.
    ///
    /// Not a red test: it is the compatibility half of the same commit, and it
    /// has to be green before AND after.
    #[tokio::test]
    async fn legacy_blobs_stay_readable_after_the_generation_key_lands() {
        let dir = TempDir::new().unwrap();
        {
            let shards = ShardSet::open(dir.path(), 1).unwrap();
            shards.vindex_create("lg", 64, 0, 1).await.unwrap();
            shards.vset("lg", 1, tvec(1), 0, None, None).await.unwrap();
        }
        // Downgrade the catalogue to what the previous version wrote: same
        // entry, no generation. The index now predates generations.
        let sdir = dir.path().join("shard-0");
        let entry = read_registry(&sdir).unwrap().remove(0);
        let mut buf = b"SVI2".to_vec();
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&(entry.name.len() as u16).to_le_bytes());
        buf.extend_from_slice(entry.name.as_bytes());
        buf.extend_from_slice(&(entry.dim as u32).to_le_bytes());
        buf.push(entry.kind.unwrap_or(1));
        fs::write(sdir.join(VINDEX_REGISTRY), &buf).unwrap();

        let shards = ShardSet::open(dir.path(), 1).unwrap();
        // The blob exactly where the previous version would have put it.
        shards
            .set(&payload_key(0, "lg", 1), b"colour=red", Durability::Relaxed)
            .await
            .unwrap();
        let hits = shards
            .vsearch("lg", tvec(1), 8, 0, 0, true, None)
            .await
            .unwrap();
        let hit = hits.iter().find(|h| h.0 == 1).expect("the row is there");
        assert_eq!(
            hit.2.as_deref(),
            Some(&b"colour=red"[..]),
            "a legacy index stopped finding the blobs it already had"
        );
    }

    #[test]
    fn the_registry_round_trips_every_entry_it_was_given() {
        let dir = TempDir::new().unwrap();
        let g = IndexGeneration::LEGACY;
        let entries = [
            ("alpha", 64usize, 1u8, g),
            ("beta", 1024, 2, g),
            ("gamma", 8, 4, g),
        ];
        write_registry(dir.path(), &entries).unwrap();
        let back = read_registry(dir.path()).unwrap();
        assert_eq!(back.len(), 3);
        for (i, (name, dim, kind, _)) in entries.iter().enumerate() {
            assert_eq!(&back[i].name, name);
            assert_eq!(back[i].dim, *dim);
            assert_eq!(back[i].kind, Some(*kind));
        }
    }

    #[tokio::test]
    async fn corrupt_framed_vindex_wal_refuses_shard_open() {
        let dir = TempDir::new().unwrap();
        let base = dir.path().to_owned();
        {
            let shards = ShardSet::open(&base, 1).unwrap();
            shards.vindex_create("idx", 4, 0, 1).await.unwrap();
            shards
                .vset("idx", 7, vec![1.0; 4], 0, None, None)
                .await
                .unwrap();
        }
        let wal_path = base.join("shard-0/vindex-idx/delta.log");
        let mut wal = fs::read(&wal_path).unwrap();
        assert!(wal.starts_with(b"SKWL\x03"));
        *wal.last_mut().unwrap() ^= 0x01;
        fs::write(wal_path, wal).unwrap();

        assert!(
            ShardSet::open(&base, 1).is_err(),
            "a corrupt registered VINDEX must refuse startup"
        );
    }

    /// A tenant-scoped key, exactly as `resp3_handler::scope_key` builds it:
    /// the tenant's 16 bytes in little-endian order, then the raw key.
    fn scoped(tenant: u128, key: &str) -> Vec<u8> {
        let mut k = tenant.to_le_bytes().to_vec();
        k.extend_from_slice(key.as_bytes());
        k
    }

    #[tokio::test]
    async fn erase_tenant_removes_only_that_tenants_keys() {
        const VICTIM: u128 = 1;
        const NEIGHBOUR: u128 = 2;
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 4).unwrap();

        // Both tenants' keys hash across all four shards, and they collide on
        // the unscoped part: erasing one must not touch the other.
        for i in 0u32..30 {
            let k = format!("k{i}");
            for t in [VICTIM, NEIGHBOUR] {
                shards
                    .set_scoped(&scoped(t, &k), b"v", Durability::Kernel, t, None)
                    .await
                    .unwrap();
            }
        }

        let (vindexes, keys) = shards
            .erase_tenant(VICTIM, Durability::Kernel)
            .await
            .unwrap();
        assert_eq!(keys, 30, "every key of the victim tenant deleted");
        assert_eq!(vindexes, 0, "no vindexes existed to drop");

        for i in 0u32..30 {
            let k = format!("k{i}");
            assert_eq!(
                shards.get(&scoped(VICTIM, &k)).await.unwrap(),
                None,
                "victim key {k} survived the erasure"
            );
            assert_eq!(
                shards.get(&scoped(NEIGHBOUR, &k)).await.unwrap().as_deref(),
                Some(b"v".as_slice()),
                "neighbour key {k} was collateral damage"
            );
        }

        // Erasure is idempotent: nothing left to find the second time.
        let (_, keys) = shards
            .erase_tenant(VICTIM, Durability::Kernel)
            .await
            .unwrap();
        assert_eq!(keys, 0, "second erase is a no-op");
    }

    #[tokio::test]
    async fn erase_tenant_drops_its_vindexes_and_leaves_no_disk_charged() {
        const VICTIM: u128 = 7;
        const NEIGHBOUR: u128 = 8;
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 2).unwrap();

        // A vector's payload blob is a KV key under the tenant's prefix. If the
        // sweep ran before the vindex drop it would delete the blob and leave
        // the index pointing at a hole, so the order is what this pins down.
        for t in [VICTIM, NEIGHBOUR] {
            let name = scope_key(t, "idx");
            shards.vindex_create_scoped(&name, 4, 0, 0).await.unwrap();
            for id in 0u64..8 {
                shards
                    .vset(
                        &name,
                        id,
                        vec![0.1, 0.2, 0.3, 0.4],
                        t,
                        None,
                        Some(Bytes::from_static(b"payload")),
                    )
                    .await
                    .unwrap();
            }
            // A plain KV key alongside the vector data.
            shards
                .set_scoped(&scoped(t, "doc"), b"v", Durability::Kernel, t, None)
                .await
                .unwrap();
        }

        let (vindexes, _) = shards
            .erase_tenant(VICTIM, Durability::Kernel)
            .await
            .unwrap();
        assert!(vindexes >= 1, "the victim's vindex was dropped");

        // Nothing of the victim's is left: no vindex, no key, no disk charged.
        let names: Vec<String> = shards
            .vindex_list()
            .await
            .unwrap()
            .into_iter()
            .map(|row| row.name)
            .collect();
        assert!(
            !names.contains(&scope_key(VICTIM, "idx")),
            "victim vindex still listed: {names:?}"
        );
        assert_eq!(
            shards.get(&scoped(VICTIM, "doc")).await.unwrap(),
            None,
            "victim KV key survived"
        );
        assert_eq!(
            shards.tenant_disk_bytes(VICTIM),
            0,
            "victim still charged for disk after erasure"
        );

        // The neighbour is untouched: index still listed, key still readable.
        assert!(
            names.contains(&scope_key(NEIGHBOUR, "idx")),
            "neighbour vindex was collateral damage: {names:?}"
        );
        assert_eq!(
            shards
                .get(&scoped(NEIGHBOUR, "doc"))
                .await
                .unwrap()
                .as_deref(),
            Some(b"v".as_slice()),
            "neighbour KV key was collateral damage"
        );
    }

    #[tokio::test]
    async fn erase_prefix_removes_one_subject_and_spares_the_others() {
        const TENANT: u128 = 5;
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 4).unwrap();

        // Two subjects under the same tenant, keys namespaced `subject/i`.
        let key = |subject: &str, i: u32| {
            let mut k = TENANT.to_le_bytes().to_vec();
            k.extend_from_slice(format!("{subject}/{i}").as_bytes());
            k
        };
        for i in 0u32..25 {
            for subj in ["alice", "bob"] {
                shards
                    .set_scoped(&key(subj, i), b"v", Durability::Kernel, TENANT, None)
                    .await
                    .unwrap();
            }
        }

        let erased = shards
            .erase_prefix(TENANT, b"alice/", Durability::Kernel)
            .await
            .unwrap();
        assert_eq!(erased, 25, "every key of the named subject deleted");

        for i in 0u32..25 {
            assert_eq!(
                shards.get(&key("alice", i)).await.unwrap(),
                None,
                "alice key {i} survived"
            );
            assert_eq!(
                shards.get(&key("bob", i)).await.unwrap().as_deref(),
                Some(b"v".as_slice()),
                "bob key {i} was collateral damage"
            );
        }
    }

    #[tokio::test]
    async fn reclaim_makes_erased_values_unrecoverable_on_disk() {
        const TENANT: u128 = 9;
        let needle = b"pii-value-to-erase";
        let base = TempDir::new().unwrap().path().to_owned();
        std::fs::create_dir_all(&base).unwrap();

        {
            let shards = ShardSet::open(&base, 2).unwrap();
            let mut k = TENANT.to_le_bytes().to_vec();
            k.extend_from_slice(b"subject/doc");
            shards
                .set_scoped(&k, needle, Durability::Kernel, TENANT, None)
                .await
                .unwrap();
            // Filler so the secret's segment seals behind a rotation.
            for i in 0u32..50 {
                let mut fk = TENANT.to_le_bytes().to_vec();
                fk.extend_from_slice(format!("filler/{i}").as_bytes());
                shards
                    .set_scoped(&fk, b"x", Durability::Kernel, TENANT, None)
                    .await
                    .unwrap();
            }
            shards
                .erase_prefix(TENANT, b"subject/", Durability::Kernel)
                .await
                .unwrap();
            let freed = shards.reclaim().await.unwrap();
            assert!(freed > 0, "reclaim freed bytes");
            // Dropping ShardSet joins the worker threads and closes the files.
        }

        // Scan every segment file: the erased value must be gone from disk.
        let mut raw = Vec::new();
        for e in walk_seg_files(&base) {
            raw.extend(std::fs::read(&e).unwrap());
        }
        assert!(
            !raw.windows(needle.len()).any(|w| w == needle),
            "erased value must not survive reclaim on disk"
        );
    }

    /// Every `.seg` file under `root`, recursively (shards live in subdirs).
    fn walk_seg_files(root: &std::path::Path) -> Vec<std::path::PathBuf> {
        let mut out = Vec::new();
        let mut stack = vec![root.to_owned()];
        while let Some(d) = stack.pop() {
            let Ok(rd) = std::fs::read_dir(&d) else {
                continue;
            };
            for entry in rd.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().is_some_and(|e| e == "seg") {
                    out.push(p);
                }
            }
        }
        out
    }

    #[tokio::test]
    async fn erase_tenant_refuses_the_anonymous_tenant() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 2).unwrap();
        shards
            .set(b"plain", b"v", Durability::Kernel)
            .await
            .unwrap();

        // Tenant 0's keys are unscoped, so a prefix sweep cannot tell them from
        // anyone else's: erasing it would be a whole-store wipe. It must refuse.
        assert!(
            shards.erase_tenant(0, Durability::Kernel).await.is_err(),
            "erasing tenant 0 must be refused"
        );
        assert_eq!(
            shards.get(b"plain").await.unwrap().as_deref(),
            Some(b"v".as_slice()),
            "the refused erase must not have deleted anything"
        );
    }

    #[tokio::test]
    async fn mset_writes_all_keys_across_shards_and_survives_reopen() {
        let dir = TempDir::new().unwrap();
        let owned: Vec<(Vec<u8>, Vec<u8>)> = (0u32..40)
            .map(|i| (format!("mk{i}").into_bytes(), format!("v{i}").into_bytes()))
            .collect();
        {
            let shards = ShardSet::open(dir.path(), 4).unwrap();
            let pairs: Vec<(&[u8], &[u8])> = owned
                .iter()
                .map(|(k, v)| (k.as_slice(), v.as_slice()))
                .collect();
            shards.mset(&pairs, Durability::Kernel).await.unwrap();
            for (k, v) in &owned {
                assert_eq!(shards.get(k).await.unwrap().as_deref(), Some(v.as_slice()));
            }
            // Dropping ShardSet joins the worker threads (files closed).
        }
        // Each shard's batch is durable: every key survives a reopen.
        let shards = ShardSet::open(dir.path(), 4).unwrap();
        for (k, v) in &owned {
            assert_eq!(
                shards.get(k).await.unwrap().as_deref(),
                Some(v.as_slice()),
                "key survived reopen"
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
        // Two tenants writing through the ShardSet have separate cache
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
    async fn test_vector_quota_enforced_per_tenant() {
        // A tenant at max_vectors is rejected on VSET; another tenant
        // is unaffected; an overwrite never consumes quota; VDEL frees a slot.
        // Indexes are tenant-scoped names, mirroring how the RESP3 handler
        // scopes them, so ids never collide across tenants.
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 2).unwrap();
        shards.vindex_create("idx7", 4, 0, 0).await.unwrap();
        shards.vindex_create("idx9", 4, 0, 0).await.unwrap();
        let v = vec![1.0f32, 0.0, 0.0, 0.0];
        let lim = Some(2u64);

        shards
            .vset("idx7", 1, v.clone(), 7, lim, None)
            .await
            .unwrap();
        shards
            .vset("idx7", 2, v.clone(), 7, lim, None)
            .await
            .unwrap();
        assert_eq!(shards.tenant_vector_count(7), 2);

        // A third new id exceeds the limit -> rejected, count unchanged.
        assert!(
            shards
                .vset("idx7", 3, v.clone(), 7, lim, None)
                .await
                .is_err()
        );
        assert_eq!(
            shards.tenant_vector_count(7),
            2,
            "rejected vset must not count"
        );

        // Overwriting an existing id is always allowed and free, even at the cap.
        shards
            .vset("idx7", 1, vec![0.0, 1.0, 0.0, 0.0], 7, lim, None)
            .await
            .unwrap();
        assert_eq!(shards.tenant_vector_count(7), 2, "overwrite is free");

        // A different tenant has its own independent budget.
        shards
            .vset("idx9", 1, v.clone(), 9, Some(1), None)
            .await
            .unwrap();
        assert_eq!(shards.tenant_vector_count(9), 1);
        assert_eq!(shards.tenant_vector_count(7), 2, "tenant 9 did not touch 7");

        // VDEL frees a slot for tenant 7, letting a new id back in.
        assert!(shards.vdel("idx7", 2, 7).await.unwrap());
        assert_eq!(shards.tenant_vector_count(7), 1);
        shards
            .vset("idx7", 3, v.clone(), 7, lim, None)
            .await
            .unwrap();
        assert_eq!(shards.tenant_vector_count(7), 2);
    }

    // ── The vector quota has to survive a restart ────────────────────────
    //
    // `TenantVectorQuota` is process state: `ShardSet::open` builds it empty
    // and nothing puts back what the shards already hold. So a tenant sitting
    // at its limit gets its entire budget back by restarting the server - the
    // limit is not a limit, it is a limit per uptime - and the same restart
    // can also count a tenant's rows twice, because a routed index keeps a
    // second physical copy of a logical row on the boundary shard.
    //
    // These pin what the count must be immediately after an open, before a
    // single write is admitted. They are written against the SCOPED name the
    // RESP3 layer builds (`scope_key`), because that is the only place a
    // stored index records which tenant reaches it.

    /// Dimension of the quota fixtures. Sixteen so a per-row fingerprint has
    /// room to make every row its own nearest neighbour, which is what lets
    /// the balanced k-means split the two clusters cleanly.
    const QUOTA_DIM: usize = 16;

    /// A row in one of two orthogonal clusters, unique within its cluster.
    /// The dominant component carries the cluster (so a reshard splits them
    /// across two shards); the rest is a discrete +/- 0.5 fingerprint.
    fn quota_vec(id: u64, cluster: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; QUOTA_DIM];
        v[cluster] = 1.0;
        let mut s = (id << 1) | 1;
        for slot in v.iter_mut().skip(2) {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *slot = if s % 2 == 0 { 0.5 } else { -0.5 };
        }
        v
    }

    /// The cluster a row belongs to, so a fixture can spread ids over both.
    fn quota_row(id: u64) -> Vec<f32> {
        quota_vec(id, (id % 2) as usize)
    }

    #[tokio::test]
    async fn test_quota_survives_restart() {
        const T: u128 = 7;
        const LIMIT: u64 = 3;
        let dir = TempDir::new().unwrap();
        let name = scope_key(T, "q");
        {
            let shards = ShardSet::open(dir.path(), 2).unwrap();
            shards
                .vindex_create_scoped(&name, QUOTA_DIM as u32, 1, 1)
                .await
                .unwrap();
            for id in 0..LIMIT {
                shards
                    .vset(&name, id, quota_row(id), T, Some(LIMIT), None)
                    .await
                    .unwrap();
            }
            assert_eq!(shards.tenant_vector_count(T), LIMIT);
        }

        let shards = ShardSet::open(dir.path(), 2).unwrap();
        assert_eq!(
            shards.tenant_vector_count(T),
            LIMIT,
            "the count a restart rebuilt must be what the shards hold"
        );
        assert!(
            shards
                .vset(&name, 99, quota_row(99), T, Some(LIMIT), None)
                .await
                .is_err(),
            "a tenant at its limit must not get its budget back by restarting"
        );
    }

    #[tokio::test]
    async fn test_quota_overwrite_same_id_counts_once() {
        const T: u128 = 11;
        let dir = TempDir::new().unwrap();
        let name = scope_key(T, "ov");
        {
            let shards = ShardSet::open(dir.path(), 2).unwrap();
            shards
                .vindex_create_scoped(&name, QUOTA_DIM as u32, 1, 1)
                .await
                .unwrap();
            for _ in 0..4 {
                shards
                    .vset(&name, 1, quota_row(1), T, Some(2), None)
                    .await
                    .unwrap();
            }
            assert_eq!(shards.tenant_vector_count(T), 1, "an overwrite is free");
        }

        let shards = ShardSet::open(dir.path(), 2).unwrap();
        assert_eq!(
            shards.tenant_vector_count(T),
            1,
            "four writes to one id are one logical row, before and after a restart"
        );
    }

    /// A tenant is charged for its OWN indexes only. The rebuild reads the
    /// scoped registry key, which is what decides who can reach the index in
    /// the first place; an index in the unscoped namespace must not land on
    /// the tenant whose id another name happens to spell.
    #[tokio::test]
    async fn test_quota_rebuild_charges_each_index_to_its_own_tenant() {
        const A: u128 = 0x2a;
        const B: u128 = 0x2b;
        let dir = TempDir::new().unwrap();
        let a = scope_key(A, "own");
        let b = scope_key(B, "own");
        {
            let shards = ShardSet::open(dir.path(), 2).unwrap();
            shards
                .vindex_create_scoped(&a, QUOTA_DIM as u32, 1, 1)
                .await
                .unwrap();
            shards
                .vindex_create_scoped(&b, QUOTA_DIM as u32, 1, 1)
                .await
                .unwrap();
            shards
                .vindex_create("plain", QUOTA_DIM as u32, 1, 1)
                .await
                .unwrap();
            for id in 0..5 {
                shards
                    .vset(&a, id, quota_row(id), A, Some(50), None)
                    .await
                    .unwrap();
            }
            for id in 0..2 {
                shards
                    .vset(&b, id, quota_row(id), B, Some(50), None)
                    .await
                    .unwrap();
            }
            for id in 0..9 {
                shards
                    .vset("plain", id, quota_row(id), 0, Some(50), None)
                    .await
                    .unwrap();
            }
        }

        let shards = ShardSet::open(dir.path(), 2).unwrap();
        assert_eq!(shards.tenant_vector_count(A), 5, "tenant A's own rows");
        assert_eq!(shards.tenant_vector_count(B), 2, "tenant B's own rows");
        assert_eq!(
            shards.tenant_vector_count(0),
            9,
            "the unscoped namespace keeps its own rows and takes nobody else's"
        );
    }

    #[tokio::test]
    async fn test_quota_survives_restart_after_move() {
        const T: u128 = 13;
        const N: u64 = 60;
        let dir = TempDir::new().unwrap();
        let name = scope_key(T, "mv");
        {
            let shards = ShardSet::open(dir.path(), 2).unwrap();
            shards
                .vindex_create_scoped(&name, QUOTA_DIM as u32, 1, 1)
                .await
                .unwrap();
            for id in 0..N {
                shards
                    .vset(&name, id, quota_row(id), T, Some(N), None)
                    .await
                    .unwrap();
            }
            let moved = shards.reshard(&name, 0.25, 10, T).await.expect("reshard");
            assert!(moved > 0, "fixture: the reshard has to move something");
            assert_eq!(
                shards.tenant_vector_count(T),
                N,
                "a move changes no cardinality"
            );
        }

        let shards = ShardSet::open(dir.path(), 2).unwrap();
        assert_eq!(
            shards.tenant_vector_count(T),
            N,
            "a moved row is one row after a restart too"
        );
    }

    #[tokio::test]
    async fn test_quota_survives_restart_with_replica() {
        const T: u128 = 17;
        const N: u64 = 60;
        let dir = TempDir::new().unwrap();
        let name = scope_key(T, "rp");
        {
            let shards = ShardSet::open(dir.path(), 2).unwrap();
            shards
                .vindex_create_scoped(&name, QUOTA_DIM as u32, 1, 1)
                .await
                .unwrap();
            for id in 0..N {
                shards
                    .vset(&name, id, quota_row(id), T, Some(N), None)
                    .await
                    .unwrap();
            }
            shards.reshard(&name, 0.25, 10, T).await.expect("reshard");
            let replicated = shards.overlap(&name, 4.0, T).await.expect("overlap");
            assert!(replicated > 0, "fixture: the overlap has to replicate");
            let physical: u64 = shards
                .vindex_list()
                .await
                .unwrap()
                .iter()
                .find(|r| r.name == name)
                .map(|r| r.n_vectors)
                .unwrap();
            assert!(
                physical > N,
                "fixture: replicas mean more physical rows than logical ones \
                 ({physical} vs {N})"
            );
        }

        let shards = ShardSet::open(dir.path(), 2).unwrap();
        assert_eq!(
            shards.tenant_vector_count(T),
            N,
            "a replica is a second copy of one logical row, not a second row"
        );
    }

    /// A crash between a move's write to the destination and the delete of
    /// its source leaves two physical copies of one logical row. The reopen
    /// has to reconcile them to ONE, the same way the owner map does.
    #[tokio::test]
    async fn test_quota_crash_between_move_vset_and_vdel_reconciles_at_reopen() {
        const T: u128 = 19;
        const N: u64 = 60;
        let dir = TempDir::new().unwrap();
        let name = scope_key(T, "hm");
        {
            let shards = ShardSet::open(dir.path(), 2).unwrap();
            shards
                .vindex_create_scoped(&name, QUOTA_DIM as u32, 1, 1)
                .await
                .unwrap();
            for id in 0..N {
                shards
                    .vset(&name, id, quota_row(id), T, Some(N), None)
                    .await
                    .unwrap();
            }
            crate::failpoint::arm(crate::failpoint::WriteFailpoint::ReshardSourceDelete);
            let outcome = shards.reshard(&name, 0.25, 10, T).await;
            crate::failpoint::disarm(crate::failpoint::WriteFailpoint::ReshardSourceDelete);
            assert!(
                crate::failpoint::fired(crate::failpoint::WriteFailpoint::ReshardSourceDelete),
                "the failpoint never fired, so this test proved nothing"
            );
            assert!(outcome.is_err(), "the source delete was refused");
            let physical: u64 = shards
                .vindex_list()
                .await
                .unwrap()
                .iter()
                .find(|r| r.name == name)
                .map(|r| r.n_vectors)
                .unwrap();
            assert!(
                physical > N,
                "fixture: the half-done move has to leave a duplicate \
                 ({physical} vs {N})"
            );
        }

        let shards = ShardSet::open(dir.path(), 2).unwrap();
        assert_eq!(
            shards.tenant_vector_count(T),
            N,
            "a duplicate left by a half-done move is one logical row"
        );
    }

    #[tokio::test]
    async fn test_quota_drop_index_credits_then_restart() {
        const T: u128 = 23;
        let dir = TempDir::new().unwrap();
        let keep = scope_key(T, "keep");
        let go = scope_key(T, "go");
        {
            let shards = ShardSet::open(dir.path(), 2).unwrap();
            for n in [&keep, &go] {
                shards
                    .vindex_create_scoped(n, QUOTA_DIM as u32, 1, 1)
                    .await
                    .unwrap();
            }
            for id in 0..3 {
                shards
                    .vset(&keep, id, quota_row(id), T, Some(50), None)
                    .await
                    .unwrap();
            }
            for id in 0..2 {
                shards
                    .vset(&go, id, quota_row(id), T, Some(50), None)
                    .await
                    .unwrap();
            }
            assert_eq!(shards.tenant_vector_count(T), 5);
            shards.vindex_drop(&go, T).await.unwrap();
            assert_eq!(shards.tenant_vector_count(T), 3, "the drop credits");
        }

        let shards = ShardSet::open(dir.path(), 2).unwrap();
        assert_eq!(
            shards.tenant_vector_count(T),
            3,
            "a dropped index is not counted again at the next open"
        );
    }

    /// A vindex directory the catalogue no longer names is an orphan: a drop
    /// whose registry commit landed and whose directory removal did not.
    /// Recovery does not serve it, so the rebuild must not count it either -
    /// otherwise a tenant pays for rows nothing can read.
    #[tokio::test]
    async fn test_quota_orphan_records_not_double_counted() {
        const T: u128 = 29;
        let dir = TempDir::new().unwrap();
        let live = scope_key(T, "live");
        let orphan = scope_key(T, "orph");
        {
            let shards = ShardSet::open(dir.path(), 2).unwrap();
            for n in [&live, &orphan] {
                shards
                    .vindex_create_scoped(n, QUOTA_DIM as u32, 1, 1)
                    .await
                    .unwrap();
            }
            for id in 0..4 {
                shards
                    .vset(&live, id, quota_row(id), T, Some(50), None)
                    .await
                    .unwrap();
            }
            for id in 0..6 {
                shards
                    .vset(&orphan, id, quota_row(id), T, Some(50), None)
                    .await
                    .unwrap();
            }
            shards.vindex_consolidate(&orphan).await.unwrap();
        }
        // The state a drop leaves when the registry commit lands and the
        // directory removal does not: entry gone, files still there.
        for shard in 0..2 {
            let sdir = dir.path().join(format!("shard-{shard}"));
            persist_registry_removing(&sdir, &RwLock::new(VindexSet::new()), Some(&orphan))
                .unwrap();
        }

        let shards = ShardSet::open(dir.path(), 2).unwrap();
        assert_eq!(
            shards.tenant_vector_count(T),
            4,
            "an orphan directory the catalogue does not name is not the tenant's"
        );
    }

    /// A DROP credited what each shard PHYSICALLY held, and a routed index
    /// keeps a second physical copy of every boundary row - so dropping one
    /// gave back more slots than the tenant ever spent, `sub` saturated at
    /// zero, and the tenant could then write a WHOLE `max_vectors` on top of
    /// what it already held, until the next restart.
    ///
    /// Measured on this fixture before the fix: 70 logical rows counted, 59
    /// replicas, 0 counted after the drop against 10 rows still on disk, then
    /// 100 more writes accepted for 110 held under a limit of 100. The whole
    /// sequence is client-reachable: create, reshard, overlap, drop.
    ///
    /// Asserted BEFORE any reopen: the reopen repairs it, and a repair that
    /// arrives at the next restart is not a limit.
    #[tokio::test]
    async fn test_quota_drop_of_a_replicated_index_credits_only_what_it_charged() {
        const T: u128 = 47;
        const LIMIT: u64 = 100;
        const KEPT: u64 = 10;
        const N: u64 = 60;
        let dir = TempDir::new().unwrap();
        let keep = scope_key(T, "keep");
        let rep = scope_key(T, "rep");
        let shards = ShardSet::open(dir.path(), 2).unwrap();
        for n in [&keep, &rep] {
            shards
                .vindex_create_scoped(n, QUOTA_DIM as u32, 1, 1)
                .await
                .unwrap();
        }
        for id in 0..KEPT {
            shards
                .vset(&keep, id, quota_row(id), T, Some(LIMIT), None)
                .await
                .unwrap();
        }
        for id in 0..N {
            shards
                .vset(&rep, id, quota_row(id), T, Some(LIMIT), None)
                .await
                .unwrap();
        }
        shards.reshard(&rep, 0.25, 10, T).await.expect("reshard");
        let replicated = shards.overlap(&rep, 4.0, T).await.expect("overlap");
        assert!(replicated > 0, "fixture: the overlap has to replicate");
        assert_eq!(shards.tenant_vector_count(T), KEPT + N);

        shards.vindex_drop(&rep, T).await.unwrap();
        assert_eq!(
            shards.tenant_vector_count(T),
            KEPT,
            "the drop must give back the rows the tenant had, not the copies \
             the index kept of them"
        );

        // And the limit still binds, without a restart being what makes it.
        for id in KEPT..LIMIT {
            shards
                .vset(&keep, id, quota_row(id), T, Some(LIMIT), None)
                .await
                .unwrap_or_else(|e| panic!("row {id} of {LIMIT} refused: {e}"));
        }
        assert!(
            shards
                .vset(&keep, LIMIT, quota_row(LIMIT), T, Some(LIMIT), None)
                .await
                .is_err(),
            "a tenant must not be able to write past its limit by dropping a \
             replicated index"
        );
        assert_eq!(shards.tenant_vector_count(T), LIMIT);
    }

    /// Erasing a tenant removes every index it has, so the one number that is
    /// certainly right afterwards is zero. The physical fragments a routed
    /// index leaves behind cannot add up to it, and this is the GDPR path -
    /// the count must not be a leftover of how the rows happened to be laid
    /// out.
    ///
    /// A CONTROL, green before the fix as well as after, and green for the
    /// wrong reason today: the same over-credit the test above is about
    /// saturates at zero, which happens to be the right answer here. What it
    /// guards is the fix - stop the shards crediting a routed index and the
    /// erase leaves the tenant counted at 65 unless the coordinator zeroes
    /// it, which is the opposite error and just as wrong.
    #[tokio::test]
    async fn test_quota_erase_tenant_zeroes_only_that_tenant() {
        const A: u128 = 0x53;
        const B: u128 = 0x59;
        const N: u64 = 60;
        let dir = TempDir::new().unwrap();
        let a_rep = scope_key(A, "rep");
        let a_plain = scope_key(A, "plain");
        let b_own = scope_key(B, "own");
        let shards = ShardSet::open(dir.path(), 2).unwrap();
        for n in [&a_rep, &a_plain, &b_own] {
            shards
                .vindex_create_scoped(n, QUOTA_DIM as u32, 1, 1)
                .await
                .unwrap();
        }
        for id in 0..N {
            shards
                .vset(&a_rep, id, quota_row(id), A, Some(500), None)
                .await
                .unwrap();
        }
        for id in 0..5 {
            shards
                .vset(&a_plain, id, quota_row(id), A, Some(500), None)
                .await
                .unwrap();
        }
        for id in 0..7 {
            shards
                .vset(&b_own, id, quota_row(id), B, Some(500), None)
                .await
                .unwrap();
        }
        shards.reshard(&a_rep, 0.25, 10, A).await.expect("reshard");
        let replicated = shards.overlap(&a_rep, 4.0, A).await.expect("overlap");
        assert!(replicated > 0, "fixture: the overlap has to replicate");
        assert_eq!(shards.tenant_vector_count(A), N + 5);

        shards.erase_tenant(A, Durability::Kernel).await.unwrap();
        assert_eq!(
            shards.tenant_vector_count(A),
            0,
            "an erased tenant holds nothing"
        );
        assert_eq!(
            shards.tenant_vector_count(B),
            7,
            "and nobody else was touched"
        );
    }

    /// The reopen guard for the drop above: whatever the drop credited, the
    /// next open counts what is on disk. Kept alongside
    /// `test_quota_drop_of_a_replicated_index_credits_only_what_it_charged`,
    /// which pins the credit itself - one of them would go green if the
    /// rebuild were removed and the other if the credit were, so both are
    /// needed to say the pair is right.
    #[tokio::test]
    async fn test_quota_drop_of_a_replicated_index_is_repaired_by_the_next_open() {
        const T: u128 = 37;
        const N: u64 = 60;
        let dir = TempDir::new().unwrap();
        let keep = scope_key(T, "keep");
        let rep = scope_key(T, "rep");
        {
            let shards = ShardSet::open(dir.path(), 2).unwrap();
            for n in [&keep, &rep] {
                shards
                    .vindex_create_scoped(n, QUOTA_DIM as u32, 1, 1)
                    .await
                    .unwrap();
            }
            for id in 0..10 {
                shards
                    .vset(&keep, id, quota_row(id), T, Some(500), None)
                    .await
                    .unwrap();
            }
            for id in 0..N {
                shards
                    .vset(&rep, id, quota_row(id), T, Some(500), None)
                    .await
                    .unwrap();
            }
            shards.reshard(&rep, 0.25, 10, T).await.expect("reshard");
            let replicated = shards.overlap(&rep, 4.0, T).await.expect("overlap");
            assert!(replicated > 0, "fixture: the overlap has to replicate");
            shards.vindex_drop(&rep, T).await.unwrap();
        }

        let shards = ShardSet::open(dir.path(), 2).unwrap();
        assert_eq!(
            shards.tenant_vector_count(T),
            10,
            "the open counts what is on disk, whatever the drop credited"
        );
    }

    /// A committed vindex that will not open is dropped as an ORPHAN: the
    /// registry entry goes, the files stay, and the quota is deliberately
    /// left alone because only the open index knew how many rows it held.
    /// The source says the count "stays high until it is rebuilt from the
    /// data"; this is that rebuild.
    #[tokio::test]
    async fn test_quota_orphan_drop_leaves_the_count_high_until_the_next_open() {
        use std::os::unix::fs::PermissionsExt;
        const T: u128 = 41;
        let dir = TempDir::new().unwrap();
        let keep = scope_key(T, "keep");
        let gone = scope_key(T, "gone");
        let blocked: Vec<std::path::PathBuf> = (0..2)
            .map(|s| {
                dir.path()
                    .join(format!("shard-{s}"))
                    .join(format!("vindex-{gone}"))
            })
            .collect();
        {
            let shards = ShardSet::open(dir.path(), 2).unwrap();
            for n in [&keep, &gone] {
                shards
                    .vindex_create_scoped(n, QUOTA_DIM as u32, 1, 1)
                    .await
                    .unwrap();
            }
            for id in 0..4 {
                shards
                    .vset(&keep, id, quota_row(id), T, Some(500), None)
                    .await
                    .unwrap();
            }
            for id in 0..6 {
                shards
                    .vset(&gone, id, quota_row(id), T, Some(500), None)
                    .await
                    .unwrap();
            }
            assert_eq!(shards.tenant_vector_count(T), 10);
            // Out of the resident map, then out of reach: the drop has to
            // decide from the catalogue and find an index it cannot reopen.
            for shard in 0..2 {
                shards
                    .call(shard, ShardReq::Evict { name: gone.clone() })
                    .await
                    .unwrap();
            }
            let saved: Vec<_> = blocked
                .iter()
                .map(|p| std::fs::metadata(p).unwrap().permissions())
                .collect();
            for p in &blocked {
                std::fs::set_permissions(p, PermissionsExt::from_mode(0o000)).unwrap();
            }
            shards.vindex_drop(&gone, T).await.unwrap();
            assert_eq!(
                shards.tenant_vector_count(T),
                10,
                "an orphan drop cannot credit what it never opened"
            );
            for (p, s) in blocked.iter().zip(saved) {
                std::fs::set_permissions(p, s).unwrap();
            }
        }

        let shards = ShardSet::open(dir.path(), 2).unwrap();
        assert_eq!(
            shards.tenant_vector_count(T),
            4,
            "and the next open is where the tenant gets those slots back"
        );
    }

    /// A write that failed BETWEEN staging its blob and committing its vector
    /// reserved a quota slot and gave it back. The rebuild must agree: the row
    /// is not there, so the slot is free - both in this process and in the one
    /// that opens the store next.
    #[tokio::test]
    async fn test_quota_retry_after_failed_insert_does_not_double_reserve() {
        const T: u128 = 31;
        const LIMIT: u64 = 3;
        let dir = TempDir::new().unwrap();
        let name = scope_key(T, "rt");
        {
            let shards = ShardSet::open(dir.path(), 2).unwrap();
            shards
                .vindex_create_scoped(&name, QUOTA_DIM as u32, 1, 1)
                .await
                .unwrap();
            for id in 0..2 {
                shards
                    .vset(&name, id, quota_row(id), T, Some(LIMIT), None)
                    .await
                    .unwrap();
            }
            // Fail before anything is staged.
            crate::failpoint::arm_at(crate::failpoint::WriteFailpoint::PayloadPrepare, &name);
            let refused = shards
                .vset(
                    &name,
                    2,
                    quota_row(2),
                    T,
                    Some(LIMIT),
                    Some(Bytes::from_static(b"{\"a\":1}")),
                )
                .await;
            crate::failpoint::disarm_at(crate::failpoint::WriteFailpoint::PayloadPrepare, &name);
            assert!(refused.is_err(), "the staging was refused");
            assert!(
                crate::failpoint::fired_at(crate::failpoint::WriteFailpoint::PayloadPrepare, &name),
                "the failpoint never fired, so this test proved nothing"
            );
            // And fail again in the window BETWEEN the staged blob and the
            // commit that would have published the row.
            crate::failpoint::arm_at(crate::failpoint::WriteFailpoint::VectorCommit, &name);
            let refused = shards
                .vset(
                    &name,
                    2,
                    quota_row(2),
                    T,
                    Some(LIMIT),
                    Some(Bytes::from_static(b"{\"a\":1}")),
                )
                .await;
            crate::failpoint::disarm_at(crate::failpoint::WriteFailpoint::VectorCommit, &name);
            assert!(refused.is_err(), "the commit was refused");
            assert!(
                crate::failpoint::fired_at(crate::failpoint::WriteFailpoint::VectorCommit, &name),
                "the failpoint never fired, so this test proved nothing"
            );
            assert_eq!(
                shards.tenant_vector_count(T),
                2,
                "a refused write leaves the count where it was"
            );
        }

        let shards = ShardSet::open(dir.path(), 2).unwrap();
        assert_eq!(
            shards.tenant_vector_count(T),
            2,
            "the rebuild counts rows, not attempts"
        );
        shards
            .vset(&name, 2, quota_row(2), T, Some(LIMIT), None)
            .await
            .expect("the slot the failed write gave back is still free after the restart");
        assert_eq!(shards.tenant_vector_count(T), 3);
        assert!(
            shards
                .vset(&name, 3, quota_row(3), T, Some(LIMIT), None)
                .await
                .is_err(),
            "and the limit still binds"
        );
    }

    /// A rebuilt count has to be a LIVE counter, not a number the open left
    /// behind. The whole point of the rebuild is that a client which retries
    /// after a restart gets the same answer it got before it - and that the
    /// answer still changes when the tenant's contents do.
    #[tokio::test]
    async fn test_quota_rebuilt_count_still_moves_with_writes_and_deletes() {
        const T: u128 = 43;
        const LIMIT: u64 = 3;
        let dir = TempDir::new().unwrap();
        let name = scope_key(T, "lv");
        {
            let shards = ShardSet::open(dir.path(), 2).unwrap();
            shards
                .vindex_create_scoped(&name, QUOTA_DIM as u32, 1, 1)
                .await
                .unwrap();
            for id in 0..LIMIT {
                shards
                    .vset(&name, id, quota_row(id), T, Some(LIMIT), None)
                    .await
                    .unwrap();
            }
            assert!(
                shards
                    .vset(&name, 9, quota_row(9), T, Some(LIMIT), None)
                    .await
                    .is_err(),
                "at the limit before the restart"
            );
        }

        let shards = ShardSet::open(dir.path(), 2).unwrap();
        // The retry a refused client makes: same request, same answer.
        assert!(
            shards
                .vset(&name, 9, quota_row(9), T, Some(LIMIT), None)
                .await
                .is_err(),
            "the retry after the restart gets the answer the first attempt got"
        );
        // An overwrite is still free, and on the hash-placed path that
        // decision is the receiving shard's own.
        shards
            .vset(&name, 0, quota_row(100), T, Some(LIMIT), None)
            .await
            .expect("an overwrite at the limit is free");
        assert_eq!(shards.tenant_vector_count(T), LIMIT);
        // And the rebuilt number is a counter, not a floor: a delete frees
        // exactly one slot and the next write takes exactly that one.
        assert!(shards.vdel(&name, 1, T).await.unwrap());
        assert_eq!(shards.tenant_vector_count(T), LIMIT - 1);
        shards
            .vset(&name, 9, quota_row(9), T, Some(LIMIT), None)
            .await
            .expect("the freed slot is usable");
        assert!(
            shards
                .vset(&name, 10, quota_row(10), T, Some(LIMIT), None)
                .await
                .is_err(),
            "and only that one"
        );
        drop(shards);

        let shards = ShardSet::open(dir.path(), 2).unwrap();
        assert_eq!(
            shards.tenant_vector_count(T),
            LIMIT,
            "the second open counts what the first one's writes and delete left"
        );
    }

    /// A corrupt semantic-router sidecar used to be a `warn!` and a skip, so
    /// the index came back UNROUTED: point ops fell through to hash placement
    /// on a store whose rows had been physically re-partitioned, and the quota
    /// rebuild counted each shard's rows instead of deduplicating them - 119
    /// against 60 logical rows on this fixture, after which the tenant is
    /// refused at half its limit.
    ///
    /// Nothing about that state is serviceable, and the only signal was one
    /// warning line. Same stance as the registry that will not round-trip: an
    /// open that cannot know how the store is routed refuses, and names the
    /// file. A sidecar that is ABSENT is a different thing entirely - an index
    /// that was never resharded has none - and still opens.
    #[tokio::test]
    async fn test_a_corrupt_router_sidecar_fails_the_open() {
        const T: u128 = 61;
        const N: u64 = 60;
        let dir = TempDir::new().unwrap();
        let name = scope_key(T, "cr");
        {
            let shards = ShardSet::open(dir.path(), 2).unwrap();
            shards
                .vindex_create_scoped(&name, QUOTA_DIM as u32, 1, 1)
                .await
                .unwrap();
            for id in 0..N {
                shards
                    .vset(&name, id, quota_row(id), T, Some(500), None)
                    .await
                    .unwrap();
            }
            shards.reshard(&name, 0.25, 10, T).await.expect("reshard");
            let replicated = shards.overlap(&name, 4.0, T).await.expect("overlap");
            assert!(replicated > 0, "fixture: the overlap has to replicate");
        }
        // A clean reopen agrees with the writes.
        {
            let shards = ShardSet::open(dir.path(), 2).unwrap();
            assert_eq!(shards.tenant_vector_count(T), N);
        }

        let sidecar = crate::router::router_path(dir.path(), &name);
        assert!(sidecar.exists(), "fixture: the reshard wrote a sidecar");
        std::fs::write(&sidecar, b"not a router").unwrap();
        let err = ShardSet::open(dir.path(), 2)
            .err()
            .expect("an open that cannot know how the store is routed must refuse");
        let msg = err.to_string();
        assert!(
            msg.contains("router-") && msg.contains(&name),
            "the error must name the file: {msg}"
        );
    }

    /// The control for the refusal above: an index with NO sidecar is the
    /// normal state of one that was never resharded, and it opens.
    #[tokio::test]
    async fn test_a_vindex_without_a_router_sidecar_opens() {
        const T: u128 = 67;
        let dir = TempDir::new().unwrap();
        let name = scope_key(T, "ns");
        {
            let shards = ShardSet::open(dir.path(), 2).unwrap();
            shards
                .vindex_create_scoped(&name, QUOTA_DIM as u32, 1, 1)
                .await
                .unwrap();
            for id in 0..5 {
                shards
                    .vset(&name, id, quota_row(id), T, Some(50), None)
                    .await
                    .unwrap();
            }
        }
        assert!(
            !crate::router::router_path(dir.path(), &name).exists(),
            "fixture: nothing resharded it, so there is no sidecar"
        );
        let shards = ShardSet::open(dir.path(), 2).unwrap();
        assert_eq!(shards.tenant_vector_count(T), 5);
    }

    #[test]
    fn test_scope_key_roundtrip() {
        // Tenant 0 is the unscoped namespace (byte-identical to pre-tenancy).
        assert_eq!(scope_key(0, "idx"), "idx");
        assert_eq!(unscope_key("idx"), (0, "idx".to_owned()));
        // A non-zero tenant round-trips through the `<32 hex>::<index>` form.
        for t in [1u128, 42, 0x0123_4567_89ab_cdef, u128::MAX] {
            let key = scope_key(t, "myindex");
            assert!(key.contains("::"));
            assert_eq!(unscope_key(&key), (t, "myindex".to_owned()));
        }
    }

    #[tokio::test]
    async fn test_evict_then_lazy_reopen() {
        // Evict is non-destructive: it frees RAM but leaves the files, so the
        // next access lazily reopens the index with its data intact.
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 2).unwrap();
        // Disk-backed (backend byte 1) so it can be evicted and reopened.
        shards.vindex_create("ev", 4, 1, 1).await.unwrap();
        for id in 1u64..=20 {
            let v = vec![id as f32, 0.0, 0.0, 0.0];
            shards.vset("ev", id, v, 0, None, None).await.unwrap();
        }
        // Fold the delta into the on-disk graph so the data is durable across
        // the evict (not just in the streaming WAL).
        shards.vindex_consolidate("ev").await.unwrap();

        let q = vec![20.0f32, 0.0, 0.0, 0.0];
        let ids = |r: &[(u64, f32, Option<Bytes>)]| r.iter().map(|h| h.0).collect::<Vec<_>>();
        let before = ids(&shards
            .vsearch("ev", q.clone(), 5, 0, 0, false, None)
            .await
            .unwrap());
        assert!(!before.is_empty());

        let ctl = shards.control_handle();
        assert!(
            ctl.open_indices()
                .await
                .iter()
                .any(|s| s.index == "ev" && s.evictable),
            "index is resident and evictable before eviction"
        );

        // Evict frees it on every shard that held it.
        assert!(ctl.evict(0, "ev").await.unwrap());
        assert!(
            ctl.open_indices().await.iter().all(|s| s.index != "ev"),
            "evicted index must not be resident"
        );
        // Evicting again is a no-op (already gone everywhere).
        assert!(!ctl.evict(0, "ev").await.unwrap());

        // Next search lazily reopens it; results match the pre-evict ones.
        let after = ids(&shards.vsearch("ev", q, 5, 0, 0, false, None).await.unwrap());
        assert_eq!(before, after, "lazy reopen returns the same results");
        assert!(
            ctl.open_indices().await.iter().any(|s| s.index == "ev"),
            "reopened index is resident again"
        );
    }

    /// A source whose headroom the test dictates.
    #[derive(Debug)]
    struct FixedHeadroom(u64);

    impl crate::memory::MemorySource for FixedHeadroom {
        fn headroom(&self) -> crate::memory::Headroom {
            crate::memory::Headroom::Known(self.0)
        }
    }

    fn shards_with_budget(dir: &std::path::Path, n: usize, headroom: u64) -> ShardSet {
        // Reserve 0: the test is about the budget being reached, not about the
        // margin held back, and a default reserve larger than the budget would
        // refuse the very first write for a different reason.
        let gov =
            crate::memory::MemoryGovernor::new(Arc::new(FixedHeadroom(headroom)), None, Some(0))
                .expect("a governor over a fixed headroom");
        ShardSet::open_full_with_memory(
            dir,
            n,
            false,
            QuantKind::TurboQuant { bits: 2 },
            0,
            false,
            false,
            Arc::new(gov),
        )
        .expect("a store")
    }

    /// The governor is built, tested, and asked by nobody: the write path never
    /// consults it. Measured in a 256 MiB container - 90,200 rows acknowledged,
    /// then OOMKilled, exit 137, not one write refused.
    ///
    /// A store whose whole budget is smaller than the vectors it is being sent
    /// must refuse, and say so, rather than accept until the kernel intervenes.
    #[tokio::test]
    async fn a_write_is_refused_once_the_memory_budget_is_gone() {
        let dir = TempDir::new().unwrap();
        // 1 MiB of headroom against 1 KiB rows: the budget is reached in about
        // a thousand rows, long before any test timeout.
        let shards = shards_with_budget(dir.path(), 2, 1024 * 1024);
        shards.vindex_create("m", 256, 4, 1).await.unwrap();

        let mut accepted = 0u64;
        let mut refusal = None;
        for id in 0..20_000u64 {
            match shards.vset("m", id, vec![0.5f32; 256], 0, None, None).await {
                Ok(()) => accepted += 1,
                Err(e) => {
                    refusal = Some(e);
                    break;
                }
            }
        }
        let err = refusal.unwrap_or_else(|| {
            panic!("{accepted} rows went in under a 1 MiB budget and none was refused")
        });
        let msg = format!("{err}");
        assert!(
            msg.contains("memory"),
            "the refusal must name the budget, got: {msg}"
        );
        assert!(
            accepted > 0,
            "the budget refused the very first row: the test measures nothing"
        );
    }

    /// A flat index is the case admission most needs to cover, not the one it
    /// can skip: it holds every vector in RAM and nothing ever flushes it, so
    /// it only grows. Reporting it as costing nothing exempted the one backend
    /// that can never give anything back.
    #[tokio::test]
    async fn a_flat_index_is_admitted_against_the_budget_too() {
        let dir = TempDir::new().unwrap();
        let shards = shards_with_budget(dir.path(), 2, 8 * 1024 * 1024);
        // backend 0 = flat, entirely resident.
        shards.vindex_create("f", 256, 0, 0).await.unwrap();

        let mut accepted = 0u64;
        let mut refusal = None;
        for id in 0..20_000u64 {
            match shards.vset("f", id, vec![0.5f32; 256], 0, None, None).await {
                Ok(()) => accepted += 1,
                Err(e) => {
                    refusal = Some(e);
                    break;
                }
            }
        }
        assert!(
            refusal.is_some(),
            "{accepted} rows went into a flat index under the budget and none \
             was refused"
        );
        // Not just "refused": refused after holding a useful amount. The first
        // version of this test asserted only that something was refused, and
        // passed while admitting ONE row per shard - the flat index reported
        // no cost, so every insert asked for a whole chunk.
        assert!(
            accepted > 1_000,
            "only {accepted} rows fit in an 8 MiB budget: admission is not \
             tracking what the index actually holds"
        );
    }

    /// The registry is the catalogue; the resident map is a cache. An evicted
    /// disk index is committed and still on disk, so a DROP must delete it -
    /// not report it absent because it happens not to be in RAM right now.
    #[tokio::test]
    async fn dropping_an_evicted_vindex_deletes_it_instead_of_reporting_it_absent() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 2).unwrap();
        // Disk-backed (backend byte 1): only these survive an evict.
        shards.vindex_create("ev", 4, 1, 1).await.unwrap();
        for id in 1u64..=20 {
            let v = vec![id as f32, 0.0, 0.0, 0.0];
            shards
                .vset(
                    "ev",
                    id,
                    v,
                    0,
                    None,
                    // Payloads, because reclaiming their blobs is work the drop
                    // can only do from the OPEN index - `live_ids` comes off the
                    // backend. A drop that reopens must reclaim them exactly
                    // like a drop that never evicted, and a test that checked
                    // only the directory would not notice if it did not.
                    Some(Bytes::from_static(b"payload")),
                )
                .await
                .unwrap();
        }
        // Fold, so the data is in the graph rather than only in the WAL.
        shards.vindex_consolidate("ev").await.unwrap();

        // Counted rather than probed by key: the key now carries the index's
        // generation and the row's version, so a test that rebuilt it would be
        // checking the implementation against itself - and would go quietly
        // vacuous the moment either changed.
        let held = shards.payload_blobs_held(0, "ev").await.unwrap();
        assert!(
            held > 0,
            "fixture: no payload blob is stored, so the reclamation \
             assertion below would be vacuous"
        );

        let ctl = shards.control_handle();
        assert!(
            ctl.evict(0, "ev").await.unwrap(),
            "fixture: the index must be resident so the evict has something to do"
        );
        let sdirs: Vec<_> = (0..2)
            .map(|i| dir.path().join(format!("shard-{i}")))
            .collect();
        assert!(
            sdirs
                .iter()
                .all(|d| read_registry(d).unwrap().iter().any(|e| e.name == "ev")),
            "fixture: the registry must still name the evicted index, or this \
             test is not about eviction at all"
        );

        shards.vindex_drop("ev", 0).await.unwrap();

        for (i, sdir) in sdirs.iter().enumerate() {
            assert!(
                !read_registry(sdir).unwrap().iter().any(|e| e.name == "ev"),
                "shard {i}: the catalogue still names a dropped vindex, so it \
                 comes back on the next open"
            );
            assert!(
                !sdir.join("vindex-ev").exists(),
                "shard {i}: the directory of a dropped vindex is still on disk"
            );
        }
        assert_eq!(
            shards.payload_blobs_held(0, "ev").await.unwrap(),
            0,
            "payload blobs survived the drop of an evicted index"
        );
    }

    /// LIST reads the resident map, so an index that is committed but evicted
    /// vanishes from it: an operator sees a store smaller than it is, and the
    /// existing erasure test could assert "not listed" about an index the
    /// erasure had silently skipped.
    #[tokio::test]
    async fn listing_shows_a_committed_vindex_that_is_not_resident() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 2).unwrap();
        shards.vindex_create("cold", 4, 1, 1).await.unwrap();
        for id in 1u64..=20 {
            shards
                .vset("cold", id, vec![id as f32, 0.0, 0.0, 0.0], 0, None, None)
                .await
                .unwrap();
        }
        shards.vindex_consolidate("cold").await.unwrap();

        let listed = |rows: Vec<VindexRow>| rows.into_iter().find(|r| r.name == "cold");
        let hot = listed(shards.vindex_list().await.unwrap())
            .expect("fixture: resident before the evict");
        assert_eq!(
            hot.shards_resident, 2,
            "fixture: resident on both shards before the evict"
        );

        let ctl = shards.control_handle();
        assert!(ctl.evict(0, "cold").await.unwrap(), "fixture: was resident");

        let cold = listed(shards.vindex_list().await.unwrap())
            .expect("a committed index must be listed even when nothing is resident");
        assert_eq!(
            (cold.shards_resident, cold.shards_total),
            (0, 2),
            "the row must say the numbers below it were not read from anywhere"
        );
        assert_eq!(cold.dim, 4, "dim is knowable from the catalogue alone");
    }

    /// CHECK answered `Problems(vec![])` - "nothing wrong" - for any index it
    /// did not find in the resident map. An fsck that reports clean about a
    /// thing it never opened is the false green the command exists to prevent.
    #[tokio::test]
    async fn check_does_not_report_clean_on_an_index_it_never_opened() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 2).unwrap();
        shards.vindex_create("cold", 4, 1, 1).await.unwrap();
        for id in 1u64..=20 {
            shards
                .vset("cold", id, vec![id as f32, 0.0, 0.0, 0.0], 0, None, None)
                .await
                .unwrap();
        }
        shards.vindex_consolidate("cold").await.unwrap();
        assert!(
            shards.check("cold").await.unwrap().is_empty(),
            "fixture: clean while resident"
        );

        let ctl = shards.control_handle();
        assert!(ctl.evict(0, "cold").await.unwrap(), "fixture: was resident");

        let lines = shards.check("cold").await.unwrap();
        assert!(
            lines.iter().any(|l| l.contains("not resident")),
            "check reported clean on an index it never opened: {lines:?}"
        );
    }

    /// `resident=k/n` is only worth printing if `n` is the store's shard count.
    /// Counting the shards that ANSWERED makes an index open on three shards of
    /// four read as 3/3 - complete - which is the exact failure the field was
    /// added to prevent, one level up.
    ///
    /// The state is built by evicting ONE shard, which is the ordinary way an
    /// index ends up unevenly resident. It used to be built from a create that
    /// failed on one shard; that is no longer constructible, because such a
    /// create is now undone.
    #[tokio::test]
    async fn an_unevenly_resident_index_does_not_read_as_complete() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 4).unwrap();
        shards.vindex_create("part", 4, 1, 1).await.unwrap();
        for id in 0u64..16 {
            shards
                .vset("part", id, vec![id as f32, 0.0, 0.0, 0.0], 0, None, None)
                .await
                .unwrap();
        }
        shards.vindex_consolidate("part").await.unwrap();

        let row_now = |rows: Vec<VindexRow>| {
            rows.into_iter()
                .find(|r| r.name == "part")
                .expect("the index is listed")
        };
        let full = row_now(shards.vindex_list().await.unwrap());
        assert_eq!(
            (full.shards_resident, full.shards_total),
            (4, 4),
            "fixture: resident everywhere before the evict"
        );

        assert!(
            matches!(
                shards
                    .call(
                        2,
                        ShardReq::Evict {
                            name: "part".into()
                        }
                    )
                    .await
                    .unwrap(),
                ShardResp::Evicted(true)
            ),
            "fixture: shard 2 held it and released it"
        );

        let uneven = row_now(shards.vindex_list().await.unwrap());
        assert_eq!(
            (uneven.shards_resident, uneven.shards_total),
            (3, 4),
            "an index open on three shards of four must not read as complete"
        );
    }

    /// Set `shard-N` read-only so any operation that must write there fails,
    /// while the other shards succeed. That is the fan-out partial failure.
    #[cfg(test)]
    fn set_shard_writable(base: &std::path::Path, shard: usize, writable: bool) {
        use std::os::unix::fs::PermissionsExt;
        let d = base.join(format!("shard-{shard}"));
        let mut perm = std::fs::metadata(&d).unwrap().permissions();
        perm.set_mode(if writable { 0o755 } else { 0o555 });
        std::fs::set_permissions(&d, perm).unwrap();
    }

    #[cfg(test)]
    fn registry_dim(base: &std::path::Path, shard: usize, name: &str) -> Option<usize> {
        read_registry(&base.join(format!("shard-{shard}")))
            .unwrap()
            .into_iter()
            .find(|e| e.name == name)
            .map(|e| e.dim)
    }

    /// A CREATE that fails on one shard commits on the others and rolls nothing
    /// back, so the store keeps an index the client was told does not exist -
    /// and it accepts writes. Recreating the name then splits the catalogue:
    /// the shard that failed takes the new dim, the rest keep the old one, and
    /// LIST reports whichever it aggregated first as if the store agreed.
    #[tokio::test]
    async fn a_create_that_fails_on_one_shard_leaves_nothing_behind() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 4).unwrap();
        set_shard_writable(dir.path(), 2, false);
        assert!(
            shards.vindex_create("x", 4, 1, 1).await.is_err(),
            "fixture: the create must fail on the read-only shard"
        );
        set_shard_writable(dir.path(), 2, true);

        for shard in 0..4 {
            assert_eq!(
                registry_dim(dir.path(), shard, "x"),
                None,
                "shard {shard} kept an index whose creation was refused"
            );
        }
        assert!(
            shards
                .vset("x", 0, vec![1.0; 4], 0, None, None)
                .await
                .is_err(),
            "a refused index accepted a write"
        );

        // And the name is clean enough to be created again, with a different
        // shape, without splitting the catalogue.
        shards.vindex_create("x", 8, 1, 1).await.unwrap();
        for shard in 0..4 {
            assert_eq!(
                registry_dim(dir.path(), shard, "x"),
                Some(8),
                "shard {shard} disagrees about the dim of a freshly created index"
            );
        }
    }

    /// Recovery removes the registry entry and the directory, then sweeps the
    /// blobs. A crash in that gap leaves the entry already gone, so the NEXT
    /// open counts zero shards holding the name, decides there is nothing to
    /// do, clears the record - and the blobs are orphaned for good. The sweep
    /// must not be conditional on the entry still being there.
    #[tokio::test]
    async fn blobs_survive_a_crash_between_removing_the_index_and_sweeping_them() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 3).unwrap();
        shards.vindex_create("gone", 4, 1, 1).await.unwrap();
        for id in 0u64..15 {
            shards
                .vset(
                    "gone",
                    id,
                    vec![id as f32, 1.0, 1.0, 1.0],
                    0,
                    None,
                    Some(Bytes::from_static(b"payload")),
                )
                .await
                .unwrap();
        }
        shards.vindex_consolidate("gone").await.unwrap();
        assert!(
            shards.payload_blobs_held(0, "gone").await.unwrap() > 0,
            "fixture: some blob is stored"
        );
        drop(shards);

        // The state a crash mid-recovery leaves: the record still standing, the
        // catalogue entries and directories already gone, the blobs untouched.
        crate::catalog_intent::record(dir.path(), crate::catalog_intent::Op::Drop, "gone").unwrap();
        for shard in 0..3 {
            let sdir = dir.path().join(format!("shard-{shard}"));
            persist_registry_removing(&sdir, &RwLock::new(VindexSet::new()), Some("gone")).unwrap();
            remove_vindex_dir(&sdir, "gone");
        }

        let reopened = ShardSet::open(dir.path(), 3).unwrap();
        assert_eq!(
            reopened.payload_blobs_held(0, "gone").await.unwrap(),
            0,
            "blobs were orphaned by a crash mid-recovery"
        );
    }

    /// The other half of the same rule: a store left genuinely half-dropped
    /// cannot be made consistent by a reader, so a read-only open refuses and
    /// names what is wrong instead of serving around it.
    #[tokio::test]
    async fn serve_mode_refuses_a_store_with_a_half_finished_fanout() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 3).unwrap();
        shards.vindex_create("half", 4, 1, 1).await.unwrap();
        for id in 0u64..12 {
            shards
                .vset("half", id, vec![id as f32, 1.0, 1.0, 1.0], 0, None, None)
                .await
                .unwrap();
        }
        shards.vindex_consolidate("half").await.unwrap();

        set_shard_writable(dir.path(), 0, false);
        assert!(
            shards.vindex_drop("half", 0).await.is_err(),
            "fixture: shard 0 cannot commit, the other two do"
        );
        set_shard_writable(dir.path(), 0, true);
        assert_eq!(
            registry_dim(dir.path(), 0, "half"),
            Some(4),
            "fixture: exactly one shard still holds it"
        );
        drop(shards);

        let msg = match ShardSet::open_mode(dir.path(), 3, true, QuantKind::TurboQuant { bits: 2 })
        {
            Ok(_) => panic!("a reader must not serve around a half-finished fan-out"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("half"),
            "the refusal must name the index: {msg}"
        );
    }

    /// A read-only open cannot resolve, but it can DECIDE - reading registries
    /// writes nothing. Declining to serve every in-flight name hides an index a
    /// refused drop left whole, which is a healthy index missing from serve mode
    /// for no reason.
    #[tokio::test]
    async fn serve_mode_still_serves_an_index_whose_drop_was_refused() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 3).unwrap();
        shards.vindex_create("kept", 4, 1, 1).await.unwrap();
        for id in 0u64..15 {
            shards
                .vset("kept", id, vec![id as f32, 1.0, 1.0, 1.0], 0, None, None)
                .await
                .unwrap();
        }
        shards.vindex_consolidate("kept").await.unwrap();
        for shard in 0..3 {
            set_shard_writable(dir.path(), shard, false);
        }
        assert!(
            shards.vindex_drop("kept", 0).await.is_err(),
            "fixture: no shard can commit the removal"
        );
        for shard in 0..3 {
            set_shard_writable(dir.path(), shard, true);
        }
        drop(shards);
        eprintln!(
            "SONDA intent dopo il drop rifiutato: {:?}",
            crate::catalog_intent::pending(dir.path())
        );

        let serve =
            ShardSet::open_mode(dir.path(), 3, true, QuantKind::TurboQuant { bits: 2 }).unwrap();
        eprintln!(
            "SONDA residente subito dopo l'apertura: {:?}",
            serve
                .vindex_list()
                .await
                .unwrap()
                .iter()
                .map(|r| format!("{} resident={}", r.name, r.shards_resident))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            serve
                .vsearch("kept", vec![1.0; 4], 15, 0, 0, false, None)
                .await
                .unwrap()
                .len(),
            15,
            "serve mode hid an index that a refused drop left whole"
        );
    }

    /// A DROP refused by EVERY shard removed nothing, and the caller was told
    /// so. Finishing it at the next open would turn a refused deletion into a
    /// silent one - and it is not a corner: on a single-shard store every
    /// failed drop looks like this. It is the reason the record names the
    /// operation instead of just the name.
    #[tokio::test]
    async fn a_drop_refused_everywhere_leaves_the_index_whole() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 3).unwrap();
        shards.vindex_create("whole", 4, 1, 1).await.unwrap();
        for id in 0u64..15 {
            shards
                .vset("whole", id, vec![id as f32, 1.0, 1.0, 1.0], 0, None, None)
                .await
                .unwrap();
        }
        shards.vindex_consolidate("whole").await.unwrap();

        for shard in 0..3 {
            set_shard_writable(dir.path(), shard, false);
        }
        assert!(
            shards.vindex_drop("whole", 0).await.is_err(),
            "fixture: no shard can commit the removal"
        );
        for shard in 0..3 {
            set_shard_writable(dir.path(), shard, true);
            assert_eq!(
                registry_dim(dir.path(), shard, "whole"),
                Some(4),
                "fixture: shard {shard} still has it, so nothing was removed"
            );
        }
        drop(shards);

        let reopened = ShardSet::open(dir.path(), 3).unwrap();
        for shard in 0..3 {
            assert_eq!(
                registry_dim(dir.path(), shard, "whole"),
                Some(4),
                "shard {shard} completed a drop that never started"
            );
        }
        assert_eq!(
            reopened
                .vsearch("whole", vec![1.0; 4], 15, 0, 0, false, None)
                .await
                .unwrap()
                .len(),
            15,
            "every row of a refused drop must still be there"
        );
    }

    /// A DROP that fails on one shard has already removed the data from the
    /// others; it cannot be undone, so it has to be finished. Until it is, the
    /// remainder is a zombie: unreachable through search, still on disk, still
    /// in that shard's catalogue, and back at the next open.
    #[tokio::test]
    async fn a_drop_that_fails_on_one_shard_is_finished_at_the_next_open() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 4).unwrap();
        shards.vindex_create("y", 4, 1, 1).await.unwrap();
        for id in 0u64..12 {
            shards
                .vset(
                    "y",
                    id,
                    vec![id as f32, 1.0, 1.0, 1.0],
                    0,
                    None,
                    Some(Bytes::from_static(b"payload")),
                )
                .await
                .unwrap();
        }
        shards.vindex_consolidate("y").await.unwrap();
        assert!(
            shards.payload_blobs_held(0, "y").await.unwrap() > 0,
            "fixture: no blob is stored, so the reclamation check is vacuous"
        );

        set_shard_writable(dir.path(), 0, false);
        assert!(
            shards.vindex_drop("y", 0).await.is_err(),
            "fixture: the drop must fail on the read-only shard"
        );
        set_shard_writable(dir.path(), 0, true);
        assert_eq!(
            registry_dim(dir.path(), 0, "y"),
            Some(4),
            "fixture: shard 0 is the one that kept it"
        );
        drop(shards);

        let reopened = ShardSet::open(dir.path(), 4).unwrap();
        for shard in 0..4 {
            assert_eq!(
                registry_dim(dir.path(), shard, "y"),
                None,
                "shard {shard} brought a half-dropped index back"
            );
        }
        assert!(
            reopened
                .vsearch("y", vec![1.0; 4], 5, 0, 0, false, None)
                .await
                .is_err(),
            "a half-dropped index is searchable again after a restart"
        );
        assert_eq!(
            reopened.payload_blobs_held(0, "y").await.unwrap(),
            0,
            "the blobs of a finished drop outlived its index"
        );
    }

    /// An unreadable registry is the single most useful thing CHECK could ever
    /// tell an operator, so it must arrive as a FINDING, not as an error that
    /// takes the whole report down with it. The fail-closed rule is about not
    /// returning a short answer that looks complete; a report which names the
    /// damage is complete.
    #[tokio::test]
    async fn check_reports_an_unreadable_registry_instead_of_going_dark() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 2).unwrap();
        shards.vindex_create("cold", 4, 1, 1).await.unwrap();
        shards
            .vset("cold", 1, vec![1.0, 0.0, 0.0, 0.0], 0, None, None)
            .await
            .unwrap();
        shards.vindex_consolidate("cold").await.unwrap();
        let ctl = shards.control_handle();
        assert!(ctl.evict(0, "cold").await.unwrap(), "fixture: was resident");

        // Break the catalogue on shard 0 only, so the coordinator still knows
        // the index exists and the report has something to be wrong about.
        let reg = dir.path().join("shard-0").join(VINDEX_REGISTRY);
        assert!(reg.exists(), "fixture: shard 0 has a registry to break");
        std::fs::write(&reg, b"not a registry").unwrap();

        let lines = shards
            .check("cold")
            .await
            .expect("an unreadable registry is a finding, not a dead report");
        assert!(
            lines.iter().any(|l| l.contains("registry")),
            "check said nothing about a registry it could not read: {lines:?}"
        );
        assert!(
            !lines.iter().any(|l| l.contains("missing on shard")),
            "an unreadable shard is UNKNOWN, not missing; saying missing sends \
             the operator after the wrong fault: {lines:?}"
        );
    }

    /// HEALTH answered "no such vindex" for an index that is committed on every
    /// shard and merely evicted from all of them - the exact false answer this
    /// command exists to refuse. It must say it could not assess it instead.
    #[tokio::test]
    async fn health_refuses_to_judge_an_index_it_could_not_read() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 2).unwrap();
        shards.vindex_create("cold", 4, 1, 1).await.unwrap();
        for id in 1u64..=20 {
            shards
                .vset("cold", id, vec![id as f32, 0.0, 0.0, 0.0], 0, None, None)
                .await
                .unwrap();
        }
        shards.vindex_consolidate("cold").await.unwrap();
        assert!(
            shards
                .health("cold")
                .await
                .unwrap()
                .iter()
                .any(|l| l == "state OK"),
            "fixture: healthy while resident"
        );

        let ctl = shards.control_handle();
        assert!(ctl.evict(0, "cold").await.unwrap(), "fixture: was resident");

        let lines = shards
            .health("cold")
            .await
            .expect("a committed index is not 'no such vindex'");
        assert!(
            !lines.iter().any(|l| l == "state OK"),
            "called an index OK on numbers it never read: {lines:?}"
        );
        assert!(
            lines.iter().any(|l| l.starts_with("not_assessed ")),
            "state must name what it could not read: {lines:?}"
        );
    }

    /// A committed index whose files will not open is still committed: it is in
    /// the catalogue, it occupies disk, and no client can get rid of it. Reading
    /// "cannot open" as "does not exist" left the entry unremovable forever -
    /// the store had a row nobody could delete, and a tenant erasure returned
    /// success with the vectors on disk.
    #[tokio::test]
    async fn dropping_a_committed_vindex_that_will_not_open_still_removes_it() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 2).unwrap();
        shards.vindex_create("orph", 4, 1, 1).await.unwrap();
        // A second index whose name has the first as a PREFIX. Reclaiming blobs
        // by key prefix would take this one's too: the key is
        // `tenant | marker | name | id`, with nothing between name and id.
        shards.vindex_create("orph2", 4, 1, 1).await.unwrap();
        for name in ["orph", "orph2"] {
            for id in 1u64..=12 {
                shards
                    .vset(
                        name,
                        id,
                        vec![id as f32, 0.0, 0.0, 0.0],
                        0,
                        None,
                        Some(Bytes::from_static(b"payload")),
                    )
                    .await
                    .unwrap();
            }
            shards.vindex_consolidate(name).await.unwrap();
        }

        let sibling_before = shards.payload_blobs_held(0, "orph2").await.unwrap();
        assert!(
            shards.payload_blobs_held(0, "orph").await.unwrap() > 0 && sibling_before > 0,
            "fixture: one of the two indexes stored no blob, so this test \
             could not tell them apart"
        );

        let ctl = shards.control_handle();
        assert!(ctl.evict(0, "orph").await.unwrap(), "fixture: was resident");

        // Break it on disk: every graph the index would load, filled with bytes
        // that are not a graph.
        let mut broken = 0usize;
        for i in 0..2 {
            let vdir = dir.path().join(format!("shard-{i}")).join("vindex-orph");
            let mut stack = vec![vdir];
            while let Some(d) = stack.pop() {
                for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
                    let p = e.path();
                    if p.is_dir() {
                        stack.push(p);
                    } else if p.file_name().is_some_and(|n| n == "graph.vmn") {
                        std::fs::write(&p, b"not a graph").unwrap();
                        broken += 1;
                    }
                }
            }
        }
        assert!(broken > 0, "fixture: found no graph to break");

        shards.vindex_drop("orph", 0).await.unwrap();

        for i in 0..2 {
            let sdir = dir.path().join(format!("shard-{i}"));
            assert!(
                !read_registry(&sdir)
                    .unwrap()
                    .iter()
                    .any(|e| e.name == "orph"),
                "shard {i}: the catalogue still names an index nobody can remove"
            );
            assert!(
                !sdir.join("vindex-orph").exists(),
                "shard {i}: the unopenable index still occupies disk"
            );
        }
        assert_eq!(
            shards.payload_blobs_held(0, "orph").await.unwrap(),
            0,
            "payload blobs outlived the index they belonged to"
        );
        assert_eq!(
            shards.payload_blobs_held(0, "orph2").await.unwrap(),
            sibling_before,
            "dropping 'orph' took blobs of 'orph2', whose name it prefixes"
        );
    }

    /// The erasure path walks the resident map, so a tenant's evicted index is
    /// never even attempted: the KV sweep deletes its payload blobs, the call
    /// reports success, and the vectors stay on disk. The existing erasure test
    /// cannot see this - it builds a flat index, which cannot be evicted.
    #[tokio::test]
    async fn erasing_a_tenant_erases_its_evicted_vindexes_too() {
        const VICTIM: u128 = 7;
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 2).unwrap();
        let name = scope_key(VICTIM, "idx");
        shards.vindex_create_scoped(&name, 4, 1, 1).await.unwrap();
        for id in 0u64..20 {
            shards
                .vset(
                    &name,
                    id,
                    vec![id as f32, 0.2, 0.3, 0.4],
                    VICTIM,
                    None,
                    Some(Bytes::from_static(b"payload")),
                )
                .await
                .unwrap();
        }
        shards.vindex_consolidate(&name).await.unwrap();

        let ctl = shards.control_handle();
        assert!(
            ctl.evict(VICTIM, "idx").await.unwrap(),
            "fixture: the index must be resident so the evict has something to do"
        );

        let (vindexes, _) = shards
            .erase_tenant(VICTIM, Durability::Kernel)
            .await
            .unwrap();
        assert!(
            vindexes >= 1,
            "erasure reported no vindex for a tenant that owns one on disk"
        );

        for i in 0..2 {
            let sdir = dir.path().join(format!("shard-{i}"));
            assert!(
                !sdir.join(format!("vindex-{name}")).exists(),
                "shard {i}: the erased tenant's vectors are still on disk"
            );
            assert!(
                !read_registry(&sdir).unwrap().iter().any(|e| e.name == name),
                "shard {i}: the catalogue still names the erased tenant's index"
            );
        }
    }

    /// Scale-to-zero: a fleet of K disk-backed tenant indices, all resident,
    /// then evict the cold majority. Measures the RAM the eviction reclaims
    /// (`resident_bytes`, allocator-independent) and proves a cold tenant reloads
    /// with its data intact. This is the engine half of the BUSL overcommit story.
    #[tokio::test]
    async fn scale_to_zero_reclaims_fleet_ram() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 2).unwrap();
        let k = 10usize; // tenants
        let m = 1000u64; // vectors each
        let dim = 128usize;
        let vec_for = |t: usize, id: u64| {
            let mut v = vec![0f32; dim];
            v[0] = id as f32;
            v[1 + (t % (dim - 1))] = 1.0; // a per-tenant component
            v
        };
        for t in 0..k {
            let name = format!("t{t}");
            // backend 1 = disk Vamana (evictable + lazily reopenable).
            shards.vindex_create(&name, dim as u32, 1, 1).await.unwrap();
            for id in 1..=m {
                shards
                    .vset(&name, id, vec_for(t, id), 0, None, None)
                    .await
                    .unwrap();
            }
            shards.vindex_consolidate(&name).await.unwrap();
        }

        let ctl = shards.control_handle();
        let resident = |ctl: ControlHandle| async move {
            ctl.open_indices()
                .await
                .iter()
                .map(|s| s.resident_bytes)
                .sum::<usize>()
        };

        let r_all = resident(ctl.clone()).await;
        // Scale-to-zero: keep 3 hot, evict the 7 cold tenants.
        for t in 3..k {
            assert!(ctl.evict(0, &format!("t{t}")).await.unwrap());
        }
        let r_hot = resident(ctl.clone()).await;
        let reclaimed = 100.0 * (r_all - r_hot) as f64 / r_all as f64;
        println!(
            "scale-to-zero: {k} tenants resident {r_all} B -> 3 hot {r_hot} B = {reclaimed:.0}% reclaimed"
        );
        assert!(r_hot < r_all, "eviction must reclaim resident RAM");
        assert!(
            reclaimed > 50.0,
            "evicting 7/10 idle tenants reclaims the majority"
        );

        // A cold (evicted) tenant reloads lazily on access, data intact.
        let q = vec_for(9, m); // closest to t9's id=m vector
        let hits = shards.vsearch("t9", q, 5, 0, 0, false, None).await.unwrap();
        assert!(
            !hits.is_empty(),
            "evicted tenant reopens and serves on next query"
        );
        assert!(
            ctl.open_indices().await.iter().any(|s| s.index == "t9"),
            "reopened tenant is resident again"
        );
    }

    #[tokio::test]
    async fn create_after_evict_keeps_evicted_reopenable() {
        // Regression: persist_registry used to rebuild from the resident map, so a
        // create/drop after an evict rewrote the registry WITHOUT the evicted
        // vindex -> its dir stayed but it was no longer reopenable ("not found").
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 2).unwrap();
        for n in ["a", "b"] {
            shards.vindex_create(n, 4, 1, 1).await.unwrap(); // disk-backed
            for id in 1u64..=20 {
                shards
                    .vset(n, id, vec![id as f32, 0., 0., 0.], 0, None, None)
                    .await
                    .unwrap();
            }
            shards.vindex_consolidate(n).await.unwrap();
        }
        shards.control_handle().evict(0, "a").await.unwrap(); // a: out of map, dir stays
        shards.vindex_create("c", 4, 1, 1).await.unwrap(); // triggers persist_registry (the bug trigger)

        // Without the fix, "a" was dropped from the registry here -> not found.
        let q = vec![20.0f32, 0., 0., 0.];
        let hits = shards.vsearch("a", q, 5, 0, 0, false, None).await.unwrap();
        assert!(
            !hits.is_empty(),
            "evicted 'a' must reopen after an intervening create"
        );
    }

    #[tokio::test]
    async fn evicted_vindex_survives_restart() {
        // The registry (not the resident map) is the source of truth for restart
        // recovery. An evicted vindex - out of the map, files + registry entry
        // still on disk - whose registry entry survived an intervening create
        // must come back on a full restart. Guards the registry-as-truth fix
        // across process boundaries, not just in-process lazy reopen.
        let dir = TempDir::new().unwrap();
        let base = dir.path().to_owned();
        {
            let shards = ShardSet::open(&base, 2).unwrap();
            for n in ["a", "b"] {
                shards.vindex_create(n, 4, 1, 1).await.unwrap(); // disk-backed
                for id in 1u64..=20 {
                    shards
                        .vset(n, id, vec![id as f32, 0., 0., 0.], 0, None, None)
                        .await
                        .unwrap();
                }
                shards.vindex_consolidate(n).await.unwrap();
            }
            shards.control_handle().evict(0, "a").await.unwrap(); // a: out of map, files + registry stay
            shards.vindex_create("c", 4, 1, 1).await.unwrap(); // rewrites the registry (must keep "a")
            // shards dropped here: worker threads flush and exit.
        }

        // Restart: recover_vindexes must reopen "a" from the registry.
        let shards = ShardSet::open(&base, 2).unwrap();
        let q = vec![20.0f32, 0., 0., 0.];
        let hits = shards.vsearch("a", q, 5, 0, 0, false, None).await.unwrap();
        assert!(
            !hits.is_empty(),
            "evicted-then-restarted 'a' must recover from the registry"
        );
    }

    /// A scoped key: 16-byte tenant prefix (LE) + raw, as the RESP3 handler
    /// builds them. The vLog derives the disk tenant from this prefix.
    fn scoped_key(tenant: u128, raw: &[u8]) -> Vec<u8> {
        let mut k = tenant.to_le_bytes().to_vec();
        k.extend_from_slice(raw);
        k
    }

    #[tokio::test]
    async fn test_disk_quota_global_per_tenant() {
        // max_disk_bytes is enforced GLOBALLY per tenant across shards,
        // not per shard. Each padded record here is 128 bytes; a 300-byte limit
        // admits exactly 2, regardless of how the 5 keys hash across the shards.
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 4).unwrap();
        let lim = Some(300u64);
        let v = b"v";
        let mut ok = 0;
        let mut rejected = 0;
        for i in 0..5u8 {
            let k = scoped_key(7, &[b'a' + i]);
            match shards
                .tenant(7)
                .with_disk_limit(lim)
                .set(&k, v, Durability::Kernel)
                .await
            {
                Ok(()) => ok += 1,
                Err(_) => rejected += 1,
            }
        }
        assert_eq!(ok, 2, "global limit admits exactly 2 records across shards");
        assert_eq!(rejected, 3);
        assert!(shards.tenant_disk_bytes(7) <= 300);

        // A different tenant has its own independent global budget.
        let k9 = scoped_key(9, b"a");
        shards
            .tenant(9)
            .with_disk_limit(lim)
            .set(&k9, v, Durability::Kernel)
            .await
            .unwrap();
        assert!(shards.tenant_disk_bytes(9) > 0);
        assert!(
            shards.tenant_disk_bytes(7) <= 300,
            "tenant 9 did not affect 7"
        );
    }

    #[tokio::test]
    async fn test_vector_quota_untracked_without_limit() {
        // No limit -> no counting; single-tenant path pays nothing.
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 2).unwrap();
        shards.vindex_create("idx", 4, 0, 0).await.unwrap();
        let v = vec![1.0f32, 0.0, 0.0, 0.0];
        for id in 0..50u64 {
            shards
                .vset("idx", id, v.clone(), 0, None, None)
                .await
                .unwrap();
        }
        assert_eq!(
            shards.tenant_vector_count(0),
            0,
            "no limit means no quota tracking"
        );
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

    /// Every item of a VMSET succeeded, and how many there were - the shape
    /// the call had before it started answering per item.
    fn all_ok(results: Vec<Result<(), ShardError>>) -> usize {
        for (i, r) in results.iter().enumerate() {
            r.as_ref().unwrap_or_else(|e| panic!("vmset item {i}: {e}"));
        }
        results.len()
    }

    /// Deterministic 64-dim test vector.
    #[allow(clippy::cast_precision_loss)]
    /// `tvec` at 16 dims: the same deterministic shape, a quarter of the
    /// graph work, for tests whose subject is the maintenance path and not the
    /// geometry.
    fn tvec16(seed: u64) -> Vec<f32> {
        let mut s = (seed << 1) | 1;
        (0..16)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s % 1000) as f32 / 1000.0
            })
            .collect()
    }

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

    // A disk-backed VINDEX must survive a server restart - its files plus
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
                shards
                    .vset("persist", id, tvec(id + 1), 0, None, None)
                    .await
                    .unwrap();
            }
            // `shards` dropped here: worker threads flush and exit.
        }

        // Restart: a fresh ShardSet on the same dir recovers the disk VINDEX
        // from the registry + WAL.
        let shards = ShardSet::open(&base, 4).unwrap();
        let hits = shards
            .vsearch("persist", tvec(89), 5, 0, 0, false, None)
            .await
            .unwrap();
        assert_eq!(hits[0].0, 88, "disk VINDEX must be recovered after restart");
        // A flat VINDEX, by contrast, is in-RAM and would not survive - so the
        // recovered set contains exactly the disk-backed index.
        assert!(shards.vget("persist", 42).await.unwrap().is_some());
    }

    // VINDEX.CONSOLIDATE folds a disk index's streaming delta into the graph;
    // the same vectors are still searchable afterwards.
    #[tokio::test]
    async fn test_vindex_consolidate() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 2).unwrap();
        shards.vindex_create("idx", 64, 0, 1).await.unwrap(); // disk
        for id in 0u64..200 {
            shards
                .vset("idx", id, tvec(id + 1), 0, None, None)
                .await
                .unwrap();
        }
        shards.vindex_consolidate("idx").await.unwrap();
        let hits = shards
            .vsearch("idx", tvec(90), 5, 0, 0, false, None)
            .await
            .unwrap();
        assert_eq!(hits[0].0, 89, "nearest still found after consolidate");
        // Consolidate is idempotent and a no-op on flat indices.
        shards.vindex_consolidate("idx").await.unwrap();
        shards.vindex_create("flat", 64, 0, 0).await.unwrap();
        shards.vindex_consolidate("flat").await.unwrap();
    }

    /// An explicit `SKEG.VINDEX.CONSOLIDATE` must not block reads on that vindex
    /// for the duration of the rebuild.
    ///
    /// The idle path already builds off-thread via `off_thread_maintenance`
    /// (brief write lock for begin, `spawn_blocking` for the build, brief lock
    /// for finish). The explicit command instead called
    /// `arc.write().backend.consolidate()`, holding the per-vindex write lock
    /// for the WHOLE rebuild: every query to that vindex queued behind it. On a
    /// public service taking nightly updates that is minutes of stalled reads.
    ///
    /// Timing-based on purpose: the property under test IS "reads proceed while
    /// the build runs", and there is no way to observe that structurally from
    /// outside. The margin is wide (a search must be at least 5x faster than the
    /// consolidate) so it does not flake on a loaded machine.
    #[tokio::test]
    async fn explicit_consolidate_does_not_block_reads() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 1).unwrap();
        shards.vindex_create("idx", 64, 0, 1).await.unwrap(); // disk
        // Enough vectors that the rebuild is measurably slower than a search.
        for id in 0u64..4000 {
            shards
                .vset("idx", id, tvec(id + 1), 0, None, None)
                .await
                .unwrap();
        }

        let reader = {
            let s = shards.clone();
            tokio::spawn(async move {
                let mut worst = std::time::Duration::ZERO;
                for i in 0..40u64 {
                    let t = std::time::Instant::now();
                    s.vsearch("idx", tvec(i * 7 + 1), 5, 0, 0, false, None)
                        .await
                        .unwrap();
                    worst = worst.max(t.elapsed());
                    tokio::time::sleep(std::time::Duration::from_millis(2)).await;
                }
                worst
            })
        };

        let t0 = std::time::Instant::now();
        shards.vindex_consolidate("idx").await.unwrap();
        let consolidate = t0.elapsed();
        let worst_read = reader.await.unwrap();

        assert!(
            worst_read * 5 < consolidate,
            "a read waited {worst_read:?} while the consolidate took \
             {consolidate:?}: the write lock is held for the whole rebuild"
        );

        // and the result must stay correct
        let hits = shards
            .vsearch("idx", tvec(90), 5, 0, 0, false, None)
            .await
            .unwrap();
        assert_eq!(hits[0].0, 89, "nearest neighbour still found");
    }

    // The off-thread maintenance helper drives a runs-merge and a delete-patch
    // through VectorBackend end to end: runs fold to one, dead base rows are
    // reclaimed, and liveness is correct throughout. Exercises the exact server
    // wiring (delegation + begin/build/finish orchestration), not just the
    // engine primitives.
    /// `maintenance_tick` picks one operation per vindex per tick, in a fixed
    /// priority order, and tells the caller whether it consolidated. That
    /// return value is what resets idle tracking, so a wrong one silently
    /// breaks the idle fold. Nothing else exercises this function: the idle
    /// loop that calls it runs on a timer.
    /// A set written with N shards must be DISCOVERED as N. Read-only serve
    /// mode hardcoded 1 and served an eighth of an eight-shard index without
    /// an error, a warning, or a hint - answering every query confidently
    /// with 12% of the data.
    #[tokio::test]
    async fn shard_count_is_discovered_from_disk_not_assumed() {
        let dir = TempDir::new().unwrap();
        {
            let shards = ShardSet::open_mode_with_workers(
                dir.path(),
                8,
                false,
                skeg_vector::QuantKind::TurboQuant { bits: 2 },
                1,
            )
            .unwrap();
            shards.vindex_create("sd", 8, 4, 1).await.unwrap();
            for id in 0..800u64 {
                let mut v = vec![0.05f32; 8];
                v[(id % 4) as usize] = 1.0;
                shards.vset("sd", id, v, 0, None, None).await.unwrap();
            }
        }
        let layout = crate::layout_manifest::LayoutManifest::open_or_migrate(
            dir.path(),
            crate::layout_manifest::OpenMode::ReadOnly,
        )
        .unwrap();
        assert_eq!(
            layout.shard_count().get(),
            8,
            "an eight-shard set must report eight shards"
        );
        // Reopening sees every row; the old hardcoded 1 would see an eighth.
        let n = layout.shard_count().get();
        let re = ShardSet::open_mode_with_workers(
            dir.path(),
            n,
            false,
            skeg_vector::QuantKind::TurboQuant { bits: 2 },
            1,
        )
        .unwrap();
        let rows = re.vindex_list().await.unwrap();
        let total: u64 = rows
            .iter()
            .filter(|r| r.name == "sd")
            .map(|r| r.n_vectors)
            .sum();
        assert_eq!(total, 800, "reopen lost rows: saw {total} of 800");
        // Fail-closed: an absent or holed layout is an ERROR, never a guess.
        assert!(
            crate::layout_manifest::LayoutManifest::open_or_migrate(
                &dir.path().join("nope"),
                crate::layout_manifest::OpenMode::ReadOnly,
            )
            .is_err()
        );
        std::fs::remove_dir_all(dir.path().join("shard-3")).unwrap();
        let holed = crate::layout_manifest::LayoutManifest::open_or_migrate(
            dir.path(),
            crate::layout_manifest::OpenMode::ReadOnly,
        );
        assert!(
            holed.is_err(),
            "a gap in the numbering must not be served around"
        );
    }

    /// Two heavy jobs must not run on the same vindex at once. The engine's
    /// comments assumed it; nothing enforced it, so an explicit command could
    /// overlap the automatic loop on the same runs.
    #[tokio::test]
    async fn one_heavy_job_per_vindex() {
        let dir = TempDir::new().unwrap();
        let vdir = dir.path().join("vindex-t");
        let mut idx = DiskVamanaIndex::create_empty_with_tier(
            &vdir,
            64,
            64,
            QuantKind::TurboQuant { bits: 2 },
        )
        .unwrap();
        idx.set_auto_flush(false);
        for id in 0u64..4000 {
            idx.insert(id, &tvec(id + 1)).unwrap();
        }
        idx.consolidate().unwrap().expect_clean();
        let arc: VectorEntry = Arc::new(RwLock::new(Vindex::new(
            VectorBackend::Disk(Box::new(idx)),
            4,
        )));
        // Hold the vindex's heavy gate, as an in-flight job would.
        // Clone the handle, DROP the read lock, then acquire: exactly the
        // discipline the production path follows.
        let gate = Arc::clone(&arc.read().heavy);
        let held = gate.acquire_owned().await.unwrap();
        let d = vdir.clone();
        let outcome = off_thread_maintenance(
            &arc,
            "consolidate",
            0,
            |b| b.consolidate_begin(),
            move |job| job.build(&d),
            |b, built| b.consolidate_finish(built),
        )
        .await;
        assert_eq!(
            outcome,
            MaintenanceOutcome::BudgetBusy,
            "a second heavy job started while one was in flight"
        );
        // A flush is NOT gated: it must still run alongside.
        {
            let mut g = arc.write();
            for id in 100_000u64..(100_000 + FLUSH_ROWS as u64 + 50) {
                let v = tvec(id);
                g.backend
                    .insert(id, &v, VectorVersion::LEGACY, PayloadRef::Unchanged)
                    .unwrap();
            }
        }
        let d = vdir.clone();
        let flushed = off_thread_maintenance(
            &arc,
            "flush",
            0,
            |b| b.flush_begin(),
            move |job| job.build(&d),
            |b, built| b.flush_finish(built),
        )
        .await;
        assert_eq!(
            flushed,
            MaintenanceOutcome::Ran,
            "the flush queued behind a heavy job it is meant to coexist with"
        );
        drop(held);
    }

    // ---- the ladder's decision, as a decision ----
    //
    // What matters about the ladder is WHICH rung it picks and in what order,
    // not the work the rung then schedules. Driving real maintenance to assert
    // that took half an hour before the threshold became an argument, and even
    // now it walks one trajectory through the state space. `ladder_plan` is
    // pure, so the space itself can be covered.

    fn st(delta: usize, runs: usize, run_rows: usize, tombs: usize, base: usize) -> LsmState {
        LsmState {
            delta,
            runs,
            run_rows,
            tombs,
            base,
            flush_streak: 0,
        }
    }

    #[test]
    fn a_quiet_index_schedules_nothing() {
        assert!(ladder_plan(&st(0, 0, 0, 0, 10_000), 4096).is_empty());
    }

    #[test]
    fn a_full_delta_flushes() {
        assert_eq!(
            ladder_plan(&st(5000, 0, 0, 0, 10_000), 4096),
            vec![Rung::Flush]
        );
    }

    #[test]
    fn the_flush_wins_the_tick_when_both_are_due() {
        // The flush is the cheapest rung and it bounds the RAM delta, so it
        // goes first - once.
        let plan = ladder_plan(&st(5000, 4, 4000, 0, 10_000), 4096);
        assert_eq!(plan.first(), Some(&Rung::Flush));
    }

    #[test]
    fn the_merge_goes_first_once_the_flush_has_already_won_a_tick() {
        // The anti-starvation rule: a permanently hot flush cannot hold the
        // tick forever while runs pile up. Measured cost of getting this
        // wrong: 0 -> 33 runs over ten turnovers, with recall falling
        // 0.9925 -> 0.7180 alongside (mechanism never isolated - see
        // `ladder_plan`).
        let mut s = st(5000, 4, 4000, 0, 10_000);
        s.flush_streak = 1;
        assert_eq!(ladder_plan(&s, 4096).first(), Some(&Rung::RunsMerge));
    }

    #[test]
    fn a_declined_merge_falls_through_to_the_flush() {
        // `NotNeeded` means the exact check found nothing worth rewriting.
        // The tick must not be consumed by a refusal - the flush is still due.
        let mut s = st(5000, 4, 4000, 0, 10_000);
        s.flush_streak = 1;
        assert_eq!(ladder_plan(&s, 4096), vec![Rung::RunsMerge, Rung::Flush]);
    }

    #[test]
    fn the_merge_is_never_planned_twice() {
        // It used to be. With a due merge and a quiet delta the ladder tried
        // the merge, took the refusal, fell past a flush that was not due, and
        // reached a SECOND runs-merge rung where `runs >= RUNS_MERGE_TRIGGER`
        // was still true - attempting the identical operation for the identical
        // refusal. That refusal is not cheap: `merge_runs_begin` builds the
        // whole survivor set, one hash insert per row of every run, before it
        // can decide there is nothing to do, holding the write lock readers
        // contend for.
        //
        // Swept, not spot-checked: no reachable state may plan it twice.
        for delta in [0, 100, 4095, 4096, 100_000] {
            for runs in 0..8 {
                for tombs in [0, 1, 5000, 60_000] {
                    for base in [0, 4096, 100_000] {
                        for streak in [0u64, 1, 7] {
                            let mut s = st(delta, runs, runs * 4096, tombs, base);
                            s.flush_streak = streak;
                            let plan = ladder_plan(&s, 4096);
                            let n = plan.iter().filter(|r| **r == Rung::RunsMerge).count();
                            assert!(n <= 1, "{s:?} plans {n} merges: {plan:?}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn every_plan_is_free_of_repeats() {
        // The same argument as above, for every rung: a plan is an ordered set
        // of things to try, and trying one twice in a tick is always waste.
        for delta in [0, 4096, 50_000] {
            for runs in 0..6 {
                for tombs in [0, 100, 30_000, 90_000] {
                    for base in [0, 4096, 100_000] {
                        let s = st(delta, runs, runs * 4096, tombs, base);
                        let plan = ladder_plan(&s, 4096);
                        let mut seen = plan.clone();
                        seen.sort_by_key(|r| format!("{r:?}"));
                        seen.dedup();
                        assert_eq!(seen.len(), plan.len(), "{s:?} repeats a rung: {plan:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn a_heavily_tombstoned_base_goes_to_the_fold_not_the_patch() {
        // Past roughly a quarter dead, delete-patch is in its measured losing
        // regime (0.3x at 40%). It fired at 61% during the demo repair before
        // this guard existed.
        let s = st(0, 0, 0, 70_000, 100_000);
        let plan = ladder_plan(&s, 4096);
        assert!(plan.contains(&Rung::Consolidate), "{plan:?}");
        assert!(!plan.contains(&Rung::DeletePatch), "{plan:?}");
    }

    #[test]
    fn a_lightly_tombstoned_base_takes_the_patch() {
        // The regime where reuse wins - the L3 verdict measured 10.6x at 1%
        // dead and 4.5x at 3%, and the trigger sits at base/16 (6.25%), well
        // inside it, with the 25% ceiling above.
        //
        // Both edges pinned, because the first version of this test asserted
        // the patch at 6% and failed: 6% is BELOW the trigger, not above it.
        assert_eq!(
            ladder_plan(&st(0, 0, 0, 10_000, 100_000), 4096),
            vec![Rung::DeletePatch]
        );
        assert!(
            ladder_plan(&st(0, 0, 0, 6_000, 100_000), 4096).is_empty(),
            "6% dead is under the base/16 trigger: nothing is due"
        );
    }

    #[test]
    fn a_lone_dirty_run_is_still_offered_to_the_merge() {
        // Count OR mass: one run is not "four runs" however much garbage it
        // holds, and a lone fat dirty run sat untouched forever. Whether it is
        // dirty enough to rewrite is decided from the survivor set, not here.
        let s = st(0, 1, 4096, 10, 100_000);
        assert_eq!(ladder_plan(&s, 4096).first(), Some(&Rung::RunsMerge));
    }

    /// Sustained writes must not starve the runs-merge. Every rung of the
    /// ladder ends in `return`, so a permanently-hot flush used to hold the
    /// tick forever while each flush quietly added another run - measured on
    /// the churn gate as 0 -> 33 runs over ten turnovers, with recall falling
    /// from 0.9925 to 0.7180 alongside. (The MECHANISM of that fall was never
    /// isolated; two stale-vector defects found later explain it at least as
    /// well as the short beam this used to assert. See `ladder_plan`.)
    ///
    /// Driven at a SMALL flush threshold. What matters is the sequence of
    /// decisions the ladder makes when both rungs are due, not the size of
    /// the work it schedules; at the production threshold the same test took
    /// thirty-two minutes, which meant it sat `#[ignore]` and protected
    /// nothing at all.
    #[tokio::test]
    async fn sustained_writes_do_not_starve_the_runs_merge() {
        const SMALL_FLUSH: usize = 200;
        const DIM: usize = 16;
        // A cheap deterministic vector at the small dim this test uses: the
        // ladder's decision does not depend on the geometry, and the full
        // 64-dim `tvec` makes every round build a real graph.
        fn small(seed: u64) -> Vec<f32> {
            let mut s = (seed << 1) | 1;
            (0..DIM)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    (s % 1000) as f32 / 1000.0
                })
                .collect()
        }
        let dir = TempDir::new().unwrap();
        let vdir = dir.path().join("vindex-t");
        let mut idx = DiskVamanaIndex::create_empty_with_tier(
            &vdir,
            DIM,
            32,
            QuantKind::TurboQuant { bits: 2 },
        )
        .unwrap();
        idx.set_auto_flush(false);
        let arc: VectorEntry = Arc::new(RwLock::new(Vindex::new(
            VectorBackend::Disk(Box::new(idx)),
            4,
        )));
        // The churn shape: refill the delta past the threshold before every
        // tick, so the flush rung is permanently hot.
        let mut worst_runs = 0usize;
        let mut next_id = 0u64;
        for _ in 0..14 {
            {
                let mut g = arc.write();
                for _ in 0..(SMALL_FLUSH + 20) {
                    let v = small(next_id + 1);
                    g.backend
                        .insert(next_id, &v, VectorVersion::LEGACY, PayloadRef::Unchanged)
                        .unwrap();
                    next_id += 1;
                }
            }
            maintenance_tick_at(&arc, &vdir, 0, false, SMALL_FLUSH).await;
            worst_runs = worst_runs.max(arc.read().backend.run_count());
        }
        // The bar is the ORDINARY trigger, not an emergency ceiling: the
        // damage measured on the churn gate happened at four runs per shard,
        // which is the trigger itself. A fix that only acts at a higher
        // ceiling would pass a test and fail the workload.
        assert!(
            worst_runs <= RUNS_MERGE_TRIGGER + 2,
            "runs reached {worst_runs}: the flush starved the merge again"
        );
    }

    #[tokio::test]
    async fn maintenance_tick_prefers_flush_then_consolidate() {
        let dir = TempDir::new().unwrap();
        let vdir = dir.path().join("vindex-t");
        let mut idx = DiskVamanaIndex::create_empty_with_tier(
            &vdir,
            64,
            64,
            QuantKind::TurboQuant { bits: 2 },
        )
        .unwrap();
        // The server drives the flush from its maintenance loop, so it turns
        // the inline auto-flush off; without this the delta drains itself and
        // the flush branch is never reached through `maintenance_tick`.
        idx.set_auto_flush(false);
        // Past FLUSH_ROWS, so flush outranks everything else this tick.
        for id in 0u64..(FLUSH_ROWS as u64 + 100) {
            idx.insert(id, &tvec(id + 1)).unwrap();
        }
        let arc: VectorEntry = Arc::new(RwLock::new(Vindex::new(
            VectorBackend::Disk(Box::new(idx)),
            4,
        )));

        // Flush wins even with `idle` set: it is first in the chain, and it
        // does not consolidate, so the tick reports false.
        let delta_before = arc.read().backend.delta_len();
        assert!(delta_before >= FLUSH_ROWS, "setup did not fill the delta");
        assert!(
            !maintenance_tick(&arc, &vdir, 0, true).await,
            "flush must not report a consolidate",
        );
        assert!(
            arc.read().backend.delta_len() < delta_before,
            "flush did not drain the delta",
        );

        // The flushed rows are now a run of 4196 against an empty base, so the
        // geometric trigger fires: run_rows >= base.max(IDLE_CONSOLIDATE_MIN).
        // It is the run size that decides, not idleness.
        //
        // Asserted as a DECISION first, which is what this test is named for
        // and the only part that is deterministic. Driving it through
        // `maintenance_tick` is not: the tick deliberately refuses to park on
        // the process-wide fold budget, so a concurrent test holding it makes
        // the tick skip. This used to be a hundred retries over five seconds -
        // mitigation, not determinism, and it still failed about one full-suite
        // run in four.
        let state = {
            let g = arc.read();
            LsmState {
                delta: g.backend.delta_len(),
                runs: g.backend.run_count(),
                run_rows: g.backend.run_rows(),
                tombs: g.backend.tombstone_count(),
                base: g.backend.main_len(),
                flush_streak: g.flush_streak.load(Ordering::Relaxed),
            }
        };
        assert_eq!(
            ladder_plan(&state, FLUSH_ROWS),
            vec![Rung::Consolidate],
            "a run past the geometric threshold must plan the fold: {state:?}"
        );

        // Then that the planned fold actually completes - WAITING for the
        // budget rather than skipping, since here there is no request path to
        // protect and no next tick to retry on.
        let d = vdir.clone();
        let ran = try_off_thread_maintenance(
            &arc,
            "consolidate",
            true,
            |b| b.consolidate_begin(),
            move |job| job.build(&d),
            |b, built| b.consolidate_finish(built),
        )
        .await
        .expect("the fold must not fail");
        assert_eq!(ran, MaintenanceOutcome::Ran);
        assert_eq!(arc.read().backend.run_count(), 0, "runs did not fold");

        // Nothing left to do: quiet tick, no consolidate reported.
        assert!(
            !maintenance_tick(&arc, &vdir, 0, true).await,
            "a quiet tick must not report work",
        );
    }

    /// A quiet store with a little pending work must not rebuild itself.
    ///
    /// The chain used to fold whenever `idle && delta + run_rows >= 4096`, so a
    /// store that went quiet with one flush behind it rebuilt its whole base.
    /// That is O(live) work to tidy up a run, and on a large index it is
    /// minutes of it, triggered by nothing more than traffic stopping. A quiet
    /// store needs its runs not to pile up, which is what runs-merge is for.
    #[tokio::test]
    async fn an_idle_tick_does_not_rebuild_a_base_that_dwarfs_its_runs() {
        let dir = TempDir::new().unwrap();
        let vdir = dir.path().join("vindex-t");
        let mut idx = DiskVamanaIndex::create_empty_with_tier(
            &vdir,
            64,
            64,
            QuantKind::TurboQuant { bits: 2 },
        )
        .unwrap();
        // A base far larger than what follows it, so the geometric trigger
        // cannot fire and only the old idle clause could have.
        for id in 0u64..20_000 {
            idx.insert(id, &tvec(id + 1)).unwrap();
        }
        idx.consolidate().unwrap().expect_clean();
        let base = idx.main_len();
        assert!(base >= 20_000, "base did not build: {base}");
        idx.set_auto_flush(false);
        // One flush worth of new rows: enough for the old clause, nowhere near
        // the base.
        for id in 20_000u64..(20_000 + FLUSH_ROWS as u64 + 100) {
            idx.insert(id, &tvec(id + 1)).unwrap();
        }
        let arc: VectorEntry = Arc::new(RwLock::new(Vindex::new(
            VectorBackend::Disk(Box::new(idx)),
            4,
        )));

        let folds_before =
            skeg_telemetry::counter_value(skeg_telemetry::Counter::MaintenanceConsolidate);
        // Flush first, as always.
        assert!(!maintenance_tick(&arc, &vdir, 0, true).await);
        // Then ticks until the store settles. None of them may fold.
        for _ in 0..6 {
            assert!(
                !maintenance_tick(&arc, &vdir, 0, true).await,
                "an idle tick folded a base {} rows against {} run rows",
                arc.read().backend.main_len(),
                arc.read().backend.run_rows(),
            );
        }
        assert_eq!(
            skeg_telemetry::counter_value(skeg_telemetry::Counter::MaintenanceConsolidate)
                - folds_before,
            0,
            "the fold ran despite the runs being a fraction of the base"
        );
    }

    /// Past a quarter of the base dead, the tick folds instead of patching.
    ///
    /// Delete-patch loses in that regime (0,3x at 40% in the L3 verdict) and
    /// during the demo repair it fired on a base 61% dead, which is how this
    /// guard was found missing. The geometric trigger only watches run growth,
    /// so without the heavy-dead arm nothing would ever reclaim such a base.
    #[tokio::test]
    async fn a_heavily_dead_base_folds_instead_of_patching() {
        let dir = TempDir::new().unwrap();
        let vdir = dir.path().join("vindex-t");
        let mut idx = DiskVamanaIndex::create_empty_with_tier(
            &vdir,
            64,
            64,
            QuantKind::TurboQuant { bits: 2 },
        )
        .unwrap();
        for id in 0u64..6_000 {
            idx.insert(id, &tvec(id + 1)).unwrap();
        }
        idx.consolidate().unwrap().expect_clean();
        // Kill 40% of the base: well past the crossover, squarely in the
        // regime where patching loses.
        for id in 0u64..2_400 {
            idx.delete(id).unwrap();
        }
        idx.set_auto_flush(false);
        let arc: VectorEntry = Arc::new(RwLock::new(Vindex::new(
            VectorBackend::Disk(Box::new(idx)),
            4,
        )));

        let patches_before =
            skeg_telemetry::counter_value(skeg_telemetry::Counter::MaintenanceDeletePatch);
        let folds_before =
            skeg_telemetry::counter_value(skeg_telemetry::Counter::MaintenanceConsolidate);
        // The budget can make a single tick skip; retry as the shard loop does.
        for _ in 0..50 {
            if maintenance_tick(&arc, &vdir, 0, true).await {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        assert_eq!(
            skeg_telemetry::counter_value(skeg_telemetry::Counter::MaintenanceDeletePatch)
                - patches_before,
            0,
            "delete-patch ran on a 40%-dead base, its measured losing regime"
        );
        assert!(
            skeg_telemetry::counter_value(skeg_telemetry::Counter::MaintenanceConsolidate)
                > folds_before,
            "nothing reclaimed the heavily-dead base"
        );
    }

    /// A reshard moves every vector to its semantic owner, point ops keep
    /// working through the id-owner map, and a reopen rebuilds the map.
    #[tokio::test]
    async fn reshard_moves_clusters_to_their_owners_and_ops_survive() {
        let dir = TempDir::new().unwrap();
        let open = || {
            ShardSet::open_mode_with_workers(
                dir.path(),
                2,
                false,
                skeg_vector::QuantKind::TurboQuant { bits: 2 },
                1,
            )
            .unwrap()
        };
        let shards = open();
        shards.vindex_create("rs", 8, 4, 1).await.unwrap();
        // Two orthogonal clusters, ids interleaved so hash placement mixes them.
        let vec_for = |id: u64| {
            let mut v = vec![0.05f32; 8];
            v[(id % 2) as usize] = 1.0;
            v
        };
        for id in 0..400u64 {
            shards
                .vset("rs", id, vec_for(id), 0, None, None)
                .await
                .unwrap();
        }
        let moved = shards.reshard("rs", 0.25, 10, 0).await.expect("reshard");
        assert!(moved > 0, "an interleaved layout must move rows");

        // Each shard now holds one cluster: per-shard counts are ~even and
        // ids of the same parity live together.
        let router = shards.router("rs").expect("router trained by reshard");
        let owner_even = router.assign(&vec_for(0));
        let owner_odd = router.assign(&vec_for(1));
        assert_ne!(owner_even, owner_odd, "clusters must have distinct owners");

        // Point ops after the move: get finds every id, delete works.
        for id in [0u64, 1, 199, 398, 399] {
            assert_eq!(
                shards.vget("rs", id).await.unwrap().expect("live id"),
                vec_for(id),
                "id {id} lost or corrupted by the move"
            );
        }
        assert!(shards.vdel("rs", 42, 0).await.unwrap());
        assert!(shards.vget("rs", 42).await.unwrap().is_none());
        // Overwrite an id with a vector of the OTHER cluster: it must follow
        // its semantics to the other shard and stay unique.
        shards
            .vset("rs", 7, vec_for(0), 0, None, None)
            .await
            .unwrap();
        assert_eq!(shards.vget("rs", 7).await.unwrap().unwrap(), vec_for(0));

        // Search still finds the right cluster.
        let hits = shards
            .vsearch("rs", vec_for(3), 5, 0, 0, false, None)
            .await
            .unwrap();
        assert!(
            hits.iter().all(|&(id, _, _)| id % 2 == 1 || id == 7),
            "odd-cluster query must return odd ids (or the re-homed 7)"
        );
        drop(shards);

        // Reopen: the id-owner map rebuilds and EVERY id resolves (a weak
        // spot-check here once passed on a 50% hash-fallback coincidence).
        let re = open();
        for id in 0..400u64 {
            if id == 42 {
                assert!(
                    re.vget("rs", id).await.unwrap().is_none(),
                    "delete survives"
                );
            } else if id == 7 {
                assert_eq!(re.vget("rs", 7).await.unwrap().unwrap(), vec_for(0));
            } else {
                assert_eq!(
                    re.vget("rs", id).await.unwrap().unwrap_or_default(),
                    vec_for(id),
                    "id {id} unreachable after reopen"
                );
            }
        }
    }

    /// Concurrent routed vset+vdel on the same id leave the map and the
    /// shards consistent: exactly one logical copy or none, never an
    /// untracked duplicate (the race the per-id stripe closes).
    #[tokio::test]
    async fn concurrent_routed_ops_on_one_id_stay_consistent() {
        let dir = TempDir::new().unwrap();
        let shards = std::sync::Arc::new(
            ShardSet::open_mode_with_workers(
                dir.path(),
                4,
                false,
                skeg_vector::QuantKind::TurboQuant { bits: 2 },
                1,
            )
            .unwrap(),
        );
        shards.vindex_create("rc", 8, 4, 1).await.unwrap();
        for id in 0..400u64 {
            let mut v = vec![0.05f32; 8];
            v[(id % 4) as usize] = 1.0;
            shards.vset("rc", id, v, 0, None, None).await.unwrap();
        }
        shards.reshard("rc", 0.25, 10, 0).await.unwrap();

        // Hammer id 7 with interleaved overwrite (to a different cluster) and
        // delete from many tasks; the stripe must serialise them.
        let mut set = tokio::task::JoinSet::new();
        for t in 0..40u64 {
            let s = shards.clone();
            set.spawn(async move {
                if t % 2 == 0 {
                    let mut v = vec![0.05f32; 8];
                    v[(t % 4) as usize] = 1.0;
                    let _ = s.vset("rc", 7, v, 0, None, None).await;
                } else {
                    let _ = s.vdel("rc", 7, 0).await;
                }
            });
        }
        while set.join_next().await.is_some() {}

        // Whatever the final state, id 7 has at most one live copy across all
        // shards, and the map agrees with the shards.
        let mut live = Vec::new();
        for shard in 0..4usize {
            if let ShardResp::Vector(Some(_)) = shards
                .call(
                    shard,
                    ShardReq::Vget {
                        name: "rc".into(),
                        id: 7,
                    },
                )
                .await
                .unwrap()
            {
                live.push(shard);
            }
        }
        assert!(
            live.len() <= 1,
            "id 7 has {} live copies: {live:?}",
            live.len()
        );
        // vget (map-routed) agrees with the shards.
        let via_map = shards.vget("rc", 7).await.unwrap().is_some();
        assert_eq!(
            via_map,
            !live.is_empty(),
            "map disagrees with shards on id 7"
        );
    }

    /// Targeted overlap: boundary rows (small margin between their two
    /// nearest centroids) get a replica on the second shard; probe-2 search
    /// then finds them from either side, deletes remove both copies, and an
    /// overwrite cannot resurrect a stale replica.
    #[tokio::test]
    async fn targeted_overlap_replicates_boundary_rows_and_ops_stay_exact() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open_mode_with_workers(
            dir.path(),
            2,
            false,
            skeg_vector::QuantKind::TurboQuant { bits: 2 },
            1,
        )
        .unwrap();
        shards.vindex_create("ov", 8, 4, 1).await.unwrap();
        // Two clusters plus BOUNDARY rows sitting between them.
        let cluster = |c: usize| {
            let mut v = vec![0.05f32; 8];
            v[c] = 1.0;
            v
        };
        let boundary = |seed: u64| {
            let mut v = vec![0.05f32; 8];
            v[0] = 0.72 + (seed as f32) * 1e-3;
            v[1] = 0.70;
            v
        };
        for id in 0..300u64 {
            shards
                .vset("ov", id, cluster((id % 2) as usize), 0, None, None)
                .await
                .unwrap();
        }
        for id in 300..340u64 {
            shards
                .vset("ov", id, boundary(id), 0, None, None)
                .await
                .unwrap();
        }
        shards.reshard("ov", 0.25, 10, 0).await.unwrap();
        // Tau comes from the trained router's own geometry: between the
        // boundary rows' margin and the cluster cores' margin, so the test
        // asserts SELECTIVITY instead of guessing a constant.
        let router = shards.router("ov").expect("router");
        let margin = |v: &[f32]| {
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            let qn: Vec<f32> = v.iter().map(|x| x / n).collect();
            let mut sims: Vec<f32> = (0..router.k)
                .map(|j| {
                    let c = &router.centroids[j * router.dim..(j + 1) * router.dim];
                    qn.iter().zip(c).map(|(a, b)| a * b).sum()
                })
                .collect();
            sims.sort_by(f32::total_cmp);
            sims[router.k - 1] - sims[router.k - 2]
        };
        let m_boundary = margin(&boundary(320));
        let m_core = margin(&cluster(0));
        assert!(
            m_boundary < m_core,
            "boundary rows must sit nearer the frontier ({m_boundary} vs {m_core})"
        );
        let tau = (m_boundary + m_core) / 2.0;
        let replicated = shards.overlap("ov", tau, 0).await.expect("overlap");
        assert!(
            (30..200).contains(&replicated),
            "overlap at tau {tau} must catch the ~40 boundary rows, not the cores              (replicated {replicated})"
        );

        // A probe-2 search from the boundary finds boundary ids.
        let hits = shards
            .vsearch_with_probe("ov", boundary(320), 10, 0, 0, false, None, 1)
            .await
            .unwrap();
        assert!(
            hits.iter().filter(|&&(id, _, _)| id >= 300).count() >= 5,
            "probe-1 from the boundary must see replicated boundary rows"
        );
        // No duplicate ids in results.
        let mut ids: Vec<u64> = hits.iter().map(|&(id, _, _)| id).collect();
        let n_before = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), n_before, "merge must dedup replicas");

        // Delete removes BOTH copies.
        assert!(shards.vdel("ov", 320, 0).await.unwrap());
        assert!(shards.vget("ov", 320).await.unwrap().is_none());
        let hits = shards
            .vsearch_with_probe("ov", boundary(320), 10, 0, 0, false, None, 0)
            .await
            .unwrap();
        assert!(
            hits.iter().all(|&(id, _, _)| id != 320),
            "a deleted id must not survive as a replica ghost"
        );
        // Overwrite lands one logical copy (replica of the old value gone).
        shards
            .vset("ov", 321, cluster(0), 0, None, None)
            .await
            .unwrap();
        assert_eq!(shards.vget("ov", 321).await.unwrap().unwrap(), cluster(0));
    }

    /// Routed probing: with a router, an unfiltered search asks only the
    /// top-P shards and returns the same answers as the full fan-out for
    /// in-cluster queries; probe=0 keeps the full fan-out.
    #[tokio::test]
    async fn routed_probe_matches_full_fanout_for_cluster_queries() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open_mode_with_workers(
            dir.path(),
            4,
            false,
            skeg_vector::QuantKind::TurboQuant { bits: 2 },
            1,
        )
        .unwrap();
        shards.vindex_create("pr", 16, 4, 1).await.unwrap();
        let vec_for = |id: u64| {
            let mut v = vec![0.02f32; 16];
            v[(id % 4) as usize] = 1.0;
            v[(4 + id % 4) as usize] = 0.5;
            v
        };
        for id in 0..800u64 {
            shards
                .vset("pr", id, vec_for(id), 0, None, None)
                .await
                .unwrap();
        }
        shards.reshard("pr", 0.25, 10, 0).await.unwrap();
        for probe_q in 0..4u64 {
            let q = vec_for(probe_q);
            let full = shards
                .vsearch_with_probe("pr", q.clone(), 10, 0, 0, false, None, 0)
                .await
                .unwrap();
            let probed = shards
                .vsearch_with_probe("pr", q, 10, 0, 0, false, None, 2)
                .await
                .unwrap();
            let ids = |v: &Vec<(u64, f32, Option<Bytes>)>| {
                let mut s: Vec<u64> = v.iter().map(|&(id, _, _)| id).collect();
                s.sort_unstable();
                s
            };
            assert_eq!(
                ids(&full),
                ids(&probed),
                "probe=2 must match full fan-out for cluster query {probe_q}"
            );
        }
    }

    /// An explicit CONSOLIDATE must leave the index in the SAME state the
    /// maintenance ladder would: router included. Without it a hand-consolidated
    /// index scans the entire match set on every filtered search.
    #[tokio::test]
    async fn explicit_consolidate_rebuilds_the_ivf_router() {
        let dir = TempDir::new().unwrap();
        // One shard, so the per-shard base clears the router's size floor.
        let shards = ShardSet::open_mode_with_workers(
            dir.path(),
            1,
            false,
            skeg_vector::QuantKind::TurboQuant { bits: 2 },
            1,
        )
        .unwrap();
        shards.vindex_create("iv", 8, 4, 1).await.unwrap();
        // wants_ivf() has a 50k floor; below it no router is wanted and the
        // exact scan is already the cheap answer. Assert the CONTRACT rather
        // than build 50k rows here: after a consolidate, an index that wants
        // a router has one.
        for id in 0..2000u64 {
            let mut v = vec![0.05f32; 8];
            v[(id % 4) as usize] = 1.0;
            shards.vset("iv", id, v, 0, None, None).await.unwrap();
        }
        shards.vindex_consolidate("iv").await.unwrap();
        let wants = match shards
            .call(0, ShardReq::WantsIvf { name: "iv".into() })
            .await
        {
            Ok(ShardResp::Count(n)) => n == 1,
            _ => false,
        };
        assert!(
            !wants,
            "after an explicit consolidate no index should still WANT a router"
        );
    }

    /// owners_of answers for a routed index from the map, and for an
    /// unrouted one from the id itself - so a benchmark can always report
    /// its placement, resharded or not.
    #[tokio::test]
    async fn owners_of_reports_placement_routed_and_unrouted() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open_mode_with_workers(
            dir.path(),
            4,
            false,
            skeg_vector::QuantKind::TurboQuant { bits: 2 },
            1,
        )
        .unwrap();
        shards.vindex_create("ow", 8, 4, 1).await.unwrap();
        for id in 0..400u64 {
            let mut v = vec![0.05f32; 8];
            v[(id % 4) as usize] = 1.0;
            shards.vset("ow", id, v, 0, None, None).await.unwrap();
        }
        let probe: Vec<u64> = (0..40).collect();

        // Unrouted: every id resolves, and to a shard in range.
        let before = shards.owners_of("ow", &probe).await.unwrap();
        assert_eq!(before.len(), probe.len());
        assert!(before.iter().all(|&(p, _)| (p as usize) < 4));

        shards.reshard("ow", 0.25, 10, 0).await.unwrap();
        let after = shards.owners_of("ow", &probe).await.unwrap();
        assert_eq!(after.len(), probe.len());
        assert!(
            after
                .iter()
                .all(|&(p, r)| (p as usize) < 4 && r.is_none_or(|s| (s as usize) < 4))
        );
        // The whole point: a semantic reshard MOVES rows, so the placement a
        // benchmark reports must change with it.
        assert_ne!(before, after, "reshard left every id on its hash shard");
    }

    /// HEALTH must never print OK over an index that is missing on some
    /// shards. A monitor reads `state` and nothing else, so an OK with a
    /// caveat below it is the false green this command exists to prevent.
    #[tokio::test]
    async fn health_state_is_partial_when_the_index_is_incomplete() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open_mode_with_workers(
            dir.path(),
            4,
            false,
            skeg_vector::QuantKind::TurboQuant { bits: 2 },
            1,
        )
        .unwrap();
        shards.vindex_create("hp", 8, 4, 1).await.unwrap();
        for id in 0..200u64 {
            let mut v = vec![0.05f32; 8];
            v[(id % 4) as usize] = 1.0;
            shards.vset("hp", id, v, 0, None, None).await.unwrap();
        }
        let ok = shards.health("hp").await.unwrap();
        assert!(
            ok.iter().any(|l| l == "state OK"),
            "a complete quiet index should read OK: {ok:?}"
        );

        // Drop the index on ONE shard only, as a partial failure would.
        shards
            .call(
                2,
                ShardReq::VindexDrop {
                    name: "hp".into(),
                    tenant: 0,
                    credit: DropCredit::Fragment,
                    require_present: true,
                },
            )
            .await
            .expect("drop on one shard");
        let partial = shards.health("hp").await.unwrap();
        assert!(
            partial.iter().any(|l| l == "state PARTIAL"),
            "an index missing on a shard must not read OK: {partial:?}"
        );
        assert!(
            partial.iter().any(|l| l.starts_with("present_on 3 of 4")),
            "the report must say where it is missing: {partial:?}"
        );
    }

    /// A healthy index checks clean through the coordinator, including the
    /// routed cross-checks; a missing index is not an error (nothing to
    /// disagree with) and an unknown one reports nothing rather than lying.
    #[tokio::test]
    async fn check_reports_clean_on_a_healthy_routed_index() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open_mode_with_workers(
            dir.path(),
            2,
            false,
            skeg_vector::QuantKind::TurboQuant { bits: 2 },
            1,
        )
        .unwrap();
        shards.vindex_create("ck", 8, 4, 1).await.unwrap();
        for id in 0..300u64 {
            let mut v = vec![0.05f32; 8];
            v[(id % 2) as usize] = 1.0;
            shards.vset("ck", id, v, 0, None, None).await.unwrap();
        }
        let clean = shards.check("ck").await.unwrap();
        assert!(clean.is_empty(), "healthy index reported: {clean:?}");

        // After a reshard the router cross-checks run too.
        shards.reshard("ck", 0.25, 10, 0).await.unwrap();
        let after = shards.check("ck").await.unwrap();
        assert!(after.is_empty(), "resharded index reported: {after:?}");

        // An unknown index is an error, not a clean bill of health.
        assert!(
            shards.check("nope").await.is_err(),
            "unknown index reported healthy"
        );
    }

    /// Dropping a resharded vindex removes its router sidecar, its in-RAM
    /// router and owner map; recreating the same name starts routing-free.
    #[tokio::test]
    async fn drop_clears_the_semantic_router() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open_mode_with_workers(
            dir.path(),
            2,
            false,
            skeg_vector::QuantKind::TurboQuant { bits: 2 },
            1,
        )
        .unwrap();
        shards.vindex_create("dr", 8, 4, 1).await.unwrap();
        for id in 0..200u64 {
            let mut v = vec![0.05f32; 8];
            v[(id % 2) as usize] = 1.0;
            shards.vset("dr", id, v, 0, None, None).await.unwrap();
        }
        shards.reshard("dr", 0.25, 10, 0).await.unwrap();
        assert!(shards.router("dr").is_some());
        let sidecar = crate::router::router_path(dir.path(), "dr");
        assert!(sidecar.exists(), "sidecar written");

        shards.vindex_drop("dr", 0).await.unwrap();
        assert!(
            shards.router("dr").is_none(),
            "router still loaded after drop"
        );
        assert!(!sidecar.exists(), "sidecar survived the drop");
        assert!(
            shards.inner.owners.read().get("dr").is_none(),
            "owner map survived the drop"
        );

        // Recreate the same name: no inherited routing.
        shards.vindex_create("dr", 8, 4, 1).await.unwrap();
        assert!(
            shards.router("dr").is_none(),
            "recreated index inherited a router"
        );
    }

    /// Training the router samples every shard, writes the sidecar, and a
    /// reopened shard set serves the same centroids (same epoch, bit-exact).
    #[tokio::test]
    async fn router_training_writes_a_sidecar_that_survives_reopen() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open_mode_with_workers(
            dir.path(),
            2,
            false,
            skeg_vector::QuantKind::TurboQuant { bits: 2 },
            1,
        )
        .unwrap();
        shards.vindex_create("rt", 8, 4, 1).await.unwrap();
        // Two clear clusters so the trained centroids are meaningful.
        for id in 0..200u64 {
            let mut v = vec![0.05f32; 8];
            v[(id % 2) as usize] = 1.0;
            shards.vset("rt", id, v, 0, None, None).await.unwrap();
        }
        let epoch = shards.train_router("rt", 0.25, 10).await.expect("training");
        assert!(epoch >= 1);
        let r1 = shards.router("rt").expect("router loaded after training");
        assert_eq!(r1.k, 2, "one centroid per shard");
        assert_eq!(r1.dim, 8);
        drop(shards);

        let re = ShardSet::open_mode_with_workers(
            dir.path(),
            2,
            false,
            skeg_vector::QuantKind::TurboQuant { bits: 2 },
            1,
        )
        .unwrap();
        let r2 = re.router("rt").expect("router reloaded at open");
        assert_eq!(r2.epoch, r1.epoch);
        assert_eq!(
            r2.centroids, r1.centroids,
            "centroids must reload bit-exact"
        );
        // The two natural clusters land on different owners.
        let mut a = vec![0.05f32; 8];
        a[0] = 1.0;
        let mut b = vec![0.05f32; 8];
        b[1] = 1.0;
        assert_ne!(
            r2.assign(&a),
            r2.assign(&b),
            "distinct clusters share an owner"
        );
    }

    /// VGET returns the stored vector for a live id across every location
    /// (delta, run, base), and nothing for a deleted or unknown id.
    #[tokio::test]
    async fn vget_returns_the_stored_vector_and_respects_deletes() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open_mode_with_workers(
            dir.path(),
            2,
            false,
            skeg_vector::QuantKind::TurboQuant { bits: 2 },
            1,
        )
        .unwrap();
        shards.vindex_create("vg", 8, 4, 1).await.unwrap();
        let v: Vec<f32> = (0..8).map(|i| i as f32 / 10.0).collect();
        shards
            .vset("vg", 7, v.clone(), 0, None, None)
            .await
            .unwrap();
        let got = shards.vget("vg", 7).await.unwrap().expect("id 7 stored");
        assert_eq!(got, v, "roundtrip must be bit-exact");
        assert!(
            shards.vget("vg", 8).await.unwrap().is_none(),
            "unknown id must be None"
        );
        shards.vdel("vg", 7, 0).await.unwrap();
        assert!(
            shards.vget("vg", 7).await.unwrap().is_none(),
            "deleted id must be None"
        );
    }

    #[tokio::test]
    async fn off_thread_maintenance_runs_merge_and_delete_patch() {
        let dir = TempDir::new().unwrap();
        let vdir = dir.path().join("vindex-t");
        let mut idx = DiskVamanaIndex::create_empty_with_tier(
            &vdir,
            64,
            64,
            QuantKind::TurboQuant { bits: 2 },
        )
        .unwrap();
        // A base of 5000 (one flush + consolidate), then 9000 more -> two runs.
        for id in 0u64..5000 {
            idx.insert(id, &tvec(id + 1)).unwrap();
        }
        idx.consolidate().unwrap().expect_clean();
        let base_before = idx.main_len();
        for id in 5000u64..14000 {
            idx.insert(id, &tvec(id + 1)).unwrap();
        }
        assert!(
            idx.run_count() >= 2,
            "need >=2 runs, got {}",
            idx.run_count()
        );

        let arc: VectorEntry = Arc::new(RwLock::new(Vindex::new(
            VectorBackend::Disk(Box::new(idx)),
            4,
        )));

        // L2: runs fold into one.
        let mut ran = false;
        for _ in 0..100 {
            let d = vdir.clone();
            if off_thread_maintenance(
                &arc,
                "runs-merge",
                0,
                |b| b.merge_runs_begin(),
                move |job| job.build(&d),
                |b, built| b.merge_runs_finish(built),
            )
            .await
                == MaintenanceOutcome::Ran
            {
                ran = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert!(ran, "runs-merge ran");
        assert_eq!(arc.read().backend.run_count(), 1, "runs folded to one");
        assert!(
            arc.read().backend.get(42).unwrap().is_some(),
            "base id live"
        );

        // Delete a fifth of the base, then L3: dead base rows reclaimed in place.
        for id in 0u64..1000 {
            assert!(
                arc.write()
                    .backend
                    .delete(id, VectorVersion::LEGACY)
                    .unwrap(),
                "deleting a live base id"
            );
        }
        assert_eq!(arc.read().backend.tombstone_count(), 1000);
        let mut ran = false;
        for _ in 0..100 {
            let d = vdir.clone();
            if off_thread_maintenance(
                &arc,
                "delete-patch",
                0,
                |b| b.delete_patch_begin(),
                move |job| job.build(&d),
                |b, built| b.delete_patch_finish(built),
            )
            .await
                == MaintenanceOutcome::Ran
            {
                ran = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert!(ran, "delete-patch ran");
        assert_eq!(
            arc.read().backend.main_len(),
            base_before - 1000,
            "base shrank by the deletes"
        );
        assert!(
            arc.read().backend.get(0).unwrap().is_none(),
            "deleted id stays gone"
        );
        assert!(
            arc.read().backend.get(4999).unwrap().is_some(),
            "surviving base id live"
        );
        assert!(
            arc.read().backend.get(9000).unwrap().is_some(),
            "a run id still live"
        );
    }

    // STRESS: sustained retract churn (insert successor + delete predecessor)
    // through the real ShardSet, with the background maintenance loop cranked
    // fast (SKEG_IDLE_MAINT_MS). Measures query latency throughout and checks
    // the index stays correct. Run: SKEG_IDLE_MAINT_MS=50 cargo test --release
    // -p skeg-server -- --ignored stress_churn_maintenance --nocapture
    #[tokio::test]
    #[ignore = "stress; run with SKEG_IDLE_MAINT_MS=50 in release"]
    #[allow(clippy::explicit_counter_loop)] // `next` is an id generator, not an index
    async fn stress_churn_maintenance() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 1).unwrap();
        shards.vindex_create("c", 64, 4, 1).await.unwrap(); // tq2 disk
        let env_usize = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(d)
        };
        let seed = env_usize("SKEG_STRESS_N", 6000) as u64;
        for id in 0..seed {
            shards
                .vset("c", id, tvec(id + 1), 0, None, None)
                .await
                .unwrap();
        }
        let mut ids: Vec<u64> = (0..seed).collect();
        let mut next = seed;
        let mut rng = 0x1234_5678u64;
        let mut lat: Vec<f64> = Vec::new();
        let ops = env_usize("SKEG_STRESS_OPS", 20_000);
        let start = std::time::Instant::now();
        for step in 0..ops {
            shards
                .vset("c", next, tvec(next + 1), 0, None, None)
                .await
                .unwrap();
            let succ = next;
            next += 1;
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let j = (rng as usize) % ids.len();
            shards.vdel("c", ids[j], 0).await.unwrap();
            ids[j] = succ;
            if step % 100 == 0 {
                let live = ids[(rng as usize) % ids.len()];
                let t = std::time::Instant::now();
                let _ = shards
                    .vsearch("c", tvec(live + 1), 10, 0, 0, false, None)
                    .await
                    .unwrap();
                lat.push(t.elapsed().as_secs_f64() * 1e3);
            }
        }
        let elapsed = start.elapsed().as_secs_f64();
        lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = lat[lat.len() / 2];
        let p99 = lat[lat.len() * 99 / 100];
        let max = *lat.last().unwrap();
        println!(
            "churn {ops} retr/{elapsed:.1}s = {:.0}/s | query p50 {p50:.1}ms p99 {p99:.1}ms max {max:.1}ms ({} samples)",
            ops as f64 / elapsed,
            lat.len()
        );
        // Correctness after churn: a live id is retrievable as its own neighbour.
        let live = ids[0];
        let hits = shards
            .vsearch("c", tvec(live + 1), 5, 0, 0, false, None)
            .await
            .unwrap();
        assert!(
            hits.iter().any(|h| h.0 == live),
            "live id retrievable after churn"
        );
    }

    // STRESS (concurrent): one task churns (vset+vdel) while ANOTHER hammers
    // queries and times them. The shard is single-threaded, so an inline
    // ingest-path fold (synchronous) blocks queued queries - this is the test
    // that actually catches a stall (the sequential stress cannot). Run:
    // SKEG_IDLE_MAINT_MS=50 SKEG_STRESS_N=50000 cargo test --release -p
    // skeg-server -- --ignored stress_concurrent_query --nocapture
    #[tokio::test]
    #[ignore = "stress; run in release with SKEG_STRESS_N set"]
    #[allow(clippy::explicit_counter_loop)] // `next` is an id generator, not an index
    async fn stress_concurrent_query_during_churn() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let dir = TempDir::new().unwrap();
        let shards = std::sync::Arc::new(ShardSet::open(dir.path(), 1).unwrap());
        shards.vindex_create("c", 64, 4, 1).await.unwrap(); // tq2 disk
        let env_usize = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(d)
        };
        let seed = env_usize("SKEG_STRESS_N", 50_000) as u64;
        let ops = env_usize("SKEG_STRESS_OPS", 120_000);
        for id in 0..seed {
            shards
                .vset("c", id, tvec(id + 1), 0, None, None)
                .await
                .unwrap();
        }
        let done = std::sync::Arc::new(AtomicBool::new(false));

        // Query task: continuous, times each search (a fixed query vector).
        let q_shards = shards.clone();
        let q_done = done.clone();
        let query = async move {
            let mut lat: Vec<f64> = Vec::new();
            while !q_done.load(Ordering::Relaxed) {
                let t = std::time::Instant::now();
                let _ = q_shards
                    .vsearch("c", tvec(7), 10, 0, 0, false, None)
                    .await
                    .unwrap();
                lat.push(t.elapsed().as_secs_f64() * 1e3);
            }
            lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
            lat
        };

        // Churn task: sustained retract.
        let c_shards = shards.clone();
        let c_done = done.clone();
        let churn = async move {
            let mut ids: Vec<u64> = (0..seed).collect();
            let mut next = seed;
            let mut rng = 0x1234_5678u64;
            let start = std::time::Instant::now();
            for _ in 0..ops {
                c_shards
                    .vset("c", next, tvec(next + 1), 0, None, None)
                    .await
                    .unwrap();
                let succ = next;
                next += 1;
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let j = (rng as usize) % ids.len();
                c_shards.vdel("c", ids[j], 0).await.unwrap();
                ids[j] = succ;
            }
            let e = start.elapsed().as_secs_f64();
            c_done.store(true, Ordering::Relaxed);
            let known_live = ids[0];
            (ops as f64 / e, e, known_live)
        };

        let ((rps, secs, known_live), lat) = tokio::join!(churn, query);
        let hits = shards
            .vsearch("c", tvec(known_live + 1), 5, 0, 0, false, None)
            .await
            .unwrap();
        assert!(
            hits.iter().any(|hit| hit.0 == known_live),
            "a live id stays retrievable after concurrent churn"
        );
        let p50 = lat[lat.len() / 2];
        let p99 = lat[lat.len() * 99 / 100];
        let max = *lat.last().unwrap();
        println!(
            "CONCURRENT | churn {rps:.0} retr/s over {secs:.1}s | {} queries: p50 {p50:.1}ms p99 {p99:.1}ms MAX {max:.1}ms",
            lat.len()
        );
    }

    // Find a hit's payload by id in a WITHPAYLOAD result.
    fn payload_of(hits: &[(u64, f32, Option<Bytes>)], id: u64) -> Option<&[u8]> {
        hits.iter().find(|h| h.0 == id).and_then(|h| h.2.as_deref())
    }

    // A payload stored with VSET comes back byte-identical with a WITHPAYLOAD
    // search, for empty, binary-with-NUL, and large (>4KB) blobs. Without
    // WITHPAYLOAD no payload is attached.
    #[tokio::test]
    async fn test_payload_round_trip() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 2).unwrap();
        shards.vindex_create("idx", 64, 0, 0).await.unwrap(); // flat

        let empty: Vec<u8> = Vec::new();
        let binary: Vec<u8> = vec![0u8, 1, 2, 0, 255, 0, 7];
        let large: Vec<u8> = (0..5000u32).map(|i| (i % 251) as u8).collect();
        for (id, blob) in [(1u64, &empty), (2, &binary), (3, &large)] {
            shards
                .vset(
                    "idx",
                    id,
                    tvec(id + 1),
                    0,
                    None,
                    Some(Bytes::from(blob.clone())),
                )
                .await
                .unwrap();
        }

        // WITHPAYLOAD: each id carries its exact blob.
        let hits = shards
            .vsearch("idx", tvec(2), 10, 0, 0, true, None)
            .await
            .unwrap();
        assert_eq!(payload_of(&hits, 1), Some(&empty[..]));
        assert_eq!(payload_of(&hits, 2), Some(&binary[..]));
        assert_eq!(payload_of(&hits, 3), Some(&large[..]));

        // Without the flag: no payload attached at all.
        let plain = shards
            .vsearch("idx", tvec(2), 10, 0, 0, false, None)
            .await
            .unwrap();
        assert!(plain.iter().all(|h| h.2.is_none()));
    }

    // Payloads survive a restart (disk index + vLog replay), and a VDEL before
    // the restart removes the blob for good.
    #[tokio::test]
    async fn test_payload_survives_restart() {
        let dir = TempDir::new().unwrap();
        let base = dir.path().to_owned();
        {
            let shards = ShardSet::open(&base, 2).unwrap();
            shards.vindex_create("persist", 64, 0, 1).await.unwrap(); // disk
            shards
                .vset(
                    "persist",
                    10,
                    tvec(11),
                    0,
                    None,
                    Some(Bytes::from_static(b"keep")),
                )
                .await
                .unwrap();
            shards
                .vset(
                    "persist",
                    20,
                    tvec(21),
                    0,
                    None,
                    Some(Bytes::from_static(b"gone")),
                )
                .await
                .unwrap();
            assert!(shards.vdel("persist", 20, 0).await.unwrap());
        }
        let shards = ShardSet::open(&base, 2).unwrap();
        let hits = shards
            .vsearch("persist", tvec(11), 10, 0, 0, true, None)
            .await
            .unwrap();
        assert_eq!(payload_of(&hits, 10), Some(&b"keep"[..]));
        // id 20 was deleted before restart: no hit, hence no payload.
        assert!(hits.iter().all(|h| h.0 != 20));
    }

    // A payload is scoped to its tenant. The vector index is shared at this
    // layer, so tenant 9 still sees id 1, but reads no payload for it.
    #[tokio::test]
    async fn test_payload_tenant_isolation() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 2).unwrap();
        shards.vindex_create("idx", 64, 0, 0).await.unwrap();
        shards
            .vset(
                "idx",
                1,
                tvec(2),
                7,
                None,
                Some(Bytes::from_static(b"tenant-7-secret")),
            )
            .await
            .unwrap();

        let as7 = shards
            .vsearch("idx", tvec(2), 5, 0, 7, true, None)
            .await
            .unwrap();
        assert_eq!(payload_of(&as7, 1), Some(&b"tenant-7-secret"[..]));

        let as9 = shards
            .vsearch("idx", tvec(2), 5, 0, 9, true, None)
            .await
            .unwrap();
        assert!(
            as9.iter().any(|h| h.0 == 1),
            "vector is shared at this layer"
        );
        assert_eq!(
            payload_of(&as9, 1),
            None,
            "tenant 9 must not read tenant 7's payload"
        );
    }

    // Dropping an index reclaims its payload blobs, so a recreated index
    // reusing the same name and id does not resurface a stale payload.
    #[tokio::test]
    async fn test_payload_dropped_with_index() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 2).unwrap();
        shards.vindex_create("idx", 64, 0, 0).await.unwrap();
        shards
            .vset(
                "idx",
                1,
                tvec(2),
                0,
                None,
                Some(Bytes::from_static(b"stale")),
            )
            .await
            .unwrap();
        shards.vindex_drop("idx", 0).await.unwrap();

        // Recreate and re-insert the same id with NO payload.
        shards.vindex_create("idx", 64, 0, 0).await.unwrap();
        shards.vset("idx", 1, tvec(2), 0, None, None).await.unwrap();
        let hits = shards
            .vsearch("idx", tvec(2), 5, 0, 0, true, None)
            .await
            .unwrap();
        assert_eq!(
            payload_of(&hits, 1),
            None,
            "dropped payload must not resurface"
        );
    }

    fn flt(s: &str) -> Option<crate::payload::Filter> {
        Some(crate::payload::parse_filter(s).unwrap())
    }

    fn ids_of(hits: &[(u64, f32, Option<Bytes>)]) -> BTreeSet<u64> {
        hits.iter().map(|h| h.0).collect()
    }

    // A FILTER returns the exact nearest among only the matching ids, not the
    // global nearest minus non-matches. id 1 is the global nearest (the query is
    // its own vector) but belongs to `bob`; a `user = alice` filter must return
    // alice's vectors and exclude id 1 entirely.
    #[tokio::test]
    async fn test_filter_exact_over_subset() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 2).unwrap();
        shards.vindex_create("idx", 64, 0, 0).await.unwrap();
        let set = |id: u64, who: &'static [u8]| {
            let s = &shards;
            async move {
                s.vset("idx", id, tvec(id), 0, None, Some(Bytes::from_static(who)))
                    .await
                    .unwrap();
            }
        };
        set(1, b"user=bob").await;
        set(2, b"user=alice type=doc").await;
        set(3, b"user=alice type=img").await;

        // Query == id 1's vector, so id 1 is the global nearest.
        let global = shards
            .vsearch("idx", tvec(1), 10, 0, 0, false, None)
            .await
            .unwrap();
        assert_eq!(
            ids_of(&global).iter().next(),
            Some(&1),
            "id 1 is global nearest"
        );

        // user = alice excludes id 1 (bob) and returns alice's two, exactly.
        let alice = shards
            .vsearch("idx", tvec(1), 10, 0, 0, false, flt("user = alice"))
            .await
            .unwrap();
        assert_eq!(ids_of(&alice), BTreeSet::from([2, 3]));

        // AND narrows further; an empty match yields zero hits.
        let one = shards
            .vsearch(
                "idx",
                tvec(1),
                10,
                0,
                0,
                false,
                flt("user = alice AND type = doc"),
            )
            .await
            .unwrap();
        assert_eq!(ids_of(&one), BTreeSet::from([2]));
        let none = shards
            .vsearch("idx", tvec(1), 10, 0, 0, false, flt("user = nobody"))
            .await
            .unwrap();
        assert!(none.is_empty());
    }

    // VMSET bulk-inserts every item (vectors searchable) and indexes each
    // supplied payload (filtered search sees it), same as a sequence of VSETs.
    #[tokio::test]
    async fn vmset_bulk_inserts_and_indexes() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 2).unwrap();
        shards.vindex_create("idx", 64, 0, 1).await.unwrap(); // f32, disk
        let items = vec![
            (1u64, tvec(1), Some(Bytes::from_static(b"user=bob"))),
            (2u64, tvec(2), Some(Bytes::from_static(b"user=alice"))),
            (3u64, tvec(3), None),
        ];
        let n = all_ok(shards.vmset("idx", items, 0, None).await);
        assert_eq!(n, 3, "all three items inserted");

        // Every vector is searchable.
        let got = shards
            .vsearch("idx", tvec(1), 10, 0, 0, false, None)
            .await
            .unwrap();
        assert!(ids_of(&got).contains(&1), "id 1 searchable after VMSET");

        // Supplied payloads are indexed: a filter selects exactly the match.
        let alice = shards
            .vsearch("idx", tvec(2), 10, 0, 0, false, flt("user = alice"))
            .await
            .unwrap();
        assert_eq!(
            ids_of(&alice),
            BTreeSet::from([2]),
            "VMSET payload is filterable"
        );
    }

    // Completeness: after a large concurrent VMSET, EVERY item's payload is
    // indexed - each even id finds itself under the matching filter. Guards the
    // ~10%-indexed bug seen when the bench queried mid-population.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn vmset_indexes_all_payloads() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 2).unwrap();
        shards.vindex_create("idx", 64, 0, 1).await.unwrap(); // f32, disk
        let n: u64 = 4000;
        let items: Vec<_> = (0..n)
            .map(|id| {
                let pl = if id % 2 == 0 {
                    b"p=yes".as_slice()
                } else {
                    b"p=no".as_slice()
                };
                (id, tvec(id), Some(Bytes::copy_from_slice(pl)))
            })
            .collect();
        let cnt = all_ok(shards.vmset("idx", items, 0, None).await);
        assert_eq!(cnt, n as usize, "all items inserted");

        // Every even id must find ITSELF (its exact vector) under `p = yes`.
        let probes: Vec<u64> = (0..n).step_by(40).collect(); // all even
        let mut found = 0;
        for &id in &probes {
            let got = shards
                .vsearch("idx", tvec(id), 5, 0, 0, false, flt("p = yes"))
                .await
                .unwrap();
            if ids_of(&got).contains(&id) {
                found += 1;
            }
        }
        assert_eq!(found, probes.len(), "every even id is indexed+self-matches");
    }

    // Same, but across SEVERAL geometric consolidates during the load (batched
    // VMSET, like the bench), to catch a payload/consolidate or Relaxed-blob
    // interaction that drops index entries.
    //
    // What this needs is the folds, not the row count. The old version got
    // them by WAITING: it wrote 20k rows at 64 dims and took eighty seconds,
    // long enough for the background maintenance loop to fire a few times -
    // which is why it cost eighty seconds, why it lived behind `--ignored`,
    // and why it guarded nothing. The folds are driven explicitly here, so
    // they are certain instead of merely likely, and the test is fast.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn vmset_indexes_all_payloads_at_scale() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 2).unwrap();
        shards.vindex_create("idx", 16, 0, 1).await.unwrap(); // f32, disk
        let n: u64 = 12_000;
        let mut id = 0u64;
        let mut folds = 0u32;
        while id < n {
            let end = (id + 256).min(n);
            let items: Vec<_> = (id..end)
                .map(|i| {
                    let pl = if i % 2 == 0 {
                        b"p=yes".as_slice()
                    } else {
                        b"p=no".as_slice()
                    };
                    (i, tvec16(i), Some(Bytes::copy_from_slice(pl)))
                })
                .collect();
            all_ok(shards.vmset("idx", items, 0, None).await);
            id = end;
            // Fold mid-load, several times, with writes still arriving after
            // each one - that ordering is the whole point.
            if id >= u64::from(folds + 1) * 3_000 {
                shards.vindex_consolidate("idx").await.unwrap();
                folds += 1;
            }
        }
        assert!(
            folds >= 3,
            "only {folds} folds: the interaction is untested"
        );
        // The premise: this load really did fold several times. Without it a
        // future shrink would leave a test that never reaches the path it
        // exists to cover, and still passes.
        let based: u64 = shards
            .vindex_list()
            .await
            .unwrap()
            .iter()
            .filter(|r| r.name == "idx")
            .map(|r| r.base)
            .sum();
        assert!(
            based >= n / 2,
            "the folds must reach the base, only {based} of {n} rows there"
        );

        let probes: Vec<u64> = (0..n).step_by(400).collect(); // all even
        let mut found = 0;
        for &id in &probes {
            let got = shards
                .vsearch("idx", tvec16(id), 5, 0, 0, false, flt("p = yes"))
                .await
                .unwrap();
            if ids_of(&got).contains(&id) {
                found += 1;
            }
        }
        assert_eq!(
            found,
            probes.len(),
            "every even id indexed after consolidates"
        );
    }

    // Filtered search stays exactly correct across sustained churn + periodic
    // consolidation - the maintenance path this branch refactored (cheap begin,
    // prebuilt-segment finish). After each consolidate a filtered query must
    // return EXACTLY the live matching set: no dead id, no wrong-group leak,
    // none missing.
    #[tokio::test]
    #[allow(clippy::explicit_counter_loop)] // `next` is an id generator, not an index
    async fn filtered_search_correct_under_churn_and_consolidate() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 1).unwrap();
        shards.vindex_create("idx", 64, 4, 1).await.unwrap(); // tq2 disk
        let grp = |id: u64| id % 8;
        let put = |id: u64| {
            let s = &shards;
            async move {
                let g = grp(id);
                s.vset(
                    "idx",
                    id,
                    tvec(id + 1),
                    0,
                    None,
                    Some(Bytes::from(format!("g={g}"))),
                )
                .await
                .unwrap();
            }
        };
        let seed = 2000u64;
        let mut live: std::collections::BTreeMap<u64, u64> = std::collections::BTreeMap::new();
        for id in 0..seed {
            put(id).await;
            live.insert(id, grp(id));
        }
        shards.vindex_consolidate("idx").await.unwrap();

        let mut next = seed;
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        for step in 0..3000u64 {
            put(next).await;
            live.insert(next, grp(next));
            next += 1;
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let keys: Vec<u64> = live.keys().copied().collect();
            let victim = keys[(rng as usize) % keys.len()];
            shards.vdel("idx", victim, 0).await.unwrap();
            live.remove(&victim);

            if step % 500 == 499 {
                // Fold delta + runs into base (exercises the refactored path).
                shards.vindex_consolidate("idx").await.unwrap();
                let want: BTreeSet<u64> = live
                    .iter()
                    .filter(|&(_, &g)| g == 3)
                    .map(|(&id, _)| id)
                    .collect();
                // k covers every match, so the exact-scan filter returns all of them.
                let got = shards
                    .vsearch("idx", tvec(3), want.len() + 50, 0, 0, false, flt("g = 3"))
                    .await
                    .unwrap();
                assert_eq!(
                    ids_of(&got),
                    want,
                    "filtered g=3 must equal the live g=3 set after churn+consolidate (step {step})"
                );
            }
        }
    }

    // VDEL drops an id from the payload index, so a later filtered search no
    // longer returns it; an overwrite VSET replaces the id's fields.
    #[tokio::test]
    async fn test_filter_index_lifecycle() {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 2).unwrap();
        shards.vindex_create("idx", 64, 0, 0).await.unwrap();
        shards
            .vset("idx", 1, tvec(1), 0, None, Some(Bytes::from_static(b"k=a")))
            .await
            .unwrap();
        shards
            .vset("idx", 2, tvec(2), 0, None, Some(Bytes::from_static(b"k=a")))
            .await
            .unwrap();

        // Overwrite id 2 to k=b: it leaves the k=a set.
        shards
            .vset("idx", 2, tvec(2), 0, None, Some(Bytes::from_static(b"k=b")))
            .await
            .unwrap();
        let a = shards
            .vsearch("idx", tvec(1), 10, 0, 0, false, flt("k = a"))
            .await
            .unwrap();
        assert_eq!(ids_of(&a), BTreeSet::from([1]));

        // VDEL id 1: the k=a set is now empty.
        assert!(shards.vdel("idx", 1, 0).await.unwrap());
        let a2 = shards
            .vsearch("idx", tvec(1), 10, 0, 0, false, flt("k = a"))
            .await
            .unwrap();
        assert!(a2.is_empty());
    }

    // After a restart the payload index is rebuilt from the stored blobs on the
    // first filtered search, so a FILTER returns the same ids as before.
    #[tokio::test]
    async fn test_filter_index_rebuilt_after_restart() {
        let dir = TempDir::new().unwrap();
        let base = dir.path().to_owned();
        {
            let shards = ShardSet::open(&base, 2).unwrap();
            shards.vindex_create("persist", 64, 0, 1).await.unwrap(); // disk
            for id in 1u64..=4 {
                let who: &[u8] = if id % 2 == 0 {
                    b"user=alice"
                } else {
                    b"user=bob"
                };
                shards
                    .vset(
                        "persist",
                        id,
                        tvec(id),
                        0,
                        None,
                        Some(Bytes::from(who.to_vec())),
                    )
                    .await
                    .unwrap();
            }
        }
        // Restart: the payload index starts empty and is rebuilt on first use.
        let shards = ShardSet::open(&base, 2).unwrap();
        let alice = shards
            .vsearch("persist", tvec(2), 10, 0, 0, false, flt("user = alice"))
            .await
            .unwrap();
        assert_eq!(ids_of(&alice), BTreeSet::from([2, 4]));
    }

    /// Two VSEARCH callers hitting **different** vindexes on the same
    /// shard must not serialize against each other.
    ///
    /// We measure two regimes back-to-back on one shard with two dedicated
    /// VSEARCH workers:
    /// - **baseline**  : both tasks search the same vindex (serialized
    ///   by the per-vindex write lock, intentionally).
    /// - **concurrent**: each task searches its own vindex (per-vindex
    ///   write locks are disjoint, so both can hold their lock at the
    ///   same time in the worker pool).
    ///
    /// SoL gate: `baseline / concurrent >= 1.2×`. The theoretical
    /// ceiling is 2.0× (perfect parallelism on two cores); a floor of
    /// 1.2× is enough to distinguish "the lock refactor parallelised
    /// the work" (always above the floor in practice) from "the
    /// searches still serialise" (a 1.0× or sub-1.0× ratio, which
    /// would have been the result on the old single-`RwLock`
    /// `VindexSet`). The gap below 2.0 absorbs worker scheduling,
    /// allocator noise from interleaved tests, and CI
    /// runners with fewer real cores than the developer M1.
    ///
    /// Measured locally on M1: 1.5×–2.0× depending on warm-up and
    /// concurrent system load; never below 1.4×.
    // Wall-clock ratio gate: inherently flaky on shared / low-core CI runners
    // (a contended ubuntu runner can't deliver the 1.2x parallel speedup even
    // though the locks parallelise correctly). Ignored by default like the other
    // perf gates; run locally with `--ignored`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore = "wall-clock perf gate, flaky on shared CI; run locally with --ignored"]
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
            shards
                .vset("a", id, v.clone(), 0, None, None)
                .await
                .unwrap();
            shards.vset("b", id, v, 0, None, None).await.unwrap();
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
                let _ = s1
                    .vsearch("a", q1.clone(), 10, 0, 0, false, None)
                    .await
                    .unwrap();
            }
        });
        let h2 = tokio::spawn(async move {
            for _ in 0..iters {
                let _ = s2
                    .vsearch("a", q2.clone(), 10, 0, 0, false, None)
                    .await
                    .unwrap();
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
                let _ = s1
                    .vsearch("a", q1.clone(), 10, 0, 0, false, None)
                    .await
                    .unwrap();
            }
        });
        let h2 = tokio::spawn(async move {
            for _ in 0..iters {
                let _ = s2
                    .vsearch("b", q2.clone(), 10, 0, 0, false, None)
                    .await
                    .unwrap();
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

/// The most `SKEG.VMSET` item writes that run at once inside one call.
///
/// The fan-out is a per-REQUEST cost, and it is the one the ingress budget
/// does not charge: the budget covers the socket buffers a connection holds,
/// not the tree a single request expands into while it is being served. One
/// maximum VMSET is 4096 items, so an unbounded fan-out was 4096 concurrent
/// tasks per connection - about 98,000 across 24 connections, and 4.2 million
/// at the default connection limit. Measured at 24 connections it cost
/// +284.9 MiB of resident memory, roughly 12 MiB per connection, none of it
/// visible to any budget.
///
/// Bounding it does not undo the reason the fan-out exists. The point of VMSET
/// is that per-vector blob writes accumulate in the group committer and flush
/// in batches instead of one barrier per vector; sixty-four writers is still a
/// batch, and the throughput measurement in the CHANGELOG says by how much.
pub const VMSET_INFLIGHT: usize = 64;

/// Item writes running inside [`ShardSet::vmset`], and the most that ever ran
/// at once.
///
/// Instrumentation, not accounting. The fan-out is memory no budget charges,
/// so the only way a test can say "at most N at a time" is to watch it happen;
/// a test that checked the constant instead would pass against code that
/// ignored it. Present only where failpoints are, and the guard is a
/// zero-sized nothing otherwise.
#[cfg(any(test, feature = "failpoints"))]
pub mod vmset_inflight {
    use std::sync::atomic::{AtomicUsize, Ordering};

    static RUNNING: AtomicUsize = AtomicUsize::new(0);
    static PEAK: AtomicUsize = AtomicUsize::new(0);

    /// Held for the life of one item write.
    pub(crate) struct Guard;

    impl Guard {
        pub(crate) fn enter() -> Self {
            let now = RUNNING.fetch_add(1, Ordering::AcqRel) + 1;
            PEAK.fetch_max(now, Ordering::AcqRel);
            Self
        }
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            RUNNING.fetch_sub(1, Ordering::AcqRel);
        }
    }

    /// The most item writes that ran at once since [`reset`].
    ///
    /// Process-wide, so a test that reads it has to be the only VMSET in its
    /// process - which an integration test target is, each one being its own
    /// binary.
    #[must_use]
    pub fn peak() -> usize {
        PEAK.load(Ordering::Acquire)
    }

    /// Start a new measurement.
    pub fn reset() {
        PEAK.store(0, Ordering::Release);
    }
}

/// The same guard where the instrumentation is compiled out: nothing at all.
#[cfg(not(any(test, feature = "failpoints")))]
pub mod vmset_inflight {
    pub(crate) struct Guard;

    impl Guard {
        pub(crate) fn enter() -> Self {
            Self
        }
    }
}
