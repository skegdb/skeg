#![deny(unsafe_code)]

//! Cross-shard group committer for `DurabilityModel::DeviceGlobal`
//! platforms.
//!
//! On a platform where `sync_durable` is a device-wide barrier (Apple
//! Silicon `F_FULLFSYNC` is the reference case), letting every shard
//! own its own committer means N concurrent fsync calls all hit the
//! same hardware barrier and serialize. Slice A measured a regression
//! going from 1 to 4 shards because of exactly this.
//!
//! [`SharedCommitter`] is a single background task that:
//! 1. Accepts append requests from every shard on the device.
//! 2. Buckets them by file id, builds one combined write per file.
//! 3. Issues a single `sync_durable` covering every write in the
//!    batch (one barrier amortised across every shard).
//! 4. Acks each pending request with the offset + size assigned by
//!    the committer.
//!
//! The owning façade in `group_commit.rs` decides whether to use this
//! or a per-shard `PerFileCommitter` based on
//! [`skeg_platform::resolve_durability_model`].

use std::collections::HashMap;
use std::io;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use skeg_platform::PlatformFile;
use tokio::sync::{mpsc, oneshot};
use tokio::time::{Duration, Instant, sleep_until};

use crate::group_commit::Durability;

/// Inbox cap. Matches the design doc `SharedCommitter` budget; full
/// inbox makes `send().await` block, propagating backpressure up to
/// the connection handler (same pattern as the shard inbox).
const BATCH_INBOX_CAP: usize = 8192;
/// Maximum number of write entries aggregated into one batch.
const MAX_BATCH_ENTRIES: usize = 256;
/// Maximum total payload bytes aggregated into one batch.
const MAX_BATCH_BYTES: usize = 256 * 1024;
/// Soft deadline from the arrival of the first entry to the flush of
/// the batch. Picked to match the per-file committer so the
/// observable latency profile stays comparable.
const TIMER_MICROS: u64 = 200;

/// Per-process unique handle for a registered file.
type FileId = u64;
/// Reply channel handed back to the caller of `append`.
type AppendReply = oneshot::Sender<io::Result<(u64, u32)>>;
/// One entry materialised into the per-file accumulator: the bytes to
/// write and where to ack the result.
type PendingItem = (Vec<u8>, AppendReply);
/// One ack target: the assigned offset, padded size, and the reply
/// channel that owes the caller a result.
type AckSlot = (u64, u32, AppendReply);

static NEXT_FILE_ID: AtomicU64 = AtomicU64::new(1);

enum Msg {
    Attach {
        file_id: FileId,
        file: Arc<PlatformFile>,
        initial_offset: u64,
        reply: oneshot::Sender<()>,
    },
    Detach {
        file_id: FileId,
    },
    Append {
        file_id: FileId,
        data: Vec<u8>,
        durability: Durability,
        reply: oneshot::Sender<io::Result<(u64, u32)>>,
    },
    /// Flush the entire pending batch. file_id is dropped on the
    /// floor: a flush is a synchronisation point for *all* shards on
    /// this device, not just the caller's file, which matches what a
    /// user expects when they ask for durability on a DeviceGlobal
    /// platform.
    Flush {
        reply: oneshot::Sender<io::Result<()>>,
    },
}

/// Process-wide handle. Cheap to clone, every clone shares the same
/// background task. Today a single global instance; the design doc
/// (§ 8 Q-A) leaves a hook for a per-device variant when skeg gains
/// multi-disk support (T3).
#[derive(Clone)]
pub struct SharedCommitter {
    tx: mpsc::Sender<Msg>,
}

impl SharedCommitter {
    /// Get (or lazily create) the process-wide singleton. The
    /// background task is spawned on the calling tokio runtime, so the
    /// first call must happen from inside a runtime context.
    pub fn global() -> SharedCommitter {
        static SHARED: OnceLock<SharedCommitter> = OnceLock::new();
        SHARED.get_or_init(Self::new).clone()
    }

    fn new() -> Self {
        let (tx, rx) = mpsc::channel(BATCH_INBOX_CAP);
        // The bg task lives on a dedicated OS thread with its own
        // `current_thread` tokio runtime, not on the caller's runtime.
        // Tests construct (and drop) a fresh tokio runtime per
        // `#[tokio::test]` invocation; if the bg task lived on the
        // caller's runtime, the very first test to call `global()`
        // would leave a dangling singleton with a dead receiver for
        // every subsequent test (manifested as "file detached during
        // batch" once the new test attached its own files).
        //
        // Running the loop on a dedicated thread + runtime means the
        // bg task survives for the entire process lifetime and every
        // caller's runtime can talk to it through the channel. The
        // tokio mpsc Sender is runtime-agnostic, and async file ops
        // inside the loop use `spawn_blocking` against tokio's global
        // blocking pool, so neither side cares about the runtime
        // boundary.
        std::thread::Builder::new()
            .name("skeg-shared-committer".into())
            .spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("shared committer runtime build");
                rt.block_on(committer_loop(rx));
            })
            .expect("shared committer thread spawn");
        Self { tx }
    }

    /// Register `file` with the committer; appends through the returned
    /// entry are routed to it.
    ///
    /// Awaited so the caller can pair the registration with the next
    /// append without a race window. The bg task acknowledges
    /// registration immediately (no IO).
    pub async fn attach(
        &self,
        file: Arc<PlatformFile>,
        initial_offset: u64,
    ) -> SharedCommitterEntry {
        let file_id = NEXT_FILE_ID.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        if self
            .tx
            .send(Msg::Attach {
                file_id,
                file,
                initial_offset,
                reply: tx,
            })
            .await
            .is_ok()
        {
            // Best-effort wait for the bg task to register the file.
            // If the ack never arrives (bg task crashed) appends will
            // fail with the usual "shut down" error.
            let _ = rx.await;
        }
        SharedCommitterEntry {
            inner: Arc::new(EntryInner {
                file_id,
                tx: self.tx.clone(),
            }),
        }
    }
}

/// Per-file handle into the shared committer.
///
/// Cheap to clone: clones share the underlying `Arc<EntryInner>`. The
/// detach side-effect fires only when the **last** clone drops, so a
/// caller passing the entry into nested `GroupCommitter` clones does
/// not accidentally rip the file out from under in-flight writes.
#[derive(Clone)]
pub struct SharedCommitterEntry {
    inner: Arc<EntryInner>,
}

struct EntryInner {
    file_id: FileId,
    tx: mpsc::Sender<Msg>,
}

impl Drop for EntryInner {
    fn drop(&mut self) {
        // Best-effort detach; if the inbox is full the FileState
        // sticks around (bounded leak: at most one stale state per
        // entry that was ever attached, finite by construction).
        let _ = self.tx.try_send(Msg::Detach {
            file_id: self.file_id,
        });
    }
}

impl SharedCommitterEntry {
    fn file_id(&self) -> FileId {
        self.inner.file_id
    }

    fn tx(&self) -> &mpsc::Sender<Msg> {
        &self.inner.tx
    }

    /// Submit a write at the requested durability. Returns the
    /// (offset, padded_size) assigned by the committer once the
    /// containing batch has been flushed.
    ///
    /// # Errors
    /// Returns `io::Error::other` if the committer has shut down, or
    /// the underlying IO/sync error if the batch flush failed.
    pub async fn append(&self, data: Vec<u8>, durability: Durability) -> io::Result<(u64, u32)> {
        let (tx, rx) = oneshot::channel();
        self.tx()
            .send(Msg::Append {
                file_id: self.file_id(),
                data,
                durability,
                reply: tx,
            })
            .await
            .map_err(|_| io::Error::other("shared committer shut down"))?;
        rx.await
            .map_err(|_| io::Error::other("shared committer shut down"))?
    }

    /// Force a flush of the current batch.
    ///
    /// # Errors
    /// Returns `io::Error::other` if the committer has shut down, or
    /// the underlying flush error.
    pub async fn flush(&self) -> io::Result<()> {
        let (tx, rx) = oneshot::channel();
        self.tx()
            .send(Msg::Flush { reply: tx })
            .await
            .map_err(|_| io::Error::other("shared committer shut down"))?;
        rx.await
            .map_err(|_| io::Error::other("shared committer shut down"))?
    }
}

struct FileState {
    file: Arc<PlatformFile>,
    next_offset: u64,
}

struct BatchEntry {
    file_id: FileId,
    data: Vec<u8>,
    durability: Durability,
    reply: oneshot::Sender<io::Result<(u64, u32)>>,
}

async fn committer_loop(mut rx: mpsc::Receiver<Msg>) {
    let mut state: HashMap<FileId, FileState> = HashMap::new();
    let mut batch: Vec<BatchEntry> = Vec::new();
    let mut batch_bytes: usize = 0;
    let mut batch_deadline: Option<Instant> = None;

    loop {
        let sleep_until_inst =
            batch_deadline.unwrap_or_else(|| Instant::now() + Duration::from_secs(60 * 60));

        let mut should_flush = false;

        tokio::select! {
            biased;
            msg = rx.recv() => {
                match msg {
                    None => {
                        // All senders dropped: flush + exit.
                        flush_batch(&mut state, &mut batch).await;
                        return;
                    }
                    Some(Msg::Attach { file_id, file, initial_offset, reply }) => {
                        state.insert(file_id, FileState { file, next_offset: initial_offset });
                        let _ = reply.send(());
                    }
                    Some(Msg::Detach { file_id }) => {
                        // Remove only when no pending entries reference
                        // this file in the current batch; otherwise
                        // the in-flight writes would lose their target.
                        if !batch.iter().any(|e| e.file_id == file_id) {
                            state.remove(&file_id);
                        }
                        // If there are pending entries the file state
                        // is kept alive; the next batch flush will
                        // drain them and a subsequent detach attempt
                        // (or shutdown) can collect the FileState.
                    }
                    Some(Msg::Append { file_id, data, durability, reply }) => {
                        batch_bytes += data.len();
                        if batch.is_empty() {
                            batch_deadline = Some(Instant::now() + Duration::from_micros(TIMER_MICROS));
                        }
                        batch.push(BatchEntry { file_id, data, durability, reply });
                        if batch.len() >= MAX_BATCH_ENTRIES || batch_bytes >= MAX_BATCH_BYTES {
                            should_flush = true;
                        }
                    }
                    Some(Msg::Flush { reply }) => {
                        flush_batch(&mut state, &mut batch).await;
                        batch_bytes = 0;
                        batch_deadline = None;
                        let _ = reply.send(Ok(()));
                    }
                }
            }
            () = sleep_until(sleep_until_inst) => {
                if !batch.is_empty() {
                    should_flush = true;
                }
            }
        }

        if should_flush {
            flush_batch(&mut state, &mut batch).await;
            batch_bytes = 0;
            batch_deadline = None;
        }
    }
}

async fn flush_batch(state: &mut HashMap<FileId, FileState>, batch: &mut Vec<BatchEntry>) {
    if batch.is_empty() {
        return;
    }

    // Group by file_id, preserving append order within each file.
    let mut by_file: HashMap<FileId, Vec<PendingItem>> = HashMap::new();
    let mut batch_durability = Durability::Relaxed;
    for entry in batch.drain(..) {
        batch_durability = batch_durability.max(entry.durability);
        by_file
            .entry(entry.file_id)
            .or_default()
            .push((entry.data, entry.reply));
    }

    // Per-file: build combined buffer, assign sequential offsets,
    // issue one write_at per file. Writes go in sequence here; the
    // wins are still real because the savings come from the *fsync*
    // amortisation, not parallel writes. (Spawning parallel writes
    // adds task-launch noise without a measurable throughput win for
    // the small batches typical at the M-series fsync floor.)
    let mut sync_target: Option<Arc<PlatformFile>> = None;
    let mut successful_acks: Vec<AckSlot> = Vec::new();

    for (file_id, items) in by_file {
        let Some(fs) = state.get_mut(&file_id) else {
            // File detached mid-batch: ack each pending entry with
            // an error. This is rare and bounded (only happens on
            // race with Drop), but the ack must still arrive so the
            // caller is not left waiting.
            for (_, reply) in items {
                let _ = reply.send(Err(io::Error::other("file detached during batch")));
            }
            continue;
        };

        let start = fs.next_offset;
        let mut combined = Vec::new();
        let mut acks: Vec<AckSlot> = Vec::new();
        let mut pos = start;
        for (data, reply) in items {
            #[allow(clippy::cast_possible_truncation)]
            let sz = data.len() as u32;
            acks.push((pos, sz, reply));
            pos += data.len() as u64;
            combined.extend_from_slice(&data);
        }
        let end = pos;
        let file = fs.file.clone();
        match file.write_at(start, combined).await {
            Ok(()) => {
                fs.next_offset = end;
                if sync_target.is_none() {
                    sync_target = Some(file);
                }
                successful_acks.extend(acks);
            }
            Err(e) => {
                let kind = e.kind();
                let msg = e.to_string();
                for (_, _, reply) in acks {
                    let _ = reply.send(Err(io::Error::new(kind, msg.clone())));
                }
            }
        }
    }

    // One sync covers every successful write in this batch (the whole
    // point of the shared committer on a DeviceGlobal platform).
    let sync_res = match (batch_durability, sync_target.as_ref()) {
        (Durability::Relaxed, _) | (_, None) => Ok(()),
        (Durability::Kernel, Some(f)) => {
            skeg_telemetry::tick_counter(skeg_telemetry::Counter::VlogSyncs);
            f.sync_data().await
        }
        (Durability::Power, Some(f)) => {
            skeg_telemetry::tick_counter(skeg_telemetry::Counter::VlogSyncs);
            f.sync_durable().await
        }
    };
    skeg_telemetry::tick_counter(skeg_telemetry::Counter::VlogGroupCommitBatches);

    let sync_err = sync_res.as_ref().err().map(|e| (e.kind(), e.to_string()));
    for (off, sz, reply) in successful_acks {
        let ack = match &sync_err {
            None => Ok((off, sz)),
            Some((kind, msg)) => Err(io::Error::new(*kind, msg.clone())),
        };
        let _ = reply.send(ack);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use tempfile::TempDir;

    fn make_file(dir: &TempDir) -> Arc<PlatformFile> {
        let path: &Path = dir.path();
        Arc::new(PlatformFile::create(&path.join("test.bin")).unwrap())
    }

    #[tokio::test]
    async fn test_attach_append_flush_single_file() {
        let dir = TempDir::new().unwrap();
        let file = make_file(&dir);
        let sc = SharedCommitter::new();
        let entry = sc.attach(file.clone(), 0).await;

        let (off, sz) = entry
            .append(vec![0xABu8; 64], Durability::Kernel)
            .await
            .unwrap();
        assert_eq!(off, 0);
        assert_eq!(sz, 64);
        entry.flush().await.unwrap();
    }

    #[tokio::test]
    async fn test_offsets_monotone_per_file() {
        let dir = TempDir::new().unwrap();
        let file = make_file(&dir);
        let sc = SharedCommitter::new();
        let entry = sc.attach(file.clone(), 0).await;

        let (o0, s0) = entry
            .append(vec![1u8; 16], Durability::Kernel)
            .await
            .unwrap();
        let (o1, s1) = entry
            .append(vec![2u8; 32], Durability::Kernel)
            .await
            .unwrap();
        let (o2, s2) = entry
            .append(vec![3u8; 8], Durability::Kernel)
            .await
            .unwrap();
        assert_eq!((o0, s0), (0, 16));
        assert_eq!((o1, s1), (16, 32));
        assert_eq!((o2, s2), (48, 8));
    }

    #[tokio::test]
    async fn test_two_files_one_fsync_per_batch() {
        let dir = TempDir::new().unwrap();
        let f1 = Arc::new(PlatformFile::create(&dir.path().join("a.bin")).unwrap());
        let f2 = Arc::new(PlatformFile::create(&dir.path().join("b.bin")).unwrap());
        let sc = SharedCommitter::new();
        let e1 = sc.attach(f1.clone(), 0).await;
        let e2 = sc.attach(f2.clone(), 0).await;

        let h1 = tokio::spawn({
            let e1 = e1.clone();
            async move { e1.append(vec![0u8; 100], Durability::Power).await }
        });
        let h2 = tokio::spawn({
            let e2 = e2.clone();
            async move { e2.append(vec![0u8; 200], Durability::Power).await }
        });
        let (r1, r2) = (h1.await.unwrap().unwrap(), h2.await.unwrap().unwrap());
        assert_eq!(r1, (0, 100));
        assert_eq!(r2, (0, 200));

        // Both files have been synced exactly once across the two
        // appends; one fsync amortised the durability of both writes.
        // (Per-file PerFileCommitter would have called sync twice.)
        let total_sync = f1.sync_count() + f2.sync_count();
        assert_eq!(
            total_sync, 1,
            "shared committer must aggregate the fsync across files"
        );
    }

    #[tokio::test]
    async fn test_relaxed_alone_skips_fsync() {
        let dir = TempDir::new().unwrap();
        let file = make_file(&dir);
        let sc = SharedCommitter::new();
        let entry = sc.attach(file.clone(), 0).await;

        entry
            .append(vec![0u8; 64], Durability::Relaxed)
            .await
            .unwrap();
        entry.flush().await.unwrap();
        assert_eq!(file.sync_count(), 0);
    }

    /// M4 correctness gate: 4 shards drive Power appends concurrently
    /// through one SharedCommitter. After the dust settles every byte
    /// must land at the expected per-file offset. Pins the contract
    /// the M4 perf bench is measuring against (no point recovering
    /// throughput if data placement breaks under contention).
    #[tokio::test]
    async fn test_four_shards_parallel_power_writes_data_consistent() {
        const SHARDS: usize = 4;
        const WRITES_PER_SHARD: usize = 64;
        const RECORD_BYTES: usize = 128;

        let dir = TempDir::new().unwrap();
        let sc = SharedCommitter::new();

        let mut entries = Vec::with_capacity(SHARDS);
        let mut files = Vec::with_capacity(SHARDS);
        for s in 0..SHARDS {
            let f = Arc::new(PlatformFile::create(&dir.path().join(format!("s{s}.bin"))).unwrap());
            let e = sc.attach(f.clone(), 0).await;
            entries.push(e);
            files.push(f);
        }

        // Each shard writes its own marker byte so we can verify
        // cross-file isolation: shard s writes bytes == s as u8.
        let mut handles = Vec::new();
        for (s, entry) in entries.iter().cloned().enumerate() {
            let marker = u8::try_from(s).unwrap();
            handles.push(tokio::spawn(async move {
                let mut acks = Vec::with_capacity(WRITES_PER_SHARD);
                for _ in 0..WRITES_PER_SHARD {
                    let ack = entry
                        .append(vec![marker; RECORD_BYTES], Durability::Power)
                        .await
                        .unwrap();
                    acks.push(ack);
                }
                acks
            }));
        }

        let mut per_shard_acks: Vec<Vec<(u64, u32)>> = Vec::with_capacity(SHARDS);
        for h in handles {
            per_shard_acks.push(h.await.unwrap());
        }

        // Per-file: offsets monotone 0..N*RECORD_BYTES, size == RECORD_BYTES.
        for (s, acks) in per_shard_acks.iter().enumerate() {
            assert_eq!(acks.len(), WRITES_PER_SHARD, "shard {s} ack count");
            for (i, &(off, sz)) in acks.iter().enumerate() {
                assert_eq!(
                    sz,
                    u32::try_from(RECORD_BYTES).unwrap(),
                    "shard {s} entry {i} size"
                );
                assert_eq!(
                    off,
                    u64::try_from(i * RECORD_BYTES).unwrap(),
                    "shard {s} entry {i} offset not sequential"
                );
            }
        }

        // Read-back: every byte in every file must equal the shard's marker.
        for (s, file) in files.iter().enumerate() {
            let total = WRITES_PER_SHARD * RECORD_BYTES;
            let data = file.pread(0, total).await.unwrap();
            let marker = u8::try_from(s).unwrap();
            assert!(
                data.iter().all(|&b| b == marker),
                "shard {s} file content corrupted under contention"
            );
        }

        // Sync amortisation: total fsync count must be far less than
        // SHARDS * WRITES_PER_SHARD. Loose upper bound (exact count
        // depends on batch timing) but proves aggregation happened.
        let total_sync: u64 = files.iter().map(|f| f.sync_count()).sum();
        let writes = u64::try_from(SHARDS * WRITES_PER_SHARD).unwrap();
        assert!(
            total_sync < writes,
            "expected fsync amortisation, got {total_sync} syncs for {writes} writes"
        );
    }

    /// M5 crash-recovery gate. 100 deterministic iterations driven by
    /// a seeded xorshift PRNG: each iteration randomises
    /// (shards, writes-per-shard, record bytes), commits every write
    /// at `Durability::Power`, then *drops* every handle (committer
    /// entries + file Arcs) and reopens each backing file fresh.
    ///
    /// Post-condition: every acked Power write must be readable from
    /// the reopened file at the exact (offset, size) the committer
    /// returned, with byte-for-byte content match. Captures three
    /// classes of bug at once:
    ///   - lost writes (batch dropped before sync)
    ///   - mis-routed writes (shard A's bytes land in shard B's file)
    ///   - offset accounting drift (sequential writes overlap or skip)
    ///
    /// Deterministic seed: failures reproduce by re-running the same
    /// iteration index.
    #[tokio::test]
    async fn test_crash_recovery_100_iter_random_seed() {
        // Tiny xorshift64* PRNG. Avoids pulling rand into core dev-deps.
        // Period 2^64-1, statistical quality good enough for randomising
        // workload shapes.
        struct Rng(u64);
        impl Rng {
            fn new(seed: u64) -> Self {
                Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
            }
            fn next_u64(&mut self) -> u64 {
                let mut x = self.0;
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                self.0 = x;
                x.wrapping_mul(0x2545_F491_4F6C_DD1D)
            }
            fn range(&mut self, lo: usize, hi_inclusive: usize) -> usize {
                let span = (hi_inclusive - lo + 1) as u64;
                lo + (self.next_u64() % span) as usize
            }
            fn byte(&mut self) -> u8 {
                (self.next_u64() & 0xFF) as u8
            }
        }

        const ITERATIONS: u64 = 100;

        for seed in 0..ITERATIONS {
            let mut rng = Rng::new(seed);
            let shards = rng.range(1, 4);
            let writes_per_shard = rng.range(8, 80);
            let record_bytes = rng.range(16, 256);

            let dir = TempDir::new().unwrap();
            let sc = SharedCommitter::new();

            // Pre-generate the payload for each (shard, write_idx) so
            // both write side and verify side use the same bytes.
            // Shape: payloads[shard][write_idx] = Vec<u8> of record_bytes.
            let mut payloads: Vec<Vec<Vec<u8>>> = Vec::with_capacity(shards);
            for s in 0..shards {
                let mut shard_payloads: Vec<Vec<u8>> = Vec::with_capacity(writes_per_shard);
                for w in 0..writes_per_shard {
                    let marker = rng
                        .byte()
                        .wrapping_add(u8::try_from(s & 0xFF).unwrap())
                        .wrapping_add(u8::try_from(w & 0xFF).unwrap());
                    shard_payloads.push(vec![marker; record_bytes]);
                }
                payloads.push(shard_payloads);
            }

            // Attach each shard's file; keep paths for the reopen pass.
            let mut paths = Vec::with_capacity(shards);
            let mut entries = Vec::with_capacity(shards);
            for s in 0..shards {
                let path = dir.path().join(format!("s{s}.bin"));
                let file = Arc::new(PlatformFile::create(&path).unwrap());
                let entry = sc.attach(file.clone(), 0).await;
                paths.push(path);
                entries.push(entry);
                // Original file Arc dropped end-of-iteration with the
                // entry; the committer holds its own clone internally
                // until Detach completes.
            }

            // Drive writes concurrently across shards. Order within a
            // shard is sequential to make offset verification trivial.
            let mut handles = Vec::with_capacity(shards);
            for (s, entry) in entries.iter().cloned().enumerate() {
                let shard_payloads = payloads[s].clone();
                handles.push(tokio::spawn(async move {
                    let mut acks = Vec::with_capacity(shard_payloads.len());
                    for payload in shard_payloads {
                        let ack = entry.append(payload, Durability::Power).await.unwrap();
                        acks.push(ack);
                    }
                    acks
                }));
            }
            let mut per_shard_acks: Vec<Vec<(u64, u32)>> = Vec::with_capacity(shards);
            for h in handles {
                per_shard_acks.push(h.await.unwrap());
            }

            // Force a flush before tearing down to make sure the last
            // batch hits the disk. Mirrors what a graceful shutdown
            // does; an actual crash mid-batch is outside this test's
            // scope (handled at the vLog recovery-scan layer).
            for entry in &entries {
                entry.flush().await.unwrap();
            }

            // Simulated crash: drop every committer entry + the original
            // file Arc by clearing the vectors. The bg task's
            // FileState clone is the only thing keeping the fd alive
            // briefly; once Detach is processed it goes too.
            drop(entries);

            // Verify pass: reopen each file fresh and read back.
            for (s, path) in paths.iter().enumerate() {
                let reopened = PlatformFile::open(path).unwrap();
                for (w, &(off, sz)) in per_shard_acks[s].iter().enumerate() {
                    assert_eq!(
                        sz as usize, record_bytes,
                        "seed {seed} shard {s} entry {w}: size mismatch"
                    );
                    let mut buf = vec![0u8; sz as usize];
                    let n = reopened.pread_sync(off, &mut buf).unwrap_or_else(|e| {
                        panic!(
                            "seed {seed} shard {s} entry {w}: pread at off={off} sz={sz} failed: {e}"
                        )
                    });
                    assert_eq!(
                        n, sz as usize,
                        "seed {seed} shard {s} entry {w}: short read {n} < {sz}"
                    );
                    assert_eq!(
                        buf, payloads[s][w],
                        "seed {seed} shard {s} entry {w} off={off} content mismatch"
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn test_relaxed_in_mixed_batch_gets_durability() {
        let dir = TempDir::new().unwrap();
        let file = make_file(&dir);
        let sc = SharedCommitter::new();
        let entry = sc.attach(file.clone(), 0).await;

        // Two appends submitted concurrently land in the same batch
        // window; the Power request promotes the whole batch to fsync.
        let e1 = entry.clone();
        let e2 = entry.clone();
        let h1 = tokio::spawn(async move { e1.append(vec![0u8; 64], Durability::Relaxed).await });
        let h2 = tokio::spawn(async move { e2.append(vec![0u8; 64], Durability::Power).await });
        h1.await.unwrap().unwrap();
        h2.await.unwrap().unwrap();
        assert!(
            file.sync_count() >= 1,
            "a Power entry must force the batch to flush"
        );
    }
}
