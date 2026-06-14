#![deny(unsafe_code)]

//! Async group committer: batches multiple write requests, flushing each batch
//! at the strongest durability any of its entries requested.
//!
//! Flush triggers (whichever fires first):
//!   - accumulated bytes ≥ 256 KB
//!   - accumulated entries ≥ 256
//!   - 200 µs timer since last message

use std::io;
use std::sync::Arc;

use skeg_platform::PlatformFile;
use tokio::sync::{mpsc, oneshot};
use tokio::time::{Duration, sleep};

use crate::shared_committer::{SharedCommitter, SharedCommitterEntry};

const MAX_BATCH_BYTES: usize = 256 * 1024; // 256 KB
const MAX_BATCH_ENTRIES: usize = 256;
const TIMER_MICROS: u64 = 200;

/// Durability requested for a write. Ordered weakest → strongest.
///
/// AI workloads rarely need power-loss durability for every write (an embedding
/// cache can be recomputed), so `Kernel` is the sensible default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Durability {
    /// Ack once the record is in the OS write buffer. Survives a process
    /// crash, not a kernel panic or power loss. No `fsync`.
    Relaxed,
    /// Ack after `fsync`/`fdatasync` - survives a kernel panic, not power loss.
    #[default]
    Kernel,
    /// Ack after `F_FULLFSYNC` - survives power loss.
    Power,
}

// ── Internal channel types ────────────────────────────────────────────────────

struct WriteReq {
    data: Vec<u8>,
    durability: Durability,
    tx: oneshot::Sender<io::Result<(u64, u32)>>,
}

enum Msg {
    Write(WriteReq),
    Flush(oneshot::Sender<io::Result<()>>),
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Façade in front of the platform-specific committer strategy.
///
/// On a [`DurabilityModel::PerFile`] platform (Linux, default), each
/// committer owns its own background task and a per-file flush loop;
/// `N` shards => `N` parallel `fdatasync` calls that the kernel
/// schedules independently.
///
/// On a [`DurabilityModel::DeviceGlobal`] platform (macOS, default),
/// every `sync_durable` is a device-wide barrier, so per-shard
/// committers serialize on the hardware. The dispatch routes through
/// the process-wide [`SharedCommitter`] which aggregates writes from
/// every shard on the device into a single `sync_durable` per batch.
///
/// Cloneable: handles share the underlying background task / entry.
///
/// [`DurabilityModel::PerFile`]: skeg_platform::DurabilityModel::PerFile
/// [`DurabilityModel::DeviceGlobal`]: skeg_platform::DurabilityModel::DeviceGlobal
#[derive(Clone)]
pub struct GroupCommitter {
    inner: CommitterImpl,
}

#[derive(Clone)]
enum CommitterImpl {
    /// Standalone background task per file. Linux default.
    PerFile(PerFileCommitter),
    /// Handle into the process-wide shared committer. Every entry
    /// routes appends to the same bg task, which amortises one
    /// `sync_durable` across all the device's shards.
    DeviceGlobal(SharedCommitterEntry),
}

impl GroupCommitter {
    /// Start a committer for `file` starting at `initial_offset`. Picks
    /// the strategy by consulting [`resolve_durability_model`]; the
    /// returned handle is cheap to clone, every clone shares the
    /// underlying background task.
    ///
    /// The async signature is required because the `DeviceGlobal` arm
    /// has to attach the file to the shared committer's registry
    /// before the first append; the attach round-trips through the
    /// bg task. The `PerFile` arm is sync underneath but pays a
    /// no-op `async` wrapper for API symmetry.
    ///
    /// [`resolve_durability_model`]: skeg_platform::resolve_durability_model
    pub async fn start(file: Arc<PlatformFile>, initial_offset: u64) -> Self {
        let model = skeg_platform::resolve_durability_model();
        let inner = match model {
            skeg_platform::DurabilityModel::PerFile => {
                CommitterImpl::PerFile(PerFileCommitter::start(file, initial_offset))
            }
            skeg_platform::DurabilityModel::DeviceGlobal => {
                let entry = SharedCommitter::global().attach(file, initial_offset).await;
                CommitterImpl::DeviceGlobal(entry)
            }
        };
        Self { inner }
    }

    /// Submit a pre-encoded record for a write at the given durability.
    ///
    /// Returns `(start_offset, padded_size_bytes)` once the containing
    /// batch has been flushed at (at least) the requested durability.
    ///
    /// # Errors
    ///
    /// Returns an IO error if the committer task has shut down or if
    /// the underlying write or flush fails.
    pub async fn append(&self, data: Vec<u8>, durability: Durability) -> io::Result<(u64, u32)> {
        match &self.inner {
            CommitterImpl::PerFile(c) => c.append(data, durability).await,
            CommitterImpl::DeviceGlobal(c) => c.append(data, durability).await,
        }
    }

    /// Force-flush all pending writes immediately.
    ///
    /// Blocks until the flush completes. Useful for graceful shutdown
    /// or tests.
    ///
    /// # Errors
    ///
    /// Returns an IO error if the committer task has shut down or if
    /// the flush fails.
    pub async fn flush(&self) -> io::Result<()> {
        match &self.inner {
            CommitterImpl::PerFile(c) => c.flush().await,
            CommitterImpl::DeviceGlobal(c) => c.flush().await,
        }
    }
}

/// Single-file group committer: one background task per file, batches
/// writes from every producer into one `sync_durable` per batch. Used
/// directly on `PerFile` platforms and as the underlying implementation
/// of the placeholder `DeviceGlobal` branch until the device-global
/// path of the shared-committer workstream lands.
#[derive(Clone)]
struct PerFileCommitter {
    tx: Arc<mpsc::UnboundedSender<Msg>>,
}

impl PerFileCommitter {
    fn start(file: Arc<PlatformFile>, initial_offset: u64) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        tokio::spawn(committer_task(file, rx, initial_offset));
        Self { tx: Arc::new(tx) }
    }

    async fn append(&self, data: Vec<u8>, durability: Durability) -> io::Result<(u64, u32)> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(Msg::Write(WriteReq {
                data,
                durability,
                tx,
            }))
            .map_err(|_| io::Error::other("committer shut down"))?;
        rx.await
            .map_err(|_| io::Error::other("committer shut down"))?
    }

    async fn flush(&self) -> io::Result<()> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(Msg::Flush(tx))
            .map_err(|_| io::Error::other("committer shut down"))?;
        rx.await
            .map_err(|_| io::Error::other("committer shut down"))?
    }
}

// ── Background task ───────────────────────────────────────────────────────────

async fn committer_task(
    file: Arc<PlatformFile>,
    mut rx: mpsc::UnboundedReceiver<Msg>,
    mut w_offset: u64,
) {
    let mut batch: Vec<WriteReq> = Vec::new();
    let mut batch_bytes: usize = 0;

    loop {
        let flush = tokio::select! {
            msg = rx.recv() => {
                match msg {
                    None => {
                        // Channel closed: flush remaining entries and exit.
                        flush_batch(&file, &mut batch, &mut w_offset).await;
                        return;
                    }
                    Some(Msg::Write(req)) => {
                        batch_bytes += req.data.len();
                        batch.push(req);
                        batch_bytes >= MAX_BATCH_BYTES || batch.len() >= MAX_BATCH_ENTRIES
                    }
                    Some(Msg::Flush(reply_tx)) => {
                        flush_batch(&file, &mut batch, &mut w_offset).await;
                        batch_bytes = 0;
                        let _ = reply_tx.send(Ok(()));
                        false
                    }
                }
            }
            () = sleep(Duration::from_micros(TIMER_MICROS)) => {
                !batch.is_empty()
            }
        };

        if flush {
            flush_batch(&file, &mut batch, &mut w_offset).await;
            batch_bytes = 0;
        }
    }
}

async fn flush_batch(file: &PlatformFile, batch: &mut Vec<WriteReq>, w_offset: &mut u64) {
    if batch.is_empty() {
        return;
    }

    // Assign sequential offsets, build the combined write buffer, and find the
    // strongest durability any entry in this batch asked for.
    let mut combined = Vec::new();
    let mut offsets: Vec<(u64, u32)> = Vec::with_capacity(batch.len());
    let mut pos = *w_offset;
    let mut batch_durability = Durability::Relaxed;
    for req in batch.iter() {
        #[allow(clippy::cast_possible_truncation)]
        offsets.push((pos, req.data.len() as u32));
        pos += req.data.len() as u64;
        combined.extend_from_slice(&req.data);
        batch_durability = batch_durability.max(req.durability);
    }

    // One write, then one flush for the whole group at the strongest tier.
    //
    // Disk-full semantics: if `write_at` fails with `ENOSPC` (or any other
    // IO error), `w_offset` is NOT advanced (see line below); the next batch
    // will retry at the same offset, overwriting any partial bytes the
    // failed write may have left behind. After a crash mid-write, the
    // vLog recovery scan stops at the first record with a bad CRC, so the
    // failed batch is correctly forgotten - no zombie partial record can
    // be mistaken for live data. We preserve the original `ErrorKind`
    // (in particular `StorageFull`) when propagating to the waiter so the
    // caller can detect ENOSPC and surface it to its own user.
    let write_result = file.write_at(*w_offset, combined).await;
    match write_result {
        Ok(()) => {
            // Telemetry: tick one batch per call regardless of durability.
            // Inexpensive (atomic fetch_add); off the hot per-op path.
            skeg_telemetry::tick_counter(skeg_telemetry::Counter::VlogGroupCommitBatches);
            let sync_result = match batch_durability {
                Durability::Relaxed => Ok(()),
                Durability::Kernel => {
                    skeg_telemetry::tick_counter(skeg_telemetry::Counter::VlogSyncs);
                    file.sync_data().await
                }
                Durability::Power => {
                    skeg_telemetry::tick_counter(skeg_telemetry::Counter::VlogSyncs);
                    file.sync_durable().await
                }
            };
            let sync_err = sync_result
                .as_ref()
                .err()
                .map(|e| (e.kind(), e.to_string()));
            if sync_result.is_ok() {
                *w_offset = pos;
            }
            for (req, (off, sz)) in batch.drain(..).zip(offsets) {
                let result = match &sync_err {
                    None => Ok((off, sz)),
                    Some((kind, msg)) => Err(io::Error::new(*kind, msg.clone())),
                };
                let _ = req.tx.send(result);
            }
        }
        Err(e) => {
            let kind = e.kind();
            let msg = e.to_string();
            for req in batch.drain(..) {
                let _ = req.tx.send(Err(io::Error::new(kind, msg.clone())));
            }
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Most tests in this module check `file.sync_count()` to verify a
    /// flush happened. That contract is per-file and matches the
    /// `PerFile` committer; under `DeviceGlobal` the shared committer
    /// would issue one fsync on whichever file in the batch happened
    /// to be the first successful write, leaving sibling files with a
    /// 0 sync_count even though they are durable (device-wide
    /// barrier). Force `PerFile` for the assertions in this module
    /// to make sense; the DeviceGlobal semantics are covered by
    /// `shared_committer::tests` instead.
    ///
    /// Idempotent across tests: every `#[tokio::test]` here calls
    /// `force_per_file()` first so test ordering does not matter.
    fn force_per_file() {
        skeg_platform::durability::set_durability_model_for_tests(
            skeg_platform::DurabilityModel::PerFile,
        );
    }

    fn make_file(dir: &TempDir) -> Arc<PlatformFile> {
        Arc::new(PlatformFile::create(dir.path().join("gc.bin").as_path()).unwrap())
    }

    #[tokio::test]
    async fn test_group_commit_single_write() {
        force_per_file();
        let dir = TempDir::new().unwrap();
        let file = make_file(&dir);
        let gc = GroupCommitter::start(file.clone(), 0).await;

        let (off, sz) = gc
            .append(vec![0xAAu8; 128], Durability::Power)
            .await
            .unwrap();
        assert_eq!(off, 0);
        assert_eq!(sz, 128);

        let data = file.pread(0, 128).await.unwrap();
        assert!(data.iter().all(|&b| b == 0xAA));
    }

    #[tokio::test]
    async fn test_group_commit_batch_of_n() {
        force_per_file();
        let dir = TempDir::new().unwrap();
        let file = make_file(&dir);
        let gc = GroupCommitter::start(file.clone(), 0).await;

        // Launch 10 concurrent writers - they should batch into ≤ 2 syncs.
        let n: u64 = 10;
        let mut handles = Vec::new();
        for i in 0u64..n {
            let gc2 = gc.clone();
            handles.push(tokio::spawn(async move {
                gc2.append(vec![i as u8; 128], Durability::Power).await
            }));
        }

        let mut results: Vec<(u64, u32)> = Vec::new();
        for h in handles {
            results.push(h.await.unwrap().unwrap());
        }

        // All acked with correct size.
        assert_eq!(results.len(), 10);
        for &(_, sz) in &results {
            assert_eq!(sz, 128);
        }

        // Offsets must cover 0..10*128 non-overlapping.
        let mut offsets: Vec<u64> = results.iter().map(|&(off, _)| off).collect();
        offsets.sort_unstable();
        for (i, &off) in offsets.iter().enumerate() {
            assert_eq!(off, i as u64 * 128);
        }

        // Batching: fewer syncs than entries.
        let syncs = file.sync_count();
        assert!(
            syncs < n,
            "expected batching but got {syncs} syncs for {n} entries"
        );
    }

    #[tokio::test]
    async fn test_group_commit_timer_flush() {
        force_per_file();
        let dir = TempDir::new().unwrap();
        let file = make_file(&dir);
        let gc = GroupCommitter::start(file.clone(), 0).await;

        // Submit a single entry - well under batch limits.
        // The committer should auto-flush after the 200 µs timer.
        let start = std::time::Instant::now();
        let (off, sz) = gc
            .append(vec![0xBBu8; 64], Durability::Power)
            .await
            .unwrap();
        let elapsed = start.elapsed();

        assert_eq!(off, 0);
        assert_eq!(sz, 64);
        assert!(
            elapsed >= Duration::from_micros(TIMER_MICROS),
            "flush should wait at least {TIMER_MICROS} µs, took {elapsed:?}"
        );
        // Upper bound is generous: shared GH Actions runners stall the
        // committer task tens of ms beyond the 200 µs timer. We only
        // care that the timer fires "soon", not that it fires inside
        // any specific budget.
        assert!(
            elapsed < Duration::from_secs(2),
            "timer took too long: {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn test_group_commit_explicit_flush() {
        force_per_file();
        let dir = TempDir::new().unwrap();
        let file = make_file(&dir);
        let gc = GroupCommitter::start(file.clone(), 0).await;

        // Start an append without awaiting it yet.
        let write_handle = tokio::spawn({
            let gc2 = gc.clone();
            async move { gc2.append(vec![0xCCu8; 128], Durability::Power).await }
        });

        // Force flush immediately.
        gc.flush().await.unwrap();

        // The write should have completed.
        let (off, sz) = write_handle.await.unwrap().unwrap();
        assert_eq!(off, 0);
        assert_eq!(sz, 128);
    }

    #[tokio::test]
    async fn test_group_commit_sequential_offsets() {
        force_per_file();
        let dir = TempDir::new().unwrap();
        let file = make_file(&dir);
        let gc = GroupCommitter::start(file.clone(), 0).await;

        let (off0, sz0) = gc.append(vec![1u8; 128], Durability::Power).await.unwrap();
        let (off1, sz1) = gc.append(vec![2u8; 256], Durability::Power).await.unwrap();
        let (off2, _) = gc.append(vec![3u8; 128], Durability::Power).await.unwrap();

        assert_eq!(off0, 0);
        assert_eq!(off1, u64::from(sz0));
        assert_eq!(off2, u64::from(sz0) + u64::from(sz1));
    }

    #[tokio::test]
    async fn test_durability_relaxed_no_sync() {
        force_per_file();
        let dir = TempDir::new().unwrap();
        let file = make_file(&dir);
        let gc = GroupCommitter::start(file.clone(), 0).await;

        let (off, sz) = gc
            .append(vec![0u8; 128], Durability::Relaxed)
            .await
            .unwrap();
        assert_eq!((off, sz), (0, 128));
        // Relaxed: the data is written, but no fsync was issued.
        assert_eq!(file.sync_count(), 0, "Relaxed must not fsync");
        assert!(file.pread(0, 128).await.unwrap().iter().all(|&b| b == 0));
    }

    #[tokio::test]
    async fn test_durability_kernel_syncs() {
        force_per_file();
        let dir = TempDir::new().unwrap();
        let file = make_file(&dir);
        let gc = GroupCommitter::start(file.clone(), 0).await;

        gc.append(vec![1u8; 128], Durability::Kernel).await.unwrap();
        assert!(file.sync_count() >= 1, "Kernel must issue a flush");
    }

    #[tokio::test]
    async fn test_durability_batch_takes_max() {
        force_per_file();
        let dir = TempDir::new().unwrap();
        let file = make_file(&dir);
        let gc = GroupCommitter::start(file.clone(), 0).await;

        // One Power write among Relaxed ones must drag the whole batch to a
        // durable flush.
        let mut handles = Vec::new();
        for i in 0u64..8 {
            let gc2 = gc.clone();
            let dur = if i == 3 {
                Durability::Power
            } else {
                Durability::Relaxed
            };
            handles.push(tokio::spawn(async move {
                gc2.append(vec![i as u8; 128], dur).await
            }));
        }
        for h in handles {
            h.await.unwrap().unwrap();
        }
        assert!(
            file.sync_count() >= 1,
            "a Power entry must force the batch to flush"
        );
    }

    /// Dispatch smoke test: the façade builds on both
    /// `DurabilityModel::PerFile` and `DurabilityModel::DeviceGlobal`
    /// and round-trips an append. Until the device-global path lands,
    /// both branches delegate to `PerFileCommitter`, so behaviour is
    /// observationally identical; this test pins that contract so the
    /// rewire is caught if it accidentally regresses the `PerFile` codepath.
    #[tokio::test]
    async fn test_facade_dispatches_on_durability_model() {
        force_per_file();
        use skeg_platform::{DurabilityModel, durability};

        for model in [DurabilityModel::PerFile, DurabilityModel::DeviceGlobal] {
            durability::set_durability_model_for_tests(model);

            let dir = TempDir::new().unwrap();
            let file = make_file(&dir);
            let gc = GroupCommitter::start(file.clone(), 0).await;

            let (off, sz) = gc
                .append(vec![0xAAu8; 64], Durability::Kernel)
                .await
                .unwrap();
            assert_eq!(off, 0, "model {model:?}: first append must start at 0");
            assert_eq!(sz, 64, "model {model:?}: padded size mismatch");
        }

        durability::reset_durability_model_cache_for_tests();
    }
}
