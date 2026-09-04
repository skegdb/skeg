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

use crate::failpoint::{self, CommitFailpoint};
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
                        //
                        // Nobody is left to answer, which is exactly why this
                        // used to be discarded - and exactly why it has to be
                        // said out loud instead: a shutdown that could not
                        // land its last batch is otherwise indistinguishable
                        // from a clean one. The waiters of that batch still
                        // get their `Err` from inside `flush_batch`; this is
                        // for the operator, who has no waiter.
                        if let Err(e) = flush_batch(&file, &mut batch, &mut w_offset).await {
                            tracing::error!(
                                error = %e,
                                "group committer: the last flush before shutdown did not land"
                            );
                        }
                        return;
                    }
                    Some(Msg::Write(req)) => {
                        batch_bytes += req.data.len();
                        batch.push(req);
                        batch_bytes >= MAX_BATCH_BYTES || batch.len() >= MAX_BATCH_ENTRIES
                    }
                    Some(Msg::Flush(reply_tx)) => {
                        let result = flush_batch(&file, &mut batch, &mut w_offset).await;
                        batch_bytes = 0;
                        let _ = reply_tx.send(result);
                        false
                    }
                }
            }
            () = sleep(Duration::from_micros(TIMER_MICROS)) => {
                !batch.is_empty()
            }
        };

        if flush {
            // A size- or timer-triggered flush has no caller waiting on a
            // result: every waiter in the batch is answered individually
            // inside `flush_batch`. The failure is still worth a line, because
            // it is the first sign the disk under this store is going.
            if let Err(e) = flush_batch(&file, &mut batch, &mut w_offset).await {
                tracing::error!(error = %e, "group committer: batch flush did not land");
            }
            batch_bytes = 0;
        }
    }
}

/// Commit `batch` to `file` and answer every waiter in it.
///
/// Returns what the batch actually achieved, which is not the same question as
/// what each waiter is owed: the waiters are answered here either way, and the
/// result is for the caller who asked for a barrier. `Ok(())` means the bytes
/// are on the disk at the durability the batch asked for; an `Err` means they
/// are not, and no caller may treat the flush as one.
///
/// An empty batch is `Ok(())`: there was nothing to make durable. Everything
/// that came before it was answered by its own flush.
async fn flush_batch(
    file: &PlatformFile,
    batch: &mut Vec<WriteReq>,
    w_offset: &mut u64,
) -> io::Result<()> {
    if batch.is_empty() {
        return Ok(());
    }
    let fp_key = failpoint::file_key(file);

    // Assign sequential offsets and find the strongest durability any entry
    // in this batch asked for. Each entry's buffer is *moved* out into
    // `chunks`, not copied into one combined buffer: `write_vectored_at`
    // (`pwritev` on Linux) writes straight from each buffer's own memory.
    let mut chunks: Vec<Vec<u8>> = Vec::with_capacity(batch.len());
    let mut offsets: Vec<(u64, u32)> = Vec::with_capacity(batch.len());
    let mut pos = *w_offset;
    let mut batch_durability = Durability::Relaxed;
    let mut total_bytes: u64 = 0;
    for req in batch.iter_mut() {
        #[allow(clippy::cast_possible_truncation)]
        offsets.push((pos, req.data.len() as u32));
        pos += req.data.len() as u64;
        total_bytes += req.data.len() as u64;
        batch_durability = batch_durability.max(req.durability);
        chunks.push(std::mem::take(&mut req.data));
    }

    // One write, then one flush for the whole group at the strongest tier.
    //
    // Disk-full semantics: if the write fails with `ENOSPC` (or any other
    // IO error), `w_offset` is NOT advanced (see line below); the next batch
    // will retry at the same offset, overwriting any partial bytes the
    // failed write may have left behind. After a crash mid-write, the
    // vLog recovery scan stops at the first record with a bad CRC, so the
    // failed batch is correctly forgotten - no zombie partial record can
    // be mistaken for live data. We preserve the original `ErrorKind`
    // (in particular `StorageFull`) when propagating to the waiter so the
    // caller can detect ENOSPC and surface it to its own user.
    let write_result = if failpoint::hit(CommitFailpoint::PerFileBatchWrite, fp_key) {
        Err(failpoint::injected("per-file batch write"))
    } else {
        file.write_vectored_at(*w_offset, chunks).await
    };
    match write_result {
        Ok(()) => {
            // Telemetry: tick one batch per call regardless of durability,
            // plus the payload bytes moved through the zero-copy path.
            // Inexpensive (atomic fetch_add); off the hot per-op path.
            skeg_telemetry::tick_counter(skeg_telemetry::Counter::VlogGroupCommitBatches);
            skeg_telemetry::add_counter(
                skeg_telemetry::Counter::VlogPwritevBytesTotal,
                total_bytes,
            );
            let sync_result = if failpoint::hit(CommitFailpoint::PerFileBatchSync, fp_key) {
                Err(failpoint::injected("per-file batch sync"))
            } else {
                match batch_durability {
                    Durability::Relaxed => Ok(()),
                    Durability::Kernel => {
                        skeg_telemetry::tick_counter(skeg_telemetry::Counter::VlogSyncs);
                        file.sync_data().await
                    }
                    Durability::Power => {
                        skeg_telemetry::tick_counter(skeg_telemetry::Counter::VlogSyncs);
                        // `is_size_fixed` is stable for the handle's whole
                        // lifetime (set once by `preallocate`), so this
                        // accurately reflects whether the flush that is about
                        // to run will take the Linux fdatasync fast path.
                        if file.is_size_fixed() {
                            skeg_telemetry::tick_counter(
                                skeg_telemetry::Counter::VlogFdatasyncFastPath,
                            );
                        }
                        file.sync_durable().await
                    }
                }
            };
            let sync_err = sync_result
                .as_ref()
                .err()
                .map(|e| (e.kind(), e.to_string()));
            // `w_offset` only advances when the sync *also* succeeds - not
            // just the write - deliberately mirroring the write-failure case
            // documented above: on a sync failure the bytes are physically
            // on disk at `[*w_offset, pos)` but not proven durable, so the
            // next batch reuses the same starting offset and overwrites
            // them. Every waiter in this batch gets an `Err` below regardless
            // (nobody is told an unsynced write is durable), so the only
            // possible skew is in the caller's favour: if a crash happens
            // before that overwrite, recovery's CRC scan can still pick up
            // the well-formed-but-unacked record. Never the reverse - a
            // record this function reports as durable always has passed sync.
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
            match sync_err {
                None => Ok(()),
                Some((kind, msg)) => Err(count_failure(io::Error::new(kind, msg))),
            }
        }
        Err(e) => {
            let kind = e.kind();
            let msg = e.to_string();
            for req in batch.drain(..) {
                let _ = req.tx.send(Err(io::Error::new(kind, msg.clone())));
            }
            Err(count_failure(e))
        }
    }
}

/// Record that one batch did not become durable, and hand the error back.
///
/// Every failed flush passes through here, whoever asked for it - an explicit
/// `flush()`, a full batch, the 200 µs timer, or the last one before shutdown -
/// and in either committer. The counter is the only signal for the three of
/// those that have no caller to answer.
pub(crate) fn count_failure(e: io::Error) -> io::Error {
    skeg_telemetry::tick_counter(skeg_telemetry::Counter::VlogFlushFailures);
    e
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

    /// Run `committer_task` over a channel every message of which is queued
    /// BEFORE the task exists.
    ///
    /// Not decoration: it is what makes these tests deterministic. The 200 µs
    /// timer is a fresh `sleep` constructed inside the `select!` on every
    /// iteration, so it cannot already be elapsed when it is first polled -
    /// which means a message that is already in the queue always wins. Queue
    /// the whole conversation first and the task's route through it is fixed:
    /// write, then flush, with no batch the timer could have taken first.
    fn queued_committer(
        file: Arc<PlatformFile>,
        initial_offset: u64,
    ) -> (mpsc::UnboundedSender<Msg>, tokio::task::JoinHandle<()>) {
        let (tx, rx) = mpsc::unbounded_channel();
        let handle = tokio::spawn(committer_task(file, rx, initial_offset));
        (tx, handle)
    }

    fn write_msg(
        data: Vec<u8>,
        durability: Durability,
    ) -> (Msg, oneshot::Receiver<io::Result<(u64, u32)>>) {
        let (tx, rx) = oneshot::channel();
        (
            Msg::Write(WriteReq {
                data,
                durability,
                tx,
            }),
            rx,
        )
    }

    fn flush_msg() -> (Msg, oneshot::Receiver<io::Result<()>>) {
        let (tx, rx) = oneshot::channel();
        (Msg::Flush(tx), rx)
    }

    /// The process-wide flush-failure counter, held across a whole test so the
    /// delta it measures is its own.
    async fn counter_guard() -> tokio::sync::MutexGuard<'static, ()> {
        crate::failpoint::FLUSH_FAILURE_COUNTER_LOCK.lock().await
    }

    fn flush_failures() -> u64 {
        skeg_telemetry::counter_value(skeg_telemetry::Counter::VlogFlushFailures)
    }

    /// An accepted barrier over bytes nothing has synced is the same lie as an
    /// accepted barrier over a failed sync - it just takes a `Relaxed` writer
    /// to reach it.
    ///
    /// SD-1: `Relaxed` is a client-reachable path, not a compaction-only one
    /// (`PAYLOAD_DURABILITY`, `shard.rs:1471`, is the durability of every VSET
    /// payload blob and of four tombstone sweeps). A flush that answers `Ok`
    /// having issued no sync at all confirms a barrier that did not happen.
    #[tokio::test]
    #[ignore = "opens in the commit that makes an explicit flush a real barrier"]
    async fn an_explicit_flush_syncs_what_a_relaxed_batch_left_unsynced() {
        force_per_file();
        let dir = TempDir::new().unwrap();
        let file = make_file(&dir);

        let (tx, _task) = queued_committer(file.clone(), 0);
        let (write, write_rx) = write_msg(vec![1u8; 64], Durability::Relaxed);
        let (flush, flush_rx) = flush_msg();
        tx.send(write).unwrap();
        tx.send(flush).unwrap();

        assert!(flush_rx.await.unwrap().is_ok(), "the disk is healthy");
        assert_eq!(write_rx.await.unwrap().unwrap(), (0, 64));
        assert_eq!(
            file.sync_count(),
            1,
            "an accepted barrier over Relaxed bytes must have synced them"
        );
    }

    /// The same lie, one step later: the batch has already gone out on the
    /// timer, so the explicit flush finds nothing pending - and answers `Ok`
    /// over bytes that are still only in the page cache.
    #[tokio::test]
    #[ignore = "opens in the commit that makes an explicit flush a real barrier"]
    async fn an_explicit_flush_over_an_empty_batch_still_syncs_earlier_relaxed_bytes() {
        force_per_file();
        let dir = TempDir::new().unwrap();
        let file = make_file(&dir);
        let gc = GroupCommitter::start(file.clone(), 0).await;

        // The ack only arrives once the containing batch has been flushed, so
        // by here the timer has already taken it - with no sync, because every
        // entry in it asked for Relaxed.
        gc.append(vec![2u8; 64], Durability::Relaxed).await.unwrap();
        assert_eq!(
            file.sync_count(),
            0,
            "fixture: a Relaxed batch is not supposed to sync"
        );

        gc.flush().await.unwrap();
        assert_eq!(
            file.sync_count(),
            1,
            "a barrier over an empty batch must still sync what an earlier Relaxed batch left"
        );
    }

    /// And when that barrier sync is the thing that fails, the caller hears
    /// about it - the same contract as a batch sync failure.
    #[tokio::test]
    #[ignore = "opens in the commit that makes an explicit flush a real barrier"]
    async fn a_barrier_that_could_not_sync_reports_the_failure() {
        force_per_file();
        let _counter = counter_guard().await;
        let before = flush_failures();
        let dir = TempDir::new().unwrap();
        let file = make_file(&dir);
        let key = crate::failpoint::file_key(&file);
        let gc = GroupCommitter::start(file.clone(), 0).await;

        gc.append(vec![3u8; 64], Durability::Relaxed).await.unwrap();
        crate::failpoint::arm_at(crate::failpoint::CommitFailpoint::PerFileBatchSync, key);
        let result = gc.flush().await;
        crate::failpoint::disarm_at(crate::failpoint::CommitFailpoint::PerFileBatchSync, key);

        assert!(
            crate::failpoint::fired_at(crate::failpoint::CommitFailpoint::PerFileBatchSync, key),
            "the sync failpoint never fired: the test proved nothing"
        );
        assert!(
            result.is_err(),
            "a barrier whose sync failed answered Ok: {result:?}"
        );
        assert_eq!(
            flush_failures(),
            before + 1,
            "a barrier that could not land must be counted exactly once"
        );
    }

    /// A write that never reached the disk must not come back as a barrier.
    ///
    /// SD-1: an acknowledged `flush()` means every record submitted before it
    /// is on stable storage. A failing disk must not be able to produce an
    /// `Ok(())` here.
    #[tokio::test]
    async fn a_flush_over_a_batch_whose_write_failed_reports_the_failure() {
        force_per_file();
        let _counter = counter_guard().await;
        let before = flush_failures();
        let dir = TempDir::new().unwrap();
        let file = make_file(&dir);
        let key = crate::failpoint::file_key(&file);
        crate::failpoint::arm_at(crate::failpoint::CommitFailpoint::PerFileBatchWrite, key);

        let (tx, _task) = queued_committer(file.clone(), 0);
        let (write, write_rx) = write_msg(vec![7u8; 64], Durability::Kernel);
        let (flush, flush_rx) = flush_msg();
        tx.send(write).unwrap();
        tx.send(flush).unwrap();

        let flush_result = flush_rx.await.unwrap();
        let write_result = write_rx.await.unwrap();
        crate::failpoint::disarm_at(crate::failpoint::CommitFailpoint::PerFileBatchWrite, key);

        assert!(
            crate::failpoint::fired_at(crate::failpoint::CommitFailpoint::PerFileBatchWrite, key),
            "the write failpoint never fired: the test proved nothing"
        );
        assert!(
            write_result.is_err(),
            "the waiter was told its record landed: {write_result:?}"
        );
        assert!(
            flush_result.is_err(),
            "flush answered Ok over a batch whose write failed: {flush_result:?}"
        );
        assert_eq!(
            file.sync_count(),
            0,
            "a batch that never wrote must not have been synced"
        );
        assert_eq!(
            flush_failures(),
            before + 1,
            "a flush that could not land must be counted exactly once"
        );
    }

    /// The bytes are on the disk; the proof that they survive is not. That is
    /// still a failed barrier, and the offset must not move.
    ///
    /// SD-1 again, for the half an environmental fault cannot express: the
    /// write succeeds and the sync does not.
    #[tokio::test]
    async fn a_flush_over_a_batch_whose_sync_failed_reports_the_failure() {
        force_per_file();
        let _counter = counter_guard().await;
        let before = flush_failures();
        let dir = TempDir::new().unwrap();
        let file = make_file(&dir);
        let key = crate::failpoint::file_key(&file);
        crate::failpoint::arm_at(crate::failpoint::CommitFailpoint::PerFileBatchSync, key);

        let (tx, _task) = queued_committer(file.clone(), 0);
        let (write, write_rx) = write_msg(vec![0xEEu8; 64], Durability::Kernel);
        let (flush, flush_rx) = flush_msg();
        tx.send(write).unwrap();
        tx.send(flush).unwrap();

        let flush_result = flush_rx.await.unwrap();
        let write_result = write_rx.await.unwrap();
        crate::failpoint::disarm_at(crate::failpoint::CommitFailpoint::PerFileBatchSync, key);

        assert!(
            crate::failpoint::fired_at(crate::failpoint::CommitFailpoint::PerFileBatchSync, key),
            "the sync failpoint never fired: the test proved nothing"
        );
        assert!(
            write_result.is_err(),
            "an unsynced record was acked as durable: {write_result:?}"
        );
        assert!(
            flush_result.is_err(),
            "flush answered Ok over a batch that was never synced: {flush_result:?}"
        );
        assert_eq!(
            flush_failures(),
            before + 1,
            "a flush that could not land must be counted exactly once"
        );
        // The write DID happen - this is the sync-only failure - and the
        // offset must not have advanced over bytes nothing proved durable.
        assert!(
            file.pread(0, 64).await.unwrap().iter().all(|&b| b == 0xEE),
            "fixture: the write itself was supposed to succeed"
        );
        let (write2, write2_rx) = write_msg(vec![1u8; 32], Durability::Kernel);
        let (flush2, flush2_rx) = flush_msg();
        tx.send(write2).unwrap();
        tx.send(flush2).unwrap();
        assert!(
            flush2_rx.await.unwrap().is_ok(),
            "the disk is healthy again"
        );
        assert_eq!(
            write2_rx.await.unwrap().unwrap().0,
            0,
            "the offset advanced over bytes no sync ever proved durable"
        );
    }

    /// The flush a committer runs on the way down has no caller left to tell,
    /// which is exactly why it used to be discarded. It is still the last
    /// barrier the store gets.
    ///
    /// SD-2: a shutdown that could not land its final batch must leave a
    /// trace. Silence is indistinguishable from success.
    #[tokio::test]
    async fn the_last_flush_on_the_way_down_is_counted_not_discarded() {
        force_per_file();
        let _counter = counter_guard().await;
        let before = flush_failures();
        let dir = TempDir::new().unwrap();
        let file = make_file(&dir);
        let key = crate::failpoint::file_key(&file);
        crate::failpoint::arm_at(crate::failpoint::CommitFailpoint::PerFileBatchWrite, key);

        let (tx, task) = queued_committer(file.clone(), 0);
        let (write, write_rx) = write_msg(vec![3u8; 48], Durability::Kernel);
        tx.send(write).unwrap();
        // Closing the channel IS the shutdown: the task flushes what is left
        // and returns.
        drop(tx);
        task.await.unwrap();
        let write_result = write_rx.await.unwrap();
        crate::failpoint::disarm_at(crate::failpoint::CommitFailpoint::PerFileBatchWrite, key);

        assert!(
            crate::failpoint::fired_at(crate::failpoint::CommitFailpoint::PerFileBatchWrite, key),
            "the write failpoint never fired: the test proved nothing"
        );
        assert!(
            write_result.is_err(),
            "the last waiter was told its record landed: {write_result:?}"
        );
        assert_eq!(
            flush_failures(),
            before + 1,
            "the shutdown flush failed and nothing counted it"
        );
    }

    /// The other half of the contract: a flush that DID land still says so,
    /// and does not tick the failure counter. Without this a fix that always
    /// answers `Err` would pass every test above.
    #[tokio::test]
    async fn a_flush_that_landed_still_answers_ok() {
        force_per_file();
        let _counter = counter_guard().await;
        let before = flush_failures();
        let dir = TempDir::new().unwrap();
        let file = make_file(&dir);

        let (tx, _task) = queued_committer(file.clone(), 0);
        let (write, write_rx) = write_msg(vec![9u8; 128], Durability::Kernel);
        let (flush, flush_rx) = flush_msg();
        tx.send(write).unwrap();
        tx.send(flush).unwrap();

        assert!(
            flush_rx.await.unwrap().is_ok(),
            "a healthy batch must still answer Ok"
        );
        assert_eq!(write_rx.await.unwrap().unwrap(), (0, 128));
        assert!(file.sync_count() >= 1, "Kernel must issue a flush");
        assert_eq!(
            flush_failures(),
            before,
            "a flush that landed must not tick the failure counter"
        );
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
