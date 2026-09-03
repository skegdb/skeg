# ADR: what an acknowledged flush means

Status: accepted, 2026-09-03. Supersedes nothing; it writes down a contract
that existed only as an intention.

## Context

skeg commits writes in batches. Two committers do it, picked at open by
`skeg_platform::resolve_durability_model`:

- `PerFileCommitter` (`group_commit.rs`), one background task per segment
  file, one `write_vectored_at` and one sync per batch. The `PerFile` model,
  Linux's default.
- `SharedCommitter` (`shared_committer.rs`), one process-wide task that
  buckets a batch by file, writes each file once, and issues **one** sync for
  the whole batch. The `DeviceGlobal` model, macOS's default, where
  `F_FULLFSYNC` is a device-wide barrier and per-shard syncs would serialise
  on the hardware.

Both had the same defect. `flush_batch` returned `()`, and the `Flush` message
answered `Ok(())` unconditionally. `VLog::flush()` therefore reported success
over a batch whose write returned `ENOSPC` or whose sync returned `EIO`, and
the shard's shutdown discarded the result outright (`let _ = vlog.flush()`).
Individual waiters were told the truth - that part was already right - but the
caller who asked for a *barrier* was not, and nothing outside the process
counted the failure.

A store may lose a write to a failing disk. It may not say the write is
durable when it is not.

## Decision

1. `flush_batch` returns `io::Result<()>` in both committers, and `Msg::Flush`
   forwards it verbatim. `Ok(())` means: every file this batch wrote was
   written, and the sync covering them succeeded at the strongest durability
   any entry in the batch asked for.
2. An **empty** batch flushes to `Ok(())`. There was nothing to make durable;
   everything before it was answered by its own flush. (See "What this does
   not fix", point 1, for the case this leaves open.)
3. The shared committer **aggregates per file**. One file's failed write does
   not fail the other files' waiters, and does not buy the flusher an
   `Ok(())`: the flush answers with an error naming how many of the batch's
   files did not commit and why. The `ErrorKind` survives when every failure
   shares one - a batch that failed entirely on `StorageFull` still reaches a
   caller looking for ENOSPC - and collapses to `Other` when they differ,
   because no single kind would then be the truth.
4. Every failed flush ticks `skeg_vlog_flush_failures_total`, wherever it came
   from: an explicit `flush()`, a full batch, the timer, or the last flush as
   the inbox closes. Three of those four have no caller to answer, so for them
   the counter is the whole signal. It reaches `/metrics` and `SKEG.STATS`
   like every other counter. There is no healthy value above zero.
5. The shard's shutdown flush logs at ERROR instead of discarding its result.

Nothing about *which bytes go where* changed. The offset still refuses to
advance past a write whose sync failed, waiters are still answered
individually and identically, and the durability tiers are untouched.

## What this does not fix

Written down because a barrier that is honest about three cases and silent
about a fourth is worse than one nobody trusts.

1. **A flush with nothing pending syncs nothing.** If every entry in the
   preceding batch asked for `Durability::Relaxed`, the batch was written and
   never synced, and a later `flush()` over an empty batch answers `Ok(())`
   having issued no barrier at all. Bounded in practice: `Relaxed` is used by
   compaction relocations only, and `compact_segment` issues its own
   `sync_durable` per destination file. Every write a client can ask for is
   `Kernel` or `Power`, which sync in their own batch. It is still a promise
   `VLog::flush()` makes and does not keep for a `Relaxed` writer.

2. **The shared committer syncs one file, not every file it wrote.** The
   single sync is the entire point of that committer on a `DeviceGlobal`
   platform, and it rests on two properties that are asserted nowhere: that
   `F_FULLFSYNC` flushes the device's write cache rather than one file's, and
   that `F_NOCACHE` (applied to every `PlatformFile` at open) has already
   pushed the other files' bytes past the page cache. The `Power` tier is
   plausible on that reasoning. The `Kernel` tier is thinner: it calls
   `sync_data`, which is `fsync` on macOS, on whichever file in the batch
   happened to be written first. Neither is verified by a test, and the sync
   target is chosen by `HashMap` iteration order.

3. **A failed shutdown flush does not reach an exit code.** `run_shard` is a
   thread with a `block_on` that nothing joins, so the error is logged and
   counted and goes no further. A supervisor watching the exit status of
   `skeg` learns nothing.

4. **No client-visible barrier exists on either wire.** RESP3 has no `FLUSH`
   or `SYNC` command; the native protocol reserves `Op::Flush` (0x82) but the
   handler does not dispatch it, so it answers `InvalidRequest: unsupported
   op`. That is honest - nothing lies to a client - but it means a client that
   wants a barrier has no way to ask for one, and this work is not yet
   observable from outside the process except through the counter.

## Testing

The failure half of a commit cannot be driven by environmental injection: the
write and the sync go through the same descriptor, and the interesting case is
the one where the write lands and the sync does not. `skeg-core` therefore has
its own typed failpoint registry (`src/failpoint.rs`), shaped like
`skeg-server`'s, keyed on the address of the `Arc<PlatformFile>` the batch is
about to write. Two live files never share an address, so isolation between
two tests in one binary does not depend on anyone choosing unique names, and
the site needs no extra plumbing: it already holds the file. Every test
asserts the point actually fired.

The tests drive `committer_task` and `committer_loop` over a channel whose
messages are all queued before the task exists. That is what makes them
deterministic rather than a race against the 200 microsecond batch timer.
