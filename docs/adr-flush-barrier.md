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
2. An explicit `flush()` is a **barrier**, not a batch. Both committers count
   the bytes they have written that no sync has covered, and a barrier syncs
   while that count is above zero, at `sync_durable`, whatever tier the batch
   in front of it asked for. `Durability::Relaxed` is the only tier that can
   leave anything there - every other one syncs before it acks - and it is a
   client-reachable tier, not a compaction-only one (see below). A barrier over
   an empty batch is therefore only `Ok(())` once nothing is owed; before this,
   it was `Ok(())` unconditionally, which confirmed a barrier that had not
   happened.

   The shared committer keeps the count **per file**, because its batch path
   syncs exactly one of the files it wrote - that is the point of it on a
   device-global platform - so every other file it wrote keeps its count and is
   owed a sync by the next barrier. A batch sync clears only the file it was
   demonstrably called on. On Apple the extra syncs a barrier then issues are
   redundant, because `F_FULLFSYNC` does flush the device; the premise is not
   trusted here because this is the one place where believing it and being
   wrong loses data.

   The cost falls only on `flush()`, which the server calls once per shard at
   shutdown (`shard.rs:3505`): one `F_FULLFSYNC` per file still carrying
   unsynced bytes, about **4.6 ms** each on APFS / M-series (measured by audit
   23, 2026-09-04, 60 iterations). The batch path is unchanged, so nothing on
   the write path pays for this.
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

   One thing deliberately does **not** tick it: an entry whose file was
   detached before the batch reached it. That detach comes from
   `EntryInner::drop`, so it means the append was cancelled by its own caller -
   a client hanging up - and not that a disk failed. It would otherwise let a
   client raise a durability alarm by closing a connection. It is counted on
   `skeg_vlog_commit_orphaned_entries_total` instead, which is benign and says
   so.

6. `SKEG_DURABILITY_MODEL=device-global` is honoured on Apple and **refused
   everywhere else**, with a `warn!` naming the reason. It is a claim about the
   hardware, not a preference: see the platform section below.
5. The shard's shutdown flush logs at ERROR instead of discarding its result,
   and the committers' own shutdown flush is a barrier, not a batch: it is the
   last one the store gets.

Nothing about *which bytes go where* changed, and the durability tiers
themselves are untouched: the offset still refuses to advance past a write
whose sync failed, and waiters are still answered individually and
identically. What changed is when an *explicit* flush syncs, which is the
barrier's meaning and not the tier's.

## The platform, stated correctly

An earlier draft of this ADR reasoned from two claims about the platform that
are false in source. Both were load-bearing, so they are set out here rather
than corrected in passing.

**On Apple, `sync_data` is `F_FULLFSYNC`, so `Kernel` and `Power` are the same
call.** `PlatformFile::sync_data` (`skeg-platform/src/file.rs:268`) calls
`std::fs::File::sync_data`, and std's `os_datasync` is
`fcntl(fd, F_FULLFSYNC)` under `#[cfg(target_vendor = "apple")]`
(`library/std/src/sys/fs/unix.rs:1409-1412` in this toolchain). It is not the
cheaper `fsync` the tier names suggest. Measured on APFS / M-series (audit 23,
2026-09-04, 60 iterations): `sync_data` median **4941 µs**, `sync_durable`
median **4437 µs** - one primitive, not two tiers. So on Apple a `Kernel`
write already costs and already buys power-loss durability, and the concern
about the `Kernel` tier in the shared committer **does not exist there**.

**The single-sync design is Apple-only, and is now enforced.** The shared
committer issues one durability call for a batch spanning several files. That
is a barrier for all of them because `F_FULLFSYNC` asks the drive to flush its
whole buffered cache, and because `F_NOCACHE` - applied to every
`PlatformFile` at open, and `#[cfg(target_os = "macos")]`
(`file.rs:354-365`) - has already taken the other files' bytes out of the page
cache. Neither premise exists on Linux: `fsync(2)` transfers the data "of the
file referred to by the file descriptor fd", and skeg applies no `F_NOCACHE`
and no `O_DIRECT` there. On Linux the same code would ack every file but one
as durable over dirty pages nothing flushed - lost on power failure and lost
on a kernel panic.

`DurabilityModel::DeviceGlobal` is the compile-time default only on macOS, so
this was never the shipped Linux behaviour; it was reachable by an operator
setting `SKEG_DURABILITY_MODEL=device-global`, which the module documented as
a supported override. `resolve_model_from`
(`skeg-platform/src/durability.rs`) now refuses that value off Apple and warns.
`per-file` stays selectable everywhere: it promises less than the platform can
do, never more.

## What this does not fix

Written down because a barrier that is honest about three cases and silent
about a fourth is worse than one nobody trusts.

1. **A failed shutdown flush does not reach an exit code.** `run_shard` is a
   thread with a `block_on` that nothing joins, so the error is logged and
   counted and goes no further. A supervisor watching the exit status of
   `skeg` learns nothing. Name it in the shutdown runbook:
   `skeg_vlog_flush_failures_total` is the signal, not the exit status.

2. **No client-visible barrier exists on either wire.** RESP3 has no `FLUSH`
   or `SYNC` command; the native protocol reserves `Op::Flush` (0x82) but the
   handler does not dispatch it, so it answers `InvalidRequest: unsupported
   op`. That is honest - nothing lies to a client - but it means a client that
   wants a barrier has no way to ask for one, and this work is not yet
   observable from outside the process except through the counter.

3. **The shared committer's batch path still syncs one file.** Unchanged, and
   deliberately: on Apple, where that committer is now the only place it can
   run, one `F_FULLFSYNC` covers the device, and syncing all four files of a
   batch instead measured **4.2x** slower (1 sync 4600 µs vs 4 sequential
   19421 µs; 4 in parallel 10008 µs; audit 23, APFS, 4 files x 4 KiB, 60
   iterations, 2026-09-04). The files a batch sync did not name keep their
   dirty count and are synced by the next barrier, so nothing rests on the
   premise at the point where being wrong would lose data - only the *cost* of
   the redundant syncs does.

### Corrected here, from audit 23

An earlier revision of this document listed two further gaps and dismissed
both on reasoning that was wrong, which is worth recording because the
reasoning is the part that failed, not the code:

- It claimed a `Relaxed`-only flush was bounded because "`Relaxed` is used by
  compaction relocations only". It is not: `PAYLOAD_DURABILITY`
  (`skeg-server/src/shard.rs:1471`) is `Durability::Relaxed`, and it is the
  durability of every VSET payload blob and of four tombstone sweeps. A sweep
  is N deletes with no durable write behind them. Fixed, not accepted: see
  decision 2.
- It claimed the `Kernel` tier was "thinner" than `Power` because `sync_data`
  is "`fsync` on macOS". It is `F_FULLFSYNC` there. See the platform section.

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

The barrier itself is tested with `sync_count()` rather than a failpoint,
because the question is not "what did it answer" but "did a sync happen":
a `Relaxed` append followed by `flush()` must leave `sync_count() == 1`, both
while its batch is still pending and after the timer has already taken it, and
on the shared committer every file the batch left dirty must be covered.

The platform rule (`device-global` off Apple) is tested through
`resolve_model_from(raw, is_apple)`, which takes the platform as a parameter,
so both branches run on any host. A rule about what Linux may not select is
worth nothing if it can only be exercised on Linux; a `#[cfg(not(target_vendor
= "apple"))]` test then asserts the same thing of the real constant where it
compiles.
