# ADR: durability model for the incremental-insert flush

Status: **accepted** (branch `feat/incremental-insert`, Phase 2b).
Context: extends [incremental-insert-design.md](incremental-insert-design.md).

## Context

Phase 2b adds `flush`: when the in-RAM L0 buffer (`delta`) fills, its vectors are
bulk-built into an immutable on-disk Vamana **run** and appended to `runs`; L0 is
cleared. Search already walks the runs (Phase 1b). The open question is durability:
what is the source of truth, and what survives a crash at the worst moment?

Quality attributes in tension: **durability** (no acknowledged write is ever
lost), **crash-safety** (a crash mid-flush leaves a consistent index),
**simplicity** (this is the one data-critical path — minimise the surface that
can corrupt data), and **restart cost** (defer-able).

## Decision

**The delta WAL (`delta.log`) stays the single source of truth for every
un-consolidated write. A flush is a pure in-process optimisation; it does NOT
truncate the WAL.**

- `insert` appends to the WAL and to L0 (unchanged).
- `flush` bulk-builds a run from L0's vectors into a fresh sub-directory, opens
  it, moves its `Segment` into `runs`, and clears L0 in RAM. **The WAL is left
  intact.**
- `consolidate` folds `base ∪ runs ∪ delta` into one fresh base, deletes the run
  directories, and only then truncates the WAL (as it does today for base+delta).

### Crash-safety, case by case

- **Crash during a flush** (run dir half-written): on restart the partial run
  dir is ignored; the WAL still holds every vector. No loss, no corruption.
- **Crash after a flush, before consolidate**: the run dirs exist but are not yet
  discovered on open (see "deferred" below); the WAL replays all of them back
  into L0. The data is intact — it is just served from L0 again until the next
  flush. No loss.
- **Crash during consolidate**: unchanged from today — the rebuild writes to the
  graph files and only truncates the WAL after a successful reopen; a crash
  before that leaves the old graph + full WAL, which replays cleanly.

## What we deliberately do NOT build yet (deferred to Phase 4)

- **Run discovery on `open`.** Flushed runs are not reloaded after a restart;
  the WAL replays their vectors into L0 instead. Correct but loses the run
  optimisation across a restart until they re-flush. Wiring run dirs into `open`
  is a persistence optimisation, not a correctness fix, so it waits.
- **Leveled compaction of runs** (Phase 3). Phase 2b flushes into same-size runs
  with no merging; the gate measures the resulting search fan-out and tells us
  whether leveling is needed to hit the ≤2× latency target.

## Consequences

- **Positive:** the durability guarantee is unchanged from the proven
  delta-WAL + consolidate model — the data-critical path gains no new way to lose
  data. Runs are a throwaway, rebuildable cache.
- **Negative:** the WAL grows with all un-consolidated writes (as the delta did
  before), and restart re-absorbs them into L0 (a known, bounded cost). Run dirs
  accumulate on disk until a consolidate cleans them.
- **Validation:** the `incremental_gate` covers recall through a flush/consolidate
  cycle and bounded latency; a crash/replay unit test covers the WAL path.

## Alternatives rejected

- **Truncate the WAL on flush, make runs the durability source.** Rejected: it
  moves the durability boundary onto the new, less-proven run-build path, and a
  crash mid-flush could lose the L0 batch. Not worth the saved WAL bytes.
- **In-place insert into the base graph (FreshVamana).** Rejected earlier:
  back-edge erosion collapsed plain recall to 0.31. Immutable runs avoid it.
