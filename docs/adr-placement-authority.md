# ADR: one placement authority per index

Status: **accepted** (branch `fix/b0-placement-authority`, audit 20 §B0).

## Context

A routed vindex keeps an `id -> (primary, replica, version)` owner map on the
coordinator. It is derived state - a reopen rebuilds it by asking every shard
for its live ids - and it is what a point op reads to decide which shard holds
a row, and writes back to record where it put one.

Three things wrote it, and nothing ordered them:

- `ShardSet::vset` (routed) picks an owner from the router, writes the new copy,
  then **publishes** `id -> (owner, None, version)`;
- `ShardSet::vdel` read the map to pick a shard, read it again for the row's
  version, again for the replica slot, and on success **removed** the entry -
  four independent acquisitions with shard round trips between them;
- `rebuild_owner_maps` scans every shard with one `await` per shard and then
  **replaces the whole map** with `owners.write().insert(name, map)`.

The first two take the id's owner stripe. The third takes nothing. So a publish
whose scan predates a point op silently undoes that op's record:

- **(a) VDEL lies.** `point_shard` reads the old map and `known` reads the new
  one. The delete then goes to the shard the OLD map named, which no longer
  holds the row, answers `Existed(false)`, and because `existed` is false the
  map entry is never removed. The client is told the id did not exist while the
  row stays live and searchable. Audit 18 reproduced this on real rows in
  30-70% of runs.
- **(b) VSET is lost.** The rebuild has already scanned shard 0; a VSET commits
  on shard 0 and publishes its entry; the rebuild scans shard 1 and then
  overwrites the map with its pre-scan copy. The entry is gone, reads route to
  the shard the VSET's own post-commit cleanup emptied, and an acknowledged
  write is unreadable.
- **(c) DROP under-credits.** `vindex_drop` counts logical rows as
  `owners[name].len()`. On a first reshard that map named only the rows the loop
  had moved so far, so the tenant stayed charged for the rest.

`docs/adr-vector-version.md` already states that "the coordinator defers to the
shard `point_shard` would place it on ... that shard checks
`backend.contains(id)` under its own write lock". That was true only when no
placement change was running underneath. This ADR is what makes it true.

## Decision

**One placement authority per index, and it is a lock, not a hope.**

`ShardSetInner::placement: Mutex<HashMap<String, Arc<tokio::sync::RwLock<u64>>>>`
- one `RwLock` per SCOPED index name, guarding a `u64` epoch.

- **Exclusive** for the whole per-name body of a rebuild (scan AND publish) and
  for `drop_router_state`.
- **Shared for the whole decision-and-commit** of everything that reads the map
  to decide and writes it to record: routed `vset` (from `router()` to the map
  commit; the post-commit cleanup VDELs may release it), `vdel` (one guard
  across all four accesses), `vget`, `owners_of`, `check`, the cardinality read
  of `vindex_drop`, and the per-row body of `reshard` and `overlap`.
- **Per name, never global.** A global lock would make one tenant's reshard
  stall every other tenant's point ops (SD5). It is also why a reshard rebuilds
  only ITS index at the end of a run (`rebuild_owner_map(name)`) instead of
  every routed one.
- **Lock order: `placement(read|write)` -> `owner_stripe(name, id)` -> shard
  mailbox.** Never re-entered, never the other way round. `tokio::sync::RwLock`
  is FIFO, so a reader that waits on something a queued writer needs is a
  deadlock; what keeps the mailbox out of that cycle is that a shard worker
  never touches `placement`. A rebuild does hold the exclusive lock across n
  `LiveIds` round trips, so an unavailable shard delays it - bounded by the
  request path's own `Unavailable`, not by a cycle.
- **`ensure_owner_map` is hoisted above every stripe** and is the only thing
  that may take the exclusive lock on behalf of a point op. Double-checked:
  `contains_key` -> exclusive -> re-check -> rebuild -> publish -> bump epoch.
  Never with a guard in hand. `vset` used to call it AFTER taking the stripe,
  which meant holding one id's stripe across a scan of every shard.
- **The map is reachable only through accessors that take the guard as a
  parameter** (`point_shard_at(&self, _g: &PlacementShared, ..)`,
  `owner_entry_at`, `publish_owner_at`, `forget_owner_at`, `set_replica_at`,
  `owner_rows_at`, `owner_out_of_range_at`), so reading it without the authority
  does not compile. The one deliberate exemption is named
  `owner_primaries_unsynchronised` and has `vsearch` as its only caller.
- **`vsearch` stays outside the authority.** It merges an atomic snapshot per
  shard and uses the map only as a tie-break AFTER the version comparison, so
  the worst a concurrent publish can do is make it prefer a different SHARD's
  copy of the same id at the same version - never a different id and never a
  different value. Putting it inside would put a read permit on the hottest path
  in the system to buy nothing.
- **The epoch is internal.** Bumped on every publish, asserted equal across a
  reshard row's read and its publish (a `debug_assert`, which fails first if the
  guard is ever released mid-body), available as a tracing field. It is NOT
  exposed on the wire: that is a protocol change, and "placement changed, retry"
  does not fix (a) - the client has no way to know it should retry.
- **Not persisted.** The map is derived, and the authority is one process; the
  store lock is what guarantees there is only one.
- Reopen needs no special case: there is no map at `open`, and the first
  operation takes the exclusive lock with no reader to fight.

### What this deliberately is not

Distributed consensus. A sweep of all 256 owner stripes. An `ArcSwap` (it makes
the READ atomic and does nothing about the publish overwriting a commit). A
persisted map. MVCC, or a "placement changed, retry" error on the wire.
Unifying `routers`/`owners`/`versions` into one structure - worth doing, not
here.

## Security Definitions

Attacker: many connections of the same tenant, any interleaving of
VSET/VDEL/VGET/VSEARCH/RESHARD/OVERLAP/DROP over their own indexes. They cannot
write the files, touch another tenant, inject into a shard mailbox, or restart
the process.

| | Goal | Severity | Before | After |
|---|---|---|---|---|
| **SD1** | An acknowledged VSET is never lost to a placement change. | Critical | not met (b) | **met** |
| **SD2** | VDEL never answers `false` for a row that stays live and searchable. | Critical | not met (a) | **met** |
| **SD3** | No row becomes unreachable because of a concurrent placement change. | High | not met | **partial**: a crash mid-reshard still leaves two copies, resolved by `max(version)` at the next rebuild. |
| **SD4** | A DROP during a reshard credits the tenant every logical row. | Medium | not met (c) | **met** |
| **SD5** | One tenant's reshard does not block another tenant's point ops. | Medium | met by accident | **met by construction**: the lock is per scoped name; a global lock violates it. |

Tests: `crates/skeg-server/tests/placement_authority.rs`, plus
`failpoint::tests::a_gate_parks_and_releases` for the gate the first four lean
on.

## What this does not cover

- **The exclusive window is O(rows x shards)** - one `LiveIds` round trip per
  shard plus the map build, and the authority is held across all of it.
  Measured 2026-09-04, release, aarch64 macOS, dim 16, disk backend, 2 shards,
  5 samples, idle machine, against the same harness on the a67258c baseline:

  | rows | baseline p50 (range) | with the authority p50 (range) |
  |---|---|---|
  | 20 000 | 799 µs (776-896) | **829 µs** (815-893) |
  | 200 000 | 10,34 ms (10,02-11,42) | **10,14 ms** (9,34-11,05) |

  The authority costs the lock, not the scan: +30 µs at 20k and nothing
  measurable at 200k. But the WINDOW at 200k is ~10 ms, and every reader of
  that index waits behind it. Above that size it wants an incremental rebuild;
  that is a follow-up, and this number is the argument for it. (An earlier
  round taken while other builds saturated the machine read 1,3/18,3 ms
  branch against 1,0/13,7 ms base - contention, not the lock, and recorded
  here so the quiet numbers are not the only ones on file.)

- **Point ops pay a read permit.** Same conditions, 20 000 routed VSETs then
  20 000 VDELs over the same index: VSET p50 43,7 -> **41,5 µs**, p99
  115,5 -> **87,2 µs**; VDEL p50 12,0 -> **11,5 µs**, p99 63,3 -> **29,4 µs**.
  Parity, inside the noise: an uncontended `RwLock` read is not measurable
  against the shard round trip the op already makes.
- **FIFO.** A queued rebuild blocks new readers for its whole duration. That is
  the point - it is what makes the authority single - but it is latency a very
  large index will feel, and it is the same number as above.
- **A crash between a move's destination write and its source delete** still
  leaves two copies; `max(version)` at the next rebuild picks the right one.
- **A tombstone's version is still lost at a fold** (see
  `docs/adr-vector-version.md`).
- **The map stays derived and process-local.**
- **`vsearch` may disagree with a point read about which SHARD served a row**
  for the duration of a rebuild. Never about the id, never about the value.


## Addendum (audit 24, 2026-09-04)

Two defects in the registry that hands out the per-index lock, both fixed:

- A dropped index's authority. `drop_router_state` removed the registry entry
  while holding the exclusive guard; a point op already queued on that lock
  was served on an orphan and `publish_owner_at` recreated the owner map of an
  index that no longer existed. Now every acquisition, shared or exclusive,
  re-checks under the registry mutex that the lock it holds is still the
  registry's entry for that name: a shared holder whose entry is gone answers
  as a dropped index does; an exclusive one resolves again.
- The registry grew without bound: any point op on any name created an
  entry, so `VGET nope-<n>` was an authenticated way to leak. The last shared
  guard on a name with no router and no owner map removes the entry (checked
  under the registry mutex, where resolvers clone, so a concurrent resolver
  keeps it alive). Entries now number at most the routed indexes plus the
  point ops in flight.
