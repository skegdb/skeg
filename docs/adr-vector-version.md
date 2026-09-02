# ADR: the authoritative vector version

Status: **accepted** (branch `fix/vector-copy-versioning`).

## Context

A vector id can have more than one physical copy at the same time, and every one
of those states is normal rather than exceptional:

- during a `SKEG.VINDEX.RESHARD`, a row exists on its old shard and its new one
  until the source copy is deleted;
- a boundary `overlap` deliberately keeps a second copy on the second-nearest
  shard;
- a routed overwrite writes the new copy, publishes it, and then deletes the old
  one **best-effort** - a failure there is logged, not reported, because the
  write is already committed and readable;
- inside one index, a fold has the same row in the delta, in a run and in the
  base at once.

Which copy is LIVE was inferred from position: the higher LSM layer, the lower
shard number, the order a hash map happened to be iterated in. Position is not
identity. A copy that moves keeps its position and loses its history, and the
engine then has no way to tell the value a write replaced from the value that
replaced it. Three concrete losses followed, all of them silent:

1. `reshard` collects a batch of rows and moves them one at a time with an
   `await` between each. A `vset` committing inside that window is acknowledged,
   and the reshard then writes the copy IT read over the top and points the owner
   map at it. On the run that pinned this, 63 of 240 overwritten rows came back
   as the value they had replaced.
2. `overlap` replicates a row a concurrent `vdel` removed. The delete takes both
   copies and the map entry while the replica is in flight; the replica lands
   afterwards on a shard nothing is left to clean up - findable by search,
   invisible to the map.
3. A reopen picks the primary by iteration order, so a restart promotes whichever
   copy of a duplicated row sits on the lower shard. Half the time that is the
   copy an overwrite replaced, and `vget` and `vsearch` then agree on it.

## Decision

**A vector version is a fact carried with the row, not inferred from where the
row is.**

`VectorVersion(u64)`, monotone per `(index, id)`:

- **allocated** by the coordinator on a user write (`vset`, `vdel`), under the
  id's owner stripe, strictly above the version already recorded for that row;
- **carried unchanged** by anything that only relocates a row: a reshard move,
  the delete on the far side of that move, a boundary replica, a fold, a flush, a
  runs merge. A relocation is the same copy in a different place, and giving it a
  new version would make it outrank the write that replaced it;
- **higher wins.** Equal versions fall back to the old positional rule, which is
  what makes this a no-op on data at rest;
- `LEGACY` is zero: every row written before this existed. It loses against every
  allocated version and ties with itself.

### The rules, in one place

1. A write - insert or delete - whose version is BELOW the version the index
   already holds for that row is dropped. Not an error: the caller is a
   relocation and the newer copy standing is the answer it wants.
2. A relocation re-reads the owner map under the row's stripe before acting, and
   skips the row when its version has advanced or its entry is gone.
3. A fold, a flush and a runs merge choose a survivor by max version, with layer
   order as the tie-break rather than the rule.
4. An owner-map rebuild names the copy with the highest version as the primary,
   and records the one it displaces as the replica so a later delete still
   reaches it.
5. A search ranks two copies of one id by version first, then by what the owner
   map calls live, then by score.

## Where it is stored

| Place | Shape | Persisted |
| --- | --- | --- |
| Delta WAL, `SKWL\x03` | `[op][id:8][ver:8][payload_ref][dim*4]`, delete `[op][id:8][ver:8]`, CRC32C-framed | yes |
| `versions.bin`, per segment | `n` u64 LE in graph row order | yes |
| `delta_ver` / `flushing_ver` / tombstones | in-RAM maps beside their rows | no (the WAL is) |
| `OwnerMap` | the primary copy's version | no, derived |
| Version allocator | one `AtomicU64` per index on the shard SET | no, derived |

The allocator lives on the set, not on a shard: a shard's counter only knows the
copies it holds, so a row moving to a shard that has never seen it would be
handed a version below the one it already carried and dropped as stale. It is
seeded past the highest version seen at every owner-map rebuild, and floored on
each allocation by the version already recorded for the row, so a value read off
disk can never be reissued.

`versions.bin` is published by the same rename that publishes the graph, so a
generation cannot be live without the column that describes it. An ABSENT column
means every row is legacy. A column of the WRONG LENGTH is an **error**, not a
fallback: `attr.bin` next door does fall back, and doing the same here would
bring an index back serving zeros - tie-breaking by shard number again - with
nothing anywhere saying so.

`payload_ref` is written as `Unchanged` by everything and read by nothing. It is
reserved in the record now so the vector/payload commit point does not have to
bump the format again.

## Migration

V1 and V2 WALs still open and decode as legacy. Promotion to V3 happens only
where the whole file is rewritten anyway - a flush's WAL compaction, or a fold -
and never at an open, which is only asked to read. Until then, a version written
into a V1 or V2 store is dropped by the encoding.

An index whose segments have no `versions.bin` reads back as all-legacy and
behaves exactly as it did; the first fold writes the column.

## What this does not cover

- **A tombstone's version is lost at a fold**, which drops the tombstone along
  with the rows it masked. A straggler arriving after that fold has nothing to
  lose against. Accepted: the alternative is keeping every delete for ever, and
  the stripe lock is what stops a straggler arriving that late.
- **Legacy rows tie with each other**, so a store that has never been written by
  a versioned client still resolves duplicates by shard number.
- **The flat backend** records a tombstone for a versioned delete of an id it has
  never held, but not for a legacy one - version zero could not win that
  comparison anyway.
- **Durability of the version is the durability of the write it belongs to.**
  The WAL is `Relaxed` (OS buffer, no fsync per record); a version is exactly as
  durable as the vector it describes, no more and no less.
