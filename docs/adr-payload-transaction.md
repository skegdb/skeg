# ADR: one commit point for a vector and its payload

Status: **accepted** (branch `fix/vector-payload-commit`).

## Context

`SKEG.VSET name id vector payload` writes three things:

1. the vector, into the engine's delta and its WAL;
2. the payload's fields, into the in-RAM postings a filtered search reads;
3. the payload blob itself, into the shard's KV vLog.

They used to happen in that order, and each step after the first could fail on
its own. The result was reported to the client as a failed write while the
vector it had already published stood: a search returned a row the client had
been told did not exist, carrying either no payload or the payload of the value
it had just replaced. A `SKEG.VDEL` had the same shape - the tombstone is
durable before the blob is reclaimed, and a failure reclaiming it reported a
delete that had happened as a delete that had not.

Two more holes sat beside it.

A vindex name is reusable. Dropping `notes` and creating `notes` again is
ordinary, and the two shared a tenant, a name and an id space - so they shared
their blob keys too. The drop's blob sweep runs AFTER the catalogue has stopped
naming the index, deliberately, so that a failure there cannot fail a committed
drop; a crash or an error in that window left blobs behind, and the next
incarnation served them as its own. `sweep_payload_blobs` documented the hazard
and could not close it.

And `SKEG.VMSET` answered n writes with one integer. Its body awaited a
`JoinSet` with `??`, and a `JoinSet` aborts its outstanding tasks when it is
dropped - so one malformed item cancelled its siblings mid-write, including a
task that had already committed and had not yet published its row in the owner
map. That row is durable, acknowledged by the engine, and unreachable.

## Decision

**The WAL record is the commit point of the pair, and the blob is staged before
it.**

A blob's key is `tenant | marker | generation | name | id | version`. Two facts
in there are new, and each closes one of the holes above:

- the **generation**, minted once by the coordinator at `VINDEX.CREATE`,
  broadcast to every shard and persisted in the registry (`SVI3`). A recreated
  name does not inherit the blobs of the index it replaced.
- the **row version** (the authoritative version from
  [`adr-vector-version.md`](adr-vector-version.md)). A blob belongs to one COPY
  of a row, so a blob can be staged for a write that has not committed without
  overwriting the blob of the value that write is replacing.

The order is therefore:

1. **Admit.** Everything that can refuse the write - the dimension, a superseded
   relocation, the tenant quota, memory admission - decided under the write
   lock, before a byte is staged. A refusal leaves nothing behind because
   nothing has been written.
2. **Stage.** The blob is written at the key of the version this write is about
   to take. No live row carries that version; a search walks live ids and a
   payload read is keyed by the row's own version, so the staged blob is
   unreachable. A payload-less overwrite carries the row's existing blob to the
   same place, so it still keeps the payload it had.
3. **Commit.** One append: `Insert{id, version, payload_ref}`. Before it neither
   half is reachable, after it both are. This is the only step allowed to fail
   the call.
4. **Reclaim.** The blob the commit superseded. Post-commit, so it logs and
   cannot fail the call.

## The durability contract

**Process death (a kill, a panic, an OOM): the pair is atomic, with no fsync.**
The blob and the WAL record both go through the OS buffer (`Durability::Relaxed`
for the blob, a plain append for the record), and a process that dies leaves
both in the kernel. The record either reached the file or it did not, and the
blob is unreadable until it does - so a restart sees the old row with the old
payload, or the new row with the new payload, and never a mixture.

**Power loss: one skew is reachable, and only one.** The blob is written before
the record, so the orders the device can commit them in are: neither, blob only,
or both. "Record only" cannot happen. So the worst state a power cut can produce
is **a vector whose payload blob did not survive** - never a vector wearing the
payload of the value it replaced, and never a payload with no vector. That row
reads back with no payload; the open-time reclamation counts and collects
nothing for it, because the blob it would have named is not there.

This is deliberately not stronger. Making the pair power-safe means an fsync per
blob, which on macOS is a device-wide barrier of about 7 ms and turns a 100k
bulk load from 31 s into roughly 13 minutes - and it would make the payload
strictly MORE durable than the vector it annotates, which is the wrong shape.
The vector's own durability checkpoint is `consolidate`; the payload's is the
same one.

## What a failed commit leaves, and who collects it

Prepare-before-commit's price is a blob at a version no row ever took. It cannot
be read, and nothing on the write path removes it - by design, since the write
path's remaining job after the commit point is to not fail.

The collector is the open. Inside the readiness barrier, once per shard, one
pass over the keyspace deletes every payload blob that no resident index names
live. Three kinds of garbage have the same shape and all go here:

- a blob staged for a commit that never landed;
- a blob a committed overwrite superseded whose reclamation did not run;
- a blob of an earlier incarnation of a name, or of an index this shard no
  longer holds, left by a drop whose sweep did not finish.

It belongs at open and nowhere else: the store is quiescent, the registry has
just been read, every live row's version is in hand, and no request has yet
staged a blob whose commit has not landed - which is the one state this must not
mistake for garbage.

A blob is kept when the index of its NAME is resident, its generation is that
index's, and its `(id, version)` is live. The name and the generation are both
facts the server holds; the tenant in the key is deliberately not consulted,
because a vindex name is a client-chosen string that can spell another tenant's
scope key, and recovering an owner by re-reading one deleted the blobs of a live
index.

**The pass is O(the shard's whole keyspace), unconditional and uncapped.** It is
proportional to the KV keys the shard holds, not to the blobs or the orphans.
Measured 2026-09-03 (release, macOS arm64, isolated behind an env var): under
the noise at 20k blobs, +220 ms on a 14,1 s open at 80k. Both of those opens are
already far outside the 14 ms / 2,1 s cold-start budget for reasons that predate
this - the vLog recovery - so the pass is invisible rather than free, and the
slope between those two points has not been measured. If it ever matters, the
answer is an index on blob keys, not a return to walking `live_ids`: that cannot
see an index which will not open.

## Compatibility

A store written before generations existed reads as `IndexGeneration::LEGACY` -
not as an error, and not as a freshly minted one, because an index recorded by
an `SVI2` registry HAS no incarnation and inventing one would move its blobs out
from under it. A legacy-generation index reads the pre-generation key
(`tenant | \x00vp | name | id`) when the new key misses, which is how such a
store keeps answering; an index created since never pays that second lookup. Its
own new writes go to the new key, so it gains the transaction as it is written
to.

`payload.idx` carries the generation too, and is refused to another one. The log
position it was already stamped with says which LOG the file describes, not
which INDEX, and the `vindex-<name>/` directory outlives a drop that failed
partway.

## Consequences, stated

- **A payload-less overwrite now copies the blob forward.** It has to: the row's
  new version has a new key, and leaving the blob at the old one would silently
  drop the payload of a row nobody asked to change. The copy is skipped for a
  row the shard does not already hold - every insert and every arriving
  relocation - so the bulk path pays nothing. A CROSS-SHARD payload-less
  overwrite still loses the payload, exactly as it did before this change: the
  blob lives in the old owner's vLog and the new owner has no way to reach it.
- **`PayloadRef::Unchanged` is now only written by the unversioned engine API**
  and by V1/V2 records. A server write says `Blob(version)` or `Cleared`, so the
  record states what the row's payload is rather than declining to say.
- **`payload_ref_of` reaches the delta and its flush staging, not the whole
  index.** A fold takes a row into a segment and no segment column holds a
  payload reference, so the answer goes back to `Unchanged`. That is the honest
  reach: the window a crash leaves behind is the window a recovery has to
  reconcile.
- **Both point paths take the row stripe now, routed or not.** A VSET stopped
  being one shard message the moment it staged, committed and reclaimed with
  awaits in between, and a shard runs its requests concurrently.
- **`SKEG.VMSET` replies with an array of n**, `+OK` or that item's error, in
  request order. This is a wire change. Nothing across items is atomic and
  nothing pretends to be.

## Rejected

- **An fsync per blob.** See the contract above: it is 13 minutes per 100k
  vectors, and it makes the annotation more durable than the thing it annotates.
- **A blob key without the version, plus a staging namespace promoted after the
  commit.** The promotion is a second write, and a crash between the commit and
  the promotion leaves a new row wearing the old payload - which is the state
  this whole change exists to remove.
- **A persisted payload-reference column per segment**, mirroring
  `versions.bin`. It would let `payload_ref_of` answer for folded rows too, but
  nothing needs that answer: the read path derives the key from the row's live
  version, which the version column already persists. A second column with no
  reader is a second thing to keep in lockstep.
- **Two-phase commit across shards for VMSET.** For a command whose reason to
  exist is throughput, and whose per-item checks (quota, admission, dimension)
  are already per item.
