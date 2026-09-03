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

## Disk quota

`max_disk_bytes` is a tenant's hard limit on live on-disk KV bytes, and a KV
`SET` enforced it from the start. `stage_payload_blob` did not: it wrote
straight to the vLog with no limit, so an authenticated tenant could fill the
shared disk through `VSET`/`VMSET` payloads alone, unbounded by the number
that already bounded everything else it wrote.

A payload blob's key carries the same 16-byte tenant prefix a KV key does
(`payload_blob_key`, above), so `VLog::tenant_disk_bytes` already counted it -
the gap was that nothing REFUSED a write against that count. The fix reuses
the KV path exactly: `stage_payload_blob` takes the tenant's `max_disk_bytes`
and passes it to the same `VLog::set` builder (`.with_disk_limit`) a `SET`
already called, checked before anything is written
(`VLog::set_scoped`: "over-limit ... rejected ... BEFORE anything is
written"). **Quota = live KV bytes plus live blob bytes**, by construction:
one counter, one key format, one check, for both.

Reached from two call sites, mirroring `limit` (the vector-count quota)
exactly: `ShardSet::vset_with_disk_limit` and the per-item calls inside
`ShardSet::vmset_with_disk_limit`, both fed from the tenant's
`max_disk_bytes` at the RESP3 entry point. `vset`/`vmset` (no `_with_disk_limit`)
still exist, unenforced, for every caller with no limit to enforce: every
internal relocation, every existing test, and the native listener, which has
no pluggable tenant backend to read a limit from at all.

**The temporary physical margin.** Prepare-before-commit means a write is, for
a while, TWO physical keys: an overwrite's staged blob at the new version's
key while the one it supersedes is still resident (reclaimed only after the
commit, per above), or - on a crash or a refused reclaim - an orphan that
lives until the next open collects it. Both are real bytes on disk, and both
are counted, because the counter is physical, not logical: it cannot tell a
candidate or an orphan from a live blob, and it must not try to, because the
same reclamation path that frees them is what makes the count exact again.
The consequence is stated plainly rather than hidden: **a tenant sitting
exactly at its `max_disk_bytes` can see a payload-less overwrite refused for
the width of the staging-to-commit window**, because the carried-forward copy
needs room for both keys at once. This is not a bug to route around - doing
so would mean an overwrite could grow a tenant's resident bytes with nothing
counting them - it is the same trade the vector-count quota already makes for
`Move`/`Replica` (uncharged, because refusing an internal relocation can
strand a row mid-move) inverted: here the write IS the tenant's own, so it is
charged, and a tenant that operates payload-heavy workloads near its limit
should leave headroom for one blob's worth of margin. `VSET`/`VMSET` staged by
an internal relocation (a reshard move, a boundary replica) pass no disk
limit at all, for the same reason `Move`/`Replica` pass no vector-count limit:
they move or duplicate bytes the tenant's writes already paid for, and a
refusal would leave a shard's data placement stuck rather than a tenant's
disk usage smaller.

**Boundary replicas: a permanent, uncounted duplicate.** The reshard move
above does not duplicate - `CollectMoves` writes the destination and deletes
the source, verified on a single-shard reshard of 64 rows (`usage` unchanged
before/after) - but the OTHER internal relocation does. A boundary replica
(`ShardReq::Vset` with `effect: Replica`, `disk_limit: None`) writes a SECOND
physical copy of the row's blob on a second shard and never deletes the
first: the two stand side by side for as long as the row's margin keeps it
replicated, which is not a window like the overwrite one above - it is
indefinite, and it is real disk this tenant's quota does not see. The bound
has a shape, even though nothing enforces it: at most one extra blob per
row this shard's overlap boundary currently replicates, so the uncounted
total is bounded by (replicated rows) x (their blob sizes), not by the whole
tenant. Left open rather than closed here: fixing it means either charging
the replica's own tenant (which the write-path shape above argues against,
for the same reason `Move`/`Replica` are uncharged) or teaching the
reclamation/quota rebuild at open to walk replicas as well as primaries -
both bigger than this mandate's boundary.

**Concurrent writers of one tenant.** The check above and the counter update
happen in the SAME critical section inside `VLog::set_scoped` (no `await`
between them), so the bound is exact, not probabilistic: N writers racing
the same tenant's headroom can never jointly land it past `max_disk_bytes`,
whatever N is and whatever order they finish in. Before this, the check read
the counter and the update wrote it on either side of the write's own
`await`, so two writers of different keys could each read the same
pre-write total and both proceed - reproduced at 5.3x a 1.5-unit budget with
8 concurrent writers.

**The refusal is typed.** A disk-quota rejection reaches both wires as
`AdmissionError::DiskQuota { tenant, limit, needed }`, classified exactly
like its sibling `QuotaExceeded` (the vector-count quota): `Permanent`,
`ERR ...` on RESP3, `InvalidRequest` on native - a tenant at its own ceiling
is the request's fault, not the server's, and telling a client `Internal`
sends it looking for a bug that is not there. One place builds it
(`disk_quota_refused` in `shard.rs`) for both the payload blob path and the
KV `SET`/`APPEND` path, so the two cannot drift the way the RESP3 and native
wires once did before `admission.rs` existed.

**Refund.** There is no separate reservation counter to refund: the count IS
the physical bytes on disk, so "refund" is exactly the reclamation path this
ADR already describes - `VLog::del` decrements the same counter `VLog::set`
incremented, called by the post-commit cleanup (superseded blob), the open-time
reclamation (orphans, superseded blobs the cleanup failed to reach, a dropped
index's stragglers) and nothing else. A write refused before staging never
incremented it. Idempotent because `VLog::del` is: a key already gone is a
no-op, not a second decrement.

**Reopen.** The counter (`SharedTenantDisk`) lives in RAM. `VLog::open`
already rebuilds it from the recovered index before `ShardSet::open`'s
readiness barrier reclaims the shard's orphans - both steps predate this
change (`VLog::recover_tenant_disk`, the reclamation described above) and
needed no new scan: a payload blob is a KV key like any other, so the
existing rebuild already covers it byte for byte, and the reclamation that
runs after it (still inside the barrier, before `ready.send`) removes what a
crash left staged. No request is admitted until both have run.

**MSET.** `VLog::set_many` shared none of the above until audit/17 round 2:
it wrote its whole batch with no limit check at all - a deterministic
bypass, not a race. `set_many_with_disk_limit` closes it with the batch's
net delta (its sum, folding a duplicate key inside one batch to its LAST
occurrence, the same way the per-key write loop already does) checked and
reserved atomically, in the SAME critical section as `set_scoped` above,
before a single byte of the batch is written - so a batch that would cross
the limit writes NONE of its members, the same all-or-nothing contract
`set_many` already has for a crash. One caveat inherited, not introduced:
an `MSET` whose keys span shards was never atomic ACROSS shards (each
shard's portion is its own commit, sequentially), so the disk quota follows
the same shape - a batch split across two shards can have one shard's
portion admitted and written before the other shard's is checked, and
refused. A single-shard deployment, or a batch whose keys all hash to one
shard, gets true all-or-nothing for the quota exactly as it already did for
the write itself.

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
- **Nothing reads `payload_ref` today, and a future reader must not trust the
  records already written.** The field is encoded, replayed and readable, and
  its only callers are the tests that pin the round trip. The commit-point
  property does not come from it - it comes from the versioned key plus the
  live row's version, which is why the per-segment column was rejected above.
  So `Blob(v)` against `Cleared` currently has NO observable effect anywhere,
  which means the two have never been distinguished by anything that could have
  caught them being wrong. Whoever writes the first real reader has to treat
  existing records as unverified: `Cleared` in particular has never had to be
  true, only written.
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
