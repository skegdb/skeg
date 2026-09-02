# Changelog

All notable changes to the engine are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file tracks the engine and the multi-tenant server, both in this
repository.

## [Unreleased]

### `create_empty` refuses a directory that already holds an index

`DiskVamanaIndex::create_empty_with_tier` called `create_dir_all` and then
wrote an empty graph, vectors, `CURRENT` and WAL over whatever was there.
The server guards its own call site; embedders did not. `skeg-rigging-skeg`
0.1.3's `Tenant::open` decided whether a tenant existed from its metadata
sidecar, which only `flush` writes, so a tenant that inserted and never
flushed lost its vectors on the next open (reproduced: one vector, then
none). A tier file or a `CURRENT` pointer now makes `create_empty` fail
with `AlreadyExists`: a repeat of that mistake by any caller is an error,
not a silent wipe. `skeg-multi-tenant` requires `skeg-rigging-skeg` 0.1.4,
which reopens such an index and writes the sidecar at create, and pins
the scenario in `tests/reopen_without_flush.rs`.

The `open_scoped` perf gate timed the creation of a new tenant per
iteration; a durable create is four fsyncs (~10 ms on APFS) and could
never meet 5 ms - which is also why the gate only turned red once the
crate embedded the workspace engine instead of the fsync-less 0.1.3. It
now creates in the warm-up and times the reopen.

### A release is one gated transaction

`release.yml` used to run `publish-crates` and `build-binaries` side by
side, both waiting only on the tag-ancestry guard: a crate could reach
crates.io - irreversibly - while a binary build was failing. And
`docker-publish.yml` answered `v*` tags on its own, without that guard, so
a tag the release workflow rejected still produced `:<version>` and
`:latest` images.

Now the workflow is split into build and promote. guard -> test ->
{build-binaries, docker-build} produce nothing public: each tarball is a
run artifact (it used to be uploaded to the GitHub Release by its own
matrix job, so one architecture could publish while another failed), and
the image is pushed by digest only - untagged, unreachable by any tag.
Only once all of them succeeded does promotion start: `promote-release`
creates the GitHub Release with every tarball at once, `docker-promote`
binds the digests under `:<version>` / `:latest` (the Docker workflow is
called twice, `stage: build` and `stage: promote`; it no longer answers
`v*` on its own), then `publish-crates` (irreversible), then Homebrew. A
failed build of any single target leaves nothing public behind.
`scripts/check-release-graph.rb` asserts that shape against the YAML:
every publishing job transitively needs every build job, no build job
publishes. What remains non-atomic is the promotion chain itself: it is
sequential and each step re-runnable, but a failure in crates.io or the
Homebrew bump leaves the GitHub Release and the image already visible.

Off the tag path, the Docker workflow keeps its own copy of the ancestry
guard and of the test job (workflows cannot share jobs): `:release-edge`
and any manual image dispatch are pushed only once fmt, clippy and the
suite are green on that commit (`release` has no CI run of its own, so
this is the only test `release-edge` ever sees). The ancestry guard runs on
every Docker dispatch (each one pushes at least its custom tag and
`sha-*`), so a dispatch from a feature branch cannot push an image.

A manual run of the release workflow is now always a dry run: guard, test,
binary and image builds, nothing promoted. The former `dry_run: false`
dispatch had no tag, so it would have shipped crates and tagged the image,
then failed the Homebrew bump looking for a GitHub Release that was never
created. Not in this change: SHA-pinned actions, SBOM, provenance,
cargo-deny, cleanup of untagged candidate digests left on ghcr.io by a tag
run that failed before promotion.

### The single-tenant servers refuse a network bind without an opt-in

`skeg` and `skeg-resp3` have no authentication. They now refuse any
non-loopback `--addr` (including `0.0.0.0` and `::`) unless
`--allow-unauthenticated-network` (env `SKEG_ALLOW_UNAUTHENTICATED_NETWORK=1`)
is given, and log a warning at startup when it is. The quickstart publishes
the container port on the host loopback only (`-p 127.0.0.1:6379:6379`);
the image itself keeps `0.0.0.0` inside the container and sets the opt-in,
because a container needs to bind all interfaces to be reachable through
`-p` at all. For network exposure use `skeg-server-tenant` with
`--tenant-auth --tenant-strict`, or an authenticating proxy.

`skeg-server-tenant` applies the same guard unless auth is actually
enforced: `--tenant-auth` alone (lenient mode) still maps an anonymous
`HELLO 3` to tenant ZERO, so only `--tenant-auth --tenant-strict` lifts the
check. The review version lifted it on `--tenant-auth` alone.

### One RESP3 connection is bounded to one legitimate frame

`MAX_BULK_LEN` drops from Redis's 512 MiB default to 64 MiB: no KV value,
vector or payload has a legitimate reason to arrive in a bigger bulk. The
per-connection ceiling is a *frame* cap (the parser yields a frame only once
the whole aggregate is buffered), so it is sized from both limits a
`SKEG.VMSET` can reach at once - 64 MiB of vectors plus a 64 MiB bulk of
ids/payloads, ~129 MiB - where it used to be ~513 MiB, more than the
256 MiB cgroup the engine is gated under. The 256 KiB read reservation is
now taken only while a frame is mid-flight and given back once the buffer
drains; a socket that never sent a byte, or that finished its last frame,
holds 4 KiB. Still open: the native protocol handler and a connection/buffer
budget owned by the governor - a thousand connections each mid-frame can
still pin a thousand ceilings.

### `skeg-multi-tenant` embeds the workspace engine

The crate reaches its vector engine through `skeg-rigging-skeg`, which
depends on `skeg-vector = "0.1"` from crates.io. That resolved to
`skeg-vector 0.1.3` next to the workspace's `0.1.9`: two engines in one
build, and the multi-tenant one was never the one this repository tests.

A `[patch.crates-io]` entry now points that dependency at
`crates/skeg-vector` (and `skeg-simd`, `skeg-platform` with it); the
rigging chain is moved to `skeg-rigging` / `skeg-rigging-skeg` 0.1.3 and
`skeg-rigging-net` / `-resp3` 0.1.1. `tests/engine_alignment.rs` asserts
that the resolved graph holds exactly one `skeg-vector`, the workspace one,
and that a tenant written through `MultiTenantRoot` reopens with
`skeg_vector::DiskVamanaIndex` directly. Still duplicated: `skeg-resp3`
(`^0.1.3` through `skeg-rigging-net-resp3`, 0.2.5 here - the caret cannot
reach a 0.2) - a transport crate, not the engine.

`skeg-multi-tenant` 0.1.0 -> 0.1.1.

### The on-disk layout is declared, not deduced

A store now carries a `LAYOUT` file at its root stating what it is: format
version, shard count, store identity, and the feature flags its files were
written with. Every server constructor asks that file and nothing else, so
there is exactly ONE place that decides how many shards a store has.

Before, the count was inferred by scanning for `shard-N` directories. The
scan was made fail-closed after serve mode was found opening one shard of an
eight-shard store - 597 rows of 5,000, recall 0.115, no error and no warning
- but a refusal still leaves the reader deducing something the writer knew
and never wrote down.

Every ambiguity is now a startup error: a failed checksum, a format version
this build does not read, a feature flag it cannot honour, a shard count that
disagrees with the caller, or a manifest that disagrees with the directories
beside it. The file is a fixed 44 bytes with a CRC32C over its body, checked
before any field is believed - the version of a corrupt file is itself
corrupt. A longer file is rejected rather than read as a prefix, so a future
writer's extra data can never be silently ignored by an older reader.

Deliberately absent: base generations and router epochs. Those advance per
vindex many times a minute and already have their own atomic per-vindex
files; putting them here would turn one rarely-written file into a contended
one, and recreate the rule-in-two-places shape behind every P0 this engine
has had.

Stores written before this existed still open. A writable open adopts a
manifest once, keeping the fail-closed rules; a read-only open reads the
layout and changes nothing on disk, because serve mode runs over copies an
operator may have mounted read-only. Directory scanning survives only as
that one-way migration.

### The catalogue decides what exists, not the resident map (P0)

A vindex can be evicted from RAM without being dropped: its files stay, its
registry entry stays, and the next access reopens it. `DROP` read absence off
the resident map, so it answered "not found" for an index that was committed,
on disk, and about to come back - and a tenant erasure, which built its work
list the same way, walked past the tenant's own vectors. The KV sweep then
took their payload blobs, the call reported success, and the vectors stayed on
disk. The existing erasure test could not see it: it builds a flat index,
which cannot be evicted, and its final assertion reads `VINDEX.LIST`, which
walks the same map.

Existence is now decided by the registry. A resident miss reopens through the
same path a search uses, and only a miss there is a real absence.

An index that is committed but will not OPEN can now be dropped too. Reading
"cannot open" as "does not exist" left the store with a catalogue row nobody
could remove. Its payload blobs are reclaimed by matching the key shape
exactly rather than by asking the index for its live ids - which needs no open
index, and which also stops `DROP ev` from taking the blobs of `ev2`, whose
name it prefixes.

`LIST`, `CHECK` and `HEALTH` stop drawing conclusions from the map's silence.
`LIST` reports committed-but-evicted indexes and ends each line
`resident=k/n`, where `n` is the store's shard count, so a row whose counts
were summed over only some shards says so. `CHECK` no longer answers
`Problems([])` - nothing wrong - about an index it never opened, and an
unreadable registry is a finding rather than an error that takes the report
down with it. `HEALTH` no longer answers "no such vindex" for an index
committed on every shard and merely evicted from all of them; it reports
`not_assessed` and has a state of its own, `UNASSESSED`, directly under
`MISSING` because it carries the same information about health: none.

`DROP` now walks the shard's key space to find blobs, so it is O(keyspace) -
the property `SKEG.COUNT` and the erase sweep already declare. Measured at
36-68 ns per key, which is +9 to +17 ms across 250k unrelated keys and, by
extrapolation, under a second at ten million. `LIST` gained a per-shard
registry read: 27-71 us per shard including the round trip, +0.9 us per index.

### A catalogue fan-out that failed left the store split (P0)

`VINDEX CREATE` and `DROP` reach every shard and each shard commits into its
own registry. Nothing coordinated them: the first error was returned and the
shards that had succeeded stayed committed, so the caller got an error
describing a store it no longer had. Measured on four shards with one unable
to write: a failed CREATE left the index on three of them accepting writes the
caller believed impossible, and recreating the name with a different dim then
took on the shard that had failed while the others refused it - one name, two
dims, reported by `LIST` as a single agreed row. A failed DROP had already
removed the data from three shards, leaving the fourth unreachable through
search, still on disk, still catalogued, and back at the next open.

A name is now recorded at the store root before the fan-out starts and cleared
before any success is returned, so its presence means exactly "not
acknowledged". A create that fails is undone at once on the shards that took
it; anything left is resolved at the next open, before a shard serves
anything, and the payload blobs and router sidecar go with it.

The record names the operation, because the two do not resolve alike in one
case: when the fan-out reached no shard at all, a create that succeeded
unacknowledged must still be undone, but a DROP the store refused has an
intact index and a caller who was told it failed. On a single-shard store that
is every failed drop, not a corner.

A read-only open cannot resolve, so it refuses to open a store with an
unfinished fan-out and names the indexes, rather than serve around a
half-state. Skipping those names at open is not a guard: the first query
reopens the index lazily from the registry.

### A point read could return a stale vector (P0)

`DiskVamanaIndex::get` checked the BASE before the RUNS, under a comment
claiming the opposite ("newest run wins on a shadowed id"). A run holds a
freshly flushed delta and is therefore NEWER than the base, so any id
present in both read back at its old value: an acknowledged overwrite
stayed invisible to point lookups (SKEG.VGET, and anything built on it)
until a consolidate happened to fold that run into the base.

Search was never affected - `score_ids_quantized` walks runs newest-first
and falls back to the base - so recall stayed high while point reads
lied. That divergence is why it survived: every recall gate passed.

Measured on a settled 60,000-row index holding one run: 3,534 rows
(5.9%) returned the previous generation. With runs and delta at zero, the
count was zero - which is why a bench that consolidated between rounds
could never see it.

Precedence is now delta -> flush staging -> runs newest-first -> base.
Pinned by a test that fails against the old order with the diagnosis in
its message (cosine 0.255 to the new vector, 1.000 to the old one).

### Read-only serve mode served a fraction of the index (P0)

`bind_serve*` opened the shard set with a hardcoded count of 1. A set
written with eight shards therefore served exactly the rows that landed
in shard 0 - an eighth of the index - with no error, no warning and no
hint, answering every query with complete confidence. Measured: 597 of
5,000 vectors, before and after a consolidate; recall against the full
corpus read 0.115.

The count now comes from the DATA (`ShardSet::discover_shard_count`
counts `shard-N` directories) and is logged at open. Pinned: an
eight-shard set is discovered as eight and a reopen sees all 800 of 800
rows.

Found while chasing something else entirely - a churn-gate recall drop -
which is the usual way: the frozen-state experiment built to isolate that
bug ran in serve mode, and its impossible numbers (more re-rank budget
producing WORSE recall) were this bug, not the one under investigation.

### mmap graph validation (P0)

The mmap open route cast `graph.vmn`'s node region straight into `&[Node]`
with no per-node pass, while the in-RAM route validated every degree and
every neighbour. So the mmap route ACCEPTED graphs the in-RAM route
refuses, and the walk then followed whatever those bytes said - and a
degree past MAX_R sliced out of range, which under `panic = "abort"` is a
dead server. Both routes now run the same structural validation at open,
and `Node::slice` clamps to MAX_R so a corrupt degree can never panic a
neighbour read (which also keeps CHECK - the tool you reach for when a
file IS corrupt - safe to run). Pinned: a dangling edge and a
degree > MAX_R are each refused by BOTH routes, and a forged degree of
u32::MAX still yields a MAX_R-long slice.

### SKEG.CHECK

The operator's fsck. `SKEG.CHECK <index>` reports every integrity problem
found across the shards, or `OK` when the index is healthy; an unknown
index is an error, not a clean bill of health. Read-only and O(rows +
edges), safe on a serving index.

What it checks is exactly the failure shapes this engine has produced:
ids/graph/vectors.bin row-count disagreement, edges pointing past a
segment's rows (the mmap reader skips the per-node validation the in-RAM
reader does), a medoid outside its segment, a run directory missing its
run.ok durability marker, a CURRENT pointer naming an absent generation
slot, and - for a resharded index - a router whose dim or centroid count
disagrees with the index, or an owner map naming a shard out of range.

### Hardening

kill -9 at the four worst moments - mid-bulk-write, mid-consolidate,
mid-reshard, mid-overlap - all four reopen with zero acked ids lost, the
reshard resumes to completion, and a post-crash delete leaves no replica
ghost (bench/hardening.py in the demo repo). C4 (probe) is parked with
its numbers: 0,9445 at probe 3 against a 0,955 bar - the dense local
graphs' recall stacks onto routing coverage, so cell health (the graph
doctor and usage-driven stitching) is the dependency that reopens it.
Targeted overlap ships with tau from the measured margin distribution.

### Semantic shards (C1-C3)

The owner's "galaxies" become engine machinery. `balanced_kmeans`
(SPANN-style size penalty) gives every shard a semantic identity;
`router-<name>.bin` persists the centroids with an epoch;
`SKEG.VINDEX.RESHARD` physically moves every live row to its owner
(payload included, vset-then-vdel so a crash duplicates and never
loses, cursor-resumable, the search merge dedups by id); an id-to-owner
map keeps point ops O(1) and rebuilds itself lazily after open.

Live baptism on the demo: 386.047 rows moved in 22 minutes with the
count exact at every check, residual debt folded in 19s. The recall
gate then caught a real -3pt regression, root-caused to lonely queries
in the dense per-shard graphs (top-1 perfect, tail scores -0,003..-0,028;
NOT a boundary effect - margins identical); beam 256 restores parity
(0,9915 vs 0,9925 hash control) and, under a 24-user storm, the
semantic layout at beam 256 beats the old hash layout's server p99
(20,0 vs ~28 ms) at double the load with zero errors. Routed probing
(C4) and boundary overlap (C5) come next.

### The full rebuild leaves the normal path

The consolidate rebuilt the whole graph from scratch on every fold: measured
O(n^1,5), 46s at 400k vectors, with the write throughput falling as 1/live-set.
The engine now folds at a cost proportional to what changed, not to what it
holds. Measured on real 1024-dim embeddings: steady fold slices of 18-21s per
125k rows at 500k and 42-54s per 250k rows at 1M (flat per-vector cost across
scale), stream/bulk recall ratio 0,995 at 500k and 0,994 at 1M, and a fold with
nothing new to add costs 13-25ms instead of a rebuild.

### Added

- **`SKEG.VGET name id`** on the RESP3 surface: the stored f32 vector as
  little-endian bytes (the encoding VSEARCH accepts), Null for unknown or
  deleted ids, tenant-scoped like its write twin. A client never re-embeds
  a document whose vector the index already holds: the demo's
  similar-search dropped from ~250ms to under 4ms by fetching the anchor
  vector instead of re-embedding its summary.

- **Permute-dot ADC kernels (default).** The 2- and 4-bit TurboQuant proxy
  scored by widening every TBL-picked centroid to f32 and paying four FMAs
  per 16 dims; the query is now quantised to i8 once per search and the
  decoded levels feed `sdot` - sixteen multiply-accumulates per instruction,
  exact i32 accumulation. Measured at dim 1024: tq2 151,6 -> 40,3 ns
  (3,76x), tq4 146,7 -> 40,9 ns (3,59x). Recall gated on 50k real mxbai
  embeddings against brute force: 0,9968 -> 0,9970. `SKEG_TQ_QI8=0`
  restores the f32 path. An `sdot` kernel also backs `dot_int8` on aarch64
  (the baseline-NEON `vmull` kernel measures slower than the
  auto-vectorized scalar and stays undispatched); the 100k-vector flat
  scan drops 23,8%.
- **Int8 walk proxy for the build** (default on; `SKEG_BUILD_INT8_WALK=0`
  restores the f32 walk): navigation ranks candidates by the i8 dot on the
  sdot kernel at a quarter of the memory traffic; the prune re-scores in
  f32 (mandatory, the int8-prune verdict stands). Gated at 150k real mxbai
  rows: fold 28,3s -> 14,9s (1,90x), recall 0,9878 -> 0,9880. An earlier
  1,09x verdict was invalid - measured on a binary that lacked the flag,
  i.e. two identical runs; the re-measure on the merged binary is the
  number that stands.

- **Patched fold.** `ConsolidateJob` now captures the base adjacency, and the
  fold reuses it: rows whose neighbours all survive are remapped verbatim at
  zero distance computations, rows touching dead neighbours are re-pruned, and
  new points are inserted with a single greedy+prune+back-edge pass. The full
  rebuild remains as fallback, chosen by measured shape: patched only while the
  base is at most 20% dead (the delete-patch verdict places the losing regime
  above ~25%) and the new rows do not outnumber the live base.
  `SKEG_PATCH_FOLD=off|force` overrides the route.
- **Fold concurrency budget.** Heavy maintenance builds (consolidate,
  runs-merge, delete-patch, IVF) share a process-wide budget
  (`SKEG_FOLD_CONCURRENCY`, default 2). Explicit commands park and wait;
  background maintenance skips and retries at the next tick, counted by
  `MaintenanceBudgetSkips` and the `skeg_folds_waiting` gauge. The delta flush
  is exempt: parking it froze search behind a growing flat-scanned delta,
  which was the actual latency degradation observed, not CPU contention.
- **Consolidate pace.** A serving store folds on a quarter of the cores; an
  idle one uses the whole machine (`SKEG_CONSOLIDATE_THREADS` overrides).
- **Per-kind maintenance counters.** Flush, consolidate, runs-merge,
  delete-patch and budget skips are now counted, so a soak can prove which
  path ran instead of guessing.

### Changed

- **Maintenance ladder order.** Cheap first, expensive last: flush, then
  delete-patch, then runs-merge, then consolidate. The fold used to be checked
  first, so whenever it was due the paths proportional to the change never got
  a turn.
- **The idle clause is gone.** A store that went quiet with one flush behind
  it used to rebuild its entire base (`idle && delta + run_rows >= 4096`).
  A quiet store needs its runs merged, which is what runs-merge is for; the
  fold now fires only on the geometric trigger or on a heavily-dead base.

### Fixed

- **Filtered scans paid four hash probes per id.** The phase decomposition
  under storm caught the hybrid route (58% of shard calls on the demo)
  scoring 12,3k ids at 212ns each against a 40ns kernel. In the folded
  steady state the loop now does one base lookup per id: scoring per call
  2,60 -> 1,85ms, re-rank 0,57 -> 0,23ms under the same storm. The
  scan-vs-IVF crossover re-measured on current kernels at the demo's
  regime: the exact scan wins the tail (the compiled threshold was right;
  `SKEG_HYBRID_SCAN_MAX` exists to re-measure as kernels move). The
  remaining 151ns/id is the per-id hash walk - the block-32 layout's
  target number.

- **A restart replayed everything since the last fold into the RAM delta.**
  `clean_stale_runs` deleted every run directory at open and recovered the
  whole WAL: on the demo, a restart after a 218k-row growth put 900 MB back
  into a flat-scanned delta. Flushed runs are durable graphs: `flush_finish`
  now fsyncs the run, writes a `run.ok` marker and compacts the WAL down to
  the current delta plus live tombstones; a reopen loads every marked run
  and replays only the WAL suffix. Unmarked (torn) runs stay WAL-covered
  and are deleted as before.

- **Delete-patch ran in its losing regime.** It had a lower tombstone bound
  but no upper one, and fired on a base 61% dead, where the measurement says
  it loses 3x. Past a quarter dead the ladder now routes to the full fold,
  which also gained a heavy-dead arm: without it, nothing would ever have
  reclaimed such a base, because the geometric trigger only watches run
  growth.
- **Connectivity repair scanned the whole index per stranded node.**
  `patch_connectivity` found each stranded node's attachment point by an exact
  scan over all rows: O(stranded x n) distances, ~0,5s per node at 460k rows,
  and the entire cost of the ~50s per-shard folds observed on dirty history
  with zero new writes. It now walks the graph itself (a walk from the medoid
  can only visit reachable nodes), and logs the stranded count and repair
  time.


What a restart and a filtered search actually cost, found while running a
471.918-vector corpus (HuggingFace model metadata, dim 1024, `tq2`, disk
backend, 8 shards) as a live service. Every number below is measured on that
corpus, not projected.

```
                                  before     after
open to queryable                   ~24s     10,6s
first filtered search after open  5247 ms     28 ms
vlog records re-decoded at open     472k         0
one client search counts as           8x        1x
```

Three of these were invisible in the configurations most people run: the
snapshot only paid off after a segment rollover, the search counter only
inflated on a multi-shard server, and the payload rebuild only showed up on a
restart with a filtered query.

### Fixed

- **A restart re-read the whole active vlog segment, however recent the
  snapshot.** `Snapshot::hwm` is a segment id, and its own documentation says it
  lets recovery skip segments with a *lower* id. With a single active segment
  there is no lower id, so the mechanism never engaged before the first segment
  rollover and a 512 MB segment was decoded again on every open. Snapshots now
  also record `hwm_offset`, the byte length of the active segment they cover,
  and recovery resumes there. Opening the corpus above went from 39,0s to 0,5s;
  in production the recovery counter now reports zero records replayed.

- **One client search counted as one operation per shard.** `VSEARCH` is the
  only operation that scatters to every shard, and the counter and latency
  histogram were recorded inside each shard worker. On an 8-shard server a
  single search reported 8 operations, and the histogram observed one shard's
  fragment of the work rather than what the client waited for, so both
  published query traffic and published query latency were wrong by the shard
  count. Measured on a live server: 5 searches moved the counter by 40. The
  measurement now happens once at the scatter, spanning fan-out, replies and
  merge. Nothing is lost: every shard receives exactly one message per search,
  so per-shard counters carry no information for a scattered operation.

- **The first filtered search after a restart paid for the whole reopening.** A
  filtered `VSEARCH` evaluates the payload index, and that index was rebuilt
  lazily on the first filter to arrive, reading every live id's payload blob
  from the vlog. On the corpus above that search took 5247 ms against 6 ms for
  every one after it. An unfiltered search neither pays this nor prevents it:
  without a filter only the k results' payloads are read. The rebuild now runs
  at open, inside the readiness barrier that already waits for recovery.

- **Explicit `SKEG.VINDEX.CONSOLIDATE` held the vindex write lock for the whole
  graph rebuild**, blocking every read for its duration: on 4.000 vectors a read
  waited 3,81s on a 3,81s consolidate. Automatic maintenance already did this
  correctly; only the explicit command was left behind. It now uses the same
  three-phase path (short lock to begin, build off-thread with no lock, short
  lock to finish). The command still returns when the work is done; it just
  stops blocking readers meanwhile.

### Changed

- **The readiness barrier now also waits for payload indexes to load.** The
  barrier exists so that an open port means queryable, and a search that still
  has to rebuild an index is not being served. Consequence to know about: there
  is no timeout on it. On a slow or degraded disk the port stays shut rather
  than opening and making the first filtered query pay, so a health check will
  time out where it previously saw one slow response. Warming is best effort per
  vindex: one that fails is logged and still loads lazily on first use.

- **Payload indexes are resident from open, for every recovered vindex.**
  Measured at 338 MB for 471.918 records across two indexes, which extrapolates
  to roughly 2 GB at 3M. Previously this was paid lazily and only for vindexes
  actually filtered on, so a deployment that never uses `FILTER` now pays memory
  and open time for nothing. On macOS the compressor hides this: an idle process
  reports 3 MB resident until the index is touched.

- **Snapshot format is now v2** (adds `hwm_offset`). A v1 file is rejected and
  recovery falls back to a full scan, which is correct and only slower, so
  downgrading is safe but gives up the faster open.

- **A bulk write no longer evicts the working set.** `set_many`, the batch
  behind MSET, wrote every value through into the hot-key cache. That is right
  for a single SET, where reading the key back next is normal, and wrong for
  the bulk primitive. Loading 669.405 keys into a store that was serving
  traffic filled the 256 MB budget exactly and evicted 150.760 entries. It now
  invalidates instead: removing rather than skipping, since a key already
  cached would otherwise keep its old value and be served stale.

- **The key index is sized from the snapshot instead of growing into it.**
  Recovery built it with `Index::new()` and let it double its way up, so the
  table ended up sized for the next power of two and kept the slack for the
  life of the process. The count was known all along. Measured on 1.383.158
  keys shaped like a real store's, 21,3 MB of actual key bytes: 149 bytes per
  key grown against 86 pre-sized, same structure and same lookups.

### Added

- **The payload index moved off the heap.** It was rebuilt at every open by
  reading each live id's blob back from the log, one random read per id, and
  then held entirely in memory. Both halves are fixed.

  It cost 449 bytes per vector on a real corpus whose payloads are 103 bytes of
  text. Two thirds of that was the posting sets, many small `BTreeSet`s that
  are mostly node overhead; the rest was a parsed copy of every payload kept
  only so an overwrite would know which postings to withdraw. The posting ids
  now live in `payload.idx` beside the vindex, sorted and delta-varint encoded,
  and the parsed copy is gone: the canonical text is kept instead and re-parsed
  on the rare path.

  ```
  one vindex, 28.019 vectors, index built in isolation
    before            ~1120 B/vector
    text not parsed     449 B/vector
    postings on disk     56 B/vector      file 0,8 MB

  production artifact, 471.918 vectors over two indexes
    open                10,5s  ->  1,6s
    process RSS          842 MB -> 159 MB
  ```

  The directory, which fields exist and which values, stays in memory as a
  `BTreeMap`. Range filters depend on `Value`'s ordering, and reimplementing
  that ordering inside a binary format is the kind of mistake that returns the
  wrong rows and says nothing; the `BTreeMap` makes it correct by construction,
  and it is not the part that weighs.

  Staleness is the risk that governs the design, because payload blobs live in
  the log and keep changing after the file is written, and a payload index that
  is quietly wrong makes filtered searches drop results with no error anywhere.
  Three rules, each with a test that fails without it:

  - the file carries the log snapshot position it reflects, is written in the
    same step as that snapshot, and is refused unless recovery seeded from
    exactly that snapshot;
  - an id whose payload key appears in the replayed log tail is refused and
    read from the log instead;
  - it is read only during the open-time warm, never when a vindex is reopened
    after an eviction, because by then writes may have landed that no tail
    records.

  Writes after the file was built go to an in-memory overlay whose ids shadow
  the file, so an overwrite never rewrites it and a delete never has to know
  which postings to withdraw. Writing it again folds the previous file in
  rather than chaining.

  The gate is equivalence rather than "it works": a disk-backed index is
  compared against an in-memory one over every shape of the filter grammar,
  including a check that the filters match enough ids to be comparing
  something. It is created `0600`, carries a crc32c, and its header is
  validated and its counts bounded before anything is allocated, so a length
  field on disk cannot drive an allocation. Truncation at every length and a
  flipped bit are refused.

  Stressed on the real corpus as well: payloads rewritten around a snapshot,
  the store restarted, and the filter's view compared against the stored blobs.
  `skeg_payload_index_from_disk_total` reports how many ids came from the file,
  and reads the total minus exactly the ids rewritten after the stamp, so the
  refusal is visible in the number rather than only asserted.

  On the RSS figures: they moved in the right direction, but macOS compresses
  idle pages and the same process reported 3 MB and 338 MB minutes apart during
  this work. The per-vector measurements, taken in isolation in separate
  processes, are the ones to trust.

- **Quantised tier cache.** The `tq2` tier was recomputed from the source
  vectors on every open; it is now serialised to `tier.cache.bin` in the vindex
  directory and read back, with a fingerprint (vector count, dim, tier tag,
  source length and mtime) that invalidates it when the corpus changes. A
  foreign, truncated or corrupt cache is ignored rather than trusted. Worth 21%
  of open time on the corpus above.

- **`skeg_vlog_recovery_records_total`** counts records decoded while replaying
  the log at open. A value close to the total key count means the snapshot is
  not doing its job.

- **`skeg_payload_index_rebuilds_total`** counts payload indexes rebuilt from
  blobs. Any increase while serving traffic means a query paid for a rebuild.

- Each payload warm logs its vector count and duration at `info`, since it is
  now a visible share of open time.

## [0.7.3] - 2026-08-22

Hardens the boundaries where skeg trusts network bytes or on-disk data, from a
security review of the workspace. No protocol or API changes; published SDKs and
adapters are unaffected.

### Security

- **Path traversal via vindex names on the native binary protocol.** Name
  validation lived only in the RESP3 layer, so a native-protocol
  `VINDEX.CREATE "../../x"` escaped the data dir. Validation now runs in the
  shard layer (create/drop/consolidate), the choke point both protocols cross.
- **Unbounded `k`/`l_search` on the disk VSEARCH path** sized an allocation
  straight from the wire (a large `l_search` could request tens of GiB and
  abort the process). Both are now clamped.
- **Recovery allocated from on-disk length fields before the CRC.** A bit-flip
  in a record header could drive a multi-GiB allocation and OOM at startup; the
  length is now bounded against the segment ceiling first. The snapshot decoder
  clamps its entry count the same way.
- **No throttle on failed authentication.** HELLO/AUTH bypass the QoS gate, so
  online guessing was unbounded. Failed attempts are now counted per source IP
  over a rolling window (blocked before the password verify) and each failure is
  tarpitted.
- **Crafted `graph.vmn` could crash the server** on open or first search: the
  owned open path trusted `n`/`degree`/`medoid`/neighbour ids off disk. These
  are validated, returning a clean error instead of an out-of-bounds panic.
- **Data and WAL files inherited the umask** (world-readable on a shared host);
  they are now created `0600`, matching the auth store. Segment opens use
  `O_NOFOLLOW`.
- **Missing parent-directory fsync** on segment rotation and snapshot rename
  could drop a `Durability::Power` write on power loss; the directory entry is
  now persisted.
- Bounded the speculative RESP3 aggregate pre-allocation and added a
  per-connection input-buffer ceiling.

## [0.7.2] - 2026-08-19

### Fixed

- **`SKEG.VINDEX.CREATE` rejected its own documented default.** The three-arg
  form `name dim backend`, which takes the server's default tier, is
  implemented in the dispatcher and documented in `skeg-resp3 --help`, but the
  command parser refused arity 3 before the dispatcher ran. The default was
  unreachable from the wire and the code implementing it was dead. Arity
  disambiguates the two forms: kind and backend share the numeric aliases 0
  and 1, so a three-arg call is always `[name, dim, backend]`.

### Added

- `conformance/`: the case files every skeg client is checked against (106
  RESP3, 49 native) and two standalone validators that speak the wire
  directly. They were in a private repo, where no public client's CI could
  read them.

## [0.7.1] - 2026-08-15

x86 support for the SIMD kernels, validated on real hardware for the first
time, and a cleanup pass over the engine that turned up a data-loss bug.

### Fixed

- **The native protocol silently created the wrong index type.** Its wire
  contract documents three kinds (0 f32, 1 int8, 2 binary), but the server
  passes that byte to the shared six-value table, where 3 is TQ1. Byte 3 was
  historically PQ, so a native client asking for PQ got a TurboQuant 1-bit
  index instead: no error, a different index. Native v1 now refuses kind 3 and
  says where to go, and the TurboQuant tiers are reachable through native v2,
  which states its kind map explicitly. Nothing about a v1 byte changed
  meaning.

- **Consolidate dropped vectors staged by an in-flight flush.** `flush_begin`
  moves the delta into a staging map and releases the write lock while the new
  segment builds off-thread; a `SKEG.VINDEX.CONSOLIDATE` arriving in that
  window folded `delta > runs > base`, skipping the staging map, then truncated
  the WAL and reopened from disk. The staged vectors were left in neither the
  rebuilt graph, the log, nor memory. Silent: no error, no crash, the ids
  simply stopped being found. The fold now follows the documented precedence,
  `delta > flushing > runs > base`.

- **A wrong vector dimension aborted a shard thread.** `DiskVamanaIndex::insert`
  and the search path asserted on a dimension mismatch, so one client's bad
  request could take down a thread serving every other vindex on that shard.
  Both already returned `io::Result`; they now return `InvalidInput`. The flat
  backend is guarded at the seam, where its signature cannot change.

- **NaN was bucketed to the top instead of the bottom on AVX-512 builds.**
  `bucketize_x8_avx512` used an unordered compare as if it were an ordered `>`,
  which is true for NaN, so any vector containing one was silently mis-encoded.
  Only on AVX-512 builds, only on x86. It had passed clippy, an assembly
  read-through and an instruction-selection guard in CI; executing it on x86
  found it in one run.

- **Public SIMD kernels did not enforce the preconditions their dispatchers
  checked.** `skeg-simd` exports the individual kernels, not only the
  dispatchers, and eleven of them index with `get_unchecked` or raw pointer
  loads. They took a loop bound from one slice and read from another, assumed
  a bit-packed layout, or relied on a `debug_assert` that a release build
  compiles away: `tq1_masked_sum_neon` given a short slice aborted the process
  instead of panicking, and a non-power-of-two length walked the FWHT
  butterfly stages off the end of its slice. The ADC entry points had a
  related gap, accepting a dimension with no packed representation at all,
  because `dim * BITS / 8` truncates to zero for `dim = 1`. Every caller
  inside the workspace goes through a dispatcher or a quantizer that
  validates, so the engine was not exposed; a direct `skeg-simd` user was.
  Each public kernel now validates for itself through the same helper its
  dispatcher uses, and the FWHT and bit-plane preconditions are runtime
  asserts rather than debug-only ones. Verified by walking every public path
  to an unchecked access: 47 reach one, none through an unguarded route.

- **The AVX-512 ADC declared fewer target features than it uses.** The kernel
  is compiled with AVX-512F, BW, VL and SSSE3, but the tq4 wrapper declared two
  of those and the tq2 wrapper three, and the tq4 dispatcher did not check
  SSSE3. Zen 4 and Zen 5 have all four so nothing surfaced there, but a CPU
  with a partial combination, or a caller following the documented contract of
  the unsafe entry point, could execute an instruction it does not have. All
  five sites now read one shared predicate, `avx512_adc_supported`.

- **VSEARCH had two separately written paths**, one for the worker pool and one
  inline. They agreed, but nothing kept them agreeing, and a fix applied to one
  and not the other would surface only where `workers > 0`. Now one path, with
  the equivalence test extended to cover filters and payloads.

- **The io_uring batch reader rebuilt its ring on every call**, an
  `io_uring_setup` plus two mmaps and two munmaps per batch, which erased the
  overlap it exists for. With the ring reused, a cold batch of 800 reads costs
  3.5 us per read against `pread`'s 100 us; the backend had measured slower
  than `pread` before the fix.

### Added

- **Native protocol v2**, negotiated rather than assumed. The 24-byte frame
  header already carried a version byte; the parser now accepts 1 and 2, and
  every response goes back at the request's version. `NativeHello` reports what
  the server supports. The default encoder still emits v1, so existing clients
  are untouched.

- **AVX2 and AVX-512 kernels for x86**, dispatched at runtime, with the
  `avx512` feature off by default. AVX-512 is selected only where the wider
  instruction set offers something AVX2 lacks: VNNI for the int8 dot,
  VPOPCNTDQ for Hamming, mask registers for tq1 and sign flips, and a
  register-resident 16-entry table for the ADC. Where it is only the same trick
  at twice the width it loses on cores that split 512-bit operations, so those
  kernels ship built and tested but not dispatched.

- **NEON kernels for the rotation path** (`fwht_f32`, `flip_signs`,
  `bucketize_x8`), which ran scalar on aarch64. At dim 1536 on an M1 Pro:
  `flip_signs` 950 ns to 157 ns, `bucketize_x8` 6.22 us to 1.59 us, `fwht_f32`
  1.94 us to 1.39 us.

- **A batch-read seam in the platform layer** with a blocking implementation
  and an optional `io_uring` one, plus `preallocate_sync`,
  `write_vectored_at_sync`, `advise_huge` and `open_populated`. The vLog
  preallocates and fsyncs segments through it.

- **An x86_64 release.** The tarball matrix gains
  `x86_64-unknown-linux-gnu`, and the container image now publishes
  `linux/amd64` alongside `linux/arm64` under one manifest, so `docker pull`
  resolves the architecture by itself. Both carry the AVX-512 kernels: kernel
  selection is a runtime CPU check, so a machine without AVX-512 simply never
  picks them. The x86 CI job runs that build on an AVX2-only runner, which is
  what turns "carries the kernels" into something other than a claim. The Cargo
  feature stays opt-in, because compiling those kernels needs Rust 1.89 and the
  MSRV promised to people building from source is 1.88.

- **A kernel coverage test**: which kernel exists for which instruction set and
  which one the dispatcher picks. A gap has to carry a reason, and a kernel
  that exists but loses to its neighbour is recorded as present-but-not-chosen
  rather than as coverage.

### Changed

- **Only one workflow publishes the container image.** `release.yml` and
  `docker-publish.yml` both fired on a `v*` tag and both pushed
  `ghcr.io/skegdb/skeg:<version>` and `:latest`; the one that finished last
  won. The images were identical, so it never showed. It would have started
  showing now that one of them publishes a multi-architecture manifest and the
  other an arm64-only image. `release.yml` still builds the Dockerfile, to fail
  the release run if it is broken, but no longer pushes.

- **The ADC kernels keep their centroid table in a register** and permute it
  instead of gathering it from memory and unpacking codes in a scalar loop. At
  dim 1536 on a Zen 4 EPYC the AVX-512 kernel went from 464 ns to 63.7 ns and
  AVX2 from 431 ns to 147 ns, so machines without AVX-512 gain too. End to end
  on 99k vectors, an AVX-512 build serves tq2 and tq4 36% to 61% more queries
  per second than an AVX2 one, at identical recall and RSS.

- **Flat 4-bit TurboQuant search uses the block-32 kernel**, which scores 32
  rows against one pre-computed table: 939 ns against 11.9 us for the scalar
  reference at dim 1536. Same candidate width and same exact-f32 rerank, so the
  answer is unchanged.

- **One ADC kernel per instruction set** over a code-width parameter, instead
  of one per tier, so a technique fix lands once per instruction set. Kernels
  are grouped by operation, each module holding every instruction set for it.

- **One source of truth for the quantization wire encoding**
  (`QuantKind::from_wire` / `to_wire` / `wire_kinds`), replacing four
  hand-written byte tables. The byte values are unchanged and now pinned by a
  test, because on-disk registries written by earlier versions depend on them.

- **The idle maintenance decision is a named function**, not a 110-line closure
  nested inside the shard loop. It chooses flush against consolidate against
  runs-merge against delete-patch, so it is what gets read during a memory or
  latency incident, and it used to appear in traces as an anonymous closure.

### Removed

- `crossbeam-channel` and `lz4_flex` from the workspace manifest. Neither was
  referenced by any crate or any source file.

### Versions bumped

- `skeg-proto` 0.2.0. A breaking bump for a 0.x crate: `Op`, `ParseError`,
  `ErrCode` and `NativeVectorKindV2` gained variants, which breaks an
  exhaustive `match` in another crate. All four are now `#[non_exhaustive]`,
  so the next op, error code or kind is an additive change instead of another
  breaking one. It costs nothing here: the server's dispatch already ends in a
  catch-all, because a wire protocol has to answer an op it does not know.
- `skeg-simd` 0.1.6, `skeg-platform` 0.1.5, `skeg-core` 0.3.4,
  `skeg-telemetry` 0.2.2, `skeg-vector` 0.1.8, `skeg-server` 0.7.1,
  `skeg-server-tenant` 0.2.4

## [0.7.0] - 2026-07-21

### Added

- **Subject and tenant erasure with physical reclaim (GDPR).** `SKEG.SUBJECT.ERASE
  <prefix>` (tenant-facing) tombstones a subject's keys; `SKEG.TENANT.ERASE` and
  `SKEG.TENANT.DELETE` (admin) erase a whole tenant's footprint, the latter also
  removing its identity so an out-of-band delete can no longer orphan data.
  `SKEG.RECLAIM` (admin) then reclaims the bytes. Erasure is a fast logical
  delete; reclaim is a heavy offline pass (tens of seconds on a large store), so
  they are separate calls. Backed by `ShardSet::{erase_tenant, erase_prefix,
  reclaim}` and zero-alloc key enumeration (`VLog::keys` / `for_each_key`). Three
  latent concurrency bugs in compaction and relocation were fixed along the way.

- **Atomic multi-key write (honest MSET).** `VLog::set_many` writes all pairs
  behind one batch header as a single group-commit append; recovery applies a
  batch only if all its members survived, dropping a torn batch whole. MSET is
  now atomic per shard (was N sequential sets, partial on crash), matching
  Redis's contract for single-shard deployments. Keys hashing across shards are
  still not globally atomic (no cross-shard coordination).

- **APPEND.** The Redis `APPEND` command, serialised per key so concurrent
  same-key appends never drop a delta.

- **Range-filtered VSEARCH.** An optional per-vector `u64` attribute column with
  zone-map pruning, so a vector search can be bounded to an attribute range
  without scanning filtered-out rows.

- **Store hardening.** An advisory open lock bars concurrent opens of the same
  store; startup fails loudly when a shard cannot open its store;
  `VLog::write_seq` exposes a monotonic per-store write counter.

- **Off-thread maintenance.** Every graph rebuild (consolidate, runs-merge,
  delete-patch, and the new delta flush) now runs as begin then build then
  finish: a short write-lock snapshot, an off-thread build on a capped pool, and
  a short write-lock swap of the prebuilt segment. On a single-threaded shard a
  synchronous build used to stall every concurrent query; under sustained churn
  the query tail drops from ~200 ms to ~17 ms. A per-shard maintenance loop
  drives flush, consolidate, runs-merge, and delete-patch by priority and
  threshold, so ingest never blocks the shard.

- **Off-thread delta flush.** A delta flush moves to a staging buffer built on a
  blocking thread and is still searched at full precedence while it runs, so the
  shard keeps serving during a flush.

- **Runs-merge and delete-patch scaling levels.** Runs-merge folds immutable run
  segments into one merged run for a bounded per-query traversal (best p99 under
  churn, largest win at high dimension). Delete-patch reclaims deleted rows in
  place in O(deleted) instead of an O(live) rebuild, about 4.5 to 10x cheaper
  while dead rows stay under a few percent; it is gated at a low tombstone
  threshold, with full consolidate as the fallback above the crossover.

- **Parallel quant-tier build.** The TurboQuant tier build on the streaming path
  encodes rows across all cores (the rotation dominates and is independent per
  row), byte-identical to the sequential output. Serve-open recovery of a 500k
  tq2 index drops from ~9.6 s to ~2.1 s.

- **Serve readiness barrier.** In serve mode the listen port opens only after the
  shard has recovered and built its tier, so a connecting client can query at
  once. First-query latency at 500k drops from ~8 s (the client used to race
  recovery and wait in the backlog) to ~15 ms.

### Changed

- **Default read-write tier is now tq2** (TurboQuant 2-bit), was int8, and
  `--tier` applies to both `rw` and `serve` modes. On real embeddings tq2 roughly
  halves index RAM versus int8 at equal recall@10, with recall@100 within about
  0.5 points at 384 to 1536 dimensions. Low-dimension corpora (for example GloVe
  104d) keep more recall on tq4 or int8. Pass `--tier int8` to restore the old
  default.

## [0.6.1] - 2026-07-11

### Fixed

- **Auth store file permissions.** `auth.kdb` (usernames plus argon2id password
  hashes) is now created `0600` on Unix instead of inheriting the process umask,
  so other local users on a shared machine cannot read it. Thanks to Simone
  Zannini and Matteo Cese ([devop.sbs](https://www.devop.sbs/it)) for the report.

## [0.6.0] - 2026-07-11

### Added

- **Filtered search that scales.** A filtered query no longer scores every
  matching vector. Its matching set is routed by a coarse k-means IVF index to
  the query-nearest cells that actually contain matches (predicate-aware, so a
  filter whose matches sit away from the query is still found), then quantized
  scored and f32 re-ranked. A tiny match set still takes an exact quantized scan;
  a large one takes the routed path, which stays sub-linear as the corpus grows.
  On mxbai 500k a 10% filter drops from ~15 ms (scan every match) to ~2.6 ms at
  the same recall. The router is built off the request path during the idle
  consolidate and persisted next to the graph, so it survives a restart.

- **`skeg-bench`, a unified benchmark tool.** One binary that reports recall@10
  and recall@100 (both from real k-searches against brute-force truth), build
  time, RSS, p50/p99 and QPS, per dataset and tier. RSS is read in a subprocess
  that opens the index but never loads the corpus, so the number is the index's
  footprint, not the harness's. Replaces the pile of ad-hoc benches.

- **Memory-mapped tier (`--tier-mmap`, `SKEG_TIER_MMAP=1`).** The TurboQuant
  codes can be backed by a file instead of owned RAM, so the OS can reclaim them
  under memory pressure and drop them when an index goes idle. Latency is
  unchanged while the codes are hot (they stay in the page cache).

### Changed

- **Payload ingest is much faster.** The RESP3 connection now dispatches a
  client's pipelined data-plane commands concurrently (bounded window, replies
  still in order), so the payload blob writes reach the vLog group committer in
  batches instead of one blob per commit. Streaming 100k vectors with payloads
  went from ~260 s to ~26 s.

- **Faster graph builds.** The on-disk build width defaults to `l_build = 48`
  (was 64): recall-neutral on the tested corpora, ~24% less build work. The
  graph build dominates consolidate, so this shortens ingest too. `SKEG_L_BUILD`
  still overrides it.

- **Leaner consolidate.** The IVF router is no longer built inline during the
  ingest-triggered consolidate (which stalled ingest); it is built in the
  background idle consolidate instead.

### Fixed

- **Pipeline command ordering.** Only the vector commands and read-only commands
  run concurrently within a single connection. The scalar KV verbs
  (`GET/SET/DEL/INCR/...`) are serial again, so a pipelined `SET k a; SET k b`
  and `INCR` keep their guaranteed order and atomicity.

- **IVF sidecar crash consistency.** The persisted router is dropped before the
  consolidate rewrites the base, so a crash cannot leave a stale router that a
  length-only load check would accept against reordered rows. `from_bytes` also
  rejects a corrupt-but-right-length file.

### Notes

- Tier guidance: `tq2` (2-bit) is the default and holds recall@10 and recall@100
  nearly flat as the corpus grows. `tq1` (1-bit) is faster and about half the
  RAM, but recall@100 falls off with scale; it fits small tenants and workloads
  that only need recall@10. Both keep the full f32 re-rank, so the top result is
  exact either way.

## [0.5.0] - 2026-06-22

### Added

- **Filtered vector search.** A vector can carry an optional payload, and a
  search can restrict its results to vectors whose payload matches a filter.
  `SKEG.VSET <name> <id> <vector> [PAYLOAD <blob>]` stores an opaque blob beside
  the vector (in the KV vLog under a reserved, tenant-scoped key, so the
  quantized graph stays dense). `SKEG.VSEARCH <name> <k> <l_search> <query>
  [WITHPAYLOAD] [FILTER <expr>]` returns the blob with each hit (`WITHPAYLOAD`)
  and/or applies a payload filter (`FILTER`). The blob's `key=value` fields are
  parsed into a per-index payload index; the filter grammar supports `field =
  value`, `field IN (...)`, the ranges `>= > <= < BETWEEN a AND b`, `field
  EXISTS`, and `AND` / `OR` / `NOT` with parentheses. A field repeated in a
  payload is multi-valued (matches any of its values).

- **Adaptive filtered-search planner.** A selective filter (small matching set)
  is scored exactly over just the matching vectors. A broad filter runs a
  filtered graph search: two complementary walks merged (one that explores only
  the matching subgraph, one that navigates the whole graph and filters at
  re-rank), so recall holds whether the matching vectors cluster together (real
  metadata) or scatter. Validated on real 1024-dim embeddings at 100k and 500k:
  recall@10 0.98 to 1.00 across selectivities and metadata shapes, at query time
  with no extra build cost. The payload index is rebuilt from the stored blobs
  on the first filtered search after a restart.

- **TurboQuant tiers on the read-write disk path.** `SKEG.VINDEX.CREATE name dim
  tq1|tq2|tq4 disk` builds a live-writable disk index whose resident tier is
  TurboQuant (`dim*bits/8` bytes/vector) instead of int8. No trained codebook
  (unlike PQ, which stays serve-only), so it works under streaming writes. The
  tier kind persists in a sidecar and is rebuilt on open and consolidate. tq2 is
  the recommended sweet spot (recall ~1.0, sub-int8 RAM, latency ~int8); tq1 is
  the leanest but best-effort below 512d.

- **`SKEG.VMSET` bulk insert.** One `name` followed by `(id, vector, payload)`
  triples; the server fans the items out concurrently so durable payload writes
  batch in the group committer. Combined with relaxed payload durability and a
  geometric delta rebuild, 100k ingest dropped from 1928s to 28s.

- **`SKEG.VINDEX.CONSOLIDATE name`** force-folds a disk index's delta into the
  graph after a bulk load. Idle indexes also self-consolidate: a per-shard
  background task folds a delta that has been stable across ticks, so an index
  is lean by default without an explicit call.

- **Vindex tiering control plane (mechanism, not policy).** A new
  `Server::control_handle()` returns a `ControlHandle` for managing resident
  vindexes out of band: `open_indices()` enumerates every open index per shard
  with its `IndexStat` (resident bytes, last-access, vector count, whether it is
  evictable), `total_resident_bytes()` sums the fleet, and `evict(tenant, index)`
  drops an index from RAM non-destructively: the `vindex-<name>/` files stay
  and the next access reopens it lazily. A disk-backed index reopens off the
  shard thread (`spawn_blocking`), so a cold-start reopen does not stall other
  indexes on the same shard. The eviction *policy* (RAM budget, LRU, hysteresis)
  is left to an external controller; the engine ships only the knobs.

- **tq2 is now the default tier.** `SKEG.VINDEX.CREATE name dim backend` (kind
  omitted, 3 args) builds a tq2 disk index, and `--mode serve` without `--tier`
  serves tq2. Validated recall-neutral vs int8 across 100-784d real embeddings
  (r@10 >= 0.999, r@100 within 0.004) at lower RAM. Pass an explicit kind / `--tier
  int8` for the prior full-fidelity tier. Behavior change on upgrade: an existing
  serve deployment that relied on the implicit int8 default now serves tq2
  (recall-neutral, leaner) unless it passes `--tier int8`.

- **Per-command admission carries the command kind.** `TenantBackend::admit` now
  takes an `Admission { tenant, op, cost }` (was `(tenant, cost)`), where `op` is
  a coarse `CommandKind`. A backend can apply command-level RBAC (e.g. refuse
  `VINDEX.DROP` for some tenants) and per-operation metering from one gate. Both
  types are `#[non_exhaustive]`. Single-tenant deployments and the default
  (admit-everything) backend are unaffected.

### Changed

- **Vindex RAM accounting (`VindexSizeBytes` gauge) is now tier-accurate.** Disk
  indexes report their own `resident_bytes()` (graph + quantized tier + delta)
  instead of an `n*dim` int8 estimate, so the gauge tracks tq1/tq2/tq4 correctly.

- **Plain (non-filtered) search and writes are unchanged.** A VSET without
  `PAYLOAD` does no extra work; a VSEARCH without `FILTER`/`WITHPAYLOAD` takes
  the existing path byte-for-byte. The native binary protocol stays
  payload/filter-free; RESP3 is the surface for the new feature.

- **`ShardSet::vset` / `ShardSet::vsearch` gained parameters** (payload, the
  tenant, `want_payload`, and an optional filter), and `skeg-vector` gained
  `DiskVamanaIndex::search_filtered` / `score_ids` / `live_ids`. Library code on
  these APIs must update its call sites, which is why `skeg-server` takes a minor
  version bump.

- **`TenantBackend::admit` signature changed** from `admit(tenant, cost)` to
  `admit(Admission)`. Backends that override it must update; the default
  (admit-everything) is unchanged, so single-tenant and non-overriding backends
  need no change.

### Versions bumped

- `skeg-vector` 0.1.5, `skeg-resp3` 0.2.1, `skeg-server` 0.5.0,
  `skeg-server-tenant` 0.2.1

## [0.4.1] - 2026-06-15

### Added

- **Fair per-tenant cache eviction.** Per-tenant cache accounting already
  shipped; eviction itself was tenant-blind, so under a noisy neighbour a
  tenant with a large hot set could evict another tenant's small hot set in
  FIFO order. The Main queue's victim selection is now share-aware: it
  computes an equal share (`cache budget / active tenants`) once per
  eviction and, when some tenant is over its share, briefly skips
  under-share victims (bounded to 16 re-queues) to evict an over-share
  tenant instead. The Small queue stays tenant-blind on purpose: it already
  absorbs scan floods, so no fairness is needed there. A flooding tenant can
  no longer starve a quiet tenant's working set out of the cache.

### Changed

- **Single-tenant and anonymous traffic is unchanged.** Fairness activates
  only when more than one tenant is resident in a cache shard; with a single
  tenant the eviction path is byte-identical to before and adds no
  measurable overhead. The over-share scan is `O(active tenants)` once per
  eviction but short-circuits on the first over-share tenant; a scaling
  bench (1 to 10000 resident tenants) shows eviction cost stays flat.

### Versions bumped

- `skeg-core` 0.3.1, `skeg-server` 0.4.1

## [0.4.0] - 2026-06-14

### Added

- **Per-tenant resource accounting and hard quotas.** The engine now
  tracks, per tenant, the hot-key cache bytes and the live on-disk KV
  bytes each tenant holds, and can enforce optional hard limits at
  admission. `VLog` gains a `tenant(id)` view that scopes cache
  residency and disk accounting to a tenant; the per-tenant disk total
  is rebuilt from the index on restart. A `TenantBackend::limits(tenant)`
  hook lets a deployment cap a tenant's vector count (`max_vectors`,
  checked on `SKEG.VSET`) and its on-disk KV bytes (`max_disk_bytes`,
  checked on `SET`); an over-limit write is rejected before anything is
  stored. The vector quota is enforced under the index write lock so an
  insert is counted exactly once and overwrites stay free; the disk
  quota counter is shared across shards so the limit is global per
  tenant. New public surface: `VLog::tenant`, `TenantView`,
  `SharedTenantDisk`, `new_shared_disk` (skeg-core); `TenantLimits`,
  `TenantVectorQuota` (skeg-server).

- **`SKEG.QUOTA.SET` / `SKEG.QUOTA.GET` admin commands.** An operator can
  set a tenant's quotas at runtime over RESP3:
  `SKEG.QUOTA.SET <tenant> <max_vectors> <max_disk_bytes>` (`*` = unlimited),
  and read them back with `SKEG.QUOTA.GET <tenant>`. The commands require
  an admin connection; the multi-tenant binary designates the admin tenant
  with `--admin-tenant <name>` and persists the limits in a sidecar next to
  `auth.kdb`, so they survive a restart.

### Changed

- **Single-tenant and anonymous traffic is unchanged.** With no limit
  configured nothing is counted and the write path is byte-identical to
  before; the per-tenant accounting adds no measurable overhead on the
  single-tenant path.

- **`skeg_core::Error` and `skeg_resp3::Command` gained variants**
  (`Error::DiskQuota`; `Command::SkegQuotaSet` / `SkegQuotaGet`). Code that
  matches these enums with a wildcard arm is unaffected; an exhaustive
  match must add the new arms. This is why `skeg-core` and `skeg-resp3`
  take a breaking version bump.

### Versions bumped

- `skeg-core` 0.3.0, `skeg-resp3` 0.2.0, `skeg-server` 0.4.0,
  `skeg-server-tenant` 0.2.0, `skeg-vector` 0.1.4

## [0.3.8] - 2026-06-09

### Added

- **`skeg-multi-tenant` first crates.io publish.** The crate moves
  back into the engine workspace now that all of its sibling rigging
  transports (notably `skeg-rigging-net-resp3`) are on the registry.
  Path overrides on the rigging deps were dropped in favour of plain
  version requirements so the local copy of `skeg-rigging` and the
  one pulled in transitively by `skeg-rigging-net-resp3` resolve to a
  single shared instance; without it the `live-attach` feature would
  refuse to compile because `TenantId` would surface as two distinct
  types.

### Changed

- **`Cargo.toml` workspace lint config** gains
  `unknown_lints = "allow"` so the MSRV CI rustc no longer screams
  when it meets the `clippy::manual_is_multiple_of` allow line we ship
  for the dev toolchain.

### Versions bumped

- `skeg-multi-tenant` 0.1.0 (first publish)

## [0.3.7] - 2026-06-08

### Added

- **Child spans inside `VSEARCH`.** `DiskVamanaIndex::search_with_l`
  now emits `vsearch.walk` (fields: `list_size`, `rerank`, `early`,
  `visited`, `returned`) and `vsearch.rerank` (fields: `candidates`,
  `disk_reads`, `skipped`) nested under the `vsearch` parent the
  handler creates. Traces shipped through the OTLP exporter carry the
  full hierarchy without extra configuration.

- **`compat-tests/redis_py_compat.py`** - end-to-end smoke against a
  live `skeg-resp3` driven through `redis-py` 5+. Exercises every
  typed command from v0.3.5, asserts byte-exact error strings, and
  validates RESP2/RESP3 negotiation. Manual gate (not part of `cargo
  test`); runs in ~3 seconds and exits non-zero on any divergence.

### Changed

- **`skeg-server::resp3_handler`** folds the v0.3.5 `_typed` wrapper
  layer into the `dispatch_command` match arms and drops the legacy
  `dispatch_unknown` / `dispatch_skeg` arms that the typed parse path
  made unreachable. Net: about 190 lines removed, identical wire
  behaviour preserved.

### Versions bumped

- `skeg-server` 0.3.2 -> 0.3.3 (handler simplification, span
  hierarchy; no new public API surface)

## [0.3.6] - 2026-06-07

### Added

- **Default-on Prometheus exporter.** The `metrics-http` cargo feature
  is now part of `skeg-server`'s default set, so released binaries
  serve `/metrics` out of the box when `--metrics-port <PORT>` is
  passed. Drop it with `--no-default-features` to get a slim build;
  the `--metrics-port` flag stays parseable and logs a warning when the
  feature is off.

- **OTLP/gRPC tracing exporter** (`tracing-otlp` feature, default-on).
  When `SKEG_TRACE_OTLP_ENDPOINT=<url>` is set, spans flow to an
  OpenTelemetry collector through a `tracing-opentelemetry` bridge with
  head-based sampling. Env knobs:

  - `SKEG_TRACE_OTLP_ENDPOINT`: OTLP/gRPC URL (unset disables export).
  - `SKEG_TRACE_SAMPLE_RATE`: `[0.0, 1.0]` head sampler, default `1.0`.
  - `SKEG_TRACE_RESOURCE_ATTRS`: `k1=v1,k2=v2` resource labels.

  Both binaries (`skeg`, `skeg-resp3`) install the layer; the exporter
  is shut down cleanly on exit so in-flight spans flush.

- **`vsearch` span** emitted from both protocol handlers (binary +
  RESP3), with structured fields: tenant, vindex, k, l_search,
  vector_dim, hits. Resource attributes include `service.name=skeg`
  and `service.version=<crate version>`.

- **Documentation and operator assets.** New `OBSERVABILITY.md` covers
  the metric schema, a Prometheus scrape config, OTel collector
  integration, and the tracing overhead numbers. `assets/grafana/`
  ships an overview dashboard JSON, `prometheus.yml`, and a starter
  `otel-collector-config.yaml`.

- **Tracing overhead microbench** at
  `crates/skeg-server/benches/tracing_overhead.rs`. Measured on M1 Pro:
  1.54 ns/span with subscriber installed and filter dropping; 1116
  ns/span when fully serialised. The G-O3.1 gate (under 5% relative to
  the no-subscriber baseline) clears with a ~70x margin.

### Changed

- **`skeg-vector` sort sites** switched from `sort_unstable_by(|a, b|
  b.0.cmp(&a.0))` to `sort_unstable_by_key(|x|
  std::cmp::Reverse(x.0))` where the key type is `Copy`. Identical
  ordering; quiets the new clippy 1.96 `unnecessary_sort_by` lint
  without changing behaviour.

### Versions bumped

- `skeg-server` 0.3.1 -> 0.3.2 (new default features, span
  instrumentation; depends on the same skeg-vector / skeg-resp3 as
  v0.3.5)

## [0.3.5] - 2026-06-04

### Added

- **`skeg-resp3` typed `Command` variants** for the 21 KV and `SKEG.*`
  verbs the server already exposed via the legacy untyped fallback. KV
  (11): `GET`, `SET`, `DEL`, `EXISTS`, `MGET`, `MSET`, `INCR`, `DECR`,
  `INCRBY`, `DECRBY`, `SELECT`. `SKEG.*` (10): `STATS`, `SHARDS`,
  `WHOAMI`, `AUTH`, `VINDEX.LIST`, `VINDEX.CREATE`, `VINDEX.DROP`,
  `VSET`, `VDEL`, `VSEARCH`. The new `CommandError` variants
  (`WrongArity`, `WrongAritySkeg`, `SelectDbOutOfRange`,
  `SelectInvalidIndex`, `NotAnInteger`) render the same error strings
  the server emitted byte-for-byte, so existing `redis-cli` scripts and
  Redis client libraries that match on `ERR ...` text keep working.

  Effect on consumers:

  - `skeg-resp3` is now the source of truth for command arity + simple
    arg parsing. Downstream crates (`skeg-server`, future
    `skeg-client-rs` / `skeg-py`) get the same typed dispatch instead
    of duplicating the validation logic.
  - 180 unit tests in `skeg-resp3` (43 new typed-command tests + 5
    proptests covering KV byte-preservation across random inputs).
  - Verified end-to-end with `redis-cli` on the live binary: 14
    commands round-trip with identical error strings.

### Changed

- **`skeg-server::resp3_handler` dispatches the new typed variants
  directly**, bypassing the legacy `dispatch_unknown` for KV and
  `SKEG.*`. The legacy untyped path is preserved for genuinely unknown
  commands (`FOO bar` -> `ERR unknown command 'FOO'`).

### Versions bumped

- `skeg-resp3` 0.1.2 -> 0.1.3 (added public `Command` and
  `CommandError` variants; backwards-compatible additive change)
- `skeg-server` 0.3.0 -> 0.3.1 (depends on `skeg-resp3` 0.1.3; dispatch
  rewired)

## [0.3.4] - 2026-06-04

### Added

- **`skeg-simd` block-32 SIMD scoring for TurboQuant 4-bit codes.**
  New public API: `build_tq4_lut_f32`, `interleave_tq4_codes`,
  `quantize_tq4_lut_u8`, `tq4_block32_score_scalar`,
  `tq4_block32_score_u8_scalar`, and (on aarch64)
  `tq4_block32_score_u8_neon`. The block kernel scores 32 vectors in
  parallel via `vqtbl1q_u8` lookups into a per-query u8 LUT with a
  periodic widening flush to f32. Measured on M1 Pro, 100k synthetic
  vectors, single-thread flat scan:

  - dim=384: row 275 QPS -> block 1359 QPS (4.93x)
  - dim=1024: row 107 QPS -> block 515 QPS (4.80x)
  - dim=1536: row 73 QPS -> block 342 QPS (4.71x)

- **`skeg-vector::FlatIndex::search_block_tq4`**: opt-in entry point
  routing the flat scan through the block-32 kernel for the
  `TurboQuant { bits: 4 }` tier. Returns `None` for other tiers so the
  caller falls back to the row-major `search` path. Recall is
  equivalent to row scoring (proptest verified, |delta| < 1e-3).

- **`skeg-vector` pq32 / pq64 aliases**: cache-fit experiments for
  PQ with smaller `m` values.

- **Diagnostic benches**: `flat_block_throughput`,
  `flat_block_pareto`, `vamana_rerank_pipe`.

### Changed

- **`skeg-simd::tq4_adc_i8_neon`**: further +100% via 8-accumulator
  unroll + native nibble unpack (on top of v0.3.3's +35-66%). 32 coords
  per outer iteration over 8 independent f32x4 accumulator chains,
  replacing the SWAR+stack-roundtrip nibble unpack with a NEON-native
  `vand_u8 + vshr_n_u8 + vzip1_u8 + vzip2_u8 + vcombine_u8` sequence
  (~6 cycles/chunk saved). Bit-equivalent to the previous kernel within
  the existing equivalence test tolerance.

- **`skeg-simd::tq2_adc_i8_neon`**: +70-74% via the same NEON-native
  nibble unpack pattern adapted to 2-bit codes.

### Versions bumped

- `skeg-simd` 0.1.3 -> 0.1.4 (added block-32 API; row kernels still
  binary-compatible)
- `skeg-vector` 0.1.2 -> 0.1.3 (added `FlatIndex::search_block_tq4`)

## [0.3.3] - 2026-06-02

### Changed

- **`skeg-simd::tq4_adc_i8_neon` and `tq2_adc_i8_neon`**: refactor the
  TurboQuant ADC NEON kernels to use multiple independent f32x4
  accumulators (4 for tq4, 8 for tq2) and defer the `i8_scale`
  multiply until after the horizontal sum. Breaks the serial-FMA
  dependency chain that previously throttled per-row throughput.

  Measured on M1 Pro, 100k synthetic vectors, single-thread flat
  scan via `cargo bench -p skeg-vector --bench flat_throughput`:

  - tq2 at dim=1024: 32 QPS -> 55 QPS (+72%)
  - tq2 at dim=1536: 21 QPS -> 38 QPS (+81%)
  - tq4 at dim=1024: 32 QPS -> 43-53 QPS (+35-66%)
  - tq4 at dim=1536: 21 QPS -> 35 QPS (+67%)

  Result is bit-equivalent to the previous kernel within the
  existing equivalence test tolerance (float multiply distributes
  over the scale factor; per-test 1e-5 budget).

### Versions bumped

- `skeg-simd` 0.1.2 -> 0.1.3 (kernel refactor, behaviour unchanged)

## [0.3.2] - 2026-06-02

### Fixed

- **`release.yml`**: the publish-crates loop hard-coded the list of
  crates and was missing `skeg-tenant` and `skeg-server-tenant`. They
  shipped to GitHub Releases in v0.3.1 but never reached crates.io.
  The loop is now extended to include them, and both crates are
  bumped to 0.1.1 so the diff-based skip logic re-publishes them.

### Versions bumped

- `skeg-tenant` 0.1.0 -> 0.1.1
- `skeg-server-tenant` 0.1.0 -> 0.1.1

## [0.3.1] - 2026-06-01

### Added

- **`skeg-tenant` crate.** Multi-tenant primitives shipped as a
  first-class workspace member: tenant id (xxh3_128), argon2id
  password hashing, HMAC-SHA256 tokens, on-disk auth store
  (`auth.kdb`), quota tracker. Apache-2.0, same as the engine.
- **`skeg-server-tenant` crate.** Multi-tenant server binary
  that wraps `skeg-server` and installs `skeg-tenant` as the
  `TenantBackend`. Ships a binary called `skeg-server` (same name
  as the OSS one; different package). Two extra flags vs the OSS
  server: `--tenant-auth <path>` (enable tenant resolution against
  an `auth.kdb`) and `--tenant-strict` (reject anonymous `HELLO 3`).
  Apache-2.0.

### Versions bumped

- `skeg-tenant` (new) 0.1.0
- `skeg-server-tenant` (new) 0.1.0

## [0.3.0] - 2026-06-01

### Added

- **`SharedCommitter` for multi-shard write throughput on Apple Silicon.**
  `F_FULLFSYNC` (the macOS power-loss durability call) is a device-wide
  barrier: concurrent fsyncs on different files serialize at the
  hardware. Previously, multi-shard write throughput on macOS regressed
  going from 1 shard to 4 shards because each shard issued its own
  barrier. The new process-wide `SharedCommitter` aggregates pending
  writes from every shard into a single fsync per batch, amortising
  the barrier across all shards.

  Measured on a MacBook Pro M1 (1000 power-durable appends per shard,
  128 byte records, 5 runs each variant, median wall clock):

  | variant       | durability model | shards | median  | ops/s   |
  | ------------- | ---------------- | ------ | ------- | ------- |
  | 1sh_perfile   | per-file fsync   | 1      | 6.60 s  | 152     |
  | 4sh_perfile   | per-file fsync   | 4      | 20.52 s | 195     |
  | 1sh_devglobal | shared committer | 1      | 6.93 s  | 144     |
  | 4sh_devglobal | shared committer | 4      | 7.87 s  | **508** |

  4-shard shared committer recovers to within 1.19x of the 1-shard
  baseline (cap 1.5x), a 2.61x improvement over the per-shard fsync
  regression (floor 1.5x). A 100-iteration random-seed crash-recovery
  test passes with zero data loss.

- **Per-platform durability dispatch via `skeg-platform::DurabilityModel`.**
  Linux keeps the per-file group committer (per-file `fdatasync`
  parallelism is already efficient there); macOS routes through the
  shared committer. The model is detected at runtime via
  `resolve_durability_model()`. Tests can override the model via the
  `testing` feature of `skeg-platform`.

### Changed

- **`skeg-core::group_commit::GroupCommitter::start` is now `async`.**
  Required because the shared-committer arm attaches the file to its
  internal registry before the first append. In-tree callers
  (`VLog::open`, `VLog::maybe_rotate`) are already updated to `.await`.
  **Breaking** for any out-of-tree user of `GroupCommitter`.

### Versions bumped

- `skeg-core` 0.1.3 -> 0.2.0 (breaking: `GroupCommitter::start` async)
- `skeg-platform` 0.1.2 -> 0.1.3 (additive: `DurabilityModel`)
- `skeg-server` 0.2.2 -> 0.3.0

## [0.2.2] - 2026-05-29

### Changed

- **Per-vindex locks (Q11 phase 2).** The shard's vindex set was
  previously wrapped in a single `RwLock<HashMap<String,
  VectorBackend>>`, so any `VSET` / `VSEARCH` held the outer write
  lock for its entire duration and blocked operations on every
  other vindex on the same shard. Each entry is now its own
  `Arc<RwLock<VectorBackend>>`:
  - The outer map is held only for the lookup, then released.
  - The per-vindex lock serialises operations on the **same**
    vindex (still required: `VectorBackend::search` mutates the
    working-set cache and the streaming-insert buffer).
  - The worker-pool path (`--workers N` since v0.1) now lifts two
    concurrent `VSEARCH` calls on different vindexes to the
    blocking pool without contention.
  - SoL gate (`test_per_vindex_locks_concurrency_gate`): two-thread
    workload at 2,000-vector, 256-dim flat indexes, `workers=2`,
    requires `baseline / concurrent >= 1.4x`. Measured 1.99x on
    Apple M1 (theoretical max 2.0x). Floor sits below the measurement
    to absorb noise on slower CI runners.

### Notes

- Wire format unchanged, public `ShardSet` API unchanged. Existing
  callers see no behavioural change beyond the new parallelism on
  multi-vindex workloads.
- `VINDEX DROP` keeps its previous semantics: the entry is popped
  from the map before the data directory is removed. In-flight
  operations on the dropped vindex hold their own `Arc` clone and
  finish their inner lock window before dropping it; on POSIX the
  directory deletion is decoupled from the file-handle lifetime.

## [0.2.1] - 2026-05-29

### Added

- All seven engine gauges now report live values; the five that read
  `0` in v0.2.0 are wired:
  - `skeg_vlog_segments_live`     refreshed on each `SKEG.STATS` call
    from `VLog::segment_count()`.
  - `skeg_vlog_total_bytes`       refreshed similarly via the new
    `VLog::disk_bytes_total()` helper (sealed × max_seg_size + active
    write offset, no `stat()`).
  - `skeg_compaction_in_progress` and
    `skeg_vlog_segments_compacting` use an RAII guard in
    `VLog::compact_segment` so every return path (including `?` and
    early no-ops) leaves the gauges balanced.
  - `skeg_vindex_vectors` and `skeg_vindex_size_bytes` are aggregated
    on `STATS` from the shard's vindex set; the size approximation is
    `n * dim * 4` for flat indexes and `n * dim` for disk (tier-1
    int8 codes only; the graph and full f32 vectors live on disk and
    are not counted as RAM).
- `skeg_telemetry::incr_gauge` / `decr_gauge` / `add_gauge` for
  delta-style updates on gauges (used by the compaction RAII guard).
  The closed-enum and dynamic registry APIs both gain the new
  helpers; same `#[inline(always)]` / no-op compile-out story.

### Notes

- v0.2.1 is an additive release: no engine behaviour change, no wire
  format change, no schema change. Dashboards built against v0.2.0
  start displaying real values on the previously-empty gauges with
  no rewrite.

## [0.2.0] - 2026-05-28

### Added

- **`skeg-telemetry` dynamic registry.** Downstream consumers
  (`skeg-kv-cache`, `skeg-tenant`, applications on top of skeg) can now
  register their own counters, histograms, and gauges without patching
  the engine's closed enums.
  - `register_counter(name) -> &'static [AtomicU64; MAX_SHARDS]`
  - `register_histogram(name) -> &'static DynHistogram`
  - `register_gauge(name) -> &'static AtomicU64`
  - `register_op!("base_name")` macro: derives `<name>_total` (sharded
    counter) and `<name>_duration_seconds` (histogram) from a single
    base, returns a `DynOp` bundling both.
  - Idempotent: repeated calls with the same name return the same
    `&'static` handle, so `OnceLock::get_or_init(|| register_*(…))`
    patterns are safe.
  - Pool sizing: 256 sharded counter slots (64 KiB), 64 histogram
    slots (14 KiB), 64 gauge slots (512 B). All `static` `AtomicU64`s,
    no allocator on the hot path.
  - Hot-path cost: same shape as the closed-enum path (one
    `OnceLock::get` branch + one or two `fetch_add(Relaxed)`). Measured
    ~2 ns on Apple M1.
- Histogram buckets extended from 22 to 26 (1 µs → 16.78 s upper
  bound, then `+Inf`). The previous range clipped any observation
  ≥ 1.05 s into the sentinel; downstream consumers with longer-tail
  operations (`skeg-kv-cache` blob restore, tenant quota probes) now
  observe the real distribution out to ~16 s.
- `SKEG.STATS` and `/metrics` output now appends the dynamic registry
  contents (sorted by name) after the engine's static metric block.
  Engine schema is grep-stable; downstream metrics are added below the
  blank-line separator.

### Changed

- **Per-crate versioning** (workspace-level): each crate now carries
  its own `version` field instead of inheriting from
  `[workspace.package]`. The release workflow diffs each crate's tree
  against the previous tag and skips `cargo publish` for unchanged
  crates, so a release that touches only `skeg-telemetry` no longer
  republishes the seven other crates as identical no-op bumps on
  crates.io.
- `skeg-server` bumped to `0.2.0` to track the new telemetry dep
  surface and stay aligned with the user-facing release tag (the
  Homebrew / GitHub Release naming follows the tag).

### Notes

- `skeg-core` keeps version `0.1.2` on crates.io: its source did not
  change in this release. The local workspace dep requirement was
  bumped to `skeg-telemetry = "0.2"` so the bench/dev path build
  works; a future skeg-core release will carry that requirement out.
- The seven unwired gauges from v0.1.2 (`VlogSegmentsLive`,
  `VlogSegmentsCompacting`, `VlogTotalBytes`, `CompactionInProgress`,
  `VindexSizeBytes`, `VindexVectors`) are still TODO. The schema is
  stable; dashboards written against v0.1.2 keep working.

## [0.1.2] - 2026-05-28

### Added

- **Telemetry.** New `skeg-telemetry` crate provides zero-overhead
  `AtomicU64` counters and exponential histograms for the hot path.
  - Per-op counters (`skeg_ops_total{op}`) and duration histograms
    (`skeg_op_duration_seconds_*`) for `get`, `set`, `del`, `vset`,
    `vsearch`, `vdel`, `ping`.
  - Cache counters: `skeg_cache_hits_total`, `_misses_total`,
    `_evictions_total` (wired in `skeg-core/cache`).
  - vLog counters: `skeg_vlog_syncs_total`,
    `_group_commit_batches_total`, `_compaction_runs_total`,
    `_compaction_bytes_total`. Gauge `skeg_vlog_live_bytes`
    refreshes on each `STATS` call.
  - Measured hot-path cost: `record_op` ≈ 4.7 ns on Apple M1;
    `criterion` gate in `crates/skeg-telemetry/benches/overhead.rs`.
  - Three exposure modes:
    - default: counters compiled in, dump via RESP3 `SKEG.STATS`.
    - `--no-default-features` on the crate: every public function
      is a compile-out `#[inline(always)]` no-op.
    - `--features metrics-http` on `skeg-server`: tiny HTTP
      exporter on a dedicated thread, surfaces `/metrics` in
      Prometheus text format.
- **`--metrics-port <PORT>` CLI flag** on `skeg` (and the matching
  `SKEG_METRICS_PORT` env). Spawns the Prometheus exporter on
  `127.0.0.1:PORT` when the binary is built with `metrics-http`.
- `SKEG.STATS` response is now extended with the full telemetry
  dump after the legacy `cache_bytes=…` summary line, separated by
  a blank line. Old clients that grep for `cache_bytes=` keep
  working unchanged.

### Notes

- Five gauges remain unwired in this release
  (`VlogSegmentsLive`, `VlogSegmentsCompacting`, `VlogTotalBytes`,
  `CompactionInProgress`, `VindexSizeBytes`, `VindexVectors`). They
  read `0` from `STATS` and `/metrics`. Wiring sites are marked
  with `TODO(telemetry):` comments in `vlog.rs` and `shard.rs` and
  will land in a follow-up; the schema is stable and dashboards
  written today will not need to change.

## [0.1.1] - 2026-05-26

### Added

- `skeg --help`, `skeg -h`, `skeg --version`, `skeg -V` (and the same on
  `skeg-resp3`). The binaries now print a usage block and exit
  cleanly instead of starting the server when these flags are
  passed. Unblocks the canonical `brew install` smoke test.

### Fixed

- README quickstart used the wrong vector command syntax. All vector
  operations are namespaced under `SKEG.*` and take positional args
  (`SKEG.VINDEX.CREATE <name> <dim> <kind> <backend>`), not the
  `VINDEX.CREATE docs DIM 1024 METRIC cosine` form that the previous
  README implied.

## [0.6.1] - 2026-07-11 - pre-release v0.1.0

### Added

- New `tenant` Cargo feature (default off) in `skeg-server`. When
  enabled at compile time, the server accepts a `TenantContext`
  provided by an external crate and scopes KV and vector ops per
  tenant. Without the feature the engine ships as a pure single-tenant
  store with byte-identical wire and disk layout to pre-tenancy code.
- `tune_socket` applies `TCP_NODELAY` and `SO_KEEPALIVE` (60s idle, 10s
  probe interval) on every accepted server connection and every client
  connection from `SkegClient`. Catches half-open TCP states that
  otherwise leak file descriptors in long-running deployments.
- `--workers <N>` (env `SKEG_WORKERS`) dispatches `VSEARCH` requests to
  `tokio::task::spawn_blocking` so KV ops on the same shard do not
  queue behind multi-ms searches. Default `0` keeps the inline
  behaviour that matches the public benchmark numbers.
- `--tier-mmap` (env `SKEG_TIER_MMAP`): the TurboQuant `codes` buffer
  is persisted to `tier.cache.bin` at open and memory-mapped. The OS
  page cache can reclaim tier pages under pressure instead of pushing
  anonymous memory to swap.
- `--graph-mmap` (env `SKEG_GRAPH_MMAP`): `graph.vmn` is opened as a
  memory map and the `Node` array is reinterpreted directly from the
  mmap'd bytes. No per-`Node` parsing at open, OS page cache reclaim
  on the graph. Combines with `--tier-mmap` to make the whole disk
  index paginable.
- New RESP3 verbs in the `SKEG.*` namespace for vector ops:
  `SKEG.VINDEX.CREATE`, `SKEG.VINDEX.DROP`, `SKEG.VINDEX.LIST`,
  `SKEG.VSET`, `SKEG.VDEL`, `SKEG.VSEARCH`. Vector payloads are RESP
  bulk-strings carrying raw little-endian `f32` bytes; length must be
  exactly `dim * 4`. When the `tenant` feature is on, names are scoped
  per tenant; otherwise they pass through unchanged.
- Workspace metadata: root `Cargo.toml` declares `[workspace.package]`
  with `edition = "2024"` and `rust-version = "1.86"`. Older
  toolchains get a clear error instead of a cryptic build failure.

### Changed

- `set_speed_enabled` returns `Result<(), SpeedAlreadySet>` instead of
  `Result<(), ()>`. Caller can format or log a meaningful message;
  behaviour is unchanged.

### Documentation

- `README.md`, `CHANGELOG.md`, `LICENSE`, `NOTICE`, and `SECURITY.md`
  added at the repo root for the public release.
