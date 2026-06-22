# ADR: the tenant-tiering seam - mechanism in the engine, policy outside

Status: **accepted** (branch `feat/tenant-admit`).
Context: extends [multi-tenancy.md](multi-tenancy.md). Sibling: the QoS admission
seam (`TenantBackend::admit`) follows the same mechanism/policy split.

## Context

A high-density multi-tenant deployment wants **RAM overcommit**: pack far more
tenants than fit in RAM, keep the hot ones resident, evict the cold ones to disk,
reload on demand. A gate spike measured the prize - an idle open disk-backed
index costs **~1 KB resident RAM per vector, linear, no fixed floor** (768-dim
int8: 1 k vec ~ 1 MB, 10 k ~ 10 MB; `benches/tiering_gate.rs`). At thousands of
tenants that is tens-to-hundreds of GB of pure idle structure. Reclaiming it is a
real, large win, so overcommit is worth building.

But overcommit is two very different things bolted together:

1. **Mechanism** - the engine must be *able* to drop an open index's RAM without
   losing data, bring it back on access, and report what each index costs.
2. **Policy** - *deciding* which tenant to evict and when: a global RAM budget,
   LRU vs working-set, anti-thrash hysteresis, hot-tenant pinning. This is a
   control loop tuned against live telemetry.

The risk of putting both in the engine: the open engine becomes opinionated about
fleet economics it cannot see, and the commercial control plane has nothing left
that is hard to replicate. The risk of putting both outside: impossible - the
policy needs primitives only the engine can provide (it owns the index lifecycle).

## Decision

**The engine ships the mechanism; the policy lives in a separate crate that drives
it.** Same rule as the QoS seam: a generic knob in the Apache engine, the
controller outside.

### What the engine adds (all in `crates/skeg-server`)

- **Non-destructive evict.** `ShardReq::Evict { name }` removes the index from the
  shard's map but **keeps the disk files** (contrast `VINDEX.DROP`, which
  `remove_dir_all`s). RAM frees when the last in-flight `Arc<RwLock<Vindex>>`
  clone drops (RAII), so an evict never races a running query. Allowed in
  read-only/serve mode - it mutates no data.
- **Lazy reopen (off-thread).** `get_or_reopen()` is async and replaces the bare
  map lookups on the six point-get paths. A hit takes only the read lock. A miss
  reopens via `Vindex::recovered` (so `payload_loaded = false` - the first
  *filtered* search rebuilds the payload index from blobs, like crash recovery)
  on `spawn_blocking`, so the shard's single-threaded executor yields to other
  tenants' requests during the disk I/O (bulkhead - one tenant's cold start does
  not stall the shard). No lock is held across the await; a short write lock then
  double-checks (a racing reopen wins, the loser's index is dropped) and inserts.
- **`last_access` tracking.** An `AtomicU64` on `Vindex`, stamped in
  `get_or_reopen` (which all real accesses funnel through). Admin polls
  (`IndexStats`, `VindexList`) deliberately do **not** stamp it - they are not
  tenant accesses.
- **Per-index RAM report.** `ShardReq::IndexStats` returns, per resident index,
  `resident_bytes` (the existing `DiskVamanaIndex::resident_bytes()`; an estimate
  for flat), `last_access_ms`, `vectors`, and `evictable`.

### What the engine exposes (the seam)

```rust
impl Server {
    pub fn control_handle(&self) -> ControlHandle;
}

pub struct ControlHandle { /* clones an Arc'd ShardSet; cheap to clone */ }

impl ControlHandle {
    /// One row per (shard, index) across all shards. The caller aggregates.
    pub async fn open_indices(&self) -> Vec<IndexStat>;
    /// Non-destructive evict on every shard; reopens lazily on next access.
    pub async fn evict(&self, tenant: u128, index: &str) -> Result<bool, ShardError>;
    pub async fn total_resident_bytes(&self) -> usize;
}

pub struct IndexStat {
    pub tenant: u128,         // unscoped from "{tenant}::{index}" (tenant 0 = raw name)
    pub index: String,
    pub shard: usize,
    pub resident_bytes: usize,
    pub last_access_ms: u64,
    pub vectors: usize,
    pub evictable: bool,
}
```

That is the whole surface: **enumerate, report RAM, evict, reload-lazily.** A fork
of the engine gets these knobs - not a controller.

### What stays out (the policy crate)

`skeg-tenant-tiering` (commercial) attaches a background task to `ControlHandle`:
global RAM budget, eviction ordering (LRU / working-set), anti-thrash hysteresis,
hot-tenant pinning, reload-cost awareness. It is almost entirely new work, which
is what makes it defensible - see the commercial repo's
`docs/adr-open-core-boundary.md`.

## Consequences

- The engine is honest open-core: single-tenant and naive-multi-tenant
  (everything resident) work fully; the engine never decides fleet economics.
- `IndexStat` is per-(shard, index), not aggregated. The policy sums across shards
  as it likes - the engine does not impose a logical-index view.
- **Reopen runs off the shard thread** (`spawn_blocking`), so only the triggering
  query pays the cold-start latency (~`resident_bytes` of reads, plus a
  payload-index rebuild if filtered); other tenants on the shard keep serving.
  Validated by `tests/tiering_bulkhead.rs` (other-tenant ops during a reopen:
  ~2 under a synchronous reopen, ~200 off-thread). Remaining ceiling: no in-flight
  dedup, so two racing accesses to the same evicted index may both open it
  (last-wins, the loser is dropped) - upgrade to an in-flight set + `Notify` only
  if that waste is ever measured.
- The mechanism composes with `--tier-mmap`: mmap lets the OS page cold tier codes
  out passively (~75 % of the per-vector cost), while evict additionally frees the
  always-resident graph + ids + delta + file handles. Two levels; the policy uses
  both.

## What this ADR does NOT decide

- The eviction policy itself, the RAM budget default, or pinning - all in the
  commercial crate.
- Multi-node tenant placement (a different concern; a server-side coordinator, not
  this single-process seam).
