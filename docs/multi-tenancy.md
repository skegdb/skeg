# Multi-tenancy

skeg runs many tenants on one node with real per-tenant isolation, not just
prefix namespacing. This guide covers the tenant binary, key scoping,
per-tenant quotas, the admin commands that set them, and the fair cache
eviction that keeps a noisy tenant from starving a quiet one.

The engine is single-tenant by default and pays nothing for any of this when
only one tenant (or anonymous traffic) is present. Everything below activates
only once a deployment opts in.

## The tenant binary

Multi-tenancy ships in three Apache-2.0 crates: `skeg-tenant` (the tenant
model), `skeg-server-tenant` (the server that resolves and isolates tenants),
and `skeg-multi-tenant`. The end-user binary is named `skeg-server`.

```sh
skeg-server \
  --data-dir ./data \
  --addr 127.0.0.1:6379 \
  --tenant-auth ./data/auth.kdb \
  --tenant-strict \
  --admin-tenant ops
```

- `--tenant-auth <path>` enables tenant resolution against an `auth.kdb` on
  disk. A client picks its tenant with `HELLO 3 AUTH <user> <pass>` (argon2id).
- `--tenant-strict` rejects anonymous `HELLO 3` (no AUTH). Without it,
  anonymous connections map to tenant zero, and the server treats the bind
  as unauthenticated: a non-loopback `--addr` is refused unless
  `--allow-unauthenticated-network` opts in.
- `--admin-tenant <name>` (or the `SKEG_ADMIN_TENANT` environment variable)
  names the one tenant allowed to run the quota admin commands below.

## Key scoping

Every tenant gets an isolated keyspace. Keys, vector indices, cache residency,
and on-disk bytes are all scoped to the resolved tenant; one tenant cannot read
or evict another's data. The scope is carried as a first-class tenant view
through the engine rather than threaded as a parameter, so the isolation
boundary is structural, not a filter applied after the fact.

## Per-tenant quotas

A deployment can cap two resources per tenant:

- `max_vectors`. The number of vectors a tenant may hold. Checked on
  `SKEG.VSET`, under the index write lock, so an insert is counted exactly once
  and overwriting an existing id stays free. An over-limit insert is rejected
  before anything is stored.
- `max_disk_bytes`. The live vLog bytes a tenant holds. Checked on `SET`,
  `MSET`, `APPEND`, and on the payload blob of `SKEG.VSET` / `SKEG.VMSET` -
  one counter, one key format, one check, because a blob key carries the same
  16-byte tenant prefix a KV key does. The counter is shared across shards, so
  the limit is global per tenant, and it is rebuilt from the index on restart.
  The check and the counter update happen in one critical section, so
  concurrent writers of one tenant cannot jointly land it past the limit.

Either limit may be left unset (unlimited). With no limit configured the write
path is byte-identical to the non-tenant build.

### What `max_disk_bytes` does and does not count

It counts what the vLog holds LIVE for the tenant, and it holds the limit
against the engine's own maintenance as well as against client writes: a
boundary replica (`SKEG.VINDEX.OVERLAP`) is a second permanent copy of a row's
payload blob, so it is charged - and when it does not fit it is SKIPPED rather
than written, leaving that part of the boundary under-replicated instead of
taking the tenant over its limit. Each skip ticks
`skeg_overlap_replicas_skipped_quota_total`; a later run replicates whatever
fits by then. A reshard MOVE is not charged, because it deletes the source
copy it wrote: it is over the limit by at most one blob record, for the width
of one row.

Outside the number, and bounded by something else:

- **The vector index files.** Vectors, graph, WAL and payload index live in
  the index directory, not the vLog, and `max_disk_bytes` does not see them. A
  payload-less `VSET` grows a tenant's disk with only `max_vectors` bounding
  it. Size the two limits together.
- **Dead records awaiting compaction.** The counter tracks live bytes: an
  overwritten or deleted record stops counting immediately, while its space in
  the segment file returns at the next compaction. The filesystem footprint of
  a churn-heavy tenant is therefore above its counted bytes until then.
- **The transient staging margin.** Prepare-before-commit means one row is two
  physical keys for the width of a write, so a tenant exactly at its limit can
  see a payload-less overwrite refused (see
  `docs/adr-payload-transaction.md`, "Disk quota").
- **The native (non-RESP3) listener**, which has no pluggable tenant backend
  to read a limit from and writes as tenant `0`.

## Setting quotas (admin commands)

An operator on the admin tenant sets quotas at runtime over RESP3:

```text
> SKEG.QUOTA.SET <tenant> <max_vectors> <max_disk_bytes>
OK
> SKEG.QUOTA.GET <tenant>
max_vectors=100000 max_disk_bytes=1073741824
```

`*` means unlimited for either field, for example
`SKEG.QUOTA.SET acme * 1073741824` caps disk only. The commands require an
admin connection (the tenant named by `--admin-tenant`); a non-admin connection
is rejected. Limits are persisted in a sidecar next to `auth.kdb`, so they
survive a restart.

## Fair cache eviction

The hot-key cache (S3-FIFO) is shared across tenants and bounded by a byte
budget. Accounting is per-tenant, and so is eviction: the Main queue's victim
selection is share-aware. It computes an equal share
(`cache budget / active tenants`) once per eviction and, when some tenant is
over its share, briefly skips under-share victims to evict an over-share tenant
instead. A tenant that floods the cache can no longer push another tenant's
small working set out.

The Small queue stays tenant-blind on purpose: it already absorbs scan floods
(one-hit-wonders die there), so no fairness is needed. Fairness runs only when
more than one tenant is resident in a shard; with a single tenant the eviction
path is byte-identical to before. The over-share check is one short-circuiting
scan per eviction, and eviction cost stays flat from one resident tenant to ten
thousand.

## Why this is the wedge

Per-collection vector engines pay a RAM floor for every isolated tenant and cap
the tenant count low, which pushes operators toward soft filter-isolation that
shares one index. skeg isolates at the engine: shared budget, per-tenant
accounting, hard quotas, and fair eviction. That is isolation that holds under a
noisy neighbour at high tenant density, not isolation traded away for density.
