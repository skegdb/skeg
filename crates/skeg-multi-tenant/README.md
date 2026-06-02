# skeg-multi-tenant

Multi-tenant orchestration layer on top of [`skeg-tenant`].

Sister crate that combines `skeg-tenant`'s per-tenant primitives
(auth, quota tracker, namespaces) with on-disk tenant directories
into a single orchestrator surface so callers can:

- Open / create a tenant by id.
- Open one by verifying a bearer token (no manual id plumbing).
- Open a *quota-scoped* handle whose every write charges the shared
  `QuotaTracker` before delegating to the storage layer.
- List the tenants currently materialised on disk.
- Hand a tenant out as a `Box<dyn ReadOnlyView>` for hansa-style peer
  queries.

## Why this crate exists separately from `skeg-tenant`

`skeg-tenant` owns the multi-tenant *concept* - auth tokens, quota
counters, tenant ids. It deliberately doesn't pull in the rigging
trait surface or any storage adapter so it stays usable on its own.

`skeg-multi-tenant` is the layer that connects those primitives to a
concrete on-disk storage engine (`skeg-rigging-skeg`) and to the
rigging trait set (`skeg-rigging`). It lives in the `skeg-tenant`
repo because it's owned by the same project, but its dependency cone
is wider.

Direction of deps:

```
skeg-tenant      (auth, quota tracker - no rigging dep)
    ▲
    │
skeg-multi-tenant
    │  ──────────►  skeg-rigging       (trait set)
    │  ──────────►  skeg-rigging-skeg  (on-disk adapter)
    ▲
    │
hansa  (or any other rigging consumer)
```

## What you get

### `MultiTenantRoot`

On-disk root holding one subdir per tenant.

```rust,ignore
use std::sync::Arc;
use std::time::Duration;
use skeg_multi_tenant::{MultiTenantRoot, SkegTenantId};
use skeg_multi_tenant::tenant_primitives::{TokenStore, QuotaTracker};

let store = Arc::new(TokenStore::from_key([7u8; 32], Duration::from_secs(60)));
let tracker = Arc::new(QuotaTracker::new());
let root = MultiTenantRoot::new("/var/lib/skeg")
    .with_tokens(store)
    .with_quota_tracker(tracker);

// Plain read-write open: bypasses auth + quota.
let t1 = root.open(SkegTenantId::from_bytes([0x11; 16]), /* dim */ 384)?;

// Token-verified open: tenant id comes from the token.
let token = /* bytes issued elsewhere */;
let t2 = root.open_with_token(&token, 384)?;

// Quota-scoped handle: every write charges the tracker first.
let h = root.open_scoped(SkegTenantId::from_bytes([0x22; 16]), 384)?;
```

### `TenantHandle`

Wraps a tenant + a shared atomic quota entry. Implements every
rigging trait via forwarding, so it can be passed anywhere a rigging
tenant is expected - including as a `Box<dyn ReadOnlyView>` to
hansa's `PeerOpener`.

```rust,ignore
use skeg_rigging::{Quota, TenantQuota, TenantStats};

let h = root.open_scoped(tid, 384)?;
h.set_quota(Quota {
    max_records: Some(10_000),
    max_bytes:   Some(64 * 1024 * 1024),
})?;

// Writes go through the quota gate first; rejection leaves disk
// state untouched.
h.insert(RecordId(1), embedding, true, vec!["topic".into()], payload)?;
let usage = h.current_usage();   // (records, bytes) snapshot

// Read-side: any rigging trait works through forwarding.
let dim = h.embedding_dim();
let bytes = h.bytes_on_disk();
let hits = h.query_filtered(&query, 10, &filter)?;
```

### Lifecycle

`TenantHandle::snapshot(dest)` forwards to the wrapped adapter.
`destroy` requires moving the inner tenant out (`handle.into_inner()`)
and boxing it as `Box<dyn TenantLifecycle>` - keeps the quota
counters consistent because the orchestrator gets to decide when to
forget the tracker entry.

## Tests + gates

- 24 tests across `multi_tenant_basic`, `lifecycle_composition`,
  `quota`, `handle_robustness` - covers trait-object dispatch,
  concurrent inserts under a shared cap, snapshot round-trip, quota
  refund on delete, adapter-rejection rollback.
- Hansa integration: 3 tests in `hansa::tests::multi_tenant_membrane`
  cover federated query fan-out, destroyed-peer skip, and
  quota-capped tenant participation.

## License

Apache-2.0, matching `skeg-tenant`. Distribution: this crate is
intentionally kept out of crates.io because it has path deps to
sibling private repos (`skeg-hull`, `skeg-rigging`, `skeg-rigging-net`)
that are not all public. Build directly from this repository:

```sh
cargo build --manifest-path crates/skeg-multi-tenant/Cargo.toml
```
