# skeg-rigging

> *Public extension points for memory engines and the plugins that
> build on top of them.*

`skeg-rigging` is the running rigging of the skeg ecosystem: a small
trait-based contract that lets plugins like
[hansa](https://github.com/skegdb/hansa) read and query a tenant
without knowing which engine is behind it. The reference adapter
`skeg-rigging-skeg` wraps `skeg-vector::FlatIndex` for the canonical
case; alternative engines (graph, multimodal, temporal, mock) can
implement the same trait set and be drop-in interchangeable.

## v0.1 scope

Three Stable traits and one Provisional trait. The set is small on
purpose - adding traits later is easy, removing them is not.

| Trait          | Stability   | Purpose                                                          |
| -------------- | ----------- | ---------------------------------------------------------------- |
| `IterVectors`  | Stable      | Stream `(RecordId, Vec<f32>)` for digests / auxiliary indexes    |
| `QueryFiltered`| Stable      | Filtered similarity search; takes `&dyn Filter` for object safety|
| `Filter`       | Stable      | Predicate over `RecordMeta` (closures auto-impl)                 |
| `ReadOnlyView` | Provisional | Concurrent multi-process read access from a path                 |

Shared types: `RecordId`, `TenantId`, `RecordMeta`, `Hit`,
`QueryError`, `OpenError`. All `#[non_exhaustive]` where it matters.

## Engine pluralism

Rigging is engine-neutral by design. The trait set is named for the
capability it exposes, not for the data model underneath. A vector +
KV engine (skeg-core), a graph engine, a temporal engine all satisfy
the contract. The convention for capability identifiers in
`TenantInfo::capabilities()` (v0.2+) is `<engine>.<capability>`:

- `vector.kv` - skeg-core baseline (and the reference adapter here)
- `graph.traverse` - hypothetical graph engine
- `temporal.windowed` - hypothetical temporal engine
- `hansa.member` / `hansa.membrane` - plugin-side capabilities

## Adapter - `skeg-rigging-skeg`

A working impl of the v0.1 trait surface backed by skeg's public
engine library:

```rust
use skeg_rigging::prelude::*;
use skeg_rigging_skeg::Tenant;

let tenant = Tenant::open("/path/to/dir", TenantId::ZERO, 768)?;
tenant.insert(
    RecordId(1),
    vec![0.1; 768],
    /* shareable */ true,
    vec!["topic".into()],
    b"payload".to_vec(),
)?;
tenant.flush()?;

let hits = tenant.query_filtered(
    &[0.1; 768],
    /* top_k */ 10,
    &|m: &RecordMeta<'_>| m.shareable,
)?;
```

The adapter uses a `meta.json` sidecar for the `shareable` flag and
tags (skeg-core does not model them natively). v0.2 will swap to
`skeg-core::VLog` + `DiskVamanaIndex` for scalable persistence.

## Naming

The capability concept is `TenantId`. The auth/quota `TenantId` in the
separate `skeg-tenant` crate is orthogonal - different namespace,
different concern, the two coexist.

## Building

```sh
cargo build --workspace
cargo test --workspace
```

## License

Apache-2.0.
