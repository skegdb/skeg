# skeg-rigging-net

> *Network transports for `skeg-rigging`: federate hansa peers across
> machines without modifying the engine.*

This workspace ships the network-attached side of the `skeg-rigging`
ecosystem. The reference single-tenant adapter
[`skeg-rigging-skeg`](https://github.com/skegdb/skeg-rigging) links
`skeg-vector` in-process; this repo's adapters talk to a remote
skeg-server over the wire.

## Crates

| Crate | Transport | Status | Use case |
| --- | --- | --- | --- |
| `skeg-rigging-net` | (none - shared types) | v0.1 | wire envelope, error, [`TenantLocation`] |
| `skeg-rigging-net-resp3` | RESP3 over TCP | **v0.1 alpha** | hansa peer queries against a skeg-server |
| `skeg-rigging-net-http` | HTTP/1.1 | planned | saga distribution + alternate query path |
| `skeg-rigging-net-http-async` | HTTP/2 async | planned | high-fanout deployments |

The shared crate is intentionally tiny: it only defines the
[`TenantLocation`] enum (so hansa can address peers across mixed
transports) and a [`RecordEnvelope`] convention for KV stores that
don't natively carry the `shareable` flag.

## Why a separate workspace

The single-tenant Apache adapter and the BUSL bridge to skeg-tenant
live in their own repos because their licenses differ
(Apache-2.0 vs BUSL-1.1). Network transports are Apache-2.0 across
the board - they only consume skeg's *public* protocol, never modify
skeg itself.

## Architecture

The federation path with RESP3 looks like:

```text
Agent A (Mac)                            Agent B (Linux box)
  ┌─────────────────┐                      ┌──────────────────┐
  │ hansa lib       │                      │ skeg-server      │
  │  membrane       │                      │   (RESP3 :6379)  │
  │                 │                      │                  │
  │ PeerOpener──RESP3──VSEARCH + MGET──────┼─→  hansa index   │
  │                 │                      │    + KV envelopes│
  └─────────────────┘                      └──────────────────┘
        │
        └── local saga dir (~/.hansa/<id>/sagas/<peer>.saga)
            populated out-of-band (e.g. via HTTP side-channel
            or shared filesystem; not in this crate yet)
```

The RESP3 path uses three commands that skeg-server already exposes:

- `SKEG.VINDEX.LIST` - resolve the tenant's vector index dim + count.
- `SKEG.VSEARCH name k l_search vector_bytes` - top-k nearest neighbours.
- `MGET hansa:rec:<id1> hansa:rec:<id2> ...` - fetch the JSON envelope
  per hit. Decode `shareable` + tags client-side, apply filter, return.

The convention key prefix `hansa:rec:` and the JSON envelope layout are
defined in [`skeg-rigging-net::RecordEnvelope`](./crates/skeg-rigging-net/src/envelope.rs).

## Quick start (mock-server example)

The integration test in
[crates/skeg-rigging-net-resp3/tests/mock_roundtrip.rs][mock] spawns a
tiny RESP3 server on loopback and drives it through a `Resp3Tenant`:

```rust
use skeg_rigging::prelude::*;
use skeg_rigging_net_resp3::Resp3Tenant;

let tenant = Resp3Tenant::connect(
    "127.0.0.1:6379",
    TenantId::from_bytes([1; 16]),
    Some(("alice", "secret")),
)?;

let hits = tenant.query_filtered(
    &[0.1; 768],
    /* top_k */ 10,
    &|m: &RecordMeta<'_>| m.shareable,
)?;
```

## What v0.1 covers

- Synchronous TCP client (`std::net::TcpStream` + `skeg-resp3` framing).
- `HELLO 3 [AUTH user pass]` handshake.
- Read-only query path (`SKEG.VSEARCH` + `MGET` + post-filter).
- `ReadOnlyView`, `IterVectors` (returns empty over the network - see
  below), `QueryFiltered` impls.
- `Resp3Writer` helper for seeding records from the owner side or in
  tests.

## What v0.1 does **not** cover

- **`iter_vectors` over the network**: skeg-server has no `VSCAN` op.
  In hansa's design `iter_vectors` is only invoked locally (saga
  build runs on the owner's machine), so the missing op never blocks
  the federation path. The trait impl returns an empty iterator and
  documents the limit.
- **Saga distribution**: peer sagas live in `~/.hansa/<id>/sagas/`.
  This crate doesn't move them across machines yet - that's the job
  of the upcoming `skeg-rigging-net-http` companion.
- **Async / HTTP/2**: planned crates listed above.

## Wiring into hansa

Hansa's `PeerOpener` is `Arc<dyn Fn(&Path) -> Result<Box<dyn ReadOnlyView>, OpenError>>`.
For the RESP3 path the "path" carried in `MemberRecord` is a
`TenantLocation` JSON blob (or a `resp3://host:port` URL via
[`parse_location`]). The opener parses the location and dispatches:

```rust
use std::sync::Arc;
use skeg_rigging::OpenError;
use skeg_rigging_net::{TenantLocation, parse_location};
use skeg_rigging_net_resp3::Resp3Tenant;

let opener = Arc::new(|path: &std::path::Path| {
    let url = path.to_str().ok_or(OpenError::NotFound)?;
    let loc = parse_location(url).map_err(|_| OpenError::NotFound)?;
    match loc {
        TenantLocation::Resp3 { endpoint, auth } => {
            let auth_ref = auth.as_ref().map(|(u, p)| (u.as_str(), p.as_str()));
            let t = Resp3Tenant::connect(&endpoint, skeg_rigging::TenantId::ZERO, auth_ref)
                .map_err(|_| OpenError::NotFound)?;
            Ok(Box::new(t) as Box<dyn skeg_rigging::ReadOnlyView>)
        }
        _ => Err(OpenError::NotFound),
    }
});
```

A future hansa minor will expose a transport-aware `PeerOpener` factory
so this boilerplate disappears.

## Building

```sh
cargo build --workspace
cargo test --workspace
```

## License

Apache-2.0.

[mock]: ./crates/skeg-rigging-net-resp3/tests/mock_roundtrip.rs
[`TenantLocation`]: ./crates/skeg-rigging-net/src/location.rs
[`RecordEnvelope`]: ./crates/skeg-rigging-net/src/envelope.rs
[`parse_location`]: ./crates/skeg-rigging-net/src/location.rs
