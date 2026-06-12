# Ecosystem

skeg is the storage core. The companion crates ship as their own published libraries.

- [hansa](https://crates.io/crates/hansa) and [hansa-cli](https://crates.io/crates/hansa-cli). Federation primitive: local AI agents form trust groups and query across each other's skeg instances.
- [skeg-rigging](https://crates.io/crates/skeg-rigging). Public trait surface (`TenantWrite`, `Embed`) for plugging skeg into ingest and federation pipelines.
- [skeg-rigging-skeg](https://crates.io/crates/skeg-rigging-skeg). The local DiskVamana backend implementation of those traits.
- [skeg-rigging-net](https://crates.io/crates/skeg-rigging-net), [-http](https://crates.io/crates/skeg-rigging-net-http), and [-resp3](https://crates.io/crates/skeg-rigging-net-resp3). Network adapters that federate against a remote skeg server through HTTP or RESP3.
- [skeg-rigging-ingest](https://crates.io/crates/skeg-rigging-ingest). Reusable text-to-embeddings-to-tenant pipeline: walk a tree, chunk, embed via Ollama, write through any `TenantWrite`. The `watch` feature enables live re-ingest.
