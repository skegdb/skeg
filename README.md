<!-- markdownlint-disable MD033 MD041 -->
<p align="center">
  <img src="assets/skeg-logo.png" alt="skeg" width="480">
</p>

<p align="center">
  <strong>The vector database that fits.</strong><br>
  Multi-tenant, disk-first, RAM-frugal. Recall 1.0 at a fraction of the memory.
</p>

<p align="center">
  <a href="https://crates.io/crates/skeg-server"><img src="https://img.shields.io/crates/v/skeg-server.svg" alt="crates.io"></a>
  <a href="https://github.com/skegdb/skeg/releases"><img src="https://img.shields.io/github/v/release/skegdb/skeg.svg" alt="release"></a>
  <a href="https://github.com/skegdb/skeg/actions"><img src="https://img.shields.io/github/actions/workflow/status/skegdb/skeg/ci.yml?branch=main" alt="CI"></a>
  <img src="https://img.shields.io/badge/MSRV-1.88-orange.svg" alt="MSRV 1.88">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="Apache-2.0"></a>
  <a href="https://github.com/skegdb/skeg-bench"><img src="https://img.shields.io/badge/benchmarks-reproducible-brightgreen.svg" alt="benchmarks"></a>
</p>
<!-- markdownlint-enable MD033 MD041 -->

---

Vector search where RAM is contested: a SaaS packing thousands of tenants on one
box, a RAG service paying for memory by the gigabyte, an agent sharing a machine
with the model it serves. skeg keeps the full vectors on SSD and only a small,
quantized working set in RAM, so it serves at **recall 1.0** on a memory
footprint the RAM-resident engines can't touch. Key-value and vectors in one
engine, a Redis-compatible wire protocol, and multi-tenancy with isolation that
is leak-free by construction.

## Benchmarks

Reproducible from [`skeg-bench`](https://github.com/skegdb/skeg-bench) (public
harness, real embeddings, brute-force ground truth). Measured single-machine on
Apple Silicon; the RAM ratios are hardware-independent.

**Lean and fast.** Single-tenant, 100K x 1024-dim, recall against exact brute
force. Every engine at a reasonable default (LanceDB tuned to recall 1.0 for a
fair fight):

| engine | serve RAM | recall@10 | p50 latency |
| --- | ---: | ---: | ---: |
| **skeg** (tq2) | **47 MB** | **1.000** | **2.5 ms** |
| Milvus Lite | 108 MB | 0.934 | 2.7 ms |
| LanceDB (IVF-PQ) | 198 MB | 0.998 | 59 ms |
| hnswlib (raw HNSW) | 426 MB | 0.985 | 2.0 ms |
| Chroma (HNSW) | 682 MB | 0.985 | 3.9 ms |
| Qdrant (HNSW, f32) | 885 MB | 0.997 | 2.6 ms |

Every other engine gives up at least one axis: RAM, recall, or latency. skeg is
the only one that is leanest, most accurate, and fast at once.

**Co-resident with a model.** A 3B LLM answering RAG over 1M vectors, both on one
M1 Pro (16 GiB). The index stays on SSD, the resident set stays flat:

| Co-resident, 1M vectors | backend RSS p50 | backend RSS max |
| --- | ---: | ---: |
| **skeg** (pq128) | **54 MiB** | **67 MiB** |
| Qdrant (HNSW) | 254 MiB | 2,387 MiB |

<!-- markdownlint-disable MD033 MD041 -->
<p align="center">
  <img src="assets/coresidence-rss.png" alt="Backend RSS while a 3B LLM serves RAG, swept from 10K to 1M vectors on an M1 Pro 16 GiB. skeg stays under 80 MiB; Qdrant climbs into multi-GiB territory." width="760">
</p>
<!-- markdownlint-enable MD033 MD041 -->

The full matrix (engine x scale x tier, p50/p99, recall, RSS), the multi-tenant
density and container-OOM runs, and a cost calculator live on the dashboard:
[`skegdb.github.io/bench`](https://skegdb.github.io/bench/). On a 50M-vector
workload at $4/GB-month that lands at ~$930/year against Qdrant's ~$9,647, at the
same recall.

## Multi-tenancy

Multi-tenancy is first-class, not a filter convention:

- **Isolation by construction:** one index per tenant, so a query physically
  cannot reach another tenant's vectors. No filter to misconfigure, no leak path.
  It holds under an adversarial leak-fuzz: query a tenant's index with another
  tenant's exact vector and zero rows cross the boundary, every time.
- **Hard quotas:** `max_vectors`, `max_disk_bytes`, set and read at runtime via
  `SKEG.QUOTA.SET` / `SKEG.QUOTA.GET`.
- **Fair eviction:** a noisy tenant can't starve a quiet one out of the cache.
- **Authentication:** `HELLO 3 AUTH user pass` (argon2id), prefix-routed namespaces.

See [`docs/multi-tenancy.md`](docs/multi-tenancy.md).

## Install

### Homebrew (macOS and Linux ARM)

```sh
brew tap skegdb/tap
brew install skeg
```

Installs both binaries (`skeg`, `skeg-resp3`) and a launchd/systemd service.

### From crates.io

```sh
cargo install skeg-server
```

Compiles from source and installs `skeg` and `skeg-resp3` into `$CARGO_HOME/bin`.
Requires a Rust toolchain (MSRV 1.88).

<!-- markdownlint-disable MD033 -->
<details>
<summary>Tarball, source, Docker, and the published crate list</summary>

#### Pre-built tarball

```sh
TARGET=aarch64-apple-darwin   # or aarch64-unknown-linux-gnu
curl -L -o skeg.tar.gz \
  "https://github.com/skegdb/skeg/releases/latest/download/skeg-$(curl -s https://api.github.com/repos/skegdb/skeg/releases/latest | grep tag_name | cut -d'"' -f4)-${TARGET}.tar.gz"
tar -xzf skeg.tar.gz
./skeg --help
```

SHA256 checksums are published alongside each tarball (`.sha256` suffix). Pin a
version from the [releases page](https://github.com/skegdb/skeg/releases).

#### From source

```sh
git clone https://github.com/skegdb/skeg
cd skeg
cargo build --release --bin skeg --bin skeg-resp3
```

Requires Rust 1.88+. Binaries land at `target/release/skeg` and `target/release/skeg-resp3`.

#### Docker

```sh
docker run -d --name skeg \
  -p 7379:7379 \
  -v skeg-data:/var/lib/skeg \
  ghcr.io/skegdb/skeg:latest
```

Bundles both binaries (`skeg` native on 7379, `skeg-resp3` Redis-compat on 6379).
Default entrypoint is `skeg`; for RESP3 override with `--entrypoint
/usr/local/bin/skeg-resp3` and publish 6379. Built for `linux/arm64`. An Ollama
companion setup lives in [`docker-compose.example.yml`](docker-compose.example.yml).

#### Published crates

`skeg-proto`, `skeg-simd`, `skeg-platform`, `skeg-telemetry`, `skeg-resp3`,
`skeg-core`, `skeg-vector`, `skeg-server`, `skeg-tenant`, `skeg-server-tenant`,
`skeg-multi-tenant`. Network adapters: [`skeg-rigging`](https://github.com/skegdb/skeg-rigging),
[`skeg-rigging-net`](https://github.com/skegdb/skeg-rigging-net).

</details>
<!-- markdownlint-enable MD033 -->

Then follow [`docs/getting-started.md`](docs/getting-started.md) to run it and
issue the first commands.

## Status

**Shipping today.** KV and vector ops on both protocols; filtered vector search
with the full filter grammar; first-class multi-tenancy with quotas and fair
eviction; three durability tiers (relaxed / kernel / power-loss); Prometheus
metrics and OTLP tracing; a workspace test suite that's clean under
`cargo clippy --workspace --all-targets`.

**Honest about the edges.** skeg is not the lowest-latency *single-query* engine.
Qdrant is comparable on p99 and raw hnswlib is faster still. A single process
saturates around 780 QPS at 1024-dim (more at lower dimensions); past that you
scale out with processes, not cores. Cold bulk-loading a fresh index is
rebuild-based and trades build time for the lean serving footprint. Release
binaries are aarch64 (Apple Silicon, Linux ARM); the source builds on x86_64 but
AVX2/AVX-512 tuning and native Linux validation are [on the
roadmap](docs/roadmap.md), not done.

## Documentation

Long-form design and benchmark write-ups on the [project blog](https://amanitaproject.com/):
*Constraints as Method*, *Seven More Hypotheses*, *The Substrate*, *What Was Measured*.

Guides in [`docs/`](docs/):

- [`getting-started.md`](docs/getting-started.md): run it, the command reference, the filter grammar.
- [`architecture.md`](docs/architecture.md): on-disk index, tiers (tq1 vs tq2), filtered-search planner.
- [`multi-tenancy.md`](docs/multi-tenancy.md): tenants, key scoping, quotas, fair eviction.
- [`filtered-search.md`](docs/filtered-search.md): payloads, filter grammar, the planner.
- [`observability.md`](docs/observability.md): Prometheus, OTel, tracing.
- [`ecosystem.md`](docs/ecosystem.md): federation (hansa) and ingest pipelines.
- [`roadmap.md`](docs/roadmap.md): what's planned, what's conditional, what's deliberately not.

Reproducible benchmark suite: [`skeg-bench`](https://github.com/skegdb/skeg-bench).
Live dashboard: [`skegdb.github.io/bench`](https://skegdb.github.io/bench/).

## Contributing

Bug reports, design discussions, and pull requests are welcome. Run `cargo fmt`,
`cargo clippy --workspace --all-targets -- -D warnings`, and `cargo test
--workspace` before opening a PR. A pre-push hook at `.githooks/pre-push` runs
the same three locally. Enable with `git config core.hooksPath .githooks`
(bypass a docs-only push with `SKIP_PREPUSH=1`).

## Security

Report security issues by opening an issue with a brief description and a request
to take the conversation private. See [`SECURITY.md`](SECURITY.md).

## License

[Apache-2.0](LICENSE). See [`NOTICE`](NOTICE) for attribution.
