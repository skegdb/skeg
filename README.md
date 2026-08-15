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

<p align="center">
  <a href="https://www.producthunt.com/products/github-449?embed=true&amp;utm_source=badge-featured&amp;utm_medium=badge&amp;utm_campaign=badge-skeg" target="_blank" rel="noopener noreferrer"><img alt="Skeg - The memory-efficient vector DB with high recall. | Product Hunt" width="250" height="54" src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1202792&amp;theme=neutral&amp;t=1784904821678"></a>
</p>
<!-- markdownlint-enable MD033 MD041 -->

---

skeg stores the full vectors on SSD and keeps only a small quantized working set
in RAM. That trade buys recall 1.0 on a memory footprint the RAM-resident
engines cannot reach, which is what matters when memory is the contested
resource: thousands of tenants on one box, or a vector store sharing a machine
with the model it serves.

Key-value and vectors live in the same engine, behind a Redis-compatible wire
protocol.

## Quickstart

```sh
docker run -d --name skeg -p 6379:6379 -v skeg-data:/var/lib/skeg \
  --entrypoint /usr/local/bin/skeg-resp3 ghcr.io/skegdb/skeg:latest
```

Any Redis client talks to it:

```text
$ redis-cli -3 -p 6379
> SET greeting "hello"
OK
> SKEG.VINDEX.CREATE docs 1024 tq2 disk
OK
> SKEG.VSET docs 1 <1024-float vector as bytes>
OK
> SKEG.VSEARCH docs 10 100 <query vector bytes>
1) "1"
2) (double) 0.987
```

Vector operations sit under `SKEG.*` so they stay clear of the Redis command
surface. The command above overrides the image entrypoint because RESP3 is the
protocol to build against; see [Protocols](#protocols) for the other one and why
it exists. Full walkthrough, command reference and filter grammar:
[`docs/getting-started.md`](docs/getting-started.md).

## Benchmarks

Reproducible from [`skeg-bench`](https://github.com/skegdb/skeg-bench) (public
harness, real embeddings, brute-force ground truth). Measured single-machine on
Apple Silicon; the RAM ratios are hardware-independent.

Single-tenant, 100K x 1024-dim, recall against exact brute force. Every engine
at a reasonable default, with LanceDB tuned to recall 1.0 for a fair fight:

| engine | serve RAM | recall@10 | p50 latency |
| --- | ---: | ---: | ---: |
| **skeg** (tq2) | **47 MB** | **1.000** | **2.5 ms** |
| Milvus Lite | 108 MB | 0.934 | 2.7 ms |
| LanceDB (IVF-PQ) | 198 MB | 0.998 | 59 ms |
| hnswlib (raw HNSW) | 426 MB | 0.985 | 2.0 ms |
| Chroma (HNSW) | 682 MB | 0.985 | 3.9 ms |
| Qdrant (HNSW, f32) | 885 MB | 0.997 | 2.6 ms |

Co-resident with a model: a 3B LLM answering RAG over 1M vectors, both on one M1
Pro (16 GiB). The index stays on SSD and the resident set stays flat.

| Co-resident, 1M vectors | backend RSS p50 | backend RSS max |
| --- | ---: | ---: |
| **skeg** (pq128) | **54 MiB** | **67 MiB** |
| Qdrant (HNSW) | 254 MiB | 2,387 MiB |

<!-- markdownlint-disable MD033 MD041 -->
<p align="center">
  <img src="assets/coresidence-rss.png" alt="Backend RSS while a 3B LLM serves RAG, swept from 10K to 1M vectors on an M1 Pro 16 GiB. skeg stays under 80 MiB; Qdrant climbs into multi-GiB territory." width="760">
</p>
<!-- markdownlint-enable MD033 MD041 -->

The full matrix, plus the multi-tenant and container-OOM runs, is on the
[dashboard](https://skegdb.github.io/bench/).

### What it does not win

skeg is not the lowest-latency single-query engine: Qdrant is comparable on p99
and raw hnswlib is faster. One process saturates near 780 QPS at 1024-dim, past
which you scale out with processes. Cold bulk-loads rebuild the index.

## Multi-tenancy

Tenancy is a property of the storage layout rather than a filter convention.
Each tenant gets its own index, so a query has no physical path to another
tenant's vectors, and there is no filter to misconfigure. An adversarial leak-fuzz
holds it to that: query one tenant's index with another tenant's exact vector
and zero rows cross the boundary, every time.

On top of that isolation:

- Hard quotas: `max_vectors` and `max_disk_bytes`, set and read at runtime
  through `SKEG.QUOTA.SET` / `SKEG.QUOTA.GET`.
- Fair eviction, so a noisy tenant cannot starve a quiet one out of the cache.
- Authentication via `HELLO 3 AUTH user pass` (argon2id), with prefix-routed
  namespaces.

Details in [`docs/multi-tenancy.md`](docs/multi-tenancy.md).

## Install

### Docker

```sh
docker run -d --name skeg -p 7379:7379 -v skeg-data:/var/lib/skeg \
  ghcr.io/skegdb/skeg:latest
```

The image carries both binaries and publishes for `linux/amd64` and
`linux/arm64`. The default entrypoint is `skeg` on 7379; for RESP3 override with
`--entrypoint /usr/local/bin/skeg-resp3` and publish 6379. An Ollama companion
setup lives in [`docker-compose.example.yml`](docker-compose.example.yml).

### Homebrew (macOS and Linux ARM)

```sh
brew tap skegdb/tap
brew install skeg
```

Installs both binaries and a launchd/systemd service.

### Pre-built tarball

```sh
TARGET=aarch64-apple-darwin   # see Platforms below for the full list
TAG=$(curl -s https://api.github.com/repos/skegdb/skeg/releases/latest | grep tag_name | cut -d'"' -f4)
curl -L -o skeg.tar.gz \
  "https://github.com/skegdb/skeg/releases/latest/download/skeg-${TAG}-${TARGET}.tar.gz"
tar -xzf skeg.tar.gz && ./skeg --help
```

Each tarball ships a `.sha256` next to it. Pin a version from the
[releases page](https://github.com/skegdb/skeg/releases).

### From crates.io

```sh
cargo install skeg-server
```

Builds from source into `$CARGO_HOME/bin`. Needs a Rust toolchain (MSRV 1.88).

### From git

```sh
git clone https://github.com/skegdb/skeg
cd skeg
cargo build --release --bin skeg --bin skeg-resp3
```

Binaries land in `target/release/`.

## Platforms

| your machine | tarball to download | container |
| --- | --- | --- |
| Mac, Apple Silicon | `aarch64-apple-darwin` | not published |
| Linux, ARM | `aarch64-unknown-linux-gnu` | `:latest` |
| Linux, x86_64 | `x86_64-unknown-linux-gnu` | `:latest` |

`:latest` carries both Linux architectures and resolves to the right one on
`docker pull`. There is no Intel Mac or Windows build.

### The second x86_64 build

x86_64 also has an `-avx512` tarball and a `:<version>-avx512` image. Take the
plain one unless you know your CPU has AVX-512, and even then the difference is
worth measuring rather than assuming.

Both builds run on any x86_64 CPU. skeg picks its kernels at runtime after a CPU
check, so the suffix describes what is compiled in, not what the machine must
have; on a CPU without AVX-512 the extra kernels are simply never chosen. They
are a separate download only because compiling them needs Rust 1.89, above the
1.88 the rest of the project builds with.

To build them yourself:

```sh
cargo build --release --bin skeg --bin skeg-resp3 --features skeg-server/avx512
```

Which kernel actually runs on which instruction set, and why some are built but
deliberately not selected, is asserted in a test rather than described in prose:
`cargo test -p skeg-simd --test coverage`.

## Protocols

Use **RESP3** for application integrations. It is the supported public API and
names the vector tiers directly: `f32`, `int8`, `tq1`, `tq2`, `tq4`, `binary`.

The native transport on 7379 exists for specialised clients. It is versioned,
and the version decides which tiers it can name:

| | v1 | v2 |
| --- | --- | --- |
| kinds | `0=f32` `1=int8` `2=binary` | the same, plus `3=tq1` `4=tq2` `5=tq4` |
| kind `3` | rejected: historical clients used it for PQ | `tq1` |

A v2 client opens with `NativeHello` (op `0x84`) and reads the tier capability
mask it gets back. v1 byte meanings are unchanged, so an existing client keeps
working.

## Documentation

Guides in [`docs/`](docs/):

- [`getting-started.md`](docs/getting-started.md): run it, command reference, filter grammar.
- [`architecture.md`](docs/architecture.md): on-disk index, tiers, filtered-search planner.
- [`multi-tenancy.md`](docs/multi-tenancy.md): tenants, key scoping, quotas, fair eviction.
- [`filtered-search.md`](docs/filtered-search.md): payloads, filter grammar, the planner.
- [`observability.md`](docs/observability.md): Prometheus, OTel, tracing.
- [`ecosystem.md`](docs/ecosystem.md): federation (hansa) and ingest pipelines.
- [`roadmap.md`](docs/roadmap.md): planned, conditional, and deliberately not.

Long-form design and benchmark write-ups are on the
[project blog](https://amanitaproject.com/): *Constraints as Method*, *Seven More
Hypotheses*, *The Substrate*, *What Was Measured*.

Published crates: `skeg-proto`, `skeg-simd`, `skeg-platform`, `skeg-telemetry`,
`skeg-resp3`, `skeg-core`, `skeg-vector`, `skeg-server`, `skeg-tenant`,
`skeg-server-tenant`, `skeg-multi-tenant`. Network adapters live in
[`skeg-rigging`](https://github.com/skegdb/skeg-rigging) and
[`skeg-rigging-net`](https://github.com/skegdb/skeg-rigging-net).

## Contributing

Bug reports, design discussions, and pull requests are welcome. Before opening a
PR run `cargo fmt`, `cargo clippy --workspace --all-targets -- -D warnings`, and
`cargo test --workspace`. The pre-push hook at `.githooks/pre-push` runs the same
three; enable it with `git config core.hooksPath .githooks` (a docs-only push can
skip it with `SKIP_PREPUSH=1`).

## Security

Report security issues by opening an issue with a brief description and a request
to take the conversation private. See [`SECURITY.md`](SECURITY.md).

## License

[Apache-2.0](LICENSE). See [`NOTICE`](NOTICE) for attribution.
