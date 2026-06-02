<!-- markdownlint-disable MD033 MD041 -->
<p align="center">
  <img src="assets/skeg-logo.png" alt="skeg" width="480">
</p>
<!-- markdownlint-enable MD033 MD041 -->

# skeg

Vector database and context layer for AI agents. Multi-tenant, RAM-frugal.

> **Hardware target.** skeg was written and optimised for Apple Silicon (M-series). The benchmarks below are from an M1; the full live dashboard is at [`skegdb.github.io/bench`](https://skegdb.github.io/bench/). Native validation and architecture-specific optimisation for x86_64 (Linux, server hardware) will follow as soon as we have the hardware to test on. The v0.1.0 release ships aarch64 binaries only (Apple Silicon, Linux ARM); the source builds on x86_64 but is not yet tuned for it.

## The point

Most vector databases are designed to live alone on the machine. They expect every gigabyte of RAM the operating system can spare, and they take it.

skeg is built for the opposite case: a machine where the language model already owns most of the memory. The index sits on the SSD, a small bounded working set stays in RAM, and the rest of the system is left alone.

The numbers below are from the same hardware that ran the model.

| 1M vectors at recall@10 >= 0.95 |         RSS |     p99 |   qps |
| ------------------------------- | ----------: | ------: | ----: |
| skeg (pq:128:256)               | **419 MiB** |  3.6 ms |   489 |
| skeg (int8)                     |    1252 MiB | 10.2 ms |   299 |
| qdrant (hnsw)                   |    4162 MiB | 38.9 ms |   168 |
| chroma (hnsw)                   |    4417 MiB |  9.7 ms |   196 |

Four gigabytes against four hundred megabytes. The ten times that come back to you are not a benchmark trick. They are the memory you can give to a larger model, a longer context window, a second model loaded alongside the first, a vision encoder, a speech pipeline, the application doing the actual work. Or you can give them to nothing and let the operating system breathe.

This is the operating envelope skeg is built for.

## What it is

skeg is the storage layer for AI agents that run on the same hardware as the model. Vectors and key-value pairs in the same engine. Multi-tenant by design. Sized for machines that already have a language model resident in memory.

The server speaks two protocols. A native binary protocol on port 7379 for clients that want the lowest overhead. RESP3 on port 6379 for compatibility with existing Redis tooling. Both protocols expose the same operations.

## Why it works

The architecture is the answer to a single constraint: the resident set of the index must hold while the model owns the rest of the memory.

The index lives on the SSD. The Vamana graph is walked on disk, with a small in-memory cache for the hot pages. Five tiers of vector quantization (int8, product quantization at 128 by 256, and TurboQuant at 1, 2, and 4 bits per coordinate) let you trade memory against precision. Re-ranking against full-precision vectors held on disk recovers the accuracy lost to quantization. The cache is S3-FIFO, bounded by a byte budget you configure, and gives evicted pages back to the operating system through jemalloc decay timers.

The substrate, the design decisions, and the eleven falsifications that produced this architecture are documented in [the series on the project blog](https://amanitaproject.com/).

## Install

Three install paths. The Homebrew tap and the pre-built tarballs ship aarch64 binaries (Apple Silicon, Linux ARM); `cargo install` and the source build work on any host with a Rust toolchain.

### Homebrew (macOS and Linux ARM)

```sh
brew tap skegdb/tap
brew install skeg
```

The formula installs both binaries (`skeg`, `skeg-resp3`) and a launchd/systemd service definition.

### Pre-built tarball

```sh
TARGET=aarch64-apple-darwin   # or aarch64-unknown-linux-gnu
curl -L -o skeg.tar.gz \
  "https://github.com/skegdb/skeg/releases/latest/download/skeg-$(curl -s https://api.github.com/repos/skegdb/skeg/releases/latest | grep tag_name | cut -d'"' -f4)-${TARGET}.tar.gz"
tar -xzf skeg.tar.gz
./skeg --help
```

Or pin a specific version from the
[releases page](https://github.com/skegdb/skeg/releases).

SHA256 checksums are published alongside each tarball at the same URL with a `.sha256` suffix.

### From crates.io

```sh
cargo install skeg-server
```

This compiles from source on the host machine and installs `skeg` and `skeg-resp3` into `$CARGO_HOME/bin`. Requires a Rust toolchain (MSRV 1.88).

The published library crates are `skeg-proto`, `skeg-simd`, `skeg-platform`, `skeg-resp3`, `skeg-core`, `skeg-vector`, and `skeg-server`.

### From source

```sh
git clone https://github.com/skegdb/skeg
cd skeg
cargo build --release --bin skeg --bin skeg-resp3
```

Requirements: Rust 1.88 or newer. The binaries are at `target/release/skeg` and `target/release/skeg-resp3`.

### Docker

```sh
docker run -d --name skeg \
  -p 7379:7379 \
  -v skeg-data:/var/lib/skeg \
  ghcr.io/skegdb/skeg:latest
```

The image bundles both `skeg` (native protocol, port 7379) and `skeg-resp3` (Redis-compat, port 6379). The default entrypoint is `skeg`; for the RESP3 surface override with `--entrypoint /usr/local/bin/skeg-resp3` and publish 6379 instead. Built for `linux/arm64`.

For an Ollama companion setup with `docker compose`, see [`docker-compose.example.yml`](docker-compose.example.yml).

## Quickstart

Start the server. The native binary protocol listens on 7379; RESP3 listens on 6379.

```sh
skeg --data-dir ./data --addr 127.0.0.1:7379 &
skeg-resp3 --data-dir ./data --addr 127.0.0.1:6379 &
```

If you built from source, the binaries are at `./target/release/skeg` and `./target/release/skeg-resp3`.

Key-value through any Redis client:

```text
$ redis-cli -3 -p 6379
> SET greeting "hello"
OK
> GET greeting
"hello"
```

Vector operations are namespaced under `SKEG.*` to stay out of the
Redis command surface. Create an index, insert, search:

```text
> SKEG.VINDEX.CREATE docs 1024 int8 flat
OK
> SKEG.VINDEX.LIST
name=docs dim=1024 kind=int8 backend=flat n_vectors=0
> SKEG.VSET docs doc1 <1024-float vector as bytes>
OK
> SKEG.VSEARCH docs 10 100 <query vector bytes>
1) doc1  0.987
2) doc7  0.954
...
```

Command form: `SKEG.VINDEX.CREATE <name> <dim> <kind> <backend>` where
`kind` is `int8 | f32 | pq | tq1 | tq2 | tq4` and `backend` is
`flat | disk`. The vector payload in `SKEG.VSET` and `SKEG.VSEARCH`
is a raw byte buffer (native protocol) or a bulk string (RESP3).
Protocol documentation will be published in this repository shortly.

## Status

**Working today.**

- KV operations on both protocols: `GET`, `SET`, `DEL`, `MGET`, `MSET`, `INCR`, `DECR`, `EXISTS`.
- Vector operations on both protocols: `SKEG.VINDEX.CREATE`, `SKEG.VINDEX.DROP`, `SKEG.VINDEX.LIST`, `SKEG.VSET`, `SKEG.VDEL`, `SKEG.VSEARCH`.
- Three durability tiers: relaxed (`sync_data`), kernel (`fsync`), and power-loss (`F_FULLFSYNC` on macOS).
- Test suite covers the workspace with no clippy warnings under `cargo clippy --workspace --all-targets`.

**Not in v0.1.**

- Native Linux validation. Linux is tested only through Docker.
- The VSEARCH worker pool is opt-in via `--workers N`; the inline default is what produced the numbers in the table above.
- Multi-tenant operations. Implemented and tested as the `skeg-tenant` and `skeg-server-tenant` crates (Apache-2.0, same as the engine); released in v0.3.
- No GPU acceleration, no horizontal scaling across nodes, no hosted service.

## Roadmap

The components listed below are functional in development and validated against the same test suite as the engine. They are not in v0.1 because they need the stabilisation and release work that accompanies a first public version. They will be released in the weeks and months following v0.1.

- Companion repositories: Rust client, Python SDK, terminal dashboard, LlamaIndex and Ollama integrations, Gleam BEAM client. PyPI distribution for the Python SDK will follow.
- Native validation and architecture-specific tuning for x86_64 (Linux server hardware). The v0.1.0 binaries are aarch64 only; the SIMD path is NEON-only, the AVX2/AVX-512 equivalents need a machine to be written and benchmarked on.

## Documentation

The project blog at [amanitaproject.com](https://amanitaproject.com/) carries the long-form design and benchmark documentation:

- *Constraints as Method.* The operating envelope and the first five falsifications.
- *Seven More Hypotheses, One That Survived.* The path to TurboQuant.
- *The Substrate.* The vLog, the group commit, the Vamana index, and the memory budget.
- *What Was Measured: The Numbers.* The full benchmark record across engines, scales, and tiers.

Live benchmark dashboard with the latest measurements (engine x scale x tier matrix, p50/p99 latency, recall, RSS): [`skegdb.github.io/bench`](https://skegdb.github.io/bench/).

Protocol and operational documentation will be added to this repository in the days following the release.

## Contributing

Bug reports, design discussions, and pull requests are welcome. Run `cargo fmt`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo test --workspace` before opening a pull request.

A pre-push hook is shipped at `.githooks/pre-push` that runs the same three commands locally so `git push` blocks if any of them would fail. Enable it once per clone with:

```
git config core.hooksPath .githooks
```

Bypass (e.g. for a docs-only commit) with `SKIP_PREPUSH=1 git push`.

## Security

Security issues should be reported by opening an issue with a brief description and a request to move the conversation private. A dedicated mailbox will be activated shortly. See [`SECURITY.md`](SECURITY.md).

## License

[Apache-2.0](LICENSE). See [`NOTICE`](NOTICE) for attribution.
