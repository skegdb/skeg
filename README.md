<!-- markdownlint-disable MD033 MD041 -->
<p align="center">
  <img src="assets/skeg-logo.png" alt="skeg" width="480">
</p>
<!-- markdownlint-enable MD033 MD041 -->

# skeg

Vector database and context layer for AI agents. Multi-tenant, RAM-frugal.

> **Pre-release.** The first public version (`v0.1.0`) is not tagged yet. The code in this repository builds and runs from source. There are no published packages and no release binaries.
>
> **Hardware target.** skeg was written and optimised for Apple Silicon (M-series). The benchmarks below are from an M1. Native validation and architecture-specific optimisation for x86_64 (Linux, server hardware) will follow as soon as we have the hardware to test on. Building from source on x86_64 is expected to work, but the binary is not yet tuned for that architecture.

## The point

Most vector databases are designed to live alone on the machine. They expect every gigabyte of RAM the operating system can spare, and they take it.

skeg is built for the opposite case: a machine where the language model already owns most of the memory. The index sits on the SSD, a small bounded working set stays in RAM, and the rest of the system is left alone.

The numbers below are from the same hardware that ran the model.

| 1M vectors at recall@10 >= 0.95 |    RSS |     p99 |   qps |
| ------------------------------- | -----: | ------: | ----: |
| skeg (pq:128:256)               | **419 MiB** |  3.6 ms |   489 |
| skeg (int8)                     | 1252 MiB | 10.2 ms |   299 |
| qdrant (hnsw)                   | 4162 MiB | 38.9 ms |   168 |
| chroma (hnsw)                   | 4417 MiB |  9.7 ms |   196 |

Four gigabytes against four hundred megabytes. The ten times that come back to you are not a benchmark trick. They are the memory you can give to a larger model, a longer context window, a second model loaded alongside the first, a vision encoder, a speech pipeline, the application doing the actual work. Or you can give them to nothing and let the operating system breathe.

This is the operating envelope skeg is built for.

## What it is

skeg is the storage layer for AI agents that run on the same hardware as the model. Vectors and key-value pairs in the same engine. Multi-tenant by design. Sized for machines that already have a language model resident in memory.

The server speaks two protocols. A native binary protocol on port 7379 for clients that want the lowest overhead. RESP3 on port 6379 for compatibility with existing Redis tooling. Both protocols expose the same operations.

## Why it works

The architecture is the answer to a single constraint: the resident set of the index must hold while the model owns the rest of the memory.

The index lives on the SSD. The Vamana graph is walked on disk, with a small in-memory cache for the hot pages. Five tiers of vector quantization (int8, product quantization at 128 by 256, and TurboQuant at 1, 2, and 4 bits per coordinate) let you trade memory against precision. Re-ranking against full-precision vectors held on disk recovers the accuracy lost to quantization. The cache is S3-FIFO, bounded by a byte budget you configure, and gives evicted pages back to the operating system through jemalloc decay timers.

The substrate, the design decisions, and the eleven falsifications that produced this architecture are documented in [the series on the project blog](https://amanitaproject.com/).

## Build from source

Build from source is the only install path right now. Distribution through a Homebrew custom tap, crates.io, and PyPI is in the roadmap.

Requirements: Rust 1.86 or newer.

```sh
git clone https://github.com/skegdb/skeg
cd skeg
cargo build --release --bin skeg --bin skeg-resp3
```

The binaries are at `target/release/skeg` and `target/release/skeg-resp3`.

## Quickstart

Start the server. The native binary protocol listens on 7379; RESP3 listens on 6379.

```sh
./target/release/skeg --data-dir ./data --addr 127.0.0.1:7379 &
./target/release/skeg-resp3 --data-dir ./data --addr 127.0.0.1:6379 &
```

Key-value through any Redis client:

```text
$ redis-cli -3 -p 6379
> SET greeting "hello"
OK
> GET greeting
"hello"
```

Vector operations: create an index, insert, search.

```text
> VINDEX.CREATE docs DIM 1024 METRIC cosine
OK
> VSET docs doc1 [0.12, 0.34, ...]
OK
> VSEARCH docs [0.13, 0.31, ...] LIMIT 10
1) doc1  0.987
2) doc7  0.954
...
```

The vector payload in `VSET` and `VSEARCH` is a binary buffer in the native protocol and a base64 string in RESP3. Protocol documentation will be published in this repository shortly.

## Status

**Working today.**

- KV operations on both protocols: `GET`, `SET`, `DEL`, `MGET`, `MSET`, `INCR`, `DECR`, `EXISTS`.
- Vector operations on both protocols: `VINDEX.CREATE`, `VINDEX.DROP`, `VINDEX.LIST`, `VSET`, `VDEL`, `VSEARCH`.
- Three durability tiers: relaxed (`sync_data`), kernel (`fsync`), and power-loss (`F_FULLFSYNC` on macOS).
- Test suite covers the workspace with no clippy warnings under `cargo clippy --workspace --all-targets`.

**Not in v0.1.**

- Native Linux validation. Linux is tested only through Docker.
- The VSEARCH worker pool is opt-in via `--workers N`; the inline default is what produced the numbers in the table above.
- Multi-tenant operations. The implementation is in the repository and tested, but is not released in v0.1.
- No GPU acceleration, no horizontal scaling across nodes, no hosted service.

## Roadmap

The components listed below are functional in development and validated against the same test suite as the engine. They are not in v0.1 because they need the stabilisation and release work that accompanies a first public version. They will be released in the weeks and months following v0.1.

- Multi-tenant operations: prefix routing, per-tenant isolation, and the `HELLO 3 AUTH` authentication path with argon2id are implemented and tested. What remains is operational hardening for public release.
- Distribution through a Homebrew custom tap (`brew tap skegdb/tap && brew install skeg`), crates.io for the Rust client, and PyPI for the Python SDK.
- Companion repositories: Rust client, Python SDK, terminal dashboard, LlamaIndex and Ollama integrations, Gleam BEAM client.
- The benchmark harness repository.
- Native validation and architecture-specific tuning for x86_64 (Linux server hardware). The current build runs through Docker; the SIMD path is NEON-only, the AVX2/AVX-512 equivalents need a machine to be written and benchmarked on.
- Build artifacts for Linux aarch64 (cloud ARM instances).

## Documentation

The project blog at [amanitaproject.com](https://amanitaproject.com/) carries the long-form design and benchmark documentation:

- *Constraints as Method.* The operating envelope and the first five falsifications.
- *Seven More Hypotheses, One That Survived.* The path to TurboQuant.
- *The Substrate.* The vLog, the group commit, the Vamana index, and the memory budget.
- *What Was Measured: The Numbers.* The full benchmark record across engines, scales, and tiers.

Protocol and operational documentation will be added to this repository in the days following the release.

## Contributing

Bug reports, design discussions, and pull requests are welcome. Run `cargo fmt`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo test --workspace` before opening a pull request. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Security

Security issues should be reported by opening an issue with a brief description and a request to move the conversation private. A dedicated mailbox will be activated shortly. See [`SECURITY.md`](SECURITY.md).

## License

[Apache-2.0](LICENSE). See [`NOTICE`](NOTICE) for attribution.
