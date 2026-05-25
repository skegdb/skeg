# skeg-fuzz

Cargo-fuzz targets for skeg (M10 hardening, PLAN.md §M10).

Three parser entry points, three fuzz binaries:

| target | entry point | what it exercises |
| --- | --- | --- |
| `fuzz_proto_frame` | `skeg_proto::FrameParser::feed` | binary protocol frame state machine on adversarial byte streams (malformed magic, oversized payload_len, partial headers, payloads spanning calls) |
| `fuzz_vlog_record` | `skeg_core::record::decode_record` | vLog record decoder (CRC mismatches with valid-looking lengths, key/value sizes overflowing the slice, kind byte outside the set, lying padding) |
| `fuzz_index_snapshot` | `skeg_core::snapshot::decode` | index snapshot decoder (bad magic/version, hwm/max_ts mismatches, entries lying about key length, truncated CRC, oversized counts driving runaway allocation) |

The contract being verified: **no panic, no OOB read** on any byte sequence.
Successful parses return `Ok(_)`, malformed inputs return `Err(_)` or `None`.

## Running

cargo-fuzz requires the nightly toolchain (`-Zsanitizer=address` is unstable).
On macOS with homebrew rust shadowing rustup, set the PATH explicitly:

```sh
# Smoke (PLAN matrix: 30s per target)
PATH=$HOME/.rustup/toolchains/nightly-aarch64-apple-darwin/bin:$PATH \
    cargo fuzz run fuzz_proto_frame -- -max_total_time=30

# Or long-running, no time cap
PATH=$HOME/.rustup/toolchains/nightly-aarch64-apple-darwin/bin:$PATH \
    cargo fuzz run fuzz_vlog_record
```

Linux runners do not need the PATH dance: `cargo +nightly fuzz run ...`.

## Status

All three targets verified to build and run on aarch64-apple-darwin
(2026-05-21). 20-second smoke per target: zero crashes across ~5-8M
runs per target. Coverage saturates quickly (proto: 108 cov / 116
features) on the tight state machines.

Corpus, crashes/, and artifacts/ directories are gitignored (cargo-fuzz
manages them per-target).
