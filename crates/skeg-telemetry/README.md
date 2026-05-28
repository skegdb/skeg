# skeg-telemetry

Zero-overhead atomic counters and histograms for skeg.

## Design

- Every public API entry is `#[inline(always)]`.
- All counters and histograms are static `AtomicU64` arrays. No `HashMap`, no
  `Mutex`, no allocation on the hot path.
- Per-op counters are sharded across `MAX_SHARDS = 32` to avoid false sharing
  on M-series P-cores.
- Histograms are fixed exponential buckets (1 µs → 1 s, plus `+Inf`).
- When the crate is compiled with no features, every public function is an
  empty body and the caller's arguments are sunk into `let _ = …` — verified
  with `cargo asm` to compile to a tail call elimination.

## Features

| feature | default | what it adds |
| --- | --- | --- |
| `stats` | yes | static counters + `stats::dump_text()` (used by RESP3 `STATS`) |
| `http`  | no  | tiny HTTP exporter on a dedicated thread (`/metrics`) |

When neither feature is on, this crate is a no-op.

## Hot-path cost budget

`record_op` is gated by `benches/overhead.rs`:

| call | budget | typical (Apple M1) |
| --- | --- | --- |
| `record_op(Op, shard, dur)` | < 50 ns | ~ 3–5 ns |
| `tick_counter(Counter)`     | < 20 ns | ~ 1–2 ns |
| `set_gauge(Gauge, u64)`     | < 20 ns | ~ 1–2 ns |

CI fails if any benchmark drifts past its budget.

## Usage

```rust
use skeg_telemetry::{record_op, Op};

let t0 = std::time::Instant::now();
do_vsearch();
record_op(Op::VSearch, shard_id, t0.elapsed());
```

For the HTTP exporter:

```rust
skeg_telemetry::http::spawn("127.0.0.1:9090".parse().unwrap())?;
```

For the RESP3 `STATS` command body:

```rust
let body = skeg_telemetry::stats::dump_text();
```
