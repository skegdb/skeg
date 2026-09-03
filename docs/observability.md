# Observability

skeg exposes operational metrics in Prometheus text format and structured
logs through the `tracing` crate. This document covers what is shipped in
v0.3.6, how to wire it up, and what is coming next.

## TL;DR

```bash
# Start skeg with the Prometheus exporter on port 9090.
# skeg has no authentication, so a non-loopback --addr needs the explicit
# opt-in below (or bind --addr 127.0.0.1:7379 instead).
skeg --mode serve --tier pq:128:256 --data-dir /var/skeg \
     --addr 0.0.0.0:7379 --metrics-port 9090 \
     --allow-unauthenticated-network

# Scrape it.
curl http://127.0.0.1:9090/metrics
```

Released binaries from v0.3.6 onwards include the exporter by default; no
extra Cargo feature is needed.

## Metric schema

The `/metrics` endpoint produces three families.

### `skeg_ops_total{op="..."}` (counter)

Total number of operations served, broken down by op. Stable labels:
`get`, `set`, `del`, `mget`, `mset`, `incr`, `decr`, `vset`, `vdel`,
`vsearch`, `ping`. Counters are monotonic for the lifetime of the process.

### `skeg_op_duration_seconds{op="...",le="..."}` (histogram)

Per-op latency, in seconds. Reported as Prometheus cumulative buckets
plus `_count` and `_sum`. The bucket bounds are fixed at compile time
and documented in [`skeg-telemetry/src/histograms.rs`](../crates/skeg-telemetry/src/histograms.rs).
Useful queries:

```promql
# p99 latency for VSEARCH over the last 5 minutes.
histogram_quantile(0.99,
  sum(rate(skeg_op_duration_seconds_bucket{op="vsearch"}[5m])) by (le))

# Throughput per op.
rate(skeg_ops_total[1m])
```

### Global counters and gauges

Cache and shard health gauges (`skeg_cache_bytes`, `skeg_cache_evictions_total`,
`skeg_n_keys`, etc.) are emitted with stable names. Run `curl /metrics |
grep '^# TYPE'` against a live binary to see the current set; the list
is grep-stable across patch releases.

### The memory budget and the ingress class

Two gauge families report what the process has promised and to whom.

**Where to read them.** The gauges below are composed by the `SKEG.STATS`
command and appear in its reply only; `/metrics` carries the counters but
not these gauges, because the exporter serves the telemetry registry and
these are assembled at the point the command runs. Scrape them with a
`SKEG.STATS` probe until that is unified.

`skeg_memory_budget_state{state="known"|"unlimited"|"unknown"}` is the
governor: whether a ceiling applies, whether none does, and whether one
applies whose headroom could not be read. The third is a refusal, not a
licence - it is the state in which writes are declined - so `unknown` is
the one to alert on. `skeg_memory_headroom_bytes` is what is left when
the state is `known`, `skeg_memory_reserved_bytes` what is promised and
not yet allocated, and `skeg_memory_reserve_bytes` the margin
deliberately held back.

`skeg_ingress_state{state="known"|"default"|"unreadable"}` is the share
of that budget the network may hold, with `skeg_ingress_cap_bytes`,
`skeg_ingress_held_bytes` and
`skeg_ingress_per_connection_max_bytes`. `held` is a SUBSET of
`skeg_memory_reserved_bytes`: the two are one total seen whole and seen
by class, which is how you tell a store that is full of buffered requests
from one that is full of data.

Four counters go with them, and these DO reach `/metrics`:
`skeg_ingress_refused_accept_total` (peers
turned away because the class was full - too many clients),
`skeg_ingress_refused_growth_total` (frames refused because the buffer
they needed was not available - frames too large, or too much other
traffic), `skeg_ingress_stalls_total` (reads paused waiting for room; a
stall that ends in room shows up to a client only as latency), and
`skeg_ingress_budget_unreadable_total` (connections served while the
budget could not be established).

#### The three charges, and why they do not overlap

1. **Ingress**: the CAPACITY of a connection's read buffer, times two for
   the copy the parser makes out of it while the buffer still holds it.
   Charged for as long as the buffer is that big, given back when it
   drains or the connection closes.
2. **Delta**: the rows a committed write puts in the in-memory delta,
   charged per megabyte as it grows. A row reaches it only after it has
   stopped being wire bytes; the buffer it arrived in is drained before
   the command runs.
3. **`MAX_VMSET_BYTES`** (64 MiB): a constant early reject on one
   command's vector bytes, with no reservation at all. It is a ceiling on
   what a single frame may ask for, applied upstream of both budgets, and
   reserving for it would charge the same bytes twice.

#### Sizing

| setting | default | what it does |
| --- | --- | --- |
| `SKEG_MEMORY_LIMIT_BYTES` | cgroup headroom | ceiling on headroom; only ever makes the budget smaller |
| `SKEG_MEMORY_RESERVE_BYTES` | max(64 MiB, limit/10) | margin held back from the budget |
| `SKEG_INGRESS_FRACTION` | 25 | percent of usable headroom the network may hold |
| `SKEG_INGRESS_BUDGET_BYTES` | derived | an explicit class cap, replacing the fraction |
| `SKEG_INGRESS_STALL_MS` | 500 | how long a connection waits for room before its frame is refused |
| `SKEG_MAX_CONNECTIONS` | 1024 | concurrent connections per listener, both protocols |
| `SKEG_MAX_FDS` | 65536 | descriptor limit raised at boot |

Worked example, a 256 MiB container with about 100 MiB resident:
headroom 150 MiB, less a 64 MiB reserve, leaves 86 MiB usable. A quarter
of that is the ingress class, 21.5 MiB; a quarter of the class is one
connection's allowance, 5.375 MiB charged, which is 2.7 MiB of actual
buffer. An idle connection holds 8 KiB, so a thousand of them hold
8 MiB - measured at exactly that against the real binary.

**Declare this to your clients.** In a container that size, a `SKEG.VMSET`
at the protocol's 129 MiB frame ceiling no longer fits one connection's
allowance and is refused by name, where an earlier build accepted it and
was killed by the kernel part-way through. A refusal that says
`ERR ingress budget: this connection may hold at most N bytes and the
frame needs M` is not retryable and means the batch must be split. A
refusal that says `BACKPRESSURE ingress budget: ...` is momentary and
should be retried.

Off Linux there is no cgroup to read, so the class cap is a static
default: 1 GiB, or four maximum frames if that is larger. It is the
larger, by three per cent, so a machine with no memory limit at all still
admits a frame at the protocol ceiling.

## Prometheus scrape config

Drop into your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: skeg
    scrape_interval: 15s
    static_configs:
      - targets: ['skeg-host:9090']
        labels:
          tier: pq128       # whatever you pass to --tier
          environment: prod
```

That is enough to populate the dashboard JSON shipped under
[`assets/grafana/`](../assets/grafana/).

## OpenTelemetry: use the collector's Prometheus receiver

skeg does not link the OpenTelemetry SDK directly on the hot path. Atomic
counters keep the per-op overhead near zero; an OpenTelemetry receiver in
your collector picks up the same `/metrics` endpoint and forwards it as
OTLP. Sample [otel-collector-config.yaml](../assets/grafana/otel-collector-config.yaml):

```yaml
receivers:
  prometheus:
    config:
      scrape_configs:
        - job_name: skeg
          scrape_interval: 15s
          static_configs:
            - targets: ['skeg-host:9090']

exporters:
  otlphttp:
    endpoint: https://your-backend.example.com/v1/metrics
    headers:
      authorization: Bearer ${OTEL_TOKEN}

service:
  pipelines:
    metrics:
      receivers: [prometheus]
      exporters: [otlphttp]
```

The collector handles batching, retry, auth, and TLS. skeg stays in its
lane (zero-overhead counters + Prometheus expose).

## Tracing

`skeg-server` uses the `tracing` crate. The default subscriber writes
structured log lines to stdout; level is controlled by `RUST_LOG`
(`RUST_LOG=info,skeg_server=debug` is a reasonable production setting).

### OTLP span export

When the binary is built with `--features tracing-otlp` (released
binaries from v0.3.6 onwards include it by default) and the env var
`SKEG_TRACE_OTLP_ENDPOINT` points at an OTLP/gRPC collector, spans flow
to the collector in addition to stdout.

```bash
export SKEG_TRACE_OTLP_ENDPOINT=http://collector:4317
export SKEG_TRACE_SAMPLE_RATE=1.0
export SKEG_TRACE_RESOURCE_ATTRS="region=eu-west-1,host=skeg-01"
skeg --mode serve --tier pq:128:256 --data-dir /var/skeg --addr :7379
```

| Env var                        | Meaning                                    | Default |
|--------------------------------|--------------------------------------------|---------|
| `SKEG_TRACE_OTLP_ENDPOINT`     | OTLP/gRPC URL. Unset = no export.          | unset   |
| `SKEG_TRACE_SAMPLE_RATE`       | Head-based sampling [0.0, 1.0].            | `1.0`   |
| `SKEG_TRACE_RESOURCE_ATTRS`    | `k1=v1,k2=v2` resource labels.             | unset   |

Spans currently emitted by `VSEARCH`:

- `vsearch` (parent): tenant, vindex, k, l_search, vector_dim, hits.

Span hierarchy expands in subsequent releases (child spans for walk and
rerank phases).

### Overhead

Microbenched on M1 Pro (`crates/skeg-server/benches/tracing_overhead.rs`):

| Configuration             | ns / span | vs baseline |
|---------------------------|-----------|-------------|
| no subscriber             | 1.64      | 1.00x       |
| subscriber + filter drops | 1.54      | 0.94x       |
| subscriber + emit to sink | 1116      | 680x        |

For VSEARCH (~1500 us per query) the overhead at full-trace
visibility is around 0.07%.

## Opting out

If you build skeg from source and want a slim binary with no HTTP
exporter, drop the default feature:

```bash
cargo build --release -p skeg-server --no-default-features
```

The `--metrics-port` flag stays parseable but logs a warning instead of
binding. No tiny_http link, no thread spawn.

## Performance notes

- Hot path: every counter increment is a single relaxed atomic add. Per
  op overhead measured around 3-5 ns on M1 Pro.
- Scrape cost: serialising the full metric set into Prometheus text takes
  ~150 us for the default schema (about 1 KB of output). Safe to scrape
  at 1 Hz; the cost is in the scrape handler, not the hot path.
- Memory: counters are static globals; total footprint stays under
  64 KB regardless of traffic volume.
