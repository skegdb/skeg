# Observability

skeg exposes operational metrics in Prometheus text format and structured
logs through the `tracing` crate. This document covers what is shipped in
v0.3.6, how to wire it up, and what is coming next.

## TL;DR

```bash
# Start skeg with the Prometheus exporter on port 9090.
skeg --mode serve --tier pq:128:256 --data-dir /var/skeg \
     --addr 0.0.0.0:7379 --metrics-port 9090

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
and documented in [`skeg-telemetry/src/histograms.rs`](crates/skeg-telemetry/src/histograms.rs).
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
[`assets/grafana/`](assets/grafana/).

## OpenTelemetry: use the collector's Prometheus receiver

skeg does not link the OpenTelemetry SDK directly on the hot path. Atomic
counters keep the per-op overhead near zero; an OpenTelemetry receiver in
your collector picks up the same `/metrics` endpoint and forwards it as
OTLP. Sample [otel-collector-config.yaml](assets/grafana/otel-collector-config.yaml):

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
visibility is around 0.07% — well inside the 5% gate defined in
[`observability/PLAN.md`](https://github.com/skegdb/skeg-internal/blob/main/observability/PLAN.md).

## Roadmap

Coming next:

- **Child spans for VSEARCH internals** (walk traversal, rerank disk
  reads + cosine) once skeg-vector grows a thin tracing entry point.
- **Grafana dashboard refinements** with span exemplars linking the
  histogram buckets to individual traces.
- **Per-tenant labels** when the multi-tenant server (`skeg-server-tenant`)
  is wired against the metrics path.

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
