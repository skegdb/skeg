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

**Where to read them.** Both places, and the same values.  The governor
and the ingress class publish themselves to the telemetry registry, which
is what `/metrics` serves and what `SKEG.STATS` appends to its reply, so
a scrape and a probe carry the same series. (Until 0.7.4 these were
assembled by hand inside the `SKEG.STATS` handler and `/metrics` did not
carry them at all.)

`skeg_memory_budget_state{state="known"|"unlimited"|"unknown"}` is the
governor: whether a ceiling applies, whether none does, and whether one
applies whose headroom could not be read. The third is a refusal, not a
licence - it is the state in which writes are declined - so `unknown` is
the one to alert on.

These are **state sets**: all three series are present on every scrape,
exactly one of them at 1. Alert on `skeg_memory_budget_state{state="unknown"} == 1`,
not on `absent()`. Earlier builds emitted only the series that was true,
which left `absent()` as the only way to write that alert and left the
previous state showing on a dashboard until its series went stale.

`skeg_memory_headroom_bytes` is what is left when the state is `known`.
It is **absent** in the other two states, deliberately: publishing 0 for
a headroom nobody could read says "no room left", which is a different
fact and the one you would page on. `skeg_memory_reserved_bytes` is what
is promised and not yet allocated, and `skeg_memory_reserve_bytes` the
margin deliberately held back.

`skeg_ingress_state{state="known"|"default"|"unreadable"}` is the share
of that budget the network may hold - a state set too, same rule - with `skeg_ingress_cap_bytes`,
`skeg_ingress_held_bytes` and
`skeg_ingress_per_connection_max_bytes`. `held` is a SUBSET of
`skeg_memory_reserved_bytes`: the two are one total seen whole and seen
by class, which is how you tell a store that is full of buffered requests
from one that is full of data.

Six counters go with them:
`skeg_ingress_refused_accept_total` (peers
turned away because the class was full - too many clients),
`skeg_ingress_refused_growth_total` (frames refused because the buffer
they needed was not available - frames too large, or too much other
traffic), `skeg_ingress_stalls_total` (reads paused waiting for room; a
stall that ends in room shows up to a client only as latency),
`skeg_ingress_budget_unreadable_total` (connections served while the
budget could not be established),
`skeg_ingress_reply_over_budget_total`, and
`skeg_quota_refused_total` (writes declined because the tenant already
held every vector its limit allows - a tenant at its ceiling used to look
from outside exactly like a tenant that had stopped writing).

Two more belong to the KV read path:
`skeg_kv_read_bytes_total` (value bytes actually materialised to answer a
`GET`/`MGET`, on either wire, counted at the shard worker beside the fetch
itself) and `skeg_kv_read_refused_total` (reads refused because the summed
value lengths did not fit the connection's allowance, or because a value
grew past the size reserved for it between the measurement and the read).
Read the pair together: a climb in refusals with a flat byte count is the
preflight doing its job, and a climb in BOTH means clients are asking for
more than the class can hand back and getting some of it. Note the second
is about a request refused an ANSWER, where
`skeg_ingress_refused_growth_total` is about a connection refused a
BUFFER.

**Read that last one as "a reply could not be charged", not "a reply was
large".** A reply is written whether or not the class can cover its
buffer, because it answers work that has already committed; the counter
records that the charge failed. Under a full class it therefore ticks for
ANY reply, including a seven-byte `+PONG` - measured at ten ticks for ten
`+PONG`s with the class at 1,040,384 of 1,048,576 bytes. It is a signal
about the class being full, read alongside
`skeg_ingress_refused_growth_total`, and not a measure of reply volume.

#### What the budget covers, and what it does not

It covers the socket buffers a connection holds, in and out, for as long
as it holds them, and - since 0.8.0 - the reply a `GET`/`MGET` is about to
build, reserved BEFORE the values are read rather than measured after.
Those two commands are the only ones whose answer is sized by the store
rather than by the request, and the size is knowable without paying for
it: a key's padded on-disk record size is in the in-RAM index. The sizes
are summed with checked arithmetic, reserved, and only then fetched, with
each fetch bounded by the size reserved for it so a concurrent overwrite
cannot make the measurement stale. A read that does not fit is refused
with nothing fetched. The reservation uses the PADDED record size, so it
over-states the reply by the record header, the key and up to 127 bytes
per key; under a tight class a large `MGET` that would just have fitted
can be refused.

It does NOT cover the peak a single request reaches
while it is being served: the decoded frame tree, the parsed vectors, and
the fan-out of one command into concurrent per-item work. That memory
belongs to a request rather than to a connection, it is gone when the
request is answered, and charging it would mean charging the same bytes
twice.

What bounds it instead is the request itself. `MAX_VMSET_ITEMS` (4096)
and `MAX_VMSET_BYTES` (64 MiB) cap what one command may ask for, and
`VMSET_INFLIGHT` (64) caps how much of it runs at once - that last one
because an unbounded fan-out made a maximum batch 4096 concurrent item
writes per connection, which is the same multiplication the ingress
budget exists to close, arriving by a different door. Measured at 24
callers with one maximum batch each (2026-09-03, macOS arm64, release):
+208 MiB resident unbounded against +25 MiB with the window.

#### The three charges, and why they do not overlap

1. **Ingress and egress**: the CAPACITY of a connection's read buffer and
   of its reply buffer, times two. That factor of two carries two
   passengers, not one - the copies the parser makes out of the buffer
   while the buffer still holds them, AND the over-allocation of
   `BytesMut::reserve`, which doubles and can leave capacity above the
   figure just charged. Measured at eight full `SKEG.VMSET`s of 623 KiB
   each (2026-09-03, macOS arm64): class peak 1.5 MiB against RSS
   +2.0 MiB, so the resident cost was 1.35x the charge. The factor holds,
   with no margin to spare if either passenger grows.
   Charged for as long as the buffers are that big, given back when they
   drain or the connection closes - the reply buffer as soon as the reply
   is on the wire.
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
| `SKEG_MAX_CONNECTIONS` | 1024 | concurrent connections per listener, both protocols. **In `unreadable` mode this is the only ceiling** - see below |
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

**`unreadable` has no class cap.** When a limit applies and its headroom
cannot be read, the governor has no answer to give, so each connection is
granted its 8 KiB floor WITHOUT a reservation and nothing may grow past
it. What bounds the total is then `SKEG_MAX_CONNECTIONS` alone:
`max_connections x 8 KiB`, which is 8 MiB at the default and rises with
the flag, counted by nothing. Treat that product as the constraint on
raising the flag, not the 8 MiB the default happens to give. Set
`SKEG_MEMORY_LIMIT_BYTES` or `SKEG_INGRESS_BUDGET_BYTES` to get a real
cap back; `skeg_ingress_budget_unreadable_total` counts every connection
served in this state.

## What a refusal tells a client

Every refusal taken BEFORE a request runs answers one question: is
sending the same request again worth doing? One classification answers it
(`crates/skeg-server/src/admission.rs`), and both wires derive their
spelling from it, so they cannot say different things.

| condition | retry? | RESP3 | native `ErrCode` |
| --- | --- | --- | --- |
| ingress class full (at accept, or mid-frame) | yes | `-BACKPRESSURE ingress budget: ...` | `4` Backpressure |
| ingress budget unreadable | yes | `-BACKPRESSURE ...` | `4` Backpressure |
| memory governor out of headroom | yes | `-BACKPRESSURE out of budget at the write: ...` | `4` Backpressure |
| VSEARCH pool saturated | yes | `-BACKPRESSURE vsearch queue is full` | `4` Backpressure |
| tenant backend refused, `RATELIMITED ...` | yes | the backend's own line, verbatim | `4` Backpressure |
| tenant backend refused, any other message | no | the backend's own line, verbatim | `3` Internal |
| frame over the connection allowance | no | `-ERR ingress budget: this connection may hold at most N ...` | `2` InvalidRequest |
| tenant vector quota exceeded | no | `-ERR tenant vector quota exceeded: ...` | `2` InvalidRequest |
| request over a fixed ceiling (`SKEG.VMSET` items or bytes, native frame payload) | no | `-ERR ...: at most N, got M ...` | `2` InvalidRequest |
| `GET`/`MGET` reply over the connection allowance | no | `-ERR ingress budget: this connection may hold at most N ...` | `2` InvalidRequest |
| a value grew past the size reserved for it | no | `-ERR stored value bytes (the value grew after its size was reserved): at most N, got M` | `2` InvalidRequest |
| headroom could not be read at all | no | `-ERR ...` | `3` Internal |
| vector of the wrong dimension | no | `-ERR vindex '...' dim N but vector has M` | `2` InvalidRequest |

**RESP3: two words mean retry, not one.** `BACKPRESSURE` for everything the
server decides, and `RATELIMITED` for a tenant backend's rate limit, which is
passed through as the backend wrote it. A client's `is_retryable` must
therefore be a TABLE of code words, not `starts_with("BACKPRESSURE")` - and a
backend can introduce a fourth word this server has never seen, which it
classifies as permanent and counts in
`skeg_backend_refusal_unclassified_total`. On the native wire there is no such
ambiguity: one byte, `0x04`.

An unclassified backend refusal is `3` Internal there, not `2`: what failed
is the server's reading of the backend's answer, and the caller's request may
have been perfectly fine.

**The quota row is RESP3-only in practice.** The native listener has no
tenant backend - `Server::run` drops it and every request on that wire is
tenant `0` - so `handler.rs` passes `limit: None` and the vector quota is
never active there. The row above says what the byte WOULD be, and a test
arms the limit to prove it, but no production native deployment reaches it.
The same applies to a tenant backend's refusal, which cannot arrive on that
wire at all.

**Native:** the code is the first byte of the `Err` payload.
`0x04 Backpressure` was added in 0.7.4 and is the only retryable code.
It is emitted without a version gate: the refusal that matters most is
taken at accept, before any request exists, so there is no negotiated
version to gate it on. Released clients carry the byte rather than match
it exhaustively - `skeg-py` hands back the integer, `skeg-gleam` keeps it
in a plain `Int`, `skeg-client-rs` falls through to `Internal` - so a
build that has not been updated behaves exactly as it does today, and a
client that wants the distinction reads
`ErrCode::from_u8(byte).is_some_and(ErrCode::is_retryable)`. A code a
build does not know is NOT retryable: guessing that an unknown refusal
clears on its own is how a client loops on a permanent one.

A frame declaring more than the connection may ever hold is now refused
by name on the header. Before 0.7.4 the server closed the socket without
a word, which a client reads as a network fault and answers by
reconnecting and sending the same frame.

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
