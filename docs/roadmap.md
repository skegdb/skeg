# Roadmap

Driven by what real workloads ask for first. Nothing here blocks the current
release; these are the next things to reach for when a workload demands them.

## Planned

- **x86_64 native tuning.** AVX2 / AVX-512 kernels and native Linux validation.
  Release binaries are aarch64 (Apple Silicon, Linux ARM) today; the source
  builds on x86_64 but the SIMD paths are not tuned for it and native Linux
  validation is not done.
- **VSEARCH child spans.** Break the query internals out as child spans through
  the OTLP exporter, so a trace shows the walk, re-rank, and filter phases
  separately.

## Conditional

Cold-build acceleration, staged. Bulk-loading a fresh index today streams
through `SKEG.VMSET` and rebuilds the graph on a geometric schedule (O(log N)
bulk builds over a load, O(N) total work at recall 1.0). If cold-load time
proves a real bottleneck on a workload, in order:

- **Batch-build mode (CPU).** Suppress the intermediate consolidates during a
  known-size load and do a single final build instead. Small change, reuses the
  existing consolidate path, no new dependency.
- **Opportunistic GPU build offload.** The consolidate already runs in the
  background when an index goes idle. On a box with a co-resident GPU that is
  lightly loaded (the model between bursts), offload that graph build to the GPU
  and fall back to the CPU when the GPU is contended. This is a scheduling
  feature on the build path, not a kernel on the query path, and it earns its
  keep only after the CPU batch-build is shown to be not enough. It borrows
  cycles the model is not using rather than competing for them.

## Needs study

- **GPU-parallel graph walk.** Graph ANN on the GPU is real (CAGRA, GGNN):
  expand many frontier neighbors per step instead of one, so a wider beam costs
  the same wall-clock. It can map onto skeg without breaking the RAM-frugal
  thesis, because the walk is already driven by the small quantized codes, not
  the f32 vectors. Mirror the tq codes and the graph adjacency into GPU memory
  (for 1M x 1024-dim tq1 that is ~128 MB of codes plus ~256 MB of a degree-64
  graph, both fit in HBM) and keep the f32 re-rank on the CPU reading from SSD.
  Two measurements decide whether it pays:
  - **Amdahl on the re-rank.** If the disk re-rank dominates query time, a faster
    walk barely moves the total. Measure the walk-vs-re-rank split first; if the
    re-rank is the bulk, GPU walk is a dead end.
  - **Where the recall headroom is.** At recall 1.0 on today's single-tenant
    benchmark there is nothing to gain on recall, only QPS or latency. The one
    place a wider parallel beam might raise recall is the tq1-at-scale falloff
    the tiers already admit (recall@100 drops on large corpora). Recover that and
    GPU earns an accuracy story, not just throughput.

  Deployment mode decides the framing. On a box shared with the model the GPU is
  already the model's, so this targets a standalone skeg server with a spare GPU,
  or model-idle windows. As a plain CPU-saturation fallback it is weakest: the
  GPU tends to be busy with the model exactly when skeg's CPU is saturated.

- **Incremental in-graph insert, re-evaluated.** Rejected once: naive per-vector
  insertion with local back-edge re-pruning eroded the long-range edges and
  recall@10 fell to ~0.31 against 1.00 for a bulk build at 100k. That was an
  older skeg, and the test measured the raw walk, not today's pipeline. Three
  things changed the calculus:
  - **The known fix was not in that attempt.** FreshDiskANN's StreamingMerge is
    the published answer to exactly this: incremental insert plus a background
    re-consolidation that re-prunes the affected region globally, not locally, so
    long-range edges survive. skeg is Vamana (DiskANN's graph), so it applies
    directly; the old attempt did the naive local prune.
  - **Every query now f32-re-ranks** (k*8, adaptive bound). A graph that
    navigates worse but still lands the true neighbors in a wide beam recovers
    much of the lost recall at re-rank. Re-measure with the current pipeline, not
    the raw "plain recall@10".
  - **The motivation is insert latency, not throughput.** Delta plus geometric
    rebuild already streams 100k in ~28s at recall 1.0. The gap it leaves is the
    p99 of a single insert: the one that triggers a doubling pays a full
    consolidate. Workloads that need smooth per-insert latency (real-time
    updates, an append-heavy fact store's retract path) are what true incremental
    insert would serve.

  Gate: only worth building if a workload needs consistent single-insert latency
  that the geometric-rebuild spikes cannot give.
