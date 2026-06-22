# Incremental insert — design

Status: **design** (branch `feat/incremental-insert`). Nothing here is built yet.

## The problem

Today a disk VINDEX is one immutable on-disk Vamana graph (`main`) plus an
in-RAM `delta` of streaming inserts. A search walks `main` and **brute-force
scans the delta**, then re-ranks the survivors with exact f32. Consolidation
folds the delta back by **rebuilding the whole graph from scratch**
(`VamanaIndex::build` over `main ∪ delta`), on a geometric schedule
(`delta_len >= max(main_len, 4096)`).

Consequences:

- **Recall is already perfect** — the delta is an exact f32 scan, so streaming
  writes never lose recall. (This is *not* the thing to fix. The old
  FreshDiskANN in-place insert that collapsed to recall 0.31 was removed; we are
  not bringing it back.)
- **Delta search is O(delta)** — latency grows linearly with un-consolidated
  writes until the next consolidation.
- **Consolidation is a full O(N log N) rebuild** — one big stall. At 500K this
  is the ~2× cold-build cost that Qdrant (incremental HNSW) and brinicle
  (streaming HNSW) beat us on. It is the *only* axis where skeg loses, and it is
  exactly where the competition aims.

**Goal:** make ingest smooth — bounded, sub-linear delta search and *no single
big rebuild stall* — without giving up the perfect recall or the lean footprint,
and **without ever mutating a mature graph in place** (the source of the 0.31
collapse).

Non-goals: changing the query API, changing the quantized tiers, distributed
ingest. Single-node, same recall, same RAM envelope.

## Why in-place insert is off the table

The removed FreshDiskANN path inserted into the *mature* `main` graph: greedy
search for neighbours, add bidirectional edges, robust-prune the new node **and
its neighbours**. Pruning existing nodes eroded the back-edges that make the
medoid-rooted greedy walk reach the whole graph. The graph stayed 100%
*reachable* but became greedy-*unnavigable* — plain recall 0.31. Any design that
mutates edges of an existing large graph risks the same. So: **segments are
immutable. We never prune an edge that exists.** Erosion cannot happen if we
never erode.

## Design: an LSM tree of immutable Vamana segments

Model the index as a log-structured merge tree whose runs are Vamana graphs.

```
L0  in-RAM f32 buffer (tiny, < FLUSH)         brute-forced, absorbs writes
L1  immutable on-disk Vamana segment(s)       bulk-built from an L0 flush
L2  immutable on-disk Vamana segment(s)       merge of L1 runs
...                                            geometric sizes
Lk  the big base segment (today's `main`)
```

- **Insert** → append to L0 (RAM + WAL), O(1). No graph touched.
- **Flush** → when L0 reaches `FLUSH` (e.g. 4096), bulk-build a small immutable
  L1 segment from it (`VamanaIndex::build` over a few thousand vectors = a few
  ms) and clear L0. Cheap because it is small and fresh — no erosion, it is a
  brand-new graph.
- **Compaction** → when a level holds too many runs (or too many bytes), merge
  its runs (+ optionally the next level) into one fresh immutable segment in the
  next level. Leveled like RocksDB: small merges are frequent and cheap, big
  merges are rare. Total write amplification is **O(log N), spread out** —
  there is no longer a single O(N log N) stall.
- **Search** → greedy-walk *every* segment's graph (they are independent, walk
  them with the existing rayon pool), brute-force the tiny L0, merge the
  per-segment top-L candidates, re-rank once with exact f32. Tombstones and
  delta-shadowing apply across all runs (newest wins).

This is exactly today's "main walk + delta scan" generalised to *N* immutable
runs instead of one, with the delta promoted from a flat scan to a small
navigable graph.

### Why this beats the current rebuild

| | today | LSM segments |
| --- | --- | --- |
| insert | O(1) to delta | O(1) to L0 |
| delta/recent search | **O(delta)** scan | **O(log·run)** graph walk |
| folding writes in | **one O(N log N) stall** | **amortised O(log N)**, no stall |
| recall | perfect (f32 rerank) | perfect (f32 rerank) |
| erosion risk | none | **none** (immutable runs) |

The win is *smoothness*: we trade one big periodic rebuild for many small merges,
which is precisely how Qdrant/brinicle avoid the stall — but our runs stay
immutable and bulk-built, so recall never drifts.

### Costs and the knobs that bound them

- **Search fan-out.** k segments → k graph walks. Bounded by leveling: with a
  size ratio of ~8–10, a 1M index is ~6–7 runs, walked in parallel. Budget: keep
  total resident graph+tier across runs within the current single-graph
  envelope; leveling keeps run count logarithmic.
- **RAM.** Each run keeps its graph + quantized tier resident (the f32 vectors
  stay on disk, as now). Sum across runs ≈ one graph's worth × small constant.
  `resident_bytes()` already accounts per-structure; extend it to sum runs.
- **Write amplification.** Leveled LSM is O(log N) total, vs the current
  geometric rebuild which is also ~2N but in one stall. Same order, spread out.

## Recall & latency targets (the gates)

Reuse the bench discipline. New gate `incremental_gate`:

- **Recall through a cycle.** Stream N real embeddings (mxbai-1024 and
  MiniLM-384) in batches; after every batch, recall@10 vs brute-force GT must
  stay **≥ 0.98** (it should stay ~1.0 — f32 rerank — this gate catches a merge
  bug, not quantization).
- **Latency stays bounded.** p50 must not grow unbounded with un-merged writes:
  measure p50 at delta=0 and at the pre-compaction high-water mark; the ratio
  must stay **≤ ~2×** (vs today's O(delta) blow-up).
- **Ingest smoothness.** No single merge may stall ingest longer than building
  the largest single level (not the whole index). Measure max merge pause.
- **No regression.** Steady-state served recall/RAM/latency must match the
  current single-graph numbers on Slice A (no worse than `main`).

## Test plan (TDD)

1. Unit: L0 flush builds a correct small segment; search merges runs correctly;
   newest-run-wins on id shadowing; tombstones honoured across runs.
2. Property: insert M random vectors in random batches, query each — every result
   id is live and the top-1 is the true nearest (small N, exact check).
3. Recall gate on real embeddings, streamed (above).
4. Latency gate (above).
5. Crash/replay: WAL replays L0; immutable segments are durable; reopen yields
   identical results.

Write the gate first (red), implement to green, then the property tests.

## Phased implementation

1. **Multi-run search.** Generalise `search_inner` to walk a `Vec<Segment>` +
   L0, merge, rerank. No write path yet — seed with the existing `main` as the
   sole run to prove no regression (gate: Slice A unchanged).
2. **L0 + flush.** Promote the delta to L0 with a `FLUSH` threshold that
   bulk-builds an immutable run. Geometric, small.
3. **Leveled compaction.** Background per-shard task merges runs by level
   (reuse the idle-consolidate plumbing). Tune the size ratio against the
   fan-out budget.
4. **Accounting + ops.** Extend `resident_bytes()` over runs; expose run count /
   level sizes via STATS; keep the explicit `VINDEX.CONSOLIDATE` as "merge all
   to one run".

## Open questions

- Size ratio / fan-out sweet spot — measure, do not guess (Slice-A + a streaming
  workload).
- Do we keep a single big base segment (today's `main`) special-cased, or treat
  it as just the top level? Start special-cased (less churn), revisit.
- Per-run tier: every run quantized, or only the base? Start all-quantized
  (uniform search path), measure the RAM cost of small-run tiers.

## Success criterion

Stream 500K vectors and serve them at recall@10 ≥ 0.98 and bounded p50, with
**no merge pause longer than building one level**, and steady-state numbers no
worse than the current single-graph index. That closes the cold-build gap — the
last axis where the competition leads.
