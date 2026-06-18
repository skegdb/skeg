# Devlog — incremental insert (v0.5.0 line)

Scope: the streaming-insert / LSM work on top of the v0.5.0 dev line, plus the
benchmark and debugging story around it. Honest record — what landed, what we
tried and dropped, and the numbers that decided each call.

## Where we started (v0.5.0 dev = `feat/tq-rw-tier`, `1a98e34`)

Before this work the engine already had, on the 0.5.0 dev line:

- **TQ-RW tier** — data-oblivious quantisation (tq1/tq2/tq4), no training, RW-capable.
  tq2 is the sweet spot: recall ~1.0 at a quarter of int8's bytes.
- **`resident_bytes()`** — a deterministic per-index RAM gauge (graph + tier + delta),
  replacing the allocator-noisy serve-RSS for reporting.
- **skeg-bench** (public) — turnkey suite, 6-engine comparison (skeg / Lance / Chroma /
  Milvus / hnswlib / Qdrant), GloVe + MNIST validation, cost calculator.

The honest weak axis vs Qdrant was **ingest/build time**. Streaming inserts went to
an in-RAM delta; serving well meant a full `VamanaIndex::build` rebuild
(`consolidate`), and at 500k that rebuild is a multi-minute stall. Closing that gap
without losing the lean+fast serve was the goal.

## The design — LSM of immutable Vamana segments (`bde59da`)

A prior attempt at in-place insertion (FreshVamana) was rejected earlier: back-edge
erosion collapsed plain recall to 0.31. So: **never mutate an existing graph.**

- `insert` → in-RAM L0 (`delta`) + WAL.
- L0 fills → **flush**: bulk-build L0 into an immutable on-disk **run** (a Vamana
  segment), append to `runs`, clear L0. No rebuild of anything existing.
- search walks `base` + every run, then re-ranks.
- `consolidate` folds `base ∪ runs ∪ delta` into one fresh base (background / idle).

Durability ADR (`docs/adr-incremental-flush.md`): **the WAL stays the single source
of truth.** A flush is a throwaway optimisation that does NOT truncate the WAL, so a
crash mid-flush loses nothing (the run is rebuildable from the WAL). Run discovery on
restart is deferred — the WAL replays runs back into L0.

## What we built (gate-driven, TDD)

A pre-registered gate (`incremental_gate`) fixed the target up front: **recall ≥ 0.98
through a flush/consolidate cycle, and high-water p50 ≤ 2× the post-consolidate p50.**
Red baseline (`5f0fe10`): recall 0.996 PASS, latency 11.5× FAIL (O(delta) brute scan).

Phases, each behaviour-preserving and green:

- **1a** (`f7de6fa`) extract `Segment` (base), **1b** (`f39bb5f`) multi-run search,
  **2a** (`528f399`) run-aware id resolution — all additive, runs empty = no-op.
- **2b** (`09468c1`) flush + `consolidate` folds runs (newest-wins). Gate 11.6× → 5.2×.
- **3a** (`a344cb8`) global bounded re-rank across segments + shallow run walks.
  Gate 5.2× → 2.8×, recall 0.997 held.

The build win, measured on the RW path: the **worst ingest stall dropped from a full
consolidate (~12s for 30k, minutes at 500k) to a single flush (~1.14s)**, with smooth
streaming in between.

## The regression hunt — the part worth reading (`6830aa0`)

A same-machine A/B (rebuilding the pre-incremental binary, not trusting saved numbers)
caught a real serve-latency regression: **post-consolidate p50 was ~1.45× slower at
500k** (p50 7.2 vs 5.0, p95 24 vs 15.7). 100k/200k showed nothing — it was
**scale-dependent**, and **tail-heavy** (the p95 gap dwarfed the p50 gap).

That fingerprint — scale-dependent + tail-heavy, recall untouched at 1.0 — is a
**cache-locality** bug, not a graph-quality one. Root cause: the re-rank reads f32 rows
from `vectors.bin` by row, and the row order is the order vectors are fed to the build.
Our `consolidate` folded `[delta, runs.reverse, base]` ≈ reverse-id, scattering a
query's near-neighbours across the 2 GB file → page-cache misses at scale (invisible at
100k where the file fits cache).

Fix: **rebuild in id order** so near-neighbours land at nearby rows. First attempt
(naive gather of a second 2 GB vector buffer) doubled the build via memory pressure;
the clean fix collects survivor *refs* (~24 B each), sorts by id, and materialises the
vectors once in id order — no doubling.

Result: not just fixed but **better than the pre-incremental baseline** — strict id
order beats the baseline's `[base, delta]` layout. (Lesson saved to memory:
`vectors.bin` row order = re-rank cache locality; diagnose latency regressions with a
same-machine A/B, never against numbers from an earlier session.)

## What we dropped — size-tiered leveling (`feat/incremental-leveling`, archived)

To serve WITHOUT a consolidate, we tried RocksDB-style leveling (fan-out 4): runs merge
geometrically so the run count stays logarithmic. It works as designed (run count
bounded, proven by test) and **recall held at 1.0/1.0**. But:

- **Serve from leveled runs is ~3× slower** (p50 12.5 vs 3.9 same-batch). Bounding the
  run count to ~16 (from 122) is not enough — 16 runs = 16 graph walks + a re-rank
  scattered across 16 files, vs one consolidated graph = one walk. Fast serve needs ~1
  run, which means consolidating.
- The merges add **write-amplification** (~3.4× rewrites at 500k), so ingest isn't free
  either.

Off skeg's serve-quality Pareto, so **rejected** — kept as an archived branch + a
negative-result memo. (If a write-heavy "can't pause to consolidate" workload ever
shows up, 3× serve with no consolidate stall might suit it. Not today's problem.)

## Results — clean same-machine A/B (pre-incremental `1a98e34` vs keeper `6830aa0`, tq2, sequential)

| N | | build_s | p50 | p95 | recall@10/@100 |
|---|---|---|---|---|---|
| 100k | pre-incr | 61.6 | 2.41 | 2.74 | 1.0/1.0 |
| | **incr** | **53.8** | **2.23** | **2.43** | 1.0/1.0 |
| 200k | pre-incr | 177.5 | 2.52 | 2.88 | 1.0/1.0 |
| | **incr** | **149.9** | **2.32** | **2.68** | 1.0/1.0 |
| 500k | pre-incr | 598.0 | 3.23 | 8.70 | 1.0/1.0 |
| | **incr** | **522.4** | **2.54** | **3.57** | 1.0/1.0 |

The keeper beats the pre-incremental line on **build, p50, and p95 at every scale**,
recall identical at 1.0, and adds the smooth-ingest (no consolidate stall) win on top.
The advantage grows with N (p95 at 500k: 8.70 → 3.57).

## What went well

- **The LSM flush model** — turned a minutes-long rebuild stall into ~1.14s flushes.
- **The id-order consolidate fix** — serve now *beats* the baseline (2× p50, ~2.4× p95
  at 500k) instead of regressing.
- **Discipline paid off** — the pre-registered gate, behaviour-preserving phases, and
  the same-machine A/B together caught a regression that the saved numbers had hidden.
- **Recall never moved** — 1.0 throughout, including the rejected leveling path.

## What didn't

- **Leveling** — correct and recall-safe but 3× serve; net negative. Dropped.
- **First id-sort attempt** — the naive 2 GB gather doubled the build (memory pressure);
  needed the ref-sort rewrite.
- **Parallel benchmark runs contaminated the numbers** (the "200k build = 612s" outlier,
  the inflated 500k pairs). Sequential, same-machine A/B is the only trustworthy form.
- **The flush still pauses ~1.14s** every L0 fill — synchronous on the insert path.
  Background flush would hide it; deferred.

## Open / deferred

- **Background flush** — build the run off the insert hot path (truly stall-free ingest).
- **Run discovery on `open`** — reload flushed runs after a restart instead of replaying
  the WAL into L0 (Phase 4; a persistence optimisation, not a correctness gap).
- **Merge the keeper into the 0.5.0 line** after a Linux/x86 validation pass.
