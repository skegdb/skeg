# Architecture

How skeg serves recall 1.0 on a RAM footprint the RAM-resident engines cannot
match. The full design and the eleven falsifications behind it are on the
[project blog](https://amanitaproject.com/).

## The index lives on the SSD

The Vamana graph is walked on disk with a small, byte-budgeted cache for the hot
pages. The cache is S3-FIFO and returns evicted pages to the OS through jemalloc
decay timers. A resident quantized tier drives the graph walk, and re-ranking
against the full-precision vectors on disk recovers the accuracy quantization
gives up. So RAM holds only the quantized working set, not the vectors.

## Tiers

The index kind trades memory for precision, chosen at `VINDEX.CREATE`:
`f32`, `int8`, `binary`, and TurboQuant at 1, 2, or 4 bits per coordinate
(`tq1`/`tq2`/`tq4`). TurboQuant needs no trained codebook, so the lean tiers work
under live writes, not just at serve time. Every tier keeps the full f32
re-rank, so the top result is exact regardless of tier.

Product Quantization (128x256) is serve-only (it needs a trained codebook); it
is not a live-writable kind.

`tq2` is the default and the right choice unless you know otherwise:

| tier | RAM vs int8 | recall | use it when |
| --- | --- | --- | --- |
| **tq2** (default) | ~1/4 | recall@10 and recall@100 flat as the corpus grows | almost always; the safe default that does not surprise you at scale |
| **tq1** | ~1/8 | recall@10 holds; recall@100 falls off at scale | small tenants, RAM-critical boxes, or workloads that only need the top handful of results |

`tq1` is a one-word opt-in (`SKEG.VINDEX.CREATE name dim tq1 disk`). It is not the
default on purpose: a default should not degrade recall silently as a corpus
grows, which is exactly what `tq1`'s recall@100 falloff would do to someone who
never tuned.

## Filtered search scales

A filter with many matches does not score every match. Its matching set is routed
by a coarse k-means index to the query-nearest cells that hold matches, then a
short list is re-ranked, so the cost stays sub-linear as the corpus grows. See
[`filtered-search.md`](filtered-search.md).

## Memory follows the workload

With `--tier-mmap` the quantized codes are backed by a file, so the OS reclaims
them when an index goes cold or memory gets tight, and pages them back in on
demand. Latency is unchanged while an index is hot.
