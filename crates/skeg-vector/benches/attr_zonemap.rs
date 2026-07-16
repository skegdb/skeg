#![allow(clippy::cast_precision_loss, clippy::needless_range_loop)]
//! Zone-map range filter vs materialise-the-id-set: does a per-cell `[min,max]`
//! on an opaque u64 attribute beat the current filtered path, which requires the
//! caller to build the full sorted match id-set `s` first?
//!
//! The current `search_filtered_hybrid` contract is: caller supplies `s` = every
//! id matching the predicate, sorted. For a RANGE predicate on a numeric column
//! (e.g. a time axis) that means an O(n) scan of all attributes plus allocating
//! a Vec of up to `selectivity * n` ids, then `probe(s)`. The zone-map path
//! (`set_zonemap` + `probe_range`) skips whole cells whose `[min,max]` misses the
//! range and never materialises `s`.
//!
//! Real vectors (mxbai-wiki), SYNTHETIC uniform-random u64 attribute standing in
//! for a time axis. Uniform is the CONSERVATIVE case: the attribute spreads
//! evenly across every cell, so almost no cell falls fully outside a range —
//! the worst case for zone-map skipping. A real time axis correlates with
//! ingest order and would skip far more. If the zone-map wins here, it wins.
//!
//! Reports, over NQ queries: wall-clock baseline (scan+probe) vs zone-map, the
//! id-set size materialised vs avoided, cells touched, and top-K overlap so the
//! zone-map is shown not to lose candidates.
//!   SKEG_BENCH_N=100000  SKEG_NQ=200  SKEG_SEL=5  SKEG_CELLS=0(=auto)
//!
//! `harness = false`, custom main, wall-clock.

use std::time::Instant;

use skeg_simd::cosine_f32;
use skeg_vector::IvfRouter;

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
const CORPUS: &str = "skeg/bench-compare/embeddings_cache/corpus_mxbai-wiki.npy";
const QUERY: &str = "skeg/bench-compare/embeddings_cache/queries_mxbai-wiki_200.npy";
const K: usize = 10;
const BUDGET: usize = 4_096; // matches SHORTLIST in search_filtered_hybrid
const ATTR_RANGE: u64 = 1_000_000; // synthetic time domain

fn load(path: &str, cap: usize) -> (Vec<f32>, usize, usize) {
    let bytes = std::fs::read(path).unwrap();
    let hl = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + hl]).unwrap();
    let sh = header.find("'shape':").unwrap();
    let lp = header[sh..].find('(').unwrap() + sh + 1;
    let rp = header[lp..].find(')').unwrap() + lp;
    let dims: Vec<usize> = header[lp..rp]
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let (rows, dim) = (dims[0], dims[1]);
    let raw: Vec<f32> = bytes[10 + hl..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let n = cap.min(rows);
    let mut out = vec![0.0f32; n * dim];
    for i in 0..n {
        let v = &raw[i * dim..i * dim + dim];
        let nrm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
        for j in 0..dim {
            out[i * dim + j] = v[j] / nrm;
        }
    }
    (out, n, dim)
}

/// Deterministic splitmix64 — a synthetic attribute without Math.random-style
/// nondeterminism, so runs reproduce.
fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Exact top-K of a shortlist against the query (cosine), for the overlap check.
fn topk(query: &[f32], rows: &[u64], data: &[f32], dim: usize, k: usize) -> Vec<u64> {
    let mut scored: Vec<(f32, u64)> = rows
        .iter()
        .map(|&r| {
            let v = &data[r as usize * dim..r as usize * dim + dim];
            (cosine_f32(query, v), r)
        })
        .collect();
    scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    scored.truncate(k);
    scored.into_iter().map(|(_, r)| r).collect()
}

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    let n = env_usize("SKEG_BENCH_N", 100_000);
    let nq = env_usize("SKEG_NQ", 200);
    let sel = env_usize("SKEG_SEL", 5); // range width as % of ATTR_RANGE
    let cells = env_usize("SKEG_CELLS", 0);

    let (data, n, dim) = load(&format!("{ROOT}/{CORPUS}"), n);
    let (queries, nq_have, qdim) = load(&format!("{ROOT}/{QUERY}"), nq);
    assert_eq!(dim, qdim, "corpus/query dim mismatch");
    let nq = nq.min(nq_have);

    let n_cells = if cells == 0 { IvfRouter::cells_for(n) } else { cells };
    eprintln!("N={n} dim={dim} cells={n_cells} nq={nq} sel={sel}% budget={BUDGET}");

    let corr = env_usize("SKEG_CORR", 0) == 1;
    let t = Instant::now();
    let mut router = IvfRouter::build(&data, n as u32, dim, n_cells, 8);
    // Synthetic attribute. Uniform = conservative (no cell-skipping possible).
    // Correlated = a monotonic time axis (attr grows with ingest row), the real
    // agent-memory case; here cell-skipping actually engages at low selectivity.
    let attr: Vec<u64> = if corr {
        (0..n).map(|r| (r as u64) * ATTR_RANGE / n as u64).collect()
    } else {
        (0..n).map(|r| splitmix(r as u64) % ATTR_RANGE).collect()
    };
    eprintln!("attr: {}", if corr { "correlated (time axis)" } else { "uniform (conservative)" });
    router.set_zonemap(&attr);
    eprintln!("build+zonemap: {:.2}s", t.elapsed().as_secs_f64());

    let width = ATTR_RANGE * sel as u64 / 100;
    let mut base_ms = 0.0f64;
    let mut zone_ms = 0.0f64;
    let mut auto_ms = 0.0f64; // estimate-then-route path
    let mut s_total = 0usize; // ids materialised by baseline
    let mut cand_total = 0usize; // rows returned by zone-map
    let mut overlap = 0usize; // top-K agreement, summed
    let mut chose_range = 0usize; // times auto picked probe_range

    for qi in 0..nq {
        // A per-query range window over the attribute domain.
        let lo = splitmix(0xABCD ^ qi as u64) % (ATTR_RANGE - width);
        let hi = lo + width;
        let q = &queries[qi * dim..qi * dim + dim];

        // --- Baseline: materialise s (O(n) scan + sorted alloc), then probe.
        let t0 = Instant::now();
        let mut s: Vec<u64> = (0..n as u64).filter(|&r| (lo..=hi).contains(&attr[r as usize])).collect();
        s.sort_unstable(); // contract: s is sorted
        let base_short = router.probe(q, &s, BUDGET);
        base_ms += t0.elapsed().as_secs_f64() * 1000.0;
        s_total += s.len();

        // --- Zone-map: no s materialised.
        let t1 = Instant::now();
        let zone_short = router.probe_range(q, lo, hi, &attr, BUDGET);
        zone_ms += t1.elapsed().as_secs_f64() * 1000.0;
        cand_total += zone_short.len();

        // --- Auto: estimate |s| from the zone-map (O(cells), no scan), then
        // route. estimate >= BUDGET -> range path; else materialise s + probe.
        let t2 = Instant::now();
        let est = router.estimate_range_count(lo, hi);
        if est >= BUDGET {
            let _ = router.probe_range(q, lo, hi, &attr, BUDGET);
            chose_range += 1;
        } else {
            let mut s2: Vec<u64> = (0..n as u64).filter(|&r| (lo..=hi).contains(&attr[r as usize])).collect();
            s2.sort_unstable();
            let _ = router.probe(q, &s2, BUDGET);
        }
        auto_ms += t2.elapsed().as_secs_f64() * 1000.0;

        // Quality: top-K after exact rerank must agree.
        let a = topk(q, &base_short, &data, dim, K);
        let b = topk(q, &zone_short, &data, dim, K);
        overlap += a.iter().filter(|r| b.contains(r)).count();
    }

    let nqf = nq as f64;
    eprintln!("\n--- per-query averages ({nq} queries, sel={sel}%) ---");
    eprintln!("baseline (scan+probe): {:.3} ms   | id-set materialised: {} ids/query", base_ms / nqf, s_total / nq);
    eprintln!("zone-map (probe_range): {:.3} ms   | candidates: {} rows/query", zone_ms / nqf, cand_total / nq);
    eprintln!("auto (estimate+route): {:.3} ms   | chose range path: {}/{} queries", auto_ms / nqf, chose_range, nq);
    eprintln!("speedup zone vs base: {:.2}x   | speedup AUTO vs base: {:.2}x", base_ms / zone_ms.max(1e-9), base_ms / auto_ms.max(1e-9));
    eprintln!("top-{K} overlap (quality, want ~1.0): {:.3}", overlap as f64 / (nqf * K as f64));
}
