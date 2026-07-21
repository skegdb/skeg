#![allow(clippy::cast_precision_loss, clippy::needless_range_loop)]
//! End-to-end range-filtered search over the REAL disk index: does
//! `DiskVamanaIndex::search_range` (self-routing on the zone-map estimate) beat
//! the pre-existing filtered path (`search_filtered_hybrid` fed a materialised
//! id-set) once the full quantised rerank + disk reads are in the loop?
//!
//! Both paths end in the same `score_ids_quantized`, so this isolates the id-set
//! cost the zone-map avoids. Real vectors (mxbai-wiki), synthetic uniform u64
//! attribute (the conservative case; see attr_zonemap.rs). Reports per-query
//! latency old-vs-new across selectivity, and top-K overlap (new must not lose
//! recall vs old).
//!   SKEG_BENCH_N=50000  SKEG_NQ=100  SKEG_SEL=5(swept if unset)
//!
//! `harness = false`, custom main, wall-clock.

use std::time::Instant;

use skeg_vector::{DiskVamanaIndex, VamanaConfig, VamanaIndex};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
const CORPUS: &str = "skeg/bench-compare/embeddings_cache/corpus_mxbai-wiki.npy";
const QUERY: &str = "skeg/bench-compare/embeddings_cache/queries_mxbai-wiki_200.npy";
const K: usize = 10;
const RERANK: usize = 80;
const ATTR_RANGE: u64 = 1_000_000;

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

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn main() {
    let n = env_usize("SKEG_BENCH_N", 50_000);
    let nq = env_usize("SKEG_NQ", 100);
    let sels: Vec<usize> = match std::env::var("SKEG_SEL") {
        Ok(v) => vec![v.parse().unwrap()],
        Err(_) => vec![2, 5, 10, 25, 50],
    };

    let (data, n, dim) = load(&format!("{ROOT}/{CORPUS}"), n);
    let (queries, nqh, qdim) = load(&format!("{ROOT}/{QUERY}"), nq);
    assert_eq!(dim, qdim);
    let nq = nq.min(nqh);
    eprintln!("N={n} dim={dim} nq={nq} k={K} rerank={RERANK}");

    let t = Instant::now();
    let ids: Vec<u64> = (0..n as u64).collect();
    let index = VamanaIndex::build(data.clone(), ids, dim, &VamanaConfig::default());
    let tmp = tempfile::TempDir::new().unwrap();
    index.save(tmp.path()).unwrap();
    let mut disk = DiskVamanaIndex::open(tmp.path()).unwrap();
    let attr: Vec<u64> = (0..n).map(|r| splitmix(r as u64) % ATTR_RANGE).collect();
    disk.set_attr(&attr).unwrap();
    disk.build_ivf(0, 8).unwrap();
    eprintln!("build+attr+ivf: {:.1}s\n", t.elapsed().as_secs_f64());

    eprintln!(
        "{:>4}  {:>10}  {:>10}  {:>8}  {:>8}",
        "sel", "old(ms)", "new(ms)", "speedup", "overlap"
    );
    for &sel in &sels {
        let width = ATTR_RANGE * sel as u64 / 100;
        let range = |qi: usize| {
            let lo = splitmix(0xBEEF ^ qi as u64) % (ATTR_RANGE - width);
            (lo, lo + width)
        };
        // Warm the page cache for BOTH paths so neither pays the other's cold
        // reads (OLD-then-NEW ordering would otherwise flatter NEW).
        for qi in 0..nq {
            let (lo, hi) = range(qi);
            let q = &queries[qi * dim..qi * dim + dim];
            let mut s: Vec<u64> = (0..n as u64)
                .filter(|&r| (lo..=hi).contains(&attr[r as usize]))
                .collect();
            s.sort_unstable();
            let _ = disk.search_filtered_hybrid(q, &s, K, RERANK).unwrap();
            let _ = disk.search_range(q, lo, hi, K, RERANK).unwrap();
        }

        let mut overlap = 0usize;
        let (mut est_sum, mut ssize, mut wide_cnt) = (0usize, 0usize, 0usize);
        let mut news: Vec<Vec<u64>> = Vec::with_capacity(nq);

        // NEW timed FIRST, in its own loop, so it never immediately follows OLD's
        // identical reads (intra-iteration cache locality would flatter it).
        let t_new = Instant::now();
        for qi in 0..nq {
            let (lo, hi) = range(qi);
            let q = &queries[qi * dim..qi * dim + dim];
            let new = disk.search_range(q, lo, hi, K, RERANK).unwrap();
            news.push(new.iter().map(|(id, _)| *id).collect());
        }
        let new_ms = t_new.elapsed().as_secs_f64() * 1000.0;

        // OLD timed SECOND (warm caches now favour OLD, so NEW's win is the
        // conservative floor).
        let t_old = Instant::now();
        for qi in 0..nq {
            let (lo, hi) = range(qi);
            let q = &queries[qi * dim..qi * dim + dim];
            let mut s: Vec<u64> = (0..n as u64)
                .filter(|&r| (lo..=hi).contains(&attr[r as usize]))
                .collect();
            s.sort_unstable();
            let old = disk.search_filtered_hybrid(q, &s, K, RERANK).unwrap();
            let a: Vec<u64> = old.iter().map(|(id, _)| *id).collect();
            overlap += a.iter().filter(|id| news[qi].contains(id)).count();
            ssize += s.len();
            let (est, wide) = disk.debug_range_plan(lo, hi);
            est_sum += est;
            wide_cnt += usize::from(wide);
        }
        let old_ms = t_old.elapsed().as_secs_f64() * 1000.0;

        let nqf = nq as f64;
        eprintln!(
            "{:>3}%  {:>10.3}  {:>10.3}  {:>7.2}x  {:>8.3}   (|s|={}, wide {}/{})",
            sel,
            old_ms / nqf,
            new_ms / nqf,
            old_ms / new_ms.max(1e-9),
            overlap as f64 / (nqf * K as f64),
            ssize / nq,
            wide_cnt,
            nq,
        );
        let _ = est_sum;
    }
}
