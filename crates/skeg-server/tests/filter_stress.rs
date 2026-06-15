//! Stress the full filtered-search pipeline (payload index + filter grammar +
//! planner + filtered walk) on real embeddings with a VARIETY of filters.
//!
//! Heavy local data, not in CI: `#[ignore]`, gated on `SKEG_CORPUS` /
//! `SKEG_QUERIES` (`.npy`, `<f4`, C-order). Run with:
//!
//! ```sh
//! SKEG_CORPUS=/path/corpus_mxbai-wiki-chunked_500k.npy \
//! SKEG_QUERIES=/path/queries_mxbai-wiki-chunked_1000.npy \
//!   cargo test -p skeg-server --test filter_stress --release -- --ignored --nocapture
//! ```
//!
//! Every vector gets a synthesised payload with fields of different shapes:
//!   - `cat`  : 10 categories from geometric anchors (CLUSTERED, the real case);
//!   - `user` : id % 1000 (SCATTERED, high-cardinality, selective);
//!   - `year` : 2018 + id % 8 (SCATTERED, for ranges);
//!   - `type` : id % 4 (SCATTERED, medium);
//!   - `flag` : present on id % 3 == 0 (for EXISTS).
//! Then a list of filter expressions is parsed, evaluated against the real
//! payload index, served by the same planner the server uses (exact below the
//! crossover, filtered walk above), and scored against the exact top-k over the
//! matching subset. Reports |S|, path, mean recall@10, mean latency per filter.

use std::collections::BTreeSet;
use std::path::Path;
use std::time::{Duration, Instant};

use skeg_server::payload::{PayloadIndex, parse_fields, parse_filter};
use skeg_vector::{DiskVamanaIndex, VamanaConfig, VamanaIndex};

const FILTER_EXACT_MAX: usize = 16_384;
const DIM: usize = 1024;
const K: usize = 10;

fn load_npy(path: &Path) -> (Vec<f32>, usize, usize) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(&bytes[0..6], b"\x93NUMPY", "not a npy file");
    let hlen = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + hlen]).unwrap();
    assert!(header.contains("'<f4'"), "expected <f4");
    let nums: Vec<usize> = header
        .split("'shape':")
        .nth(1)
        .unwrap()
        .chars()
        .map(|c| if c.is_ascii_digit() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    let flat: Vec<f32> = bytes[10 + hlen..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (flat, nums[0], nums[1])
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-12)
}

fn row(flat: &[f32], i: usize) -> &[f32] {
    &flat[i * DIM..(i + 1) * DIM]
}

/// 10 categories by nearest geometric anchor (clustered, correlated with vectors).
fn cat_labels(corpus: &[f32], n: usize) -> Vec<u32> {
    let c = 10usize;
    let anchors: Vec<usize> = (0..c).map(|i| (i * n) / c).collect();
    (0..n)
        .map(|i| {
            let v = row(corpus, i);
            let mut best = (f32::MIN, 0u32);
            for (ai, &a) in anchors.iter().enumerate() {
                let s = cosine(v, row(corpus, a));
                if s > best.0 {
                    best = (s, ai as u32);
                }
            }
            best.1
        })
        .collect()
}

fn seeds_of(s: &BTreeSet<u64>) -> Vec<u64> {
    const SEEDS: usize = 32;
    let step = (s.len() / SEEDS).max(1);
    s.iter().copied().step_by(step).take(SEEDS).collect()
}

/// The server planner: exact over `s` below the crossover, filtered walk above.
fn planner_search(disk: &DiskVamanaIndex, q: &[f32], s: &BTreeSet<u64>) -> (Vec<u64>, Duration) {
    let ids: Vec<u64> = s.iter().copied().collect();
    let t = Instant::now();
    let hits = if s.len() > FILTER_EXACT_MAX {
        disk.search_filtered(q, K, 0, &|id| s.contains(&id), &seeds_of(s))
            .unwrap()
    } else {
        disk.score_ids(q, &ids, K).unwrap()
    };
    (hits.into_iter().map(|(id, _)| id).collect(), t.elapsed())
}

/// Exact top-K over the matching subset (ground truth).
fn exact_over(corpus: &[f32], q: &[f32], s: &BTreeSet<u64>) -> Vec<u64> {
    let mut scored: Vec<(f32, u64)> = s
        .iter()
        .map(|&id| (cosine(q, row(corpus, id as usize)), id))
        .collect();
    scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    scored.into_iter().take(K).map(|(_, id)| id).collect()
}

#[test]
#[ignore = "needs SKEG_CORPUS/SKEG_QUERIES with real embeddings"]
fn filter_grammar_stress_on_real_embeddings() {
    let Ok(corpus_path) = std::env::var("SKEG_CORPUS") else {
        eprintln!("set SKEG_CORPUS+SKEG_QUERIES; skipping");
        return;
    };
    let queries_path = std::env::var("SKEG_QUERIES").expect("set SKEG_QUERIES too");
    let q_cap: usize = std::env::var("SKEG_BENCH_QUERIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(25);

    let (corpus, n, cols) = load_npy(Path::new(&corpus_path));
    let (queries, nq, _) = load_npy(Path::new(&queries_path));
    assert_eq!(cols, DIM);
    let q_used = nq.min(q_cap);
    println!("corpus={n} queries={q_used} dim={DIM}");

    // Build the vector index.
    let t0 = Instant::now();
    let ids: Vec<u64> = (0..n as u64).collect();
    let index = VamanaIndex::build(corpus.clone(), ids, DIM, &VamanaConfig::default());
    let tmp = tempfile::TempDir::new().unwrap();
    index.save(tmp.path()).unwrap();
    let disk = DiskVamanaIndex::open(tmp.path()).unwrap();
    drop(index);
    println!("index built+opened in {:?}", t0.elapsed());

    // Synthesise payloads + build the real payload index.
    let cats = cat_labels(&corpus, n);
    let mut pidx = PayloadIndex::default();
    for id in 0..n as u64 {
        let i = id as usize;
        let mut p = format!(
            "cat=c{} user=u{} year={} type=t{}",
            cats[i],
            id % 1000,
            2018 + (id % 8),
            id % 4
        );
        if id % 3 == 0 {
            p.push_str(" flag=1");
        }
        pidx.upsert(id, parse_fields(p.as_bytes()));
    }
    println!("payload index built\n");

    // Plain (no-filter) baseline: confirms the filtered-search work did not
    // regress ordinary nearest-neighbour search. Ground truth is the exact
    // top-K over the whole corpus.
    {
        let all: BTreeSet<u64> = (0..n as u64).collect();
        let (mut recall, mut lat) = (0.0f64, Duration::ZERO);
        for qi in 0..q_used {
            let q = row(&queries, qi);
            let want = exact_over(&corpus, q, &all);
            let t = Instant::now();
            let got: Vec<u64> = disk
                .search(q, K)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            lat += t.elapsed();
            let hit = got.iter().filter(|id| want.contains(id)).count();
            recall += hit as f64 / K as f64;
        }
        println!(
            "{:42} |S|={n:7} path=plain recall@10={:.3} lat/q={:?}",
            "(no filter)",
            recall / q_used as f64,
            lat / q_used as u32,
        );
    }

    let filters = [
        "cat = c0",                                // clustered equality
        "user = u7",                               // scattered, selective -> exact
        "type IN (t0, t1)",                        // membership, broad
        "year >= 2023",                            // range, broad
        "year BETWEEN 2019 AND 2021",              // range window
        "cat = c0 OR cat = c1",                    // union
        "cat = c0 AND year >= 2022",               // clustered AND range
        "cat = c0 AND NOT type = t0",              // AND NOT (the difference path)
        "flag EXISTS",                             // presence
        "NOT flag EXISTS",                         // absence
        "(cat = c0 OR cat = c1) AND year >= 2022", // composite
        "type = t0 AND user = u3",                 // two scattered, very selective
    ];

    for expr in filters {
        let filter = parse_filter(expr).unwrap();
        let s = filter.evaluate(&pidx);
        if s.is_empty() {
            println!("{expr:42} |S|=      0  (empty)");
            continue;
        }
        let path = if s.len() > FILTER_EXACT_MAX {
            "walk "
        } else {
            "exact"
        };
        let (mut recall, mut lat) = (0.0f64, Duration::ZERO);
        for qi in 0..q_used {
            let q = row(&queries, qi);
            let want = exact_over(&corpus, q, &s);
            let (got, dt) = planner_search(&disk, q, &s);
            lat += dt;
            assert!(
                got.iter().all(|id| s.contains(id)),
                "{expr}: returned a non-matching id"
            );
            let hit = got.iter().filter(|id| want.contains(id)).count();
            recall += hit as f64 / want.len().max(1) as f64;
        }
        println!(
            "{expr:42} |S|={:7} path={path} recall@10={:.3} lat/q={:?}",
            s.len(),
            recall / q_used as f64,
            lat / q_used as u32,
        );
    }
}
