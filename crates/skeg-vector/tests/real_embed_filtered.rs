//! Filtered-search quality on real mxbai embeddings (1024-dim), at scale.
//!
//! Heavy local data, not in CI: `#[ignore]`, gated on env vars. Defaults to the
//! 10k mxbai-embed-large set; point `SKEG_CORPUS` / `SKEG_QUERIES` at the
//! wiki-chunked 500k / 1m `.npy` for a scale run. Example:
//!
//! ```sh
//! SKEG_CORPUS=/path/corpus_mxbai-wiki-chunked_500k.npy \
//! SKEG_QUERIES=/path/queries_mxbai-wiki-chunked_1000.npy \
//!   cargo test -p skeg-vector --test real_embed_filtered --release \
//!   -- --ignored --nocapture
//! ```
//!
//! For each selectivity it labels every vector two ways and compares the
//! filtered search against the exact top-k over the matching subset:
//! - `id % C` (uncorrelated: matches scattered uniformly, easy for the walk);
//! - nearest of C geometric anchors (correlated: matches cluster in one region,
//!   the realistic "topic" case and the hard one for the walk).

use std::path::Path;
use std::time::{Duration, Instant};

use skeg_vector::{DiskVamanaIndex, VamanaConfig, VamanaIndex};

/// Mirrors the server planner's crossover (`shard::VectorBackend::filtered_search`).
const FILTER_EXACT_MAX: usize = 16_384;
const DIM: usize = 1024;
const K: usize = 10;
const MAX_QUERIES: usize = 50;

fn load_npy(path: &Path) -> (Vec<f32>, usize, usize) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(&bytes[0..6], b"\x93NUMPY", "not a npy file");
    let hlen = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + hlen]).unwrap();
    assert!(header.contains("'<f4'"), "expected <f4, got {header}");
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
    let (rows, cols) = (nums[0], nums[1]);
    let flat: Vec<f32> = bytes[10 + hlen..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(flat.len(), rows * cols, "data size mismatch");
    (flat, rows, cols)
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

/// Category per vector by nearest of `c` anchors (evenly spaced corpus rows).
/// Filtering `cat == 0` selects a geometric cluster: the correlated, hard case.
fn anchor_labels(corpus: &[f32], n: usize, c: usize) -> Vec<u32> {
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

fn exact_matching(corpus: &[f32], n: usize, q: &[f32], m: &dyn Fn(u64) -> bool) -> Vec<u64> {
    let mut scored: Vec<(f32, u64)> = (0..n as u64)
        .filter(|&id| m(id))
        .map(|id| (cosine(q, row(corpus, id as usize)), id))
        .collect();
    scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    scored.into_iter().take(K).map(|(_, id)| id).collect()
}

fn overlap(got: &[u64], want: &[u64]) -> f64 {
    got.iter().filter(|id| want.contains(id)).count() as f64 / K as f64
}

/// A spread of matching ids, used to seed the filtered walk.
fn seeds_of(s: &[u64]) -> Vec<u64> {
    const SEEDS: usize = 32;
    let step = (s.len() / SEEDS).max(1);
    s.iter().copied().step_by(step).take(SEEDS).collect()
}

/// Run the planner's exact-vs-walk choice for one query and time it.
fn planner_search(disk: &DiskVamanaIndex, q: &[f32], m: &dyn Fn(u64) -> bool, s: &[u64]) -> (Vec<u64>, Duration) {
    let t = Instant::now();
    let got: Vec<u64> = if s.len() > FILTER_EXACT_MAX {
        disk.search_filtered(q, K, 0, m, &seeds_of(s)).unwrap()
    } else {
        disk.score_ids(q, s, K).unwrap()
    }
    .into_iter()
    .map(|(id, _)| id)
    .collect();
    (got, t.elapsed())
}

#[test]
#[ignore = "needs SKEG_CORPUS/SKEG_QUERIES (or SKEG_EMBED_DIR) with mxbai embeddings"]
fn filtered_search_quality_on_real_embeddings() {
    let (corpus_path, queries_path) = match std::env::var("SKEG_CORPUS") {
        Ok(c) => (c, std::env::var("SKEG_QUERIES").expect("set SKEG_QUERIES too")),
        Err(_) => {
            let Ok(d) = std::env::var("SKEG_EMBED_DIR") else {
                eprintln!("set SKEG_CORPUS+SKEG_QUERIES or SKEG_EMBED_DIR; skipping");
                return;
            };
            (
                format!("{d}/corpus_mxbai-embed-large_10000.npy"),
                format!("{d}/queries_mxbai-embed-large_200.npy"),
            )
        }
    };
    let (corpus, n, cols) = load_npy(Path::new(&corpus_path));
    let (queries, nq, _) = load_npy(Path::new(&queries_path));
    assert_eq!(cols, DIM);
    let q_cap: usize = std::env::var("SKEG_BENCH_QUERIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(MAX_QUERIES);
    println!("corpus={n} queries={nq} dim={DIM} q_cap={q_cap}");

    let t0 = Instant::now();
    let ids: Vec<u64> = (0..n as u64).collect();
    let index = VamanaIndex::build(corpus.clone(), ids, DIM, &VamanaConfig::default());
    let tmp = tempfile::TempDir::new().unwrap();
    index.save(tmp.path()).unwrap();
    let disk = DiskVamanaIndex::open(tmp.path()).unwrap();
    drop(index);
    println!("index built+opened in {:?}\n", t0.elapsed());
    let q_used = nq.min(q_cap);

    for &c in &[2usize, 5, 20, 100] {
        // Uncorrelated labels: id % c.
        run_scheme(&disk, &corpus, n, &queries, q_used, c, "id%C ", &|i, _| {
            (i % c as u64) == 0
        });
    }
    // Correlated labels: nearest-anchor clusters (the hard case for the walk).
    for &c in &[2usize, 5, 20, 100] {
        let labels = anchor_labels(&corpus, n, c);
        let lab = labels.clone();
        run_scheme(&disk, &corpus, n, &queries, q_used, c, "anchor", &move |i, _| {
            lab[i as usize] == 0
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn run_scheme(
    disk: &DiskVamanaIndex,
    corpus: &[f32],
    n: usize,
    queries: &[f32],
    q_used: usize,
    c: usize,
    tag: &str,
    m: &dyn Fn(u64, usize) -> bool,
) {
    let pred = |id: u64| m(id, 0);
    let s: Vec<u64> = (0..n as u64).filter(|&id| pred(id)).collect();
    let use_walk = s.len() > FILTER_EXACT_MAX;
    let (mut recall_p, mut recall_w, mut lat) = (0.0f64, 0.0f64, Duration::ZERO);
    for qi in 0..q_used {
        let q = row(queries, qi);
        let want = exact_matching(corpus, n, q, &pred);
        let (got, dt) = planner_search(disk, q, &pred, &s);
        lat += dt;
        assert!(got.iter().all(|&id| pred(id)), "non-matching id returned");
        recall_p += overlap(&got, &want);
        let gw: Vec<u64> = disk
            .search_filtered(q, K, 0, &pred, &seeds_of(&s))
            .unwrap()
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        recall_w += overlap(&gw, &want);
    }
    let qf = q_used as f64;
    println!(
        "{tag} sel~1/{c:<3} |S|={:7} path={:5} recall@10 planner={:.3} walk={:.3} lat/q={:?}",
        s.len(),
        if use_walk { "walk" } else { "exact" },
        recall_p / qf,
        recall_w / qf,
        lat / q_used as u32,
    );
}
