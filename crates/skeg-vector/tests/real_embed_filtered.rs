//! Filtered-search quality on real mxbai-embed-large embeddings (1024-dim).
//!
//! Heavy local data, not in CI: `#[ignore]`, gated on `SKEG_EMBED_DIR` pointing
//! at the `embeddings_cache` dir with `corpus_mxbai-embed-large_10000.npy` and
//! `queries_mxbai-embed-large_200.npy`. Run with:
//!
//! ```sh
//! SKEG_EMBED_DIR=/path/to/embeddings_cache \
//!   cargo test -p skeg-vector --test real_embed_filtered -- --ignored --nocapture
//! ```
//!
//! It builds a disk Vamana index over 10k real vectors, synthesises a category
//! label per id, and at several selectivities compares the filtered search
//! (exact-over-S below the planner threshold, filtered walk above it) against
//! the exact top-k over the matching subset. Reports recall@10 and latency.

use std::path::Path;
use std::time::Instant;

use skeg_vector::{DiskVamanaIndex, VamanaConfig, VamanaIndex};

/// Mirrors the server planner's crossover (`shard::VectorBackend::filtered_search`).
const FILTER_EXACT_MAX: usize = 2048;
const DIM: usize = 1024;
const K: usize = 10;

/// Read a C-order `<f4` numpy array, returning `(flat_f32, rows, cols)`.
fn load_npy(path: &Path) -> (Vec<f32>, usize, usize) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(&bytes[0..6], b"\x93NUMPY", "not a npy file");
    let hlen = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + hlen]).unwrap();
    assert!(header.contains("'<f4'"), "expected <f4, got {header}");
    let shape_str = header.split("'shape':").nth(1).unwrap();
    let nums: Vec<usize> = shape_str
        .chars()
        .map(|c| if c.is_ascii_digit() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    let (rows, cols) = (nums[0], nums[1]);
    let data = &bytes[10 + hlen..];
    let flat: Vec<f32> = data
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

/// Exact top-K ids over the matching subset (the ground truth).
fn exact_matching(corpus: &[f32], n: usize, query: &[f32], matches: &dyn Fn(u64) -> bool) -> Vec<u64> {
    let mut scored: Vec<(f32, u64)> = (0..n as u64)
        .filter(|&id| matches(id))
        .map(|id| (cosine(query, row(corpus, id as usize)), id))
        .collect();
    scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    scored.into_iter().take(K).map(|(_, id)| id).collect()
}

#[test]
#[ignore = "needs SKEG_EMBED_DIR with the mxbai embeddings"]
fn filtered_search_quality_on_real_embeddings() {
    let Ok(dir) = std::env::var("SKEG_EMBED_DIR") else {
        eprintln!("SKEG_EMBED_DIR unset; skipping");
        return;
    };
    let dir = Path::new(&dir);
    let (corpus, n, cols) = load_npy(&dir.join("corpus_mxbai-embed-large_10000.npy"));
    let (queries, nq, qcols) = load_npy(&dir.join("queries_mxbai-embed-large_200.npy"));
    assert_eq!(cols, DIM);
    assert_eq!(qcols, DIM);
    println!("corpus={n} queries={nq} dim={DIM}");

    let t0 = Instant::now();
    let ids: Vec<u64> = (0..n as u64).collect();
    let index = VamanaIndex::build(corpus.clone(), ids, DIM, &VamanaConfig::default());
    let tmp = tempfile::TempDir::new().unwrap();
    index.save(tmp.path()).unwrap();
    let disk = DiskVamanaIndex::open(tmp.path()).unwrap();
    println!("index built+opened in {:?}", t0.elapsed());

    let q_used = nq.min(50);
    // Selectivity buckets: a category in [0, C); `cat == 0` matches ~1/C.
    for &c in &[2usize, 5, 20, 100] {
        let matches = move |id: u64| (id as usize % c) == 0;
        let s_size = (0..n as u64).filter(|&id| matches(id)).count();
        let use_walk = s_size > FILTER_EXACT_MAX;

        let mut recall_planner = 0.0f64;
        let mut recall_walk = 0.0f64;
        let mut lat_planner = std::time::Duration::ZERO;
        for qi in 0..q_used {
            let query = row(&queries, qi);
            let want = exact_matching(&corpus, n, query, &matches);

            // Planner path: exact-over-S below the threshold, walk above it.
            let t = Instant::now();
            let got: Vec<u64> = if use_walk {
                disk.search_filtered(query, K, 0, &matches).unwrap()
            } else {
                let s: Vec<u64> = (0..n as u64).filter(|&id| matches(id)).collect();
                disk.score_ids(query, &s, K).unwrap()
            }
            .into_iter()
            .map(|(id, _)| id)
            .collect();
            lat_planner += t.elapsed();
            assert!(got.iter().all(|&id| matches(id)), "non-matching id returned");
            recall_planner += overlap(&got, &want);

            // Walk path always, for the crossover picture.
            let gw: Vec<u64> = disk
                .search_filtered(query, K, 0, &matches)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            recall_walk += overlap(&gw, &want);
        }
        let qf = q_used as f64;
        println!(
            "sel~1/{c:<3} |S|={s_size:5} path={:5} recall@10 planner={:.3} walk={:.3} lat/q={:?}",
            if use_walk { "walk" } else { "exact" },
            recall_planner / qf,
            recall_walk / qf,
            lat_planner / q_used as u32,
        );
    }
}

fn overlap(got: &[u64], want: &[u64]) -> f64 {
    got.iter().filter(|id| want.contains(id)).count() as f64 / K as f64
}
