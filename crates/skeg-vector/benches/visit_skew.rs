//! Per-node visit-count skew, hot/cold tier gate.
//!
//! Not a Criterion bench: a reporting harness (`harness = false`).
//!
//! Hypothesis: if the per-node visit count during the greedy walk is
//! Pareto-like, a hot set in RAM (~5% of nodes) + a cold set on disk shrinks
//! the tier. Gate: the top 5% of nodes covers >= 50% of visits. Plus a
//! cross-query validation: does the hot set identified on half the queries
//! recognize the visits of the other half?
//!
//! Reuses `DiskVamanaIndex::search_node_trace`, the already-exposed walk traces.

#![allow(clippy::cast_precision_loss)]

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use skeg_vector::{DiskVamanaIndex, VamanaConfig, VamanaIndex};

const CORPUS_NPY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../bench-compare/embeddings_cache/corpus_mxbai-embed-large_10000.npy"
);
const QUERY_NPY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../bench-compare/embeddings_cache/queries_mxbai-embed-large_200.npy"
);

fn load_npy(path: &str) -> Option<(Vec<f32>, usize, usize)> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        return None;
    }
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + header_len]).ok()?;
    let sh = header.find("'shape':")?;
    let lp = header[sh..].find('(')? + sh + 1;
    let rp = header[lp..].find(')')? + lp;
    let dims: Vec<usize> = header[lp..rp]
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if dims.len() != 2 {
        return None;
    }
    let data: Vec<f32> = bytes[10 + header_len..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Some((data, dims[0], dims[1]))
}

/// One standard-normal sample (Box-Muller).
fn gaussian(rng: &mut StdRng) -> f32 {
    let u1 = rng.random::<f32>().max(1e-9);
    let u2 = rng.random::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// `n` vectors uniform on the unit sphere.
fn uniform_sphere(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n * dim);
    for _ in 0..n {
        let mut v: Vec<f32> = (0..dim).map(|_| gaussian(&mut rng)).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for x in &mut v {
            *x /= norm;
        }
        out.extend(v);
    }
    out
}

/// Visit count per node accumulated over `traces`.
fn visit_counts(n: usize, traces: &[Vec<u32>]) -> Vec<u64> {
    let mut visit = vec![0u64; n];
    for tr in traces {
        for &node in tr {
            visit[node as usize] += 1;
        }
    }
    visit
}

/// Print the CDF (top X% nodes -> Y% visits) and the gate verdict.
fn cdf_report(label: &str, n: usize, traces: &[Vec<u32>]) {
    let visit = visit_counts(n, traces);
    let total: u64 = visit.iter().sum();
    let walks = traces.len();
    let avg = traces.iter().map(Vec::len).sum::<usize>() as f64 / walks as f64;

    let mut sorted = visit.clone();
    sorted.sort_unstable_by(|a, b| b.cmp(a));

    println!("\n== {label} (N={n}, {walks} queries, walk avg {avg:.0} nodes) ==");
    println!(
        "  total visits {total}  visited {} distinct nodes",
        visit.iter().filter(|&&v| v > 0).count()
    );
    println!("  {:>10}{:>14}", "top nodes %", "coverage %");
    let mut top5_cov = 0.0;
    for &pct in &[1.0f64, 5.0, 10.0, 25.0, 50.0] {
        let k = ((n as f64 * pct / 100.0).round() as usize).max(1);
        let cov = sorted[..k].iter().sum::<u64>() as f64 / total as f64 * 100.0;
        if (pct - 5.0).abs() < 1e-9 {
            top5_cov = cov;
        }
        println!("  {pct:>9.0}%{cov:>13.1}%");
    }
    // gate
    let top1 = sorted[..(n / 100).max(1)].iter().sum::<u64>() as f64 / total as f64 * 100.0;
    let verdict = if top1 >= 30.0 {
        "EXTREME skew -> hot/cold ideal"
    } else if top5_cov >= 50.0 {
        "STRONG skew -> hot/cold effective"
    } else {
        let top10 = sorted[..(n / 10).max(1)].iter().sum::<u64>() as f64 / total as f64 * 100.0;
        if top10 >= 60.0 {
            "moderate skew -> hot/cold limited"
        } else {
            "UNIFORM -> hot/cold dead"
        }
    };
    println!("  gate: {verdict}");
}

/// Cross-query validation: train the hot set (top 5%) on the first half of the
/// traces, measure what fraction of the second half's visits it covers.
fn cross_query(label: &str, n: usize, traces: &[Vec<u32>]) {
    let mid = traces.len() / 2;
    if mid == 0 {
        return;
    }
    let visit_a = visit_counts(n, &traces[..mid]);
    let hot_k = (n / 20).max(1); // top 5%
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_unstable_by(|&a, &b| visit_a[b].cmp(&visit_a[a]));
    let mut is_hot = vec![false; n];
    for &node in &idx[..hot_k] {
        is_hot[node] = true;
    }

    let mut b_total = 0u64;
    let mut b_in_hot = 0u64;
    for tr in &traces[mid..] {
        for &node in tr {
            b_total += 1;
            if is_hot[node as usize] {
                b_in_hot += 1;
            }
        }
    }
    // self-coverage on the training half, for reference
    let mut a_total = 0u64;
    let mut a_in_hot = 0u64;
    for tr in &traces[..mid] {
        for &node in tr {
            a_total += 1;
            if is_hot[node as usize] {
                a_in_hot += 1;
            }
        }
    }
    let cov_a = a_in_hot as f64 / a_total.max(1) as f64 * 100.0;
    let cov_b = b_in_hot as f64 / b_total.max(1) as f64 * 100.0;
    println!("  cross-query ({label}): hot set 5% on half A covers A {cov_a:.1}% / B {cov_b:.1}%");
}

/// Build the graph, trace the walks, run the CDF + cross-query gate.
fn run(label: &str, corpus: Vec<f32>, n: usize, dim: usize, queries: &[f32], n_q: usize) {
    let ids: Vec<u64> = (0..n as u64).collect();
    let index = VamanaIndex::build(corpus, ids, dim, &VamanaConfig::default());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    index.save(tmp.path()).expect("save");
    drop(index);
    let disk = DiskVamanaIndex::open(tmp.path()).expect("open");

    let traces: Vec<Vec<u32>> = (0..n_q)
        .map(|qi| {
            disk.search_node_trace(&queries[qi * dim..(qi + 1) * dim])
                .expect("trace")
        })
        .collect();

    cdf_report(label, n, &traces);
    cross_query(label, n, &traces);
}

fn main() {
    eprintln!("Phase 0.d - visit count skew, hot/cold gate\n");

    // Real mxbai-embed-large 10K with real queries.
    if let (Some((corpus, n, dim)), Some((queries, n_q, q_dim))) =
        (load_npy(CORPUS_NPY), load_npy(QUERY_NPY))
    {
        if dim == q_dim {
            run("mxbai real 10K", corpus, n, dim, &queries, n_q);
        }
    } else {
        eprintln!("  (mxbai npy not found, skipping the real case)");
    }

    // uniform-sphere at scale: the validated proxy for real embeddings.
    let n = 100_000usize;
    let dim = 1024;
    let corpus = uniform_sphere(n, dim, 7);
    let queries = uniform_sphere(200, dim, 99);
    run(
        "uniform-sphere 100K (real-scale proxy)",
        corpus,
        n,
        dim,
        &queries,
        200,
    );
}
