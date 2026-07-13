//! RW-under-churn recall diagnostic. Real embeddings, live index continuously
//! updated (insert + delete), recall measured against the brute-force truth of
//! the CURRENT LIVE SET at the churned state.
//!
//! Three indices over the SAME operation history and the SAME final live set:
//!   gold   - the final live set built fresh and consolidated once (the ceiling)
//!   inline - churn + the server's inline delta-size consolidate
//!   bg     - churn + the background begin/build/finish consolidate (new path)
//! For each, sweep the query budget (l_search) to separate a graph-quality loss
//! (does not recover with budget) from a budget artifact (does).
//!
//! Env: SKEG_BENCH_N (seed live), SKEG_CHURN, SKEG_NQ, SKEG_DIM, SKEG_BITS,
//!      SKEG_LS (csv l_search sweep), SKEG_L_BUILD (build width), SKEG_MAXRUNS
//!      (bg fold trigger), SKEG_CORPUS, SKEG_QUERY.
//! Run: cargo bench -p skeg-vector --bench churn_recall

use std::time::Instant;

use skeg_vector::{DiskVamanaIndex, QuantKind};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");

fn load_npy(path: &str, cap: usize, pad: usize) -> (Vec<f32>, usize) {
    let bytes = std::fs::read(path).unwrap_or_else(|_| panic!("read {path}"));
    let hlen = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + hlen]).unwrap();
    let sh = header.find("'shape':").unwrap();
    let lp = header[sh..].find('(').unwrap() + sh + 1;
    let rp = header[lp..].find(')').unwrap() + lp;
    let dims: Vec<usize> = header[lp..rp]
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let (rows, native) = (dims[0].min(cap), dims[1]);
    let start = 10 + hlen;
    let mut out = vec![0f32; rows * pad];
    for r in 0..rows {
        for c in 0..native {
            let off = start + (r * native + c) * 4;
            out[r * pad + c] =
                f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
        }
    }
    for row in out.chunks_exact_mut(pad) {
        let nrm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for x in row.iter_mut() {
            *x /= nrm;
        }
    }
    (out, rows)
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// (successor_id, successor_src_row, victim_id) per churn step, plus the corpus
/// rows that are live at the end. id == corpus row, so pool must exceed n+churn.
fn gen_ops(n: usize, churn: usize) -> (Vec<(u64, u64)>, Vec<u64>) {
    let mut ops = Vec::with_capacity(churn);
    let mut ids: Vec<u64> = (0..n as u64).collect();
    let mut s = 0x1234_5678_9ABC_DEF0u64;
    for t in 0..churn {
        let succ = (n + t) as u64;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let j = (s as usize) % ids.len();
        let victim = ids[j];
        ops.push((succ, victim));
        ids[j] = succ;
    }
    (ops, ids)
}

fn truth(
    live_rows: &[u64],
    corpus: &[f32],
    dim: usize,
    queries: &[f32],
    k: usize,
) -> Vec<Vec<u64>> {
    queries
        .chunks_exact(dim)
        .map(|q| {
            let mut sc: Vec<(f32, u64)> = live_rows
                .iter()
                .map(|&r| (cos(q, &corpus[r as usize * dim..(r as usize + 1) * dim]), r))
                .collect();
            sc.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            sc.iter().take(k).map(|(_, id)| *id).collect()
        })
        .collect()
}

fn recall(got: &[(u64, f32)], want: &[u64], k: usize) -> f64 {
    let g: std::collections::HashSet<u64> = got.iter().take(k).map(|(id, _)| *id).collect();
    want.iter().take(k).filter(|id| g.contains(id)).count() as f64 / k.min(want.len()) as f64
}

fn vat(corpus: &[f32], row: u64, dim: usize) -> Vec<f32> {
    corpus[row as usize * dim..(row as usize + 1) * dim].to_vec()
}

fn build_gold(
    live_rows: &[u64],
    corpus: &[f32],
    dim: usize,
    tier: QuantKind,
    tag: &str,
) -> DiskVamanaIndex {
    let dir = std::env::temp_dir().join(format!("skeg_cr_gold_{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    let mut idx = DiskVamanaIndex::create_empty_with_tier(&dir, dim, 300, tier).unwrap();
    for &r in live_rows {
        idx.insert(r, &vat(corpus, r, dim)).unwrap();
    }
    idx.consolidate().unwrap();
    idx
}

fn build_churned(
    mode: &str,
    max_runs: usize,
    n: usize,
    ops: &[(u64, u64)],
    corpus: &[f32],
    dim: usize,
    tier: QuantKind,
    tag: &str,
) -> DiskVamanaIndex {
    let dir = std::env::temp_dir().join(format!("skeg_cr_{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    let mut idx = DiskVamanaIndex::create_empty_with_tier(&dir, dim, 300, tier).unwrap();
    for r in 0..n as u64 {
        idx.insert(r, &vat(corpus, r, dim)).unwrap();
    }
    idx.consolidate().unwrap();
    let mut base_job: Option<
        std::thread::JoinHandle<std::io::Result<skeg_vector::ConsolidateBuilt>>,
    > = None;
    let mut merge_job: Option<std::thread::JoinHandle<std::io::Result<skeg_vector::RunMergeBuilt>>> =
        None;
    let mut next_base = n; // op index for the next rare base rebuild (l2)
    for (op, &(succ, victim)) in ops.iter().enumerate() {
        idx.insert(succ, &vat(corpus, succ, dim)).unwrap();
        idx.delete(victim).unwrap();
        match mode {
            "bg" => {
                if base_job.is_none() && idx.run_count() >= max_runs {
                    if let Some(jb) = idx.consolidate_begin().unwrap() {
                        let d = dir.clone();
                        base_job = Some(std::thread::spawn(move || jb.build(&d)));
                    }
                }
                if base_job
                    .as_ref()
                    .is_some_and(std::thread::JoinHandle::is_finished)
                {
                    let built = base_job.take().unwrap().join().unwrap().unwrap();
                    idx.consolidate_finish(built).unwrap();
                }
            }
            "l2" => {
                // Two-tier: frequent runs-merge, rare base rebuild, one slot, off-thread.
                if base_job
                    .as_ref()
                    .is_some_and(std::thread::JoinHandle::is_finished)
                {
                    let built = base_job.take().unwrap().join().unwrap().unwrap();
                    idx.consolidate_finish(built).unwrap();
                } else if merge_job
                    .as_ref()
                    .is_some_and(std::thread::JoinHandle::is_finished)
                {
                    let built = merge_job.take().unwrap().join().unwrap().unwrap();
                    idx.merge_runs_finish(built).unwrap();
                }
                if base_job.is_none() && merge_job.is_none() {
                    if op >= next_base {
                        if let Some(jb) = idx.consolidate_begin().unwrap() {
                            let d = dir.clone();
                            base_job = Some(std::thread::spawn(move || jb.build(&d)));
                            next_base += n;
                        }
                    } else if idx.run_count() >= max_runs {
                        if let Some(jb) = idx.merge_runs_begin().unwrap() {
                            let d = dir.clone();
                            merge_job = Some(std::thread::spawn(move || jb.build(&d)));
                        }
                    }
                }
            }
            _ => {
                // "inline": the server's delta-size-triggered consolidate.
                if idx.delta_len() >= idx.main_len().max(4096) {
                    idx.consolidate().unwrap();
                }
            }
        }
    }
    if let Some(h) = base_job.take() {
        idx.consolidate_finish(h.join().unwrap().unwrap()).unwrap();
    }
    if let Some(h) = merge_job.take() {
        idx.merge_runs_finish(h.join().unwrap().unwrap()).unwrap();
    }
    idx
}

fn measure(
    idx: &DiskVamanaIndex,
    queries: &[f32],
    dim: usize,
    nq: usize,
    t10: &[Vec<u64>],
    t100: &[Vec<u64>],
    ls: usize,
) -> (f64, f64, f64, f64) {
    let (mut r10, mut r100) = (0.0, 0.0);
    let mut lat = Vec::with_capacity(nq);
    for (qi, q) in queries.chunks_exact(dim).enumerate().take(nq) {
        let t = Instant::now();
        let got = idx.search_with_params(q, 100, ls, ls).unwrap();
        lat.push(t.elapsed().as_secs_f64() * 1000.0);
        r10 += recall(&got, &t10[qi], 10);
        r100 += recall(&got, &t100[qi], 100);
    }
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (
        r10 / nq as f64,
        r100 / nq as f64,
        lat[lat.len() / 2],
        lat[lat.len() * 99 / 100],
    )
}

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}

fn main() {
    let n: usize = env("SKEG_BENCH_N", 40_000);
    let churn: usize = env("SKEG_CHURN", 40_000);
    let nq: usize = env("SKEG_NQ", 200);
    let dim_native: usize = env("SKEG_DIM", 1024);
    let bits: u8 = env("SKEG_BITS", 2);
    let max_runs: usize = env("SKEG_MAXRUNS", 4);
    let ls_sweep: Vec<usize> = std::env::var("SKEG_LS")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![300usize, 600, 1200]);
    let dim = dim_native.next_multiple_of(8);
    let tier = QuantKind::TurboQuant { bits };
    let name = std::env::var("SKEG_NAME").unwrap_or_else(|_| "wiki".into());
    let cpath = std::env::var("SKEG_CORPUS")
        .unwrap_or_else(|_| format!("{ROOT}/hf_data/wiki-cohere_corpus.npy"));
    let qpath = std::env::var("SKEG_QUERY")
        .unwrap_or_else(|_| format!("{ROOT}/hf_data/wiki-cohere_queries.npy"));

    let (corpus, rows) = load_npy(&cpath, n + churn + 8, dim);
    assert!(
        rows >= n + churn,
        "need {} real vecs, have {rows}",
        n + churn
    );
    let (queries, _) = load_npy(&qpath, nq, dim);

    let (ops, live_rows) = gen_ops(n, churn);
    let t10 = truth(&live_rows, &corpus, dim, &queries, 10);
    let t100 = truth(&live_rows, &corpus, dim, &queries, 100);

    let l_build: usize = env("SKEG_L_BUILD", 48);
    println!(
        "\n### {name} dim={dim} tq{bits} | seed={n} churn={churn} live={} | l_build={l_build} max_runs={max_runs} | {nq} queries",
        live_rows.len()
    );
    println!("| mode   | l_search | runs | recall@10 | recall@100 | p50 ms | p99 ms |");
    println!("|--------|---------:|-----:|----------:|-----------:|-------:|-------:|");

    let modes: Vec<(&str, DiskVamanaIndex)> = vec![
        ("gold", build_gold(&live_rows, &corpus, dim, tier, "g")),
        (
            "inline",
            build_churned("inline", max_runs, n, &ops, &corpus, dim, tier, "i"),
        ),
        (
            "bg",
            build_churned("bg", max_runs, n, &ops, &corpus, dim, tier, "b"),
        ),
        (
            "l2",
            build_churned("l2", max_runs, n, &ops, &corpus, dim, tier, "l"),
        ),
    ];
    for (label, idx) in &modes {
        let runs = idx.run_count();
        for &ls in &ls_sweep {
            let (r10, r100, p50, p99) = measure(idx, &queries, dim, nq, &t10, &t100, ls);
            println!(
                "| {label:<6} | {ls:>8} | {runs:>4} | {r10:>9.4} | {r100:>10.4} | {p50:>6.2} | {p99:>6.2} |"
            );
        }
    }
    for (_, idx) in modes {
        drop(idx);
    }
    for tag in ["g", "i", "b", "l"] {
        let _ = std::fs::remove_dir_all(std::env::temp_dir().join(format!("skeg_cr_gold_{tag}")));
        let _ = std::fs::remove_dir_all(std::env::temp_dir().join(format!("skeg_cr_{tag}")));
    }
}
