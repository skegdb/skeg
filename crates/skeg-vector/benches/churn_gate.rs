//! Delete-churn gate: steady-state update workloads (delete + insert pairs).
//!
//! A workload that continuously replaces vectors (streaming updates, TTL'd
//! entries, supersede-style writes) keeps the delta small on its own: deletes
//! cancel inserts, so the delta-size consolidate trigger never fires. But every
//! time the delta does touch the FLUSH threshold, a navigable run is born, and
//! nothing folds runs back: they accumulate, and every query pays one extra
//! graph walk per run.
//!
//! This gate measures that accumulation and its query cost. Per config:
//!   1. seed `n` vectors, consolidate (clean baseline)
//!   2. churn one full turnover (insert successor, delete a random live id)
//!      with only the engine's own triggers active
//!   3. report run count + delta at steady state, query p50/p99 there
//!   4. force one consolidate, report the stall and the clean query p50/p99
//!
//! Pass: q@churn p50 <= 2x q@clean p50 (same bar as incremental_gate) with
//! recall preserved (checked by the recall benches, not here).
//!
//! Env: SKEG_CHURN_DIMS, SKEG_CHURN_NS, SKEG_CHURN_TIERS (csv overrides).
//! Run: cargo bench -p skeg-vector --bench churn_gate

use std::time::Instant;

use skeg_vector::{DiskVamanaIndex, QuantKind};

const CONSOLIDATE_MIN: usize = 4096; // mirror the server schedule
const N_QUERIES: usize = 200;
const K: usize = 10;

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

fn unit_vec(dim: usize, s: &mut u64) -> Vec<f32> {
    let mut v = vec![0f32; dim];
    for x in v.iter_mut() {
        *x = (xorshift(s) >> 11) as f32 / (1u64 << 53) as f32 - 0.5;
    }
    let norm = v.iter().map(|a| a * a).sum::<f32>().sqrt().max(1e-9);
    for x in v.iter_mut() {
        *x /= norm;
    }
    v
}

fn pctl(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 * p) as usize).min(sorted.len() - 1);
    sorted[idx]
}

fn query_lat(idx: &DiskVamanaIndex, dim: usize, s: &mut u64) -> (f64, f64) {
    let mut lat = Vec::with_capacity(N_QUERIES);
    for _ in 0..N_QUERIES {
        let q = unit_vec(dim, s);
        let t = Instant::now();
        let _ = idx.search(&q, K).unwrap();
        lat.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (pctl(&lat, 0.50), pctl(&lat, 0.99))
}

/// Fold trigger for the background mode: begin a background consolidate once
/// this many runs have accumulated.
const MAX_RUNS: usize = 4;

fn run(bits: u8, dim: usize, n: usize, background: bool) {
    let root = std::env::temp_dir().join(format!("skeg_churn_{dim}_{n}_tq{bits}_{background}"));
    let _ = std::fs::remove_dir_all(&root);
    let mut idx =
        DiskVamanaIndex::create_empty_with_tier(&root, dim, 200, QuantKind::TurboQuant { bits })
            .unwrap();

    let mut s = 0xD1B5_4A32_D192_ED03u64 ^ (u64::from(bits) << 40) ^ (dim as u64) ^ (n as u64);
    let mut next_id: u64 = 0;

    let mut ids: Vec<u64> = Vec::with_capacity(n);
    for _ in 0..n {
        idx.insert(next_id, &unit_vec(dim, &mut s)).unwrap();
        ids.push(next_id);
        next_id += 1;
    }
    idx.consolidate().unwrap();

    // churn: one full turnover
    let mut job: Option<std::thread::JoinHandle<std::io::Result<skeg_vector::ConsolidateBuilt>>> =
        None;
    let mut max_pause_ms = 0f64; // longest synchronous begin/finish pause
    let mut folds = 0usize;
    let tc = Instant::now();
    for _ in 0..n {
        idx.insert(next_id, &unit_vec(dim, &mut s)).unwrap();
        let succ = next_id;
        next_id += 1;
        let j = (xorshift(&mut s) as usize) % ids.len();
        idx.delete(ids[j]).unwrap();
        ids[j] = succ;
        if background {
            // Kick a background fold when runs pile up; land it when ready.
            if job.is_none() && idx.run_count() >= MAX_RUNS {
                let t = Instant::now();
                if let Some(j) = idx.consolidate_begin().unwrap() {
                    max_pause_ms = max_pause_ms.max(t.elapsed().as_secs_f64() * 1000.0);
                    let dir = root.clone();
                    job = Some(std::thread::spawn(move || j.build(&dir)));
                }
            }
            if job
                .as_ref()
                .is_some_and(std::thread::JoinHandle::is_finished)
            {
                let built = job.take().unwrap().join().unwrap().unwrap();
                let t = Instant::now();
                idx.consolidate_finish(built).unwrap();
                max_pause_ms = max_pause_ms.max(t.elapsed().as_secs_f64() * 1000.0);
                folds += 1;
            }
        } else if idx.delta_len() >= idx.main_len().max(CONSOLIDATE_MIN) {
            idx.consolidate().unwrap();
        }
    }
    // Land an in-flight fold before measuring steady state.
    if let Some(h) = job.take() {
        let built = h.join().unwrap().unwrap();
        let t = Instant::now();
        idx.consolidate_finish(built).unwrap();
        max_pause_ms = max_pause_ms.max(t.elapsed().as_secs_f64() * 1000.0);
        folds += 1;
    }
    let churn_s = tc.elapsed().as_secs_f64();
    let (runs_ss, delta_ss) = (idx.run_count(), idx.delta_len());

    let (qd_p50, qd_p99) = query_lat(&idx, dim, &mut s);

    let tk = Instant::now();
    idx.consolidate().unwrap();
    let cons_s = tk.elapsed().as_secs_f64();
    let (qc_p50, qc_p99) = query_lat(&idx, dim, &mut s);

    let ratio = qd_p50 / qc_p50.max(1e-9);
    let verdict = if ratio <= 2.0 { "PASS" } else { "FAIL" };
    let mode = if background { "bg " } else { "inl" };
    println!(
        "dim={dim:<4} n={n:<6} tq{bits} {mode} | churn {:>5.0}/s | runs={runs_ss:<3} delta={delta_ss:<5} folds={folds} max_pause={max_pause_ms:>6.1}ms | q@churn p50={qd_p50:>6.3} p99={qd_p99:>6.3} | q@clean p50={qc_p50:>6.3} p99={qc_p99:>6.3} | ratio {ratio:>4.1}x {verdict} | final consolidate {cons_s:>5.1}s",
        n as f64 / churn_s,
    );

    let _ = std::fs::remove_dir_all(&root);
}

fn csv<T: std::str::FromStr + Clone>(key: &str, default: &[T]) -> Vec<T> {
    match std::env::var(key) {
        Ok(v) => v.split(',').filter_map(|x| x.trim().parse().ok()).collect(),
        Err(_) => default.to_vec(),
    }
}

fn main() {
    let dims: Vec<usize> = csv("SKEG_CHURN_DIMS", &[768usize, 1024]);
    let ns: Vec<usize> = csv("SKEG_CHURN_NS", &[10_000usize, 100_000]);
    let tiers: Vec<u8> = csv("SKEG_CHURN_TIERS", &[2u8]);
    println!("# churn gate: dims={dims:?} ns={ns:?} tiers={tiers:?}");
    for &dim in &dims {
        for &n in &ns {
            for &bits in &tiers {
                run(bits, dim, n, false);
                run(bits, dim, n, true);
            }
        }
    }
}
