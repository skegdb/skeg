//! Scaling ceiling of the background consolidate under SUSTAINED churn.
//!
//! For each live-set size, run several full turnovers of retract churn while the
//! background begin/build/finish fold runs continuously (one in flight). The
//! question: does the fold keep the run count bounded as the live set grows, or
//! does the O(live-set) rebuild fall behind so runs pile up (query cost) and/or
//! retract throughput collapses?
//!
//! Reports per size: sustained retract/s, the max run count seen, folds done,
//! the average fold build time, and the fraction of wall time a fold was in
//! flight (~1.0 means the fold is the bottleneck, running back to back).
//!
//! Env: SKEG_SIZES (csv), SKEG_TURNS, SKEG_DIM, SKEG_MAXRUNS, SKEG_BITS.
//! Run: cargo bench -p skeg-vector --bench scale_churn

#![allow(clippy::explicit_counter_loop)] // `next` is an id generator, not an index

use std::time::Instant;

use skeg_vector::{ConsolidateBuilt, DiskVamanaIndex, QuantKind, RunMergeBuilt};

/// L3 reclaim A/B: build a base of `n`, delete `del_frac` of it, then time the
/// two ways to reclaim the dead rows - a full consolidate (O(live) greedy
/// rebuild) vs a delete-patch (O(deleted) in-place prune). Two independent
/// indices built from the same seed so the workloads are identical.
fn run_l3(n: usize, del_frac: f64, dim: usize, tier: QuantKind) {
    let build = |tag: &str| -> (DiskVamanaIndex, Vec<u64>) {
        let dir = std::env::temp_dir().join(format!("skeg_l3_{tag}_{n}"));
        let _ = std::fs::remove_dir_all(&dir);
        let mut idx = DiskVamanaIndex::create_empty_with_tier(&dir, dim, 300, tier).unwrap();
        let mut s = 0xD00D_1234u64 ^ (n as u64);
        for i in 0..n as u64 {
            idx.insert(i, &uvec(dim, &mut s)).unwrap();
        }
        idx.consolidate().unwrap();
        // Deterministic delete set (same for both indices).
        let d = (n as f64 * del_frac) as usize;
        let mut r = 0xBEEF_5678u64 ^ (n as u64);
        let mut deleted = Vec::with_capacity(d);
        for _ in 0..d {
            let id = (xs(&mut r) as usize % n) as u64;
            idx.delete(id).unwrap();
            deleted.push(id);
        }
        (idx, deleted)
    };

    let (mut a, _) = build("cons");
    let t = Instant::now();
    a.consolidate().unwrap();
    let cons_s = t.elapsed().as_secs_f64();
    let base_a = a.main_len();
    let dir_a = std::env::temp_dir().join(format!("skeg_l3_cons_{n}"));
    drop(a);
    let _ = std::fs::remove_dir_all(&dir_a);

    let (mut b, _) = build("patch");
    let dir_b = std::env::temp_dir().join(format!("skeg_l3_patch_{n}"));
    let t = Instant::now();
    let patch_s = if let Some(job) = b.delete_patch_begin().unwrap() {
        let built = job.build(&dir_b).unwrap();
        b.delete_patch_finish(built).unwrap();
        t.elapsed().as_secs_f64()
    } else {
        0.0
    };
    let base_b = b.main_len();
    drop(b);
    let _ = std::fs::remove_dir_all(&dir_b);

    let speedup = if patch_s > 0.0 { cons_s / patch_s } else { 0.0 };
    println!(
        "| {n:>7} | {:>4.0} | {cons_s:>8.2} | {patch_s:>8.2} | {speedup:>6.1}x | {base_a:>7} | {base_b:>7} |",
        del_frac * 100.0,
    );
}

fn xs(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}
fn uvec(dim: usize, s: &mut u64) -> Vec<f32> {
    let mut v = vec![0f32; dim];
    for x in v.iter_mut() {
        *x = (xs(s) >> 11) as f32 / (1u64 << 53) as f32 - 0.5;
    }
    let n = v.iter().map(|a| a * a).sum::<f32>().sqrt().max(1e-9);
    for x in v.iter_mut() {
        *x /= n;
    }
    v
}

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}

fn run(n: usize, turns: usize, dim: usize, max_runs: usize, tier: QuantKind, mode: &str) {
    if mode == "l2" {
        return run_l2(n, turns, dim, max_runs, tier);
    }
    let dir = std::env::temp_dir().join(format!("skeg_scale_{n}"));
    let _ = std::fs::remove_dir_all(&dir);
    let mut idx = DiskVamanaIndex::create_empty_with_tier(&dir, dim, 300, tier).unwrap();
    let mut s = 0xC0FF_EE00_1234u64 ^ (n as u64);
    let mut ids: Vec<u64> = Vec::with_capacity(n);
    for i in 0..n as u64 {
        idx.insert(i, &uvec(dim, &mut s)).unwrap();
        ids.push(i);
    }
    idx.consolidate().unwrap();

    let ops = turns * n;
    let mut next = n as u64;
    let mut job: Option<std::thread::JoinHandle<std::io::Result<ConsolidateBuilt>>> = None;
    let mut fold_start = Instant::now();
    let mut max_runs_seen = 0usize;
    let mut folds = 0usize;
    let mut fold_busy = std::time::Duration::ZERO;
    let mut build_total = std::time::Duration::ZERO;

    let start = Instant::now();
    for _ in 0..ops {
        idx.insert(next, &uvec(dim, &mut s)).unwrap();
        let succ = next;
        next += 1;
        let j = (xs(&mut s) as usize) % ids.len();
        idx.delete(ids[j]).unwrap();
        ids[j] = succ;

        let rc = idx.run_count();
        max_runs_seen = max_runs_seen.max(rc);

        if job.is_none()
            && rc >= max_runs
            && let Some(jb) = idx.consolidate_begin().unwrap()
        {
            let d = dir.clone();
            fold_start = Instant::now();
            job = Some(std::thread::spawn(move || jb.build(&d)));
        }
        if job
            .as_ref()
            .is_some_and(std::thread::JoinHandle::is_finished)
        {
            let built = job.take().unwrap().join().unwrap().unwrap();
            idx.consolidate_finish(built).unwrap();
            let dt = fold_start.elapsed();
            fold_busy += dt;
            build_total += dt;
            folds += 1;
        }
    }
    if let Some(h) = job.take() {
        idx.consolidate_finish(h.join().unwrap().unwrap()).unwrap();
        let dt = fold_start.elapsed();
        fold_busy += dt;
        build_total += dt;
        folds += 1;
    }
    let elapsed = start.elapsed();

    let rps = ops as f64 / elapsed.as_secs_f64();
    let busy_frac = fold_busy.as_secs_f64() / elapsed.as_secs_f64();
    let avg_build = if folds > 0 {
        build_total.as_secs_f64() / folds as f64
    } else {
        0.0
    };
    println!(
        "| {n:>7} | {turns} | {rps:>7.0} | {max_runs_seen:>4} | {:>4} | {folds:>3} | {avg_build:>6.1} | {busy_frac:>5.2} |",
        idx.run_count(),
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Two-tier: frequent off-thread runs-merge (O(runs)) keeps the run count
/// bounded; a full base rebuild runs inline once per turnover to reclaim base
/// tombstones. The claim: retract/s decouples from the O(live-set) base cost.
fn run_l2(n: usize, turns: usize, dim: usize, max_runs: usize, tier: QuantKind) {
    let dir = std::env::temp_dir().join(format!("skeg_scale_l2_{n}"));
    let _ = std::fs::remove_dir_all(&dir);
    let mut idx = DiskVamanaIndex::create_empty_with_tier(&dir, dim, 300, tier).unwrap();
    let mut s = 0xC0FF_EE00_1234u64 ^ (n as u64);
    let mut ids: Vec<u64> = Vec::with_capacity(n);
    for i in 0..n as u64 {
        idx.insert(i, &uvec(dim, &mut s)).unwrap();
        ids.push(i);
    }
    idx.consolidate().unwrap();

    let ops = turns * n;
    let mut next = n as u64;
    // One background slot, holding EITHER a runs-merge or a base rebuild; the two
    // never overlap (base consolidate discards the runs a merge would touch).
    let mut merge_job: Option<std::thread::JoinHandle<std::io::Result<RunMergeBuilt>>> = None;
    let mut base_job: Option<std::thread::JoinHandle<std::io::Result<ConsolidateBuilt>>> = None;
    let mut job_start = Instant::now();
    let mut max_runs_seen = 0usize;
    let mut merges = 0usize;
    let mut merge_busy = std::time::Duration::ZERO;
    let mut merge_total = std::time::Duration::ZERO;
    let mut base_folds = 0usize;
    let mut base_busy = std::time::Duration::ZERO;
    let mut base_total = std::time::Duration::ZERO;
    let mut next_base = n; // op index at which the next base rebuild is due

    let start = Instant::now();
    for op in 0..ops {
        idx.insert(next, &uvec(dim, &mut s)).unwrap();
        let succ = next;
        next += 1;
        let j = (xs(&mut s) as usize) % ids.len();
        idx.delete(ids[j]).unwrap();
        ids[j] = succ;

        let rc = idx.run_count();
        max_runs_seen = max_runs_seen.max(rc);

        // Reap a finished background job.
        if base_job
            .as_ref()
            .is_some_and(std::thread::JoinHandle::is_finished)
        {
            let built = base_job.take().unwrap().join().unwrap().unwrap();
            idx.consolidate_finish(built).unwrap();
            let dt = job_start.elapsed();
            base_busy += dt;
            base_total += dt;
            base_folds += 1;
        } else if merge_job
            .as_ref()
            .is_some_and(std::thread::JoinHandle::is_finished)
        {
            let built = merge_job.take().unwrap().join().unwrap().unwrap();
            idx.merge_runs_finish(built).unwrap();
            let dt = job_start.elapsed();
            merge_busy += dt;
            merge_total += dt;
            merges += 1;
        }

        if base_job.is_none() && merge_job.is_none() {
            if op >= next_base {
                // Rare: base rebuild off-thread, reclaims base tombstones.
                if let Some(jb) = idx.consolidate_begin().unwrap() {
                    let d = dir.clone();
                    job_start = Instant::now();
                    base_job = Some(std::thread::spawn(move || jb.build(&d)));
                    next_base += n;
                }
            } else if rc >= max_runs {
                // Frequent: fold the runs off-thread, O(runs) not O(live).
                if let Some(jb) = idx.merge_runs_begin().unwrap() {
                    let d = dir.clone();
                    job_start = Instant::now();
                    merge_job = Some(std::thread::spawn(move || jb.build(&d)));
                }
            }
        }
    }
    if let Some(h) = base_job.take() {
        idx.consolidate_finish(h.join().unwrap().unwrap()).unwrap();
        let dt = job_start.elapsed();
        base_busy += dt;
        base_total += dt;
        base_folds += 1;
    }
    if let Some(h) = merge_job.take() {
        idx.merge_runs_finish(h.join().unwrap().unwrap()).unwrap();
        let dt = job_start.elapsed();
        merge_busy += dt;
        merge_total += dt;
        merges += 1;
    }
    let elapsed = start.elapsed();

    let rps = ops as f64 / elapsed.as_secs_f64();
    let busy_frac = (merge_busy + base_busy).as_secs_f64() / elapsed.as_secs_f64();
    let avg_merge = if merges > 0 {
        merge_total.as_secs_f64() / merges as f64
    } else {
        0.0
    };
    let avg_base = if base_folds > 0 {
        base_total.as_secs_f64() / base_folds as f64
    } else {
        0.0
    };
    println!(
        "| {n:>7} | {turns} | {rps:>7.0} | {max_runs_seen:>4} | {:>4} | {merges:>3}/{base_folds} | {avg_merge:>5.2}/{avg_base:>4.1} | {busy_frac:>5.2} |",
        idx.run_count(),
    );
    let _ = std::fs::remove_dir_all(&dir);
}

fn main() {
    let sizes: Vec<usize> = std::env::var("SKEG_SIZES")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![25_000usize, 50_000, 100_000]);
    let turns: usize = env("SKEG_TURNS", 3);
    let dim: usize = env("SKEG_DIM", 768);
    let max_runs: usize = env("SKEG_MAXRUNS", 4);
    let bits: u8 = env("SKEG_BITS", 2);
    let tier = QuantKind::TurboQuant { bits };
    let mode = std::env::var("SKEG_MODE").unwrap_or_else(|_| "full".into());
    if mode == "l3" {
        let del: f64 = env("SKEG_DELFRAC", 0.4);
        println!(
            "# scale_churn L3 reclaim: dim={dim} tq{bits} | consolidate O(live) vs delete-patch O(deleted)"
        );
        println!(
            "# del% deleted before reclaim; cons/patch = reclaim wall seconds; base = rows after"
        );
        println!("| live    | del% | cons_s   | patch_s  | speedup | base_c  | base_p  |");
        println!("|---------|-----:|---------:|---------:|--------:|--------:|--------:|");
        for &n in &sizes {
            run_l3(n, del, dim, tier);
        }
        return;
    }
    println!("# scale_churn: mode={mode} turns={turns} dim={dim} max_runs={max_runs} tq{bits}");
    println!("# retract/s = sustained; maxR = peak run count (bounded => fold keeps pace);");
    if mode == "l2" {
        println!("# folds = runsMerges/baseFolds; build_s = avgMerge/avgBase (s)");
    }
    println!(
        "# busy = fraction of wall time a (runs-)fold was in flight (~1.0 => fold is the bottleneck)"
    );
    println!("| live    | T | retr/s | maxR | endR | folds | build_s | busy |");
    println!("|---------|---|-------:|-----:|-----:|------:|--------:|-----:|");
    for &n in &sizes {
        run(n, turns, dim, max_runs, tier, &mode);
    }
}
