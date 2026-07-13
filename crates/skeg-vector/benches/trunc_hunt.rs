//! Truncation hunt: after a churned build, verify every vectors.bin on disk is
//! exactly `64 + n*dim*4` bytes (n/dim read from its own header). Isolates
//! whether the short-write happens in the STANDARD flush/inline-consolidate path
//! or the background begin/build/finish path.
//!
//! Env: SKEG_MODE (inline|bg), SKEG_ITERS, SKEG_BENCH_N, SKEG_CHURN, SKEG_DIM,
//!      SKEG_MAXRUNS, SKEG_FSYNC (1 = the candidate fix is compiled in, see note).
//! Run: SKEG_MODE=inline SKEG_ITERS=30 cargo bench -p skeg-vector --bench trunc_hunt

use std::path::Path;

use skeg_vector::{ConsolidateBuilt, DiskVamanaIndex, QuantKind};

const HEADER_LEN: u64 = 64;

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

/// Read n and dim from a vectors.bin header and compare the file length.
fn check_vbin(path: &Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < 16 {
        return Some(format!(
            "{}: {} bytes (< header)",
            path.display(),
            bytes.len()
        ));
    }
    let n = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as u64;
    let dim = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as u64;
    let expect = HEADER_LEN + n * dim * 4;
    let actual = std::fs::metadata(path).ok()?.len();
    (actual != expect).then(|| {
        format!(
            "{}: n={n} dim={dim} expect={expect} actual={actual} (short by {})",
            path.display(),
            expect.saturating_sub(actual)
        )
    })
}

/// Every vectors.bin under the index dir: the base plus each run-*.
fn check_dir(dir: &Path) -> Vec<String> {
    let mut bad = Vec::new();
    if let Some(m) = check_vbin(&dir.join("vectors.bin")) {
        bad.push(format!("BASE {m}"));
    }
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir()
                && p.file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with("run-"))
            {
                if let Some(m) = check_vbin(&p.join("vectors.bin")) {
                    bad.push(format!("RUN  {m}"));
                }
            }
        }
    }
    bad
}

fn build_churned(
    background: bool,
    max_runs: usize,
    n: usize,
    churn: usize,
    dim: usize,
    tier: QuantKind,
    dir: &Path,
) {
    let _ = std::fs::remove_dir_all(dir);
    let mut idx = DiskVamanaIndex::create_empty_with_tier(dir, dim, 300, tier).unwrap();
    let mut s = 0xDEAD_BEEF_1234u64 ^ (dim as u64);
    let mut ids: Vec<u64> = Vec::with_capacity(n);
    for i in 0..n as u64 {
        idx.insert(i, &uvec(dim, &mut s)).unwrap();
        ids.push(i);
    }
    idx.consolidate().unwrap();
    let mut next = n as u64;
    let mut job: Option<std::thread::JoinHandle<std::io::Result<ConsolidateBuilt>>> = None;
    for _ in 0..churn {
        idx.insert(next, &uvec(dim, &mut s)).unwrap();
        let succ = next;
        next += 1;
        let j = (xs(&mut s) as usize) % ids.len();
        idx.delete(ids[j]).unwrap();
        ids[j] = succ;
        if background {
            if job.is_none() && idx.run_count() >= max_runs {
                if let Some(jb) = idx.consolidate_begin().unwrap() {
                    let d = dir.to_path_buf();
                    job = Some(std::thread::spawn(move || jb.build(&d)));
                }
            }
            if job
                .as_ref()
                .is_some_and(std::thread::JoinHandle::is_finished)
            {
                idx.consolidate_finish(job.take().unwrap().join().unwrap().unwrap())
                    .unwrap();
            }
        } else if idx.delta_len() >= idx.main_len().max(4096) {
            idx.consolidate().unwrap();
        }
    }
    if let Some(h) = job.take() {
        idx.consolidate_finish(h.join().unwrap().unwrap()).unwrap();
    }
    // keep idx alive until after the caller checks files
    std::mem::forget(idx);
}

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}

fn main() {
    let mode = std::env::var("SKEG_MODE").unwrap_or_else(|_| "inline".into());
    let bg = mode == "bg";
    let iters: usize = env("SKEG_ITERS", 20);
    let n: usize = env("SKEG_BENCH_N", 8000);
    let churn: usize = env("SKEG_CHURN", 32000);
    let dim: usize = env("SKEG_DIM", 1024);
    let max_runs: usize = env("SKEG_MAXRUNS", 3);
    let tier = QuantKind::TurboQuant { bits: 2 };
    let dir = std::env::temp_dir().join(format!("skeg_trunc_{mode}"));
    println!("# trunc_hunt mode={mode} iters={iters} n={n} churn={churn} dim={dim}");
    let mut truncations = 0usize;
    for it in 0..iters {
        build_churned(bg, max_runs, n, churn, dim, tier, &dir);
        let bad = check_dir(&dir);
        if !bad.is_empty() {
            truncations += 1;
            println!("iter {it}: TRUNCATED");
            for b in &bad {
                println!("    {b}");
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    println!("=== mode={mode}: {truncations}/{iters} builds produced a truncated vectors.bin ===");
}
