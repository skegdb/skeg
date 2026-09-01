//! Where the memory goes during a fold, attributed rather than guessed.
//!
//! The churn gate reports RSS peaking near 800 MB on a 100k x 1024 corpus and
//! settling back to tens of megabytes, so nothing is retained - but "not a
//! leak" is not an explanation. This walks one shard-sized index through the
//! phases and prints, at each one, what the engine says it holds against what
//! the process actually has.
//!
//!   cargo bench -p skeg-vector --bench ram_attribution
//!
//! Env: RAM_N (rows, default 12500 - one shard of the 100k gate), RAM_DIM.

use skeg_vector::{DiskVamanaIndex, QuantKind};

fn rss_mb() -> f64 {
    skeg_platform::rss_bytes() as f64 / (1024.0 * 1024.0)
}

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}

fn row(phase: &str, reported: usize, base_rss: f64) {
    println!(
        "{phase:<34} riportato {:>7.1} MiB   RSS {:>7.1} MiB   delta {:>7.1} MiB",
        reported as f64 / (1024.0 * 1024.0),
        rss_mb(),
        rss_mb() - base_rss
    );
}

fn main() {
    let n: u64 = env("RAM_N", 12_500);
    let dim: usize = env("RAM_DIM", 1024);
    let tier = QuantKind::TurboQuant { bits: 2 };
    let dir = std::env::temp_dir().join("skeg_ram_attribution");
    let _ = std::fs::remove_dir_all(&dir);

    let base_rss = rss_mb();
    println!(
        "# {n} righe x {dim} dim, f32 grezzi = {:.0} MiB",
        (n as f64 * dim as f64 * 4.0) / (1024.0 * 1024.0)
    );
    println!("# RSS di partenza {base_rss:.1} MiB\n");

    let mut idx = DiskVamanaIndex::create_empty_with_tier(&dir, dim, 64, tier).unwrap();
    idx.set_auto_flush(false);
    let mut v = vec![0f32; dim];
    for id in 0..n {
        for (i, x) in v.iter_mut().enumerate() {
            *x = ((id as usize + i) % 97) as f32 / 97.0;
        }
        idx.insert(id, &v).unwrap();
    }
    row("delta pieno (nessun flush)", idx.resident_bytes(), base_rss);

    let job = idx.flush_begin().unwrap().unwrap();
    row(
        "flush_begin (delta -> staging)",
        idx.resident_bytes(),
        base_rss,
    );
    let built = job.build(&dir).unwrap();
    row("dopo build della run", idx.resident_bytes(), base_rss);
    idx.flush_finish(built).unwrap().expect_clean();
    row(
        "flush_finish (run al posto)",
        idx.resident_bytes(),
        base_rss,
    );

    let job = idx.consolidate_begin().unwrap().unwrap();
    row("consolidate_begin", idx.resident_bytes(), base_rss);
    let built = job.build(&dir).unwrap();
    row("dopo build del fold", idx.resident_bytes(), base_rss);
    idx.consolidate_finish(built).unwrap().expect_clean();
    row("consolidate_finish", idx.resident_bytes(), base_rss);

    drop(idx);
    let re = DiskVamanaIndex::open_with_tier(&dir, tier).unwrap();
    row("riaperto, a riposo", re.resident_bytes(), base_rss);
    let _ = std::fs::remove_dir_all(&dir);
}
