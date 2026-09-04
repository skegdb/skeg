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

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use skeg_vector::{DiskVamanaIndex, MaintenanceKind, QuantKind};

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

fn peak_while<T>(work: impl FnOnce() -> T) -> (T, u64) {
    let running = Arc::new(AtomicBool::new(true));
    let peak = Arc::new(AtomicU64::new(skeg_platform::rss_bytes()));
    let sampler_running = Arc::clone(&running);
    let sampler_peak = Arc::clone(&peak);
    let sampler = std::thread::spawn(move || {
        while sampler_running.load(Ordering::Acquire) {
            sampler_peak.fetch_max(skeg_platform::rss_bytes(), Ordering::AcqRel);
            std::thread::sleep(Duration::from_millis(2));
        }
        sampler_peak.fetch_max(skeg_platform::rss_bytes(), Ordering::AcqRel);
    });
    let result = work();
    running.store(false, Ordering::Release);
    sampler.join().unwrap();
    (result, peak.load(Ordering::Acquire))
}

fn verdict(kind: &str, estimate: u64, before: u64, peak: u64) {
    let observed = peak.saturating_sub(before);
    println!(
        "{kind:<34} estimate {:>7.1} MiB   observed peak +{:>7.1} MiB   margin {:>6.2}x",
        estimate as f64 / (1024.0 * 1024.0),
        observed as f64 / (1024.0 * 1024.0),
        estimate as f64 / observed.max(1) as f64,
    );
    if env("RAM_ASSERT_ESTIMATE", 0_u8) != 0 {
        assert!(
            observed <= estimate,
            "{kind} estimator under-counted: estimate={estimate}, observed={observed}"
        );
    }
}

fn main() {
    let n: u64 = env("RAM_N", 12_500);
    let dim: usize = env("RAM_DIM", 1024);
    let tier = QuantKind::TurboQuant { bits: 2 };
    let tmp = tempfile::TempDir::new().unwrap();
    let dir = tmp.path().to_path_buf();

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

    let flush_estimate = idx.maintenance_working_set_bytes(MaintenanceKind::Flush);
    let before = skeg_platform::rss_bytes();
    let (_, peak) = peak_while(|| {
        let job = idx.flush_begin().unwrap().unwrap();
        let built = job.build(&dir).unwrap();
        idx.flush_finish(built).unwrap().expect_clean();
    });
    verdict("flush build", flush_estimate, before, peak);
    row(
        "flush_finish (run al posto)",
        idx.resident_bytes(),
        base_rss,
    );

    let fold_estimate = idx.maintenance_working_set_bytes(MaintenanceKind::Consolidate);
    let before = skeg_platform::rss_bytes();
    let (_, peak) = peak_while(|| {
        let job = idx.consolidate_begin().unwrap().unwrap();
        let built = job.build(&dir).unwrap();
        idx.consolidate_finish(built).unwrap().expect_clean();
    });
    verdict("consolidate build", fold_estimate, before, peak);
    row("consolidate_finish", idx.resident_bytes(), base_rss);

    drop(idx);
    let re = DiskVamanaIndex::open_with_tier(&dir, tier).unwrap();
    row("riaperto, a riposo", re.resident_bytes(), base_rss);
}
