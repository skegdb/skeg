//! Consolidate phase timing on the cached 500k tq2 index: build (graph) vs save
//! vs reopen(reread+requant). Answers where consolidate's ~220-250s goes.
//!   SKEG_STUDY_DIR=<cache> SKEG_BENCH_N=500000 SKEG_CONSOLIDATE_TIMING=1
use skeg_vector::{DiskVamanaIndex, QuantKind};
use std::path::PathBuf;
fn main() {
    let n = std::env::var("SKEG_BENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(500_000);
    let dir: PathBuf = std::env::var("SKEG_STUDY_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("skeg_tq1_study"))
        .join(format!("rw_tq2_n{n}"));
    let mut idx = DiskVamanaIndex::open_with_tier(&dir, QuantKind::TurboQuant { bits: 2 }).unwrap();
    let t = std::time::Instant::now();
    idx.consolidate().unwrap();
    eprintln!("total consolidate: {:.1}s", t.elapsed().as_secs_f64());
}
