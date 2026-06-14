//! Perf gate for the shared-committer workstream.
//!
//! Drives 4 variants of {durability model × shard count} through the
//! `GroupCommitter` façade and prints median wall-clock + ops/s for
//! each. Gates:
//!
//!   RECOVERY    (macOS): 4sh_devglobal / 1sh_perfile ≤ 1.5×
//!   IMPROVEMENT (macOS): 4sh_perfile  / 4sh_devglobal ≥ 1.5×
//!   PERFILE     (linux): 4sh_perfile  / 1sh_perfile  ≤ 1.5×
//!
//! Invoked as `cargo bench -p skeg-core --bench sharded_commit`.
//! Run-tunable via env:
//!   SKEG_M4_WRITES   per-shard Power appends (default 2000)
//!   SKEG_M4_BYTES    bytes per record         (default 128)
//!   SKEG_M4_RUNS     runs per variant         (default 5, median wins)
//!   SKEG_M4_SHARDS   shards in the "big" tier (default 4)

#![deny(unsafe_code)]

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use skeg_core::group_commit::{Durability, GroupCommitter};
use skeg_platform::{DurabilityModel, PlatformFile, durability};
use tempfile::TempDir;

#[derive(Debug, Clone, Copy)]
struct Config {
    writes_per_shard: usize,
    record_bytes: usize,
    runs: usize,
    shards_big: usize,
}

impl Config {
    fn from_env() -> Self {
        Self {
            writes_per_shard: env_usize("SKEG_M4_WRITES", 2000),
            record_bytes: env_usize("SKEG_M4_BYTES", 128),
            runs: env_usize("SKEG_M4_RUNS", 5),
            shards_big: env_usize("SKEG_M4_SHARDS", 4),
        }
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

async fn one_run(model: DurabilityModel, shards: usize, dir: &Path, cfg: &Config) -> Duration {
    // Pin dispatch for this run. Safe to flip per-run: the global
    // SharedCommitter thread is dormant when no DeviceGlobal entry
    // is attached, so the PerFile variants pay no SC overhead.
    durability::set_durability_model_for_tests(model);

    let mut committers = Vec::with_capacity(shards);
    for s in 0..shards {
        let file = Arc::new(PlatformFile::create(&dir.join(format!("s{s}.bin"))).unwrap());
        let gc = GroupCommitter::start(file, 0).await;
        committers.push(gc);
    }

    let writes = cfg.writes_per_shard;
    let bytes = cfg.record_bytes;
    let start = Instant::now();
    let mut handles = Vec::with_capacity(shards);
    for gc in committers {
        handles.push(tokio::spawn(async move {
            for _ in 0..writes {
                gc.append(vec![0xABu8; bytes], Durability::Power)
                    .await
                    .unwrap();
            }
            // Force-drain any tail before we stop the clock; otherwise
            // the last partial batch would still be in flight after the
            // futures return, biasing the throughput upward.
            gc.flush().await.unwrap();
        }));
    }
    for h in handles {
        h.await.unwrap();
    }
    start.elapsed()
}

fn median(mut xs: Vec<Duration>) -> Duration {
    xs.sort();
    xs[xs.len() / 2]
}

async fn measure_variant(
    name: &str,
    model: DurabilityModel,
    shards: usize,
    cfg: &Config,
    parent: &Path,
) -> (Duration, f64) {
    let mut runs = Vec::with_capacity(cfg.runs);

    // One warmup run absorbs page-cache cold-start + first-call thread
    // spawn for SharedCommitter::global(). Discarded from the median.
    let warmup_dir = parent.join(format!("{name}_warmup"));
    std::fs::create_dir_all(&warmup_dir).unwrap();
    let _ = one_run(model, shards, &warmup_dir, cfg).await;

    for run in 0..cfg.runs {
        let run_dir = parent.join(format!("{name}_r{run}"));
        std::fs::create_dir_all(&run_dir).unwrap();
        runs.push(one_run(model, shards, &run_dir, cfg).await);
    }

    let m = median(runs.clone());
    let total_writes = shards * cfg.writes_per_shard;
    let ops = total_writes as f64 / m.as_secs_f64();
    println!("{name:>16}: median {m:>10.2?}  ({ops:>10.0} ops/s)  runs={runs:?}");
    (m, ops)
}

fn main() {
    let cfg = Config::from_env();
    let tmp = TempDir::new().unwrap();

    println!("=== perf gate (shared-committer) ===");
    println!(
        "config: writes/shard={} bytes={} runs={} shards_big={}",
        cfg.writes_per_shard, cfg.record_bytes, cfg.runs, cfg.shards_big
    );
    println!(
        "host: target_os={} target_arch={}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    println!();

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    let (base, four_pf, one_dg, four_dg) = rt.block_on(async {
        let base =
            measure_variant("1sh_perfile", DurabilityModel::PerFile, 1, &cfg, tmp.path()).await;
        let four_pf = measure_variant(
            "4sh_perfile",
            DurabilityModel::PerFile,
            cfg.shards_big,
            &cfg,
            tmp.path(),
        )
        .await;
        let one_dg = measure_variant(
            "1sh_devglobal",
            DurabilityModel::DeviceGlobal,
            1,
            &cfg,
            tmp.path(),
        )
        .await;
        let four_dg = measure_variant(
            "4sh_devglobal",
            DurabilityModel::DeviceGlobal,
            cfg.shards_big,
            &cfg,
            tmp.path(),
        )
        .await;
        (base, four_pf, one_dg, four_dg)
    });

    let baseline_t = base.0.as_secs_f64();
    let four_pf_t = four_pf.0.as_secs_f64();
    let _one_dg_t = one_dg.0.as_secs_f64();
    let four_dg_t = four_dg.0.as_secs_f64();

    let recovery = four_dg_t / baseline_t;
    let improvement = four_pf_t / four_dg_t;
    let perfile_scaling = four_pf_t / baseline_t;

    println!();
    println!("--- gates ---");
    println!("RECOVERY    (4sh_devglobal / 1sh_perfile): {recovery:.2}×  (≤ 1.5)");
    println!("IMPROVEMENT (4sh_perfile  / 4sh_devglobal): {improvement:.2}×  (≥ 1.5 macOS)");
    println!("PERFILE     (4sh_perfile  / 1sh_perfile ): {perfile_scaling:.2}×  (≤ 1.5 linux)");
    println!();

    // Hard gates: only enforce on the platform the gate is meaningful
    // on. Print on every platform for visibility.
    let mut failures: Vec<String> = Vec::new();

    if cfg!(target_os = "macos") {
        if recovery > 1.5 {
            failures.push(format!("RECOVERY {recovery:.2}× > 1.5×"));
        }
        if improvement < 1.5 {
            failures.push(format!("IMPROVEMENT {improvement:.2}× < 1.5×"));
        }
    }
    if cfg!(target_os = "linux") && perfile_scaling > 1.5 {
        failures.push(format!("PERFILE {perfile_scaling:.2}× > 1.5×"));
    }

    if failures.is_empty() {
        println!("GATE: PASS");
    } else {
        println!("GATE: FAIL");
        for f in &failures {
            println!("  - {f}");
        }
        std::process::exit(1);
    }
}
