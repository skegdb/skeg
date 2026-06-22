//! Tiering gate spike - the load-bearing measurement for tenant RAM overcommit.
//!
//! Question: how much *fixed* RAM does an idle, open, disk-backed tenant index
//! cost (graph metadata, tier codebook, ids, file handles, allocator arenas)?
//! Disk-backed vectors page out via the S3-FIFO cache, so the only thing
//! overcommit (evict cold tenants, keep hot ones in RAM) can reclaim is this
//! fixed per-open-index footprint.
//!
//!   ~KB per idle index  -> overcommit saves little, do NOT build it (YAGNI).
//!   ~MB per idle index  -> large win at thousands of tenants, build it.
//!
//! Method: build K disk-backed indices (one per simulated tenant) at several
//! vector counts N, close them, take a quiet RSS baseline, then OPEN all K and
//! run zero queries. Two signals:
//!   - `resident_bytes()` summed across the K - what the indices themselves hold
//!     in RAM (allocator-independent; the primary number).
//!   - process RSS delta via `ps` - ground truth incl. allocator / mmap / handles.
//!
//! Run on a QUIET machine (no concurrent build/bench - it poisons RSS):
//!   cargo bench -p skeg-vector --bench tiering_gate
//! Tunables (env):
//!   GATE_K=32            indices per N
//!   GATE_DIM=768         vector dimension
//!   GATE_NS=0,1000,10000 comma list of vectors-per-index to sweep
//!   GATE_TIER=int8       int8 | tq2 | tq4 | tq1

use std::path::Path;
use std::process::Command;
use std::time::Instant;

use skeg_vector::{DiskVamanaIndex, QuantKind};

const L_SEARCH: usize = 100;

/// Resident set of this process in MiB, via `ps` (matches benches/vamana.rs).
fn self_rss_mib() -> f64 {
    let pid = std::process::id().to_string();
    Command::new("ps")
        .args(["-o", "rss=", "-p", &pid])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .map_or(0.0, |kb| kb / 1024.0)
}

/// Deterministic synthetic vectors (xorshift; no rand dep). RAM, not recall, is
/// what we measure, so distribution doesn't matter.
fn fill(buf: &mut [f32], state: &mut u64) {
    for x in buf {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        // map to roughly [-1, 1)
        *x = ((*state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn parse_tier(s: &str) -> QuantKind {
    match s {
        "tq1" => QuantKind::TurboQuant { bits: 1 },
        "tq2" => QuantKind::TurboQuant { bits: 2 },
        "tq4" => QuantKind::TurboQuant { bits: 4 },
        _ => QuantKind::Int8,
    }
}

/// Build K indices of N vectors each under `root`, returning their dirs. Each is
/// created, filled, consolidated, then dropped (closed) so the build allocator
/// state does not linger as an open index.
fn build(root: &Path, k: usize, n: usize, dim: usize, tier: QuantKind) -> Vec<std::path::PathBuf> {
    let mut dirs = Vec::with_capacity(k);
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut v = vec![0f32; dim];
    for t in 0..k {
        let dir = root.join(format!("tenant_{t}"));
        std::fs::create_dir_all(&dir).unwrap();
        let mut idx = DiskVamanaIndex::create_empty_with_tier(&dir, dim, L_SEARCH, tier).unwrap();
        for id in 0..n as u64 {
            fill(&mut v, &mut state);
            idx.insert(id, &v).unwrap();
        }
        if n > 0 {
            idx.consolidate().unwrap();
        }
        drop(idx);
        dirs.push(dir);
    }
    dirs
}

/// Total bytes on disk for one index dir.
fn disk_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| e.metadata().ok())
        .map(|m| m.len())
        .sum()
}

fn main() {
    let k = env_usize("GATE_K", 32);
    let dim = env_usize("GATE_DIM", 768);
    let tier_s = std::env::var("GATE_TIER").unwrap_or_else(|_| "int8".into());
    let tier = parse_tier(&tier_s);
    let ns: Vec<usize> = std::env::var("GATE_NS")
        .unwrap_or_else(|_| "0,1000,10000".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    println!("# tiering gate: K={k} dim={dim} tier={tier_s}");
    println!(
        "# idle per-open-index RAM. resident_bytes = index structures; \
         RSS = process ground truth.\n"
    );
    println!(
        "{:>9}  {:>11}  {:>13}  {:>11}  {:>11}  {:>9}",
        "N/idx", "disk/idx", "resident/idx", "RSS/idx", "RSS d tot", "build"
    );
    println!("{}", "-".repeat(74));

    for &n in &ns {
        let tmp = tempfile::TempDir::new().unwrap();
        let t0 = Instant::now();
        let dirs = build(tmp.path(), k, n, dim, tier);
        let build_s = t0.elapsed().as_secs_f64();
        let disk_per = disk_bytes(&dirs[0]);

        // Quiet baseline after the build allocator settles.
        let rss_base = self_rss_mib();

        // Open all K, hold them, zero queries.
        let mut open: Vec<DiskVamanaIndex> = Vec::with_capacity(k);
        for d in &dirs {
            open.push(DiskVamanaIndex::open_with_tier(d, tier).unwrap());
        }
        let rss_open = self_rss_mib();
        let resident_tot: usize = open.iter().map(DiskVamanaIndex::resident_bytes).sum();

        let resident_per = resident_tot as f64 / k as f64;
        let rss_delta = (rss_open - rss_base).max(0.0);
        let rss_per_kib = rss_delta * 1024.0 / k as f64;

        println!(
            "{:>9}  {:>9.1}K  {:>11.1}K  {:>9.1}K  {:>8.1}M  {:>7.1}s",
            n,
            disk_per as f64 / 1024.0,
            resident_per / 1024.0,
            rss_per_kib,
            rss_delta,
            build_s,
        );

        drop(open);
    }

    println!(
        "\n# verdict: resident/idx in KB -> skip tiering (YAGNI). \
         In MB -> build it (x{k} tenants = the win)."
    );
}
