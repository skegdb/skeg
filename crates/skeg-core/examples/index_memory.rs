//! What the KV index costs in memory, isolated from everything else.
//!
//! `cargo run --release -p skeg-core --example index_memory -- <n_keys>`
//!
//! Keys shaped like the ones a real store holds: short display keys and longer
//! tenant-scoped payload keys. Reports both shapes, because the difference
//! between them is the point: grown by doubling, the table keeps the slack of
//! the next power of two for the life of the process.
use skeg_core::index::{Index, IndexEntry};

fn rss_kb() -> u64 {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .unwrap();
    String::from_utf8_lossy(&out.stdout).trim().parse().unwrap_or(0)
}

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1_383_158);
    let presized = std::env::var("PRESIZED").is_ok();
    let base = rss_kb();
    let mut idx = if presized { Index::with_capacity(n) } else { Index::new() };
    let mut key_bytes = 0usize;
    for i in 0..n {
        let key: Vec<u8> = match i % 4 {
            0 => format!("n:{i}").into_bytes(),
            1 => format!("s:{i}").into_bytes(),
            2 => format!("d:{i}").into_bytes(),
            _ => {
                let mut k = Vec::with_capacity(37);
                k.extend_from_slice(&[0u8; 16]);
                k.extend_from_slice(b"mg_summary_v2");
                k.extend_from_slice(&(i as u64).to_le_bytes());
                k
            }
        };
        key_bytes += key.len();
        idx.set(
            key,
            IndexEntry { fingerprint: i as u32, segment_id: 0, _pad: 0, offset: i as u32, size: 128 },
        );
    }
    let used = (rss_kb() - base) as f64;
    println!("  {n} keys, {:.1} MB of key bytes", key_bytes as f64 / 1e6);
    println!(
        "  index, {:<12} {:>5.0} MB   {:>4.0} B/key",
        if presized { "pre-sized:" } else { "grown:" },
        used / 1024.0,
        used * 1024.0 / n as f64
    );
    std::hint::black_box(&idx);
}
