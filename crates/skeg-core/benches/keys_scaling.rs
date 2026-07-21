//! How does `VLog::keys()` scale with the number of live keys? It walks the
//! in-RAM index and allocates one `Vec<u8>` per key plus the outer `Vec`. Times
//! `keys()` at increasing key counts to size the materialisation cost and judge
//! whether a zero-alloc streaming variant is worth adding.
//!
//! `harness = false`, custom main, min-of-rounds ms.

use std::time::Instant;

use skeg_core::VLog;
use skeg_core::group_commit::Durability;
use tempfile::TempDir;

fn main() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let rounds = 5;
    println!("VLog key enumeration: keys() (materialise) vs for_each_key (stream)");
    println!("  (min of {rounds} rounds, ~13-byte keys)");
    for &n in &[10_000u32, 100_000, 500_000, 1_000_000] {
        let dir = TempDir::new().unwrap();
        let v = rt.block_on(async {
            let v = VLog::open(dir.path()).await.unwrap();
            for i in 0..n {
                v.set(format!("k:{i:010}").as_bytes(), b"v", Durability::Relaxed)
                    .await
                    .unwrap();
            }
            v
        });
        assert_eq!(v.len(), n as usize);

        let mut best_keys = f64::MAX;
        let mut best_stream = f64::MAX;
        for _ in 0..rounds {
            let t0 = Instant::now();
            let ks = v.keys();
            best_keys = best_keys.min(t0.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box(&ks);

            let mut count = 0u64;
            let t1 = Instant::now();
            v.for_each_key(|k| count += k.len() as u64);
            best_stream = best_stream.min(t1.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box(count);
        }
        // Bytes the returned Vec holds live: outer Vec of `Vec<u8>` headers
        // (24 B each on 64-bit) plus each key's heap bytes (~13). for_each_key
        // allocates none of this.
        let mib = f64::from(n) * (std::mem::size_of::<Vec<u8>>() as f64 + 13.0) / (1024.0 * 1024.0);
        println!(
            "  n={n:7}: keys() {best_keys:7.2} ms (~{mib:4.1} MiB)   for_each_key {best_stream:7.2} ms (0 MiB)   {:.1}x",
            best_keys / best_stream,
        );
    }
}
