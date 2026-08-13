//! `BlockingBatchReader` against `UringBatchReader` on the read pattern the
//! engine actually issues.
//!
//! The batch that matters is the rerank loop: one query reads up to `rerank`
//! vectors of `dim * 4` bytes from `vectors.bin` at scattered rows. MGET has
//! the same shape with smaller records. Both are "many small reads, known up
//! front", which is the only shape a batch reader can improve.
//!
//! Two cache states, because they answer different questions:
//!
//! - **warm**: every page is in the page cache, so `pread` is a memcpy plus a
//!   syscall. This isolates syscall overhead, which is the only thing
//!   `io_uring` can remove here.
//! - **cold**: the file's pages are evicted with `POSIX_FADV_DONTNEED` before
//!   each batch, so the reads reach the device. This is where overlapping
//!   requests can win.
//!
//! Run with: `cargo bench -p skeg-platform --bench batch_read --features uring`
//! Without the feature it still runs and reports the blocking path alone,
//! which is what CI can check on any machine.

use std::fs::File;
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use skeg_platform::{BatchReader, BlockingBatchReader};

const FILE_BYTES: u64 = 2 << 30; // 2 GiB, comfortably past any CPU cache
const RECORD: usize = 6144; // 1536 f32, the openai3-large row size
const ROUNDS: usize = 20;

/// Deterministic scattered offsets, record-aligned, no repeats within a batch.
fn offsets(n: usize, seed: u64) -> Vec<(u64, usize)> {
    let records = FILE_BYTES / RECORD as u64;
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s % records) * RECORD as u64, RECORD)
        })
        .collect()
}

/// Pull the whole file into the page cache. The warm case must state its
/// precondition rather than assume it: a previous cold measurement leaves the
/// file evicted, and then "warm" silently measures cold IO instead.
fn warm(path: &str) {
    use std::io::Read;
    let mut f = File::open(path).expect("open for warming");
    let mut buf = vec![0u8; 1 << 22];
    while f.read(&mut buf).expect("warm read") > 0 {}
}

/// Evict this file's pages so the next read reaches the device. Without this
/// the second round of any measurement is a page-cache hit and says nothing
/// about IO.
#[cfg(target_os = "linux")]
fn evict(file: &File) {
    use std::os::unix::io::AsRawFd;
    // SAFETY: `posix_fadvise` takes a valid fd and a range; it is advisory,
    // cannot fault, and does not alter file contents.
    unsafe {
        libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
    }
}

#[cfg(not(target_os = "linux"))]
fn evict(_file: &File) {}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn time_batches(reader: &dyn BatchReader, file: &File, batch: usize, cold: bool) -> f64 {
    let mut per_read_us = Vec::with_capacity(ROUNDS);
    for round in 0..ROUNDS {
        let reqs = offsets(batch, round as u64 + 1);
        if cold {
            evict(file);
        }
        let t = Instant::now();
        let out = reader.read_many(&reqs).expect("batch read");
        let elapsed = t.elapsed().as_secs_f64();
        assert_eq!(out.len(), batch);
        per_read_us.push(elapsed * 1e6 / batch as f64);
    }
    median(per_read_us)
}

fn main() {
    let path = std::env::var("SKEG_BATCH_BENCH_FILE")
        .unwrap_or_else(|_| "/tmp/skeg-batch-bench.bin".to_owned());

    if std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0) < FILE_BYTES {
        eprintln!("creating {path} ({} MiB)", FILE_BYTES >> 20);
        let mut f = File::create(&path).expect("create bench file");
        let chunk = vec![0x5Au8; 1 << 20];
        let mut written = 0u64;
        while written < FILE_BYTES {
            f.write_all(&chunk).expect("fill bench file");
            written += chunk.len() as u64;
        }
        f.sync_all().expect("sync bench file");
    }

    let file = Arc::new(File::open(&path).expect("open bench file"));
    println!(
        "{:<10} {:>6} {:>14} {:>14}",
        "cache", "batch", "blocking us/rd", "uring us/rd"
    );

    for cold in [false, true] {
        for batch in [8usize, 64, 256, 800] {
            if !cold {
                warm(&path);
            }
            let blocking = BlockingBatchReader::new(Arc::clone(&file));
            let b = time_batches(&blocking, &file, batch, cold);
            if !cold {
                warm(&path);
            }

            let u = {
                #[cfg(all(target_os = "linux", feature = "uring"))]
                {
                    match skeg_platform::UringBatchReader::probe(Arc::clone(&file)) {
                        Some(r) => format!("{:14.2}", time_batches(&r, &file, batch, cold)),
                        None => format!("{:>14}", "probe failed"),
                    }
                }
                #[cfg(not(all(target_os = "linux", feature = "uring")))]
                {
                    format!("{:>14}", "not built")
                }
            };
            println!(
                "{:<10} {:>6} {:>14.2} {}",
                if cold { "cold" } else { "warm" },
                batch,
                b,
                u
            );
        }
    }
}
