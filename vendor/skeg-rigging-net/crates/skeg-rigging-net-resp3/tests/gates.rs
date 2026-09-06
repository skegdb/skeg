//! Performance gates for the Resp3 connection pool (F.54).
//!
//! Run with:
//!   cargo test --release --test gates -p skeg-rigging-net-resp3
//!
//! Gates are skipped in debug mode. Thresholds set with 2-3x
//! headroom over best-of-N on M-series Apple Silicon.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use skeg_rigging_net_resp3::{PoolConfig, Resp3Pool};

fn skip_unless_release() -> bool {
    if cfg!(debug_assertions) {
        eprintln!("[gates] skipping in debug mode");
        true
    } else {
        false
    }
}

// ── Thresholds ──────────────────────────────────────────────────────

/// Acquire from a warm pool (idle queue non-empty). Best-of-100 below
/// 5 us — pure mutex + VecDeque pop_front + counter bump.
const GATE_ACQUIRE_WARM_US: u128 = 5;

/// 1000 sequential acquire+release cycles on a single warm pool.
/// Mostly stresses the Drop path. Best-of-3 below 5 ms.
const GATE_THROUGHPUT_1K_MS: u128 = 5;

// ── Helpers ─────────────────────────────────────────────────────────

fn spawn_hello_server() -> (String, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = listener.local_addr().unwrap().to_string();
    let connects = Arc::new(AtomicUsize::new(0));
    let counter = connects.clone();
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut s) = stream else { continue };
            counter.fetch_add(1, Ordering::SeqCst);
            thread::spawn(move || {
                let mut buf = [0u8; 256];
                let _ = s.read(&mut buf);
                let _ = s.write_all(b"%0\r\n");
                loop {
                    match s.read(&mut buf) {
                        Ok(0) | Err(_) => break,
                        Ok(_) => {}
                    }
                }
            });
        }
    });
    (endpoint, connects)
}

// ── Gates ───────────────────────────────────────────────────────────

#[test]
fn gate_acquire_from_warm_pool_under_threshold() {
    if skip_unless_release() {
        return;
    }
    let (endpoint, _) = spawn_hello_server();
    let pool = Resp3Pool::with_config(
        endpoint,
        None,
        PoolConfig {
            max_idle: 4,
            max_total: 4,
            idle_timeout: Duration::from_secs(60),
        },
    );
    // Warm: pre-open one connection so the idle queue is non-empty.
    drop(pool.acquire().unwrap());

    let mut best_us = u128::MAX;
    for _ in 0..100 {
        let t = Instant::now();
        let conn = pool.acquire().unwrap();
        let elapsed = t.elapsed().as_micros();
        drop(conn);
        best_us = best_us.min(elapsed);
    }
    eprintln!("[gate] acquire(warm) best-of-100 = {best_us} us (cap {GATE_ACQUIRE_WARM_US})",);
    assert!(
        best_us <= GATE_ACQUIRE_WARM_US,
        "warm acquire best-of-100 = {best_us} us, gate {GATE_ACQUIRE_WARM_US} us"
    );
}

#[test]
fn gate_acquire_release_throughput_under_threshold() {
    if skip_unless_release() {
        return;
    }
    let (endpoint, _) = spawn_hello_server();
    let pool = Resp3Pool::new(endpoint);
    // Warm.
    drop(pool.acquire().unwrap());

    let mut best_ms = u128::MAX;
    for _ in 0..3 {
        let t = Instant::now();
        for _ in 0..1000 {
            let _ = pool.acquire().unwrap();
        }
        best_ms = best_ms.min(t.elapsed().as_millis());
    }
    eprintln!("[gate] 1000 acquire+release best-of-3 = {best_ms} ms (cap {GATE_THROUGHPUT_1K_MS})",);
    assert!(
        best_ms <= GATE_THROUGHPUT_1K_MS,
        "throughput 1000 cycles best-of-3 = {best_ms} ms, gate \
         {GATE_THROUGHPUT_1K_MS} ms"
    );
}
