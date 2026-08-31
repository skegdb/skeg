//! The tier cache: opening an index must not re-quantise the whole corpus.
//!
//! `open_with_tier_full` streamed all of `vectors.bin` (f32) and rebuilt the
//! quantised tier from scratch on every open. Measured on Model Graveyard, that
//! is ~16 s per index for 223k x 1024 vectors (914 MB read and re-quantised for
//! ~26 MB of codes), and two indexes took 32 s. Projected to 3M that is over
//! three minutes of unavailability on every restart, crash recovery, or deploy.
//!
//! The codes are deterministic from the parent index (fixed rotation seed), so
//! they can be persisted once and reloaded. These tests pin that: the cache is
//! written, reused, and - most importantly - REJECTED when it does not match the
//! index it claims to describe.

use skeg_vector::{DiskVamanaIndex, QuantKind, VamanaConfig, VamanaIndex};
use std::time::Instant;

const DIM: usize = 128;
const N: usize = 800;

fn corpus(n: usize, dim: usize) -> Vec<f32> {
    // Deterministic and spread out: a real quantiser must have something to do.
    let mut s = 0x9E3779B97F4A7C15u64;
    (0..n * dim)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / 8192.0 - 1.0
        })
        .collect()
}

fn build(dir: &std::path::Path) {
    let v = corpus(N, DIM);
    let ids: Vec<u64> = (0..N as u64).collect();
    let idx = VamanaIndex::build(v, ids, DIM, &VamanaConfig::default());
    idx.save(dir).unwrap();
}

fn tier() -> QuantKind {
    QuantKind::TurboQuant { bits: 2 }
}

fn open(dir: &std::path::Path) -> DiskVamanaIndex {
    DiskVamanaIndex::open_with_tier_full(dir, tier(), false, false).unwrap()
}

#[test]
fn first_open_writes_the_cache() {
    let tmp = tempfile::TempDir::new().unwrap();
    build(tmp.path());
    let cache = tmp.path().join("tier.cache.bin");
    assert!(!cache.exists(), "no cache before the first open");
    let _idx = open(tmp.path());
    assert!(cache.exists(), "the first open must persist the tier codes");
    assert!(cache.metadata().unwrap().len() > 0);
}

#[test]
fn second_open_reuses_the_cache_and_is_faster() {
    let tmp = tempfile::TempDir::new().unwrap();
    build(tmp.path());

    let t0 = Instant::now();
    let cold = open(tmp.path());
    let cold_ms = t0.elapsed();

    let t1 = Instant::now();
    let warm = open(tmp.path());
    let warm_ms = t1.elapsed();

    assert_eq!(cold.len(), warm.len(), "same number of vectors");
    assert!(
        warm_ms * 2 < cold_ms,
        "a warm open must be at least 2x faster: cold {cold_ms:?}, warm {warm_ms:?}"
    );
}

#[test]
fn reused_cache_gives_identical_search_results() {
    // The whole point is a faster open, not a different index. If the cached
    // codes ever diverge from the rebuilt ones, recall changes silently.
    let tmp = tempfile::TempDir::new().unwrap();
    build(tmp.path());
    let v = corpus(N, DIM);

    let cold = open(tmp.path());
    let warm = open(tmp.path());

    for q in 0..25usize {
        let query = &v[q * 7 * DIM..q * 7 * DIM + DIM];
        let a = cold.search_with_l(query, 10, 64).unwrap();
        let b = warm.search_with_l(query, 10, 64).unwrap();
        assert_eq!(a, b, "query {q}: the cached tier must rank identically");
    }
}

#[test]
fn cache_from_a_different_index_is_rejected() {
    // The dangerous failure: a cache file whose bytes belong to other vectors.
    // Size alone does not catch it - two indexes with the same n/dim produce
    // caches of the same length - so the fingerprint must cover the source.
    let a = tempfile::TempDir::new().unwrap();
    let b = tempfile::TempDir::new().unwrap();
    build(a.path());
    let _ = open(a.path()); // writes a's cache

    // b holds a DIFFERENT corpus of the same shape.
    let mut s = 12345u64;
    let v: Vec<f32> = (0..N * DIM)
        .map(|_| {
            s ^= s << 7;
            s ^= s >> 9;
            (s >> 40) as f32 / 8192.0 - 1.0
        })
        .collect();
    let ids: Vec<u64> = (0..N as u64).collect();
    VamanaIndex::build(v.clone(), ids, DIM, &VamanaConfig::default())
        .save(b.path())
        .unwrap();
    // Plant a's cache into b.
    std::fs::copy(
        a.path().join("tier.cache.bin"),
        b.path().join("tier.cache.bin"),
    )
    .unwrap();

    let idx = open(b.path());
    // Rebuilt from b's own vectors: a self-query must find itself.
    let hits = idx.search_with_l(&v[0..DIM], 1, 64).unwrap();
    assert_eq!(
        hits[0].0, 0,
        "a foreign cache must be rejected, not trusted"
    );
}

#[test]
fn truncated_cache_falls_back_to_rebuild() {
    let tmp = tempfile::TempDir::new().unwrap();
    build(tmp.path());
    let _ = open(tmp.path());
    let cache = tmp.path().join("tier.cache.bin");
    let bytes = std::fs::read(&cache).unwrap();
    std::fs::write(&cache, &bytes[..bytes.len() / 3]).unwrap();

    // Must not panic and must still answer correctly.
    let idx = open(tmp.path());
    let v = corpus(N, DIM);
    let hits = idx.search_with_l(&v[0..DIM], 1, 64).unwrap();
    assert_eq!(hits[0].0, 0, "a truncated cache is rebuilt, not trusted");
}

#[test]
fn garbage_cache_does_not_panic() {
    let tmp = tempfile::TempDir::new().unwrap();
    build(tmp.path());
    std::fs::write(
        tmp.path().join("tier.cache.bin"),
        b"not a tier cache at all",
    )
    .unwrap();
    let idx = open(tmp.path());
    assert_eq!(idx.len(), N);
}
