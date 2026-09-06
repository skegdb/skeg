//! Performance gates for the binary envelope (F.55).
//!
//! Run with:
//!   cargo test --release --test gates -p skeg-rigging-net
//!
//! Gates are skipped in debug mode; release-only thresholds are set
//! with 2-3x headroom over measured bests on M-series Apple Silicon.

use std::time::Instant;

use skeg_rigging_net::{DEFAULT_ZSTD_LEVEL, RecordEnvelope};

fn skip_unless_release() -> bool {
    if cfg!(debug_assertions) {
        eprintln!("[gates] skipping in debug mode");
        true
    } else {
        false
    }
}

// ── Thresholds ──────────────────────────────────────────────────────

/// Binary encoding of a 1 KB-payload envelope must be at least 3x
/// smaller than the JSON encoding. Measured ratio ~3.8x.
const GATE_BIN_VS_JSON_SIZE_RATIO: f32 = 3.0;

/// Encoding a 1 KB envelope in the binary form. Best-of-100 below
/// 5 us (a u8 push + 3 LE writes + one crc32c pass + one Vec extend).
const GATE_ENCODE_BINARY_US: u128 = 5;

/// Decoding the same. Best-of-100 below 5 us. Same shape as encode
/// (crc + bounds checks).
const GATE_DECODE_BINARY_US: u128 = 5;

/// F.20: zstd on a 10 KB English-prose payload. Measured ~50x ratio;
/// gate at 10x so a regression in compressibility (or a zstd switch
/// that disables long-range matching) shows up loudly.
const GATE_ZSTD_PROSE_RATIO: f32 = 10.0;

/// Encoding a 10 KB prose envelope under zstd level 3. Best-of-50
/// below 1 ms; zstd compression dominates.
const GATE_ENCODE_ZSTD_MS: u128 = 1;

/// Decoding a 10 KB prose envelope under zstd level 3. Best-of-100
/// below 100 us; zstd decompression is much cheaper than compression.
const GATE_DECODE_ZSTD_US: u128 = 100;

// ── Helpers ─────────────────────────────────────────────────────────

fn sample_kb() -> RecordEnvelope {
    let payload: Vec<u8> = (0..1024).map(|i| ((i % 26) as u8) + b'a').collect();
    RecordEnvelope::new(true, vec!["topic".into(), "skill:python".into()], payload)
}

fn sample_prose_10kb() -> RecordEnvelope {
    let payload = "the quick brown fox jumps over the lazy dog. ".repeat(230);
    RecordEnvelope::new(true, vec!["doc".into()], payload.into_bytes())
}

// ── Gates ───────────────────────────────────────────────────────────

#[test]
fn gate_binary_is_smaller_than_json_by_ratio() {
    if skip_unless_release() {
        return;
    }
    let env = sample_kb();
    let json = env.encode();
    let bin = env.encode_binary();
    let ratio = json.len() as f32 / bin.len() as f32;
    eprintln!(
        "[gate] envelope size json={} bin={} ratio={:.2}x (gate >= {})",
        json.len(),
        bin.len(),
        ratio,
        GATE_BIN_VS_JSON_SIZE_RATIO,
    );
    assert!(
        ratio >= GATE_BIN_VS_JSON_SIZE_RATIO,
        "binary envelope only {ratio:.2}x smaller than JSON; gate \
         {GATE_BIN_VS_JSON_SIZE_RATIO}x"
    );
}

#[test]
fn gate_encode_binary_under_threshold() {
    if skip_unless_release() {
        return;
    }
    let env = sample_kb();
    for _ in 0..16 {
        let _ = env.encode_binary();
    }
    let mut best_us = u128::MAX;
    for _ in 0..100 {
        let t = Instant::now();
        let _ = env.encode_binary();
        best_us = best_us.min(t.elapsed().as_micros());
    }
    eprintln!("[gate] encode_binary best-of-100 = {best_us} us (cap {GATE_ENCODE_BINARY_US})",);
    assert!(
        best_us <= GATE_ENCODE_BINARY_US,
        "encode_binary best-of-100 = {best_us} us, gate \
         {GATE_ENCODE_BINARY_US} us"
    );
}

#[test]
fn gate_decode_binary_under_threshold() {
    if skip_unless_release() {
        return;
    }
    let env = sample_kb();
    let bytes = env.encode_binary();
    for _ in 0..16 {
        let _ = RecordEnvelope::decode_binary(&bytes).unwrap();
    }
    let mut best_us = u128::MAX;
    for _ in 0..100 {
        let t = Instant::now();
        let _ = RecordEnvelope::decode_binary(&bytes).unwrap();
        best_us = best_us.min(t.elapsed().as_micros());
    }
    eprintln!("[gate] decode_binary best-of-100 = {best_us} us (cap {GATE_DECODE_BINARY_US})",);
    assert!(
        best_us <= GATE_DECODE_BINARY_US,
        "decode_binary best-of-100 = {best_us} us, gate \
         {GATE_DECODE_BINARY_US} us"
    );
}

// ── F.20 zstd gates ─────────────────────────────────────────────────

#[test]
fn gate_zstd_prose_ratio_under_threshold() {
    if skip_unless_release() {
        return;
    }
    let env = sample_prose_10kb();
    let plain = env.encode_binary();
    let zstd = env.encode_binary_zstd(DEFAULT_ZSTD_LEVEL);
    let ratio = plain.len() as f32 / zstd.len() as f32;
    eprintln!(
        "[gate] zstd prose ratio plain={} zstd={} ratio={:.1}x (gate >= {})",
        plain.len(),
        zstd.len(),
        ratio,
        GATE_ZSTD_PROSE_RATIO,
    );
    assert!(
        ratio >= GATE_ZSTD_PROSE_RATIO,
        "zstd prose ratio {ratio:.1}x, gate {GATE_ZSTD_PROSE_RATIO}x"
    );
}

#[test]
fn gate_encode_zstd_under_threshold() {
    if skip_unless_release() {
        return;
    }
    let env = sample_prose_10kb();
    for _ in 0..3 {
        let _ = env.encode_binary_zstd(DEFAULT_ZSTD_LEVEL);
    }
    let mut best_ms = u128::MAX;
    for _ in 0..50 {
        let t = Instant::now();
        let _ = env.encode_binary_zstd(DEFAULT_ZSTD_LEVEL);
        best_ms = best_ms.min(t.elapsed().as_millis());
    }
    eprintln!("[gate] encode_zstd best-of-50 = {best_ms} ms (cap {GATE_ENCODE_ZSTD_MS})",);
    assert!(
        best_ms <= GATE_ENCODE_ZSTD_MS,
        "encode_binary_zstd best-of-50 = {best_ms} ms, gate \
         {GATE_ENCODE_ZSTD_MS} ms"
    );
}

#[test]
fn gate_decode_zstd_under_threshold() {
    if skip_unless_release() {
        return;
    }
    let env = sample_prose_10kb();
    let bytes = env.encode_binary_zstd(DEFAULT_ZSTD_LEVEL);
    for _ in 0..5 {
        let _ = RecordEnvelope::decode_binary(&bytes).unwrap();
    }
    let mut best_us = u128::MAX;
    for _ in 0..100 {
        let t = Instant::now();
        let _ = RecordEnvelope::decode_binary(&bytes).unwrap();
        best_us = best_us.min(t.elapsed().as_micros());
    }
    eprintln!("[gate] decode_zstd best-of-100 = {best_us} us (cap {GATE_DECODE_ZSTD_US})",);
    assert!(
        best_us <= GATE_DECODE_ZSTD_US,
        "decode zstd best-of-100 = {best_us} us, gate \
         {GATE_DECODE_ZSTD_US} us"
    );
}
