//! Platform-level model of the durability primitive (`fsync` /
//! `F_FULLFSYNC`).
//!
//! The committer in `skeg-core` picks a strategy based on
//! [`DurabilityModel`]:
//!
//! - [`DurabilityModel::DeviceGlobal`]: the platform's durability call
//!   is a barrier across the whole device. N concurrent calls are
//!   serialized by the hardware, so the only way to scale write
//!   throughput with shard count is to aggregate writes from every
//!   shard into a single fsync. Apple Silicon `F_FULLFSYNC` is the
//!   reference case (slice A measured a 0.37× regression going from
//!   1 to 4 shards because each shard paid its own barrier).
//! - [`DurabilityModel::PerFile`]: the durability call only flushes the
//!   pages of the file descriptor it is called on. N file descriptors
//!   on N files can be flushed in parallel, so per-shard committers
//!   scale linearly. Linux ext4/xfs/btrfs with `fdatasync` are the
//!   reference case.
//!
//! The const [`DURABILITY_MODEL`] is the platform default, picked at
//! compile time. A runtime override is exposed through
//! [`resolve_durability_model`] for tests (and operators who need to
//! force a specific strategy): see the `SKEG_DURABILITY_MODEL` env var.

use core::sync::atomic::{AtomicU8, Ordering};
use std::env;

/// How `sync_durable` behaves on this platform's filesystem stack.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum DurabilityModel {
    /// `sync_durable` is a device-wide barrier. Concurrent calls on
    /// different files serialize on the hardware. Aggregating writes
    /// from N shards into a single `sync_durable` recovers the
    /// throughput that N parallel barriers would lose.
    DeviceGlobal = 1,
    /// `sync_durable` only waits for the open file's dirty pages.
    /// Per-shard committers scale linearly with shard count.
    PerFile = 2,
}

impl DurabilityModel {
    const fn from_u8(v: u8) -> Self {
        match v {
            2 => Self::PerFile,
            // 0 (uninitialised) and 1 both map to the safe default. The
            // env override below normalises to the explicit variants
            // before storing, so `from_u8(0)` only happens before any
            // call to `resolve_durability_model`.
            _ => Self::DeviceGlobal,
        }
    }
}

/// Platform default, picked at compile time. Const so the compiler can
/// see through the dispatch in the release build and prune the
/// unreached branch.
pub const DURABILITY_MODEL: DurabilityModel = {
    #[cfg(target_os = "macos")]
    {
        DurabilityModel::DeviceGlobal
    }
    #[cfg(target_os = "linux")]
    {
        DurabilityModel::PerFile
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        // Conservative default for platforms we have not measured:
        // assume the worst, aggregate writes, accept the small extra
        // latency over the throughput regression that an unmodelled
        // device-global barrier would cause.
        DurabilityModel::DeviceGlobal
    }
};

/// Cached, runtime-overridable model. `0` = not yet resolved.
static CACHED_MODEL: AtomicU8 = AtomicU8::new(0);

/// Return the active [`DurabilityModel`].
///
/// First call reads `SKEG_DURABILITY_MODEL` (`device-global` /
/// `per-file`, case-insensitive) and caches the answer in an
/// `AtomicU8`. Subsequent calls are a single relaxed load (~1 ns) so
/// the dispatcher in `skeg-core` can branch on the result on every
/// `start` without measurable cost.
///
/// Unknown env values fall back to [`DURABILITY_MODEL`] silently. The
/// override is intended for tests and operators who already know what
/// they are doing; a typo should not crash the engine.
pub fn resolve_durability_model() -> DurabilityModel {
    let raw = CACHED_MODEL.load(Ordering::Relaxed);
    if raw != 0 {
        return DurabilityModel::from_u8(raw);
    }

    let resolved = match env::var("SKEG_DURABILITY_MODEL")
        .ok()
        .map(|s| s.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("device-global" | "deviceglobal") => DurabilityModel::DeviceGlobal,
        Some("per-file" | "perfile") => DurabilityModel::PerFile,
        // Empty, missing, or unrecognised value: use the platform
        // default. No logging here, the platform crate is below
        // `tracing` in the dep graph.
        _ => DURABILITY_MODEL,
    };

    CACHED_MODEL.store(resolved as u8, Ordering::Relaxed);
    resolved
}

/// **Tests only.** Force a specific model, bypassing the env / cache.
///
/// Used by `skeg-core`'s committer tests to exercise both code paths
/// regardless of the host platform. Not exposed in release builds.
#[cfg(any(test, feature = "testing"))]
pub fn set_durability_model_for_tests(model: DurabilityModel) {
    CACHED_MODEL.store(model as u8, Ordering::Relaxed);
}

/// **Tests only.** Reset the cache so the next [`resolve_durability_model`]
/// re-reads the env. Paired with `set_durability_model_for_tests`.
#[cfg(any(test, feature = "testing"))]
pub fn reset_durability_model_cache_for_tests() {
    CACHED_MODEL.store(0, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The platform const must match the build target.
    #[test]
    fn const_matches_target() {
        #[cfg(target_os = "macos")]
        assert_eq!(DURABILITY_MODEL, DurabilityModel::DeviceGlobal);
        #[cfg(target_os = "linux")]
        assert_eq!(DURABILITY_MODEL, DurabilityModel::PerFile);
    }

    /// Override paths are honoured and idempotent.
    #[test]
    fn override_round_trip() {
        // Tests run in random order; clear the cache up front.
        reset_durability_model_cache_for_tests();
        set_durability_model_for_tests(DurabilityModel::PerFile);
        assert_eq!(resolve_durability_model(), DurabilityModel::PerFile);
        set_durability_model_for_tests(DurabilityModel::DeviceGlobal);
        assert_eq!(resolve_durability_model(), DurabilityModel::DeviceGlobal);
        reset_durability_model_cache_for_tests();
    }

    /// `from_u8` is a closed mapping with `DeviceGlobal` as the safe
    /// catch-all.
    #[test]
    fn from_u8_safe_default() {
        assert_eq!(DurabilityModel::from_u8(0), DurabilityModel::DeviceGlobal);
        assert_eq!(DurabilityModel::from_u8(1), DurabilityModel::DeviceGlobal);
        assert_eq!(DurabilityModel::from_u8(2), DurabilityModel::PerFile);
        assert_eq!(DurabilityModel::from_u8(99), DurabilityModel::DeviceGlobal);
    }
}
