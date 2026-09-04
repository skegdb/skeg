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
//!   reference case (benchmarks measured a 0.37× regression going from
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

/// Is this build targeting an Apple platform?
///
/// The one place the question is asked, because two properties the
/// [`DurabilityModel::DeviceGlobal`] strategy depends on exist only there:
/// `F_FULLFSYNC` asks the drive to flush its whole buffered cache, and
/// `F_NOCACHE` (applied to every `PlatformFile` at open, and `#[cfg(target_os
/// = "macos")]`) has already taken other files' bytes out of the page cache.
pub const IS_APPLE: bool = cfg!(target_vendor = "apple");

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

    let raw_env = env::var("SKEG_DURABILITY_MODEL")
        .ok()
        .map(|s| s.trim().to_ascii_lowercase());
    let (resolved, _refused) = resolve_model_from(raw_env.as_deref(), IS_APPLE);

    CACHED_MODEL.store(resolved as u8, Ordering::Relaxed);
    resolved
}

/// The override decision, with the platform passed in rather than read from
/// `cfg!`.
///
/// Split out so both branches are testable on any host: a rule about what
/// Linux may not select is worth nothing if it can only be exercised on Linux.
///
/// Returns the model, and whether a `device-global` request was REFUSED
/// because this platform cannot honour it.
fn resolve_model_from(raw: Option<&str>, _is_apple: bool) -> (DurabilityModel, bool) {
    match raw {
        Some("device-global" | "deviceglobal") => (DurabilityModel::DeviceGlobal, false),
        Some("per-file" | "perfile") => (DurabilityModel::PerFile, false),
        // Empty, missing, or unrecognised value: use the platform default. A
        // typo must not crash the engine.
        _ => (DURABILITY_MODEL, false),
    }
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

    /// `device-global` is a claim about the hardware, not a preference.
    ///
    /// The shared committer issues ONE durability call for a batch spanning
    /// several files. That is only a barrier for all of them where the call is
    /// device-wide - Apple's `F_FULLFSYNC`, with `F_NOCACHE` keeping the other
    /// files' bytes out of the page cache. Linux `fsync(2)` transfers the data
    /// "of the file referred to by the file descriptor fd" and nothing else,
    /// so the same code there acks the other files' writers as durable over
    /// pages no one ever flushed. An operator setting the env var must not be
    /// able to buy that.
    #[test]
    #[ignore = "opens in the commit that refuses device-global off Apple"]
    fn device_global_is_refused_where_it_is_not_a_device_barrier() {
        let (model, refused) = resolve_model_from(Some("device-global"), true);
        assert_eq!(model, DurabilityModel::DeviceGlobal, "Apple may ask for it");
        assert!(!refused, "Apple may ask for it");

        let (model, refused) = resolve_model_from(Some("device-global"), false);
        assert!(
            refused,
            "device-global was honoured on a platform whose durability call is per-file"
        );
        assert_eq!(
            model, DURABILITY_MODEL,
            "a refused override must fall back to the platform default"
        );

        // Everything else is unaffected: per-file is safe everywhere, and an
        // unknown value still falls back silently rather than crashing.
        assert_eq!(
            resolve_model_from(Some("per-file"), false),
            (DurabilityModel::PerFile, false)
        );
        assert_eq!(
            resolve_model_from(Some("nonsense"), false),
            (DURABILITY_MODEL, false)
        );
        assert_eq!(resolve_model_from(None, false), (DURABILITY_MODEL, false));
    }

    /// The same rule, asked of the REAL platform constant rather than a
    /// parameter. Only meaningful off Apple, so only compiled there.
    #[cfg(not(target_vendor = "apple"))]
    #[test]
    #[ignore = "opens in the commit that refuses device-global off Apple"]
    fn device_global_is_not_reachable_by_env_on_this_platform() {
        let (model, refused) = resolve_model_from(Some("device-global"), IS_APPLE);
        assert!(refused, "this platform must refuse device-global");
        assert_ne!(
            model,
            DurabilityModel::DeviceGlobal,
            "one file synced is not a device barrier here"
        );
        assert_eq!(model, DURABILITY_MODEL);
    }

    /// The mirror of the test above, on the platform where the answer is yes.
    ///
    /// Not red before the fix - it pins the half that already holds - but
    /// without it a "fix" that refuses `device-global` everywhere would pass
    /// the two tests above and quietly cost this platform its shared
    /// committer.
    #[cfg(target_vendor = "apple")]
    #[test]
    fn device_global_stays_reachable_by_env_on_this_platform() {
        let (model, refused) = resolve_model_from(Some("device-global"), IS_APPLE);
        assert!(!refused, "Apple's F_FULLFSYNC is a device barrier");
        assert_eq!(model, DurabilityModel::DeviceGlobal);
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
