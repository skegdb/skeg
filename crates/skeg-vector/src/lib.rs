#![deny(unsafe_code)]
// Quantizer hot loops use explicit indexing because the inner kernels
// stride multiple buffers in lockstep; rewriting with enumerate hurts
// codegen for the NEON path.
#![allow(clippy::needless_range_loop)]
// Bench scaffolds and a few large-arity tuning functions get flagged
// for too_many_arguments. Refactoring to config structs would only add
// boilerplate without improving call sites.
#![allow(clippy::too_many_arguments)]
// `Box<dyn Fn(&[...]) -> Vec<...>>` style aliases in the quant trait
// are clearer than typedefs in context.
#![allow(clippy::type_complexity)]

//! `skeg-vector` - the vector tier.
//!
//! The flat-scan tier: [`FlatIndex`] holds vectors at f32 precision
//! and, for the int8 and binary kinds, a compact [`QuantizedVectors`] form
//! that a brute-force scan walks fast. A search scans the quantized proxy for
//! a candidate set, then re-ranks survivors with exact f32 cosine.
//!
//! The Vamana graph is layered on top of the same vector storage.

mod balance;
pub mod failpoint;
mod flat;
mod ivf_router;
mod quant;
mod source;
mod tq1_control;
mod turboquant;
mod vamana;
mod visited;

/// Which physical copy of a vector id is the live one.
///
/// A vector id can exist in more than one place at once - two shards during a
/// reshard, a base row and a delta row during a fold, a boundary replica - and
/// "newest wins" has until now been inferred from WHERE a copy sits: a higher
/// LSM layer, a lower shard number, the order a map happened to be iterated
/// in. Position is not identity. A copy that moves keeps its position and
/// loses its history, and the engine then has no way to tell the value a write
/// replaced from the value that replaced it.
///
/// This is that missing fact, carried with the row: monotone per `(index, id)`,
/// allocated by whoever performs a user write, and CARRIED unchanged by
/// anything that only relocates a row (a reshard move, a boundary replica, a
/// fold). Higher wins. Ties fall back to the old positional rule, so a store
/// where nothing has been versioned behaves exactly as it did.
///
/// [`LEGACY`](VectorVersion::LEGACY) - zero - is what every row written before
/// this existed carries. It loses against every real version and ties with
/// itself, which is what makes the upgrade a no-op on data at rest.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct VectorVersion(u64);

impl VectorVersion {
    /// The version of a row written before versions existed. Loses against
    /// every allocated version; ties with itself.
    pub const LEGACY: VectorVersion = VectorVersion(0);

    /// A version from its raw counter value.
    #[must_use]
    pub const fn new(v: u64) -> Self {
        Self(v)
    }

    /// The raw counter value. For encoding and for the shard-side allocator.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// True for a row that predates versioning.
    #[must_use]
    pub const fn is_legacy(self) -> bool {
        self.0 == 0
    }

    /// The next version after this one. Saturating: an allocator that has
    /// issued 2^64 versions for one id has other problems, and wrapping to
    /// zero would turn every later row into a legacy one.
    #[must_use]
    pub const fn next(self) -> Self {
        Self(self.0.saturating_add(1))
    }
}

impl std::fmt::Display for VectorVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

pub use balance::{balanced_kmeans, nearest_centroid};
pub use flat::FlatIndex;
pub use ivf_router::IvfRouter;
pub use quant::{
    QuantKind, QuantizedVectors, QueryCode, TQ1_HYBRID_MIN_DIM, Tq1ProxyMode, tq1_proxy_mode_for,
};
pub use source::{InMemoryVectorSource, MmapVectorSource, VectorSource};
pub use tq1_control::{SHADOW_EVERY, Tq1ProxyController};
pub use turboquant::{FastRotation, Rotation, TurboQuant1, TurboQuant2, TurboQuant4};
pub use vamana::{
    ConsolidateBuilt, ConsolidateJob, ConsolidatePace, DeletePatchBuilt, DeletePatchJob,
    DiskVamanaIndex, FinishOutcome, FinishResult, FlushBuilt, FlushJob, IvfBuilt, IvfJob,
    RunMergeBuilt, RunMergeJob, VamanaConfig, VamanaIndex, build_phase_times_ns,
    reset_build_phase_times, run_vacuum_debt, set_speed_enabled,
};
pub use visited::VisitedBitset;
