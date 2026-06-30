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

mod flat;
mod quant;
mod source;
mod turboquant;
mod vamana;
mod visited;

pub use flat::FlatIndex;
pub use quant::{QuantKind, QuantizedVectors, QueryCode};
pub use source::{InMemoryVectorSource, MmapVectorSource, VectorSource};
pub use turboquant::{FastRotation, Rotation, TurboQuant1, TurboQuant2, TurboQuant4};
pub use vamana::{
    DiskVamanaIndex, VamanaConfig, VamanaIndex, build_phase_times_ns, reset_build_phase_times,
    set_speed_enabled,
};
pub use visited::VisitedBitset;
