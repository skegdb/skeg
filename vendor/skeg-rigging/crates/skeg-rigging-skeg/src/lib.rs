#![deny(unsafe_code)]
#![warn(missing_docs)]

//! Reference implementation of `skeg-rigging` backed by skeg's public
//! engine APIs.
//!
//! The adapter is **embedded**: it links `skeg-vector::FlatIndex` as a
//! library and adds a parallel metadata sidecar (JSON) for the
//! `shareable` flag and tag set that skeg-core does not natively model.
//! This satisfies the rigging capability surface (capability
//! `vector.kv`, see `skeg_rigging` §11.2) without modifying skeg.
//!
//! v0.1 persistence is a single JSON snapshot per tenant. v0.2 will
//! swap in skeg-core's vlog + DiskVamanaIndex for scalable on-disk
//! storage once the relevant skeg public surface stabilises.
//!
//! ## Naming
//!
//! Per rigging's engine-pluralism position, the engine-neutral capability
//! concept is `TenantId`. This crate's concrete struct is named
//! [`Tenant`] - a skeg-specific implementor of the surface, not a
//! redefinition of the concept.

mod meta_sidecar;
mod tenant;

pub use meta_sidecar::MetadataSidecar;
pub use tenant::{Tenant, TenantError, open_readonly};
