#![deny(unsafe_code)]
#![warn(missing_docs)]

//! `skeg-rigging-net` - shared, transport-agnostic types for the
//! family of network-attached adapters that implement
//! [`skeg-rigging`](https://crates.io/crates/skeg-rigging) over the
//! wire.
//!
//! This crate does **not** open any sockets by itself. It provides:
//!
//! - [`TenantLocation`]: enum the membrane uses to decide which
//!   transport to dispatch to.
//! - [`RecordEnvelope`]: the JSON shape that hansa-side adapters wrap
//!   around payloads so the `shareable` flag and tags survive a
//!   transport that doesn't model them natively (e.g. plain
//!   skeg-server, which only stores `(id, bytes)`).
//! - [`NetError`]: the umbrella error type re-used by the concrete
//!   transport crates.
//!
//! Concrete transports live in sibling crates:
//!
//! - `skeg-rigging-net-resp3`: talks to a skeg-server via RESP3.
//! - (planned) `skeg-rigging-net-http`: tiny HTTP for saga distribution
//!   plus an optional alternate query path.

mod envelope;
mod error;
mod location;

pub use envelope::{
    DEFAULT_ZSTD_LEVEL, ENVELOPE_KEY_PREFIX, EnvelopeError, RecordEnvelope, envelope_key_for,
};
pub use error::NetError;
pub use location::{TenantLocation, parse_location};
