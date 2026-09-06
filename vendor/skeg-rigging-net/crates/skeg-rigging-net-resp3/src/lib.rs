#![deny(unsafe_code)]
#![warn(missing_docs)]

//! RESP3 transport for `skeg-rigging`.
//!
//! This crate lets a hansa consumer (or any other `skeg-rigging`
//! client) talk to a `skeg-server` over RESP3 - the protocol the
//! engine already speaks. Read-only in v0.1.
//!
//! The bridge sits on three primitives that ship in skeg-server today:
//!
//! - `SKEG.VINDEX.LIST` - discover the vector index for this tenant.
//! - `SKEG.VSEARCH` - query the top-k nearest neighbours.
//! - `GET` / `MGET` - fetch a [`RecordEnvelope`](skeg_rigging_net::RecordEnvelope)
//!   for each hit, decode the `shareable` flag and tags, apply the
//!   filter client-side.
//!
//! `iter_vectors` is **not supported** in v0.1: skeg-server lacks a
//! `VSCAN` op. Hansa only calls `iter_vectors` on the owner's tenant
//! (in-process), so the missing op never blocks the federation path.

mod connection;
mod pool;
mod tenant;
mod writer;

pub use connection::Resp3Connection;
pub use pool::{PoolConfig, PooledConnection, Resp3Pool};
pub use tenant::Resp3Tenant;
pub use writer::Resp3Writer;
