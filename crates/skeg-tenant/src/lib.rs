#![deny(unsafe_code)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

//! Multi-tenant support for skeg.
//!
//! This crate is intentionally decoupled from the core. It owns
//!
//! - `TenantId` and the trivial `ZERO` tenant for back-compat
//! - the `TenantResolver` trait and the four canonical resolvers
//!   (`NullResolver`, `AuthBound`, `PerRequest`, `ConnectionBound`)
//! - the password / token layer in `auth::*`
//! - per-tenant quota accounting in `quota`
//! - namespacing helpers in `namespace`
//!
//! The crate ships as an opt-in dependency. With no resolver wired in,
//! every operation resolves to `TenantId::ZERO` and the binary behaves
//! exactly like a single-tenant server. Wiring in one of the resolvers
//! enables the soft-isolation path for multi-tenant deployments.

pub mod auth;
pub mod id;
pub mod namespace;
pub mod quota;
pub mod resolver;

pub use auth::{AuthError, AuthStore, TokenStore, VerifiedToken};
pub use id::TenantId;
pub use namespace::{scoped_key, scoped_vindex_name, split_scoped_key};
pub use quota::{QuotaError, QuotaTracker, TenantQuota};
pub use resolver::{
    AuthBoundResolver, ConnectionBoundResolver, NullResolver, PerRequestResolver, ResolveError,
    ResolverChain, TenantResolver,
};
