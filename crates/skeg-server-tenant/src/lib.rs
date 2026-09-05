//! Compatibility shim.
//!
//! The authenticated multi-tenant profile moved into `skeg-server` itself,
//! behind its `tenant-auth` feature, and is served by the one public RESP3
//! binary: `skeg-resp3 --tenant-auth <auth.kdb> --tenant-strict`. There is no
//! second executable to start by mistake, and no second image whose
//! entrypoint an operator has to remember.
//!
//! This crate stays so that code depending on
//! `skeg_server_tenant::AuthStoreBackend` keeps building; new code should use
//! [`skeg_server::tenant_auth`] directly.

#![deny(unsafe_code)]

pub use skeg_server::tenant_auth::AuthStoreBackend;
