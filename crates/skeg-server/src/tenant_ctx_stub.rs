//! Tenant-API stub used when the `tenant` feature is disabled.
//!
//! This keeps `skeg-server` buildable without pulling `skeg-tenant` into the
//! dependency graph. Runtime tenant authentication is unavailable in this mode.

use std::fmt;
use std::io;
use std::sync::Arc;

use xxhash_rust::xxh3::xxh3_128;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TenantId([u8; Self::LEN]);

impl TenantId {
    pub const LEN: usize = 16;
    pub const ZERO: Self = Self([0; Self::LEN]);

    #[must_use]
    pub fn is_zero(self) -> bool {
        self == Self::ZERO
    }

    #[must_use]
    pub fn as_bytes(&self) -> &[u8; Self::LEN] {
        &self.0
    }

    #[must_use]
    pub fn from_bytes(bytes: [u8; Self::LEN]) -> Self {
        Self(bytes)
    }

    #[must_use]
    pub fn from_name(name: &str) -> Self {
        Self(xxh3_128(name.as_bytes()).to_le_bytes())
    }
}

impl fmt::Display for TenantId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in &self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

#[must_use]
pub fn scoped_vindex_name(tenant: TenantId, name: &str) -> String {
    if tenant.is_zero() {
        name.to_owned()
    } else {
        format!("{tenant}::{name}")
    }
}

#[derive(Debug, Default)]
pub struct TenantContext;

impl TenantContext {
    pub fn open_lenient(_auth_path: impl AsRef<std::path::Path>) -> io::Result<Arc<Self>> {
        Err(io::Error::other(
            "tenant support is disabled at compile time; rebuild skeg-server with --features tenant",
        ))
    }

    pub fn open_strict(_auth_path: impl AsRef<std::path::Path>) -> io::Result<Arc<Self>> {
        Err(io::Error::other(
            "tenant support is disabled at compile time; rebuild skeg-server with --features tenant",
        ))
    }

    /// Always fails when the `tenant` feature is off.
    ///
    /// # Errors
    ///
    /// Always returns [`std::io::Error::other`].
    pub fn verify_login(&self, _user: &str, _pass: &[u8]) -> io::Result<TenantId> {
        Err(io::Error::other(
            "tenant support is disabled at compile time",
        ))
    }

    #[must_use]
    pub fn has_tenant(&self, _candidate: TenantId) -> bool {
        false
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NullResolver;

#[must_use]
pub fn null_resolver() -> NullResolver {
    NullResolver
}
