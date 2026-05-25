//! Authentication primitives.
//!
//! Three layers:
//!
//! 1. `password`: argon2id-based password hashing and verification.
//!    Tunable cost so embedded deployments can dial memory cost down.
//! 2. `token`: short-lived HMAC-SHA256 bearer tokens with TTL +
//!    rotation. Constant-time verify via `subtle`. Tokens are signed
//!    with a process-secret HMAC key so a tenant cannot forge one.
//! 3. `store`: durable mapping from user-name to (tenant id, password
//!    hash). Persists to `auth.kdb` with a CRC-checked record format and
//!    optional fsync hook supplied by the embedder.
//!
//! The crate stops at primitive layer. Wiring into the RESP3 HELLO
//! handler / binary AUTH op is done by the server crate.

pub mod password;
pub mod store;
pub mod token;

pub use argon2::Params as Argon2Params;
pub use password::{
    PasswordError, PasswordHash, cheap_test_cost, hash_password, hash_password_with,
    verify_password,
};
pub use store::{AuthRecord, AuthStore, StoreError};
pub use token::{TokenError, TokenStore, VerifiedToken};

use thiserror::Error;

/// Catch-all error for the auth surface.
#[derive(Debug, Error)]
pub enum AuthError {
    #[error(transparent)]
    Password(#[from] PasswordError),
    #[error(transparent)]
    Token(#[from] TokenError),
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error("unknown user")]
    UnknownUser,
    #[error("invalid credentials")]
    InvalidCredentials,
}
