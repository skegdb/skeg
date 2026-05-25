//! Argon2id password hashing.
//!
//! Cost parameters are tunable. The default (`PasswordHash::default_cost`)
//! aims at ~50 ms on an Apple M1 P-core: m=19456 (~19 MiB), t=2, p=1.
//! For a Personal AI deployment on an embedded box you can dial cost down
//! via `hash_password_with`. For a public-facing server you should dial up.
//!
//! We use `argon2`'s password-hash string format (PHC `$argon2id$...`) on
//! disk so future implementations can re-verify without sharing the cost
//! parameters out of band.

use argon2::password_hash::{PasswordHasher, PasswordVerifier, SaltString};
use argon2::{Algorithm, Argon2, Params, Version};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PasswordError {
    #[error("argon2 configuration invalid: {0}")]
    BadParams(String),
    #[error("hashing failed: {0}")]
    HashFailed(String),
    #[error("verification failed")]
    VerifyFailed,
    #[error("stored hash is malformed: {0}")]
    MalformedHash(String),
}

/// PHC-encoded argon2id hash. Cheap to clone; this is just a string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PasswordHash(pub String);

impl PasswordHash {
    /// Default cost picked for Personal AI deployment.
    /// `m=19456 KiB`, `t=2`, `p=1`. Approximately 50 ms on Apple M1.
    ///
    /// # Panics
    ///
    /// Never: the parameters are constants accepted by argon2.
    #[must_use]
    pub fn default_cost() -> Params {
        Params::new(19456, 2, 1, None).expect("argon2 default params known-good")
    }
}

/// Hash a password with the default cost.
///
/// # Errors
///
/// Returns `PasswordError::HashFailed` if the underlying argon2 crate
/// reports an error (essentially never, given fixed params).
pub fn hash_password(password: &[u8]) -> Result<PasswordHash, PasswordError> {
    hash_password_with(password, PasswordHash::default_cost())
}

/// Hash with caller-supplied cost. Use this for tests that want fast
/// hashing.
///
/// # Errors
///
/// Returns `PasswordError::BadParams` if `params` are rejected by the
/// argon2 library, or `PasswordError::HashFailed` on any other internal
/// failure.
pub fn hash_password_with(password: &[u8], params: Params) -> Result<PasswordHash, PasswordError> {
    // argon2 0.5 depends on rand_core 0.6 for its `SaltString::generate`
    // helper. We are on rand 0.9 in this workspace, so we sidestep the
    // mismatch by sampling salt bytes ourselves and encoding them.
    use rand::RngCore;
    let mut raw = [0u8; 16];
    rand::rng().fill_bytes(&mut raw);
    let salt = SaltString::encode_b64(&raw).map_err(|e| PasswordError::BadParams(e.to_string()))?;
    let argon = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
    let hash = argon
        .hash_password(password, &salt)
        .map_err(|e| PasswordError::HashFailed(e.to_string()))?;
    Ok(PasswordHash(hash.to_string()))
}

/// Verify a password against a stored PHC hash. Constant-time comparison
/// is handled inside the argon2 crate.
///
/// # Errors
///
/// Returns `PasswordError::MalformedHash` if `stored` is not a valid PHC
/// string, or `PasswordError::VerifyFailed` if the password does not match.
pub fn verify_password(password: &[u8], stored: &PasswordHash) -> Result<(), PasswordError> {
    let parsed = argon2::password_hash::PasswordHash::new(&stored.0)
        .map_err(|e| PasswordError::MalformedHash(e.to_string()))?;
    Argon2::default()
        .verify_password(password, &parsed)
        .map_err(|_| PasswordError::VerifyFailed)
}

/// Cheap test cost (m=8 KiB, t=1, p=1). For unit tests only. DO NOT
/// use outside test code; it is far too weak for production.
///
/// # Panics
///
/// Never: the parameters are constants accepted by argon2.
#[must_use]
pub fn cheap_test_cost() -> Params {
    Params::new(8, 1, 1, None).expect("test cost params known-good")
}

#[cfg(test)]
pub(crate) use cheap_test_cost as test_cost;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_then_verify_roundtrip() {
        let h = hash_password_with(b"hunter2", test_cost()).unwrap();
        verify_password(b"hunter2", &h).unwrap();
    }

    #[test]
    fn wrong_password_fails() {
        let h = hash_password_with(b"hunter2", test_cost()).unwrap();
        let e = verify_password(b"hunter3", &h).unwrap_err();
        assert!(matches!(e, PasswordError::VerifyFailed));
    }

    #[test]
    fn salts_make_distinct_hashes() {
        let h1 = hash_password_with(b"same", test_cost()).unwrap();
        let h2 = hash_password_with(b"same", test_cost()).unwrap();
        assert_ne!(h1, h2);
        // Both still verify the same password.
        verify_password(b"same", &h1).unwrap();
        verify_password(b"same", &h2).unwrap();
    }

    #[test]
    fn malformed_hash_rejected() {
        let bogus = PasswordHash("not-a-phc-string".into());
        let e = verify_password(b"x", &bogus).unwrap_err();
        assert!(matches!(e, PasswordError::MalformedHash(_)));
    }
}
