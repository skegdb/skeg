//! HMAC-SHA256 bearer tokens with TTL and rotation.
//!
//! Token wire form (43 bytes raw, 64 bytes base64url):
//!
//! ```text
//! [tenant_id 16B][issued_unix_secs u64 LE 8B][nonce 7B][hmac_sha256_trunc 16B]
//! ```
//!
//! The signing key is held in process memory only. Restarting the server
//! invalidates every outstanding token; we accept this for the v0.2 model
//! because token lifetime is configurable (default 1h) and it eliminates
//! a whole class of "stale token still valid after key rotation" bugs.
//!
//! Verification compares the HMAC tag in constant time via `subtle`.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use hmac::{Hmac, Mac};
use parking_lot::RwLock;
use rand::RngCore;
use sha2::Sha256;
use subtle::ConstantTimeEq;
use thiserror::Error;

use crate::id::TenantId;

const NONCE_LEN: usize = 7;
const TAG_LEN: usize = 16;
const TOKEN_LEN: usize = TenantId::LEN + 8 + NONCE_LEN + TAG_LEN; // 47

#[derive(Debug, Error, PartialEq, Eq, Clone)]
pub enum TokenError {
    #[error("token length wrong: got {0}, want {1}")]
    BadLength(usize, usize),
    #[error("token HMAC mismatch")]
    BadMac,
    #[error("token expired")]
    Expired,
    #[error("token tenant {got} does not match expected {expected}")]
    TenantMismatch { got: TenantId, expected: TenantId },
    #[error("system clock is before unix epoch")]
    BadClock,
}

type HmacSha256 = Hmac<Sha256>;

/// Result of a successful verify. The caller can trust `tenant`.
#[derive(Debug, Clone, Copy)]
pub struct VerifiedToken {
    pub tenant: TenantId,
    pub issued_at: u64,
    pub expires_at: u64,
}

/// Source of tokens. Holds the signing key and TTL.
#[derive(Debug)]
pub struct TokenStore {
    key: RwLock<[u8; 32]>,
    ttl_secs: u64,
}

impl TokenStore {
    /// Generate a fresh signing key. Use this on server start.
    #[must_use]
    pub fn fresh(ttl: Duration) -> Self {
        let mut k = [0u8; 32];
        rand::rng().fill_bytes(&mut k);
        Self {
            key: RwLock::new(k),
            ttl_secs: ttl.as_secs().max(1),
        }
    }

    /// Construct from an externally-supplied key. Useful for tests and
    /// for clusters that want a deterministic key across processes (but
    /// note: such clusters lose the restart-invalidates-tokens property).
    #[must_use]
    pub fn from_key(key: [u8; 32], ttl: Duration) -> Self {
        Self {
            key: RwLock::new(key),
            ttl_secs: ttl.as_secs().max(1),
        }
    }

    /// Rotate the signing key. Outstanding tokens become invalid.
    pub fn rotate(&self) {
        let mut k = [0u8; 32];
        rand::rng().fill_bytes(&mut k);
        *self.key.write() = k;
    }

    /// Mint a bearer token for `tenant`.
    ///
    /// # Errors
    ///
    /// Returns `TokenError::BadClock` if the host clock is before the
    /// unix epoch.
    pub fn issue(&self, tenant: TenantId) -> Result<[u8; TOKEN_LEN], TokenError> {
        let now = now_secs()?;
        let mut nonce = [0u8; NONCE_LEN];
        rand::rng().fill_bytes(&mut nonce);
        Ok(self.encode(tenant, now, nonce))
    }

    fn encode(&self, tenant: TenantId, issued_at: u64, nonce: [u8; NONCE_LEN]) -> [u8; TOKEN_LEN] {
        let mut buf = [0u8; TOKEN_LEN];
        buf[..TenantId::LEN].copy_from_slice(tenant.as_bytes());
        buf[TenantId::LEN..TenantId::LEN + 8].copy_from_slice(&issued_at.to_le_bytes());
        buf[TenantId::LEN + 8..TenantId::LEN + 8 + NONCE_LEN].copy_from_slice(&nonce);
        let mac = self.mac(&buf[..TenantId::LEN + 8 + NONCE_LEN]);
        buf[TenantId::LEN + 8 + NONCE_LEN..].copy_from_slice(&mac[..TAG_LEN]);
        buf
    }

    fn mac(&self, msg: &[u8]) -> [u8; 32] {
        let key = self.key.read();
        let mut mac = HmacSha256::new_from_slice(&*key).expect("hmac sha256 accepts any key len");
        mac.update(msg);
        let result = mac.finalize().into_bytes();
        let mut out = [0u8; 32];
        out.copy_from_slice(&result);
        out
    }

    /// Verify a token. Returns the trusted `VerifiedToken` view on success.
    ///
    /// # Errors
    ///
    /// Returns one of `TokenError::BadLength`, `BadMac`, `Expired`,
    /// `TenantMismatch`, or `BadClock`.
    pub fn verify(
        &self,
        token: &[u8],
        expected: Option<TenantId>,
    ) -> Result<VerifiedToken, TokenError> {
        if token.len() != TOKEN_LEN {
            return Err(TokenError::BadLength(token.len(), TOKEN_LEN));
        }
        let signed = &token[..TenantId::LEN + 8 + NONCE_LEN];
        let tag = &token[TenantId::LEN + 8 + NONCE_LEN..];
        let computed = self.mac(signed);
        // Constant-time compare: this is the security-critical step.
        if !bool::from(computed[..TAG_LEN].ct_eq(tag)) {
            return Err(TokenError::BadMac);
        }

        let mut id_bytes = [0u8; TenantId::LEN];
        id_bytes.copy_from_slice(&token[..TenantId::LEN]);
        let tenant = TenantId::from_bytes(id_bytes);

        if let Some(exp) = expected
            && exp != tenant
        {
            return Err(TokenError::TenantMismatch {
                got: tenant,
                expected: exp,
            });
        }

        let mut iat_bytes = [0u8; 8];
        iat_bytes.copy_from_slice(&token[TenantId::LEN..TenantId::LEN + 8]);
        let issued_at = u64::from_le_bytes(iat_bytes);
        let expires_at = issued_at.saturating_add(self.ttl_secs);
        let now = now_secs()?;
        if now > expires_at {
            return Err(TokenError::Expired);
        }
        Ok(VerifiedToken {
            tenant,
            issued_at,
            expires_at,
        })
    }
}

fn now_secs() -> Result<u64, TokenError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .map_err(|_| TokenError::BadClock)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(ttl: Duration) -> TokenStore {
        TokenStore::from_key([0xAB; 32], ttl)
    }

    #[test]
    fn issue_then_verify_roundtrip() {
        let s = store(Duration::from_secs(60));
        let alice = TenantId::from_name("alice");
        let tok = s.issue(alice).unwrap();
        let v = s.verify(&tok, Some(alice)).unwrap();
        assert_eq!(v.tenant, alice);
    }

    #[test]
    fn verify_rejects_wrong_tenant() {
        let s = store(Duration::from_secs(60));
        let alice = TenantId::from_name("alice");
        let bob = TenantId::from_name("bob");
        let tok = s.issue(alice).unwrap();
        let e = s.verify(&tok, Some(bob)).unwrap_err();
        match e {
            TokenError::TenantMismatch { got, expected } => {
                assert_eq!(got, alice);
                assert_eq!(expected, bob);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn verify_rejects_tampered_byte() {
        let s = store(Duration::from_secs(60));
        let alice = TenantId::from_name("alice");
        let mut tok = s.issue(alice).unwrap();
        // Flip a byte in the signed portion: HMAC must catch it.
        tok[5] ^= 0x01;
        let e = s.verify(&tok, None).unwrap_err();
        assert_eq!(e, TokenError::BadMac);
    }

    #[test]
    fn verify_rejects_tampered_tag() {
        let s = store(Duration::from_secs(60));
        let alice = TenantId::from_name("alice");
        let mut tok = s.issue(alice).unwrap();
        // Flip a tag byte.
        let last = tok.len() - 1;
        tok[last] ^= 0x01;
        let e = s.verify(&tok, None).unwrap_err();
        assert_eq!(e, TokenError::BadMac);
    }

    #[test]
    fn verify_rejects_truncated() {
        let s = store(Duration::from_secs(60));
        let tok = s.issue(TenantId::from_name("alice")).unwrap();
        let e = s.verify(&tok[..tok.len() - 1], None).unwrap_err();
        matches!(e, TokenError::BadLength(_, _));
    }

    #[test]
    fn verify_rejects_after_rotation() {
        let s = store(Duration::from_secs(60));
        let tok = s.issue(TenantId::from_name("alice")).unwrap();
        s.rotate();
        let e = s.verify(&tok, None).unwrap_err();
        assert_eq!(e, TokenError::BadMac);
    }

    #[test]
    fn verify_rejects_expired() {
        // TTL = 1 second; backdate the token by setting a tiny ttl and
        // encoding with a stale issued_at.
        let s = store(Duration::from_secs(1));
        let alice = TenantId::from_name("alice");
        let nonce = [0u8; NONCE_LEN];
        let stale_iat = now_secs().unwrap() - 100;
        let tok = s.encode(alice, stale_iat, nonce);
        let e = s.verify(&tok, None).unwrap_err();
        assert_eq!(e, TokenError::Expired);
    }

    #[test]
    fn different_stores_dont_cross_verify() {
        let s1 = TokenStore::from_key([0xAA; 32], Duration::from_secs(60));
        let s2 = TokenStore::from_key([0xBB; 32], Duration::from_secs(60));
        let tok = s1.issue(TenantId::from_name("alice")).unwrap();
        assert_eq!(s2.verify(&tok, None).unwrap_err(), TokenError::BadMac);
    }
}
