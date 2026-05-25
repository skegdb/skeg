//! Key/name scoping helpers.
//!
//! Tenancy in skeg uses a prefix scheme rather than separate vLogs:
//!
//! - KV: `[tenant_id 16B][key]`
//! - VINDEX: `<tenant_id_hex>::<name>`
//!
//! The prefix is invisible to the user. It is applied at the protocol
//! handler boundary, so the core never sees a bare key from a multi-tenant
//! connection. With `TenantId::ZERO` we skip prefixing so single-tenant
//! deployments hit the exact same byte layout as before.

use crate::id::TenantId;

/// Prefix a raw key with the tenant id. For `TenantId::ZERO` the key is
/// returned unchanged, preserving wire/disk layout for single-tenant.
#[must_use]
pub fn scoped_key(tenant: TenantId, key: &[u8]) -> Vec<u8> {
    if tenant.is_zero() {
        return key.to_vec();
    }
    let mut out = Vec::with_capacity(TenantId::LEN + key.len());
    out.extend_from_slice(tenant.as_bytes());
    out.extend_from_slice(key);
    out
}

/// Inverse of [`scoped_key`]. Returns `(tenant, raw_key)` if the scoped
/// representation matches; `None` if the key is too short to contain a
/// tenant prefix (and so the caller should treat it as a `ZERO` key).
#[must_use]
pub fn split_scoped_key(scoped: &[u8]) -> Option<(TenantId, &[u8])> {
    if scoped.len() < TenantId::LEN {
        return None;
    }
    let mut id = [0u8; TenantId::LEN];
    id.copy_from_slice(&scoped[..TenantId::LEN]);
    Some((TenantId::from_bytes(id), &scoped[TenantId::LEN..]))
}

/// Decorate a VINDEX name with the tenant scope. We use a textual form
/// (`<hex>::<name>`) because VINDEX names are surfaced in error messages
/// and config files; a byte prefix would be opaque there.
#[must_use]
pub fn scoped_vindex_name(tenant: TenantId, name: &str) -> String {
    if tenant.is_zero() {
        return name.to_string();
    }
    format!("{tenant}::{name}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoped_key_zero_is_passthrough() {
        let k = b"hello";
        assert_eq!(scoped_key(TenantId::ZERO, k), k);
    }

    #[test]
    fn scoped_key_prefixes_with_id() {
        let id = TenantId::from_bytes([0xAB; 16]);
        let k = b"foo";
        let s = scoped_key(id, k);
        assert_eq!(s.len(), 16 + 3);
        assert_eq!(&s[..16], &[0xAB; 16]);
        assert_eq!(&s[16..], k);
    }

    #[test]
    fn split_inverts_scoped_key() {
        let id = TenantId::from_bytes([0x12; 16]);
        let s = scoped_key(id, b"bar");
        let (t, k) = split_scoped_key(&s).expect("split");
        assert_eq!(t, id);
        assert_eq!(k, b"bar");
    }

    #[test]
    fn split_short_input_returns_none() {
        assert!(split_scoped_key(b"abc").is_none());
    }

    #[test]
    fn vindex_name_zero_is_bare() {
        assert_eq!(scoped_vindex_name(TenantId::ZERO, "docs"), "docs");
    }

    #[test]
    fn vindex_name_includes_tenant_hex() {
        let id = TenantId::from_bytes([0xFF; 16]);
        let s = scoped_vindex_name(id, "docs");
        assert!(s.starts_with("ffffffffffffffffffffffffffffffff::"));
        assert!(s.ends_with("::docs"));
    }

    #[test]
    fn distinct_tenants_distinct_scoped_keys() {
        let a = TenantId::from_bytes([0x01; 16]);
        let b = TenantId::from_bytes([0x02; 16]);
        assert_ne!(scoped_key(a, b"k"), scoped_key(b, b"k"));
    }
}
