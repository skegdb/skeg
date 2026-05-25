//! `TenantId`: 16-byte opaque identifier, fixed-size for header alignment.
//!
//! `ZERO` is reserved for the "no tenant" / single-tenant default. Wiring a
//! resolver that returns `ZERO` for every connection is equivalent to running
//! without tenancy, which is what the `NullResolver` does.

use std::fmt;

/// 16-byte tenant identifier. Fits in a single SSE register and in the
/// reserved 16 bytes of an extended binary-protocol header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TenantId([u8; 16]);

impl TenantId {
    /// The "no tenant" sentinel. Used as the default when no resolver is
    /// configured and as the bucket every single-tenant key lives under.
    pub const ZERO: TenantId = TenantId([0u8; 16]);

    pub const LEN: usize = 16;

    /// Construct from raw bytes.
    #[must_use]
    pub const fn from_bytes(b: [u8; 16]) -> Self {
        TenantId(b)
    }

    /// Hash a UTF-8 name into a stable `TenantId` (`xxh3_128`). Two servers
    /// resolve the same name to the same id, so this is safe for federation.
    #[must_use]
    pub fn from_name(name: &str) -> Self {
        let h = xxhash_rust::xxh3::xxh3_128(name.as_bytes());
        Self::from_bytes(h.to_le_bytes())
    }

    /// Underlying byte view.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    /// True if this is the `ZERO` sentinel.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.0 == [0u8; 16]
    }
}

impl Default for TenantId {
    fn default() -> Self {
        TenantId::ZERO
    }
}

impl fmt::Display for TenantId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_is_default() {
        assert_eq!(TenantId::default(), TenantId::ZERO);
        assert!(TenantId::ZERO.is_zero());
    }

    #[test]
    fn from_name_is_stable() {
        let a = TenantId::from_name("alice");
        let b = TenantId::from_name("alice");
        assert_eq!(a, b);
        let c = TenantId::from_name("bob");
        assert_ne!(a, c);
    }

    #[test]
    fn from_name_does_not_collide_zero() {
        // Vanishingly unlikely, but lock it in: a real tenant name should not
        // resolve to the ZERO sentinel.
        let a = TenantId::from_name("default");
        assert!(!a.is_zero());
    }

    #[test]
    fn display_is_hex_32_chars() {
        let id = TenantId::from_bytes([0x01; 16]);
        assert_eq!(format!("{id}"), "01010101010101010101010101010101");
    }

    #[test]
    fn roundtrip_bytes() {
        let raw = [
            0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66,
            0x77, 0x88,
        ];
        let id = TenantId::from_bytes(raw);
        assert_eq!(id.as_bytes(), &raw);
    }
}
