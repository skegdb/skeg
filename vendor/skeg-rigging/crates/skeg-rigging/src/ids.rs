//! Stable id newtypes shared across rigging traits.

use std::fmt;

/// Record identifier. Stable u64 newtype; the on-disk representation
/// won't change within v0.x.
///
/// Stability: Stable.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Ord, PartialOrd)]
pub struct RecordId(pub u64);

impl RecordId {
    /// Raw numeric value.
    pub fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for RecordId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u64> for RecordId {
    fn from(v: u64) -> Self {
        Self(v)
    }
}

/// Tenant identifier - engine-neutral handle on one isolated memory unit.
///
/// 16 bytes. Typically derived as `xxh3_128` of a name or a UUID;
/// rigging itself does not prescribe a derivation. A "tenant" here is
/// the engine pluralism concept (see crate-level docs §11): any engine
/// that implements rigging's capability surface refers to its isolated
/// units with this id. The auth/quota scoping `TenantId` in the
/// separate `skeg-tenant` crate is orthogonal - different purpose,
/// different namespace.
///
/// Stability: Stable.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Ord, PartialOrd)]
pub struct TenantId(pub [u8; 16]);

impl TenantId {
    /// The all-zero id, used as anonymous / single-tenant sentinel.
    pub const ZERO: Self = Self([0; 16]);

    /// Raw bytes view.
    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    /// Construct from raw bytes.
    pub fn from_bytes(b: [u8; 16]) -> Self {
        Self(b)
    }

    /// True iff this is the all-zero sentinel.
    pub fn is_zero(&self) -> bool {
        self.0 == [0; 16]
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
    fn tenant_id_hex_display() {
        let id = TenantId::from_bytes([0xab; 16]);
        assert_eq!(id.to_string(), "abababababababababababababababab");
    }

    #[test]
    fn record_id_eq_and_order() {
        assert!(RecordId(1) < RecordId(2));
        assert_eq!(RecordId(7), RecordId(7));
    }

    #[test]
    fn zero_sentinel() {
        assert!(TenantId::ZERO.is_zero());
        assert!(!TenantId::from_bytes([1; 16]).is_zero());
    }
}
