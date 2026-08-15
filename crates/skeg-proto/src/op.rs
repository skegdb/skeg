//! Wire op codes. Values are fixed - changing them breaks the protocol.

/// Op codes used in the frame header.
///
/// Request ops (0x01..=0x7F) are sent by the client.
/// Response ops (0xC0..=0xFF) are sent by the server.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
#[non_exhaustive]
pub enum Op {
    // KV scalar
    Get = 0x01,
    Set = 0x02,
    Del = 0x03,
    Mget = 0x04,
    Mset = 0x05,
    Exists = 0x06,
    Mexists = 0x07,

    // Vector
    VindexCreate = 0x10,
    VindexDrop = 0x11,
    Vset = 0x12,
    Vget = 0x13,
    Vdel = 0x14,
    Vsearch = 0x15,
    VindexList = 0x16,

    // Admin
    Ping = 0x80,
    Stats = 0x81,
    Flush = 0x82,
    Shards = 0x83,
    /// Native v2 capability negotiation. Valid only in a v2 frame.
    NativeHello = 0x84,

    // Responses
    Ok = 0xC0,
    Err = 0xC1,
    Continued = 0xC2,
}

impl Op {
    #[must_use]
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0x01 => Some(Op::Get),
            0x02 => Some(Op::Set),
            0x03 => Some(Op::Del),
            0x04 => Some(Op::Mget),
            0x05 => Some(Op::Mset),
            0x06 => Some(Op::Exists),
            0x07 => Some(Op::Mexists),
            0x10 => Some(Op::VindexCreate),
            0x11 => Some(Op::VindexDrop),
            0x12 => Some(Op::Vset),
            0x13 => Some(Op::Vget),
            0x14 => Some(Op::Vdel),
            0x15 => Some(Op::Vsearch),
            0x16 => Some(Op::VindexList),
            0x80 => Some(Op::Ping),
            0x81 => Some(Op::Stats),
            0x82 => Some(Op::Flush),
            0x83 => Some(Op::Shards),
            0x84 => Some(Op::NativeHello),
            0xC0 => Some(Op::Ok),
            0xC1 => Some(Op::Err),
            0xC2 => Some(Op::Continued),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_roundtrip_all_known() {
        let ops = [
            Op::Get,
            Op::Set,
            Op::Del,
            Op::Mget,
            Op::Mset,
            Op::Exists,
            Op::Mexists,
            Op::VindexCreate,
            Op::VindexDrop,
            Op::Vset,
            Op::Vget,
            Op::Vdel,
            Op::Vsearch,
            Op::VindexList,
            Op::Ping,
            Op::Stats,
            Op::Flush,
            Op::Shards,
            Op::NativeHello,
            Op::Ok,
            Op::Err,
            Op::Continued,
        ];
        for op in ops {
            assert_eq!(Op::from_u8(op as u8), Some(op));
        }

        // The list above is written by hand, so on its own it proves nothing
        // about an op added to the enum and forgotten here. Sweeping the byte
        // space closes that: every discriminant the decoder accepts has to be
        // one the list already covers.
        for byte in 0..=u8::MAX {
            if let Some(op) = Op::from_u8(byte) {
                assert!(
                    ops.contains(&op),
                    "{op:?} decodes from 0x{byte:02x} but is missing from the list above",
                );
            }
        }
    }

    #[test]
    fn op_from_u8_unknown_returns_none() {
        assert!(Op::from_u8(0x00).is_none());
        assert!(Op::from_u8(0xFF).is_none());
        assert!(Op::from_u8(0x09).is_none());
    }
}
