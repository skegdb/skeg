//! Frame flags bitmap.

use bitflags::bitflags;

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct Flags: u32 {
        /// Block until data is flushed to durable storage (F_FULLFSYNC / fdatasync).
        const WAIT_DURABLE   = 1 << 0;
        /// Server must not send a response frame for this request.
        const NO_REPLY       = 1 << 1;
        /// First frame of a logical batch (processed atomically with BATCH_END).
        const BATCH          = 1 << 2;
        /// Last frame of a logical batch.
        const BATCH_END      = 1 << 3;
        /// Payload is LZ4-compressed.
        const COMPRESSED_LZ4 = 1 << 4;
        /// This frame is a continuation; more frames with same req_id follow.
        const CONTINUATION   = 1 << 5;
        /// SET/VSET: only if the key does NOT already exist.
        const SET_NX         = 1 << 6;
        /// SET/VSET: only if the key DOES already exist.
        const SET_XX         = 1 << 7;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_roundtrip() {
        let f = Flags::NO_REPLY | Flags::WAIT_DURABLE;
        assert_eq!(Flags::from_bits(f.bits()), Some(f));
    }

    #[test]
    fn flags_unknown_bits_rejected() {
        assert!(Flags::from_bits(1 << 31).is_none());
    }

    #[test]
    fn flags_empty_is_zero() {
        assert_eq!(Flags::empty().bits(), 0);
    }
}
