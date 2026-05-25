#![deny(unsafe_code)]

//! `skeg-proto` - binary protocol for skeg.
//!
//! Frame layout (24-byte header, all little-endian):
//! ```text
//! [magic u16][version u8][op u8][flags u32][req_id u64][payload_len u32][reserved u32]
//! ```

pub mod flags;
pub mod frame;
pub mod op;
pub mod request;
pub mod response;
pub mod vector;

pub use flags::Flags;
pub use frame::{
    Frame, FrameHeader, FrameParser, HEADER_LEN, MAGIC, MAX_FRAME_SIZE, VERSION, encode_frame,
};
pub use op::Op;
pub use request::{
    decode_key_payload, decode_mget_payload, decode_set_payload, encode_del, encode_get,
    encode_mget, encode_ping, encode_set, encode_shards, encode_stats,
};
pub use response::{
    ErrCode, ServerStats, ShardStats, VindexInfo, decode_bool_response, decode_mget_response,
    decode_shards_response, decode_stats_response, decode_u64_response, decode_value_response,
    decode_vindex_list_response, encode_err, encode_ok, encode_ok_bool, encode_ok_mget,
    encode_ok_shards, encode_ok_stats, encode_ok_u64, encode_ok_value, encode_ok_vindex_list,
};
pub use vector::{
    bytes_to_f32_vec, decode_vindex_create_payload, decode_vname_id_payload, decode_vname_payload,
    decode_vsearch_payload, decode_vsearch_response, decode_vset_payload, encode_ok_vsearch,
    encode_vdel, encode_vget, encode_vindex_create, encode_vindex_drop, encode_vindex_list,
    encode_vsearch, encode_vset, f32_vec_to_bytes,
};

/// Parse errors for frame decoding.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ParseError {
    #[error("invalid magic: expected 0x{expected:04X}, got 0x{got:04X}")]
    InvalidMagic { expected: u16, got: u16 },

    #[error("invalid version: expected {expected}, got {got}")]
    InvalidVersion { expected: u8, got: u8 },

    #[error("unknown op code: 0x{got:02X}")]
    UnknownOp { got: u8 },

    #[error("frame too large: {len} bytes (max {max})")]
    FrameTooLarge { len: u32, max: u32 },

    #[error("invalid flags bits: 0x{bits:08X}")]
    InvalidFlags { bits: u32 },

    #[error("invalid payload: {msg}")]
    InvalidPayload { msg: &'static str },
}
