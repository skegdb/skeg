#![deny(unsafe_code)]

//! `skeg-core` - vLog, index, compaction, cache, group commit.

pub mod cache;
pub mod group_commit;
pub mod index;
pub mod record;
pub mod segment;
pub mod snapshot;
pub mod vlog;

pub use cache::S3Fifo;
pub use group_commit::Durability;
pub use index::{Index, IndexEntry};
pub use record::{Record, RecordKind};
pub use vlog::VLog;

pub type Result<T> = std::result::Result<T, Error>;

/// Errors produced by skeg-core operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("CRC mismatch: computed {expected:#010x}, stored {got:#010x}")]
    CrcMismatch { expected: u32, got: u32 },

    #[error("invalid record: {msg}")]
    InvalidRecord { msg: &'static str },

    #[error("unknown record kind: {kind:#04x}")]
    UnknownKind { kind: u8 },
}
