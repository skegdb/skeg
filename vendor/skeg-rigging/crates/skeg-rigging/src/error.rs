//! Error types surfaced by rigging traits.

/// Errors returned by [`QueryFiltered::query_filtered`](crate::QueryFiltered::query_filtered).
///
/// Stability: Stable. `#[non_exhaustive]` so variants can be added in
/// minor releases.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum QueryError {
    /// Query embedding had a dimension different from the vault's.
    #[error("embedding dim mismatch: expected {expected}, got {got}")]
    EmbeddingDimMismatch {
        /// Dimension the vault was built with.
        expected: u32,
        /// Dimension of the query embedding.
        got: u32,
    },
    /// Underlying index is corrupted in a way the search cannot recover from.
    #[error("index corrupted: {0}")]
    IndexCorrupted(String),
    /// I/O failure during search.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Errors returned by [`open_readonly`](crate::open_readonly).
///
/// Stability: Stable. `#[non_exhaustive]`.
///
/// **Format-agnostic by design.** `rigging` is the public contract crate: it
/// deliberately does not depend on any concrete on-disk format (e.g.
/// `skeg-hull`). Adapter crates convert their format-specific errors into
/// [`OpenError::Format`] at the impl boundary, carrying the underlying error
/// as a `dyn Error` source.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum OpenError {
    /// No vault at the given path.
    #[error("vault not found")]
    NotFound,
    /// Permission denied opening one of the vault's files.
    #[error("permission denied")]
    PermissionDenied,
    /// On-disk format error. Adapter-specific cause (if any) is in `source`.
    #[error("format error: {message}")]
    Format {
        /// Human-readable summary supplied by the adapter.
        message: String,
        /// Underlying adapter-specific error, if available. Boxed to keep the
        /// rigging crate free of any concrete format dependency.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
    /// Generic I/O failure.
    #[error("I/O error: {0}")]
    Io(std::io::Error),
}

impl From<std::io::Error> for OpenError {
    fn from(e: std::io::Error) -> Self {
        use std::io::ErrorKind::*;
        match e.kind() {
            NotFound => Self::NotFound,
            PermissionDenied => Self::PermissionDenied,
            _ => Self::Io(e),
        }
    }
}
