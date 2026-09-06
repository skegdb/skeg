//! Umbrella network error.

/// Errors surfaced by network-attached rigging adapters.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum NetError {
    /// I/O failure on the underlying socket.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Wire protocol error (malformed frame, unexpected reply shape).
    #[error("protocol error: {0}")]
    Protocol(String),

    /// Authentication failed.
    #[error("authentication failed: {0}")]
    Auth(String),

    /// The remote returned an explicit error.
    #[error("remote error: {0}")]
    Remote(String),

    /// JSON decoding of a [`crate::RecordEnvelope`] failed.
    #[error("envelope decode: {0}")]
    Envelope(#[from] serde_json::Error),

    /// Server reply did not arrive within the configured timeout.
    #[error("timeout")]
    Timeout,

    /// Op not supported over this transport (e.g. iter_vectors over RESP3
    /// in v0.1).
    #[error("operation not supported over this transport: {0}")]
    Unsupported(&'static str),

    /// Malformed [`crate::TenantLocation`] string.
    #[error("malformed location: {0}")]
    BadLocation(String),
}

impl NetError {
    /// Whether retrying the operation may succeed without changing the
    /// request.
    ///
    /// RESP3 carries admission classes as the first whitespace-delimited
    /// word of an error. Match complete codes so similarly prefixed server
    /// errors cannot accidentally enter a retry loop.
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        let Self::Remote(message) = self else {
            return false;
        };
        matches!(
            message.split_ascii_whitespace().next(),
            Some("BACKPRESSURE" | "RATELIMITED")
        )
    }
}
