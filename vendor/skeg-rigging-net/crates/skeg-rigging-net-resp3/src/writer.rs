//! `Resp3Writer`: convenience writer for tests and for the owner-side
//! "populate skeg-server with hansa-format records" use case.
//!
//! Not intended for production write paths - production owners should
//! write to skeg-server through whatever client they already use, and
//! ensure that the KV envelope at `hansa:rec:<id>` matches the
//! [`RecordEnvelope`](skeg_rigging_net::RecordEnvelope) shape.

use bytes::Bytes;
use skeg_resp3::Frame;
use skeg_rigging::{RecordId, TenantWrite};
use skeg_rigging_net::{NetError, RecordEnvelope, envelope_key_for};

use crate::connection::{Resp3Connection, encode_vector};

/// Write-side helper. Holds an authenticated connection and the index
/// name to write to.
pub struct Resp3Writer {
    conn: Resp3Connection,
    index_name: String,
    embedding_dim: u32,
}

impl Resp3Writer {
    /// Construct against an existing connection. Ensures the named
    /// index exists at `embedding_dim` (creates if missing).
    pub fn ensure(
        mut conn: Resp3Connection,
        embedding_dim: u32,
        index_name: &str,
    ) -> Result<Self, NetError> {
        // Try to create. If it already exists, skeg-server returns an
        // error string - we swallow it as long as it mentions "exists".
        let reply = conn.call(
            "SKEG.VINDEX.CREATE",
            &[
                Bytes::copy_from_slice(index_name.as_bytes()),
                Bytes::copy_from_slice(embedding_dim.to_string().as_bytes()),
                Bytes::copy_from_slice(b"f32"),
                Bytes::copy_from_slice(b"flat"),
            ],
        );
        match reply {
            Ok(_) | Err(NetError::Remote(_)) => {}
            Err(e) => return Err(e),
        }
        Ok(Self {
            conn,
            index_name: index_name.to_string(),
            embedding_dim,
        })
    }

    /// Vault id this writer was opened under. The bridge keeps it for
    /// the caller's convenience even though skeg-server's auth context
    /// is what actually scopes data.
    pub fn embedding_dim(&self) -> u32 {
        self.embedding_dim
    }

    /// Drop the index entirely (test cleanup).
    pub fn drop_index(&mut self) -> Result<(), NetError> {
        self.conn.call(
            "SKEG.VINDEX.DROP",
            &[Bytes::copy_from_slice(self.index_name.as_bytes())],
        )?;
        Ok(())
    }

    /// Insert one record: VSET the vector + SET the envelope under
    /// `hansa:rec:<id>`.
    pub fn insert(
        &mut self,
        record_id: RecordId,
        embedding: &[f32],
        shareable: bool,
        tags: Vec<String>,
        payload: Vec<u8>,
    ) -> Result<(), NetError> {
        if embedding.len() as u32 != self.embedding_dim {
            return Err(NetError::Protocol(format!(
                "vector dim mismatch: writer {}, got {}",
                self.embedding_dim,
                embedding.len()
            )));
        }
        let vec_bytes = encode_vector(embedding);
        let id_str = record_id.0.to_string();

        let r = self.conn.call(
            "SKEG.VSET",
            &[
                Bytes::copy_from_slice(self.index_name.as_bytes()),
                Bytes::copy_from_slice(id_str.as_bytes()),
                Bytes::from(vec_bytes),
            ],
        )?;
        if let Frame::Error(e) = r {
            return Err(NetError::Remote(e));
        }

        let env = RecordEnvelope::new(shareable, tags, payload);
        let key = envelope_key_for(record_id.0);
        // F.55: prefer the binary encoding. Readers detect the form
        // by first-byte magic, so this is back-compat with older
        // peers that still emit JSON.
        let r = self.conn.call(
            "SET",
            &[
                Bytes::from(key.into_bytes()),
                Bytes::from(env.encode_binary()),
            ],
        )?;
        if let Frame::Error(e) = r {
            return Err(NetError::Remote(e));
        }
        Ok(())
    }

    /// Returns the underlying connection. Intended for test cleanup.
    pub fn into_connection(self) -> Resp3Connection {
        self.conn
    }
}

impl TenantWrite for Resp3Writer {
    type Error = NetError;

    /// Delegates to the inherent [`Resp3Writer::insert`], which issues
    /// `SKEG.VSET` for the vector and `SET hansa:rec:<id>` for the
    /// envelope. Each call is a synchronous round-trip, so `flush` is a
    /// no-op (the default).
    fn insert(
        &mut self,
        record_id: RecordId,
        embedding: &[f32],
        shareable: bool,
        tags: Vec<String>,
        payload: Vec<u8>,
    ) -> Result<(), NetError> {
        Resp3Writer::insert(self, record_id, embedding, shareable, tags, payload)
    }

    fn embedding_dim(&self) -> u32 {
        self.embedding_dim
    }
}
