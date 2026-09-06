//! `Resp3Tenant`: a remote tenant served by skeg-server.
//!
//! Read-only by design. The owner side writes records via
//! [`crate::Resp3Writer`] (or any other path); peers connect via this
//! struct and run filtered queries.

use std::sync::Arc;

use bytes::Bytes;
use parking_lot::Mutex;
use skeg_resp3::Frame;
use skeg_rigging::{
    Filter, Hit, IterVectors, OpenError, QueryError, QueryFiltered, ReadOnlyView, RecordId,
    RecordMeta, TenantId,
};
use skeg_rigging_net::{NetError, RecordEnvelope, envelope_key_for};

use crate::connection::{Resp3Connection, encode_vector};
use crate::pool::Resp3Pool;

/// Default index name the bridge writes to / reads from. Single index
/// per tenant; multiple hansas inside one tenant are not supported in
/// v0.1.
pub const DEFAULT_INDEX_NAME: &str = "hansa";

/// Oversample factor for filtered queries: VSEARCH returns
/// `top_k * OVERSAMPLE` candidates so post-filtering can still leave us
/// with `top_k`. 4 is enough when shareable selectivity is 25%+.
const OVERSAMPLE: u32 = 4;
/// Lower bound on the candidate set regardless of `top_k`.
const MIN_CANDIDATES: u32 = 64;
/// Default `L_search` passed to VSEARCH. Tunable per query later.
const DEFAULT_L_SEARCH: u32 = 100;

/// One peer's worth of state: a connection source + cached tenant facts.
///
/// The connection source can be either:
///
/// - **Owned** — a single `Resp3Connection` behind a `Mutex`,
///   serialising all calls against this tenant. Cheap to set up;
///   builders [`Self::connect`] / [`Self::from_connection`] use this
///   shape.
/// - **Pooled** — an `Arc<Resp3Pool>` shared with other tenants on
///   the same endpoint. Lets concurrent queries against the same
///   peer use distinct connections (subject to `max_total`) instead
///   of serialising on a single mutex. Builder [`Self::from_pool`].
pub struct Resp3Tenant {
    inner: ConnSource,
    tenant_id: TenantId,
    embedding_dim: u32,
    record_count: u64,
    index_name: String,
}

/// Where a `Resp3Tenant` finds its connection. Private detail; the
/// public surface chooses between the two via the constructors.
enum ConnSource {
    /// One connection, serialised by a mutex. Used by
    /// [`Resp3Tenant::connect`] / [`Resp3Tenant::from_connection`].
    Owned(Arc<Mutex<Resp3Connection>>),
    /// A shared pool. Used by [`Resp3Tenant::from_pool`].
    Pooled(Arc<Resp3Pool>),
}

impl ConnSource {
    /// Run `f` against a connection, locking the owned mutex or
    /// borrowing from the pool. On a pooled connection, an error
    /// inside `f` drops the connection (`PooledConnection::discard`)
    /// so the pool opens a fresh one on the next acquire — a
    /// command that fails mid-call may have left the socket in an
    /// undefined state.
    fn with_conn<F, R>(&self, f: F) -> Result<R, NetError>
    where
        F: FnOnce(&mut Resp3Connection) -> Result<R, NetError>,
    {
        match self {
            ConnSource::Owned(arc) => {
                let mut guard = arc.lock();
                f(&mut guard)
            }
            ConnSource::Pooled(pool) => {
                let mut pc = pool.acquire()?;
                let result = f(pc.conn_mut());
                if result.is_err() {
                    pc.discard();
                }
                result
            }
        }
    }
}

impl Resp3Tenant {
    /// Connect to `endpoint` and resolve the tenant's vector index.
    ///
    /// The index name defaults to `"hansa"`; pass [`Self::connect_with_index`]
    /// to override.
    pub fn connect(
        endpoint: &str,
        tenant_id: TenantId,
        auth: Option<(&str, &str)>,
    ) -> Result<Self, NetError> {
        Self::connect_with_index(endpoint, tenant_id, auth, DEFAULT_INDEX_NAME)
    }

    /// Variant of [`Self::connect`] that uses a custom index name.
    pub fn connect_with_index(
        endpoint: &str,
        tenant_id: TenantId,
        auth: Option<(&str, &str)>,
        index_name: &str,
    ) -> Result<Self, NetError> {
        let mut conn = Resp3Connection::connect(endpoint, auth)?;
        let (dim, count) = vindex_info(&mut conn, index_name)?;
        Ok(Self {
            inner: ConnSource::Owned(Arc::new(Mutex::new(conn))),
            tenant_id,
            embedding_dim: dim,
            record_count: count,
            index_name: index_name.to_string(),
        })
    }

    /// Wrap an already-open connection. Tests use this with a mock
    /// server.
    pub fn from_connection(
        mut conn: Resp3Connection,
        tenant_id: TenantId,
        index_name: &str,
    ) -> Result<Self, NetError> {
        let (dim, count) = vindex_info(&mut conn, index_name)?;
        Ok(Self {
            inner: ConnSource::Owned(Arc::new(Mutex::new(conn))),
            tenant_id,
            embedding_dim: dim,
            record_count: count,
            index_name: index_name.to_string(),
        })
    }

    /// Build a tenant backed by a shared [`Resp3Pool`]. Concurrent
    /// queries against this tenant draw distinct connections from
    /// the pool instead of serialising on a single mutex; multiple
    /// `Resp3Tenant`s targeting the same endpoint can share the
    /// same pool by cloning the `Arc` and passing it here.
    ///
    /// The constructor acquires one connection to run `VINDEX.LIST`
    /// for the initial `(dim, record_count)`; that connection is
    /// returned to the pool before the tenant becomes usable.
    pub fn from_pool(
        pool: Arc<Resp3Pool>,
        tenant_id: TenantId,
        index_name: &str,
    ) -> Result<Self, NetError> {
        let (dim, count) = {
            let mut pc = pool.acquire()?;
            let result = vindex_info(pc.conn_mut(), index_name);
            if result.is_err() {
                pc.discard();
            }
            result?
        };
        Ok(Self {
            inner: ConnSource::Pooled(pool),
            tenant_id,
            embedding_dim: dim,
            record_count: count,
            index_name: index_name.to_string(),
        })
    }

    /// Pull a fresh `record_count` from the server. Useful to refresh
    /// the cached value before saga rebuilds (although owner-side, not
    /// peer-side, would normally do that).
    pub fn refresh_record_count(&mut self) -> Result<u64, NetError> {
        let index_name = self.index_name.clone();
        let (_, count) = self
            .inner
            .with_conn(|conn| vindex_info(conn, &index_name))?;
        self.record_count = count;
        Ok(count)
    }
}

/// Issue `SKEG.VINDEX.LIST` and parse out `(dim, n_vectors)` for the
/// named index.
pub(crate) fn vindex_info(
    conn: &mut Resp3Connection,
    index_name: &str,
) -> Result<(u32, u64), NetError> {
    let reply = conn.call("SKEG.VINDEX.LIST", &[])?;
    let rows = match reply {
        Frame::Array(rows) => rows,
        other => {
            return Err(NetError::Protocol(format!(
                "VINDEX.LIST: expected Array, got {other:?}"
            )));
        }
    };
    for row in rows {
        let line = match row {
            Frame::Bulk(b) => String::from_utf8_lossy(&b).into_owned(),
            Frame::Simple(s) => s,
            _ => continue,
        };
        let mut name = None;
        let mut dim = None;
        let mut count = None;
        for kv in line.split_whitespace() {
            if let Some((k, v)) = kv.split_once('=') {
                match k {
                    "name" => name = Some(v.to_string()),
                    "dim" => dim = v.parse().ok(),
                    "n_vectors" => count = v.parse().ok(),
                    _ => {}
                }
            }
        }
        if name.as_deref() == Some(index_name) {
            let dim =
                dim.ok_or_else(|| NetError::Protocol("VINDEX.LIST row missing dim=".into()))?;
            let count = count.unwrap_or(0);
            return Ok((dim, count));
        }
    }
    Err(NetError::Protocol(format!(
        "index '{index_name}' not found on remote"
    )))
}

impl IterVectors for Resp3Tenant {
    fn iter_vectors(&self) -> Box<dyn Iterator<Item = (RecordId, Vec<f32>)> + '_> {
        // v0.1: skeg-server has no VSCAN. iter_vectors is owner-only
        // by hansa's design (saga build is local), so over the network
        // we return an empty iterator. Callers that *need* iteration
        // (e.g. analytics) should connect locally instead.
        Box::new(std::iter::empty())
    }

    fn record_count(&self) -> u64 {
        self.record_count
    }

    fn embedding_dim(&self) -> u32 {
        self.embedding_dim
    }
}

impl QueryFiltered for Resp3Tenant {
    fn query_filtered(
        &self,
        embedding: &[f32],
        top_k: u32,
        filter: &dyn Filter,
    ) -> Result<Vec<Hit>, QueryError> {
        if embedding.len() as u32 != self.embedding_dim {
            return Err(QueryError::EmbeddingDimMismatch {
                expected: self.embedding_dim,
                got: embedding.len() as u32,
            });
        }

        let oversample = (top_k.saturating_mul(OVERSAMPLE)).max(MIN_CANDIDATES);
        let vec_bytes = encode_vector(embedding);
        let index_name = self.index_name.clone();

        // Acquire a connection (owned mutex OR pool) for both
        // round-trips. Returning from the closure releases it.
        // Protocol parsing happens inside so we don't hold the
        // lock through Hit assembly.
        let (candidates, envelopes) = self
            .inner
            .with_conn(|conn| -> Result<_, NetError> {
                let search_reply = conn.call(
                    "SKEG.VSEARCH",
                    &[
                        Bytes::copy_from_slice(index_name.as_bytes()),
                        Bytes::copy_from_slice(oversample.to_string().as_bytes()),
                        Bytes::copy_from_slice(DEFAULT_L_SEARCH.to_string().as_bytes()),
                        Bytes::copy_from_slice(&vec_bytes),
                    ],
                )?;
                let candidates = parse_vsearch(search_reply)?;
                if candidates.is_empty() {
                    return Ok((candidates, Vec::new()));
                }
                let mget_args: Vec<Bytes> = candidates
                    .iter()
                    .map(|(id, _)| Bytes::from(envelope_key_for(*id).into_bytes()))
                    .collect();
                let mget_reply = conn.call("MGET", &mget_args)?;
                let envelopes = match mget_reply {
                    Frame::Array(rows) => rows,
                    other => {
                        return Err(NetError::Protocol(format!(
                            "MGET reply not Array: {other:?}"
                        )));
                    }
                };
                Ok((candidates, envelopes))
            })
            .map_err(net_to_query)?;

        // Filter + assemble hits outside the connection scope.
        let mut out: Vec<Hit> = Vec::with_capacity(top_k as usize);
        for ((id, sim), env_frame) in candidates.into_iter().zip(envelopes) {
            let bytes = match env_frame {
                Frame::Bulk(b) => b,
                Frame::Null => continue, // missing envelope: skip
                other => {
                    return Err(QueryError::IndexCorrupted(format!(
                        "MGET row not Bulk: {other:?}"
                    )));
                }
            };
            let env = RecordEnvelope::decode(&bytes)
                .map_err(|e| QueryError::IndexCorrupted(format!("envelope decode: {e}")))?;
            let tags: Vec<&str> = env.tags.iter().map(String::as_str).collect();
            let meta = RecordMeta {
                record_id: RecordId(id),
                shareable: env.shareable,
                tags: &tags,
            };
            if !filter.accept(&meta) {
                continue;
            }
            out.push(Hit {
                record_id: RecordId(id),
                similarity: sim,
                payload: Bytes::from(env.payload),
                // RESP3 VSEARCH doesn't ship the raw vector back; leave
                // None so semantic dedup degrades to byte/sentence dedup
                // for remote hits.
                embedding: None,
            });
            if out.len() as u32 >= top_k {
                break;
            }
        }
        Ok(out)
    }
}

/// Parse a VSEARCH reply into `(id, score)` pairs. Lives outside the
/// trait impl so the connection closure stays focused on network I/O.
/// Returns `NetError::Protocol` on shape mismatch so the error type
/// matches the closure signature.
fn parse_vsearch(reply: Frame) -> Result<Vec<(u64, f32)>, NetError> {
    let pairs = match reply {
        Frame::Array(pairs) => pairs,
        other => {
            return Err(NetError::Protocol(format!(
                "VSEARCH reply not Array: {other:?}"
            )));
        }
    };
    if pairs.len() % 2 != 0 {
        return Err(NetError::Protocol(format!(
            "VSEARCH returned odd-length array: {}",
            pairs.len()
        )));
    }
    let mut candidates: Vec<(u64, f32)> = Vec::with_capacity(pairs.len() / 2);
    let mut iter = pairs.into_iter();
    while let (Some(id_frame), Some(score_frame)) = (iter.next(), iter.next()) {
        let id = match id_frame {
            Frame::Bulk(b) => std::str::from_utf8(&b)
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
                .ok_or_else(|| NetError::Protocol("VSEARCH id not utf8 u64".into()))?,
            Frame::Integer(i) => i as u64,
            other => {
                return Err(NetError::Protocol(format!("VSEARCH id frame: {other:?}")));
            }
        };
        let score = match score_frame {
            Frame::Double(d) => d as f32,
            Frame::Bulk(b) => std::str::from_utf8(&b)
                .ok()
                .and_then(|s| s.parse::<f32>().ok())
                .ok_or_else(|| NetError::Protocol("VSEARCH score not utf8 f32".into()))?,
            other => {
                return Err(NetError::Protocol(format!(
                    "VSEARCH score frame: {other:?}"
                )));
            }
        };
        candidates.push((id, score));
    }
    Ok(candidates)
}

impl ReadOnlyView for Resp3Tenant {
    fn tenant_id(&self) -> TenantId {
        self.tenant_id
    }
    fn close(self: Box<Self>) -> Result<(), OpenError> {
        Ok(())
    }
}

fn net_to_query(e: NetError) -> QueryError {
    match e {
        NetError::Io(io) => QueryError::Io(io),
        other => QueryError::IndexCorrupted(format!("net: {other}")),
    }
}
