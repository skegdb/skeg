//! F.41 - Live attach mode: route ops to a running skeg-server-tenant.
//!
//! The on-disk path ([`crate::MultiTenantRoot`]) opens tenants as
//! local directories backed by `skeg-rigging-skeg::Tenant`. In a
//! cluster deployment that doesn't scale: every node needs the same
//! disk view, durability is the user's problem, and there's no
//! sharding. `LiveAttachRoot` is the opposite shape — every read
//! goes over RESP3 to a running `skeg-server-tenant` which owns the
//! data and applies its own auth/quota/sharding.
//!
//! ## Indexing
//!
//! Each `SkegTenantId` maps to one vector index inside the remote
//! skeg-server. The default naming is `tenant_<32-char-hex>` so two
//! tenants on the same server cannot collide. Override with
//! [`LiveAttachRoot::with_index_name`] when a single shared index
//! is the intended layout (the legacy "hansa"-index convention used
//! by hansa peers).
//!
//! ## Connection pooling
//!
//! All tenants opened from one root share a single
//! [`skeg_rigging_net_resp3::Resp3Pool`]. Concurrent queries against
//! the same root draw distinct connections (subject to `max_total`
//! default 16) instead of serialising on one socket.
//!
//! ## Auth
//!
//! A [`TokenStore`] is required at construction. Plain `open(tid)`
//! still works for orchestrator-side direct calls; `open_with_token`
//! resolves the tenant from the token's claim.
//!
//! ## What's NOT in v0.1 of live attach
//!
//! - Quota tracker integration. v0.1 leans on the remote
//!   skeg-server's own quota enforcement; the bridge does not
//!   double-count locally. Adding a tracker shim is a follow-up.
//! - Writes through the bridge. v0.1 is read-only (peers fan out
//!   queries). Owners populate skeg-server through whatever client
//!   they already use, matching the on-disk story.
//! - Multi-endpoint sharding. One `LiveAttachRoot` targets one
//!   endpoint; cluster-aware routing is a v0.2 concern.

use std::sync::Arc;

use skeg_rigging_net::NetError;
use skeg_rigging_net_resp3::{Resp3Pool, Resp3Tenant};
use skeg_tenant::TenantId as SkegTenantId;
use skeg_tenant::auth::{TokenError, TokenStore};

use crate::rigging_tenant_id;

/// Default index naming scheme: `tenant_<32-char-hex>`. Each
/// skeg-tenant id maps to its own RESP3 vector index, isolating
/// namespaces between tenants on the same server.
pub const DEFAULT_LIVE_INDEX_PREFIX: &str = "tenant_";

/// Multi-tenant root that dispatches every op to a running
/// `skeg-server-tenant` over RESP3.
pub struct LiveAttachRoot {
    pool: Arc<Resp3Pool>,
    token_store: Arc<TokenStore>,
    /// Either a fixed index name (every tenant uses this name) or
    /// `None` to use the per-tenant `tenant_<hex>` scheme.
    fixed_index: Option<String>,
}

impl LiveAttachRoot {
    /// Build a live root targeting `endpoint`. The `token_store`
    /// is used by [`Self::open_with_token`]; direct [`Self::open`]
    /// calls bypass token verification (an orchestrator might
    /// already know the tenant id from its own context).
    ///
    /// The internal connection pool defaults to `max_idle = 4`,
    /// `max_total = 16`, 60s idle timeout — same as
    /// [`Resp3Pool::new`]. Use [`Self::with_pool`] for explicit
    /// pool config.
    pub fn new(endpoint: impl Into<String>, token_store: Arc<TokenStore>) -> Self {
        Self {
            pool: Arc::new(Resp3Pool::new(endpoint)),
            token_store,
            fixed_index: None,
        }
    }

    /// Build a live root with an explicit pool. Use this to share
    /// one pool across multiple `LiveAttachRoot`s targeting the
    /// same endpoint, or to override the default pool config
    /// (max_total, idle_timeout).
    pub fn with_pool(pool: Arc<Resp3Pool>, token_store: Arc<TokenStore>) -> Self {
        Self {
            pool,
            token_store,
            fixed_index: None,
        }
    }

    /// Use one fixed index name across every tenant opened from
    /// this root. Default: per-tenant `tenant_<hex>` derivation.
    /// Pass `"hansa"` to match the legacy convention used by hansa
    /// peers connecting to a shared skeg-server.
    pub fn with_index_name(mut self, name: impl Into<String>) -> Self {
        self.fixed_index = Some(name.into());
        self
    }

    /// Open a tenant by id, bypassing token verification. The
    /// `embedding_dim` is informational on the bridge side (the
    /// server validates the actual index dim during the initial
    /// `VINDEX.LIST` round-trip).
    pub fn open(
        &self,
        tenant: SkegTenantId,
        _embedding_dim: u32,
    ) -> Result<Resp3Tenant, LiveAttachError> {
        let index_name = self.index_name_for(tenant);
        let rigging_id = rigging_tenant_id(tenant);
        let tenant_resp = Resp3Tenant::from_pool(self.pool.clone(), rigging_id, &index_name)
            .map_err(LiveAttachError::Net)?;
        Ok(tenant_resp)
    }

    /// Verify `token` against the configured [`TokenStore`] and
    /// open the tenant it names.
    pub fn open_with_token(
        &self,
        token: &[u8],
        embedding_dim: u32,
    ) -> Result<Resp3Tenant, LiveAttachError> {
        let verified = self
            .token_store
            .verify(token, None)
            .map_err(LiveAttachError::TokenError)?;
        self.open(verified.tenant, embedding_dim)
    }

    /// Borrow the underlying connection pool. Useful for metrics
    /// (idle / in-use counts) and for sharing the pool with other
    /// roots targeting the same endpoint.
    pub fn pool(&self) -> &Arc<Resp3Pool> {
        &self.pool
    }

    fn index_name_for(&self, tenant: SkegTenantId) -> String {
        match &self.fixed_index {
            Some(name) => name.clone(),
            None => {
                let mut s = String::with_capacity(DEFAULT_LIVE_INDEX_PREFIX.len() + 32);
                s.push_str(DEFAULT_LIVE_INDEX_PREFIX);
                for b in tenant.as_bytes() {
                    s.push_str(&format!("{b:02x}"));
                }
                s
            }
        }
    }
}

/// Errors surfaced by [`LiveAttachRoot`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum LiveAttachError {
    /// Underlying RESP3 / TCP error.
    #[error("network: {0}")]
    Net(#[from] NetError),

    /// Token verification failed.
    #[error("token verification failed: {0}")]
    TokenError(#[from] TokenError),
}
