//! `MultiTenantRoot`: on-disk multi-tenant directory + auth + quota.
//!
//! On-disk layout:
//!
//! ```text
//! <root>/
//!   <tenant_id_hex>/        # one subdir per tenant
//!     meta.json             # the skeg-rigging-skeg sidecar
//! ```
//!
//! Listing the root yields every present tenant; opening one returns
//! a [`Tenant`] (read-only / read-write) or a [`TenantHandle`]
//! (quota-scoped, requires a tracker).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use skeg_rigging::{OpenError, ReadOnlyView};
use skeg_rigging_skeg::Tenant;
use skeg_tenant::TenantId as SkegTenantId;
use skeg_tenant::auth::TokenStore;
use skeg_tenant::quota::QuotaTracker;

use crate::error::MultiTenantError;
use crate::handle::TenantHandle;
use crate::rigging_tenant_id;

/// On-disk multi-tenant root with optional auth + quota machinery.
pub struct MultiTenantRoot {
    root: PathBuf,
    token_store: Option<Arc<TokenStore>>,
    quota_tracker: Option<Arc<QuotaTracker>>,
}

impl MultiTenantRoot {
    /// Build a root pointing at `root`. The directory is created
    /// lazily on first open / list.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            token_store: None,
            quota_tracker: None,
        }
    }

    /// Attach a [`TokenStore`] so [`Self::open_with_token`] can verify
    /// bearer tokens.
    pub fn with_tokens(mut self, store: Arc<TokenStore>) -> Self {
        self.token_store = Some(store);
        self
    }

    /// Attach a [`QuotaTracker`] so [`Self::open_scoped`] can enforce
    /// per-tenant caps. Shared across tenants; counters are isolated
    /// per id inside the tracker.
    pub fn with_quota_tracker(mut self, tracker: Arc<QuotaTracker>) -> Self {
        self.quota_tracker = Some(tracker);
        self
    }

    /// Root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Path of a tenant's on-disk subdirectory.
    pub fn tenant_dir(&self, tenant: SkegTenantId) -> PathBuf {
        let mut buf = String::with_capacity(32);
        for b in tenant.as_bytes() {
            buf.push_str(&format!("{b:02x}"));
        }
        self.root.join(buf)
    }

    /// Open (or create) a tenant for read-write access. **Bypasses
    /// auth + quota.** Use [`Self::open_scoped`] for the quota path.
    pub fn open(
        &self,
        tenant: SkegTenantId,
        embedding_dim: u32,
    ) -> Result<Tenant, MultiTenantError> {
        let dir = self.tenant_dir(tenant);
        let id = rigging_tenant_id(tenant);
        Tenant::open(&dir, id, embedding_dim).map_err(MultiTenantError::from)
    }

    /// Open a tenant scoped against the attached [`QuotaTracker`].
    /// Returns a [`TenantHandle`] whose insert path charges the
    /// tenant's quota entry before delegating to the adapter.
    pub fn open_scoped(
        &self,
        tenant: SkegTenantId,
        embedding_dim: u32,
    ) -> Result<TenantHandle, MultiTenantError> {
        let tracker = self
            .quota_tracker
            .as_ref()
            .ok_or(MultiTenantError::NoQuotaTracker)?;
        let inner = self.open(tenant, embedding_dim)?;
        let entry = tracker.entry(tenant);
        Ok(TenantHandle::new(inner, entry, tenant))
    }

    /// Verify `token` against the attached [`TokenStore`] and, on
    /// success, open the tenant the token names. The caller does not
    /// supply a tenant id - the token carries it.
    pub fn open_with_token(
        &self,
        token: &[u8],
        embedding_dim: u32,
    ) -> Result<Tenant, MultiTenantError> {
        let store = self
            .token_store
            .as_ref()
            .ok_or(MultiTenantError::NoTokenStore)?;
        let verified = store.verify(token, None)?;
        self.open(verified.tenant, embedding_dim)
    }

    /// Open a tenant in read-only mode for hansa-style peer queries.
    pub fn open_readonly(&self, tenant: SkegTenantId) -> Result<Box<dyn ReadOnlyView>, OpenError> {
        let dir = self.tenant_dir(tenant);
        if !dir.exists() {
            return Err(OpenError::NotFound);
        }
        skeg_rigging_skeg::open_readonly(&dir)
    }

    /// Enumerate tenants currently present on disk. Subdirs whose name
    /// doesn't parse as a 32-char hex id are ignored silently.
    pub fn list_tenants(&self) -> Result<Vec<SkegTenantId>, MultiTenantError> {
        let mut out = Vec::new();
        if !self.root.exists() {
            return Ok(out);
        }
        for entry in std::fs::read_dir(&self.root)? {
            let entry = entry?;
            let name = entry.file_name();
            let Some(name_str) = name.to_str() else {
                continue;
            };
            if name_str.len() != 32 {
                continue;
            }
            let mut id = [0u8; 16];
            let mut ok = true;
            for i in 0..16 {
                match u8::from_str_radix(&name_str[i * 2..i * 2 + 2], 16) {
                    Ok(b) => id[i] = b,
                    Err(_) => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                continue;
            }
            if entry.file_type()?.is_dir() {
                out.push(SkegTenantId::from_bytes(id));
            }
        }
        out.sort_by_key(|id| *id.as_bytes());
        Ok(out)
    }
}
