//! Public extension points for an external multi-tenant layer.
//!
//! `skeg-server` ships single-tenant by default. A separate crate
//! (`skeg-server-tenant`) can install an implementation of
//! [`TenantBackend`] via
//! [`Server::with_tenant_backend`](crate::Server::with_tenant_backend),
//! at which point the RESP3 handler honours `HELLO 3 AUTH` and scopes
//! KV / vector ops per tenant.
//!
//! The interface lives here so the public engine has no compile-time
//! dependency on any specific tenant implementation. The trait is
//! object-safe; consumers pass `Arc<dyn TenantBackend>`.

/// Fixed-width tenant identifier. 16 bytes is enough to embed any
/// 128-bit hash (we use `xxh3_128` of the tenant name in the standard
/// implementation, but the trait does not require it). The all-zero
/// id is reserved as the anonymous / single-tenant sentinel.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TenantId(pub [u8; 16]);

impl TenantId {
    /// The anonymous / single-tenant sentinel.
    pub const ZERO: Self = Self([0; 16]);
    /// Byte length of the identifier.
    pub const LEN: usize = 16;

    /// True for the `ZERO` sentinel.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.0 == [0; 16]
    }

    /// Raw bytes view.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    /// Construct from raw bytes.
    #[must_use]
    pub fn from_bytes(b: [u8; 16]) -> Self {
        Self(b)
    }
}

impl std::fmt::Display for TenantId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for b in self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

/// What to do when a RESP3 client sends `HELLO 3` without AUTH.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum AnonymousPolicy {
    /// Anonymous connections are accepted and resolved to
    /// [`TenantId::ZERO`]. Single-tenant deployments behave this way.
    #[default]
    Lenient,
    /// Anonymous connections are rejected with `-NOAUTH`.
    Strict,
}

/// Coarse classification of the command reaching the admission gate. Carried in
/// [`Admission`] so a backend can apply per-command policy - per-operation
/// metering and command-level RBAC (e.g. a tenant that may not run
/// [`VindexDrop`](CommandKind::VindexDrop)) - without the engine leaking its
/// internal RESP3 command type across the public trait.
///
/// Grouped by (resource, action) rather than one-per-command: the distinctions
/// that matter for authz/metering, no finer. Index-lifecycle ops are kept
/// individual because they are the natural RBAC target. `#[non_exhaustive]` so
/// new commands can add kinds without breaking backends (keep a `_` arm).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CommandKind {
    /// Read a KV key (GET, MGET, EXISTS).
    KvRead,
    /// Write a KV key (SET, MSET, DEL, INCR/DECR family).
    KvWrite,
    /// Read a vector (VSEARCH, VGET).
    VectorRead,
    /// Write a vector (VSET, VMSET, VDEL).
    VectorWrite,
    /// Create a vector index (VINDEX.CREATE).
    VindexCreate,
    /// Drop a vector index (VINDEX.DROP) - destructive.
    VindexDrop,
    /// Consolidate a vector index's delta (VINDEX.CONSOLIDATE).
    VindexConsolidate,
    /// List vector indexes (VINDEX.LIST).
    VindexList,
    /// Administrative command (QUOTA / QOS SET or GET).
    Admin,
    /// Connection or introspection command, never resource-gated (PING, ECHO,
    /// SELECT, WHOAMI, STATS, SHARDS, and anything else that does not consume a
    /// tenant's data-plane budget).
    Meta,
}

/// Everything the admission gate knows about one command. Passed by value to
/// [`TenantBackend::admit`]. `#[non_exhaustive]`: future fields (e.g. payload
/// size, a fairness key) can be added without breaking the trait's signature.
/// The engine constructs it; a backend only reads its fields.
#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
pub struct Admission {
    /// Tenant issuing the command (`TenantId::ZERO` = single-tenant / anonymous).
    pub tenant: TenantId,
    /// What kind of command it is, for RBAC and per-operation metering.
    pub op: CommandKind,
    /// Coarse compute weight in QoS credits (see the engine cost model).
    pub cost: u32,
}

/// External hook for the multi-tenant layer.
///
/// Implementations must be `Send + Sync` so they can be shared across
/// per-connection async tasks. The trait is object-safe.
pub trait TenantBackend: Send + Sync {
    /// Verify a `(user, password)` pair. `Some(id)` on success,
    /// `None` on any failure (wrong password, unknown user, malformed
    /// hash). Implementations are expected to be constant-time wrt
    /// user existence, to avoid leaking valid usernames via timing.
    fn verify_login(&self, user: &str, password: &[u8]) -> Option<TenantId>;

    /// True if any record in the backing store is bound to `id`. Used
    /// by the anonymous-prefix forgery defense in the RESP3 handler:
    /// a `TenantId::ZERO` client cannot forge a key whose first 16
    /// bytes match a real tenant id.
    fn has_tenant(&self, id: TenantId) -> bool;

    /// Strict or lenient handling of `HELLO 3` without AUTH.
    fn anonymous_policy(&self) -> AnonymousPolicy {
        AnonymousPolicy::Lenient
    }

    /// Hard resource limits for `id`. Default is unlimited, so existing
    /// backends and single-tenant deployments are unaffected. The server
    /// enforces these at admission (e.g. `max_vectors` on VSET).
    fn limits(&self, id: TenantId) -> crate::quota::TenantLimits {
        let _ = id;
        crate::quota::TenantLimits::default()
    }

    /// True if `id` may run admin commands (`SKEG.QUOTA.SET/GET` on other
    /// tenants). Default `false`: no tenant is an admin.
    fn is_admin(&self, id: TenantId) -> bool {
        let _ = id;
        false
    }

    /// Resolve a tenant name to its id, if such a tenant exists. Used by the
    /// admin quota commands, which target a tenant by name. Default `None`.
    fn resolve_tenant(&self, name: &str) -> Option<TenantId> {
        let _ = name;
        None
    }

    /// Set hard limits for `id`. Default: unsupported (no writable store).
    ///
    /// # Errors
    ///
    /// Returns [`QuotaAdminError`] if the backend cannot store limits.
    fn set_limits(
        &self,
        id: TenantId,
        limits: crate::quota::TenantLimits,
    ) -> Result<(), QuotaAdminError> {
        let _ = (id, limits);
        Err(QuotaAdminError::Unsupported)
    }

    /// Read a tenant's QoS limits. Default: all-unlimited, so existing backends
    /// and single-tenant deployments are unaffected.
    fn qos(&self, id: TenantId) -> crate::quota::TenantQos {
        let _ = id;
        crate::quota::TenantQos::default()
    }

    /// Set a tenant's QoS limits. Default: unsupported (no writable store).
    ///
    /// # Errors
    ///
    /// Returns [`QuotaAdminError`] if the backend cannot store QoS limits.
    fn set_qos(&self, id: TenantId, qos: crate::quota::TenantQos) -> Result<(), QuotaAdminError> {
        let _ = (id, qos);
        Err(QuotaAdminError::Unsupported)
    }

    /// Per-command admission + authorization gate. Called once per command after
    /// the tenant is resolved and before execution. The returned [`AdmitGuard`]
    /// is held by the engine for the command's whole lifetime and dropped after
    /// the response, so a backend can reserve a concurrency slot in `admit` and
    /// release it via the guard's `Drop`. `Err` refuses the command.
    ///
    /// [`Admission`] carries the tenant, the [`CommandKind`], and the coarse
    /// compute `cost` (QoS credits; see the engine cost model). A backend can
    /// charge `cost` against the tenant's budget AND apply command-level RBAC off
    /// `op` (e.g. refuse [`VindexDrop`](CommandKind::VindexDrop) for some
    /// tenants) - one choke point for both rate limiting and authorization.
    ///
    /// The default admits everything and ignores the input, so existing backends
    /// and single-tenant deployments are unaffected.
    fn admit(&self, admission: Admission) -> Result<AdmitGuard, AdmitRejected> {
        let _ = admission;
        Ok(AdmitGuard::allow())
    }
}

/// Opaque admission guard held by the engine for the duration of one command.
/// Dropping it runs the backend's RAII cleanup (e.g. releasing a per-tenant
/// concurrency slot). The default path holds nothing.
pub struct AdmitGuard(Option<Box<dyn Send>>);

impl std::fmt::Debug for AdmitGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The inner permit is `dyn Send`, not `Debug`; report only whether the
        // guard holds one.
        let held = if self.0.is_some() { "holding" } else { "allow" };
        f.debug_tuple("AdmitGuard").field(&held).finish()
    }
}

impl AdmitGuard {
    /// A guard that holds nothing - the default "admitted" path.
    #[must_use]
    pub fn allow() -> Self {
        Self(None)
    }

    /// Wrap a backend RAII permit so the engine keeps it alive while the
    /// command runs; its `Drop` fires when the engine drops the guard.
    #[must_use]
    pub fn holding(permit: impl Send + 'static) -> Self {
        Self(Some(Box::new(permit)))
    }
}

/// A refused command. `message` is sent verbatim as the RESP3 error reply, so
/// the backend should format a leading uppercase code, e.g.
/// `"RATELIMITED tenant request rate exceeded"`.
#[derive(Debug)]
pub struct AdmitRejected {
    /// The full RESP3 error string (code word + human text).
    pub message: String,
}

/// Why an admin quota write could not be applied. The dispatcher resolves the
/// tenant before calling `set_limits`, so "unknown tenant" never reaches here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuotaAdminError {
    /// The backend has no writable per-tenant limits store.
    Unsupported,
}
